"""Lazy safetensors-backed image model weights.

The model is built on the meta device; only individual tensors required by a
module are opened when Accelerate moves that module to its execution device.
"""
from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
import json
import math
import struct


_DTYPE_BYTES = {'BOOL': 1, 'U8': 1, 'I8': 1, 'F8_E4M3': 1, 'I16': 2, 'I32': 4, 'I64': 8,
                'F16': 2, 'BF16': 2, 'F32': 4, 'F64': 8}


def _read_header(path: Path) -> dict:
    """Read only the bounded safetensors header, never map its payload."""
    path = Path(path)
    with path.open('rb') as file:
        size = path.stat().st_size
        prefix = file.read(8)
        if len(prefix) != 8:
            raise RuntimeError(f'Invalid safetensors header: {path}')
        length = struct.unpack('<Q', prefix)[0]
        if length == 0 or length > 64 * 1024**2 or length + 8 > size:
            raise RuntimeError(f'Invalid safetensors header size: {path}')
        raw = file.read(length)
        if len(raw) != length:
            raise RuntimeError(f'Truncated safetensors header: {path}')
    try:
        header = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise RuntimeError(f'Invalid safetensors header JSON: {path}') from exc
    if not isinstance(header, dict):
        raise RuntimeError(f'Invalid safetensors tensor list: {path}')
    result = {}
    intervals = []
    for name, info in header.items():
        if name == '__metadata__':
            continue
        try:
            dtype, shape, offsets = info['dtype'], info['shape'], info['data_offsets']
            if dtype not in _DTYPE_BYTES or not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
                raise ValueError()
            if not isinstance(offsets, list) or len(offsets) != 2 or any(type(n) is not int for n in offsets):
                raise ValueError()
            start, end = offsets
            if start < 0 or end < start or end + 8 + length > size or end - start != math.prod(shape) * _DTYPE_BYTES[dtype]:
                raise ValueError()
        except (TypeError, KeyError, ValueError) as exc:
            raise RuntimeError(f'Invalid safetensors tensor metadata for {name} in {path}') from exc
        result[name] = {'file': str(path), 'offset': 8 + length + start, 'size': end - start,
                        'shape': shape, 'source_dtype': dtype}
        if end > start:
            intervals.append((start, end))
    intervals.sort()
    if any(end > next_start for (_, end), (next_start, _) in zip(intervals, intervals[1:])):
        raise RuntimeError(f'Overlapping safetensors tensor ranges: {path}')
    return result


def checkpoint_index(model, checkpoint: Path, adapter: Path | None = None, adapter_keys: dict[str, str] | None = None) -> dict:
    adapter_keys = adapter_keys or {}
    index = {}
    base_names = _read_header(checkpoint)
    adapter_names = _read_header(adapter) if adapter else {}
    for name, tensor in model.state_dict().items():
        if name in adapter_keys:
            key, names = adapter_keys[name], adapter_names
        else:
            key = name.replace('.base_layer.', '.')
            if key.startswith('base_layer.'):
                key = key[len('base_layer.'):]
            names = base_names
        if key not in names:
            raise RuntimeError(f'Image checkpoint missing tensor {name} (source key {key})')
        record = names[key]
        if tuple(record['shape']) != tuple(tensor.shape):
            raise RuntimeError(f'Image checkpoint tensor shape mismatch for {name}')
        index[name] = {**record, 'dtype': str(tensor.dtype).removeprefix('torch.')}
    extra = set(adapter_keys) - index.keys()
    if extra:
        raise RuntimeError(f'Image adapter tensors do not match model: {len(extra)} unmatched')
    return index


class _BoundedTensorLoader(Mapping):
    """Read one tensor's bytes into writable memory, without whole-file maps."""
    def __init__(self, index: dict):
        self.index = index

    def __len__(self):
        return len(self.index)

    def __iter__(self):
        return iter(self.index)

    def __getitem__(self, key):
        import torch
        info = self.index[key]
        types = {'BOOL': torch.bool, 'U8': torch.uint8, 'I8': torch.int8, 'F8_E4M3': torch.float8_e4m3fn, 'I16': torch.int16,
                 'I32': torch.int32, 'I64': torch.int64, 'F16': torch.float16,
                 'BF16': torch.bfloat16, 'F32': torch.float32, 'F64': torch.float64}
        if info['size'] == 0:
            return torch.empty(info['shape'], dtype=getattr(torch, info['dtype']))
        data = bytearray(info['size'])
        with open(info['file'], 'rb', buffering=0) as file:
            file.seek(info['offset'])
            view = memoryview(data)
            count = 0
            while count < len(view):
                read = file.readinto(view[count:])
                if not read:
                    raise RuntimeError(f'Truncated image checkpoint tensor {key}')
                count += read
        value = torch.frombuffer(data, dtype=types[info['source_dtype']]).reshape(info['shape'])
        return value.to(dtype=getattr(torch, info['dtype']))


def attach_checkpoint_hooks(model, index: dict, device: str) -> None:
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module, attach_align_device_hook
    import torch

    model.eval()
    loader = _BoundedTensorLoader(index)
    add_hook_to_module(model, AlignDevicesHook(io_same_device=True), append=True)
    attach_align_device_hook(model, execution_device=torch.device(device), offload=True,
                             offload_buffers=True, weights_map=loader)


def _create_text_encoder(model_class, config, dtype):
    # A saved CLIP config can override _from_config's torch_dtype for child
    # modules, leaving a mixed FP16/FP32 model on CPU. Convert all meta params.
    return model_class._from_config(config, torch_dtype=dtype).to(dtype=dtype)


def load_streamed_pipeline(model_paths, *, img2img: bool, strength: float, device: str, dtype, guard):
    """Build SDXL on meta and page original model/adapter tensors on demand."""
    from accelerate import init_empty_weights
    from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
    from diffusers.utils import convert_unet_state_dict_to_peft
    from safetensors.torch import load_file
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection
    from .image import _assert_lora_coverage

    base, vae_path, adapter = (Path(model_paths[key]) for key in ('base', 'vae', 'lora'))
    # Buffers are initialized normally: CLIP's non-persistent position_ids do
    # not exist in checkpoints and must not be initialized on meta.
    with init_empty_weights():
        vae = AutoencoderKL.from_config(AutoencoderKL.load_config(str(vae_path))).to(dtype=dtype)
        unet = UNet2DConditionModel.from_config(UNet2DConditionModel.load_config(str(base/'unet'))).to(dtype=dtype)
        text = _create_text_encoder(CLIPTextModel, CLIPTextConfig.from_pretrained(base/'text_encoder', local_files_only=True), dtype)
        text2 = _create_text_encoder(CLIPTextModelWithProjection, CLIPTextConfig.from_pretrained(base/'text_encoder_2', local_files_only=True), dtype)
    guard.sample()
    cls = StableDiffusionXLImg2ImgPipeline if img2img else StableDiffusionXLPipeline
    pipe = cls.from_pretrained(str(base), vae=vae, unet=unet, text_encoder=text, text_encoder_2=text2,
                               torch_dtype=dtype, variant='fp16', local_files_only=True, use_safetensors=True)
    guard.sample()
    # Diffusers creates PEFT layers on meta. Preserve the original adapter's
    # keys through its Kohya→Diffusers→PEFT conversion, so these meta tensors
    # are served by the actual adapter file during inference.
    raw = load_file(str(adapter), device='cpu')
    original = {id(tensor): name for name, tensor in raw.items()}
    converted, _ = pipe.lora_state_dict(raw.copy(), unet_config=unet.config, local_files_only=True)
    converted = convert_unet_state_dict_to_peft({name.removeprefix('unet.'): tensor for name, tensor in converted.items() if name.startswith('unet.')})
    adapter_keys = {}
    for name, tensor in converted.items():
        target = name.replace('.lora_A.weight', '.lora_A.pixel.weight').replace('.lora_B.weight', '.lora_B.pixel.weight')
        if target == name or id(tensor) not in original or target in adapter_keys:
            raise RuntimeError(f'Could not map image adapter tensor {name} to its source')
        adapter_keys[target] = original[id(tensor)]
    pipe.load_lora_weights(raw.copy(), adapter_name='pixel', local_files_only=True, low_cpu_mem_usage=True)
    pipe.set_adapters('pixel', adapter_weights=strength)
    if 'pixel' not in pipe.get_active_adapters():
        raise RuntimeError('Pixel Art XL adapter did not activate; refusing to generate without it.')
    coverage = _assert_lora_coverage(pipe, adapter)
    actual_adapter = {name for name in unet.state_dict() if '.lora_A.pixel.' in name or '.lora_B.pixel.' in name}
    if actual_adapter != adapter_keys.keys():
        raise RuntimeError('Pixel Art XL adapter tensor mapping incomplete; refusing to continue')
    del converted, raw, original
    guard.sample()
    components = (
        (vae, vae_path/'diffusion_pytorch_model.safetensors', None, None),
        (unet, base/'unet/diffusion_pytorch_model.fp16.safetensors', adapter, adapter_keys),
        (text, base/'text_encoder/model.fp16.safetensors', None, None),
        (text2, base/'text_encoder_2/model.fp16.safetensors', None, None),
    )
    for model, checkpoint, adapter_file, mapping in components:
        index = checkpoint_index(model, checkpoint, adapter_file, mapping)
        attach_checkpoint_hooks(model, index, device)
        guard.sample()
    return pipe, coverage
