"""ComfyUI node that refuses incomplete pixel LoRA coverage."""
import functools
import json
import torch

import comfy.lora
import comfy.model_management
import comfy.utils
import folder_paths

from pixcli.wan_coverage import adapter_targets
from pixcli.memory_telemetry import report_memory_phase
from .mmap_loader import PixcliMappedGGUFLoader


@functools.lru_cache(maxsize=1)
def _load(path, adapter_device):
    return comfy.utils.load_torch_file(path, safe_load=True, device=torch.device(adapter_device))


class PixcliCheckedPixelLoRA:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'model': ('MODEL',),
            'lora_name': (folder_paths.get_filename_list('loras'),),
            'strength': ('FLOAT', {'default': 1.0, 'min': 0.01, 'max': 2.0}),
            'expert': (['high', 'low'],),
            'adapter_device': (['cpu', 'cuda'],),
        }}

    RETURN_TYPES = ('MODEL',)
    FUNCTION = 'load'
    CATEGORY = 'pixcli'

    def load(self, model, lora_name, strength, expert, adapter_device='cpu'):
        if adapter_device not in ('cpu', 'cuda'):
            raise ValueError(f'Unsupported adapter device: {adapter_device}')
        if adapter_device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA adapter device requested but CUDA is unavailable; no fallback was used')
        path = folder_paths.get_full_path_or_raise('loras', lora_name)
        report_memory_phase(expert, 'before_load', comfy.model_management.current_loaded_models)
        tensors = _load(path, adapter_device)
        devices = {tensor.device.type for tensor in tensors.values()}
        if devices != {adapter_device}:
            raise RuntimeError(f'Pixel adapter device mismatch: requested {adapter_device}, got {sorted(devices)}')
        report_memory_phase(expert, 'after_load', comfy.model_management.current_loaded_models)
        # GGUF's state_dict can contain packed meta tensors for VRAM estimates.
        # The actual parameter exposes the original logical tensor_shape.
        state = dict(model.model.named_parameters())
        expected = adapter_targets({k: tuple(v.shape) for k, v in tensors.items()},
                                   {k: tuple(getattr(v, 'tensor_shape', v.shape)) for k, v in state.items()})
        if len(tensors) != 800 or len(expected) != 400:
            raise ValueError(f'Expected 800 pixel adapter tensors and 400 targets; found {len(tensors)} and {len(expected)}')
        keys = comfy.lora.model_lora_keys_unet(model.model, {})
        patches = comfy.lora.load_lora(tensors, keys, log_missing=True)
        used = set().union(*(getattr(p, 'loaded_keys', set()) for p in patches.values()))
        unmatched = set(tensors) - used
        if unmatched or set(patches) != expected:
            raise ValueError(f'Pixel adapter coverage incomplete for {expert}: {len(expected)} expected, '
                             f'{len(patches)} mapped, {len(unmatched)} unmatched tensors')
        result = model.clone()
        report_memory_phase(expert, 'before_apply', comfy.model_management.current_loaded_models)
        applied = set(result.add_patches(patches, strength))
        report_memory_phase(expert, 'after_apply', comfy.model_management.current_loaded_models)
        if applied != expected:
            raise ValueError(f'Pixel adapter registration incomplete for {expert}: {len(applied)}/{len(expected)}')
        print('PIXCLI_LORA_COVERAGE ' + json.dumps({'expert': expert, 'targets': len(applied),
                                                      'tensors': len(used), 'unmatched': len(unmatched),
                                                      'adapter_device': adapter_device}), flush=True)
        return (result,)


NODE_CLASS_MAPPINGS = {'PixcliCheckedPixelLoRA': PixcliCheckedPixelLoRA, 'PixcliMappedGGUFLoader': PixcliMappedGGUFLoader}
NODE_DISPLAY_NAME_MAPPINGS = {'PixcliCheckedPixelLoRA': 'pixcli checked pixel LoRA'}
