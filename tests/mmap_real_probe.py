"""Opt-in GPU integration probe; reads one tensor and two LoRA pairs locally.

Run with the installed Comfy backend Python and ComfyUI on PYTHONPATH. This is
not part of the ordinary suite and does not perform full model inference.
"""
import gc
import hashlib
import importlib
import importlib.util
import sys
from pathlib import Path

import gguf
import torch
from safetensors import safe_open

import comfy.lora
from comfy.model_patcher import ModelPatcher


ROOT = Path(__file__).resolve().parents[1]
CACHE = Path.home() / '.cache' / 'huggingface' / 'hub'
q2 = next(CACHE.glob('models--bullerwins--Wan2.2-I2V-A14B-GGUF/snapshots/*/wan2.2_i2v_high_noise_14B_Q2_K.gguf'))
pixel = next(CACHE.glob('models--styly-agents--Wan2-2-pixel-animate/snapshots/*/*.safetensors'))
distill = next(CACHE.glob('models--Comfy-Org--Wan_2.2_ComfyUI_Repackaged/snapshots/*/split_files/loras/*high_noise.safetensors'))

gguf_nodes = importlib.import_module('custom_nodes.ComfyUI-GGUF.nodes')
ops = importlib.import_module('custom_nodes.ComfyUI-GGUF.ops')
dequant = importlib.import_module('custom_nodes.ComfyUI-GGUF.dequant')
loader = importlib.import_module('custom_nodes.ComfyUI-GGUF.loader')
spec = importlib.util.spec_from_file_location('pixcli_mmap_probe', ROOT / 'pixcli/comfy_custom_nodes/pixcli_wan/mmap_loader.py')
mapped = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mapped)

reader = gguf.GGUFReader(q2)
entry = next(t for t in reader.tensors if t.name == 'blocks.0.cross_attn.k.weight')
assert entry.tensor_type == gguf.GGMLQuantizationType.Q2_K
shape = loader.get_orig_shape(reader, entry.name) or torch.Size(tuple(reversed(entry.shape)))
mapped_bytes = torch.from_numpy(entry.data)
digest_before = hashlib.sha256(entry.data).digest()
tensor = ops.GGMLTensor(mapped_bytes, tensor_type=entry.tensor_type, tensor_shape=shape)
del entry, reader, mapped_bytes
gc.collect()

linear = ops.GGMLOps.Linear(shape[1], shape[0], bias=False)
linear.weight = torch.nn.Parameter(tensor, requires_grad=False)
model = torch.nn.Module()
model.layer = linear
patcher = gguf_nodes.GGUFModelPatcher(model, torch.device('cuda'), torch.device('cpu'))
patcher.__class__ = mapped.mmap_patcher_type(type(patcher), ModelPatcher)

prefix = 'diffusion_model.blocks.0.cross_attn.k'
all_adapters = []
mapped_values = []
for path, suffixes in ((distill, ('alpha', 'lora_up.weight', 'lora_down.weight')),
                       (pixel, ('lora_A.weight', 'lora_B.weight'))):
    if '--mapped-adapters' in sys.argv:
        from pixcli.mapped_safetensors import load_readonly_safetensors
        tensors, _ = load_readonly_safetensors(path)
        state = {prefix + '.' + suffix: tensors[prefix + '.' + suffix] for suffix in suffixes}
        mapped_values.extend((value, value.clone()) for value in state.values())
        del tensors
    else:
        with safe_open(path, framework='pt', device='cpu') as f:
            state = {prefix + '.' + suffix: f.get_tensor(prefix + '.' + suffix) for suffix in suffixes}
    adapted = comfy.lora.load_lora(state, {prefix: 'layer.weight'})
    assert set(adapted) == {'layer.weight'}
    assert set(adapted['layer.weight'].loaded_keys) == set(state)
    all_adapters.append(adapted['layer.weight'])
patcher.patches['layer.weight'] = [(1.0, a, 1.0, None, None) for a in all_adapters]

# Test actual upstream GGUF patch attachment, CPU-mapped storage surviving GC,
# and GPU layer output against dequantization plus both LoRA weight products.
patcher.patch_weight_to_device('layer.weight')
assert linear.weight.device.type == 'cpu'
assert len(linear.weight.patches) == 1
torch.manual_seed(42)
x = torch.randn(1, shape[1], device='cuda', dtype=torch.float32)
actual = linear(x)
base = dequant.dequantize_tensor(linear.weight.to('cuda'), dtype=torch.float32)
expected_weight = base
for adapter in all_adapters:
    up, down, alpha = adapter.weights[:3]
    scale = 1.0 if alpha is None else alpha / down.shape[0]
    expected_weight = expected_weight + scale * (up.to('cuda', torch.float32) @ down.to('cuda', torch.float32))
expected = torch.nn.functional.linear(x, expected_weight)
torch.testing.assert_close(actual, expected, rtol=0.002, atol=0.002)
del actual, base, expected_weight, expected, x
gc.collect()
torch.cuda.empty_cache()
assert hashlib.sha256(linear.weight.numpy()).digest() == digest_before
clone = patcher.clone().clone()
assert isinstance(clone, type(patcher))
torch.manual_seed(42)
x = torch.randn(1, shape[1], device='cuda', dtype=torch.float32)
before_load = linear(x)
clone.load(torch.device('cuda'), lowvram_model_memory=0.1)
after_load = linear(x)
torch.testing.assert_close(after_load, before_load, rtol=0.002, atol=0.002)
assert hashlib.sha256(linear.weight.numpy()).digest() == digest_before
try:
    clone.patch_weight_to_device('layer.weight', inplace_update=True)
    raise AssertionError('in-place patch should fail')
except RuntimeError as exc:
    assert 'in-place' in str(exc)
assert all(torch.equal(value, original) for value, original in mapped_values)
print('Q2 mmap after GC, two real LoRA pairs, CUDA layer parity, clone/low-VRAM load and unchanged bytes: PASS')
