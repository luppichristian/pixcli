"""Compare bounded text-embedding conversion with the pinned GGUF decoder.

Run with the animation backend Python; reads a small slice of the installed
Q3 UMT5 file and does all decoding on the CPU. Not part of the normal suite.
"""
import importlib.util
import importlib
import sys
import types
from pathlib import Path

import gguf
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL = (Path.home() / '.cache/huggingface/hub/models--city96--umt5-xxl-encoder-gguf'
         / 'snapshots/b535255bee98c2b0a59ea7c0ae2dcd0c6657b3b7'
         / 'umt5-xxl-encoder-Q3_K_S.gguf')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    if not MODEL.is_file():
        sys.exit('The pinned Q3 UMT5 model is not installed')
    comfy = ROOT / '.backend/ComfyUI'
    sys.path.insert(0, str(comfy))
    # Force ComfyUI to CPU; this probe must not disturb a running GPU job.
    sys.argv = [sys.argv[0], '--cpu']
    package = types.ModuleType('pixcli_gguf_probe')
    package.__path__ = [str(comfy / 'custom_nodes/ComfyUI-GGUF')]
    sys.modules[package.__name__] = package
    decoder = importlib.import_module(package.__name__ + '.dequant')
    PackedTensor = importlib.import_module(package.__name__ + '.ops').GGMLTensor
    chunked = load(ROOT / 'pixcli/comfy_custom_nodes/pixcli_wan/chunked_text.py', 'probe_chunked')
    reader = gguf.GGUFReader(str(MODEL))
    embedding = next(t for t in reader.tensors if t.name == 'token_embd.weight')
    assert embedding.tensor_type == gguf.GGMLQuantizationType.Q6_K
    assert embedding.data.shape == (256384, 3360)
    # Copy only a few rows to ensure this probe is non-mutating and tiny.
    packed = torch.from_numpy(embedding.data[701:770].copy())
    source = PackedTensor(packed, tensor_type=embedding.tensor_type, tensor_shape=(69, 4096))
    reference = decoder.dequantize_tensor(source, dtype=torch.float16)
    actual = chunked.chunk_dequantize_embedding(source, PackedTensor, decoder.dequantize_tensor, rows_per_chunk=17)
    assert actual.shape == (69, 4096)
    assert torch.isfinite(reference).all()
    assert torch.equal(actual, reference), f'max difference: {(actual-reference).abs().max()}'
    tokens = torch.tensor([[0, 68, 4, 0], [40, 4, 12, 67]])
    gathered = chunked.gather_dequantize_embedding(source, tokens, PackedTensor, decoder.dequantize_tensor, dtype=torch.float32)
    assert torch.equal(gathered, reference[tokens].float())
    print('CPU sparse lookup passed for real Q6_K rows, including repeated indices')
    gguf_ops = importlib.import_module(package.__name__ + '.ops')
    module = chunked.mapped_umt5_ops(gguf_ops.GGMLOps, PackedTensor, decoder.dequantize_tensor).Embedding(69, 4096)
    module.load_state_dict({'weight': source})
    module.to('meta')
    assert module.weight.device.type == 'cpu'
    assert torch.equal(module(tokens, out_dtype=torch.float32), reference[tokens].float())
    print('Real GGUF mapped embedding module passed CPU forward and preserves CPU-mapped weight under device moves')
    print('CPU parity passed: 69 actual Q6_K embedding rows, 17 rows per chunk, bitwise equal to pinned decoder')


if __name__ == '__main__':
    main()
