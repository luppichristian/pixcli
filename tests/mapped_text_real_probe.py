"""Opt-in CPU probe of the real mapped UMT5 loader; no inference server needed."""
from pathlib import Path
import sys
import time
import importlib
import psutil

ROOT = Path(__file__).resolve().parents[1]
RUN_FORWARD = '--forward' in sys.argv[1:]
COMFY = ROOT / '.backend' / 'ComfyUI'
MODEL = (Path.home() / '.cache/huggingface/hub/models--city96--umt5-xxl-encoder-gguf'
         / 'snapshots/b535255bee98c2b0a59ea7c0ae2dcd0c6657b3b7'
         / 'umt5-xxl-encoder-Q3_K_S.gguf')
if not MODEL.is_file():
    raise SystemExit('Pinned Q3 UMT5 model is missing')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(COMFY))
sys.argv = [sys.argv[0], '--cpu', '--disable-dynamic-vram']
import comfy.options
comfy.options.enable_args_parsing()
import folder_paths
import nodes
import torch
from pixcli.residency import resident_memory_limit

gguf_nodes = importlib.import_module('custom_nodes.ComfyUI-GGUF.nodes')
decoder = importlib.import_module('custom_nodes.ComfyUI-GGUF.dequant')
ops = importlib.import_module('custom_nodes.ComfyUI-GGUF.ops')
nodes.NODE_CLASS_MAPPINGS['CLIPLoaderGGUF'] = gguf_nodes.CLIPLoaderGGUF
folder_paths.add_model_folder_path('text_encoders', str(MODEL.parent), is_default=True)
mapped = importlib.import_module('custom_nodes.pixcli_wan.chunked_text')
started = time.monotonic()
with resident_memory_limit(5.0):
    clip, = mapped.PixcliMappedUMT5Loader().load_clip(MODEL.name)
    embeddings = [m for m in clip.cond_stage_model.modules() if type(m).__name__ == 'MappedEmbedding'
                  and tuple(m.weight.shape) == (256384, 4096)]
    assert len(embeddings) == 1
    assert not any(p.is_meta for p in clip.cond_stage_model.parameters())
    embedding = embeddings[0]
    tokens = torch.tensor([[0, 1, 7, 19, 1, 255]])
    result = embedding(tokens, out_dtype=torch.float32)
    packed = embedding.weight.data.as_subclass(torch.Tensor).index_select(0, tokens.flatten())
    actual = ops.GGMLTensor(packed, tensor_type=embedding.weight.tensor_type, tensor_shape=(6, 4096))
    expected = decoder.dequantize_tensor(actual, dtype=torch.float16).reshape(1, 6, 4096).float()
    assert torch.equal(result, expected)
    assert embedding.weight.device.type == 'cpu'
    assert torch.isfinite(result).all()
    print('Mapped UMT5 loader and token parity passed; elapsed seconds:', round(time.monotonic()-started, 2))
    print('Resident bytes:', psutil.Process().memory_info().rss)
    if RUN_FORWARD:
        import threading
        peak = [psutil.Process().memory_info().rss]
        stopped = threading.Event()
        def sample():
            process = psutil.Process()
            while not stopped.wait(0.25):
                peak[0] = max(peak[0], process.memory_info().rss)
        monitor = threading.Thread(target=sample, daemon=True)
        monitor.start()
        try:
            tokens = clip.tokenize('pixel art knight')
            with torch.inference_mode():
                output = clip.encode_from_tokens(tokens)
            assert torch.isfinite(output).all()
            print('Full CPU UMT5 text forward passed; output shape:', tuple(output.shape),
                  'elapsed seconds:', round(time.monotonic()-started, 2), 'sampled peak resident bytes:', peak[0])
        finally:
            stopped.set()
            monitor.join()
