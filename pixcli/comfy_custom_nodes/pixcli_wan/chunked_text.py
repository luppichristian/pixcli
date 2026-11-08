"""Experimental bounded conversion of the pinned Q3 UMT5 GGUF encoder."""
import importlib
import torch


def chunk_dequantize_embedding(source, tensor_type, dequantize, *, rows_per_chunk=512):
    """Convert a packed two-dimensional embedding to FP16 a few rows at a time."""
    if type(rows_per_chunk) is not int or rows_per_chunk <= 0:
        raise ValueError('Embedding chunk size must be a positive integer')
    if len(source.shape) != 2 or source.data.shape[0] != source.shape[0]:
        raise ValueError('Unexpected packed embedding shape')
    rows, columns = source.shape
    output = torch.empty((rows, columns), dtype=torch.float16, device='cpu')
    for start in range(0, rows, rows_per_chunk):
        stop = min(start + rows_per_chunk, rows)
        chunk = tensor_type(source.data[start:stop], tensor_type=source.tensor_type,
                            tensor_shape=(stop - start, columns))
        output[start:stop].copy_(dequantize(chunk, dtype=torch.float16))
    return output


class PixcliChunkedUMT5Loader:
    @classmethod
    def INPUT_TYPES(cls):
        import nodes
        return nodes.NODE_CLASS_MAPPINGS['CLIPLoaderGGUF'].INPUT_TYPES()

    RETURN_TYPES = ('CLIP',)
    FUNCTION = 'load_clip'
    CATEGORY = 'pixcli'

    def load_clip(self, clip_name, type='wan'):
        if type != 'wan':
            raise ValueError('Chunked UMT5 loader supports the Wan text encoder only')
        import gguf
        import nodes
        import comfy.sd
        import folder_paths
        from comfy.model_patcher import ModelPatcher
        from .mmap_loader import mmap_patcher_type

        upstream_cls = nodes.NODE_CLASS_MAPPINGS['CLIPLoaderGGUF']
        package = upstream_cls.__module__.rsplit('.', 1)[0]
        gguf_loader = importlib.import_module(package + '.loader')
        gguf_ops = importlib.import_module(package + '.ops')
        gguf_dequant = importlib.import_module(package + '.dequant')
        path = folder_paths.get_full_path('clip', clip_name)
        if path is None or not path.lower().endswith('.gguf'):
            raise ValueError('Chunked UMT5 loader requires a local GGUF encoder')

        state, extra = gguf_loader.gguf_sd_loader(path, is_text_model=True)
        embedding = state.get('token_embd.weight')
        if (extra.get('arch_str') not in ('t5', 't5encoder') or embedding is None
                or tuple(embedding.shape) != (256384, 4096)
                or embedding.tensor_type != gguf.GGMLQuantizationType.Q6_K
                or tuple(embedding.data.shape) != (256384, 3360)):
            raise ValueError('Unsupported UMT5 GGUF embedding architecture, shape or quantization')
        state['spiece_model'] = gguf_loader.gguf_tokenizer_loader(path, embedding.shape)
        state['token_embd.weight'] = chunk_dequantize_embedding(
            embedding, gguf_ops.GGMLTensor, gguf_dequant.dequantize_tensor)
        del embedding
        state = gguf_loader.sd_map_replace(state, gguf_loader.T5_SD_MAP)
        clip = upstream_cls().load_patcher([path], comfy.sd.CLIPType.WAN, [state])
        parent = clip.patcher.__class__
        if parent.__name__ != 'GGUFModelPatcher' or not issubclass(parent, ModelPatcher):
            raise RuntimeError('Unexpected UMT5 GGUF patcher; refusing experimental loader')
        if clip.patcher.weight_inplace_update:
            raise RuntimeError('Mapped GGUF does not permit in-place weight updates')
        clip.patcher.__class__ = mmap_patcher_type(parent, ModelPatcher)
        return (clip,)
