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


def gather_dequantize_embedding(source, tokens, tensor_type, dequantize, *, dtype=None):
    """Page in and decode only token rows, preserving the upstream FP16 rounding."""
    if len(source.shape) != 2 or source.data.shape[0] != source.shape[0]:
        raise ValueError('Unexpected packed embedding shape')
    rows = tokens.reshape(-1).to(device=source.data.device, dtype=torch.long)
    packed = source.data
    if type(packed) is not torch.Tensor:
        packed = packed.as_subclass(torch.Tensor)
    selected = torch.index_select(packed, 0, rows)
    chunk = tensor_type(selected, tensor_type=source.tensor_type,
                        tensor_shape=(rows.numel(), source.shape[1]))
    result = dequantize(chunk, dtype=torch.float16)
    return result.reshape(*tokens.shape, source.shape[1]).to(device=tokens.device, dtype=dtype)


def mapped_umt5_ops(ops, tensor_type, dequantize):
    """Keep the huge quantized embedding file-backed and gather requested rows."""
    class MappedEmbedding(ops.Embedding):
        def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None,
                     norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None,
                     _freeze=False, device=None, dtype=None):
            torch.nn.Module.__init__(self)
            self.num_embeddings = num_embeddings
            self.embedding_dim = embedding_dim
            self.padding_idx, self.max_norm = padding_idx, max_norm
            self.norm_type, self.scale_grad_by_freq, self.sparse = norm_type, scale_grad_by_freq, sparse
            self.weight = torch.nn.Parameter(torch.empty((num_embeddings, embedding_dim), device='meta',
                                                          dtype=dtype or torch.float32), requires_grad=False)
            self.bias = None

        def _apply(self, fn, recurse=True):
            # Comfy's patcher moves whole modules. The token table stays mapped
            # on CPU and only selected rows cross to the execution device.
            return self

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                                  missing_keys, unexpected_keys, error_msgs):
            # This includes the small unquantized relative-position tables;
            # copying into the meta placeholder would silently do nothing.
            self.ggml_load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                           missing_keys, unexpected_keys, error_msgs)

        def forward_ggml_cast_weights(self, input, out_dtype=None):
            if self.weight_function or self.bias_function or getattr(self.weight, 'patches', []):
                raise RuntimeError('Mapped UMT5 embedding does not support weight patches')
            return gather_dequantize_embedding(self.weight, input, tensor_type, dequantize, dtype=out_dtype)

    custom = ops()
    custom.Embedding = MappedEmbedding
    return custom


class PixcliMappedUMT5Loader:
    @classmethod
    def INPUT_TYPES(cls):
        import nodes
        return nodes.NODE_CLASS_MAPPINGS['CLIPLoaderGGUF'].INPUT_TYPES()

    RETURN_TYPES = ('CLIP',)
    FUNCTION = 'load_clip'
    CATEGORY = 'pixcli'

    def load_clip(self, clip_name, type='wan'):
        if type != 'wan':
            raise ValueError('Mapped UMT5 loader supports the Wan text encoder only')
        import gguf
        import nodes
        import comfy.sd
        import comfy.model_management
        import folder_paths
        from comfy.model_patcher import ModelPatcher
        from .mmap_loader import mmap_patcher_type

        upstream = nodes.NODE_CLASS_MAPPINGS['CLIPLoaderGGUF']
        package = upstream.__module__.rsplit('.', 1)[0]
        loader = importlib.import_module(package + '.loader')
        ops = importlib.import_module(package + '.ops')
        decoder = importlib.import_module(package + '.dequant')
        path = folder_paths.get_full_path('clip', clip_name)
        if path is None or not path.lower().endswith('.gguf'):
            raise ValueError('Mapped UMT5 loader requires a local GGUF encoder')
        state, extra = loader.gguf_sd_loader(path, is_text_model=True)
        embedding = state.get('token_embd.weight')
        if (extra.get('arch_str') not in ('t5', 't5encoder') or embedding is None
                or tuple(embedding.shape) != (256384, 4096)
                or embedding.tensor_type != gguf.GGMLQuantizationType.Q6_K
                or tuple(embedding.data.shape) != (256384, 3360)):
            raise ValueError('Unsupported UMT5 GGUF embedding architecture, shape or quantization')
        state['spiece_model'] = loader.gguf_tokenizer_loader(path, embedding.shape)
        state = loader.sd_map_replace(state, loader.T5_SD_MAP)
        clip = comfy.sd.load_text_encoder_state_dicts(
            clip_type=comfy.sd.CLIPType.WAN, state_dicts=[state],
            model_options={'custom_operations': mapped_umt5_ops(ops.GGMLOps, ops.GGMLTensor,
                                                               decoder.dequantize_tensor),
                           'initial_device': comfy.model_management.text_encoder_offload_device()},
            embedding_directory=folder_paths.get_folder_paths('embeddings'))
        gguf_nodes = importlib.import_module(package + '.nodes')
        clip.patcher = gguf_nodes.GGUFModelPatcher.clone(clip.patcher)
        parent = clip.patcher.__class__
        if parent.__name__ != 'GGUFModelPatcher' or not issubclass(parent, ModelPatcher):
            raise RuntimeError('Unexpected UMT5 GGUF patcher; refusing mapped loader')
        clip.patcher.__class__ = mmap_patcher_type(parent, ModelPatcher)
        return (clip,)


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
