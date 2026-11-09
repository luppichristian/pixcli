"""File-backed loading for the pinned scaled-FP8 Wan text encoder.

The normal Comfy mixed-precision linear operations and scales are retained.
Only the large FP16 embedding table is specialized: token rows are gathered
on the CPU directly from its read-only file mapping.
"""
from __future__ import annotations

from pathlib import Path

import torch


def mapped_fp16_embedding_type(base_embedding):
    """Provide Comfy's embedding interface without materializing the table."""
    class MappedFP16Embedding(base_embedding):
        def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None,
                     norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None,
                     _freeze=False, device=None, dtype=None):
            if max_norm is not None:
                raise ValueError('Mapped FP16 embedding cannot use max_norm, which mutates weights')
            torch.nn.Module.__init__(self)
            self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
            self.padding_idx, self.max_norm, self.norm_type = padding_idx, max_norm, norm_type
            self.scale_grad_by_freq, self.sparse = scale_grad_by_freq, sparse
            self.weight = torch.nn.Parameter(torch.empty((num_embeddings, embedding_dim),
                                                          dtype=dtype or torch.float16, device='meta'),
                                             requires_grad=False)
            self.bias = None

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                                  missing_keys, unexpected_keys, error_msgs):
            key = prefix + 'weight'
            weight = state_dict.get(key)
            if weight is None:
                missing_keys.append(key)
                return
            if weight.device.type != 'cpu' or weight.dtype != torch.float16:
                error_msgs.append(f'{key}: mapped embedding must be FP16 on CPU')
                return
            if tuple(weight.shape) != (self.num_embeddings, self.embedding_dim):
                error_msgs.append(f'{key}: mapped embedding shape mismatch')
                return
            # Parameter() preserves the underlying read-only mapping; the
            # weights are consumed by embedding lookup and never mutated.
            self.weight = torch.nn.Parameter(weight, requires_grad=False)

        def _apply(self, fn, recurse=True):
            # ModelPatcher may move the model; leave this table file-backed.
            return self

        def forward(self, input, out_dtype=None):
            if getattr(self, 'weight_function', ()) or getattr(self, 'bias_function', ()):
                raise RuntimeError('Mapped FP16 embedding does not support weight patches')
            rows = torch.nn.functional.embedding(input.to('cpu'), self.weight,
                                                 self.padding_idx, None, self.norm_type,
                                                 self.scale_grad_by_freq, self.sparse)
            return rows.to(device=input.device, dtype=out_dtype)

    return MappedFP16Embedding


def load_mapped_fp8_text(path, embedding_directory=None):
    """Load the pinned FP8 encoder with Comfy's original quantization math."""
    import comfy.ops
    import comfy.sd
    import comfy.utils
    from pixcli.mapped_safetensors import load_readonly_safetensors

    path = Path(path)
    if path.name != 'umt5_xxl_fp8_e4m3fn_scaled.safetensors':
        raise ValueError('Mapped FP8 text loading requires the pinned Wan text encoder')
    state, metadata = load_readonly_safetensors(path)
    shared = state.get('shared.weight')
    marker = state.get('scaled_fp8')
    if (len(state) != 412 or shared is None or tuple(shared.shape) != (256384, 4096)
            or shared.dtype != torch.float16 or marker is None
            or marker.dtype != torch.float8_e4m3fn):
        raise ValueError('Unsupported mapped FP8 Wan text encoder structure')
    state, metadata = comfy.utils.convert_old_quants(state, model_prefix='', metadata=metadata)
    quant = comfy.utils.detect_layer_quantization(state, '')
    dtype = state['encoder.final_layer_norm.weight'].dtype
    if not quant or dtype != torch.float16:
        raise ValueError('Unsupported mapped FP8 Wan quantization metadata')
    # This is the same operation family and full-precision matrix-multiply
    # setting selected by Comfy's Wan text encoder after quant detection.
    operations = comfy.ops.mixed_precision_ops(quant, dtype, full_precision_mm=True)
    operations.Embedding = mapped_fp16_embedding_type(operations.Embedding)
    cpu = torch.device('cpu')
    return comfy.sd.load_text_encoder_state_dicts(
        [state], embedding_directory=embedding_directory, clip_type=comfy.sd.CLIPType.WAN,
        model_options={'load_device': cpu, 'offload_device': cpu, 'initial_device': cpu,
                       'custom_operations': operations}, disable_dynamic=True)


class PixcliMappedFP8UMT5Loader:
    @classmethod
    def INPUT_TYPES(cls):
        import nodes
        return nodes.NODE_CLASS_MAPPINGS['CLIPLoader'].INPUT_TYPES()

    RETURN_TYPES = ('CLIP',)
    FUNCTION = 'load_clip'
    CATEGORY = 'pixcli'

    def load_clip(self, clip_name, type='wan', device='cpu'):
        if type != 'wan' or device != 'cpu':
            raise ValueError('Mapped FP8 Wan text loading requires the CPU device')
        import folder_paths
        path = folder_paths.get_full_path_or_raise('text_encoders', clip_name)
        return (load_mapped_fp8_text(path, folder_paths.get_folder_paths('embeddings')),)
