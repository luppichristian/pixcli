"""Read-only, file-backed safetensors for the pinned Wan LoRA adapters.

PyTorch does not enforce buffer read-only status. Callers must only use the
returned tensors as inputs and must never modify them in place.
"""
from __future__ import annotations

import json
import mmap
from pathlib import Path
import struct
import warnings


def load_readonly_safetensors(path: str | Path):
    """Return CPU tensor views and metadata while their mapping stays alive."""
    import torch
    from .image_stream import _read_header

    path = Path(path)
    entries = _read_header(path)
    types = {'BOOL': torch.bool, 'U8': torch.uint8, 'I8': torch.int8, 'F8_E4M3': torch.float8_e4m3fn, 'I16': torch.int16,
             'I32': torch.int32, 'I64': torch.int64, 'F16': torch.float16,
             'BF16': torch.bfloat16, 'F32': torch.float32, 'F64': torch.float64}
    with path.open('rb') as file:
        header_size = struct.unpack('<Q', file.read(8))[0]
        metadata = json.loads(file.read(header_size)).get('__metadata__')
        mapping = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
    tensors = {}
    for name, info in entries.items():
        dtype = types[info['source_dtype']]
        if not info['size']:
            tensors[name] = torch.empty(info['shape'], dtype=dtype)
            continue
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='The given buffer is not writable')
            value = torch.frombuffer(mapping, dtype=dtype, count=info['size'] // dtype.itemsize,
                                     offset=info['offset']).reshape(info['shape'])
        # Keep the OS mapping alive as long as any tensor storage survives.
        value.untyped_storage()._pixcli_readonly_mapping = mapping
        tensors[name] = value
    return tensors, metadata
