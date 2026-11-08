"""Inspect bounded remote GGUF headers against the locally cached pixel adapter.

This optional research check contacts Hugging Face and reads at most 2 MiB per
expert. It does not validate model payloads or prove runtime patching.
"""
import json
from pathlib import Path
import struct
import sys
import tempfile

import gguf
import requests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pixcli.models import list_models, model_path
from pixcli.wan_coverage import adapter_targets


class HeaderReader(gguf.GGUFReader):
    def _build_tensors(self, start, fields):
        self.header_shapes = {}
        for field in fields:
            _, raw_name, _, dims, _, _ = field.parts
            name = bytes(raw_name).decode('utf-8')
            original = self.get_field('comfy.gguf.orig_shape.' + name)
            shape = tuple(int(original.parts[i][0]) for i in original.data) if original else tuple(int(x) for x in reversed(dims.tolist()))
            if name.startswith('model.diffusion_model.'):
                name = name[len('model.'):]
            elif not name.startswith('diffusion_model.'):
                name = 'diffusion_model.' + name
            self.header_shapes[name] = shape


def main():
    with model_path('wan_pixel').open('rb') as file:
        size = struct.unpack('<Q', file.read(8))[0]
        adapter = {key: tuple(value['shape']) for key, value in json.loads(file.read(size)).items() if key != '__metadata__'}
    for key in ('wan_high', 'wan_low'):
        item = list_models()[key]['files'][0]
        with requests.get(item['url'], headers={'Range': 'bytes=0-2097151'}, stream=True, timeout=(30, 120)) as response:
            if response.status_code != 206 or not response.headers.get('Content-Range', '').startswith('bytes 0-2097151/'):
                raise RuntimeError(f'{key}: server rejected a bounded header range with status {response.status_code}')
            with tempfile.TemporaryDirectory(prefix='pixcli-gguf-header-') as directory:
                path = Path(directory) / 'header.gguf'
                with path.open('wb') as out:
                    total = 0
                    for chunk in response.iter_content(1024 * 1024):
                        total += len(chunk)
                        if total > 2 * 1024 * 1024:
                            raise RuntimeError(f'{key}: server exceeded the requested header range')
                        out.write(chunk)
                reader = HeaderReader(path)
                matched = adapter_targets(adapter, reader.header_shapes)
                print(f'{key}: {len(reader.header_shapes)} header tensors, {len(matched)} adapter targets matched', flush=True)
                del reader


if __name__ == '__main__':
    main()
