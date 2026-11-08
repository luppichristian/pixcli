"""Verify animation export layout and adapter audit without loading models."""
import json
from pathlib import Path
import sys
from PIL import Image
from pixcli.comfy_backend import _coverage_complete

root = Path(sys.argv[1])
meta = json.loads((root / 'metadata.json').read_text(encoding='utf-8'))
layout = json.loads((root / 'sheet.json').read_text(encoding='utf-8'))
settings = meta['settings']
count, width, height = settings['frames'], settings['pixel_width'], settings['pixel_height']
if layout['frame_count'] != count or len(layout['frames']) != count:
    raise SystemExit('Frame count mismatch')
for index, record in enumerate(layout['frames']):
    path = (root / record['file']).resolve()
    if not path.is_relative_to(root.resolve()):
        raise SystemExit('Frame path escapes output directory')
    with Image.open(path) as frame:
        if frame.size != (width, height):
            raise SystemExit(f'Incorrect dimensions for {path}')
    if (record['x'], record['y'], record['width'], record['height']) != (index * width, 0, width, height):
        raise SystemExit(f'Incorrect sprite rectangle for frame {index}')
    if abs(record['duration_ms'] - 1000 / settings['fps']) > 1e-6:
        raise SystemExit(f'Incorrect frame timing for frame {index}')
with Image.open(root / 'sheet.png') as sheet:
    if sheet.size != (width * count, height):
        raise SystemExit('Sprite sheet dimensions mismatch')
with Image.open(root / 'preview.gif') as preview:
    if preview.size != (width, height) or preview.n_frames < 1:
        raise SystemExit('Invalid GIF preview')
coverage = meta['runtime']['adapter_coverage']
adapter_device = settings['adapter_device']
if adapter_device not in ('cpu', 'cuda') or not _coverage_complete(coverage, adapter_device):
    raise SystemExit('Both experts must report complete pixel adapter coverage')
if settings['mp4']:
    import imageio.v2 as imageio
    reader = imageio.get_reader(root / 'preview.mp4')
    try:
        video = reader.get_meta_data()
        if tuple(video['size']) != (width, height) or abs(video['fps'] - settings['fps']) > 0.01:
            raise SystemExit('MP4 dimensions or FPS mismatch')
        if sum(1 for _ in reader) != count:
            raise SystemExit('MP4 frame count mismatch')
    finally:
        reader.close()
print(f'Animation export verified: {count} frames, {width}x{height}, {settings["fps"]} fps; metadata reports coverage for both experts.')

