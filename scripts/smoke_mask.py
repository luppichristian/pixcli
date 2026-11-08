"""Make a centered mask or verify an edit outside that mask."""
import sys
from pathlib import Path
from PIL import Image, ImageDraw

command, source, mask_path, *rest = sys.argv[1:]
with Image.open(source) as im:
    original = im.convert('RGBA')
if command == 'mask':
    mask = Image.new('L', original.size, 0)
    draw = ImageDraw.Draw(mask)
    w, h = original.size
    draw.rectangle((w // 4, h // 4, 3 * w // 4, 3 * h // 4), fill=255)
    Path(mask_path).parent.mkdir(parents=True, exist_ok=True)
    mask.save(mask_path)
elif command == 'verify' and len(rest) == 1:
    with Image.open(mask_path) as im:
        mask = im.convert('L')
    with Image.open(rest[0]) as im:
        result = im.convert('RGBA')
    if original.size != result.size or mask.size != original.size:
        raise SystemExit('Size mismatch')
    if any(a != b for a, b, m in zip(original.getdata(), result.getdata(), mask.getdata()) if m == 0):
        raise SystemExit('Pixels outside the mask changed')
    print('All black-mask pixels preserved exactly.')
else:
    raise SystemExit('Use: smoke_mask.py mask SOURCE MASK | verify SOURCE MASK EDIT')
