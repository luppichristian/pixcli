"""Read named RGB palettes and apply their colors without dithering."""
from __future__ import annotations

from pathlib import Path
import re
from typing import Sequence

from PIL import Image

Color = tuple[int, int, int]
_EXTENSIONS = ('.hex', '.gpl', '.pal', '.txt')


def _rgb_line(line: str) -> Color:
    fields = line.split()
    if len(fields) < 3:
        raise ValueError('expected three RGB values')
    rgb = tuple(int(value) for value in fields[:3])
    if any(value < 0 or value > 255 for value in rgb):
        raise ValueError('RGB values must be between 0 and 255')
    return rgb  # type: ignore[return-value]


def _validate_colors(colors: Sequence[Color]) -> None:
    if not 1 <= len(colors) <= 256:
        raise ValueError('palette must contain between 1 and 256 colors')
    if any(len(color) != 3 or any(not isinstance(c, int) or not 0 <= c <= 255 for c in color)
           for color in colors):
        raise ValueError('palette colors must contain three RGB values between 0 and 255')


def load_palette(name: str) -> list[Color]:
    """Load a palette by filename or unambiguous stem from ./palettes/."""
    if not name or name in ('.', '..') or any(c in name for c in '/\\:'):
        raise ValueError('palette name must be a filename without a directory')
    directory = Path.cwd() / 'palettes'
    suffix = Path(name).suffix.lower()
    if suffix:
        if suffix not in _EXTENSIONS:
            raise ValueError(f'unsupported palette format: {suffix}')
        matches = [directory / name] if (directory / name).is_file() else []
    else:
        matches = [directory / (name + extension) for extension in _EXTENSIONS
                   if (directory / (name + extension)).is_file()]
    if not matches:
        raise ValueError(f'palette not found in {directory}: {name}')
    if len(matches) != 1:
        raise ValueError(f'ambiguous palette name {name!r}; specify the file extension')
    path = matches[0]
    try:
        lines = path.read_text(encoding='utf-8-sig').splitlines()
        fmt = path.suffix.lower()
        colors: list[Color] = []
        if fmt == '.hex':
            for line in lines:
                value = line.strip().removeprefix('#')
                if not value:
                    continue
                if not re.fullmatch(r'[0-9a-fA-F]{6}', value):
                    raise ValueError('expected one RRGGBB color per line')
                colors.append(tuple(bytes.fromhex(value)))  # type: ignore[arg-type]
        elif fmt == '.txt':
            for line in lines:
                value = line.strip()
                if not value or value.startswith(';'):
                    continue
                if not re.fullmatch(r'[0-9a-fA-F]{8}', value):
                    raise ValueError('expected one AARRGGBB color per line')
                if value[:2].upper() != 'FF':
                    raise ValueError('Paint.NET palette colors must be opaque (alpha FF)')
                colors.append(tuple(bytes.fromhex(value[2:])))  # type: ignore[arg-type]
        elif fmt == '.gpl':
            if not lines or lines[0].strip() != 'GIMP Palette':
                raise ValueError('missing GIMP Palette header')
            for line in lines[1:]:
                value = line.strip()
                if not value or value.startswith('#') or value.startswith(('Name:', 'Columns:')):
                    continue
                colors.append(_rgb_line(value))
        else:
            if len(lines) < 3 or lines[0].strip() != 'JASC-PAL' or lines[1].strip() != '0100':
                raise ValueError('invalid JASC-PAL header')
            count = int(lines[2].strip())
            colors = [_rgb_line(line) for line in lines[3:] if line.strip()]
            if count != len(colors):
                raise ValueError('JASC-PAL color count does not match its entries')
        _validate_colors(colors)
        return colors
    except (UnicodeError, OSError, ValueError) as exc:
        raise ValueError(f'invalid palette {path}: {exc}') from exc


def pillow_palette(colors: Sequence[Color]) -> Image.Image:
    """Construct an indexed Pillow palette without introducing extra colors."""
    _validate_colors(colors)
    entries = list(colors) + [colors[0]] * (256 - len(colors))
    palette = Image.new('P', (1, 1))
    palette.putpalette([channel for color in entries for channel in color])
    return palette


def apply_palette(image: Image.Image, colors: Sequence[Color]) -> Image.Image:
    """Map RGB values to the palette while preserving any alpha channel."""
    alpha = image.convert('RGBA').getchannel('A') if 'A' in image.getbands() or 'transparency' in image.info else None
    result = image.convert('RGB').quantize(palette=pillow_palette(colors), dither=Image.Dither.NONE).convert('RGB')
    if alpha is not None:
        result.putalpha(alpha)
    return result
