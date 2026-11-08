"""Pixel-art transforms shared by inference and tests."""
from PIL import Image, ImageColor


def pixelate(image: Image.Image, size: tuple[int, int], colors: int | None = None) -> Image.Image:
    """Downsample to the requested sprite size without smoothing."""
    if min(size) < 1:
        raise ValueError('pixel dimensions must be positive')
    result = image.resize(size, Image.Resampling.NEAREST)
    if colors is not None:
        if not 2 <= colors <= 256:
            raise ValueError('palette colors must be between 2 and 256')
        # RGBA quantization uses Pillow's fast octree and retains transparency.
        method = Image.Quantize.FASTOCTREE if result.mode == 'RGBA' else Image.Quantize.MEDIANCUT
        result = result.quantize(colors=colors, method=method, dither=Image.Dither.NONE).convert(result.mode)
    return result


def prepare_edit(original: Image.Image, size: tuple[int, int], background: str) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Fit into inference canvas without stretching; return its crop rectangle."""
    w, h = original.size
    width, height = size
    if min(w, h, width, height) < 1:
        raise ValueError('image dimensions must be positive')
    rgb = ImageColor.getrgb(background)
    if len(rgb) != 3:
        raise ValueError('background must be an RGB color')
    scale = min(width / w, height / h)
    fitted = (max(1, round(w * scale)), max(1, round(h * scale)))
    left, top = (width - fitted[0]) // 2, (height - fitted[1]) // 2
    rgba = original.convert('RGBA').resize(fitted, Image.Resampling.NEAREST)
    canvas = Image.new('RGBA', size, (*rgb, 255))
    canvas.alpha_composite(rgba, (left, top))
    return canvas.convert('RGB'), (left, top, left + fitted[0], top + fitted[1])


def finish_edit(original: Image.Image, generated: Image.Image, box: tuple[int, int, int, int], mask: Image.Image | None = None) -> Image.Image:
    """Restore source dimensions and alpha; black mask pixels are exactly preserved."""
    mode = 'RGBA' if 'A' in original.getbands() or 'transparency' in original.info else 'RGB'
    source = original.convert(mode)
    result = generated.crop(box).resize(source.size, Image.Resampling.NEAREST).convert(mode)
    if mode == 'RGBA':
        result.putalpha(source.getchannel('A'))
    if mask is None:
        return result
    if mask.size != source.size:
        raise ValueError('mask dimensions must match the input image')
    return Image.composite(result, source, mask.convert('L'))
