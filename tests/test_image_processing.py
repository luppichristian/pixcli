from PIL import Image
import pytest
from pixcli.processing import prepare_edit, finish_edit, pixelate


def test_mask_preserves_original_rgba_exactly_outside_editable_pixels():
    original = Image.new('RGBA', (4, 2), (17, 31, 47, 83))
    mask = Image.new('L', (4, 2), 0)
    mask.putpixel((1, 0), 255)
    inference, box = prepare_edit(original, (8, 8), '#ffffff')
    edited = Image.new('RGB', inference.size, (220, 30, 20))
    output = finish_edit(original, edited, box, mask)
    assert output.size == original.size
    assert output.getpixel((0, 0)) == (17, 31, 47, 83)
    assert output.getpixel((1, 0)) == (220, 30, 20, 83)
    assert output.getpixel((3, 1)) == (17, 31, 47, 83)


def test_letterbox_preserves_aspect_ratio_and_pixel_size():
    original = Image.new('RGB', (4, 2), (255, 0, 0))
    inference, box = prepare_edit(original, (8, 8), '#000000')
    assert box == (0, 2, 8, 6)
    assert inference.getpixel((0, 0)) == (0, 0, 0)
    assert inference.getpixel((0, 2)) == (255, 0, 0)
    assert finish_edit(original, inference, box).size == (4, 2)


def test_mask_size_mismatch_is_rejected():
    with pytest.raises(ValueError, match='mask'):
        finish_edit(Image.new('RGB', (4, 2)), Image.new('RGB', (8, 8)), (0, 2, 8, 6), Image.new('L', (2, 2)))


def test_pixelate_uses_nearest_neighbor_and_requested_size():
    original = Image.new('RGB', (4, 4), 'white')
    original.putpixel((1, 1), (255, 0, 0))
    result = pixelate(original, (2, 2))
    assert result.size == (2, 2)
    assert result.getpixel((0, 0)) == (255, 0, 0)
