import pytest
from PIL import Image

from pixcli.palette import apply_palette, load_palette, pillow_palette


@pytest.mark.parametrize(('filename', 'content'), [
    ('colors.hex', '#ff0000\n00FF00\n'),
    ('colors.gpl', 'GIMP Palette\nName: Example\nColumns: 2\n#\n255 0 0 Red\n0 255 0 Green\n'),
    ('colors.pal', 'JASC-PAL\n0100\n2\n255 0 0\n0 255 0\n'),
    ('colors.txt', '; paint.net Palette File\nFFFF0000\nFF00FF00\n'),
])
def test_loads_supported_formats(tmp_path, monkeypatch, filename, content):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'palettes').mkdir()
    (tmp_path / 'palettes' / filename).write_text(content, encoding='utf-8')
    assert load_palette('colors') == [(255, 0, 0), (0, 255, 0)]
    assert load_palette(filename) == [(255, 0, 0), (0, 255, 0)]


def test_palette_selection_rejects_missing_ambiguous_and_path_names(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'palettes').mkdir()
    (tmp_path / 'palettes' / 'same.hex').write_text('ffffff\n')
    (tmp_path / 'palettes' / 'same.txt').write_text('FFFFFFFF\n')
    with pytest.raises(ValueError, match='ambiguous'):
        load_palette('same')
    with pytest.raises(ValueError, match='name'):
        load_palette('../same.hex')
    with pytest.raises(ValueError, match='not found'):
        load_palette('missing')


@pytest.mark.parametrize(('extension', 'content'), [
    ('hex', ''), ('hex', 'not-a-color'),
    ('gpl', 'GIMP Palette\n256 0 0 red\n'),
    ('pal', 'JASC-PAL\n0100\n2\n0 0 0\n'),
    ('txt', '12FF0000\n'),
])
def test_invalid_palettes_are_rejected(tmp_path, monkeypatch, extension, content):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'palettes').mkdir()
    (tmp_path / 'palettes' / f'bad.{extension}').write_text(content)
    with pytest.raises(ValueError):
        load_palette('bad')


def test_mapping_uses_only_exact_palette_colors_and_preserves_alpha():
    colors = [(255, 255, 255), (255, 0, 0)]
    source = Image.new('RGBA', (2, 1))
    source.putdata([(0, 0, 0, 29), (255, 30, 20, 175)])
    result = apply_palette(source, colors)
    assert result.mode == 'RGBA'
    assert [p[3] for p in result.getdata()] == [29, 175]
    assert all(p[:3] in colors for p in result.getdata())
    paletted = source.convert('RGB').quantize(palette=pillow_palette(colors), dither=Image.Dither.NONE)
    assert set(paletted.convert('RGB').getdata()) <= set(colors)


def test_rejects_more_than_256_palette_colors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'palettes').mkdir()
    (tmp_path / 'palettes' / 'big.hex').write_text('\n'.join(f'{i:06x}' for i in range(257)))
    with pytest.raises(ValueError, match='256'):
        load_palette('big')
