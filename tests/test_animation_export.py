import json
from pathlib import Path

import pytest
from PIL import Image

from pixcli.animation_export import export_animation


def test_exports_consistent_frames_sheet_and_timing(tmp_path):
    frames = [Image.new('RGB', (8, 4), c) for c in ('red', 'green', 'blue')]
    out = tmp_path / 'walk'
    result = export_animation(frames, out, final_size=(8, 8), fps=10)
    assert Image.open(out / 'original_frames' / '000001.png').size == (8, 4)
    assert Image.open(out / 'frames' / '000001.png').size == (8, 8)
    assert Image.open(out / 'sheet.png').size == (24, 8)
    assert Image.open(out / 'preview.gif').n_frames == 3
    meta = json.loads((out / 'sheet.json').read_text(encoding='utf-8'))
    assert [f['x'] for f in meta['frames']] == [0, 8, 16]
    assert all(f['duration_ms'] == 100 for f in meta['frames'])
    assert meta['transform']['paste_offset'] == [0, 2]
    assert result['sheet'] == str(out / 'sheet.png')


def test_export_checks_memory_before_committing_output(tmp_path):
    out = tmp_path / 'walk'
    def stop():
        raise MemoryError('test memory limit')
    with pytest.raises(MemoryError, match='test memory limit'):
        export_animation([Image.new('RGB', (8, 8))], out, final_size=(8, 8), fps=8, memory_guard=stop)
    assert not out.exists()


def test_refuses_existing_output_and_mismatched_frames(tmp_path):
    out = tmp_path / 'walk'
    out.mkdir()
    with pytest.raises(FileExistsError):
        export_animation([Image.new('RGB', (8, 8))], out, final_size=(8, 8), fps=10)
    with pytest.raises(ValueError, match='same dimensions'):
        export_animation([Image.new('RGB', (8, 8)), Image.new('RGB', (4, 4))], tmp_path / 'other', final_size=(8, 8), fps=10)


def test_shared_palette_applied_to_all_png_frames(tmp_path):
    frames = [Image.new('RGB', (2, 2), c) for c in ('red', 'blue')]
    out = tmp_path / 'walk'
    export_animation(frames, out, final_size=(2, 2), fps=8, palette_colors=2)
    first = Image.open(out / 'frames' / '000000.png')
    second = Image.open(out / 'frames' / '000001.png')
    assert first.mode == second.mode == 'P'
    assert first.getpalette() == second.getpalette()
    assert first.convert('RGB').getpixel((0, 0)) != second.convert('RGB').getpixel((0, 0))


def test_original_frame_alpha_and_overwrite_are_preserved_safely(tmp_path):
    frame = Image.new('RGBA', (3, 3), (255, 0, 0, 0))
    out = tmp_path / 'walk'
    export_animation([frame], out, final_size=(3, 3), fps=10)
    assert Image.open(out / 'original_frames' / '000000.png').getpixel((0, 0)) == (255, 0, 0, 0)
    assert Image.open(out / 'frames' / '000000.png').convert('RGB').getpixel((0, 0)) == (255, 255, 255)
    export_animation([frame], out, final_size=(4, 4), fps=10, overwrite=True)
    assert Image.open(out / 'sheet.png').size == (4, 4)
    unrelated = tmp_path / 'unrelated'
    unrelated.mkdir()
    (unrelated / 'important.txt').write_text('keep')
    with pytest.raises(FileExistsError):
        export_animation([frame], unrelated, final_size=(3, 3), fps=10, overwrite=True)
    assert (unrelated / 'important.txt').read_text() == 'keep'
