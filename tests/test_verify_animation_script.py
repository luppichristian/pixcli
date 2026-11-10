"""Exercise the real export verifier with CPU and CUDA adapter audit records."""
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest

from pixcli.animation_export import export_animation


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_verifier_uses_recorded_adapter_device(tmp_path, device):
    out = tmp_path / 'walk'
    export_animation([Image.new('RGB', (8, 8), color) for color in ('red', 'blue')],
                     out, final_size=(8, 8), fps=8, mp4=False, save_json=True)
    reports = [{'expert': expert, 'targets': 400, 'tensors': 800,
                'unmatched': 0, 'adapter_device': device} for expert in ('high', 'low')]
    metadata = {'settings': {'frames': 2, 'pixel_width': 8, 'pixel_height': 8,
                             'fps': 8, 'mp4': False, 'adapter_device': device},
                'runtime': {'adapter_coverage': reports}}
    (out / 'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'verify_animation.py'
    result = subprocess.run([sys.executable, str(script), str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr

    metadata['runtime']['adapter_coverage'][1]['adapter_device'] = 'cuda' if device == 'cpu' else 'cpu'
    (out / 'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
    result = subprocess.run([sys.executable, str(script), str(out)], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'complete pixel adapter coverage' in result.stderr
