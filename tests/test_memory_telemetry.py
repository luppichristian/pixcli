import json
from types import SimpleNamespace


def test_memory_phase_reports_actual_process_and_loaded_model(capsys):
    from pixcli.memory_telemetry import report_memory_phase
    class FakeTextEncoder:
        pass
    patcher = SimpleNamespace(model=FakeTextEncoder(), is_clip=True, load_device='cpu')
    report_memory_phase('high', 'before_load', [SimpleNamespace(model=patcher)])
    line = capsys.readouterr().out.strip()
    prefix = 'PIXCLI_MEMORY_PHASE '
    assert line.startswith(prefix)
    value = json.loads(line[len(prefix):])
    assert value['expert'] == 'high' and value['phase'] == 'before_load'
    assert value['private_bytes'] > 0 and value['rss_bytes'] > 0
    assert value['physical_available_bytes'] > 0
    assert value['loaded_models'] == [{'class': 'FakeTextEncoder', 'is_text_encoder': True, 'device': 'cpu'}]
