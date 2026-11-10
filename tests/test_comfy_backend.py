import pytest
from pixcli.comfy_backend import build_graph, validate_animation


def test_validates_frame_count_and_dimensions_before_load():
    with pytest.raises(ValueError, match='4n'):
        validate_animation(256, 256, 16)
    with pytest.raises(ValueError, match='16'):
        validate_animation(255, 256, 17)


def test_graph_routes_both_experts_and_pixel_adapter():
    graph = build_graph('source.png', 'walk', 'avoid blur', 42, 256, 256, 17, 1.0)
    assert graph['high_pixel']['inputs']['expert'] == 'high'
    assert graph['low_pixel']['inputs']['expert'] == 'low'
    assert graph['high_sample']['inputs']['start_at_step'] == 0
    assert graph['high_sample']['inputs']['end_at_step'] == 2
    assert graph['low_sample']['inputs']['start_at_step'] == 2
    assert graph['low_sample']['inputs']['end_at_step'] == 4
    assert graph['low_sample']['inputs']['latent_image'] == ['high_sample', 0]
    assert graph['decode']['inputs']['samples'] == ['low_sample', 0]


def test_disk_mapped_experts_are_selected_explicitly_for_both_stages():
    args = ('source.png', 'walk', '', 42, 256, 256, 5, 1.0)
    ordinary = build_graph(*args)
    mapped = build_graph(*args, weight_storage='mmap')
    for expert in ('high', 'low'):
        assert ordinary[expert + '_model']['class_type'] == 'UnetLoaderGGUF'
        assert mapped[expert + '_model']['class_type'] == 'PixcliMappedGGUFLoader'
        assert mapped[expert + '_distill']['inputs']['model'] == [expert + '_model', 0]
    with pytest.raises(ValueError, match='weight storage'):
        build_graph(*args, weight_storage='unsupported')


def test_adapter_device_is_explicit_for_both_experts():
    args = ('source.png', 'walk', '', 42, 256, 256, 5, 1.0)
    default = build_graph(*args)
    assert all(default[expert + '_pixel']['inputs']['adapter_device'] == 'cpu' for expert in ('high', 'low'))
    cuda = build_graph(*args, adapter_device='cuda')
    assert all(cuda[expert + '_pixel']['inputs']['adapter_device'] == 'cuda' for expert in ('high', 'low'))
    with pytest.raises(ValueError, match='adapter device'):
        build_graph(*args, adapter_device='bogus')


def test_cuda_adapter_requires_gpu_resident_preset(tmp_path):
    from pixcli.comfy_backend import run_comfy
    args = (tmp_path/'x.png', 'walk', '', 1, 256, 256, 5, 1.0, {})
    with pytest.raises(ValueError, match='gpu-resident'):
        run_comfy(*args, adapter_device='cuda')
    with pytest.raises(ValueError, match='fp8'):
        run_comfy(*args, adapter_device='cuda', preset='gpu-resident', text_encoder='q3')


def test_missing_animation_models_fail_before_startup(tmp_path):
    from pixcli.comfy_backend import run_comfy
    with pytest.raises(FileNotFoundError, match='models download -profile animation'):
        run_comfy(tmp_path / 'missing.png', 'walking', '', 42, 256, 256, 17, 1.0, {})


def test_rejects_ambiguous_model_directory_shadowing(tmp_path):
    from pixcli.comfy_backend import validate_model_locations
    first = tmp_path / 'a'
    second = tmp_path / 'b'
    first.mkdir(); second.mkdir()
    paths = {}
    for key, name, folder in [
        ('wan_high', 'wan2.2_i2v_high_noise_14B_Q3_K_S.gguf', first),
        ('wan_low', 'wan2.2_i2v_low_noise_14B_Q3_K_S.gguf', second),
        ('wan_text', 'umt5_xxl_fp8_e4m3fn_scaled.safetensors', first),
        ('wan_vae', 'wan_2.1_vae.safetensors', first),
        ('wan_distill_high', 'wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors', first),
        ('wan_distill_low', 'wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors', second),
        ('wan_pixel', 'wan2.2_animate_adapter_model.safetensors', first),
    ]:
        paths[key] = folder / name
        paths[key].touch()
    validate_model_locations(paths)
    (second / paths['wan_distill_high'].name).touch()
    with pytest.raises(ValueError, match='shadow'):
        validate_model_locations(paths)


def test_graph_rejects_invalid_seed_before_startup(tmp_path):
    from pixcli.comfy_backend import run_comfy
    with pytest.raises(ValueError, match='seed'):
        run_comfy(tmp_path / 'missing.png', 'walking', '', -1, 256, 256, 17, 1.0, {})


def test_readiness_rejects_missing_or_modified_lock_and_node(tmp_path):
    import hashlib
    import json
    from pixcli.comfy_backend import validate_ready
    node = tmp_path / 'node.py'
    node.write_text('original')
    lock = tmp_path / 'requirements.lock'
    lock.write_text('torch==2.8')
    marker = tmp_path / 'ready.json'
    commits = {'ComfyUI': 'abc'}
    with pytest.raises(RuntimeError, match='setup'):
        validate_ready(marker, lock, node, commits)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    marker.write_text(json.dumps({'schema_version': 1, 'lock_sha256': digest(lock), 'node_sha256': digest(node), 'commits': commits}), encoding='utf-8-sig')
    validate_ready(marker, lock, node, commits)
    mmap_node = tmp_path / 'mmap_loader.py'
    mmap_node.write_text('mapped original')
    ready = json.loads(marker.read_text(encoding='utf-8-sig'))
    ready['mmap_node_sha256'] = digest(mmap_node)
    marker.write_text(json.dumps(ready), encoding='utf-8')
    validate_ready(marker, lock, node, commits, mmap_node=mmap_node)
    mmap_node.write_text('mapped changed')
    with pytest.raises(RuntimeError, match='drift'):
        validate_ready(marker, lock, node, commits, mmap_node=mmap_node)
    lock.write_text('changed')
    with pytest.raises(RuntimeError, match='drift'):
        validate_ready(marker, lock, node, commits)


def test_readiness_checks_streamed_text_loader(tmp_path):
    import hashlib
    import json
    from pixcli.comfy_backend import validate_ready
    lock, node, text_node = (tmp_path / name for name in ('lock', 'node', 'chunked_text.py'))
    for path in (lock, node, text_node):
        path.write_text(path.name)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    marker = tmp_path / 'ready.json'
    marker.write_text(json.dumps({'schema_version': 1, 'lock_sha256': digest(lock),
                                  'node_sha256': digest(node), 'chunked_node_sha256': digest(text_node),
                                  'commits': {}}))
    validate_ready(marker, lock, node, {}, chunked_node=text_node)
    text_node.write_text('changed')
    with pytest.raises(RuntimeError, match='drift'):
        validate_ready(marker, lock, node, {}, chunked_node=text_node)

def test_pixel_adapter_strength_reaches_both_experts():
    graph = build_graph('source.png', 'walk', '', 42, 256, 256, 17, 1.0, lora_strength=0.7)
    assert graph['high_pixel']['inputs']['strength'] == 0.7
    assert graph['low_pixel']['inputs']['strength'] == 0.7
    assert graph['high_distill']['inputs']['strength_model'] == 1.0

def test_local_backend_requests_ignore_proxy_settings(monkeypatch):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from threading import Thread
    from pixcli.comfy_backend import _local_opener
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'ok')
        def log_message(self, *args):
            pass
    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv('HTTP_PROXY', 'http://127.0.0.1:9')
        monkeypatch.setenv('NO_PROXY', '')
        monkeypatch.setenv('no_proxy', '')
        with _local_opener().open(f'http://127.0.0.1:{server.server_port}/', timeout=2) as response:
            assert response.read() == b'ok'
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

def test_gpu_memory_probe_parses_nvidia_smi_and_handles_unavailable(monkeypatch):
    from types import SimpleNamespace
    from pixcli import comfy_backend
    monkeypatch.setattr(comfy_backend, '_run_supervised_probe', lambda *a, **kw: SimpleNamespace(stdout='1024\n'))
    assert comfy_backend._gpu_memory_used() == 1024 * 1024 * 1024
    def missing(*args, **kwargs):
        raise FileNotFoundError('nvidia-smi')
    monkeypatch.setattr(comfy_backend, '_run_supervised_probe', missing)
    assert comfy_backend._gpu_memory_used() is None

def test_optional_gpu_telemetry_does_not_spawn_for_cpu_or_ram_budget(monkeypatch):
    from pixcli import comfy_backend
    calls = []
    monkeypatch.setattr(comfy_backend, '_gpu_memory_used', lambda **kwargs: calls.append(kwargs) or 123)
    assert comfy_backend._optional_gpu_memory_used('cpu', None, 2, 2) is None
    assert comfy_backend._optional_gpu_memory_used('cuda', 4, 2, 2) is None
    assert calls == []
    assert comfy_backend._optional_gpu_memory_used('cuda', None, 3, 4) == 123
    assert calls == [{'min_free_commit_gib': 3, 'min_free_physical_gib': 4}]

def test_cuda_oom_reports_actual_configuration_and_no_fallback():
    from pixcli.comfy_backend import _oom_explanation
    text = _oom_explanation('torch.OutOfMemoryError: CUDA out of memory', 512, 512, 9)
    assert '512x512' in text and '9 frames' in text
    assert '-inference-width 384' in text and '-frames 5' in text
    assert 'no automatic fallback' in text.lower()
    assert _oom_explanation('some unrelated failure', 512, 512, 9) is None

def test_cuda_oom_advice_never_increases_canvas_size():
    from pixcli.comfy_backend import _oom_explanation
    text = _oom_explanation('CUDA out of memory', 256, 256, 5)
    assert '-inference-width 384' not in text

def test_backend_refuses_cpu_fallback():
    from pixcli.comfy_backend import _require_cuda_device
    with pytest.raises(RuntimeError, match='CUDA'):
        _require_cuda_device({'devices': [{'type': 'cpu', 'name': 'CPU'}]})
    assert _require_cuda_device({'devices': [{'type': 'cuda', 'name': 'GPU 0'}]})['name'] == 'GPU 0'

def test_runtime_coverage_requires_all_tensors_for_both_experts():
    from pixcli.comfy_backend import _coverage_complete
    reports = [{'expert': e, 'targets': 400, 'tensors': 800, 'unmatched': 0, 'adapter_device': 'cuda'} for e in ('high', 'low')]
    assert _coverage_complete(reports, 'cuda')
    assert not _coverage_complete(reports, 'cpu')
    reports[1]['tensors'] = 799
    assert not _coverage_complete(reports, 'cuda')

def test_rejects_unrelated_local_server():
    from pixcli.comfy_backend import _verify_backend_identity
    expected = {'system': {'argv': ['main.py', '--port', '4321', '--user-directory', 'expected-user', '--lowvram', '--disable-dynamic-vram', '--disable-pinned-memory', '--cache-none']}}
    _verify_backend_identity(expected, 4321, 'expected-user')
    with pytest.raises(RuntimeError, match='backend identity'):
        _verify_backend_identity({'system': {'argv': ['main.py', '--port', '4321', '--user-directory', 'other']}}, 4321, 'expected-user')

def test_backend_presets_are_explicit_and_q3_text_is_rejected_for_gpu_resident(tmp_path):
    from pixcli.comfy_backend import _backend_preset_flags, run_comfy
    common = {'--disable-dynamic-vram', '--disable-pinned-memory', '--cache-none'}
    assert set(_backend_preset_flags('low-vram', 'q3')) == common | {'--lowvram'}
    assert set(_backend_preset_flags('gpu-resident', 'fp8')) == common | {'--highvram'}
    with pytest.raises(ValueError, match='fp8'):
        run_comfy(tmp_path/'x.png', 'walk', '', 1, 256, 256, 5, 1.0, {}, preset='gpu-resident', text_encoder='q3')
    with pytest.raises(ValueError, match='preset'):
        _backend_preset_flags('unknown', 'fp8')

def test_backend_identity_checks_effective_vram_flags():
    from pixcli.comfy_backend import _verify_backend_identity
    base = ['main.py', '--port', '4321', '--user-directory', 'user', '--disable-dynamic-vram', '--disable-pinned-memory', '--cache-none']
    _verify_backend_identity({'system': {'argv': base + ['--highvram']}}, 4321, 'user', 'gpu-resident')
    for flags in (['--lowvram'], ['--highvram', '--lowvram'], []):
        with pytest.raises(RuntimeError, match='backend identity'):
            _verify_backend_identity({'system': {'argv': base + flags}}, 4321, 'user', 'gpu-resident')
    with pytest.raises(RuntimeError, match='backend identity'):
        _verify_backend_identity({'system': {'argv': base[:-1] + ['--highvram']}}, 4321, 'user', 'gpu-resident')

def test_oom_reports_selected_preset():
    from pixcli.comfy_backend import _oom_explanation
    text = _oom_explanation('CUDA out of memory', 256, 256, 5, preset='gpu-resident')
    assert 'preset=gpu-resident' in text

def test_memory_guard_uses_requested_budget():
    from pixcli.comfy_backend import memory_guard_violation
    gib = 1024**3
    assert memory_guard_violation(15*gib, 5*gib, 4*gib, 16, 3, 4) is None
    assert memory_guard_violation(17*gib, 5*gib, 4*gib, 16, 3, 4) == 'process private memory'
    assert memory_guard_violation(15*gib, 3*gib, 4*gib, 16, 3, 4) == 'physical memory'
    assert memory_guard_violation(15*gib, 5*gib, 2*gib, 16, 3, 4) == 'commit memory'
    assert memory_guard_violation(15*gib, 5*gib, None, 16, 3, 4) is None


def test_explicit_ram_budget_does_not_add_implicit_private_cap():
    from pixcli.comfy_backend import _effective_private_budget, memory_guard_violation
    gib = 1024**3
    assert _effective_private_budget(None, None) == 14
    assert _effective_private_budget(None, 8) is None
    assert _effective_private_budget(14, 8) == 14
    assert memory_guard_violation(20*gib, 5*gib, 5*gib, None, 2, 2) is None


def test_mapped_preflight_uses_requested_reserves():
    from pixcli.comfy_backend import _validate_animation_headroom
    gib = 1024**3
    _validate_animation_headroom(3*gib, 3*gib, 2, 2, 'mmap')
    with pytest.raises(RuntimeError, match='physical'):
        _validate_animation_headroom(gib, 3*gib, 2, 2, 'mmap')
    with pytest.raises(RuntimeError, match='commit'):
        _validate_animation_headroom(3*gib, gib, 2, 2, 'mmap')
    with pytest.raises(RuntimeError):
        _validate_animation_headroom(3*gib, 3*gib, 2, 2, 'normal')


def test_vram_reservation_steers_low_vram_and_is_verified():
    from pixcli.comfy_backend import _vram_reserve_flags, _verify_backend_identity
    flags = _vram_reserve_flags(12 * 1024**3, 4)
    assert flags == ('--reserve-vram', '8')
    args = ['main.py', '--port', '4321', '--user-directory', 'user', '--lowvram',
            '--disable-dynamic-vram', '--disable-pinned-memory', '--cache-none', *flags]
    _verify_backend_identity({'system': {'argv': args}}, 4321, 'user', reserve_flags=flags)
    with pytest.raises(RuntimeError, match='identity'):
        _verify_backend_identity({'system': {'argv': args[:-1] + ['2']}}, 4321, 'user', reserve_flags=flags)


def test_cuda_budget_configures_allocator_and_comfy_reservation():
    from types import SimpleNamespace
    from pixcli.comfy_launch import configure_cuda_budget
    calls = []
    cuda = SimpleNamespace(get_device_properties=lambda device: SimpleNamespace(total_memory=12 * 1024**3),
                           set_per_process_memory_fraction=lambda fraction, device: calls.append((fraction, device)),
                           get_per_process_memory_fraction=lambda device: calls[-1][0])
    flags, report = configure_cuda_budget(SimpleNamespace(cuda=cuda), 4)
    assert calls == [(1 / 3, 0)]
    assert flags == ('--reserve-vram', '8')
    assert report['allocator_fraction'] == 1 / 3


def test_launch_configures_backend_before_torch_and_updates_parsed_reservation(tmp_path):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    comfy = tmp_path / 'comfy'
    comfy.mkdir()
    (comfy / '__init__.py').write_text('')
    (comfy / 'options.py').write_text('def enable_args_parsing(): pass\n')
    (comfy / 'cli_args.py').write_text(
        'import argparse\nparser = argparse.ArgumentParser()\n'
        'parser.add_argument("--reserve-vram", type=float)\n'
        'parser.add_argument("--cuda-device")\nparser.add_argument("--default-device", type=int)\n'
        'args = parser.parse_args()\n')
    (tmp_path / 'cuda_malloc.py').write_text('import os\nos.environ["PIXCLI_TEST_ALLOCATOR"] = "configured"\n')
    (tmp_path / 'torch.py').write_text(
        'import os\nassert os.environ.get("PIXCLI_TEST_ALLOCATOR") == "configured"\n'
        'assert os.environ.get("CUDA_VISIBLE_DEVICES") == "3"\n'
        'class Device: total_memory = 12 * 1024**3\n'
        'class Cuda:\n'
        '    fraction = 0\n'
        '    def get_device_properties(self, device): return Device()\n'
        '    def set_per_process_memory_fraction(self, value, device): self.fraction = value\n'
        '    def get_per_process_memory_fraction(self, device): return self.fraction\n'
        'cuda = Cuda()\n')
    (tmp_path / 'main.py').write_text(
        'import json, sys\nfrom comfy.cli_args import args\n'
        'print("STUB " + json.dumps({"reserve": args.reserve_vram, "argv": sys.argv}))\n')
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update({'PIXCLI_VRAM_BUDGET_GIB': '4', 'PYTHONPATH': str(root)})
    env.pop('CUDA_VISIBLE_DEVICES', None)
    output = subprocess.check_output(
        [sys.executable, str(root / 'pixcli/comfy_launch.py'), str(tmp_path / 'main.py'), '--cuda-device', '3'],
        env=env, text=True)
    report = json.loads(next(line.removeprefix('STUB ') for line in output.splitlines() if line.startswith('STUB ')))
    assert report['reserve'] == 8
    assert report['argv'][-2:] == ['--reserve-vram', '8']


def test_ram_only_launch_starts_without_initializing_cuda(tmp_path):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path
    (tmp_path / 'main.py').write_text('print("RAM-ONLY-MAIN")\n')
    (tmp_path / 'torch.py').write_text('raise AssertionError("torch must be initialized by the backend")\n')
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update({'PIXCLI_RAM_BUDGET_GIB': '1', 'PIXCLI_RAM_PROCESS_LIMIT_BYTES': str(512 * 1024**2), 'PYTHONPATH': str(root)})
    env.pop('PIXCLI_VRAM_BUDGET_GIB', None)
    output = subprocess.check_output([sys.executable, str(root / 'pixcli/comfy_launch.py'),
                                      str(tmp_path / 'main.py')], env=env, text=True)
    report = json.loads(next(line.removeprefix('PIXCLI_RAM_BUDGET ') for line in output.splitlines()
                             if line.startswith('PIXCLI_RAM_BUDGET ')))
    assert report['budget_gib'] == 1
    assert report['process_budget_gib'] == .5
    assert isinstance(report['pid'], int) and report['pid'] > 0
    assert 'RAM-ONLY-MAIN' in output


def test_animation_device_auto_chooses_cpu_when_cuda_is_unavailable():
    from pixcli.comfy_backend import (_resolve_animation_device, _backend_preset_flags,
                                      _require_backend_device, _verify_backend_identity)
    assert _resolve_animation_device('auto', False, None, 'low-vram', 'cpu') == 'cpu'
    assert _resolve_animation_device('auto', True, None, 'low-vram', 'cpu') == 'cuda'
    assert '--cpu' in _backend_preset_flags('low-vram', 'q3', device='cpu')
    device = {'type': 'cpu', 'name': 'cpu'}
    assert _require_backend_device({'devices': [device]}, 'cpu') == device
    args = ['main.py', '--port', '4321', '--user-directory', 'user',
            *_backend_preset_flags('low-vram', 'q3', device='cpu')]
    _verify_backend_identity({'system': {'argv': args}}, 4321, 'user', device='cpu')
    with pytest.raises(RuntimeError, match='identity'):
        _verify_backend_identity({'system': {'argv': args}}, 4321, 'user', device='cuda')
    with pytest.raises(ValueError, match='VRAM'):
        _resolve_animation_device('cpu', False, 2, 'low-vram', 'cpu')
    with pytest.raises(RuntimeError, match='CUDA'):
        _resolve_animation_device('cuda', False, None, 'low-vram', 'cpu')


def test_backend_ram_allocation_reserves_parent_and_launcher_memory():
    from pixcli.comfy_backend import _backend_ram_limit_bytes
    gib = 1024**3
    assert _backend_ram_limit_bytes(2, 300 * 1024**2) == 2*gib - 300*1024**2 - 128*1024**2
    with pytest.raises(MemoryError, match='parent'):
        _backend_ram_limit_bytes(.25, 250 * 1024**2)


def test_mapped_graph_uses_read_only_adapter_nodes():
    args = ('source.png', 'walk', '', 42, 256, 256, 5, 1.0)
    mapped = build_graph(*args, weight_storage='mmap')
    normal = build_graph(*args, weight_storage='normal')
    for expert in ('high', 'low'):
        assert mapped[expert + '_distill']['class_type'] == 'PixcliMappedDistillLoRA'
        assert normal[expert + '_distill']['class_type'] == 'LoraLoaderModelOnly'
        assert mapped[expert + '_pixel']['inputs']['weight_storage'] == 'mmap'
        assert normal[expert + '_pixel']['inputs']['weight_storage'] == 'normal'


def test_command_tree_sampling_includes_parent_and_deduplicates_children():
    from types import SimpleNamespace
    from pixcli.comfy_backend import _command_resident_sample
    class Process:
        def __init__(self, pid, rss, children=()):
            self.pid, self.rss, self._children = pid, rss, children
        def children(self, recursive=False): return list(self._children)
        def memory_info(self): return SimpleNamespace(rss=self.rss)
        def create_time(self): return self.pid + .5
    backend = Process(3, 300)
    launcher = Process(2, 50)
    parent = Process(1, 100, (launcher, backend, backend))
    total, resident, created = _command_resident_sample(parent, 3)
    assert (total, resident, created) == (450, 300, 3.5)


def test_command_ram_supervisor_reserves_parent_memory_and_rechecks(monkeypatch):
    from pixcli import comfy_backend, residency
    mib = 1024**2
    samples = iter([(980*mib, 800*mib, 123.0), (860*mib, 680*mib, 123.0)])
    monkeypatch.setattr(comfy_backend, '_command_resident_sample', lambda parent, pid: next(samples))
    calls = []
    monkeypatch.setattr(residency, 'lower_process_residency_limit',
                        lambda pid, maximum, **kwargs: calls.append((pid, maximum, kwargs)) or maximum)
    report = {'pid': 42, 'hard_limit_applied': True, 'process_budget_gib': 1}
    total, created, limit = comfy_backend._fit_command_residency(None, report, 1, None, None)
    assert total == 860*mib and created == 123.0
    assert limit == (1024 - 180 - 32)*mib
    assert calls == [(42, limit, {'expected_create_time': 123.0})]


def test_command_ram_supervisor_rejects_replaced_backend(monkeypatch):
    from pixcli import comfy_backend
    monkeypatch.setattr(comfy_backend, '_command_resident_sample', lambda parent, pid: (1, 1, 124.0))
    with pytest.raises(RuntimeError, match='identity'):
        comfy_backend._fit_command_residency(None, {'pid': 42, 'hard_limit_applied': True,
                                                   'process_budget_gib': 1}, 1, 123.0, None)


@pytest.mark.skipif(__import__('sys').platform != 'win32', reason='Windows process working-set limits')
def test_command_ram_supervisor_caps_actual_backend_and_counts_parent(tmp_path):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path
    import psutil
    from pixcli.comfy_backend import _fit_command_residency, _process_tree_resident_bytes
    from pixcli.processes import terminate_process_tree

    main = tmp_path / 'main.py'
    main.write_text('import sys\ndata = bytearray(96 * 1024**2)\nprint("ready", flush=True)\nsys.stdin.read()\n')
    parent = psutil.Process()
    initial = _process_tree_resident_bytes(parent)
    budget = (initial + 256 * 1024**2) / 1024**3
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update({'PIXCLI_RAM_BUDGET_GIB': str(budget), 'PIXCLI_RAM_PROCESS_LIMIT_BYTES': str(128 * 1024**2),
                'PYTHONPATH': str(root)})
    env.pop('PIXCLI_VRAM_BUDGET_GIB', None)
    process = subprocess.Popen([sys.executable, str(root / 'pixcli/comfy_launch.py'), str(main)],
                               env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True)
    try:
        report = json.loads(process.stdout.readline().removeprefix('PIXCLI_RAM_BUDGET '))
        assert process.stdout.readline().strip() == 'ready'
        _, created, limit = _fit_command_residency(parent, report, budget, None, None)
        extra = bytearray(96 * 1024**2)
        total, _, lowered = _fit_command_residency(parent, report, budget, created, limit)
        assert lowered < limit
        assert total <= budget * 1024**3
        del extra
    finally:
        terminate_process_tree(process)


def test_history_poll_retries_transient_timeout_but_reports_http_error():
    import urllib.error
    from pixcli.comfy_backend import _history_or_pending
    def timeout(endpoint):
        raise TimeoutError('backend is busy')
    assert _history_or_pending(timeout, '123') is None
    assert _history_or_pending(lambda endpoint: {'123': {'status': 'ok'}}, '123') == {'status': 'ok'}
    def error(endpoint):
        raise urllib.error.HTTPError(endpoint, 500, 'server error', {}, None)
    with pytest.raises(urllib.error.HTTPError):
        _history_or_pending(error, '123')


def test_budget_guard_stops_on_observed_overages_and_missing_samples():
    from pixcli.comfy_backend import _check_sampled_budgets
    gib = 1024**3
    _check_sampled_budgets(3*gib, 2*gib, 4, 3)
    with pytest.raises(MemoryError, match='RAM'):
        _check_sampled_budgets(5*gib, 2*gib, 4, 3)
    with pytest.raises(MemoryError, match='VRAM'):
        _check_sampled_budgets(3*gib, 4*gib, 4, 3)
    with pytest.raises(RuntimeError, match='VRAM.*unavailable'):
        _check_sampled_budgets(3*gib, None, 4, 3)

def test_log_line_chunks_preserve_split_telemetry():
    from pixcli.comfy_backend import _complete_log_lines
    lines, pending = _complete_log_lines('', 'PIXCLI_MEMORY_')
    assert lines == [] and pending == 'PIXCLI_MEMORY_'
    lines, pending = _complete_log_lines(pending, 'PHASE {"phase":"after_load"}\npartial')
    assert lines == ['PIXCLI_MEMORY_PHASE {"phase":"after_load"}']
    assert pending == 'partial'

def test_backend_rejects_silently_failed_lora_patch():
    from pixcli.comfy_backend import _assert_lora_log_clean
    _assert_lora_log_clean('[INFO] sampled all layers\n')
    with pytest.raises(RuntimeError, match='failed to apply.*LoRA'):
        _assert_lora_log_clean('[ERROR] ERROR lora diffusion_model.blocks.0.attn.weight CUDA out of memory\n')
    with pytest.raises(RuntimeError, match='unmatched LoRA'):
        _assert_lora_log_clean('[WARNING] lora key not loaded: missing\n')

def test_cpu_device_memory_scope_does_not_claim_gpu_sampling():
    from pixcli.comfy_backend import _device_memory_scope
    assert _device_memory_scope('cpu', None) is None
    assert _device_memory_scope('cuda', None) is None
    assert 'device-wide' in _device_memory_scope('cuda', 123)

def test_q2_graph_uses_both_experts_and_retains_adapter_and_routing():
    graph = build_graph('source.png', 'walk', '', 42, 256, 256, 5, 1.0, expert_quant='q2', text_encoder='q3')
    assert graph['high_model']['inputs']['unet_name'] == 'wan2.2_i2v_high_noise_14B_Q2_K.gguf'
    assert graph['low_model']['inputs']['unet_name'] == 'wan2.2_i2v_low_noise_14B_Q2_K.gguf'
    assert graph['high_pixel']['inputs']['expert'] == 'high'
    assert graph['low_pixel']['inputs']['expert'] == 'low'
    assert graph['text']['class_type'] == 'CLIPLoaderGGUF'
    assert graph['high_sample']['inputs']['end_at_step'] == graph['low_sample']['inputs']['start_at_step'] == 2
    with pytest.raises(ValueError, match='expert'):
        build_graph('source.png', 'walk', '', 42, 256, 256, 5, 1.0, expert_quant='q4')

def test_q2_model_locations_and_missing_error_use_requested_profile(tmp_path):
    from pixcli.comfy_backend import validate_model_locations, run_comfy
    names = {
        'wan_high_q2': 'wan2.2_i2v_high_noise_14B_Q2_K.gguf',
        'wan_low_q2': 'wan2.2_i2v_low_noise_14B_Q2_K.gguf',
        'wan_text_q3': 'umt5-xxl-encoder-Q3_K_S.gguf',
        'wan_vae': 'wan_2.1_vae.safetensors',
        'wan_distill_high': 'wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors',
        'wan_distill_low': 'wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors',
        'wan_pixel': 'wan2.2_animate_adapter_model.safetensors',
    }
    paths = {key: tmp_path/name for key,name in names.items()}
    for path in paths.values():
        path.touch()
    validate_model_locations(paths, text_encoder='q3', expert_quant='q2')
    paths['wan_low_q2'].unlink()
    with pytest.raises(FileNotFoundError, match='-profile animation'):
        run_comfy(tmp_path/'source.png', 'walk', '', 42, 256, 256, 5, 1.0, paths, text_encoder='q3', expert_quant='q2')
