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
    with pytest.raises(FileNotFoundError, match='models download --profile animation'):
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
    import subprocess
    from pixcli.comfy_backend import _gpu_memory_used
    monkeypatch.setattr(subprocess, 'check_output', lambda *a, **kw: '1024\n')
    assert _gpu_memory_used() == 1024 * 1024 * 1024
    def missing(*args, **kwargs):
        raise FileNotFoundError('nvidia-smi')
    monkeypatch.setattr(subprocess, 'check_output', missing)
    assert _gpu_memory_used() is None

def test_cuda_oom_reports_actual_configuration_and_no_fallback():
    from pixcli.comfy_backend import _oom_explanation
    text = _oom_explanation('torch.OutOfMemoryError: CUDA out of memory', 512, 512, 9)
    assert '512x512' in text and '9 frames' in text
    assert '--inference-width 384' in text and '--frames 5' in text
    assert 'no automatic fallback' in text.lower()
    assert _oom_explanation('some unrelated failure', 512, 512, 9) is None

def test_cuda_oom_advice_never_increases_canvas_size():
    from pixcli.comfy_backend import _oom_explanation
    text = _oom_explanation('CUDA out of memory', 256, 256, 5)
    assert '--inference-width 384' not in text

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

def test_log_line_chunks_preserve_split_telemetry():
    from pixcli.comfy_backend import _complete_log_lines
    lines, pending = _complete_log_lines('', 'PIXCLI_MEMORY_')
    assert lines == [] and pending == 'PIXCLI_MEMORY_'
    lines, pending = _complete_log_lines(pending, 'PHASE {"phase":"after_load"}\npartial')
    assert lines == ['PIXCLI_MEMORY_PHASE {"phase":"after_load"}']
    assert pending == 'partial'

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
    with pytest.raises(FileNotFoundError, match='--profile animation'):
        run_comfy(tmp_path/'source.png', 'walk', '', 42, 256, 256, 5, 1.0, paths, text_encoder='q3', expert_quant='q2')
