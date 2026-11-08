"""Start the installed local ComfyUI backend without models and check required nodes."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import psutil
from pixcli.processes import terminate_process_tree
from pixcli.comfy_backend import _backend_preset_flags, _local_opener, _verify_backend_identity, build_graph

root = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', action='store_true', help='Start with the low-VRAM CUDA configuration')
parser.add_argument('--preset', choices=('low-vram', 'gpu-resident'), default='low-vram', help='GPU memory preset to validate')
parser.add_argument('--text-encoder', choices=('fp8', 'q3'), default='fp8', help='Select the graph text encoder to check')
parser.add_argument('--expert-quant', choices=('q2', 'q3'), default='q3', help='Select the expert quantization to check')
parser.add_argument('--adapter-device', choices=('cpu', 'cuda'), default='cpu', help='Select the pixel adapter device to check')
parser.add_argument('--weight-storage', choices=('normal', 'mmap'), default='normal', help='Select expert weight storage to check')
args = parser.parse_args()
comfy = root / '.backend' / 'ComfyUI'
python = root / '.backend' / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
if not python.is_file() or not (root / '.backend' / 'ready.json').is_file():
    raise SystemExit('Backend setup is incomplete. Run setup.ps1.')
with tempfile.TemporaryDirectory(prefix='pixcli-startup-') as temp:
    temp = Path(temp)
    for name in ('models', 'user', 'input', 'output', 'temp'):
        (temp / name).mkdir()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    command = [str(python), '-s', str(comfy / 'main.py'), '--listen', '127.0.0.1', '--port', str(port),
            *(_backend_preset_flags(args.preset, args.text_encoder) if args.gpu else ['--cpu']), '--offline', '--disable-auto-launch', '--disable-all-custom-nodes',
            '--whitelist-custom-nodes', 'ComfyUI-GGUF', 'ComfyUI-PainterI2V', 'pixcli_wan',
            '--models-directory', str(temp / 'models'), '--user-directory', str(temp / 'user'),
            '--input-directory', str(temp / 'input'), '--output-directory', str(temp / 'output'),
            '--temp-directory', str(temp / 'temp')]
    env = os.environ.copy()
    env.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTHONPATH': str(root)})
    opener = _local_opener()
    log_path = temp / 'startup.log'
    with log_path.open('w+', encoding='utf-8') as log:
        proc = subprocess.Popen(command, cwd=comfy, env=env, stdout=log, stderr=subprocess.STDOUT,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        process = psutil.Process(proc.pid)
        known_children = set()
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                try:
                    known_children.update(process.children(recursive=True))
                except psutil.Error:
                    pass
                if proc.poll() is not None:
                    raise RuntimeError(f'Backend exited with code {proc.returncode}')
                try:
                    with opener.open(f'http://127.0.0.1:{port}/object_info', timeout=2) as response:
                        nodes = json.load(response)
                    break
                except (OSError, urllib.error.URLError):
                    time.sleep(0.5)
            else:
                raise TimeoutError('Backend did not start within 120 seconds')
            with opener.open(f'http://127.0.0.1:{port}/system_stats', timeout=2) as response:
                stats = json.load(response)
            if args.gpu:
                _verify_backend_identity(stats, port, str(temp / 'user'), args.preset)
            else:
                pairs = list(zip(stats['system']['argv'], stats['system']['argv'][1:]))
                if ('--port', str(port)) not in pairs or ('--user-directory', str(temp / 'user')) not in pairs or '--cpu' not in stats['system']['argv']:
                    raise RuntimeError('Local backend identity does not match the CPU test process')
            if args.gpu and not any(device.get('type') == 'cuda' for device in stats.get('devices', [])):
                raise RuntimeError('Backend started without a CUDA device')
            required = {'UnetLoaderGGUF', 'PainterI2V', 'PixcliCheckedPixelLoRA'}
            if args.weight_storage == 'mmap':
                required.add('PixcliMappedGGUFLoader')
            if args.text_encoder == 'q3':
                required.add('CLIPLoaderGGUF')
            missing = required - nodes.keys()
            if missing:
                raise RuntimeError(f'Required nodes missing: {sorted(missing)}')
            graph = build_graph('source.png', 'test motion', '', 42, 512, 512, 9, 1.0, text_encoder=args.text_encoder, expert_quant=args.expert_quant, adapter_device=args.adapter_device, weight_storage=args.weight_storage)
            for name, spec in graph.items():
                schema = nodes.get(spec['class_type'])
                if schema is None:
                    raise RuntimeError(f"Graph node {name} has unavailable type {spec['class_type']}")
                inputs = schema['input']
                required_inputs = set(inputs.get('required', {}))
                allowed = required_inputs | set(inputs.get('optional', {}))
                present = set(spec['inputs'])
                if required_inputs - present or present - allowed:
                    raise RuntimeError(f'Graph node {name}: missing {sorted(required_inputs-present)}, unexpected {sorted(present-allowed)}')
                for input_name, value in spec['inputs'].items():
                    definition = (inputs.get('required', {}) | inputs.get('optional', {}))[input_name][0]
                    if isinstance(value, list) and len(value) == 2 and value[0] in graph and isinstance(value[1], int):
                        source = nodes[graph[value[0]]['class_type']]['output']
                        if value[1] >= len(source):
                            raise RuntimeError(f'Graph node {name}: invalid output index for {input_name}')
                        expected_type = definition
                        if source[value[1]] != expected_type:
                            raise RuntimeError(f'Graph node {name}: {input_name} expects {expected_type}, got {source[value[1]]}')
                    elif input_name not in {'image', 'vae_name', 'clip_name', 'unet_name', 'lora_name'} and isinstance(definition, list) and definition and value not in definition:
                        raise RuntimeError(f'Graph node {name}: {input_name} value {value!r} is unavailable')
            print('Local ComfyUI started; required nodes and graph input names are registered.')
            if args.gpu:
                print('CUDA device:', next(d['name'] for d in stats['devices'] if d.get('type') == 'cuda'))
        except Exception:
            log.flush()
            print(log_path.read_text(encoding='utf-8', errors='replace')[-6000:], file=sys.stderr)
            raise
        finally:
            terminate_process_tree(proc, tracked=known_children)

