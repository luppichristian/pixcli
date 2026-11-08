"""Best-effort process memory telemetry emitted by the checked Wan LoRA node."""
from __future__ import annotations

import json


def report_memory_phase(expert: str, phase: str, loaded_models) -> None:
    """Report the backend process and registered live model classes to its log."""
    try:
        import psutil
        from .comfy_backend import _free_commit_bytes
        process = psutil.Process()
        memory = process.memory_info()
        models = []
        for loaded in list(loaded_models):
            patcher = loaded.model
            if patcher is None:
                continue
            model = getattr(patcher, 'model', None)
            if model is None:
                continue
            models.append({'class': type(model).__name__,
                           'is_text_encoder': bool(getattr(patcher, 'is_clip', False)),
                           'device': str(getattr(patcher, 'load_device', 'unknown'))})
        event = {'expert': expert, 'phase': phase,
                 'private_bytes': getattr(memory, 'private', memory.rss),
                 'rss_bytes': memory.rss,
                 'physical_available_bytes': psutil.virtual_memory().available,
                 'commit_available_bytes': _free_commit_bytes(),
                 'loaded_models': models}
    except Exception as exc:
        event = {'expert': expert, 'phase': phase, 'telemetry_error': str(exc)}
    print('PIXCLI_MEMORY_PHASE ' + json.dumps(event), flush=True)
