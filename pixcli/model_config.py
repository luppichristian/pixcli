"""User model catalog and per-operation defaults.

Configuration is TOML version 1.  Local paths are resolved relative to the
configuration file; loading and selecting models never access the network.
"""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path, PurePosixPath
import re
import tomllib
from typing import Any


class ConfigError(ValueError):
    """An invalid model configuration or selection."""


BUILTINS = {
    'default-image': {'backend': 'builtin-image'},
    'default-animation': {'backend': 'builtin-animation'},
}
DEFAULTS = {'generation': 'default-image', 'editing': 'default-image', 'animation': 'default-animation'}
_MODEL_NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]*$')


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f'{where} must be a TOML table.')
    return value


def _fields(value: dict[str, Any], allowed: set[str], where: str) -> None:
    unexpected = set(value) - allowed
    if unexpected:
        raise ConfigError(f'{where} has unknown fields: {", ".join(sorted(unexpected))}.')


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f'{where} must be a nonempty string.')
    return value


def _relative_filename(value: Any, where: str) -> str:
    value = _string(value, where)
    path = PurePosixPath(value)
    if '\\' in value or ':' in value or path.is_absolute() or any(p in ('.', '..') for p in value.split('/')):
        raise ConfigError(f'{where} must be a safe relative filename.')
    return value


def _source(value: Any, where: str, root: Path, *, component: bool) -> dict[str, Any]:
    source = _mapping(value, where)
    kind = source.get('type')
    if kind == 'local':
        _fields(source, {'type', 'path'}, where)
        path = Path(_string(source.get('path'), f'{where}.path')).expanduser()
        if not path.is_absolute():
            path = root / path
        return {'type': 'local', 'path': str(path.resolve())}
    if kind == 'huggingface':
        allowed = {'type', 'repo_id', 'revision', 'filename' if component else 'allow_patterns'}
        _fields(source, allowed, where)
        repo = _string(source.get('repo_id'), f'{where}.repo_id')
        from huggingface_hub.utils import HFValidationError, validate_repo_id
        try:
            validate_repo_id(repo)
        except HFValidationError as exc:
            raise ConfigError(f'{where}.repo_id is invalid: {exc}') from exc
        revision = _string(source.get('revision'), f'{where}.revision')
        result = {'type': kind, 'repo_id': repo, 'revision': revision}
        if component:
            result['filename'] = _relative_filename(source.get('filename'), f'{where}.filename')
        elif 'allow_patterns' in source:
            patterns = source['allow_patterns']
            if not isinstance(patterns, list) or not patterns:
                raise ConfigError(f'{where}.allow_patterns must be a nonempty array of patterns.')
            result['allow_patterns'] = [_relative_filename(pattern, f'{where}.allow_patterns') for pattern in patterns]
        return result
    raise ConfigError(f'{where}.type must be local or huggingface.')


def _definition(name: str, value: Any, root: Path) -> dict[str, Any]:
    where = f'models.{name}'
    model = _mapping(value, where)
    backend = model.get('backend')
    if backend == 'diffusers':
        _fields(model, {'backend', 'source', 'variant'}, where)
        result = {'backend': backend, 'source': _source(model.get('source'), f'{where}.source', root, component=False)}
        if 'variant' in model:
            result['variant'] = _string(model['variant'], f'{where}.variant')
        return result
    if backend == 'wan-i2v':
        _fields(model, {'backend', 'text_encoder', 'expert_quant', 'components'}, where)
        text = model.get('text_encoder', 'q3')
        quant = model.get('expert_quant', 'q2')
        if text not in ('q3', 'fp8'):
            raise ConfigError(f'{where}.text_encoder must be q3 or fp8.')
        if quant not in ('q2', 'q3'):
            raise ConfigError(f'{where}.expert_quant must be q2 or q3.')
        from . import models
        profile = models.animation_profile(text, quant)
        allowed = set(models.list_models(profile))
        components = _mapping(model.get('components', {}), f'{where}.components')
        unknown = set(components) - allowed
        if unknown:
            raise ConfigError(f'{where}.components contains unsupported roles for {profile}: {", ".join(sorted(unknown))}.')
        return {'backend': backend, 'text_encoder': text, 'expert_quant': quant,
                'components': {key: _source(value, f'{where}.components.{key}', root, component=True)
                               for key, value in components.items()}}
    raise ConfigError(f'{where}.backend must be diffusers or wan-i2v.')


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Read an explicit config, PIXCLI_CONFIG, or ./pixcli.toml, in that order."""
    explicit = path is not None or bool(os.environ.get('PIXCLI_CONFIG'))
    selected = Path(path if path is not None else os.environ.get('PIXCLI_CONFIG', 'pixcli.toml')).expanduser()
    selected = selected.resolve()
    if not selected.is_file():
        if explicit:
            raise ConfigError(f'Model configuration does not exist: {selected}.')
        return {'version': 1, 'defaults': dict(DEFAULTS), 'models': deepcopy(BUILTINS), 'config_path': None}
    try:
        with selected.open('rb') as source:
            data = tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f'Cannot read model configuration {selected}: {exc}') from exc
    _fields(data, {'version', 'defaults', 'models'}, 'configuration')
    if data.get('version') != 1 or isinstance(data.get('version'), bool):
        raise ConfigError(f'{selected}: model configuration version must be 1.')
    defaults = dict(DEFAULTS)
    configured_defaults = _mapping(data.get('defaults', {}), 'defaults')
    _fields(configured_defaults, set(DEFAULTS), 'defaults')
    defaults.update({key: _string(value, f'defaults.{key}') for key, value in configured_defaults.items()})
    definitions = deepcopy(BUILTINS)
    for name, value in _mapping(data.get('models', {}), 'models').items():
        if name in BUILTINS:
            raise ConfigError(f'Model name {name!r} is reserved for a built-in model.')
        if not _MODEL_NAME.fullmatch(name):
            raise ConfigError(f'Invalid model name {name!r}; use letters, numbers, dot, dash or underscore.')
        definitions[name] = _definition(name, value, selected.parent)
    for operation, name in defaults.items():
        if name not in definitions:
            raise ConfigError(f'defaults.{operation} names unknown model {name!r}.')
    return {'version': 1, 'defaults': defaults, 'models': definitions, 'config_path': str(selected)}


def select_model(config: dict[str, Any], operation: str, name: str | None = None) -> dict[str, Any]:
    """Select a definition compatible with generation, editing or animation."""
    if operation not in DEFAULTS:
        raise ConfigError(f'Unknown operation {operation!r}; choose generation, editing, or animation.')
    name = name if name is not None else config['defaults'][operation]
    if name not in config['models']:
        raise ConfigError(f'Unknown model {name!r}.')
    result = deepcopy(config['models'][name])
    allowed = {'builtin-animation', 'wan-i2v'} if operation == 'animation' else {'builtin-image', 'diffusers'}
    if result.get('backend') not in allowed:
        raise ConfigError(f'Model {name!r} with backend {result.get("backend")!r} is incompatible with {operation}.')
    result['name'] = name
    return result
