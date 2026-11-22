"""Offline model resolution and a pixcli-owned installation store.

The legacy Hub cache remains readable but is never deleted by this module.
Managed installations live under a fingerprint of the model definition.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from typing import Any

from . import models


def _fingerprint(selection: dict[str, Any]) -> str:
    normalized = dict(selection)
    if normalized.get('backend') == 'builtin-animation':
        normalized.setdefault('text_encoder', 'q3')
        normalized.setdefault('expert_quant', 'q2')
    data = json.dumps(normalized, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def _root(cache_dir: str | Path | None) -> Path:
    cache = models.cache_location(cache_dir)
    root = cache / 'pixcli-managed'
    # Disallow a redirected store root or entry, including Windows junctions.
    if root.is_symlink() or root.resolve() != root:
        raise models.ModelError(f'Unsafe managed model store: {root}.')
    return root


def _entry(root: Path, identifier: str) -> Path:
    if not isinstance(identifier, str) or not re.fullmatch(r'[0-9a-f]{64}', identifier):
        raise models.ModelError('Invalid managed model installation ID.')
    entry = root / identifier
    if entry.is_symlink() or entry.resolve() != entry:
        raise models.ModelError(f'Unsafe managed model directory: {entry}.')
    return entry


def _locations(selection: dict[str, Any], cache_dir: str | Path | None) -> tuple[Path, Path]:
    root = _root(cache_dir)
    return root, _entry(root, _fingerprint(selection))


@contextmanager
def _lock(root: Path, entry: Path):
    root.mkdir(parents=True, exist_ok=True)
    lock = root / (entry.name + '.lock')
    stream = lock.open('a+b')
    try:
        if os.name == 'nt':
            import msvcrt
            stream.seek(0, os.SEEK_END)
            if not stream.tell():
                stream.write(b'0')
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise models.ModelError(f'Another operation is using model {entry.name}; retry after it finishes.') from exc
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise models.ModelError(f'Another operation is using model {entry.name}; retry after it finishes.') from exc
        yield
    finally:
        stream.close()  # OS releases the lock even if the process exits unexpectedly.


def _safe_path(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value or '\\' in value or ':' in value:
        raise models.ModelError('Invalid or unsafe managed model path in installation record.')
    parts = value.split('/')
    if any(part in ('', '.', '..') for part in parts) or PurePosixPath(value).is_absolute():
        raise models.ModelError('Invalid or unsafe managed model path in installation record.')
    path = root.joinpath(*parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise models.ModelError('Invalid or unsafe managed model path in installation record.')
    return path


def _read_marker(entry: Path, selection: dict[str, Any]) -> dict[str, Any] | None:
    path = entry / 'installed.json'
    if not path.is_file():
        return None
    if path.is_symlink() or path.resolve() != path:
        raise models.ModelError(f'Unsafe model installation record at {path}.')
    try:
        record = json.loads(path.read_text(encoding='utf-8'))
        if (not isinstance(record, dict) or record.get('version') != 1 or
                record.get('fingerprint') != _fingerprint(selection) or
                not isinstance(record.get('sources'), dict)):
            raise ValueError('mismatched fingerprint or schema')
        for source in record['sources'].values():
            if not isinstance(source, dict):
                raise ValueError('invalid source')
            if source.get('type') == 'huggingface':
                _safe_path(entry, source['path'])
                if not isinstance(source.get('files'), list) or not source['files']:
                    raise ValueError('missing file inventory')
                for item in source['files']:
                    _safe_path(entry, item['path'])
                    if not isinstance(item['size'], int) or not isinstance(item['sha256'], str):
                        raise ValueError('invalid file inventory')
        return record
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise models.ModelError(f'Invalid model installation record at {path}: {exc}. Reinstall the model.') from exc


def _write_marker(entry: Path, selection: dict[str, Any], sources: dict[str, Any]) -> None:
    record = {'version': 1, 'fingerprint': _fingerprint(selection), 'name': selection['name'],
              'backend': selection['backend'], 'sources': sources}
    fd, temporary = tempfile.mkstemp(prefix='.marker-', suffix='.tmp', dir=entry)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as output:
            json.dump(record, output, indent=2, ensure_ascii=False)
            output.write('\n')
        os.replace(temporary, entry / 'installed.json')
    finally:
        Path(temporary).unlink(missing_ok=True)


def _write_installing_marker(entry: Path, selection: dict[str, Any]) -> None:
    """Identify an incomplete download so it can be inspected and removed."""
    record = {'version': 1, 'fingerprint': _fingerprint(selection), 'name': selection['name'],
              'backend': selection['backend']}
    path = entry / 'installing.json'
    fd, temporary = tempfile.mkstemp(prefix='.installing-', suffix='.tmp', dir=entry)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as output:
            json.dump(record, output, ensure_ascii=False)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _profile(selection: dict[str, Any]) -> str:
    if selection['backend'] == 'builtin-image':
        return 'image'
    return models.animation_profile(selection.get('text_encoder', 'q3'), selection.get('expert_quant', 'q2'))


def _local_path(source: dict[str, Any], key: str, *, directory: bool) -> Path:
    path = Path(source['path'])
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        kind = 'directory' if directory else 'file'
        raise models.ModelError(f'Local model {key} {kind} does not exist: {path}.')
    if directory and not (path / 'model_index.json').is_file():
        raise models.ModelError(f'Local Diffusers model {path} has no model_index.json.')
    if directory and not any(item.is_file() for item in path.rglob('*.safetensors')):
        raise models.ModelError(f'Local Diffusers model {path} has no safetensors weights.')
    return path


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _inventory(entry: Path, path: Path, *, directory: bool) -> list[dict[str, Any]]:
    files = sorted(path.rglob('*')) if directory else [path]
    result = []
    for file in files:
        relative = file.relative_to(entry)
        if '.cache' in relative.parts:
            continue
        if file.is_symlink() or not file.resolve().is_relative_to(entry.resolve()):
            raise models.ModelError(f'Unsafe model file: {file}.')
        if file.is_file():
            result.append({'path': relative.as_posix(), 'size': file.stat().st_size, 'sha256': _digest(file)})
    if not result:
        raise models.ModelError(f'Model download contained no files: {path}.')
    return result


def _source_path(entry: Path, source: dict[str, Any], key: str, *, directory: bool) -> Path:
    try:
        path = _safe_path(entry, source['path'])
        if not (path.is_dir() if directory else path.is_file()):
            raise ValueError('missing model payload')
        for item in source['files']:
            file = _safe_path(entry, item['path'])
            if not file.is_file() or file.stat().st_size != item['size']:
                raise ValueError(f'missing or incomplete file {item["path"]}')
        return path
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise models.ModelError(f'Incomplete installation of {key}: {exc}; reinstall the model.') from exc


def _pinned_provenance(key: str, data: dict[str, Any], storage: str) -> dict[str, Any]:
    return {'type': 'pinned', 'storage': storage, 'repo_id': data['repo_id'], 'revision': data['revision'],
            'files': [item['filename'] for item in data['files']]}


def resolve(selection: dict[str, Any], cache_dir: str | Path | None = None) -> tuple[dict[str, Path], dict[str, Any]]:
    """Resolve the chosen model strictly offline, returning paths and provenance."""
    backend = selection['backend']
    _, entry = _locations(selection, cache_dir)
    marker = _read_marker(entry, selection)
    paths: dict[str, Path] = {}
    sources: dict[str, Any] = {}
    if backend == 'diffusers':
        source = selection['source']
        if source['type'] == 'local':
            paths['base'] = _local_path(source, 'base', directory=True)
            sources['base'] = {**source, 'storage': 'external'}
        else:
            saved = marker and marker['sources'].get('base')
            if not saved:
                raise models.ModelError(f'Model {selection["name"]!r} is not installed; run pixcli models install {selection["name"]}.')
            paths['base'] = _source_path(entry, saved, 'base', directory=True)
            sources['base'] = {key: value for key, value in saved.items() if key not in ('path', 'files')}
            sources['base']['storage'] = 'managed'
        return paths, {'name': selection['name'], 'backend': backend, 'sources': sources}
    if backend not in ('builtin-image', 'builtin-animation', 'wan-i2v'):
        raise models.ModelError(f'Unsupported model backend {backend!r}.')
    profile = _profile(selection)
    pinned = models.list_models(profile)
    overrides = selection.get('components', {}) if backend == 'wan-i2v' else {}
    if set(overrides) - set(pinned):
        raise models.ModelError(f'Configured animation component roles do not match profile {profile}.')
    managed_cache = entry / 'hub'
    for key, data in pinned.items():
        source = overrides.get(key)
        if source and source['type'] == 'local':
            paths[key] = _local_path(source, key, directory=False)
            sources[key] = {**source, 'storage': 'external'}
        elif source:
            saved = marker and marker['sources'].get(key)
            if not saved:
                raise models.ModelError(f'Model {selection["name"]!r} is not installed; run pixcli models install {selection["name"]}.')
            paths[key] = _source_path(entry, saved, key, directory=False)
            sources[key] = {name: value for name, value in saved.items() if name not in ('path', 'files')}
            sources[key]['storage'] = 'managed'
        else:
            storage = 'shared'
            if marker:
                try:
                    paths[key] = (models.model_path(key, managed_cache) if backend == 'builtin-image'
                                  else models.model_path(key, managed_cache, profile=profile))
                    storage = 'managed'
                except models.ModelError:
                    paths[key] = (models.model_path(key, cache_dir) if backend == 'builtin-image'
                                  else models.model_path(key, cache_dir, profile=profile))
            else:
                paths[key] = (models.model_path(key, cache_dir) if backend == 'builtin-image'
                              else models.model_path(key, cache_dir, profile=profile))
            sources[key] = _pinned_provenance(key, data, storage)
    return paths, {'name': selection['name'], 'backend': backend, 'profile': profile, 'sources': sources}


def _download_hub_source(source: dict[str, Any], entry: Path, key: str, *, directory: bool) -> dict[str, Any]:
    hub = models._hub()
    try:
        revision = hub.HfApi().model_info(source['repo_id'], revision=source['revision']).sha
        if not isinstance(revision, str) or len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision.lower()):
            raise models.ModelError('Hugging Face did not return an immutable commit revision.')
        dest = entry / 'payload' / key
        dest.mkdir(parents=True, exist_ok=True)
        if directory:
            hub.snapshot_download(repo_id=source['repo_id'], revision=revision, local_dir=str(dest),
                                  allow_patterns=source.get('allow_patterns'))
            path = dest
            if not (path / 'model_index.json').is_file():
                raise models.ModelError('Downloaded Diffusers model has no model_index.json; check allow_patterns and model format.')
            if not any(item.is_file() for item in path.rglob('*.safetensors')):
                raise models.ModelError('Downloaded Diffusers model has no safetensors weights; check allow_patterns and model format.')
        else:
            filename = source['filename']
            hub.hf_hub_download(repo_id=source['repo_id'], revision=revision, filename=filename, local_dir=str(dest))
            path = _safe_path(dest, filename)
            if not path.is_file():
                raise models.ModelError(f'Download did not produce {filename}.')
        files = _inventory(entry, path, directory=directory)
    except models.ModelError:
        raise
    except Exception as exc:
        raise models.ModelError(f'Download failed for {source["repo_id"]}: {exc}. Check connectivity, login and model terms; rerun install.') from exc
    record = {'type': 'huggingface', 'repo_id': source['repo_id'], 'requested_revision': source['revision'],
              'resolved_revision': revision, 'path': path.relative_to(entry).as_posix(), 'files': files}
    if not directory:
        record['filename'] = source['filename']
    return record


def _install_pinned(profile: str, keys: set[str], cache: Path) -> None:
    if not keys:
        return
    cache.mkdir(parents=True, exist_ok=True)
    pending = []
    for key, data in models.list_models(profile).items():
        if key not in keys:
            continue
        for item in data['files']:
            path = models._cached(data, item['filename'], cache)
            if path is None or not models._digest(path, item):
                pending.append((key, data, item, path is not None))
    total = sum(item['size'] for _, _, item, _ in pending)
    reserve = total + max((item['size'] for _, _, item, _ in pending), default=0) + (1024 ** 3 if pending else 0)
    if shutil.disk_usage(cache).free < reserve:
        raise models.ModelError('Insufficient disk space in the managed model store.')
    if not pending:
        return
    hub = models._hub()
    for key, data, item, corrupt in pending:
        try:
            hub.hf_hub_download(repo_id=data['repo_id'], filename=item['filename'], revision=data['revision'],
                                cache_dir=str(cache), force_download=corrupt)
            path = models._cached(data, item['filename'], cache)
            if path is None or not models._digest(path, item):
                raise models.ModelError(f'Pinned model verification failed for {key}: {item["filename"]}.')
        except models.ModelError:
            raise
        except Exception as exc:
            raise models.ModelError(f'Download failed for {key}: {exc}. Rerun install to resume.') from exc


def install(selection: dict[str, Any], cache_dir: str | Path | None = None) -> dict[str, Any]:
    """Install a model to the isolated store, or validate a local model reference."""
    backend = selection['backend']
    if backend == 'diffusers' and selection['source']['type'] == 'local':
        _local_path(selection['source'], 'base', directory=True)
        return status(selection, cache_dir)
    if backend == 'wan-i2v':
        for key, source in selection.get('components', {}).items():
            if source['type'] == 'local':
                _local_path(source, key, directory=False)
    root, entry = _locations(selection, cache_dir)
    with _lock(root, entry):
        marker = _read_marker(entry, selection)
        if marker is not None:
            current = status(selection, cache_dir)
            if current['installed'] and all(item['ok'] for item in verify(selection, cache_dir)):
                return current

        shared_keys: set[str] = set()
        if backend == 'wan-i2v':
            overrides = selection.get('components', {})
            profile = _profile(selection)
            for key, data in models.list_models(profile).items():
                if key in overrides:
                    continue
                try:
                    cached = [models._cached(data, item['filename'], cache_dir) for item in data['files']]
                    if all(path is not None and models._digest(path, item)
                           for path, item in zip(cached, data['files'])):
                        shared_keys.add(key)
                except OSError:
                    pass
            if len(shared_keys) == len(models.list_models(profile)) - len(overrides):
                if all(source['type'] == 'local' for source in overrides.values()):
                    return status(selection, cache_dir)

        entry.mkdir(parents=True, exist_ok=True)
        _write_installing_marker(entry, selection)
        # A failed install has no completion marker. Pinned Hub downloads can
        # resume in place; remote payloads are rebuilt to avoid stale branch files.
        (entry / 'installed.json').unlink(missing_ok=True)
        sources: dict[str, Any] = {}
        if backend in ('builtin-image', 'builtin-animation'):
            models.download(_profile(selection), entry / 'hub')
        elif backend == 'diffusers':
            payload = entry / 'payload'
            if payload.exists():
                shutil.rmtree(payload)
            sources['base'] = _download_hub_source(selection['source'], entry, 'base', directory=True)
        elif backend == 'wan-i2v':
            overrides = selection.get('components', {})
            profile = _profile(selection)
            _install_pinned(profile, set(models.list_models(profile)) - set(overrides) - shared_keys, entry / 'hub')
            payload = entry / 'payload'
            if payload.exists():
                shutil.rmtree(payload)
            for key, source in overrides.items():
                if source['type'] == 'huggingface':
                    sources[key] = _download_hub_source(source, entry, key, directory=False)
        else:
            raise models.ModelError(f'Unsupported model backend {backend!r}.')
        _write_marker(entry, selection, sources)
        (entry / 'installing.json').unlink(missing_ok=True)
    return status(selection, cache_dir)


def status(selection: dict[str, Any], cache_dir: str | Path | None = None) -> dict[str, Any]:
    """Report availability without downloading or performing a full hash check."""
    _, entry = _locations(selection, cache_dir)
    try:
        paths, provenance = resolve(selection, cache_dir)
        marker = _read_marker(entry, selection)
        return {'name': selection['name'], 'backend': selection['backend'], 'installed': True,
                'managed': marker is not None, 'paths': {key: str(path) for key, path in paths.items()},
                'provenance': provenance}
    except (models.ModelError, OSError) as exc:
        return {'name': selection['name'], 'backend': selection['backend'], 'installed': False,
                'managed': False, 'detail': str(exc)}


def verify(selection: dict[str, Any], cache_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Hash-check recorded downloads and pinned files; check local path existence."""
    _, entry = _locations(selection, cache_dir)
    try:
        paths, _ = resolve(selection, cache_dir)
        marker = _read_marker(entry, selection)
    except (models.ModelError, OSError) as exc:
        return [{'key': selection['name'], 'filename': '', 'ok': False, 'detail': str(exc)}]
    results = []
    backend = selection['backend']
    profile = _profile(selection) if backend != 'diffusers' else None
    pinned = models.list_models(profile) if profile else {}
    overrides = selection.get('components', {}) if backend == 'wan-i2v' else {}
    for key, path in paths.items():
        source = selection['source'] if backend == 'diffusers' else overrides.get(key)
        if source and source['type'] == 'local':
            results.append({'key': key, 'filename': str(path), 'ok': True, 'detail': 'exists; external file not hash-verified'})
        elif source and source['type'] == 'huggingface':
            saved = marker['sources'][key]
            for item in saved['files']:
                try:
                    file = _safe_path(entry, item['path'])
                    ok = file.is_file() and file.stat().st_size == item['size'] and _digest(file) == item['sha256']
                    detail = 'verified against installation hash' if ok else 'missing, hash or size mismatch'
                except OSError as exc:
                    ok, detail = False, str(exc)
                results.append({'key': key, 'filename': item['path'], 'ok': ok, 'detail': detail})
        else:
            data = pinned[key]
            for item in data['files']:
                try:
                    managed_cache = entry / 'hub'
                    file = models._cached(data, item['filename'], managed_cache) if marker else None
                    if file is None:
                        file = models._cached(data, item['filename'], cache_dir)
                    ok = file is not None and models._digest(file, item)
                    detail = 'verified' if ok else ('missing' if file is None else 'hash or size mismatch')
                except OSError as exc:
                    ok, detail = False, str(exc)
                results.append({'key': key, 'filename': item['filename'], 'ok': ok, 'detail': detail})
    return results


def _owned_record(entry: Path) -> tuple[dict[str, Any], bool]:
    """Read an ownership record without trusting any paths stored in it."""
    for filename, complete in (('installed.json', True), ('installing.json', False)):
        marker = entry / filename
        if not marker.exists():
            continue
        if marker.is_symlink() or marker.resolve() != marker:
            raise models.ModelError(f'Unsafe model installation record: {marker}.')
        try:
            record = json.loads(marker.read_text(encoding='utf-8'))
            if (not isinstance(record, dict) or record.get('version') != 1
                    or record.get('fingerprint') != entry.name
                    or not isinstance(record.get('name'), str)
                    or not isinstance(record.get('backend'), str)):
                raise ValueError('invalid ownership record')
        except (OSError, ValueError, TypeError) as exc:
            raise models.ModelError(f'Invalid managed model record at {marker}: {exc}.') from exc
        return record, complete
    raise models.ModelError(f'No ownership record for managed model directory {entry}.')


def _owned_size(entry: Path) -> int:
    """Count files in the store without following symlinks or junctions."""
    total = 0
    for directory, subdirectories, files in os.walk(entry, followlinks=False):
        base = Path(directory)
        for name in list(subdirectories):
            info = (base / name).lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
                subdirectories.remove(name)
        for name in files:
            info = (base / name).lstat()
            if stat.S_ISREG(info.st_mode):
                total += info.st_size
    return total


def installed(cache_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """List managed installations even if their configuration was removed."""
    root = _root(cache_dir)
    if not root.exists():
        return []
    result = []
    for path in sorted(root.iterdir()):
        if not re.fullmatch(r'[0-9a-f]{64}', path.name):
            continue
        entry = _entry(root, path.name)
        if not entry.is_dir():
            continue
        try:
            record, complete = _owned_record(entry)
            result.append({'id': entry.name, 'name': record['name'], 'backend': record['backend'],
                           'complete': complete, 'bytes': _owned_size(entry)})
        except (models.ModelError, OSError) as exc:
            result.append({'id': entry.name, 'name': None, 'complete': False, 'error': str(exc)})
    return result


def uninstall_id(identifier: str, cache_dir: str | Path | None = None) -> dict[str, Any]:
    """Delete a single identified pixcli-owned installation, including partial ones."""
    root = _root(cache_dir)
    entry = _entry(root, identifier)
    if not entry.exists():
        return {'id': identifier, 'removed': False, 'detail': 'No managed installation found.'}
    with _lock(root, entry):
        entry = _entry(root, identifier)
        if not entry.is_dir():
            raise models.ModelError(f'Unsafe managed model directory: {entry}.')
        record, _ = _owned_record(entry)
        shutil.rmtree(entry)
    return {'id': identifier, 'name': record['name'], 'removed': True,
            'detail': 'Removed managed installation; external files and the shared Hugging Face cache were preserved.'}


def uninstall(selection: dict[str, Any], cache_dir: str | Path | None = None) -> dict[str, Any]:
    """Remove only this selection's managed directory, preserving shared files."""
    result = uninstall_id(_fingerprint(selection), cache_dir)
    result['name'] = selection['name']
    if not result['removed']:
        result['detail'] = 'No managed installation; external files and the shared Hugging Face cache were preserved.'
    return result
