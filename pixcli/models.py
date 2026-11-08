"""Pinned model inventory and resumable Hugging Face cache management.

``model_path(key, cache_dir)`` is always offline: base and vae return snapshot
folders, other keys return a single model file. ``verify`` hashes every file; it
returns a list of dictionaries with key, filename, ok, and detail. ``download``
checks space, fetches only pinned files, and verifies their cryptographic hashes.
A None cache_dir uses the existing standard Hugging Face Hub cache.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
from typing import Any


class ModelError(RuntimeError):
    """Missing, corrupt, inaccessible, or unsupported model."""


MANIFEST_PATH = Path(__file__).with_name("model_manifest.json")


def manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def list_models(profile: str | None = None) -> dict[str, Any]:
    data = manifest()
    keys = list(data["models"]) if profile is None else _profile(data, profile)
    return {key: data["models"][key] for key in keys}


def animation_profile(text_encoder: str = 'q3', expert_quant: str = 'q2') -> str:
    choices = {('fp8', 'q3'): 'animation-q3-fp8', ('q3', 'q3'): 'animation-q3-text',
               ('fp8', 'q2'): 'animation-q2-experts', ('q3', 'q2'): 'animation'}
    try:
        return choices[(text_encoder, expert_quant)]
    except (KeyError, TypeError) as exc:
        raise ModelError('Unsupported animation selection: text encoder must be fp8 or q3 and expert quantization q3 or q2.') from exc


def _profile(data: dict[str, Any], profile: str) -> list[str]:
    try:
        return data["profiles"][profile]
    except KeyError as exc:
        raise ModelError(f"Unknown model profile {profile!r}; choose one of {', '.join(data['profiles'])}.") from exc


def _hub():
    try:
        import huggingface_hub
        return huggingface_hub
    except ImportError as exc:
        raise ModelError("huggingface_hub is missing; run the project setup script.") from exc


def cache_location(cache_dir: str | Path | None = None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).expanduser().resolve()
    # Respect both standard HF_HUB_CACHE and HF_HOME conventions.
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser().resolve()
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):
        return Path(os.environ["HUGGINGFACE_HUB_CACHE"]).expanduser().resolve()
    if os.environ.get("HF_HOME"):
        return (Path(os.environ["HF_HOME"]).expanduser() / "hub").resolve()
    if os.environ.get("XDG_CACHE_HOME"):
        return (Path(os.environ["XDG_CACHE_HOME"]).expanduser() / "huggingface" / "hub").resolve()
    return Path.home() / ".cache" / "huggingface" / "hub"


def _cached(model: dict[str, Any], filename: str, cache_dir: str | Path | None) -> Path | None:
    hub = _hub()
    result = hub.try_to_load_from_cache(model["repo_id"], filename, cache_dir=str(cache_location(cache_dir)), revision=model["revision"])
    return Path(result) if isinstance(result, str) and Path(result).is_file() else None


def _digest(path: Path, item: dict[str, Any]) -> bool:
    if path.stat().st_size != item["size"]:
        return False
    if "sha256" in item:
        h = hashlib.sha256()
        expected = item["sha256"]
    else:
        # Hugging Face exposes the immutable Git blob ID for non-LFS files.
        h = hashlib.sha1()
        h.update(f"blob {item['size']}\0".encode())
        expected = item["git_blob_sha1"]
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest() == expected


def verify(profile: str = "image", cache_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Offline size and SHA-256/Git blob verification of every pinned file."""
    results = []
    for key, model in list_models(profile).items():
        for item in model["files"]:
            path = _cached(model, item["filename"], cache_dir)
            try:
                ok = path is not None and _digest(path, item)
                detail = "verified" if ok else ("missing" if path is None else "hash or size mismatch")
            except OSError as exc:
                ok, detail = False, str(exc)
            results.append({"key": key, "filename": item["filename"], "ok": ok, "detail": detail})
    return results


def model_path(key: str, cache_dir: str | Path | None = None, *, profile: str | None = None) -> Path:
    """Return the pinned local snapshot directory, or adapter file, offline.

    Checks all required files and their sizes. Use ``verify`` for full hashes.
    """
    models = list_models()
    if key not in models:
        raise ModelError(f"Unknown model {key!r}.")
    if profile is not None and key not in list_models(profile):
        raise ModelError(f"Model {key!r} is not in profile {profile!r}.")
    model = models[key]
    paths = []
    for item in model["files"]:
        path = _cached(model, item["filename"], cache_dir)
        if path is None or path.stat().st_size != item["size"]:
            selected_profile = profile or ("animation-q3-fp8" if key in ('wan_high', 'wan_low', 'wan_text') else ("animation" if key.startswith("wan_") else "image"))
            raise ModelError(f"Missing or incomplete {key}: {item['filename']}. Run `pixcli models download --profile {selected_profile}`, then `pixcli models verify --profile {selected_profile}`.")
        paths.append(path)
    if key not in ("base", "vae"):
        return paths[0]
    return paths[0].parents[len(PurePosixPath(model["files"][0]["filename"]).parts) - 1]


def download(profile: str = "image", cache_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Download only pinned files with Hub resumability and verify them offline."""
    models = list_models(profile)  # reject unsupported profiles before network access
    if profile.startswith("animation"):
        print("Animation models are pinned; runtime memory needs and output quality depend on the selected profile and machine.", flush=True)
    hub = _hub()
    target = cache_location(cache_dir)
    target.mkdir(parents=True, exist_ok=True)
    pending = []
    for key, model in models.items():
        for item in model["files"]:
            path = _cached(model, item["filename"], target)
            good = path is not None and _digest(path, item)
            if not good:
                pending.append((key, model, item, path is not None))
    required = sum(item["size"] for _, _, item, _ in pending)
    # Reserve the full missing size plus the largest file for temporary/resumed
    # data and 1 GiB overhead; no model conversion is performed.
    reserve = required + max((item["size"] for _, _, item, _ in pending), default=0) + (1024**3 if pending else 0)
    free = shutil.disk_usage(target).free
    print(f"Model cache: {target}; up to {required / 1024**3:.2f} GiB to download; reserve {reserve / 1024**3:.2f} GiB including temporary space; free {free / 1024**3:.2f} GiB.")
    if free < reserve:
        raise ModelError("Insufficient disk space in the model cache. Set HF_HUB_CACHE or choose a cache directory on a larger disk.")
    for key, model, item, corrupt in pending:
        print(f"Downloading {key}: {item['filename']} ({item['size'] / 1024**3:.2f} GiB)", flush=True)
        try:
            hub.hf_hub_download(repo_id=model["repo_id"], filename=item["filename"], revision=model["revision"], cache_dir=str(target), force_download=corrupt)
        except Exception as exc:
            raise ModelError(f"Download failed for {model['repo_id']}/{item['filename']}: {exc}. Check connectivity, Hugging Face login and license acceptance if required; rerun to resume.") from exc
    results = verify(profile, target)
    failed = [r for r in results if not r["ok"]]
    if failed:
        raise ModelError(f"Downloaded model verification failed: {failed}. Rerun download to repair.")
    return results
