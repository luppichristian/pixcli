# `pixcli models`

Manage the model files used by image generation, editing, and animation. Setup downloads and verifies the default image and animation profiles. Use these commands to inspect, download, or verify files later.

```powershell
pixcli models [OPTIONS] COMMAND [ARGS]
```

| Command | What it does |
| --- | --- |
| [`models list`](models-list.md) | List all pinned model repositories, revisions, and files. |
| [`models download`](models-download.md) | Download and verify the files for a profile. |
| [`models verify`](models-verify.md) | Check cached files without network access. |

The group option `-help` displays the available commands and exits. Each command also accepts `-help`.

## Profiles

| Profile | Contents |
| --- | --- |
| `image` | Image generation and editing models. This is the default for `download` and `verify`. |
| `animation` | Q2 experts and Q3 text encoder, the default animation configuration. |
| `animation-q3-fp8` | Q3 experts and FP8 text encoder. |
| `animation-q3-text` | Q3 experts and Q3 text encoder. |
| `animation-q2-experts` | Q2 experts and FP8 text encoder. |
| `animation-q2-q3-text` | Q2 experts and Q3 text encoder; the same model selection as `animation`. |

If you change the animation text encoder or expert quantization, download the matching profile before running the job. The pinned files and their model pages are listed in the [model manifest](../pixcli/model_manifest.json).

## Cache

By default, pixcli uses the Hugging Face cache, normally `C:\Users\<your-name>\.cache\huggingface\hub` on Windows. Set `HF_HUB_CACHE` before setup to use a different location, or pass `-cache-dir PATH` to `download`, `verify`, `doctor`, and inference commands. Use the same location consistently.

## Examples

```powershell
pixcli models -help
pixcli models list
pixcli models download -profile animation
pixcli models verify -profile animation
```
