# `pixcli gen`

Generate a pixel-art PNG from a text prompt. The command uses the local image models and writes a JSON sidecar with the settings and runtime details. Run it from the repository directory after [setup](../README.md#setup).

```powershell
pixcli gen --prompt "pixel art, a knight with blue armor" --out outputs/knight.png
```

This creates `outputs/knight.png` and `outputs/knight.json`. Existing outputs are protected unless you pass `--overwrite`.

## Options

| Option | Default | What it does |
| --- | --- | --- |
| `--prompt TEXT` | Required | Description of the image; it cannot be blank. |
| `--out PATH` | Required | Output PNG path; the extension must be `.png`. |
| `--negative-prompt TEXT` | Empty | Describe features to avoid. |
| `--seed INTEGER` | `42` | Set the random seed, from 0 to 2⁶³−1. Identical seeds do not guarantee identical results across systems. |
| `--steps INTEGER` | `30` | Set the number of inference steps, from 1 to 150. |
| `--guidance FLOAT` | `7.0` | Set prompt guidance, from 0 to 30. |
| `--lora-strength FLOAT` | `1.0` | Set the pixel-art style strength; greater than 0 and at most 3. |
| `--inference-width INTEGER`, `--inference-height INTEGER` | `1024`, `1024` | Set the image dimensions used for inference. Each must be a multiple of 8 from 512 to 2048. |
| `--pixel-width INTEGER`, `--pixel-height INTEGER` | `128`, `128` | Set the final sprite dimensions, each from 1 to 2048. Their aspect ratio must match the inference dimensions. |
| `--preview-scale INTEGER` | `1` | Enlarge the sprite with nearest-neighbor scaling for a preview. Values from 2 to 16 create a separate `.preview.png` file. |
| `--palette-colors INTEGER` | None | Reduce the final sprite to 2–256 colors. |
| `--save-original` / `--no-save-original` | Off | Save or skip the inference-resolution image as `.original.png`. |
| `--overwrite` / `--no-overwrite` | Off | Allow or prevent replacing existing output files. |
| `--cache-dir PATH` | Hugging Face cache | Load models from this cache directory; use the same location used for downloading. |
| `--offline` / `--no-offline` | `--no-offline` | State offline intent explicitly; inference always uses local files. |
| `--help` | — | Show command help and exit. |

## Examples

Choose matching rectangular inference and sprite dimensions, save the original, and create an enlarged preview:

```powershell
pixcli gen --prompt "pixel art, a forest clearing" --out outputs/forest.png `
  --inference-width 768 --inference-height 512 --pixel-width 192 --pixel-height 128 `
  --save-original --preview-scale 4
```

Reduce the final sprite to 16 colors:

```powershell
pixcli gen --prompt "pixel art, a treasure chest" --out outputs/chest.png --palette-colors 16
```

Image commands have no configurable RAM or VRAM budget. They check host-memory headroom and stop when sampled available physical or Windows commit memory falls below 2 GiB. Sampling cannot catch every short allocation spike. If GPU memory is exhausted, close other GPU applications or try smaller matching inference dimensions.
