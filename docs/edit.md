# `pixcli edit`

Edit an image from a text prompt, optionally restricting the edit with a mask. The output keeps the input dimensions. The command uses local image models and writes a JSON sidecar with the settings and runtime details. Run it from the repository directory after [setup](../README.md#setup).

```powershell
pixcli edit --input outputs/knight.png --prompt "the same knight wearing red armor" --out outputs/red-knight.png
```

This writes `outputs/red-knight.png` and `outputs/red-knight.json`. Lower edit strength tends to preserve more of the source; higher strength can alter its pose or identity. A shared seed does not guarantee identity preservation.

## Options

| Option | Default | What it does |
| --- | --- | --- |
| `--input PATH` | Required | Source image to edit. |
| `--prompt TEXT` | Required | Description of the desired edit; it cannot be blank. |
| `--out PATH` | Required | Output PNG path; the extension must be `.png`. |
| `--mask PATH` | None | Grayscale mask with the same dimensions as the input. White allows edits, black preserves the original pixels, and gray blends them. |
| `--strength FLOAT` | `0.3` | Set edit strength; greater than 0 and at most 1. The selected strength and step count must result in at least one inference step. |
| `--negative-prompt TEXT` | Empty | Describe features to avoid. |
| `--seed INTEGER` | `42` | Set the random seed, from 0 to 2⁶³−1. Identical seeds do not guarantee identical results across systems. |
| `--steps INTEGER` | `30` | Set the number of inference steps, from 1 to 150. |
| `--guidance FLOAT` | `7.0` | Set prompt guidance, from 0 to 30. |
| `--lora-strength FLOAT` | `1.0` | Set the pixel-art style strength; greater than 0 and at most 3. |
| `--inference-width INTEGER`, `--inference-height INTEGER` | `1024`, `1024` | Set the dimensions used for inference. Each must be a multiple of 8 from 512 to 2048. |
| `--background COLOR` | `#ffffff` | Set the RGB color used behind transparent input and for padding, for example `'#000000'`. |
| `--preview-scale INTEGER` | `1` | Enlarge the edited image with nearest-neighbor scaling for a preview. Values from 2 to 16 create a separate `.preview.png` file. |
| `--palette-colors INTEGER` | None | This option appears in help, but specifying a value for an edit is rejected to preserve untouched pixels. |
| `--save-original` / `--no-save-original` | Off | Save or skip the inference-resolution result as `.original.png`. |
| `--overwrite` / `--no-overwrite` | Off | Allow or prevent replacement of existing outputs or inputs. |
| `--cache-dir PATH` | Hugging Face cache | Load models from this cache directory; use the same location used for downloading. |
| `--offline` / `--no-offline` | `--no-offline` | State offline intent explicitly; inference always uses local files. |
| `--help` | — | Show command help and exit. |

## Examples

Use a hand-drawn grayscale mask to select the editable region:

```powershell
pixcli edit --input outputs/knight.png --mask armor-mask.png `
  --prompt "red armor" --out outputs/masked-knight.png --strength 0.5
```

The mask controls the final blend after the whole image has been edited; this is not a dedicated inpainting model. Transparent input is composited onto `--background` for inference and retains its original alpha in the result, so new shapes do not acquire a new alpha channel.

Save an enlarged preview and use a smaller inference canvas:

```powershell
pixcli edit --input outputs/knight.png --prompt "gold armor" --out outputs/gold-knight.png `
  --inference-width 768 --inference-height 768 --preview-scale 4
```

Image commands have no configurable RAM or VRAM budget. They check host-memory headroom and stop when sampled available physical or Windows commit memory falls below 2 GiB. Sampling cannot catch every short allocation spike. If GPU memory is exhausted, close other GPU applications or try smaller inference dimensions.
