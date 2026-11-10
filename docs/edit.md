# `pixcli edit`

Edit an image from a text prompt, optionally restricting the edit with a mask. The output keeps the input dimensions. The command uses local image models. Add `-json` to print the settings and runtime details as JSON and save a JSON sidecar. Run it from the repository directory after [setup](../README.md#setup).

```powershell
pixcli edit -input outputs/knight.png -prompt "the same knight wearing red armor" -out outputs/red-knight.png
```

This writes `outputs/red-knight.png`. Add `-json` to also save `outputs/red-knight.json`. Lower edit strength tends to preserve more of the source; higher strength can alter its pose or identity. A shared seed does not guarantee identity preservation.

## Options

| Option | Default | What it does |
| --- | --- | --- |
| `-input PATH` | Required | Source image to edit. |
| `-prompt TEXT` | Required | Description of the desired edit; it cannot be blank. |
| `-out PATH` | Required | Output PNG path; the extension must be `.png`. |
| `-mask PATH` | None | Grayscale mask with the same dimensions as the input. White allows edits, black preserves the original pixels, and gray blends them. |
| `-strength FLOAT` | `0.3` | Set edit strength; greater than 0 and at most 1. The selected strength and step count must result in at least one inference step. |
| `-negative-prompt TEXT` | Empty | Describe features to avoid. |
| `-seed INTEGER` | `42` | Set the random seed, from 0 to 2⁶³−1. Identical seeds do not guarantee identical results across systems. |
| `-steps INTEGER` | `30` | Set the number of inference steps, from 1 to 150. |
| `-guidance FLOAT` | `7.0` | Set prompt guidance, from 0 to 30. |
| `-lora-strength FLOAT` | `1.0` | Set the pixel-art style strength; greater than 0 and at most 3. |
| `-inference-width INTEGER`, `-inference-height INTEGER` | `1024`, `1024` | Set the dimensions used for inference. Each must be a multiple of 8 from 512 to 2048. |
| `-background COLOR` | `#ffffff` | Set the RGB color used behind transparent input and for padding, for example `'#000000'`. |
| `-preview-scale INTEGER` | `1` | Enlarge the edited image with nearest-neighbor scaling for a preview. Values from 2 to 16 create a separate `.preview.png` file. |
| `-palette-colors INTEGER` | None | This option appears in help, but specifying a value for an edit is rejected to preserve untouched pixels. |
| `-palette NAME` | None | Map the edited image to a palette in `palettes/`. Black mask pixels stay unchanged; gray mask pixels blend with the original and may fall outside the palette. |
| `-style NAME` | None | Append the prompt in `styles/NAME.txt` to the positive prompt. |
| `-save-original` | Off | Save or skip the inference-resolution result as `.original.png`. |
| `-overwrite` | Off | Allow or prevent replacement of existing outputs or inputs. |
| `-device TEXT` | `auto` | Choose `auto`, `cuda`, or `cpu`. Auto selects CUDA if available, otherwise CPU. |
| `-ram-budget-gib FLOAT` | None | Set a positive process resident RAM budget in GiB; `auto` storage selects disk streaming. |
| `-vram-budget-gib FLOAT` | None | Limit the PyTorch CUDA allocator to this positive GiB value and use layer-level offload. |
| `-min-free-commit-gib FLOAT`, `-min-free-physical-gib FLOAT` | `2`, `2` | Minimum sampled free Windows commit and system RAM in GiB. Each must be at least 1. |
| `-weight-storage TEXT` | `auto` | `auto` chooses disk streaming for a RAM budget or CPU execution, otherwise memory. Choose `disk` or `memory` explicitly to override it. |
| `-cache-dir PATH` | Hugging Face cache | Load models from this cache directory; use the same location used for downloading. |
| `-offline` | Off | State offline intent explicitly; inference always uses local files. |
| `-json` | Off | Print JSON output and save the settings and runtime details in a JSON sidecar. |
| `-help` | — | Show command help and exit. |

## Examples

Use a hand-drawn grayscale mask to select the editable region:

```powershell
pixcli edit -input outputs/knight.png -mask armor-mask.png `
  -prompt "red armor" -out outputs/masked-knight.png -strength 0.5
```

The mask controls the final blend after the whole image has been edited; this is not a dedicated inpainting model. Transparent input is composited onto `-background` for inference and retains its original alpha in the result, so new shapes do not acquire a new alpha channel.

Add `-palette soft -style storybook` to use the supplied examples. See [Palettes and styles](../README.md#palettes-and-styles) for file formats and how to add your own.

Save an enlarged preview and use a smaller inference canvas:

```powershell
pixcli edit -input outputs/knight.png -prompt "gold armor" -out outputs/gold-knight.png `
  -inference-width 768 -inference-height 768 -preview-scale 4
```

Disk streaming constructs the model without loading its full weights, then reads tensors from the original checkpoint files as needed. It trades disk access and speed for lower memory use; keep model files unchanged while the job runs. On Windows, the RAM budget sets a process working-set maximum and the job also checks its sampled resident memory. The VRAM limit applies to PyTorch's allocator, not all GPU allocations or other applications, and cannot be used with CPU execution. Both commands stop if sampled available physical or Windows commit memory falls below its selected reserve; the default is 2 GiB for each. Model activations, operating-system memory and other processes still need headroom, and short spikes can exceed sampled limits.
