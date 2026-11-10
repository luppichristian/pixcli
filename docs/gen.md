# `pixcli gen`

Generate a pixel-art PNG from a text prompt. The command uses the local image models. Add `-json` to print the settings and runtime details as JSON and save a JSON sidecar. Run it from the repository directory after [setup](../README.md#setup).

```powershell
pixcli gen -prompt "pixel art, a knight with blue armor" -out outputs/knight.png
```

This creates `outputs/knight.png`. Add `-json` to also save `outputs/knight.json`. Existing outputs are protected unless you pass `-overwrite`.

## Options

| Option | Default | What it does |
| --- | --- | --- |
| `-prompt TEXT` | Required | Description of the image; it cannot be blank. |
| `-out PATH` | Required | Output PNG path; the extension must be `.png`. |
| `-negative-prompt TEXT` | Empty | Describe features to avoid. |
| `-seed INTEGER` | `42` | Set the random seed, from 0 to 2⁶³−1. Identical seeds do not guarantee identical results across systems. |
| `-steps INTEGER` | `30` | Set the number of inference steps, from 1 to 150. |
| `-guidance FLOAT` | `7.0` | Set prompt guidance, from 0 to 30. |
| `-lora-strength FLOAT` | `1.0` | Set the pixel-art style strength; greater than 0 and at most 3. |
| `-inference-width INTEGER`, `-inference-height INTEGER` | `1024`, `1024` | Set the image dimensions used for inference. Each must be a multiple of 8 from 512 to 2048. |
| `-pixel-width INTEGER`, `-pixel-height INTEGER` | `128`, `128` | Set the final sprite dimensions, each from 1 to 2048. Their aspect ratio must match the inference dimensions. |
| `-preview-scale INTEGER` | `1` | Enlarge the sprite with nearest-neighbor scaling for a preview. Values from 2 to 16 create a separate `.preview.png` file. |
| `-palette-colors INTEGER` | None | Reduce the final sprite to 2–256 colors. |
| `-palette NAME` | None | Map the final sprite to a palette in `palettes/`. Cannot be combined with `-palette-colors`. |
| `-style NAME` | None | Append the prompt in `styles/NAME.txt` to the positive prompt. |
| `-save-original` | Off | Save or skip the inference-resolution image as `.original.png`. |
| `-overwrite` | Off | Allow or prevent replacing existing output files. |
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

Choose matching rectangular inference and sprite dimensions, save the original, and create an enlarged preview:

```powershell
pixcli gen -prompt "pixel art, a forest clearing" -out outputs/forest.png `
  -inference-width 768 -inference-height 512 -pixel-width 192 -pixel-height 128 `
  -save-original -preview-scale 4
```

Reduce the final sprite to 16 colors:

```powershell
pixcli gen -prompt "pixel art, a treasure chest" -out outputs/chest.png -palette-colors 16
```

Use the example palette and style, or add your own files as described in [Palettes and styles](../README.md#palettes-and-styles):

```powershell
pixcli gen -prompt "a treasure chest" -out outputs/chest.png -palette soft -style storybook
```

Disk streaming constructs the model without loading its full weights, then reads tensors from the original checkpoint files as needed. It trades disk access and speed for lower memory use; keep model files unchanged while the job runs. On Windows, the RAM budget sets a process working-set maximum and the job also checks its sampled resident memory. The VRAM limit applies to PyTorch's allocator, not all GPU allocations or other applications, and cannot be used with CPU execution. Both commands stop if sampled available physical or Windows commit memory falls below its selected reserve; the default is 2 GiB for each. Model activations, operating-system memory and other processes still need headroom, and short spikes can exceed sampled limits.
