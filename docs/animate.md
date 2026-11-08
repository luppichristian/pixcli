# `pixcli animate`

Animate an existing image from a text prompt. The command runs locally using the models installed by setup and writes frames, a sprite sheet, a GIF, an MP4, and metadata to an output directory.

```powershell
pixcli animate --input outputs/knight.png --prompt "the knight walks in place, fixed camera" --out outputs/walk
```

Animation is experimental. A five-frame run completed on the hardware described in the [README](../README.md), but the requested walking motion was subtle. Output quality, memory use, and runtime vary with the image and settings. A looping preview does not guarantee a seamless animation.

## Required options

| Option | Description |
| --- | --- |
| `--input PATH` | Source image. It must exist outside the output directory. |
| `--prompt TEXT` | Nonempty description of the motion to generate. |
| `--out DIR` | Directory for the animation and its metadata. |

## Generation and output options

| Option | Default | Description |
| --- | --- | --- |
| `--negative-prompt TEXT` | Empty | Describe things to avoid. |
| `--seed INTEGER` | `42` | Random seed, from 0 to 2⁶⁴−1. Results may differ between machines even with the same seed. |
| `--steps INTEGER` | `4` | Inference steps. The current workflow requires exactly 4. |
| `--guidance FLOAT` | `1.0` | Prompt guidance, from 0 to 100. |
| `--lora-strength FLOAT` | `1.0` | Pixel-animation style strength, from 0.01 to 2. |
| `--inference-width INTEGER` | `256` | Width used to generate the animation. |
| `--inference-height INTEGER` | `256` | Height used to generate the animation. |
| `--frames INTEGER` | `5` | Number of frames, from 5 to 81 in increments of 4: 5, 9, 13, and so on. |
| `--fps FLOAT` | `8` | Playback frames per second; greater than 0 and at most 100. |
| `--pixel-width INTEGER` | `128` | Width of each output frame, from 1 to 2048. |
| `--pixel-height INTEGER` | `128` | Height of each output frame, from 1 to 2048. |
| `--palette-colors INTEGER` | None | Reduce output frames to a shared palette of 2–256 colors. |
| `--palette-image PATH` | None | Derive the shared palette from this image. Requires `--palette-colors`; the image must be outside the output directory. Without it, the palette is derived from the generated frames. |
| `--background COLOR` | `#ffffff` | RGB color used for transparent source pixels and padding. |
| `--mp4` / `--no-mp4` | `--mp4` | Include or skip MP4 export. |
| `--overwrite` / `--no-overwrite` | `--no-overwrite` | Replace an existing output directory only if it was created by pixcli. |
| `--cache-dir PATH` | Hugging Face cache | Use a specific model cache. Use the same path as when downloading the models. |
| `--offline` / `--no-offline` | `--no-offline` | Express offline intent. Animation inference always uses local model files. |
| `--help` | — | Show command options. |

Inference dimensions must each be multiples of 16 from 16 to 832, with an area of at most 832×480 pixels. Some combinations of frame count and dimensions also exceed an internal allocation limit. Larger settings can require more memory even when they satisfy these checks.

Input images are fitted without stretching and padded as needed. Transparent pixels are composited over the background; generated frames are opaque. Output includes the original generated frames, resized frames, `sheet.png`, `preview.gif`, `preview.mp4` unless disabled, and JSON metadata.

## Model and memory options

| Option | Default | Description |
| --- | --- | --- |
| `--preset TEXT` | `low-vram` | Choose `low-vram` or `gpu-resident`. `gpu-resident` requires `--text-encoder fp8`. |
| `--backend TEXT` | `comfyui-gguf` | Animation backend. `comfyui-gguf` is the only supported value. |
| `--text-encoder TEXT` | `q3` | Choose `q3` or `fp8`; the corresponding model profile must be downloaded. |
| `--expert-quant TEXT` | `q2` | Choose `q2` or `q3` expert weights; the corresponding model profile must be downloaded. |
| `--adapter-device TEXT` | `cpu` | Choose `cpu` or `cuda` for the pixel adapter. `cuda` requires the `gpu-resident` preset. |
| `--weight-storage TEXT` | `mmap` | Choose `mmap` to keep eligible weights backed by model files, or `normal` for normal loading. Keep the files unchanged while a job runs. |
| `--max-private-gib FLOAT` | `14` | Maximum sampled private memory for the backend process tree, in GiB. Must be positive. |
| `--min-free-physical-gib FLOAT` | `2` | Minimum sampled free system RAM in GiB. Must be at least 1. |
| `--min-free-commit-gib FLOAT` | `2` | Minimum sampled free Windows commit memory in GiB. Must be at least 1. |

Memory thresholds are sampled stop conditions, not hard allocation caps: a short allocation can exceed them between samples. Startup requires at least 4 GiB of free physical RAM and 6 GiB of free commit memory, or the selected reserve if higher. There is no numerical VRAM budget option. Smaller inference dimensions or fewer frames may reduce memory use.

Setup downloads the default `animation` profile, which uses Q2 experts and the Q3 text encoder. If you change either setting, download the matching profile:

| Expert quantization | Text encoder | Model profile |
| --- | --- | --- |
| `q2` | `q3` | `animation` |
| `q2` | `fp8` | `animation-q2-experts` |
| `q3` | `q3` | `animation-q3-text` |
| `q3` | `fp8` | `animation-q3-fp8` |

Use `pixcli models download --profile NAME` to download a different profile. See [model downloads](models-download.md) for details.

## Examples

Set the output dimensions and skip MP4:

```powershell
pixcli animate --input sprite.png --prompt "a torch flickers" --out outputs/torch `
  --frames 9 --fps 8 --pixel-width 64 --pixel-height 64 --no-mp4
```

Set host-memory stop thresholds:

```powershell
pixcli animate --input sprite.png --prompt "walking in place" --out outputs/walk `
  --max-private-gib 12 --min-free-physical-gib 4 --min-free-commit-gib 4
```

The examples use PowerShell and assume the environment is activated from the repository directory, as described in [setup](../README.md#setup).
