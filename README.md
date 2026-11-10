# pixcli

[![License: MIT](https://img.shields.io/github/license/luppichristian/pixcli)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/downloads/)
[![Windows](https://img.shields.io/badge/platform-Windows-0078D4)](#requirements)

Generate and edit pixel art from the command line, or animate an existing image with a motion prompt. pixcli runs inference locally using downloaded models and exports PNG sprites, sprite sheets, GIFs, and MP4s.

## Usage

### Generate

```powershell
pixcli gen -prompt "pixel art, a knight with blue armor" -out outputs/knight.png -seed 42
```

This creates a 128×128 PNG by default. Set `-pixel-width` and `-pixel-height` to change the sprite size; its aspect ratio must match the inference dimensions. Add `-preview-scale 8` for an enlarged preview or `-save-original` to keep the inference-resolution image. See the [`gen` reference](docs/gen.md).

### Edit

```powershell
pixcli edit -input outputs/knight.png -prompt "the same knight wearing red armor" -out outputs/red-knight.png -strength 0.3 -seed 42
```

The output keeps the input dimensions. Lower strength tends to preserve more of the original; higher strength permits larger changes but can alter pose or identity. To restrict the final blend, add `-mask path/to/mask.png`: the grayscale mask must match the input size, with white for editable pixels, black for preserved pixels, and gray for blending. Transparent input is composited over a background during inference and retains its original alpha. See the [`edit` reference](docs/edit.md).

### Animate

```powershell
pixcli animate -input outputs/knight.png -prompt "the knight walks in place, fixed camera" -out outputs/walk -seed 42
```

The output directory contains original and resized PNG frames, a sprite sheet, a GIF, and an MP4. Add `-no-mp4` to skip the MP4. The default is five 128×128 frames at 8 fps; `-frames`, `-fps`, `-pixel-width`, and `-pixel-height` adjust the output. Transparent input is composited over a background, and generated frames are opaque.

Animation is experimental and memory-intensive. A five-frame run completed on the GPU above, but its requested walking motion was subtle. Larger or longer jobs can require more memory, and a looping preview does not guarantee a seamless loop. See the [`animate` reference](docs/animate.md) for frame constraints, model profiles, and memory options.

### Output and offline use

Inference uses local model files. Once setup is complete, you can run without an internet connection; `-offline` makes that intent explicit. Add `-json` to print machine-readable results and save metadata alongside image outputs or in the animation directory. Existing outputs are protected by default; use a new path or pass `-overwrite` to replace them. Animation only replaces an existing directory if pixcli created it.

## Palettes and styles

Select a palette or style by name on `gen`, `edit`, or `animate`:

```powershell
pixcli gen -prompt "a knight" -out outputs/storybook.png -style storybook
```

pixcli looks for palettes in `palettes/` and style prompts in `styles/` under the current working directory. The included palettes are `soft`, `handheld-green`, `grayscale`, `cozy`, `neon`, and `retro`. The included styles are `8bit`, `16bit`, `gameboy`, `isometric`, `cozy`, `dark-fantasy`, `cyberpunk`, `monochrome`, and `storybook`.

To add a palette, create a `.hex` file with one six-digit RGB color per line. GIMP `.gpl`, JASC `.pal`, and Paint.NET `.txt` files are also supported. Palettes must contain 1 to 256 colors; Paint.NET alpha values must be opaque (`FF`). Include the extension when an extensionless name matches multiple files.

A style is a `.txt` file containing text appended to the prompt. It can select a palette with `palette: NAME` on its first line, followed by the prompt text. The supplied `storybook` style selects `soft`; an explicit `-palette` overrides it. A selected palette cannot be combined with `-palette-colors` or animation's `-palette-image`.

For masked edits, preserved and blended pixels may fall outside the palette. For animation, palette mapping applies to resized frames and the sprite sheet, not the original generated frames; MP4 compression can also change colors. With `-json`, the chosen style text and palette colors are recorded in metadata.

## Memory budgets

Generation, editing, and animation accept `-ram-budget-gib` and `-vram-budget-gib`:

```powershell
pixcli gen -prompt "pixel art, a knight" -out outputs/knight.png -ram-budget-gib 16 -vram-budget-gib 8
```

On Windows, RAM budgets use working-set limits and sampled resident-memory checks. Animation accounts for the command and backend processes together. With the default storage setting, image jobs stream weights from disk when a RAM budget is set; animation uses mapped weights by default. Streaming can be slower, and active tensors and the operating system still need memory. Sampled checks can miss short peaks.

The VRAM budget limits PyTorch's CUDA allocator, not every GPU allocation. See the [image](docs/gen.md) and [animation](docs/animate.md) references for storage choices, memory guards, and budget limits.

## Command reference

| Command | Purpose |
| --- | --- |
| [`gen`](docs/gen.md) | Generate an image from a prompt. |
| [`edit`](docs/edit.md) | Edit an image, optionally using a mask. |
| [`animate`](docs/animate.md) | Animate an image. |
| [`doctor`](docs/doctor.md) | Check the environment, hardware, and model availability. |
| [`models`](docs/models.md) | Manage model files. |
| [`models list`](docs/models-list.md) | List pinned model files and revisions. |
| [`models download`](docs/models-download.md) | Download and verify a model profile. |
| [`models verify`](docs/models-verify.md) | Check a model profile locally. |

For example, `pixcli models verify -profile animation` checks the default animation models. If using `-cache-dir PATH`, pass the same path when downloading, verifying, and running jobs. Top-level options `-install-completion` and `-show-completion` install or display shell completion.


## Requirements

- 64-bit Windows, [uv](https://docs.astral.sh/uv/getting-started/installation/), and [Git](https://git-scm.com/). Setup uses Python 3.11 and requires internet access for dependencies and models.
- About 24.12 GiB for the default models (6.78 GiB for images and 17.34 GiB for animation), plus space for software and temporary downloads.
- An NVIDIA GPU with a compatible driver is recommended. CPU execution is available but can be much slower, especially for animation.

Image generation, editing, and a short animation have been tested on Windows 11 with 32 GiB RAM and an RTX 4070 SUPER with 12 GB VRAM, as well as on CPU. Other systems have not been verified. Memory needs depend on the settings and available system headroom.

## Setup

Clone or download the repository, open PowerShell in its directory, and run:

```powershell
.\setup.ps1
. .\.venv\Scripts\Activate.ps1
pixcli doctor
```

Setup installs the image environment and separate animation backend, then downloads and verifies both default model profiles. `doctor` reports hardware and model availability. If a download is interrupted, rerun setup to resume it.

If a model requires access approval, accept its terms on Hugging Face and sign in with `.\.venv\Scripts\hf.exe auth login` or set `HF_TOKEN`, then rerun setup. Models are stored in the Hugging Face cache by default; set `HF_HUB_CACHE` before setup to change its location.

Run the examples below from the repository directory with the environment activated. In a new PowerShell session, activate it again with `. .\.venv\Scripts\Activate.ps1`; run `deactivate` when finished. Commands use single-dash option names: use `pixcli -help` or, for example, `pixcli gen -help` to see them.

## Troubleshooting

- **CUDA unavailable:** check the NVIDIA driver and run `pixcli doctor` again.
- **Missing model:** run `pixcli models download -profile image` or `-profile animation`, then verify the same profile. Alternative animation configurations require their matching [model profile](docs/models.md#profiles).
- **Out of memory:** close memory-heavy applications and try smaller inference dimensions. Animation may still need substantial RAM and free Windows commit memory, even with the default settings.

## Licenses

The pixcli source code is available under the [MIT License](LICENSE). Models have separate licenses and terms: review the pages for [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), [Pixel Art XL](https://huggingface.co/nerijs/pixel-art-xl), [SDXL VAE](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix), [Wan 2.2 GGUF experts](https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF), [Wan 2.2 components](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged), [UMT5 GGUF](https://huggingface.co/city96/umt5-xxl-encoder-gguf), and the [pixel animation adapter](https://huggingface.co/styly-agents/Wan2-2-pixel-animate). The [model manifest](pixcli/model_manifest.json) lists exact files and revisions.
