# pixcli

[![GitHub stars](https://img.shields.io/github/stars/luppichristian/pixcli)](https://github.com/luppichristian/pixcli/stargazers)
[![License: MIT](https://img.shields.io/github/license/luppichristian/pixcli)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/downloads/)
[![Windows](https://img.shields.io/badge/platform-Windows-0078D4)](#requirements)

Create pixel art and animations locally from the command line. Generate an image from a prompt, edit an existing image, or turn an image into an animation. Once the models are downloaded, inference runs without an internet connection.

## Requirements

- 64-bit Windows. An NVIDIA GPU with a compatible driver accelerates inference; CPU execution can be much slower. Image generation, editing and a short animation have been tested on Windows 11 with an RTX 4070 SUPER (12 GB VRAM) and 32 GiB RAM, and on CPU. Other systems have not been verified.
- [uv](https://docs.astral.sh/uv/getting-started/installation/), [Git](https://git-scm.com/), and an internet connection for setup and model downloads.
- About 24.12 GiB for the models (6.78 GiB for images and 17.34 GiB for animation); allow extra space for temporary downloads and software.

## Setup

Download or clone this repository, open PowerShell in its directory, and run:

```powershell
.\setup.ps1
. .\.venv\Scripts\Activate.ps1
pixcli doctor
```

Setup installs the environments for all three commands, then downloads and verifies both model profiles. `doctor` reports your GPU, memory, and model availability. You can use your own input image for editing or animation.

Activation makes `pixcli` available in the current PowerShell session. Stay in the repository directory for the commands below. In a new PowerShell session, return to the repository directory and run `. .\.venv\Scripts\Activate.ps1` again. Run `deactivate` when finished.

If a download is interrupted, run `setup.ps1` again to resume it. If a model requires access approval, visit its Hugging Face page, accept its terms, and sign in with `.\.venv\Scripts\hf.exe auth login` or set the `HF_TOKEN` environment variable, then rerun setup.

Run the commands below from the repository directory. For a list of options for any command, add `--help`, for example:

```powershell
pixcli --help
pixcli gen --help
```

## Generate an image

```powershell
pixcli gen --prompt "pixel art, a knight with blue armor" --out outputs/knight.png --seed 42
```

This writes `outputs/knight.png` and a JSON file containing the settings used. The default sprite is 128×128 pixels. Use `--pixel-width` and `--pixel-height` to change its size; the output aspect ratio must match the inference dimensions. Add `--preview-scale 8` for an enlarged preview or `--save-original` to keep the full-resolution image.

## Edit an image

```powershell
pixcli edit --input outputs/knight.png --prompt "the same knight wearing red armor" --out outputs/red-knight.png --strength 0.3 --seed 42
```

The edited image keeps the input dimensions. Lower `--strength` tends to preserve more of the original; higher values allow larger changes but can alter the pose or identity.

To edit a selected area, add `--mask path/to/mask.png`. Use a grayscale mask with the same dimensions as the input: white areas can change, black areas are preserved, and gray areas blend the original and edit. Transparent input is placed on the chosen background for editing; its original alpha is retained.

## Animate an image

Provide an image, a motion prompt, and an output directory:

```powershell
pixcli animate --input outputs/knight.png --prompt "the knight walks in place, fixed camera" --out outputs/walk --seed 42
```

The output directory contains PNG frames, a sprite sheet, a GIF, an MP4, and metadata. Add `--no-mp4` to skip the MP4. Use `--frames`, `--fps`, `--pixel-width`, and `--pixel-height` to adjust the output.

Animation is experimental and memory-intensive. The default produces five 128×128 frames at 8 fps. A test on the hardware above completed, but the requested walking motion was subtle. Longer or larger animations can require more memory, and looping playback does not guarantee a seamless loop. Transparent input is composited over a background; generated frames are opaque.

## Memory budgets

Each generation, edit, or animation command accepts `--ram-budget-gib` and `--vram-budget-gib`. On Windows, a RAM budget bounds working sets and allows file-backed weights to be paged in as needed; animation accounts for the command and its backend processes together. RAM checks are also sampled, so brief peaks can exceed a limit. The VRAM budget limits PyTorch's CUDA allocator; it does not include every GPU allocation. Image jobs stream tensors from their model files when a RAM budget is set, and animation uses mapped expert weights by default. Streaming can be slower, and inference still needs enough memory for its active tensors and operating-system overhead. See the command references for the budget scope and storage options.

```powershell
pixcli gen --prompt "pixel art, a knight" --out outputs/knight.png --ram-budget-gib 16 --vram-budget-gib 8
```

## Commands

Each command has its own reference with options, defaults, and examples:

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

Every command accepts `--help`. At the top level, `pixcli --install-completion` installs shell completion and `pixcli --show-completion` displays the completion script.

Existing output files are protected by default. Choose a new output path or add `--overwrite` to replace them. Inference uses local files; `--offline` makes that explicit. Models use the Hugging Face cache by default. To use another location, pass the same `--cache-dir PATH` when downloading, verifying, and running a job.

## Troubleshooting

- **CUDA unavailable:** check your NVIDIA driver, then run `doctor` again.
- **Missing model:** connect to the internet and run the appropriate `models download` command, then verify it.
- **Out of memory:** close other GPU or memory-heavy applications. For images, try smaller matching `--inference-width` and `--inference-height` values. Animation settings can require substantial memory even on the tested hardware.

## Model licenses

Models have their own licenses and terms. Review the pages for [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), [Pixel Art XL](https://huggingface.co/nerijs/pixel-art-xl), [SDXL VAE](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix), [Wan 2.2 GGUF experts](https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF), [Wan 2.2 components](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged), [UMT5 GGUF](https://huggingface.co/city96/umt5-xxl-encoder-gguf), and the [pixel animation adapter](https://huggingface.co/styly-agents/Wan2-2-pixel-animate). The [model manifest](pixcli/model_manifest.json) lists the exact files and revisions.

The pixcli source code is available under the [MIT License](LICENSE).
