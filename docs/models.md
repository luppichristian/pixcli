# Model configuration and management

pixcli uses named models for generation, editing, and animation. The checked-in `pixcli.toml` selects `default-image` and `default-animation`, the same pinned models used by setup. Inference always uses local files; it does not start a download.

```powershell
pixcli models list
pixcli models status
pixcli models install NAME
pixcli models verify -model NAME
pixcli models uninstall NAME
```

`models list` shows the built-in manifest and configured names. `models status` checks local availability. `models install` downloads a configured Hugging Face model into pixcli-managed storage; a configured local model remains at its original path. `models uninstall` removes pixcli-managed downloads for that name and leaves local paths and the shared Hugging Face cache intact.

If you change or remove a model definition, run `pixcli models status -installed` to list managed installations, including older definitions. Remove an old installation with `pixcli models uninstall-id ID` using the ID shown in that list.

Setup and the existing `models download -profile NAME` command use the shared Hugging Face cache. Files there may also be used by other applications or other pixcli profiles, so uninstall does not remove them. To inspect or delete those shared cache entries, use `hf cache scan` and `hf cache delete`; review the affected repositories and revisions before deleting them. Add `--dir PATH` to those commands if you use a custom cache.

Run `./setup.ps1 -SkipModels` if you want to configure and install your own models without downloading the default profiles. Installing `default-image` or `default-animation` through `models install` creates a separate managed copy; an existing shared-cache copy can still be available after that managed copy is removed.

By default, pixcli checks `pixcli.toml` in the current working directory. You can select a file with `-config PATH` on the model and inference commands, or set `PIXCLI_CONFIG`. Local paths in the file are relative to the configuration file. An absent default file uses the built-in defaults.

## Choose image models

Create or edit `pixcli.toml`, replacing the example repository with a model you have access to:

```toml
version = 1

[defaults]
generation = "my-generator"
editing = "my-editor"
animation = "default-animation"

[models.my-generator]
backend = "diffusers"

[models.my-generator.source]
type = "huggingface"
repo_id = "owner/model-name"
revision = "main"

[models.my-editor]
backend = "diffusers"

[models.my-editor.source]
type = "local"
path = "D:/models/my-editor"
```

Then install and use the generator:

```powershell
pixcli models install my-generator
pixcli models status
pixcli gen -prompt "a knight" -out knight.png
pixcli edit -input knight.png -prompt "red armor" -out red.png
pixcli gen -model my-editor -prompt "a tree" -out tree.png
```

The local path must point to a Diffusers model directory, including `model_index.json` and safetensors weights. A Hugging Face model needs a compatible Diffusers repository layout. `revision` can be a branch, tag, or commit; installation resolves it to a commit, which is recorded with the installed files. Use a commit to specify an immutable revision in the configuration. Add `variant = "fp16"` under a model only when that snapshot provides the corresponding weight variant.

Installing a Hugging Face image model downloads the repository snapshot, which can include several large weight variants. You can set `allow_patterns` in the source table to restrict the downloaded files, but the selected patterns must include every configuration, tokenizer, and weight file required by that pipeline. Installing the same definition again keeps the installed revision; change `revision` in the configuration to select another version.

Image backends and checkpoints have different capabilities. The selected model must be supported by the installed Diffusers version and by the requested text-to-image or image-to-image operation. A file name alone cannot turn an incompatible model into a compatible pipeline. The built-in SDXL model uses the pinned Pixel Art XL adapter and supports disk streaming; custom Diffusers models use their own pipeline without that adapter and currently use resident model loading. `-weight-storage disk` is unavailable for custom models.

## Choose animation components

The built-in animation workflow uses Wan2.2 I2V with high- and low-noise experts, a text encoder, VAE, two distillation LoRAs, and a pixel adapter. You can provide compatible replacements for any of these roles. Components omitted from the table use the selected pinned profile.

```toml
version = 1

[defaults]
animation = "my-wan"

[models.my-wan]
backend = "wan-i2v"
text_encoder = "q3"
expert_quant = "q2"

[models.my-wan.components.wan_high_q2]
type = "local"
path = "D:/models/high-noise.gguf"

[models.my-wan.components.wan_low_q2]
type = "huggingface"
repo_id = "owner/wan-model"
revision = "main"
filename = "low-noise.gguf"
```

```powershell
pixcli models install my-wan
pixcli animate -input knight.png -prompt "walking" -out walk
```

For `q2` or `q3` experts, use `wan_high_q2` / `wan_low_q2` or `wan_high` / `wan_low` respectively. The text role is `wan_text_q3` for `q3` or `wan_text` for `fp8`. The remaining roles are `wan_vae`, `wan_distill_high`, `wan_distill_low`, and `wan_pixel`. The selected files must match the existing Wan2.2 I2V graph and loader formats; pixcli checks pixel-adapter coverage for both experts. Other video architectures need their own backend and workflow.

## Profiles

The older profile commands remain available:

```powershell
pixcli models download -profile image
pixcli models verify -profile animation
pixcli models download -profile animation-q3-text -cache-dir D:\huggingface-cache
```

| Profile | Contents |
| --- | --- |
| `image` | Pinned SDXL image model, VAE, and adapter. |
| `animation` | Q2 experts and Q3 text encoder, the default animation configuration. |
| `animation-q3-fp8` | Q3 experts and FP8 text encoder. |
| `animation-q3-text` | Q3 experts and Q3 text encoder. |
| `animation-q2-experts` | Q2 experts and FP8 text encoder. |
| `animation-q2-q3-text` | Q2 experts and Q3 text encoder; the same selection as `animation`. |

The shared cache normally lives at `C:\Users\<your-name>\.cache\huggingface\hub` on Windows. Set `HF_HUB_CACHE` or pass the same `-cache-dir PATH` to model-management and inference commands. The pinned files, sources, revisions, sizes, and hashes are listed in the [model manifest](../pixcli/model_manifest.json).
