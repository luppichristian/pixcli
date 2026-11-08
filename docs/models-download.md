# `pixcli models download`

Download and verify the pinned files for a model profile from Hugging Face. The command checks disk space, reuses valid cached files, and verifies file sizes and hashes. Rerun it to resume an interrupted download or repair invalid files.

```powershell
pixcli models download [OPTIONS]
```

The first download can require substantial disk space and an internet connection. Some models may require you to accept their terms and sign in to Hugging Face. The default image and animation profiles are already downloaded by `setup.ps1`.

## Options

| Option | Default | Description |
| --- | --- | --- |
| `--profile NAME` | `image` | Select the model profile to download. See [profiles](models.md#profiles) for all names and their model selections. |
| `--cache-dir PATH` | Hugging Face cache | Download to a custom cache. Use the same location for verification and inference. |
| `--offline` / `--no-offline` | `--no-offline` | `--offline` disallows downloads and exits with an error. Use [`models verify`](models-verify.md) to inspect cached files offline. |
| `--help` | | Show command help and exit. |

## Examples

```powershell
pixcli models download --profile image
pixcli models download --profile animation
pixcli models download --profile animation-q3-text --cache-dir D:\huggingface-cache
```

If access is denied, visit the relevant model page, accept its terms if required, run `hf auth login`, and retry. You can also set `HF_TOKEN` in your environment.
