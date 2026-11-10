# `pixcli models verify`

Check the cached files for a model profile against their pinned sizes and hashes. This command works offline and prints a result for each file. It exits with an error if any file is missing, unreadable, or mismatched.

```powershell
pixcli models verify [OPTIONS]
```

## Options

| Option | Default | Description |
| --- | --- | --- |
| `-profile NAME` | `image` | Select the profile to verify. See [profiles](models.md#profiles) for the available names and model selections. |
| `-cache-dir PATH` | Hugging Face cache | Check a custom cache location; use the same location used when downloading. |
| `-json` | Off | Print the result as JSON. |
| `-help` | | Show command help and exit. |

## Examples

```powershell
pixcli models verify -profile image
pixcli models verify -profile animation
pixcli models verify -profile animation-q3-text -cache-dir D:\huggingface-cache
```

Verification reads the model files and can take time. If files are missing or invalid, connect to the internet and run [`models download`](models-download.md) with the same profile and cache location.
