# `pixcli doctor`

Report system details and whether the required models and animation backend appear to be available locally. The JSON report includes the operating system, Python and package versions, GPU and CUDA availability, VRAM, system memory, free disk space, the model cache, and local model paths or errors.

```powershell
pixcli doctor [OPTIONS]
```

The backend status is an installation check; it does not run inference or establish that the backend works. Model availability checks do not verify file hashes. Use [`models verify`](models-verify.md) for that.

## Options

| Option | Default | Description |
| --- | --- | --- |
| `--cache-dir PATH` | Hugging Face cache | Check models and free disk space at a custom cache location. Use the same path when downloading and running jobs. |
| `--help` | | Show command help and exit. |

## Examples

```powershell
pixcli doctor
pixcli doctor --cache-dir D:\huggingface-cache
```

If `cuda_available` is false, check your NVIDIA driver and installation. The cache path shown in the report is the one used for the check.
