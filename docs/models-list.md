# `pixcli models list`

List all pinned model repositories, revisions, and files as JSON. The list includes alternatives as well as the default image and animation models; it does not indicate which files are already downloaded. Use [`models verify`](models-verify.md) or [`doctor`](doctor.md) to inspect local availability.

```powershell
pixcli models list [OPTIONS]
```

## Options

| Option | Default | Description |
| --- | --- | --- |
| `--help` | | Show command help and exit. |

There is no profile filter for this command. See [model profiles](models.md#profiles) for the available selections.

## Example

```powershell
pixcli models list
```
