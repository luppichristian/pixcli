# `pixcli models list`

List pinned model repositories, revisions, files, configured model names, and operation defaults. Add `-json` to print the list as JSON. The list includes alternatives as well as the default image and animation models; use `models status` to inspect local availability.

```powershell
pixcli models list [OPTIONS]
```

## Options

| Option | Default | Description |
| --- | --- | --- |
| `-json` | Off | Print the list as JSON. |
| `-config PATH` | `pixcli.toml` | Read configured models from this file. |
| `-help` | | Show command help and exit. |

There is no profile filter for this command. See [model profiles](models.md#profiles) for the available selections.

## Example

```powershell
pixcli models list
```
