# Configurable Models Implementation Plan

**Goal:** Let users select compatible image-generation, editing, and animation models from a configuration file, install Hugging Face assets, and remove pixcli-managed downloads, while keeping the existing models and behavior as defaults.

**Architecture:** A versioned TOML catalog describes named model selections and operation defaults. A model store resolves either local user-owned paths or explicitly installed Hugging Face assets; existing pinned profiles remain supported. Image selections use compatible Diffusers pipelines, while animation selections describe the seven roles required by the existing Wan2.2 I2V workflow. Actual selections, rather than built-in manifest values, flow into preflight checks and output metadata.

**Tech stack:** Python 3.11, Typer, `tomllib`, the existing pinned Diffusers/Hugging Face Hub and ComfyUI backends, pytest.

## Decisions and alternatives

- Keep the current pinned models and profile options as the no-configuration default. A configuration file may independently select named generation, editing and animation models; CLI overrides make a one-off selection possible.
- Support Hugging Face repositories and local assets. Install is explicit; inference stays offline. Unsupported pipeline architectures, layouts and animation workflows fail clearly. An arbitrary checkpoint cannot be made compatible just by changing its name.
- Use a pixcli-owned model store for new installs, leaving user-provided local assets and the existing shared Hugging Face cache untouched by normal uninstall. Existing downloads remain usable. An alternative that directly deletes Hub cache revisions risks breaking other programs or shared profiles.
- Preserve the existing optimized SDXL route and verified Wan adapter coverage. Compatible general Diffusers pipelines use a separate route, and unsupported disk streaming is rejected rather than claimed to work. Wan remains a seven-component, two-expert graph; a different video architecture requires its own workflow integration.
- Keep immutable pinned-default hash verification. For custom models, report the source and resolved revision and distinguish any locally recorded integrity checks from independent authentication of upstream contents.

## Tasks

- [x] Define and validate the TOML catalog, default resolution, local path behavior, and compatible model/source schemas. Test absent, invalid, duplicate, incompatible and relative-path configurations.
- [x] Add explicit install, status, offline resolution, verification and uninstall for named models, preserving shared or externally owned files and handling interrupted installations safely. Test lifecycle, corruption, shared-cache fallback, isolation and path containment.
- [x] Integrate compatible Diffusers model selection into generation and editing, with correct device/offline flags, memory preflight, masks, and actual model metadata. Test the loader against a small real local pipeline as well as errors and default regressions.
- [x] Pass configured compatible Wan assets through graph construction and checks, retain both expert-stage LoRA coverage and existing defaults, and record the actual model selection. Test model shadowing, custom names and default graphs; if custom node sources change, refresh and smoke-test the backend without changing its pinned dependencies.
- [x] Expose configuration and model selection on the CLI, add install/uninstall commands and useful installed status, and document complete local and Hub examples. Preserve `models download`, `models verify`, `-cache-dir` and existing profiles. Provide a setup option to skip the default downloads.
- [x] Run focused tests, the full relevant suite, package build, CLI smoke tests and backend startup validation where applicable; inspect the diff and report what was and was not tested.

## Review focus

- A local source may never be deleted by uninstall; managed-path traversal, symlinks and unfinished installs must not escape the store.
- Multiple profiles and other programs can share the old Hub cache; normal removal must not claim to reclaim or delete those files.
- Local-only inference must not access the network or silently switch models, device, storage mode, or generation settings.
- A model may support text-to-image but not image-to-image; reject unsupported operations instead of reporting a false capability.
- Custom Wan assets must reach the correct graph nodes while preserving location checks and coverage for both experts; model registration alone is not evidence of successful inference.
