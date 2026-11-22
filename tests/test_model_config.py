from pathlib import Path

import pytest


def test_no_config_preserves_builtin_defaults(tmp_path, monkeypatch):
    from pixcli.model_config import load_config, select_model

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('PIXCLI_CONFIG', raising=False)
    config = load_config()
    assert select_model(config, 'generation')['name'] == 'default-image'
    assert select_model(config, 'editing')['backend'] == 'builtin-image'
    assert select_model(config, 'animation')['backend'] == 'builtin-animation'


def test_config_resolves_local_paths_relative_to_file(tmp_path, monkeypatch):
    from pixcli.model_config import load_config, select_model

    config_path = tmp_path / 'pixcli.toml'
    config_path.write_text('''version = 1
[defaults]
generation = "sprite"
editing = "sprite"
animation = "motion"
[models.sprite]
backend = "diffusers"
variant = "fp16"
[models.sprite.source]
type = "local"
path = "./assets/sprite"
[models.motion]
backend = "wan-i2v"
text_encoder = "q3"
expert_quant = "q2"
[models.motion.components.wan_high_q2]
type = "local"
path = "./assets/high.gguf"
''', encoding='utf-8')
    monkeypatch.chdir(tmp_path.parent)
    config = load_config(config_path)
    image = select_model(config, 'editing')
    animation = select_model(config, 'animation')
    assert image['source']['path'] == str((tmp_path / 'assets/sprite').resolve())
    assert image['variant'] == 'fp16'
    assert animation['components']['wan_high_q2']['path'] == str((tmp_path / 'assets/high.gguf').resolve())


def test_discovery_and_explicit_missing_config(tmp_path, monkeypatch):
    from pixcli.model_config import ConfigError, load_config, select_model

    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / 'pixcli.toml'
    cfg.write_text('version = 1\n[defaults]\nediting = "default-image"\n', encoding='utf-8')
    assert select_model(load_config(), 'editing')['name'] == 'default-image'
    with pytest.raises(ConfigError, match='does not exist'):
        load_config(tmp_path / 'missing.toml')
    monkeypatch.setenv('PIXCLI_CONFIG', str(tmp_path / 'missing.toml'))
    with pytest.raises(ConfigError, match='does not exist'):
        load_config()


def test_select_model_rejects_incompatible_backend(tmp_path):
    from pixcli.model_config import ConfigError, select_model

    config = load_config_from(tmp_path, '''
[models.still]
backend = "diffusers"
[models.still.source]
type = "local"
path = "assets"
''')
    with pytest.raises(ConfigError, match='animation'):
        select_model(config, 'animation', 'still')
    with pytest.raises(ConfigError, match='generation'):
        select_model(config, 'generation', 'default-animation')


def load_config_from(tmp_path: Path, body: str):
    from pixcli.model_config import load_config

    path = tmp_path / 'config.toml'
    path.write_text('version = 1\n' + body, encoding='utf-8')
    return load_config(path)


@pytest.mark.parametrize('body, message', [
    ('version = 2\n', 'version'),
    ('version = 1\n[models.default-image]\nbackend = "diffusers"\n', 'reserved'),
    ('version = 1\n[models.x]\nbackend = "unknown"\n', 'backend'),
    ('version = 1\n[models.x]\nbackend = "diffusers"\n[models.x.source]\ntype = "huggingface"\nrepo_id = "../outside"\nrevision = "main"\n', 'repo_id'),
    ('version = 1\n[models.x]\nbackend = "wan-i2v"\n[models.x.components.wan_vae]\ntype = "huggingface"\nrepo_id = "owner/repo"\nrevision = "main"\nfilename = "../escape"\n', 'filename'),
])
def test_invalid_configs_rejected(tmp_path, body, message):
    from pixcli.model_config import ConfigError, load_config

    path = tmp_path / 'config.toml'
    path.write_text(body, encoding='utf-8')
    with pytest.raises(ConfigError, match=message):
        load_config(path)


def test_huggingface_root_repository_names_are_supported(tmp_path):
    from pixcli.model_config import select_model

    config = load_config_from(tmp_path, '''
[models.root]
backend = "diffusers"
[models.root.source]
type = "huggingface"
repo_id = "example-model"
revision = "main"
''')
    assert select_model(config, 'generation', 'root')['source']['repo_id'] == 'example-model'
