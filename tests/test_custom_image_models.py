import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from pixcli.image import ImageSettings, run_image


@pytest.fixture
def custom_spec():
    return {
        'name': 'custom-art', 'backend': 'diffusers',
        'source': {'type': 'local', 'path': 'model'},
        'provenance': {'path': 'model'}, 'variant': 'fp16',
    }


@pytest.fixture
def fake_diffusers(monkeypatch):
    calls = []
    class Pipe:
        scheduler = SimpleNamespace(config={})

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert os.environ['HF_HUB_OFFLINE'] == '1'
            assert os.environ['TRANSFORMERS_OFFLINE'] == '1'
            calls.append((cls.operation, path, kwargs))
            return cls()

        def enable_vae_slicing(self): pass
        def enable_vae_tiling(self): pass
        def to(self, device): pass
        def __call__(self, *, prompt, negative_prompt, num_inference_steps, guidance_scale,
                     generator, callback_on_step_end, image=None, strength=None, **kwargs):
            if self.operation == 'edit':
                assert image.size == (512, 512)
                assert strength == 0.5
            else:
                assert (kwargs['width'], kwargs['height']) == (512, 512)
            return SimpleNamespace(images=[Image.new('RGB', (512, 512), 'red')])

    class Text(Pipe): operation = 'gen'
    class Edit(Pipe): operation = 'edit'
    class Generator:
        def __init__(self, **kwargs): pass
        def manual_seed(self, seed): return self
    class FakeOOM(Exception): pass
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False, OutOfMemoryError=FakeOOM),
                            float32=object(), Generator=Generator, inference_mode=contextlib.nullcontext)
    diffusers = SimpleNamespace(AutoPipelineForText2Image=Text, AutoPipelineForImage2Image=Edit)
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'diffusers', diffusers)
    monkeypatch.setattr('pixcli.image._memory_status', lambda: {
        'physical_available': 64 * 1024**3, 'commit_available': 64 * 1024**3})
    return calls, Text, Edit


@pytest.mark.parametrize('editing', [False, True])
def test_custom_image_pipeline_runs_offline_and_records_selected_model(tmp_path, monkeypatch, custom_spec, fake_diffusers, editing):
    monkeypatch.setenv('HF_HUB_OFFLINE', 'before')
    monkeypatch.delenv('TRANSFORMERS_OFFLINE', raising=False)
    snapshot = tmp_path / 'snapshot'
    snapshot.mkdir()
    (snapshot / 'weights.safetensors').write_bytes(b'weights')
    kwargs = {}
    if editing:
        source, mask = tmp_path / 'source.png', tmp_path / 'mask.png'
        Image.new('RGB', (2, 1), 'blue').save(source)
        m = Image.new('L', (2, 1), 255)
        m.putpixel((0, 0), 0)
        m.save(mask)
        kwargs.update(input_path=source, mask_path=mask, strength=0.5)
    settings = ImageSettings(prompt='cat', out=tmp_path / 'out.png', inference_width=512,
                             inference_height=512, pixel_width=8, pixel_height=8,
                             save_json=True, **kwargs)
    result = run_image(settings, {'base': snapshot}, custom_spec)
    assert fake_diffusers[0][0][0] == ('edit' if editing else 'gen')
    assert fake_diffusers[0][0][1] == str(snapshot)
    assert fake_diffusers[0][0][2]['local_files_only'] is True
    assert fake_diffusers[0][0][2]['use_safetensors'] is True
    assert fake_diffusers[0][0][2]['variant'] == 'fp16'
    assert result['models']['base'] == {'name': 'custom-art', 'backend': 'diffusers', 'path': 'model',
                                        'variant': 'fp16'}
    assert result['runtime']['adapter_active'] is False
    assert result['runtime']['adapter_coverage'] is None
    assert json.loads((tmp_path / 'out.json').read_text())['models'] == result['models']
    assert os.environ['HF_HUB_OFFLINE'] == 'before'
    assert 'TRANSFORMERS_OFFLINE' not in os.environ
    if editing:
        with Image.open(settings.out) as result_image:
            assert result_image.getpixel((0, 0)) == (0, 0, 255)
            assert result_image.getpixel((1, 0)) == (255, 0, 0)


def test_custom_preflight_counts_actual_safetensors_not_pinned_manifest(tmp_path, monkeypatch, custom_spec):
    from pixcli.image import _memory_preflight
    from pixcli import models
    (tmp_path / 'one.safetensors').write_bytes(b'1234')
    (tmp_path / 'nested').mkdir()
    (tmp_path / 'nested' / 'two.safetensors').write_bytes(b'123456')
    (tmp_path / 'readme.txt').write_bytes(b'not weights')
    monkeypatch.setattr(models, 'list_models', lambda *_: pytest.fail('must not use pinned image manifest'))
    monkeypatch.setattr('pixcli.image._memory_status', lambda: {
        'physical_available': 64 * 1024**3, 'commit_available': 64 * 1024**3})
    result = _memory_preflight({'base': tmp_path}, model_spec=custom_spec)
    assert result['checkpoint_bytes'] == 10


def test_custom_disk_streaming_rejected_before_model_load(tmp_path, custom_spec, fake_diffusers):
    (tmp_path / 'weights.safetensors').write_bytes(b'weights')
    settings = ImageSettings(prompt='cat', out=tmp_path / 'out.png', weight_storage='disk')
    with pytest.raises((ValueError, RuntimeError), match='(?i)disk.*(custom|diffusers)|(?:custom|diffusers).*disk'):
        run_image(settings, {'base': tmp_path}, custom_spec)
    assert not fake_diffusers[0]
    assert not settings.out.exists()


def test_custom_incompatible_pipeline_has_clear_error(tmp_path, monkeypatch, custom_spec, fake_diffusers):
    (tmp_path / 'weights.safetensors').write_bytes(b'weights')
    def fail(*args, **kwargs):
        raise ValueError('Unsupported model architecture')
    monkeypatch.setattr(fake_diffusers[1], 'from_pretrained', fail)
    settings = ImageSettings(prompt='cat', out=tmp_path / 'out.png')
    with pytest.raises(RuntimeError, match='(?i)(unsupported|incompatible|cannot load|could not load)'):
        run_image(settings, {'base': tmp_path}, custom_spec)
    assert not settings.out.exists()


def test_custom_image_rejects_unused_pixel_lora_strength(tmp_path, custom_spec, fake_diffusers):
    (tmp_path / 'weights.safetensors').write_bytes(b'weights')
    settings = ImageSettings(prompt='cat', out=tmp_path / 'out.png', lora_strength=0.7)
    with pytest.raises(ValueError, match='(?i)(lora|adapter)'):
        run_image(settings, {'base': tmp_path}, custom_spec)
    assert not fake_diffusers[0]
    assert not settings.out.exists()


@pytest.mark.parametrize('negative_prompt', ['', 'avoid cats'])
def test_custom_pipeline_without_negative_prompt_never_silently_ignores_it(
        tmp_path, monkeypatch, custom_spec, fake_diffusers, negative_prompt):
    (tmp_path / 'weights.safetensors').write_bytes(b'weights')
    class NoNegative(fake_diffusers[1]):
        def __call__(self, *, prompt, num_inference_steps, guidance_scale, generator,
                     callback_on_step_end, width, height, **kwargs):
            assert 'negative_prompt' not in kwargs
            return SimpleNamespace(images=[Image.new('RGB', (width, height), 'red')])
    monkeypatch.setattr(fake_diffusers[1], 'from_pretrained', lambda *args, **kwargs: NoNegative())
    settings = ImageSettings(prompt='cat', negative_prompt=negative_prompt,
                             out=tmp_path / 'out.png', inference_width=512, inference_height=512)
    if negative_prompt:
        with pytest.raises((ValueError, RuntimeError), match='(?i)negative.prompt'):
            run_image(settings, {'base': tmp_path}, custom_spec)
        assert not settings.out.exists()
    else:
        run_image(settings, {'base': tmp_path}, custom_spec)
        assert settings.out.exists()


def test_custom_legacy_callback_keeps_memory_checks_active(tmp_path, monkeypatch, custom_spec, fake_diffusers):
    (tmp_path / 'weights.safetensors').write_bytes(b'weights')
    class Legacy(fake_diffusers[1]):
        def __call__(self, *, prompt, negative_prompt, num_inference_steps, guidance_scale,
                     generator, callback, callback_steps, width, height):
            assert callback_steps == 1
            callback(0, 0, None)
            return SimpleNamespace(images=[Image.new('RGB', (width, height), 'red')])
    monkeypatch.setattr(fake_diffusers[1], 'from_pretrained', lambda *args, **kwargs: Legacy())
    settings = ImageSettings(prompt='cat', out=tmp_path / 'out.png', inference_width=512, inference_height=512)
    result = run_image(settings, {'base': tmp_path}, custom_spec)
    assert result['runtime']['minimum_sampled_physical_available_bytes'] == 64 * 1024**3
    assert settings.out.exists()


def test_custom_pipeline_rejects_ignored_guidance(tmp_path, monkeypatch, custom_spec, fake_diffusers):
    (tmp_path / 'weights.safetensors').write_bytes(b'weights')
    class NoGuidance(fake_diffusers[1]):
        def __call__(self, *, prompt, negative_prompt, num_inference_steps, generator,
                     callback_on_step_end, width, height, **kwargs):
            return SimpleNamespace(images=[Image.new('RGB', (width, height))])
    monkeypatch.setattr(fake_diffusers[1], 'from_pretrained', lambda *args, **kwargs: NoGuidance())
    settings = ImageSettings(prompt='cat', out=tmp_path / 'out.png')
    with pytest.raises((ValueError, RuntimeError), match='(?i)guidance'):
        run_image(settings, {'base': tmp_path}, custom_spec)
    assert not settings.out.exists()


def test_custom_edit_rejects_ignored_strength(tmp_path, monkeypatch, custom_spec, fake_diffusers):
    (tmp_path / 'weights.safetensors').write_bytes(b'weights')
    source = tmp_path / 'source.png'
    Image.new('RGB', (1, 1)).save(source)
    class NoStrength(fake_diffusers[2]):
        def __call__(self, *, prompt, negative_prompt, num_inference_steps, guidance_scale,
                     generator, callback_on_step_end, image, **kwargs):
            return SimpleNamespace(images=[image])
    monkeypatch.setattr(fake_diffusers[2], 'from_pretrained', lambda *args, **kwargs: NoStrength())
    settings = ImageSettings(prompt='cat', input_path=source, out=tmp_path / 'out.png')
    with pytest.raises((ValueError, RuntimeError), match='(?i)strength'):
        run_image(settings, {'base': tmp_path}, custom_spec)
    assert not settings.out.exists()


def _run_real_tiny_pipeline_probe(tmp_path, editing):
    """Run in a child process so Torch releases its memory before other tests."""
    import torch
    from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
    from transformers.models.clip.tokenization_clip import bytes_to_unicode
    from pixcli import image as image_module

    tokenizer_dir = tmp_path / 'tokenizer'
    tokenizer_dir.mkdir()
    tokens = list(bytes_to_unicode().values())
    tokens += [token + '</w>' for token in tokens]
    tokens += ['<|startoftext|>', '<|endoftext|>']
    (tokenizer_dir / 'vocab.json').write_text(json.dumps({word: i for i, word in enumerate(tokens)}))
    (tokenizer_dir / 'merges.txt').write_text('#version: 0.2\n')
    tokenizer = CLIPTokenizer(str(tokenizer_dir / 'vocab.json'), str(tokenizer_dir / 'merges.txt'),
                              model_max_length=16)
    text = CLIPTextModel(CLIPTextConfig(vocab_size=len(tokens), hidden_size=16, intermediate_size=32,
                                        num_hidden_layers=1, num_attention_heads=2,
                                        max_position_embeddings=16))
    unet = UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4, layers_per_block=1,
        block_out_channels=(16, 32), down_block_types=('DownBlock2D', 'CrossAttnDownBlock2D'),
        up_block_types=('CrossAttnUpBlock2D', 'UpBlock2D'), cross_attention_dim=16,
        attention_head_dim=4, norm_num_groups=4)
    vae = AutoencoderKL(in_channels=3, out_channels=3, down_block_types=('DownEncoderBlock2D',)*2,
        up_block_types=('UpDecoderBlock2D',)*2, block_out_channels=(16, 32), layers_per_block=1,
        latent_channels=4, norm_num_groups=4, sample_size=16)
    pipe = StableDiffusionPipeline(vae=vae, text_encoder=text, tokenizer=tokenizer, unet=unet,
                                   scheduler=DDIMScheduler(steps_offset=1, clip_sample=False), safety_checker=None,
                                   feature_extractor=None, requires_safety_checker=False)
    snapshot = tmp_path / 'snapshot'
    pipe.save_pretrained(snapshot, safe_serialization=True)
    image_module._memory_status = lambda: {
        'physical_available': 64 * 1024**3, 'commit_available': 64 * 1024**3}
    spec = {'name': 'tiny', 'backend': 'diffusers', 'source': {'type': 'local', 'path': str(snapshot)},
            'provenance': {'path': str(snapshot)}}
    settings = ImageSettings(prompt='cat', out=tmp_path / 'out.png', device='cpu', steps=2,
                             inference_width=16, inference_height=16, strength=1.0, guidance=1.0)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result, runtime = image_module._infer(settings, {'base': snapshot}, Image.new('RGB', (16, 16)) if editing else None, spec)
    finally:
        torch.set_num_threads(previous_threads)
    assert result.size == (16, 16)
    assert runtime['adapter_active'] is False


@pytest.mark.parametrize('editing', [False, True])
def test_custom_pipeline_loads_and_infers_with_real_tiny_diffusers_model(tmp_path, editing):
    """Exercise both auto loaders on a locally constructed, real Diffusers pipeline."""
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(tmp_path), str(int(editing))],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + '\n' + result.stderr


if __name__ == '__main__':
    _run_real_tiny_pipeline_probe(Path(sys.argv[1]), bool(int(sys.argv[2])))
