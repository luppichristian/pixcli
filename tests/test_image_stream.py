import pytest


def test_streamed_preflight_does_not_require_loading_every_weight(tmp_path, monkeypatch):
    from pixcli.image import _memory_preflight
    from pixcli import models
    monkeypatch.setattr(models, 'list_models', lambda profile: {key: {'files': []} for key in ('base', 'vae', 'lora')})
    monkeypatch.setattr('pixcli.image._memory_status', lambda: {'physical_available': 4 * 1024**3, 'commit_available': 4 * 1024**3})
    result = _memory_preflight({'base': tmp_path, 'vae': tmp_path, 'lora': tmp_path}, streaming=True)
    assert result['estimated_commit_needed'] <= 4 * 1024**3


def test_image_device_selection_rejects_impossible_choices(tmp_path):
    from pixcli.image import ImageSettings, validate_image
    with pytest.raises(ValueError, match='device'):
        validate_image(ImageSettings(prompt='x', out=tmp_path/'out.png', device='other'))
    with pytest.raises(ValueError, match='VRAM'):
        validate_image(ImageSettings(prompt='x', out=tmp_path/'out.png', device='cpu', vram_budget_gib=1))


def test_streamed_text_encoder_uses_execution_dtype_even_if_config_requests_half(tmp_path):
    import torch
    from accelerate import init_empty_weights
    from transformers import CLIPTextConfig, CLIPTextModelWithProjection
    from safetensors.torch import save_file
    from pixcli.image_stream import _create_text_encoder, attach_checkpoint_hooks, checkpoint_index
    config = CLIPTextConfig(vocab_size=32, hidden_size=8, intermediate_size=16,
                            projection_dim=8, num_attention_heads=2, num_hidden_layers=1,
                            torch_dtype='float16')
    reference = CLIPTextModelWithProjection(config).float().eval()
    path = tmp_path/'clip.safetensors'
    save_file({key: value.contiguous() for key, value in reference.state_dict().items()}, str(path))
    ids = torch.tensor([[0, 1, 2, 3]])
    with torch.inference_mode():
        expected = reference(ids).text_embeds
    with init_empty_weights():
        model = _create_text_encoder(CLIPTextModelWithProjection, config, torch.float32)
    assert {p.dtype for p in model.parameters()} == {torch.float32}
    attach_checkpoint_hooks(model, checkpoint_index(model, path), 'cpu')
    with torch.inference_mode():
        actual = model(ids).text_embeds
    assert torch.allclose(actual, expected, atol=1e-6)
    assert all(param.is_meta for param in model.parameters())


def test_image_resident_budget_covers_preprocessing_and_export(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from PIL import Image
    from pixcli.image import ImageSettings, run_image
    state = {'active': False, 'entered': False}
    @contextmanager
    def scope(budget):
        assert budget == 3
        state['entered'] = state['active'] = True
        try:
            yield True
        finally:
            state['active'] = False
    def infer(*args):
        assert state['active']
        return Image.new('RGB', (8, 8)), {}
    from pixcli import image
    original_pixelate = image.pixelate
    def pixelate(*args):
        assert state['active']
        return original_pixelate(*args)
    monkeypatch.setattr('pixcli.residency.resident_memory_limit', scope)
    monkeypatch.setattr(image, '_infer', infer)
    monkeypatch.setattr(image, 'pixelate', pixelate)
    monkeypatch.setattr(image, '_memory_status', lambda: {'physical_available': 8 * 1024**3, 'commit_available': 8 * 1024**3})
    monkeypatch.setattr(image, '_resident_memory_bytes', lambda: 1024**3)
    run_image(ImageSettings(prompt='x', out=tmp_path/'out.png', pixel_width=4, pixel_height=4, ram_budget_gib=3), {})
    assert state == {'entered': True, 'active': False}


def test_image_checks_budget_after_postprocessing_before_writing_outputs(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from PIL import Image
    from pixcli import image
    from pixcli.image import ImageSettings, run_image
    output = tmp_path/'out.png'
    resident = [1024**3]
    monkeypatch.setattr('pixcli.residency.resident_memory_limit', lambda budget: nullcontext(False))
    monkeypatch.setattr(image, '_memory_status', lambda: {'physical_available': 8 * 1024**3, 'commit_available': 8 * 1024**3})
    monkeypatch.setattr(image, '_resident_memory_bytes', lambda: resident[0])
    monkeypatch.setattr(image, '_infer', lambda *args: (Image.new('RGB', (8, 8)), {}))
    def pixelate(*args):
        resident[0] = 4 * 1024**3
        return Image.new('RGB', (4, 4))
    monkeypatch.setattr(image, 'pixelate', pixelate)
    with pytest.raises(MemoryError, match='RAM budget'):
        run_image(ImageSettings(prompt='x', out=output, pixel_width=4, pixel_height=4, ram_budget_gib=2), {})
    assert not output.exists()


def test_lazy_checkpoint_hooks_stream_original_weights(tmp_path):
    import torch
    from accelerate import init_empty_weights
    from safetensors.torch import save_file
    from pixcli.image_stream import attach_checkpoint_hooks, checkpoint_index

    class AdapterLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_layer = torch.nn.Linear(3, 2)
            self.lora_A = torch.nn.ModuleDict({'pixel': torch.nn.Linear(3, 1, bias=False)})
            self.lora_B = torch.nn.ModuleDict({'pixel': torch.nn.Linear(1, 2, bias=False)})
            self.dropout = torch.nn.Dropout(p=1.0)
        def forward(self, x):
            return self.dropout(self.base_layer(x) + self.lora_B['pixel'](self.lora_A['pixel'](x)))

    original = AdapterLayer().eval()
    inputs = torch.tensor([[1., 2., 3.]])
    expected = original(inputs).detach()
    base, adapter = tmp_path/'base.safetensors', tmp_path/'adapter.safetensors'
    save_file({'weight': original.base_layer.weight.detach(), 'bias': original.base_layer.bias.detach()}, str(base))
    save_file({'down': original.lora_A['pixel'].weight.detach(), 'up': original.lora_B['pixel'].weight.detach()}, str(adapter))
    with init_empty_weights():
        model = AdapterLayer().train()
    index = checkpoint_index(model, base, adapter, {'lora_A.pixel.weight': 'down', 'lora_B.pixel.weight': 'up'})
    attach_checkpoint_hooks(model, index, 'cpu')
    assert not model.training and not model.dropout.training
    assert all(p.device.type == 'meta' for p in model.parameters())
    assert torch.allclose(model(inputs), expected)
    assert all(p.device.type == 'meta' for p in model.parameters())


def test_lazy_checkpoint_index_rejects_missing_or_wrong_shape(tmp_path):
    import torch
    from accelerate import init_empty_weights
    from safetensors.torch import save_file
    from pixcli.image_stream import checkpoint_index
    with init_empty_weights():
        model = torch.nn.Linear(3, 2)
    path = tmp_path/'bad.safetensors'
    save_file({'weight': torch.ones(3, 3), 'bias': torch.ones(2)}, str(path))
    with pytest.raises(RuntimeError, match='shape'):
        checkpoint_index(model, path)
    save_file({'weight': torch.ones(2, 3)}, str(path))
    with pytest.raises(RuntimeError, match='missing'):
        checkpoint_index(model, path)


def test_lazy_checkpoint_reads_without_safetensors_mapping(tmp_path, monkeypatch):
    import torch
    import safetensors
    from accelerate import init_empty_weights
    from safetensors.torch import save_file
    from pixcli.image_stream import attach_checkpoint_hooks, checkpoint_index
    reference = torch.nn.Linear(4, 2).eval()
    source = torch.randn(1, 4)
    expected = reference(source).detach()
    path = tmp_path/'weights.safetensors'
    save_file(reference.state_dict(), str(path))
    with init_empty_weights():
        model = torch.nn.Linear(4, 2)
    def fail(*args, **kwargs):
        raise AssertionError('whole-file safetensors mapping is forbidden')
    monkeypatch.setattr(safetensors, 'safe_open', fail)
    index = checkpoint_index(model, path)
    attach_checkpoint_hooks(model, index, 'cpu')
    assert torch.allclose(model(source), expected)


@pytest.mark.parametrize('edit', [False, True])
def test_lazy_cpu_sdxl_pipeline_matches_resident_generation_and_edit(tmp_path, edit):
    import numpy as np
    import torch
    from PIL import Image
    from accelerate import init_empty_weights
    from diffusers import (AutoencoderKL, EulerDiscreteScheduler, StableDiffusionXLPipeline,
                           StableDiffusionXLImg2ImgPipeline, UNet2DConditionModel)
    from peft import LoraConfig
    from safetensors.torch import save_file
    from pixcli.image_stream import attach_checkpoint_hooks, checkpoint_index

    def unet():
        model = UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4, layers_per_block=1,
            block_out_channels=(32, 64), down_block_types=('DownBlock2D', 'CrossAttnDownBlock2D'),
            up_block_types=('CrossAttnUpBlock2D', 'UpBlock2D'), cross_attention_dim=32,
            attention_head_dim=4, addition_embed_type='text_time', addition_time_embed_dim=8,
            projection_class_embeddings_input_dim=80, norm_num_groups=8)
        model.add_adapter(LoraConfig(r=4, lora_alpha=4, target_modules=['to_q']))
        return model

    def vae():
        return AutoencoderKL(in_channels=3, out_channels=3, down_block_types=('DownEncoderBlock2D',)*2,
            up_block_types=('UpDecoderBlock2D',)*2, block_out_channels=(32, 64), layers_per_block=1,
            latent_channels=4, norm_num_groups=8, sample_size=16, force_upcast=True)

    def pipeline(u, v):
        cls = StableDiffusionXLImg2ImgPipeline if edit else StableDiffusionXLPipeline
        result = cls(vae=v, text_encoder=None, text_encoder_2=None, tokenizer=None, tokenizer_2=None,
                     unet=u, scheduler=EulerDiscreteScheduler())
        result.set_progress_bar_config(disable=True)
        return result

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(17)
        original_unet, original_vae = unet().eval(), vae().eval()
        for name, param in original_unet.named_parameters():
            if '.lora_B.' in name:
                param.data.fill_(0.1)
        unet_base, unet_adapter = {}, {}
        for name, value in original_unet.state_dict().items():
            if '.lora_A.' in name or '.lora_B.' in name:
                unet_adapter[name] = value.contiguous()
            else:
                unet_base[name.replace('.base_layer.', '.')] = value.contiguous()
        base_file, adapter_file, vae_file = (tmp_path/'unet.safetensors', tmp_path/'adapter.safetensors', tmp_path/'vae.safetensors')
        save_file(unet_base, str(base_file))
        save_file(unet_adapter, str(adapter_file))
        save_file({name: value.contiguous() for name, value in original_vae.state_dict().items()}, str(vae_file))
        kwargs = {'prompt_embeds': torch.randn(1, 4, 32), 'pooled_prompt_embeds': torch.randn(1, 32),
                  'num_inference_steps': 2, 'guidance_scale': 1, 'output_type': 'np'}
        if edit:
            kwargs.update(image=Image.new('RGB', (16, 16), 'gray'), strength=1.0)
        else:
            kwargs.update(height=16, width=16)
        with torch.inference_mode():
            reference = pipeline(original_unet, original_vae)(generator=torch.Generator().manual_seed(42), **kwargs).images
        with init_empty_weights():
            lazy_unet, lazy_vae = unet(), vae()
        attach_checkpoint_hooks(lazy_unet, checkpoint_index(lazy_unet, base_file, adapter_file,
                                {key: key for key in unet_adapter}), 'cpu')
        attach_checkpoint_hooks(lazy_vae, checkpoint_index(lazy_vae, vae_file), 'cpu')
        with torch.inference_mode():
            actual = pipeline(lazy_unet, lazy_vae)(generator=torch.Generator().manual_seed(42), **kwargs).images
        assert np.isfinite(actual).all()
        assert np.allclose(actual, reference, atol=1e-5)
        assert all(param.is_meta for model in (lazy_unet, lazy_vae) for param in model.parameters())
    finally:
        torch.set_num_threads(previous_threads)
