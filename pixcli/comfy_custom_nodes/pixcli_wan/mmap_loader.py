"""Experimental GGUF loader retaining read-only mapped weights during offload.

This deliberately bypasses upstream GGUF's eager mmap detachment. The mapped
checkpoint must remain unchanged and accessible until the backend process exits.
"""
from functools import lru_cache


@lru_cache(maxsize=None)
def mmap_patcher_type(gguf_parent, base_patcher):
    """Derive a GGUF patcher that keeps upstream quantization/LoRA behavior."""
    class MappedGGUFModelPatcher(gguf_parent):
        def load(self, *args, force_patch_weights=False, **kwargs):
            if self.weight_inplace_update:
                raise RuntimeError('Mapped GGUF does not permit in-place weight updates')
            # GGUF patching must still be installed even for offloaded layers.
            # Calling the grandparent skips upstream's eager mmap detachment.
            return base_patcher.load(self, *args, force_patch_weights=True, **kwargs)

        def pin_weight_to_device(self, key):
            return base_patcher.pin_weight_to_device(self, key)

        def patch_weight_to_device(self, key, device_to=None, inplace_update=False):
            if self.weight_inplace_update or inplace_update:
                raise RuntimeError('Mapped GGUF does not permit in-place weight updates')
            return super().patch_weight_to_device(key, device_to=device_to, inplace_update=False)

        def clone(self, *args, **kwargs):
            clone = super().clone(*args, **kwargs)
            # Upstream clone deliberately resets the class to GGUFModelPatcher.
            clone.__class__ = type(self)
            return clone

    return MappedGGUFModelPatcher


class PixcliMappedGGUFLoader:
    @classmethod
    def INPUT_TYPES(cls):
        import nodes
        return nodes.NODE_CLASS_MAPPINGS['UnetLoaderGGUF'].INPUT_TYPES()

    RETURN_TYPES = ('MODEL',)
    FUNCTION = 'load_unet'
    CATEGORY = 'pixcli'

    def load_unet(self, unet_name):
        import nodes
        from comfy.model_patcher import ModelPatcher

        loader = nodes.NODE_CLASS_MAPPINGS['UnetLoaderGGUF']()
        model, = loader.load_unet(unet_name)
        parent = type(model)
        if parent.__name__ != 'GGUFModelPatcher' or not issubclass(parent, ModelPatcher):
            raise RuntimeError('Unexpected GGUF patcher; refusing experimental mmap loader')
        if model.weight_inplace_update:
            raise RuntimeError('Mapped GGUF does not permit in-place weight updates')
        model.__class__ = mmap_patcher_type(parent, ModelPatcher)
        return (model,)
