"""Strict pixel adapter coverage checks, independent of the inference runtime."""
from __future__ import annotations


def adapter_targets(adapter: dict, model: dict) -> set[str]:
    """Check that every A/B pair targets one correctly shaped model matrix.

    Values are shapes. Safetensors metadata must be excluded before calling.
    """
    suffix_a = '.lora_A.weight'
    suffix_b = '.lora_B.weight'
    unexpected = sorted(k for k in adapter if not k.endswith((suffix_a, suffix_b)))
    if unexpected:
        raise ValueError(f'Unexpected adapter tensors: {unexpected[:5]}')
    a_keys = {k[:-len(suffix_a)] for k in adapter if k.endswith(suffix_a)}
    b_keys = {k[:-len(suffix_b)] for k in adapter if k.endswith(suffix_b)}
    if not a_keys or a_keys != b_keys:
        raise ValueError(f'Missing adapter pair tensors: {sorted(a_keys ^ b_keys)[:5]}')
    result = set()
    for prefix in sorted(a_keys):
        target = prefix + '.weight'
        if target not in model:
            raise ValueError(f'Missing model target: {target}')
        a, b, expected = tuple(adapter[prefix + suffix_a]), tuple(adapter[prefix + suffix_b]), tuple(model[target])
        if len(a) != 2 or len(b) != 2 or len(expected) != 2 or b[1] != a[0] or (b[0], a[1]) != expected:
            raise ValueError(f'Incompatible adapter shape for {target}: A={a} B={b} model={expected}')
        result.add(target)
    return result
