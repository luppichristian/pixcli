"""Load optional text styles and combine them with user prompts."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Style:
    prompt: str
    palette: str | None = None


def load_style(name: str | None) -> Style | None:
    if name is None:
        return None
    if not name or name in ('.', '..') or any(char in name for char in '/\\:\0'):
        raise ValueError('style must be a name in the styles directory')
    filename = name if name.lower().endswith('.txt') else name + '.txt'
    path = Path('styles') / filename
    try:
        value = path.read_text(encoding='utf-8').strip()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f'Could not read style {name!r} from {path}: {exc}') from exc
    if not value:
        raise ValueError(f'Style {name!r} is empty: {path}')
    first, separator, rest = value.partition('\n')
    palette = None
    if first.startswith('palette:'):
        palette = first[len('palette:'):].strip()
        suffix = Path(palette).suffix.lower()
        if (not palette or palette in ('.', '..') or any(char in palette for char in '/\\:\0')
                or (suffix and suffix not in ('.hex', '.gpl', '.pal', '.txt'))):
            raise ValueError(f'Style {name!r} has an invalid palette name')
        value = rest.strip() if separator else ''
        if not value:
            raise ValueError(f'Style {name!r} has an empty prompt: {path}')
    return Style(prompt=value, palette=palette)


def styled_prompt(prompt: str, style_prompt: str | None) -> str:
    return f'{prompt}, {style_prompt}' if style_prompt else prompt
