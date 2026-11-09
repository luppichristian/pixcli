"""Deterministic, sequence-wide processing and export of generated video frames."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable, Sequence

from PIL import Image


def _rgb(image: Image.Image, background: tuple[int, int, int]) -> Image.Image:
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        base = Image.new("RGBA", rgba.size, (*background, 255))
        return Image.alpha_composite(base, rgba).convert("RGB")
    return image.convert("RGB")


def _palette(frames: Sequence[Image.Image], colors: int, palette_image: Image.Image | None) -> Image.Image:
    # One palette derived from the whole sequence, with bounded sampling memory.
    samples = [palette_image.convert("RGB")] if palette_image is not None else list(frames)
    thumbs = []
    for sample in samples:
        thumb = sample.resize((128, 128), Image.Resampling.NEAREST)
        thumbs.append(thumb)
    atlas = Image.new("RGB", (128 * len(thumbs), 128))
    for i, thumb in enumerate(thumbs):
        atlas.paste(thumb, (128 * i, 0))
    return atlas.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)


def export_animation(
    frames: Sequence[Image.Image],
    output_dir: str | Path,
    *,
    final_size: tuple[int, int],
    fps: float,
    palette_colors: int | None = None,
    palette_image: Image.Image | None = None,
    background: tuple[int, int, int] = (255, 255, 255),
    overwrite: bool = False,
    mp4: bool = False,
    memory_guard: Callable[[], None] | None = None,
) -> dict:
    """Export original and processed frames, GIF, sprite sheet and layout JSON.

    Frame processing uses one aspect-preserving fit/padding transform. A fixed palette
    is used across every frame. MP4 is optional and requires imageio-ffmpeg.
    Existing output directories are replaced only if created by this function.
    """
    output = Path(output_dir)
    check = memory_guard or (lambda: None)
    check()
    if not frames:
        raise ValueError("At least one animation frame is required")
    if any(frame.size != frames[0].size for frame in frames):
        raise ValueError("All source frames must have the same dimensions")
    if len(final_size) != 2 or any(not isinstance(x, int) or x <= 0 for x in final_size):
        raise ValueError("Final dimensions must be positive integers")
    if not math.isfinite(fps) or not 0 < fps <= 100:
        raise ValueError("FPS must be greater than zero and no greater than 100")
    if palette_colors is not None and not 2 <= palette_colors <= 256:
        raise ValueError("Palette colors must be between 2 and 256")
    if palette_image is not None and palette_colors is None:
        raise ValueError("A palette image requires palette_colors")
    if output.is_symlink():
        raise ValueError("Output directory may not be a symbolic link")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output}; use --overwrite")
        if not output.is_dir() or not (output / ".pixcli-animation").is_file():
            raise FileExistsError(f"Refusing to replace a directory not created by pixcli: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    backup = None
    try:
        (stage / "original_frames").mkdir()
        (stage / "frames").mkdir()
        (stage / ".pixcli-animation").write_text("pixcli animation output\n", encoding="utf-8")
        originals = [_rgb(frame, background) for frame in frames]
        for i, frame in enumerate(frames):
            frame.save(stage / "original_frames" / f"{i:06d}.png")
            check()
        source_w, source_h = originals[0].size
        dest_w, dest_h = final_size
        scale = min(dest_w / source_w, dest_h / source_h)
        scaled = (max(1, min(dest_w, round(source_w * scale))),
                  max(1, min(dest_h, round(source_h * scale))))
        offset = ((dest_w - scaled[0]) // 2, (dest_h - scaled[1]) // 2)
        processed = []
        for frame in originals:
            canvas = Image.new("RGB", final_size, background)
            canvas.paste(frame.resize(scaled, Image.Resampling.NEAREST), offset)
            processed.append(canvas)
            check()
        global_palette = _palette(processed, palette_colors or 256, palette_image)
        if palette_colors is not None:
            processed = [frame.quantize(palette=global_palette, dither=Image.Dither.NONE) for frame in processed]
        for i, frame in enumerate(processed):
            frame.save(stage / "frames" / f"{i:06d}.png")
            check()
        sheet = Image.new("RGB", (dest_w * len(processed), dest_h))
        for i, frame in enumerate(processed):
            sheet.paste(frame.convert("RGB"), (i * dest_w, 0))
        sheet.save(stage / "sheet.png")
        gif_frames = [f.convert("RGB").quantize(palette=global_palette, dither=Image.Dither.NONE) for f in processed]
        gif_duration_ms = max(10, round(100 / fps) * 10)
        gif_frames[0].save(stage / "preview.gif", save_all=True, append_images=gif_frames[1:],
                           duration=gif_duration_ms, loop=0, optimize=False, disposal=2)
        if mp4:
            try:
                import imageio.v2 as imageio
                import numpy as np
                import imageio_ffmpeg  # noqa: F401
            except ImportError as exc:
                raise RuntimeError("MP4 export requires imageio[ffmpeg] and numpy") from exc
            pixel_format = "yuv420p" if dest_w % 2 == 0 and dest_h % 2 == 0 else "yuv444p"
            with imageio.get_writer(stage / "preview.mp4", fps=fps, macro_block_size=1,
                                    pixelformat=pixel_format) as writer:
                for frame in processed:
                    writer.append_data(np.asarray(frame.convert("RGB")))
                    check()
        check()
        info = {
            "fps": fps,
            "gif_duration_ms": gif_duration_ms,
            "frame_count": len(processed),
            "frames": [{"file": f"frames/{i:06d}.png", "x": i * dest_w, "y": 0,
                        "width": dest_w, "height": dest_h, "duration_ms": 1000 / fps}
                       for i in range(len(processed))],
            "transform": {"source_size": [source_w, source_h], "scaled_size": list(scaled),
                          "final_size": [dest_w, dest_h], "paste_offset": list(offset),
                          "resampling": "nearest", "background": list(background)},
            "palette_colors": palette_colors,
            "palette_source": "image" if palette_image is not None else ("sequence" if palette_colors else None),
        }
        (stage / "sheet.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        check()
        if output.exists():
            backup = Path(tempfile.mkdtemp(prefix=f".{output.name}-backup-", dir=output.parent))
            backup.rmdir()
            os.replace(output, backup)
        try:
            os.replace(stage, output)
        except BaseException:
            if backup is not None:
                os.replace(backup, output)
                backup = None
            raise
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
        return {"frames": str(output / "frames"), "original_frames": str(output / "original_frames"),
                "gif": str(output / "preview.gif"), "sheet": str(output / "sheet.png"),
                "sheet_json": str(output / "sheet.json"), "mp4": str(output / "preview.mp4") if mp4 else None,
                "transform": info["transform"]}
    finally:
        if stage.exists():
            shutil.rmtree(stage)
