"""Headless Blender 5.3 subprocess driver for one-command `.blend` export."""

from __future__ import annotations

import subprocess
from pathlib import Path

from hfw_toolkit.config import ToolkitConfig

BUILDER_SCRIPT = Path(__file__).resolve().parent / "blender_builder.py"


def build_blend_headless(
    bundle_dir: Path,
    output_blend: Path,
    config: ToolkitConfig | None = None,
    pack_textures: bool = False,
    enable_detail_nodes: bool = True,
    export_fbx: Path | None = None,
) -> Path:
    """Invoke Blender 5.3 in `--background` mode to assemble `bundle_dir` into `output_blend` (and optional rigged `.fbx`)."""
    cfg = config or ToolkitConfig.load()
    if not cfg.blender_exe.exists():
        raise FileNotFoundError(f"Blender executable not found at: {cfg.blender_exe}")

    output_blend = Path(output_blend).resolve()
    output_blend.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(cfg.blender_exe),
        "--background",
        "--factory-startup",
        "--python",
        str(BUILDER_SCRIPT),
        "--",
        str(Path(bundle_dir).resolve()),
        str(output_blend),
    ]
    if pack_textures:
        cmd.append("--pack")
    if not enable_detail_nodes:
        cmd.append("--no-detail")
    if export_fbx is not None:
        cmd.extend(["--export-fbx", str(Path(export_fbx).resolve())])

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not output_blend.exists():
        raise RuntimeError(
            f"Blender headless build failed (exit {proc.returncode}):\n"
            f"STDOUT:\n{proc.stdout[-2000:]}\nSTDERR:\n{proc.stderr[-2000:]}"
        )
    return output_blend
