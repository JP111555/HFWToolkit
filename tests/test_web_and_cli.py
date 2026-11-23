"""Phase 5 tests: FastAPI endpoints, texture PNG streaming, material diff, CLI commands & Blender 5.3 Add-on registration."""

from __future__ import annotations

import json
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import uvicorn

from hfw_toolkit.config import ToolkitConfig
from hfw_toolkit.web import create_app


def _get_json(url: str):
    with urllib.request.urlopen(url, timeout=60) as resp:
        assert resp.status == 200
        return json.loads(resp.read().decode("utf-8"))


def _get_bytes(url: str) -> tuple[bytes, str]:
    with urllib.request.urlopen(url, timeout=60) as resp:
        assert resp.status == 200
        return resp.read(), resp.headers.get("Content-Type", "")


def test_web_app_endpoints() -> None:
    cfg = ToolkitConfig.load()
    app = create_app(cfg)
    port = 18421
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    try:
        base = f"http://127.0.0.1:{port}"
        # Wait for server socket
        for _ in range(40):
            try:
                _get_json(f"{base}/api/stats")
                break
            except Exception:
                time.sleep(0.25)

        html, ctype = _get_bytes(f"{base}/")
        assert "HFWToolkit" in html.decode("utf-8")

        st = _get_json(f"{base}/api/stats")
        assert st["catalog"]["machines"] >= 20
        assert st["dxcompiler_available"] is True
        assert st["rtti_type_count"] >= 12000

        doc = _get_json(f"{base}/api/doctor")
        assert doc["ready"] is True
        assert doc["rtti_ok"] is True

        ms = _get_json(f"{base}/api/machines?q=Slaughterspine")
        assert len(ms["items"]) >= 1
        sp_eid = f"{ms['items'][0]['group_id']}:{ms['items'][0]['object_index']}"

        mach = _get_json(f"{base}/api/machine/{sp_eid}?variant=0")
        assert mach["display_name"] == "Slaughterspine"
        assert len(mach["parts"]) >= 50
        assert len(mach["materials"]) >= 10

        mat_id = mach["materials"][0]["material_id"]
        rep = _get_json(f"{base}/api/material/{mat_id}")
        assert len(rep["explainer"]["cards"]) >= 3
        assert len(rep["decima_graph"]["nodes"]) >= 4
        assert len(rep["decima_detailed_graph"]["nodes"]) >= 10
        assert len(rep["unreal_graph"]["nodes"]) >= 4
        assert "Begin Object" in rep["ue5_clipboard_t3d"]

        # Stream a real decoded texture PNG (rgba & normal_dx)
        first_tex = next(t["resource_id"] for t in rep["primary_textures"] if t.get("resource_id"))
        png, ptype = _get_bytes(f"{base}/api/texture/{first_tex}/image?mode=rgba&max_dim=256")
        assert "image/png" in ptype and png[:8] == b"\x89PNG\r\n\x1a\n"
        npng, _ = _get_bytes(f"{base}/api/texture/{first_tex}/image?mode=normal_dx&max_dim=256")
        assert npng[:8] == b"\x89PNG\r\n\x1a\n"

        # Test TextureSet search & diff endpoint
        tss = _get_json(f"{base}/api/texturesets?q=plasmaspine")
        assert len(tss["items"]) >= 5
        if len(mach["materials"]) >= 2:
            m2 = mach["materials"][1]["material_id"]
            diff = _get_json(f"{base}/api/diff?a={mat_id}&b={m2}")
            assert "textures" in diff and "variables" in diff
    finally:
        server.should_exit = True
        th.join(timeout=5)


def test_blender_addon_registers_in_blender53() -> None:
    cfg = ToolkitConfig.load()
    addon_file = Path(__file__).resolve().parent.parent / "hfw_toolkit" / "blender" / "addon" / "__init__.py"
    code = (
        "import importlib.util, bpy; "
        f"spec = importlib.util.spec_from_file_location('hfw_toolkit_addon', r'{addon_file}'); "
        "mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); "
        "mod.register(); "
        "assert hasattr(bpy.types.Scene, 'hfw_toolkit'); "
        "mod.unregister(); "
        "print('ADDON_OK')"
    )
    proc = subprocess.run(
        [str(cfg.blender_exe), "--background", "--factory-startup", "--python-expr", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Blender Add-on failed:\n{proc.stderr[-1000:]}"
    assert "ADDON_OK" in proc.stdout


def test_cli_machines() -> None:
    venv_cli = Path(__file__).resolve().parent.parent / ".venv" / "Scripts" / "hfw-toolkit.exe"
    proc = subprocess.run([str(venv_cli), "machines", "Slitherfang"], capture_output=True, text=True, check=False)
    assert proc.returncode == 0
    assert "Slitherfang" in proc.stdout


def test_cli_doctor() -> None:
    venv_cli = Path(__file__).resolve().parent.parent / ".venv" / "Scripts" / "hfw-toolkit.exe"
    proc = subprocess.run([str(venv_cli), "doctor"], capture_output=True, text=True, check=False)
    assert proc.returncode == 0
    assert "12,173 Decima types ready" in proc.stdout
    assert "Ready!" in proc.stdout
