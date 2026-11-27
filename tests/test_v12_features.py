"""Comprehensive verification tests for HFWToolkit v1.2 features."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from hfw_toolkit.blender.bundle import MACHINE_VARIANT_PROFILES, export_assembly_bundle
from hfw_toolkit.blender.control_rig import analyze_rig_chains
from hfw_toolkit.config import ToolkitConfig
from hfw_toolkit.decima.catalog import MACHINE_CODENAMES
from hfw_toolkit.shaders.character_dxil import (
    infer_character_eye_color,
    infer_character_hair_color,
    infer_character_skin_tone,
)
from hfw_toolkit.updater import (
    DEFAULT_UPDATE_BRANCH,
    _is_windows_store_stub,
    _parse_version_tuple,
    detect_system_pythons,
    get_local_version_info,
)


def test_machine_codename_mapping_glider_and_swooper():
    """Verify Glinthawk = glider and Skydrifter = swooper (fixing the swapped mapping)."""
    assert MACHINE_CODENAMES["glider"] == "Glinthawk"
    assert MACHINE_CODENAMES["swooper"] == "Skydrifter"


def test_eye_and_muscle_state_decoupling():
    """Overridden machines change ONLY optics (Cyan); muscles stay Standard Amber."""
    std = MACHINE_VARIANT_PROFILES["standard"]
    ovr = MACHINE_VARIANT_PROFILES["overridden"]
    apx = MACHINE_VARIANT_PROFILES["apex"]
    assert ovr["muscle_color"] == std["muscle_color"]
    assert ovr["eye_color"] != std["eye_color"]
    assert apx["muscle_color"] != std["muscle_color"]


def test_character_authentic_iris_colors():
    """Aloy gets green irises; Seyka/Kotallo/Zo/Sylens get dark brown; Erend gets hazel; Tilda gets ice-blue."""
    aloy = infer_character_eye_color("Aloy")
    seyka = infer_character_eye_color("Seyka")
    erend = infer_character_eye_color("Erend")
    tilda = infer_character_eye_color("Tilda")
    assert aloy["preset"] == "aloy_green"
    assert seyka["preset"] == "dark_brown"
    assert erend["preset"] == "warm_hazel"
    assert tilda["preset"] == "zenith_blue"
    # Verify linear RGB differs substantially between Aloy and Seyka
    assert aloy["iris_color_linear"] != seyka["iris_color_linear"]
    assert infer_character_skin_tone("Aloy")["preset"] == "fair_rosy"
    assert infer_character_hair_color("Seyka")["preset"] == "espresso_black"


def test_updater_config_defaults_and_persistence(tmp_path: Path):
    """Auto-update must default to opt-in False on `main`, and round-trip cleanly via save/load."""
    cfg_file = tmp_path / "hfw_toolkit_config.json"
    cfg = ToolkitConfig()
    assert cfg.auto_update is False
    assert cfg.update_branch == DEFAULT_UPDATE_BRANCH
    assert cfg.check_updates_on_startup is True

    cfg.auto_update = True
    cfg.check_updates_on_startup = False
    cfg.save(cfg_file)

    reloaded = ToolkitConfig.load(cfg_file)
    assert reloaded.auto_update is True
    assert reloaded.check_updates_on_startup is False
    assert reloaded.update_branch == "main"


def test_python_runtime_detector_and_helpers():
    """System Python detector filters WindowsApps stubs and returns valid sorted versions >= 3.10."""
    assert _parse_version_tuple("Python 3.14.10") == (3, 14, 10)
    assert _parse_version_tuple("3.12") == (3, 12, 0)
    assert _is_windows_store_stub(Path(r"C:\Users\me\AppData\Local\Microsoft\WindowsApps\python.exe")) is True
    assert _is_windows_store_stub(Path(r"C:\Program Files\Python314\python.exe")) is False

    info = detect_system_pythons(ToolkitConfig().toolkit_dir)
    assert tuple(info["current_version_tuple"][:2]) >= (3, 10)
    versions = [tuple(p["version_tuple"]) for p in info["installed_pythons"]]
    assert versions == sorted(versions, reverse=True)


def test_local_version_info_reports_valid_mode():
    loc = get_local_version_info(ToolkitConfig().toolkit_dir)
    assert loc["install_mode"] in ("git", "zip")
    assert loc["short_sha"]


def test_avian_leg_ik_and_wing_feather_analysis():
    """Synthetic avian skeleton test confirming ground-level talon IK and wing feather collection."""
    joints = [
        {"name": "Root", "parent_name": None, "head": [0.0, 0.0, 0.0]},
        {"name": "C_pelvis_sjnt_0", "parent_name": "Root", "head": [0.0, 0.0, 1.5]},
        {"name": "C_spine_sjnt_0", "parent_name": "C_pelvis_sjnt_0", "head": [0.0, 0.3, 1.6]},
        {"name": "C_chest_sjnt_0", "parent_name": "C_spine_sjnt_0", "head": [0.0, 0.6, 1.7]},
        # Elevated avian leg descending to ground talon hub (Z=0.10m)
        {"name": "L_leg_sjnt_0", "parent_name": "C_pelvis_sjnt_0", "head": [-0.3, 0.0, 1.4]},
        {"name": "L_leg_sjnt_1", "parent_name": "L_leg_sjnt_0", "head": [-0.32, 0.1, 0.95]},
        {"name": "L_leg_sjnt_2", "parent_name": "L_leg_sjnt_1", "head": [-0.32, -0.1, 0.55]},
        {"name": "L_Toe_sjnt_0", "parent_name": "L_leg_sjnt_2", "head": [-0.32, 0.0, 0.10]},
        {"name": "L_Toe_01_sjnt_0", "parent_name": "L_Toe_sjnt_0", "head": [-0.36, 0.18, 0.04]},
        {"name": "L_Toe_01_sjnt_1", "parent_name": "L_Toe_01_sjnt_0", "head": [-0.38, 0.30, 0.02]},
        {"name": "L_Toe_02_sjnt_0", "parent_name": "L_Toe_sjnt_0", "head": [-0.28, 0.18, 0.04]},
        {"name": "L_Toe_02_sjnt_1", "parent_name": "L_Toe_02_sjnt_0", "head": [-0.26, 0.30, 0.02]},
        # Avian wing chain + primary/secondary feathers
        {"name": "L_wing_sjnt_0", "parent_name": "C_chest_sjnt_0", "head": [-0.4, 0.5, 1.7]},
        {"name": "L_wing_sjnt_1", "parent_name": "L_wing_sjnt_0", "head": [-1.2, 0.4, 1.75]},
        {"name": "L_wing_sjnt_2", "parent_name": "L_wing_sjnt_1", "head": [-2.2, 0.3, 1.8]},
        {"name": "L_wing_sjnt_3", "parent_name": "L_wing_sjnt_2", "head": [-3.0, 0.2, 1.82]},
        {"name": "L_wingFeather_01_sjnt_0", "parent_name": "L_wing_sjnt_1", "head": [-1.3, 0.1, 1.72]},
        {"name": "L_wingFeather_02_sjnt_0", "parent_name": "L_wing_sjnt_2", "head": [-2.3, 0.0, 1.76]},
        {"name": "L_wingFeather_03_sjnt_0", "parent_name": "L_wing_sjnt_3", "head": [-3.1, -0.1, 1.78]},
    ]
    spec = analyze_rig_chains(joints)
    assert len(spec["legs"]) == 1
    assert spec["legs"][0]["foot_bone"] == "L_Toe_sjnt_0"
    assert len(spec["wings"]) == 1
    assert len(spec["wings"][0].get("feather_chains", [])) == 3


@pytest.mark.skipif(
    not ToolkitConfig.load().streaming_graph_path.is_file(),
    reason="Requires local Horizon Forbidden West game archives",
)
def test_live_game_machines_variants_sacs_and_characters(tmp_path: Path):
    """End-to-end verification against live HFW game archives (codenames, variants, Apex overrides, liquid sacs, characters)."""
    from hfw_toolkit.decima import AssetCatalog, HFWGame, ObjectId
    from hfw_toolkit.extract import (
        assemble_character,
        assemble_machine,
        resolve_machine_entity_id,
    )

    cfg = ToolkitConfig.load()
    game = HFWGame(cfg)
    cat = AssetCatalog(cfg.index_db_path)
    try:
        # 1. Skydrifter != Glinthawk
        sky_oid, sky_code, _ = resolve_machine_entity_id(game, cat, "Skydrifter")
        gli_oid, gli_code, _ = resolve_machine_entity_id(game, cat, "Glinthawk")
        assert sky_code == "swooper"
        assert gli_code == "glider"
        assert str(sky_oid) != str(gli_oid)

        # 2. Stingspawn has zero Bilegut contamination
        st_oid, _, _ = resolve_machine_entity_id(game, cat, "Stingspawn")
        st_asm = assemble_machine(game, st_oid, display_name="Stingspawn", variant_index=0)
        assert len(st_asm.parts) <= 20
        assert not any("frog" in m.name.lower() for m in st_asm.materials.values())

        # 3. Apex Clawstrider applies genuine *_BVar_* texture overrides
        cw_oid, _, _ = resolve_machine_entity_id(game, cat, "Clawstrider")
        cw_base = assemble_machine(game, cw_oid, display_name="Clawstrider", variant_index=0)
        apex_idx = next(
            (v.variant_index for v in cw_base.available_variants if "apex" in v.label.lower()),
            None,
        )
        assert apex_idx is not None
        cw_apex = assemble_machine(game, cw_oid, display_name="Clawstrider", variant_index=apex_idx)
        assert any("bvar" in m.name.lower() for m in cw_apex.materials.values())

        # 4. Bilegut liquid sacs: exactly true liquid sacs, zero canisters/armor false positives
        bg_asm = assemble_machine(game, ObjectId.parse("2880:1"), display_name="Bilegut", variant_index=0)
        bg_dir = tmp_path / "Bilegut"
        export_assembly_bundle(game, bg_asm, bg_dir, max_texture_size=256)
        bg_meta = json.loads((bg_dir / "assembly.json").read_text(encoding="utf-8"))
        bg_sacs = [
            v["name"]
            for v in bg_meta["materials"].values()
            if (v.get("emissive_overlay") or {}).get("kind") == "elemental_sack_overlay"
        ]
        assert 3 <= len(bg_sacs) <= 6
        assert all("canister" not in n.lower() for n in bg_sacs)
        ast.parse((bg_dir / "ue5_import_machine.py").read_text(encoding="utf-8"))

        # 5. Character shader metadata (Aloy green eyes + Seyka dark brown eyes + UE5 script AST)
        for prompt, exp_eye in [("Aloy Carja Blazon no headgear", "aloy_green"), ("Seyka", "dark_brown")]:
            c_asm = assemble_character(game, catalog=cat, prompt=prompt)
            c_dir = tmp_path / prompt.split()[0]
            export_assembly_bundle(game, c_asm, c_dir, max_texture_size=256, asset_kind="character")
            c_meta = json.loads((c_dir / "assembly.json").read_text(encoding="utf-8"))
            eye_presets = {v.get("eye_profile") for v in c_meta["materials"].values() if v.get("surface_profile") == "eye"}
            assert exp_eye in eye_presets
            ast.parse((c_dir / "ue5_import_character.py").read_text(encoding="utf-8"))
    finally:
        cat.close()
        game.close()


def test_windows_batch_launchers_find_python_and_crlf(tmp_path):
    """Both Windows .bat launchers must use CRLF and cleanly discover Python 3.10+ (even with spaces in path)."""
    import subprocess
    import sys

    if sys.platform != "win32":
        pytest.skip("Windows batch launcher check only applies on Windows")

    root = Path(__file__).resolve().parent.parent
    for fname in ("Launch_HFW_Studio.bat", "Setup_And_Doctor.bat"):
        bat_path = root / fname
        raw = bat_path.read_bytes()
        assert b"\r\n" in raw, f"{fname} must use Windows CRLF line endings"
        assert b"\n" not in raw.replace(b"\r\n", b""), f"{fname} contains bare LF line endings"

        text = raw.decode("ascii")
        idx = text.rfind(":FIND_NEWEST_PYTHON")
        assert idx > 0
        probe_bat = tmp_path / f"probe_{fname}"
        probe_src = (
            "@echo off\r\n"
            "setlocal EnableDelayedExpansion\r\n"
            "call :FIND_NEWEST_PYTHON\r\n"
            "echo BEST_PY=!BEST_PY!\r\n"
            "echo BEST_PY_VER=!BEST_PY_VER!\r\n"
            "echo BEST_SCORE=!BEST_SCORE!\r\n"
            "exit /b 0\r\n"
            + text[idx:]
        )
        probe_bat.write_bytes(probe_src.encode("ascii"))

        res = subprocess.run(
            ["cmd", "/c", str(probe_bat)],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert res.returncode == 0, f"{fname} probe failed: {res.stdout}\n{res.stderr}"
        combined = (res.stdout or "") + "\n" + (res.stderr or "")
        assert "cannot find the path" not in combined.lower(), combined
        assert "not recognized" not in combined.lower(), combined

        kv = dict(line.split("=", 1) for line in res.stdout.splitlines() if "=" in line)
        assert kv.get("BEST_PY") and Path(kv["BEST_PY"]).exists(), f"{fname} failed to find Python: {kv}"
        assert int(kv.get("BEST_SCORE", "0")) >= 31000, f"{fname} score too low: {kv}"

