"""v9 Regression Tests:
- Pillar 1: Machine muscle glow DXIL*40 fix, wet liquid/glass shaders, Apex/Standard variant profiles, HFW_MachineShaderControl & UE5 MPC
- Pillar 2: Full HFW Character support (Aloy outfits + headgear + braided hair + cinematic face, companions, NPCs, prompt resolver, unified poseable rig, 8-layer outfit shader)
- Pillar 3: General Asset support & Guerrilla Developer-Directory Asset Catalog (Weapons, Wildlife, Building Blocks, Eco Trees, Web API & CLI)
"""

from __future__ import annotations

import ast
import json
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pytest
import uvicorn

from hfw_toolkit.blender import build_blend_headless, export_assembly_bundle
from hfw_toolkit.blender.bundle import infer_machine_variant_profile
from hfw_toolkit.config import ToolkitConfig
from hfw_toolkit.decima import AssetCatalog, HFWGame
from hfw_toolkit.extract import (
    assemble_character,
    assemble_general_asset,
    assemble_machine,
    resolve_character_prompt,
    resolve_machine_entity_id,
)
from hfw_toolkit.web import create_app


@pytest.fixture(scope="module")
def hfw_env():
    cfg = ToolkitConfig.load()
    game = HFWGame(cfg)
    cat = AssetCatalog(cfg.index_db_path)
    if not cat.is_characters_assets_built():
        cat.index_characters_and_assets(game)
    yield cfg, game, cat
    cat.close()
    game.close()


# ==============================================================================
# PILLAR 1 — MACHINE SHADERS, VARIANT PALETTES, LIQUIDS & UE5 MPC
# ==============================================================================


def test_machine_variant_profiles_authentic() -> None:
    std = infer_machine_variant_profile("raider Clawstrider", "Standard Melee")
    apx = infer_machine_variant_profile("raider Clawstrider", "Apex Clawstrider")
    cor = infer_machine_variant_profile("warrobot Corruptor", "Corrupted")
    ovr = infer_machine_variant_profile("horse Charger", "Overridden")

    assert std["profile"] == "standard"
    assert std["muscle_color"][0] > std["muscle_color"][2]  # Warm amber (R > B)
    assert std["eye_color"][2] > std["eye_color"][0]        # Patrol blue (B > R)

    assert apx["profile"] == "apex"
    assert apx["muscle_color"][2] > 0.8 and apx["muscle_color"][0] > 0.4  # Apex purple-violet!
    assert apx["eye_color"][2] > 0.8 and apx["eye_color"][0] > 0.5        # Apex purple eyes!

    assert cor["profile"] == "corrupted"
    assert cor["muscle_color"][0] > 0.85 and cor["muscle_color"][1] < 0.15

    assert ovr["profile"] == "overridden"
    # Overridden machines keep Standard Amber muscles and change ONLY their optical sensors to Cyan!
    assert ovr["muscle_color"] == std["muscle_color"]
    assert ovr["eye_color"][2] > 0.85 and ovr["eye_color"][1] > 0.65


def test_apex_machine_shader_controls_and_ue5_mpc(hfw_env, tmp_path: Path) -> None:
    cfg, game, cat = hfw_env
    oid, code, disp = resolve_machine_entity_id(game, cat, "Clawstrider")
    # Pick Apex variant if available (Clawstrider variant 3 is Apex)
    asm = assemble_machine(game, oid, codename=code, display_name=disp, variant_index=3, lod_index=0)
    bdir = tmp_path / "clawstrider_apex_bundle"
    export_assembly_bundle(game, asm, bdir, unpack_textures=False, asset_kind="machine")

    manifest = json.loads((bdir / "assembly.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] >= 3
    assert manifest["asset_kind"] == "machine"
    sc = manifest["shader_controls"]
    assert sc["profile"] == "apex"
    assert sc["muscle_color"][2] > 0.8  # Apex purple muscle default!

    # Verify UE5 script uses MPC_HFW_MachineControl & parses cleanly
    ue_path = bdir / "ue5_import_machine.py"
    assert ue_path.exists()
    ue_src = ue_path.read_text(encoding="utf-8")
    ast.parse(ue_src)
    assert "MPC_HFW_MachineControl" in ue_src
    assert "MaterialParameterCollection" in ue_src
    assert "IS_SKELETAL = True" in ue_src

    # Build in Blender 5.3 headless & verify HFW_MachineShaderControl + DXIL*40 muscle shader
    out_blend = tmp_path / "clawstrider_apex.blend"
    build_blend_headless(bdir, out_blend, config=cfg)
    assert out_blend.exists()

    verify_code = (
        f"import bpy; bpy.ops.wm.open_mainfile(filepath=r'{out_blend}'); "
        "ng = bpy.data.node_groups.get('HFW_MachineShaderControl'); "
        "assert ng is not None, 'Missing HFW_MachineShaderControl'; "
        "mc = next(n for n in ng.nodes if n.label == 'Muscle Color').outputs[0].default_value; "
        "assert mc[2] > 0.8 and mc[0] > 0.4, f'Expected Apex purple muscle color, got {tuple(mc)}'; "
        "print('BLENDER_SHADER_CTRL_OK')"
    )
    proc = subprocess.run(
        [str(cfg.blender_exe), "--background", "--python-exit-code", "1", "--python-expr", verify_code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    assert "BLENDER_SHADER_CTRL_OK" in proc.stdout


def test_elemental_liquid_shader_ue5_flowmap(hfw_env, tmp_path: Path) -> None:
    cfg, game, cat = hfw_env
    oid, code, disp = resolve_machine_entity_id(game, cat, "Bellowback")
    asm = assemble_machine(game, oid, codename=code, display_name=disp, variant_index=0, lod_index=0)
    bdir = tmp_path / "bellowback_bundle"
    export_assembly_bundle(game, asm, bdir, unpack_textures=False, asset_kind="machine")

    manifest = json.loads((bdir / "assembly.json").read_text(encoding="utf-8"))
    sacks = [m for m in manifest["materials"].values() if (m.get("emissive_overlay") or {}).get("kind") == "elemental_sack_overlay"]
    assert len(sacks) >= 1
    assert sacks[0]["emissive_overlay"]["flow_file"] != ""
    assert sacks[0]["emissive_overlay"]["fluid_file"] != ""

    ue_src = (bdir / "ue5_import_machine.py").read_text(encoding="utf-8")
    ast.parse(ue_src)
    assert "FlowMap" in ue_src and "BLEND_TRANSLUCENT" in ue_src


# ==============================================================================
# PILLAR 2 — CHARACTERS, OUTFITS, HAIR, PROMPT RESOLVER & UNIFIED RIG
# ==============================================================================


def test_character_catalog_and_prompt_resolver(hfw_env) -> None:
    _cfg, _game, cat = hfw_env
    st = cat.stats()
    assert st["characters"] >= 1000
    assert st["character_outfits"] >= 60
    assert st["character_heads"] >= 200

    aloy = cat.get_character("aloy")
    assert aloy is not None
    assert aloy["character_kind"] == "hero"
    assert aloy["facial_id"] == "48346:0"
    assert aloy["hair_id"] == "43305:37"
    aloy_outfits = cat.list_character_outfits("aloy", limit=100)
    assert len(aloy_outfits) >= 30

    # Key companions all present with authentic genders!
    for cid, expected_gender in [("erend", "male"), ("seyka", "female"), ("kotallo", "male"), ("alva", "female"), ("regalla", "female")]:
        c = cat.get_character(cid)
        assert c is not None, f"Missing companion {cid}"
        assert c["gender"] == expected_gender

    # Natural-language prompt resolution
    r1 = resolve_character_prompt(cat, "Aloy Carja Blazon no headgear")
    assert r1["matched"] is True
    assert r1["character"]["char_id"] == "aloy"
    assert "Carja Blazon" in r1["outfit"]["display_name"]
    assert r1["headgear"] is False
    assert r1["include_hair"] is True

    r2 = resolve_character_prompt(cat, "Aloy")
    assert r2["character"]["char_id"] == "aloy"
    assert "Nora Anointed" in r2["outfit"]["display_name"]
    assert r2["headgear"] is True

    r3 = resolve_character_prompt(cat, "Tenakth female warrior")
    assert r3["matched"] is True
    assert r3["character"]["tribe"] == "Tenakth"
    assert r3["character"]["gender"] == "female"


def test_aloy_character_unified_rig_layered_shader_and_pose_deformation(hfw_env, tmp_path: Path) -> None:
    cfg, game, cat = hfw_env
    asm = assemble_character(game, catalog=cat, prompt="Aloy Carja Blazon no headgear", include_hair=True, hair_lod=2)
    assert asm.display_name == "Aloy"
    assert "Carja Blazon" in asm.variant_label
    assert len(asm.skeleton.joints) >= 1200  # Body + Outfit + Merged Facial Skeleton!
    assert "C_Head_sjnt_0" in asm.skeleton.joint_by_name
    assert any(p.source_kind == "FacialMesh" for p in asm.parts)
    assert any(p.source_kind == "HairSkinnedMesh" for p in asm.parts)

    bdir = tmp_path / "aloy_carja_bundle"
    export_assembly_bundle(game, asm, bdir, unpack_textures=False, asset_kind="character")
    manifest = json.loads((bdir / "assembly.json").read_text(encoding="utf-8"))
    assert manifest["asset_kind"] == "character"

    # Verify clean character texture filtering + DXIL 8-corner outfit shader + DXIL Skin LUT + dual-UV hair!
    mats = list(manifest["materials"].values())
    dxil_outfits = [m["outfit_layered"] for m in mats if m.get("outfit_layered")]
    assert len(dxil_outfits) >= 1, "Expected HFW 8-layer outfit_layered material on Aloy Carja Blazon"
    assert any(o.get("mode") == "outfit_dxil" and len(o.get("layers", [])) == 8 for o in dxil_outfits), (
        f"Expected DXIL 8-corner outfit extraction, got: {[o.get('mode') for o in dxil_outfits]}"
    )
    skin_mats = [m for m in mats if m.get("surface_profile") == "skin" and not m.get("allow_alpha_cutout")]
    assert len(skin_mats) >= 1
    assert any(m.get("uses_skin_lut") and m.get("skin_tone_linear") for m in skin_mats), (
        "Expected 570:0 Skin LUT tint on Aloy skin materials"
    )
    assert manifest.get("shader_controls", {}).get("skin_tone_linear"), "Missing manifest shader_controls.skin_tone_linear"
    assert any(m.get("surface_profile") == "hair_face" for m in mats)
    assert any(
        any(e.get("uv_index") == 1 for e in m.get("packed_entries", []))
        for m in mats
        if m.get("surface_profile") == "hair"
    ), "Expected UV1 mapping on AloyHair_Cards_UvSet02_CLR"
    for m in mats:
        for e in m.get("packed_entries", []):
            fn = e["filename"].lower()
            assert "placeholder" not in fn and "outfit_fuzz_atlas" not in fn, f"Junk texture leaked: {fn}"

    # Verify UE5 character importer AST
    ue_char = bdir / "ue5_import_character.py"
    assert ue_char.exists()
    ue_src = ue_char.read_text(encoding="utf-8")
    ast.parse(ue_src)
    assert "OutfitMaskRGB" in ue_src
    assert "IS_SKELETAL = True" in ue_src

    # Build in Blender 5.3 headless & verify unified rig Pose Mode deformation (Head + Hair follow Armature together!)
    out_blend = tmp_path / "Aloy_Carja.blend"
    build_blend_headless(bdir, out_blend, config=cfg)
    assert out_blend.exists()

    pose_script = tmp_path / "verify_aloy_pose.py"
    pose_script.write_text(
        f"""
import bpy
from mathutils import Vector, Euler

bpy.ops.wm.open_mainfile(filepath=r'{out_blend}')
char_ng = bpy.data.node_groups.get('HFW_CharacterControl')
assert char_ng is not None, 'Missing HFW_CharacterControl skin node group'
sk_rgb = next(n for n in char_ng.nodes if n.label == 'Skin Tone').outputs[0].default_value
assert sk_rgb[0] > sk_rgb[1] > sk_rgb[2] > 0.1, f'Unexpected Aloy linear skin tone: {{tuple(sk_rgb)}}'
arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
bcolls = {{bc.name for bc in arm.data.collections}}
assert {{'Body', 'Face'}}.issubset(bcolls), f'Missing bone collections: {{bcolls}}'

def center(name):
    deps = bpy.context.evaluated_depsgraph_get()
    o = bpy.data.objects[name]
    eo = o.evaluated_get(deps)
    me = eo.to_mesh()
    step = max(1, len(me.vertices) // 100)
    acc = Vector((0.0, 0.0, 0.0))
    c = 0
    for i in range(0, len(me.vertices), step):
        acc += eo.matrix_world @ me.vertices[i].co
        c += 1
    eo.to_mesh_clear()
    return acc / max(1, c)

head_obj = next(o.name for o in bpy.data.objects if o.type == 'MESH' and o.name.startswith('Head_') and not o.hide_viewport)
hair_obj = next(o.name for o in bpy.data.objects if o.type == 'MESH' and o.name.startswith('Aloy_Hair_') and not o.hide_viewport)
h0, r0 = center(head_obj), center(hair_obj)

# Pose via the live Control Rig (`CTRL_Head`) when present (hfw_rig_master = 1.0), proving Head + Hair deform cleanly!
pb = arm.pose.bones.get('CTRL_Head') or arm.pose.bones['C_Head_sjnt_0']
pb.rotation_mode = 'XYZ'
pb.rotation_euler = Euler((0.25, 0.0, -0.45), 'XYZ')
bpy.context.view_layer.update()

dh = (center(head_obj) - h0).length
dr = (center(hair_obj) - r0).length
assert dh > 0.01 and dr > 0.01, f'Head/hair failed to deform: dh={{dh}}, dr={{dr}}'
print(f'POSE_DEFORM_OK dh={{dh:.4f}} dr={{dr:.4f}}')
""",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [str(cfg.blender_exe), "--factory-startup", "--background", "--python-exit-code", "1", "--python", str(pose_script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    assert "POSE_DEFORM_OK" in proc.stdout


# ==============================================================================
# PILLAR 3 — GENERAL ASSETS & GUERRILLA DEV-DIRECTORY ASSET CATALOG
# ==============================================================================


def test_dev_directory_asset_catalog_and_exports(hfw_env, tmp_path: Path) -> None:
    cfg, game, cat = hfw_env
    st = cat.stats()
    assert st["dev_assets"] >= 1000

    # Verify Guerrilla developer tree traversal
    root_tree = cat.list_dev_tree_children("")
    root_names = {f["name"].lower() for f in root_tree["folders"]}
    assert "models" in root_names and "dlc" in root_names

    weapons_tree = cat.list_dev_tree_children("models/weapons")
    assert len(weapons_tree["assets"]) >= 50

    # 1. Rigged Weapon export (Bow_Heavy_Quen_SE) -> IS_SKELETAL = True + real non-origin bind-pose bones!
    w_row = cat.get_dev_asset("DLC/models/weapons/Bow_Heavy_Quen_SE")
    assert w_row is not None and w_row["is_skinned"] is True
    asm_w = assemble_general_asset(
        game, w_row["resource_id"], display_name=w_row["display_name"], dev_path=w_row["dev_path"], category=w_row["category"]
    )
    assert len(asm_w.skeleton.joints) >= 15 and len(asm_w.parts) >= 1
    non_origin_joints = sum(
        1
        for j in asm_w.skeleton.joints
        if float((j.world_matrix[0, 3] ** 2 + j.world_matrix[1, 3] ** 2 + j.world_matrix[2, 3] ** 2) ** 0.5) > 0.05
    )
    assert non_origin_joints >= 12, f"Bow bones stuck at origin: only {non_origin_joints}/{len(asm_w.skeleton.joints)} non-origin!"
    bdir_w = tmp_path / "weapon_bundle"
    export_assembly_bundle(game, asm_w, bdir_w, unpack_textures=False, asset_kind="asset")
    w_manifest = json.loads((bdir_w / "assembly.json").read_text(encoding="utf-8"))
    assert any(m.get("packed_entries") for m in w_manifest["materials"].values()), "Bow missing textured materials"
    ue_w = (bdir_w / "ue5_import_asset.py").read_text(encoding="utf-8")
    ast.parse(ue_w)
    assert "IS_SKELETAL = True" in ue_w

    # 2. Static Eco Tree export (ponderosa_pine) -> IS_SKELETAL = False + Blender Root Empty
    t_row = cat.get_dev_asset("models/eco_assets_hrz2/trees/ponderosa_pine")
    assert t_row is not None and t_row["is_skinned"] is False
    asm_t = assemble_general_asset(
        game, t_row["resource_id"], display_name=t_row["display_name"], dev_path=t_row["dev_path"], category=t_row["category"]
    )
    assert len(asm_t.skeleton.joints) == 0 and len(asm_t.parts) >= 1
    bdir_t = tmp_path / "tree_bundle"
    export_assembly_bundle(game, asm_t, bdir_t, unpack_textures=False, asset_kind="asset")
    ue_t = (bdir_t / "ue5_import_asset.py").read_text(encoding="utf-8")
    ast.parse(ue_t)
    assert "IS_SKELETAL = False" in ue_t

    out_blend = tmp_path / "Ponderosa_Pine.blend"
    build_blend_headless(bdir_t, out_blend, config=cfg)
    assert out_blend.exists()


def test_web_api_and_cli_characters_and_assets() -> None:
    cfg = ToolkitConfig.load()
    venv_cli = Path(__file__).resolve().parent.parent / ".venv" / "Scripts" / "hfw-toolkit.exe"

    # CLI checks
    p1 = subprocess.run([str(venv_cli), "characters", "resolve", "Aloy Carja Blazon no headgear"], capture_output=True, text=True, check=False)
    assert p1.returncode == 0 and "Carja Blazon" in p1.stdout and "headgear=False" in p1.stdout

    p2 = subprocess.run([str(venv_cli), "assets", "tree", "models/weapons"], capture_output=True, text=True, check=False)
    assert p2.returncode == 0 and "Bow_Aloy" in p2.stdout

    # Web API checks
    app = create_app(cfg)
    port = 18429
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(40):
            try:
                with urllib.request.urlopen(f"{base}/api/stats", timeout=10):
                    break
            except Exception:
                time.sleep(0.25)

        def get_json(path: str):
            with urllib.request.urlopen(f"{base}{path}", timeout=30) as resp:
                assert resp.status == 200
                return json.loads(resp.read().decode("utf-8"))

        chars = get_json("/api/characters?limit=30")
        assert chars["counts"]["characters"] >= 1000
        assert any(c["char_id"] == "aloy" for c in chars["items"])

        res = get_json("/api/characters/resolve?prompt=" + urllib.parse.quote("Aloy Carja Blazon no headgear"))
        assert res["matched"] is True and res["headgear"] is False and res["character"]["char_id"] == "aloy"

        outfits = get_json("/api/characters/aloy/outfits")
        assert len(outfits["items"]) >= 30

        heads = get_json("/api/characters/heads?limit=50")
        assert len(heads["items"]) >= 50

        cats = get_json("/api/assets/categories")
        assert cats["total"] >= 1000

        tree = get_json("/api/assets/tree?dir=models/weapons")
        assert len(tree["assets"]) >= 50

        srch = get_json("/api/assets/search?category=weapons&q=Bow")
        assert len(srch["items"]) >= 5
    finally:
        server.should_exit = True
        th.join(timeout=5)


# ==============================================================================
# v10 REGRESSION TESTS — CLI FACIAL_OVERRIDE, DLC DXIL OUTFITS & ASSET MATERIALS
# ==============================================================================


def test_v10_fixes_cli_character_import_dlc_dxil_and_general_assets(hfw_env, tmp_path: Path) -> None:
    cfg, game, cat = hfw_env
    venv_cli = Path(__file__).resolve().parent.parent / ".venv" / "Scripts" / "hfw-toolkit.exe"

    # 1. Backward-compatible `facial_override` kwarg + CLI `characters export-bundle --prompt`
    asm_ov = assemble_character(game, catalog=cat, char_id="aloy", facial_override="48346:0", include_hair=False)
    assert asm_ov.display_name == "Aloy" and len(asm_ov.parts) >= 1

    cli_out = tmp_path / "cli_aloy_bundle"
    proc = subprocess.run(
        [
            str(venv_cli), "characters", "export-bundle",
            "--prompt", "Aloy Carja Blazon no headgear",
            "--no-hair",
            "--out", str(cli_out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"CLI character export failed:\n{proc.stderr}\n{proc.stdout}"
    assert (cli_out / "assembly.json").exists()

    # 2. Burning Shores DLC Companion (Seyka) -> multi-technique DXIL 8-corner outfit + warm golden skin LUT + dark hair LUT
    asm_seyka = assemble_character(game, catalog=cat, char_id="seyka", include_hair=False)
    bdir_seyka = tmp_path / "seyka_bundle"
    export_assembly_bundle(game, asm_seyka, bdir_seyka, unpack_textures=False, asset_kind="character")
    s_man = json.loads((bdir_seyka / "assembly.json").read_text(encoding="utf-8"))
    s_mats = list(s_man["materials"].values())
    assert any(m.get("uses_skin_lut") and m.get("skin_tone_linear") for m in s_mats), "Seyka missing 570:0 skin tone"
    assert any(
        m.get("surface_profile") == "hair" and m.get("hair_color_linear")
        for m in s_mats
    ), "Seyka QuenHair_Alpha missing hair surface_profile / 382:0 Hair LUT color"
    assert any(
        (m.get("outfit_layered") or {}).get("mode") == "outfit_dxil"
        and len((m.get("outfit_layered") or {}).get("layers", [])) == 8
        for m in s_mats
    ), "Seyka DLC outfit failed multi-technique DXIL 8-corner extraction"
    # Verify pure-dielectric tassels (`f0_rel == []`) are NOT falsely metallic!
    tassels = [m for m in s_mats if "tassel" in m.get("name", "").lower() and m.get("outfit_layered")]
    for tm in tassels:
        assert all(not lyr.get("is_metallic") for lyr in tm["outfit_layered"]["layers"]), (
            "Pure-dielectric Seyka tassels falsely marked metallic!"
        )

    # 2b. Verify Aloy Nora Anointed scalar-F0 (`f0_rel == [-7]`) leather sleeves/pants are NOT falsely metallic!
    asm_nora = assemble_character(game, catalog=cat, prompt="Aloy", include_hair=False)
    bdir_nora = tmp_path / "aloy_nora_bundle"
    export_assembly_bundle(game, asm_nora, bdir_nora, unpack_textures=False, asset_kind="character")
    n_man = json.loads((bdir_nora / "assembly.json").read_text(encoding="utf-8"))
    arms_mat = n_man["materials"].get("75185:1033")
    assert arms_mat and arms_mat.get("outfit_layered")
    c0_leather = arms_mat["outfit_layered"]["layers"][0]
    assert c0_leather["is_metallic"] is False, f"Aloy Nora leather falsely metallic: {c0_leather}"
    assert c0_leather["dye_color_linear"][0] < 0.35, f"Aloy Nora leather got corrupted F0 tint: {c0_leather}"

    # 3. General dev-directory assets previously missing/broken materials
    asset_checks = [
        ("DLC/models/landscape_assets/rock/volcanic/flow_b201", "rock_volcanic"),
        ("models/building_blocks/ruins_sf/dressing", "ruins_sf"),
        ("Models/landscape_assets/snow/snowcover_cb_b201", "snowcover"),
    ]
    for dev_path, slug in asset_checks:
        row = cat.get_dev_asset(dev_path)
        assert row is not None, f"Missing dev_asset {dev_path}"
        asm_a = assemble_general_asset(
            game, row["resource_id"], display_name=row["display_name"], dev_path=row["dev_path"], category=row["category"]
        )
        assert len(asm_a.parts) >= 1
        bdir_a = tmp_path / f"{slug}_bundle"
        export_assembly_bundle(game, asm_a, bdir_a, unpack_textures=False, asset_kind="asset")
        a_man = json.loads((bdir_a / "assembly.json").read_text(encoding="utf-8"))
        a_mats = list(a_man["materials"].values())
        has_tex = any(
            m.get("packed_entries") or m.get("outfit_layered") or m.get("surface_profile") == "snow"
            for m in a_mats
        )
        assert has_tex, f"{dev_path} exported with 0 textured/shaded materials!"
        if slug == "rock_volcanic":
            assert any(m.get("surface_profile") == "rock_volcanic" for m in a_mats), (
                "Expected rock_volcanic _TNT shader profile on flow_b201"
            )
            assert any(m.get("asset_detail_normal_file") for m in a_mats), (
                "Expected 2K basalt detail normal map (`5211:1`) exported on flow_b201"
            )
        if slug == "ruins_sf":
            assert sum(1 for m in a_mats if m.get("packed_entries") or m.get("outfit_layered")) >= 2, (
                "Expected >= 2 textured materials on ruins_sf/dressing"
            )
            assert all("starmoss" not in m.get("name", "").lower() for m in a_mats), (
                "StarMoss world overlay should not overwrite unique ruins_sf materials"
            )
        if slug == "snowcover":
            assert any(m.get("surface_profile") == "snow" for m in a_mats), (
                "Expected snow surface_profile on snowcover_cb_b201"
            )
        ue_a = (bdir_a / "ue5_import_asset.py").read_text(encoding="utf-8")
        ast.parse(ue_a)
        if slug == "rock_volcanic":
            assert "AssetDetailNormal8x" in ue_a

    # 4. v11 fixes: Calibrated muscle glow, Seyka hybrid Torso/TrimsUV2 stem-matched normals, & Axe_Nora Detail_* maps!
    std_p = infer_machine_variant_profile("slaughterspine", "Standard")
    apx_p = infer_machine_variant_profile("clawstrider", "Apex")
    assert std_p["muscle_glow"] <= 3.0 and std_p["cable_glow"] <= 3.4
    assert apx_p["muscle_glow"] <= 3.0 and apx_p["cable_glow"] <= 3.4

    s_torso = s_man["materials"].get("21350:2823")
    assert s_torso and (s_torso.get("outfit_layered") or {}).get("mode") == "outfit_dxil"
    assert s_torso["outfit_layered"].get("macro_color_mode") == "multiply2"
    assert s_torso["outfit_layered"].get("misc_file")

    s_trims = s_man["materials"].get("21350:3947") or s_man["materials"].get("21350:798")
    assert s_trims and (s_trims.get("outfit_layered") or {}).get("mode") == "outfit_dxil"
    assert "seykatrimsuv2" in s_trims["outfit_layered"].get("normal_file", "").lower(), (
        f"SeykaTrimsUV2 picked wrong normal map: {s_trims['outfit_layered'].get('normal_file')}"
    )
    assert all(e.get("uv_index", 0) == 0 for e in s_trims.get("packed_entries", [])), (
        "SeykaTrimsUV2 should sample UV0 (not dead UV1)"
    )

    ue_s = (bdir_seyka / "ue5_import_character.py").read_text(encoding="utf-8")
    ast.parse(ue_s)
    assert "_DIELECTRIC_PROFILES" in ue_s and "MicroNrmRgh" in ue_s

    axe_row = cat.get_dev_asset("models/weapons/Axe_Nora")
    assert axe_row is not None
    asm_axe = assemble_general_asset(
        game, axe_row["resource_id"], display_name=axe_row["display_name"], dev_path=axe_row["dev_path"], category=axe_row["category"]
    )
    bdir_axe = tmp_path / "axe_nora_bundle"
    export_assembly_bundle(game, asm_axe, bdir_axe, unpack_textures=False, asset_kind="asset")
    axe_man = json.loads((bdir_axe / "assembly.json").read_text(encoding="utf-8"))
    assert len(axe_man["materials"]) == 4
    for mid, am in axe_man["materials"].items():
        assert len(am.get("packed_entries", [])) >= 4, f"Axe_Nora material {mid} ({am['name']}) missing Detail_* textures!"
        assert all("snow_clumps" not in e.get("filename", "").lower() for e in am.get("packed_entries", []))
    ast.parse((bdir_axe / "ue5_import_asset.py").read_text(encoding="utf-8"))
