"""Phase 3 tests: End-to-end Headless Blender 5.3 `.blend` generation & rig/plate/material verification
on Slitherfang, Clawstrider, and Slaughterspine!
"""

from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

import pytest

from hfw_toolkit.blender import build_blend_headless, export_assembly_bundle
from hfw_toolkit.config import ToolkitConfig
from hfw_toolkit.decima import AssetCatalog, HFWGame
from hfw_toolkit.extract import assemble_machine, resolve_machine_entity_id

VERIFY_SCRIPT = r"""
import json, math, sys, bpy
from mathutils import Euler, Vector

out_json = sys.argv[sys.argv.index("--") + 1]
arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
meshes = [o for o in bpy.data.objects if o.type == "MESH"]
bone_parented = [o for o in meshes if o.parent_type == "BONE" and o.parent_bone]
skinned = [o for o in meshes if any(m.type == "ARMATURE" for m in o.modifiers)]
mats_with_tex = 0
mats_with_detail_array = 0
mats_with_decima_pbr = 0
mats_with_muscle_shader = 0
mats_with_emissive_overlay = 0
mats_with_plasma_charge = 0
mats_with_time_driver = 0
for m in bpy.data.materials:
    if not m.use_nodes:
        continue
    nodes = m.node_tree.nodes
    if any(n.type == "TEX_IMAGE" and n.image is not None for n in nodes):
        mats_with_tex += 1
    if m.node_tree.animation_data and len(m.node_tree.animation_data.drivers) > 0:
        mats_with_time_driver += 1
    for n in nodes:
        if n.type == "GROUP" and n.node_tree is not None:
            gname = n.node_tree.name
            if gname.startswith("HFW_MachineDetailArray"):
                mats_with_detail_array += 1
            elif gname == "HFW_DecimaPBR":
                mats_with_decima_pbr += 1
            elif gname in ("HFW_MachineMuscleFiber", "HFW_MachineMuscleCables"):
                mats_with_muscle_shader += 1
            elif gname.startswith("HFW_PlasmaCharge") and gname != "HFW_PlasmaChargeControl":
                mats_with_plasma_charge += 1
            elif gname.startswith(("HFW_ElementalSack", "HFW_CanisterGlow", "HFW_MachineEyeLens")):
                mats_with_emissive_overlay += 1

mf_ng = bpy.data.node_groups.get("HFW_MachineMuscleFiber")
muscle_fiber_cycles_safe = False
if mf_ng is not None:
    ntypes = {n.type for n in mf_ng.nodes}
    muscle_fiber_cycles_safe = ("LIGHT_PATH" in ntypes) and ("NEW_GEOMETRY" in ntypes) and ("MAP_RANGE" in ntypes)

meshes_with_color0 = sum(1 for o in meshes if "Color0" in o.data.color_attributes)
skinned_tris = sum(len(o.data.polygons) for o in skinned if not o.hide_viewport)
has_plasma_control = (
    bpy.data.node_groups.get("HFW_PlasmaChargeControl") is not None
    and any("hfw_plasma_charge" in a for a in arms)
)
cycles_trans_bounces = getattr(getattr(bpy.context.scene, "cycles", None), "transparent_max_bounces", 0)

dists = [o.matrix_world.translation.length for o in bone_parented]
nonzero_plates = sum(1 for d in dists if d > 0.1)

# Rigging & Pose-Mode Lockstep Verification:
# Rotate a real Skeleton bone that BOTH weights skinned body vertices (>0.85) AND has a bone-parented plate!
# Verify that:
#   1. Deformed skinned body vertices actually move (> 0.05m) via Armature modifier!
#   2. The attached armor plate moves in exact lockstep with the deformed body vertices (drift < 0.05m)!
plate_moved = False
skin_deformed = False
lockstep_ok = False
lockstep_drift = -1.0

if arms and bone_parented and skinned:
    arm = arms[0]
    bpy.context.view_layer.objects.active = arm

    # Map skeleton bone name -> list of plates attached directly or via child helper bone
    plates_by_skel_bone = {}
    for plate in bone_parented:
        if plate.hide_viewport:
            continue
        pb = arm.pose.bones.get(plate.parent_bone)
        if not pb:
            continue
        # If parented to a helper bone, check its parent skeleton bone too
        cand_bones = [pb.name]
        if pb.parent:
            cand_bones.append(pb.parent.name)
        for bn in cand_bones:
            plates_by_skel_bone.setdefault(bn, []).append(plate)

    # Find a skinned mesh + skeleton bone with both high-weight vertices and an attached plate
    tested_bone = None
    tested_body = None
    tested_vidxs = []
    tested_plate = None
    for body in skinned:
        if body.hide_viewport or not body.vertex_groups:
            continue
        for vg in body.vertex_groups:
            if vg.name not in plates_by_skel_bone:
                continue
            pb = arm.pose.bones.get(vg.name)
            if not pb or not pb.parent:
                continue
            # Sample vertices heavily weighted (> 0.85) to this bone
            hi_verts = []
            vg_idx = vg.index
            for v in body.data.vertices:
                for g in v.groups:
                    if g.group == vg_idx and g.weight > 0.85:
                        hi_verts.append(v.index)
                        break
                if len(hi_verts) >= 40:
                    break
            if len(hi_verts) >= 10:
                tested_bone = pb
                tested_body = body
                tested_vidxs = hi_verts
                tested_plate = plates_by_skel_bone[vg.name][0]
                break
        if tested_bone is not None:
            break

    def _eval_center(obj, vidxs):
        deps = bpy.context.evaluated_depsgraph_get()
        eval_obj = obj.evaluated_get(deps)
        me = eval_obj.to_mesh()
        mw = eval_obj.matrix_world
        acc = Vector((0.0, 0.0, 0.0))
        for vi in vidxs:
            acc += mw @ me.vertices[vi].co
        eval_obj.to_mesh_clear()
        return acc / max(len(vidxs), 1)

    if tested_bone is not None:
        bpy.context.view_layer.update()
        body_before = _eval_center(tested_body, tested_vidxs)
        plate_before = tested_plate.matrix_world.translation.copy()
        dist_before = (plate_before - body_before).length

        # Rotate the bone by 45 degrees!
        tested_bone.rotation_mode = "XYZ"
        tested_bone.rotation_euler = Euler((0.785, 0.35, 0.0), "XYZ")
        bpy.context.view_layer.update()

        body_after = _eval_center(tested_body, tested_vidxs)
        plate_after = tested_plate.matrix_world.translation.copy()
        dist_after = (plate_after - body_after).length

        body_delta = (body_after - body_before).length
        plate_delta = (plate_after - plate_before).length
        lockstep_drift = abs(dist_after - dist_before)

        skin_deformed = body_delta > 0.02
        plate_moved = plate_delta > 0.02
        lockstep_ok = skin_deformed and plate_moved and (lockstep_drift < 0.05)

res = {
    "armatures": len(arms),
    "bones": len(arms[0].data.bones) if arms else 0,
    "meshes": len(meshes),
    "skinned": len(skinned),
    "bone_parented": len(bone_parented),
    "nonzero_plates": nonzero_plates,
    "materials": len(bpy.data.materials),
    "mats_with_tex": mats_with_tex,
    "mats_with_detail_array": mats_with_detail_array,
    "mats_with_decima_pbr": mats_with_decima_pbr,
    "mats_with_muscle_shader": mats_with_muscle_shader,
    "mats_with_emissive_overlay": mats_with_emissive_overlay,
    "mats_with_plasma_charge": mats_with_plasma_charge,
    "mats_with_time_driver": mats_with_time_driver,
    "muscle_fiber_cycles_safe": muscle_fiber_cycles_safe,
    "meshes_with_color0": meshes_with_color0,
    "skinned_tris": skinned_tris,
    "has_plasma_control": has_plasma_control,
    "cycles_trans_bounces": cycles_trans_bounces,
    "skin_deformed_in_pose_mode": skin_deformed,
    "plate_moved_in_pose_mode": plate_moved,
    "plate_body_lockstep_ok": lockstep_ok,
    "lockstep_drift_m": round(lockstep_drift, 5),
}
with open(out_json, "w", encoding="utf-8") as fp:
    json.dump(res, fp)
"""


@pytest.fixture(scope="module")
def env():
    cfg = ToolkitConfig.load()
    game = HFWGame(cfg)
    catalog = AssetCatalog(cfg.index_db_path)
    yield cfg, game, catalog
    catalog.close()
    game.close()


@pytest.mark.parametrize("machine_name", ["Slitherfang", "Clawstrider", "Slaughterspine", "Thunderjaw"])
def test_headless_blender_export_and_verify(env, machine_name: str, tmp_path: Path) -> None:
    cfg, game, catalog = env
    oid, code, disp = resolve_machine_entity_id(game, catalog, machine_name)
    asm = assemble_machine(game, oid, code, disp, variant_index=0, lod_index=0, include_broken_states=True)

    # Export fast verification bundle into `tmp_path` so pytest NEVER overwrites full-res user exports in `exports/`!
    bundle_dir = tmp_path / f"{disp}_bundle"
    out_blend = tmp_path / f"{disp}.blend"
    export_assembly_bundle(game, asm, bundle_dir, max_texture_size=1024, unpack_textures=False)
    build_blend_headless(bundle_dir, out_blend, config=cfg, pack_textures=False, enable_detail_nodes=True)

    assert out_blend.exists() and out_blend.stat().st_size > 100_000
    # Verify native packed textures + 23-slice detail array atlas PNG exported!
    manifest_data = json.loads((bundle_dir / "assembly.json").read_text(encoding="utf-8"))
    assert manifest_data["texture_mode"] == "packed"
    tex_names = [p.name for p in (bundle_dir / "textures").glob("*.png")]
    assert len(tex_names) >= 3
    assert any(n.startswith("HFW_DetailArray_") for n in tex_names), f"Missing detail array atlas in {tex_names}"
    assert not any(n.endswith("_Normal_GL.png") or n.endswith("_Roughness.png") for n in tex_names)

    # Re-open in Blender 5.3 headless to verify rig, skin weights, bone-parented plate lockstep, and materials!
    vscript = tmp_path / "verify.py"
    vjson = tmp_path / f"verify_{disp}.json"
    vscript.write_text(VERIFY_SCRIPT, encoding="utf-8")
    proc = subprocess.run(
        [str(cfg.blender_exe), "--factory-startup", "--background", str(out_blend), "--python", str(vscript), "--", str(vjson)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Blender verify failed:\n{proc.stderr[-1500:]}"
    stats = json.loads(vjson.read_text(encoding="utf-8"))
    print(f"\n[Blender 5.3 Verified {out_blend.name}]: {stats}")

    assert stats["armatures"] == 1
    assert stats["bones"] >= 100
    assert stats["skinned"] >= 1
    assert stats["bone_parented"] >= 50
    assert stats["nonzero_plates"] >= 50
    assert stats["mats_with_tex"] >= 5
    assert stats["mats_with_detail_array"] >= 40, f"Detail Array not applied to materials: {stats}"
    assert stats["mats_with_decima_pbr"] >= 40, f"HFW_DecimaPBR not applied to materials: {stats}"
    assert stats["mats_with_muscle_shader"] >= 1, f"Machine muscle shaders missing: {stats}"
    assert stats["mats_with_time_driver"] >= 1, f"Animated #frame/24.0 shader time drivers missing: {stats}"
    assert stats["skin_deformed_in_pose_mode"] is True
    assert stats["plate_moved_in_pose_mode"] is True
    assert stats["plate_body_lockstep_ok"] is True, f"Plate/body lockstep drift too high: {stats}"

    # Verify generated UE5 full machine importer script exists, has valid Python syntax, and includes animated expressions!
    ue5_script = bundle_dir / "ue5_import_machine.py"
    assert ue5_script.exists()
    ue5_src = ue5_script.read_text(encoding="utf-8")
    ast.parse(ue5_src)
    assert "MaterialExpressionTime" in ue5_src
    assert "MaterialExpressionPanner" in ue5_src
    assert "MaterialExpressionSine" in ue5_src


def test_frostclaw_elemental_sacks_and_animation_drivers(env, tmp_path: Path) -> None:
    cfg, game, catalog = env
    oid, code, disp = resolve_machine_entity_id(game, catalog, "Frostclaw")
    asm = assemble_machine(game, oid, code, disp, variant_index=0, lod_index=0, include_broken_states=False)

    bundle_dir = tmp_path / "Frostclaw_bundle"
    out_blend = tmp_path / "Frostclaw.blend"
    export_assembly_bundle(game, asm, bundle_dir, max_texture_size=1024, unpack_textures=False)
    build_blend_headless(bundle_dir, out_blend, config=cfg, pack_textures=False, enable_detail_nodes=True)

    vscript = tmp_path / "verify_fc.py"
    vjson = tmp_path / "verify_fc.json"
    vscript.write_text(VERIFY_SCRIPT, encoding="utf-8")
    proc = subprocess.run(
        [str(cfg.blender_exe), "--factory-startup", "--background", str(out_blend), "--python", str(vscript), "--", str(vjson)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    stats = json.loads(vjson.read_text(encoding="utf-8"))
    assert stats["mats_with_emissive_overlay"] >= 2, f"Expected glowing Frostclaw Cryo sac/eye overlays: {stats}"
    assert stats["mats_with_time_driver"] >= 2, f"Expected animated Time drivers on Frostclaw sacks/muscles: {stats}"


def test_slaughterspine_plasma_charge_and_redeye_watcher_blender(env, tmp_path: Path) -> None:
    cfg, game, catalog = env

    # 1. Slaughterspine Plasma Charge-Up (Color0 vertex colors + HFW_PlasmaCharge + live slider + Cycles-safe muscles + UE5!)
    ss_oid, ss_code, ss_disp = resolve_machine_entity_id(game, catalog, "Slaughterspine")
    ss_asm = assemble_machine(game, ss_oid, ss_code, ss_disp, variant_index=0, lod_index=0, include_broken_states=False)
    ss_bdir = tmp_path / "Slaughterspine_plasma_bundle"
    ss_blend = tmp_path / "Slaughterspine_plasma.blend"
    export_assembly_bundle(game, ss_asm, ss_bdir, max_texture_size=512, unpack_textures=False)
    build_blend_headless(ss_bdir, ss_blend, config=cfg, pack_textures=False, enable_detail_nodes=True)

    vscript = tmp_path / "verify_v6.py"
    vscript.write_text(VERIFY_SCRIPT, encoding="utf-8")
    ss_json = tmp_path / "verify_ss.json"
    proc = subprocess.run(
        [str(cfg.blender_exe), "--factory-startup", "--background", str(ss_blend), "--python", str(vscript), "--", str(ss_json)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-1000:]
    ss_stats = json.loads(ss_json.read_text(encoding="utf-8"))
    assert ss_stats["has_plasma_control"] is True, f"Missing live HFW_PlasmaChargeControl slider: {ss_stats}"
    assert ss_stats["mats_with_plasma_charge"] >= 10, f"Expected >= 10 Slaughterspine plasma materials: {ss_stats}"
    assert ss_stats["meshes_with_color0"] >= 10, f"Expected Color0 vertex colors on plasma meshes: {ss_stats}"
    assert ss_stats["muscle_fiber_cycles_safe"] is True, f"Missing Cycles shadow/backface bypass on muscles: {ss_stats}"
    assert ss_stats["cycles_trans_bounces"] >= 16

    ue5_src = (ss_bdir / "ue5_import_machine.py").read_text(encoding="utf-8")
    ast.parse(ue5_src)
    assert "MaterialExpressionVertexColor" in ue5_src
    assert "PlasmaCharge" in ue5_src
    assert "VertexNormalWS" in ue5_src
    assert "VertexColorImportOption.REPLACE" in ue5_src

    # Verify Standard Slaughterspine AcidSac uses Acid Green (NOT default placeholder blue!),
    # while Fire Bellowback uses Blaze Orange and Acid Bellowback uses Acid Green!
    ss_man = json.loads((ss_bdir / "assembly.json").read_text(encoding="utf-8"))
    ss_sacs = [
        mm["emissive_overlay"]["glow_color"]
        for mm in ss_man["materials"].values()
        if (mm.get("emissive_overlay") or {}).get("kind") == "elemental_sack_overlay"
    ]
    assert ss_sacs and all(c[1] > 0.75 and c[2] < 0.20 for c in ss_sacs), f"Slaughterspine AcidSacs must glow Acid Green, got {ss_sacs}"

    bb_oid, bb_code, bb_disp = resolve_machine_entity_id(game, catalog, "Bellowback")
    bb_vars = catalog.list_machine_variants(bb_oid.group_id, bb_oid.object_index)
    fire_idx = next(v["variant_index"] for v in bb_vars if "Fire Bellowback" in v["label"])
    acid_idx = next(v["variant_index"] for v in bb_vars if "Acid Bellowback" in v["label"])
    for lbl, v_idx, check_fn in (
        ("Fire Bellowback", fire_idx, lambda c: c[0] > 0.85 and c[2] < 0.10),
        ("Acid Bellowback", acid_idx, lambda c: c[1] > 0.75 and c[2] < 0.15),
    ):
        bb_asm = assemble_machine(game, bb_oid, bb_code, bb_disp, variant_index=v_idx, lod_index=0, include_broken_states=False)
        bb_bdir = tmp_path / f"bb_{v_idx}"
        export_assembly_bundle(game, bb_asm, bb_bdir, max_texture_size=256, unpack_textures=False)
        bb_man = json.loads((bb_bdir / "assembly.json").read_text(encoding="utf-8"))
        bb_sacs = [
            mm["emissive_overlay"]["glow_color"]
            for mm in bb_man["materials"].values()
            if (mm.get("emissive_overlay") or {}).get("kind") == "elemental_sack_overlay"
        ]
        assert bb_sacs and all(check_fn(c) for c in bb_sacs), f"{lbl} sac colors wrong: {bb_sacs}"

    # 2. Redeye Watcher in Blender: full rigged 104k-tri body + BOTH legs + plates!
    rw_oid, rw_code, rw_disp = resolve_machine_entity_id(game, catalog, "Redeye Watcher")
    rw_asm = assemble_machine(game, rw_oid, rw_code, rw_disp, variant_index=0, lod_index=0, include_broken_states=False)
    rw_bdir = tmp_path / "Redeye_Watcher_bundle"
    rw_blend = tmp_path / "Redeye_Watcher.blend"
    export_assembly_bundle(game, rw_asm, rw_bdir, max_texture_size=512, unpack_textures=False)
    build_blend_headless(rw_bdir, rw_blend, config=cfg, pack_textures=False, enable_detail_nodes=True)

    rw_json = tmp_path / "verify_rw.json"
    proc2 = subprocess.run(
        [str(cfg.blender_exe), "--factory-startup", "--background", str(rw_blend), "--python", str(vscript), "--", str(rw_json)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc2.returncode == 0, proc2.stderr[-1000:]
    rw_stats = json.loads(rw_json.read_text(encoding="utf-8"))
    assert rw_stats["skinned"] >= 1 and rw_stats["skinned_tris"] >= 100_000, f"Redeye Watcher missing legs/body in Blender: {rw_stats}"
    assert rw_stats["bone_parented"] >= 5, f"Redeye Watcher missing plates in Blender: {rw_stats}"
