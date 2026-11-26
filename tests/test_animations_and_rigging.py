"""Automated tests for Sony Edge Anim (`EA08`) decoder, HFW Control Rig, bidirectional baking, and UE5 export."""

from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from hfw_toolkit.blender.bundle import export_assembly_bundle
from hfw_toolkit.blender.control_rig import analyze_rig_chains, compute_dynamic_collection_labels
from hfw_toolkit.config import ToolkitConfig
from hfw_toolkit.decima.catalog import AssetCatalog
from hfw_toolkit.decima.streaming_graph import HFWGame
from hfw_toolkit.extract.animations import (
    decode_animation_resource,
    discover_entity_animations,
    export_animation_pack,
    remap_animation_to_blender_basis,
    resolve_animations_for_export,
    select_diverse_animations,
    summarize_animation_categories,
)
from hfw_toolkit.extract.character_assembler import assemble_character
from hfw_toolkit.extract.machine_assembler import assemble_machine, resolve_machine_entity_id


@pytest.fixture(scope="module")
def game_and_catalog():
    cfg = ToolkitConfig.load()
    if not cfg.streaming_graph_path.is_file():
        pytest.skip("HFW PC game archive not present")
    g = HFWGame(cfg)
    cat = AssetCatalog(cfg.index_db_path)
    yield g, cat, cfg
    cat.close()
    g.close()


def _skel_specs(skel) -> list[dict]:
    return [
        {
            "name": j.name,
            "parent_name": skel.joints[j.parent_index].name if 0 <= j.parent_index < len(skel.joints) else None,
            "head": j.world_matrix[:3, 3].tolist(),
        }
        for j in skel.joints
    ]


def test_ea08_decode_and_remap_clawstrider(game_and_catalog):
    g, cat, _ = game_and_catalog
    oid, code, disp = resolve_machine_entity_id(g, cat, "clawstrider")
    asm = assemble_machine(g, oid, codename=code, display_name=disp, variant_index=0, lod_index=0, include_broken_states=False)
    clips = discover_entity_animations(g, oid.group_id, asset_kind="machine", target_skeleton_id=asm.skeleton.object_id, limit=30)
    assert len(clips) >= 10
    picked = select_diverse_animations(clips, 2)
    assert len(picked) == 2

    for clip in picked:
        dec = decode_animation_resource(g, clip.anim_id, name=clip.name, category=clip.category)
        assert dec.num_frames >= 2
        assert np.isfinite(dec.rotations_xyzw).all()
        assert np.isfinite(dec.translations).all()
        qnorms = np.linalg.norm(dec.rotations_xyzw, axis=-1)
        assert float(qnorms.min()) > 0.999
        assert float(qnorms.max()) < 1.001

        rem = remap_animation_to_blender_basis(dec, asm.skeleton)
        assert len(rem["joint_names"]) >= 100
        assert np.isfinite(rem["loc"]).all()
        assert np.isfinite(rem["rot_wxyz"]).all()


def test_ea08_all_animations_categories_and_export_pack(game_and_catalog, tmp_path: Path):
    """Verify unlimited discovery (no 25 cap!), category counts/descriptions, 'all' export, and fast export_animation_pack."""
    g, cat, _ = game_and_catalog
    oid, code, disp = resolve_machine_entity_id(g, cat, "clawstrider")
    asm = assemble_machine(g, oid, codename=code, display_name=disp, variant_index=0, lod_index=0, include_broken_states=False)

    all_clips = discover_entity_animations(g, oid.group_id, asset_kind="machine", target_skeleton_id=asm.skeleton.object_id, limit=None)
    assert len(all_clips) > 100
    counts = summarize_animation_categories(all_clips)
    assert counts["all"] == len(all_clips)
    assert counts["combat"] >= 20
    assert counts["movement"] >= 40
    assert all(c.description for c in all_clips[:20])

    # Category-filtered discovery & fast animation-only pack export
    jump_pack = tmp_path / "claw_jump_pack"
    export_animation_pack(g, asm.skeleton, jump_pack, entity_group_id=oid.group_id, asset_kind="machine", animations_spec="category:jump")
    pack_manifest = json.loads((jump_pack / "assembly.json").read_text(encoding="utf-8"))
    assert len(pack_manifest["animations"]) == counts["jump"]
    assert all(a["category"] == "jump" and a.get("description") for a in pack_manifest["animations"])

    # Verify 'all' mode resolves far beyond the old 25-clip cap on Behemoth
    beh_oid, beh_code, beh_disp = resolve_machine_entity_id(g, cat, "behemoth")
    beh_asm = assemble_machine(g, beh_oid, codename=beh_code, display_name=beh_disp, variant_index=0, lod_index=0, include_broken_states=False)
    beh_all = resolve_animations_for_export(g, beh_asm.skeleton, entity_group_id=beh_oid.group_id, asset_kind="machine", animations_spec="all")
    assert len(beh_all) >= 60


def test_ea08_humanoid_remap_onto_aloy_1881j(game_and_catalog):
    g, cat, _ = game_and_catalog
    aloy = assemble_character(g, catalog=cat, prompt="Aloy Nora Anointed", hair_lod=2, lod_index=0)
    aspec = analyze_rig_chains(_skel_specs(aloy.skeleton))
    assert aspec["archetype"] == "humanoid"
    assert aspec["chest"] == "C_Spine_sjnt_4"
    assert aspec["head_ik_supported"] is True
    assert "C_ChestOri_fjnt_0" not in aspec["spine"]
    aloy_labels = compute_dynamic_collection_labels(aspec)
    assert aloy_labels["body"] == "Torso, Chest & Head"
    assert aloy_labels["head_ik"] == "Head & Neck IK"
    assert aloy_labels["hands_feet"] == "Fingers & Toes"
    assert aloy_labels["hf_tweaks"] == "Fingers & Toes Tweaks"
    assert aloy_labels["frills"] == "Belt Flaps"
    assert aloy_labels["weapons"] == "Weapon Sockets"
    assert aloy_labels["arms_ik"] == "Arms IK"
    assert len(aspec["legs"]) == 2
    assert len(aspec["arms"]) == 2
    assert len(aspec["fingers"]) >= 8
    assert len(aspec["toes"]) >= 2
    clips = discover_entity_animations(g, None, asset_kind="character", gender="female", limit=20)
    assert len(clips) >= 5
    dec = decode_animation_resource(g, clips[0].anim_id, name=clips[0].name, category=clips[0].category)
    assert dec.rotations_xyzw.shape[1] == 82
    rem = remap_animation_to_blender_basis(dec, aloy.skeleton, preserve_non_root_bone_lengths=True)
    assert len(rem["joint_names"]) >= 50
    assert np.isfinite(rem["rot_wxyz"]).all()
    qnorms = np.linalg.norm(rem["rot_wxyz"], axis=-1)
    assert float(qnorms.min()) > 0.999


@pytest.mark.parametrize(
    ("query", "expected_archetype", "min_legs"),
    [
        ("clawstrider", "theropod", 2),
        ("slaughterspine", "theropod", 2),
        ("behemoth", "quadruped", 4),
        ("ravager", "quadruped", 4),
        ("tremortusk", "quadruped", 4),
        ("bilegut", "quadruped", 4),
        ("sunwing", "flyer", 2),
        ("dreadwing", "flyer", 2),
        ("shell-walker", "hexapod", 6),
        ("tideripper", "aquatic", 2),
        ("slitherfang", "serpentine", 0),
    ],
)
def test_universal_control_rig_chain_detection(game_and_catalog, query, expected_archetype, min_legs):
    g, cat, _ = game_and_catalog
    oid, code, disp = resolve_machine_entity_id(g, cat, query)
    asm = assemble_machine(g, oid, codename=code, display_name=disp, variant_index=0, lod_index=0, include_broken_states=False)
    spec = analyze_rig_chains(_skel_specs(asm.skeleton))
    assert spec["archetype"] == expected_archetype
    assert spec["root"] is not None
    assert spec["hips"] is not None
    assert spec["chest"] is not None
    assert spec["chest"] not in spec["spine"]
    assert spec["head"] is not None
    assert len(spec["legs"]) >= min_legs
    dyn = compute_dynamic_collection_labels(spec)
    if expected_archetype == "flyer":
        assert len(spec["wings"]) == 2
        assert dyn["wings_fk"] == "Wings FK"
        assert dyn["wings_ik"] == "Wings IK"
    if expected_archetype == "serpentine":
        assert len(spec["spine"]) >= 20
        assert spec["torso_ik_mode"] == "hood"
        assert dyn["torso_ik"] == "Cobra Hood IK"
        assert dyn["hood_fk"] == "Cobra Hood FK"
        assert dyn["frills"] == "Hood Feathers"
        assert any(s.get("kind") == "spine" for s in spec.get("spline_chains") or [])
        assert dyn["spline_ik"] == "Serpent Spline IK"
        assert len(spec.get("spinners") or []) >= 40
        assert dyn["spinners"] in ("Earthgrinders", "Drills & Grinders")
    if query == "behemoth":
        assert spec["chest"] == "C_Chest_sjnt_0"
        assert dyn["front_legs_ik"] == "Front Legs IK"
        assert dyn["legs_ik"] == "Rear Legs IK"
        assert dyn["hands_feet"] == "Hooves"
    if query == "ravager":
        assert len(spec["legs"]) == 4
        assert any(lg.get("shoulder") for lg in spec["legs"])
        assert dyn["front_legs_ik"] == "Front Legs IK"
    if query == "tremortusk":
        assert spec["appendages"] and spec["appendages"][0]["name"] == "Trunk"
        assert dyn["appendages"] == "Trunk FK"
        assert any(s.get("kind") == "appendage" and s.get("name") == "Trunk" for s in spec.get("spline_chains") or [])
    if query == "bilegut":
        assert spec["appendages"] and spec["appendages"][0]["name"] == "Tongue"
        assert dyn["frills"] == "Vocal Sacs"
    if query == "shell-walker":
        assert len(spec["legs"]) == 6
        assert len(spec["arms"]) == 2
        assert dyn["arms_ik"] == "Claw & Gun Arms IK"
    if query == "tideripper":
        assert all(lg.get("is_flipper") for lg in spec["legs"])
        assert all(ar.get("is_flipper") for ar in spec["arms"])
        assert spec["head_ik_supported"] is True
        assert len(spec.get("spinners") or []) >= 6
        assert dyn["arms_ik"] == "Front Flippers IK"
        assert dyn["legs_ik"] == "Rear Flippers IK"
        assert dyn["spinners"] == "Drills & Rotors"
    if query == "clawstrider":
        assert spec["chest"] == "C_spine_sjnt_6"
        assert spec["head_ik_supported"] is True
        assert len(spec.get("chainsaws") or []) == 2
        assert sum(len(c["blades"]) for c in spec["chainsaws"]) == 52
        assert dyn["spinners"] == "Chainsaws"
        assert dyn["frills"] == "Feathers & Blades"
        assert dyn["weapons"] == "Jaw Saws & Launchers"
        assert dyn["hands_feet"] == "Claws & Talons"
        assert dyn["arms_ik"] == "Forelimbs IK"
        assert len(spec["tails"][0]["bones"]) == 8  # Pure core tail (no endBlade polluted!)
        assert len(spec["frills"]) >= 30
        assert len(spec["weapons"]) >= 4
    if query == "slaughterspine":
        assert spec["chest"] == "C_spine_sjnt_6"
        assert dyn["frills"] == "Plasma Intakes & Spikes"
        assert dyn["weapons"] == "Cannons & Tail Blasters"
        assert len(spec["tails"][0]["bones"]) == 11  # Pure 11-segment tail (separated from TailBlaster!)
        assert len(spec["frills"]) >= 12  # Plasma intakes + dorsal spikes + antenna
        assert len(spec["plates"]) >= 12  # Neck/spine/tail plates + canisters
        assert len(spec["weapons"]) >= 4  # HeadCannon + 3 TailBlasters


def test_blender_control_rig_and_bidirectional_bake_and_ue5_ast(game_and_catalog, tmp_path: Path):
    g, cat, cfg = game_and_catalog
    if not cfg.blender_exe.exists():
        pytest.skip("Blender executable not available")

    oid, code, disp = resolve_machine_entity_id(g, cat, "clawstrider")
    asm = assemble_machine(g, oid, codename=code, display_name=disp, variant_index=0, lod_index=0, include_broken_states=False)
    bdir = tmp_path / "claw_bundle"
    export_assembly_bundle(g, asm, bdir, max_texture_size=256, animations="1587:321")

    assert (bdir / "animations.npz").is_file()
    manifest = json.loads((bdir / "assembly.json").read_text(encoding="utf-8"))
    assert manifest["rig_spec"]["archetype"] == "theropod"
    assert len(manifest["animations"]) == 1

    # Verify UE5 Control Rig + AnimSequence Python importer AST (including shoulders/wings/flippers/detail controls)
    ue5_script = (bdir / "ue5_import_machine.py").read_text(encoding="utf-8")
    ast.parse(ue5_script)
    assert "build_ue5_control_rig" in ue5_script
    assert "CR_Chest" in ue5_script
    assert "CR_IK_Head" in ue5_script
    assert "CR_Spline_" in ue5_script
    assert "CR_Saw_Spin_" in ue5_script
    assert "CR_Spin_" in ue5_script
    assert "CR_Shoulder_" in ue5_script
    assert "CR_Clavicle_" in ue5_script
    assert "_Blade_" in ue5_script
    assert "RigUnit_InverseExecution" in ue5_script
    assert "import_animations_fbx" in ue5_script
    assert "detail_groups" in ue5_script

    # Headless Blender 5.3 test: zero ghost-anim rest-lock, IK+FK limb switching, talons/fingers, dual bake, and DEF export!
    builder_path = Path(__file__).resolve().parents[1] / "hfw_toolkit" / "blender" / "blender_builder.py"
    cr_path = Path(__file__).resolve().parents[1] / "hfw_toolkit" / "blender" / "control_rig.py"
    test_py = tmp_path / "verify_rig.py"
    test_py.write_text(
        f'''
import importlib.util
import json
import math
from pathlib import Path
import bpy
from mathutils import Euler, Matrix

def load_m(name, p):
    spec = importlib.util.spec_from_file_location(name, str(p))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

builder = load_m("b", Path(r"{builder_path}"))
cr = load_m("c", Path(r"{cr_path}"))

bpy.ops.wm.read_factory_settings(use_empty=True)
arm = builder.import_bundle_into_scene(Path(r"{bdir}"), enable_detail_nodes=False, build_control_rig_layer=True, import_animations=True)
assert arm.get("hfw_has_control_rig") is True
assert arm.get("hfw_rig_archetype") == "theropod"
assert float(arm["hfw_rig_master"]) == 0.0  # Starts in Anim Mode when HFW clips are imported!
assert "hfw_ik_fk_legs" in arm and "hfw_rig_lock_secondary" in arm and "hfw_rig_detail" in arm
assert arm.animation_data is not None and arm.animation_data.action is not None

# 1. Sample pure game animation pose at frame 20 (hfw_rig_master = 0.0)
bpy.context.scene.frame_set(20)
bpy.context.view_layer.update()
hips_anim = arm.pose.bones["hipsBone"].matrix.to_translation().copy()
foot_anim = arm.pose.bones["L_leg_sjnt_3"].matrix.to_translation().copy()

# Verify animation actually moves joints between frame 1 and frame 20
bpy.context.scene.frame_set(1)
bpy.context.view_layer.update()
hips_f1 = arm.pose.bones["hipsBone"].matrix.to_translation().copy()
assert (hips_anim - hips_f1).length > 0.05

# 1b. ZERO GHOST ANIMATION CHECK: at mid-motion frame 20 with game_def Action STILL active,
# switching hfw_rig_master = 1.0 must snap 100% of DEF bones (controlled & uncontrolled) to exact rest!
bpy.context.scene.frame_set(20)
for ik_val in (1.0, 0.5, 0.0):
    arm["hfw_rig_master"] = 1.0
    arm["hfw_ik_fk_legs"] = ik_val
    arm["hfw_ik_fk_arms"] = ik_val
    arm["hfw_rig_lock_secondary"] = 1.0
    arm.update_tag()
    bpy.context.view_layer.update()
    for pb in arm.pose.bones:
        if pb.name.startswith(("CTRL_", "MCH_", "WGT_")):
            continue
        dp = (pb.matrix.to_translation() - pb.bone.matrix_local.to_translation()).length
        q1 = pb.bone.matrix_local.to_quaternion().normalized()
        q2 = pb.matrix.to_quaternion().normalized()
        if q1.dot(q2) < 0.0:
            q2 = -q2
        ang = math.degrees(q1.rotation_difference(q2).angle)
        assert dp < 0.002, f"Ghost pos leak on {{pb.name}} (ik={{ik_val}}): {{dp}}m"
        assert ang < 0.5, f"Ghost rot leak on {{pb.name}} (ik={{ik_val}}): {{ang}}deg"

# 1c. Verify Multi-Bone CTRL_Chest IK <-> Spine FK switch, BIG FK widgets, show_in_front, dynamic collections & IK<->FK snap!
assert arm.show_in_front is True
assert "CTRL_Chest" in arm.pose.bones and "CTRL_Spine_Mid" in arm.pose.bones and "hfw_ik_fk_torso" in arm
assert arm.pose.bones["CTRL_Chest"].custom_shape.name == "WGT_HFW_Chest"
assert arm.pose.bones["CTRL_FK_Leg_L_0"].custom_shape_scale_xyz[0] >= 0.24
rig_meta = json.loads(arm["hfw_ctrl_rig_meta"])
slots = rig_meta["collection_slots"]
for skey, expected_nm in (
    ("body", "Torso, Chest & Head"),
    ("torso_ik", "Chest & Back IK"),
    ("spine_fk", "Spine & Chest FK"),
    ("tail", "Tail FK"),
    ("legs_ik", "Legs IK"),
    ("legs_fk", "Legs FK"),
    ("arms_ik", "Forelimbs IK"),
    ("arms_fk", "Forelimbs FK"),
    ("hands_feet", "Claws & Talons"),
    ("frills", "Feathers & Blades"),
    ("weapons", "Jaw Saws & Launchers"),
):
    assert slots.get(skey) == expected_nm, f"Expected slot {{skey}}={{expected_nm}}, got {{slots.get(skey)}}"
    assert expected_nm in arm.data.collections, f"Missing dynamic collection {{expected_nm}}"
assert "plates" in slots and slots["plates"] in arm.data.collections

# Verify Multi-Bone CTRL_Chest Move (G) arches LOWER + MID + UPPER SPINE, `hfw_rig_chest = 0.0` disables Chest IK cleanly, AND switches cleanly to Pure Spine FK!
arm["hfw_rig_master"] = 1.0
arm["hfw_ik_fk_torso"] = 1.0
arm["hfw_rig_chest"] = 1.0
arm.update_tag()
bpy.context.view_layer.update()
mid_sp0 = arm.pose.bones["C_spine_sjnt_3"].matrix.to_translation().copy()
head_c0 = arm.pose.bones["C_head_sjnt_0"].matrix.to_translation().copy()
arm.pose.bones["CTRL_Chest"].location = (0.0, 0.0, 0.35)
arm.pose.bones["CTRL_Chest"].rotation_mode = "XYZ"
arm.pose.bones["CTRL_Chest"].rotation_euler = Euler((math.radians(18), 0.0, math.radians(-20)), "XYZ")
bpy.context.view_layer.update()
mid_sp1 = arm.pose.bones["C_spine_sjnt_3"].matrix.to_translation().copy()
head_c1 = arm.pose.bones["C_head_sjnt_0"].matrix.to_translation().copy()
assert (mid_sp1 - mid_sp0).length > 0.06, f"Multi-bone Chest IK did not arch mid-spine! ({{(mid_sp1 - mid_sp0).length}}m)"
assert (head_c1 - head_c0).length > 0.25
# Dragging `hfw_rig_chest = 0.0` must smoothly disable Chest IK back to Spine FK rest pose!
arm["hfw_rig_chest"] = 0.0
arm.update_tag()
bpy.context.view_layer.update()
assert (arm.pose.bones["C_head_sjnt_0"].matrix.to_translation() - head_c0).length < 0.005
arm["hfw_rig_chest"] = 1.0
arm.update_tag()
bpy.context.view_layer.update()
# Turn Chest IK OFF -> Spine FK (zero-pop snap + hides Chest IK + shows Spine FK!)
cr.switch_limb_ik_fk(arm, group="torso", to_ik=False, snap_pose=True)
assert float(arm["hfw_ik_fk_torso"]) == 0.0
assert arm.data.collections[slots["spine_fk"]].is_visible is True
assert arm.data.collections[slots["torso_ik"]].is_visible is False
assert (arm.pose.bones["C_head_sjnt_0"].matrix.to_translation() - head_c1).length < 0.01
cr.switch_limb_ik_fk(arm, group="torso", to_ik=True, snap_pose=True)
cr.reset_control_rig_pose(arm)

# Verify exact L->R Mirror Pose across IK legs AND sequenced feathers!
arm.pose.bones["CTRL_IK_Leg_L"].location = (0.18, 0.25, 0.42)
bpy.context.view_layer.update()
n_mir = cr.mirror_control_rig_pose(arm, source_side="L")
assert n_mir >= 80
p_ll = arm.pose.bones["L_leg_sjnt_3"].matrix.to_translation()
p_lr = arm.pose.bones["R_leg_sjnt_3"].matrix.to_translation()
assert abs(p_ll.x + p_lr.x) < 0.015 and abs(p_ll.y - p_lr.y) < 0.015 and abs(p_ll.z - p_lr.z) < 0.015
cr.reset_control_rig_pose(arm)

# Move IK Leg and assert thigh armor plate + dewclaw + foot move together, then zero-pop snap to FK!
arm["hfw_rig_master"] = 1.0
arm["hfw_ik_fk_legs"] = 1.0
arm.update_tag()
bpy.context.view_layer.update()
plate_r0 = arm.pose.bones["BackLegPlate_L_03_Lx_helper"].matrix.to_translation().copy()
dew_r0 = arm.pose.bones["L_thumb_sjnt_0"].matrix.to_translation().copy()
arm.pose.bones["CTRL_IK_Leg_L"].location = (0.0, 0.25, 0.40)
bpy.context.view_layer.update()
foot_ik_p = arm.pose.bones["L_leg_sjnt_3"].matrix.to_translation().copy()
assert (arm.pose.bones["BackLegPlate_L_03_Lx_helper"].matrix.to_translation() - plate_r0).length > 0.04
assert (arm.pose.bones["L_thumb_sjnt_0"].matrix.to_translation() - dew_r0).length > 0.20

cr.switch_limb_ik_fk(arm, group="legs", to_ik=False, snap_pose=True)
assert float(arm["hfw_ik_fk_legs"]) == 0.0
assert arm.data.collections[slots["legs_fk"]].is_visible is True
assert arm.data.collections[slots["legs_ik"]].is_visible is False
assert (arm.pose.bones["L_leg_sjnt_3"].matrix.to_translation() - foot_ik_p).length < 0.005

# Edit in pure FK mode -> must visibly move leg -> then zero-pop snap back to IK!
pb_fk = arm.pose.bones["CTRL_FK_Leg_L_0"]
pb_fk.rotation_mode = "XYZ"
pb_fk.rotation_euler = Euler((math.radians(-35), 0.0, 0.0), "XYZ")
bpy.context.view_layer.update()
foot_fk_moved = arm.pose.bones["L_leg_sjnt_3"].matrix.to_translation().copy()
assert (foot_fk_moved - foot_ik_p).length > 0.15

cr.switch_limb_ik_fk(arm, group="legs", to_ik=True, snap_pose=True)
assert float(arm["hfw_ik_fk_legs"]) == 1.0
assert arm.data.collections[slots["legs_ik"]].is_visible is True
assert arm.data.collections[slots["legs_fk"]].is_visible is False
assert (arm.pose.bones["L_leg_sjnt_3"].matrix.to_translation() - foot_fk_moved).length < 0.005
cr.reset_control_rig_pose(arm)

# 1d. Verify v1.0.0 Studio Rig features: Head IK + Chest<->World Space, Master Talon Curls, Looping Chainsaws & Tail Spline IK!
assert "CTRL_IK_Head" in arm.pose.bones and "hfw_ik_fk_head" in arm and "hfw_head_ik_space" in arm
assert "CTRL_Saw_Spin" in arm.pose.bones and "hfw_saw_speed" in arm
assert len(rig_meta.get("curl_ctrls") or []) >= 6
assert len(rig_meta.get("spline_ctrls") or []) >= 4

# Head IK + Chest<->World lock + zero-pop IK->FK snap
arm["hfw_ik_fk_head"] = 1.0
arm["hfw_head_ik_space"] = 0.0
arm.update_tag(); bpy.context.view_layer.update()
hd_r0 = arm.pose.bones["C_head_sjnt_0"].matrix.to_translation().copy()
arm.pose.bones["CTRL_IK_Head"].location = (0.30, 0.35, 0.20)
arm.update_tag(); bpy.context.view_layer.update()
hd_ik_p = arm.pose.bones["C_head_sjnt_0"].matrix.to_translation().copy()
assert (hd_ik_p - hd_r0).length > 0.20
cr.switch_limb_ik_fk(arm, group="head", to_ik=False, snap_pose=True)
assert (arm.pose.bones["C_head_sjnt_0"].matrix.to_translation() - hd_ik_p).length < 0.015
arm["hfw_ik_fk_head"] = 1.0; arm["hfw_head_ik_space"] = 1.0
arm.pose.bones["CTRL_IK_Head"].location = (0.0, 0.0, 0.0)
arm.update_tag(); bpy.context.view_layer.update()
hd_w0 = arm.pose.bones["C_head_sjnt_0"].matrix.to_translation().copy()
arm.pose.bones["CTRL_COG"].location = (0.0, 0.0, -0.30)
arm.update_tag(); bpy.context.view_layer.update()
assert (arm.pose.bones["C_head_sjnt_0"].matrix.to_translation() - hd_w0).length < 0.06
arm["hfw_ik_fk_head"] = 0.0; arm["hfw_head_ik_space"] = 0.0
cr.reset_control_rig_pose(arm)

# Looping Mouth Chainsaws (auto-spin speed AND manual dial, zero drift at rest!)
saw_teeth = [pb.name for pb in arm.pose.bones if not pb.name.startswith(("CTRL_", "MCH_", "WGT_")) and "sawblade" in pb.name.lower()]
assert len(saw_teeth) == 52
bpy.context.scene.frame_set(1); arm["hfw_saw_speed"] = 0.0
arm.update_tag(); bpy.context.view_layer.update()
saw_p0 = {{b: arm.pose.bones[b].matrix.to_translation().copy() for b in saw_teeth}}
arm["hfw_saw_speed"] = 1.0; arm.update_tag()
bpy.context.scene.frame_set(15); bpy.context.view_layer.update()
assert min((arm.pose.bones[b].matrix.to_translation() - saw_p0[b]).length for b in saw_teeth) > 0.10
arm["hfw_saw_speed"] = 0.0; bpy.context.scene.frame_set(1)
arm.pose.bones["CTRL_Saw_Spin"].rotation_euler = Euler((0.0, 2.0, 0.0), "XYZ")
arm.update_tag(); bpy.context.view_layer.update()
assert min((arm.pose.bones[b].matrix.to_translation() - saw_p0[b]).length for b in saw_teeth) > 0.08
cr.reset_control_rig_pose(arm)

# Master Finger/Talon Curl lever curls all phalanges cleanly
curl0 = rig_meta["curl_ctrls"][0]
arm.pose.bones[curl0].rotation_euler = Euler((1.0, 0.0, 0.0), "XYZ")
arm.update_tag(); bpy.context.view_layer.update()
curled_cnt = sum(
    1 for pb in arm.pose.bones
    if not pb.name.startswith(("CTRL_", "MCH_", "WGT_"))
    and any(k in pb.name.lower() for k in ("finger", "toe", "claw", "talon"))
    and math.degrees(pb.bone.matrix_local.to_quaternion().rotation_difference(pb.matrix.to_quaternion()).angle) > 15.0
)
assert curled_cnt >= 2
cr.reset_control_rig_pose(arm)

# Toggleable B-Bone Spline IK on Tail (rigid uniform armor scale!)
arm["hfw_ik_fk_spline"] = 1.0; arm.update_tag(); bpy.context.view_layer.update()
t_tip0 = arm.pose.bones["C_tail_sjnt_7"].matrix.to_translation().copy()
arm.pose.bones[rig_meta["spline_ctrls"][-1]].location = (0.6, 0.0, 0.5)
arm.update_tag(); bpy.context.view_layer.update()
assert (arm.pose.bones["C_tail_sjnt_7"].matrix.to_translation() - t_tip0).length > 0.35
assert max(abs(s - 1.0) for s in arm.pose.bones["C_tail_sjnt_7"].matrix.to_scale()) < 0.005
arm["hfw_ik_fk_spline"] = 0.0
cr.reset_control_rig_pose(arm)

# Switch back to pure game anim before baking
arm["hfw_rig_master"] = 0.0
arm["hfw_ik_fk_legs"] = 1.0
arm["hfw_ik_fk_arms"] = 1.0
arm.update_tag()
bpy.context.view_layer.update()

# 2. Bake Game Anim -> Control Rig (frames 18..22) and verify millimeter accuracy in BOTH IK & FK!
baked_ctrl = cr.bake_action_to_control_rig(arm, frame_start=18, frame_end=22, activate_baked=True)
assert baked_ctrl is not None
assert float(arm["hfw_rig_master"]) == 1.0
assert float(arm["hfw_saw_speed"]) == 0.0
assert float(arm["hfw_head_ik_space"]) == 0.0
bpy.context.scene.frame_set(20)
for ik_val in (1.0, 0.0):
    arm["hfw_ik_fk_legs"] = ik_val
    arm.update_tag()
    bpy.context.view_layer.update()
    hips_ctrl = arm.pose.bones["hipsBone"].matrix.to_translation().copy()
    foot_ctrl = arm.pose.bones["L_leg_sjnt_3"].matrix.to_translation().copy()
    assert (hips_ctrl - hips_anim).length < 0.015, f"Hips bake error (ik={{ik_val}}): {{(hips_ctrl - hips_anim).length}}"
    assert (foot_ctrl - foot_anim).length < 0.025, f"Foot bake error (ik={{ik_val}}): {{(foot_ctrl - foot_anim).length}}"

# 3. Verify live IK edit moves DEF foot cleanly and survives Bake Control Rig -> Game Skeleton (`DEF`)!
arm["hfw_ik_fk_legs"] = 1.0
arm.update_tag()
bpy.context.view_layer.update()
foot_ctrl = arm.pose.bones["L_leg_sjnt_3"].matrix.to_translation().copy()
arm.pose.bones["CTRL_IK_Leg_L"].location.z += 0.50
arm.pose.bones["CTRL_IK_Leg_L"].keyframe_insert(data_path="location", frame=20)
bpy.context.view_layer.update()
foot_lifted = arm.pose.bones["L_leg_sjnt_3"].matrix.to_translation().copy()
assert (foot_lifted - foot_ctrl).length > 0.30

# 4. Bake Control Rig -> Game Skeleton (`DEF`) and verify clean exportable DEF keys!
baked_def = cr.bake_control_rig_to_deform(arm, frame_start=19, frame_end=21, activate_baked=True)
assert baked_def is not None
assert float(arm["hfw_rig_master"]) == 0.0
bpy.context.scene.frame_set(20)
bpy.context.view_layer.update()
foot_final = arm.pose.bones["L_leg_sjnt_3"].matrix.to_translation().copy()
assert (foot_final - foot_lifted).length < 0.015, f"DEF bake error too high: {{(foot_final - foot_lifted).length}}"
print("BLENDER_RIG_TEST_OK")
''',
        encoding="utf-8",
    )

    proc = subprocess.run(
        [str(cfg.blender_exe), "--background", "--factory-startup", "--python-exit-code", "1", "--python", str(test_py)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Blender test failed:\nSTDOUT:\n{proc.stdout[-2000:]}\nSTDERR:\n{proc.stderr[-2000:]}"
    assert "BLENDER_RIG_TEST_OK" in proc.stdout
    combined_log = proc.stdout + "\n" + proc.stderr
    assert "Dependency cycle" not in combined_log
    assert "Error in PyDriver" not in combined_log
