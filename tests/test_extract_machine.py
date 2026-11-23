"""Phase 2 tests: Machine assembly, plate attachment resolution, skinning, and TextureSet decoding
on Slitherfang, Clawstrider, and Slaughterspine.
"""

from __future__ import annotations

import numpy as np
import pytest

from hfw_toolkit.config import ToolkitConfig
from hfw_toolkit.decima import AssetCatalog, HFWGame, ObjectId
from hfw_toolkit.extract import assemble_machine, resolve_machine_entity_id, unpack_texture_set


@pytest.fixture(scope="module")
def env():
    cfg = ToolkitConfig.load()
    game = HFWGame(cfg)
    catalog = AssetCatalog(cfg.index_db_path)
    yield game, catalog
    catalog.close()
    game.close()


@pytest.mark.parametrize(
    "machine_name",
    ["Slitherfang", "Clawstrider", "Slaughterspine", "Thunderjaw"],
)
def test_assemble_target_machines(env, machine_name: str) -> None:
    game, catalog = env
    oid, code, disp = resolve_machine_entity_id(game, catalog, machine_name)
    asm = assemble_machine(game, oid, codename=code, display_name=disp, variant_index=0, lod_index=0)

    n_joints = len(asm.skeleton.joints)
    n_helpers = len(asm.skeleton.helpers)
    skinned = [p for p in asm.parts if p.is_skinned]
    static_attached = [p for p in asm.parts if not p.is_skinned and p.parent_joint_index is not None]
    static_unattached = [p for p in asm.parts if not p.is_skinned and p.parent_joint_index is None]

    print(
        f"\n[{disp} ({code}) | {oid} | variant='{asm.variant_label}' ({len(asm.available_variants)} variants)]\n"
        f"  Skeleton: {n_joints} joints, {n_helpers} helpers\n"
        f"  Parts: total={len(asm.parts)} | skinned={len(skinned)} | static_attached={len(static_attached)} | static_unattached={len(static_unattached)}\n"
        f"  Materials: {len(asm.materials)} unique RenderEffectResources"
    )
    for v in asm.available_variants[:5]:
        print(f"    variant[{v.variant_index}]: {v.label}")

    assert n_joints >= 50
    assert len(asm.parts) >= 10
    assert len(skinned) >= 1
    assert len(static_attached) >= 5
    assert len(asm.materials) >= 3

    # Verify all skinned bone indices fall within [0, n_joints)
    for sp in skinned:
        for prim in sp.mesh.primitives:
            assert np.isfinite(prim.positions).all()
            if prim.bone_indices is not None:
                assert prim.bone_indices.max() < n_joints

    # Verify all attached static plate transforms are finite and not all collapsed at origin
    nonzero_offsets = 0
    for ap in static_attached:
        assert np.isfinite(ap.world_matrix).all()
        if np.linalg.norm(ap.world_matrix[:3, 3]) > 0.01:
            nonzero_offsets += 1
    assert nonzero_offsets >= 5

    # Verify at least one TextureSet from this machine unpacks cleanly with semantic maps!
    unpacked_ok = 0
    for mat in list(asm.materials.values())[:5]:
        for ts_id in mat.texture_set_ids[:2]:
            ts_obj = game.read_object(ObjectId.parse(ts_id), read_subgroups=True)
            uts = unpack_texture_set(game, ts_obj, max_dimension=512)
            if uts.semantic_maps:
                unpacked_ok += 1
                print(f"    Unpacked TextureSet {uts.name} ({ts_id}): maps={list(uts.semantic_maps.keys())}")
                break
        if unpacked_ok >= 2:
            break
    assert unpacked_ok >= 1


def test_deduplicated_machine_catalog(env) -> None:
    _game, catalog = env
    machines = catalog.list_machines()
    names = [m["display_name"] for m in machines]
    # Zero duplicate display names and no child turrets / cutscene props
    assert len(names) == len(set(names)), f"Duplicate machines in catalog: {names}"
    assert 38 <= len(machines) <= 46
    by_name = {m["display_name"]: m for m in machines}
    for req in ("Thunderjaw", "Tremortusk", "Slitherfang", "Slaughterspine", "Clawstrider", "Frostclaw", "Fireclaw", "Bellowback", "Shell-Walker", "Shellsnapper"):
        assert req in by_name, f"Missing expected machine {req}"
        assert by_name[req]["part_count"] >= 15


def test_thunderjaw_bind_pose_and_child_entities(env) -> None:
    game, catalog = env
    # 1. Thunderjaw InitialRepresentationSkeleton 163/163 bind pose: ZERO plates collapsed at (0,0,0)!
    oid, code, disp = resolve_machine_entity_id(game, catalog, "Thunderjaw")
    tj = assemble_machine(game, oid, codename=code, display_name=disp, include_broken_states=False)
    static_att = [p for p in tj.parts if not p.is_skinned and p.parent_joint_index is not None]
    at_origin = [p.name for p in static_att if np.linalg.norm(p.world_matrix[:3, 3]) < 0.05]
    assert len(at_origin) == 0, f"Thunderjaw still has parts at origin: {at_origin}"
    for jn in ("R_backShield", "L_backShield", "R_BeltGun", "L_BeltGun", "L_TailShieldJnt"):
        j = tj.skeleton.joint_by_name[jn]
        assert np.linalg.norm(j.world_matrix[:3, 3]) > 2.0, f"Thunderjaw joint {jn} collapsed: {j.world_matrix[:3, 3]}"

    # 2. ChildEntityComponentResource attachments (Tremortusk cannons & Shell-Walker cargo/shield!)
    for mname, min_children in (("Tremortusk", 2), ("Shell-Walker", 1)):
        c_oid, c_code, c_disp = resolve_machine_entity_id(game, catalog, mname)
        asm = assemble_machine(game, c_oid, codename=c_code, display_name=c_disp, include_broken_states=False)
        child_parts = [p for p in asm.parts if p.name.startswith("Child_")]
        assert len(child_parts) >= min_children, f"{mname} missing ChildEntity parts: {len(child_parts)}"

    # 3. PhysicsSkeleton.ReferencePose recovery on Frostclaw (HasBindPose=False, no IRS)
    f_oid, f_code, f_disp = resolve_machine_entity_id(game, catalog, "Frostclaw")
    fc = assemble_machine(game, f_oid, codename=f_code, display_name=f_disp, include_broken_states=False)
    nz_joints = sum(1 for j in fc.skeleton.joints if np.linalg.norm(j.world_matrix[:3, 3]) > 0.01)
    assert nz_joints >= len(fc.skeleton.joints) - 5


def test_clawstrider_variants_and_redeye_watcher_body(env) -> None:
    game, catalog = env

    # 1. Clawstrider must default to base Melee Clawstrider (NOT Elemental!) and expose official Acid/Fire/Apex variants in SQLite
    cs_oid, cs_code, cs_disp = resolve_machine_entity_id(game, catalog, "Clawstrider")
    assert cs_code == "raider", f"Clawstrider codename regressed to {cs_code}"
    sql_vars = catalog.list_machine_variants(cs_oid.group_id, cs_oid.object_index)
    assert len(sql_vars) >= 6, f"Expected >= 6 indexed Clawstrider variants, got {sql_vars}"
    assert sql_vars[0]["label"].startswith("Clawstrider (Group 19913"), f"Clawstrider v0 must be Melee Clawstrider 19913, got {sql_vars[0]}"
    assert any("Acid Clawstrider" in v["label"] for v in sql_vars), f"Missing Acid Clawstrider variant: {sql_vars}"
    assert any("Fire Clawstrider" in v["label"] for v in sql_vars), f"Missing Fire Clawstrider variant: {sql_vars}"
    assert any("Apex" in v["label"] for v in sql_vars), f"Missing Apex Clawstrider variant: {sql_vars}"

    cs_melee = assemble_machine(game, cs_oid, codename=cs_code, display_name=cs_disp, variant_index=0, include_broken_states=False)
    assert cs_melee.variant_label.startswith("Clawstrider (Group 19913")
    assert len(cs_melee.parts) >= 65
    cs_elem = assemble_machine(game, cs_oid, codename=cs_code, display_name=cs_disp, variant_index=1, include_broken_states=False)
    assert any(w in cs_elem.variant_label for w in ("Acid", "Fire", "Elemental"))
    assert len(cs_elem.parts) >= 55
    assert cs_melee.available_variants[0].smr_id != cs_elem.available_variants[1].smr_id

    # 2. Bellowback must expose all 4 official elemental variants (Acid, Fire, Frost, Apex/Adhesive)!
    bb_oid, _, _ = resolve_machine_entity_id(game, catalog, "Bellowback")
    bb_vars = [v["label"] for v in catalog.list_machine_variants(bb_oid.group_id, bb_oid.object_index)]
    for req_bb in ("Acid Bellowback", "Fire Bellowback", "Frost Bellowback", "Apex Bellowback"):
        assert any(req_bb in lbl for lbl in bb_vars), f"Missing {req_bb} in {bb_vars}"

    # 3. Redeye Watcher (4760:42) must load FULL LaserScout 4267:3 body + BOTH LEGS (44.5k legs + 59.5k torso >= 100k tris, Z_min <= 0.05)!
    rw_oid, rw_code, rw_disp = resolve_machine_entity_id(game, catalog, "Redeye Watcher")
    rw = assemble_machine(game, rw_oid, codename=rw_code, display_name=rw_disp, variant_index=0, include_broken_states=False)
    skinned_parts = [p for p in rw.parts if p.is_skinned]
    skinned_tris = sum(sum(len(pr.indices) // 3 for pr in p.mesh.primitives) for p in skinned_parts)
    all_pts = np.concatenate([pr.positions for p in skinned_parts for pr in p.mesh.primitives], axis=0)
    static_plates = sum(1 for p in rw.parts if not p.is_skinned)
    assert skinned_tris >= 100_000, f"Redeye Watcher missing legs or body mesh: skinned_tris={skinned_tris}"
    assert float(all_pts[:, 2].min()) <= 0.05, f"Redeye Watcher legs missing (Z_min={all_pts[:, 2].min():.3f})"
    assert static_plates >= 5, f"Redeye Watcher missing static plates: {static_plates}"
