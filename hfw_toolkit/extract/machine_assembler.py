"""Complete Horizon Forbidden West Machine & Armor-Plate Assembler.

Resolves:
- ControlledEntityResource -> SkinnedModelResource (Skeleton + EdgeAnim bind pose + SkeletonHelpers + base parts)
- All associated DestructibilityResources (base + supergroup variants such as Standard / Apex!)
- Every DestructibilityPart's InitialState (Intact) + AllAvailableStates (Damaged / Broken / Exposed)
- Exact attachment transforms:
  * RegularSkinnedMeshResource -> model bind-pose space (identity transform + full Skeleton joint weights)
  * StaticMeshResource -> local helper/joint space (parented to resolved Skeleton joint / OrientationHelper + Mat44 Offset)
- All referenced Materials (RenderEffectResource) and TextureSets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hfw_toolkit.decima.catalog import AssetCatalog, MACHINE_CODENAMES
from hfw_toolkit.decima.reader import ObjectId, UUIDRef
from hfw_toolkit.decima.streaming_graph import HFWGame
from hfw_toolkit.extract.materials import MaterialInfo, apply_shader_overrides, parse_material
from hfw_toolkit.extract.meshes import DecodedMesh, decode_mesh_hierarchy
from hfw_toolkit.extract.skeleton import (
    SkeletonData,
    add_skeleton_helpers,
    decima_mat44_to_np,
    extract_physics_skeleton_world_matrices,
    extract_skeleton,
)


@dataclass(slots=True)
class AssembledPart:
    part_id: str
    name: str
    collection: str  # "Body" | "Plates_Intact" | "Plates_Broken_States" | "Weapons_And_Components"
    source_kind: str  # "SkinnedModelPart" | "DestructibilityIntact" | "DestructibilityAltState"
    is_skinned: bool
    parent_joint_index: int | None
    parent_joint_name: str | None
    helper_name: str | None
    local_offset_matrix: np.ndarray  # (4, 4) relative to parent joint (for static parts)
    world_matrix: np.ndarray  # (4, 4) bind-pose world matrix (for static parts; Identity for skinned)
    mesh: DecodedMesh
    enabled_by_default: bool = True
    state_index: int = 0
    catalog_part_name: str = ""  # Official HFW Focus/Catalog part name (e.g. "Metalbite Sac", "Blaze Sac", "Adhesive Sac")


@dataclass(slots=True)
class MachineVariantOption:
    variant_index: int
    label: str
    destructibility_id: str
    group_id: int
    part_count: int
    smr_id: str | None = None
    official_name: str = ""
    official_description: str = ""


@dataclass(slots=True)
class MachineAssembly:
    entity_id: str
    codename: str
    display_name: str
    variant_label: str
    available_variants: list[MachineVariantOption]
    skeleton: SkeletonData
    parts: list[AssembledPart] = field(default_factory=list)
    materials: dict[str, MaterialInfo] = field(default_factory=dict)  # material_id -> MaterialInfo
    official_variant_name: str = ""
    official_variant_description: str = ""
    shader_override_count: int = 0
    shader_override_resource_ids: list[str] = field(default_factory=list)


def _classify_part_collection(name: str, is_skinned: bool, is_alt_state: bool) -> str:
    if is_alt_state:
        return "Plates_Broken_States"
    low = name.lower()
    if any(
        kw in low
        for kw in (
            "blaster",
            "cannon",
            "launcher",
            "gun",
            "weapon",
            "canister",
            "sac",
            "battery",
            "sparker",
            "cell",
            "tank",
            "container",
            "grinder",
            "saw",
            "claw",
            "tail",
            "radar",
            "antenna",
            "emitter",
            "coil",
            "resource",
            "cargo",
        )
    ):
        return "Weapons_And_Components"
    if not is_skinned or "plate" in low or "armor" in low or "armour" in low or "shell" in low or "fin" in low:
        return "Plates_Intact"
    return "Body"


def _skeleton_matches(skel_a: dict[str, Any] | None, skel_b: dict[str, Any] | None) -> bool:
    if skel_a is None or skel_b is None:
        return False
    if skel_a.get("_id") == skel_b.get("_id"):
        return True
    ja = skel_a.get("Joints", ())
    jb = skel_b.get("Joints", ())
    if len(ja) < 10 or len(jb) < 10:
        return False
    if len(ja) == len(jb):
        la = skel_a.get("SkeletonLayoutHash")
        lb = skel_b.get("SkeletonLayoutHash")
        if la is None or lb is None or la == lb:
            return True
    # Allow subspecies skeletons (e.g. Grazer vs Fanghorn, Charger vs Lancehorn) sharing >= 65% joint UUIDs/names
    ua = {j["JointID"].uuid for j in ja if isinstance(j.get("JointID"), UUIDRef)}
    ub = {j["JointID"].uuid for j in jb if isinstance(j.get("JointID"), UUIDRef)}
    if ua and ub:
        overlap = len(ua & ub) / max(1, min(len(ua), len(ub)))
        if overlap >= 0.65:
            return True
    na = {j.get("Name", "") for j in ja if j.get("Name")}
    nb = {j.get("Name", "") for j in jb if j.get("Name")}
    if na and nb:
        return (len(na & nb) / max(1, min(len(na), len(nb)))) >= 0.65
    return False


def _smr_part_count(smr_obj: dict[str, Any] | None) -> int:
    if not smr_obj:
        return 0
    return len(smr_obj.get("ModelPartResources", ()))


def _mesh_has_geometry(game: HFWGame, mesh_obj: dict[str, Any] | None, depth: int = 0) -> bool:
    """Fast check (no vertex decoding) whether a MeshResource hierarchy contains at least one real geometry mesh."""
    if mesh_obj is None or depth > 5:
        return False
    mt = mesh_obj.get("_type", "")
    if mt in ("RegularSkinnedMeshResource", "StaticMeshResource"):
        if not mesh_obj.get("Primitives"):
            return False
        # Skip shadow-caster-only meshes (exact same DrawFlags rule as `meshes.py:_is_shadow_caster_only`)
        df = mesh_obj.get("DrawFlags")
        if isinstance(df, dict) and ((df.get("Data", 0) >> 4) & 3) == 1:
            return False
        return True
    if mt == "LodMeshResource":
        # Check LOD 0 (`RuntimeMeshes[0]`), matching `decode_mesh_hierarchy(..., lod_index=0)` so shadow-only LOD0
        # proxies like Redeye Watcher `47852:0` (`61995:1`) are NOT mistaken for a visible body mesh!
        rms = mesh_obj.get("RuntimeMeshes", ())
        if not rms:
            return False
        mr0 = rms[0]
        sub = game.resolve_ref(mr0.get("Mesh") if isinstance(mr0, dict) else mr0)
        return _mesh_has_geometry(game, sub, depth + 1)
    if mt == "MultiMeshResource":
        for mr in mesh_obj.get("Meshes", ()):
            sub = game.resolve_ref(mr)
            if _mesh_has_geometry(game, sub, depth + 1):
                return True
        return False
    if "Mesh" in mesh_obj:
        return _mesh_has_geometry(game, game.resolve_ref(mesh_obj.get("Mesh")), depth + 1)
    return False


def _smr_mesh_part_count(game: HFWGame, smr_obj: dict[str, Any] | None) -> int:
    """Count `ModelPartResources` in `smr_obj` that actually resolve to real geometry (ignores empty placeholders like Redeye Watcher 47852:0!)."""
    if not smr_obj:
        return 0
    count = 0
    for mref in smr_obj.get("ModelPartResources", ()):
        mpr = game.resolve_ref(mref)
        if not mpr:
            continue
        mroot = game.resolve_ref(mpr.get("MeshResource"))
        if _mesh_has_geometry(game, mroot):
            count += 1
    return count


def _extract_collectable_robot_info(
    game: HFWGame,
    group_id: int,
) -> tuple[str, str, dict[str, str]]:
    """Read official HFW variant name, description & DestructibilityPartTag->PartName map from `CollectableRobotComponentResource`.

    Cached on `game` so repeated queries across variant discovery and assembly are instant (<0.1ms).
    """
    cache: dict[int, tuple[str, str, dict[str, str]]] = getattr(game, "_cr_info_cache", None)
    if cache is None:
        cache = {}
        setattr(game, "_cr_info_cache", cache)
    if group_id in cache:
        return cache[group_id]

    result: tuple[str, str, dict[str, str]] = ("", "", {})
    if (
        "CollectableRobotComponentResource" not in game.graph.group_types(group_id)
        or game.graph.group(group_id).num_objects > 8_000
    ):
        cache[group_id] = result
        return result

    try:
        objs = game.read_group(group_id, read_subgroups=True)
    except Exception:
        cache[group_id] = result
        return result

    def _en_text(ref_or_obj: Any) -> str:
        o = game.resolve_ref(ref_or_obj) if not isinstance(ref_or_obj, (dict, str, type(None))) else ref_or_obj
        if isinstance(o, dict) and o.get("_texts"):
            return str(o["_texts"][0]).strip()
        return ""

    for o in objs:
        if o.get("_type") != "CollectableRobotComponentResource":
            continue
        for cref in o.get("CollectableRobots", ()):
            cr = game.resolve_ref(cref)
            if not cr:
                continue
            name = _en_text(cr.get("LocalizedName"))
            desc = _en_text(cr.get("LocalizedDescription"))
            tag_to_part: dict[str, str] = {}
            for pref in cr.get("RobotParts", ()):
                po = game.resolve_ref(pref)
                if not po:
                    continue
                pnm = _en_text(po.get("Name"))
                if not pnm:
                    continue
                sc = game.resolve_ref(po.get("PartScannableBase"))
                if sc:
                    dtag = sc.get("DestructibilityPartTag")
                    tu = getattr(dtag, "uuid", None) if dtag is not None else None
                    if tu:
                        tag_to_part[tu] = pnm
            if name or tag_to_part:
                result = (name, desc, tag_to_part)
                break
        if result[0] or result[2]:
            break

    cache[group_id] = result
    return result


def collect_variant_shader_overrides(
    game: HFWGame,
    variant_group_id: int,
) -> tuple[dict[int, ObjectId], dict[int, dict[str, Any]], list[str]]:
    """Collect Decima `RenderEffectShaderOverridesResource` texture & variable overrides for a variant group.

    Fast (<5ms): uses `game.graph.group_types(gid)` to only read groups that actually contain
    `RenderEffectShaderOverridesResource` across depth 0..2 subgroups (skipping shared root groups <= 20).
    """
    tex_overrides: dict[int, ObjectId] = {}
    var_overrides: dict[int, dict[str, Any]] = {}
    override_ids: list[str] = []

    visited: set[int] = {variant_group_id}
    frontier: list[int] = [variant_group_id]
    candidate_gids: list[int] = []

    for depth in range(3):
        nxt: list[int] = []
        for g in frontier:
            try:
                gmeta = game.graph.group(g)
            except Exception:
                continue
            if gmeta.num_objects > 12_000:
                continue
            if "RenderEffectShaderOverridesResource" in game.graph.group_types(g):
                candidate_gids.append(g)
            if depth < 2:
                for sg in game.graph.group_subgroups(g):
                    if sg not in visited and sg > 20:
                        visited.add(sg)
                        nxt.append(sg)
        frontier = nxt[:240]

    for cg in candidate_gids:
        try:
            # Must read with read_subgroups=True so external/subgroup TextureResource refs populate!
            objs = game.read_group(cg, read_subgroups=True)
        except Exception:
            continue
        for o in objs:
            if o.get("_type") != "RenderEffectShaderOverridesResource":
                continue
            oid_str = str(o.get("_id", ""))
            if oid_str and oid_str not in override_ids:
                override_ids.append(oid_str)
            so = o.get("ShaderOverrides") or {}
            for tb in so.get("ShaderTextureOverrides", ()):
                bh = int(tb.get("BindingNameHash", 0))
                tr = tb.get("TextureResource")
                if not bh or tr is None or bh in tex_overrides:
                    continue
                if isinstance(tr, dict) and tr.get("_id") is not None:
                    tex_overrides[bh] = tr
                else:
                    tid = getattr(tr, "target_id", None)
                    if tid is not None:
                        tex_overrides[bh] = tid
                    else:
                        resolved = game.resolve_ref(tr)
                        if isinstance(resolved, dict) and resolved.get("_id") is not None:
                            tex_overrides[bh] = resolved
            for vb in so.get("ShaderVariableOverrides", ()):
                bh = int(vb.get("BindingNameHash", 0))
                if bh and bh not in var_overrides:
                    var_overrides[bh] = dict(vb)

    return tex_overrides, var_overrides, override_ids


def _has_variant_shader_overrides_fast(game: HFWGame, variant_group_id: int) -> bool:
    """Zero-I/O graph-metadata check (<0.1ms) for `RenderEffectShaderOverridesResource` in depth 0..2 subgroups."""
    visited: set[int] = {variant_group_id}
    frontier: list[int] = [variant_group_id]
    for depth in range(3):
        nxt: list[int] = []
        for g in frontier:
            try:
                gmeta = game.graph.group(g)
            except Exception:
                continue
            if gmeta.num_objects > 12_000:
                continue
            if "RenderEffectShaderOverridesResource" in game.graph.group_types(g):
                return True
            if depth < 2:
                for sg in game.graph.group_subgroups(g):
                    if sg not in visited and sg > 20:
                        visited.add(sg)
                        nxt.append(sg)
        frontier = nxt[:240]
    return False


_ALL_KNOWN_SPECIES_NAMES: frozenset[str] = frozenset(v.lower() for v in MACHINE_CODENAMES.values())

# Shared-chassis ControlledEntityResource groups that legitimately host multiple subspecies
_ALLOWED_SHARED_SPECIES: dict[int, frozenset[str]] = {
    4262: frozenset({"grazer", "fanghorn", "lancehorn"}),
    2911: frozenset({"charger", "strider", "lancehorn"}),
    3385: frozenset({"plowhorn", "grimhorn"}),
    1587: frozenset({"clawstrider"}),
}


def _infer_base_species_names(game: HFWGame, entity_oid: ObjectId, smr: dict[str, Any] | None) -> set[str]:
    """Infer the canonical HFW species display name(s) belonging to this entity's own CE/SMR group."""
    if entity_oid.group_id in _ALLOWED_SHARED_SPECIES:
        return set(_ALLOWED_SHARED_SPECIES[entity_oid.group_id])
    species: set[str] = set()
    # 1. Check CE's own CollectableRobotComponentResource if present
    cr_name, _, _ = _extract_collectable_robot_info(game, entity_oid.group_id)
    if cr_name:
        low = cr_name.lower()
        for sp in _ALL_KNOWN_SPECIES_NAMES:
            if sp in low:
                species.add(sp)
    # 2. Check TextureSet paths in CE and SMR groups (and 1-hop small subgroups)
    scan_gids: list[int] = [entity_oid.group_id]
    if smr is not None:
        sg = smr["_id"].group_id
        if sg not in scan_gids:
            scan_gids.append(sg)
    for base_g in list(scan_gids):
        for sub in game.graph.group_subgroups(base_g):
            if sub > 20 and sub not in scan_gids and "TextureSet" in game.graph.group_types(sub):
                if game.graph.group(sub).num_objects < 400:
                    scan_gids.append(sub)
    for g in scan_gids[:12]:
        try:
            objs = game.read_group(g, read_subgroups=False)
        except Exception:
            continue
        for o in objs:
            if o.get("_type") != "TextureSet":
                continue
            for desc in o.get("TextureDesc", ()):
                p = str(desc.get("Path", "")).lower().replace("\\", "/")
                if "models/characters/robots/" not in p:
                    continue
                after = p.split("models/characters/robots/", 1)[1]
                code = after.split("/", 1)[0]
                if code in ("dlc1", "shared", "common", "generic", "canisters", "weapons"):
                    if "inferno" in p or "firebear" in p:
                        code = "infernobear"
                    elif "cryo" in p or "frost" in p:
                        code = "cryobear"
                    else:
                        continue
                if code in MACHINE_CODENAMES:
                    species.add(MACHINE_CODENAMES[code].lower())
    return species


def _dr_matches_skeleton(
    game: HFWGame,
    dr: dict[str, Any],
    base_joint_uuids: set[str],
    base_joint_names: set[str],
    base_helper_names: set[str],
    local_objs: list[dict[str, Any]] | None = None,
) -> bool:
    """Fast local-only validation that a `DestructibilityResource`'s parts attach to `base_skel` (zero subgroup I/O!)."""
    if not base_joint_uuids and not base_joint_names:
        return True
    cp = dr.get("ConvertedParts", ())
    if not cp:
        return False
    dr_gid = dr["_id"].group_id if dr.get("_id") is not None else -1

    def _local_resolve(ref: Any) -> dict[str, Any] | None:
        if isinstance(ref, dict):
            return ref
        if ref is None:
            return None
        # Only dereference already-resolved `.target` or same-group objects (NEVER trigger recursive subgroup reads!)
        tgt = getattr(ref, "target", None)
        if isinstance(tgt, dict):
            return tgt
        tid = getattr(ref, "target_id", None)
        if tid is not None and local_objs is not None and tid.group_id == dr_gid and 0 <= tid.object_index < len(local_objs):
            return local_objs[tid.object_index]
        return None

    checked = 0
    matched = 0
    for pref in cp[:32]:
        dp = _local_resolve(pref)
        if not isinstance(dp, dict):
            continue
        jref = dp.get("AttachToJointID")
        juuid = jref.uuid if isinstance(jref, UUIDRef) else None
        bname = str(dp.get("BoneName", "") or "")
        if not juuid and not bname:
            st = _local_resolve(dp.get("InitialState"))
            mpr = _local_resolve(st.get("ModelPartResource")) if isinstance(st, dict) else None
            if isinstance(mpr, dict):
                mj = mpr.get("AttachToJointID")
                juuid = mj.uuid if isinstance(mj, UUIDRef) else None
                bname = str(mpr.get("HelperNode", "") or "")
        if not juuid and not bname:
            continue
        checked += 1
        bclean = bname.removesuffix("_Lx_helper").removesuffix("_Cx_helper").removesuffix("_helper")
        if (
            (juuid and juuid in base_joint_uuids)
            or (bname and (bname in base_joint_names or bname in base_helper_names))
            or (bclean and (bclean in base_joint_names or bclean in base_helper_names))
        ):
            matched += 1
    if checked >= 3:
        return (matched / checked) >= 0.25
    return True


def discover_machine_variants(
    game: HFWGame,
    entity_oid: ObjectId,
) -> tuple[dict[str, Any], dict[str, Any] | None, list[MachineVariantOption]]:
    """Locate a machine's `ControlledEntityResource`, its `SkinnedModelResource`, and all associated `DestructibilityResource` variants."""
    ce = game.read_object(entity_oid, read_subgroups=True)
    comps = [game.resolve_ref(r) for r in ce.get("EntityComponentResources", ())]
    direct_smr = next((c for c in comps if c and c["_type"] == "SkinnedModelResource"), None)
    direct_mesh_count = _smr_mesh_part_count(game, direct_smr)
    smr = direct_smr if direct_mesh_count > 0 else None

    # Resolve base Skeleton (from SMR if present, else from GraphAnimationManagerResource — e.g. Clawstrider/Grazer/Charger!)
    base_skel: dict[str, Any] | None = None
    if direct_smr is not None:
        base_skel = game.resolve_ref(direct_smr.get("Skeleton"))
    if base_skel is None:
        for c in comps:
            if c and c["_type"] == "GraphAnimationManagerResource":
                cand = game.resolve_ref(c.get("Skeleton"))
                if cand is not None:
                    base_skel = cand
                    break

    # Multi-subspecies machines (e.g. Clawstrider 1587:51, Grazer/Fanghorn 4262:32, Charger/Lancehorn 2911:47,
    # Plowhorn/Grimhorn 3385:27, Burrower) AND empty-placeholder-SMR machines (e.g. Redeye Watcher 4760:42 -> 4267:3)
    # store their real SkinnedModelResource(s) in Skeleton supergroups or variant subgroups!
    extra_smrs: dict[int, dict[str, Any]] = {}  # smr_gid -> SMR object
    extra_smr_counts: dict[int, int] = {}       # smr_gid -> real mesh part count
    if smr is not None:
        extra_smrs[smr["_id"].group_id] = smr
        extra_smr_counts[smr["_id"].group_id] = direct_mesh_count

    if base_skel is not None and smr is None:
        skel_gid = base_skel["_id"].group_id
        for cand_gid in game.graph.supergroups.get(skel_gid, ()):
            if "SkinnedModelResource" not in game.graph.group_types(cand_gid):
                continue
            if game.graph.group(cand_gid).num_objects > 15_000:
                continue
            # Fast flat pre-check before reading recursive subgroups
            try:
                flat_objs = game.read_group(cand_gid, read_subgroups=False)
            except Exception:
                continue
            has_candidate_smr = any(
                o["_type"] == "SkinnedModelResource" and _smr_part_count(o) >= 1
                for o in flat_objs
            )
            if not has_candidate_smr:
                continue
            try:
                objs = game.read_group(cand_gid, read_subgroups=True)
            except Exception:
                continue
            for o in objs:
                if o["_type"] != "SkinnedModelResource":
                    continue
                oskel = game.resolve_ref(o.get("Skeleton"))
                if not _skeleton_matches(oskel, base_skel):
                    continue
                mc = _smr_mesh_part_count(game, o)
                if mc < 1:
                    continue
                prev_mc = extra_smr_counts.get(cand_gid, -1)
                prev_o = extra_smrs.get(cand_gid)
                if prev_o is None or (mc, _smr_part_count(o)) > (prev_mc, _smr_part_count(prev_o)):
                    extra_smrs[cand_gid] = o
                    extra_smr_counts[cand_gid] = mc
        if extra_smrs:
            best_gid = max(extra_smrs, key=lambda g: (extra_smr_counts.get(g, 0), _smr_part_count(extra_smrs[g])))
            smr = extra_smrs[best_gid]

    def _find_matching_smr_for_group(target_gid: int) -> dict[str, Any] | None:
        """Find the best matching non-empty SMR belonging to `target_gid` or its subgroups (up to 2 hops)."""
        if target_gid in extra_smrs:
            return extra_smrs[target_gid]
        seen_sub: set[int] = {target_gid}
        frontier = list(game.graph.group_subgroups(target_gid))
        found_extra: dict[str, Any] | None = None
        cand_smr_gids: list[int] = []
        for _hop in range(2):
            next_frontier: list[int] = []
            for sg in frontier:
                if sg in (1, 2, 3) or sg in seen_sub:
                    continue
                seen_sub.add(sg)
                if sg in extra_smrs:
                    if found_extra is None or (extra_smr_counts.get(sg, 0), _smr_part_count(extra_smrs[sg])) > (
                        extra_smr_counts.get(found_extra["_id"].group_id, 0),
                        _smr_part_count(found_extra),
                    ):
                        found_extra = extra_smrs[sg]
                elif "SkinnedModelResource" in game.graph.group_types(sg) and game.graph.group(sg).num_objects < 6_000:
                    cand_smr_gids.append(sg)
                next_frontier.extend(game.graph.group_subgroups(sg))
            frontier = next_frontier
        direct_smr_gid = direct_smr["_id"].group_id if direct_smr is not None else -1
        if found_extra is not None and found_extra["_id"].group_id != direct_smr_gid:
            return found_extra

        # Also inspect 1-2 hop subgroup SMRs directly:
        # - When CE has no direct SMR (Grazer/Fanghorn/Charger/Lancehorn/Plowhorn/Grimhorn/Clawstrider), accept any matching SMR (mc >= 1).
        # - When CE HAS a direct SMR (e.g. Redeye Watcher CE 4760 -> fallback Scout 47852:0 mc=1), still check small variant-specific
        #   subgroup SMRs (num_objects < 2000, e.g. LaserScout 4267:3 mc=2 with BOTH legs + torso + eye!) and prefer if mc > direct_mesh_count!
        if direct_mesh_count > 0:
            cand_smr_gids = [
                sg for sg in cand_smr_gids
                if sg != direct_smr_gid and game.graph.group(sg).num_objects < 2_000
            ]
            if not cand_smr_gids:
                return found_extra or smr
        best_sub_smr: dict[str, Any] | None = None
        best_sub_score = (direct_mesh_count, 0) if direct_mesh_count > 0 else (0, 0)
        for sg in cand_smr_gids:
            try:
                flat_objs = game.read_group(sg, read_subgroups=False)
            except Exception:
                continue
            if not any(o["_type"] == "SkinnedModelResource" and _smr_part_count(o) >= 2 for o in flat_objs):
                continue
            try:
                objs = game.read_group(sg, read_subgroups=True)
            except Exception:
                continue
            for o in objs:
                if o["_type"] != "SkinnedModelResource":
                    continue
                pc = _smr_part_count(o)
                if pc < 2:
                    continue  # Ignore 1-part shared canister SMRs (e.g. 1479:*)
                oskel = game.resolve_ref(o.get("Skeleton"))
                jcnt = len(oskel.get("Joints", ())) if oskel else 0
                if jcnt < 20:
                    continue
                if base_skel is None or _skeleton_matches(oskel, base_skel):
                    mc = _smr_mesh_part_count(game, o)
                    if mc >= 1 and (mc, pc) > best_sub_score:
                        best_sub_score = (mc, pc)
                        best_sub_smr = o
        if best_sub_smr is not None:
            extra_smrs[best_sub_smr["_id"].group_id] = best_sub_smr
            extra_smr_counts[best_sub_smr["_id"].group_id] = best_sub_score[0]
            return best_sub_smr
        return smr

    # Build base skeleton joint/helper signature sets to reject cross-species DestructibilityResources
    # (e.g. Bilegut 85-part DR inside Stingspawn egg-spawner supergroups, Dreadwing 95-part DR inside Stormbird duo supergroups!)
    base_joint_uuids: set[str] = set()
    base_joint_names: set[str] = set()
    base_helper_names: set[str] = set()
    if base_skel is not None:
        for j in base_skel.get("Joints", ()):
            jid = j.get("JointID")
            if isinstance(jid, UUIDRef) and jid.uuid:
                base_joint_uuids.add(jid.uuid)
            jnm = str(j.get("Name", "") or "")
            if jnm:
                base_joint_names.add(jnm)
    for cand_smr_h in (smr, direct_smr):
        if cand_smr_h is None:
            continue
        for href in cand_smr_h.get("Helpers", ()):
            ho = game.resolve_ref(href)
            if isinstance(ho, dict):
                for h in ho.get("OrientationHelpers", ()):
                    hnm = str(h.get("Name", "") or "")
                    if hnm:
                        base_helper_names.add(hnm)

    base_species_names = _infer_base_species_names(game, entity_oid, smr or direct_smr)

    # Candidate groups that may hold this machine's DestructibilityResource(s):
    # 1. Direct components on CE
    # 2. Objects in CE's own group
    # 3. 1-hop (and for multi-subspecies machines, 2-hop) supergroups of CE and matching SMRs!
    seen_dr_ids: set[str] = set()
    variants: list[MachineVariantOption] = []

    def consider_dr(
        dr: dict[str, Any],
        hint_gid: int,
        matched_smr: dict[str, Any] | None = None,
        local_objs: list[dict[str, Any]] | None = None,
    ) -> bool:
        dr_id = str(dr["_id"])
        if dr_id in seen_dr_ids:
            return False
        cp = dr.get("ConvertedParts", ())
        if len(cp) < 2:
            return False
        # Reject foreign-species DRs whose parts don't attach to this machine's skeleton!
        if not _dr_matches_skeleton(game, dr, base_joint_uuids, base_joint_names, base_helper_names, local_objs=local_objs):
            return False
        seen_dr_ids.add(dr_id)
        use_smr = matched_smr or smr or direct_smr
        variants.append(
            MachineVariantOption(
                variant_index=len(variants),
                label=f"Variant_{len(variants) + 1}_Group{hint_gid}",
                destructibility_id=dr_id,
                group_id=hint_gid,
                part_count=len(cp),
                smr_id=str(use_smr["_id"]) if use_smr is not None else None,
            )
        )
        return True

    # 1. Direct CE component
    for c in comps:
        if c and c["_type"] == "DestructibilityResource":
            consider_dr(c, entity_oid.group_id)

    # 2. Direct CE group
    ce_objs = game.read_group(entity_oid.group_id, read_subgroups=True)
    for o in ce_objs:
        if o["_type"] == "DestructibilityResource":
            consider_dr(o, entity_oid.group_id)

    # 3. Supergroups of CE and matching SMR(s)
    super_gids: list[int] = []
    seen_super: set[int] = set()
    seeds = [entity_oid.group_id, *extra_smrs.keys()]
    for seed in seeds:
        for sg in game.graph.supergroups.get(seed, ()):
            if sg not in seen_super:
                seen_super.add(sg)
                super_gids.append(sg)

    # Check 2-hop supergroups for multi-subspecies machines (Clawstrider/Grazer/Charger) AND for any 2-hop
    # official CollectableRobot variant (e.g. Canister Burrowers: Fire/Frost/Acid/Purgewater/Plasma/Shock!)
    for seed_gid in list(seeds):
        for sup1 in game.graph.supergroups.get(seed_gid, ()):
            if game.graph.group(sup1).num_objects > 15_000:
                continue
            for sup2 in game.graph.supergroups.get(sup1, ()):
                if sup2 in seen_super:
                    continue
                if direct_mesh_count == 0 or "CollectableRobotComponentResource" in game.graph.group_types(sup2):
                    seen_super.add(sup2)
                    super_gids.append(sup2)

    # Keep track of ALL candidate supergroups (including shared-DR Apex/variant groups that carry
    # `CollectableRobotComponentResource` + `RenderEffectShaderOverridesResource` without duplicating the DR!)
    all_candidate_super_gids = [
        sg for sg in super_gids
        if game.graph.group(sg).num_objects <= 18_000
    ]

    # Prioritize official CollectableRobot variant groups first, then smaller clean groups
    super_gids = [
        sg for sg in all_candidate_super_gids
        if "DestructibilityResource" in game.graph.group_types(sg)
    ]
    super_gids.sort(
        key=lambda sg: (
            0 if "CollectableRobotComponentResource" in game.graph.group_types(sg) else 1,
            game.graph.group(sg).num_objects,
        )
    )

    for sgid in super_gids[:45]:
        matched_smr = _find_matching_smr_for_group(sgid)
        if len(extra_smrs) > 1 and matched_smr is None and "CollectableRobotComponentResource" not in game.graph.group_types(sgid):
            continue
        try:
            # Flat read is 100x faster than read_subgroups=True and still reads exact DestructibilityResource ConvertedParts count!
            objs = game.read_group(sgid, read_subgroups=False)
        except Exception:
            continue
        drs = [o for o in objs if o["_type"] == "DestructibilityResource" and len(o.get("ConvertedParts", ())) >= 5]
        if drs:
            # Sort largest first, and pick the first DR whose parts actually attach to this machine's skeleton!
            # (Prevents a larger foreign machine in a duo-encounter supergroup from masking this machine's DR!)
            for cand_dr in sorted(drs, key=lambda d: len(d.get("ConvertedParts", ())), reverse=True):
                if consider_dr(cand_dr, sgid, matched_smr=matched_smr, local_objs=objs):
                    break

    # 3b. Shared-DR Apex / named variant groups (e.g. Apex Slitherfang `61843`, Apex Tideripper `68979`) that inherit
    # their base machine's DestructibilityResource from a subgroup while supplying their own CollectableRobot +
    # `RenderEffectShaderOverridesResource` (`*_BVar_*` textures)!
    if variants:
        base_dr_var = max(variants, key=lambda vv: vv.part_count)
        seen_var_gids = {vv.group_id for vv in variants}
        cr_ov_gids = [
            sg for sg in all_candidate_super_gids
            if sg not in seen_var_gids
            and "CollectableRobotComponentResource" in game.graph.group_types(sg)
            and _has_variant_shader_overrides_fast(game, sg)
        ]
        cr_ov_gids.sort(key=lambda sg: game.graph.group(sg).num_objects)
        for sgid in cr_ov_gids[:16]:
            matched_smr = _find_matching_smr_for_group(sgid)
            use_smr = matched_smr or smr or direct_smr
            variants.append(
                MachineVariantOption(
                    variant_index=len(variants),
                    label=f"Variant_{len(variants) + 1}_Group{sgid}",
                    destructibility_id=base_dr_var.destructibility_id,
                    group_id=sgid,
                    part_count=base_dr_var.part_count,
                    smr_id=str(use_smr["_id"]) if use_smr is not None else base_dr_var.smr_id,
                )
            )

    # If smr was still None (e.g. Grazer/Fanghorn where SMR lived inside variant subgroup), pick the best variant SMR!
    if smr is None and extra_smrs:
        best_gid = max(extra_smrs, key=lambda g: (extra_smr_counts.get(g, 0), _smr_part_count(extra_smrs[g])))
        smr = extra_smrs[best_gid]
    if smr is None:
        smr = direct_smr

    # Backfill any variant that had an empty/None smr_id
    for v in variants:
        if not v.smr_id and smr is not None:
            v.smr_id = str(smr["_id"])

    # Detect official HFW variant names from CollectableRobotComponentResource + TextureSet subspecies/Apex tags
    def _inspect_variant_tags(v: MachineVariantOption) -> tuple[str, str, str, bool]:
        cr_name, cr_desc, _ = _extract_collectable_robot_info(game, v.group_id)
        v.official_name = cr_name
        v.official_description = cr_desc
        sub_type = ""
        is_apex = "apex" in cr_name.lower()

        def _scan_path_tokens(p: str, *, allow_apex: bool) -> None:
            nonlocal sub_type, is_apex
            if "canister" in p or "robotguts" in p or "shared_textures" in p:
                return
            if allow_apex:
                # Only trigger Apex on real BVar color/normal maps or explicit apex/hunterkiller paths
                # (never on shared base roughness filenames like `raidershared_body-noplate-muscle_bvar_rgh.png`!).
                if (
                    "apex" in p
                    or "hunterkiller" in p
                    or "_bvar_clr" in p
                    or "_bvar_nmt" in p
                    or "_bvar_msk" in p
                    or "bvar_clr" in p
                ):
                    is_apex = True
            if not sub_type:
                if "raidermelee" in p:
                    sub_type = "Melee"
                elif "raiderelemental" in p:
                    sub_type = "Elemental"
                elif "/reindeer/" in p or "reindeer_" in p:
                    sub_type = "Fanghorn"
                elif "/antelope/" in p or "/deer/" in p or "antelope_" in p:
                    sub_type = "Grazer"
                elif "/goat/" in p or "goat_" in p or "ibex" in p:
                    sub_type = "Lancehorn"
                elif "/horse/" in p or "horse_" in p or "bighorn" in p:
                    sub_type = "Charger"
                elif "wardredger" in p or "dredgerwar" in p or "grimhorn" in p:
                    sub_type = "Grimhorn"
                elif "/dredger/" in p or "dredger_" in p:
                    sub_type = "Plowhorn"
                elif "_fire" in p or "fire_" in p:
                    sub_type = "Fire"
                elif "_freeze" in p or "_frost" in p or "_cryo" in p:
                    sub_type = "Frost"
                elif "_acid" in p or "acid_" in p:
                    sub_type = "Acid"
                elif "_plasma" in p:
                    sub_type = "Plasma"
                elif "_purge" in p or "_water" in p:
                    sub_type = "Purgewater"
                elif "_shock" in p:
                    sub_type = "Shock"

        # 1. Check SMR group ONLY for subspecies name (never Apex, since base SMRs are shared by Standard & Apex!)
        if v.smr_id:
            try:
                for o in game.read_group(ObjectId.parse(v.smr_id).group_id, read_subgroups=False):
                    if o["_type"] == "TextureSet":
                        for desc in o.get("TextureDesc", ()):
                            _scan_path_tokens(desc.get("Path", "").lower().replace("\\", "/"), allow_apex=False)
            except Exception:
                pass

        # 2. Check v.group_id itself and small variant-specific TextureSet subgroups for Apex & elemental overrides
        check_gids = [v.group_id]
        for sg in game.graph.group_subgroups(v.group_id):
            if sg not in (1, 2, 3) and "TextureSet" in game.graph.group_types(sg):
                if game.graph.group(sg).num_objects < 400:
                    check_gids.append(sg)

        for igid in check_gids:
            try:
                objs = game.read_group(igid, read_subgroups=False)
            except Exception:
                continue
            for o in objs:
                if o["_type"] == "TextureSet":
                    for desc in o.get("TextureDesc", ()):
                        _scan_path_tokens(desc.get("Path", "").lower().replace("\\", "/"), allow_apex=not bool(cr_name))
        # 3. Authoritative Apex signal: zero-I/O graph check for RenderEffectShaderOverridesResource in variant group
        if not is_apex and not cr_name:
            if _has_variant_shader_overrides_fast(game, v.group_id):
                is_apex = True
        return cr_name, cr_desc, sub_type, is_apex

    tagged_raw: list[tuple[MachineVariantOption, str, str, str, bool]] = [(v, *_inspect_variant_tags(v)) for v in variants]

    # Species consistency filter: reject cross-species duo-encounter variants (e.g. Bilegut on Stingspawn,
    # Dreadwing on Stormbird) whose CollectableRobot official name explicitly names a DIFFERENT species!
    def _species_is_compatible(cr_name: str) -> bool:
        if not cr_name or not base_species_names:
            return True
        low = cr_name.lower()
        if any(sp in low for sp in base_species_names):
            return True
        # Check if cr_name explicitly names a known OTHER machine species
        for other_sp in _ALL_KNOWN_SPECIES_NAMES:
            if other_sp in low and other_sp not in base_species_names:
                return False
        return True

    tagged = [t for t in tagged_raw if _species_is_compatible(t[1])]
    if not tagged:
        tagged = tagged_raw
    # Prefer Standard (non-Apex) full-part variants first so default export is the iconic complete machine!
    max_parts = max((t[0].part_count for t in tagged), default=0)
    # Filter out tiny incomplete stub variants (keep any official CollectableRobot variant!)
    if max_parts >= 20:
        filtered = [t for t in tagged if t[1] or t[0].part_count >= max_parts * 0.45]
        if filtered:
            tagged = filtered

    # If official CollectableRobot variants exist for this machine, drop anonymous quest duplicates —
    # EXCEPT preserve a full-part Standard (non-Apex) variant if ONLY Apex has a CollectableRobot (e.g. Ravager, Spikesnout, Rollerback)!
    has_official = any(bool(t[1]) for t in tagged)
    has_official_standard = any(
        bool(t[1]) and not t[4] and t[0].part_count >= max_parts * 0.5
        for t in tagged
    )

    def _is_elemental_variant_name(nm: str) -> bool:
        low = nm.lower()
        return any(w in low for w in ("acid ", "fire ", "frost ", "freeze ", "purgewater ", "plasma ", "shock ", "canister ", "tracker "))

    tagged.sort(
        key=lambda item: (
            0 if (not item[4] and item[0].part_count >= max_parts * 0.5) else 1,  # Non-Apex full variants first
            0 if item[1] else 1,                                                  # Official CollectableRobot variants before anonymous groups!
            0 if (item[1] and not _is_elemental_variant_name(item[1])) else 1,    # Base species (e.g. "Clawstrider", "Burrower") before elemental variants!
            -item[0].part_count,
            0 if item[3] == "Melee" else 1,
            item[1].lower() if item[1] else "zzz",
            item[0].group_id,
        )
    )

    # Deduplicate: keep 1 best entry per unique official CollectableRobot name (preserving Fire/Frost/Acid/Apex Bellowback,
    # Melee/Acid/Fire Clawstrider, Fire/Acid Bristleback, all Canister Burrowers, etc.), and drop anonymous clones when
    # official named variants already cover the machine!
    deduped: list[tuple[MachineVariantOption, str, str, str, bool]] = []
    seen_official_names: set[str] = set()
    bucket_counts: dict[tuple[str, bool, str | None, int], int] = {}
    for item in tagged:
        v, cr_name, _cr_desc, sub_type, is_apex = item
        if cr_name:
            k_name = cr_name.lower()
            if k_name in seen_official_names:
                continue
            seen_official_names.add(k_name)
            deduped.append(item)
            continue
        # Anonymous variant (no CollectableRobot): skip if official variants already cover Standard (or if this is Apex/partial)
        if has_official and (is_apex or has_official_standard or v.part_count < max_parts * 0.5):
            continue
        bkey = (sub_type, is_apex, v.smr_id, v.part_count)
        cnt = bucket_counts.get(bkey, 0)
        limit = 2 if sub_type == "Elemental" else 1
        if cnt >= limit:
            continue
        bucket_counts[bkey] = cnt + 1
        deduped.append(item)

    # Build clean human-readable labels, preferring official HFW names!
    prefix_totals: dict[str, int] = {}
    raw_prefixes: list[str] = []
    for i, (v, cr_name, _cr_desc, sub_type, is_apex) in enumerate(deduped):
        if cr_name:
            prefix = cr_name
        elif sub_type and is_apex:
            prefix = f"{sub_type} Apex"
        elif is_apex:
            prefix = "Apex"
        elif sub_type:
            prefix = f"{sub_type} Standard"
        elif i == 0:
            prefix = "Standard"
        else:
            prefix = f"Variant {i + 1}"
        raw_prefixes.append(prefix)
        prefix_totals[prefix] = prefix_totals.get(prefix, 0) + 1

    prefix_seen: dict[str, int] = {}
    variants = []
    for i, ((v, _cr_name, _cr_desc, _sub_type, _is_apex), prefix) in enumerate(zip(deduped, raw_prefixes)):
        v.variant_index = i
        if prefix_totals[prefix] > 1:
            idx_in = prefix_seen.get(prefix, 0)
            prefix_seen[prefix] = idx_in + 1
            suffix = chr(ord("A") + idx_in)
            disp_prefix = f"{prefix} {suffix}"
        else:
            disp_prefix = prefix
        v.label = f"{disp_prefix} (Group {v.group_id}, {v.part_count} parts)"
        variants.append(v)

    return ce, smr, variants


def resolve_machine_entity_id(
    game: HFWGame,
    catalog: AssetCatalog,
    machine_query: str,
) -> tuple[ObjectId, str, str]:
    """Resolve a machine name, codename, or `'group:idx'` string to `(ObjectId, codename, display_name)`."""
    if ":" in machine_query:
        oid = ObjectId.parse(machine_query)
        row = catalog.conn.execute(
            "SELECT codename, display_name FROM entities WHERE group_id = ? AND object_index = ?",
            (oid.group_id, oid.object_index),
        ).fetchone()
        if row:
            return oid, row["codename"], row["display_name"]
        return oid, f"group_{oid.group_id}", f"Entity {machine_query}"

    matches = catalog.list_machines(machine_query)
    if not matches:
        matches = catalog.search_entities(machine_query, limit=10)
    if not matches:
        raise LookupError(f"No machine or entity matching {machine_query!r} found in catalog.")

    # Prefer exact display_name or codename match with highest part_count & skeleton
    qlow = machine_query.strip().lower()
    exact = [m for m in matches if m["display_name"].lower() == qlow or m["codename"].lower() == qlow]
    pool = exact or matches
    best = max(pool, key=lambda m: (m["has_skeleton"], m["has_destructibility"], m["part_count"]))
    return ObjectId(best["group_id"], best["object_index"]), best["codename"], best["display_name"]


def assemble_machine(
    game: HFWGame,
    entity_oid: ObjectId,
    codename: str = "",
    display_name: str = "",
    variant_index: int = 0,
    lod_index: int = 0,
    include_broken_states: bool = True,
) -> MachineAssembly:
    """Assemble a complete HFW machine with unified skeleton, skinned body, bone-parented armor plates, and materials."""
    ce, smr, variants = discover_machine_variants(game, entity_oid)

    selected_variant: MachineVariantOption | None = None
    selected_dr: dict[str, Any] | None = None
    if variants:
        vidx = max(0, min(variant_index, len(variants) - 1))
        selected_variant = variants[vidx]
        if selected_variant.smr_id:
            smr = game.read_object(ObjectId.parse(selected_variant.smr_id), read_subgroups=True)

    if smr is None:
        raise ValueError(f"Entity {entity_oid} has no SkinnedModelResource")

    skel_obj = game.resolve_ref(smr.get("Skeleton"))
    if skel_obj is None:
        raise ValueError(f"SkinnedModelResource {smr['_id']} has no Skeleton")

    # Gather ALL SkeletonHelpers from SMR + Skeleton group + CE group + selected variant group & its helper subgroups
    helper_objs: list[dict[str, Any]] = []
    for href in smr.get("Helpers", ()):
        h = game.resolve_ref(href)
        if h is not None:
            helper_objs.append(h)

    skeleton = extract_skeleton(game, skel_obj, helper_objs)

    helper_gids: set[int] = {entity_oid.group_id, smr["_id"].group_id, skel_obj["_id"].group_id}
    if selected_variant is not None:
        selected_dr = game.read_object(ObjectId.parse(selected_variant.destructibility_id), read_subgroups=True)
        helper_gids.add(selected_variant.group_id)
        for sg in game.graph.group_subgroups(selected_variant.group_id):
            if sg not in (1, 2, 3) and "SkeletonHelpers" in game.graph.group_types(sg):
                helper_gids.add(sg)

    for hgid in helper_gids:
        if "SkeletonHelpers" not in game.graph.group_types(hgid):
            continue
        try:
            grp_objs = game.read_group(hgid, read_subgroups=True)
            extra_helpers = [o for o in grp_objs if o["_type"] == "SkeletonHelpers"]
            if extra_helpers:
                add_skeleton_helpers(game, skeleton, extra_helpers)
        except Exception:
            pass

    # 4-Tier Bind-Pose Resolution (preserving `skel_obj`'s exact 0..N-1 joint indices for skinning!):
    # Tier 1 : smr.Skeleton itself has HasBindPose=True (e.g. Slitherfang, Clawstrider, Tremortusk)
    # Tier 2a: smr.InitialRepresentationSkeleton with HasBindPose=True (e.g. Thunderjaw 755:4, Bellowback, Snapmaw, Rockbreaker!)
    # Tier 2b: Superset/matching Skeleton with HasBindPose=True in CE/SMR/Skel/Variant groups (e.g. Slaughterspine 1127:0, Lancehorn 2202:6!)
    # Tier 2c: PhysicsSkeleton.ReferencePose in CE/SMR/Skel/Variant groups (e.g. Behemoth, Fireclaw, Frostclaw, Scorcher, Shell-Walker, Skydrifter, Stalker, Stormbird!)
    # Tier 3 : InverseBindMatrices from SkinnedMeshIndexedJointBindings (last-resort fallback)
    if not skel_obj.get("HasBindPose"):
        target_jcount = max(1, len(skeleton.joints))
        pose_covered = 0

        # Tier 2a: InitialRepresentationSkeleton on SMR
        irs_obj = game.resolve_ref(smr.get("InitialRepresentationSkeleton"))
        if irs_obj is not None and irs_obj.get("HasBindPose"):
            irs_sd = extract_skeleton(game, irs_obj)
            pose_covered = skeleton.apply_pose_from_skeleton(irs_sd)

        # Tier 2b: Any Skeleton with HasBindPose=True in GraphAnimationManagerResource or SMR/Skel/CE/Variant groups
        if pose_covered < int(target_jcount * 0.90):
            cand_skels: list[dict[str, Any]] = []
            for comp_ref in ce.get("EntityComponentResources", ()):
                comp = game.resolve_ref(comp_ref)
                if comp and comp["_type"] == "GraphAnimationManagerResource":
                    cand = game.resolve_ref(comp.get("Skeleton"))
                    if cand and cand.get("HasBindPose"):
                        cand_skels.append(cand)
            search_pose_gids = {smr["_id"].group_id, skel_obj["_id"].group_id, entity_oid.group_id}
            if selected_variant is not None:
                search_pose_gids.add(selected_variant.group_id)
            for pgid in list(search_pose_gids):
                for sg in game.graph.group_subgroups(pgid):
                    if sg not in (1, 2, 3) and ("Skeleton" in game.graph.group_types(sg) or "PhysicsSkeleton" in game.graph.group_types(sg)):
                        if game.graph.group(sg).num_objects < 8_000:
                            search_pose_gids.add(sg)
            for pgid in search_pose_gids:
                if "Skeleton" not in game.graph.group_types(pgid):
                    continue
                try:
                    for o in game.read_group(pgid, read_subgroups=False):
                        if o["_type"] == "Skeleton" and o.get("HasBindPose") and len(o.get("Joints", ())) >= int(target_jcount * 0.80):
                            cand_skels.append(o)
                except Exception:
                    continue

            best_sd: SkeletonData | None = None
            best_hits = pose_covered
            for cs in cand_skels:
                csd = extract_skeleton(game, cs)
                hits = sum(
                    1
                    for ji in skeleton.joints
                    if (ji.uuid and ji.uuid in csd.joint_by_uuid) or (ji.name and ji.name in csd.joint_by_name)
                )
                if hits > best_hits:
                    best_hits = hits
                    best_sd = csd
            if best_sd is not None and best_hits >= int(target_jcount * 0.80):
                pose_covered = skeleton.apply_pose_from_skeleton(best_sd)

        # Tier 2c: PhysicsSkeleton.ReferencePose (verified < 0.5 mm error across all 163 Thunderjaw bones!)
        if pose_covered < int(target_jcount * 0.90):
            ps_gids = {entity_oid.group_id, smr["_id"].group_id, skel_obj["_id"].group_id}
            if selected_variant is not None:
                ps_gids.add(selected_variant.group_id)
            for pgid in list(ps_gids):
                for sg in game.graph.group_subgroups(pgid):
                    if sg not in (1, 2, 3) and "PhysicsSkeleton" in game.graph.group_types(sg):
                        if game.graph.group(sg).num_objects < 12_000:
                            ps_gids.add(sg)
            best_ps_map: dict[str, np.ndarray] | None = None
            best_ps_hits = pose_covered
            for pgid in ps_gids:
                if "PhysicsSkeleton" not in game.graph.group_types(pgid):
                    continue
                try:
                    for o in game.read_group(pgid, read_subgroups=False):
                        if o["_type"] != "PhysicsSkeleton":
                            continue
                        wmap = extract_physics_skeleton_world_matrices(o)
                        hits = sum(1 for ji in skeleton.joints if ji.uuid and ji.uuid in wmap)
                        if hits > best_ps_hits:
                            best_ps_hits = hits
                            best_ps_map = wmap
                except Exception:
                    continue
            if best_ps_map is not None and best_ps_hits >= int(target_jcount * 0.80):
                pose_covered = skeleton.apply_world_matrices_by_uuid(best_ps_map)

        # Tier 3 fallback: InverseBindMatrices (only if Tiers 2a-2c couldn't cover >= 80% of joints)
        if pose_covered < int(target_jcount * 0.80):
            ibm_world: dict[int, np.ndarray] = {}
            search_gids = {smr["_id"].group_id, entity_oid.group_id}
            if selected_variant is not None:
                search_gids.add(selected_variant.group_id)
            for gid in list(search_gids):
                search_gids.update(game.graph.group_subgroups(gid))
            for gid in search_gids:
                if "SkinnedMeshIndexedJointBindings" not in game.graph.group_types(gid):
                    continue
                try:
                    objs = game.read_group(gid, read_subgroups=False)
                except Exception:
                    continue
                for o in objs:
                    if o["_type"] == "SkinnedMeshIndexedJointBindings":
                        jlist = o.get("JointIndexList", ())
                        mats = o.get("InverseBindMatrices", ())
                        for jidx, m in zip(jlist, mats):
                            if jidx not in ibm_world:
                                try:
                                    ibm_world[int(jidx)] = np.linalg.inv(decima_mat44_to_np(m))
                                except np.linalg.LinAlgError:
                                    pass
            if ibm_world:
                skeleton.apply_inverse_bind_matrices(ibm_world)

    parts: list[AssembledPart] = []
    seen_mesh_attach_keys: set[tuple[str, str | None, int | None, int]] = set()
    used_material_ids: set[str] = set()
    material_objs: dict[str, dict[str, Any]] = {}

    def _register_materials(dm: DecodedMesh) -> None:
        for p in dm.primitives:
            if p.material_id and p.material_id not in used_material_ids:
                used_material_ids.add(p.material_id)
                try:
                    material_objs[p.material_id] = game.read_object(ObjectId.parse(p.material_id), read_subgroups=True)
                except Exception:
                    pass

    def _align_skinned_bone_indices(dm: DecodedMesh) -> None:
        """Align a skinned mesh's bone indices to `skeleton` if the mesh references a different Skeleton object."""
        if not dm.is_skinned or not dm.skeleton_joint_names:
            return
        if dm.skeleton_id == skeleton.object_id and len(dm.skeleton_joint_names) == len(skeleton.joints):
            return
        # Check if 1:1 identity already holds by UUID/name
        same = len(dm.skeleton_joint_names) == len(skeleton.joints) and all(
            (u and u == j.uuid) or (n == j.name)
            for u, n, j in zip(dm.skeleton_joint_uuids, dm.skeleton_joint_names, skeleton.joints)
        )
        if same:
            return
        # Build lookup from mesh_skel index -> unified skeleton index (UUID first, then Name)
        lut = np.zeros(len(dm.skeleton_joint_names), dtype=np.uint16)
        for src_i, (su, sn) in enumerate(zip(dm.skeleton_joint_uuids, dm.skeleton_joint_names)):
            tgt = skeleton.joint_by_uuid.get(su) if su else None
            if tgt is None and sn:
                tgt = skeleton.joint_by_name.get(sn)
            if tgt is not None:
                lut[src_i] = tgt.index
            elif src_i < len(skeleton.joints):
                lut[src_i] = src_i
        for pr in dm.primitives:
            if pr.bone_indices is not None and len(lut) > 0:
                safe = np.clip(pr.bone_indices, 0, len(lut) - 1)
                pr.bone_indices = lut[safe]

    def _add_model_part(
        mpr: dict[str, Any],
        source_kind: str,
        override_joint_uuid: str | None = None,
        override_bone_name: str = "",
        enabled_default: bool = True,
        state_index: int = 0,
        part_prefix: str = "",
        catalog_part_name: str = "",
    ) -> None:
        mesh_ref = mpr.get("MeshResource")
        if mesh_ref is None:
            return
        mesh_root = game.resolve_ref(mesh_ref)
        if mesh_root is None:
            return

        # Determine attachment UUID / helper / bone
        mpr_jref = mpr.get("AttachToJointID")
        mpr_juuid = mpr_jref.uuid if isinstance(mpr_jref, UUIDRef) else None
        target_uuid = override_joint_uuid or mpr_juuid
        helper_node_str = mpr.get("HelperNode", "")

        att_name, parent_jidx, helper_name, local_off, world_mat = skeleton.resolve_attachment(
            joint_uuid=target_uuid,
            helper_name=helper_node_str,
            bone_name=override_bone_name,
        )
        parent_jname = skeleton.joints[parent_jidx].name if (parent_jidx is not None and 0 <= parent_jidx < len(skeleton.joints)) else None

        decoded_meshes = decode_mesh_hierarchy(game, mesh_root, lod_index=lod_index)
        for dm_i, dm in enumerate(decoded_meshes):
            # Deduplicate identical (mesh_id, helper/joint, state) across SMR + DestructibilityResource
            dedup_key = (dm.object_id, helper_name or parent_jname, parent_jidx, state_index)
            if dedup_key in seen_mesh_attach_keys:
                continue
            seen_mesh_attach_keys.add(dedup_key)

            _align_skinned_bone_indices(dm)
            _register_materials(dm)

            # Clean human-readable part name
            clean_att = att_name.removesuffix("_Lx_helper").removesuffix("_Cx_helper").removesuffix("_helper")
            if clean_att == "Unattached":
                clean_att = f"BodyMesh_{dm.object_id.replace(':', '_')}"
            pname = f"{part_prefix}{clean_att}" if part_prefix else clean_att
            if state_index > 0:
                pname = f"{pname}_State{state_index}"
            if len(decoded_meshes) > 1:
                pname = f"{pname}_{dm_i}"

            # CRITICAL VERIFIED RULE:
            # - RegularSkinnedMeshResource vertices are already in model bind-pose world space (identity transform).
            # - StaticMeshResource armor plates are centered at (0,0,0) in local helper/joint space!
            if dm.is_skinned:
                part_local = np.eye(4, dtype=np.float64)
                part_world = np.eye(4, dtype=np.float64)
                eff_parent_idx = None
                eff_parent_name = None
                eff_helper_name = None
            else:
                extra_local = dm.local_transform if dm.local_transform is not None else np.eye(4, dtype=np.float64)
                part_local = local_off @ extra_local
                part_world = world_mat @ extra_local
                eff_parent_idx = parent_jidx
                eff_parent_name = parent_jname
                eff_helper_name = helper_name

            coll = _classify_part_collection(pname, dm.is_skinned, is_alt_state=(state_index > 0))
            parts.append(
                AssembledPart(
                    part_id=f"{mpr['_id']}_{dm_i}_s{state_index}",
                    name=pname,
                    collection=coll,
                    source_kind=source_kind,
                    is_skinned=dm.is_skinned,
                    parent_joint_index=eff_parent_idx,
                    parent_joint_name=eff_parent_name,
                    helper_name=eff_helper_name,
                    local_offset_matrix=part_local,
                    world_matrix=part_world,
                    mesh=dm,
                    enabled_by_default=enabled_default and (state_index == 0),
                    state_index=state_index,
                    catalog_part_name=catalog_part_name,
                )
            )

    # Resolve official Focus/Catalog part names (DestructibilityPartTag UUID -> "Metalbite Sac" / "Blaze Sac" / "Adhesive Sac" / etc.)
    cr_name = ""
    cr_desc = ""
    tag_to_catalog_part: dict[str, str] = {}
    if selected_variant is not None:
        cr_name, cr_desc, tag_to_catalog_part = _extract_collectable_robot_info(game, selected_variant.group_id)
    if not tag_to_catalog_part:
        ce_cr_name, ce_cr_desc, ce_tags = _extract_collectable_robot_info(game, entity_oid.group_id)
        if ce_tags:
            tag_to_catalog_part = ce_tags
        if not cr_name:
            cr_name = ce_cr_name
        if not cr_desc:
            cr_desc = ce_cr_desc

    # Step 1: Assemble all parts from the selected DestructibilityResource (gives full intact plates + broken states!)
    if selected_dr is not None:
        for pref in selected_dr.get("ConvertedParts", ()):
            dp = game.resolve_ref(pref)
            if not dp:
                continue
            dp_jref = dp.get("AttachToJointID")
            dp_juuid = dp_jref.uuid if isinstance(dp_jref, UUIDRef) else None
            dp_bone = dp.get("BoneName", "")
            dp_enabled = bool(dp.get("Enabled", True))

            # Match DestructibilityPartTags UUIDs to official HFW Focus Catalog part name (e.g. "Metalbite Sac", "Purgewater Sac")
            dp_cat_names = [
                tag_to_catalog_part[tu]
                for t in dp.get("DestructibilityPartTags", ())
                if (tu := getattr(t, "uuid", None)) and tu in tag_to_catalog_part
            ]
            dp_cat_name = " / ".join(dict.fromkeys(dp_cat_names))

            init_state = game.resolve_ref(dp.get("InitialState"))
            init_state_id = init_state["_id"] if init_state else None
            if init_state:
                mpr = game.resolve_ref(init_state.get("ModelPartResource"))
                if mpr:
                    _add_model_part(
                        mpr,
                        source_kind="DestructibilityIntact",
                        override_joint_uuid=dp_juuid,
                        override_bone_name=dp_bone,
                        enabled_default=dp_enabled,
                        state_index=0,
                        catalog_part_name=dp_cat_name,
                    )

            if include_broken_states:
                alt_idx = 0
                for sref in dp.get("AllAvailableStates", ()):
                    st = game.resolve_ref(sref)
                    if not st or st["_id"] == init_state_id:
                        continue
                    mpr = game.resolve_ref(st.get("ModelPartResource"))
                    if mpr:
                        alt_idx += 1
                        _add_model_part(
                            mpr,
                            source_kind="DestructibilityAltState",
                            override_joint_uuid=dp_juuid,
                            override_bone_name=dp_bone,
                            enabled_default=False,
                            state_index=alt_idx,
                            catalog_part_name=dp_cat_name,
                        )

    # Step 2: Assemble base SkinnedModelResource ModelPartResources (core body + any non-destructible parts!)
    init_states = smr.get("InitiallyEnabledStates") or ()
    for idx, mref in enumerate(smr.get("ModelPartResources", ())):
        mpr = game.resolve_ref(mref)
        if not mpr:
            continue
        enabled = bool(init_states[idx]) if idx < len(init_states) else True
        _add_model_part(
            mpr,
            source_kind="SkinnedModelPart",
            enabled_default=enabled,
            state_index=0,
        )

    # Step 2b: Assemble attached `ChildEntityComponentResource` modules (Tremortusk cannons, Shellsnapper 4x turrets,
    # Shell-Walker cargo crate/shield, Behemoth cargo container, Tallneck head module, Sunwing/Waterwing shields!)
    def _assemble_child_entity(child_comp: dict[str, Any], child_idx: int) -> None:
        child_ent = game.resolve_ref(child_comp.get("EntityResource"))
        if not child_ent:
            return
        mover = game.resolve_ref(child_comp.get("MoverResource"))
        ohl = (mover.get("OrientationHelperLink") or {}) if mover else {}
        ohl_jref = ohl.get("Joint")
        ohl_juuid = ohl_jref.uuid if isinstance(ohl_jref, UUIDRef) else None
        ohl_offset = decima_mat44_to_np(ohl["Offset"]) if "Offset" in ohl else np.eye(4, dtype=np.float64)
        mover_local = decima_mat44_to_np(mover["LocalMatrix"]) if (mover and "LocalMatrix" in mover) else np.eye(4, dtype=np.float64)
        helper_str = (mover.get("Helper") or "") if mover else ""

        att_name, mount_jidx, mount_helper, base_local, base_world = skeleton.resolve_attachment(
            joint_uuid=ohl_juuid,
            helper_name=helper_str,
        )
        if mount_jidx is None and mount_helper is None and not ohl_juuid and not helper_str:
            return  # Skip unanchored logical/audio child entities without spatial attachment
        mount_jname = skeleton.joints[mount_jidx].name if (mount_jidx is not None and 0 <= mount_jidx < len(skeleton.joints)) else None
        mount_local = base_local @ ohl_offset @ mover_local
        mount_world = base_world @ ohl_offset @ mover_local

        child_comps = [game.resolve_ref(cr) for cr in child_ent.get("EntityComponentResources", ())]
        child_smr = next((cc for cc in child_comps if cc and cc["_type"] == "SkinnedModelResource"), None)
        child_static = next((cc for cc in child_comps if cc and cc["_type"] == "StaticModelResource"), None)
        child_dr = next((cc for cc in child_comps if cc and cc["_type"] == "DestructibilityResource"), None)
        if child_smr is None and child_static is None and child_dr is None:
            return

        # Build child's own local skeleton if it has one (e.g. Tremortusk cannons, Shellsnapper turrets)
        child_skel: SkeletonData | None = None
        if child_smr is not None:
            c_skel_obj = game.resolve_ref(child_smr.get("Skeleton"))
            if c_skel_obj is not None:
                c_helpers = [h for href in child_smr.get("Helpers", ()) if (h := game.resolve_ref(href)) is not None]
                child_skel = extract_skeleton(game, c_skel_obj, c_helpers)
                if not c_skel_obj.get("HasBindPose"):
                    c_irs = game.resolve_ref(child_smr.get("InitialRepresentationSkeleton"))
                    if c_irs and c_irs.get("HasBindPose"):
                        child_skel.apply_pose_from_skeleton(extract_skeleton(game, c_irs))

        clean_mount = att_name.removesuffix("_Lx_helper").removesuffix("_Cx_helper").removesuffix("_helper")
        if clean_mount == "Unattached":
            clean_mount = f"Child{child_idx}"
        child_seen: set[tuple[str, int]] = set()

        def _emit_child_mesh(
            mesh_root: dict[str, Any],
            part_label: str,
            child_sub_local: np.ndarray,
            source_kind: str,
            enabled_default: bool,
            state_index: int,
        ) -> None:
            decoded_list = decode_mesh_hierarchy(game, mesh_root, lod_index=lod_index)
            for dm_i, dm in enumerate(decoded_list):
                ckey = (dm.object_id, state_index)
                if ckey in child_seen:
                    continue
                child_seen.add(ckey)
                _register_materials(dm)

                # In child space: skinned meshes already sit in child bind-pose; static meshes sit at child_sub_local
                sub_mat = np.eye(4, dtype=np.float64) if dm.is_skinned else child_sub_local
                extra_local = dm.local_transform if (dm.local_transform is not None and not dm.is_skinned) else np.eye(4, dtype=np.float64)
                final_local = mount_local @ sub_mat @ extra_local
                final_world = mount_world @ sub_mat @ extra_local

                pname = f"Child_{clean_mount}_{part_label}"
                if state_index > 0:
                    pname = f"{pname}_State{state_index}"
                if len(decoded_list) > 1:
                    pname = f"{pname}_{dm_i}"
                coll = "Plates_Broken_States" if state_index > 0 else "Weapons_And_Components"
                parts.append(
                    AssembledPart(
                        part_id=f"child{child_idx}_{dm.object_id}_{dm_i}_s{state_index}",
                        name=pname,
                        collection=coll,
                        source_kind=source_kind,
                        is_skinned=False,  # Rigidly parented to parent machine's mounting joint/helper!
                        parent_joint_index=mount_jidx,
                        parent_joint_name=mount_jname,
                        helper_name=mount_helper,
                        local_offset_matrix=final_local,
                        world_matrix=final_world,
                        mesh=dm,
                        enabled_by_default=enabled_default and (state_index == 0),
                        state_index=state_index,
                    )
                )

        def _emit_child_mpr(
            mpr: dict[str, Any],
            source_kind: str,
            override_juuid: str | None = None,
            override_bname: str = "",
            enabled_default: bool = True,
            state_index: int = 0,
            fallback_lbl: str = "Part",
        ) -> None:
            mroot = game.resolve_ref(mpr.get("MeshResource"))
            if mroot is None:
                return
            sub_mat = np.eye(4, dtype=np.float64)
            plbl = fallback_lbl
            if child_skel is not None:
                mj = mpr.get("AttachToJointID")
                tu = override_juuid or (mj.uuid if isinstance(mj, UUIDRef) else None)
                c_att, _, _, _, c_wmat = child_skel.resolve_attachment(
                    joint_uuid=tu, helper_name=mpr.get("HelperNode", ""), bone_name=override_bname
                )
                if c_att != "Unattached":
                    plbl = c_att.removesuffix("_Lx_helper").removesuffix("_Cx_helper").removesuffix("_helper")
                    sub_mat = c_wmat
            _emit_child_mesh(mroot, plbl, sub_mat, source_kind, enabled_default, state_index)

        if child_dr is not None:
            for pi, pref in enumerate(child_dr.get("ConvertedParts", ())):
                dp = game.resolve_ref(pref)
                if not dp:
                    continue
                dp_j = dp.get("AttachToJointID")
                dp_ju = dp_j.uuid if isinstance(dp_j, UUIDRef) else None
                dp_bn = dp.get("BoneName", "")
                init_st = game.resolve_ref(dp.get("InitialState"))
                init_id = init_st["_id"] if init_st else None
                if init_st:
                    mpr = game.resolve_ref(init_st.get("ModelPartResource"))
                    if mpr:
                        _emit_child_mpr(mpr, "DestructibilityIntact", dp_ju, dp_bn, True, 0, f"Part{pi}")
                if include_broken_states:
                    alt_i = 0
                    for sref in dp.get("AllAvailableStates", ()):
                        st = game.resolve_ref(sref)
                        if not st or st["_id"] == init_id:
                            continue
                        mpr = game.resolve_ref(st.get("ModelPartResource"))
                        if mpr:
                            alt_i += 1
                            _emit_child_mpr(mpr, "DestructibilityAltState", dp_ju, dp_bn, False, alt_i, f"Part{pi}")

        if child_smr is not None:
            c_inits = child_smr.get("InitiallyEnabledStates") or ()
            for mi_idx, mref in enumerate(child_smr.get("ModelPartResources", ())):
                mpr = game.resolve_ref(mref)
                if mpr:
                    en = bool(c_inits[mi_idx]) if mi_idx < len(c_inits) else True
                    _emit_child_mpr(mpr, "ChildEntityPart", None, "", en, 0, f"Mesh{mi_idx}")

        if child_static is not None:
            s_mesh = game.resolve_ref(child_static.get("MeshResource"))
            if s_mesh is not None:
                _emit_child_mesh(s_mesh, "Static", np.eye(4, dtype=np.float64), "ChildEntityPart", True, 0)
            for mi_idx, mref in enumerate(child_static.get("ModelPartResources", ())):
                mpr = game.resolve_ref(mref)
                if mpr:
                    _emit_child_mpr(mpr, "ChildEntityPart", None, "", True, 0, f"Static{mi_idx}")

    child_idx_counter = 0
    seen_child_ids: set[str] = set()
    for comp_ref in ce.get("EntityComponentResources", ()):
        comp = game.resolve_ref(comp_ref)
        if comp and comp["_type"] == "ChildEntityComponentResource":
            cid = str(comp.get("_id", ""))
            if cid:
                seen_child_ids.add(cid)
            try:
                _assemble_child_entity(comp, child_idx_counter)
            except Exception:
                pass
            child_idx_counter += 1

    # Also check selected variant group for any variant-exclusive ChildEntityComponentResource (e.g. elemental weapons/turrets)
    if selected_variant is not None and selected_variant.group_id != entity_oid.group_id:
        if "ChildEntityComponentResource" in game.graph.group_types(selected_variant.group_id):
            try:
                for v_obj in game.read_group(selected_variant.group_id, read_subgroups=True):
                    if v_obj.get("_type") != "ChildEntityComponentResource":
                        continue
                    cid = str(v_obj.get("_id", ""))
                    if cid in seen_child_ids:
                        continue
                    seen_child_ids.add(cid)
                    try:
                        _assemble_child_entity(v_obj, child_idx_counter)
                    except Exception:
                        pass
                    child_idx_counter += 1
            except Exception:
                pass

    # Step 3: Parse all referenced materials AND apply Decima Variant/Apex RenderEffectShaderOverridesResource!
    tex_overrides: dict[int, ObjectId] = {}
    var_overrides: dict[int, dict[str, Any]] = {}
    override_res_ids: list[str] = []
    if selected_variant is not None:
        tex_overrides, var_overrides, override_res_ids = collect_variant_shader_overrides(game, selected_variant.group_id)

    materials: dict[str, MaterialInfo] = {}
    total_override_hits = 0
    for mid, mobj in material_objs.items():
        mat_info = parse_material(game, mobj)
        if tex_overrides or var_overrides:
            total_override_hits += apply_shader_overrides(game, mat_info, tex_overrides, var_overrides)
        materials[mid] = mat_info

    # Step 4: Route Focus-Scan wireframe overlays (ForwardForeground-only with no PBR maps),
    # unattached origin proxy parts, and intact DestructibilityParts with combat Enabled=False
    # (such as Slaughterspine's intact Antenna 1818:237!) cleanly into their proper collections.
    for part in parts:
        p_mids = [pr.material_id for pr in part.mesh.primitives if pr.material_id]
        mat_names_low = [materials[mid].name.lower() for mid in p_mids if mid in materials]
        if p_mids:
            all_focus_overlay = True
            for mid in p_mids:
                mi = materials.get(mid)
                pt = mi.primary_technique if mi else None
                passes = pt.available_passes if pt else []
                has_pbr = any(
                    r in mi.roles
                    for r in ("Color", "Normal", "Surface_RMA", "Emissive", "MuscleFiber_Color", "MuscleFiber_Normal")
                ) if mi else False
                if not (passes == ["ForwardForeground"] and not has_pbr):
                    all_focus_overlay = False
                    break
            if all_focus_overlay:
                part.collection = "Focus_Overlay"
                part.enabled_by_default = False
                continue

        if not part.is_skinned and part.parent_joint_index is None:
            part.collection = "Unattached_Misc"
            part.enabled_by_default = False
            continue

        # In Decima, `DestructibilityPart.Enabled=False` disables combat shootability rather than
        # hiding intact parts (e.g. Slaughterspine Antenna 1818:237 uses intact Plates_02!).
        # Enable intact `InitialState` parts unless their material is genuinely damaged/destroyed.
        if part.source_kind == "DestructibilityIntact" and part.state_index == 0 and not part.enabled_by_default:
            is_damaged_mat = bool(mat_names_low) and all(
                any(tok in mn for tok in ("_des_", "_dmg_", "destroyed", "_broken")) for mn in mat_names_low
            )
            if is_damaged_mat:
                part.collection = "Plates_Broken_States"
                part.enabled_by_default = False
            else:
                part.enabled_by_default = True
                if part.name.endswith("_Des"):
                    part.name = part.name[:-4]

    return MachineAssembly(
        entity_id=str(entity_oid),
        codename=codename or f"group_{entity_oid.group_id}",
        display_name=display_name or codename or str(entity_oid),
        variant_label=selected_variant.label if selected_variant else "Base",
        available_variants=variants,
        skeleton=skeleton,
        parts=parts,
        materials=materials,
        official_variant_name=cr_name or (selected_variant.official_name if selected_variant else ""),
        official_variant_description=cr_desc or (selected_variant.official_description if selected_variant else ""),
        shader_override_count=total_override_hits,
        shader_override_resource_ids=override_res_ids,
    )
