"""Universal HFW General Asset Assembler & Guerrilla Developer-Directory Asset Catalog.

Organizes ALL renderable Horizon Forbidden West assets (Weapons, Wildlife/Animals,
Building Blocks / Ruins / Architecture, Eco Assets / Trees / Foliage, Landscape / Props,
Machines, Characters & Shaders) into Guerrilla's original `work:` developer directory tree
(`models/weapons/...`, `models/characters/animals/...`, `models/building_blocks/...`,
`models/eco_assets_hrz2/...`, `dlc/models/...`, etc.) and exports any asset with full
geometry, transforms, optional skeleton/skinning, and Decima PBR shaders to Blender & UE5!
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from hfw_toolkit.decima.catalog import AssetCatalog
from hfw_toolkit.decima.reader import ObjectId
from hfw_toolkit.decima.streaming_graph import HFWGame
from hfw_toolkit.extract.character_assembler import _remap_mesh_weights
from hfw_toolkit.extract.machine_assembler import AssembledPart, MachineAssembly
from hfw_toolkit.extract.materials import MaterialInfo, parse_material
from hfw_toolkit.extract.meshes import DecodedMesh, decode_mesh_hierarchy
from hfw_toolkit.extract.skeleton import (
    SkeletonData,
    add_skeleton_helpers,
    decima_mat44_to_np,
    extract_skeleton,
)

# Priority order when picking the cleanest renderable root RTTI type inside an asset group
_ROOT_RENDER_TYPES_RIGGED: list[tuple[str, int, bool]] = [
    ("SkinnedModelResource", 100, True),
    ("StaticModelResource", 90, False),
    ("ModelPartResource", 85, False),
    ("LodMeshResource", 80, False),
    ("MultiMeshResource", 78, False),
    ("RegularSkinnedMeshResource", 74, True),
    ("StaticMeshResource", 68, False),
    ("PrefabResource", 60, False),
    ("EntityResource", 55, False),
]
_ROOT_RENDER_TYPES_STATIC: list[tuple[str, int, bool]] = [
    ("StaticModelResource", 100, False),
    ("MultiMeshResource", 94, False),
    ("LodMeshResource", 90, False),
    ("StaticMeshResource", 86, False),
    ("ModelPartResource", 78, False),
    ("PrefabResource", 72, False),
    ("SkinnedModelResource", 64, True),
    ("RegularSkinnedMeshResource", 60, True),
    ("EntityResource", 55, False),
]
_ROOT_RENDER_TYPES = _ROOT_RENDER_TYPES_RIGGED
_RENDER_TYPE_SET = {t for t, _, _ in _ROOT_RENDER_TYPES_RIGGED}


@dataclass(slots=True)
class DevAssetRecord:
    asset_id: str         # e.g. "asset:14476:0" or "char:aloy" or "machine:1587:51"
    dev_path: str         # Guerrilla dev folder e.g. "models/weapons/bow_precision"
    parent_dir: str       # e.g. "models/weapons"
    category: str         # "weapons" | "animals" | "building_blocks" | "eco_assets" | "landscape" | "characters" | "machines" | "props"
    display_name: str     # Clean human name e.g. "Bow Precision"
    resource_type: str    # "SkinnedModelResource" | "StaticModelResource" | "LodMeshResource" | ...
    resource_id: str      # "14476:0"
    group_id: int
    is_skinned: bool
    tset_count: int = 1
    shader_count: int = 0
    is_dlc: bool = False


def _normalize_dev_asset_path(raw_path: str) -> str:
    """Convert `work:Models/weapons/Bow_Precision/textures/Precision_Bow_CLR.psd` -> `Models/weapons/Bow_Precision`."""
    s = str(raw_path).replace("\\", "/")
    if s.lower().startswith("work:"):
        s = s[5:]
    s = s.strip("/")
    low = s.lower()
    for marker in ("/textures/", "/texture/", "/source/", "/_source/"):
        idx = low.find(marker)
        if idx > 0:
            s = s[:idx]
            low = s.lower()
            break
    else:
        parts = [p for p in s.split("/") if p]
        if len(parts) > 1 and "." in parts[-1]:
            s = "/".join(parts[:-1])
    return s.strip("/")


def _classify_asset_category(dev_path: str) -> str:
    low = dev_path.lower()
    if "models/characters/robots" in low or "dlc/models/characters/robots" in low:
        return "machines"
    if "models/characters/humans" in low or "dlc/models/characters/humans" in low:
        return "characters"
    if "models/characters/animals" in low or "dlc/models/characters/animals" in low:
        return "animals"
    if "/weapons" in low or low.startswith("models/weapons"):
        return "weapons"
    if "eco_assets" in low:
        return "eco_assets"
    if "building_blocks" in low or "compound_blocks" in low:
        return "building_blocks"
    if "landscape_assets" in low or "levels/worlds" in low:
        return "landscape"
    return "props"


def _humanize_asset_name(dev_path: str) -> str:
    leaf = dev_path.rstrip("/").split("/")[-1]
    s = re.sub(r"^(?:h2_|hrz2_)", "", leaf, flags=re.IGNORECASE)
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    s = re.sub(r"[_\-]+", " ", s).strip()
    return " ".join(w.capitalize() for w in s.split()) or leaf


def discover_dev_directory_assets(
    game: HFWGame,
    catalog: AssetCatalog,
) -> list[DevAssetRecord]:
    """Build the complete Guerrilla Developer Directory Asset Catalog from all `10,660` TextureSets
    and streaming-graph supergroup edges in ~1–2 seconds!

    Maps every developer folder (`models/weapons/...`, `models/characters/animals/...`,
    `models/building_blocks/...`, `models/eco_assets_hrz2/...`, `dlc/models/...`, etc.)
    to its cleanest dedicated renderable model/mesh group and resource ID.
    """
    graph = game.graph
    rev = graph.supergroups
    names_by_hash = graph.type_names_by_hash
    tt_idx = graph.type_table_indices

    # Pre-resolve type indices for O(1) slice lookups (no list allocations!)
    tidx_by_name = {name: i for i, name in enumerate(names_by_hash)}
    shader_tidx = tidx_by_name.get("ShaderResource", -1)
    skel_tidx = tidx_by_name.get("Skeleton", -1)
    reg_sk_tidx = tidx_by_name.get("RegularSkinnedMeshResource", -1)
    rigged_prios = [(tidx_by_name[t], t, sc, is_sk) for t, sc, is_sk in _ROOT_RENDER_TYPES_RIGGED if t in tidx_by_name]
    static_prios = [(tidx_by_name[t], t, sc, is_sk) for t, sc, is_sk in _ROOT_RENDER_TYPES_STATIC if t in tidx_by_name]
    all_render_tidxs = {ti for ti, _, _, _ in rigged_prios}

    # 1. Group TextureSets by canonical developer directory (`dev_path.lower()`) + track per-group folder diversity!
    dir_entries: dict[str, dict[str, Any]] = {}
    group_folders: dict[int, set[str]] = {}
    for tgid, _toid, _name, pjson in catalog.conn.execute(
        "SELECT group_id, object_index, name, paths_json FROM texture_sets"
    ).fetchall():
        try:
            items = json.loads(pjson) if pjson else []
        except Exception:
            continue
        p0 = ""
        for it in items:
            cand = it.get("path", "") if isinstance(it, dict) else str(it)
            if cand:
                p0 = cand
                break
        if not p0:
            continue

        dev_path = _normalize_dev_asset_path(p0)
        low = dev_path.lower()
        if not (low.startswith("models/") or low.startswith("dlc/models/")):
            continue
        if any(skip in low for skip in ("/luts", "/vfx", "/particles", "/fx_", "/test", "detailarray", "global_")):
            continue

        ent = dir_entries.get(low)
        if ent is None:
            ent = {
                "dev_path": dev_path,
                "tset_groups": set(),
                "tset_count": 0,
            }
            dir_entries[low] = ent
        ig = int(tgid)
        ent["tset_groups"].add(ig)
        ent["tset_count"] += 1
        group_folders.setdefault(ig, set()).add(low)

    # 2. Category-aware & purity-aware group scorer
    def _score_group(gid: int, prefer_static: bool, low_path: str) -> tuple[int, str, int, bool] | None:
        if gid <= 5 or gid not in graph.groups:
            return None
        gm = graph.groups[gid]
        n_objs = gm.type_count
        if n_objs == 0 or n_objs > 12000:
            return None
        sl = tt_idx[gm.type_start : gm.type_start + n_objs]
        tset = set(sl)
        if not (tset & all_render_tidxs):
            return None
        prios = static_prios if prefer_static else rigged_prios
        for ti, rtype, base_sc, is_sk in prios:
            if ti in tset:
                oid = sl.index(ti)
                size_bonus = 30 if n_objs <= 250 else (16 if n_objs <= 1200 else (-18 if n_objs > 4000 else 0))
                # Purity bonus: strongly favor groups dedicated to THIS folder over mixed world chunks!
                gf = group_folders.get(gid)
                purity_bonus = 24 if (gf and len(gf) == 1 and low_path in gf) else (8 if (gf and low_path in gf and len(gf) <= 3) else 0)
                has_sk = is_sk or (not prefer_static and ((skel_tidx in tset) or (reg_sk_tidx in tset)))
                return (base_sc + size_bonus + purity_bonus, rtype, oid, has_sk)
        return None

    records: list[DevAssetRecord] = []
    for low_path, info in sorted(dir_entries.items()):
        dev_path = info["dev_path"]
        cat_name = _classify_asset_category(dev_path)
        prefer_static = cat_name in ("building_blocks", "eco_assets", "landscape", "props")
        tgroups = sorted(info["tset_groups"])
        best: tuple[int, int, str, int, bool] | None = None

        # Check same group first (self-contained dedicated asset groups!)
        for tg in tgroups:
            if tg <= 5:
                continue
            sc = _score_group(tg, prefer_static, low_path)
            if sc is not None:
                cand = (sc[0] + 20, tg, sc[1], sc[2], sc[3])
                if best is None or cand[0] > best[0]:
                    best = cand

        # Check 1-hop parent groups referencing these TextureSets
        if best is None or best[0] < 130:
            checked_p: set[int] = set()
            for tg in tgroups:
                if tg <= 5:
                    continue
                parents = rev.get(tg, ())
                if len(parents) > 200:
                    parents = [p for p in parents if graph.groups[p].type_count <= 600][:30]
                for pg in parents[:40]:
                    if pg in checked_p:
                        continue
                    checked_p.add(pg)
                    sc = _score_group(pg, prefer_static, low_path)
                    if sc is not None:
                        cand = (sc[0], pg, sc[1], sc[2], sc[3])
                        if best is None or cand[0] > best[0]:
                            best = cand

        if best is None:
            continue

        _score, best_gid, rtype, roid, is_sk = best
        bgm = graph.groups[best_gid]
        shader_cnt = tt_idx[bgm.type_start : bgm.type_start + bgm.type_count].count(shader_tidx) if shader_tidx >= 0 else 0
        parent_dir = "/".join(dev_path.split("/")[:-1])
        records.append(
            DevAssetRecord(
                asset_id=f"asset:{best_gid}:{roid}",
                dev_path=dev_path,
                parent_dir=parent_dir,
                category=cat_name,
                display_name=_humanize_asset_name(dev_path),
                resource_type=rtype,
                resource_id=f"{best_gid}:{roid}",
                group_id=best_gid,
                is_skinned=is_sk,
                tset_count=info["tset_count"],
                shader_count=shader_cnt,
                is_dlc=low_path.startswith("dlc/"),
            )
        )

    return records


# ==============================================================================
# UNIVERSAL GENERAL ASSET ASSEMBLER
# ==============================================================================

def _collect_meshes_from_object(
    game: HFWGame,
    obj: dict[str, Any] | None,
    lod_index: int = 0,
    depth: int = 0,
    seen_ids: set[str] | None = None,
    out_skeletons: list[dict[str, Any]] | None = None,
    out_helpers: list[dict[str, Any]] | None = None,
) -> list[tuple[str, DecodedMesh, np.ndarray]]:
    """Recursively extract `(part_name, DecodedMesh, world_4x4)` from ANY Decima model/mesh/entity/prefab object."""
    if obj is None or depth > 6:
        return []
    if seen_ids is None:
        seen_ids = set()
    oid = str(obj.get("_id", ""))
    if oid and oid in seen_ids:
        return []
    if oid:
        seen_ids.add(oid)

    otype = obj.get("_type", "")
    ident = np.eye(4, dtype=np.float64)
    results: list[tuple[str, DecodedMesh, np.ndarray]] = []

    if otype in ("StaticMeshResource", "RegularSkinnedMeshResource", "LodMeshResource", "MultiMeshResource"):
        dms = decode_mesh_hierarchy(game, obj, lod_index=lod_index)
        for i, dm in enumerate(dms):
            wmat = dm.local_transform if dm.local_transform is not None else ident
            results.append((f"{otype}_{oid.replace(':', '_')}_{i}", dm, wmat))
        return results

    if otype == "ModelPartResource":
        mres = game.resolve_ref(obj.get("MeshResource"))
        return _collect_meshes_from_object(game, mres, lod_index, depth + 1, seen_ids, out_skeletons, out_helpers)

    if otype in ("SkinnedModelResource", "StaticModelResource"):
        if out_skeletons is not None:
            for sk_fld in ("InitialRepresentationSkeleton", "Skeleton"):
                sk = game.resolve_ref(obj.get(sk_fld))
                if sk is not None:
                    out_skeletons.append(sk)
        if out_helpers is not None:
            sh = game.resolve_ref(obj.get("SkeletonHelpers"))
            if sh is not None:
                out_helpers.append(sh)
        mps = obj.get("ModelPartResources") or ()
        init_states = obj.get("InitiallyEnabledStates") or ()
        for i, mref in enumerate(mps):
            if i < len(init_states) and not init_states[i]:
                continue
            mp = game.resolve_ref(mref)
            results.extend(_collect_meshes_from_object(game, mp, lod_index, depth + 1, seen_ids, out_skeletons, out_helpers))
        # StaticModelResource may also reference a direct `MeshResource`
        if obj.get("MeshResource"):
            results.extend(
                _collect_meshes_from_object(
                    game, game.resolve_ref(obj.get("MeshResource")), lod_index, depth + 1, seen_ids, out_skeletons, out_helpers
                )
            )
        return results

    if otype in ("EntityResource", "ControlledEntityResource", "HumanoidEntityResource", "DestructibleEntityResource"):
        for comp_ref in obj.get("Components") or obj.get("EntityComponentResources") or ():
            comp = game.resolve_ref(comp_ref)
            if comp is None:
                continue
            ct = comp.get("_type", "")
            if ct in ("SkinnedModelResource", "StaticModelResource", "ModelPartResource", "PrefabResource", "EntityComponentSetResource"):
                results.extend(_collect_meshes_from_object(game, comp, lod_index, depth + 1, seen_ids, out_skeletons, out_helpers))
            elif comp.get("ModelResource"):
                results.extend(
                    _collect_meshes_from_object(
                        game, game.resolve_ref(comp.get("ModelResource")), lod_index, depth + 1, seen_ids, out_skeletons, out_helpers
                    )
                )
            elif comp.get("MeshResource"):
                results.extend(
                    _collect_meshes_from_object(
                        game, game.resolve_ref(comp.get("MeshResource")), lod_index, depth + 1, seen_ids, out_skeletons, out_helpers
                    )
                )
        return results

    if otype == "PrefabResource":
        for child_ref in obj.get("ObjectCollection") or obj.get("Objects") or obj.get("Instances") or ():
            child = game.resolve_ref(child_ref) if not isinstance(child_ref, dict) else child_ref
            if isinstance(child, dict):
                # Optional prefab transform
                tmat = ident
                if isinstance(child.get("Orientation"), dict) and "Col0" in child["Orientation"]:
                    tmat = decima_mat44_to_np(child["Orientation"])
                sub_res = _collect_meshes_from_object(game, child, lod_index, depth + 1, seen_ids, out_skeletons, out_helpers)
                for nm, dm, wm in sub_res:
                    results.append((nm, dm, tmat @ wm))
        return results

    if otype == "EntityComponentSetResource":
        for ref in obj.get("ComponentResources") or ():
            results.extend(
                _collect_meshes_from_object(game, game.resolve_ref(ref), lod_index, depth + 1, seen_ids, out_skeletons, out_helpers)
            )
        return results

    # Generic fallback: inspect common mesh/model link fields
    for fld in ("MeshResource", "ModelResource", "SkinnedModelResource", "StaticModelResource", "Prefab"):
        if obj.get(fld):
            results.extend(
                _collect_meshes_from_object(game, game.resolve_ref(obj.get(fld)), lod_index, depth + 1, seen_ids, out_skeletons, out_helpers)
            )
    return results


def assemble_general_asset(
    game: HFWGame,
    resource_id: str,
    *,
    display_name: str = "",
    dev_path: str = "",
    category: str = "props",
    lod_index: int = 0,
    max_parts: int = 64,
) -> MachineAssembly:
    """Assemble ANY HFW static or rigged asset (`resource_id` = `"group:obj"` or `"asset:group:obj"` or group int)
    into a canonical `MachineAssembly` ready for Blender `.blend` and UE5 `.fbx` + script export!
    """
    clean_id = resource_id.split(":", 1)[1] if resource_id.startswith("asset:") else resource_id
    if ":" in clean_id:
        oid = ObjectId.parse(clean_id)
        root_obj = game.read_object(oid, read_subgroups=True)
        gid = oid.group_id
    else:
        gid = int(clean_id)
        objs = game.read_group(gid, read_subgroups=True)
        root_obj = None
        for rtype, _, _ in _ROOT_RENDER_TYPES:
            for o in objs:
                if o.get("_type") == rtype:
                    root_obj = o
                    break
            if root_obj is not None:
                break
        if root_obj is None:
            raise ValueError(f"No renderable model/mesh found in Group {gid}")
        oid = root_obj["_id"]

    skel_objs: list[dict[str, Any]] = []
    helper_objs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    extracted = _collect_meshes_from_object(
        game, root_obj, lod_index=lod_index, seen_ids=seen_ids, out_skeletons=skel_objs, out_helpers=helper_objs
    )

    # Helper to check whether an extracted mesh list actually carries real shaded color-pass materials
    # (filters out DepthOnly / Shadowmap occluder proxy LodMeshResources like `rock/volcanic 705:1`!)
    materials_cache: dict[str, MaterialInfo] = {}
    _COLOR_PASSES = frozenset({
        "Deferred", "DeferredEmissive", "DeferredForeground", "CustomDeferred",
        "Transparency", "CustomForward", "Forward", "ForwardForeground", "ForwardBackground", "Unlit",
    })

    def _eval_visual_score(mesh_list: list[tuple[str, DecodedMesh, np.ndarray]]) -> tuple[int, int, int]:
        textured_mats = 0
        total_tbs = 0
        total_verts = 0
        seen_m: set[str] = set()
        for _nm, dm, _wm in mesh_list:
            for pr in dm.primitives:
                total_verts += len(pr.positions)
                mid = pr.material_id
                if not mid or mid in seen_m:
                    continue
                seen_m.add(mid)
                mi = materials_cache.get(mid)
                if mi is None:
                    try:
                        m_obj = game.read_object(ObjectId.parse(mid), read_subgroups=True)
                        mi = parse_material(game, m_obj)
                        materials_cache[mid] = mi
                    except Exception:
                        continue
                pt = mi.primary_technique
                if pt and pt.texture_bindings and (set(pt.available_passes) & _COLOR_PASSES):
                    textured_mats += 1
                    total_tbs += len(pt.texture_bindings)
        return (textured_mats, total_tbs, total_verts)

    # If the selected root object yielded 0 geometry OR only a DepthOnly/Shadowmap occluder mesh with 0 textured materials,
    # scan all renderable model resources in the same group and pick the richest shaded visual asset!
    cur_score = _eval_visual_score(extracted) if extracted else (0, 0, 0)
    if (not extracted) or (cur_score[0] == 0):
        best_cand = extracted
        best_score = cur_score
        best_skels = list(skel_objs)
        best_helpers = list(helper_objs)
        grp_objs = game.read_group(gid, read_subgroups=True)
        for cand_type in ("SkinnedModelResource", "StaticModelResource", "MultiMeshResource", "LodMeshResource", "RegularSkinnedMeshResource", "StaticMeshResource"):
            for o in grp_objs:
                if o.get("_type") != cand_type:
                    continue
                c_skels: list[dict[str, Any]] = []
                c_helpers: list[dict[str, Any]] = []
                c_meshes = _collect_meshes_from_object(
                    game, o, lod_index=lod_index, seen_ids=set(), out_skeletons=c_skels, out_helpers=c_helpers
                )
                if not c_meshes:
                    continue
                sc = _eval_visual_score(c_meshes[:max_parts])
                if sc > best_score:
                    best_score = sc
                    best_cand = c_meshes
                    best_skels = c_skels
                    best_helpers = c_helpers
            if best_score[0] > 0:
                break
        if best_cand:
            extracted = best_cand
            skel_objs = best_skels
            helper_objs = best_helpers

    if not extracted:
        raise ValueError(f"Asset {oid} ({root_obj.get('_type')}) yielded 0 renderable meshes.")

    # Trim overly huge multi-object sector dumps cleanly while preserving primary asset
    extracted = extracted[:max_parts]

    # Build skeleton if any mesh is skinned (Weapons/Bows, Wildlife/Animals, Animated Props!).
    # CRITICAL: On HFW weapons like `Bow_Heavy_Quen_SE`, `SMR.Skeleton` often has `HasBindPose=False`
    # (ALL joints at `(0,0,0)`!), whereas `dm.skeleton_id` (`RegularSkinnedMeshResource.Skeleton`)
    # and `InitialRepresentationSkeleton` carry the real bind-pose joint transforms AND match the vertex weights!
    master_skel = SkeletonData(object_id="none")
    any_skinned = any(dm.is_skinned for _nm, dm, _wm in extracted)
    if any_skinned or skel_objs:
        direct_mesh_skel_ids: set[str] = set()
        cand_skel_objs: dict[str, dict[str, Any]] = {}

        # Priority 1: Skeletons referenced directly by decoded skinned meshes (`dm.skeleton_id`)
        for _nm, dm, _wm in extracted:
            if dm.is_skinned and dm.skeleton_id:
                direct_mesh_skel_ids.add(dm.skeleton_id)
                if dm.skeleton_id not in cand_skel_objs:
                    try:
                        cand_skel_objs[dm.skeleton_id] = game.read_object(ObjectId.parse(dm.skeleton_id), read_subgroups=True)
                    except Exception:
                        pass
        # Priority 2: Skeletons collected from SkinnedModelResource (`InitialRepresentationSkeleton`, `Skeleton`)
        for sko in skel_objs:
            sid = str(sko.get("_id", ""))
            if sid and sid not in cand_skel_objs:
                cand_skel_objs[sid] = sko

        skel_cache: dict[str, SkeletonData] = {}
        best_sk_key: tuple[int, int, int, int, int] | None = None
        best_sk: SkeletonData | None = None

        def _score_skel(sid: str, sko: dict[str, Any]) -> tuple[tuple[int, int, int, int, int], SkeletonData] | None:
            try:
                sd = extract_skeleton(game, sko)
            except Exception:
                return None
            skel_cache[sid] = sd
            nz = sum(1 for j in sd.joints if float(np.linalg.norm(j.world_matrix[:3, 3])) > 1e-4)
            key = (
                1 if nz > 0 else 0,
                1 if sid in direct_mesh_skel_ids else 0,
                1 if sko.get("HasBindPose") else 0,
                nz,
                len(sd.joints),
            )
            return key, sd

        for sid, sko in cand_skel_objs.items():
            res = _score_skel(sid, sko)
            if res is not None and (best_sk_key is None or res[0] > best_sk_key):
                best_sk_key, best_sk = res

        # Priority 3: If all direct candidates still have 0 non-zero bone positions, scan group `Skeleton` objects!
        if best_sk_key is None or best_sk_key[0] == 0:
            for o in game.read_group(gid, read_subgroups=True):
                if o.get("_type") == "Skeleton":
                    sid = str(o.get("_id", ""))
                    if sid and sid not in skel_cache:
                        res = _score_skel(sid, o)
                        if res is not None and (best_sk_key is None or res[0] > best_sk_key):
                            best_sk_key, best_sk = res

        if best_sk is not None:
            master_skel = best_sk
            for sh in helper_objs:
                try:
                    add_skeleton_helpers(master_skel, sh)
                except Exception:
                    pass
            # Remap any skinned mesh whose `skeleton_id` differs from `master_skel.object_id`!
            merge_maps: dict[str, dict[int, int]] = {}
            for _nm, dm, _wm in extracted:
                if dm.is_skinned and dm.skeleton_id and dm.skeleton_id != master_skel.object_id:
                    _remap_mesh_weights(game, dm, master_skel, skel_cache, merge_maps)

    materials: dict[str, MaterialInfo] = {}
    parts: list[AssembledPart] = []
    ident = np.eye(4, dtype=np.float64)

    for idx, (pname, dm, wmat) in enumerate(extracted):
        for pr in dm.primitives:
            if pr.material_id and pr.material_id not in materials:
                if pr.material_id in materials_cache:
                    materials[pr.material_id] = materials_cache[pr.material_id]
                else:
                    try:
                        m_obj = game.read_object(ObjectId.parse(pr.material_id), read_subgroups=True)
                        materials[pr.material_id] = parse_material(game, m_obj)
                    except Exception:
                        pass
        parts.append(
            AssembledPart(
                part_id=f"{dm.object_id}_{idx}",
                name=pname,
                collection="Assets",
                source_kind=dm.mesh_type,
                is_skinned=dm.is_skinned and bool(master_skel.joints),
                parent_joint_index=None,
                parent_joint_name=None,
                helper_name=None,
                local_offset_matrix=ident if dm.is_skinned else wmat.copy(),
                world_matrix=ident if dm.is_skinned else wmat.copy(),
                mesh=dm,
                enabled_by_default=True,
                state_index=0,
            )
        )

    disp = display_name or (_humanize_asset_name(dev_path) if dev_path else f"Asset_{gid}_{oid.object_index}")
    return MachineAssembly(
        entity_id=f"asset:{oid}",
        codename=re.sub(r"[^a-z0-9_]+", "_", disp.lower()).strip("_") or f"asset_{gid}",
        display_name=disp,
        variant_label=dev_path or category,
        available_variants=[],
        skeleton=master_skel,
        parts=parts,
        materials=materials,
        official_variant_name=disp,
        official_variant_description=dev_path,
    )
