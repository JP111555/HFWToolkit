"""Universal Non-Destructive HFW Control Rig (Full IK + FK + Hands/Feet/Detail + Rest-Lock) & Bidirectional Baker.

100% pure Blender Python (`bpy`, `mathutils`, `numpy`, stdlib) — zero external dependencies.
The pure-Python chain analyzer (`analyze_rig_chains`) can ALSO be imported outside Blender
(e.g. by `hfw_toolkit/shaders/unreal_export.py` and `pytest`) so Blender and UE5 Control Rigs
share the exact same skeleton archetype and limb/detail chain detection.

Non-Destructive `DEF + MCH + CTRL` Architecture:
  - Original Decima deform bones (`DEF`) are NEVER renamed, reparented, or modified.
  - Imported HFW `EA08` animations key directly onto untouched `DEF` pose bones.
  - EVERY leg and EVERY arm supports LIVE `FK (0.0) <-> IK (1.0)` blending (`hfw_ik_fk_legs`,
    `hfw_ik_fk_arms`) with complete `CTRL_FK_*` limb chains AND zero-drift `MCH_IK_*` solvers!
  - ALL smaller articulated joints (humanoid fingers/thumbs/toes, machine foot talons/hooves,
    forearm claws, head fans/crests, antennae, dorsal spikes, jaw sub-segments, weapon turrets)
    get clean FK controls in `Controls_Detail` riding smoothly on `MCH_Follow_*` in BOTH IK & FK!
  - Every remaining non-control `DEF` bone gets a driven `HFW_RestLock` (`LOCAL` space zero-clamp
    driven by `hfw_rig_master * hfw_rig_lock_secondary`) so switching to Control Rig mode (`1.0`)
    eliminates 100% of leftover "ghost" game animation on unconstrained bones!
  - Bidirectional 1-click baking:
      * `bake_action_to_control_rig(arm_obj)` -> snaps & keys ALL IK, FK, Spine, Head, Tail,
        Finger, Toe, and Detail controls onto any playing HFW `DEF` animation!
      * `bake_control_rig_to_deform(arm_obj)` -> bakes your Control Rig edits back onto clean
        `DEF` keyframes (`hfw_rig_master = 0.0`) ready for FBX / Unreal Engine 5 export!
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

try:
    import bpy
    from mathutils import Euler, Matrix, Quaternion, Vector
except ImportError:
    bpy = None  # type: ignore[assignment]
    Matrix = Quaternion = Vector = Euler = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Pure-Python Universal HFW Skeleton Chain Analyzer (Shared by Blender & UE5)
# ---------------------------------------------------------------------------
_IGNORE_TOKENS = (
    "facial_",
    "_pbd_",
    "_jbd",
    "_push_",
    "_corrective",
    "_twist",
    "_helper",
    "_cloth",
    "_hair",
    "_braid",
    "_skirt",
    "_strap",
    "_tassel",
    "_flap",
    "_rope",
    "_ribbon",
    "_cape",
    "ikleft",
    "ikright",
    "ikhips",
    "ikfoot",
    "ikhand",
    "_ik_",
    "weapon",
    "quiver",
    "focus",
    "attach",
    "socket",
    "camera",
    "mount",
    "rider",
    "canister",
    "armor",
    "plate",
    "antenna",
    "fin_",
    "spike",
    "feather",
    "toe",
    "finger",
    "thumb",
    "index",
    "middle",
    "ring",
    "pinky",
    "claw",
    "talon",
    "eye",
    "_ear_",
    "earjoint",
    "ear_sjnt",
    "earring",
    "gear",
    "wear",
    "sleeve",
    "bracer",
    "pauldron",
    "gauntlet",
    "pouch",
    "brow",
    "lid",
    "lip",
    "cheek",
    "nose",
    "tongue",
)

# Bones NEVER given FK detail controls (cloth sim, helpers, IK dummies, corrective twist/push)
_DETAIL_FORBIDDEN_TOKENS = (
    "ctrl_",
    "mch_",
    "wgt_",
    "facial_",
    "_pbd_",
    "_jbd",
    "_fjnt",
    "proc",
    "roll",
    "push",
    "aim",
    "mover",
    "corrective",
    "twist",
    "helper",
    "forearm",
    "upperarm",
    "thigh",
    "shin",
    "calf",
    "scapula",
    "clavicle",
    "hrz",
    "wristup",
    "wristbt",
    "wristin",
    "wristot",
    "elbow",
    "_cloth",
    "_hair",
    "_braid",
    "_skirt",
    "_strap",
    "_rope",
    "_ribbon",
    "_cape",
    "ikleft",
    "ikright",
    "ikhips",
    "ikfoot",
    "ikhand",
    "_ik_",
    "attach",
    "socket",
    "camera",
    "mount",
    "rider",
    "quiver",
    "focus",
    "sleeve",
    "bracer",
    "pauldron",
    "gauntlet",
    "pouch",
    "tassel",
    "earring",
    "cloth",
    "wear",
    "gear",
)


def _is_ignored_bone(name: str) -> bool:
    low = name.lower()
    if low.startswith(("ctrl_", "mch_", "wgt_", "facial_")):
        return True
    if "_aim_" in low or low.endswith("_aim") or low.startswith("aim_"):
        return True
    return any(tok in low for tok in _IGNORE_TOKENS)


def _is_detail_forbidden(name: str) -> bool:
    low = name.lower()
    if low.endswith("_aim") or low.startswith("aim_"):
        return True
    return any(tok in low for tok in _DETAIL_FORBIDDEN_TOKENS)


def _extract_side(name: str) -> str:
    low = name.lower()
    if low.startswith(("lf_", "lb_", "lm_", "lfleg", "lbleg")):
        return "L"
    if low.startswith(("rf_", "rb_", "rm_", "rfleg", "rbleg")):
        return "R"
    if low.startswith(("l_", "left_", "left")) or "_l_" in low or low.endswith("_l"):
        return "L"
    if low.startswith(("r_", "right_", "right")) or "_r_" in low or low.endswith("_r"):
        return "R"
    if low.startswith("c_") or "_c_" in low:
        return "C"
    return "C"


def _numeric_sort_key(name: str) -> tuple[int, str]:
    nums = re.findall(r"\d+", name)
    return (int(nums[-1]) if nums else -1, name.lower())


def _order_chain_topologically(names: list[str], parent_map: dict[str, str | None]) -> list[str]:
    if len(names) <= 1:
        return list(names)
    depth_cache: dict[str, int] = {}

    def depth(n: str) -> int:
        if n in depth_cache:
            return depth_cache[n]
        seen = {n}
        cur = parent_map.get(n)
        d = 0
        while cur and cur not in seen:
            seen.add(cur)
            d += 1
            cur = parent_map.get(cur)
        depth_cache[n] = d
        return d

    return sorted(dict.fromkeys(names), key=lambda n: (depth(n), _numeric_sort_key(n)))


def _longest_parent_chain(candidates: list[str], parent_map: dict[str, str | None]) -> list[str]:
    if not candidates:
        return []
    cand_set = set(candidates)
    best: list[str] = []
    for leaf in candidates:
        chain: list[str] = [leaf]
        cur = parent_map.get(leaf)
        seen = {leaf}
        while cur and cur not in seen:
            seen.add(cur)
            if cur in cand_set:
                chain.append(cur)
            cur = parent_map.get(cur)
        chain.reverse()
        if len(chain) > len(best):
            best = chain
    return best


def _collect_sub_chains(
    root_bones: list[str],
    children_map: dict[str, list[str]],
    head_map: dict[str, tuple[float, float, float]],
    excluded: set[str],
    max_depth: int = 4,
) -> list[list[str]]:
    """Trace clean linear descendant FK chains starting from `root_bones` (fingers, toes, claws, antennae)."""
    chains: list[list[str]] = []
    visited: set[str] = set()

    def _trace(start: str) -> None:
        if start in excluded or start in visited or _is_detail_forbidden(start):
            return
        chain: list[str] = []
        cur: str | None = start
        while cur and len(chain) < max_depth:
            if cur in excluded or cur in visited or _is_detail_forbidden(cur):
                break
            visited.add(cur)
            chain.append(cur)
            kids = [
                k
                for k in children_map.get(cur, [])
                if k not in excluded and k not in visited and not _is_detail_forbidden(k)
            ]
            if len(kids) == 1:
                cur = kids[0]
            else:
                if chain:
                    chains.append(chain)
                for k in kids:
                    _trace(k)
                return
        if chain:
            chains.append(chain)

    for rb in root_bones:
        _trace(rb)
    return chains


def analyze_rig_chains(joint_specs: list[dict[str, Any]]) -> dict[str, Any]:
    """Detect universal HFW Control Rig chains (Main FK/IK + Hands/Fingers + Feet/Toes + Secondary Details).

    Supports ALL HFW machine archetypes (Clawstrider, Slaughterspine, Thunderjaw, Behemoth,
    Charger, Sunwing, Slitherfang, etc.) AND Humanoids (Aloy, Companions, Tribal NPCs).
    """
    names: list[str] = []
    parent_map: dict[str, str | None] = {}
    head_map: dict[str, tuple[float, float, float]] = {}
    children_map: dict[str, list[str]] = {}

    for js in joint_specs:
        nm = str(js["name"])
        pn = js.get("parent_name")
        if pn is not None:
            pn = str(pn)
        hd = tuple(float(v) for v in js["head"][:3])
        names.append(nm)
        parent_map[nm] = pn
        head_map[nm] = hd
        if pn:
            children_map.setdefault(pn, []).append(nm)

    by_lower = {n.lower(): n for n in names}

    # 1. Root, Ground, Hips (COG)
    root_bone = (
        by_lower.get("rootbone")
        or by_lower.get("root")
        or next((n for n in names if not parent_map.get(n) and not _is_ignored_bone(n)), names[0] if names else None)
    )
    ground_bone = by_lower.get("groundbone")
    hips_bone = (
        by_lower.get("hipsbone")
        or by_lower.get("c_hips_sjnt_0")
        or by_lower.get("pelvis")
        or by_lower.get("hips")
    )
    if hips_bone is None and root_bone:
        cur = ground_bone or root_bone
        for _ in range(4):
            kids = [k for k in children_map.get(cur, []) if not _is_ignored_bone(k)]
            if len(kids) == 1:
                cur = kids[0]
            else:
                break
        if cur != root_bone:
            hips_bone = cur

    # 2. Spine & Chest (exclude helper orientation `_fjnt` bones such as humanoid `C_ChestOri_fjnt_0`)
    spine_cands = [
        n
        for n in names
        if not _is_ignored_bone(n)
        and "_fjnt" not in n.lower()
        and _extract_side(n) == "C"
        and any(k in n.lower() for k in ("spine", "chest", "torso", "abdomen", "pelvis_upper", "frontbody"))
        and "back" not in n.lower()
    ]
    spine_chain = _longest_parent_chain(spine_cands, parent_map)
    if not spine_chain and spine_cands:
        spine_chain = _order_chain_topologically(spine_cands, parent_map)

    # 3. Neck
    neck_cands = [
        n
        for n in names
        if not _is_ignored_bone(n) and "_fjnt" not in n.lower() and _extract_side(n) == "C" and "neck" in n.lower()
    ]
    neck_chain = _longest_parent_chain(neck_cands, parent_map)
    if not neck_chain and neck_cands:
        neck_chain = _order_chain_topologically(neck_cands, parent_map)

    # 4. Head & Jaws
    jaw_bones: list[str] = []
    for n in names:
        low = n.lower()
        if low.startswith(("ctrl_", "mch_", "wgt_")):
            continue
        if low == "facial_c_jaw" or (
            not _is_ignored_bone(n)
            and ("jaw" in low or "mandible" in low)
            and "piston" not in low
            and "muscle" not in low
            and "plate" not in low
        ):
            jaw_bones.append(n)
    jaw_set = set(jaw_bones)
    jaw_bones = [j for j in jaw_bones if parent_map.get(j) not in jaw_set]

    head_bone = (
        by_lower.get("c_head_sjnt_0")
        or by_lower.get("headjoint")
        or by_lower.get("head")
        or next(
            (
                n
                for n in names
                if not _is_ignored_bone(n)
                and _extract_side(n) == "C"
                and ("head" in n.lower())
                and not any(x in n.lower() for x in ("headgear", "headfan", "headgun", "headcanister", "headflap"))
            ),
            None,
        )
    )
    if head_bone is None and jaw_bones:
        jp = parent_map.get(jaw_bones[0])
        if jp and jp in head_map:
            head_bone = jp
    if head_bone is None and neck_chain:
        head_bone = neck_chain[-1]
    if head_bone and head_bone in neck_chain:
        neck_chain = [b for b in neck_chain if b != head_bone]

    # Compute skeleton height & ground level early for geometric limb/wing/quadruped disambiguation
    zs_all = [hd[2] for nm, hd in head_map.items() if not _is_ignored_bone(nm)]
    min_z = min(zs_all) if zs_all else 0.0
    max_z = max(zs_all) if zs_all else 1.8
    rig_height = max(0.5, max_z - min_z)

    # 5. Tails (Pure Core Tail Chains — include Stormbird FlexTail; exclude blades, feathers, brackets, plates, blasters, claws!)
    _TAIL_NON_CORE = (
        "blade",
        "feather",
        "bracket",
        "hinge",
        "shield",
        "canister",
        "weapon",
        "blaster",
        "exhaust",
        "launcher",
        "fin",
        "spike",
        "plate",
        "detail",
        "flap",
        "antenna",
        "control",
        "claw",
        "shell",
        "can",
    )
    tails: list[dict[str, Any]] = []
    for side in ("C", "L", "R"):
        tcands = [
            n
            for n in names
            if not _is_ignored_bone(n)
            and "_fjnt" not in n.lower()
            and _extract_side(n) == side
            and ("tail" in n.lower())
            and not any(tok in n.lower() for tok in _TAIL_NON_CORE)
        ]
        tchain = _longest_parent_chain(tcands, parent_map)
        if len(tchain) >= 2:
            tails.append({"name": f"Tail_{side}" if side != "C" else "Tail", "side": side, "bones": tchain})

    # 5b. Signature Articulated Appendages (Tremortusk Elephant Trunk, Bilegut Extendable Tongue!)
    appendages: list[dict[str, Any]] = []
    for app_kind, app_tok, app_title in (("trunk", "trunk", "Trunk"), ("tongue", "tongue", "Tongue")):
        acands = [
            n
            for n in names
            if not n.lower().startswith(("ctrl_", "mch_", "wgt_", "facial_"))
            and "_fjnt" not in n.lower()
            and "helper" not in n.lower()
            and "drum" not in n.lower()
            and "tip" not in n.lower()
            and _extract_side(n) == "C"
            and app_tok in n.lower()
        ]
        ach = _longest_parent_chain(acands, parent_map)
        if len(ach) >= 3:
            appendages.append(
                {
                    "name": app_title,
                    "kind": app_kind,
                    "side": "C",
                    "parent_def": parent_map.get(ach[0]),
                    "bones": ach,
                }
            )

    # 6. Legs (ALL HFW/HZD Conventions: 2-bone human/quad, 3-bone digitigrade theropod/flyer, 4-leg quad, 6-leg hexapod, aquatic flippers!)
    legs: list[dict[str, Any]] = []

    def _build_leg_entry(
        label: str,
        side: str,
        chain: list[str],
        *,
        allow_flipper: bool = False,
    ) -> dict[str, Any] | None:
        cleaned: list[str] = []
        for b in chain:
            if cleaned:
                p0, p1 = head_map[cleaned[-1]], head_map[b]
                if math.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2 + (p1[2] - p0[2]) ** 2) < 0.015:
                    continue
            cleaned.append(b)
        if len(cleaned) == 2:
            last_b = cleaned[-1]
            p_last = head_map[last_b]
            kids = [k for k in children_map.get(last_b, []) if not _is_ignored_bone(k) and k in head_map]
            valid_kids = [
                k for k in kids if math.sqrt(sum((head_map[k][ax] - p_last[ax]) ** 2 for ax in range(3))) > 0.05
            ]
            if valid_kids:
                valid_kids.sort(key=lambda k: head_map[k][2])
                cleaned.append(valid_kids[0])
        if len(cleaned) < 3:
            return None

        # Extract leading scapula/shoulder/hip-root joint if chain has >= 4 bones (gives quadrupeds clean shoulder articulation!)
        shoulder_bone: str | None = None
        if len(cleaned) >= 4 and any(
            k in cleaned[0].lower() for k in ("scapula", "shoulder", "hipbase", "fronthip", "backhip", "midhip")
        ):
            shoulder_bone = cleaned[0]
            cleaned = cleaned[1:]
        # Skip flat hip-offset lead bone on 5-6 bone machine legs (e.g. Dreadwing/Glinthawk/Shellsnapper `leg_0 -> leg_1` horizontal offset)
        while (
            len(cleaned) >= 5
            and not allow_flipper
            and len(cleaned) <= 6
            and abs(head_map[cleaned[1]][2] - head_map[cleaned[0]][2]) < rig_height * 0.12
            and head_map[cleaned[4]][2] < head_map[cleaned[3]][2] - rig_height * 0.06
        ):
            if shoulder_bone is None:
                shoulder_bone = cleaned[0]
            cleaned = cleaned[1:]

        extra_fk: list[str] = []
        is_flipper = False
        if allow_flipper and len(cleaned) >= 7:
            is_flipper = True
            ik_bones = cleaned[:3]
            foot_bone = cleaned[3]
            extra_fk = cleaned[4:]
            chain_len = 3
        elif len(cleaned) >= 4:
            if len(cleaned) > 4:
                # Prefer cutting at an explicit ankle/foot/hoof bone if present, else pick the 4 bones landing nearest ground
                foot_idx = next(
                    (
                        i
                        for i, b in enumerate(cleaned)
                        if i >= 3 and any(tk in b.lower() for tk in ("foot_0", "foot_sjnt_0", "hoof_sjnt_0", "foot_root", "ankle"))
                    ),
                    None,
                )
                if foot_idx is not None:
                    cleaned = cleaned[max(0, foot_idx - 3) : foot_idx + 1]
                elif any("hip" in b.lower() or "pelvis" in b.lower() or "shoulder" in b.lower() for b in cleaned[:2]):
                    cleaned = cleaned[-4:]
                elif head_map[cleaned[4]][2] < head_map[cleaned[3]][2] - rig_height * 0.08:
                    cleaned = cleaned[1:5]
                else:
                    cleaned = cleaned[:4]
            ik_bones = cleaned[:3]
            foot_bone = cleaned[3]
            chain_len = 3
        else:
            ik_bones = cleaned[:2]
            foot_bone = cleaned[2]
            chain_len = 2

        return {
            "name": label,
            "side": side,
            "shoulder": shoulder_bone,
            "ik_bones": ik_bones,
            "foot_bone": foot_bone,
            "extra_fk_bones": extra_fk,
            "is_flipper": is_flipper,
            "chain_length": chain_len,
            "is_digitigrade": chain_len == 3,
        }

    low_non_ignored = [n.lower() for n in names if not _is_ignored_bone(n)]
    has_quad_explicit = any(
        any(k in ln for k in ("frontleg", "rearleg", "backleg", "hindleg", "lfleg", "rfleg", "lf_leg", "rf_leg", "_front_knee", "_back_knee"))
        for ln in low_non_ignored
    )
    has_mid_legs = any("midleg" in ln for ln in low_non_ignored)

    if has_quad_explicit:
        for side in ("L", "R"):
            s_pref = side.lower()
            front_cands = [
                n
                for n in names
                if not _is_ignored_bone(n)
                and _extract_side(n) == side
                and not any(bk in n.lower() for bk in ("back", "rear", "hind", "mid", "flap"))
                and (
                    any(k in n.lower() for k in ("frontleg", "frontfoot", "fronthoof", "frontankle", "fronthip", "_front_shoulder", "_front_knee", "_front_ankle", "_front_foot", "scapula", "shoulder"))
                    or n.lower().startswith((f"{s_pref}fleg", f"{s_pref}f_leg", f"{s_pref}f_shoulder"))
                )
            ]
            fchain = _longest_parent_chain(front_cands, parent_map)
            # Wolf/Cat HZD/HFW quadrupeds (Ravager, Scorcher, Scrapper, Scrounger, Stalker): front legs are `L/R_scapula -> L/R_Leg -> L/R_Foot` alongside `L/R_backLeg`!
            if len(fchain) < 3:
                front_cands = [
                    n
                    for n in names
                    if not _is_ignored_bone(n)
                    and _extract_side(n) == side
                    and not any(bk in n.lower() for bk in ("back", "rear", "hind", "mid", "flap"))
                    and any(k in n.lower() for k in ("_leg_", "leg_sjnt", "_foot_", "foot_sjnt", "scapula", "shoulder"))
                ]
                fchain = _longest_parent_chain(front_cands, parent_map)
            if len(fchain) > 4 and any("hoof_sjnt_0" in b.lower() for b in fchain):
                idx_h = next(i for i, b in enumerate(fchain) if "hoof_sjnt_0" in b.lower())
                fchain = fchain[: idx_h + 1]
            # Tallneck `Front_Shoulder_01..03 -> Knee -> Ankle -> Foot_Root`: drop stacked shoulder offsets cleanly
            while len(fchain) > 5 and "shoulder" in fchain[0].lower() and "shoulder" in fchain[1].lower():
                fchain = fchain[1:]
            entry_f = _build_leg_entry(f"FrontLeg_{side}", side, fchain)
            if entry_f:
                legs.append(entry_f)

            if has_mid_legs:
                mid_cands = [
                    n
                    for n in names
                    if not _is_ignored_bone(n)
                    and _extract_side(n) == side
                    and any(k in n.lower() for k in ("midleg", "midhip", "midfoot"))
                ]
                mchain = _longest_parent_chain(mid_cands, parent_map)
                entry_m = _build_leg_entry(f"MidLeg_{side}", side, mchain)
                if entry_m:
                    legs.append(entry_m)

            rear_cands = [
                n
                for n in names
                if not _is_ignored_bone(n)
                and _extract_side(n) == side
                and (
                    any(
                        k in n.lower()
                        for k in (
                            "rearleg", "rearfoot", "rearhoof", "rearankle", "rearhip",
                            "backleg", "backfoot", "backhip", "hindleg",
                            "_back_shoulder", "_back_knee", "_back_ankle", "_back_foot",
                        )
                    )
                    or n.lower().startswith((f"{s_pref}bleg", f"{s_pref}b_leg"))
                )
            ]
            rchain = _longest_parent_chain(rear_cands, parent_map)
            if len(rchain) > 4 and any("hoof_sjnt_0" in b.lower() for b in rchain):
                idx_h = next(i for i, b in enumerate(rchain) if "hoof_sjnt_0" in b.lower())
                rchain = rchain[: idx_h + 1]
            while len(rchain) > 5 and "shoulder" in rchain[0].lower() and "shoulder" in rchain[1].lower():
                rchain = rchain[1:]
            entry_r = _build_leg_entry(f"RearLeg_{side}", side, rchain)
            if entry_r:
                legs.append(entry_r)
    else:
        for side in ("L", "R"):
            leg_cands = [
                n
                for n in names
                if not _is_ignored_bone(n)
                and _extract_side(n) == side
                and any(
                    k in n.lower()
                    for k in (
                        "_leg_", "legjoint", "leg_sjnt", "upleg", "dwleg", "ftleg",
                        "_hip", "_knee", "_calf", "_shin", "_thigh", "_ankle", "_foot_",
                    )
                )
                and "footfinger" not in n.lower()
                and "ball" not in n.lower()
                and "anklewing" not in n.lower()
                and "leghook" not in n.lower()
            ]
            lchain = _longest_parent_chain(leg_cands, parent_map)
            if len(lchain) >= 4 and any("foot_sjnt_0" in b.lower() for b in lchain):
                idx_f0 = next(i for i, b in enumerate(lchain) if "foot_sjnt_0" in b.lower())
                lchain = lchain[: idx_f0 + 1]
            entry = _build_leg_entry(f"Leg_{side}", side, lchain, allow_flipper=True)
            if entry:
                legs.append(entry)

    # 7. Wings (ALL Flyers including Dreadwing bat-wings!) & Arms (Humanoid, Theropod, Shell-Walker, Flippers, Grounded Quadrupeds)
    arms: list[dict[str, Any]] = []
    wings: list[dict[str, Any]] = []
    used_leg_bones = {
        b
        for lg in legs
        for b in (
            ([lg["shoulder"]] if lg.get("shoulder") else [])
            + lg["ik_bones"]
            + [lg["foot_bone"]]
            + list(lg.get("extra_fk_bones") or ())
        )
    }

    _WING_EXCL = ("wingrib", "wingknife", "wingplate", "anklewing", "rearwing", "reartopwing", "sidewing", "fronttopwing", "underwing", "winghandhinge", "wingtip")
    has_wing_ribs = any("wingrib" in n.lower() for n in names if not _is_ignored_bone(n))

    for side in ("L", "R"):
        wing_cands = [
            n
            for n in names
            if not _is_ignored_bone(n)
            and n not in used_leg_bones
            and _extract_side(n) == side
            and "wing" in n.lower()
            and not any(ex in n.lower() for ex in _WING_EXCL)
        ]
        wchain = _longest_parent_chain(wing_cands, parent_map)

        # Dreadwing bat-wings: main wing arm is `L/R_arm_sjnt_0..4` carrying `L/R_wingRib0..4`!
        is_bat_wing = False
        if len(wchain) < 3 and has_wing_ribs:
            bat_cands = [
                n
                for n in names
                if not _is_ignored_bone(n)
                and n not in used_leg_bones
                and _extract_side(n) == side
                and any(k in n.lower() for k in ("_arm_sjnt", "_arm_"))
                and "handhinge" not in n.lower()
            ]
            bchain = _longest_parent_chain(bat_cands, parent_map)
            if len(bchain) >= 3:
                wchain = bchain
                is_bat_wing = True

        if len(wchain) >= 2:
            w_clean: list[str] = []
            for b in wchain:
                if w_clean:
                    p0, p1 = head_map[w_clean[-1]], head_map[b]
                    if math.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2 + (p1[2] - p0[2]) ** 2) < 0.015:
                        continue
                w_clean.append(b)
            if len(w_clean) >= 3:
                tip_idx = min(3, len(w_clean) - 1)
                wings.append(
                    {
                        "name": f"Wing_{side}",
                        "side": side,
                        "bones": w_clean,
                        "ik_bones": w_clean[:tip_idx],
                        "tip_bone": w_clean[tip_idx],
                        "extra_fk_bones": w_clean[tip_idx + 1 :],
                        "is_bat_wing": is_bat_wing,
                    }
                )
            elif len(w_clean) == 2:
                wings.append(
                    {
                        "name": f"Wing_{side}",
                        "side": side,
                        "bones": w_clean,
                        "ik_bones": [],
                        "tip_bone": w_clean[-1],
                        "extra_fk_bones": [],
                        "is_bat_wing": is_bat_wing,
                    }
                )

        wing_used_side = {b for w in wings if w["side"] == side for b in w["bones"]}
        has_elbow_side = any(
            not _is_ignored_bone(n) and _extract_side(n) == side and "elbow" in n.lower()
            for n in names
        )
        arm_tokens = ("_arm_", "arm_sjnt", "armjoint", "upperarm", "forearm", "elbow", "wrist", "hand")
        if has_elbow_side:
            arm_tokens = arm_tokens + ("shoulder",)
        arm_cands = [
            n
            for n in names
            if not _is_ignored_bone(n)
            and n not in used_leg_bones
            and n not in wing_used_side
            and _extract_side(n) == side
            and any(k in n.lower() for k in arm_tokens)
            and "armor" not in n.lower()
            and "palm" not in n.lower()
            and "thumb" not in n.lower()
            and "finger" not in n.lower()
            and "handhinge" not in n.lower()
            and "armhook" not in n.lower()
            and "shoulderassist" not in n.lower()
            and "shouldergun" not in n.lower()
        ]
        canon_arm = [
            by_lower.get(f"{side.lower()}_arm_sjnt_0"),
            by_lower.get(f"{side.lower()}_arm_sjnt_1"),
            by_lower.get(f"{side.lower()}_wrist_sjnt_0"),
        ]
        if all(canon_arm) and not any(b in wing_used_side for b in canon_arm if b):
            achain = [b for b in canon_arm if b]
        else:
            achain = _longest_parent_chain(arm_cands, parent_map)
        clav = next(
            (
                n
                for n in names
                if not _is_ignored_bone(n)
                and n not in used_leg_bones
                and n not in wing_used_side
                and _extract_side(n) == side
                and ("clavicle" in n.lower() or "collar" in n.lower())
            ),
            None,
        )
        aclean: list[str] = []
        for b in achain:
            if aclean:
                p0, p1 = head_map[aclean[-1]], head_map[b]
                if math.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2 + (p1[2] - p0[2]) ** 2) < 0.015:
                    continue
            aclean.append(b)
        if len(aclean) >= 3:
            full_arm_clean = list(aclean)
            is_flipper = len(full_arm_clean) >= 7
            extra_arm_fk: list[str] = []
            if is_flipper:
                ik_b = full_arm_clean[:3]
                hand_b = full_arm_clean[3]
                extra_arm_fk = full_arm_clean[4:]
            else:
                if len(aclean) > 3 and any("wrist" in b.lower() for b in aclean):
                    wi = next(i for i, b in enumerate(aclean) if "wrist" in b.lower())
                    if wi >= 2:
                        aclean = aclean[wi - 2 : wi + 1]
                elif len(aclean) > 4:
                    aclean = aclean[:4]
                ik_b = aclean[:-1]
                hand_b = aclean[-1]
            arms.append(
                {
                    "name": f"Arm_{side}",
                    "side": side,
                    "clavicle": clav,
                    "ik_bones": ik_b,
                    "hand_bone": hand_b,
                    "extra_fk_bones": extra_arm_fk,
                    "is_flipper": is_flipper,
                    "full_chain": full_arm_clean,
                    "chain_length": len(ik_b),
                }
            )

    # 7a. Promote Grounded Quadruped Front Legs (New HFW quadrupeds like Tremortusk, Plowhorn, Shellsnapper, Widemaw,
    #     Bilegut, Clamberjaw name their front weight-bearing legs `L/R_arm_sjnt_0..5` reaching all the way to ground!)
    has_any_flipper = any(lg.get("is_flipper") for lg in legs) or any(ar.get("is_flipper") for ar in arms)
    if len(legs) == 2 and len(arms) == 2 and not wings and not has_any_flipper:
        min_leg_z = min(head_map[lg["foot_bone"]][2] for lg in legs)
        min_arm_z = min(min(head_map[b][2] for b in ar["full_chain"]) for ar in arms)
        # Ensure we NEVER mistake an upright humanoid (head directly above hips) for a quadruped
        is_upright_humanoid = False
        if head_bone and hips_bone and head_bone in head_map and hips_bone in head_map:
            dz_hh = abs(head_map[head_bone][2] - head_map[hips_bone][2])
            dxy_hh = math.hypot(head_map[head_bone][0] - head_map[hips_bone][0], head_map[head_bone][1] - head_map[hips_bone][1])
            if dz_hh > max(0.35, dxy_hh * 1.6):
                is_upright_humanoid = True
        if not is_upright_humanoid and min_arm_z <= min_z + rig_height * 0.19 and min_arm_z <= min_leg_z + rig_height * 0.16:
            promoted_legs: list[dict[str, Any]] = []
            for ar in arms:
                side = ar["side"]
                fc = list(ar["full_chain"])
                sh_b = ar.get("clavicle")
                # Pick 4-bone ground-contact front leg from full_chain (preserving leading shoulder if >= 5 bones!)
                if len(fc) >= 5:
                    if sh_b is None:
                        sh_b = fc[0]
                    fc = fc[-4:]
                entry_fl = _build_leg_entry(f"FrontLeg_{side}", side, fc)
                if entry_fl:
                    if sh_b and sh_b not in entry_fl["ik_bones"] and sh_b != entry_fl["foot_bone"]:
                        entry_fl["shoulder"] = sh_b
                    promoted_legs.append(entry_fl)
            if len(promoted_legs) == 2:
                for lg in legs:
                    lg["name"] = f"RearLeg_{lg['side']}"
                legs = promoted_legs + legs
                arms = []

    # 7b. Dedicated Chest Anchor Detection (explicit chest, arm/wing/shoulder upper-torso anchor, or chassis/neck-base fallback!)
    def _spine_ancestor(bname: str | None) -> str | None:
        cur = bname
        seen_a: set[str] = set()
        spine_set = set(spine_chain)
        while cur and cur not in seen_a:
            seen_a.add(cur)
            if cur in spine_set:
                return cur
            cur = parent_map.get(cur)
        return None

    chest_bone: str | None = None
    # Priority 1: explicit chest bone already in spine_chain (e.g. Behemoth `C_Chest_sjnt_0`, Charger `chestJoint`)
    explicit_chests = [b for b in spine_chain if "chest" in b.lower() and "_fjnt" not in b.lower()]
    if explicit_chests:
        chest_bone = explicit_chests[-1]

    # Priority 2: spine bone anchoring arms, clavicles, wings, or quadruped front-leg shoulders
    if chest_bone is None and spine_chain:
        upper_limb_roots: list[str] = []
        for ar in arms:
            upper_limb_roots.append(ar["clavicle"] or ar["ik_bones"][0])
        for wg in wings:
            upper_limb_roots.append(wg["bones"][0])
        for lg in legs:
            if lg["name"].startswith("FrontLeg_"):
                upper_limb_roots.append(lg.get("shoulder") or lg["ik_bones"][0])
        limb_spine_hits = [a for r in upper_limb_roots if (a := _spine_ancestor(r)) is not None]
        if limb_spine_hits:
            chest_bone = min(limb_spine_hits, key=lambda b: spine_chain.index(b))

    # Priority 3: spine ancestor of neck_chain[0] (or head_bone)
    if chest_bone is None and spine_chain:
        nk_anc = _spine_ancestor(neck_chain[0] if neck_chain else head_bone)
        if nk_anc is not None:
            chest_bone = nk_anc

    # Priority 4: central non-hips parent of neck_chain[0] (e.g. Grazer `frontBodyJoint1`) or upper body chassis (`C_Body_0_jnt`)
    if chest_bone is None and neck_chain:
        np0 = parent_map.get(neck_chain[0])
        if (
            np0
            and np0 not in (root_bone, ground_bone, hips_bone)
            and not _is_ignored_bone(np0)
            and "_fjnt" not in np0.lower()
            and _extract_side(np0) == "C"
        ):
            chest_bone = np0
    if chest_bone is None and not spine_chain:
        body_jnt = by_lower.get("c_body_0_jnt") or by_lower.get("c_body_sjnt_0")
        if body_jnt and body_jnt != hips_bone:
            chest_bone = body_jnt

    # Priority 5: promote neck_chain[0] when spine is empty (Thunderjaw `neckBase`) or neck attaches directly to hips (Slitherfang/Rockbreaker/Snapmaw)
    if chest_bone is None:
        if len(neck_chain) >= 2 and (not spine_chain or parent_map.get(neck_chain[0]) == hips_bone):
            chest_bone = neck_chain[0]
            neck_chain = neck_chain[1:]
        elif spine_chain:
            chest_bone = spine_chain[-1]

    # Keep spine_chain strictly BELOW chest_bone; any spine bone above a wing/arm-anchor chest (e.g. Sunwing `C_spine_sjnt_6`)
    # joins the base of neck_chain so CTRL_Chest cleanly drives BOTH wings/front legs AND upper neck/head with zero duplicate controls!
    if chest_bone is not None and chest_bone in spine_chain:
        c_idx = spine_chain.index(chest_bone)
        above_chest = [b for b in spine_chain[c_idx + 1 :] if b != head_bone and b not in neck_chain]
        spine_chain = spine_chain[:c_idx]
        if above_chest:
            neck_chain = above_chest + neck_chain
    if chest_bone is not None and chest_bone in neck_chain:
        neck_chain = [b for b in neck_chain if b != chest_bone]

    # Multi-bone Torso/Chest IK is supported whenever chest_bone sits directly at the top of spine_chain (`spine_0..k -> chest`)!
    # Serpentine rigs (Slitherfang) instead support Cobra Hood IK across `chest_bone -> neck_0..N` (`torso_ik_mode = "hood"`)!
    spine_torso_ik = bool(spine_chain and chest_bone and parent_map.get(chest_bone) == spine_chain[-1])
    hood_torso_ik = bool(
        not spine_torso_ik
        and len(legs) == 0
        and len(spine_chain) >= 8
        and chest_bone
        and len(neck_chain) >= 4
        and parent_map.get(neck_chain[0]) == chest_bone
        and all(parent_map.get(neck_chain[i]) == neck_chain[i - 1] for i in range(1, len(neck_chain)))
    )
    torso_ik_mode = "spine" if spine_torso_ik else ("hood" if hood_torso_ik else "none")
    torso_ik_supported = torso_ik_mode != "none"

    # 8. Smaller Articulated FK Chains: Hand Fingers/Claws, Wing Ribs/Fingers, Foot Toes/Talons/Hooves, Secondary Machine Details!
    primary_used: set[str] = set()
    for b in (root_bone, ground_bone, hips_bone, chest_bone, head_bone):
        if b:
            primary_used.add(b)
    primary_used.update(spine_chain)
    primary_used.update(neck_chain)
    primary_used.update(jaw_bones)
    for t in tails:
        primary_used.update(t["bones"])
    for app in appendages:
        primary_used.update(app["bones"])
    for w in wings:
        primary_used.update(w["bones"])
    for lg in legs:
        if lg.get("shoulder"):
            primary_used.add(lg["shoulder"])
        primary_used.update(lg["ik_bones"])
        primary_used.add(lg["foot_bone"])
        primary_used.update(lg.get("extra_fk_bones") or ())
    for ar in arms:
        if ar.get("clavicle"):
            primary_used.add(ar["clavicle"])
        primary_used.update(ar["ik_bones"])
        primary_used.add(ar["hand_bone"])
        primary_used.update(ar.get("extra_fk_bones") or ())

    # 8a. Hand Fingers, Forearm Claws & Flyer Wing Fingers / Bat Wing Ribs
    fingers: list[dict[str, Any]] = []
    for ar in arms:
        hb = ar["hand_bone"]
        side = ar["side"]
        direct_kids = [k for k in children_map.get(hb, []) if k not in primary_used and not _is_detail_forbidden(k)]
        sub_chains = _collect_sub_chains(direct_kids, children_map, head_map, primary_used, max_depth=4)
        for ci, ch in enumerate(sub_chains[:10]):
            stem = re.sub(r"^[LlRrCc]_", "", ch[0])
            stem = re.sub(r"_?sjnt_?\d+$", "", stem, flags=re.IGNORECASE)
            stem = re.sub(r"[^A-Za-z0-9]+", "", stem)[:14] or f"Finger{ci}"
            fingers.append({"name": f"{stem}_{side}", "side": side, "parent_def": hb, "bones": ch})
            primary_used.update(ch)

    # Flyer wing fingers & bat wing ribs hanging off wing joints (`wingHandHinge -> handFinger0..2`, `wingRib0..6`)
    for wg in wings:
        side = wg["side"]
        for wb in wg["bones"]:
            w_kids = [
                k
                for k in children_map.get(wb, [])
                if k not in primary_used
                and not _is_detail_forbidden(k)
                and any(tk in k.lower() for tk in ("wingrib", "gliderrib", "winghandhinge", "armhandhinge", "handfinger", "wingfinger", "claw", "thumb"))
            ]
            for ci, ch in enumerate(_collect_sub_chains(w_kids, children_map, head_map, primary_used, max_depth=5)):
                stem = re.sub(r"^[LlRrCc]_", "", ch[0])
                stem = re.sub(r"_?sjnt_?\d+$", "", stem, flags=re.IGNORECASE)
                stem = re.sub(r"[^A-Za-z0-9]+", "", stem)[:14] or f"WingRib{ci}"
                fingers.append({"name": f"{stem}_{side}", "side": side, "parent_def": parent_map.get(ch[0]) or wb, "bones": ch})
                primary_used.update(ch)

    # 8b. Foot Toes, Talons, Dewclaws, Balls & Hooves (descendants of foot_bone + raptor dewclaw thumbs on lower leg!)
    toes: list[dict[str, Any]] = []
    for lg in legs:
        fb = lg["foot_bone"]
        side = lg["side"]
        direct_kids = [k for k in children_map.get(fb, []) if k not in primary_used and not _is_detail_forbidden(k)]
        sub_chains = _collect_sub_chains(direct_kids, children_map, head_map, primary_used, max_depth=4)
        for ci, ch in enumerate(sub_chains[:10]):
            stem = re.sub(r"^[LlRrCc]_", "", ch[0])
            stem = re.sub(r"_?sjnt_?\d+$", "", stem, flags=re.IGNORECASE)
            stem = re.sub(r"[^A-Za-z0-9]+", "", stem)[:14] or f"Toe{ci}"
            toes.append({"name": f"{stem}_{side}", "side": side, "parent_def": fb, "bones": ch})
            primary_used.update(ch)
        for lb in lg["ik_bones"]:
            dew_kids = [
                k
                for k in children_map.get(lb, [])
                if k not in primary_used
                and not _is_detail_forbidden(k)
                and any(tk in k.lower() for tk in ("thumb", "toe", "claw", "talon", "spur", "dewclaw", "wingrib"))
            ]
            for ci, ch in enumerate(_collect_sub_chains(dew_kids, children_map, head_map, primary_used, max_depth=4)):
                stem = re.sub(r"^[LlRrCc]_", "", ch[0])
                stem = re.sub(r"_?sjnt_?\d+$", "", stem, flags=re.IGNORECASE)
                stem = re.sub(r"[^A-Za-z0-9]+", "", stem)[:14] or f"Dewclaw{ci}"
                toes.append({"name": f"{stem}_{side}", "side": side, "parent_def": lb, "bones": ch})
                primary_used.update(ch)

    # 8c. Dedicated Looping Chainsaws (`Clawstrider` 26-tooth jaw saws, `Charger/Lancehorn` horn saws) & Rotary Drills/Grinders/Rotors!
    chainsaws: list[dict[str, Any]] = []
    saw_blade_groups: dict[str, list[str]] = {}
    for n in names:
        low = n.lower()
        if n in primary_used or _is_detail_forbidden(n):
            continue
        if "sawblade" in low or "chainsaw" in low:
            p = parent_map.get(n)
            if p and p in head_map:
                saw_blade_groups.setdefault(p, []).append(n)
    for par_b, bl_list in saw_blade_groups.items():
        if len(bl_list) >= 4:
            # Sort cyclically by numeric index if numbered (`sawBlade0..25`), else angularly around their centroid
            if all(re.search(r"\d+", bn) for bn in bl_list):
                bl_sorted = sorted(bl_list, key=_numeric_sort_key)
            else:
                pts_s = np.array([head_map[b] for b in bl_list], dtype=np.float64)
                ctr = pts_s.mean(axis=0)
                rel = pts_s - ctr
                _, _, vh = np.linalg.svd(rel, full_matrices=False)
                ang = np.arctan2(rel @ vh[1], rel @ vh[0])
                bl_sorted = [bl_list[i] for i in np.argsort(ang)]
            side = _extract_side(par_b)
            stem = re.sub(r"^[LlRrCc]_", "", par_b)
            stem = re.sub(r"_?s?jnt_?\d*$", "", stem, flags=re.IGNORECASE)
            stem = re.sub(r"[^A-Za-z0-9]+", "", stem)[:14] or "Saw"
            chainsaws.append(
                {
                    "name": f"{stem}_{side}" if side != "C" else stem,
                    "side": side,
                    "parent_def": par_b,
                    "blades": bl_sorted,
                }
            )
            primary_used.update(bl_sorted)

    _SPINNER_KEYWORDS = (
        "jawdrill",
        "headdrill",
        "drillbit",
        "_drill_",
        "drill_sjnt",
        "drill_jnt",
        "bodygrinder",
        "slotmachinegrinder",
        "_grinder_",
        "grinder_sjnt",
        "grinder_jnt",
        "legrotor",
        "longhornrotor",
        "rotor_jnt",
        "rotor_sjnt",
        "turbine_sjnt",
        "turbine_jnt",
        "tonguedrum",
        "drumturret",
    )
    _SPINNER_EXCL = ("plate", "clip", "helper", "muzzle", "eject", "danger", "scrape", "arm")
    spinners: list[dict[str, Any]] = []
    for n in names:
        low = n.lower()
        if n in primary_used or _is_detail_forbidden(n):
            continue
        if any(ex in low for ex in _SPINNER_EXCL):
            continue
        if any(kw in low for kw in _SPINNER_KEYWORDS):
            side = _extract_side(n)
            stem = re.sub(r"^[LlRrCc]_", "", n)
            stem = re.sub(r"_?s?jnt_?\d*$", "", stem, flags=re.IGNORECASE)
            stem = re.sub(r"[^A-Za-z0-9]+", "", stem)[:18] or "Spin"
            kind = (
                "grinder"
                if "grinder" in low
                else ("rotor" if ("rotor" in low or "turbine" in low or "drum" in low) else "drill")
            )
            spinners.append(
                {
                    "name": f"{stem}_{side}",
                    "kind": kind,
                    "side": side,
                    "parent_def": parent_map.get(n),
                    "bone": n,
                }
            )
            primary_used.add(n)

    # 8d. Categorized Machine Secondary Controls
    _FRILL_KEYWORDS = (
        "feather",
        "frill",
        "headblade",
        "topblade",
        "midblade",
        "neckblade",
        "upperblade",
        "lowerblade",
        "tailleftendblade",
        "tailrightendblade",
        "tailleftblade",
        "tailrightblade",
        "tailblade",
        "wingknife",
        "rearwing",
        "reartopwing",
        "fronttopwing",
        "sidewing",
        "anklewing",
        "frontguide",
        "guidespike",
        "plasmaspike",
        "spinespike",
        "spike",
        "fin",
        "crest",
        "headfan",
        "fan",
        "head_flap",
        "headflap",
        "flap",
        "antenna",
        "antena",
        "extender",
        "exhaust",
        "engine",
        "wingtip",
        "gill",
        "plume",
        "mane",
        "ribbit",
        "_ear_",
        "eartip",
    )
    _PLATE_KEYWORDS = (
        "plate",
        "shield",
        "shell",
        "hinge",
        "canister",
        "cannister",
        "clip",
        "clamp",
        "door",
        "bellyopener",
        "bellyflap",
        "bellycompress",
        "armor",
        "hatch",
        "cover",
        "lid",
        "cargo",
        "pod",
        "sac",
        "sack",
        "tank",
        "core",
        "cell",
        "battery",
        "vent",
        "hipbase",
        "chestrod",
    )
    _WEAPON_KEYWORDS = (
        "weapon",
        "gun",
        "cannon",
        "turret",
        "blaster",
        "launcher",
        "bracket",
        "beltgun",
        "beltjoint",
        "horn",
        "hammer",
        "drill",
        "grinder",
        "nozzle",
        "saw_sjnt",
        "mandible",
        "tusk",
        "pincer",
        "stinger",
        "laser",
        "mortar",
        "radar",
        "scanner",
        "jaw",
        "tailclaw",
    )

    def _classify_secondary_bone(bname: str) -> str | None:
        low = bname.lower()
        if "sawblade" in low:
            return None
        if any(kw in low for kw in _WEAPON_KEYWORDS):
            return "weapons"
        if any(kw in low for kw in _FRILL_KEYWORDS):
            return "frills"
        if any(kw in low for kw in _PLATE_KEYWORDS):
            return "plates"
        return None

    frills: list[dict[str, Any]] = []
    plates: list[dict[str, Any]] = []
    weapons: list[dict[str, Any]] = []
    cat_lists = {"frills": frills, "plates": plates, "weapons": weapons}
    cat_counters = {"frills": 0, "plates": 0, "weapons": 0}

    for _pass in range(3):
        pass_roots: list[tuple[str, str]] = []
        for n in names:
            if n in primary_used or _is_detail_forbidden(n):
                continue
            p = parent_map.get(n)
            if p not in primary_used:
                continue
            cat = _classify_secondary_bone(n)
            if cat is not None:
                pass_roots.append((n, cat))
        if not pass_roots:
            break
        for rname, rcat in pass_roots:
            if rname in primary_used or cat_counters[rcat] >= 120:
                continue
            ch: list[str] = []
            cur: str | None = rname
            while cur and len(ch) < 6:
                if cur in primary_used or _is_detail_forbidden(cur):
                    break
                ch.append(cur)
                primary_used.add(cur)
                kids = [
                    k
                    for k in children_map.get(cur, [])
                    if k not in primary_used and not _is_detail_forbidden(k)
                ]
                if len(kids) == 1:
                    kcat = _classify_secondary_bone(kids[0])
                    if kcat in (None, rcat) and ("spike" not in kids[0].lower() or "spike" in cur.lower()):
                        cur = kids[0]
                    else:
                        break
                else:
                    break
            if ch:
                ci = cat_counters[rcat]
                cat_counters[rcat] += 1
                side = _extract_side(ch[0])
                stem = re.sub(r"^[LlRrCc]_", "", ch[0])
                stem = re.sub(r"_?sjnt_?\d+$", "", stem, flags=re.IGNORECASE)
                stem = re.sub(r"[^A-Za-z0-9]+", "", stem)[:18] or f"{rcat[:-1].title()}{ci}"
                cat_lists[rcat].append(
                    {
                        "name": f"{stem}_{side}_{ci:02d}",
                        "category": rcat,
                        "side": side,
                        "parent_def": parent_map.get(ch[0]),
                        "bones": ch,
                    }
                )

    details: list[dict[str, Any]] = frills + plates + weapons

    # 9. Universal Archetype Classification & Forward Axis
    # Only mark `aquatic` when BOTH legs and arms are long multi-bone flippers (e.g. Tideripper),
    # so bipedal Specter / Specter Prime keep `theropod` while still getting full extra FK tendril/arm segments!
    is_true_aquatic = bool(
        legs and arms and all(lg.get("is_flipper") for lg in legs) and all(ar.get("is_flipper") for ar in arms)
    )
    # Distinguish real flying machines (wingspan >= 0.85 * height or >= 4 wing bones) from flightless Longleg (tiny flare wings!)
    wing_span = 0.0
    if len(wings) >= 2:
        p_wl = head_map.get(wings[0]["bones"][-1], (0.0, 0.0, 0.0))
        p_wr = head_map.get(wings[1]["bones"][-1], (0.0, 0.0, 0.0))
        wing_span = math.sqrt(sum((p_wl[ax] - p_wr[ax]) ** 2 for ax in range(3)))
    is_true_flyer = bool(wings) and (wing_span >= rig_height * 0.85 or any(len(w["bones"]) >= 4 for w in wings) or len(legs) == 0)

    if len(legs) >= 6:
        archetype = "hexapod"
    elif is_true_aquatic:
        archetype = "aquatic"
    elif len(legs) >= 4:
        archetype = "quadruped"
    elif is_true_flyer:
        archetype = "flyer"
    elif len(legs) == 0 and len(spine_chain) >= 8:
        archetype = "serpentine"
    elif len(legs) == 2 and any(lg["is_digitigrade"] for lg in legs):
        archetype = "theropod"
    elif len(legs) == 2:
        archetype = "humanoid"
    else:
        archetype = "creature"

    fwd_vec = (0.0, 1.0, 0.0)
    if head_bone and hips_bone and head_bone in head_map and hips_bone in head_map:
        dh = (head_map[head_bone][0] - head_map[hips_bone][0], head_map[head_bone][1] - head_map[hips_bone][1], 0.0)
        hl = math.hypot(dh[0], dh[1])
        if hl > rig_height * 0.12:
            fwd_vec = (dh[0] / hl, dh[1] / hl, 0.0)
        elif legs:
            for n in names:
                nl = n.lower()
                if "foot_sjnt_1" in nl or "toe" in nl:
                    p = parent_map.get(n)
                    if p and p in head_map and n in head_map:
                        dt = (head_map[n][0] - head_map[p][0], head_map[n][1] - head_map[p][1], 0.0)
                        tl = math.hypot(dt[0], dt[1])
                        if tl > 0.01:
                            fwd_vec = (dt[0] / tl, dt[1] / tl, 0.0)
                            break

    # 9b. Toggleable Head/Neck IK & Spline IK Chain Detection
    head_ik_supported = bool(
        head_bone
        and len(neck_chain) >= 1
        and torso_ik_mode != "hood"
        and parent_map.get(head_bone) == neck_chain[-1]
        and all(parent_map.get(neck_chain[i]) == neck_chain[i - 1] for i in range(1, len(neck_chain)))
    )

    spline_chains: list[dict[str, Any]] = []
    if archetype == "serpentine" and len(spine_chain) >= 8:
        spline_chains.append(
            {
                "name": "SerpentBody",
                "kind": "spine",
                "group_prop": "hfw_rig_spine",
                "bones": list(spine_chain),
                "default_on": True,
                "knots": min(7, max(4, len(spine_chain) // 6 + 2)),
            }
        )
    else:
        for tinfo in tails:
            if len(tinfo["bones"]) >= 7:
                spline_chains.append(
                    {
                        "name": tinfo["name"],
                        "kind": "tail",
                        "group_prop": "hfw_rig_tail",
                        "bones": list(tinfo["bones"]),
                        "default_on": False,
                        "knots": min(6, max(4, len(tinfo["bones"]) // 3 + 1)),
                    }
                )
        for app in appendages:
            if len(app["bones"]) >= 7:
                spline_chains.append(
                    {
                        "name": app["name"],
                        "kind": "appendage",
                        "group_prop": "hfw_rig_tail",
                        "bones": list(app["bones"]),
                        "default_on": False,
                        "knots": min(6, max(4, len(app["bones"]) // 3 + 1)),
                    }
                )

    return {
        "archetype": archetype,
        "rig_height": round(rig_height, 4),
        "forward": [round(fwd_vec[0], 4), round(fwd_vec[1], 4), 0.0],
        "root": root_bone,
        "ground": ground_bone,
        "hips": hips_bone,
        "spine": spine_chain,
        "chest": chest_bone,
        "torso_ik_supported": torso_ik_supported,
        "torso_ik_mode": torso_ik_mode,
        "head_ik_supported": head_ik_supported,
        "spline_chains": spline_chains,
        "neck": neck_chain,
        "head": head_bone,
        "jaws": jaw_bones,
        "tails": tails,
        "appendages": appendages,
        "legs": legs,
        "arms": arms,
        "wings": wings,
        "fingers": fingers,
        "toes": toes,
        "chainsaws": chainsaws,
        "spinners": spinners,
        "frills": frills,
        "plates": plates,
        "weapons": weapons,
        "details": details,
    }


def analyze_armature_rig_spec(arm_obj: Any) -> dict[str, Any]:
    """Analyze a Blender Armature object's deform skeleton into a Control Rig spec."""
    specs = []
    for b in arm_obj.data.bones:
        if b.name.startswith(("CTRL_", "MCH_", "WGT_")):
            continue
        specs.append(
            {
                "name": b.name,
                "parent_name": b.parent.name if b.parent else None,
                "head": (float(b.head_local.x), float(b.head_local.y), float(b.head_local.z)),
            }
        )
    return analyze_rig_chains(specs)


# ---------------------------------------------------------------------------
# Blender Custom Wireframe Widget Builders (Zero-Render Control Shapes)
# ---------------------------------------------------------------------------
def _get_or_create_widget_collection() -> Any:
    wcoll = bpy.data.collections.get("HFW_Rig_Widgets")
    if wcoll is None:
        wcoll = bpy.data.collections.new("HFW_Rig_Widgets")
        bpy.context.scene.collection.children.link(wcoll)
    wcoll.hide_viewport = True
    wcoll.hide_render = True
    return wcoll


def _make_wire_widget(name: str, verts: list[tuple[float, float, float]], edges: list[tuple[int, int]]) -> Any:
    existing = bpy.data.objects.get(name)
    if existing is not None:
        return existing
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, edges, [])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    _get_or_create_widget_collection().objects.link(obj)
    obj.hide_viewport = True
    obj.hide_render = True
    return obj


def _ensure_control_widgets() -> dict[str, Any]:
    segs = 28
    # Bold double-ring FK controller with 4 radial tick marks (super visible around thick machine limbs & tails!)
    fk_verts: list[tuple[float, float, float]] = []
    fk_edges: list[tuple[int, int]] = []
    for rad in (0.90, 1.06):
        off = len(fk_verts)
        for i in range(segs):
            a = 2.0 * math.pi * i / segs
            fk_verts.append((rad * math.cos(a), 0.0, rad * math.sin(a)))
            fk_edges.append((off + i, off + (i + 1) % segs))
    for i in range(0, segs, segs // 4):
        fk_edges.append((i, segs + i))
    # Subtle forward indicator notch on +Z so rotation orientation reads clearly
    tbase = len(fk_verts)
    fk_verts.extend([(0.0, 0.0, 1.32), (-0.16, 0.0, 1.06), (0.16, 0.0, 1.06)])
    fk_edges.extend([(tbase, tbase + 1), (tbase, tbase + 2)])
    w_fk_ring = _make_wire_widget("WGT_HFW_FK_Ring", fk_verts, fk_edges)

    # Root ring on XY ground plane with bold forward arrow
    r_verts = [(math.cos(2.0 * math.pi * i / segs), math.sin(2.0 * math.pi * i / segs), 0.0) for i in range(segs)]
    r_edges = [(i, (i + 1) % segs) for i in range(segs)]
    base = len(r_verts)
    r_verts.extend([(0.0, 1.50, 0.0), (-0.32, 1.05, 0.0), (0.32, 1.05, 0.0)])
    r_edges.extend([(base, base + 1), (base, base + 2), (base + 1, base + 2)])
    w_root = _make_wire_widget("WGT_HFW_Root", r_verts, r_edges)

    # COG double hoop + inner diamond (unmistakable center-of-gravity grab handle!)
    cog_verts = [(math.cos(2.0 * math.pi * i / segs), math.sin(2.0 * math.pi * i / segs), 0.0) for i in range(segs)]
    cog_edges = [(i, (i + 1) % segs) for i in range(segs)]
    cbase = len(cog_verts)
    cog_verts.extend([(0.0, 1.28, 0.0), (1.28, 0.0, 0.0), (0.0, -1.28, 0.0), (-1.28, 0.0, 0.0)])
    cog_edges.extend([(cbase, cbase + 1), (cbase + 1, cbase + 2), (cbase + 2, cbase + 3), (cbase + 3, cbase)])
    w_cog = _make_wire_widget("WGT_HFW_COG", cog_verts, cog_edges)

    # Bold Foot / Hand / Head IK Cube
    b_verts = [
        (-0.55, -0.55, -0.38), (0.55, -0.55, -0.38), (0.55, 0.55, -0.38), (-0.55, 0.55, -0.38),
        (-0.55, -0.55, 0.38), (0.55, -0.55, 0.38), (0.55, 0.55, 0.38), (-0.55, 0.55, 0.38),
    ]
    b_edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    w_box = _make_wire_widget("WGT_HFW_Box", b_verts, b_edges)

    # Knee / Elbow Pole Target Diamond
    d_verts = [
        (0.0, 0.0, 1.0), (0.0, 0.0, -1.0),
        (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
    ]
    d_edges = [
        (0, 2), (0, 3), (0, 4), (0, 5),
        (1, 2), (1, 3), (1, 4), (1, 5),
        (2, 4), (4, 3), (3, 5), (5, 2),
    ]
    w_diamond = _make_wire_widget("WGT_HFW_Diamond", d_verts, d_edges)

    # 3D Dual-Ring Lollipop Pin for Fingers, Toes, Talons & Hooves (visible from EVERY camera angle!)
    p_verts = [(0.0, 0.0, 0.0), (0.0, 0.85, 0.0)]
    p_edges = [(0, 1)]
    p1 = len(p_verts)
    for i in range(12):
        a = 2.0 * math.pi * i / 12
        p_verts.append((0.32 * math.cos(a), 0.85 + 0.32 * math.sin(a), 0.0))
        p_edges.append((p1 + i, p1 + (i + 1) % 12))
    p2 = len(p_verts)
    for i in range(12):
        a = 2.0 * math.pi * i / 12
        p_verts.append((0.0, 0.85 + 0.32 * math.sin(a), 0.32 * math.cos(a)))
        p_edges.append((p2 + i, p2 + (i + 1) % 12))
    w_pin = _make_wire_widget("WGT_HFW_Pin", p_verts, p_edges)

    # Elongated Feather / Frill / Plasma Intake Paddle extending clearly outward!
    pad_verts = [
        (0.0, 0.0, 0.0), (0.0, 1.45, 0.0),
        (-0.36, 0.55, 0.0), (0.36, 0.55, 0.0),
        (-0.26, 1.15, 0.0), (0.26, 1.15, 0.0),
        (0.0, 0.55, 0.24), (0.0, 0.55, -0.24),
    ]
    pad_edges = [
        (0, 1), (0, 2), (0, 3), (2, 4), (3, 5), (4, 1), (5, 1),
        (2, 3), (4, 5), (0, 6), (6, 1), (0, 7), (7, 1),
    ]
    w_paddle = _make_wire_widget("WGT_HFW_Paddle", pad_verts, pad_edges)

    # Beveled Armor Plate / Shield / Canister Square Frame
    pl_verts = [
        (-0.65, 0.0, -0.45), (0.65, 0.0, -0.45), (0.65, 0.0, 0.45), (-0.65, 0.0, 0.45),
        (-0.42, 0.0, -0.26), (0.42, 0.0, -0.26), (0.42, 0.0, 0.26), (-0.42, 0.0, 0.26),
        (0.0, 0.0, 0.0), (0.0, 0.45, 0.0),
    ]
    pl_edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 9),
    ]
    w_plate = _make_wire_widget("WGT_HFW_Plate", pl_verts, pl_edges)

    # Weapon / Turret / Blaster / Horn Crosshair Sight
    wp_verts = [(0.85 * math.cos(2.0 * math.pi * i / 16), 0.0, 0.85 * math.sin(2.0 * math.pi * i / 16)) for i in range(16)]
    wp_edges = [(i, (i + 1) % 16) for i in range(16)]
    wb = len(wp_verts)
    wp_verts.extend([(-1.2, 0.0, 0.0), (1.2, 0.0, 0.0), (0.0, 0.0, -1.2), (0.0, 0.0, 1.2), (0.0, 0.0, 0.0), (0.0, 1.2, 0.0)])
    wp_edges.extend([(wb, wb + 1), (wb + 2, wb + 3), (wb + 4, wb + 5)])
    w_weapon = _make_wire_widget("WGT_HFW_Weapon", wp_verts, wp_edges)

    # Contoured 3D Double-Tier Chest / Ribcage Cage (distinct from spine rings & COG!)
    ch_verts: list[tuple[float, float, float]] = []
    ch_edges: list[tuple[int, int]] = []
    ch_segs = 16
    for tier_y, rx, rz in ((-0.16, 1.05, 0.88), (0.34, 1.18, 0.96)):
        off = len(ch_verts)
        for i in range(ch_segs):
            a = 2.0 * math.pi * i / ch_segs
            ch_verts.append((rx * math.cos(a), tier_y, rz * math.sin(a)))
            ch_edges.append((off + i, off + (i + 1) % ch_segs))
    # 4 vertical rib struts linking lower and upper ribcage hoops
    for i in (0, ch_segs // 4, ch_segs // 2, (3 * ch_segs) // 4):
        ch_edges.append((i, ch_segs + i))
    # Front sternum chevron pointing clearly along +Z
    sbase = len(ch_verts)
    ch_verts.extend([(0.0, 0.09, 1.32), (-0.28, -0.16, 0.88), (0.28, -0.16, 0.88), (-0.28, 0.34, 0.96), (0.28, 0.34, 0.96)])
    ch_edges.extend([(sbase, sbase + 1), (sbase, sbase + 2), (sbase, sbase + 3), (sbase, sbase + 4)])
    w_chest = _make_wire_widget("WGT_HFW_Chest", ch_verts, ch_edges)

    # 3-Axis Spline IK Knot Gimbal Sphere (Move G + Rotate R + Scale S readable from every view!)
    sp_verts: list[tuple[float, float, float]] = []
    sp_edges: list[tuple[int, int]] = []
    sp_n = 20
    for plane in (0, 1, 2):
        off = len(sp_verts)
        for i in range(sp_n):
            a = 2.0 * math.pi * i / sp_n
            ca, sa = 1.05 * math.cos(a), 1.05 * math.sin(a)
            if plane == 0:
                sp_verts.append((ca, 0.0, sa))
            elif plane == 1:
                sp_verts.append((ca, sa, 0.0))
            else:
                sp_verts.append((0.0, ca, sa))
            sp_edges.append((off + i, off + (i + 1) % sp_n))
    sb = len(sp_verts)
    sp_verts.extend([(0.0, 1.45, 0.0), (-0.22, 1.05, 0.0), (0.22, 1.05, 0.0)])
    sp_edges.extend([(sb, sb + 1), (sb, sb + 2)])
    w_spline = _make_wire_widget("WGT_HFW_Spline", sp_verts, sp_edges)

    # Compact Visored 3D Head IK Target Cage with forward aiming chevron
    hi_verts = [
        (-0.46, -0.34, -0.34), (0.46, -0.34, -0.34), (0.46, 0.50, -0.34), (-0.46, 0.50, -0.34),
        (-0.46, -0.34, 0.34), (0.46, -0.34, 0.34), (0.46, 0.50, 0.34), (-0.46, 0.50, 0.34),
        (0.0, 0.88, 0.0), (-0.28, 0.50, 0.0), (0.28, 0.50, 0.0),
    ]
    hi_edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
        (8, 9), (8, 10), (8, 2), (8, 3), (8, 6), (8, 7),
    ]
    w_head_ik = _make_wire_widget("WGT_HFW_HeadIK", hi_verts, hi_edges)

    # Sleek Master Finger / Toe / Talon Curl Lever (Rigify super-finger inspired, never clutters hands/feet!)
    cl_verts = [
        (0.0, 0.0, 0.0), (0.0, 0.96, 0.0),
        (-0.075, 0.10, 0.0), (0.075, 0.10, 0.0),
        (-0.055, 0.82, 0.0), (0.055, 0.82, 0.0),
        (0.0, 0.82, 0.075), (0.0, 0.82, -0.075),
        (0.0, 1.04, 0.0),
    ]
    cl_edges = [
        (0, 2), (0, 3), (2, 4), (3, 5), (4, 8), (5, 8),
        (0, 1), (6, 8), (7, 8), (4, 6), (6, 5), (5, 7), (7, 4),
    ]
    w_curl = _make_wire_widget("WGT_HFW_Curl", cl_verts, cl_edges)

    # Crisp 8-Tooth Mechanical Spin Gear Wheel for Chainsaws, Drills, Grinders & Rotors!
    gr_verts: list[tuple[float, float, float]] = []
    gr_edges: list[tuple[int, int]] = []
    teeth = 8
    pts_per = teeth * 4
    for i in range(pts_per):
        a = 2.0 * math.pi * i / pts_per
        rad = 1.18 if (i % 4 in (1, 2)) else 0.82
        gr_verts.append((rad * math.cos(a), 0.0, rad * math.sin(a)))
        gr_edges.append((i, (i + 1) % pts_per))
    ig_off = len(gr_verts)
    for i in range(16):
        a = 2.0 * math.pi * i / 16
        gr_verts.append((0.42 * math.cos(a), 0.0, 0.42 * math.sin(a)))
        gr_edges.append((ig_off + i, ig_off + (i + 1) % 16))
    # Spin axis spindle indicator along +Y
    ax_off = len(gr_verts)
    gr_verts.extend([(0.0, -0.45, 0.0), (0.0, 0.85, 0.0), (-0.18, 0.55, 0.0), (0.18, 0.55, 0.0)])
    gr_edges.extend([(ax_off, ax_off + 1), (ax_off + 1, ax_off + 2), (ax_off + 1, ax_off + 3)])
    w_gear = _make_wire_widget("WGT_HFW_Gear", gr_verts, gr_edges)

    return {
        "circle": w_fk_ring,
        "fk_ring": w_fk_ring,
        "chest": w_chest,
        "spline": w_spline,
        "head_ik": w_head_ik,
        "curl": w_curl,
        "gear": w_gear,
        "root": w_root,
        "cog": w_cog,
        "box": w_box,
        "diamond": w_diamond,
        "pin": w_pin,
        "paddle": w_paddle,
        "plate": w_plate,
        "weapon": w_weapon,
    }


def _analyze_armature_mesh_geometry(arm_obj: Any) -> tuple[dict[str, float], dict[str, tuple[float, float, float]]]:
    """Measure real per-bone mesh cross-section radii (for snug, non-overlapping widgets!) and 3D PCA spin axes for drills/grinders."""
    # Flush depsgraph first so BONE-parented meshes have exact evaluated `matrix_world` (never stale Identity)!
    try:
        bpy.context.view_layer.update()
    except Exception:
        pass
    bones = arm_obj.data.bones
    bone_inv = {b.name: b.matrix_local.inverted() for b in bones if not b.name.startswith(("CTRL_", "MCH_", "WGT_"))}
    # Precompute each bone's true anatomical axial unit vector in bone-local space (head -> primary child head)
    bone_axis_local: dict[str, np.ndarray] = {}
    for b in bones:
        if b.name not in bone_inv:
            continue
        ax = Vector((0.0, 1.0, 0.0))
        ch_dirs = [
            (c.head_local - b.head_local)
            for c in b.children
            if not c.name.startswith(("CTRL_", "MCH_", "WGT_"))
            and not c.name.lower().endswith("_helper")
            and (c.head_local - b.head_local).length > 0.02
        ]
        if ch_dirs:
            avg_d = sum(ch_dirs, Vector((0.0, 0.0, 0.0)))
            if avg_d.length > 1e-4:
                ax_l = bone_inv[b.name].to_3x3() @ avg_d.normalized()
                if ax_l.length > 1e-4:
                    ax = ax_l.normalized()
        bone_axis_local[b.name] = np.array((ax.x, ax.y, ax.z), dtype=np.float32)
    pts_by_bone: dict[str, list[np.ndarray]] = {}

    for obj in bpy.data.objects:
        if obj.type != "MESH" or obj.hide_viewport:
            continue
        if any("broken" in c.name.lower() or "widget" in c.name.lower() for c in obj.users_collection):
            continue
        if obj.parent == arm_obj and obj.parent_type == "BONE" and obj.parent_bone in bone_inv:
            pb_nm = obj.parent_bone
            mesh = obj.data
            if not mesh or len(mesh.vertices) == 0:
                continue
            # Relative matrix from mesh local to armature local -> bone local
            m_to_arm = arm_obj.matrix_world.inverted() @ obj.matrix_world
            m_to_bone = bone_inv[pb_nm] @ m_to_arm
            step = max(1, len(mesh.vertices) // 96)
            verts = [m_to_bone @ mesh.vertices[i].co for i in range(0, len(mesh.vertices), step)]
            if verts:
                pts_by_bone.setdefault(pb_nm, []).append(np.array([(v.x, v.y, v.z) for v in verts], dtype=np.float32))
        elif any(m.type == "ARMATURE" and m.object == arm_obj for m in obj.modifiers):
            mesh = obj.data
            if not mesh or len(mesh.vertices) == 0 or not obj.vertex_groups:
                continue
            vg_to_bone = {vg.index: vg.name for vg in obj.vertex_groups if vg.name in bone_inv}
            if not vg_to_bone:
                continue
            m_to_arm = arm_obj.matrix_world.inverted() @ obj.matrix_world
            v_step = max(1, len(mesh.vertices) // 2500)
            local_buckets: dict[str, list[tuple[float, float, float]]] = {}
            for vi in range(0, len(mesh.vertices), v_step):
                v = mesh.vertices[vi]
                best_g = None
                best_w = 0.28
                for g in v.groups:
                    if g.group in vg_to_bone and g.weight > best_w:
                        best_w = g.weight
                        best_g = vg_to_bone[g.group]
                if best_g is not None:
                    p_arm = m_to_arm @ v.co
                    p_b = bone_inv[best_g] @ p_arm
                    local_buckets.setdefault(best_g, []).append((p_b.x, p_b.y, p_b.z))
            for bnm, plist in local_buckets.items():
                if plist:
                    pts_by_bone.setdefault(bnm, []).append(np.array(plist, dtype=np.float32))

    radii: dict[str, float] = {}
    spin_axes: dict[str, tuple[float, float, float]] = {}
    for bnm, chunks in pts_by_bone.items():
        arr = np.concatenate(chunks, axis=0)
        if len(arr) < 4:
            continue
        # Measure perpendicular radius around the bone's TRUE anatomical axis (head->child), 75th percentile ignores thin fins/plates!
        u_ax = bone_axis_local.get(bnm, np.array((0.0, 1.0, 0.0), dtype=np.float32))
        proj = (arr @ u_ax)[:, None] * u_ax[None, :]
        perp = np.linalg.norm(arr - proj, axis=1)
        r75 = float(np.percentile(perp, 75))
        if r75 > 0.005:
            radii[bnm] = r75
        # 3D PCA spin-axis detection for cylindrical drills / flat discs / grinders
        centered = arr - arr.mean(axis=0, keepdims=True)
        cov = (centered.T @ centered) / max(1, len(arr))
        evals, evecs = np.linalg.eigh(cov)
        # evals sorted ascending: e0 <= e1 <= e2
        e0, e1, e2 = float(evals[0]), float(evals[1]), float(evals[2])
        if e2 > 1e-6:
            # Elongated cylinder/cone drill (`e2 >> e1 ~ e0`) vs flat disc/wheel (`e0 << e1 ~ e2`)
            ratio_long = e2 / max(e1, 1e-6)
            ratio_flat = e1 / max(e0, 1e-6)
            ax_vec = evecs[:, 2] if ratio_long >= ratio_flat else evecs[:, 0]
            # Snap cleanly to cardinal bone axis if within ~20 deg (matches Decima authoring convention!)
            abs_ax = np.abs(ax_vec)
            dom = int(np.argmax(abs_ax))
            if float(abs_ax[dom]) > 0.93:
                snap = [0.0, 0.0, 0.0]
                snap[dom] = 1.0 if ax_vec[dom] >= 0.0 else -1.0
                spin_axes[bnm] = (snap[0], snap[1], snap[2])
            else:
                nrm = float(np.linalg.norm(ax_vec)) or 1.0
                spin_axes[bnm] = (float(ax_vec[0] / nrm), float(ax_vec[1] / nrm), float(ax_vec[2] / nrm))

    # Propagate helper/armor radii gently onto immediate parent core bone so armored torsos get true outer armor clearance!
    for b in bones:
        if b.name in radii and b.parent is not None and b.parent.name in bone_inv:
            if b.name.lower().endswith("_helper") or "plate" in b.name.lower() or "armor" in b.name.lower():
                p_name = b.parent.name
                off_d = (b.head_local - b.parent.head_local).length
                eff_r = min(off_d + radii[b.name] * 0.5, max(radii.get(p_name, 0.0) * 1.45, radii[b.name] * 1.15))
                if eff_r > radii.get(p_name, 0.0) and off_d < 2.2:
                    radii[p_name] = eff_r

    return radii, spin_axes


# ---------------------------------------------------------------------------
# Dynamic Asset-Aware Bone Collection Naming (Shared by Control Rig & N-Panel)
# ---------------------------------------------------------------------------
def compute_dynamic_collection_labels(
    spec: dict[str, Any],
    helper_plate_bones: list[str] | tuple[str, ...] = (),
) -> dict[str, str]:
    """Compute descriptive, human-readable Bone Collection names based on what each group controls."""
    arch = str(spec.get("archetype") or "creature")
    has_chest = bool(spec.get("chest"))
    has_spine = bool(spec.get("spine"))
    has_wings = bool(spec.get("wings"))
    legs = spec.get("legs") or []
    arms = spec.get("arms") or []
    tails = spec.get("tails") or []
    fingers = spec.get("fingers") or []
    toes = spec.get("toes") or []
    frills = spec.get("frills") or []
    weapons = spec.get("weapons") or []
    plates = spec.get("plates") or []

    def _collect_lower(items: list[dict[str, Any]]) -> list[str]:
        out: list[str] = []
        for it in items:
            out.append(str(it.get("name", "")).lower())
            for b in it.get("bones") or ():
                out.append(str(b).lower())
        return out

    def _pick_labels(
        names_low: list[str],
        rules: tuple[tuple[tuple[str, ...], str, str], ...],
        fallback: str,
    ) -> str:
        """Return 1–2 matched semantic group names joined with ' & '. Each rule is (tokens, solo_label, short_pair_label)."""
        hits: list[tuple[str, str]] = []
        seen_solo: set[str] = set()
        covered: set[int] = set()
        for tokens, solo_lbl, short_lbl in rules:
            matched_idxs = [i for i, nm in enumerate(names_low) if i not in covered and any(tok in nm for tok in tokens)]
            if matched_idxs:
                covered.update(matched_idxs)
                if solo_lbl not in seen_solo:
                    seen_solo.add(solo_lbl)
                    hits.append((solo_lbl, short_lbl))
            if len(hits) >= 2:
                break
        if not hits:
            return fallback
        if len(hits) == 1:
            return hits[0][0]
        return f"{hits[0][1]} & {hits[1][1]}"

    # 1. Body / Torso / Chest / Head + Multi-Bone Torso IK / Cobra Hood IK / Spine FK + Appendages
    torso_ik_mode = str(spec.get("torso_ik_mode") or ("spine" if spec.get("torso_ik_supported") else "none"))
    if arch == "serpentine":
        body_lbl = "Body, Chest & Hood" if has_chest else "Serpent Body & Head"
    elif has_chest and has_spine:
        body_lbl = "Torso, Chest & Head"
    elif has_chest:
        body_lbl = "Hips, Chest & Head"
    else:
        body_lbl = "Body & Head"

    torso_ik_lbl = "Cobra Hood IK" if torso_ik_mode == "hood" else "Chest & Back IK"
    hood_fk_lbl = "Cobra Hood FK"
    spine_fk_lbl = "Serpent Body FK" if arch == "serpentine" else "Spine & Chest FK"

    # 2. Tail, Appendages (Trunk/Tongue) & Wings (Dual IK / FK!)
    tail_lbl = "Tails FK" if len(tails) > 1 else "Tail FK"
    apps = spec.get("appendages") or []
    if apps:
        app_names = list(dict.fromkeys(str(a.get("name", "Appendage")) for a in apps))
        app_lbl = " & ".join(app_names[:2]) + " FK"
    else:
        app_lbl = "Appendage FK"
    wings_ik_lbl = "Wings IK"
    wings_fk_lbl = "Wings FK"

    # 3. Legs (Biped / Quadruped Front+Rear / Hexapod / Aquatic Flippers) & Arms
    has_front_legs = any(str(lg.get("name", "")).startswith("FrontLeg_") for lg in legs)
    if arch == "aquatic":
        leg_prefix = "Rear Flippers"
        front_leg_prefix = "Front Legs"
        arm_prefix = "Front Flippers"
    elif arch == "hexapod":
        leg_prefix = "Mid & Rear Legs"
        front_leg_prefix = "Front Legs"
        arm_prefix = "Claw & Gun Arms"
    elif has_front_legs:
        leg_prefix = "Rear Legs"
        front_leg_prefix = "Front Legs"
        arm_prefix = "Forelimbs" if arch == "theropod" else "Arms"
    else:
        leg_prefix = "Quad Legs" if len(legs) >= 4 else "Legs"
        front_leg_prefix = "Front Legs"
        arm_prefix = "Forelimbs" if arch == "theropod" else "Arms"

    # 4. Hands / Fingers / Wing Ribs / Feet / Toes / Talons / Hooves
    f_low = _collect_lower(fingers)
    t_low = _collect_lower(toes)
    hand_part = ""
    if f_low:
        if any("wingrib" in nm for nm in f_low):
            hand_part = "Wing Ribs"
        elif any(k in nm for nm in f_low for k in ("winghandhinge", "wingfinger")) or (arch == "flyer" and f_low):
            hand_part = "Wing Fingers"
        elif any(k in nm for nm in f_low for k in ("thumb", "index", "middle", "ring", "pinky")):
            hand_part = "Fingers"
        elif arch == "theropod" or any("claw" in nm for nm in f_low):
            hand_part = "Claws"
        else:
            hand_part = "Hands"
    foot_part = ""
    if t_low:
        if any("hoof" in nm for nm in t_low):
            foot_part = "Hooves"
        elif arch in ("theropod", "flyer") and any(
            k in nm for nm in t_low for k in ("footfinger", "claw", "talon", "thumb", "toe")
        ):
            foot_part = "Talons"
        elif arch == "humanoid" or any(k in nm for nm in t_low for k in ("toe", "ball")):
            foot_part = "Toes"
        elif any("finger" in nm for nm in t_low):
            foot_part = "Foot Claws"
        else:
            foot_part = "Feet"
    if hand_part and foot_part:
        hf_lbl = f"{hand_part} & {foot_part}"
    elif hand_part:
        hf_lbl = hand_part
    elif foot_part:
        hf_lbl = f"{foot_part} & Claws" if foot_part == "Talons" and arch == "flyer" else (
            f"{foot_part} & Feet" if foot_part == "Toes" else foot_part
        )
    else:
        hf_lbl = "Hands & Feet"

    # 5. Frills / Feathers / Plasma Intakes / Fans / Flaps / Antennae / Wing Engines
    fr_low = _collect_lower(frills)
    frill_lbl = _pick_labels(
        fr_low,
        (
            (("plasmaspike", "frontguide", "guidespike"), "Plasma Intakes", "Plasma Intakes"),
            (("ribbit",), "Vocal Sacs", "Vocal Sacs"),
            (
                ("feather", "plume"),
                "Hood Feathers"
                if arch == "serpentine"
                else ("Tail Feathers" if all("tail" in nm for nm in fr_low if "feather" in nm) and fr_low else "Feathers"),
                "Hood Feathers" if arch == "serpentine" else "Feathers",
            ),
            (("headfan", "fan"), "Head Fans", "Head Fans"),
            (("engine",), "Wing Engines", "Engines"),
            (("rearwing", "reartopwing", "fronttopwing", "sidewing", "anklewing", "wingknife"), "Aux Wings", "Aux Wings"),
            (("blade",), "Blades", "Blades"),
            (("spinespike", "spike"), "Dorsal Spikes", "Spikes"),
            (("antenna", "antena"), "Antennae", "Antennae"),
            (("headflap", "head_flap"), "Head Flaps", "Head Flaps"),
            (("ventflap",), "Vent Flaps", "Vent Flaps"),
            (("beltflap",), "Belt Flaps", "Belt Flaps"),
            (("ear",), "Ears", "Ears"),
            (("flap",), "Flaps", "Flaps"),
            (("frill", "crest", "mane"), "Frills & Crests", "Frills"),
            (("fin", "gill"), "Fins", "Fins"),
            (("exhaust",), "Exhausts", "Exhausts"),
            (("extender",), "Extenders", "Extenders"),
        ),
        "Frills & Details",
    )

    # 6. Weapons / Saws / Cannons / Tail Blasters / Guns / Horns / Tusks / Jaws
    wp_low = _collect_lower(weapons)
    weapon_lbl = _pick_labels(
        wp_low,
        (
            (("saw_sjnt", "saw_", "chainsaw"), "Jaw Saws", "Jaw Saws"),
            (("watercannon",), "Water Cannons", "Water Cannons"),
            (("headcannon", "cannon"), "Cannons", "Cannons"),
            (("tailbracket", "bottomtailbracket", "blaster"), "Tail Blasters", "Tail Blasters"),
            (("gunbone", "gun_bone", "gun_end", "secondary_gun"), "Weapon Sockets", "Weapon Sockets"),
            (("beltgun", "headgun", "ballgun", "wolfgun", "shouldergun", "turret", "gun"), "Guns & Turrets", "Guns"),
            (("launcher", "mortar"), "Launchers", "Launchers"),
            (("tusk",), "Tusks", "Tusks"),
            (("horn",), "Horns", "Horns"),
            (("drill", "grinder"), "Drills & Grinders", "Drills"),
            (("hammer",), "Hammers", "Hammers"),
            (("mandible",), "Mandibles", "Mandibles"),
            (("jawplate", "jaw"), "Jaws", "Jaws"),
            (("bracket",), "Weapon Mounts", "Weapon Mounts"),
            (("pincer", "stinger", "tailclaw"), "Pincers & Claws", "Pincers"),
            (("laser", "nozzle"), "Lasers & Nozzles", "Nozzles"),
            (("radar", "scanner"), "Scanners", "Scanners"),
        ),
        "Weapons",
    )

    # 7. Armor Plates / Canisters / Shields / Shells / Hinges / Pods / Doors
    pl_low = _collect_lower(plates) + [str(h).lower() for h in helper_plate_bones]
    plate_lbl = _pick_labels(
        pl_low,
        (
            (("plate", "armor"), "Armor Plates", "Armor Plates"),
            (("canister", "cannister", "cell", "battery", "charger", "tank", "sac", "sack", "cargo", "core"), "Canisters & Cells", "Canisters"),
            (("shield", "shell"), "Shields & Shells", "Shields"),
            (("missilepod", "pod"), "Missile Pods", "Pods"),
            (("door", "hatch", "lid", "cover", "bellyopener"), "Doors & Hatches", "Doors"),
            (("clamp",), "Cargo Clamps", "Clamps"),
            (("hinge",), "Hinges", "Hinges"),
            (("vent",), "Vents", "Vents"),
            (("hipbase",), "Hip Guards", "Hip Guards"),
            (("clip",), "Clips", "Clips"),
        ),
        "Armor Plates",
    )

    spline_chains = spec.get("spline_chains") or []
    if spline_chains:
        sk0 = spline_chains[0].get("kind")
        spline_ik_lbl = (
            "Serpent Spline IK"
            if sk0 == "spine"
            else ("Trunk Spline IK" if sk0 == "appendage" else "Tail Spline IK")
        )
    else:
        spline_ik_lbl = "Spline IK"

    chainsaws = spec.get("chainsaws") or []
    spinners = spec.get("spinners") or []
    if chainsaws and spinners:
        spin_lbl = "Chainsaws & Drills"
    elif chainsaws:
        spin_lbl = "Chainsaws"
    elif spinners:
        sp_kinds = {s.get("kind") for s in spinners}
        if "grinder" in sp_kinds and "drill" in sp_kinds:
            spin_lbl = "Drills & Grinders"
        elif "grinder" in sp_kinds:
            spin_lbl = "Earthgrinders" if arch == "serpentine" else "Grinders"
        elif "rotor" in sp_kinds and "drill" not in sp_kinds:
            spin_lbl = "Rotors & Turbines"
        else:
            spin_lbl = "Drills & Rotors"
    else:
        spin_lbl = "Spinners"

    slots: dict[str, str] = {
        "body": body_lbl,
        "head_ik": "Head & Neck IK",
        "torso_ik": torso_ik_lbl,
        "hood_fk": hood_fk_lbl,
        "spline_ik": spline_ik_lbl,
        "spine_fk": spine_fk_lbl,
        "tail": tail_lbl,
        "appendages": app_lbl,
        "wings_ik": wings_ik_lbl,
        "wings_fk": wings_fk_lbl,
        "wings": wings_fk_lbl,
        "front_legs_ik": f"{front_leg_prefix} IK",
        "front_legs_fk": f"{front_leg_prefix} FK",
        "legs_ik": f"{leg_prefix} IK",
        "legs_fk": f"{leg_prefix} FK",
        "arms_ik": f"{arm_prefix} IK",
        "arms_fk": f"{arm_prefix} FK",
        "hands_feet": hf_lbl,
        "hf_tweaks": f"{hf_lbl} Tweaks",
        "spinners": spin_lbl,
        "frills": frill_lbl,
        "weapons": weapon_lbl,
        "plates": plate_lbl,
        "mechanism": "Rig Mechanism (Hidden)",
    }

    # Defensive uniqueness guarantee (preserve intentional `wings` alias to `wings_fk`)
    used_vals: set[str] = set()
    for skey in list(slots.keys()):
        if skey == "wings":
            slots["wings"] = slots["wings_fk"]
            continue
        val = slots[skey]
        if val in used_vals:
            val = f"{val} ({skey.replace('_', ' ').title()})"
            slots[skey] = val
        used_vals.add(val)

    return slots


# ---------------------------------------------------------------------------
# Blender Control Rig Builder (Full IK + FK + Multi-Bone Chest IK + All Archetypes + Rest-Lock!)
# ---------------------------------------------------------------------------
_RIG_GROUP_PROPS = (
    ("hfw_rig_master", "Master Control Rig Influence (0 = Raw HFW Game Anim, 1 = Full Control Rig)", 1.0),
    ("hfw_ik_fk_torso", "Torso / Cobra Hood Mode (0 = Pure FK, 1 = Multi-Bone Chest / Hood IK)", 1.0),
    ("hfw_ik_fk_spline", "Spline IK Mode for Serpent Body / Long Tail / Trunk (0 = Pure FK, 1 = Smooth B-Bone Spline IK)", 0.0),
    ("hfw_ik_fk_head", "Head & Neck Mode (0 = Pure Neck/Head FK, 1 = Head IK Position & Aim)", 0.0),
    ("hfw_head_ik_space", "Head IK Parent Space (0 = Follow Chest/Torso, 1 = World Lock / Stalk Target)", 0.0),
    ("hfw_ik_fk_legs", "Legs Mode (0 = Pure FK Legs, 1 = Pure IK Foot Controls)", 1.0),
    ("hfw_ik_fk_front_legs", "Quadruped Front Legs Mode (0 = Pure FK Front Legs, 1 = Pure IK Front Feet)", 1.0),
    ("hfw_ik_fk_arms", "Arms / Flippers Mode (0 = Pure FK, 1 = Pure IK)", 1.0),
    ("hfw_ik_fk_wings", "Flyer Wings Mode (0 = Pure FK Flap/Fold, 1 = Wingtip IK)", 0.0),
    ("hfw_saw_speed", "Chainsaw Auto-Spin Speed during playback (0 = Manual Dial Only, 1 = In-Game Rev Speed)", 0.0),
    ("hfw_drill_speed", "Drills / Grinders / Rotors Auto-Spin Speed during playback (0 = Manual Dial Only, 1 = Active Spin)", 0.0),
    ("hfw_rig_legs_ik", "Legs Control Influence (IK & FK)", 1.0),
    ("hfw_rig_arms_ik", "Arms / Wings Control Influence (IK & FK)", 1.0),
    ("hfw_rig_spine", "Root / COG / Spine Influence", 1.0),
    ("hfw_rig_chest", "Chest / Upper Torso Control Influence", 1.0),
    ("hfw_rig_head", "Neck & Head Control Influence", 1.0),
    ("hfw_rig_tail", "Tail / Trunk / Tongue FK & Spline Influence", 1.0),
    ("hfw_rig_jaw", "Jaw / Mandibles Control Influence", 1.0),
    ("hfw_rig_detail", "Hands, Fingers, Wing Ribs, Feet & Toes Influence", 1.0),
    ("hfw_rig_frills", "Feathers, Frills, Plasma Intakes, Fins & Antennae Influence", 1.0),
    ("hfw_rig_plates", "Armor Plates, Shields, Canisters & Cargo Doors Influence", 1.0),
    ("hfw_rig_weapons", "Weapons, Chainsaws, Drills, Cannons & Tail Blasters Influence", 1.0),
    ("hfw_rig_lock_secondary", "Lock Uncontrolled Secondary Bones to Rest Pose when Control Rig = 1 (prevents ghost game anim!)", 1.0),
)


def _add_influence_driver(
    arm_obj: Any,
    constraint: Any,
    group_prop: str,
    *,
    ik_prop: str | None = None,
    ik_invert: bool = False,
    extra_prop: str | None = None,
    use_master: bool = True,
) -> None:
    """Drive `constraint.influence` using Blender's safe Simple Expression evaluator (works without autoexec!)."""
    fc = constraint.driver_add("influence")
    drv = fc.driver
    drv.type = "SCRIPTED"

    terms: list[str] = []

    def _add_var(vname: str, prop_key: str) -> None:
        v = drv.variables.new()
        v.name = vname
        v.type = "SINGLE_PROP"
        v.targets[0].id_type = "OBJECT"
        v.targets[0].id = arm_obj
        v.targets[0].data_path = f'["{prop_key}"]'

    if use_master:
        _add_var("master", "hfw_rig_master")
        terms.append("master")
    if group_prop and group_prop != "hfw_rig_master":
        _add_var("grp", group_prop)
        terms.append("grp")
    if ik_prop:
        _add_var("ik", ik_prop)
        terms.append("(1.0 - ik)" if ik_invert else "ik")
    if extra_prop:
        _add_var("ext", extra_prop)
        terms.append("ext")

    drv.expression = " * ".join(terms) if terms else "1.0"


def _ensure_rig_custom_properties(arm_obj: Any, default_master: float = 1.0) -> None:
    for prop_name, desc, def_val in _RIG_GROUP_PROPS:
        val = default_master if prop_name == "hfw_rig_master" else def_val
        if prop_name not in arm_obj:
            arm_obj[prop_name] = float(val)
        try:
            ui = arm_obj.id_properties_ui(prop_name)
            if prop_name in ("hfw_saw_speed", "hfw_drill_speed"):
                ui.update(min=-5.0, max=5.0, soft_min=-3.0, soft_max=3.0, description=desc)
            else:
                ui.update(min=0.0, max=1.0, soft_min=0.0, soft_max=1.0, description=desc)
        except Exception:
            pass


def _set_bone_color(
    pbone: Any,
    side: str,
    is_root: bool = False,
    is_detail: bool = False,
    category: str = "body",
) -> None:
    try:
        if is_root:
            pbone.color.palette = "THEME03"  # Bright Green (Root / COG)
        elif category == "chest":
            pbone.color.palette = "THEME02"  # Vibrant Orange-Gold (Dedicated Chest Controller)
        elif category == "spline":
            pbone.color.palette = "THEME07"  # Bright Aqua/Teal (Spline IK Gimbal Knots)
        elif category == "head_ik":
            pbone.color.palette = "THEME02"  # Vibrant Orange-Gold (Head IK Target)
        elif category == "spin":
            pbone.color.palette = "THEME10"  # Vivid Magenta-Pink (Chainsaw & Drill Spin Dials)
        elif category == "frills":
            pbone.color.palette = "THEME12" if side == "L" else ("THEME05" if side == "R" else "THEME07")  # Cyan / Teal
        elif category == "plates":
            pbone.color.palette = "THEME02"  # Warm Amber / Orange
        elif category == "weapons":
            pbone.color.palette = "THEME06" if side != "R" else "THEME10"  # Vivid Purple / Magenta
        elif category == "tail":
            pbone.color.palette = "THEME09"  # Gold / Yellow
        elif is_detail:
            pbone.color.palette = "THEME04" if side == "L" else ("THEME01" if side == "R" else "THEME09")
        elif side == "L":
            pbone.color.palette = "THEME04"  # Bright Blue
        elif side == "R":
            pbone.color.palette = "THEME01"  # Bright Red
        else:
            pbone.color.palette = "THEME09"  # Gold / Yellow
    except Exception:
        pass


def _compute_pole_pos_and_angle(
    p0: Any,
    p_mid: Any,
    p_end: Any,
    chain_root_mat: Any,
    fallback_dir: Any,
    dist_scale: float = 0.85,
) -> tuple[Any, float]:
    chord = p_end - p0
    chord_len = max(chord.length, 0.1)
    chord_dir = chord.normalized()
    v_mid = p_mid - p0
    perp = v_mid - chord_dir * v_mid.dot(chord_dir)
    if perp.length < chord_len * 0.03:
        bend_dir = fallback_dir - chord_dir * fallback_dir.dot(chord_dir)
        if bend_dir.length < 1e-4:
            bend_dir = Vector((0.0, 1.0, 0.0))
        bend_dir.normalize()
    else:
        bend_dir = perp.normalized()

    pole_pos = p_mid + bend_dir * (chord_len * dist_scale)

    rot3 = chain_root_mat.to_3x3()
    x_ax = (rot3 @ Vector((1.0, 0.0, 0.0))).normalized()
    y_ax = (rot3 @ Vector((0.0, 1.0, 0.0))).normalized()
    z_ax = (rot3 @ Vector((0.0, 0.0, 1.0))).normalized()
    to_pole = pole_pos - p0
    proj = to_pole - y_ax * to_pole.dot(y_ax)
    if proj.length > 1e-6:
        proj.normalize()
        pole_angle = math.atan2(proj.dot(z_ax), proj.dot(x_ax))
    else:
        pole_angle = 0.0
    return pole_pos, pole_angle


def _sample_chain_handles(chain: list[str], max_handles: int = 12) -> list[str]:
    if len(chain) <= max_handles:
        return list(chain)
    idxs = sorted({int(round(i * (len(chain) - 1) / (max_handles - 1))) for i in range(max_handles)})
    return [chain[i] for i in idxs]


_ALL_CTRL_COLLECTION_NAMES = (
    "Body_Main",
    "Tail",
    "Legs_IK",
    "Legs_FK",
    "Arms_IK",
    "Arms_FK",
    "Hands_Feet",
    "Frills_Feathers",
    "Plates_Armor",
    "Weapons_Aux",
    "Mechanism",
    "Rig Mechanism (Hidden)",
    # Legacy v0.6/v0.7 names for clean rebuilds
    "Controls",
    "Controls_IK",
    "Controls_FK",
    "Controls_Detail",
)

_LEGACY_SLOT_DEFAULTS = {
    "body": "Body_Main",
    "head_ik": "Head & Neck IK",
    "torso_ik": "Chest & Back IK",
    "hood_fk": "Cobra Hood FK",
    "spline_ik": "Spline IK",
    "spine_fk": "Spine & Chest FK",
    "tail": "Tail",
    "appendages": "Appendage FK",
    "wings_ik": "Wings IK",
    "wings_fk": "Wings FK",
    "wings": "Wings FK",
    "front_legs_ik": "Front Legs IK",
    "front_legs_fk": "Front Legs FK",
    "legs_ik": "Legs_IK",
    "legs_fk": "Legs_FK",
    "arms_ik": "Arms_IK",
    "arms_fk": "Arms_FK",
    "hands_feet": "Hands_Feet",
    "hf_tweaks": "Finger & Toe Tweaks",
    "spinners": "Chainsaws & Drills",
    "frills": "Frills_Feathers",
    "weapons": "Weapons_Aux",
    "plates": "Plates_Armor",
    "mechanism": "Mechanism",
}


def _get_rig_collection_slots(arm_obj: Any) -> dict[str, str]:
    """Read dynamically renamed Bone Collection slots from `hfw_ctrl_rig_meta` (with legacy fallback)."""
    slots = dict(_LEGACY_SLOT_DEFAULTS)
    meta_raw = arm_obj.get("hfw_ctrl_rig_meta") if arm_obj is not None else None
    if meta_raw:
        try:
            meta = json.loads(meta_raw)
            saved = meta.get("collection_slots") or {}
            if isinstance(saved, dict):
                for k, v in saved.items():
                    if isinstance(v, str) and v:
                        slots[k] = v
        except Exception:
            pass
    return slots


def remove_control_rig(arm_obj: Any) -> None:
    """Cleanly remove any existing HFW Control Rig bones, constraints, drivers, and dynamic Bone Collections."""
    if arm_obj is None or arm_obj.type != "ARMATURE":
        return
    prev_mode = bpy.context.mode
    bpy.context.view_layer.objects.active = arm_obj
    if prev_mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    # Identify all Control Rig collections BEFORE removing CTRL_/MCH_ bones (handles dynamic names + legacy names!)
    colls_to_remove: set[str] = set(_ALL_CTRL_COLLECTION_NAMES)
    meta_raw = arm_obj.get("hfw_ctrl_rig_meta")
    if meta_raw:
        try:
            meta = json.loads(meta_raw)
            for cname in (meta.get("collection_slots") or {}).values():
                if isinstance(cname, str) and cname:
                    colls_to_remove.add(cname)
        except Exception:
            pass
    _RAW_DEF_COLLS = {"Skeleton", "Body", "Helpers"}
    for bc in arm_obj.data.collections:
        if bc.name in _RAW_DEF_COLLS:
            continue
        bones_in = list(bc.bones)
        if bones_in and all(b.name.startswith(("CTRL_", "MCH_")) for b in bones_in):
            colls_to_remove.add(bc.name)

    for pb in arm_obj.pose.bones:
        if not pb.name.startswith(("CTRL_", "MCH_")):
            for con in list(pb.constraints):
                if con.name.startswith("HFW_"):
                    try:
                        con.driver_remove("influence")
                    except Exception:
                        pass
                    pb.constraints.remove(con)

    bpy.ops.object.mode_set(mode="EDIT")
    ebones = arm_obj.data.edit_bones
    for eb in [b for b in ebones if b.name.startswith(("CTRL_", "MCH_"))]:
        ebones.remove(eb)
    bpy.ops.object.mode_set(mode="OBJECT")

    for cname in colls_to_remove:
        if cname in _RAW_DEF_COLLS:
            continue
        bc = arm_obj.data.collections.get(cname)
        if bc is not None:
            try:
                arm_obj.data.collections.remove(bc)
            except Exception:
                pass


def build_control_rig(
    arm_obj: Any,
    *,
    default_master: float = 1.0,
    rebuild: bool = True,
) -> dict[str, Any]:
    """Build a non-destructive layered `DEF + MCH + CTRL` Control Rig with large intuitive widgets,
    full FK/IK limb switching + snapping, full FK Tails, categorized Frills/Intakes/Plates/Weapons & Rest-Lock."""
    if bpy is None:
        raise RuntimeError("build_control_rig must be executed inside Blender.")
    if arm_obj is None or arm_obj.type != "ARMATURE":
        raise ValueError("Target object must be a Blender ARMATURE.")

    if rebuild:
        remove_control_rig(arm_obj)

    spec = analyze_armature_rig_spec(arm_obj)
    rig_h = max(0.5, float(spec["rig_height"]))
    fwd_vec = Vector((spec["forward"][0], spec["forward"][1], 0.0))
    if fwd_vec.length < 1e-4:
        fwd_vec = Vector((0.0, 1.0, 0.0))
    fwd_vec.normalize()

    widgets = _ensure_control_widgets()
    _ensure_rig_custom_properties(arm_obj, default_master=default_master)
    # Smart per-archetype defaults:
    # - Torso: Multi-bone Chest IK (`1.0`) when spine->chest chain exists, switchable to Pure Spine FK (`0.0`)!
    # - Legs / Quad Front Legs: IK (`1.0`) so all feet plant cleanly!
    # - Arms: FK (`0.0`) on theropods (tiny forelimbs ride chest!), IK (`1.0`) on humanoids/hexapods/aquatic!
    # - Wings: FK (`0.0`) by default (classic wing flap/fold!), 1-click switchable to Wingtip IK (`1.0`)!
    default_torso_ik = 1.0 if spec.get("torso_ik_supported") else 0.0
    spline_chains_spec = spec.get("spline_chains") or []
    default_spline_ik = 1.0 if any(sc.get("default_on") for sc in spline_chains_spec) else 0.0
    default_head_ik = 0.0
    default_arms_ik = 0.0 if (spec["archetype"] == "theropod" and spec["arms"]) else 1.0
    default_wings_ik = 0.0
    arm_obj["hfw_ik_fk_torso"] = float(default_torso_ik)
    arm_obj["hfw_ik_fk_spline"] = float(default_spline_ik)
    arm_obj["hfw_ik_fk_head"] = float(default_head_ik)
    arm_obj["hfw_head_ik_space"] = 0.0
    arm_obj["hfw_ik_fk_legs"] = 1.0
    arm_obj["hfw_ik_fk_front_legs"] = 1.0
    arm_obj["hfw_ik_fk_arms"] = float(default_arms_ik)
    arm_obj["hfw_ik_fk_wings"] = float(default_wings_ik)
    arm_obj["hfw_saw_speed"] = 0.0
    arm_obj["hfw_drill_speed"] = 0.0
    arm_obj["hfw_rig_archetype"] = spec["archetype"]
    arm_obj["hfw_has_control_rig"] = True
    # Always display Control Rig widgets clearly IN FRONT of thick machine armor in Viewport!
    arm_obj.show_in_front = True

    arm_data = arm_obj.data
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="OBJECT")

    # Measure REAL per-bone mesh cross-section radii (for clean, non-overlapping body-fitted widgets!)
    # and 3D PCA spin axes for rotary drills, grinders, rotors, turbines & Tidal Discs!
    bone_radii, spinner_axes = _analyze_armature_mesh_geometry(arm_obj)

    # Find visible non-broken mesh-owning `*_helper` bones BEFORE entering Edit Mode so we can give
    # every detachable armor plate & canister an optional `CTRL_Plate_*` handle (and spin Tideripper's Tidal Disc!)
    visible_helper_meshes: dict[str, str] = {}
    helper_spinners: list[dict[str, Any]] = []
    for obj in bpy.data.objects:
        if obj.type != "MESH" or obj.parent != arm_obj or obj.parent_type != "BONE":
            continue
        pb_nm = obj.parent_bone
        if not pb_nm or not pb_nm.lower().endswith("_helper"):
            continue
        if obj.hide_viewport or any("broken" in c.name.lower() for c in obj.users_collection):
            continue
        if pb_nm not in visible_helper_meshes:
            visible_helper_meshes[pb_nm] = obj.name
            pl = pb_nm.lower()
            if any(tk in pl for tk in ("disc_c_01", "tidaldisc", "grinder_disc", "legrotor_grinder", "headgrinder")) and "dmg" not in pl:
                helper_spinners.append(
                    {
                        "name": re.sub(r"(_[LlRrCc]x)?_helper$", "", pb_nm, flags=re.IGNORECASE)[:18],
                        "kind": "rotor",
                        "side": _extract_side(pb_nm),
                        "parent_def": arm_data.bones[pb_nm].parent.name if (pb_nm in arm_data.bones and arm_data.bones[pb_nm].parent) else None,
                        "bone": pb_nm,
                    }
                )

    all_spinners = list(spec.get("spinners") or []) + helper_spinners

    def _get_coll(cname: str, visible: bool = True) -> Any:
        bc = arm_data.collections.get(cname) or arm_data.collections.new(cname)
        bc.is_visible = visible
        return bc

    # Compute descriptive, asset-specific Bone Collection names based on what each group controls!
    spec_for_labels = dict(spec)
    if all_spinners:
        spec_for_labels["spinners"] = all_spinners
    dyn_labels = compute_dynamic_collection_labels(spec_for_labels, list(visible_helper_meshes.keys()))
    active_slots: dict[str, str] = {}

    def _slot_coll(slot_key: str, condition: bool, visible: bool) -> Any:
        if not condition:
            return None
        cname = dyn_labels[slot_key]
        active_slots[slot_key] = cname
        return _get_coll(cname, visible)

    torso_ik_mode = str(spec.get("torso_ik_mode") or ("spine" if spec.get("torso_ik_supported") else "none"))
    has_torso_ik_spec = torso_ik_mode != "none"
    has_head_ik_spec = bool(spec.get("head_ik_supported"))
    has_spline_spec = bool(spline_chains_spec)
    spline_spine_active = any(sc.get("kind") == "spine" for sc in spline_chains_spec)
    spline_tail_active = any(sc.get("kind") == "tail" for sc in spline_chains_spec)
    spline_app_active = any(sc.get("kind") == "appendage" for sc in spline_chains_spec)
    has_front_legs = any(lg["name"].startswith("FrontLeg_") for lg in spec["legs"])
    has_other_legs = any(not lg["name"].startswith("FrontLeg_") for lg in spec["legs"])
    has_wing_ik = any(len(w.get("ik_bones") or ()) >= 2 for w in spec["wings"])
    has_spin_rig = bool(spec.get("chainsaws") or all_spinners)
    has_multi_hf = any(len(ch.get("bones") or ()) >= 2 for ch in list(spec.get("fingers") or []) + list(spec.get("toes") or []))

    body_coll = _slot_coll("body", True, True)
    head_ik_coll = _slot_coll("head_ik", has_head_ik_spec, default_head_ik >= 0.5)
    torso_ik_coll = _slot_coll("torso_ik", has_torso_ik_spec, True)
    hood_fk_coll = _slot_coll("hood_fk", torso_ik_mode == "hood", False)
    spline_ik_coll = _slot_coll("spline_ik", has_spline_spec, default_spline_ik >= 0.5)
    spine_fk_coll = _slot_coll(
        "spine_fk",
        bool(spec["spine"]),
        (torso_ik_mode != "spine") and (not (spline_spine_active and default_spline_ik >= 0.5)),
    )
    tail_coll = _slot_coll("tail", bool(spec["tails"]), not (spline_tail_active and default_spline_ik >= 0.5))
    app_coll = _slot_coll("appendages", bool(spec.get("appendages")), not (spline_app_active and default_spline_ik >= 0.5))
    wings_ik_coll = _slot_coll("wings_ik", has_wing_ik, default_wings_ik >= 0.5)
    wings_fk_coll = _slot_coll("wings_fk", bool(spec["wings"]), default_wings_ik < 0.5)
    if wings_fk_coll is not None:
        active_slots["wings"] = active_slots["wings_fk"]
    front_legs_ik_coll = _slot_coll("front_legs_ik", has_front_legs, True)
    front_legs_fk_coll = _slot_coll("front_legs_fk", has_front_legs, False)
    legs_ik_coll = _slot_coll("legs_ik", has_other_legs, True)
    legs_fk_coll = _slot_coll("legs_fk", has_other_legs, False)
    arms_ik_coll = _slot_coll("arms_ik", bool(spec["arms"]), default_arms_ik >= 0.5)
    arms_fk_coll = _slot_coll("arms_fk", bool(spec["arms"]), default_arms_ik < 0.5)
    hf_coll = _slot_coll("hands_feet", bool(spec["fingers"] or spec["toes"]), True)
    hf_tweaks_coll = _slot_coll("hf_tweaks", has_multi_hf, False)
    spin_coll = _slot_coll("spinners", has_spin_rig, True)
    frill_coll = _slot_coll("frills", bool(spec["frills"]), True)
    weapon_coll = _slot_coll("weapons", bool(spec["weapons"]), True)
    plate_coll = _slot_coll("plates", bool(spec["plates"] or visible_helper_meshes), False)
    mch_coll = _slot_coll("mechanism", True, False)

    for raw_cname in ("Skeleton", "Body", "Helpers"):
        rc = arm_data.collections.get(raw_cname)
        if rc is not None:
            rc.is_visible = False

    bpy.ops.object.mode_set(mode="EDIT")
    ebones = arm_data.edit_bones

    def_rest: dict[str, tuple[Vector, Vector, Matrix, float]] = {}
    for eb in ebones:
        if not eb.name.startswith(("CTRL_", "MCH_", "WGT_")):
            def_rest[eb.name] = (eb.head.copy(), eb.tail.copy(), eb.matrix.copy(), float(eb.length))

    def_bindings: list[dict[str, Any]] = []
    follow_setups: list[dict[str, Any]] = []
    extra_constraints: list[dict[str, Any]] = []
    ik_setups: list[dict[str, Any]] = []
    spline_setups: list[dict[str, Any]] = []
    curl_setups: list[dict[str, Any]] = []
    saw_driver_setups: list[dict[str, Any]] = []
    spin_driver_setups: list[dict[str, Any]] = []
    ctrl_styles: dict[str, dict[str, Any]] = {}
    bake_fk_pairs: list[dict[str, str]] = []
    bake_ik_limbs: list[dict[str, Any]] = []
    limb_fk_chains: dict[str, list[dict[str, Any]]] = {
        "torso": [],
        "head": [],
        "spline": [],
        "legs": [],
        "front_legs": [],
        "arms": [],
        "wings": [],
    }
    torso_ik_meta: dict[str, Any] = {}
    head_ik_meta: dict[str, Any] = {}
    spline_ik_meta: list[dict[str, Any]] = []
    tail_ctrls: list[str] = []
    spine_ctrls: list[str] = []
    spline_ctrls: list[str] = []
    neck_ctrls: list[str] = []
    curl_ctrls: list[str] = []
    spin_ctrls: list[str] = []
    wing_ctrls: list[str] = []
    wing_ik_ctrls: list[str] = []
    appendage_ctrls: list[str] = []
    driven_def_bones: set[str] = set()
    core_def_bones: set[str] = set()

    def _fit_scale(
        bname: str | None,
        fallback: float,
        min_s: float,
        max_s: float,
        mult: float = 1.18,
    ) -> float:
        """Fit widget scale cleanly to the real cross-section of the bone's mesh geometry so controls hug body parts snugly!"""
        if bname and bname in bone_radii:
            r_fit = bone_radii[bname] * mult
            # Keep widgets cleanly bounded: never microscopic, never giant overlapping spiderwebs!
            return max(min_s, min(max_s, r_fit))
        return max(min_s, min(max_s, fallback))

    def _unique_name(base: str) -> str:
        if base not in ebones:
            return base
        idx = 1
        while f"{base}_{idx}" in ebones:
            idx += 1
        return f"{base}_{idx}"

    def _new_bone(
        name: str,
        head: Vector,
        tail: Vector,
        parent_eb: Any = None,
        *,
        matrix: Matrix | None = None,
        colls: tuple[Any, ...] = (),
        use_connect: bool = False,
    ) -> Any:
        eb = ebones.new(name)
        eb.head = head
        if (tail - head).length < 1e-4:
            tail = head + Vector((0.0, 0.06, 0.0))
        eb.tail = tail
        if matrix is not None:
            eb.matrix = matrix
        eb.use_deform = False
        if parent_eb is not None:
            eb.parent = parent_eb
            eb.use_connect = use_connect
        for c in colls:
            if c is not None:
                c.assign(eb)
        return eb

    def _make_bind_child(
        def_name: str,
        parent_eb: Any,
        group_prop: str,
        mode: str = "TRANSFORM",
        *,
        prefix: str = "MCH_Bind_",
        ik_prop: str | None = None,
        ik_invert: bool = False,
        extra_prop: str | None = None,
        con_name: str = "HFW_CTRL",
    ) -> str:
        h, t, m, l = def_rest[def_name]
        bname = _unique_name(f"{prefix}{def_name}")
        eb = _new_bone(bname, h, t, parent_eb, matrix=m, colls=(mch_coll,))
        eb.length = max(0.015, l)
        def_bindings.append(
            {
                "def_name": def_name,
                "target_mch": bname,
                "group_prop": group_prop,
                "mode": mode,
                "ik_prop": ik_prop,
                "ik_invert": ik_invert,
                "extra_prop": extra_prop,
                "con_name": con_name,
            }
        )
        driven_def_bones.add(def_name)
        return bname

    # --- 1. CTRL_Root ---
    is_humanoid = spec["archetype"] == "humanoid"
    root_def = spec["root"]
    root_pos = def_rest[root_def][0] if (root_def and root_def in def_rest) else Vector((0.0, 0.0, 0.0))
    ctrl_root = _new_bone(
        "CTRL_Root",
        Vector((root_pos.x, root_pos.y, 0.0)),
        Vector((root_pos.x, root_pos.y, 0.0)) + fwd_vec * max(0.35, rig_h * 0.30),
        None,
        colls=(body_coll,),
    )
    ctrl_styles["CTRL_Root"] = {
        "widget": "root", "scale": max(0.38, rig_h * 0.28) if is_humanoid else max(0.55, rig_h * 0.42), "side": "C",
        "is_root": True, "is_detail": False, "category": "body", "lock_loc": False,
    }
    if root_def and root_def in def_rest:
        _make_bind_child(root_def, ctrl_root, "hfw_rig_spine", "TRANSFORM")
        bake_fk_pairs.append({"ctrl": "CTRL_Root", "def_bone": root_def, "mode": "TRANSFORM"})
        core_def_bones.add(root_def)
    if spec["ground"] and spec["ground"] in def_rest and spec["ground"] != root_def:
        _make_bind_child(spec["ground"], ctrl_root, "hfw_rig_spine", "TRANSFORM")
        core_def_bones.add(spec["ground"])

    # --- 2. CTRL_COG (Hips / Center of Gravity) ---
    hips_def = spec["hips"]
    ctrl_cog = ctrl_root
    if hips_def and hips_def in def_rest:
        h_pos = def_rest[hips_def][0]
        ctrl_cog = _new_bone(
            "CTRL_COG",
            h_pos,
            h_pos + fwd_vec * max(0.24, rig_h * 0.18),
            ctrl_root,
            colls=(body_coll,),
        )
        ctrl_styles["CTRL_COG"] = {
            "widget": "cog",
            "scale": _fit_scale(
                hips_def,
                max(0.24, rig_h * 0.16) if is_humanoid else max(0.38, rig_h * 0.28),
                min_s=0.22 if is_humanoid else 0.30,
                max_s=0.40 if is_humanoid else max(1.45, rig_h * 0.34),
                mult=1.22,
            ),
            "side": "C",
            "is_root": True, "is_detail": False, "category": "body", "lock_loc": False, "rot_mode": "XYZ",
        }
        _make_bind_child(hips_def, ctrl_cog, "hfw_rig_spine", "TRANSFORM")
        bake_fk_pairs.append({"ctrl": "CTRL_COG", "def_bone": hips_def, "mode": "TRANSFORM"})
        core_def_bones.add(hips_def)

    def_to_driver_eb: dict[str, Any] = {}
    if root_def:
        def_to_driver_eb[root_def] = ctrl_root
    if hips_def:
        def_to_driver_eb[hips_def] = ctrl_cog

    def _find_parent_driver(def_name: str, default_eb: Any) -> Any:
        eb = ebones.get(def_name)
        cur = eb.parent if eb else None
        while cur is not None:
            if cur.name in def_to_driver_eb:
                return def_to_driver_eb[cur.name]
            cur = cur.parent
        return default_eb

    # Helper to attach dual-space `MCH_Follow_<bname>` to EVERY spine/chest/wing/limb segment so ALL child controls
    # (shoulders, wings, neck, head, dorsal spikes, plasma intakes, armor plates, toes, fingers) track BOTH IK & FK!
    def _make_dual_follow_socket(
        bname: str,
        fk_bind: str,
        ik_bind: str,
        ik_prop: str,
        *,
        extra_prop: str | None = None,
    ) -> Any:
        bh, bt, bm, bl = def_rest[bname]
        mf = _new_bone(_unique_name(f"MCH_Follow_{bname}"), bh, bt, ctrl_root, matrix=bm, colls=(mch_coll,))
        mf.length = max(0.02, bl)
        follow_setups.append(
            {
                "follow_mch": mf.name,
                "fk_mch": fk_bind,
                "ik_mch": ik_bind,
                "ik_prop": ik_prop,
                "extra_prop": extra_prop,
            }
        )
        def_to_driver_eb[bname] = mf
        return mf

    # Helper: Cycle-Free Blender-Studio B-Bone Spline IK overlaid cleanly on any long FK chain (`hfw_ik_fk_spline`)!
    def _build_bbone_spline_for_chain(
        chain_name: str,
        chain_bones: list[str],
        fk_binds_map: dict[str, str],
        *,
        anchor_parent: Any,
        knot_parent: Any,
        grp_prop: str,
        num_knots: int,
        side: str = "C",
    ) -> list[str]:
        n_b = len(chain_bones)
        if n_b < 5:
            return []
        k_cnt = max(3, min(num_knots, n_b))
        knot_idxs = sorted({int(round(i * (n_b - 1) / (k_cnt - 1))) for i in range(k_cnt)})
        k_cnt = len(knot_idxs)
        knot_ctrls: list[Any] = []
        knot_ctrl_names: list[str] = []
        knot_pts = [def_rest[chain_bones[ki]][0] for ki in knot_idxs]

        # Build clean, bold 3D gimbal Spline Knot handles (`CTRL_Spline_<name>_00..K-1`)
        for ki_pos, bi in enumerate(knot_idxs):
            bname = chain_bones[bi]
            p_cur = knot_pts[ki_pos]
            if ki_pos + 1 < k_cnt:
                tang = knot_pts[ki_pos + 1] - p_cur
            else:
                tang = p_cur - knot_pts[ki_pos - 1]
            if tang.length < 1e-4:
                tang = -fwd_vec
            tang = tang.normalized()
            seg_len_hint = (
                (knot_pts[min(ki_pos + 1, k_cnt - 1)] - knot_pts[max(ki_pos - 1, 0)]).length * 0.14
            )
            c_len = max(0.18, min(1.2, seg_len_hint))
            par_eb = anchor_parent if ki_pos == 0 else knot_parent
            kn_nm = _unique_name(f"CTRL_Spline_{chain_name}_{ki_pos:02d}")
            k_eb = _new_bone(kn_nm, p_cur, p_cur + tang * c_len, par_eb, colls=(spline_ik_coll or body_coll,))
            sc_fit = _fit_scale(
                bname,
                max(0.28, rig_h * 0.22),
                min_s=0.18 if not is_humanoid else 0.12,
                max_s=max(1.45, rig_h * 0.45),
                mult=1.22,
            )
            ctrl_styles[kn_nm] = {
                "widget": "spline",
                "scale": sc_fit,
                "side": side,
                "is_root": False,
                "is_detail": False,
                "category": "spline",
                "lock_loc": False,
                "rot_mode": "XYZ",
            }
            knot_ctrls.append(k_eb)
            knot_ctrl_names.append(kn_nm)
            spline_ctrls.append(kn_nm)

        # Build K-1 Bendy-Bone cubic Bézier bridge segments (`MCH_SplineSeg_*`) with exact-rest tangent handle children -> 0.000mm rest drift!
        seg_ebs: list[Any] = []
        for si in range(k_cnt - 1):
            p0, p1 = knot_pts[si], knot_pts[si + 1]
            seg_nm = _unique_name(f"MCH_SplineSeg_{chain_name}_{si:02d}")
            s_eb = _new_bone(seg_nm, p0, p1, knot_ctrls[si], colls=(mch_coll,))
            s_eb.bbone_segments = 16
            s_eb.bbone_handle_type_start = "TANGENT"
            s_eb.bbone_handle_type_end = "TANGENT"
            s_eb.bbone_easein = 1.0
            s_eb.bbone_easeout = 1.0
            s_mat3 = s_eb.matrix.to_3x3().to_4x4()
            s_dir = (p1 - p0).normalized() * max(0.08, (p1 - p0).length * 0.12)
            h0_eb = _new_bone(
                _unique_name(f"MCH_SplineH0_{chain_name}_{si:02d}"),
                p0,
                p0 + s_dir,
                knot_ctrls[si],
                matrix=Matrix.Translation(p0) @ s_mat3,
                colls=(mch_coll,),
            )
            h1_eb = _new_bone(
                _unique_name(f"MCH_SplineH1_{chain_name}_{si:02d}"),
                p1,
                p1 + s_dir,
                knot_ctrls[si + 1],
                matrix=Matrix.Translation(p1) @ s_mat3,
                colls=(mch_coll,),
            )
            s_eb.bbone_custom_handle_start = h0_eb
            s_eb.bbone_custom_handle_end = h1_eb
            seg_ebs.append(s_eb)
            spline_setups.append(
                {
                    "type": "seg",
                    "bone": seg_nm,
                    "end_ctrl": knot_ctrl_names[si + 1],
                }
            )

        # Sample EVERY bone in `chain_bones` cleanly along its B-Bone bridge segment (rigid scale = 1.0, exact rest bind offset!)
        for bi, bname in enumerate(chain_bones):
            # Locate segment `si` such that knot_idxs[si] <= bi <= knot_idxs[si+1]
            si = 0
            for cand_s in range(k_cnt - 1):
                if knot_idxs[cand_s] <= bi <= knot_idxs[cand_s + 1]:
                    si = cand_s
                    break
            i0, i1 = knot_idxs[si], knot_idxs[si + 1]
            p0, p1 = knot_pts[si], knot_pts[si + 1]
            seg_vec = p1 - p0
            seg_len2 = max(1e-6, seg_vec.length_squared)
            p_b = def_rest[bname][0]
            # Exact projection `t_i` along straight rest B-Bone segment (handles non-uniform bone spacing!)
            t_i = max(0.0, min(1.0, (p_b - p0).dot(seg_vec) / seg_len2)) if i1 > i0 else 0.0
            p_on_seg = p0.lerp(p1, t_i)
            s_mat3 = seg_ebs[si].matrix.to_3x3().to_4x4()
            s_dir = seg_vec.normalized() * 0.12 if seg_vec.length > 1e-4 else -fwd_vec * 0.12

            pt_nm = _unique_name(f"MCH_SplinePt_{chain_name}_{bi:02d}")
            _new_bone(pt_nm, p_on_seg, p_on_seg + s_dir, ctrl_root, matrix=Matrix.Translation(p_on_seg) @ s_mat3, colls=(mch_coll,))
            rg_nm = _unique_name(f"MCH_SplineRigid_{chain_name}_{bi:02d}")
            rg_eb = _new_bone(rg_nm, p_on_seg, p_on_seg + s_dir, ctrl_root, matrix=Matrix.Translation(p_on_seg) @ s_mat3, colls=(mch_coll,))

            spline_setups.append(
                {
                    "type": "sample",
                    "pt_bone": pt_nm,
                    "rigid_bone": rg_nm,
                    "seg_bone": seg_ebs[si].name,
                    "head_tail": float(t_i),
                }
            )
            ik_b = _make_bind_child(
                bname,
                rg_eb,
                grp_prop,
                "TRANSFORM",
                prefix="MCH_BindSpline_",
                ik_prop="hfw_ik_fk_spline",
                ik_invert=False,
                con_name="HFW_Spline_Xform",
            )
            if bname in fk_binds_map:
                _make_dual_follow_socket(bname, fk_binds_map[bname], ik_b, "hfw_ik_fk_spline")
            else:
                def_to_driver_eb[bname] = rg_eb

        spline_ik_meta.append(
            {
                "name": chain_name,
                "kind": "spine" if grp_prop == "hfw_rig_spine" else "tail",
                "knots": [
                    {"ctrl": knot_ctrl_names[ki_pos], "def_bone": chain_bones[bi]}
                    for ki_pos, bi in enumerate(knot_idxs)
                ],
                "bones": list(chain_bones),
            }
        )
        return knot_ctrl_names

    # --- 3. Multi-Bone Chest / Back IK <-> Pure Spine FK (AND Serpentine Cobra Hood IK + B-Bone Spline IK!) ---
    spine_full = [b for b in spec["spine"] if b in def_rest]
    neck_full = [b for b in spec["neck"] if b in def_rest]
    chest_def = spec.get("chest")
    use_spine_ik = bool(torso_ik_mode == "spine" and spine_full and chest_def and chest_def in def_rest)
    use_hood_ik = bool(torso_ik_mode == "hood" and chest_def and chest_def in def_rest and len(neck_full) >= 4)
    use_torso_ik = use_spine_ik or use_hood_ik
    spine_parent_eb = ctrl_cog
    chest_ctrl_name = ""
    skip_standard_neck = False

    def _build_multi_bone_ik_block(
        chain_bones: list[str],
        tip_bone: str,
        *,
        anchor_parent: Any,
        grp_prop: str,
        fk_coll: Any,
        fk_prefix: str,
        fk_ctrl_list: list[str],
        nxt_after_tip: str | None,
    ) -> Any:
        nonlocal chest_ctrl_name, torso_ik_meta
        fk_t_parent = anchor_parent
        fk_binds_t: dict[str, str] = {}
        torso_fk_pairs: list[dict[str, str]] = []
        torso_full = list(chain_bones) + [tip_bone]
        for idx, tb in enumerate(torso_full):
            core_def_bones.add(tb)
            h_pos = def_rest[tb][0]
            nxt_b = torso_full[idx + 1] if (idx + 1 < len(torso_full)) else nxt_after_tip
            if nxt_b and nxt_b in def_rest and (def_rest[nxt_b][0] - h_pos).length > 0.02:
                t_pos = def_rest[nxt_b][0]
            else:
                t_pos = h_pos + Vector((0.0, 0.0, max(0.14, rig_h * 0.10)))
            is_ch = idx == len(torso_full) - 1
            fk_nm = "CTRL_FK_Chest" if is_ch else f"{fk_prefix}_{idx:02d}"
            fk_eb = _new_bone(fk_nm, h_pos, t_pos, fk_t_parent, colls=(fk_coll or body_coll,))
            ctrl_styles[fk_nm] = {
                "widget": "chest" if is_ch else "fk_ring",
                "scale": (
                    _fit_scale(
                        tb,
                        max(0.21, rig_h * 0.14) if is_humanoid else max(0.36, rig_h * 0.27),
                        min_s=0.18 if is_humanoid else 0.28,
                        max_s=0.38 if is_humanoid else max(1.15, rig_h * 0.34),
                        mult=1.18,
                    )
                    if is_ch
                    else _fit_scale(
                        tb,
                        max(0.17, rig_h * 0.115) if is_humanoid else max(0.28, rig_h * 0.22),
                        min_s=0.14 if is_humanoid else 0.22,
                        max_s=0.32 if is_humanoid else max(0.95, rig_h * 0.28),
                        mult=1.16,
                    )
                ),
                "side": "C",
                "is_root": False,
                "is_detail": False,
                "category": "chest" if is_ch else "body",
                "lock_loc": True,
                "rot_mode": "XYZ",
            }
            fk_binds_t[tb] = _make_bind_child(
                tb, fk_eb, grp_prop, "TRANSFORM",
                prefix="MCH_BindFK_", ik_prop="hfw_ik_fk_torso", ik_invert=True, con_name="HFW_FK_Xform",
            )
            pair = {"ctrl": fk_nm, "def_bone": tb, "mode": "ROTATION"}
            bake_fk_pairs.append(pair)
            torso_fk_pairs.append(pair)
            fk_ctrl_list.append(fk_nm)
            fk_t_parent = fk_eb
        limb_fk_chains["torso"].append({"name": "Torso", "fk_pairs": torso_fk_pairs})

        # Multi-Bone Chest / Cobra Hood IK Solver (`CTRL_Chest` + `CTRL_Spine_Mid`)
        # Moving (G) OR Rotating (R) `CTRL_Chest` arches the ENTIRE back or cobra hood smoothly!
        pts_t = [def_rest[b][0] for b in chain_bones] + [def_rest[tip_bone][0]]
        ch_pos = pts_t[-1]
        if nxt_after_tip and nxt_after_tip in def_rest and (def_rest[nxt_after_tip][0] - ch_pos).length > 0.02:
            ch_tail = def_rest[nxt_after_tip][0]
        else:
            ch_tail = ch_pos + Vector((0.0, 0.0, max(0.18, rig_h * 0.14)))
        ctrl_chest = _new_bone("CTRL_Chest", ch_pos, ch_tail, anchor_parent, colls=(torso_ik_coll or body_coll,))
        ctrl_styles["CTRL_Chest"] = {
            "widget": "chest",
            "scale": _fit_scale(
                tip_bone,
                max(0.21, rig_h * 0.14) if is_humanoid else max(0.36, rig_h * 0.27),
                min_s=0.18 if is_humanoid else 0.28,
                max_s=0.38 if is_humanoid else max(1.25, rig_h * 0.36),
                mult=1.18,
            ),
            "side": "C",
            "is_root": False,
            "is_detail": False,
            "category": "chest",
            "lock_loc": False,  # UNLOCKED so Move (G) & Rotate (R) arch the entire back/hood!
            "rot_mode": "XYZ",
        }
        chest_ctrl_name = "CTRL_Chest"

        k_sp = len(chain_bones)
        mch_t_chain: list[Any] = []
        ik_binds_t: dict[str, str] = {}
        for idx, sb in enumerate(chain_bones):
            m_eb = _new_bone(
                f"MCH_TorsoIK_{idx}",
                pts_t[idx],
                pts_t[idx + 1],
                anchor_parent if idx == 0 else mch_t_chain[-1],
                colls=(mch_coll,),
                use_connect=(idx > 0),
            )
            mch_t_chain.append(m_eb)
            ik_binds_t[sb] = _make_bind_child(
                sb, m_eb, grp_prop, "TRANSFORM",
                prefix="MCH_BindIK_", ik_prop="hfw_ik_fk_torso", ik_invert=False,
                extra_prop="hfw_rig_chest", con_name="HFW_IK_Xform",
            )

        mch_t_chest = _new_bone(
            "MCH_TorsoIK_Chest",
            ch_pos,
            ch_tail,
            mch_t_chain[-1],
            matrix=ctrl_chest.matrix.copy(),
            colls=(mch_coll,),
        )
        extra_constraints.append(
            {"bone": "MCH_TorsoIK_Chest", "type": "COPY_ROTATION", "target": "CTRL_Chest", "influence": 1.0}
        )
        ik_binds_t[tip_bone] = _make_bind_child(
            tip_bone, mch_t_chest, grp_prop, "TRANSFORM",
            prefix="MCH_BindIK_", ik_prop="hfw_ik_fk_torso", ik_invert=False,
            extra_prop="hfw_rig_chest", con_name="HFW_IK_Xform",
        )

        # Exact-rest-aligned upper IK target child of CTRL_Chest (guarantees 0.000mm / 0.000° rest drift!)
        up_mat = Matrix.Translation(pts_t[-1]) @ mch_t_chain[-1].matrix.to_3x3().to_4x4()
        _new_bone(
            "MCH_TorsoIK_UpperTarget",
            pts_t[-1],
            pts_t[-1] + (pts_t[-1] - pts_t[-2]).normalized() * 0.12,
            ctrl_chest,
            matrix=up_mat,
            colls=(mch_coll,),
        )

        mid_ctrl_name = ""
        mid_def_name = ""
        if k_sp >= 2:
            mid_idx = k_sp // 2
            mid_def_name = chain_bones[mid_idx]
            mid_pos = pts_t[mid_idx]
            mid_rot4 = mch_t_chain[mid_idx - 1].matrix.to_3x3().to_4x4()
            mid_mat = Matrix.Translation(mid_pos) @ mid_rot4
            mid_dir = (pts_t[mid_idx] - pts_t[mid_idx - 1]).normalized() * 0.12

            _new_bone("MCH_COG_MidRef", mid_pos, mid_pos + mid_dir, anchor_parent, matrix=mid_mat, colls=(mch_coll,))
            _new_bone("MCH_Chest_MidRef", mid_pos, mid_pos + mid_dir, ctrl_chest, matrix=mid_mat, colls=(mch_coll,))
            m_anc = _new_bone("MCH_TorsoMidAnchor", mid_pos, mid_pos + mid_dir, ctrl_root, matrix=mid_mat, colls=(mch_coll,))
            extra_constraints.append({"bone": m_anc.name, "type": "COPY_TRANSFORMS", "target": "MCH_COG_MidRef", "influence": 1.0})
            extra_constraints.append({"bone": m_anc.name, "type": "COPY_TRANSFORMS", "target": "MCH_Chest_MidRef", "influence": 0.5})

            ctrl_mid = _new_bone(
                "CTRL_Spine_Mid",
                mid_pos,
                pts_t[mid_idx + 1] if (mid_idx + 1 < len(pts_t)) else (mid_pos + mid_dir),
                m_anc,
                colls=(torso_ik_coll or body_coll,),
            )
            ctrl_styles["CTRL_Spine_Mid"] = {
                "widget": "diamond",
                "scale": max(0.13, rig_h * 0.085) if is_humanoid else max(0.22, rig_h * 0.15),
                "side": "C",
                "is_root": False,
                "is_detail": False,
                "category": "chest",
                "lock_loc": False,
            }
            mid_ctrl_name = "CTRL_Spine_Mid"
            _new_bone("MCH_TorsoIK_LowerTarget", mid_pos, mid_pos + mid_dir, ctrl_mid, matrix=mid_mat, colls=(mch_coll,))

            ik_setups.append(
                {
                    "mch_solver": mch_t_chain[mid_idx - 1].name,
                    "mch_chain": [m.name for m in mch_t_chain[:mid_idx]],
                    "target_ctrl": "MCH_TorsoIK_LowerTarget",
                    "pole_ctrl": "",
                    "chain_len": mid_idx,
                    "pole_angle": 0.0,
                    "use_rotation": True,
                    "orient_weight": 0.50,
                }
            )
            ik_setups.append(
                {
                    "mch_solver": mch_t_chain[-1].name,
                    "mch_chain": [m.name for m in mch_t_chain[mid_idx:]],
                    "target_ctrl": "MCH_TorsoIK_UpperTarget",
                    "pole_ctrl": "",
                    "chain_len": k_sp - mid_idx,
                    "pole_angle": 0.0,
                    "use_rotation": True,
                    "orient_weight": 0.85,
                }
            )
        else:
            ik_setups.append(
                {
                    "mch_solver": mch_t_chain[0].name,
                    "mch_chain": [mch_t_chain[0].name],
                    "target_ctrl": "MCH_TorsoIK_UpperTarget",
                    "pole_ctrl": "",
                    "chain_len": 1,
                    "pole_angle": 0.0,
                    "use_rotation": True,
                    "orient_weight": 0.85,
                }
            )

        for tb in torso_full:
            _make_dual_follow_socket(tb, fk_binds_t[tb], ik_binds_t[tb], "hfw_ik_fk_torso", extra_prop="hfw_rig_chest")
        torso_ik_meta = {
            "mode": torso_ik_mode,
            "chest_ctrl": "CTRL_Chest",
            "chest_def": tip_bone,
            "mid_ctrl": mid_ctrl_name,
            "mid_def": mid_def_name,
            "mid_anchor": "MCH_TorsoMidAnchor" if mid_ctrl_name else "",
        }
        return def_to_driver_eb[tip_bone]

    if use_spine_ik:
        spine_parent_eb = _build_multi_bone_ik_block(
            spine_full,
            chest_def,
            anchor_parent=ctrl_cog,
            grp_prop="hfw_rig_spine",
            fk_coll=spine_fk_coll,
            fk_prefix="CTRL_FK_Spine",
            fk_ctrl_list=spine_ctrls,
            nxt_after_tip=spec["neck"][0] if spec["neck"] else spec["head"],
        )
    else:
        # Pure FK Spine (AND B-Bone Spline IK on Slitherfang's 40-bone serpent body!)
        spine_spline_spec = next((sc for sc in spline_chains_spec if sc.get("kind") == "spine"), None)
        spine_handles = _sample_chain_handles(spine_full, max_handles=14)
        spine_fk_binds: dict[str, str] = {}
        spine_fk_pairs: list[dict[str, str]] = []
        for i, sb in enumerate(spine_full):
            core_def_bones.add(sb)
            if sb in spine_handles:
                h_pos = def_rest[sb][0]
                nxt_b = (
                    spine_full[i + 1]
                    if (i + 1 < len(spine_full))
                    else (chest_def if not use_hood_ik else None)
                )
                if nxt_b and nxt_b in def_rest and (def_rest[nxt_b][0] - h_pos).length > 0.02:
                    t_pos = def_rest[nxt_b][0]
                else:
                    t_pos = h_pos + Vector((0.0, 0.0, max(0.14, rig_h * 0.10)))
                cname = f"CTRL_Spine_{spine_handles.index(sb):02d}"
                c_eb = _new_bone(cname, h_pos, t_pos, spine_parent_eb, colls=(spine_fk_coll or body_coll,))
                ctrl_styles[cname] = {
                    "widget": "fk_ring",
                    "scale": _fit_scale(
                        sb,
                        max(0.17, rig_h * 0.115) if is_humanoid else max(0.30, rig_h * 0.24),
                        min_s=0.14 if is_humanoid else 0.24,
                        max_s=0.32 if is_humanoid else max(1.35, rig_h * 0.35),
                        mult=1.18,
                    ),
                    "side": "C",
                    "is_root": False,
                    "is_detail": False,
                    "category": "body",
                    "lock_loc": True,
                    "rot_mode": "XYZ",
                }
                if spine_spline_spec is not None:
                    spine_fk_binds[sb] = _make_bind_child(
                        sb, c_eb, "hfw_rig_spine", "TRANSFORM",
                        prefix="MCH_BindFK_", ik_prop="hfw_ik_fk_spline", ik_invert=True, con_name="HFW_FK_Xform",
                    )
                else:
                    _make_bind_child(sb, c_eb, "hfw_rig_spine", "ROTATION")
                pair = {"ctrl": cname, "def_bone": sb, "mode": "ROTATION"}
                bake_fk_pairs.append(pair)
                spine_fk_pairs.append(pair)
                spine_ctrls.append(cname)
                spine_parent_eb = c_eb
                def_to_driver_eb[sb] = c_eb
            else:
                if spine_spline_spec is not None:
                    spine_fk_binds[sb] = _make_bind_child(
                        sb, spine_parent_eb, "hfw_rig_spine", "TRANSFORM",
                        prefix="MCH_BindFK_", ik_prop="hfw_ik_fk_spline", ik_invert=True, con_name="HFW_FK_Xform",
                    )
                def_to_driver_eb[sb] = spine_parent_eb

        if spine_spline_spec is not None:
            limb_fk_chains["spline"].append({"name": spine_spline_spec["name"], "fk_pairs": spine_fk_pairs})
            _build_bbone_spline_for_chain(
                spine_spline_spec["name"],
                spine_full,
                spine_fk_binds,
                anchor_parent=ctrl_cog,
                knot_parent=ctrl_root,
                grp_prop="hfw_rig_spine",
                num_knots=int(spine_spline_spec.get("knots", 7)),
                side="C",
            )

        if use_hood_ik:
            # Serpentine Cobra Hood IK <-> Hood FK across `chest_def -> neck_0..N-1 -> neck_N` (12-bone arching cobra hood!)
            hood_anchor = _find_parent_driver(chest_def, ctrl_cog)
            spine_parent_eb = _build_multi_bone_ik_block(
                [chest_def] + neck_full[:-1],
                neck_full[-1],
                anchor_parent=hood_anchor,
                grp_prop="hfw_rig_head",
                fk_coll=hood_fk_coll,
                fk_prefix="CTRL_FK_Hood",
                fk_ctrl_list=neck_ctrls,
                nxt_after_tip=spec["head"],
            )
            skip_standard_neck = True
        elif chest_def and chest_def in def_rest:
            core_def_bones.add(chest_def)
            ch_pos = def_rest[chest_def][0]
            ch_parent = _find_parent_driver(chest_def, spine_parent_eb)
            nxt_ch = spec["neck"][0] if spec["neck"] else spec["head"]
            if nxt_ch and nxt_ch in def_rest and (def_rest[nxt_ch][0] - ch_pos).length > 0.02:
                ch_tail = def_rest[nxt_ch][0]
            else:
                ch_tail = ch_pos + Vector((0.0, 0.0, max(0.18, rig_h * 0.14)))
            ctrl_chest = _new_bone("CTRL_Chest", ch_pos, ch_tail, ch_parent, colls=(body_coll,))
            ctrl_styles["CTRL_Chest"] = {
                "widget": "chest",
                "scale": _fit_scale(
                    chest_def,
                    max(0.21, rig_h * 0.14) if is_humanoid else max(0.36, rig_h * 0.27),
                    min_s=0.18 if is_humanoid else 0.28,
                    max_s=0.38 if is_humanoid else max(1.25, rig_h * 0.36),
                    mult=1.18,
                ),
                "side": "C",
                "is_root": False,
                "is_detail": False,
                "category": "chest",
                "lock_loc": True,
                "rot_mode": "XYZ",
            }
            _make_bind_child(chest_def, ctrl_chest, "hfw_rig_chest", "ROTATION")
            bake_fk_pairs.append({"ctrl": "CTRL_Chest", "def_bone": chest_def, "mode": "ROTATION"})
            spine_ctrls.append("CTRL_Chest")
            def_to_driver_eb[chest_def] = ctrl_chest
            spine_parent_eb = ctrl_chest
            chest_ctrl_name = "CTRL_Chest"

    # --- 4. Neck, Head (with Toggleable Head/Neck IK + Chest/World Space Switch!) & Jaws ---
    head_def = spec["head"]
    use_head_ik = bool(has_head_ik_spec and not skip_standard_neck and neck_full and head_def and head_def in def_rest)
    if skip_standard_neck:
        neck_parent_eb = spine_parent_eb
        ctrl_head = neck_parent_eb
        if head_def and head_def in def_rest:
            core_def_bones.add(head_def)
            h_pos = def_rest[head_def][0]
            h_parent = _find_parent_driver(head_def, neck_parent_eb)
            ctrl_head = _new_bone("CTRL_Head", h_pos, h_pos + fwd_vec * max(0.20, rig_h * 0.16), h_parent, colls=(body_coll,))
            ctrl_styles["CTRL_Head"] = {
                "widget": "box",
                "scale": _fit_scale(head_def, max(0.24, rig_h * 0.16), 0.16, max(1.15, rig_h * 0.30), 1.15),
                "side": "C", "is_root": False, "is_detail": False, "category": "body", "lock_loc": True, "rot_mode": "XYZ",
            }
            _make_bind_child(head_def, ctrl_head, "hfw_rig_head", "ROTATION")
            bake_fk_pairs.append({"ctrl": "CTRL_Head", "def_bone": head_def, "mode": "ROTATION"})
            def_to_driver_eb[head_def] = ctrl_head
    elif use_head_ik:
        # 4a. Dual FK Neck+Head (`CTRL_Neck_00..N`, `CTRL_Head`) <-> Pole-Free Soft-Stiffness Head/Neck IK (`CTRL_IK_Head`)!
        neck_anchor = _find_parent_driver(neck_full[0], spine_parent_eb)
        neck_handles = _sample_chain_handles(neck_full, max_handles=8)
        fk_np = neck_anchor
        fk_binds_h: dict[str, str] = {}
        head_fk_pairs: list[dict[str, str]] = []
        for i, nb in enumerate(neck_full):
            core_def_bones.add(nb)
            if nb in neck_handles:
                h_pos = def_rest[nb][0]
                nxt_b = neck_full[i + 1] if (i + 1 < len(neck_full)) else head_def
                t_pos = def_rest[nxt_b][0] if (nxt_b in def_rest and (def_rest[nxt_b][0] - h_pos).length > 0.02) else (h_pos + Vector((0.0, 0.0, max(0.10, rig_h * 0.08))))
                cname = f"CTRL_Neck_{neck_handles.index(nb):02d}"
                c_eb = _new_bone(cname, h_pos, t_pos, fk_np, colls=(body_coll,))
                ctrl_styles[cname] = {
                    "widget": "fk_ring",
                    "scale": _fit_scale(
                        nb,
                        max(0.11, rig_h * 0.075) if is_humanoid else max(0.20, rig_h * 0.12),
                        min_s=0.09 if is_humanoid else 0.16,
                        max_s=0.22 if is_humanoid else max(0.72, rig_h * 0.18),
                        mult=1.18,
                    ),
                    "side": "C", "is_root": False, "is_detail": False, "category": "body", "lock_loc": True, "rot_mode": "XYZ",
                }
                fk_binds_h[nb] = _make_bind_child(
                    nb, c_eb, "hfw_rig_head", "TRANSFORM",
                    prefix="MCH_BindFK_", ik_prop="hfw_ik_fk_head", ik_invert=True, con_name="HFW_FK_Xform",
                )
                pair = {"ctrl": cname, "def_bone": nb, "mode": "ROTATION"}
                bake_fk_pairs.append(pair)
                head_fk_pairs.append(pair)
                neck_ctrls.append(cname)
                fk_np = c_eb
            else:
                fk_binds_h[nb] = _make_bind_child(
                    nb, fk_np, "hfw_rig_head", "TRANSFORM",
                    prefix="MCH_BindFK_", ik_prop="hfw_ik_fk_head", ik_invert=True, con_name="HFW_FK_Xform",
                )

        core_def_bones.add(head_def)
        hd_pos = def_rest[head_def][0]
        hd_tail = hd_pos + fwd_vec * max(0.18, rig_h * 0.14)
        ctrl_head_fk = _new_bone("CTRL_Head", hd_pos, hd_tail, fk_np, colls=(body_coll,))
        hd_sc = _fit_scale(
            head_def,
            max(0.15, rig_h * 0.095) if is_humanoid else max(0.24, rig_h * 0.13),
            min_s=0.13 if is_humanoid else 0.20,
            max_s=0.22 if is_humanoid else min(0.95, max(0.42, rig_h * 0.16)),
            mult=1.08,
        )
        ctrl_styles["CTRL_Head"] = {
            "widget": "box", "scale": hd_sc, "side": "C",
            "is_root": False, "is_detail": False, "category": "body", "lock_loc": True, "rot_mode": "XYZ",
        }
        fk_binds_h[head_def] = _make_bind_child(
            head_def, ctrl_head_fk, "hfw_rig_head", "TRANSFORM",
            prefix="MCH_BindFK_", ik_prop="hfw_ik_fk_head", ik_invert=True, con_name="HFW_FK_Xform",
        )
        hpair = {"ctrl": "CTRL_Head", "def_bone": head_def, "mode": "ROTATION"}
        bake_fk_pairs.append(hpair)
        head_fk_pairs.append(hpair)
        limb_fk_chains["head"].append({"name": "Head", "fk_pairs": head_fk_pairs})

        # 4b. Head IK Target with Chest <-> World Space Switch (`hfw_head_ik_space`) + Pole-Free Soft-Stiffness Neck IK!
        m_ch_ref = _new_bone("MCH_HeadIK_ChestRef", hd_pos, hd_tail, neck_anchor, colls=(mch_coll,))
        m_rt_ref = _new_bone("MCH_HeadIK_RootRef", hd_pos, hd_tail, ctrl_root, colls=(mch_coll,))
        m_hd_par = _new_bone("MCH_HeadIK_Parent", hd_pos, hd_tail, ctrl_root, matrix=m_ch_ref.matrix.copy(), colls=(mch_coll,))
        extra_constraints.append({"bone": m_hd_par.name, "type": "COPY_TRANSFORMS", "target": m_ch_ref.name, "influence": 1.0})
        extra_constraints.append({
            "bone": m_hd_par.name, "type": "COPY_TRANSFORMS", "target": m_rt_ref.name,
            "influence": 0.0, "driver_prop": "hfw_head_ik_space",
        })

        ctrl_head_ik = _new_bone("CTRL_IK_Head", hd_pos, hd_tail, m_hd_par, colls=(head_ik_coll or body_coll,))
        ctrl_styles["CTRL_IK_Head"] = {
            "widget": "head_ik", "scale": hd_sc * 0.96, "side": "C",
            "is_root": False, "is_detail": False, "category": "head_ik", "lock_loc": False, "rot_mode": "XYZ",
        }

        # Build exact zero-drift Neck IK chain: skip coincident (<1.5cm) neck joints (e.g. Stormbird C_Neck_sjnt_3 == C_Head_sjnt_0!)
        nk_seg_bones: list[tuple[str, Vector, Vector]] = []
        nk_coincident_head: list[str] = []
        nk_coincident_prev: list[tuple[str, int]] = []
        for idx, nb in enumerate(neck_full):
            p_cur = def_rest[nb][0]
            # Find next distinct position along neck -> head
            p_nxt = hd_pos
            for j in range(idx + 1, len(neck_full)):
                cand_p = def_rest[neck_full[j]][0]
                if (cand_p - p_cur).length >= 0.015:
                    p_nxt = cand_p
                    break
            if (p_nxt - p_cur).length < 0.015:
                if (hd_pos - p_cur).length < 0.015:
                    nk_coincident_head.append(nb)
                else:
                    nk_coincident_prev.append((nb, max(0, len(nk_seg_bones) - 1)))
            else:
                nk_seg_bones.append((nb, p_cur, p_nxt))
        if not nk_seg_bones:
            nk_seg_bones = [(neck_full[0], def_rest[neck_full[0]][0], hd_pos)]

        mch_nk_chain: list[Any] = []
        ik_binds_h: dict[str, str] = {}
        for idx, (nb, p0_n, p1_n) in enumerate(nk_seg_bones):
            m_eb = _new_bone(
                f"MCH_NeckIK_{idx}",
                p0_n,
                p1_n,
                neck_anchor if idx == 0 else mch_nk_chain[-1],
                colls=(mch_coll,),
                use_connect=(idx > 0),
            )
            mch_nk_chain.append(m_eb)
            ik_binds_h[nb] = _make_bind_child(
                nb, m_eb, "hfw_rig_head", "TRANSFORM",
                prefix="MCH_BindIK_", ik_prop="hfw_ik_fk_head", ik_invert=False, con_name="HFW_IK_Xform",
            )
        for nb_c, prev_i in nk_coincident_prev:
            ik_binds_h[nb_c] = _make_bind_child(
                nb_c, mch_nk_chain[min(prev_i, len(mch_nk_chain) - 1)], "hfw_rig_head", "TRANSFORM",
                prefix="MCH_BindIK_", ik_prop="hfw_ik_fk_head", ik_invert=False, con_name="HFW_IK_Xform",
            )

        mch_nk_head = _new_bone(
            "MCH_NeckIK_Head", hd_pos, hd_tail, mch_nk_chain[-1],
            matrix=ctrl_head_ik.matrix.copy(), colls=(mch_coll,),
        )
        extra_constraints.append({"bone": mch_nk_head.name, "type": "COPY_ROTATION", "target": "CTRL_IK_Head", "influence": 1.0})
        for nb_h in nk_coincident_head:
            ik_binds_h[nb_h] = _make_bind_child(
                nb_h, mch_nk_head, "hfw_rig_head", "TRANSFORM",
                prefix="MCH_BindIK_", ik_prop="hfw_ik_fk_head", ik_invert=False, con_name="HFW_IK_Xform",
            )
        ik_binds_h[head_def] = _make_bind_child(
            head_def, mch_nk_head, "hfw_rig_head", "TRANSFORM",
            prefix="MCH_BindIK_", ik_prop="hfw_ik_fk_head", ik_invert=False, con_name="HFW_IK_Xform",
        )

        # Exact-rest-aligned Neck IK target child of CTRL_IK_Head (0.000mm rest drift, NO pole target per Blender Studio rules!)
        nk_tgt_mat = Matrix.Translation(hd_pos) @ mch_nk_chain[-1].matrix.to_3x3().to_4x4()
        last_dir = mch_nk_chain[-1].tail - mch_nk_chain[-1].head
        nk_tgt_dir = last_dir.normalized() * 0.12 if last_dir.length > 1e-4 else Vector((0, 0, 0.12))
        _new_bone("MCH_NeckIK_Target", hd_pos, hd_pos + nk_tgt_dir, ctrl_head_ik, matrix=nk_tgt_mat, colls=(mch_coll,))
        ik_setups.append(
            {
                "mch_solver": mch_nk_chain[-1].name,
                "mch_chain": [m.name for m in mch_nk_chain],
                "target_ctrl": "MCH_NeckIK_Target",
                "pole_ctrl": "",
                "chain_len": len(mch_nk_chain),
                "pole_angle": 0.0,
                "use_rotation": True,
                "orient_weight": 0.65,
                "soft_stiffness": True,
            }
        )

        # Dual-space follow sockets on EVERY neck bone + head_def so jaws, saws, tusks, trunks & head plates follow Head IK & FK!
        for hb_n in list(neck_full) + [head_def]:
            _make_dual_follow_socket(hb_n, fk_binds_h[hb_n], ik_binds_h[hb_n], "hfw_ik_fk_head")
        ctrl_head = def_to_driver_eb[head_def]
        neck_parent_eb = def_to_driver_eb[neck_full[-1]]
        head_ik_meta = {"head_ctrl": "CTRL_IK_Head", "head_def": head_def, "neck_defs": list(neck_full)}
    else:
        neck_handles = _sample_chain_handles(neck_full, max_handles=8)
        neck_parent_eb = _find_parent_driver(neck_full[0], spine_parent_eb) if neck_full else spine_parent_eb
        for i, nb in enumerate(neck_full):
            core_def_bones.add(nb)
            if nb in neck_handles:
                h_pos = def_rest[nb][0]
                nxt_b = neck_full[i + 1] if (i + 1 < len(neck_full)) else head_def
                t_pos = def_rest[nxt_b][0] if (nxt_b and nxt_b in def_rest and (def_rest[nxt_b][0] - h_pos).length > 0.02) else (h_pos + Vector((0.0, 0.0, max(0.10, rig_h * 0.08))))
                cname = f"CTRL_Neck_{neck_handles.index(nb):02d}"
                c_eb = _new_bone(cname, h_pos, t_pos, neck_parent_eb, colls=(body_coll,))
                ctrl_styles[cname] = {
                    "widget": "fk_ring",
                    "scale": _fit_scale(nb, max(0.12, rig_h * 0.08) if is_humanoid else max(0.20, rig_h * 0.12), 0.10, max(0.72, rig_h * 0.18), 1.18),
                    "side": "C", "is_root": False, "is_detail": False, "category": "body", "lock_loc": True, "rot_mode": "XYZ",
                }
                _make_bind_child(nb, c_eb, "hfw_rig_head", "ROTATION")
                bake_fk_pairs.append({"ctrl": cname, "def_bone": nb, "mode": "ROTATION"})
                neck_ctrls.append(cname)
                neck_parent_eb = c_eb
                def_to_driver_eb[nb] = c_eb
            else:
                def_to_driver_eb[nb] = neck_parent_eb

        ctrl_head = neck_parent_eb
        if head_def and head_def in def_rest:
            core_def_bones.add(head_def)
            h_pos = def_rest[head_def][0]
            h_parent = _find_parent_driver(head_def, neck_parent_eb)
            ctrl_head = _new_bone("CTRL_Head", h_pos, h_pos + fwd_vec * max(0.20, rig_h * 0.16), h_parent, colls=(body_coll,))
            ctrl_styles["CTRL_Head"] = {
                "widget": "box",
                "scale": _fit_scale(head_def, max(0.17, rig_h * 0.11) if is_humanoid else max(0.24, rig_h * 0.15), 0.13, max(0.95, rig_h * 0.22), 1.18),
                "side": "C", "is_root": False, "is_detail": False, "category": "body", "lock_loc": True, "rot_mode": "XYZ",
            }
            _make_bind_child(head_def, ctrl_head, "hfw_rig_head", "ROTATION")
            bake_fk_pairs.append({"ctrl": "CTRL_Head", "def_bone": head_def, "mode": "ROTATION"})
            def_to_driver_eb[head_def] = ctrl_head

    for ji, jb in enumerate(spec["jaws"]):
        if jb not in def_rest:
            continue
        core_def_bones.add(jb)
        side = _extract_side(jb)
        j_pos = def_rest[jb][0]
        j_parent = _find_parent_driver(jb, ctrl_head)
        cname = _unique_name(f"CTRL_Jaw_{side}" if len(spec["jaws"]) <= 3 else f"CTRL_Jaw_{ji:02d}")
        c_jaw = _new_bone(
            cname,
            j_pos,
            j_pos + fwd_vec * max(0.12, rig_h * 0.10) - Vector((0.0, 0.0, max(0.04, rig_h * 0.04))),
            j_parent,
            colls=(body_coll,),
        )
        ctrl_styles[cname] = {
            "widget": "fk_ring",
            "scale": _fit_scale(jb, max(0.16, rig_h * 0.10), min_s=0.08, max_s=max(0.55, rig_h * 0.14), mult=1.15),
            "side": side, "is_root": False, "is_detail": False, "category": "body", "lock_loc": True, "rot_mode": "XYZ",
        }
        _make_bind_child(jb, c_jaw, "hfw_rig_jaw", "ROTATION")
        bake_fk_pairs.append({"ctrl": cname, "def_bone": jb, "mode": "ROTATION"})
        def_to_driver_eb[jb] = c_jaw

    # --- 5. Dedicated Full FK Tail Chains (+ Toggleable B-Bone Spline IK on Long Tails >= 7 joints!) ---
    for tinfo in spec["tails"]:
        t_full = [b for b in tinfo["bones"] if b in def_rest]
        t_handles = _sample_chain_handles(t_full, max_handles=16)
        side = tinfo["side"]
        t_root_anchor = _find_parent_driver(t_full[0], ctrl_cog) if t_full else ctrl_cog
        t_parent = t_root_anchor
        t_spline_spec = next((sc for sc in spline_chains_spec if sc.get("kind") == "tail" and sc.get("name") == tinfo["name"]), None)
        t_fk_binds: dict[str, str] = {}
        t_fk_pairs: list[dict[str, str]] = []
        n_h = max(1, len(t_handles))
        for i, tb in enumerate(t_full):
            core_def_bones.add(tb)
            if tb in t_handles:
                hi = t_handles.index(tb)
                frac = hi / max(1, n_h - 1) if n_h > 1 else 0.0
                h_pos = def_rest[tb][0]
                t_pos = def_rest[t_full[i + 1]][0] if (i + 1 < len(t_full)) else (h_pos - fwd_vec * max(0.20, rig_h * 0.14))
                cname = f"CTRL_{tinfo['name']}_{hi:02d}"
                c_eb = _new_bone(cname, h_pos, t_pos, t_parent, colls=(tail_coll or body_coll,))
                sc_base = max(0.24, rig_h * 0.16)
                sc_tip = max(0.11, rig_h * 0.075)
                sc_fb = sc_base * (1.0 - frac) + sc_tip * frac
                sc_val = _fit_scale(tb, sc_fb, min_s=0.10, max_s=max(0.85, sc_fb * 1.35), mult=1.18)
                ctrl_styles[cname] = {
                    "widget": "fk_ring", "scale": sc_val, "side": side,
                    "is_root": False, "is_detail": False, "category": "tail", "lock_loc": True, "rot_mode": "XYZ",
                }
                if t_spline_spec is not None:
                    t_fk_binds[tb] = _make_bind_child(
                        tb, c_eb, "hfw_rig_tail", "TRANSFORM",
                        prefix="MCH_BindFK_", ik_prop="hfw_ik_fk_spline", ik_invert=True, con_name="HFW_FK_Xform",
                    )
                else:
                    _make_bind_child(tb, c_eb, "hfw_rig_tail", "ROTATION")
                pair = {"ctrl": cname, "def_bone": tb, "mode": "ROTATION"}
                bake_fk_pairs.append(pair)
                t_fk_pairs.append(pair)
                tail_ctrls.append(cname)
                t_parent = c_eb
                def_to_driver_eb[tb] = c_eb
            else:
                if t_spline_spec is not None:
                    t_fk_binds[tb] = _make_bind_child(
                        tb, t_parent, "hfw_rig_tail", "TRANSFORM",
                        prefix="MCH_BindFK_", ik_prop="hfw_ik_fk_spline", ik_invert=True, con_name="HFW_FK_Xform",
                    )
                def_to_driver_eb[tb] = t_parent
        if t_spline_spec is not None and t_full:
            limb_fk_chains["spline"].append({"name": tinfo["name"], "fk_pairs": t_fk_pairs})
            _build_bbone_spline_for_chain(
                tinfo["name"],
                t_full,
                t_fk_binds,
                anchor_parent=t_root_anchor,
                knot_parent=t_root_anchor,
                grp_prop="hfw_rig_tail",
                num_knots=int(t_spline_spec.get("knots", 5)),
                side=side,
            )

    # --- 5b. Signature Articulated Appendages (Tremortusk Trunk FK + Toggleable Trunk Spline IK, Bilegut Tongue FK!) ---
    for app in spec.get("appendages") or []:
        a_full = [b for b in app["bones"] if b in def_rest]
        a_handles = _sample_chain_handles(a_full, max_handles=16)
        side = app.get("side", "C")
        a_root_anchor = _find_parent_driver(a_full[0], ctrl_head) if a_full else ctrl_head
        a_parent = a_root_anchor
        a_spline_spec = next((sc for sc in spline_chains_spec if sc.get("kind") == "appendage" and sc.get("name") == app["name"]), None)
        a_fk_binds: dict[str, str] = {}
        a_fk_pairs: list[dict[str, str]] = []
        n_ah = max(1, len(a_handles))
        for i, ab in enumerate(a_full):
            core_def_bones.add(ab)
            if ab in a_handles:
                hi = a_handles.index(ab)
                frac = hi / max(1, n_ah - 1) if n_ah > 1 else 0.0
                h_pos = def_rest[ab][0]
                t_pos = def_rest[a_full[i + 1]][0] if (i + 1 < len(a_full)) else (h_pos + fwd_vec * max(0.18, rig_h * 0.10))
                cname = _unique_name(f"CTRL_{app['name']}_{hi:02d}")
                c_eb = _new_bone(cname, h_pos, t_pos, a_parent, colls=(app_coll or body_coll,))
                sc_fb = max(0.20, rig_h * 0.095) * (1.0 - frac * 0.5)
                sc_val = _fit_scale(ab, sc_fb, min_s=0.09, max_s=max(0.65, sc_fb * 1.3), mult=1.18)
                ctrl_styles[cname] = {
                    "widget": "fk_ring", "scale": sc_val, "side": side,
                    "is_root": False, "is_detail": False, "category": "tail", "lock_loc": True, "rot_mode": "XYZ",
                }
                if a_spline_spec is not None:
                    a_fk_binds[ab] = _make_bind_child(
                        ab, c_eb, "hfw_rig_tail", "TRANSFORM",
                        prefix="MCH_BindFK_", ik_prop="hfw_ik_fk_spline", ik_invert=True, con_name="HFW_FK_Xform",
                    )
                else:
                    _make_bind_child(ab, c_eb, "hfw_rig_tail", "ROTATION")
                pair = {"ctrl": cname, "def_bone": ab, "mode": "ROTATION"}
                bake_fk_pairs.append(pair)
                a_fk_pairs.append(pair)
                appendage_ctrls.append(cname)
                a_parent = c_eb
                def_to_driver_eb[ab] = c_eb
            else:
                if a_spline_spec is not None:
                    a_fk_binds[ab] = _make_bind_child(
                        ab, a_parent, "hfw_rig_tail", "TRANSFORM",
                        prefix="MCH_BindFK_", ik_prop="hfw_ik_fk_spline", ik_invert=True, con_name="HFW_FK_Xform",
                    )
                def_to_driver_eb[ab] = a_parent
        if a_spline_spec is not None and a_full:
            limb_fk_chains["spline"].append({"name": app["name"], "fk_pairs": a_fk_pairs})
            _build_bbone_spline_for_chain(
                app["name"],
                a_full,
                a_fk_binds,
                anchor_parent=a_root_anchor,
                knot_parent=a_root_anchor,
                grp_prop="hfw_rig_tail",
                num_knots=int(a_spline_spec.get("knots", 5)),
                side=side,
            )

    # --- 6. Flyer Wing Chains: DUAL Wing FK (Flap/Fold) + Wingtip IK (`hfw_ik_fk_wings`) + Full-Segment Follow Sockets! ---
    for winfo in spec["wings"]:
        w_full = [b for b in winfo["bones"] if b in def_rest]
        if not w_full:
            continue
        side = winfo["side"]
        wname = winfo["name"]
        w_ik_defs = [b for b in (winfo.get("ik_bones") or []) if b in def_rest]
        w_tip_def = winfo.get("tip_bone")
        w_extra_fk = [b for b in (winfo.get("extra_fk_bones") or []) if b in def_rest]
        w_root_parent = _find_parent_driver(w_full[0], spine_parent_eb)

        if len(w_ik_defs) >= 2 and w_tip_def in def_rest:
            pts_w = [def_rest[b][0] for b in w_ik_defs] + [def_rest[w_tip_def][0]]
            for b in w_ik_defs:
                core_def_bones.add(b)
            core_def_bones.add(w_tip_def)
            out_dir = Vector(((-1.0 if side == "L" else 1.0), 0.0, 0.0))

            # 6a. Wing FK Chain (`CTRL_FK_Wing_L_0..N`, `CTRL_FK_Wing_L_Tip`)
            fk_wp = w_root_parent
            fk_binds_w: dict[str, str] = {}
            wg_fk_pairs: list[dict[str, str]] = []
            for idx, bname in enumerate(w_ik_defs):
                fk_nm = f"CTRL_FK_{wname}_{idx}"
                seg_len = max(0.18, (pts_w[idx + 1] - pts_w[idx]).length)
                fk_eb = _new_bone(fk_nm, pts_w[idx], pts_w[idx + 1], fk_wp, colls=(wings_fk_coll or body_coll,))
                ctrl_styles[fk_nm] = {
                    "widget": "fk_ring",
                    "scale": _fit_scale(bname, max(0.20, rig_h * 0.14, seg_len * 0.28), min_s=0.14, max_s=max(0.68, rig_h * 0.18), mult=1.18),
                    "side": side, "is_root": False, "is_detail": False, "category": "body", "lock_loc": True, "rot_mode": "XYZ",
                }
                fk_binds_w[bname] = _make_bind_child(
                    bname, fk_eb, "hfw_rig_arms_ik", "TRANSFORM",
                    prefix="MCH_BindFK_", ik_prop="hfw_ik_fk_wings", ik_invert=True, con_name="HFW_FK_Xform",
                )
                pair = {"ctrl": fk_nm, "def_bone": bname, "mode": "ROTATION"}
                bake_fk_pairs.append(pair)
                wg_fk_pairs.append(pair)
                wing_ctrls.append(fk_nm)
                fk_wp = fk_eb

            tip_pos = def_rest[w_tip_def][0]
            tip_tail = (
                def_rest[w_extra_fk[0]][0]
                if w_extra_fk and (def_rest[w_extra_fk[0]][0] - tip_pos).length > 0.05
                else (tip_pos + out_dir * max(0.28, rig_h * 0.22))
            )
            fk_tip_nm = f"CTRL_FK_{wname}_Tip"
            fk_tip_eb = _new_bone(fk_tip_nm, tip_pos, tip_tail, fk_wp, colls=(wings_fk_coll or body_coll,))
            ctrl_styles[fk_tip_nm] = {
                "widget": "fk_ring",
                "scale": _fit_scale(w_tip_def, max(0.17, rig_h * 0.12), min_s=0.12, max_s=max(0.52, rig_h * 0.15), mult=1.18),
                "side": side, "is_root": False, "is_detail": False, "category": "body", "lock_loc": True, "rot_mode": "XYZ",
            }
            fk_binds_w[w_tip_def] = _make_bind_child(
                w_tip_def, fk_tip_eb, "hfw_rig_arms_ik", "TRANSFORM",
                prefix="MCH_BindFK_", ik_prop="hfw_ik_fk_wings", ik_invert=True, con_name="HFW_FK_Xform",
            )
            tpair = {"ctrl": fk_tip_nm, "def_bone": w_tip_def, "mode": "ROTATION"}
            bake_fk_pairs.append(tpair)
            wg_fk_pairs.append(tpair)
            wing_ctrls.append(fk_tip_nm)
            limb_fk_chains["wings"].append({"name": wname, "fk_pairs": wg_fk_pairs})

            # 6b. Wing IK Chain (`MCH_IK_Wing_L_*`, `CTRL_IK_Wing_L` wingtip box, `CTRL_Pole_Wing_L` elbow diamond!)
            mch_w_chain: list[Any] = []
            ik_binds_w: dict[str, str] = {}
            for idx, bname in enumerate(w_ik_defs):
                m_eb = _new_bone(
                    f"MCH_IK_{wname}_{idx}",
                    pts_w[idx],
                    pts_w[idx + 1],
                    w_root_parent if idx == 0 else mch_w_chain[-1],
                    colls=(mch_coll,),
                    use_connect=(idx > 0),
                )
                mch_w_chain.append(m_eb)
                ik_binds_w[bname] = _make_bind_child(
                    bname, m_eb, "hfw_rig_arms_ik", "TRANSFORM",
                    prefix="MCH_BindIK_", ik_prop="hfw_ik_fk_wings", ik_invert=False, con_name="HFW_IK_Xform",
                )

            ctrl_wtip_nm = f"CTRL_IK_{wname}"
            ctrl_wtip = _new_bone(ctrl_wtip_nm, tip_pos, tip_tail, ctrl_cog, colls=(wings_ik_coll or body_coll,))
            wing_ik_ctrls.append(ctrl_wtip_nm)
            ctrl_styles[ctrl_wtip_nm] = {
                "widget": "box",
                "scale": _fit_scale(w_tip_def, max(0.22, rig_h * 0.14), min_s=0.16, max_s=max(0.62, rig_h * 0.17), mult=1.22),
                "side": side, "is_root": False, "is_detail": False, "category": "body", "lock_loc": False, "rot_mode": "XYZ",
            }
            ik_binds_w[w_tip_def] = _make_bind_child(
                w_tip_def, ctrl_wtip, "hfw_rig_arms_ik", "TRANSFORM",
                prefix="MCH_BindIK_", ik_prop="hfw_ik_fk_wings", ik_invert=False, con_name="HFW_IK_Xform",
            )

            # Dual-space Follow sockets on EVERY wing segment so wing ribs, wing fingers, engines & wing plates track BOTH Wing IK & FK!
            for bname in list(w_ik_defs) + [w_tip_def]:
                _make_dual_follow_socket(bname, fk_binds_w[bname], ik_binds_w[bname], "hfw_ik_fk_wings")

            w_fb = -fwd_vec + Vector((0.0, 0.0, 0.35))
            w_mid_pt = pts_w[1] if len(pts_w) == 3 else (pts_w[1] * 0.6 + pts_w[2] * 0.4)
            wpole_pos, wpole_ang = _compute_pole_pos_and_angle(pts_w[0], w_mid_pt, pts_w[-1], mch_w_chain[0].matrix, w_fb.normalized(), dist_scale=0.65)
            ctrl_wpole_nm = f"CTRL_Pole_{wname}"
            _new_bone(ctrl_wpole_nm, wpole_pos, wpole_pos + w_fb.normalized() * max(0.14, rig_h * 0.09), ctrl_cog, colls=(wings_ik_coll or body_coll,))
            ctrl_styles[ctrl_wpole_nm] = {
                "widget": "diamond", "scale": max(0.16, rig_h * 0.11), "side": side,
                "is_root": False, "is_detail": False, "category": "body", "lock_loc": False,
            }
            ik_setups.append(
                {
                    "mch_solver": mch_w_chain[-1].name,
                    "mch_chain": [m.name for m in mch_w_chain],
                    "target_ctrl": ctrl_wtip_nm,
                    "pole_ctrl": ctrl_wpole_nm,
                    "chain_len": len(mch_w_chain),
                    "pole_angle": wpole_ang,
                }
            )
            bake_ik_limbs.append(
                {
                    "group": "wings",
                    "target_ctrl": ctrl_wtip_nm,
                    "pole_ctrl": ctrl_wpole_nm,
                    "end_def": w_tip_def,
                    "ik_defs": list(w_ik_defs),
                    "fallback_dir": [float(w_fb.x), float(w_fb.y), float(w_fb.z)],
                    "dist_scale": 0.65,
                }
            )

            # Extra outer wing tip FK segments riding on `MCH_Follow_<w_tip_def>`
            ex_wp = def_to_driver_eb[w_tip_def]
            for ei, wb in enumerate(w_extra_fk):
                core_def_bones.add(wb)
                h_p = def_rest[wb][0]
                t_p = def_rest[w_extra_fk[ei + 1]][0] if (ei + 1 < len(w_extra_fk)) else (h_p + out_dir * max(0.24, rig_h * 0.18))
                ex_nm = f"CTRL_FK_{wname}_Outer_{ei:02d}"
                ex_eb = _new_bone(ex_nm, h_p, t_p, ex_wp, colls=(wings_fk_coll or body_coll,))
                ctrl_styles[ex_nm] = {
                    "widget": "fk_ring", "scale": max(0.18, rig_h * 0.13), "side": side,
                    "is_root": False, "is_detail": False, "category": "body", "lock_loc": True,
                }
                _make_bind_child(wb, ex_eb, "hfw_rig_arms_ik", "ROTATION")
                bake_fk_pairs.append({"ctrl": ex_nm, "def_bone": wb, "mode": "ROTATION"})
                wing_ctrls.append(ex_nm)
                ex_wp = ex_eb
                def_to_driver_eb[wb] = ex_eb
        else:
            w_handles = _sample_chain_handles(w_full, max_handles=10)
            w_parent = w_root_parent
            for i, wb in enumerate(w_full):
                core_def_bones.add(wb)
                if wb in w_handles:
                    h_pos = def_rest[wb][0]
                    t_pos = (
                        def_rest[w_full[i + 1]][0]
                        if (i + 1 < len(w_full))
                        else (h_pos + Vector(((-1.0 if side == "R" else 1.0) * max(0.28, rig_h * 0.22), 0.0, 0.0)))
                    )
                    cname = f"CTRL_{wname}_{w_handles.index(wb):02d}"
                    c_eb = _new_bone(cname, h_pos, t_pos, w_parent, colls=(wings_fk_coll or body_coll,))
                    ctrl_styles[cname] = {
                        "widget": "fk_ring", "scale": max(0.26, rig_h * 0.18), "side": side,
                        "is_root": False, "is_detail": False, "category": "body", "lock_loc": True,
                    }
                    _make_bind_child(wb, c_eb, "hfw_rig_arms_ik", "ROTATION")
                    bake_fk_pairs.append({"ctrl": cname, "def_bone": wb, "mode": "ROTATION"})
                    wing_ctrls.append(cname)
                    w_parent = c_eb
                    def_to_driver_eb[wb] = c_eb
                else:
                    def_to_driver_eb[wb] = w_parent

    # --- 7. Legs: Quad/Hex Independent Front+Rear IK/FK + Scapula/Shoulders + Aquatic Flipper Tips + Full Follow Sockets! ---
    for lg in spec["legs"]:
        ik_defs = [b for b in lg["ik_bones"] if b in def_rest]
        foot_def = lg["foot_bone"]
        if len(ik_defs) < 2 or foot_def not in def_rest:
            continue
        side = lg["side"]
        lname = lg["name"]
        is_front_lg = lname.startswith("FrontLeg_")
        lg_group = "front_legs" if is_front_lg else "legs"
        lg_ik_prop = "hfw_ik_fk_front_legs" if is_front_lg else "hfw_ik_fk_legs"
        lg_ik_coll = (front_legs_ik_coll if is_front_lg else legs_ik_coll) or body_coll
        lg_fk_coll = (front_legs_fk_coll if is_front_lg else legs_fk_coll) or body_coll

        hip_parent = _find_parent_driver(ik_defs[0], ctrl_cog)
        sh_bone = lg.get("shoulder")
        if sh_bone and sh_bone in def_rest and sh_bone not in driven_def_bones:
            core_def_bones.add(sh_bone)
            sh_parent = _find_parent_driver(sh_bone, spine_parent_eb if is_front_lg else ctrl_cog)
            sh_ctrl = _new_bone(
                _unique_name(f"CTRL_Shoulder_{lname}"),
                def_rest[sh_bone][0],
                def_rest[ik_defs[0]][0],
                sh_parent,
                colls=(body_coll,),
            )
            ctrl_styles[sh_ctrl.name] = {
                "widget": "fk_ring",
                "scale": _fit_scale(sh_bone, max(0.18, rig_h * 0.12), min_s=0.14, max_s=max(0.75, rig_h * 0.18), mult=1.18),
                "side": side, "is_root": False, "is_detail": False, "category": "body", "lock_loc": True, "rot_mode": "XYZ",
            }
            _make_bind_child(sh_bone, sh_ctrl, "hfw_rig_legs_ik", "ROTATION")
            bake_fk_pairs.append({"ctrl": sh_ctrl.name, "def_bone": sh_bone, "mode": "ROTATION"})
            def_to_driver_eb[sh_bone] = sh_ctrl
            hip_parent = sh_ctrl

        pts = [def_rest[b][0] for b in ik_defs] + [def_rest[foot_def][0]]
        for b in ik_defs:
            core_def_bones.add(b)
        core_def_bones.add(foot_def)

        # 7a. Full BIG FK Leg Chain (`CTRL_FK_<lname>_0..N`, `CTRL_FK_<lname>_Foot`)
        fk_parent = hip_parent
        fk_binds: dict[str, str] = {}
        lg_fk_chain_pairs: list[dict[str, str]] = []
        for idx, bname in enumerate(ik_defs):
            fk_name = f"CTRL_FK_{lname}_{idx}"
            seg_len = max(0.15, (pts[idx + 1] - pts[idx]).length)
            fk_eb = _new_bone(fk_name, pts[idx], pts[idx + 1], fk_parent, colls=(lg_fk_coll,))
            ctrl_styles[fk_name] = {
                "widget": "fk_ring",
                "scale": _fit_scale(
                    bname,
                    max(0.24, rig_h * 0.15, seg_len * 0.38),
                    min_s=0.14 if is_humanoid else 0.24,
                    max_s=0.34 if is_humanoid else max(0.95, rig_h * 0.22),
                    mult=1.22,
                ),
                "side": side, "is_root": False, "is_detail": False, "category": "body", "lock_loc": True, "rot_mode": "XYZ",
            }
            fk_binds[bname] = _make_bind_child(
                bname, fk_eb, "hfw_rig_legs_ik", "TRANSFORM",
                prefix="MCH_BindFK_", ik_prop=lg_ik_prop, ik_invert=True, con_name="HFW_FK_Xform",
            )
            pair = {"ctrl": fk_name, "def_bone": bname, "mode": "ROTATION"}
            bake_fk_pairs.append(pair)
            lg_fk_chain_pairs.append(pair)
            fk_parent = fk_eb

        foot_pos = def_rest[foot_def][0]
        fk_foot_name = f"CTRL_FK_{lname}_Foot"
        fk_foot_eb = _new_bone(
            fk_foot_name,
            foot_pos,
            foot_pos + fwd_vec * max(0.20, rig_h * 0.14),
            fk_parent,
            colls=(lg_fk_coll,),
        )
        ctrl_styles[fk_foot_name] = {
            "widget": "fk_ring",
            "scale": _fit_scale(
                foot_def,
                max(0.18, rig_h * 0.12),
                min_s=0.12 if is_humanoid else 0.18,
                max_s=0.28 if is_humanoid else max(0.75, rig_h * 0.18),
                mult=1.20,
            ),
            "side": side, "is_root": False, "is_detail": False, "category": "body", "lock_loc": True, "rot_mode": "XYZ",
        }
        fk_binds[foot_def] = _make_bind_child(
            foot_def, fk_foot_eb, "hfw_rig_legs_ik", "TRANSFORM",
            prefix="MCH_BindFK_", ik_prop=lg_ik_prop, ik_invert=True, con_name="HFW_FK_Xform",
        )
        fpair = {"ctrl": fk_foot_name, "def_bone": foot_def, "mode": "ROTATION"}
        bake_fk_pairs.append(fpair)
        lg_fk_chain_pairs.append(fpair)
        limb_fk_chains[lg_group].append({"name": lname, "fk_pairs": lg_fk_chain_pairs})

        # 7b. IK Chain (`MCH_IK_<lname>_*`, `CTRL_IK_<lname>`, `CTRL_Pole_<lname>`)
        mch_chain: list[Any] = []
        ik_binds: dict[str, str] = {}
        for idx, bname in enumerate(ik_defs):
            m_eb = _new_bone(
                f"MCH_IK_{lname}_{idx}",
                pts[idx],
                pts[idx + 1],
                hip_parent if idx == 0 else mch_chain[-1],
                colls=(mch_coll,),
                use_connect=(idx > 0),
            )
            mch_chain.append(m_eb)
            ik_binds[bname] = _make_bind_child(
                bname, m_eb, "hfw_rig_legs_ik", "TRANSFORM",
                prefix="MCH_BindIK_", ik_prop=lg_ik_prop, ik_invert=False, con_name="HFW_IK_Xform",
            )

        ctrl_foot_name = f"CTRL_IK_{lname}"
        ctrl_foot = _new_bone(
            ctrl_foot_name,
            foot_pos,
            foot_pos + fwd_vec * max(0.24, rig_h * 0.15),
            ctrl_root,
            colls=(lg_ik_coll,),
        )
        ctrl_styles[ctrl_foot_name] = {
            "widget": "box",
            "scale": _fit_scale(
                foot_def,
                max(0.20, rig_h * 0.14),
                min_s=0.16 if is_humanoid else 0.22,
                max_s=0.34 if is_humanoid else max(0.85, rig_h * 0.20),
                mult=1.25,
            ),
            "side": side, "is_root": False, "is_detail": False, "category": "body", "lock_loc": False, "rot_mode": "XYZ",
        }
        ik_binds[foot_def] = _make_bind_child(
            foot_def, ctrl_foot, "hfw_rig_legs_ik", "TRANSFORM",
            prefix="MCH_BindIK_", ik_prop=lg_ik_prop, ik_invert=False, con_name="HFW_IK_Xform",
        )

        # 7c. Dual-space Follow sockets on EVERY leg segment + foot
        for bname in list(ik_defs) + [foot_def]:
            _make_dual_follow_socket(bname, fk_binds[bname], ik_binds[bname], lg_ik_prop)

        # 7d. Aquatic Flipper Extended FK Blade Chain (e.g. Tideripper 12-bone rear flippers!)
        lg_extra = [b for b in (lg.get("extra_fk_bones") or []) if b in def_rest]
        if lg_extra:
            ex_p = def_to_driver_eb[foot_def]
            n_ex = max(1, len(lg_extra))
            for ei, eb_nm in enumerate(lg_extra):
                core_def_bones.add(eb_nm)
                frac = ei / max(1, n_ex - 1) if n_ex > 1 else 0.0
                h_p = def_rest[eb_nm][0]
                t_p = def_rest[lg_extra[ei + 1]][0] if (ei + 1 < len(lg_extra)) else (h_p - fwd_vec * max(0.22, rig_h * 0.10))
                cname_ex = f"CTRL_FK_{lname}_Blade_{ei:02d}"
                ex_eb = _new_bone(cname_ex, h_p, t_p, ex_p, colls=(hf_coll or body_coll,))
                ctrl_styles[cname_ex] = {
                    "widget": "fk_ring",
                    "scale": _fit_scale(eb_nm, max(0.16, rig_h * 0.09) * (1.0 - frac * 0.5), min_s=0.08, max_s=max(0.50, rig_h * 0.13), mult=1.15),
                    "side": side, "is_root": False, "is_detail": False, "category": "body", "lock_loc": True, "rot_mode": "XYZ",
                }
                _make_bind_child(eb_nm, ex_eb, "hfw_rig_legs_ik", "ROTATION")
                bake_fk_pairs.append({"ctrl": cname_ex, "def_bone": eb_nm, "mode": "ROTATION"})
                ex_p = ex_eb
                def_to_driver_eb[eb_nm] = ex_eb

        mid_pt = pts[1] if len(pts) == 3 else (pts[1] * 0.7 + pts[2] * 0.3)
        pole_pos, pole_angle = _compute_pole_pos_and_angle(pts[0], mid_pt, pts[-1], mch_chain[0].matrix, fwd_vec, dist_scale=0.85)
        ctrl_pole_name = f"CTRL_Pole_{lname}"
        _new_bone(ctrl_pole_name, pole_pos, pole_pos + fwd_vec * max(0.12, rig_h * 0.08), ctrl_root, colls=(lg_ik_coll,))
        ctrl_styles[ctrl_pole_name] = {
            "widget": "diamond", "scale": max(0.17, rig_h * 0.11), "side": side,
            "is_root": False, "is_detail": False, "category": "body", "lock_loc": False,
        }

        ik_setups.append(
            {
                "mch_solver": mch_chain[-1].name,
                "mch_chain": [m.name for m in mch_chain],
                "target_ctrl": ctrl_foot_name,
                "pole_ctrl": ctrl_pole_name,
                "chain_len": len(mch_chain),
                "pole_angle": pole_angle,
            }
        )
        bake_ik_limbs.append(
            {
                "group": lg_group,
                "target_ctrl": ctrl_foot_name,
                "pole_ctrl": ctrl_pole_name,
                "end_def": foot_def,
                "ik_defs": list(ik_defs),
                "fallback_dir": [float(fwd_vec.x), float(fwd_vec.y), float(fwd_vec.z)],
                "dist_scale": 0.85,
            }
        )

    # --- 8. Arms: Clavicle FK + DUAL IK + FULL BIG FK CHAINS + FULL-SEGMENT DUAL-FOLLOW SOCKETS! ---
    for arm in spec["arms"]:
        ik_defs = [b for b in arm["ik_bones"] if b in def_rest]
        hand_def = arm["hand_bone"]
        if len(ik_defs) < 2 or hand_def not in def_rest:
            continue
        side = arm["side"]
        aname = arm["name"]

        arm_parent = _find_parent_driver(ik_defs[0], spine_parent_eb)
        clav = arm.get("clavicle")
        if clav and clav in def_rest:
            core_def_bones.add(clav)
            c_parent = _find_parent_driver(clav, spine_parent_eb)
            c_ctrl = _new_bone(
                f"CTRL_Clavicle_{side}",
                def_rest[clav][0],
                def_rest[ik_defs[0]][0],
                c_parent,
                colls=(body_coll,),
            )
            ctrl_styles[c_ctrl.name] = {
                "widget": "fk_ring",
                "scale": _fit_scale(clav, max(0.15, rig_h * 0.10), min_s=0.10, max_s=0.28 if is_humanoid else max(0.65, rig_h * 0.15), mult=1.15),
                "side": side, "is_root": False, "is_detail": False, "category": "body", "lock_loc": True, "rot_mode": "XYZ",
            }
            _make_bind_child(clav, c_ctrl, "hfw_rig_arms_ik", "ROTATION")
            bake_fk_pairs.append({"ctrl": c_ctrl.name, "def_bone": clav, "mode": "ROTATION"})
            def_to_driver_eb[clav] = c_ctrl
            arm_parent = c_ctrl

        pts = [def_rest[b][0] for b in ik_defs] + [def_rest[hand_def][0]]
        for b in ik_defs:
            core_def_bones.add(b)
        core_def_bones.add(hand_def)

        # 8a. Full BIG FK Arm & Hand Chain (`CTRL_FK_<aname>_0..N`, `CTRL_FK_<aname>_Hand`)
        fk_parent = arm_parent
        fk_binds_a: dict[str, str] = {}
        ar_fk_chain_pairs: list[dict[str, str]] = []
        for idx, bname in enumerate(ik_defs):
            fk_name = f"CTRL_FK_{aname}_{idx}"
            seg_len = max(0.10, (pts[idx + 1] - pts[idx]).length)
            fk_eb = _new_bone(fk_name, pts[idx], pts[idx + 1], fk_parent, colls=(arms_fk_coll,))
            ctrl_styles[fk_name] = {
                "widget": "fk_ring",
                "scale": _fit_scale(
                    bname,
                    max(0.15, rig_h * 0.10, seg_len * 0.38),
                    min_s=0.10 if is_humanoid else 0.16,
                    max_s=0.24 if is_humanoid else max(0.75, rig_h * 0.17),
                    mult=1.22,
                ),
                "side": side, "is_root": False, "is_detail": False, "category": "body", "lock_loc": True, "rot_mode": "XYZ",
            }
            fk_binds_a[bname] = _make_bind_child(
                bname, fk_eb, "hfw_rig_arms_ik", "TRANSFORM",
                prefix="MCH_BindFK_", ik_prop="hfw_ik_fk_arms", ik_invert=True, con_name="HFW_FK_Xform",
            )
            pair = {"ctrl": fk_name, "def_bone": bname, "mode": "ROTATION"}
            bake_fk_pairs.append(pair)
            ar_fk_chain_pairs.append(pair)
            fk_parent = fk_eb

        hand_pos = def_rest[hand_def][0]
        fk_hand_name = f"CTRL_FK_{aname}_Hand"
        fk_hand_eb = _new_bone(
            fk_hand_name,
            hand_pos,
            hand_pos + fwd_vec * max(0.14, rig_h * 0.10),
            fk_parent,
            colls=(arms_fk_coll,),
        )
        ctrl_styles[fk_hand_name] = {
            "widget": "fk_ring",
            "scale": _fit_scale(
                hand_def,
                max(0.13, rig_h * 0.09),
                min_s=0.09 if is_humanoid else 0.15,
                max_s=0.22 if is_humanoid else max(0.65, rig_h * 0.15),
                mult=1.20,
            ),
            "side": side, "is_root": False, "is_detail": False, "category": "body", "lock_loc": True, "rot_mode": "XYZ",
        }
        fk_binds_a[hand_def] = _make_bind_child(
            hand_def, fk_hand_eb, "hfw_rig_arms_ik", "TRANSFORM",
            prefix="MCH_BindFK_", ik_prop="hfw_ik_fk_arms", ik_invert=True, con_name="HFW_FK_Xform",
        )
        hpair = {"ctrl": fk_hand_name, "def_bone": hand_def, "mode": "ROTATION"}
        bake_fk_pairs.append(hpair)
        ar_fk_chain_pairs.append(hpair)
        limb_fk_chains["arms"].append({"name": aname, "fk_pairs": ar_fk_chain_pairs})

        # 8b. IK Arm Chain (`MCH_IK_<aname>_*`, `CTRL_IK_<aname>`, `CTRL_Pole_<aname>`)
        mch_chain = []
        ik_binds_a: dict[str, str] = {}
        for idx, bname in enumerate(ik_defs):
            m_eb = _new_bone(
                f"MCH_IK_{aname}_{idx}",
                pts[idx],
                pts[idx + 1],
                arm_parent if idx == 0 else mch_chain[-1],
                colls=(mch_coll,),
                use_connect=(idx > 0),
            )
            mch_chain.append(m_eb)
            ik_binds_a[bname] = _make_bind_child(
                bname, m_eb, "hfw_rig_arms_ik", "TRANSFORM",
                prefix="MCH_BindIK_", ik_prop="hfw_ik_fk_arms", ik_invert=False, con_name="HFW_IK_Xform",
            )

        ctrl_hand_name = f"CTRL_IK_{aname}"
        ctrl_hand = _new_bone(
            ctrl_hand_name,
            hand_pos,
            hand_pos + fwd_vec * max(0.16, rig_h * 0.11),
            ctrl_root,
            colls=(arms_ik_coll,),
        )
        ctrl_styles[ctrl_hand_name] = {
            "widget": "box",
            "scale": _fit_scale(
                hand_def,
                max(0.14, rig_h * 0.10),
                min_s=0.11 if is_humanoid else 0.18,
                max_s=0.24 if is_humanoid else max(0.70, rig_h * 0.16),
                mult=1.25,
            ),
            "side": side, "is_root": False, "is_detail": False, "category": "body", "lock_loc": False, "rot_mode": "XYZ",
        }
        ik_binds_a[hand_def] = _make_bind_child(
            hand_def, ctrl_hand, "hfw_rig_arms_ik", "TRANSFORM",
            prefix="MCH_BindIK_", ik_prop="hfw_ik_fk_arms", ik_invert=False, con_name="HFW_IK_Xform",
        )

        # 8c. Dual-space Follow sockets on EVERY arm segment + hand so arm feathers, blades, plates & fingers follow IK & FK!
        for bname in list(ik_defs) + [hand_def]:
            _make_dual_follow_socket(bname, fk_binds_a[bname], ik_binds_a[bname], "hfw_ik_fk_arms")

        # 8d. Extended FK Flipper Blade / Tendril Tip Chain (e.g. Tideripper 11-bone front flippers, Specter nano-arms!)
        ar_extra = [b for b in (arm.get("extra_fk_bones") or []) if b in def_rest]
        if ar_extra:
            ex_ap = def_to_driver_eb[hand_def]
            n_aex = max(1, len(ar_extra))
            for ei, eb_nm in enumerate(ar_extra):
                core_def_bones.add(eb_nm)
                frac = ei / max(1, n_aex - 1) if n_aex > 1 else 0.0
                h_p = def_rest[eb_nm][0]
                t_p = def_rest[ar_extra[ei + 1]][0] if (ei + 1 < len(ar_extra)) else (h_p - fwd_vec * max(0.20, rig_h * 0.09))
                cname_aex = f"CTRL_FK_{aname}_Blade_{ei:02d}"
                ex_eb = _new_bone(cname_aex, h_p, t_p, ex_ap, colls=(hf_coll or body_coll,))
                ctrl_styles[cname_aex] = {
                    "widget": "fk_ring",
                    "scale": _fit_scale(eb_nm, max(0.14, rig_h * 0.085) * (1.0 - frac * 0.5), min_s=0.08, max_s=max(0.45, rig_h * 0.12), mult=1.15),
                    "side": side, "is_root": False, "is_detail": False, "category": "body", "lock_loc": True, "rot_mode": "XYZ",
                }
                _make_bind_child(eb_nm, ex_eb, "hfw_rig_arms_ik", "ROTATION")
                bake_fk_pairs.append({"ctrl": cname_aex, "def_bone": eb_nm, "mode": "ROTATION"})
                ex_ap = ex_eb
                def_to_driver_eb[eb_nm] = ex_eb

        elbow_fallback = -fwd_vec
        mid_pt = pts[1] if len(pts) == 3 else (pts[1] * 0.5 + pts[2] * 0.5)
        pole_pos, pole_angle = _compute_pole_pos_and_angle(pts[0], mid_pt, pts[-1], mch_chain[0].matrix, elbow_fallback, dist_scale=0.85)
        ctrl_pole_name = f"CTRL_Pole_{aname}"
        _new_bone(ctrl_pole_name, pole_pos, pole_pos + elbow_fallback * max(0.10, rig_h * 0.07), ctrl_root, colls=(arms_ik_coll,))
        ctrl_styles[ctrl_pole_name] = {
            "widget": "diamond", "scale": max(0.12, rig_h * 0.085), "side": side,
            "is_root": False, "is_detail": False, "category": "body", "lock_loc": False,
        }

        ik_setups.append(
            {
                "mch_solver": mch_chain[-1].name,
                "mch_chain": [m.name for m in mch_chain],
                "target_ctrl": ctrl_hand_name,
                "pole_ctrl": ctrl_pole_name,
                "chain_len": len(mch_chain),
                "pole_angle": pole_angle,
            }
        )
        bake_ik_limbs.append(
            {
                "group": "arms",
                "target_ctrl": ctrl_hand_name,
                "pole_ctrl": ctrl_pole_name,
                "end_def": hand_def,
                "ik_defs": list(ik_defs),
                "fallback_dir": [float(elbow_fallback.x), float(elbow_fallback.y), float(elbow_fallback.z)],
                "dist_scale": 0.85,
            }
        )

    # --- 9. Categorized Small-Part Controls: Master Finger/Toe/Talon Curl Rigs, Frills, Plates, Weapons, Chainsaws & Drills ---
    cat_counts: dict[str, int] = {"hands_feet": 0, "frills": 0, "plates": 0, "weapons": 0, "spinners": 0}

    def _build_finger_toe_curl_chains(
        chain_list: list[dict[str, Any]],
        *,
        is_foot: bool,
    ) -> None:
        """Rigify `super_finger`-inspired Finger / Toe / Talon Curl Rig:
        - Auto-aligns every multi-bone chain to its exact anatomical curl plane (`+X` = curl inward/downward, `+Z` = splay).
        - Places ONE clean, sleek `CTRL_Curl_*` master lever per finger/toe/talon in `Hands & Feet` (zero viewport clutter!).
        - Tucks fine individual `CTRL_FK_*` phalanx tweak rings under additive `MCH_Curl_*` parents in `hf_tweaks_coll`!
        """
        for ch_info in chain_list:
            ch_bones = [b for b in ch_info["bones"] if b in def_rest and b not in driven_def_bones]
            if not ch_bones:
                continue
            side = ch_info["side"]
            p_def = ch_info.get("parent_def")
            c_parent = def_to_driver_eb.get(p_def) if p_def else None
            if c_parent is None:
                continue

            # Single-bone hoof/ball/dewclaw -> clean mesh-fitted FK ring/pin directly in `Hands & Feet`
            if len(ch_bones) == 1:
                bname = ch_bones[0]
                bh, bt, bm, bl = def_rest[bname]
                cname = _unique_name(f"CTRL_FK_{ch_info['name']}_00")
                c_eb = _new_bone(cname, bh, bt, c_parent, matrix=bm, colls=(hf_coll,))
                sc = _fit_scale(bname, max(0.05, min(0.22, bl * 1.1)), 0.04, 0.24, 1.15)
                ctrl_styles[cname] = {
                    "widget": "pin", "scale": sc, "side": side,
                    "is_root": False, "is_detail": True, "category": "hands_feet",
                    "lock_loc": True, "rot_mode": "XYZ",
                }
                _make_bind_child(bname, c_eb, "hfw_rig_detail", "ROTATION")
                bake_fk_pairs.append({"ctrl": cname, "def_bone": bname, "mode": "ROTATION"})
                def_to_driver_eb[bname] = c_eb
                cat_counts["hands_feet"] += 1
                continue

            # Multi-bone Finger / Toe / Talon / Wing-Finger chain -> Master `CTRL_Curl_*` + aligned `MCH_Curl_*` + `CTRL_FK_*` tweaks!
            pts_c = [def_rest[b][0] for b in ch_bones]
            # Derive true anatomical tip purely from real joint-to-joint spacing (NEVER Decima's arbitrary 20-45cm leaf stub tail!)
            last_seg = pts_c[-1] - pts_c[-2]
            if last_seg.length > 0.004:
                end_tip = pts_c[-1] + last_seg * 0.65
            else:
                end_tip = pts_c[-1] + (pts_c[-1] - pts_c[0]).normalized() * 0.03
            chain_vec = end_tip - pts_c[0]
            chain_len = max(0.025, chain_vec.length)
            y_base = (pts_c[1] - pts_c[0]).normalized() if (pts_c[1] - pts_c[0]).length > 1e-4 else chain_vec.normalized()

            # Compute consistent anatomical curl axis (+X so positive X rotation curls downward/inward cleanly!)
            curl_x: Vector | None = None
            if len(pts_c) >= 3:
                v01 = (pts_c[1] - pts_c[0]).normalized()
                v12 = (pts_c[2] - pts_c[1]).normalized()
                cr_v = v01.cross(v12)
                if cr_v.length > 0.05:
                    curl_x = cr_v.normalized()
                    # Ensure +X curls DOWNWARD/INWARD (towards -Z on feet, towards palm on hands)
                    bend_dir = (v12 - v01 * v01.dot(v12)).normalized()
                    if curl_x.cross(v01).dot(bend_dir) < 0.0:
                        curl_x = -curl_x
            if curl_x is None:
                down_hint = Vector((0.0, 0.0, -1.0)) if (is_foot or not is_humanoid) else Vector((0.0, 0.0, -1.0))
                cr_v = y_base.cross(down_hint)
                if cr_v.length < 0.05:
                    cr_v = y_base.cross(Vector((1.0 if side == "L" else -1.0, 0.0, 0.0)))
                curl_x = cr_v.normalized() if cr_v.length > 1e-4 else Vector((1.0, 0.0, 0.0))

            def _make_curl_mat(origin: Vector, y_dir: Vector, x_hint: Vector) -> Matrix:
                yd = y_dir.normalized() if y_dir.length > 1e-4 else Vector((0.0, 1.0, 0.0))
                xd = x_hint - yd * x_hint.dot(yd)
                if xd.length < 1e-4:
                    xd = Vector((1.0, 0.0, 0.0))
                xd.normalize()
                zd = xd.cross(yd).normalized()
                m4 = Matrix((
                    (xd.x, yd.x, zd.x, origin.x),
                    (xd.y, yd.y, zd.y, origin.y),
                    (xd.z, yd.z, zd.z, origin.z),
                    (0.0, 0.0, 0.0, 1.0),
                ))
                return m4

            curl_nm = _unique_name(f"CTRL_Curl_{ch_info['name']}")
            curl_mat = _make_curl_mat(pts_c[0], chain_vec, curl_x)
            curl_eb = _new_bone(curl_nm, pts_c[0], end_tip, c_parent, matrix=curl_mat, colls=(hf_coll,))
            curl_eb.length = chain_len
            ctrl_styles[curl_nm] = {
                "widget": "curl",
                "scale": max(0.025, min(0.32, chain_len * 0.90)),
                "side": side,
                "is_root": False,
                "is_detail": True,
                "category": "hands_feet",
                "lock_loc": True,
                "lock_rot": (False, True, False),  # X = Curl, Z = Spread, Y roll locked!
                "rot_mode": "XYZ",
            }
            curl_ctrls.append(curl_nm)
            cat_counts["hands_feet"] += 1

            fk_par = c_parent
            tweak_coll = hf_tweaks_coll or hf_coll
            for idx, bname in enumerate(ch_bones):
                bh, bt, _bm, bl = def_rest[bname]
                nxt_p = pts_c[idx + 1] if (idx + 1 < len(pts_c)) else end_tip
                seg_v = nxt_p - bh
                seg_l = max(0.02, seg_v.length)
                ph_mat = _make_curl_mat(bh, seg_v, curl_x)

                # Hidden `MCH_Curl_*` receives additive curl/splay from `CTRL_Curl_*`!
                mc_nm = _unique_name(f"MCH_Curl_{ch_info['name']}_{idx:02d}")
                mc_eb = _new_bone(mc_nm, bh, nxt_p, fk_par, matrix=ph_mat, colls=(mch_coll,))
                mc_eb.length = seg_l
                curl_setups.append(
                    {
                        "mch_bone": mc_nm,
                        "curl_ctrl": curl_nm,
                        "is_base": idx == 0,
                        "influence": 1.0,
                    }
                )

                # Individual phalanx FK tweak sits cleanly under `MCH_Curl_*` (works ON TOP of master curl AND bakes cleanly!)
                cname = _unique_name(f"CTRL_FK_{ch_info['name']}_{idx:02d}")
                c_eb = _new_bone(cname, bh, nxt_p, mc_eb, matrix=ph_mat, colls=(tweak_coll,))
                c_eb.length = seg_l
                tw_sc = _fit_scale(bname, max(0.018, min(0.16, seg_l * 0.55)), 0.015, 0.18, 1.15)
                ctrl_styles[cname] = {
                    "widget": "fk_ring",
                    "scale": tw_sc,
                    "side": side,
                    "is_root": False,
                    "is_detail": True,
                    "category": "hands_feet",
                    "lock_loc": True,
                    "rot_mode": "XYZ",
                }
                _make_bind_child(bname, c_eb, "hfw_rig_detail", "ROTATION")
                bake_fk_pairs.append({"ctrl": cname, "def_bone": bname, "mode": "ROTATION"})
                fk_par = c_eb
                def_to_driver_eb[bname] = c_eb

    def _build_categorized_chains(
        chain_list: list[dict[str, Any]],
        *,
        prefix_tag: str,
        target_coll: Any,
        group_prop: str,
        widget_name: str,
        cat_key: str,
        min_sc: float,
        max_sc: float,
        scale_frac: float,
    ) -> None:
        for ch_info in chain_list:
            ch_bones = [b for b in ch_info["bones"] if b in def_rest and b not in driven_def_bones]
            if not ch_bones:
                continue
            side = ch_info["side"]
            p_def = ch_info.get("parent_def")
            c_parent = def_to_driver_eb.get(p_def) if p_def else None
            if c_parent is None:
                continue
            for idx, bname in enumerate(ch_bones):
                bh, bt, bm, bl = def_rest[bname]
                if idx + 1 < len(ch_bones):
                    nxt_h = def_rest[ch_bones[idx + 1]][0]
                    tail_p = nxt_h if (nxt_h - bh).length > 0.008 else bt
                else:
                    tail_p = bt
                cname = _unique_name(f"CTRL_{prefix_tag}_{ch_info['name']}_{idx:02d}")
                c_eb = _new_bone(cname, bh, tail_p, c_parent, matrix=bm if (tail_p == bt) else None, colls=(target_coll,))
                sc = _fit_scale(bname, min(max_sc, rig_h * scale_frac), min_sc, max_sc, 1.10)
                ctrl_styles[cname] = {
                    "widget": widget_name, "scale": sc, "side": side,
                    "is_root": False, "is_detail": True, "category": cat_key, "lock_loc": True, "rot_mode": "XYZ",
                }
                _make_bind_child(bname, c_eb, group_prop, "ROTATION")
                bake_fk_pairs.append({"ctrl": cname, "def_bone": bname, "mode": "ROTATION"})
                c_parent = c_eb
                def_to_driver_eb[bname] = c_eb
                cat_counts[cat_key] = cat_counts.get(cat_key, 0) + 1

    _build_finger_toe_curl_chains(spec.get("fingers") or [], is_foot=False)
    _build_finger_toe_curl_chains(spec.get("toes") or [], is_foot=True)
    _build_categorized_chains(
        spec.get("frills") or [],
        prefix_tag="Frill", target_coll=frill_coll, group_prop="hfw_rig_frills",
        widget_name="paddle", cat_key="frills", min_sc=0.07, max_sc=0.34, scale_frac=0.075,
    )
    _build_categorized_chains(
        spec.get("plates") or [],
        prefix_tag="Plate", target_coll=plate_coll, group_prop="hfw_rig_plates",
        widget_name="plate", cat_key="plates", min_sc=0.08, max_sc=0.32, scale_frac=0.070,
    )
    _build_categorized_chains(
        spec.get("weapons") or [],
        prefix_tag="Weapon", target_coll=weapon_coll, group_prop="hfw_rig_weapons",
        widget_name="weapon", cat_key="weapons", min_sc=0.11, max_sc=0.40, scale_frac=0.090,
    )

    # --- 9b. True Closed-Loop Parametric Stadium Chainsaw Rig (`Clawstrider` 2x26-tooth Jaw Saws, `Charger/Lancehorn` Horn Saws!) ---
    chainsaws_spec = spec.get("chainsaws") or []
    saw_master_ctrl = ""
    if chainsaws_spec and spin_coll is not None:
        # Place ONE bold magenta gear wheel `CTRL_Saw_Spin` clearly ABOVE the snout/jaw (outside armor!) so 1 dial revs ALL chainsaws!
        saw_pts_all = [def_rest[b][0] for cs in chainsaws_spec for b in cs["blades"] if b in def_rest]
        if saw_pts_all:
            saw_ctr = sum(saw_pts_all, Vector((0.0, 0.0, 0.0))) / len(saw_pts_all)
            dial_pos = saw_ctr + Vector((0.0, 0.0, max(0.58, rig_h * 0.30))) + fwd_vec * 0.16
            saw_master_par = ctrl_head if ctrl_head is not None else ctrl_cog
            c_saw_eb = _new_bone("CTRL_Saw_Spin", dial_pos, dial_pos + Vector((0.20, 0.0, 0.0)), saw_master_par, colls=(spin_coll,))
            ctrl_styles["CTRL_Saw_Spin"] = {
                "widget": "gear",
                "scale": max(0.22, min(0.42, rig_h * 0.14)),
                "side": "C",
                "is_root": False,
                "is_detail": False,
                "category": "spin",
                "lock_loc": True,
                "lock_rot": (True, False, True),  # Rotate Y ONLY -> instant chainsaw spin!
                "rot_mode": "XYZ",
            }
            saw_master_ctrl = "CTRL_Saw_Spin"
            spin_ctrls.append(saw_master_ctrl)
            cat_counts["spinners"] += 1

        for cs in chainsaws_spec:
            bl_bones = [b for b in cs["blades"] if b in def_rest and b not in driven_def_bones]
            if len(bl_bones) < 4:
                continue
            par_def = cs.get("parent_def")
            saw_par_eb = def_to_driver_eb.get(par_def) or _find_parent_driver(par_def or bl_bones[0], ctrl_head)
            pts_w = np.array([[def_rest[b][0].x, def_rest[b][0].y, def_rest[b][0].z] for b in bl_bones], dtype=np.float64)
            ctr_w = pts_w.mean(axis=0)
            rel_w = pts_w - ctr_w
            _, _, vh = np.linalg.svd(rel_w, full_matrices=False)
            u_ax = Vector(vh[0]).normalized()  # Long stadium axis (+Y of MCH_SawFrame)
            v_ax = Vector(vh[1]).normalized()  # Short stadium axis (+Z of MCH_SawFrame)
            # Keep u_ax pointing generally forward (+fwd_vec) and v_ax generally upward (+Z) for intuitive spin direction
            if u_ax.dot(fwd_vec) < 0.0:
                u_ax = -u_ax
            if v_ax.z < 0.0:
                v_ax = -v_ax
            n_ax = u_ax.cross(v_ax).normalized()
            v_ax = n_ax.cross(u_ax).normalized()  # Exact orthonormal right-handed basis (X=n, Y=u, Z=v)!

            frame_mat = Matrix((
                (n_ax.x, u_ax.x, v_ax.x, float(ctr_w[0])),
                (n_ax.y, u_ax.y, v_ax.y, float(ctr_w[1])),
                (n_ax.z, u_ax.z, v_ax.z, float(ctr_w[2])),
                (0.0, 0.0, 0.0, 1.0),
            ))
            fr_nm = _unique_name(f"MCH_SawFrame_{cs['name']}")
            fr_eb = _new_bone(fr_nm, Vector(ctr_w), Vector(ctr_w) + u_ax * 0.25, saw_par_eb, matrix=frame_mat, colls=(mch_coll,))
            fr_inv = frame_mat.inverted()

            # Measure exact stadium dimensions (L = straight rail length, r = semicircular tip radius, P = 2L + 2*pi*r)
            local_yz = [fr_inv @ def_rest[b][0] for b in bl_bones]
            ys = [p.y for p in local_yz]
            zs = [p.z for p in local_yz]
            y_mid = 0.5 * (min(ys) + max(ys))
            z_mid = 0.5 * (min(zs) + max(zs))
            span_y = max(0.05, max(ys) - min(ys))
            span_z = max(0.02, max(zs) - min(zs))
            rad = max(0.01, 0.5 * min(span_y, span_z))
            straight_L = max(0.002, max(span_y, span_z) - 2.0 * rad)
            half_a = 0.5 * straight_L
            half_P = straight_L + math.pi * rad
            perim_P = 2.0 * half_P

            def _stadium_eval(q_val: float) -> tuple[float, float, float]:
                qm = q_val - perim_P * math.floor(q_val / perim_P)
                u1 = min(max(qm, 0.0), straight_L)
                a1 = min(max(qm - straight_L, 0.0), math.pi * rad) / rad
                u2 = min(max(qm - half_P, 0.0), straight_L)
                a2 = min(max(qm - half_P - straight_L, 0.0), math.pi * rad) / rad
                yv = y_mid - half_a + u1 - u2 + rad * math.sin(a1) - rad * math.sin(a2)
                zv = z_mid + rad * math.cos(a1) - rad * math.cos(a2) + rad
                phv = -(a1 + a2)
                return yv, zv, phv

            # Find exact initial arc-length parameter q0 for each tooth by dense lookup along the closed stadium
            grid_n = 360
            grid_qs = [perim_P * gi / grid_n for gi in range(grid_n)]
            grid_pts = [_stadium_eval(gq) for gq in grid_qs]

            for bi, bname in enumerate(bl_bones):
                p_loc = local_yz[bi]
                best_gi = min(
                    range(grid_n),
                    key=lambda gi: (grid_pts[gi][0] - p_loc.y) ** 2 + (grid_pts[gi][1] - p_loc.z) ** 2,
                )
                q0 = grid_qs[best_gi]
                y0, z0, phi0 = _stadium_eval(q0)
                bh = def_rest[bname][0]
                t_mat = Matrix.Translation(bh) @ frame_mat.to_3x3().to_4x4()
                t_nm = _unique_name(f"MCH_SawTooth_{bname}")
                t_eb = _new_bone(t_nm, bh, bh + u_ax * 0.05, fr_eb, matrix=t_mat, colls=(mch_coll,))
                _make_bind_child(bname, t_eb, "hfw_rig_weapons", "TRANSFORM", prefix="MCH_BindSaw_", con_name="HFW_Saw")
                saw_driver_setups.append(
                    {
                        "tooth_bone": t_nm,
                        "dial_ctrl": saw_master_ctrl,
                        "q0": q0,
                        "y0": y0,
                        "z0": z0,
                        "phi0": phi0,
                        "y_mid": y_mid,
                        "z_mid": z_mid,
                        "half_a": half_a,
                        "L": straight_L,
                        "r": rad,
                        "h": half_P,
                        "P": perim_P,
                    }
                )

    # --- 9c. Universal PCA-Axis Rotary Drills, Grinders, Rotors, Turbines & Tidal Discs Spin Rig! ---
    if all_spinners and spin_coll is not None:
        valid_spinners = [sp for sp in all_spinners if sp["bone"] in def_rest and sp["bone"] not in driven_def_bones]
        if valid_spinners:
            # Always create ONE unmistakable master gear wheel `CTRL_Spin_Drills` cleanly offset above/beside the rig!
            sp_pts = [def_rest[sp["bone"]][0] for sp in valid_spinners]
            sp_ctr = sum(sp_pts, Vector((0.0, 0.0, 0.0))) / len(sp_pts)
            sp_kind_main = "Grinders" if all(sp["kind"] == "grinder" for sp in valid_spinners) else "Drills"
            master_sp_nm = f"CTRL_Spin_{sp_kind_main}"
            sp_dial_pos = sp_ctr + Vector((0.0, 0.0, max(0.35, rig_h * 0.20)))
            sp_dial_par = ctrl_cog if sp_kind_main == "Grinders" else (ctrl_head if len(valid_spinners) <= 8 else ctrl_cog)
            _new_bone(master_sp_nm, sp_dial_pos, sp_dial_pos + Vector((0.0, 0.25, 0.0)), sp_dial_par, colls=(spin_coll,))
            ctrl_styles[master_sp_nm] = {
                "widget": "gear",
                "scale": max(0.18, min(0.55, rig_h * 0.14)),
                "side": "C",
                "is_root": False,
                "is_detail": False,
                "category": "spin",
                "lock_loc": True,
                "lock_rot": (True, False, True),  # Rotate Y ONLY -> instant smooth drill/grinder spin!
                "rot_mode": "XYZ",
            }
            spin_ctrls.append(master_sp_nm)
            cat_counts["spinners"] += 1

            make_individual_dials = len(valid_spinners) <= 10
            for sp in valid_spinners:
                bname = sp["bone"]
                bh, bt, bm, bl = def_rest[bname]
                p_def = sp.get("parent_def")
                sp_par = def_to_driver_eb.get(p_def) if p_def else None
                if sp_par is None:
                    sp_par = _find_parent_driver(bname, ctrl_cog)

                # Compute exact 3D mechanical spin axis in armature space (from PCA of real drill/disc mesh!)
                loc_ax_t = spinner_axes.get(bname, (0.0, 1.0, 0.0))
                ax_world = (bm.to_3x3() @ Vector(loc_ax_t)).normalized()
                if ax_world.length < 1e-4:
                    ax_world = Vector((0.0, 1.0, 0.0))
                # Build clean right-handed matrix whose +Y axis IS the exact mechanical spin axis!
                ref_up = Vector((0.0, 0.0, 1.0)) if abs(ax_world.z) < 0.92 else Vector((1.0, 0.0, 0.0))
                x_sp = ax_world.cross(ref_up).normalized()
                z_sp = x_sp.cross(ax_world).normalized()
                sp_mat = Matrix((
                    (x_sp.x, ax_world.x, z_sp.x, bh.x),
                    (x_sp.y, ax_world.y, z_sp.y, bh.y),
                    (x_sp.z, ax_world.z, z_sp.z, bh.z),
                    (0.0, 0.0, 0.0, 1.0),
                ))

                local_dial_nm = ""
                mch_par = sp_par
                if make_individual_dials:
                    local_dial_nm = _unique_name(f"CTRL_Spin_{sp['name']}")
                    d_eb = _new_bone(local_dial_nm, bh, bh + ax_world * max(0.12, bl), sp_par, matrix=sp_mat, colls=(spin_coll,))
                    ctrl_styles[local_dial_nm] = {
                        "widget": "gear",
                        "scale": _fit_scale(bname, max(0.10, rig_h * 0.07), 0.08, 0.50, 1.25),
                        "side": sp["side"],
                        "is_root": False,
                        "is_detail": True,
                        "category": "spin",
                        "lock_loc": True,
                        "lock_rot": (True, False, True),
                        "rot_mode": "XYZ",
                    }
                    spin_ctrls.append(local_dial_nm)
                    cat_counts["spinners"] += 1
                    mch_par = d_eb

                m_sp_nm = _unique_name(f"MCH_Spin_{bname}")
                m_sp_eb = _new_bone(m_sp_nm, bh, bh + ax_world * max(0.08, bl), mch_par, matrix=sp_mat, colls=(mch_coll,))
                _make_bind_child(bname, m_sp_eb, "hfw_rig_weapons", "ROTATION", prefix="MCH_BindSpin_", con_name="HFW_Spin")
                def_to_driver_eb[bname] = m_sp_eb
                spin_driver_setups.append(
                    {
                        "mch_bone": m_sp_nm,
                        "master_ctrl": master_sp_nm,
                        "sign": -1.0 if sp["side"] == "R" else 1.0,
                    }
                )

    # --- 10. Optional Detachable Armor Plate / Canister Controls (Mesh-owning `*_helper` bones on core joints!) ---
    if plate_coll is not None and visible_helper_meshes:
        for hp_bone, mesh_name in sorted(visible_helper_meshes.items()):
            if hp_bone not in def_rest or hp_bone in driven_def_bones:
                continue
            eb_hp = ebones.get(hp_bone)
            p_eb = eb_hp.parent if eb_hp else None
            if p_eb is None or p_eb.name not in core_def_bones:
                continue
            c_parent = def_to_driver_eb.get(p_eb.name)
            if c_parent is None:
                continue
            hh, ht, hm, hl = def_rest[hp_bone]
            stem = re.sub(r"(_[LlRrCc]x)?_helper$", "", hp_bone, flags=re.IGNORECASE)
            stem = re.sub(r"[^A-Za-z0-9_]+", "", stem)[:24] or "Plate"
            cname = _unique_name(f"CTRL_Plate_{stem}")
            side = _extract_side(stem)
            c_eb = _new_bone(cname, hh, ht, c_parent, matrix=hm, colls=(plate_coll,))
            ctrl_styles[cname] = {
                "widget": "plate", "scale": max(0.11, min(0.36, rig_h * 0.08)), "side": side,
                "is_root": False, "is_detail": True, "category": "plates", "lock_loc": False,
            }
            _make_bind_child(hp_bone, c_eb, "hfw_rig_plates", "TRANSFORM")
            cat_counts["plates"] += 1

    # Switch to POSE mode to attach IK constraints, dual FK/IK DEF bindings, follow sockets & rest-lock!
    bpy.ops.object.mode_set(mode="POSE")
    pbones = arm_obj.pose.bones

    # 1. Extra helper constraints (e.g., Torso Mid-Anchor 50/50 blend, Chest tip rotation alignment, Head IK space switch!)
    for ec in extra_constraints:
        pb_ec = pbones.get(ec["bone"])
        if pb_ec is None:
            continue
        con_ec = pb_ec.constraints.new(type=ec["type"])
        con_ec.name = _unique_name(f"HFW_MCH_{ec['type'][:4]}")
        con_ec.target = arm_obj
        con_ec.subtarget = ec["target"]
        con_ec.target_space = "WORLD"
        con_ec.owner_space = "WORLD"
        con_ec.influence = float(ec.get("influence", 1.0))
        if ec.get("driver_prop"):
            _add_influence_driver(arm_obj, con_ec, ec["driver_prop"], use_master=False)

    # 1b. IK constraints on MCH_IK, MCH_TorsoIK & pole-free soft-stiffness MCH_NeckIK solvers
    for iks in ik_setups:
        pb = pbones[iks["mch_solver"]]
        con = pb.constraints.new(type="IK")
        con.name = "HFW_IK"
        con.target = arm_obj
        con.subtarget = iks["target_ctrl"]
        if iks.get("pole_ctrl"):
            con.pole_target = arm_obj
            con.pole_subtarget = iks["pole_ctrl"]
            con.pole_angle = float(iks["pole_angle"])
        if iks.get("use_rotation"):
            con.use_rotation = True
            con.orient_weight = float(iks.get("orient_weight", 0.75))
        if iks.get("soft_stiffness"):
            m_list = iks.get("mch_chain") or []
            n_m = max(1, len(m_list))
            for mi, mn in enumerate(m_list):
                pbm = pbones.get(mn)
                if pbm is not None:
                    # Gentle stiffness gradient (slightly stiffer near base -> smooth natural distributed neck bend!)
                    stiff = 0.25 * (1.0 - mi / max(1, n_m))
                    pbm.ik_stiffness_x = stiff
                    pbm.ik_stiffness_y = stiff * 0.5
                    pbm.ik_stiffness_z = stiff
        con.chain_count = int(iks["chain_len"])
        con.use_stretch = False

    # 1c. Cycle-Free B-Bone Spline IK Constraints (`STRETCH_TO` bridge + `COPY_TRANSFORMS(use_bbone_shape=True)` + rigid scale!)
    for ss in spline_setups:
        if ss["type"] == "seg":
            pb_s = pbones.get(ss["bone"])
            if pb_s is None:
                continue
            st = pb_s.constraints.new(type="STRETCH_TO")
            st.name = "HFW_SplineStretch"
            st.target = arm_obj
            st.subtarget = ss["end_ctrl"]
            st.volume = "NO_VOLUME"
            st.keep_axis = "PLANE_Z"
        elif ss["type"] == "sample":
            pb_pt = pbones.get(ss["pt_bone"])
            pb_rg = pbones.get(ss["rigid_bone"])
            if pb_pt is not None:
                ct = pb_pt.constraints.new(type="COPY_TRANSFORMS")
                ct.name = "HFW_BBoneSample"
                ct.target = arm_obj
                ct.subtarget = ss["seg_bone"]
                ct.head_tail = float(ss["head_tail"])
                ct.use_bbone_shape = True
                ct.target_space = "WORLD"
                ct.owner_space = "WORLD"
            if pb_rg is not None:
                cl = pb_rg.constraints.new(type="COPY_LOCATION")
                cl.name = "HFW_SplineLoc"
                cl.target = arm_obj
                cl.subtarget = ss["pt_bone"]
                cl.target_space = "WORLD"
                cl.owner_space = "WORLD"
                cr_s = pb_rg.constraints.new(type="COPY_ROTATION")
                cr_s.name = "HFW_SplineRot"
                cr_s.target = arm_obj
                cr_s.subtarget = ss["pt_bone"]
                cr_s.target_space = "WORLD"
                cr_s.owner_space = "WORLD"

    # 1d. Master Finger / Toe / Talon Additive Curl Constraints (`CTRL_Curl_*` -> `MCH_Curl_*` in LOCAL space!)
    for cs in curl_setups:
        pb_mc = pbones.get(cs["mch_bone"])
        if pb_mc is None:
            continue
        cr_c = pb_mc.constraints.new(type="COPY_ROTATION")
        cr_c.name = "HFW_MasterCurl"
        cr_c.target = arm_obj
        cr_c.subtarget = cs["curl_ctrl"]
        cr_c.target_space = "LOCAL"
        cr_c.owner_space = "LOCAL"
        cr_c.use_x = True
        cr_c.use_y = False
        cr_c.use_z = bool(cs["is_base"])  # Only knuckle splays in Z; all phalanges curl cleanly in X!
        if hasattr(cr_c, "mix_mode"):
            cr_c.mix_mode = "ADD"
        cr_c.influence = float(cs.get("influence", 1.0))

    # 1e. Exact Closed-Loop Parametric Stadium Chainsaw Drivers (2-stage Simple Expressions, <=175 chars, 0 cycles!)
    for sd in saw_driver_setups:
        pb_t = pbones.get(sd["tooth_bone"])
        if pb_t is None:
            continue
        pb_t.rotation_mode = "XYZ"
        P_v = sd["P"]
        h_v = sd["h"]
        L_v = sd["L"]
        r_v = max(1e-4, sd["r"])
        r_inv = 1.0 / r_v
        a_v = sd["half_a"]
        pi_r = math.pi * r_v
        q0_v = sd["q0"]
        y_mid_v = sd["y_mid"]
        z_mid_v = sd["z_mid"]
        y0_v = sd["y0"]
        z0_v = sd["z0"]
        phi0_v = sd["phi0"]
        dial_nm = sd["dial_ctrl"]

        # Stage 1: wrapped phase `q` in [0, P) stored on hidden `MCH_SawTooth_*["q"]` (~110 chars, pure C Simple Expression!)
        pb_t["q"] = float(q0_v)
        fc_q = pb_t.driver_add('["q"]')
        drv_q = fc_q.driver
        drv_q.type = "SCRIPTED"
        v_sp = drv_q.variables.new()
        v_sp.name = "s"
        v_sp.type = "SINGLE_PROP"
        v_sp.targets[0].id_type = "OBJECT"
        v_sp.targets[0].id = arm_obj
        v_sp.targets[0].data_path = '["hfw_saw_speed"]'
        v_dl = drv_q.variables.new()
        v_dl.name = "d"
        v_dl.type = "TRANSFORMS"
        v_dl.targets[0].id = arm_obj
        v_dl.targets[0].bone_target = dial_nm
        v_dl.targets[0].transform_type = "ROT_Y"
        v_dl.targets[0].transform_space = "LOCAL_SPACE"
        ph = f"{q0_v:.5f}+d*{P_v / (2.0 * math.pi):.5f}+frame*s*{P_v / 18.0:.5f}"
        drv_q.expression = f"({ph})-{P_v:.5f}*floor(({ph})/{P_v:.5f})"

        # Stage 2: closed-form stadium `(y, z, rx)` deltas reading single variable `q` (<= 175 chars, way under 255-char C limit!)
        c_y = (y_mid_v - a_v) - y0_v
        c_z = (z_mid_v + r_v) - z0_v
        a1 = f"min(max(q-{L_v:.5f},0),{pi_r:.5f})*{r_inv:.4f}"
        a2 = f"min(max(q-{h_v + L_v:.5f},0),{pi_r:.5f})*{r_inv:.4f}"
        expr_y = (
            f"{c_y:.5f}+min(max(q,0),{L_v:.5f})-min(max(q-{h_v:.5f},0),{L_v:.5f})"
            f"+{r_v:.5f}*(sin({a1})-sin({a2}))"
        )
        expr_z = f"{c_z:.5f}+{r_v:.5f}*(cos({a1})-cos({a2}))"
        expr_rx = f"-({a1}+{a2})-{phi0_v:.5f}"
        q_dpath = f'pose.bones["{pb_t.name}"]["q"]'

        for dpath, axis_i, expr_s in (
            ("location", 1, expr_y),
            ("location", 2, expr_z),
            ("rotation_euler", 0, expr_rx),
        ):
            fc = pb_t.driver_add(dpath, axis_i)
            drv = fc.driver
            drv.type = "SCRIPTED"
            vq = drv.variables.new()
            vq.name = "q"
            vq.type = "SINGLE_PROP"
            vq.targets[0].id_type = "OBJECT"
            vq.targets[0].id = arm_obj
            vq.targets[0].data_path = q_dpath
            drv.expression = expr_s

    # 1f. Universal PCA-Axis Drill / Grinder / Rotor / Tidal-Disc Spin Drivers (Simple Expression!)
    for spd_item in spin_driver_setups:
        pb_sp = pbones.get(spd_item["mch_bone"])
        if pb_sp is None:
            continue
        pb_sp.rotation_mode = "XYZ"
        fc = pb_sp.driver_add("rotation_euler", 1)
        drv = fc.driver
        drv.type = "SCRIPTED"
        v_sp = drv.variables.new()
        v_sp.name = "spd"
        v_sp.type = "SINGLE_PROP"
        v_sp.targets[0].id_type = "OBJECT"
        v_sp.targets[0].id = arm_obj
        v_sp.targets[0].data_path = '["hfw_drill_speed"]'
        v_dl = drv.variables.new()
        v_dl.name = "dial"
        v_dl.type = "TRANSFORMS"
        v_dl.targets[0].id = arm_obj
        v_dl.targets[0].bone_target = spd_item["master_ctrl"]
        v_dl.targets[0].transform_type = "ROT_Y"
        v_dl.targets[0].transform_space = "LOCAL_SPACE"
        sgn = float(spd_item.get("sign", 1.0))
        drv.expression = f"{sgn:.1f} * (dial + frame * spd * 0.45)"

    # 2. Dual-space follow sockets (`MCH_Follow_*`) for fingers/toes so they track FK OR IK seamlessly!
    for fs in follow_setups:
        pb_f = pbones.get(fs["follow_mch"])
        if pb_f is None:
            continue
        c_fk = pb_f.constraints.new(type="COPY_TRANSFORMS")
        c_fk.name = "HFW_Follow_FK"
        c_fk.target = arm_obj
        c_fk.subtarget = fs["fk_mch"]
        c_fk.target_space = "WORLD"
        c_fk.owner_space = "WORLD"
        c_fk.influence = 1.0

        c_ik = pb_f.constraints.new(type="COPY_TRANSFORMS")
        c_ik.name = "HFW_Follow_IK"
        c_ik.target = arm_obj
        c_ik.subtarget = fs["ik_mch"]
        c_ik.target_space = "WORLD"
        c_ik.owner_space = "WORLD"
        _add_influence_driver(
            arm_obj,
            c_ik,
            "",
            ik_prop=fs["ik_prop"],
            ik_invert=False,
            extra_prop=fs.get("extra_prop"),
            use_master=False,
        )

    # 3. Driven constraints onto controlled DEF bones (supports both single FK and dual IK/FK limbs!)
    rot_only_defs: set[str] = set()
    xform_defs: set[str] = set()
    for bd in def_bindings:
        pb = pbones.get(bd["def_name"])
        if pb is None:
            continue
        if bd["mode"] == "ROTATION":
            con = pb.constraints.new(type="COPY_ROTATION")
            con.name = bd["con_name"] + "_Rot"
            rot_only_defs.add(bd["def_name"])
        else:
            con = pb.constraints.new(type="COPY_TRANSFORMS")
            con.name = bd["con_name"] + "_Xform"
            xform_defs.add(bd["def_name"])
        con.target = arm_obj
        con.subtarget = bd["target_mch"]
        con.target_space = "WORLD"
        con.owner_space = "WORLD"
        # For dual FK+IK limbs, FK runs first at `master * grp` and IK runs second at `master * grp * ik (* ext)`
        # so even mid-blend (ik=0.5) interpolates cleanly FK<->IK with ZERO raw game-anim leak!
        use_ik_prop = None if (bd["ik_prop"] and bd["ik_invert"]) else bd["ik_prop"]
        _add_influence_driver(
            arm_obj,
            con,
            bd["group_prop"],
            ik_prop=use_ik_prop,
            ik_invert=False,
            extra_prop=bd.get("extra_prop"),
            use_master=True,
        )

    # 4. ZERO-GHOST REST-LOCK on ALL uncontrolled DEF bones + loc/scale lock on rotation-only controlled DEF bones!
    # When `hfw_rig_master = 1.0` & `hfw_rig_lock_secondary = 1.0`, ZERO leftover game animation leaks anywhere!
    def _attach_loc_scale_lock(pb_t: Any) -> None:
        cr_loc = pb_t.constraints.new(type="LIMIT_LOCATION")
        cr_loc.name = "HFW_RestLock_Loc"
        cr_loc.owner_space = "LOCAL"
        cr_loc.use_min_x = cr_loc.use_max_x = True
        cr_loc.use_min_y = cr_loc.use_max_y = True
        cr_loc.use_min_z = cr_loc.use_max_z = True
        cr_loc.min_x = cr_loc.max_x = 0.0
        cr_loc.min_y = cr_loc.max_y = 0.0
        cr_loc.min_z = cr_loc.max_z = 0.0
        _add_influence_driver(arm_obj, cr_loc, "hfw_rig_lock_secondary", use_master=True)

        cr_scl = pb_t.constraints.new(type="LIMIT_SCALE")
        cr_scl.name = "HFW_RestLock_Scale"
        cr_scl.owner_space = "LOCAL"
        cr_scl.use_min_x = cr_scl.use_max_x = True
        cr_scl.use_min_y = cr_scl.use_max_y = True
        cr_scl.use_min_z = cr_scl.use_max_z = True
        cr_scl.min_x = cr_scl.max_x = 1.0
        cr_scl.min_y = cr_scl.max_y = 1.0
        cr_scl.min_z = cr_scl.max_z = 1.0
        _add_influence_driver(arm_obj, cr_scl, "hfw_rig_lock_secondary", use_master=True)

    for dname in rot_only_defs - xform_defs:
        pb = pbones.get(dname)
        if pb is not None:
            _attach_loc_scale_lock(pb)

    locked_secondary_count = 0
    for dname in def_rest:
        if dname in driven_def_bones:
            continue
        pb = pbones.get(dname)
        if pb is None:
            continue
        cr_rot = pb.constraints.new(type="LIMIT_ROTATION")
        cr_rot.name = "HFW_RestLock_Rot"
        cr_rot.owner_space = "LOCAL"
        cr_rot.use_limit_x = cr_rot.use_limit_y = cr_rot.use_limit_z = True
        cr_rot.min_x = cr_rot.max_x = 0.0
        cr_rot.min_y = cr_rot.max_y = 0.0
        cr_rot.min_z = cr_rot.max_z = 0.0
        _add_influence_driver(arm_obj, cr_rot, "hfw_rig_lock_secondary", use_master=True)
        _attach_loc_scale_lock(pb)
        locked_secondary_count += 1

    # 5. Style CTRL bones with mesh-fitted color-coded wire shapes, clean Euler modes & intuitive transform locks!
    for cname, st in ctrl_styles.items():
        pb = pbones.get(cname)
        if pb is None:
            continue
        pb.rotation_mode = str(st.get("rot_mode", "QUATERNION"))
        if st.get("lock_loc", False):
            pb.lock_location = (True, True, True)
        if st.get("lock_rot"):
            pb.lock_rotation = tuple(bool(x) for x in st["lock_rot"])
        wobj = widgets.get(st["widget"])
        if wobj is not None:
            pb.custom_shape = wobj
            pb.use_custom_shape_bone_size = False
            sc = float(st["scale"])
            pb.custom_shape_scale_xyz = (sc, sc, sc)
        _set_bone_color(
            pb,
            st["side"],
            is_root=bool(st["is_root"]),
            is_detail=bool(st.get("is_detail", False)),
            category=str(st.get("category", "body")),
        )

    bpy.ops.object.mode_set(mode="OBJECT")

    # Auto-calibrate `pole_angle` in Blender 5.3 across full 360° minimizing joint position + orientation drift
    arm_obj["hfw_rig_master"] = 1.0
    arm_obj["hfw_ik_fk_torso"] = float(default_torso_ik)
    arm_obj["hfw_ik_fk_legs"] = 1.0
    arm_obj["hfw_ik_fk_front_legs"] = 1.0
    arm_obj["hfw_ik_fk_arms"] = 1.0
    arm_obj["hfw_ik_fk_wings"] = 1.0
    arm_obj.update_tag()
    bpy.context.view_layer.update()
    for iks in ik_setups:
        if not iks.get("pole_ctrl"):
            continue
        m_pb = arm_obj.pose.bones.get(iks["mch_solver"])
        if m_pb is None:
            continue
        con = m_pb.constraints.get("HFW_IK")
        if con is None:
            continue
        chain_names = iks.get("mch_chain") or [iks["mch_solver"]]
        rest_heads = [arm_obj.data.bones[cn].head_local.copy() for cn in chain_names]
        rest_tails = [arm_obj.data.bones[cn].tail_local.copy() for cn in chain_names]
        rest_x = [arm_obj.data.bones[cn].matrix_local.to_3x3().col[0].copy() for cn in chain_names]

        def _chain_rest_err() -> float:
            err = 0.0
            for idx_c, cn in enumerate(chain_names):
                pb_c = arm_obj.pose.bones[cn]
                err += (pb_c.head - rest_heads[idx_c]).length
                err += (pb_c.tail - rest_tails[idx_c]).length
                err += 0.30 * (pb_c.matrix.to_3x3().col[0] - rest_x[idx_c]).length
            return err

        best_ang = con.pole_angle
        best_err = _chain_rest_err()
        if best_err > 0.002:
            for deg in range(0, 360, 15):
                cand = math.radians(deg) - math.pi
                con.pole_angle = cand
                bpy.context.view_layer.update()
                e = _chain_rest_err()
                if e < best_err:
                    best_err = e
                    best_ang = cand
            center_ang = best_ang
            for d_deg in range(-14, 15, 2):
                cand = center_ang + math.radians(d_deg)
                con.pole_angle = cand
                bpy.context.view_layer.update()
                e = _chain_rest_err()
                if e < best_err:
                    best_err = e
                    best_ang = cand
            fine_ang = best_ang
            for step in range(-8, 9):
                cand = fine_ang + math.radians(step * 0.25)
                con.pole_angle = cand
                bpy.context.view_layer.update()
                e = _chain_rest_err()
                if e < best_err:
                    best_err = e
                    best_ang = cand
            con.pole_angle = best_ang
            bpy.context.view_layer.update()

    arm_obj["hfw_rig_master"] = float(default_master)
    arm_obj["hfw_ik_fk_torso"] = float(default_torso_ik)
    arm_obj["hfw_ik_fk_spline"] = float(default_spline_ik)
    arm_obj["hfw_ik_fk_head"] = float(default_head_ik)
    arm_obj["hfw_head_ik_space"] = 0.0
    arm_obj["hfw_ik_fk_legs"] = 1.0
    arm_obj["hfw_ik_fk_front_legs"] = 1.0
    arm_obj["hfw_ik_fk_arms"] = float(default_arms_ik)
    arm_obj["hfw_ik_fk_wings"] = float(default_wings_ik)
    arm_obj["hfw_saw_speed"] = 0.0
    arm_obj["hfw_drill_speed"] = 0.0
    arm_obj.update_tag()
    bpy.context.view_layer.update()

    # Drop any empty dynamic collection (e.g., if every plate helper was filtered out) so the Bone Collections list stays spotless!
    for skey in list(active_slots.keys()):
        if skey in ("body", "mechanism", "wings"):
            continue
        cname = active_slots[skey]
        bc = arm_data.collections.get(cname)
        if bc is not None and len(bc.bones) == 0:
            try:
                arm_data.collections.remove(bc)
                del active_slots[skey]
            except Exception:
                pass
    if "wings_fk" not in active_slots and "wings" in active_slots:
        del active_slots["wings"]

    rig_meta = {
        "archetype": spec["archetype"],
        "chest_ctrl": chest_ctrl_name,
        "torso_ik_mode": torso_ik_mode,
        "torso_ik": torso_ik_meta,
        "head_ik": head_ik_meta,
        "spline_ik": spline_ik_meta,
        "has_chainsaws": bool(saw_driver_setups),
        "has_spinners": bool(spin_driver_setups),
        "collection_slots": active_slots,
        "ctrl_bones": list(ctrl_styles.keys()),
        "fk_pairs": bake_fk_pairs,
        "ik_limbs": bake_ik_limbs,
        "limb_fk_chains": limb_fk_chains,
        "tail_ctrls": tail_ctrls,
        "spine_ctrls": spine_ctrls,
        "spline_ctrls": spline_ctrls,
        "neck_ctrls": neck_ctrls,
        "curl_ctrls": curl_ctrls,
        "spin_ctrls": spin_ctrls,
        "wing_ctrls": wing_ctrls,
        "wing_ik_ctrls": wing_ik_ctrls,
        "appendage_ctrls": appendage_ctrls,
        "category_counts": cat_counts,
    }
    arm_obj["hfw_ctrl_rig_meta"] = json.dumps(rig_meta)

    return {
        "archetype": spec["archetype"],
        "chest_ctrl": chest_ctrl_name,
        "torso_ik_supported": use_torso_ik,
        "has_head_ik": bool(head_ik_meta),
        "has_spline_ik": bool(spline_ik_meta),
        "chainsaw_count": len(spec.get("chainsaws") or []),
        "chainsaw_teeth": len(saw_driver_setups),
        "spinner_count": len(spin_driver_setups),
        "fitted_bone_radii": len(bone_radii),
        "collection_slots": active_slots,
        "control_count": len(ctrl_styles),
        "ik_limb_count": len(ik_setups),
        "tail_ctrl_count": len(tail_ctrls),
        "category_counts": cat_counts,
        "fk_detail_count": sum(cat_counts.values()),
        "driven_def_count": len(driven_def_bones),
        "locked_secondary_count": locked_secondary_count,
        "spec": spec,
    }


# ---------------------------------------------------------------------------
# Blender 5.3 Layered Action / Channelbag Helpers & Animation Importer
# ---------------------------------------------------------------------------
def _create_action_channelbag(arm_obj: Any, action_name: str) -> tuple[Any, Any, Any]:
    act = bpy.data.actions.new(name=action_name)
    act.use_fake_user = True
    if hasattr(act, "slots") and hasattr(act, "layers"):
        slot = act.slots.new(id_type="OBJECT", name=arm_obj.name)
        layer = act.layers.new(name="HFW_Anim")
        strip = layer.strips.new(type="KEYFRAME")
        cb = strip.channelbag(slot, ensure=True)
        return act, slot, cb
    return act, None, act


def assign_action_to_armature(arm_obj: Any, action: Any) -> None:
    if arm_obj.animation_data is None:
        arm_obj.animation_data_create()
    arm_obj.animation_data.action = action
    if action is not None and hasattr(action, "slots") and hasattr(arm_obj.animation_data, "action_slot"):
        slot = None
        for s in action.slots:
            if getattr(s, "target_id_type", "OBJECT") == "OBJECT":
                slot = s
                break
        if slot is None and len(action.slots) > 0:
            slot = action.slots[0]
        if slot is not None:
            try:
                arm_obj.animation_data.action_slot = slot
            except Exception:
                pass


def _write_fcurve_bulk(fcurves: Any, data_path: str, index: int, frames: np.ndarray, values: np.ndarray) -> None:
    fc = fcurves.new(data_path=data_path, index=index)
    n = len(frames)
    fc.keyframe_points.add(n)
    co = np.empty(n * 2, dtype=np.float32)
    co[0::2] = frames
    co[1::2] = values
    fc.keyframe_points.foreach_set("co", co)
    fc.update()


def import_animations_into_armature(
    arm_obj: Any,
    bundle_dir: str | Path,
    *,
    activate_first: bool = True,
    switch_to_anim_mode: bool = True,
) -> list[Any]:
    """Import all decoded `animations.npz` clips from a bundle or animation pack as native Blender Actions."""
    if bpy is None or arm_obj is None or arm_obj.type != "ARMATURE":
        return []

    bundle_path = Path(bundle_dir)
    manifest_path = bundle_path / "assembly.json"
    anim_npz_path = bundle_path / "animations.npz"
    if not manifest_path.exists() or not anim_npz_path.exists():
        return []

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    anim_metas = manifest.get("animations") or []
    if not anim_metas:
        return []

    anim_npz = np.load(anim_npz_path)
    created_actions: list[Any] = []
    pbones = arm_obj.pose.bones
    existing_by_id = {
        str(a.get("hfw_anim_id", "")): a
        for a in bpy.data.actions
        if a.get("hfw_anim_kind") == "game_def" and a.get("hfw_anim_id")
    }

    for am in anim_metas:
        aid = str(am.get("anim_id", ""))
        if aid and aid in existing_by_id:
            created_actions.append(existing_by_id[aid])
            continue
        key = am["key"]
        aname = am["name"]
        num_frames = int(am["num_frames"])
        jnames: list[str] = list(am.get("joint_names") or ())
        if num_frames < 1 or not jnames:
            continue

        loc_arr = anim_npz[f"{key}_loc"]
        rot_arr = anim_npz[f"{key}_rot"]
        scl_arr = anim_npz[f"{key}_scale"]
        has_loc = anim_npz[f"{key}_has_loc"]
        has_rot = anim_npz[f"{key}_has_rot"]
        has_scl = anim_npz[f"{key}_has_scale"]

        act, _slot, cb = _create_action_channelbag(arm_obj, f"HFW_{aname}")
        act["hfw_anim_id"] = aid
        act["hfw_anim_category"] = str(am.get("category", ""))
        act["hfw_anim_description"] = str(am.get("description", ""))
        act["hfw_anim_kind"] = "game_def"
        act["hfw_sample_rate"] = float(am.get("sample_rate", 30.0))

        frames = np.arange(1, num_frames + 1, dtype=np.float32)
        fcurves = cb.fcurves

        for k, bname in enumerate(jnames):
            pb = pbones.get(bname)
            if pb is None:
                continue
            pb.rotation_mode = "QUATERNION"
            if bool(has_loc[k]):
                dp = f'pose.bones["{bname}"].location'
                for axis in range(3):
                    _write_fcurve_bulk(fcurves, dp, axis, frames, loc_arr[:, k, axis])
            if bool(has_rot[k]):
                dp = f'pose.bones["{bname}"].rotation_quaternion'
                for axis in range(4):
                    _write_fcurve_bulk(fcurves, dp, axis, frames, rot_arr[:, k, axis])
            if bool(has_scl[k]):
                dp = f'pose.bones["{bname}"].scale'
                for axis in range(3):
                    _write_fcurve_bulk(fcurves, dp, axis, frames, scl_arr[:, k, axis])

        created_actions.append(act)

    anim_npz.close()

    if created_actions and activate_first:
        first_act = created_actions[0]
        assign_action_to_armature(arm_obj, first_act)
        try:
            fps = int(round(float(first_act.get("hfw_sample_rate", 30.0))))
            bpy.context.scene.render.fps = max(24, min(60, fps))
            f0, f1 = first_act.frame_range
            bpy.context.scene.frame_start = int(round(f0))
            bpy.context.scene.frame_end = max(int(round(f0)) + 1, int(round(f1)))
            bpy.context.scene.frame_set(bpy.context.scene.frame_start)
        except Exception:
            pass
        if switch_to_anim_mode and "hfw_rig_master" in arm_obj:
            arm_obj["hfw_rig_master"] = 0.0
            arm_obj.update_tag()
            bpy.context.view_layer.update()

    return created_actions


# ---------------------------------------------------------------------------
# Bidirectional 1-Click Baking (Game DEF Anim <-> Full IK + FK + Detail Control Rig)
# ---------------------------------------------------------------------------
def _pose_world_to_basis(arm_obj: Any, pb: Any, target_arm_space_mat: Any) -> tuple[Vector, Quaternion, Vector]:
    bone = pb.bone
    if bone.parent is not None:
        parent_pb = arm_obj.pose.bones[bone.parent.name]
        rest_local = bone.parent.matrix_local.inverted() @ bone.matrix_local
        basis = (parent_pb.matrix @ rest_local).inverted() @ target_arm_space_mat
    else:
        basis = bone.matrix_local.inverted() @ target_arm_space_mat
    loc, rot, scl = basis.decompose()
    return loc, rot, scl


def bake_action_to_control_rig(
    arm_obj: Any,
    *,
    frame_start: int | None = None,
    frame_end: int | None = None,
    activate_baked: bool = True,
) -> Any:
    """Bake the active HFW `DEF` game animation onto ALL IK, FK, Spine, Head, Tail, Finger, Toe & Detail `CTRL` bones!"""
    if bpy is None:
        raise RuntimeError("bake_action_to_control_rig must run inside Blender.")
    if arm_obj is None or arm_obj.type != "ARMATURE":
        raise ValueError("Active object must be an Armature.")
    meta_raw = arm_obj.get("hfw_ctrl_rig_meta")
    if not meta_raw:
        build_control_rig(arm_obj, default_master=0.0, rebuild=False)
        meta_raw = arm_obj.get("hfw_ctrl_rig_meta")
    meta = json.loads(meta_raw)

    src_act = arm_obj.animation_data.action if arm_obj.animation_data else None
    if src_act is None:
        raise RuntimeError("Armature has no active Animation Action to bake onto the Control Rig.")

    scene = bpy.context.scene
    if frame_start is None or frame_end is None:
        f0, f1 = src_act.frame_range
        if frame_start is None:
            frame_start = int(round(f0))
        if frame_end is None:
            frame_end = max(frame_start, int(round(f1)))
    frame_list = list(range(int(frame_start), int(frame_end) + 1))
    n_frames = len(frame_list)

    bones = arm_obj.data.bones
    pbones = arm_obj.pose.bones

    fk_pairs = [
        {
            "ctrl": p["ctrl"],
            "def_bone": p["def_bone"],
            "mode": p["mode"],
            "rest_offset": bones[p["def_bone"]].matrix_local.inverted() @ bones[p["ctrl"]].matrix_local,
        }
        for p in meta.get("fk_pairs", [])
        if p["ctrl"] in bones and p["def_bone"] in bones
    ]
    ik_limbs = [
        {
            "target_ctrl": lm["target_ctrl"],
            "pole_ctrl": lm["pole_ctrl"],
            "end_def": lm["end_def"],
            "ik_defs": [b for b in lm["ik_defs"] if b in bones],
            "fallback_dir": Vector(lm["fallback_dir"]),
            "dist_scale": float(lm.get("dist_scale", 0.85)),
            "end_rest_offset": bones[lm["end_def"]].matrix_local.inverted() @ bones[lm["target_ctrl"]].matrix_local,
            "pole_rest_rot": bones[lm["pole_ctrl"]].matrix_local.to_3x3().to_4x4(),
        }
        for lm in meta.get("ik_limbs", [])
        if lm["target_ctrl"] in bones and lm["pole_ctrl"] in bones and lm["end_def"] in bones
    ]

    torso_ik = meta.get("torso_ik") or {}
    has_bake_torso_ik = bool(
        torso_ik.get("chest_ctrl") in bones and torso_ik.get("chest_def") in bones
    )
    torso_extra_ctrls: list[str] = []
    if has_bake_torso_ik:
        torso_extra_ctrls.append(torso_ik["chest_ctrl"])
        if torso_ik.get("mid_ctrl") in bones and torso_ik.get("mid_def") in bones:
            torso_extra_ctrls.append(torso_ik["mid_ctrl"])

    head_ik_m = meta.get("head_ik") or {}
    has_bake_head_ik = bool(
        head_ik_m.get("head_ctrl") in bones and head_ik_m.get("head_def") in bones
    )
    head_extra_ctrls: list[str] = [head_ik_m["head_ctrl"]] if has_bake_head_ik else []

    spline_bake_knots: list[dict[str, Any]] = []
    for sp_m in meta.get("spline_ik") or []:
        for kn in sp_m.get("knots") or []:
            cn, dn = kn.get("ctrl"), kn.get("def_bone")
            if cn in bones and dn in bones:
                spline_bake_knots.append(
                    {
                        "ctrl": cn,
                        "def_bone": dn,
                        "rest_offset": bones[dn].matrix_local.inverted() @ bones[cn].matrix_local,
                    }
                )
    spline_extra_ctrls = [sk["ctrl"] for sk in spline_bake_knots]

    all_ctrls = list(
        dict.fromkeys(
            [p["ctrl"] for p in fk_pairs]
            + [c for lm in ik_limbs for c in (lm["target_ctrl"], lm["pole_ctrl"])]
            + torso_extra_ctrls
            + head_extra_ctrls
            + spline_extra_ctrls
        )
    )
    rec_loc: dict[str, np.ndarray] = {c: np.zeros((n_frames, 3), dtype=np.float32) for c in all_ctrls}
    rec_rot: dict[str, np.ndarray] = {c: np.zeros((n_frames, 4), dtype=np.float32) for c in all_ctrls}
    for c in all_ctrls:
        rec_rot[c][:, 0] = 1.0

    # Map `MCH_Follow_<def>` -> `<def>` so finger/toe/claw FK controls compute exact basis relative to their animated foot/hand!
    follow_to_def: dict[str, str] = {}
    for b in bones:
        if b.name.startswith("MCH_Follow_"):
            cand = b.name[len("MCH_Follow_") :]
            if cand in bones:
                follow_to_def[b.name] = cand

    prev_master = float(arm_obj.get("hfw_rig_master", 1.0))
    prev_frame = scene.frame_current
    arm_obj["hfw_rig_master"] = 0.0
    arm_obj.update_tag()

    # Reset ALL CTRL_* bones (including CTRL_Curl_*, CTRL_Saw_Spin, etc.) to identity before sampling DEF frames
    for pb_c in pbones:
        if pb_c.name.startswith("CTRL_"):
            pb_c.matrix_basis = Matrix.Identity(4)

    for fi, fr in enumerate(frame_list):
        scene.frame_set(fr)
        bpy.context.view_layer.update()

        ctrl_solved_mat: dict[str, Matrix] = {}
        # Populate follow socket world matrices directly from evaluated DEF foot/hand/spine/neck bones!
        for fm_name, d_name in follow_to_def.items():
            ctrl_solved_mat[fm_name] = pbones[d_name].matrix.copy()

        def _resolve_mat(b_obj: Any) -> Matrix:
            """Resolve any parent bone's solved world matrix cleanly, propagating through passive MCH wrappers (e.g. `MCH_Curl_*`)."""
            bn = b_obj.name
            if bn in ctrl_solved_mat:
                return ctrl_solved_mat[bn]
            if b_obj.parent is not None:
                pm = _resolve_mat(b_obj.parent)
                rl = b_obj.parent.matrix_local.inverted() @ b_obj.matrix_local
                res = pm @ rl
            else:
                res = b_obj.matrix_local.copy()
            ctrl_solved_mat[bn] = res
            return res

        for p in fk_pairs:
            cname = p["ctrl"]
            dname = p["def_bone"]
            pb_d = pbones[dname]
            target_mat = pb_d.matrix @ p["rest_offset"]

            bone_c = bones[cname]
            if bone_c.parent is not None:
                p_mat = _resolve_mat(bone_c.parent)
                rest_local = bone_c.parent.matrix_local.inverted() @ bone_c.matrix_local
                basis = (p_mat @ rest_local).inverted() @ target_mat
            else:
                basis = bone_c.matrix_local.inverted() @ target_mat
            loc, rot, _scl = basis.decompose()
            if p["mode"] == "ROTATION":
                loc = Vector((0.0, 0.0, 0.0))
                basis = rot.to_matrix().to_4x4()
            if bone_c.parent is not None:
                p_mat = _resolve_mat(bone_c.parent)
                rest_local = bone_c.parent.matrix_local.inverted() @ bone_c.matrix_local
                ctrl_solved_mat[cname] = p_mat @ rest_local @ basis
            else:
                ctrl_solved_mat[cname] = bone_c.matrix_local @ basis

            rec_loc[cname][fi] = (loc.x, loc.y, loc.z)
            rec_rot[cname][fi] = (rot.w, rot.x, rot.y, rot.z)

        root_solved = ctrl_solved_mat.get("CTRL_Root", Matrix.Identity(4))
        for lm in ik_limbs:
            t_name = lm["target_ctrl"]
            p_name = lm["pole_ctrl"]
            pb_end = pbones[lm["end_def"]]
            target_mat = pb_end.matrix @ lm["end_rest_offset"]

            bone_t = bones[t_name]
            if bone_t.parent is not None:
                p_mat = _resolve_mat(bone_t.parent)
                rest_local = bone_t.parent.matrix_local.inverted() @ bone_t.matrix_local
                basis_t = (p_mat @ rest_local).inverted() @ target_mat
            else:
                basis_t = bone_t.matrix_local.inverted() @ target_mat
            lt, rt, _st = basis_t.decompose()
            rec_loc[t_name][fi] = (lt.x, lt.y, lt.z)
            rec_rot[t_name][fi] = (rt.w, rt.x, rt.y, rt.z)

            pts = [pbones[b].matrix.to_translation() for b in lm["ik_defs"]] + [pb_end.matrix.to_translation()]
            p0 = pts[0]
            p_mid = pts[1] if len(pts) == 3 else (pts[1] * 0.7 + pts[2] * 0.3)
            p_end = pts[-1]
            chord = p_end - p0
            chord_len = max(chord.length, 0.1)
            chord_dir = chord.normalized()
            perp = (p_mid - p0) - chord_dir * (p_mid - p0).dot(chord_dir)
            if perp.length < chord_len * 0.03:
                fb = (root_solved.to_3x3() @ lm["fallback_dir"]).normalized()
                bend_dir = fb - chord_dir * fb.dot(chord_dir)
                bend_dir = bend_dir.normalized() if bend_dir.length > 1e-4 else Vector((0.0, 1.0, 0.0))
            else:
                bend_dir = perp.normalized()
            pole_pos = p_mid + bend_dir * (chord_len * lm["dist_scale"])
            pole_mat = Matrix.Translation(pole_pos) @ lm["pole_rest_rot"]

            bone_p = bones[p_name]
            if bone_p.parent is not None:
                p_mat = _resolve_mat(bone_p.parent)
                rest_local = bone_p.parent.matrix_local.inverted() @ bone_p.matrix_local
                basis_p = (p_mat @ rest_local).inverted() @ pole_mat
            else:
                basis_p = bone_p.matrix_local.inverted() @ pole_mat
            lp, rp, _sp = basis_p.decompose()
            rec_loc[p_name][fi] = (lp.x, lp.y, lp.z)
            rec_rot[p_name][fi] = (rp.w, rp.x, rp.y, rp.z)

        if has_bake_torso_ik:
            cc_nm = torso_ik["chest_ctrl"]
            cd_nm = torso_ik["chest_def"]
            ch_rest_off = bones[cd_nm].matrix_local.inverted() @ bones[cc_nm].matrix_local
            ch_tmat = pbones[cd_nm].matrix @ ch_rest_off
            bone_cc = bones[cc_nm]
            if bone_cc.parent is not None:
                p_mat = _resolve_mat(bone_cc.parent)
                rest_local = bone_cc.parent.matrix_local.inverted() @ bone_cc.matrix_local
                basis_cc = (p_mat @ rest_local).inverted() @ ch_tmat
            else:
                basis_cc = bone_cc.matrix_local.inverted() @ ch_tmat
            lcc, rcc, _scc = basis_cc.decompose()
            rec_loc[cc_nm][fi] = (lcc.x, lcc.y, lcc.z)
            rec_rot[cc_nm][fi] = (rcc.w, rcc.x, rcc.y, rcc.z)

            mc_nm = torso_ik.get("mid_ctrl") or ""
            md_nm = torso_ik.get("mid_def") or ""
            if mc_nm and md_nm and mc_nm in bones and md_nm in bones:
                # Approximate mid-anchor from 50% blend of solved COG & Chest ref positions
                cog_solved = ctrl_solved_mat.get("CTRL_COG", root_solved)
                cog_ref_b = bones.get("MCH_COG_MidRef")
                ch_ref_b = bones.get("MCH_Chest_MidRef")
                if cog_ref_b and ch_ref_b:
                    m_cog_ref = cog_solved @ (bones["CTRL_COG"].matrix_local.inverted() @ cog_ref_b.matrix_local)
                    m_ch_ref = ch_tmat @ (bone_cc.matrix_local.inverted() @ ch_ref_b.matrix_local)
                    anc_pos = (m_cog_ref.to_translation() + m_ch_ref.to_translation()) * 0.5
                    anc_rot = m_cog_ref.to_quaternion().slerp(m_ch_ref.to_quaternion(), 0.5)
                    anc_mat = Matrix.Translation(anc_pos) @ anc_rot.to_matrix().to_4x4()
                else:
                    anc_mat = bones[mc_nm].parent.matrix_local
                md_rest_off = bones[md_nm].matrix_local.inverted() @ bones[mc_nm].matrix_local
                md_tmat = pbones[md_nm].matrix @ md_rest_off
                rest_l_m = bones[mc_nm].parent.matrix_local.inverted() @ bones[mc_nm].matrix_local
                basis_mc = (anc_mat @ rest_l_m).inverted() @ md_tmat
                lmc, rmc, _smc = basis_mc.decompose()
                rec_loc[mc_nm][fi] = (lmc.x, lmc.y, lmc.z)
                rec_rot[mc_nm][fi] = (rmc.w, rmc.x, rmc.y, rmc.z)

        if has_bake_head_ik:
            hc_nm = head_ik_m["head_ctrl"]
            hd_nm = head_ik_m["head_def"]
            hd_rest_off = bones[hd_nm].matrix_local.inverted() @ bones[hc_nm].matrix_local
            hd_tmat = pbones[hd_nm].matrix @ hd_rest_off
            bone_hc = bones[hc_nm]
            hd_sp = max(0.0, min(1.0, float(arm_obj.get("hfw_head_ik_space", 0.0))))
            ch_ref = bones.get("MCH_HeadIK_ChestRef")
            rt_ref = bones.get("MCH_HeadIK_RootRef")
            if ch_ref is not None and rt_ref is not None and bone_hc.parent is not None:
                m_ch = _resolve_mat(ch_ref)
                m_rt = _resolve_mat(rt_ref)
                anc_pos = m_ch.to_translation().lerp(m_rt.to_translation(), hd_sp)
                anc_rot = m_ch.to_quaternion().slerp(m_rt.to_quaternion(), hd_sp)
                p_mat = Matrix.Translation(anc_pos) @ anc_rot.to_matrix().to_4x4()
                rest_local = bone_hc.parent.matrix_local.inverted() @ bone_hc.matrix_local
                basis_hc = (p_mat @ rest_local).inverted() @ hd_tmat
            elif bone_hc.parent is not None:
                p_mat = _resolve_mat(bone_hc.parent)
                rest_local = bone_hc.parent.matrix_local.inverted() @ bone_hc.matrix_local
                basis_hc = (p_mat @ rest_local).inverted() @ hd_tmat
            else:
                basis_hc = bone_hc.matrix_local.inverted() @ hd_tmat
            lhc, rhc, _shc = basis_hc.decompose()
            rec_loc[hc_nm][fi] = (lhc.x, lhc.y, lhc.z)
            rec_rot[hc_nm][fi] = (rhc.w, rhc.x, rhc.y, rhc.z)

        for sk in spline_bake_knots:
            sc_nm = sk["ctrl"]
            sd_nm = sk["def_bone"]
            st_mat = pbones[sd_nm].matrix @ sk["rest_offset"]
            bone_sc = bones[sc_nm]
            if bone_sc.parent is not None:
                p_mat = _resolve_mat(bone_sc.parent)
                rest_local = bone_sc.parent.matrix_local.inverted() @ bone_sc.matrix_local
                basis_sc = (p_mat @ rest_local).inverted() @ st_mat
            else:
                basis_sc = bone_sc.matrix_local.inverted() @ st_mat
            lsc, rsc, _ssc = basis_sc.decompose()
            rec_loc[sc_nm][fi] = (lsc.x, lsc.y, lsc.z)
            rec_rot[sc_nm][fi] = (rsc.w, rsc.x, rsc.y, rsc.z)

    base_name = re.sub(r"_CTRL$", "", src_act.name)
    baked_act, _slot, cb = _create_action_channelbag(arm_obj, f"{base_name}_CTRL")
    baked_act["hfw_anim_id"] = str(src_act.get("hfw_anim_id", ""))
    baked_act["hfw_anim_kind"] = "control_rig"
    baked_act["hfw_source_action"] = src_act.name

    frames_np = np.array(frame_list, dtype=np.float32)
    fcurves = cb.fcurves
    for cname in all_ctrls:
        rq = rec_rot[cname]
        for fi in range(1, n_frames):
            if float(np.dot(rq[fi - 1], rq[fi])) < 0.0:
                rq[fi] = -rq[fi]
        if cname in pbones:
            pbones[cname].rotation_mode = "QUATERNION"
        dp_loc = f'pose.bones["{cname}"].location'
        dp_rot = f'pose.bones["{cname}"].rotation_quaternion'
        if float(np.max(np.abs(rec_loc[cname]))) > 1e-5:
            for ax in range(3):
                _write_fcurve_bulk(fcurves, dp_loc, ax, frames_np, rec_loc[cname][:, ax])
        for ax in range(4):
            _write_fcurve_bulk(fcurves, dp_rot, ax, frames_np, rq[:, ax])

    if activate_baked:
        assign_action_to_armature(arm_obj, baked_act)
        arm_obj["hfw_rig_master"] = 1.0
        for p_nm, _, _ in _RIG_GROUP_PROPS:
            if p_nm.startswith("hfw_rig_"):
                arm_obj[p_nm] = 1.0
    else:
        arm_obj["hfw_rig_master"] = prev_master

    arm_obj.update_tag()
    scene.frame_set(prev_frame)
    bpy.context.view_layer.update()
    return baked_act


def bake_control_rig_to_deform(
    arm_obj: Any,
    *,
    frame_start: int | None = None,
    frame_end: int | None = None,
    activate_baked: bool = True,
) -> Any:
    """Bake the current Control Rig animation back onto pure `DEF` skeleton keyframes (`hfw_rig_master = 0.0`) for UE5/FBX export!"""
    if bpy is None:
        raise RuntimeError("bake_control_rig_to_deform must run inside Blender.")
    if arm_obj is None or arm_obj.type != "ARMATURE":
        raise ValueError("Active object must be an Armature.")

    scene = bpy.context.scene
    src_act = arm_obj.animation_data.action if arm_obj.animation_data else None
    if frame_start is None:
        frame_start = int(round(src_act.frame_range[0])) if src_act else scene.frame_start
    if frame_end is None:
        frame_end = int(round(src_act.frame_range[1])) if src_act else scene.frame_end
    frame_list = list(range(int(frame_start), max(int(frame_start), int(frame_end)) + 1))
    n_frames = len(frame_list)

    bones = arm_obj.data.bones
    pbones = arm_obj.pose.bones
    def_names = [b.name for b in bones if not b.name.startswith(("CTRL_", "MCH_", "WGT_"))]

    rec_loc = np.zeros((n_frames, len(def_names), 3), dtype=np.float32)
    rec_rot = np.zeros((n_frames, len(def_names), 4), dtype=np.float32)
    rec_rot[:, :, 0] = 1.0

    prev_frame = scene.frame_current
    for fi, fr in enumerate(frame_list):
        scene.frame_set(fr)
        bpy.context.view_layer.update()
        for di, dname in enumerate(def_names):
            pb = pbones[dname]
            loc, rot, _scl = _pose_world_to_basis(arm_obj, pb, pb.matrix)
            rec_loc[fi, di] = (loc.x, loc.y, loc.z)
            rec_rot[fi, di] = (rot.w, rot.x, rot.y, rot.z)

    base_name = re.sub(r"(_CTRL|_DEF)$", "", src_act.name) if src_act else f"{arm_obj.name}_Pose"
    def_act, _slot, cb = _create_action_channelbag(arm_obj, f"{base_name}_DEF")
    def_act["hfw_anim_kind"] = "baked_def"

    frames_np = np.array(frame_list, dtype=np.float32)
    fcurves = cb.fcurves
    for di, dname in enumerate(def_names):
        rq = rec_rot[:, di, :]
        for fi in range(1, n_frames):
            if float(np.dot(rq[fi - 1], rq[fi])) < 0.0:
                rq[fi] = -rq[fi]
        has_t = float(np.max(np.abs(rec_loc[:, di, :]))) > 1e-4
        has_r = float(np.max(np.abs(rq - np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)))) > 1e-4
        if not (has_t or has_r):
            continue
        pbones[dname].rotation_mode = "QUATERNION"
        if has_t:
            dp_l = f'pose.bones["{dname}"].location'
            for ax in range(3):
                _write_fcurve_bulk(fcurves, dp_l, ax, frames_np, rec_loc[:, di, ax])
        if has_r:
            dp_r = f'pose.bones["{dname}"].rotation_quaternion'
            for ax in range(4):
                _write_fcurve_bulk(fcurves, dp_r, ax, frames_np, rq[:, ax])

    if activate_baked:
        assign_action_to_armature(arm_obj, def_act)
        arm_obj["hfw_rig_master"] = 0.0
        arm_obj.update_tag()

    scene.frame_set(prev_frame)
    bpy.context.view_layer.update()
    return def_act


def sync_ik_fk_layer_visibility(arm_obj: Any) -> None:
    """Sync Torso/Chest, Head IK, Spline IK, Legs, Front Legs, Arms/Forelimbs & Wings dynamic Bone Collection visibility."""
    if bpy is None or arm_obj is None or arm_obj.type != "ARMATURE":
        return
    colls = arm_obj.data.collections
    slots = _get_rig_collection_slots(arm_obj)
    meta = {}
    try:
        meta = json.loads(arm_obj.get("hfw_ctrl_rig_meta") or "{}")
    except Exception:
        meta = {}
    torso_ik = float(arm_obj.get("hfw_ik_fk_torso", 1.0)) >= 0.5
    head_ik = float(arm_obj.get("hfw_ik_fk_head", 0.0)) >= 0.5
    spline_ik = float(arm_obj.get("hfw_ik_fk_spline", 0.0)) >= 0.5
    legs_ik = float(arm_obj.get("hfw_ik_fk_legs", 1.0)) >= 0.5
    flegs_ik = float(arm_obj.get("hfw_ik_fk_front_legs", 1.0)) >= 0.5
    arms_ik = float(arm_obj.get("hfw_ik_fk_arms", 1.0)) >= 0.5
    wings_ik = float(arm_obj.get("hfw_ik_fk_wings", 0.0)) >= 0.5
    pairs = [
        ("head_ik", head_ik),
        ("legs_ik", legs_ik),
        ("legs_fk", not legs_ik),
        ("front_legs_ik", flegs_ik),
        ("front_legs_fk", not flegs_ik),
        ("arms_ik", arms_ik),
        ("arms_fk", not arms_ik),
        ("wings_ik", wings_ik),
        ("wings_fk", not wings_ik),
    ]
    if slots.get("torso_ik") in colls:
        fk_slot = "hood_fk" if (slots.get("hood_fk") in colls) else "spine_fk"
        pairs.extend([("torso_ik", torso_ik), (fk_slot, not torso_ik)])
    if slots.get("spline_ik") in colls:
        pairs.append(("spline_ik", spline_ik))
        for sp_info in meta.get("spline_ik") or []:
            sk = sp_info.get("kind")
            if sk == "spine" and slots.get("spine_fk") in colls:
                pairs.append(("spine_fk", not spline_ik))
            elif sk == "tail" and slots.get("tail") in colls:
                pairs.append(("tail", not spline_ik))
            elif sk == "appendage" and slots.get("appendages") in colls:
                pairs.append(("appendages", not spline_ik))
    for skey, vis in pairs:
        cname = slots.get(skey)
        if cname and cname in colls:
            colls[cname].is_visible = vis


def snap_limb_fk_to_ik(arm_obj: Any, group: str = "legs") -> int:
    """Snap `CTRL_FK_*` limb controls cleanly onto the current evaluated IK/DEF limb pose (zero pop!)."""
    if bpy is None or arm_obj is None or arm_obj.type != "ARMATURE":
        return 0
    meta_raw = arm_obj.get("hfw_ctrl_rig_meta")
    if not meta_raw:
        return 0
    meta = json.loads(meta_raw)
    chains = (meta.get("limb_fk_chains") or {}).get(group, [])
    if not chains:
        return 0

    bpy.context.view_layer.update()
    bones = arm_obj.data.bones
    pbones = arm_obj.pose.bones

    # Snapshot current world (`arm`-space) matrices of every limb DEF bone BEFORE modifying FK controls
    def_mats: dict[str, Matrix] = {}
    for ch in chains:
        for p in ch.get("fk_pairs", []):
            dn = p["def_bone"]
            if dn in pbones:
                def_mats[dn] = pbones[dn].matrix.copy()

    snapped = 0
    for ch in chains:
        solved_ctrl: dict[str, Matrix] = {}
        for p in ch.get("fk_pairs", []):
            cn = p["ctrl"]
            dn = p["def_bone"]
            if cn not in bones or dn not in bones or dn not in def_mats:
                continue
            bone_c = bones[cn]
            pb_c = pbones[cn]
            rest_offset = bones[dn].matrix_local.inverted() @ bone_c.matrix_local
            target_mat = def_mats[dn] @ rest_offset
            if bone_c.parent is not None:
                p_mat = solved_ctrl.get(bone_c.parent.name, pbones[bone_c.parent.name].matrix)
                rest_local = bone_c.parent.matrix_local.inverted() @ bone_c.matrix_local
                basis = (p_mat @ rest_local).inverted() @ target_mat
            else:
                basis = bone_c.matrix_local.inverted() @ target_mat
            _loc, rot, _scl = basis.decompose()
            rot_mat = rot.to_matrix().to_4x4()
            pb_c.rotation_mode = "QUATERNION"
            pb_c.matrix_basis = rot_mat
            if bone_c.parent is not None:
                p_mat = solved_ctrl.get(bone_c.parent.name, pbones[bone_c.parent.name].matrix)
                rest_local = bone_c.parent.matrix_local.inverted() @ bone_c.matrix_local
                solved_ctrl[cn] = p_mat @ rest_local @ rot_mat
            else:
                solved_ctrl[cn] = bone_c.matrix_local @ rot_mat
            snapped += 1

    arm_obj.update_tag()
    bpy.context.view_layer.update()
    return snapped


def snap_limb_ik_to_fk(arm_obj: Any, group: str = "legs") -> int:
    """Snap `CTRL_IK_*` foot/hand boxes & `CTRL_Pole_*` knee/elbow diamonds onto the current FK/DEF pose (zero pop!)."""
    if bpy is None or arm_obj is None or arm_obj.type != "ARMATURE":
        return 0
    meta_raw = arm_obj.get("hfw_ctrl_rig_meta")
    if not meta_raw:
        return 0
    meta = json.loads(meta_raw)
    limbs = [lm for lm in (meta.get("ik_limbs") or []) if lm.get("group", "legs") == group]
    if not limbs:
        return 0

    bpy.context.view_layer.update()
    bones = arm_obj.data.bones
    pbones = arm_obj.pose.bones
    root_pb = pbones.get("CTRL_Root")
    root_mat = root_pb.matrix.copy() if root_pb is not None else Matrix.Identity(4)

    snapped = 0
    for lm in limbs:
        t_name = lm["target_ctrl"]
        p_name = lm["pole_ctrl"]
        end_def = lm["end_def"]
        ik_defs = [b for b in lm["ik_defs"] if b in pbones]
        if t_name not in bones or p_name not in bones or end_def not in pbones or len(ik_defs) < 2:
            continue

        pb_end = pbones[end_def]
        end_rest_offset = bones[end_def].matrix_local.inverted() @ bones[t_name].matrix_local
        target_mat = pb_end.matrix @ end_rest_offset

        bone_t = bones[t_name]
        if bone_t.parent is not None:
            p_mat = pbones[bone_t.parent.name].matrix
            rest_local = bone_t.parent.matrix_local.inverted() @ bone_t.matrix_local
            basis_t = (p_mat @ rest_local).inverted() @ target_mat
        else:
            basis_t = bone_t.matrix_local.inverted() @ target_mat
        pbones[t_name].rotation_mode = "QUATERNION"
        pbones[t_name].matrix_basis = basis_t

        pts = [pbones[b].matrix.to_translation() for b in ik_defs] + [pb_end.matrix.to_translation()]
        p0 = pts[0]
        p_mid = pts[1] if len(pts) == 3 else (pts[1] * 0.7 + pts[2] * 0.3)
        p_end = pts[-1]
        chord = p_end - p0
        chord_len = max(chord.length, 0.1)
        chord_dir = chord.normalized()
        perp = (p_mid - p0) - chord_dir * (p_mid - p0).dot(chord_dir)
        if perp.length < chord_len * 0.03:
            fb = (root_mat.to_3x3() @ Vector(lm["fallback_dir"])).normalized()
            bend_dir = fb - chord_dir * fb.dot(chord_dir)
            bend_dir = bend_dir.normalized() if bend_dir.length > 1e-4 else Vector((0.0, 1.0, 0.0))
        else:
            bend_dir = perp.normalized()
        pole_pos = p_mid + bend_dir * (chord_len * float(lm.get("dist_scale", 0.85)))
        pole_mat = Matrix.Translation(pole_pos) @ bones[p_name].matrix_local.to_3x3().to_4x4()

        bone_p = bones[p_name]
        if bone_p.parent is not None:
            p_mat = pbones[bone_p.parent.name].matrix
            rest_local = bone_p.parent.matrix_local.inverted() @ bone_p.matrix_local
            basis_p = (p_mat @ rest_local).inverted() @ pole_mat
        else:
            basis_p = bone_p.matrix_local.inverted() @ pole_mat
        pbones[p_name].rotation_mode = "QUATERNION"
        pbones[p_name].matrix_basis = basis_p
        snapped += 1

    arm_obj.update_tag()
    bpy.context.view_layer.update()
    return snapped


def snap_torso_ik_to_fk(arm_obj: Any) -> int:
    """Snap `CTRL_Chest` & `CTRL_Spine_Mid` cleanly onto the current evaluated Spine/Chest FK pose (minimal pop!)."""
    if bpy is None or arm_obj is None or arm_obj.type != "ARMATURE":
        return 0
    meta_raw = arm_obj.get("hfw_ctrl_rig_meta")
    if not meta_raw:
        return 0
    meta = json.loads(meta_raw)
    tik = meta.get("torso_ik") or {}
    cc_nm = tik.get("chest_ctrl")
    cd_nm = tik.get("chest_def")
    if not cc_nm or not cd_nm:
        return 0
    bones = arm_obj.data.bones
    pbones = arm_obj.pose.bones
    if cc_nm not in bones or cd_nm not in pbones:
        return 0

    bpy.context.view_layer.update()
    cd_mat = pbones[cd_nm].matrix.copy()
    mc_nm = tik.get("mid_ctrl") or ""
    md_nm = tik.get("mid_def") or ""
    md_mat = pbones[md_nm].matrix.copy() if (md_nm and md_nm in pbones) else None

    # 1. Snap CTRL_Chest to evaluated chest_def world transform
    ch_off = bones[cd_nm].matrix_local.inverted() @ bones[cc_nm].matrix_local
    ch_target = cd_mat @ ch_off
    bone_cc = bones[cc_nm]
    if bone_cc.parent is not None:
        p_mat = pbones[bone_cc.parent.name].matrix
        rest_local = bone_cc.parent.matrix_local.inverted() @ bone_cc.matrix_local
        basis_cc = (p_mat @ rest_local).inverted() @ ch_target
    else:
        basis_cc = bone_cc.matrix_local.inverted() @ ch_target
    pbones[cc_nm].rotation_mode = "QUATERNION"
    pbones[cc_nm].matrix_basis = basis_cc
    snapped = 1

    # 2. Update MCH_TorsoMidAnchor then snap CTRL_Spine_Mid to evaluated mid_def!
    if mc_nm and md_mat is not None and mc_nm in bones:
        arm_obj.update_tag()
        bpy.context.view_layer.update()
        md_off = bones[md_nm].matrix_local.inverted() @ bones[mc_nm].matrix_local
        md_target = md_mat @ md_off
        bone_mc = bones[mc_nm]
        if bone_mc.parent is not None:
            p_mat = pbones[bone_mc.parent.name].matrix
            rest_local = bone_mc.parent.matrix_local.inverted() @ bone_mc.matrix_local
            basis_mc = (p_mat @ rest_local).inverted() @ md_target
        else:
            basis_mc = bone_mc.matrix_local.inverted() @ md_target
        pbones[mc_nm].rotation_mode = "QUATERNION"
        pbones[mc_nm].matrix_basis = basis_mc
        snapped += 1

    arm_obj.update_tag()
    bpy.context.view_layer.update()
    return snapped


def snap_head_ik_to_fk(arm_obj: Any) -> int:
    """Snap `CTRL_IK_Head` cleanly onto the current evaluated Neck/Head FK pose (zero pop!)."""
    if bpy is None or arm_obj is None or arm_obj.type != "ARMATURE":
        return 0
    meta_raw = arm_obj.get("hfw_ctrl_rig_meta")
    if not meta_raw:
        return 0
    meta = json.loads(meta_raw)
    hik = meta.get("head_ik") or {}
    hc_nm = hik.get("head_ctrl")
    hd_nm = hik.get("head_def")
    if not hc_nm or not hd_nm:
        return 0
    bones = arm_obj.data.bones
    pbones = arm_obj.pose.bones
    if hc_nm not in bones or hd_nm not in pbones:
        return 0

    bpy.context.view_layer.update()
    hd_mat = pbones[hd_nm].matrix.copy()
    h_off = bones[hd_nm].matrix_local.inverted() @ bones[hc_nm].matrix_local
    h_target = hd_mat @ h_off
    bone_hc = bones[hc_nm]
    if bone_hc.parent is not None:
        p_mat = pbones[bone_hc.parent.name].matrix
        rest_local = bone_hc.parent.matrix_local.inverted() @ bone_hc.matrix_local
        basis_hc = (p_mat @ rest_local).inverted() @ h_target
    else:
        basis_hc = bone_hc.matrix_local.inverted() @ h_target
    pbones[hc_nm].matrix_basis = basis_hc
    arm_obj.update_tag()
    bpy.context.view_layer.update()
    return 1


def snap_spline_ik_to_fk(arm_obj: Any) -> int:
    """Snap all `CTRL_Spline_*` gimbal knots cleanly onto the current evaluated FK spine/tail/trunk pose!"""
    if bpy is None or arm_obj is None or arm_obj.type != "ARMATURE":
        return 0
    meta_raw = arm_obj.get("hfw_ctrl_rig_meta")
    if not meta_raw:
        return 0
    meta = json.loads(meta_raw)
    splines = meta.get("spline_ik") or []
    if not splines:
        return 0
    bones = arm_obj.data.bones
    pbones = arm_obj.pose.bones
    bpy.context.view_layer.update()

    # Snapshot current DEF world transforms first
    def_mats = {
        kn["def_bone"]: pbones[kn["def_bone"]].matrix.copy()
        for sp in splines
        for kn in sp.get("knots") or []
        if kn["def_bone"] in pbones
    }
    snapped = 0
    for sp in splines:
        for kn in sp.get("knots") or []:
            cn = kn["ctrl"]
            dn = kn["def_bone"]
            if cn not in bones or dn not in def_mats:
                continue
            rest_off = bones[dn].matrix_local.inverted() @ bones[cn].matrix_local
            target_mat = def_mats[dn] @ rest_off
            bone_c = bones[cn]
            if bone_c.parent is not None:
                p_mat = pbones[bone_c.parent.name].matrix
                rest_local = bone_c.parent.matrix_local.inverted() @ bone_c.matrix_local
                basis = (p_mat @ rest_local).inverted() @ target_mat
            else:
                basis = bone_c.matrix_local.inverted() @ target_mat
            pbones[cn].matrix_basis = basis
            snapped += 1
    arm_obj.update_tag()
    bpy.context.view_layer.update()
    return snapped


def pose_serpent_wave(
    arm_obj: Any,
    *,
    style: str = "wave",
    amplitude: float = 1.0,
    cycles: float = 1.25,
    phase: float = 0.0,
) -> int:
    """Apply an instant smooth serpentine S-wave (`wave`) or coiled ambush (`coil`) pose onto Spline IK handles!"""
    if bpy is None or arm_obj is None or arm_obj.type != "ARMATURE":
        return 0
    meta_raw = arm_obj.get("hfw_ctrl_rig_meta")
    if not meta_raw:
        return 0
    meta = json.loads(meta_raw)
    splines = meta.get("spline_ik") or []
    if not splines:
        return 0

    arm_obj["hfw_rig_master"] = 1.0
    arm_obj["hfw_ik_fk_spline"] = 1.0
    sync_ik_fk_layer_visibility(arm_obj)
    arm_obj.update_tag()
    bpy.context.view_layer.update()

    bones = arm_obj.data.bones
    pbones = arm_obj.pose.bones
    posed = 0
    for sp in splines:
        knots = [kn["ctrl"] for kn in (sp.get("knots") or []) if kn["ctrl"] in bones]
        if len(knots) < 3:
            continue
        p0 = bones[knots[0]].head_local.copy()
        p_end = bones[knots[-1]].head_local.copy()
        chain_len = max(1.0, (p_end - p0).length)
        axis_dir = (p_end - p0).normalized() if (p_end - p0).length > 1e-4 else Vector((0.0, -1.0, 0.0))
        side_dir = axis_dir.cross(Vector((0.0, 0.0, 1.0)))
        side_dir = side_dir.normalized() if side_dir.length > 1e-4 else Vector((1.0, 0.0, 0.0))

        # Compute target knot world positions along a smooth sinusoidal slither wave OR spiral ground coil!
        n_k = len(knots)
        new_pts: list[Vector] = []
        for ki in range(n_k):
            u = ki / max(1, n_k - 1)
            if style == "coil":
                # Smooth ground-hugging spiral coil tucked around the base!
                theta = u * math.pi * 2.15 + phase
                rad = chain_len * (0.14 + 0.16 * u) * max(0.3, amplitude)
                if ki == 0:
                    pt = p0.copy()
                else:
                    cx = p0.x + side_dir.x * (rad * math.sin(theta)) + axis_dir.x * (rad * (1.0 - math.cos(theta)))
                    cy = p0.y + side_dir.y * (rad * math.sin(theta)) + axis_dir.y * (rad * (1.0 - math.cos(theta)))
                    cz = max(0.45, p0.z * (1.0 - 0.18 * u))
                    pt = Vector((cx, cy, cz))
            else:
                # Classic lateral sinusoidal S-wave with gentle vertical undulation & natural foreshortening!
                ang = u * math.pi * 2.0 * cycles + phase
                amp_lat = chain_len * 0.11 * amplitude * math.sin(u * math.pi * 0.85)
                amp_z = chain_len * 0.015 * amplitude * math.sin(ang * 2.0) * u
                fwd_dist = chain_len * u * (1.0 - 0.08 * min(1.5, abs(amplitude)))
                pt = p0 + axis_dir * fwd_dist + side_dir * (amp_lat * math.sin(ang)) + Vector((0.0, 0.0, amp_z))
            new_pts.append(pt)

        # Orient every knot cleanly along its local spline tangent so B-Bone segments stay C1-smooth with zero kinks!
        for ki, kn_nm in enumerate(knots):
            if ki == 0:
                tang = new_pts[1] - new_pts[0]
            elif ki == n_k - 1:
                tang = new_pts[-1] - new_pts[-2]
            else:
                tang = new_pts[ki + 1] - new_pts[ki - 1]
            tang = tang.normalized() if tang.length > 1e-4 else axis_dir
            up_v = Vector((0.0, 0.0, 1.0))
            x_v = tang.cross(up_v)
            x_v = x_v.normalized() if x_v.length > 1e-4 else Vector((1.0, 0.0, 0.0))
            z_v = x_v.cross(tang).normalized()
            target_m = Matrix((
                (x_v.x, tang.x, z_v.x, new_pts[ki].x),
                (x_v.y, tang.y, z_v.y, new_pts[ki].y),
                (x_v.z, tang.z, z_v.z, new_pts[ki].z),
                (0.0, 0.0, 0.0, 1.0),
            ))
            bone_k = bones[kn_nm]
            if bone_k.parent is not None:
                p_mat = pbones[bone_k.parent.name].matrix
                rest_local = bone_k.parent.matrix_local.inverted() @ bone_k.matrix_local
                basis = (p_mat @ rest_local).inverted() @ target_m
            else:
                basis = bone_k.matrix_local.inverted() @ target_m
            pbones[kn_nm].matrix_basis = basis
            posed += 1

    arm_obj.update_tag()
    bpy.context.view_layer.update()
    return posed


def switch_limb_ik_fk(
    arm_obj: Any,
    group: str = "legs",
    *,
    to_ik: bool = True,
    snap_pose: bool = True,
) -> dict[str, Any]:
    """Switch Torso/Chest, Head, Spline, Legs, Front Legs, Arms, or Wings between IK and FK with auto zero-pop snapping!"""
    if bpy is None or arm_obj is None or arm_obj.type != "ARMATURE":
        return {"snapped": 0}
    prop_map = {
        "torso": "hfw_ik_fk_torso",
        "head": "hfw_ik_fk_head",
        "spline": "hfw_ik_fk_spline",
        "legs": "hfw_ik_fk_legs",
        "front_legs": "hfw_ik_fk_front_legs",
        "arms": "hfw_ik_fk_arms",
        "wings": "hfw_ik_fk_wings",
    }
    prop_key = prop_map.get(group, "hfw_ik_fk_legs")
    snapped = 0
    if snap_pose:
        if group == "torso":
            snapped = snap_torso_ik_to_fk(arm_obj) if to_ik else snap_limb_fk_to_ik(arm_obj, group="torso")
        elif group == "head":
            snapped = snap_head_ik_to_fk(arm_obj) if to_ik else snap_limb_fk_to_ik(arm_obj, group="head")
        elif group == "spline":
            snapped = snap_spline_ik_to_fk(arm_obj) if to_ik else snap_limb_fk_to_ik(arm_obj, group="spline")
        else:
            snapped = snap_limb_ik_to_fk(arm_obj, group=group) if to_ik else snap_limb_fk_to_ik(arm_obj, group=group)
    arm_obj[prop_key] = 1.0 if to_ik else 0.0
    sync_ik_fk_layer_visibility(arm_obj)
    arm_obj.update_tag()
    bpy.context.view_layer.update()
    return {"group": group, "mode": "IK" if to_ik else "FK", "snapped": snapped}


def select_ctrl_chain(arm_obj: Any, group: str = "tail") -> int:
    """Select Chest/Hood, Head, Spline, Neck, Spine, Tail, Wings, Trunk, Curls, or Spinners in Pose Mode!"""
    if bpy is None or arm_obj is None or arm_obj.type != "ARMATURE":
        return 0
    meta_raw = arm_obj.get("hfw_ctrl_rig_meta")
    if not meta_raw:
        return 0
    meta = json.loads(meta_raw)
    if group == "chest":
        use_fk_ch = (
            bool(meta.get("torso_ik"))
            and float(arm_obj.get("hfw_ik_fk_torso", 1.0)) < 0.5
            and "CTRL_FK_Chest" in arm_obj.pose.bones
        )
        cc = "CTRL_FK_Chest" if use_fk_ch else (meta.get("chest_ctrl") or ("CTRL_Chest" if "CTRL_Chest" in arm_obj.pose.bones else ""))
        names = [cc] if cc else []
    elif group == "head":
        use_ik_hd = bool(meta.get("head_ik")) and float(arm_obj.get("hfw_ik_fk_head", 0.0)) >= 0.5 and "CTRL_IK_Head" in arm_obj.pose.bones
        hc = "CTRL_IK_Head" if use_ik_hd else ("CTRL_Head" if "CTRL_Head" in arm_obj.pose.bones else "")
        names = [hc] if hc else []
    elif group == "spline":
        names = list(meta.get("spline_ctrls") or [])
    elif group == "curls":
        names = list(meta.get("curl_ctrls") or [])
    elif group == "spinners":
        names = list(meta.get("spin_ctrls") or [])
    elif group == "neck":
        names = list(meta.get("neck_ctrls") or [])
    elif group == "wings":
        if float(arm_obj.get("hfw_ik_fk_wings", 0.0)) >= 0.5 and meta.get("wing_ik_ctrls"):
            names = list(meta.get("wing_ik_ctrls") or [])
        else:
            names = list(meta.get("wing_ctrls") or [])
    elif group == "appendages":
        names = list(meta.get("appendage_ctrls") or [])
    else:
        names = list(meta.get("tail_ctrls" if group == "tail" else "spine_ctrls", []))
    if not names:
        return 0

    bpy.context.view_layer.objects.active = arm_obj
    if bpy.context.mode != "POSE":
        bpy.ops.object.mode_set(mode="POSE")
    slots = _get_rig_collection_slots(arm_obj)
    if group == "tail":
        tc = slots.get("tail", "Tail")
        if tc in arm_obj.data.collections:
            arm_obj.data.collections[tc].is_visible = True
    elif group == "spline":
        sc = slots.get("spline_ik")
        if sc and sc in arm_obj.data.collections:
            arm_obj.data.collections[sc].is_visible = True
    elif group == "head":
        for sk in ("head_ik", "body"):
            c = slots.get(sk)
            if c and c in arm_obj.data.collections:
                arm_obj.data.collections[c].is_visible = True
    elif group == "curls":
        hc = slots.get("hands_feet")
        if hc and hc in arm_obj.data.collections:
            arm_obj.data.collections[hc].is_visible = True
    elif group == "spinners":
        sp = slots.get("spinners")
        if sp and sp in arm_obj.data.collections:
            arm_obj.data.collections[sp].is_visible = True
    elif group == "chest":
        for sk in ("torso_ik", "hood_fk", "spine_fk", "body") if names[0] == "CTRL_FK_Chest" else ("torso_ik", "body"):
            c = slots.get(sk)
            if c and c in arm_obj.data.collections:
                arm_obj.data.collections[c].is_visible = True
    elif group == "neck":
        for sk in ("hood_fk", "body"):
            c = slots.get(sk)
            if c and c in arm_obj.data.collections:
                arm_obj.data.collections[c].is_visible = True
    elif group == "spine":
        for sk in ("spine_fk", "body"):
            c = slots.get(sk)
            if c and c in arm_obj.data.collections:
                arm_obj.data.collections[c].is_visible = True
    elif group == "wings":
        sk = "wings_ik" if float(arm_obj.get("hfw_ik_fk_wings", 0.0)) >= 0.5 else "wings_fk"
        wc = slots.get(sk) or slots.get("wings")
        if wc and wc in arm_obj.data.collections:
            arm_obj.data.collections[wc].is_visible = True
    elif group == "appendages":
        ac = slots.get("appendages")
        if ac and ac in arm_obj.data.collections:
            arm_obj.data.collections[ac].is_visible = True

    for pb in arm_obj.pose.bones:
        try:
            pb.select = pb.name in names
        except Exception:
            pass
        try:
            pb.bone.select = pb.name in names
        except Exception:
            pass
    if names and names[0] in arm_obj.data.bones:
        arm_obj.data.bones.active = arm_obj.data.bones[names[0]]
    try:
        bpy.context.scene.tool_settings.transform_pivot_point = "INDIVIDUAL_ORIGINS"
    except Exception:
        pass
    return len(names)


def _find_mirror_ctrl_pairs(arm_obj: Any, source_side: str = "L") -> list[tuple[str, str]]:
    """Pair left/right `CTRL_*` bones cleanly by exact name swap AND geometric + stem fallback (handles sequenced frills/plates!)."""
    src_s = "L" if str(source_side).upper().startswith("L") else "R"
    dst_s = "R" if src_s == "L" else "L"
    bones = arm_obj.data.bones
    ctrl_names = [b.name for b in bones if b.name.startswith("CTRL_")]
    ctrl_set = set(ctrl_names)

    def _side_of_ctrl(nm: str) -> str:
        if re.search(r"_L(?:_|$)", nm):
            return "L"
        if re.search(r"_R(?:_|$)", nm):
            return "R"
        return "C"

    def _swap_name(nm: str) -> str:
        return re.sub(rf"_{src_s}(?=_|$)", f"_{dst_s}", nm)

    pairs: list[tuple[str, str]] = []
    used_src: set[str] = set()
    used_dst: set[str] = set()

    # Pass 1: Direct exact name swap (`CTRL_IK_Leg_L` <-> `CTRL_IK_Leg_R`, `CTRL_FK_Wing_L_0` <-> `CTRL_FK_Wing_R_0`, etc.)
    for nm in ctrl_names:
        if _side_of_ctrl(nm) != src_s:
            continue
        cand = _swap_name(nm)
        if cand != nm and cand in ctrl_set and cand not in used_dst:
            pairs.append((nm, cand))
            used_src.add(nm)
            used_dst.add(cand)

    # Pass 2: Geometric + stem match for sequenced secondary controls (`CTRL_Frill_feather_L_00_00` <-> `CTRL_Frill_feather_R_01_00`)
    def _norm_stem(nm: str) -> str:
        s = re.sub(r"_[LR]_\d+(?=_|$)", "_X", nm)
        s = re.sub(r"_[LR](?=_|$)", "_X", s)
        return s.lower()

    rem_src = [nm for nm in ctrl_names if _side_of_ctrl(nm) == src_s and nm not in used_src]
    rem_dst = [nm for nm in ctrl_names if _side_of_ctrl(nm) == dst_s and nm not in used_dst]
    if rem_src and rem_dst:
        dst_by_stem: dict[str, list[str]] = {}
        for d in rem_dst:
            dst_by_stem.setdefault(_norm_stem(d), []).append(d)
        for s in rem_src:
            cands = [d for d in dst_by_stem.get(_norm_stem(s), []) if d not in used_dst]
            if not cands:
                continue
            hs = bones[s].head_local
            mir_hs = Vector((-hs.x, hs.y, hs.z))
            best_d = min(cands, key=lambda d: (bones[d].head_local - mir_hs).length)
            if (bones[best_d].head_local - mir_hs).length < 0.35:
                pairs.append((s, best_d))
                used_src.add(s)
                used_dst.add(best_d)

    # Sort parent-to-child so mirrored parent FK joints are solved before their child joints!
    def _depth(nm: str) -> int:
        d = 0
        cur = bones[nm].parent
        while cur is not None:
            d += 1
            cur = cur.parent
        return d

    pairs.sort(key=lambda pr: _depth(pr[1]))
    return pairs


def mirror_control_rig_pose(arm_obj: Any, source_side: str = "L") -> int:
    """Mirror all paired `CTRL_*` pose transforms cleanly across X=0 (`L -> R` or `R -> L`), invariant to Decima bone rolls!

    Runs in two dependency-safe passes:
      1. Primary/limb/wing/shoulder controls (everything not under an `MCH_Follow_*` socket), followed by a depsgraph update
         so every right/left `MCH_Follow_*` socket lands at its exact new mirrored pose.
      2. Child controls (fingers, toes, talons, wing ribs, flipper blades, frills, weapons, armor plates), preserving
         untouched Identity controls cleanly and reflecting controls sharing a posed Center parent (`CTRL_Head`/`CTRL_Chest`)
         across that parent's own symmetry plane so turned heads/chests never detach plates or frills!
    """
    if bpy is None or arm_obj is None or arm_obj.type != "ARMATURE":
        return 0
    pairs = _find_mirror_ctrl_pairs(arm_obj, source_side=source_side)
    if not pairs:
        return 0

    bpy.context.view_layer.update()
    bones = arm_obj.data.bones
    pbones = arm_obj.pose.bones
    S3 = Matrix.Diagonal((-1.0, 1.0, 1.0))

    def _under_follow_socket(bname: str) -> bool:
        cur = bones[bname].parent if bname in bones else None
        while cur is not None:
            if cur.name.startswith("MCH_Follow_"):
                return True
            cur = cur.parent
        return False

    def _is_identity_basis(pb: Any) -> bool:
        loc, rot, _scl = pb.matrix_basis.decompose()
        return loc.length < 1e-5 and abs(rot.angle) < 1e-5

    pass1 = [pr for pr in pairs if not _under_follow_socket(pr[1])]
    pass2 = [pr for pr in pairs if _under_follow_socket(pr[1])]

    mirrored = 0
    for batch in (pass1, pass2):
        if not batch:
            continue
        bpy.context.view_layer.update()
        src_mats = {s: pbones[s].matrix.copy() for s, _ in batch if s in pbones}
        src_is_ident = {s: _is_identity_basis(pbones[s]) for s, _ in batch if s in pbones}
        solved_dst: dict[str, Matrix] = {}

        for s, d in batch:
            if s not in src_mats or d not in pbones:
                continue
            bone_s = bones[s]
            bone_d = bones[d]
            pb_d = pbones[d]

            # If source control was never moved/rotated from rest AND has the same or mirrored parent, reset destination cleanly!
            if src_is_ident[s] and (
                bone_s.parent is not None
                and bone_d.parent is not None
                and (bone_s.parent.name == bone_d.parent.name or bone_d.parent.name.startswith("MCH_Follow_") or bone_d.parent.name in solved_dst)
            ):
                pb_d.rotation_mode = "QUATERNION"
                pb_d.matrix_basis = Matrix.Identity(4)
                p_mat = solved_dst.get(bone_d.parent.name, pbones[bone_d.parent.name].matrix)
                rest_local = bone_d.parent.matrix_local.inverted() @ bone_d.matrix_local
                solved_dst[d] = p_mat @ rest_local
                mirrored += 1
                continue

            r_s = bone_s.matrix_local
            r_d = bone_d.matrix_local
            m_s = src_mats[s]

            # If both L & R share the exact same Center parent (e.g. turned CTRL_Head / arched CTRL_Chest / tilted CTRL_COG),
            # reflect in that Center parent's own rest-aligned symmetry frame so turned heads/chests never throw plates off-axis!
            shared_center = (
                bone_s.parent is not None
                and bone_d.parent is not None
                and bone_s.parent.name == bone_d.parent.name
                and bone_s.parent.name in pbones
            )
            if shared_center:
                pc_name = bone_s.parent.name
                r_c = bones[pc_name].matrix_local
                p_c = pbones[pc_name].matrix
                m_s_ref = r_c @ p_c.inverted() @ m_s
            else:
                m_s_ref = m_s

            # Exact SO(3) reflection of rotation delta: det(S3 @ dR @ S3) = +1 (zero handedness flip!)
            delta_rot_s = m_s_ref.to_3x3() @ r_s.to_3x3().inverted()
            delta_rot_d = S3 @ delta_rot_s @ S3
            target_rot_d = delta_rot_d @ r_d.to_3x3()

            delta_pos_s = m_s_ref.to_translation() - r_s.to_translation()
            target_pos_d = r_d.to_translation() + (S3 @ delta_pos_s)
            target_mat_d = Matrix.Translation(target_pos_d) @ target_rot_d.to_4x4()
            if shared_center:
                target_mat_d = p_c @ r_c.inverted() @ target_mat_d

            if bone_d.parent is not None:
                p_mat = solved_dst.get(bone_d.parent.name, pbones[bone_d.parent.name].matrix)
                rest_local = bone_d.parent.matrix_local.inverted() @ bone_d.matrix_local
                basis_d = (p_mat @ rest_local).inverted() @ target_mat_d
            else:
                basis_d = bone_d.matrix_local.inverted() @ target_mat_d

            loc_d, rot_d, _scl_d = basis_d.decompose()
            if all(pb_d.lock_location):
                basis_d = rot_d.to_matrix().to_4x4()
            else:
                basis_d = Matrix.Translation(loc_d) @ rot_d.to_matrix().to_4x4()

            pb_d.rotation_mode = "QUATERNION"
            pb_d.matrix_basis = basis_d

            if bone_d.parent is not None:
                p_mat = solved_dst.get(bone_d.parent.name, pbones[bone_d.parent.name].matrix)
                rest_local = bone_d.parent.matrix_local.inverted() @ bone_d.matrix_local
                solved_dst[d] = p_mat @ rest_local @ basis_d
            else:
                solved_dst[d] = bone_d.matrix_local @ basis_d
            mirrored += 1

        arm_obj.update_tag()
        bpy.context.view_layer.update()

    return mirrored


def reset_control_rig_pose(arm_obj: Any) -> None:
    """Reset all `CTRL_*` pose bones back to their clean rest transforms."""
    if bpy is None or arm_obj is None or arm_obj.type != "ARMATURE":
        return
    for pb in arm_obj.pose.bones:
        if pb.name.startswith("CTRL_"):
            pb.matrix_basis = Matrix.Identity(4)
    arm_obj.update_tag()
    bpy.context.view_layer.update()
