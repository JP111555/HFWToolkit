"""HFWToolkit Native Embedded Rig & Shader UI (Add-on Independent!).

Automatically embedded as `hfw_rig_ui.py` (`use_module = True`) inside every exported `.blend`
so Rig Controls, IK/FK zero-pop snapping, Chainsaws & Drills, Pose Mirroring, and live
Armature-driven Shader Controls work 100% natively in Blender's `Item` sidebar tab even
WITHOUT the HFWToolkit add-on installed!
"""
import json
import math
import re
import sys
import types

import bpy
from mathutils import Euler, Matrix, Vector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _active_hfw_armature(context):
    obj = getattr(context, "active_object", None)
    if obj is not None:
        if obj.type == "ARMATURE" and (obj.get("hfw_has_control_rig") or obj.get("hfw_shader_kind")):
            return obj
        if obj.parent is not None and obj.parent.type == "ARMATURE" and (
            obj.parent.get("hfw_has_control_rig") or obj.parent.get("hfw_shader_kind")
        ):
            return obj.parent
    return None


def _all_bcolls(arm):
    return getattr(arm.data, "collections_all", arm.data.collections)


def _read_meta(arm) -> tuple[dict, dict[str, str]]:
    meta = {}
    raw = arm.get("hfw_ctrl_rig_meta")
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                meta = parsed
        except Exception:
            meta = {}
    slots: dict[str, str] = {}
    saved = meta.get("collection_slots")
    all_c = _all_bcolls(arm)
    if isinstance(saved, dict):
        for k, v in saved.items():
            if isinstance(v, str) and v in all_c:
                slots[k] = v
    return meta, slots


def _sync_layer_visibility(arm) -> None:
    colls = _all_bcolls(arm)
    meta, slots = _read_meta(arm)
    torso_ik = float(arm.get("hfw_ik_fk_torso", 1.0)) >= 0.5
    head_ik = float(arm.get("hfw_ik_fk_head", 0.0)) >= 0.5
    spline_ik = float(arm.get("hfw_ik_fk_spline", 0.0)) >= 0.5
    legs_ik = float(arm.get("hfw_ik_fk_legs", 1.0)) >= 0.5
    flegs_ik = float(arm.get("hfw_ik_fk_front_legs", 1.0)) >= 0.5
    arms_ik = float(arm.get("hfw_ik_fk_arms", 1.0)) >= 0.5
    wings_ik = float(arm.get("hfw_ik_fk_wings", 0.0)) >= 0.5
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
    if slots.get("torso_ik", "") in colls:
        fk_slot = "hood_fk" if (slots.get("hood_fk", "") in colls) else "spine_fk"
        pairs.extend([("torso_ik", torso_ik), (fk_slot, not torso_ik)])
    if slots.get("spline_ik", "") in colls:
        pairs.append(("spline_ik", spline_ik))
        for sp_info in meta.get("spline_ik") or []:
            sk = sp_info.get("kind")
            if sk in ("spine", "serpent_full") and slots.get("spine_fk", "") in colls:
                pairs.append(("spine_fk", not spline_ik))
            elif sk == "tail" and slots.get("tail", "") in colls:
                pairs.append(("tail", not spline_ik))
            elif sk == "appendage" and slots.get("appendages", "") in colls:
                pairs.append(("appendages", not spline_ik))
    for skey, vis in pairs:
        cname = slots.get(skey)
        if cname and cname in colls:
            colls[cname].is_visible = vis


def _snap_fk_to_ik(arm, group: str) -> int:
    meta, _ = _read_meta(arm)
    chains = (meta.get("limb_fk_chains") or {}).get(group, [])
    if not chains:
        return 0
    bpy.context.view_layer.update()
    bones = arm.data.bones
    pbones = arm.pose.bones
    def_mats = {
        p["def_bone"]: pbones[p["def_bone"]].matrix.copy()
        for ch in chains
        for p in ch.get("fk_pairs", [])
        if p["def_bone"] in pbones
    }
    snapped = 0
    for ch in chains:
        solved = {}
        for p in ch.get("fk_pairs", []):
            cn, dn = p["ctrl"], p["def_bone"]
            if cn not in bones or dn not in def_mats:
                continue
            bc = bones[cn]
            rest_off = bones[dn].matrix_local.inverted() @ bc.matrix_local
            tgt = def_mats[dn] @ rest_off
            if bc.parent is not None:
                pm = solved.get(bc.parent.name, pbones[bc.parent.name].matrix)
                rl = bc.parent.matrix_local.inverted() @ bc.matrix_local
                basis = (pm @ rl).inverted() @ tgt
            else:
                basis = bc.matrix_local.inverted() @ tgt
            _loc, rot, _scl = basis.decompose()
            rmat = rot.to_matrix().to_4x4()
            pbones[cn].matrix_basis = rmat
            if bc.parent is not None:
                pm = solved.get(bc.parent.name, pbones[bc.parent.name].matrix)
                rl = bc.parent.matrix_local.inverted() @ bc.matrix_local
                solved[cn] = pm @ rl @ rmat
            else:
                solved[cn] = bc.matrix_local @ rmat
            snapped += 1
    arm.update_tag()
    bpy.context.view_layer.update()
    return snapped


def _snap_ik_to_fk(arm, group: str) -> int:
    meta, _ = _read_meta(arm)
    bones = arm.data.bones
    pbones = arm.pose.bones
    bpy.context.view_layer.update()
    snapped = 0
    if group == "torso":
        tik = meta.get("torso_ik") or {}
        cc, cd = tik.get("chest_ctrl"), tik.get("chest_def")
        if cc in bones and cd in pbones:
            tgt = pbones[cd].matrix @ (bones[cd].matrix_local.inverted() @ bones[cc].matrix_local)
            bc = bones[cc]
            pm = pbones[bc.parent.name].matrix if bc.parent else Matrix.Identity(4)
            rl = (bc.parent.matrix_local.inverted() @ bc.matrix_local) if bc.parent else bc.matrix_local
            pbones[cc].matrix_basis = (pm @ rl).inverted() @ tgt
            snapped += 1
            mc, md = tik.get("mid_ctrl") or "", tik.get("mid_def") or ""
            if mc in bones and md in pbones:
                md_mat = pbones[md].matrix.copy()
                arm.update_tag()
                bpy.context.view_layer.update()
                mtgt = md_mat @ (bones[md].matrix_local.inverted() @ bones[mc].matrix_local)
                bm = bones[mc]
                pm2 = pbones[bm.parent.name].matrix if bm.parent else Matrix.Identity(4)
                rl2 = (bm.parent.matrix_local.inverted() @ bm.matrix_local) if bm.parent else bm.matrix_local
                pbones[mc].matrix_basis = (pm2 @ rl2).inverted() @ mtgt
                snapped += 1
    elif group == "head":
        hik = meta.get("head_ik") or {}
        hc, hd = hik.get("head_ctrl"), hik.get("head_def")
        if hc in bones and hd in pbones:
            tgt = pbones[hd].matrix @ (bones[hd].matrix_local.inverted() @ bones[hc].matrix_local)
            bh = bones[hc]
            pm = pbones[bh.parent.name].matrix if bh.parent else Matrix.Identity(4)
            rl = (bh.parent.matrix_local.inverted() @ bh.matrix_local) if bh.parent else bh.matrix_local
            pbones[hc].matrix_basis = (pm @ rl).inverted() @ tgt
            snapped += 1
    elif group == "spline":
        splines = meta.get("spline_ik") or []
        dmats = {
            kn["def_bone"]: pbones[kn["def_bone"]].matrix.copy()
            for sp in splines
            for kn in sp.get("knots") or []
            if kn["def_bone"] in pbones
        }
        for sp in splines:
            sl_c = sp.get("slide_ctrl")
            if sl_c and sl_c in pbones:
                pbones[sl_c].matrix_basis = Matrix.Identity(4)
            for kn in sp.get("knots") or []:
                cn, dn = kn["ctrl"], kn["def_bone"]
                if cn not in bones or dn not in dmats:
                    continue
                tgt = dmats[dn] @ (bones[dn].matrix_local.inverted() @ bones[cn].matrix_local)
                bc = bones[cn]
                pm = pbones[bc.parent.name].matrix if bc.parent else Matrix.Identity(4)
                rl = (bc.parent.matrix_local.inverted() @ bc.matrix_local) if bc.parent else bc.matrix_local
                pbones[cn].matrix_basis = (pm @ rl).inverted() @ tgt
                snapped += 1
    else:
        limbs = [lm for lm in (meta.get("ik_limbs") or []) if lm.get("group", "legs") == group]
        root_mat = pbones["CTRL_Root"].matrix.copy() if "CTRL_Root" in pbones else Matrix.Identity(4)
        for lm in limbs:
            tn, pn, ed = lm["target_ctrl"], lm["pole_ctrl"], lm["end_def"]
            ik_defs = [b for b in lm["ik_defs"] if b in pbones]
            if tn not in bones or pn not in bones or ed not in pbones or len(ik_defs) < 2:
                continue
            tgt = pbones[ed].matrix @ (bones[ed].matrix_local.inverted() @ bones[tn].matrix_local)
            bt = bones[tn]
            pm = pbones[bt.parent.name].matrix if bt.parent else Matrix.Identity(4)
            rl = (bt.parent.matrix_local.inverted() @ bt.matrix_local) if bt.parent else bt.matrix_local
            pbones[tn].matrix_basis = (pm @ rl).inverted() @ tgt

            pts = [pbones[b].matrix.to_translation() for b in ik_defs] + [pbones[ed].matrix.to_translation()]
            p0, p_end = pts[0], pts[-1]
            p_mid = pts[1] if len(pts) == 3 else (pts[1] * 0.7 + pts[2] * 0.3)
            chord = p_end - p0
            clen = max(chord.length, 0.1)
            cdir = chord.normalized()
            perp = (p_mid - p0) - cdir * (p_mid - p0).dot(cdir)
            if perp.length < clen * 0.03:
                fb = (root_mat.to_3x3() @ Vector(lm["fallback_dir"])).normalized()
                bdir = fb - cdir * fb.dot(cdir)
                bdir = bdir.normalized() if bdir.length > 1e-4 else Vector((0.0, 1.0, 0.0))
            else:
                bdir = perp.normalized()
            ppos = p_mid + bdir * (clen * float(lm.get("dist_scale", 0.85)))
            pmat = Matrix.Translation(ppos) @ bones[pn].matrix_local.to_3x3().to_4x4()
            bp = bones[pn]
            pmp = pbones[bp.parent.name].matrix if bp.parent else Matrix.Identity(4)
            rlp = (bp.parent.matrix_local.inverted() @ bp.matrix_local) if bp.parent else bp.matrix_local
            pbones[pn].matrix_basis = (pmp @ rlp).inverted() @ pmat
            snapped += 1
    arm.update_tag()
    bpy.context.view_layer.update()
    return snapped


def _mirror_pose(arm, source_side: str = "L") -> int:
    src = "L" if source_side.upper().startswith("L") else "R"
    dst = "R" if src == "L" else "L"
    bones = arm.data.bones
    pbones = arm.pose.bones
    ctrls = [b.name for b in bones if b.name.startswith("CTRL_")]
    cset = set(ctrls)
    pairs = {}
    for nm in ctrls:
        if f"_{src}_" in nm or nm.endswith(f"_{src}"):
            cand = re.sub(rf"_{src}(_|$)", f"_{dst}\\1", nm)
            if cand != nm and cand in cset:
                pairs[nm] = cand
    if not pairs:
        return 0
    s4 = Matrix.Diagonal((-1.0, 1.0, 1.0, 1.0))
    s3 = Matrix.Diagonal((-1.0, 1.0, 1.0))

    def _apply_pair(sn: str, dn: str):
        bs, bd = bones[sn], bones[dn]
        ps, pd = pbones[sn], pbones[dn]
        ds = ps.matrix @ bs.matrix_local.inverted()
        md_t = s4 @ (ds @ bs.matrix_local)
        d_pos = md_t.to_translation()
        d_rot = s3 @ ds.to_3x3() @ s3 @ bd.matrix_local.to_3x3()
        tgt = Matrix.Translation(d_pos) @ d_rot.to_4x4()
        if bd.parent is not None:
            pm = pbones[bd.parent.name].matrix
            rl = bd.parent.matrix_local.inverted() @ bd.matrix_local
            basis = (pm @ rl).inverted() @ tgt
        else:
            basis = bd.matrix_local.inverted() @ tgt
        loc, rot, scl = basis.decompose()
        if all(pd.lock_location):
            loc = Vector((0.0, 0.0, 0.0))
        pd.matrix_basis = Matrix.Translation(loc) @ rot.to_matrix().to_4x4() @ Matrix.Diagonal((scl.x, scl.y, scl.z, 1.0))

    bpy.context.view_layer.update()
    primary = [(s, d) for s, d in pairs.items() if any(t in s for t in ("_IK_", "_Pole_", "_FK_Leg", "_FK_FrontLeg", "_FK_RearLeg", "_FK_Arm", "_FK_Wing", "_Shoulder_", "_Clavicle_"))]
    secondary = [(s, d) for s, d in pairs.items() if (s, d) not in primary]
    for s, d in primary:
        _apply_pair(s, d)
    arm.update_tag()
    bpy.context.view_layer.update()
    for s, d in secondary:
        _apply_pair(s, d)
    arm.update_tag()
    bpy.context.view_layer.update()
    return len(pairs)


def _pose_serpent(arm, style: str = "wave") -> int:
    meta, _ = _read_meta(arm)
    splines = meta.get("spline_ik") or []
    if not splines:
        return 0
    arm["hfw_rig_master"] = 1.0
    arm["hfw_ik_fk_spline"] = 1.0
    _sync_layer_visibility(arm)
    arm.update_tag()
    bpy.context.view_layer.update()
    bones = arm.data.bones
    pbones = arm.pose.bones
    posed = 0
    for sp in splines:
        sl_c = sp.get("slide_ctrl")
        if sl_c and sl_c in pbones:
            pbones[sl_c].matrix_basis = Matrix.Identity(4)
        knots = [kn["ctrl"] for kn in (sp.get("knots") or []) if kn["ctrl"] in bones]
        if len(knots) < 3:
            continue
        p0 = bones[knots[0]].head_local.copy()
        p_end = bones[knots[-1]].head_local.copy()
        clen = max(1.0, (p_end - p0).length)
        adir = (p_end - p0).normalized() if (p_end - p0).length > 1e-4 else Vector((0.0, -1.0, 0.0))
        sdir = adir.cross(Vector((0.0, 0.0, 1.0)))
        sdir = sdir.normalized() if sdir.length > 1e-4 else Vector((1.0, 0.0, 0.0))
        n_k = len(knots)
        is_full = sp.get("kind") == "serpent_full"
        pts = []
        for ki in range(n_k):
            u = ki / max(1, n_k - 1)
            if style == "coil":
                if is_full and ki == 0:
                    pt = Vector((p0.x, p0.y - clen * 0.08, p0.z + clen * 0.14))
                elif is_full and ki == 1:
                    pt = Vector((p0.x + sdir.x * clen * 0.04, p0.y - clen * 0.14, p0.z + clen * 0.07))
                else:
                    u_c = max(0.0, (u - 0.18) / 0.82) if is_full else u
                    th = u_c * math.pi * 2.15
                    rad = clen * (0.12 + 0.15 * u_c)
                    pt = Vector((
                        p0.x + sdir.x * (rad * math.sin(th)) + adir.x * (rad * (1.0 - math.cos(th))),
                        p0.y + sdir.y * (rad * math.sin(th)) + adir.y * (rad * (1.0 - math.cos(th))),
                        max(0.55, p0.z * (1.0 - 0.12 * u)),
                    ))
            else:
                ang = u * math.pi * 2.5
                alat = clen * 0.105 * math.sin(max(0.08, u) * math.pi * 0.90)
                hlift = clen * 0.10 * math.exp(-((u / 0.22) ** 2)) if is_full else 0.0
                az = hlift + clen * 0.012 * math.sin(ang * 2.0) * u
                pt = p0 + adir * (clen * u * 0.91) + sdir * (alat * math.sin(ang)) + Vector((0.0, 0.0, az))
            pts.append(pt)
        for ki, kn in enumerate(knots):
            tang = (pts[1] - pts[0]) if ki == 0 else ((pts[-1] - pts[-2]) if ki == n_k - 1 else (pts[ki + 1] - pts[ki - 1]))
            tang = tang.normalized() if tang.length > 1e-4 else adir
            xv = tang.cross(Vector((0.0, 0.0, 1.0)))
            xv = xv.normalized() if xv.length > 1e-4 else Vector((1.0, 0.0, 0.0))
            zv = xv.cross(tang).normalized()
            tm = Matrix(((xv.x, tang.x, zv.x, pts[ki].x), (xv.y, tang.y, zv.y, pts[ki].y), (xv.z, tang.z, zv.z, pts[ki].z), (0, 0, 0, 1)))
            bk = bones[kn]
            pm = pbones[bk.parent.name].matrix if bk.parent else Matrix.Identity(4)
            rl = (bk.parent.matrix_local.inverted() @ bk.matrix_local) if bk.parent else bk.matrix_local
            pbones[kn].matrix_basis = (pm @ rl).inverted() @ tm
            posed += 1
    arm.update_tag()
    bpy.context.view_layer.update()
    return posed


# ---------------------------------------------------------------------------
# Native Self-Contained Operators (Zero Add-on Dependency!)
# ---------------------------------------------------------------------------
class HFW_OT_NativeSetMode(bpy.types.Operator):
    bl_idname = "hfw_rig.set_mode"
    bl_label = "Set Rig Mode"
    bl_options = {"REGISTER", "UNDO"}
    to_rig: bpy.props.BoolProperty(default=True)

    def execute(self, context):
        arm = _active_hfw_armature(context)
        if arm is None:
            return {"CANCELLED"}
        arm["hfw_rig_master"] = 1.0 if self.to_rig else 0.0
        arm.update_tag()
        bpy.context.view_layer.update()
        return {"FINISHED"}


class HFW_OT_NativeResetPose(bpy.types.Operator):
    bl_idname = "hfw_rig.reset_pose"
    bl_label = "Reset Rig Pose"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        arm = _active_hfw_armature(context)
        if arm is None:
            return {"CANCELLED"}
        for pb in arm.pose.bones:
            if pb.name.startswith("CTRL_"):
                pb.matrix_basis = Matrix.Identity(4)
        for pk in (
            "hfw_wing_fold_L", "hfw_wing_fold_R",
            "hfw_wing_fan_L", "hfw_wing_fan_R",
            "hfw_wing_camber_L", "hfw_wing_camber_R",
        ):
            if pk in arm:
                arm[pk] = 0.0
        arm.update_tag()
        bpy.context.view_layer.update()
        return {"FINISHED"}


class HFW_OT_NativeSwitchIKFK(bpy.types.Operator):
    bl_idname = "hfw_rig.switch_ik_fk"
    bl_label = "Switch IK / FK (Auto-Snap)"
    bl_options = {"REGISTER", "UNDO"}
    group: bpy.props.StringProperty(default="legs")
    to_ik: bpy.props.BoolProperty(default=True)

    def execute(self, context):
        arm = _active_hfw_armature(context)
        if arm is None:
            return {"CANCELLED"}
        pmap = {
            "torso": "hfw_ik_fk_torso", "head": "hfw_ik_fk_head", "spline": "hfw_ik_fk_spline",
            "legs": "hfw_ik_fk_legs", "front_legs": "hfw_ik_fk_front_legs",
            "arms": "hfw_ik_fk_arms", "wings": "hfw_ik_fk_wings",
        }
        if self.to_ik:
            _snap_ik_to_fk(arm, self.group)
        else:
            _snap_fk_to_ik(arm, self.group)
        arm[pmap.get(self.group, "hfw_ik_fk_legs")] = 1.0 if self.to_ik else 0.0
        _sync_layer_visibility(arm)
        arm.update_tag()
        bpy.context.view_layer.update()
        return {"FINISHED"}


class HFW_OT_NativeSelectBones(bpy.types.Operator):
    bl_idname = "hfw_rig.select_bones"
    bl_label = "Select Rig Controls"
    bl_options = {"REGISTER", "UNDO"}
    bone_names_csv: bpy.props.StringProperty(default="")
    group: bpy.props.StringProperty(default="")

    def execute(self, context):
        arm = _active_hfw_armature(context)
        if arm is None:
            return {"CANCELLED"}
        meta, slots = _read_meta(arm)
        if self.bone_names_csv:
            names = [n.strip() for n in self.bone_names_csv.split(",") if n.strip() in arm.pose.bones]
        else:
            g = self.group
            if g == "chest":
                names = [meta.get("chest_ctrl") or "CTRL_Chest"]
            elif g == "head":
                names = ["CTRL_IK_Head" if float(arm.get("hfw_ik_fk_head", 0.0)) >= 0.5 else "CTRL_Head"]
            elif g == "spline":
                names = [kn["ctrl"] for sp in (meta.get("spline_ik") or []) for kn in (sp.get("knots") or [])]
            elif g == "slide":
                names = [sp.get("slide_ctrl", "") for sp in (meta.get("spline_ik") or []) if sp.get("slide_ctrl")]
            elif g == "curls":
                names = list(meta.get("curl_ctrls") or [])
            elif g == "tail":
                names = list(meta.get("tail_ctrls") or [])
            elif g == "wings":
                names = list(meta.get("wing_ik_ctrls") if float(arm.get("hfw_ik_fk_wings", 0.0)) >= 0.5 else (meta.get("wing_ctrls") or []))
            else:
                names = list(meta.get("spine_ctrls") or [])
            names = [n for n in names if n in arm.pose.bones]
        if not names:
            return {"CANCELLED"}
        context.view_layer.objects.active = arm
        if context.mode != "POSE":
            bpy.ops.object.mode_set(mode="POSE")
        for pb in arm.pose.bones:
            sel = pb.name in names
            pb.select = sel
            pb.bone.select = sel
        arm.data.bones.active = arm.data.bones[names[0]]
        # Ensure bone's collection (and its parent folder) is visible
        for bc in _all_bcolls(arm):
            if names[0] in bc.bones and not bc.is_visible:
                bc.is_visible = True
                if getattr(bc, "parent", None) is not None:
                    bc.parent.is_visible = True
        return {"FINISHED"}


class HFW_OT_NativeSerpentPose(bpy.types.Operator):
    bl_idname = "hfw_rig.serpent_pose"
    bl_label = "Serpent Pose"
    bl_options = {"REGISTER", "UNDO"}
    style: bpy.props.StringProperty(default="wave")

    def execute(self, context):
        arm = _active_hfw_armature(context)
        if arm is None:
            return {"CANCELLED"}
        _pose_serpent(arm, style=self.style)
        return {"FINISHED"}


class HFW_OT_NativeWingPose(bpy.types.Operator):
    bl_idname = "hfw_rig.wing_pose"
    bl_label = "Apply Wing Preset"
    bl_options = {"REGISTER", "UNDO"}
    preset: bpy.props.StringProperty(default="SPREAD")

    def execute(self, context):
        arm = _active_hfw_armature(context)
        if arm is None:
            return {"CANCELLED"}
        if self.preset == "SYNC_LR":
            for stem in ("hfw_wing_fold", "hfw_wing_fan", "hfw_wing_camber"):
                kl, kr = f"{stem}_L", f"{stem}_R"
                if kl in arm and kr in arm:
                    arm[kr] = float(arm[kl])
        else:
            # (fold, fan, camber)
            presets = {
                "SPREAD": (0.0, 0.35, 0.05),
                "GLIDE": (-0.08, 0.60, 0.25),
                "FLAP": (0.08, 0.25, 0.85),
                "DIVE": (0.58, -0.45, -0.25),
                "FOLD": (0.95, -0.75, 0.10),
            }
            fv, fnv, cv = presets.get(self.preset, (0.0, 0.0, 0.0))
            for s in ("L", "R"):
                if f"hfw_wing_fold_{s}" in arm:
                    arm[f"hfw_wing_fold_{s}"] = fv
                if f"hfw_wing_fan_{s}" in arm:
                    arm[f"hfw_wing_fan_{s}"] = fnv
                if f"hfw_wing_camber_{s}" in arm:
                    arm[f"hfw_wing_camber_{s}"] = cv
                mc = f"CTRL_Wing_Master_{s}"
                if mc in arm.pose.bones:
                    arm.pose.bones[mc].matrix_basis = Matrix.Identity(4)
        arm.update_tag()
        bpy.context.view_layer.update()
        return {"FINISHED"}


class HFW_OT_NativeMirrorPose(bpy.types.Operator):
    bl_idname = "hfw_rig.mirror_pose"
    bl_label = "Mirror Pose"
    bl_options = {"REGISTER", "UNDO"}
    source_side: bpy.props.StringProperty(default="L")

    def execute(self, context):
        arm = _active_hfw_armature(context)
        if arm is None:
            return {"CANCELLED"}
        _mirror_pose(arm, source_side=self.source_side)
        return {"FINISHED"}


class HFW_OT_NativeStopSpin(bpy.types.Operator):
    bl_idname = "hfw_rig.stop_spin"
    bl_label = "Stop Spin"
    bl_options = {"REGISTER", "UNDO"}
    prop_key: bpy.props.StringProperty(default="ALL")

    def execute(self, context):
        arm = _active_hfw_armature(context)
        if arm is None:
            return {"CANCELLED"}
        if self.prop_key == "ALL":
            for k in list(arm.keys()):
                if k in ("hfw_saw_speed", "hfw_drill_speed") or (k.startswith(("hfw_saw_", "hfw_spin_")) and k.endswith("_speed")):
                    arm[k] = 0.0
        elif self.prop_key in arm:
            arm[self.prop_key] = 0.0
        arm.update_tag()
        bpy.context.view_layer.update()
        return {"FINISHED"}


class HFW_OT_NativeToggleLayer(bpy.types.Operator):
    bl_idname = "hfw_rig.toggle_layer"
    bl_label = "Toggle Layer"
    coll_name: bpy.props.StringProperty(default="")

    def execute(self, context):
        arm = _active_hfw_armature(context)
        if arm is None:
            return {"CANCELLED"}
        all_c = _all_bcolls(arm)
        if self.coll_name == "RAW_DEF":
            raws = [all_c.get(n) for n in ("Skeleton", "Body", "Helpers") if all_c.get(n)]
            if raws:
                nv = not any(c.is_visible for c in raws)
                for c in raws:
                    c.is_visible = nv
            return {"FINISHED"}
        bc = all_c.get(self.coll_name)
        if bc is not None:
            bc.is_visible = not bc.is_visible
            if bc.is_visible and getattr(bc, "parent", None) is not None:
                bc.parent.is_visible = True
        return {"FINISHED"}


class HFW_OT_NativeShaderPreset(bpy.types.Operator):
    bl_idname = "hfw_rig.shader_preset"
    bl_label = "Apply Machine Eye or Muscle Preset"
    bl_options = {"REGISTER", "UNDO"}
    preset: bpy.props.StringProperty(default="EYE_PATROL")

    def execute(self, context):
        arm = _active_hfw_armature(context)
        if arm is None:
            return {"CANCELLED"}
        # Lore-accurate decoupling:
        #   - EYE_* presets change ONLY eye optics (Overridden turns eyes Cyan while keeping muscles unchanged!)
        #   - MUSCLE_* presets change ONLY muscle fibers (Standard Amber / Apex Purple / Corrupted Crimson!)
        eye_presets = {
            "EYE_PATROL": ((0.04, 0.56, 1.0), 0.0),      # Calm Patrol Blue
            "EYE_SUSPICIOUS": ((1.0, 0.72, 0.03), 0.0),  # Suspicious Yellow
            "EYE_ALERT": ((1.0, 0.06, 0.02), 1.0),       # Hostile Combat Red
            "EYE_OVERRIDDEN": ((0.03, 0.82, 1.0), 0.0),  # Overridden Electric Cyan (Eyes ONLY!)
            # Legacy preset keys mapped safely without falsely coloring muscles blue on Override
            "PATROL": ((0.04, 0.56, 1.0), 0.0),
            "ALERT": ((1.0, 0.06, 0.02), 1.0),
            "OVERRIDDEN": ((0.03, 0.82, 1.0), 0.0),
        }
        muscle_presets = {
            "MUSCLE_STANDARD": (0.95, 0.28, 0.02),       # Standard Warm Amber
            "MUSCLE_APEX": (0.54, 0.05, 1.0),            # Apex Violet-Purple
            "MUSCLE_CORRUPTED": (1.0, 0.05, 0.01),       # Corrupted Crimson
            "APEX": (0.54, 0.05, 1.0),
        }
        if self.preset in eye_presets:
            ec, alert_v = eye_presets[self.preset]
            if "hfw_sh_eye_color" in arm and self.preset not in ("EYE_ALERT", "ALERT"):
                arm["hfw_sh_eye_color"] = ec
            if "hfw_sh_eye_alert" in arm:
                arm["hfw_sh_eye_alert"] = alert_v
        if self.preset in muscle_presets:
            mc = muscle_presets[self.preset]
            if "hfw_sh_muscle_color" in arm:
                arm["hfw_sh_muscle_color"] = mc
            if self.preset == "APEX" and "hfw_sh_eye_color" in arm:
                arm["hfw_sh_eye_color"] = (0.64, 0.05, 1.0)
                if "hfw_sh_eye_alert" in arm:
                    arm["hfw_sh_eye_alert"] = 0.0
        arm.update_tag()
        bpy.context.view_layer.update()
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Clean Collapsible Native Panels in Blender's `Item` Sidebar Tab!
# ---------------------------------------------------------------------------
class HFW_PT_RigUI(bpy.types.Panel):
    bl_label = "HFW Rig Controls"
    bl_idname = "HFW_PT_rig_ui_native"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Item"

    @classmethod
    def poll(cls, context):
        arm = _active_hfw_armature(context)
        return bool(arm and arm.get("hfw_has_control_rig") and "hfw_rig_master" in arm)

    def draw(self, context):
        layout = self.layout
        arm = _active_hfw_armature(context)
        is_rig = float(arm["hfw_rig_master"]) >= 0.5
        row = layout.row(align=True)
        row.scale_y = 1.15
        o1 = row.operator("hfw_rig.set_mode", text="Game Anim", icon="PLAY", depress=not is_rig)
        o1.to_rig = False
        o2 = row.operator("hfw_rig.set_mode", text="Control Rig", icon="POSE_HLT", depress=is_rig)
        o2.to_rig = True
        row.operator("hfw_rig.reset_pose", text="", icon="LOOP_BACK")
        layout.prop(arm, '["hfw_rig_master"]', text="Rig Blend", slider=True)


class HFW_PT_RigUI_IKFK(bpy.types.Panel):
    bl_label = "IK / FK Switches"
    bl_idname = "HFW_PT_rig_ui_ikfk"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Item"
    bl_parent_id = "HFW_PT_rig_ui_native"

    def draw(self, context):
        layout = self.layout
        arm = _active_hfw_armature(context)
        meta, slots = _read_meta(arm)

        def _row_ik(label: str, grp: str, prop: str, ik_lbl: str = "IK", fk_lbl: str = "FK"):
            is_ik = float(arm[prop]) >= 0.5
            r = layout.row(align=True)
            sp = r.split(factor=0.38, align=True)
            sp.label(text=label)
            sub = sp.row(align=True)
            a = sub.operator("hfw_rig.switch_ik_fk", text=ik_lbl, depress=is_ik)
            a.group = grp; a.to_ik = True
            b = sub.operator("hfw_rig.switch_ik_fk", text=fk_lbl, depress=not is_ik)
            b.group = grp; b.to_ik = False

        if "torso_ik" in slots and "hfw_ik_fk_torso" in arm:
            _row_ik("Torso", "torso", "hfw_ik_fk_torso", "Chest IK", "Spine FK")
        if "head_ik" in slots and "hfw_ik_fk_head" in arm:
            _row_ik("Head", "head", "hfw_ik_fk_head", "Head IK", "Neck FK")
            if float(arm["hfw_ik_fk_head"]) >= 0.5 and "hfw_head_ik_space" in arm:
                layout.prop(arm, '["hfw_head_ik_space"]', text="Head World Lock", slider=True)
        if "spline_ik" in slots and "hfw_ik_fk_spline" in arm:
            sp_lbl = re.sub(r"\s+Spline\s+IK$", "", slots.get("spline_ik", "Spline")).strip() or "Spline"
            _row_ik(sp_lbl, "spline", "hfw_ik_fk_spline", "Spline IK", "FK")
            if float(arm["hfw_ik_fk_spline"]) >= 0.5:
                sr = layout.row(align=True)
                os_sl = sr.operator("hfw_rig.select_bones", text="Slide", icon="ARROW_LEFTRIGHT")
                os_sl.group = "slide"
                ow = sr.operator("hfw_rig.serpent_pose", text="S-Wave", icon="MOD_WAVE")
                ow.style = "wave"
                oc = sr.operator("hfw_rig.serpent_pose", text="Coil", icon="FORCE_VORTEX")
                oc.style = "coil"
        if "front_legs_ik" in slots and "hfw_ik_fk_front_legs" in arm:
            _row_ik("Front Legs", "front_legs", "hfw_ik_fk_front_legs")
        if "legs_ik" in slots and "hfw_ik_fk_legs" in arm:
            lbl = re.sub(r"\s+IK$", "", slots["legs_ik"]).strip() or "Legs"
            _row_ik(lbl, "legs", "hfw_ik_fk_legs")
        if "arms_ik" in slots and "hfw_ik_fk_arms" in arm:
            lbl = re.sub(r"\s+IK$", "", slots["arms_ik"]).strip() or "Arms"
            _row_ik(lbl, "arms", "hfw_ik_fk_arms")
        if "wings_ik" in slots and "hfw_ik_fk_wings" in arm:
            _row_ik("Wings", "wings", "hfw_ik_fk_wings")


class HFW_PT_RigUI_Wings(bpy.types.Panel):
    bl_label = "Flyer Wings (Fold, Fan & Camber)"
    bl_idname = "HFW_PT_rig_ui_wings"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Item"
    bl_parent_id = "HFW_PT_rig_ui_native"

    @classmethod
    def poll(cls, context):
        arm = _active_hfw_armature(context)
        if not arm:
            return False
        meta, _ = _read_meta(arm)
        return bool(meta.get("wing_rig"))

    def draw(self, context):
        layout = self.layout
        arm = _active_hfw_armature(context)
        meta, slots = _read_meta(arm)
        wrig = meta.get("wing_rig") or []

        # 1-Click Avian Flight & Fold Poses
        pr = layout.row(align=True)
        for pid, plbl in (("SPREAD", "Spread"), ("GLIDE", "Glide"), ("FLAP", "Downstroke"), ("DIVE", "Dive"), ("FOLD", "Fold")):
            op = pr.operator("hfw_rig.wing_pose", text=plbl)
            op.preset = pid

        # Select Wing Masters + Sync L=R + Toggle Feather Tweaks
        tr = layout.row(align=True)
        masters = [w["master_ctrl"] for w in wrig if w.get("master_ctrl") in arm.pose.bones]
        if masters:
            osel = tr.operator("hfw_rig.select_bones", text="Wing Masters", icon="RESTRICT_SELECT_OFF")
            osel.bone_names_csv = ",".join(masters)
        osync = tr.operator("hfw_rig.wing_pose", text="Sync L=R", icon="MOD_MIRROR")
        osync.preset = "SYNC_LR"
        fcoll_nm = slots.get("wings_feathers")
        all_c = _all_bcolls(arm)
        if fcoll_nm and fcoll_nm in all_c:
            of = tr.operator("hfw_rig.toggle_layer", text="Feathers", depress=all_c[fcoll_nm].is_visible)
            of.coll_name = fcoll_nm

        for w in wrig:
            s = w.get("side", "L")
            box = layout.box()
            h = box.row(align=True)
            h.label(text=f"{w.get('name', 'Wing')} ({w.get('feather_bones', 0)} feathers)", icon="OUTLINER_OB_CURVE")
            mc = w.get("master_ctrl", "")
            if mc in arm.pose.bones:
                om = h.operator("hfw_rig.select_bones", text="Ctrl", icon="BONE_DATA")
                om.bone_names_csv = mc
            if w.get("fold_prop") in arm:
                box.prop(arm, f'["{w["fold_prop"]}"]', text="Wing Fold", slider=True)
            if w.get("fan_prop") in arm:
                box.prop(arm, f'["{w["fan_prop"]}"]', text="Feather Fan", slider=True)
            if w.get("camber_prop") in arm:
                box.prop(arm, f'["{w["camber_prop"]}"]', text="Airfoil Camber", slider=True)


class HFW_PT_RigUI_Spinners(bpy.types.Panel):
    bl_label = "Chainsaws & Drills"
    bl_idname = "HFW_PT_rig_ui_spinners"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Item"
    bl_parent_id = "HFW_PT_rig_ui_native"

    @classmethod
    def poll(cls, context):
        arm = _active_hfw_armature(context)
        if not arm:
            return False
        meta, _ = _read_meta(arm)
        return bool(meta.get("has_chainsaws") or meta.get("has_spinners"))

    def draw(self, context):
        layout = self.layout
        arm = _active_hfw_armature(context)
        meta, _ = _read_meta(arm)

        top = layout.row(align=True)
        top.label(text="Rotate dials to keyframe, or set Speed", icon="MOD_SCREW")
        o_stop = top.operator("hfw_rig.stop_spin", text="Stop All", icon="SNAP_FACE")
        o_stop.prop_key = "ALL"

        saws = meta.get("chainsaw_groups") or []
        if saws:
            sbox = layout.box()
            sh = sbox.row(align=True)
            sh.label(text="Chainsaws", icon="MOD_SOLIDIFY")
            if "CTRL_Saw_Spin" in arm.pose.bones:
                om = sh.operator("hfw_rig.select_bones", text="All Dial", icon="RESTRICT_SELECT_OFF")
                om.bone_names_csv = "CTRL_Saw_Spin"
            if "hfw_saw_speed" in arm:
                sbox.prop(arm, '["hfw_saw_speed"]', text="All Saws Speed", slider=True)
            for cg in saws:
                r = sbox.row(align=True)
                op_d = r.operator("hfw_rig.select_bones", text=cg.get("label", cg["name"]), icon="DRIVER_ROTATIONAL_DIFFERENCE")
                op_d.bone_names_csv = cg["dial_ctrl"]
                sp = cg["speed_prop"]
                if sp in arm:
                    r.prop(arm, f'["{sp}"]', text="Spd", slider=True)
                    oz = r.operator("hfw_rig.stop_spin", text="0")
                    oz.prop_key = sp

        grps = meta.get("spinner_groups") or []
        if grps:
            dbox = layout.box()
            dh = dbox.row(align=True)
            dh.label(text="Drills & Grinders", icon="MOD_SCREW")
            m_ctrl = grps[0].get("master_ctrl", "")
            if len(grps) > 1 and m_ctrl and m_ctrl in arm.pose.bones:
                om = dh.operator("hfw_rig.select_bones", text="All Dial", icon="RESTRICT_SELECT_OFF")
                om.bone_names_csv = m_ctrl
            if len(grps) > 1 and "hfw_drill_speed" in arm:
                dbox.prop(arm, '["hfw_drill_speed"]', text="All Drills Speed", slider=True)
            for sg in grps:
                r = dbox.row(align=True)
                lbl = f"{sg['label']} ({sg['count']})"
                op_d = r.operator("hfw_rig.select_bones", text=lbl, icon="DRIVER_ROTATIONAL_DIFFERENCE")
                op_d.bone_names_csv = sg["dial_ctrl"]
                sp = sg["speed_prop"]
                if sp in arm:
                    r.prop(arm, f'["{sp}"]', text="Spd", slider=True)
                    oz = r.operator("hfw_rig.stop_spin", text="0")
                    oz.prop_key = sp


class HFW_PT_RigUI_Layers(bpy.types.Panel):
    bl_label = "Control Layers"
    bl_idname = "HFW_PT_rig_ui_layers"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Item"
    bl_parent_id = "HFW_PT_rig_ui_native"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        arm = _active_hfw_armature(context)
        _, slots = _read_meta(arm)
        all_c = _all_bcolls(arm)
        top = layout.row(align=True)
        top.prop(arm, "show_in_front", text="In Front", toggle=True)
        raw_vis = any((all_c.get(n) and all_c[n].is_visible) for n in ("Skeleton", "Body", "Helpers"))
        oraw = top.operator("hfw_rig.toggle_layer", text="Raw DEF", depress=raw_vis)
        oraw.coll_name = "RAW_DEF"

        grid = layout.grid_flow(columns=2, align=True)
        seen = set()
        for skey in (
            "body", "head_ik", "torso_ik", "spline_ik", "spine_fk", "tail", "appendages",
            "wings_ik", "wings_fk", "wings_feathers", "front_legs_ik", "front_legs_fk", "legs_ik", "legs_fk",
            "arms_ik", "arms_fk", "hands_feet", "hf_tweaks", "spinners", "frills", "weapons", "plates",
        ):
            cn = slots.get(skey)
            if not cn or cn in seen or cn not in all_c:
                continue
            seen.add(cn)
            bc = all_c[cn]
            op = grid.operator("hfw_rig.toggle_layer", text=cn, depress=bc.is_visible)
            op.coll_name = cn


class HFW_PT_RigUI_Tools(bpy.types.Panel):
    bl_label = "Pose & Symmetry Tools"
    bl_idname = "HFW_PT_rig_ui_tools"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Item"
    bl_parent_id = "HFW_PT_rig_ui_native"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        arm = _active_hfw_armature(context)
        meta, _ = _read_meta(arm)
        mr = layout.row(align=True)
        o1 = mr.operator("hfw_rig.mirror_pose", text="Mirror L -> R", icon="MOD_MIRROR")
        o1.source_side = "L"
        o2 = mr.operator("hfw_rig.mirror_pose", text="Mirror R -> L", icon="MOD_MIRROR")
        o2.source_side = "R"

        qr = layout.grid_flow(columns=3, align=True)
        for lbl, grp, cond in (
            ("Chest", "chest", bool(meta.get("chest_ctrl")) or "CTRL_Chest" in arm.pose.bones),
            ("Head", "head", "CTRL_Head" in arm.pose.bones or "CTRL_IK_Head" in arm.pose.bones),
            ("Spline", "spline", bool(meta.get("spline_ik"))),
            ("Curls", "curls", bool(meta.get("curl_ctrls"))),
            ("Tail", "tail", bool(meta.get("tail_ctrls"))),
            ("Wings", "wings", bool(meta.get("wing_ctrls"))),
        ):
            if cond:
                op = qr.operator("hfw_rig.select_bones", text=lbl)
                op.group = grp
        if "hfw_rig_lock_secondary" in arm:
            layout.prop(arm, '["hfw_rig_lock_secondary"]', text="Lock Uncontrolled Bones", slider=True)
        if arm.animation_data is not None:
            layout.prop(arm.animation_data, "action", text="Action")


class HFW_PT_ShaderUI(bpy.types.Panel):
    bl_label = "HFW Shader Controls"
    bl_idname = "HFW_PT_shader_ui_native"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Item"

    @classmethod
    def poll(cls, context):
        arm = _active_hfw_armature(context)
        return bool(arm and arm.get("hfw_shader_kind") in ("machine", "character"))

    def draw(self, context):
        layout = self.layout
        arm = _active_hfw_armature(context)
        kind = arm.get("hfw_shader_kind")
        if kind == "machine":
            # Decoupled Eye State presets (NEVER change muscles when overriding or alerting!)
            ebox = layout.box()
            ebox.label(text="Eye State (Optics Only)", icon="HIDE_OFF")
            pr = ebox.row(align=True)
            for lbl, key in (
                ("Patrol", "EYE_PATROL"),
                ("Suspicious", "EYE_SUSPICIOUS"),
                ("Alert", "EYE_ALERT"),
                ("Overridden", "EYE_OVERRIDDEN"),
            ):
                op = pr.operator("hfw_rig.shader_preset", text=lbl)
                op.preset = key
            if "hfw_sh_eye_color" in arm:
                r = ebox.row(align=True)
                r.prop(arm, '["hfw_sh_eye_color"]', text="Eyes")
                if "hfw_sh_eye_alert_color" in arm:
                    r.prop(arm, '["hfw_sh_eye_alert_color"]', text="Alert")
                if "hfw_sh_eye_alert" in arm:
                    ebox.prop(arm, '["hfw_sh_eye_alert"]', text="Alert Blend", slider=True)
                if "hfw_sh_eye_glow" in arm:
                    ebox.prop(arm, '["hfw_sh_eye_glow"]', text="Eye Glow", slider=True)

            # Separate Muscle Fiber Variant State (only changes when muscles actually change in-game!)
            mbox = layout.box()
            mbox.label(text="Muscle Fiber State", icon="MOD_SIMPLIFY")
            mr = mbox.row(align=True)
            for lbl, key in (("Standard", "MUSCLE_STANDARD"), ("Apex", "MUSCLE_APEX"), ("Corrupted", "MUSCLE_CORRUPTED")):
                op = mr.operator("hfw_rig.shader_preset", text=lbl)
                op.preset = key
            if "hfw_sh_plasma_charge" in arm:
                pcol = layout.column(align=True)
                pcol.prop(arm, '["hfw_sh_plasma_charge"]', text="Plasma Charge", slider=True)
                if "hfw_sh_plasma_color" in arm:
                    pcol.prop(arm, '["hfw_sh_plasma_color"]', text="Plasma Color")
        elif kind == "character":
            col = layout.column(align=True)
            if "hfw_sh_skin_tone" in arm:
                col.prop(arm, '["hfw_sh_skin_tone"]', text="Skin Tone")
            if "hfw_sh_skin_brightness" in arm:
                col.prop(arm, '["hfw_sh_skin_brightness"]', text="Brightness", slider=True)
            if "hfw_sh_sss_weight" in arm:
                col.prop(arm, '["hfw_sh_sss_weight"]', text="Subsurface", slider=True)
            if "hfw_sh_peach_fuzz" in arm:
                col.prop(arm, '["hfw_sh_peach_fuzz"]', text="Peach Fuzz", slider=True)
            if "hfw_sh_iris_color" in arm:
                col.prop(arm, '["hfw_sh_iris_color"]', text="Iris Color")
            if "hfw_sh_hair_color" in arm:
                col.prop(arm, '["hfw_sh_hair_color"]', text="Hair Color")


class HFW_PT_ShaderUI_Muscles(bpy.types.Panel):
    bl_label = "Muscles, Micro-Detail & Liquids"
    bl_idname = "HFW_PT_shader_ui_muscles"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Item"
    bl_parent_id = "HFW_PT_shader_ui_native"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        arm = _active_hfw_armature(context)
        return bool(arm and arm.get("hfw_shader_kind") == "machine")

    def draw(self, context):
        layout = self.layout
        arm = _active_hfw_armature(context)
        if "hfw_sh_muscle_color" in arm:
            layout.prop(arm, '["hfw_sh_muscle_color"]', text="Muscle Color")
        if "hfw_sh_muscle_glow" in arm:
            layout.prop(arm, '["hfw_sh_muscle_glow"]', text="Muscle Glow", slider=True)
        if "hfw_sh_detail_scale" in arm:
            layout.prop(arm, '["hfw_sh_detail_scale"]', text="Detail Tiling", slider=True)
        if "hfw_sh_detail_strength" in arm:
            layout.prop(arm, '["hfw_sh_detail_strength"]', text="Detail Strength", slider=True)
        if "hfw_sh_liquid_mix" in arm:
            layout.prop(arm, '["hfw_sh_liquid_mix"]', text="Override Liquid", slider=True)
            if float(arm["hfw_sh_liquid_mix"]) > 0.01 and "hfw_sh_liquid_color" in arm:
                layout.prop(arm, '["hfw_sh_liquid_color"]', text="Liquid Color")
        if "hfw_sh_liquid_glow" in arm:
            layout.prop(arm, '["hfw_sh_liquid_glow"]', text="Liquid Glow", slider=True)
        if "hfw_sh_anim_speed" in arm:
            layout.prop(arm, '["hfw_sh_anim_speed"]', text="Anim Speed", slider=True)


_NATIVE_CLASSES = (
    HFW_OT_NativeSetMode,
    HFW_OT_NativeResetPose,
    HFW_OT_NativeSwitchIKFK,
    HFW_OT_NativeSelectBones,
    HFW_OT_NativeSerpentPose,
    HFW_OT_NativeWingPose,
    HFW_OT_NativeMirrorPose,
    HFW_OT_NativeStopSpin,
    HFW_OT_NativeToggleLayer,
    HFW_OT_NativeShaderPreset,
    HFW_PT_RigUI,
    HFW_PT_RigUI_IKFK,
    HFW_PT_RigUI_Wings,
    HFW_PT_RigUI_Spinners,
    HFW_PT_RigUI_Layers,
    HFW_PT_RigUI_Tools,
    HFW_PT_ShaderUI,
    HFW_PT_ShaderUI_Muscles,
)


def register():
    # Blender 5.3 / Python 3.14 PEP 649 calls `typing.get_type_hints(cls)` on Operator annotations,
    # which looks up `sys.modules[cls.__module__]`. Ensure this module is always registered in `sys.modules`!
    mod_name = __name__
    mod = sys.modules.get(mod_name)
    if mod is None or not hasattr(mod, "bpy"):
        mod = types.ModuleType(mod_name)
        mod.__dict__.update(globals())
        sys.modules[mod_name] = mod
    for cls in _NATIVE_CLASSES:
        prev = getattr(bpy.types, cls.__name__, None)
        if prev is not None:
            try:
                bpy.utils.unregister_class(prev)
            except Exception:
                pass
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_NATIVE_CLASSES):
        prev = getattr(bpy.types, cls.__name__, None)
        if prev is not None:
            try:
                bpy.utils.unregister_class(prev)
            except Exception:
                pass


if __name__ == "__main__":
    register()
