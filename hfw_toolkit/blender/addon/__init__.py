"""HFW Toolkit — Blender 4.2+ / 5.3 Add-on (N-Panel Machine, Character & Dev-Asset Importer + Live Shader Controls).

Installable via Blender Preferences -> Add-ons -> Install from Disk (or `hfw-toolkit install-blender-addon`).
Uses only Blender's built-in Python (`bpy`, `sqlite3`, `subprocess`, `webbrowser`).
"""

import importlib.util
import json
import os
import re
import shutil
import sqlite3
import subprocess
import webbrowser
from pathlib import Path

import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    StringProperty,
)
from bpy.types import AddonPreferences, Operator, Panel, PropertyGroup, UIList

bl_info = {
    "name": "HFW Toolkit — Machines, Characters, Animations, Control Rig & Live Shaders",
    "author": "HFWToolkit",
    "version": (1, 1, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > HFW Toolkit (Import) & Item (Native Rig/Shader Controls)",
    "description": (
        "One-click import rigged Horizon Forbidden West machines (all 8 archetypes), characters, ALL/per-category EA08 animations, "
        "zero-stretch Spline IK, linked chainsaws, accurate per-bank rotary drills, and native add-on-free Rig & Shader Controls in Blender's Item tab."
    ),
    "category": "Import-Export",
}


def _resolve_default_toolkit_dir() -> str:
    env_dir = os.environ.get("HFW_TOOLKIT_DIR", "").strip()
    if env_dir and Path(env_dir).exists():
        return env_dir
    try:
        candidate = Path(__file__).resolve().parents[3]
        if (candidate / "hfw_toolkit").exists():
            return str(candidate)
    except Exception:
        pass
    return ""


DEFAULT_TOOLKIT_DIR = _resolve_default_toolkit_dir()

# Authentic Horizon Forbidden West Muscle & Eye Color Presets (Linear RGB)
MUSCLE_PRESETS: dict[str, tuple[float, float, float]] = {
    "STANDARD": (0.85, 0.24, 0.04),   # Warm HFW Standard Amber-Orange
    "APEX": (0.58, 0.07, 0.96),       # Authentic HFW Apex Purple-Violet
    "OVERRIDDEN": (0.04, 0.56, 1.00), # Aloy Overridden Electric Cyan
    "CORRUPTED": (0.95, 0.05, 0.04),  # Hades / Corrupted Crimson Red
    "PLASMA": (0.52, 0.06, 1.00),     # Slaughterspine Plasma Violet
    "ACID": (0.36, 0.85, 0.06),       # Acid / Bilegut Toxic Lime
}

EYE_PRESETS: dict[str, tuple[float, float, float]] = {
    "PATROL_BLUE": (0.06, 0.42, 1.00),   # Unaware Patrol Blue
    "SUSPICIOUS": (1.00, 0.72, 0.04),    # Suspicious Alert Yellow
    "HOSTILE_RED": (1.00, 0.05, 0.03),   # Hostile Combat Red
    "OVERRIDDEN": (0.03, 0.72, 1.00),    # Overridden Blue-Cyan
    "APEX_PURPLE": (0.70, 0.10, 1.00),   # Apex Machine Violet
    "CORRUPTED": (0.95, 0.02, 0.06),     # Corrupted Crimson
}


class HFWToolkitPreferences(AddonPreferences):
    bl_idname = __name__

    toolkit_dir: StringProperty(
        name="HFWToolkit Directory",
        description="Root folder of HFWToolkit (containing .venv and cache/hfw_index.sqlite)",
        subtype="DIR_PATH",
        default=DEFAULT_TOOLKIT_DIR,
    )
    web_url: StringProperty(
        name="Studio Web URL",
        description="Local URL of `hfw-toolkit web`",
        default="http://127.0.0.1:8420",
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "toolkit_dir")
        layout.prop(self, "web_url")


def _get_prefs(context) -> tuple[Path, str]:
    addon = context.preferences.addons.get(__name__)
    if addon is not None:
        return Path(addon.preferences.toolkit_dir), addon.preferences.web_url.rstrip("/")
    return Path(DEFAULT_TOOLKIT_DIR), "http://127.0.0.1:8420"


def _find_toolkit_python(toolkit_dir: Path) -> Path | None:
    venv_python = toolkit_dir / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return venv_python
    venv_unix = toolkit_dir / ".venv" / "bin" / "python"
    if venv_unix.exists():
        return venv_unix
    sys_py = shutil.which("python")
    return Path(sys_py) if sys_py else None


def _load_control_rig_module(toolkit_dir: Path):
    cr_path = toolkit_dir / "hfw_toolkit" / "blender" / "control_rig.py"
    spec = importlib.util.spec_from_file_location("hfw_control_rig_addon", str(cr_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _find_active_hfw_armature(context):
    obj = context.active_object
    if obj is not None and obj.type == "ARMATURE":
        return obj
    if obj is not None and obj.parent is not None and obj.parent.type == "ARMATURE":
        return obj.parent
    for o in context.scene.objects:
        if o.type == "ARMATURE" and (o.get("hfw_entity_id") or o.get("hfw_has_control_rig")):
            return o
    return None


def _import_bundle_direct(
    toolkit_dir: Path,
    bundle_dir: Path,
    enable_detail: bool,
    pack_tex: bool,
    build_control_rig: bool = True,
):
    builder_path = toolkit_dir / "hfw_toolkit" / "blender" / "blender_builder.py"
    spec = importlib.util.spec_from_file_location("hfw_blender_builder", str(builder_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.import_bundle_into_scene(
        bundle_dir,
        enable_detail_nodes=enable_detail,
        pack_textures=pack_tex,
        build_control_rig_layer=build_control_rig,
        import_animations=True,
    )


# ----------------------------------------------------------------------
# SQLite-backed dynamic Enum caches (sub-millisecond, zero UI lag)
# ----------------------------------------------------------------------
_MACHINE_ITEMS_CACHE: list[tuple[str, str, str]] = []
_VARIANT_ITEMS_CACHE: list[tuple[str, str, str]] = []
_CHAR_ITEMS_CACHE: list[tuple[str, str, str]] = []
_OUTFIT_ITEMS_CACHE: list[tuple[str, str, str]] = []
_HEAD_ITEMS_CACHE: list[tuple[str, str, str]] = []
_ASSET_ITEMS_CACHE: list[tuple[str, str, str]] = []


def _query_machines(self, context):
    global _MACHINE_ITEMS_CACHE
    toolkit_dir, _ = _get_prefs(context)
    db_path = toolkit_dir / "cache" / "hfw_index.sqlite"
    if not db_path.exists():
        _MACHINE_ITEMS_CACHE = [("NONE", "Run `hfw-toolkit index` first", "Index DB not found")]
        return _MACHINE_ITEMS_CACHE

    props = context.scene.hfw_toolkit
    q = (props.search_query or "").strip()
    try:
        conn = sqlite3.connect(str(db_path))
        if q:
            like = f"%{q}%"
            rows = conn.execute(
                """
                SELECT group_id, object_index, display_name, codename, part_count
                FROM entities
                WHERE category = 'machine'
                  AND (display_name LIKE ? OR codename LIKE ? OR CAST(group_id AS TEXT) = ?)
                ORDER BY display_name, part_count DESC
                LIMIT 60
                """,
                (like, like, q),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT group_id, object_index, display_name, codename, part_count
                FROM entities
                WHERE category = 'machine'
                ORDER BY display_name, part_count DESC
                LIMIT 60
                """
            ).fetchall()
        conn.close()
    except Exception as exc:
        _MACHINE_ITEMS_CACHE = [("NONE", f"DB error: {exc}", "")]
        return _MACHINE_ITEMS_CACHE

    if not rows:
        _MACHINE_ITEMS_CACHE = [("NONE", "No matching machines", "Try another search query")]
        return _MACHINE_ITEMS_CACHE

    _MACHINE_ITEMS_CACHE = [
        (f"{gid}:{oidx}", f"{disp} ({code}) [{gid}:{oidx}]", f"Group {gid}, {pcnt} parts")
        for gid, oidx, disp, code, pcnt in rows
    ]
    return _MACHINE_ITEMS_CACHE


def _query_variants(self, context):
    global _VARIANT_ITEMS_CACHE
    props = context.scene.hfw_toolkit
    oid = props.selected_entity
    if not oid or oid == "NONE" or ":" not in oid:
        _VARIANT_ITEMS_CACHE = [("0", "Standard (Default)", "Default variant 0")]
        return _VARIANT_ITEMS_CACHE

    toolkit_dir, _ = _get_prefs(context)
    db_path = toolkit_dir / "cache" / "hfw_index.sqlite"
    if not db_path.exists():
        _VARIANT_ITEMS_CACHE = [("0", "Standard (Default)", "Index DB not found")]
        return _VARIANT_ITEMS_CACHE

    try:
        gid_s, oidx_s = oid.split(":", 1)
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            """
            SELECT variant_index, label, destructibility_id, smr_id, part_count
            FROM entity_variants
            WHERE entity_group_id = ? AND entity_object_index = ?
            ORDER BY variant_index
            """,
            (int(gid_s), int(oidx_s)),
        ).fetchall()
        conn.close()
    except Exception:
        rows = []

    if not rows:
        _VARIANT_ITEMS_CACHE = [
            ("0", "0: Standard (Default)", "Variant 0"),
            ("1", "1: Alternate / Apex", "Variant 1"),
            ("2", "2: Variant 2", "Variant 2"),
            ("3", "3: Variant 3", "Variant 3"),
        ]
        return _VARIANT_ITEMS_CACHE

    _VARIANT_ITEMS_CACHE = [
        (str(vidx), f"{vidx}: {label}", f"Destructibility {dr_id or 'base'} | Body {smr_id or 'default'} ({pcnt} parts)")
        for vidx, label, dr_id, smr_id, pcnt in rows
    ]
    return _VARIANT_ITEMS_CACHE


def _query_characters(self, context):
    global _CHAR_ITEMS_CACHE
    toolkit_dir, _ = _get_prefs(context)
    db_path = toolkit_dir / "cache" / "hfw_index.sqlite"
    if not db_path.exists():
        _CHAR_ITEMS_CACHE = [("NONE", "Run `hfw-toolkit index` first", "Index DB not found")]
        return _CHAR_ITEMS_CACHE

    props = context.scene.hfw_toolkit
    q = (props.char_search or "").strip()
    kind_f = props.char_kind_filter
    try:
        conn = sqlite3.connect(str(db_path))
        sql = "SELECT char_id, display_name, character_kind, tribe, gender, outfit_count FROM characters WHERE 1=1"
        params = []
        if kind_f == "MAIN":
            sql += " AND character_kind IN ('hero', 'companion')"
        elif kind_f == "NPC":
            sql += " AND character_kind = 'npc'"
        if q:
            like = f"%{q}%"
            sql += " AND (display_name LIKE ? OR char_id LIKE ? OR tribe LIKE ? OR dev_folder LIKE ? OR tags LIKE ?)"
            params.extend([like, like, like, like, like])
        sql += (
            " ORDER BY CASE character_kind WHEN 'hero' THEN 0 WHEN 'companion' THEN 1 ELSE 2 END, "
            "CASE WHEN facial_id != '' THEN 0 ELSE 1 END, outfit_count DESC, display_name LIMIT 100"
        )
        rows = conn.execute(sql, params).fetchall()
        conn.close()
    except Exception as exc:
        _CHAR_ITEMS_CACHE = [("NONE", f"Characters not indexed ({exc})", "Run `hfw-toolkit characters list` to index")]
        return _CHAR_ITEMS_CACHE

    if not rows:
        _CHAR_ITEMS_CACHE = [("NONE", "No matching characters", "Try another search or prompt")]
        return _CHAR_ITEMS_CACHE

    items = []
    for cid, disp, kind, tribe, gender, ocnt in rows:
        badge = "★" if kind == "hero" else ("●" if kind == "companion" else "○")
        sub = f"{tribe} • {gender}" + (f" • {ocnt} outfits" if ocnt > 1 else "")
        items.append((cid, f"{badge} {disp} ({tribe})", sub))
    _CHAR_ITEMS_CACHE = items
    return _CHAR_ITEMS_CACHE


def _query_character_outfits(self, context):
    global _OUTFIT_ITEMS_CACHE
    props = context.scene.hfw_toolkit
    cid = props.selected_character
    if not cid or cid == "NONE":
        _OUTFIT_ITEMS_CACHE = [("DEFAULT", "Default Outfit", "Default character appearance")]
        return _OUTFIT_ITEMS_CACHE

    toolkit_dir, _ = _get_prefs(context)
    db_path = toolkit_dir / "cache" / "hfw_index.sqlite"
    if not db_path.exists():
        _OUTFIT_ITEMS_CACHE = [("DEFAULT", "Default Outfit", "Index DB not found")]
        return _OUTFIT_ITEMS_CACHE

    try:
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            """
            SELECT outfit_id, display_name, tribe, is_dlc, has_headgear_toggle, dev_folder
            FROM character_outfits
            WHERE char_id = ?
            ORDER BY is_dlc ASC, tribe, display_name
            LIMIT 100
            """,
            (cid,),
        ).fetchall()
        conn.close()
    except Exception:
        rows = []

    if not rows:
        _OUTFIT_ITEMS_CACHE = [("DEFAULT", "Default Appearance", "Character's built-in BodyVariant")]
        return _OUTFIT_ITEMS_CACHE

    items = [("DEFAULT", "Default Outfit", "Character's default outfit")]
    for oid, disp, tribe, is_dlc, has_hg, dev_f in rows:
        tag = " [DLC]" if is_dlc else ""
        items.append((oid, f"{disp} ({tribe}){tag}", dev_f))
    _OUTFIT_ITEMS_CACHE = items
    return _OUTFIT_ITEMS_CACHE


def _query_character_heads(self, context):
    global _HEAD_ITEMS_CACHE
    toolkit_dir, _ = _get_prefs(context)
    db_path = toolkit_dir / "cache" / "hfw_index.sqlite"
    items = [("DEFAULT", "Default Character Head", "Keep character's built-in head")]
    if not db_path.exists():
        _HEAD_ITEMS_CACHE = items
        return _HEAD_ITEMS_CACHE
    try:
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT facial_id, display_name, gender, dev_folder FROM character_heads ORDER BY display_name LIMIT 120"
        ).fetchall()
        conn.close()
        for fid, disp, gender, dev_f in rows:
            items.append((fid, f"{disp} ({gender}) [{fid}]", dev_f))
    except Exception:
        pass
    _HEAD_ITEMS_CACHE = items
    return _HEAD_ITEMS_CACHE


def _query_dev_assets(self, context):
    global _ASSET_ITEMS_CACHE
    toolkit_dir, _ = _get_prefs(context)
    db_path = toolkit_dir / "cache" / "hfw_index.sqlite"
    if not db_path.exists():
        _ASSET_ITEMS_CACHE = [("NONE", "Run `hfw-toolkit index` first", "Index DB not found")]
        return _ASSET_ITEMS_CACHE

    props = context.scene.hfw_toolkit
    q = (props.asset_search or "").strip()
    cat_f = props.asset_category
    try:
        conn = sqlite3.connect(str(db_path))
        sql = "SELECT dev_path, display_name, category, resource_type, resource_id, is_skinned FROM dev_assets WHERE 1=1"
        params = []
        if cat_f and cat_f != "ALL":
            sql += " AND category = ?"
            params.append(cat_f)
        if q:
            like = f"%{q}%"
            sql += " AND (display_name LIKE ? OR dev_path LIKE ? OR resource_id = ?)"
            params.extend([like, like, q])
        sql += " ORDER BY is_skinned DESC, tset_count DESC, dev_path LIMIT 120"
        rows = conn.execute(sql, params).fetchall()
        conn.close()
    except Exception as exc:
        _ASSET_ITEMS_CACHE = [("NONE", f"Assets not indexed ({exc})", "Run `hfw-toolkit assets list` to index")]
        return _ASSET_ITEMS_CACHE

    if not rows:
        _ASSET_ITEMS_CACHE = [("NONE", "No matching dev assets", "Try another folder or keyword")]
        return _ASSET_ITEMS_CACHE

    _ASSET_ITEMS_CACHE = [
        (
            dev_path,
            f"{disp} ({'Rigged' if is_sk else 'Static'}) — {dev_path}",
            f"{cat} | {rtype} ({rid})",
        )
        for dev_path, disp, cat, rtype, rid, is_sk in rows
    ]
    return _ASSET_ITEMS_CACHE


# ----------------------------------------------------------------------
# Live Shader Control Sync Callbacks (`HFW_MachineShaderControl`)
# ----------------------------------------------------------------------
_UPDATING_SHADERS = False


def _sync_machine_shader_controls(props, context) -> None:
    """Push N-Panel Muscle/Eye/Liquid/Plasma/Animation controls live into `HFW_MachineShaderControl` & scene rigs!"""
    global _UPDATING_SHADERS
    if _UPDATING_SHADERS:
        return
    _UPDATING_SHADERS = True
    try:
        mc = tuple(props.muscle_color)
        ec = tuple(props.eye_color)
        eac = tuple(props.eye_alert_color)
        lc = tuple(props.liquid_override_color)
        pc = tuple(props.plasma_color)

        col_updates = {
            "Muscle Color": (mc[0], mc[1], mc[2], 1.0),
            "Eye Color": (ec[0], ec[1], ec[2], 1.0),
            "Eye Alert Color": (eac[0], eac[1], eac[2], 1.0),
            "Liquid Override Color": (lc[0], lc[1], lc[2], 1.0),
            "Plasma Color": (pc[0], pc[1], pc[2], 1.0),
        }
        val_updates = {
            "Muscle Glow Scale": float(props.muscle_glow_scale),
            "Eye Alert State": float(props.eye_alert_state),
            "Eye Glow Scale": float(props.eye_glow_scale),
            "Liquid Override Mix": 1.0 if props.liquid_override_enable else 0.0,
            "Liquid Glow Scale": float(props.liquid_glow_scale),
            "Plasma Charge": float(props.plasma_charge),
            "Animation Speed": float(props.animation_speed),
        }

        ctrl_ng = bpy.data.node_groups.get("HFW_MachineShaderControl")
        if ctrl_ng is not None:
            for n in ctrl_ng.nodes:
                if n.type == "RGB" and n.label in col_updates:
                    n.outputs[0].default_value = col_updates[n.label]
                elif n.type == "VALUE" and n.label in val_updates:
                    n.outputs[0].default_value = val_updates[n.label]

        # Also keep legacy/Slaughterspine PlasmaChargeControl + armature custom property synced
        pval = float(props.plasma_charge)
        for o in context.scene.objects:
            if "hfw_plasma_charge" in o:
                o["hfw_plasma_charge"] = pval
                o.update_tag()
        p_ng = bpy.data.node_groups.get("HFW_PlasmaChargeControl")
        if p_ng is not None:
            for n in p_ng.nodes:
                if n.type == "VALUE" and n.label == "Plasma Charge":
                    n.outputs[0].default_value = pval

        # Also sync shared Character Skin & Hair controls (`HFW_CharacterControl`)
        char_ng = bpy.data.node_groups.get("HFW_CharacterControl")
        if char_ng is not None:
            sc = tuple(props.skin_tone)
            hc = tuple(props.hair_color)
            for n in char_ng.nodes:
                if n.type == "RGB" and n.label == "Skin Tone":
                    n.outputs[0].default_value = (sc[0], sc[1], sc[2], 1.0)
                elif n.type == "RGB" and n.label == "Hair Color":
                    n.outputs[0].default_value = (hc[0], hc[1], hc[2], 1.0)
                elif n.type == "VALUE" and n.label == "Skin Brightness":
                    n.outputs[0].default_value = float(props.skin_brightness)
                elif n.type == "VALUE" and n.label == "Subsurface Weight":
                    n.outputs[0].default_value = float(props.skin_subsurface)
    finally:
        _UPDATING_SHADERS = False


def _on_shader_control_changed(self, context):
    _sync_machine_shader_controls(self, context)


def _on_muscle_preset_changed(self, context):
    if self.muscle_preset in MUSCLE_PRESETS:
        self.muscle_color = MUSCLE_PRESETS[self.muscle_preset]
    _sync_machine_shader_controls(self, context)


def _on_eye_preset_changed(self, context):
    if self.eye_preset in EYE_PRESETS:
        self.eye_color = EYE_PRESETS[self.eye_preset]
        if self.eye_preset in ("HOSTILE_RED", "CORRUPTED"):
            self.eye_alert_state = 0.0
    _sync_machine_shader_controls(self, context)


def _pull_shader_controls_from_scene(props) -> bool:
    """Populate N-Panel shader controls from the scene's `HFW_MachineShaderControl` and `HFW_CharacterControl` groups."""
    global _UPDATING_SHADERS
    ctrl_ng = bpy.data.node_groups.get("HFW_MachineShaderControl")
    char_ng = bpy.data.node_groups.get("HFW_CharacterControl")
    if ctrl_ng is None and char_ng is None:
        return False
    _UPDATING_SHADERS = True
    try:
        if ctrl_ng is not None:
            for n in ctrl_ng.nodes:
                if n.type == "RGB":
                    c = tuple(n.outputs[0].default_value)[:3]
                    if n.label == "Muscle Color":
                        props.muscle_color = c
                    elif n.label == "Eye Color":
                        props.eye_color = c
                    elif n.label == "Eye Alert Color":
                        props.eye_alert_color = c
                    elif n.label == "Liquid Override Color":
                        props.liquid_override_color = c
                    elif n.label == "Plasma Color":
                        props.plasma_color = c
                elif n.type == "VALUE":
                    v = float(n.outputs[0].default_value)
                    if n.label == "Muscle Glow Scale":
                        props.muscle_glow_scale = v
                    elif n.label == "Eye Alert State":
                        props.eye_alert_state = v
                    elif n.label == "Eye Glow Scale":
                        props.eye_glow_scale = v
                    elif n.label == "Liquid Override Mix":
                        props.liquid_override_enable = v >= 0.5
                    elif n.label == "Liquid Glow Scale":
                        props.liquid_glow_scale = v
                    elif n.label == "Plasma Charge":
                        props.plasma_charge = v
                    elif n.label == "Animation Speed":
                        props.animation_speed = v
        if char_ng is not None:
            for n in char_ng.nodes:
                if n.type == "RGB" and n.label == "Skin Tone":
                    props.skin_tone = tuple(n.outputs[0].default_value)[:3]
                elif n.type == "RGB" and n.label == "Hair Color":
                    props.hair_color = tuple(n.outputs[0].default_value)[:3]
                elif n.type == "VALUE" and n.label == "Skin Brightness":
                    props.skin_brightness = float(n.outputs[0].default_value)
                elif n.type == "VALUE" and n.label == "Subsurface Weight":
                    props.skin_subsurface = float(n.outputs[0].default_value)
        return True
    finally:
        _UPDATING_SHADERS = False


def _on_entity_changed(self, context):
    try:
        items = _query_variants(self, context)
        if items:
            self.selected_variant = items[0][0]
            self.variant_index = int(items[0][0])
    except Exception:
        pass


def _on_variant_changed(self, context):
    try:
        self.variant_index = int(self.selected_variant)
    except Exception:
        pass


def _on_char_changed(self, context):
    try:
        items = _query_character_outfits(self, context)
        if items:
            self.selected_outfit = items[0][0]
    except Exception:
        pass


# ----------------------------------------------------------------------
# Live In-Blender Animation Browser Item & UIList
# ----------------------------------------------------------------------
_ANIM_CATEGORY_ENUM_ITEMS = [
    ("combat", "Combat & Attacks", "Melee strikes, bites, stomps, tail whips, ranged weapon fire & hit reactions"),
    ("movement", "Movement & Locomotion", "Walk, trot, run, sprint, strafe, turns, crouch, stealth & climbs"),
    ("idle", "Idle & Stances", "Idle loops, alert stances, look-arounds & transitions"),
    ("jump", "Jumps, Leaps & Vaults", "Jumps, leaps, vaults, takeoff, airborne & landings"),
    ("emote", "Emotes, Roars & Gestures", "Roars, snarls, grazing, calls, eating/drinking & gestures"),
    ("mounted", "Mount / Rider & Special", "Mount, rider, swim & special actions"),
]

_ANIM_FILTER_ENUM_ITEMS = [
    ("ALL", "All Categories", "Show every animation category"),
] + _ANIM_CATEGORY_ENUM_ITEMS


class HFWAnimationBrowserItem(PropertyGroup):
    selected: BoolProperty(name="Import", default=False)
    anim_id: StringProperty(name="ID", default="")
    name: StringProperty(name="Name", default="")
    category: StringProperty(name="Category", default="")
    duration: FloatProperty(name="Duration", default=0.0)
    num_frames: IntProperty(name="Frames", default=0)
    description: StringProperty(name="Description", default="")
    events: StringProperty(name="Events", default="")


def _anim_item_matches_filter(item, props) -> bool:
    cat_f = props.anim_browser_category
    if cat_f and cat_f != "ALL" and item.category != cat_f:
        return False
    q = (props.anim_browser_search or "").strip().lower()
    if q:
        hay = f"{item.name} {item.description} {item.events} {item.anim_id}".lower()
        if q not in hay:
            return False
    return True


class HFW_UL_AnimationList(UIList):
    bl_idname = "HFW_UL_AnimationList"

    def filter_items(self, context, data, propname):
        items = getattr(data, propname)
        props = context.scene.hfw_toolkit
        flt_flags = [
            self.bitflag_filter_item if _anim_item_matches_filter(it, props) else 0
            for it in items
        ]
        return flt_flags, []

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "selected", text="")
        sub = row.split(factor=0.64, align=True)
        sub.label(text=item.name)
        meta = f"[{item.category[:4].upper()}] {item.duration:.1f}s/{item.num_frames}f"
        sub.label(text=meta)


def _resolve_addon_anim_arg(props) -> str | None:
    mode = props.anim_import_mode
    if mode == "NONE":
        return None
    if mode == "ALL":
        return "all"
    if mode == "CATEGORY":
        return f"category:{props.anim_category}"
    if mode == "SHOWCASE":
        return str(max(1, int(props.import_anim_count)))
    if mode == "SELECTED":
        ids = [it.anim_id for it in props.anim_browser_items if it.selected and it.anim_id]
        return ",".join(ids) if ids else None
    return None


def _resolve_anim_scan_target(context) -> tuple[str, str, str]:
    """Return `(kind, cli_arg_value, display_label)` for scanning or fast-importing animations."""
    props = context.scene.hfw_toolkit
    arm = _find_active_hfw_armature(context)
    if props.import_mode == "MACHINES":
        oid = props.selected_entity
        if oid and oid != "NONE" and ":" in oid:
            return ("machine", oid, f"Machine {oid}")
    elif props.import_mode == "CHARACTERS":
        cid = props.selected_character
        if cid and cid != "NONE":
            return ("character", cid, f"Character {cid}")
        if (props.char_prompt or "").strip():
            return ("character", props.char_prompt.strip(), props.char_prompt.strip())
    if arm is not None:
        akind = str(arm.get("hfw_asset_kind") or "machine")
        eid = str(arm.get("hfw_entity_id") or "")
        if akind == "machine" and eid and ":" in eid:
            return ("machine", eid, arm.name)
        if akind == "character" or "aloy" in arm.name.lower():
            return ("character", arm.name, arm.name)
    return ("character", "Aloy", "Humanoid (Female)")


# ----------------------------------------------------------------------
# Scene Properties
# ----------------------------------------------------------------------
class HFWToolkitSceneProps(PropertyGroup):
    # Top mode switcher
    import_mode: EnumProperty(
        name="Mode",
        items=[
            ("MACHINES", "Machines", "Import rigged HFW machines & customize glowing muscles/eyes/liquids", "ARMATURE_DATA", 0),
            ("CHARACTERS", "Characters", "Import Aloy (+ outfits & hair), Companions, and Tribal NPCs via prompt or browser", "OUTLINER_OB_ARMATURE", 1),
            ("ASSETS", "Assets", "Browse & import weapons, animals, ruins & trees in Guerrilla's dev directory layout", "ASSET_MANAGER", 2),
        ],
        default="MACHINES",
    )

    # --- Machine Importer Props ---
    search_query: StringProperty(
        name="Search",
        description="Filter machines by name or codename (e.g. Slitherfang, Clawstrider, Slaughterspine)",
        default="",
        update=_on_entity_changed,
    )
    selected_entity: EnumProperty(name="Machine", items=_query_machines, update=_on_entity_changed)
    selected_variant: EnumProperty(name="Variant", items=_query_variants, update=_on_variant_changed)
    variant_index: IntProperty(name="Variant Index", default=0, min=0, max=20)
    lod_index: IntProperty(name="LOD", description="0 = Highest detail", default=0, min=0, max=4)
    include_broken_states: BoolProperty(name="Include Broken Plate States", default=True)
    enable_detail_nodes: BoolProperty(name="Wire Detail Array Nodes", default=True)
    build_control_rig: BoolProperty(
        name="Auto-Build IK/FK Control Rig",
        description="Attach non-destructive DEF+MCH+CTRL Control Rig with full IK/FK limbs, fingers, toes & zero ghost animation",
        default=True,
    )
    anim_import_mode: EnumProperty(
        name="Animations",
        description="Choose how many / which HFW EA08 in-game animations to import (no 25-clip limit!)",
        items=[
            ("NONE", "None (Rest Pose)", "Import mesh & Control Rig at clean rest pose without animations"),
            ("SHOWCASE", "Quick Showcase Pack", "Import N diverse clips spanning Combat, Movement, Idle, Jump & Emote"),
            ("CATEGORY", "All of a Category", "Import EVERY animation of one chosen category (e.g. all Combat or Movement)"),
            ("ALL", "Import ALL Animations", "Import EVERY compatible animation clip for this machine/character (no cap!)"),
            ("SELECTED", "Selected Clips (Browser)", "Import only the clips checked in the Live Animation Browser below"),
        ],
        default="NONE",
    )
    anim_category: EnumProperty(
        name="Category",
        description="Animation category to import when mode is 'All of a Category'",
        items=_ANIM_CATEGORY_ENUM_ITEMS,
        default="combat",
    )
    import_anim_count: IntProperty(
        name="Showcase Clips",
        description="Number of diverse showcase clips to import (no 25-clip limit!)",
        default=5,
        min=1,
        max=2000,
    )
    anim_browser_items: CollectionProperty(type=HFWAnimationBrowserItem)
    anim_browser_index: IntProperty(name="Active Animation", default=0)
    anim_browser_category: EnumProperty(
        name="Filter Category",
        items=_ANIM_FILTER_ENUM_ITEMS,
        default="ALL",
    )
    anim_browser_search: StringProperty(
        name="Search Clips",
        description="Filter discovered animations by name, description, or game event",
        default="",
    )
    anim_browser_summary: StringProperty(name="Summary", default="")
    anim_browser_target_label: StringProperty(name="Target", default="")
    show_anim_browser: BoolProperty(name="Animation Browser", default=False)
    show_rig_overrides: BoolProperty(name="Category Influences", default=False)
    unpack_textures: BoolProperty(name="Unpack Textures (Separate Maps)", default=False)
    pack_textures: BoolProperty(name="Pack Textures Into .blend", default=False)

    # --- Live Machine Shader Controls ---
    muscle_preset: EnumProperty(
        name="Muscle Preset",
        items=[
            ("CUSTOM", "Custom Color", "Pick any custom muscle fiber glow color"),
            ("STANDARD", "Standard Amber", "Authentic HFW Standard Machine warm amber-orange muscles"),
            ("APEX", "Apex Purple", "Authentic HFW Apex Machine violet-purple muscles"),
            ("OVERRIDDEN", "Overridden Cyan", "Aloy Overridden electric cyan muscles"),
            ("CORRUPTED", "Corrupted Red", "Corrupted / Hades crimson-red muscles"),
            ("PLASMA", "Plasma Violet", "Slaughterspine charged electric plasma violet"),
            ("ACID", "Acid Lime", "Acid / Bilegut toxic lime-green glow"),
        ],
        default="CUSTOM",
        update=_on_muscle_preset_changed,
    )
    muscle_color: FloatVectorProperty(
        name="Muscle Color",
        subtype="COLOR",
        size=3,
        min=0.0,
        max=1.0,
        default=(0.85, 0.24, 0.04),
        update=_on_shader_control_changed,
    )
    muscle_glow_scale: FloatProperty(
        name="Muscle Glow",
        default=1.0,
        min=0.0,
        max=3.0,
        update=_on_shader_control_changed,
    )
    eye_preset: EnumProperty(
        name="Eye Preset",
        items=[
            ("CUSTOM", "Custom Eye Color", "Pick any custom eye optics color"),
            ("PATROL_BLUE", "Patrol Blue", "Unaware / Patrol calm blue eye optics"),
            ("SUSPICIOUS", "Suspicious Yellow", "Alert / Investigating amber-yellow eye optics"),
            ("HOSTILE_RED", "Hostile Red", "Aggressive combat red eye optics"),
            ("OVERRIDDEN", "Overridden Cyan", "Aloy Overridden bright cyan eye optics"),
            ("APEX_PURPLE", "Apex Purple", "Apex machine violet-purple eye optics"),
            ("CORRUPTED", "Corrupted Crimson", "Corrupted crimson eye optics"),
        ],
        default="CUSTOM",
        update=_on_eye_preset_changed,
    )
    eye_color: FloatVectorProperty(
        name="Eye Color",
        subtype="COLOR",
        size=3,
        min=0.0,
        max=1.0,
        default=(0.06, 0.42, 1.00),
        update=_on_shader_control_changed,
    )
    eye_alert_color: FloatVectorProperty(
        name="Alert Eye Color",
        subtype="COLOR",
        size=3,
        min=0.0,
        max=1.0,
        default=(1.00, 0.06, 0.03),
        update=_on_shader_control_changed,
    )
    eye_alert_state: FloatProperty(
        name="Alert Blend",
        description="Blend smoothly between Calm Eye Color (0.0) and Alert Eye Color (1.0)",
        default=0.0,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
        update=_on_shader_control_changed,
    )
    eye_glow_scale: FloatProperty(
        name="Eye Glow",
        default=1.0,
        min=0.0,
        max=3.0,
        update=_on_shader_control_changed,
    )
    liquid_override_enable: BoolProperty(
        name="Override Liquid Color",
        description="Override native per-elemental sac & canister liquid colors with a custom color",
        default=False,
        update=_on_shader_control_changed,
    )
    liquid_override_color: FloatVectorProperty(
        name="Liquid Color",
        subtype="COLOR",
        size=3,
        min=0.0,
        max=1.0,
        default=(0.08, 0.62, 1.00),
        update=_on_shader_control_changed,
    )
    liquid_glow_scale: FloatProperty(
        name="Liquid Glow",
        default=1.0,
        min=0.0,
        max=3.0,
        update=_on_shader_control_changed,
    )
    plasma_color: FloatVectorProperty(
        name="Plasma Color",
        subtype="COLOR",
        size=3,
        min=0.0,
        max=1.0,
        default=(0.46, 0.05, 1.00),
        update=_on_shader_control_changed,
    )
    plasma_charge: FloatProperty(
        name="Plasma Charge",
        description="Slaughterspine plasma charge-up state (0.0 = Idle, 1.0 = Full Electric Purple Charge)",
        default=1.0,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
        update=_on_shader_control_changed,
    )
    animation_speed: FloatProperty(
        name="Animation Speed",
        description="Global speed multiplier for animated muscles, flowing elemental liquids, eye scans, and plasma",
        default=1.0,
        min=0.0,
        max=4.0,
        update=_on_shader_control_changed,
    )

    # --- Character Importer Props ---
    char_prompt: StringProperty(
        name="Prompt",
        description="Natural-language character prompt (e.g. 'Aloy Carja Blazon no headgear', 'Seyka', 'Erend', 'Tenakth female warrior')",
        default="Aloy Carja Blazon no headgear",
    )
    char_kind_filter: EnumProperty(
        name="Group",
        items=[
            ("MAIN", "Hero & Companions", "Aloy + Main Story Characters (Erend, Seyka, Kotallo, Zo, Alva, Beta, Sylens...)"),
            ("ALL", "All Characters (1,200)", "Aloy, Companions, and all Modular Tribal NPCs"),
            ("NPC", "Tribal NPCs Only", "Hundreds of Nora, Carja, Tenakth, Utaru, Oseram & Quen NPCs"),
        ],
        default="MAIN",
        update=_on_char_changed,
    )
    char_search: StringProperty(
        name="Filter",
        description="Filter characters by name, tribe, or folder",
        default="",
        update=_on_char_changed,
    )
    selected_character: EnumProperty(name="Character", items=_query_characters, update=_on_char_changed)
    selected_outfit: EnumProperty(name="Outfit", items=_query_character_outfits)
    selected_head_swap: EnumProperty(name="Head Swap", items=_query_character_heads)
    char_headgear: BoolProperty(name="Show Headgear", default=True)
    char_include_hair: BoolProperty(name="Include Braided Hair", default=True)
    char_hair_lod: IntProperty(name="Hair LOD", description="0 = Ultra Cinematic, 2 = Balanced (Recommended), 3 = Light", default=2, min=0, max=3)

    # --- Live Character Skin Controls (`HFW_CharacterControl`) ---
    skin_tone: FloatVectorProperty(
        name="Skin Tone",
        description="Shared DXIL 570:0 Skin LUT base tone applied seamlessly across Head & Body",
        subtype="COLOR",
        size=3,
        min=0.0,
        max=1.0,
        default=(0.453, 0.246, 0.209),
        update=_on_shader_control_changed,
    )
    skin_brightness: FloatProperty(
        name="Skin Brightness",
        default=1.0,
        min=0.25,
        max=2.0,
        update=_on_shader_control_changed,
    )
    skin_subsurface: FloatProperty(
        name="Subsurface Weight",
        default=0.14,
        min=0.0,
        max=0.6,
        update=_on_shader_control_changed,
    )
    hair_color: FloatVectorProperty(
        name="Hair Color",
        description="Shared DXIL 382:0 Hair LUT base color for hair, eyebrow, and eyelash cards",
        subtype="COLOR",
        size=3,
        min=0.0,
        max=1.0,
        default=(0.045, 0.032, 0.025),
        update=_on_shader_control_changed,
    )

    # --- Dev-Directory General Asset Catalog Props ---
    asset_category: EnumProperty(
        name="Category",
        items=[
            ("ALL", "All Categories", "All indexed Guerrilla developer-directory assets"),
            ("weapons", "Weapons (146)", "Bows, spears, axes, gauntlets, Zenith weapons"),
            ("animals", "Wildlife / Animals (44)", "Rigged wildlife (boar, sheep, raccoon, scorpion, birds...)"),
            ("building_blocks", "Building Blocks / Ruins (453)", "Architecture, Old World ruins, vehicles, facilities"),
            ("eco_assets", "Eco Assets / Trees (135)", "Trees, pines, shrubs, vines, kelp, foliage"),
            ("landscape", "Landscape Assets (60)", "Cliffs, rocks, geological props"),
            ("props", "Interactive & Cinematic Props", "Interactive objects & cinematic props"),
            ("characters", "Raw Character Folders", "Guerrilla models/characters/humans/... folders"),
            ("machines", "Raw Machine Folders", "Guerrilla models/characters/robots/... folders"),
        ],
        default="weapons",
    )
    asset_search: StringProperty(
        name="Search / Folder",
        description="Filter by asset name or Guerrilla developer folder (e.g. Bow, Pine, Ruins, Boar)",
        default="",
    )
    selected_asset: EnumProperty(name="Dev Asset", items=_query_dev_assets)


# ----------------------------------------------------------------------
# Operators
# ----------------------------------------------------------------------
class HFW_OT_ImportMachine(Operator):
    bl_idname = "hfw_toolkit.import_machine"
    bl_label = "Import Machine Into Scene"
    bl_description = "Extract & assemble the selected HFW machine with armature, bone-parented armor plates & animated shaders"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.hfw_toolkit
        oid = props.selected_entity
        if not oid or oid == "NONE":
            self.report({"ERROR"}, "Select a valid machine first.")
            return {"CANCELLED"}
        try:
            vidx = int(props.selected_variant)
        except Exception:
            vidx = int(props.variant_index)

        toolkit_dir, _ = _get_prefs(context)
        py_exe = _find_toolkit_python(toolkit_dir)
        if py_exe is None:
            self.report({"ERROR"}, f"HFWToolkit Python not found in: {toolkit_dir}")
            return {"CANCELLED"}

        bundle_dir = toolkit_dir / "exports" / "bundles" / f"entity_{oid.replace(':', '_')}_v{vidx}_lod{props.lod_index}"
        cmd = [
            str(py_exe), "-m", "hfw_toolkit.cli", "export-bundle",
            "--machine", oid, "--variant", str(vidx), "--lod", str(props.lod_index), "--out", str(bundle_dir),
        ]
        if not props.include_broken_states:
            cmd.append("--no-broken-states")
        if props.unpack_textures:
            cmd.append("--unpack-textures")
        anim_arg = _resolve_addon_anim_arg(props)
        if anim_arg:
            cmd.extend(["--animations", anim_arg])

        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            self.report({"ERROR"}, f"Bundle extraction failed: {proc.stderr[-300:] or proc.stdout[-300:]}")
            return {"CANCELLED"}

        arm = _import_bundle_direct(
            toolkit_dir,
            bundle_dir,
            props.enable_detail_nodes,
            props.pack_textures,
            build_control_rig=props.build_control_rig,
        )
        _pull_shader_controls_from_scene(props)
        self.report({"INFO"}, f"Imported {arm.name} cleanly!")
        return {"FINISHED"}


class HFW_OT_SyncShaderControls(Operator):
    bl_idname = "hfw_toolkit.sync_shader_controls"
    bl_label = "Read Colors From Scene Rig"
    bl_description = "Pull current Muscle, Eye, Liquid, and Plasma shader settings from the imported HFW rig into the panel"

    def execute(self, context):
        if _pull_shader_controls_from_scene(context.scene.hfw_toolkit):
            self.report({"INFO"}, "Synced live shader controls from scene.")
        else:
            self.report({"WARNING"}, "No HFW machine or character shader control group found in this scene yet.")
        return {"FINISHED"}


class HFW_OT_ImportCharacterPrompt(Operator):
    bl_idname = "hfw_toolkit.import_character_prompt"
    bl_label = "Import Character From Prompt"
    bl_description = "Resolve natural-language prompt (e.g. 'Aloy Carja Blazon no headgear', 'Seyka', 'Tenakth female') and import unified rig"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.hfw_toolkit
        prompt = (props.char_prompt or "").strip()
        if not prompt:
            self.report({"ERROR"}, "Enter a character prompt first (e.g. 'Aloy Carja Blazon no headgear').")
            return {"CANCELLED"}

        toolkit_dir, _ = _get_prefs(context)
        py_exe = _find_toolkit_python(toolkit_dir)
        if py_exe is None:
            self.report({"ERROR"}, f"HFWToolkit Python not found in: {toolkit_dir}")
            return {"CANCELLED"}

        safe = re.sub(r"[^A-Za-z0-9_]+", "_", prompt.lower()).strip("_")[:48] or "character"
        bundle_dir = toolkit_dir / "exports" / "bundles" / f"char_prompt_{safe}"
        cmd = [
            str(py_exe), "-m", "hfw_toolkit.cli", "characters", "export-bundle",
            "--prompt", prompt,
            "--hair-lod", str(props.char_hair_lod),
            "--lod", str(props.lod_index),
            "--out", str(bundle_dir),
        ]
        if props.unpack_textures:
            cmd.append("--unpack-textures")
        anim_arg = _resolve_addon_anim_arg(props)
        if anim_arg:
            cmd.extend(["--animations", anim_arg])

        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            self.report({"ERROR"}, f"Character prompt import failed: {proc.stderr[-300:] or proc.stdout[-300:]}")
            return {"CANCELLED"}

        arm = _import_bundle_direct(
            toolkit_dir,
            bundle_dir,
            False,
            props.pack_textures,
            build_control_rig=props.build_control_rig,
        )
        _pull_shader_controls_from_scene(props)
        self.report({"INFO"}, f"Imported character rig '{arm.name}' from prompt '{prompt}'!")
        return {"FINISHED"}


class HFW_OT_ImportCharacterSelected(Operator):
    bl_idname = "hfw_toolkit.import_character_selected"
    bl_label = "Import Selected Character & Outfit"
    bl_description = "Assemble and import the chosen character, outfit, optional modular head swap, and hair onto ONE unified rig"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.hfw_toolkit
        cid = props.selected_character
        if not cid or cid == "NONE":
            self.report({"ERROR"}, "Select a valid character first.")
            return {"CANCELLED"}

        toolkit_dir, _ = _get_prefs(context)
        py_exe = _find_toolkit_python(toolkit_dir)
        if py_exe is None:
            self.report({"ERROR"}, f"HFWToolkit Python not found in: {toolkit_dir}")
            return {"CANCELLED"}

        oid = props.selected_outfit if props.selected_outfit != "DEFAULT" else ""
        fid = props.selected_head_swap if props.selected_head_swap != "DEFAULT" else ""
        safe = re.sub(r"[^A-Za-z0-9_]+", "_", f"{cid}_{oid or 'def'}").strip("_")[:52]
        bundle_dir = toolkit_dir / "exports" / "bundles" / f"char_{safe}_hg{int(props.char_headgear)}"

        cmd = [
            str(py_exe), "-m", "hfw_toolkit.cli", "characters", "export-bundle",
            "--char", cid,
            "--hair-lod", str(props.char_hair_lod),
            "--lod", str(props.lod_index),
            "--out", str(bundle_dir),
        ]
        if oid:
            cmd.extend(["--outfit", oid])
        if fid:
            cmd.extend(["--facial", fid])
        if not props.char_headgear:
            cmd.append("--no-headgear")
        if not props.char_include_hair:
            cmd.append("--no-hair")
        if props.unpack_textures:
            cmd.append("--unpack-textures")
        anim_arg = _resolve_addon_anim_arg(props)
        if anim_arg:
            cmd.extend(["--animations", anim_arg])

        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            self.report({"ERROR"}, f"Character import failed: {proc.stderr[-300:] or proc.stdout[-300:]}")
            return {"CANCELLED"}

        arm = _import_bundle_direct(
            toolkit_dir,
            bundle_dir,
            False,
            props.pack_textures,
            build_control_rig=props.build_control_rig,
        )
        _pull_shader_controls_from_scene(props)
        self.report({"INFO"}, f"Imported character rig '{arm.name}' cleanly!")
        return {"FINISHED"}


class HFW_OT_ImportDevAsset(Operator):
    bl_idname = "hfw_toolkit.import_dev_asset"
    bl_label = "Import Selected Dev Asset"
    bl_description = "Extract & import the selected Guerrilla developer-directory asset (Weapon, Animal, Ruin, Tree, Prop) with materials"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.hfw_toolkit
        apath = props.selected_asset
        if not apath or apath == "NONE":
            self.report({"ERROR"}, "Select a valid developer asset first.")
            return {"CANCELLED"}

        toolkit_dir, _ = _get_prefs(context)
        py_exe = _find_toolkit_python(toolkit_dir)
        if py_exe is None:
            self.report({"ERROR"}, f"HFWToolkit Python not found in: {toolkit_dir}")
            return {"CANCELLED"}

        safe = re.sub(r"[^A-Za-z0-9_]+", "_", apath.lower()).strip("_")[-52:] or "asset"
        bundle_dir = toolkit_dir / "exports" / "bundles" / f"asset_{safe}_lod{props.lod_index}"
        cmd = [
            str(py_exe), "-m", "hfw_toolkit.cli", "assets", "export-bundle",
            "--asset", apath,
            "--lod", str(props.lod_index),
            "--out", str(bundle_dir),
        ]
        if props.unpack_textures:
            cmd.append("--unpack-textures")

        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            self.report({"ERROR"}, f"Asset import failed: {proc.stderr[-300:] or proc.stdout[-300:]}")
            return {"CANCELLED"}

        root_obj = _import_bundle_direct(toolkit_dir, bundle_dir, props.enable_detail_nodes, props.pack_textures)
        self.report({"INFO"}, f"Imported asset '{root_obj.name}' ({apath}) cleanly!")
        return {"FINISHED"}


class HFW_OT_BuildControlRig(Operator):
    bl_idname = "hfw_toolkit.build_control_rig"
    bl_label = "Build / Rebuild Control Rig"
    bl_description = "Generate non-destructive IK/FK Control Rig on the active HFW armature (preserves all game animations!)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        arm = _find_active_hfw_armature(context)
        if arm is None:
            self.report({"ERROR"}, "Select an HFW Armature first.")
            return {"CANCELLED"}
        toolkit_dir, _ = _get_prefs(context)
        cr = _load_control_rig_module(toolkit_dir)
        has_act = bool(arm.animation_data and arm.animation_data.action)
        info = cr.build_control_rig(arm, default_master=0.0 if has_act else 1.0, rebuild=True)
        self.report(
            {"INFO"},
            f"Built {info['archetype']} Control Rig ({info['control_count']} controls, {info['ik_limb_count']} IK limbs)!",
        )
        return {"FINISHED"}


class HFW_OT_BakeAnimToControlRig(Operator):
    bl_idname = "hfw_toolkit.bake_anim_to_ctrl"
    bl_label = "Bake Game Anim -> Control Rig"
    bl_description = "Transfer the active HFW game animation onto IK/FK Control Rig handles so you can edit feet, spine & head freely!"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        arm = _find_active_hfw_armature(context)
        if arm is None:
            self.report({"ERROR"}, "Select an HFW Armature first.")
            return {"CANCELLED"}
        toolkit_dir, _ = _get_prefs(context)
        cr = _load_control_rig_module(toolkit_dir)
        try:
            baked = cr.bake_action_to_control_rig(arm, activate_baked=True)
            self.report({"INFO"}, f"Baked editable Control Rig Action '{baked.name}' (Control Rig = 1.0)!")
            return {"FINISHED"}
        except Exception as exc:
            self.report({"ERROR"}, f"Bake to Control Rig failed: {exc}")
            return {"CANCELLED"}


class HFW_OT_BakeControlRigToDeform(Operator):
    bl_idname = "hfw_toolkit.bake_ctrl_to_def"
    bl_label = "Bake Control Rig -> Game Skeleton"
    bl_description = "Bake your Control Rig edits back onto clean DEF game skeleton keyframes (ready for FBX / UE5 export!)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        arm = _find_active_hfw_armature(context)
        if arm is None:
            self.report({"ERROR"}, "Select an HFW Armature first.")
            return {"CANCELLED"}
        toolkit_dir, _ = _get_prefs(context)
        cr = _load_control_rig_module(toolkit_dir)
        try:
            baked = cr.bake_control_rig_to_deform(arm, activate_baked=True)
            self.report({"INFO"}, f"Baked clean game-skeleton Action '{baked.name}' (ready for UE5/FBX)!")
            return {"FINISHED"}
        except Exception as exc:
            self.report({"ERROR"}, f"Bake to Game Skeleton failed: {exc}")
            return {"CANCELLED"}


class HFW_OT_ResetControlRigPose(Operator):
    bl_idname = "hfw_toolkit.reset_ctrl_pose"
    bl_label = "Reset Control Rig Pose"
    bl_description = "Reset all CTRL bones back to their clean bind pose"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        arm = _find_active_hfw_armature(context)
        if arm is None:
            self.report({"ERROR"}, "Select an HFW Armature first.")
            return {"CANCELLED"}
        toolkit_dir, _ = _get_prefs(context)
        cr = _load_control_rig_module(toolkit_dir)
        cr.reset_control_rig_pose(arm)
        self.report({"INFO"}, "Reset Control Rig pose.")
        return {"FINISHED"}


class HFW_OT_MirrorControlRigPose(Operator):
    bl_idname = "hfw_toolkit.mirror_ctrl_pose"
    bl_label = "Mirror Control Rig Pose"
    bl_description = "Mirror paired Left/Right Control Rig poses symmetrically across X=0 (wings, legs, arms, fingers, feathers, plasma intakes)"
    bl_options = {"REGISTER", "UNDO"}

    source_side: StringProperty(default="L")

    def execute(self, context):
        arm = _find_active_hfw_armature(context)
        if arm is None:
            self.report({"ERROR"}, "Select an HFW Armature first.")
            return {"CANCELLED"}
        toolkit_dir, _ = _get_prefs(context)
        cr = _load_control_rig_module(toolkit_dir)
        n = cr.mirror_control_rig_pose(arm, source_side=self.source_side)
        arrow = "L -> R" if self.source_side.upper().startswith("L") else "R -> L"
        if n == 0:
            self.report({"WARNING"}, f"No paired L/R controls found on '{arm.name}'.")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Mirrored {n} control poses ({arrow}) on '{arm.name}'!")
        return {"FINISHED"}


class HFW_OT_PoseSerpentWave(Operator):
    bl_idname = "hfw_toolkit.pose_serpent_wave"
    bl_label = "Serpent S-Wave / Coil Pose"
    bl_description = "Shape the Spline IK body / tail / trunk into a smooth sinusoidal slithering S-wave or ground coil"
    bl_options = {"REGISTER", "UNDO"}

    style: StringProperty(default="wave")

    def execute(self, context):
        arm = _find_active_hfw_armature(context)
        if arm is None:
            return {"CANCELLED"}
        toolkit_dir, _ = _get_prefs(context)
        cr = _load_control_rig_module(toolkit_dir)
        n = cr.pose_serpent_wave(arm, style=self.style)
        if n == 0:
            self.report({"WARNING"}, f"No Spline IK controls found on '{arm.name}'.")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Applied smooth Spline IK '{self.style}' pose across {n} gimbal handles!")
        return {"FINISHED"}


class HFW_OT_ToggleSpinSpeed(Operator):
    bl_idname = "hfw_toolkit.toggle_spin_speed"
    bl_label = "Toggle Chainsaw / Drill Auto-Spin"
    bl_description = "Start or stop real-time playback spinning for machine Chainsaws or Drills/Grinders/Rotors"
    bl_options = {"REGISTER", "UNDO"}

    prop_key: StringProperty(default="hfw_saw_speed")
    target_speed: FloatProperty(default=1.0)

    def execute(self, context):
        arm = _find_active_hfw_armature(context)
        if arm is None or self.prop_key not in arm:
            return {"CANCELLED"}
        cur = float(arm[self.prop_key])
        arm[self.prop_key] = 0.0 if abs(cur) > 0.05 else float(self.target_speed)
        arm.update_tag()
        context.view_layer.update()
        state = "ON" if abs(float(arm[self.prop_key])) > 0.05 else "OFF"
        self.report({"INFO"}, f"Auto-spin ({self.prop_key}) turned {state} (press Spacebar to watch it spin!).")
        return {"FINISHED"}


class HFW_OT_ScanAnimations(Operator):
    bl_idname = "hfw_toolkit.scan_animations"
    bl_label = "Scan Available Animations"
    bl_description = "Discover ALL compatible HFW animations with plain-English descriptions & category counts before importing"

    def execute(self, context):
        props = context.scene.hfw_toolkit
        toolkit_dir, _ = _get_prefs(context)
        py_exe = _find_toolkit_python(toolkit_dir)
        if py_exe is None:
            self.report({"ERROR"}, f"HFWToolkit Python not found in: {toolkit_dir}")
            return {"CANCELLED"}

        kind, target_val, label = _resolve_anim_scan_target(context)
        cmd = [
            str(py_exe), "-m", "hfw_toolkit.cli", "animations", "list",
            "--machine" if kind == "machine" else "--character", target_val,
            "--limit", "0",
            "--json",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            self.report({"ERROR"}, f"Animation scan failed: {proc.stderr[-250:] or proc.stdout[-250:]}")
            return {"CANCELLED"}

        try:
            payload = json.loads(proc.stdout)
        except Exception as exc:
            self.report({"ERROR"}, f"Invalid JSON from animation scan: {exc}")
            return {"CANCELLED"}

        props.anim_browser_items.clear()
        for row in payload.get("items") or ():
            it = props.anim_browser_items.add()
            it.selected = False
            it.anim_id = str(row.get("anim_id", ""))
            it.name = str(row.get("name", ""))
            it.category = str(row.get("category", "mounted"))
            it.duration = float(row.get("duration", 0.0))
            it.num_frames = int(row.get("num_frames", 0))
            it.description = str(row.get("description", ""))
            it.events = ", ".join(row.get("events") or ())

        props.anim_browser_index = 0
        counts = payload.get("category_counts") or {}
        parts = [f"{counts.get('all', len(props.anim_browser_items))} clips"]
        for k, short_lbl in (("combat", "Combat"), ("movement", "Move"), ("idle", "Idle"), ("jump", "Jump"), ("emote", "Emote"), ("mounted", "Mount")):
            n = int(counts.get(k, 0) or 0)
            if n > 0:
                parts.append(f"{short_lbl}:{n}")
        props.anim_browser_summary = " | ".join(parts)
        props.anim_browser_target_label = str(payload.get("target") or label)
        props.show_anim_browser = True
        self.report({"INFO"}, f"Found {props.anim_browser_summary} for {props.anim_browser_target_label}")
        return {"FINISHED"}


class HFW_OT_SelectVisibleAnimations(Operator):
    bl_idname = "hfw_toolkit.select_visible_animations"
    bl_label = "Select / Clear Visible Clips"
    bl_description = "Check or uncheck all animations currently visible under the active category/search filter"

    select_all: BoolProperty(default=True)

    def execute(self, context):
        props = context.scene.hfw_toolkit
        count = 0
        for it in props.anim_browser_items:
            if not self.select_all or _anim_item_matches_filter(it, props):
                it.selected = bool(self.select_all)
                if self.select_all:
                    count += 1
        if self.select_all and count > 0:
            props.anim_import_mode = "SELECTED"
        self.report({"INFO"}, f"Selected {count} clips." if self.select_all else "Cleared animation selection.")
        return {"FINISHED"}


class HFW_OT_ImportAnimationsToActiveRig(Operator):
    bl_idname = "hfw_toolkit.import_anims_to_rig"
    bl_label = "Import Animations Onto Active Rig"
    bl_description = "Fast-import ALL, Category, Showcase, or Checked animations straight onto the active HFW armature without re-exporting meshes!"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        arm = _find_active_hfw_armature(context)
        if arm is None:
            self.report({"ERROR"}, "Select an imported HFW Armature first.")
            return {"CANCELLED"}
        props = context.scene.hfw_toolkit
        anim_arg = _resolve_addon_anim_arg(props)
        if not anim_arg:
            self.report({"ERROR"}, "Choose an animation mode (All / Category / Showcase / Selected clips) first.")
            return {"CANCELLED"}

        toolkit_dir, _ = _get_prefs(context)
        py_exe = _find_toolkit_python(toolkit_dir)
        if py_exe is None:
            self.report({"ERROR"}, f"HFWToolkit Python not found in: {toolkit_dir}")
            return {"CANCELLED"}

        akind = str(arm.get("hfw_asset_kind") or "machine")
        eid = str(arm.get("hfw_entity_id") or "")
        if akind == "machine" and eid and ":" in eid:
            flag, target_val = "--machine", eid
        else:
            kind, target_val, _ = _resolve_anim_scan_target(context)
            flag = "--machine" if kind == "machine" else "--character"

        safe_t = re.sub(r"[^A-Za-z0-9_]+", "_", f"{target_val}_{props.anim_import_mode}").strip("_")[:48] or "pack"
        pack_dir = toolkit_dir / "exports" / "anim_packs" / safe_t
        cmd = [
            str(py_exe), "-m", "hfw_toolkit.cli", "animations", "export-pack",
            flag, target_val,
            "--animations", anim_arg,
            "--out", str(pack_dir),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            self.report({"ERROR"}, f"Animation pack export failed: {proc.stderr[-250:] or proc.stdout[-250:]}")
            return {"CANCELLED"}

        cr = _load_control_rig_module(toolkit_dir)
        actions = cr.import_animations_into_armature(arm, pack_dir, activate_first=True)
        if actions and "hfw_rig_master" in arm:
            arm["hfw_rig_master"] = 0.0
            arm.update_tag()
        self.report({"INFO"}, f"Imported {len(actions)} HFW animation Action(s) onto '{arm.name}'!")
        return {"FINISHED"}


class HFW_OT_SetRigMode(Operator):
    bl_idname = "hfw_toolkit.set_rig_mode"
    bl_label = "Switch Anim / Control Rig Mode"
    bl_description = "Switch cleanly between Playing HFW Game Animation (0.0) and Full Player Control Rig Posing (1.0)"
    bl_options = {"REGISTER", "UNDO"}

    to_control_rig: BoolProperty(default=True)

    def execute(self, context):
        arm = _find_active_hfw_armature(context)
        if arm is None:
            return {"CANCELLED"}
        arm["hfw_rig_master"] = 1.0 if self.to_control_rig else 0.0
        arm.update_tag()
        context.view_layer.update()
        mode_lbl = "Control Rig (Pose Mode)" if self.to_control_rig else "Game Animation Playback"
        self.report({"INFO"}, f"Switched '{arm.name}' to {mode_lbl}.")
        return {"FINISHED"}


class HFW_OT_SwitchLimbIKFK(Operator):
    bl_idname = "hfw_toolkit.switch_limb_ik_fk"
    bl_label = "Switch Limb IK / FK (Keep Pose)"
    bl_description = "Switch Legs or Arms between IK and FK, auto-snapping the pose (zero pop!) and showing the active controls"
    bl_options = {"REGISTER", "UNDO"}

    group: StringProperty(default="legs")
    to_ik: BoolProperty(default=True)
    snap_pose: BoolProperty(default=True)

    def execute(self, context):
        arm = _find_active_hfw_armature(context)
        if arm is None:
            return {"CANCELLED"}
        toolkit_dir, _ = _get_prefs(context)
        cr = _load_control_rig_module(toolkit_dir)
        res = cr.switch_limb_ik_fk(arm, group=self.group, to_ik=self.to_ik, snap_pose=self.snap_pose)
        self.report({"INFO"}, f"Switched {self.group.title()} to {res['mode']} (snapped pose cleanly!).")
        return {"FINISHED"}


class HFW_OT_SelectCtrlChain(Operator):
    bl_idname = "hfw_toolkit.select_ctrl_chain"
    bl_label = "Select Control / FK Chain"
    bl_description = "Select the Chest controller or all Tail / Spine FK controls in Pose Mode with Individual Origins for smooth multi-bend"
    bl_options = {"REGISTER", "UNDO"}

    group: StringProperty(default="tail")

    def execute(self, context):
        arm = _find_active_hfw_armature(context)
        if arm is None:
            return {"CANCELLED"}
        toolkit_dir, _ = _get_prefs(context)
        cr = _load_control_rig_module(toolkit_dir)
        n = cr.select_ctrl_chain(arm, group=self.group)
        if n == 0:
            self.report({"WARNING"}, f"No {self.group} controls found on '{arm.name}'.")
            return {"CANCELLED"}
        if self.group == "chest":
            self.report({"INFO"}, f"Selected chest/hood controller on '{arm.name}' (press G to arch or R to twist/lean!).")
        elif self.group == "wings" and float(arm.get("hfw_ik_fk_wings", 0.0)) >= 0.5:
            self.report({"INFO"}, f"Selected {n} Wing IK tip controls (press G to position wingtips!).")
        else:
            self.report({"INFO"}, f"Selected {n} {self.group} FK controls (Individual Origins ON — press R to bend smoothly!).")
        return {"FINISHED"}


class HFW_OT_ToggleBoneCollection(Operator):
    bl_idname = "hfw_toolkit.toggle_bone_coll"
    bl_label = "Toggle Control Category"
    bl_description = "Show or hide this category of Control Rig / Skeleton bones"

    coll_name: StringProperty(default="Body_Main")

    def execute(self, context):
        arm = _find_active_hfw_armature(context)
        if arm is None:
            return {"CANCELLED"}
        all_c = getattr(arm.data, "collections_all", arm.data.collections)
        if self.coll_name == "RAW_DEF":
            raw_colls = [all_c.get(n) for n in ("Skeleton", "Body", "Helpers")]
            raw_colls = [c for c in raw_colls if c is not None]
            if not raw_colls:
                return {"CANCELLED"}
            new_vis = not any(c.is_visible for c in raw_colls)
            for c in raw_colls:
                c.is_visible = new_vis
            return {"FINISHED"}
        bc = all_c.get(self.coll_name)
        if bc is not None:
            bc.is_visible = not bc.is_visible
            if bc.is_visible and getattr(bc, "parent", None) is not None:
                bc.parent.is_visible = True
        return {"FINISHED"}


class HFW_OT_InspectMaterialWeb(Operator):
    bl_idname = "hfw_toolkit.inspect_material_web"
    bl_label = "Open in Web Studio / Shader Explorer"
    bl_description = "Open the active object's HFW material (or Web Studio) in your browser"

    def execute(self, context):
        _, web_url = _get_prefs(context)
        obj = context.active_object
        mat_id = None
        if obj and obj.active_material:
            mat_id = obj.active_material.get("hfw_material_id")
        if mat_id:
            webbrowser.open(f"{web_url}/#/material/{mat_id}")
        else:
            ent_id = obj.get("hfw_entity_id") if obj else None
            if ent_id:
                webbrowser.open(f"{web_url}/#/machine/{ent_id}")
            else:
                webbrowser.open(web_url)
        return {"FINISHED"}


def _draw_anim_options_block(layout, props, arm=None):
    abox = layout.box()
    hdr = abox.row(align=True)
    hdr.label(text="In-Game Animations (No Cap!)", icon="ACTION")
    hdr.operator("hfw_toolkit.scan_animations", text="Scan / Browse", icon="VIEWZOOM")
    abox.prop(props, "anim_import_mode", text="Mode")
    if props.anim_import_mode == "CATEGORY":
        abox.prop(props, "anim_category", text="Type")
    elif props.anim_import_mode == "SHOWCASE":
        abox.prop(props, "import_anim_count", text="Clips")
    elif props.anim_import_mode == "SELECTED":
        n_sel = sum(1 for it in props.anim_browser_items if it.selected)
        abox.label(text=f"Selected in Browser: {n_sel} clip(s)", icon="CHECKBOX_HLT" if n_sel else "INFO")

    if len(props.anim_browser_items) > 0:
        row_tog = abox.row(align=True)
        row_tog.prop(
            props,
            "show_anim_browser",
            icon="TRIA_DOWN" if props.show_anim_browser else "TRIA_RIGHT",
            text=f"Browser ({props.anim_browser_summary or len(props.anim_browser_items)})",
            emboss=False,
        )
        if props.show_anim_browser:
            if props.anim_browser_target_label:
                abox.label(text=f"Target: {props.anim_browser_target_label}", icon="ARMATURE_DATA")
            row_f = abox.row(align=True)
            row_f.prop(props, "anim_browser_category", text="")
            row_f.prop(props, "anim_browser_search", text="", icon="VIEWZOOM")
            abox.template_list(
                "HFW_UL_AnimationList",
                "",
                props,
                "anim_browser_items",
                props,
                "anim_browser_index",
                rows=5,
            )
            if 0 <= props.anim_browser_index < len(props.anim_browser_items):
                cur = props.anim_browser_items[props.anim_browser_index]
                dbox = abox.box()
                dbox.scale_y = 0.85
                dbox.label(text=cur.description[:76] if cur.description else cur.name, icon="INFO")
                if cur.description and len(cur.description) > 76:
                    dbox.label(text=cur.description[76:152])
            row_sel = abox.row(align=True)
            op_all = row_sel.operator("hfw_toolkit.select_visible_animations", text="Check Visible", icon="CHECKBOX_HLT")
            op_all.select_all = True
            op_clr = row_sel.operator("hfw_toolkit.select_visible_animations", text="Clear", icon="CHECKBOX_DEHLT")
            op_clr.select_all = False

    if arm is not None and props.anim_import_mode != "NONE":
        abox.operator("hfw_toolkit.import_anims_to_rig", icon="IMPORT")


def _get_arm_rig_meta_and_slots(arm) -> tuple[dict, dict[str, str]]:
    """Read `hfw_ctrl_rig_meta` and dynamically renamed Bone Collection slots (with clean legacy fallback)."""
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
    all_c = getattr(arm.data, "collections_all", arm.data.collections)
    saved = meta.get("collection_slots")
    if isinstance(saved, dict) and saved:
        for k, v in saved.items():
            if isinstance(v, str) and v in all_c:
                slots[k] = v
    else:
        for skey, legacy_nm in (
            ("body", "Body_Main"),
            ("tail", "Tail"),
            ("wings", "Wings FK"),
            ("legs_ik", "Legs_IK"),
            ("legs_fk", "Legs_FK"),
            ("arms_ik", "Arms_IK"),
            ("arms_fk", "Arms_FK"),
            ("hands_feet", "Hands_Feet"),
            ("frills", "Frills_Feathers"),
            ("weapons", "Weapons_Aux"),
            ("plates", "Plates_Armor"),
        ):
            if legacy_nm in all_c:
                slots[skey] = legacy_nm
    return meta, slots


# ----------------------------------------------------------------------
# Clean, Decluttered Importer & Pipeline Panels (`HFW Toolkit` Tab)
# Live Rig Controls & Shader Controls now live NATIVELY in `Sidebar -> Item`
# (embedded inside every `.blend` so rigs & shaders work WITHOUT this add-on!)
# ----------------------------------------------------------------------
class HFW_PT_MainPanel(Panel):
    bl_label = "HFW Asset Importer"
    bl_idname = "HFW_PT_main_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "HFW Toolkit"

    def draw(self, context):
        layout = self.layout
        props = context.scene.hfw_toolkit

        row = layout.row(align=True)
        row.prop(props, "import_mode", expand=True)

        if props.import_mode == "MACHINES":
            col = layout.column(align=True)
            col.prop(props, "search_query", text="", icon="VIEWZOOM")
            col.prop(props, "selected_entity", text="Machine")
            col.prop(props, "selected_variant", text="Variant")
            row_lod = layout.row(align=True)
            row_lod.prop(props, "lod_index")
            row_lod.prop(props, "build_control_rig", text="Control Rig", toggle=True)
            row_opt = layout.row(align=True)
            row_opt.prop(props, "include_broken_states", text="Broken Plates")
            row_opt.prop(props, "enable_detail_nodes", text="Micro-Detail")
            row_tx = layout.row(align=True)
            row_tx.prop(props, "pack_textures", text="Pack Tex")
            row_tx.prop(props, "unpack_textures", text="Unpack")
            op_r = layout.row()
            op_r.scale_y = 1.25
            op_r.operator("hfw_toolkit.import_machine", icon="IMPORT")

        elif props.import_mode == "CHARACTERS":
            pbox = layout.box()
            pbox.label(text="Quick Prompt Import", icon="CONSOLE")
            pbox.prop(props, "char_prompt", text="")
            pbox.operator("hfw_toolkit.import_character_prompt", icon="IMPORT")

            cbox = layout.box()
            cbox.label(text="Character & Outfit Browser", icon="OUTLINER_OB_ARMATURE")
            cbox.prop(props, "char_kind_filter", text="Group")
            cbox.prop(props, "char_search", text="", icon="VIEWZOOM")
            cbox.prop(props, "selected_character", text="Character")
            cbox.prop(props, "selected_outfit", text="Outfit")
            cbox.prop(props, "selected_head_swap", text="Head")
            row_opt = cbox.row(align=True)
            row_opt.prop(props, "char_headgear", text="Headgear", toggle=True)
            row_opt.prop(props, "char_include_hair", text="Hair", toggle=True)
            row_lod = cbox.row(align=True)
            row_lod.prop(props, "char_hair_lod", text="Hair LOD")
            row_lod.prop(props, "build_control_rig", text="Control Rig", toggle=True)
            cbox.prop(props, "pack_textures")
            op_r = cbox.row()
            op_r.scale_y = 1.2
            op_r.operator("hfw_toolkit.import_character_selected", icon="IMPORT")

        else:  # ASSETS
            col = layout.column(align=True)
            col.prop(props, "asset_category", text="Category")
            col.prop(props, "asset_search", text="", icon="VIEWZOOM")
            col.prop(props, "selected_asset", text="Asset")
            row_lod = layout.row(align=True)
            row_lod.prop(props, "lod_index")
            row_lod.prop(props, "pack_textures", text="Pack Tex", toggle=True)
            op_r = layout.row()
            op_r.scale_y = 1.25
            op_r.operator("hfw_toolkit.import_dev_asset", icon="IMPORT")


class HFW_PT_AnimLibraryPanel(Panel):
    bl_label = "In-Game Animations"
    bl_idname = "HFW_PT_anim_library"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "HFW Toolkit"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        props = context.scene.hfw_toolkit
        arm = _find_active_hfw_armature(context)
        _draw_anim_options_block(layout, props, arm=arm)


class HFW_PT_PipelinePanel(Panel):
    bl_label = "Baking & Rig Utilities"
    bl_idname = "HFW_PT_pipeline_tools"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "HFW Toolkit"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        arm = _find_active_hfw_armature(context)

        tip = layout.box()
        tip.scale_y = 0.85
        tip.label(text="Rig & Shader Controls are in Sidebar -> Item", icon="INFO")
        tip.label(text="(Embedded in .blend — works without add-on!)")

        if arm is not None:
            col = layout.column(align=True)
            col.operator("hfw_toolkit.bake_anim_to_ctrl", icon="KEYFRAME_HLT")
            col.operator("hfw_toolkit.bake_ctrl_to_def", icon="EXPORT")
            row = layout.row(align=True)
            row.operator("hfw_toolkit.build_control_rig", text="Rebuild Rig", icon="ARMATURE_DATA")
            row.operator("hfw_toolkit.reset_ctrl_pose", text="Reset Pose", icon="LOOP_BACK")

        layout.separator()
        obj = context.active_object
        if obj and obj.active_material and obj.active_material.get("hfw_material_id"):
            layout.label(text=f"Material: {obj.active_material['hfw_material_id']}", icon="NODE_MATERIAL")
        layout.operator("hfw_toolkit.inspect_material_web", icon="URL")


_CLASSES = (
    HFWToolkitPreferences,
    HFWAnimationBrowserItem,
    HFW_UL_AnimationList,
    HFWToolkitSceneProps,
    HFW_OT_ImportMachine,
    HFW_OT_SyncShaderControls,
    HFW_OT_ImportCharacterPrompt,
    HFW_OT_ImportCharacterSelected,
    HFW_OT_ImportDevAsset,
    HFW_OT_BuildControlRig,
    HFW_OT_BakeAnimToControlRig,
    HFW_OT_BakeControlRigToDeform,
    HFW_OT_ResetControlRigPose,
    HFW_OT_MirrorControlRigPose,
    HFW_OT_PoseSerpentWave,
    HFW_OT_ToggleSpinSpeed,
    HFW_OT_ScanAnimations,
    HFW_OT_SelectVisibleAnimations,
    HFW_OT_ImportAnimationsToActiveRig,
    HFW_OT_SetRigMode,
    HFW_OT_SwitchLimbIKFK,
    HFW_OT_SelectCtrlChain,
    HFW_OT_ToggleBoneCollection,
    HFW_OT_InspectMaterialWeb,
    HFW_PT_MainPanel,
    HFW_PT_AnimLibraryPanel,
    HFW_PT_PipelinePanel,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.hfw_toolkit = bpy.props.PointerProperty(type=HFWToolkitSceneProps)


def unregister():
    del bpy.types.Scene.hfw_toolkit
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
