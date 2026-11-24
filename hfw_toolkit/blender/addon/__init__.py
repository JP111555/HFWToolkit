"""HFW Toolkit — Blender 4.2+ / 5.3 Add-on (N-Panel Machine, Character & Dev-Asset Importer + Live Shader Controls).

Installable via Blender Preferences -> Add-ons -> Install from Disk (or `hfw-toolkit install-blender-addon`).
Uses only Blender's built-in Python (`bpy`, `sqlite3`, `subprocess`, `webbrowser`).
"""

import importlib.util
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
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    StringProperty,
)
from bpy.types import AddonPreferences, Operator, Panel, PropertyGroup

bl_info = {
    "name": "HFW Toolkit — Machines, Characters, Assets & Live Shaders",
    "author": "HFWToolkit",
    "version": (0, 5, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar (N-Panel) > HFW Toolkit",
    "description": (
        "One-click import rigged Horizon Forbidden West machines, characters (natural-language prompt + outfits + "
        "hair + unified rig), and Guerrilla dev-directory assets — with live Muscle/Eye/Liquid/Plasma shader controls."
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


def _import_bundle_direct(toolkit_dir: Path, bundle_dir: Path, enable_detail: bool, pack_tex: bool):
    builder_path = toolkit_dir / "hfw_toolkit" / "blender" / "blender_builder.py"
    spec = importlib.util.spec_from_file_location("hfw_blender_builder", str(builder_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.import_bundle_into_scene(
        bundle_dir,
        enable_detail_nodes=enable_detail,
        pack_textures=pack_tex,
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

        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            self.report({"ERROR"}, f"Bundle extraction failed: {proc.stderr[-300:] or proc.stdout[-300:]}")
            return {"CANCELLED"}

        arm = _import_bundle_direct(toolkit_dir, bundle_dir, props.enable_detail_nodes, props.pack_textures)
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

        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            self.report({"ERROR"}, f"Character prompt import failed: {proc.stderr[-300:] or proc.stdout[-300:]}")
            return {"CANCELLED"}

        arm = _import_bundle_direct(toolkit_dir, bundle_dir, False, props.pack_textures)
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

        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            self.report({"ERROR"}, f"Character import failed: {proc.stderr[-300:] or proc.stdout[-300:]}")
            return {"CANCELLED"}

        arm = _import_bundle_direct(toolkit_dir, bundle_dir, False, props.pack_textures)
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


# ----------------------------------------------------------------------
# Main N-Panel UI
# ----------------------------------------------------------------------
class HFW_PT_MainPanel(Panel):
    bl_label = "HFW Toolkit"
    bl_idname = "HFW_PT_main_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "HFW Toolkit"

    def draw(self, context):
        layout = self.layout
        props = context.scene.hfw_toolkit

        # Mode Switcher Row
        row = layout.row(align=True)
        row.prop(props, "import_mode", expand=True)

        if props.import_mode == "MACHINES":
            box = layout.box()
            box.label(text="Machine & Rig Importer", icon="ARMATURE_DATA")
            box.prop(props, "search_query", icon="VIEWZOOM")
            box.prop(props, "selected_entity")
            box.prop(props, "selected_variant")
            box.prop(props, "lod_index")
            box.prop(props, "include_broken_states")
            box.prop(props, "enable_detail_nodes")
            box.prop(props, "unpack_textures")
            box.prop(props, "pack_textures")
            box.operator("hfw_toolkit.import_machine", icon="IMPORT")

        elif props.import_mode == "CHARACTERS":
            pbox = layout.box()
            pbox.label(text="Quick Prompt Character Import", icon="CONSOLE")
            pbox.prop(props, "char_prompt", text="")
            pbox.operator("hfw_toolkit.import_character_prompt", icon="IMPORT")

            cbox = layout.box()
            cbox.label(text="Browse Characters, Outfits & Heads", icon="OUTLINER_OB_ARMATURE")
            cbox.prop(props, "char_kind_filter")
            cbox.prop(props, "char_search", icon="VIEWZOOM")
            cbox.prop(props, "selected_character")
            cbox.prop(props, "selected_outfit")
            cbox.prop(props, "selected_head_swap")
            row_opt = cbox.row(align=True)
            row_opt.prop(props, "char_headgear")
            row_opt.prop(props, "char_include_hair")
            cbox.prop(props, "char_hair_lod")
            cbox.prop(props, "pack_textures")
            cbox.operator("hfw_toolkit.import_character_selected", icon="IMPORT")

        else:  # ASSETS
            abox = layout.box()
            abox.label(text="Guerrilla Dev-Directory Asset Catalog", icon="ASSET_MANAGER")
            abox.prop(props, "asset_category")
            abox.prop(props, "asset_search", icon="VIEWZOOM")
            abox.prop(props, "selected_asset")
            abox.prop(props, "lod_index")
            abox.prop(props, "pack_textures")
            abox.operator("hfw_toolkit.import_dev_asset", icon="IMPORT")

        # Live Machine Shader Controls (automatically shown whenever an HFW machine control group exists!)
        has_ctrl = (
            bpy.data.node_groups.get("HFW_MachineShaderControl") is not None
            or bpy.data.node_groups.get("HFW_PlasmaChargeControl") is not None
        )
        if has_ctrl:
            sbox = layout.box()
            hdr = sbox.row()
            hdr.label(text="Live Machine Shader Controls", icon="SHADING_RENDERED")
            hdr.operator("hfw_toolkit.sync_shader_controls", text="", icon="FILE_REFRESH")

            # Muscles
            mbox = sbox.box()
            mbox.label(text="Muscle Fibers & Cables", icon="MOD_WAVE")
            mbox.prop(props, "muscle_preset", text="Preset")
            r_m = mbox.row(align=True)
            r_m.prop(props, "muscle_color", text="")
            r_m.prop(props, "muscle_glow_scale", slider=True)

            # Eyes
            ebox = sbox.box()
            ebox.label(text="Machine Eye Optics", icon="HIDE_OFF")
            ebox.prop(props, "eye_preset", text="Preset")
            r_e = ebox.row(align=True)
            r_e.prop(props, "eye_color", text="Calm")
            r_e.prop(props, "eye_alert_color", text="Alert")
            ebox.prop(props, "eye_alert_state", slider=True)
            ebox.prop(props, "eye_glow_scale", slider=True)

            # Liquids & Elemental Sacs
            lbox = sbox.box()
            lbox.label(text="Elemental Liquids & Sacs", icon="MATFLUID")
            lbox.prop(props, "liquid_glow_scale", slider=True)
            lbox.prop(props, "liquid_override_enable")
            if props.liquid_override_enable:
                lbox.prop(props, "liquid_override_color", text="Custom Liquid")

            # Plasma & Animation
            pbox = sbox.box()
            pbox.label(text="Plasma & Shader Animation", icon="LIGHT_POINT")
            r_p = pbox.row(align=True)
            r_p.prop(props, "plasma_color", text="")
            r_p.prop(props, "plasma_charge", slider=True)
            pbox.prop(props, "animation_speed", slider=True)

        # Live Character Skin & Hair Controls (shown whenever an HFW character is in the scene!)
        if bpy.data.node_groups.get("HFW_CharacterControl") is not None:
            cctrl = layout.box()
            chdr = cctrl.row()
            chdr.label(text="Live Character Skin & Hair Controls", icon="USER")
            chdr.operator("hfw_toolkit.sync_shader_controls", text="", icon="FILE_REFRESH")
            cctrl.prop(props, "skin_tone")
            r_sk = cctrl.row(align=True)
            r_sk.prop(props, "skin_brightness", slider=True)
            r_sk.prop(props, "skin_subsurface", slider=True)
            cctrl.prop(props, "hair_color")

        box2 = layout.box()
        box2.label(text="Material & Shader Explorer", icon="NODE_MATERIAL")
        obj = context.active_object
        if obj and obj.active_material and obj.active_material.get("hfw_material_id"):
            box2.label(text=f"Material: {obj.active_material['hfw_material_id']}")
        box2.operator("hfw_toolkit.inspect_material_web", icon="URL")


_CLASSES = (
    HFWToolkitPreferences,
    HFWToolkitSceneProps,
    HFW_OT_ImportMachine,
    HFW_OT_SyncShaderControls,
    HFW_OT_ImportCharacterPrompt,
    HFW_OT_ImportCharacterSelected,
    HFW_OT_ImportDevAsset,
    HFW_OT_InspectMaterialWeb,
    HFW_PT_MainPanel,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.hfw_toolkit = bpy.props.PointerProperty(type=HFWToolkitSceneProps)


def unregister():
    del bpy.types.Scene.hfw_toolkit
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
