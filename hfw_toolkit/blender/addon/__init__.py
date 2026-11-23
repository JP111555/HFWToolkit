"""HFW Toolkit — Blender 5.3 Add-on (N-Panel Machine Importer, Variant Selector & Shader Controls).

Installable via Blender Preferences -> Add-ons -> Install from Disk (or `hfw-toolkit install-blender-addon`).
Uses only Blender's built-in Python (`bpy`, `sqlite3`, `subprocess`, `webbrowser`).
"""

import importlib.util
import os
import shutil
import sqlite3
import subprocess
import webbrowser
from pathlib import Path

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty
from bpy.types import AddonPreferences, Operator, Panel, PropertyGroup

bl_info = {
    "name": "HFW Toolkit — Machine Importer & Shader Explorer",
    "author": "HFWToolkit",
    "version": (0, 3, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar (N-Panel) > HFW Toolkit",
    "description": "Search and one-click import rigged Horizon Forbidden West machines with live Variant selection, bone-parented armor plates, animated shaders & Web Shader Explorer link.",
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


class HFWToolkitPreferences(AddonPreferences):
    bl_idname = __name__

    toolkit_dir: StringProperty(
        name="HFWToolkit Directory",
        description="Root folder of HFWToolkit (containing .venv and cache/hfw_index.sqlite)",
        subtype="DIR_PATH",
        default=DEFAULT_TOOLKIT_DIR,
    )
    web_url: StringProperty(
        name="Shader Explorer URL",
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


_MACHINE_ITEMS_CACHE: list[tuple[str, str, str]] = []
_VARIANT_ITEMS_CACHE: list[tuple[str, str, str]] = []


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

    items = []
    for gid, oidx, disp, code, pcnt in rows:
        oid = f"{gid}:{oidx}"
        items.append((oid, f"{disp} ({code}) [{oid}]", f"Group {gid}, {pcnt} parts"))
    _MACHINE_ITEMS_CACHE = items
    return _MACHINE_ITEMS_CACHE


def _query_variants(self, context):
    """Populate live Variant dropdown straight from SQLite `entity_variants` (<1ms, zero UI freeze)."""
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

    items = []
    for vidx, label, dr_id, smr_id, pcnt in rows:
        desc = f"Destructibility {dr_id or 'base'} | Body {smr_id or 'default'} ({pcnt} parts)"
        items.append((str(vidx), f"{vidx}: {label}", desc))
    _VARIANT_ITEMS_CACHE = items
    return _VARIANT_ITEMS_CACHE


def _on_entity_changed(self, context):
    """Reset selected variant to 0 whenever switching machines so enum index stays valid."""
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


def _find_plasma_controller_armature(context):
    """Find active (or any scene) HFW armature carrying the `hfw_plasma_charge` shader controller property."""
    obj = context.active_object
    if obj is not None:
        if "hfw_plasma_charge" in obj:
            return obj
        if obj.parent and "hfw_plasma_charge" in obj.parent:
            return obj.parent
    for o in context.scene.objects:
        if "hfw_plasma_charge" in o:
            return o
    return None


def _on_plasma_charge_slider(self, context):
    """Sync N-Panel slider directly onto Slaughterspine's Armature property & HFW_PlasmaChargeControl node group."""
    val = float(self.plasma_charge)
    for o in context.scene.objects:
        if "hfw_plasma_charge" in o:
            o["hfw_plasma_charge"] = val
            o.update_tag()
    ctrl_ng = bpy.data.node_groups.get("HFW_PlasmaChargeControl")
    if ctrl_ng is not None:
        for n in ctrl_ng.nodes:
            if n.type == "VALUE" and n.label == "Plasma Charge":
                n.outputs[0].default_value = val
    for mat in bpy.data.materials:
        if mat.use_nodes and mat.node_tree:
            for n in mat.node_tree.nodes:
                if n.type == "GROUP" and n.node_tree:
                    if n.node_tree.name.startswith("HFW_MachineMuscleFiber") and "Charge State" in n.inputs:
                        if not n.inputs["Charge State"].is_linked:
                            n.inputs["Charge State"].default_value = val
                    elif n.node_tree.name.startswith("HFW_PlasmaCharge") and "Plasma Charge" in n.inputs:
                        if not n.inputs["Plasma Charge"].is_linked:
                            n.inputs["Plasma Charge"].default_value = val


class HFWToolkitSceneProps(PropertyGroup):
    search_query: StringProperty(
        name="Search",
        description="Filter machines by name or Guerrilla codename (e.g. Slitherfang, Clawstrider, Slaughterspine, Redeye Watcher)",
        default="",
        update=_on_entity_changed,
    )
    selected_entity: EnumProperty(
        name="Machine",
        description="Select a machine entity from HFWToolkit's index",
        items=_query_machines,
        update=_on_entity_changed,
    )
    selected_variant: EnumProperty(
        name="Variant",
        description="Select the exact subspecies / Standard / Apex variant to import (e.g. Melee Standard vs Elemental vs Apex)",
        items=_query_variants,
        update=_on_variant_changed,
    )
    variant_index: IntProperty(
        name="Variant Index",
        description="Numeric variant index synced with Variant selector",
        default=0,
        min=0,
        max=20,
    )
    lod_index: IntProperty(
        name="LOD",
        description="0 = Highest detail",
        default=0,
        min=0,
        max=4,
    )
    include_broken_states: BoolProperty(
        name="Include Broken Plate States",
        description="Import damaged/destroyed states into a hidden collection",
        default=True,
    )
    enable_detail_nodes: BoolProperty(
        name="Wire Detail Array Nodes",
        description="Build HFW_MachineDetailArray (Mask-driven 23-slice detail atlas) shader node groups",
        default=True,
    )
    unpack_textures: BoolProperty(
        name="Unpack Textures (Separate Maps)",
        description="Off (default): export native game-packed textures and unpack in shader nodes. On: split into separate Color/Normal/Roughness/AO/Mask PNG files",
        default=False,
    )
    pack_textures: BoolProperty(
        name="Pack Textures Into .blend",
        description="Embed imported PNG textures inside the current .blend file",
        default=False,
    )
    plasma_charge: FloatProperty(
        name="Plasma Charge",
        description="Live Slaughterspine plasma charge-up state (0.0 = Idle/Uncharged, 1.0 = Full Electric Purple Plasma Charge!)",
        default=1.0,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
        update=_on_plasma_charge_slider,
    )


class HFW_OT_ImportMachine(Operator):
    bl_idname = "hfw_toolkit.import_machine"
    bl_label = "Import Machine Into Scene"
    bl_description = "Extract & assemble the selected HFW machine with armature, bone-parented armor plates, and materials"
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
        venv_python = toolkit_dir / ".venv" / "Scripts" / "python.exe"
        if not venv_python.exists():
            sys_py = shutil.which("python")
            if sys_py:
                venv_python = Path(sys_py)
            else:
                self.report({"ERROR"}, f"HFWToolkit venv Python not found: {venv_python}")
                return {"CANCELLED"}

        bundle_dir = toolkit_dir / "exports" / "bundles" / f"entity_{oid.replace(':', '_')}_v{vidx}_lod{props.lod_index}"
        cmd = [
            str(venv_python),
            "-m",
            "hfw_toolkit.cli",
            "export-bundle",
            "--machine",
            oid,
            "--variant",
            str(vidx),
            "--lod",
            str(props.lod_index),
            "--out",
            str(bundle_dir),
        ]
        if not props.include_broken_states:
            cmd.append("--no-broken-states")
        if props.unpack_textures:
            cmd.append("--unpack-textures")

        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            self.report({"ERROR"}, f"Bundle extraction failed: {proc.stderr[-300:] or proc.stdout[-300:]}")
            return {"CANCELLED"}

        builder_path = toolkit_dir / "hfw_toolkit" / "blender" / "blender_builder.py"
        spec = importlib.util.spec_from_file_location("hfw_blender_builder", str(builder_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        arm = mod.import_bundle_into_scene(
            bundle_dir,
            enable_detail_nodes=props.enable_detail_nodes,
            pack_textures=props.pack_textures,
        )
        self.report({"INFO"}, f"Imported {arm.name} cleanly!")
        return {"FINISHED"}


class HFW_OT_InspectMaterialWeb(Operator):
    bl_idname = "hfw_toolkit.inspect_material_web"
    bl_label = "Inspect Active Material in Shader Explorer"
    bl_description = "Open the active object's HFW material in the local Web Material & Shader Deep-Dive Explorer"

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


class HFW_PT_MainPanel(Panel):
    bl_label = "HFW Toolkit"
    bl_idname = "HFW_PT_main_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "HFW Toolkit"

    def draw(self, context):
        layout = self.layout
        props = context.scene.hfw_toolkit

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

        # Live Slaughterspine Plasma Charge control (shown automatically when a Plasma machine is in the scene!)
        plasma_arm = _find_plasma_controller_armature(context)
        if plasma_arm is not None or bpy.data.node_groups.get("HFW_PlasmaChargeControl") is not None:
            pbox = layout.box()
            pbox.label(text="Plasma Charge-Up Control", icon="LIGHT_POINT")
            pbox.prop(props, "plasma_charge", slider=True)

        box2 = layout.box()
        box2.label(text="Material & Shader Explorer", icon="SHADING_RENDERED")
        obj = context.active_object
        if obj and obj.active_material and obj.active_material.get("hfw_material_id"):
            box2.label(text=f"Material: {obj.active_material['hfw_material_id']}")
        box2.operator("hfw_toolkit.inspect_material_web", icon="URL")


_CLASSES = (
    HFWToolkitPreferences,
    HFWToolkitSceneProps,
    HFW_OT_ImportMachine,
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
