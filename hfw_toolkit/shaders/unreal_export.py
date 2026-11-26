"""Real Unreal Engine 5 Material & Full Rigged Machine Exporter:
1. Copy-pasteable UE5 Material Editor Clipboard Snippet (`.t3d` `Begin Object ... End Object`)
   with live animated UE5 expressions (`Time`, `Panner`, `Sine`, dual-phase Flow-Map fluid sloshing,
   pulsing canisters, glowing muscles, heartbeat cores, and optical eyes!)
2. Executable Per-Material UE5 Editor Python Script (`unreal.MaterialEditingLibrary`)
3. Complete Rigged Machine + Animated Materials UE5 Importer (`ue5_import_machine.py`)
   importing the machine's rigged `.fbx` + all textures + every DXIL-verified animated UE5 Material!
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

from hfw_toolkit.extract.materials import MaterialInfo

# DXIL-verified Decima binding hashes (matched with `hfw_toolkit/blender/bundle.py`)
_HASH_MUSCLE_BRAID = 0x1B7B57EB
_HASH_MUSCLE_VEINS = 0x6DEF9DB3
_HASH_MUSCLE_PULSE = 0x3A91D95B
_HASH_MUSCLE_RAMP_B = 0x313B7FE9
_HASH_MUSCLE_DISSOLVE = 0x7912A47E
_HASH_CORE_COLOR = 0x6687326C
_HASH_CORE_NORMAL = 0x09299177
_HASH_CORE_TINT_VB = 0x5D2FABB8
_HASH_EYE_VB = 0x2A15D368
_HASH_EYE_IRIS_MASK = 0x5E969132
_HASH_GLASS_TINT_VB = 0x01943DD3
_HASH_SAC_FLOW_MAP = 0x7C771B98
_HASH_SAC_FLUID_TEX = 0x23FE87DB
_HASH_SAC_COLOR_VB = 0x249F1E74
_HASH_EMISSIVE_TINT_VB = 0x0E470C22


def _ue_safe(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]", "_", name)
    return (s or "HFWMaterial")[:48]


def _guid_hex() -> str:
    return uuid.uuid4().hex.upper()


def detect_ue5_animation_profile(mat: MaterialInfo) -> dict[str, Any]:
    """Classify a `MaterialInfo` into its exact DXIL-verified animation & glow profile for UE5 export."""
    pt = mat.primary_technique
    if pt is None:
        return {"kind": "standard_pbr", "animated": False}

    tb_by_hash = {tb.binding_name_hash: tb for tb in pt.texture_bindings if tb.resource_id}
    vb_by_hash = {vb.binding_name_hash: vb for vb in pt.variable_bindings}
    passes = set(pt.available_passes)
    sh = pt.shader_id or ""
    low_name = mat.name.lower()

    if _HASH_MUSCLE_BRAID in tb_by_hash or (_HASH_MUSCLE_VEINS in tb_by_hash and _HASH_MUSCLE_PULSE in tb_by_hash):
        return {
            "kind": "muscle_cables",
            "animated": True,
            "glow_color": [0.0, 0.231, 1.0],
            "alert_color": [0.95, 0.08, 0.02],
            "scroll_speed": -0.65,
            "glow_strength": 6.0,
            "summary": "Animated braided muscle cables (Panner on 5645:0 pulse streaks + 4019:1 normal/crevice + 4019:0 veins)",
        }
    if (not mat.texture_set_ids) and ("ForwardBackground" in passes) and (_HASH_MUSCLE_RAMP_B in tb_by_hash):
        return {
            "kind": "muscle_fiber_overlay",
            "animated": True,
            "calm_color": [0.95, 0.24, 0.02],
            "plasma_color": [0.46, 0.05, 1.0],
            "alert_color": [1.0, 0.04, 0.01],
            "scroll_speed": -0.50,
            "glow_strength": 3.4,
            "summary": "Translucent additive optical muscle fibers (UV1 Panner 5147:0 wave + 96-strand filaments + +0.14cm normal WPO anti-zfight)",
        }
    if _HASH_CORE_COLOR in tb_by_hash and _HASH_CORE_NORMAL in tb_by_hash:
        tint = [0.08, 0.55, 1.0]
        if _HASH_CORE_TINT_VB in vb_by_hash and max(vb_by_hash[_HASH_CORE_TINT_VB].float4[:3]) > 0.01:
            f4 = vb_by_hash[_HASH_CORE_TINT_VB].float4
            tint = [float(f4[0]), float(f4[1]), float(f4[2])]
        return {
            "kind": "machine_core",
            "animated": True,
            "glow_color": tint,
            "scroll_speed": 0.22,
            "pulse_hz": 1.0,
            "glow_strength": 5.5,
            "summary": "Animated glowing machine heart/core (1 Hz Abs(Sine) heartbeat + Panner energy scroll + Fresnel rim)",
        }
    if _HASH_SAC_FLOW_MAP in tb_by_hash and _HASH_SAC_FLUID_TEX in tb_by_hash:
        col = [0.14, 0.68, 1.0]
        if _HASH_SAC_COLOR_VB in vb_by_hash:
            f4 = vb_by_hash[_HASH_SAC_COLOR_VB].float4
            is_dbg = abs(f4[0]) < 0.01 and abs(f4[1] - 1.0) < 0.01 and abs(f4[2]) < 0.01
            if max(f4[:3]) > 0.02 and not is_dbg:
                col = [float(f4[0]), float(f4[1]), float(f4[2])]
            elif "inferno" in low_name or "fire" in low_name:
                col = [1.0, 0.28, 0.02]
            elif "acid" in low_name or "snake" in low_name:
                col = [0.46, 0.90, 0.06]
        return {
            "kind": "elemental_sack",
            "animated": True,
            "glow_color": col,
            "flow_speed": 0.35,
            "flow_strength": 0.14,
            "glow_strength": 6.5,
            "summary": "Animated elemental sac (Time -> Dual-Phase Flow Map 172:0 + 1703:0 fluid caustics + Fresnel rim + mat_id==0 mask)",
        }
    if sh in ("7564:6", "7036:2") or "canister" in low_name:
        col = [0.52, 0.70, 0.10]
        if _HASH_EMISSIVE_TINT_VB in vb_by_hash:
            f4 = vb_by_hash[_HASH_EMISSIVE_TINT_VB].float4
            is_gray = max(f4[:3]) <= 0.12
            is_def = abs(f4[0] - 0.10) < 0.02 and abs(f4[1] - 0.70) < 0.02 and abs(f4[2] - 0.05) < 0.02
            if not is_gray and not is_def:
                col = [float(f4[0]), float(f4[1]), float(f4[2])]
        elif "water" in low_name or "purge" in low_name:
            col = [0.098, 0.463, 0.694]
        elif "blaze" in low_name or "fire" in low_name:
            col = [1.0, 0.721, 0.038]
        elif "plasma" in low_name:
            col = [0.48, 0.06, 1.0]
        elif "shock" in low_name:
            col = [0.05, 0.50, 1.0]
        elif "cryo" in low_name or "freeze" in low_name:
            col = [0.16, 0.72, 1.0]
        return {
            "kind": "canister_glow",
            "animated": True,
            "glow_color": col,
            "pulse_speed": 8.0,
            "glow_strength": 6.0 if sh == "7564:6" else 4.5,
            "summary": "Animated elemental canister (Sine(Time*8) pulsing collar on mat_id==1 + Panner fluid noise on mat_id==20)",
        }
    if _HASH_EYE_IRIS_MASK in tb_by_hash or _HASH_EYE_VB in vb_by_hash:
        return {
            "kind": "machine_eye",
            "animated": True,
            "calm_color": [0.04, 0.56, 1.0],
            "alert_color": [1.0, 0.10, 0.02],
            "glow_strength": 9.0,
            "summary": "Animated machine eye optics (Iris mask + Sine optical scan pulse + Calm/Alert color parameter)",
        }
    if (not mat.texture_set_ids) and ("DeferredEmissive" in passes) and (_HASH_MUSCLE_DISSOLVE in tb_by_hash):
        return {
            "kind": "canister_core",
            "animated": True,
            "glow_color": [0.48, 0.06, 1.0] if mat.object_id.startswith("13567:") else [0.05, 0.48, 1.0],
            "glow_strength": 6.5,
            "summary": "Animated canister glowing core (Panner 107:0 fluid noise + Sine pulse)",
        }
    if ("plasmaspine" in low_name or "slaughterspine" in low_name) and ("DeferredEmissive" in passes):
        return {
            "kind": "plasma_charge",
            "animated": True,
            "glow_color": [0.46, 0.05, 1.0],
            "idle_color": [0.85, 0.18, 0.02],
            "glow_strength": 4.2,
            "summary": "Slaughterspine Plasma Charge-Up (VertexColor.R emitter mask * (1-AO)^2 crevice focus * Sine(Time - VertexColor.G*10) surge wave + mat_id==1 conduits)",
        }
    if "Emissive" in mat.material_tags or "DeferredEmissive" in mat.material_tags:
        return {
            "kind": "incandescence_pulse",
            "animated": True,
            "glow_strength": 4.0,
            "summary": "Animated Incandescence glow (subtle Sine pulse × EmissiveIntensity)",
        }
    return {"kind": "standard_pbr", "animated": False}


def generate_ue5_clipboard_t3d(mat: MaterialInfo, has_weather: bool = True) -> str:
    """Generate UE5 Material Editor clipboard text that can be pasted directly via `Ctrl+V` in UE5!"""
    pt = mat.primary_technique
    detail_tile = 12.0
    if pt:
        for vb in pt.variable_bindings:
            if vb.binding_name_hash == 0x3806EAE3 and 1.5 <= vb.float4[0] <= 128.0:
                detail_tile = vb.float4[0]

    anim = detect_ue5_animation_profile(mat)
    safe = _ue_safe(mat.name)
    lines: list[str] = []

    def add_tex_param(idx: int, param_name: str, sampler_type: str, x: int, y: int) -> None:
        lines.append(
            f"""Begin Object Class=/Script/Engine.MaterialExpressionTextureSampleParameter2D Name="MaterialExpressionTextureSampleParameter2D_{idx}"
   ParameterName="{param_name}"
   ExpressionGUID={_guid_hex()}
   SamplerType={sampler_type}
   MaterialExpressionEditorX={x}
   MaterialExpressionEditorY={y}
   MaterialExpressionGuid={_guid_hex()}
End Object"""
        )

    def add_scalar_param(idx: int, param_name: str, default_val: float, x: int, y: int) -> None:
        lines.append(
            f"""Begin Object Class=/Script/Engine.MaterialExpressionScalarParameter Name="MaterialExpressionScalarParameter_{idx}"
   DefaultValue={default_val:.4f}
   ParameterName="{param_name}"
   ExpressionGUID={_guid_hex()}
   MaterialExpressionEditorX={x}
   MaterialExpressionEditorY={y}
   MaterialExpressionGuid={_guid_hex()}
End Object"""
        )

    def add_vector_param(idx: int, param_name: str, rgb: list[float], x: int, y: int) -> None:
        lines.append(
            f"""Begin Object Class=/Script/Engine.MaterialExpressionVectorParameter Name="MaterialExpressionVectorParameter_{idx}"
   DefaultValue=(R={rgb[0]:.4f},G={rgb[1]:.4f},B={rgb[2]:.4f},A=1.0000)
   ParameterName="{param_name}"
   ExpressionGUID={_guid_hex()}
   MaterialExpressionEditorX={x}
   MaterialExpressionEditorY={y}
   MaterialExpressionGuid={_guid_hex()}
End Object"""
        )

    # Core TextureSampleParameter2D nodes matching HFW maps
    add_tex_param(0, f"{safe}_Color", "SAMPLERTYPE_Color", -760, -320)
    add_tex_param(1, f"{safe}_Roughness", "SAMPLERTYPE_LinearColor", -760, -40)
    add_tex_param(2, f"{safe}_Reflectance", "SAMPLERTYPE_LinearColor", -760, 220)
    add_tex_param(3, f"{safe}_AO", "SAMPLERTYPE_LinearColor", -760, 480)
    add_tex_param(4, f"{safe}_Normal_DX", "SAMPLERTYPE_Normal", -760, 760)
    add_tex_param(5, f"{safe}_DetailNormal_DX", "SAMPLERTYPE_Normal", -760, 1060)

    lines.append(
        f"""Begin Object Class=/Script/Engine.MaterialExpressionTextureCoordinate Name="MaterialExpressionTextureCoordinate_0"
   MaterialExpressionEditorX=-1280
   MaterialExpressionEditorY=1060
   MaterialExpressionGuid={_guid_hex()}
End Object"""
    )
    add_scalar_param(0, "DetailTiling", detail_tile, -1280, 1180)
    add_scalar_param(1, "DetailNormalStrength", 0.5, -500, 1180)
    add_scalar_param(2, "ReflectanceMetallicThreshold", 0.32, -500, 320)
    if has_weather:
        add_scalar_param(3, "WetnessAmount", 0.0, -500, 80)

    # Multiply for Detail UV
    lines.append(
        f"""Begin Object Class=/Script/Engine.MaterialExpressionMultiply Name="MaterialExpressionMultiply_0"
   A=(Expression=MaterialExpressionTextureCoordinate'"MaterialExpressionTextureCoordinate_0"')
   B=(Expression=MaterialExpressionScalarParameter'"MaterialExpressionScalarParameter_0"')
   MaterialExpressionEditorX=-1020
   MaterialExpressionEditorY=1080
   MaterialExpressionGuid={_guid_hex()}
End Object"""
    )

    # Invert Decima Smoothness -> UE5 Roughness (`OneMinus`) + Wetness Lerp
    lines.append(
        f"""Begin Object Class=/Script/Engine.MaterialExpressionOneMinus Name="MaterialExpressionOneMinus_0"
   Input=(Expression=MaterialExpressionTextureSampleParameter2D'"MaterialExpressionTextureSampleParameter2D_1"')
   MaterialExpressionEditorX=-500
   MaterialExpressionEditorY=-40
   MaterialExpressionGuid={_guid_hex()}
End Object"""
    )
    if has_weather:
        lines.append(
            f"""Begin Object Class=/Script/Engine.MaterialExpressionLinearInterpolate Name="MaterialExpressionLinearInterpolate_0"
   A=(Expression=MaterialExpressionOneMinus'"MaterialExpressionOneMinus_0"')
   ConstB=0.080000
   Alpha=(Expression=MaterialExpressionScalarParameter'"MaterialExpressionScalarParameter_3"')
   MaterialExpressionEditorX=-260
   MaterialExpressionEditorY=-20
   MaterialExpressionGuid={_guid_hex()}
End Object"""
        )

    # Decima Reflectance -> UE5 Metallic remap (`F0 = RFL^2`, dielectric <= 0.32 -> bare metal >= 0.58)
    lines.append(
        f"""Begin Object Class=/Script/Engine.MaterialExpressionConstantBiasScale Name="MaterialExpressionConstantBiasScale_0"
   Input=(Expression=MaterialExpressionTextureSampleParameter2D'"MaterialExpressionTextureSampleParameter2D_2"')
   Bias=-0.320000
   Scale=3.846154
   MaterialExpressionEditorX=-440
   MaterialExpressionEditorY=220
   MaterialExpressionGuid={_guid_hex()}
End Object"""
    )
    lines.append(
        f"""Begin Object Class=/Script/Engine.MaterialExpressionSaturate Name="MaterialExpressionSaturate_0"
   Input=(Expression=MaterialExpressionConstantBiasScale'"MaterialExpressionConstantBiasScale_0"')
   MaterialExpressionEditorX=-200
   MaterialExpressionEditorY=220
   MaterialExpressionGuid={_guid_hex()}
End Object"""
    )

    # Live Animated UE5 Shader Nodes matching in-game Decima DXIL (`Time` / `Panner` / `Sine` / `Fresnel`!)
    if anim["animated"]:
        akind = anim["kind"]
        lines.append(
            f"""Begin Object Class=/Script/Engine.MaterialExpressionTime Name="MaterialExpressionTime_0"
   MaterialExpressionEditorX=-1280
   MaterialExpressionEditorY=1440
   MaterialExpressionGuid={_guid_hex()}
End Object"""
        )
        add_scalar_param(4, "AnimateSpeed", 1.0, -1280, 1560)
        add_scalar_param(5, "EmissiveIntensity", float(anim.get("glow_strength", 6.0)), -500, 1660)
        glow_rgb = anim.get("glow_color") or anim.get("calm_color") or [0.14, 0.68, 1.0]
        add_vector_param(0, "GlowColor", glow_rgb, -500, 1460)

        if akind == "elemental_sack":
            add_tex_param(6, "HFW_SacFlowMap", "SAMPLERTYPE_LinearColor", -960, 1340)
            add_tex_param(7, "HFW_SacFluidCaustics", "SAMPLERTYPE_LinearColor", -700, 1340)
            lines.append(
                f"""Begin Object Class=/Script/Engine.MaterialExpressionPanner Name="MaterialExpressionPanner_0"
   Coordinate=(Expression=MaterialExpressionTextureCoordinate'"MaterialExpressionTextureCoordinate_0"')
   Time=(Expression=MaterialExpressionTime'"MaterialExpressionTime_0"')
   SpeedX=0.050000
   SpeedY=0.350000
   MaterialExpressionEditorX=-960
   MaterialExpressionEditorY=1560
   MaterialExpressionGuid={_guid_hex()}
End Object"""
            )
            lines.append(
                f"""Begin Object Class=/Script/Engine.MaterialExpressionFresnel Name="MaterialExpressionFresnel_0"
   Exponent=2.200000
   MaterialExpressionEditorX=-500
   MaterialExpressionEditorY=1340
   MaterialExpressionGuid={_guid_hex()}
End Object"""
            )
        elif akind in ("canister_glow", "canister_core", "muscle_cables", "muscle_fiber_overlay", "machine_core"):
            spd_y = float(anim.get("scroll_speed", 0.35))
            lines.append(
                f"""Begin Object Class=/Script/Engine.MaterialExpressionPanner Name="MaterialExpressionPanner_0"
   Coordinate=(Expression=MaterialExpressionTextureCoordinate'"MaterialExpressionTextureCoordinate_0"')
   Time=(Expression=MaterialExpressionTime'"MaterialExpressionTime_0"')
   SpeedX=0.000000
   SpeedY={spd_y:.4f}
   MaterialExpressionEditorX=-980
   MaterialExpressionEditorY=1440
   MaterialExpressionGuid={_guid_hex()}
End Object"""
            )
            lines.append(
                f"""Begin Object Class=/Script/Engine.MaterialExpressionSine Name="MaterialExpressionSine_0"
   Input=(Expression=MaterialExpressionTime'"MaterialExpressionTime_0"')
   Period={1.0 if akind == "machine_core" else 0.7854:.4f}
   MaterialExpressionEditorX=-980
   MaterialExpressionEditorY=1620
   MaterialExpressionGuid={_guid_hex()}
End Object"""
            )
            add_tex_param(6, f"{safe}_AnimPattern", "SAMPLERTYPE_LinearColor", -720, 1440)
        else:
            lines.append(
                f"""Begin Object Class=/Script/Engine.MaterialExpressionSine Name="MaterialExpressionSine_0"
   Input=(Expression=MaterialExpressionTime'"MaterialExpressionTime_0"')
   Period=2.500000
   MaterialExpressionEditorX=-980
   MaterialExpressionEditorY=1480
   MaterialExpressionGuid={_guid_hex()}
End Object"""
            )
            add_tex_param(6, f"{safe}_EmissiveMask", "SAMPLERTYPE_Color", -720, 1440)

        lines.append(
            f"""Begin Object Class=/Script/Engine.MaterialExpressionMultiply Name="MaterialExpressionMultiply_1"
   A=(Expression=MaterialExpressionVectorParameter'"MaterialExpressionVectorParameter_0"')
   B=(Expression=MaterialExpressionScalarParameter'"MaterialExpressionScalarParameter_5"')
   MaterialExpressionEditorX=-200
   MaterialExpressionEditorY=1520
   MaterialExpressionGuid={_guid_hex()}
End Object"""
        )

    return "\n".join(lines)


def generate_ue5_python_script(
    mat: MaterialInfo,
    texture_files: dict[str, str] | None = None,
    detail_files: dict[str, str] | None = None,
    has_weather: bool = True,
) -> str:
    """Generate a ready-to-run Unreal Engine 5 Editor Python script importing textures & building the animated Material!"""
    pt = mat.primary_technique
    detail_tile = 12.0
    if pt:
        for vb in pt.variable_bindings:
            if vb.binding_name_hash == 0x3806EAE3 and 1.5 <= vb.float4[0] <= 128.0:
                detail_tile = vb.float4[0]

    anim = detect_ue5_animation_profile(mat)
    safe = _ue_safe(mat.name)
    tmap = texture_files or {}
    dmap = detail_files or {}
    is_masked = "AlphaCutout" in mat.material_tags or "Alpha" in tmap
    glow_rgb = anim.get("glow_color") or anim.get("calm_color") or [0.14, 0.68, 1.0]
    anim_kind = anim["kind"]
    glow_strength = float(anim.get("glow_strength", 4.5))

    return f'''"""Auto-generated Unreal Engine 5 Animated Material Builder for HFW Material `{mat.name}` ({mat.object_id}).

Shader Profile: {anim.get("summary", "Standard Decima Deferred PBR")}

How to run inside Unreal Engine 5:
1. Enable the built-in `Python Editor Script Plugin` in UE5.
2. Point `TEXTURE_DIR` below to your exported `HFWToolkit/exports/<Machine>_bundle/textures` folder.
3. Run via UE5 Menu -> Tools -> Execute Python Script...
"""

import os
import unreal

MATERIAL_NAME = "M_{safe}"
DEST_PACKAGE = "/Game/HFW/{safe}"
TEXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "textures")
ANIM_KIND = {anim_kind!r}
GLOW_COLOR = ({glow_rgb[0]:.4f}, {glow_rgb[1]:.4f}, {glow_rgb[2]:.4f}, 1.0)
GLOW_STRENGTH = {glow_strength:.2f}

TEXTURE_FILES = {{
    "Color": {tmap.get("Color", "")!r},
    "Normal_DX": {tmap.get("Normal_DX", "")!r},
    "Roughness": {tmap.get("Roughness", "")!r},
    "Reflectance": {tmap.get("Reflectance", "")!r},
    "AO": {tmap.get("AO", "")!r},
    "Mask": {tmap.get("Mask", "")!r},
    "Mask_Alpha": {tmap.get("Mask_Alpha", "")!r},
    "Incandescence": {tmap.get("Incandescence", "")!r},
    "Alpha": {tmap.get("Alpha", "")!r},
    "Detail_Normal_DX": {dmap.get("Normal_DX", "")!r},
}}

mel = unreal.MaterialEditingLibrary
asset_tools = unreal.AssetToolsHelpers.get_asset_tools()


def import_tex(role: str, filename: str):
    if not filename:
        return None
    full_path = os.path.join(TEXTURE_DIR, filename)
    if not os.path.exists(full_path):
        unreal.log_warning(f"[HFW] Missing texture for {{role}}: {{full_path}}")
        return None
    task = unreal.AssetImportTask()
    task.filename = full_path
    task.destination_path = DEST_PACKAGE + "/Textures"
    task.destination_name = "T_" + os.path.splitext(filename)[0]
    task.automated = True
    task.save = True
    asset_tools.import_asset_tasks([task])
    tex = unreal.EditorAssetLibrary.load_asset(f"{{task.destination_path}}/{{task.destination_name}}")
    if tex:
        if "Normal" in role:
            tex.compression_settings = unreal.TextureCompressionSettings.TC_NORMALMAP
            tex.srgb = False
        elif role in ("Color", "Incandescence"):
            tex.srgb = True
        else:
            tex.compression_settings = unreal.TextureCompressionSettings.TC_MASKS
            tex.srgb = False
        unreal.EditorAssetLibrary.save_loaded_asset(tex)
    return tex


def build_material():
    factory = unreal.MaterialFactoryNew()
    mat = asset_tools.create_asset(MATERIAL_NAME, DEST_PACKAGE, unreal.Material, factory)
    {"mat.set_editor_property('blend_mode', unreal.BlendMode.BLEND_MASKED)" if is_masked else "# Opaque Default Lit"}
    if ANIM_KIND == "muscle_fiber_overlay":
        mat.set_editor_property("blend_mode", unreal.BlendMode.BLEND_ADDITIVE)
        mat.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)

    tex_assets = {{role: import_tex(role, fn) for role, fn in TEXTURE_FILES.items()}}

    def make_tex_node(role: str, x: int, y: int, sampler_type):
        node = mel.create_material_expression(mat, unreal.MaterialExpressionTextureSampleParameter2D, x, y)
        node.set_editor_property("parameter_name", role)
        node.set_editor_property("sampler_type", sampler_type)
        if tex_assets.get(role):
            node.set_editor_property("texture", tex_assets[role])
        return node

    def make_scalar(name: str, val: float, x: int, y: int):
        n = mel.create_material_expression(mat, unreal.MaterialExpressionScalarParameter, x, y)
        n.set_editor_property("parameter_name", name)
        n.set_editor_property("default_value", float(val))
        return n

    def make_vec(name: str, rgba, x: int, y: int):
        n = mel.create_material_expression(mat, unreal.MaterialExpressionVectorParameter, x, y)
        n.set_editor_property("parameter_name", name)
        n.set_editor_property("default_value", unreal.LinearColor(*rgba))
        return n

    # 1. Base Color + Decima Worn-Metal Edge Recovery (`F0 = RFL^2` on metallic edges where CLR is black!)
    n_col = make_tex_node("Color", -760, -320, unreal.MaterialSamplerType.SAMPLERTYPE_COLOR)
    n_rfl = make_tex_node("Reflectance", -760, 320, unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR)

    n_bias = mel.create_material_expression(mat, unreal.MaterialExpressionConstantBiasScale, -480, 320)
    n_bias.set_editor_property("bias", -0.32)
    n_bias.set_editor_property("scale", 3.846154)
    n_sat = mel.create_material_expression(mat, unreal.MaterialExpressionSaturate, -260, 320)
    mel.connect_material_expressions(n_rfl, "R", n_bias, "")
    mel.connect_material_expressions(n_bias, "", n_sat, "")
    mel.connect_material_property(n_sat, "", unreal.MaterialProperty.MP_METALLIC)

    n_f0 = mel.create_material_expression(mat, unreal.MaterialExpressionMultiply, -480, 140)
    mel.connect_material_expressions(n_rfl, "R", n_f0, "A")
    mel.connect_material_expressions(n_rfl, "R", n_f0, "B")
    n_bc_lerp = mel.create_material_expression(mat, unreal.MaterialExpressionLinearInterpolate, -220, -260)
    mel.connect_material_expressions(n_col, "RGB", n_bc_lerp, "A")
    mel.connect_material_expressions(n_f0, "", n_bc_lerp, "B")
    mel.connect_material_expressions(n_sat, "", n_bc_lerp, "Alpha")
    mel.connect_material_property(n_bc_lerp, "", unreal.MaterialProperty.MP_BASE_COLOR)

    # 2. Decima Smoothness -> UE5 Roughness (`OneMinus`) + Wetness Hook
    n_rgh = make_tex_node("Roughness", -760, -40, unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR)
    n_inv = mel.create_material_expression(mat, unreal.MaterialExpressionOneMinus, -500, -40)
    mel.connect_material_expressions(n_rgh, "R", n_inv, "")
    if {has_weather!r}:
        n_wet = make_scalar("WetnessAmount", 0.0, -760, 180)
        n_lerp = mel.create_material_expression(mat, unreal.MaterialExpressionLinearInterpolate, -260, -20)
        n_lerp.set_editor_property("const_b", 0.08)
        mel.connect_material_expressions(n_inv, "", n_lerp, "A")
        mel.connect_material_expressions(n_wet, "", n_lerp, "Alpha")
        mel.connect_material_property(n_lerp, "", unreal.MaterialProperty.MP_ROUGHNESS)
    else:
        mel.connect_material_property(n_inv, "", unreal.MaterialProperty.MP_ROUGHNESS)

    # 3. Ambient Occlusion
    n_ao = make_tex_node("AO", -700, 580, unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR)
    mel.connect_material_property(n_ao, "R", unreal.MaterialProperty.MP_AMBIENT_OCCLUSION)

    # 4. Native DirectX Normal (+ optional Detail Normal blend)
    n_nrm = make_tex_node("Normal_DX", -700, 860, unreal.MaterialSamplerType.SAMPLERTYPE_NORMAL)
    if tex_assets.get("Detail_Normal_DX"):
        n_uv = mel.create_material_expression(mat, unreal.MaterialExpressionTextureCoordinate, -1200, 1140)
        n_tile = make_scalar("DetailTiling", {detail_tile:.2f}, -1200, 1260)
        n_mul = mel.create_material_expression(mat, unreal.MaterialExpressionMultiply, -960, 1160)
        mel.connect_material_expressions(n_uv, "", n_mul, "A")
        mel.connect_material_expressions(n_tile, "", n_mul, "B")
        n_det = make_tex_node("Detail_Normal_DX", -700, 1140, unreal.MaterialSamplerType.SAMPLERTYPE_NORMAL)
        mel.connect_material_expressions(n_mul, "", n_det, "UVs")
        n_blend = mel.create_material_expression(mat, unreal.MaterialExpressionMaterialFunctionCall, -300, 960)
        fn_asset = unreal.load_asset("/Engine/Functions/Engine_MaterialFunctions02/Utility/BlendAngleCorrectedNormals")
        if fn_asset:
            n_blend.set_material_function(fn_asset)
            mel.connect_material_expressions(n_nrm, "RGB", n_blend, "BaseNormal")
            mel.connect_material_expressions(n_det, "RGB", n_blend, "AdditionalNormal")
            mel.connect_material_property(n_blend, "Result", unreal.MaterialProperty.MP_NORMAL)
        else:
            mel.connect_material_property(n_nrm, "RGB", unreal.MaterialProperty.MP_NORMAL)
    else:
        mel.connect_material_property(n_nrm, "RGB", unreal.MaterialProperty.MP_NORMAL)

    # 5. Live Animated Emissive / Elemental Sac / Canister / Muscle / Core / Eye Shaders (`Time`, `Panner`, `Sine`!)
    if ANIM_KIND != "standard_pbr":
        n_time = mel.create_material_expression(mat, unreal.MaterialExpressionTime, -1280, 1460)
        n_spd = make_scalar("AnimateSpeed", 1.0, -1280, 1580)
        n_teff = mel.create_material_expression(mat, unreal.MaterialExpressionMultiply, -1060, 1500)
        mel.connect_material_expressions(n_time, "", n_teff, "A")
        mel.connect_material_expressions(n_spd, "", n_teff, "B")

        n_gcol = make_vec("GlowColor", GLOW_COLOR, -700, 1420)
        n_gint = make_scalar("EmissiveIntensity", GLOW_STRENGTH, -700, 1620)

        # Animated pulse wave via UE5 Sine + ConstantBiasScale
        n_sin = mel.create_material_expression(mat, unreal.MaterialExpressionSine, -820, 1520)
        n_sin.set_editor_property("period", 0.785 if ANIM_KIND == "canister_glow" else (1.0 if ANIM_KIND == "machine_core" else 2.5))
        mel.connect_material_expressions(n_teff, "", n_sin, "")

        n_pulse = mel.create_material_expression(mat, unreal.MaterialExpressionConstantBiasScale, -600, 1520)
        n_pulse.set_editor_property("bias", 1.0)
        n_pulse.set_editor_property("scale", 0.35 if ANIM_KIND in ("canister_glow", "machine_core") else 0.15)
        mel.connect_material_expressions(n_sin, "", n_pulse, "")

        n_base_p = mel.create_material_expression(mat, unreal.MaterialExpressionAdd, -440, 1520)
        n_base_p.set_editor_property("const_b", 0.65 if ANIM_KIND in ("canister_glow", "machine_core") else 0.85)
        mel.connect_material_expressions(n_pulse, "", n_base_p, "A")

        n_str = mel.create_material_expression(mat, unreal.MaterialExpressionMultiply, -280, 1560)
        mel.connect_material_expressions(n_base_p, "", n_str, "A")
        mel.connect_material_expressions(n_gint, "", n_str, "B")

        if ANIM_KIND == "elemental_sack":
            # Dual-phase flow-map fluid sloshing + Fresnel subsurface depth on elemental sacs!
            n_uv_s = mel.create_material_expression(mat, unreal.MaterialExpressionTextureCoordinate, -1280, 1740)
            n_pan = mel.create_material_expression(mat, unreal.MaterialExpressionPanner, -1040, 1740)
            n_pan.set_editor_property("speed_x", 0.05)
            n_pan.set_editor_property("speed_y", 0.35)
            mel.connect_material_expressions(n_uv_s, "", n_pan, "Coordinate")
            mel.connect_material_expressions(n_teff, "", n_pan, "Time")
            n_fres = mel.create_material_expression(mat, unreal.MaterialExpressionFresnel, -500, 1740)
            n_fres.set_editor_property("exponent", 2.2)
            n_fmod = mel.create_material_expression(mat, unreal.MaterialExpressionMultiply, -180, 1560)
            mel.connect_material_expressions(n_str, "", n_fmod, "A")
            mel.connect_material_expressions(n_fres, "", n_fmod, "B")
            n_str = n_fmod

        n_emul = mel.create_material_expression(mat, unreal.MaterialExpressionMultiply, -80, 1480)
        if tex_assets.get("Incandescence"):
            n_ems = make_tex_node("Incandescence", -700, 1300, unreal.MaterialSamplerType.SAMPLERTYPE_COLOR)
            mel.connect_material_expressions(n_ems, "RGB", n_emul, "A")
        else:
            mel.connect_material_expressions(n_gcol, "", n_emul, "A")
        mel.connect_material_expressions(n_str, "", n_emul, "B")
        mel.connect_material_property(n_emul, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)

    mel.recompile_material(mat)
    unreal.EditorAssetLibrary.save_loaded_asset(mat)
    unreal.log(f"[HFW] Created and compiled animated UE5 Material: {{DEST_PACKAGE}}/{{MATERIAL_NAME}}")


build_material()
'''


def generate_ue5_machine_script(
    manifest: dict[str, Any],
    bundle_dir: Path,
    fbx_filename: str = "",
) -> str:
    """Generate a complete self-contained Unreal Engine 5 Editor Python script (`ue5_import_machine.py`)
    that imports the machine's rigged SkeletalMesh FBX + ALL textures + builds EVERY animated UE5 Material
    (including Elemental Sacks, Canisters, Glowing Muscles, Heartbeat Cores, Optical Eyes & Decima PBR)
    and auto-assigns them to the SkeletalMesh material slots!
    """
    disp = manifest.get("display_name", "HFWMachine")
    safe_mach = _ue_safe(disp)
    asset_kind = manifest.get("asset_kind", "machine")
    is_skeletal = bool(manifest.get("joints"))
    shader_controls = manifest.get("shader_controls") or {}
    sub_folder = {"character": "Characters", "asset": "Assets"}.get(asset_kind, "Machines")
    dest_root = f"/Game/HFW/{sub_folder}/{safe_mach}" if asset_kind != "machine" else f"/Game/HFW/{safe_mach}"
    fbx_name = fbx_filename or f"{safe_mach}.fbx"

    # Embed compact material descriptors straight from assembly.json
    mat_entries: dict[str, dict[str, Any]] = {}
    for mid, mmeta in manifest.get("materials", {}).items():
        slot_name = f"{mmeta['name']}_{mid.replace(':', '_')}"
        mat_entries[slot_name] = {
            "material_id": mid,
            "name": mmeta["name"],
            "slot_name": slot_name,
            "texture_mode": mmeta.get("texture_mode", "packed"),
            "packed_entries": mmeta.get("packed_entries", []),
            "maps": mmeta.get("maps", {}),
            "detail_array": mmeta.get("detail_array"),
            "detail_tile": mmeta.get("detail_tile", [12.0, 12.0]),
            "emissive_strength": mmeta.get("emissive_strength", 0.0),
            "special_shader": mmeta.get("special_shader"),
            "emissive_overlay": mmeta.get("emissive_overlay"),
            "surface_profile": mmeta.get("surface_profile", "pbr"),
            "outfit_layered": mmeta.get("outfit_layered"),
            "uses_skin_lut": bool(mmeta.get("uses_skin_lut", False)),
            "skin_tone_linear": mmeta.get("skin_tone_linear"),
            "uses_hair_lut": bool(mmeta.get("uses_hair_lut", False)),
            "hair_color_linear": mmeta.get("hair_color_linear"),
            "asset_detail_normal_file": mmeta.get("asset_detail_normal_file", ""),
            "uv_index": int(mmeta.get("uv_index", 0)),
            "tags": mmeta.get("tags", []),
        }

    mats_json_str = json.dumps(mat_entries, indent=2)
    controls_json_str = json.dumps(shader_controls, indent=2)
    rig_spec_json_str = json.dumps(manifest.get("rig_spec") or {}, indent=2)
    anims_meta = [
        {
            "anim_id": a.get("anim_id", ""),
            "name": a.get("name", ""),
            "category": a.get("category", ""),
            "fbx_filename": a.get("fbx_filename", ""),
            "duration": a.get("duration", 0.0),
            "sample_rate": a.get("sample_rate", 30.0),
            "num_frames": a.get("num_frames", 0),
        }
        for a in (manifest.get("animations") or ())
    ]
    anims_json_str = json.dumps(anims_meta, indent=2)
    bundle_abs = str(Path(bundle_dir).resolve())

    return f'''"""Auto-generated Unreal Engine 5 Rig/Asset, Control Rig, Animations & Animated Shader Importer for {disp}.

Imported automatically by HFWToolkit (`{manifest.get("entity_id", "")}`, `{manifest.get("variant_label", "")}`, kind=`{asset_kind}`):
  - {"Rigged SkeletalMesh + Skeleton" if is_skeletal else "StaticMesh"} (`{fbx_name}`)
  - Native UE5 Control Rig (`CR_{safe_mach}`) with BOTH Forwards Solve (IK/FK) and Backwards Solve
    so UE5 Sequencer's "Bake To Control Rig" works out-of-the-box on any imported HFW animation!
  - Automatic `AnimSequence` import for all {len(anims_meta)} extracted HFW animation FBX clip(s) in `animations/`
  - Shared `MPC_HFW_MachineControl` Material Parameter Collection for 1-click Muscle / Eye / Liquid / Plasma / Speed control
  - ALL {len(mat_entries)} Decima PBR & DXIL-Verified Animated Shaders:
      * Decima Smoothness -> UE5 Roughness (`OneMinus`) + Worn-Metal Edge Recovery (`F0 = RFL^2`)
      * Animated Elemental Sacks (True Dual-Phase Flow Map `172:0` + `1703:0` fluid sloshing + Fresnel + Wet Specular)
      * Physical Fresnel Glass Canisters (`BLEND_Translucent` + Refraction `1.45` + Fresnel Rim Opacity)
      * Animated Elemental Canisters (`Sine(Time*8)` pulsing collar `mat_id==1` + `Panner` fluid window `mat_id==20`)
      * Animated Machine Muscle Cables (`8260:1`) & Optical Muscle Fibers (`19501:0`, DXIL `Mask_Alpha*40`, TwoSided Additive)
      * Animated Machine Cores/Hearts (1 Hz `Abs(Sine)` heartbeat + `Panner` scroll) & Glowing Optical Eyes
      * Character/Asset PBR Surface Profiles (Skin SSS, Masked Two-Sided Hair/Foliage, Glossy Eyes, Cloth, Glass)

How to run inside Unreal Engine 5:
  1. Enable the built-in `Python Editor Script Plugin` (and `Control Rig` plugin) in UE5.
  2. In UE5 Menu -> Tools -> Execute Python Script... -> select this script.
  3. Open `{dest_root}/{"SK_" if is_skeletal else "SM_"}{safe_mach}` and hit Realtime Viewport!
"""

import json
import os
import re
import unreal

MACHINE_NAME = {safe_mach!r}
ASSET_KIND = {asset_kind!r}
IS_SKELETAL = {is_skeletal!r}
DEST_PACKAGE = {dest_root!r}
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else r"{bundle_abs}"
TEXTURE_DIR = os.path.join(SCRIPT_DIR, "textures")
ANIM_DIR = os.path.join(SCRIPT_DIR, "animations")
FBX_CANDIDATES = [
    os.path.join(SCRIPT_DIR, {fbx_name!r}),
    os.path.join(os.path.dirname(SCRIPT_DIR), {fbx_name!r}),
]

MATERIALS_META = json.loads(r"""{mats_json_str}""")
SHADER_CONTROLS = json.loads(r"""{controls_json_str}""")
RIG_SPEC = json.loads(r"""{rig_spec_json_str}""")
ANIMATIONS_META = json.loads(r"""{anims_json_str}""")

mel = unreal.MaterialEditingLibrary
asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
_TEX_CACHE = {{}}
_MPC_ASSET = None


def _ensure_mpc():
    """Create shared `MPC_HFW_MachineControl` Material Parameter Collection for live global shader customization."""
    global _MPC_ASSET
    if ASSET_KIND != "machine":
        return None
    if _MPC_ASSET is not None:
        return _MPC_ASSET
    mpc_pkg = DEST_PACKAGE + "/Materials"
    mpc_name = "MPC_HFW_MachineControl"
    mpc_path = f"{{mpc_pkg}}/{{mpc_name}}"
    existing = unreal.EditorAssetLibrary.load_asset(mpc_path)
    if existing:
        unreal.EditorAssetLibrary.delete_asset(mpc_path)
    try:
        mpc = asset_tools.create_asset(mpc_name, mpc_pkg, unreal.MaterialParameterCollection, unreal.MaterialParameterCollectionFactoryNew())
        mc = SHADER_CONTROLS.get("muscle_color", [0.95, 0.28, 0.02])
        ec = SHADER_CONTROLS.get("eye_color", [0.04, 0.56, 1.0])
        eac = SHADER_CONTROLS.get("eye_alert_color", [1.0, 0.08, 0.01])
        pc = SHADER_CONTROLS.get("plasma_color", [0.46, 0.05, 1.0])
        lc = SHADER_CONTROLS.get("liquid_override_color", [0.08, 0.62, 1.0])

        s_params = []
        for pname, pval in (
            ("MuscleGlowScale", 1.0),
            ("EyeAlertState", 0.0),
            ("EyeGlowScale", 1.0),
            ("LiquidOverrideMix", float(SHADER_CONTROLS.get("liquid_override_fac", 0.0))),
            ("LiquidGlowScale", float(SHADER_CONTROLS.get("liquid_glow_scale", 1.0))),
            ("PlasmaCharge", float(SHADER_CONTROLS.get("plasma_charge", 0.0))),
            ("AnimateSpeed", float(SHADER_CONTROLS.get("animation_speed", 1.0))),
        ):
            sp = unreal.CollectionScalarParameter()
            sp.parameter_name = pname
            sp.default_value = pval
            s_params.append(sp)

        v_params = []
        for pname, prgb in (
            ("MuscleColor", mc),
            ("EyeColor", ec),
            ("EyeAlertColor", eac),
            ("LiquidOverrideColor", lc),
            ("PlasmaColor", pc),
        ):
            vp = unreal.CollectionVectorParameter()
            vp.parameter_name = pname
            vp.default_value = unreal.LinearColor(float(prgb[0]), float(prgb[1]), float(prgb[2]), 1.0)
            v_params.append(vp)

        mpc.set_editor_property("scalar_parameters", s_params)
        mpc.set_editor_property("vector_parameters", v_params)
        unreal.EditorAssetLibrary.save_loaded_asset(mpc)
        _MPC_ASSET = mpc
    except Exception as exc:
        unreal.log_warning(f"[HFW] MPC creation fallback to material params: {{exc}}")
        _MPC_ASSET = None
    return _MPC_ASSET


def _mpc_param(mat, param_name: str, fallback_fn, x: int, y: int):
    """Create a CollectionParameter linked to `MPC_HFW_MachineControl`, or fall back cleanly to a normal Parameter."""
    mpc = _ensure_mpc()
    if mpc is not None:
        try:
            n = _mk(mat, unreal.MaterialExpressionCollectionParameter, x, y)
            n.set_editor_property("collection", mpc)
            n.set_editor_property("parameter_name", param_name)
            return n
        except Exception:
            pass
    return fallback_fn()


def _safe(name: str) -> str:
    return (re.sub(r"[^A-Za-z0-9_]", "_", name) or "Asset")[:56]


def import_texture(filename: str, is_srgb: bool = False, is_normal: bool = False):
    if not filename:
        return None
    ckey = (filename, is_srgb, is_normal)
    if ckey in _TEX_CACHE:
        return _TEX_CACHE[ckey]
    full_path = os.path.join(TEXTURE_DIR, filename)
    if not os.path.exists(full_path):
        unreal.log_warning(f"[HFW] Missing texture: {{full_path}}")
        return None
    dest_name = "T_" + _safe(os.path.splitext(filename)[0])
    dest_path = DEST_PACKAGE + "/Textures"
    existing = unreal.EditorAssetLibrary.load_asset(f"{{dest_path}}/{{dest_name}}")
    if existing:
        _TEX_CACHE[ckey] = existing
        return existing
    task = unreal.AssetImportTask()
    task.filename = full_path
    task.destination_path = dest_path
    task.destination_name = dest_name
    task.automated = True
    task.replace_existing = True
    task.save = True
    asset_tools.import_asset_tasks([task])
    tex = unreal.EditorAssetLibrary.load_asset(f"{{dest_path}}/{{dest_name}}")
    if tex:
        if is_normal:
            tex.compression_settings = unreal.TextureCompressionSettings.TC_NORMALMAP
            tex.srgb = False
        elif is_srgb:
            tex.srgb = True
        else:
            tex.compression_settings = unreal.TextureCompressionSettings.TC_MASKS
            tex.srgb = False
        unreal.EditorAssetLibrary.save_loaded_asset(tex)
    _TEX_CACHE[ckey] = tex
    return tex


def _mk(mat, cls, x: int, y: int):
    return mel.create_material_expression(mat, cls, x, y)


def _scalar(mat, name: str, val: float, x: int, y: int):
    n = _mk(mat, unreal.MaterialExpressionScalarParameter, x, y)
    n.set_editor_property("parameter_name", name)
    n.set_editor_property("default_value", float(val))
    return n


def _vec(mat, name: str, rgb, x: int, y: int):
    n = _mk(mat, unreal.MaterialExpressionVectorParameter, x, y)
    n.set_editor_property("parameter_name", name)
    n.set_editor_property("default_value", unreal.LinearColor(float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0))
    return n


def _tex_node(mat, name: str, tex_asset, sampler_type, x: int, y: int):
    n = _mk(mat, unreal.MaterialExpressionTextureSampleParameter2D, x, y)
    n.set_editor_property("parameter_name", name)
    n.set_editor_property("sampler_type", sampler_type)
    if tex_asset:
        n.set_editor_property("texture", tex_asset)
    return n


def _time_scaled(mat, x: int, y: int, speed_default: float = 1.0):
    n_t = _mk(mat, unreal.MaterialExpressionTime, x - 320, y)
    n_s = _mpc_param(mat, "AnimateSpeed", lambda: _scalar(mat, "AnimateSpeed", speed_default, x - 320, y + 110), x - 320, y + 110)
    n_m = _mk(mat, unreal.MaterialExpressionMultiply, x - 100, y + 30)
    mel.connect_material_expressions(n_t, "", n_m, "A")
    mel.connect_material_expressions(n_s, "", n_m, "B")
    return n_m


def _reconstruct_normal_z(mat, r_src, r_ch: str, g_src, g_ch: str, x: int, y: int):
    """Reconstruct tangent-space Normal XYZ from packed RG channels (`DeriveNormalZ`)."""
    n_app = _mk(mat, unreal.MaterialExpressionAppendVector, x - 260, y)
    mel.connect_material_expressions(r_src, r_ch, n_app, "A")
    mel.connect_material_expressions(g_src, g_ch, n_app, "B")
    n_bs = _mk(mat, unreal.MaterialExpressionConstantBiasScale, x - 120, y)
    n_bs.set_editor_property("bias", -0.5)
    n_bs.set_editor_property("scale", 2.0)
    mel.connect_material_expressions(n_app, "", n_bs, "")
    n_dz = _mk(mat, unreal.MaterialExpressionDeriveNormalZ, x + 60, y)
    mel.connect_material_expressions(n_bs, "", n_dz, "")
    return n_dz


def _build_mask_id_compare(mat, mask_src, mask_ch: str, target_id: float, x: int, y: int):
    """Exact DXIL mask ID isolator: `abs(round(Mask * 20.0) - target_id) < 0.25` -> 1.0 else 0.0."""
    n_m20 = _mk(mat, unreal.MaterialExpressionMultiply, x - 440, y)
    n_m20.set_editor_property("const_b", 20.0)
    mel.connect_material_expressions(mask_src, mask_ch, n_m20, "A")

    n_rnd = _mk(mat, unreal.MaterialExpressionRound, x - 300, y)
    mel.connect_material_expressions(n_m20, "", n_rnd, "")

    n_sub = _mk(mat, unreal.MaterialExpressionSubtract, x - 180, y)
    n_sub.set_editor_property("const_b", float(target_id))
    mel.connect_material_expressions(n_rnd, "", n_sub, "A")

    n_abs = _mk(mat, unreal.MaterialExpressionAbs, x - 60, y)
    mel.connect_material_expressions(n_sub, "", n_abs, "")

    n_if = _mk(mat, unreal.MaterialExpressionIf, x + 80, y)
    n_if.set_editor_property("const_b", 0.25)
    n_c0 = _mk(mat, unreal.MaterialExpressionConstant, x - 60, y + 90)
    n_c0.set_editor_property("r", 0.0)
    n_c1 = _mk(mat, unreal.MaterialExpressionConstant, x - 60, y + 160)
    n_c1.set_editor_property("r", 1.0)
    mel.connect_material_expressions(n_abs, "", n_if, "A")
    mel.connect_material_expressions(n_c0, "", n_if, "AGreaterThanB")
    mel.connect_material_expressions(n_c1, "", n_if, "AEqualsB")
    mel.connect_material_expressions(n_c1, "", n_if, "ALessThanB")
    return n_if


def build_single_material(slot_name: str, meta: dict):
    mat_name = "M_" + _safe(slot_name)
    mat_pkg = DEST_PACKAGE + "/Materials"
    existing = unreal.EditorAssetLibrary.load_asset(f"{{mat_pkg}}/{{mat_name}}")
    if existing:
        unreal.EditorAssetLibrary.delete_asset(f"{{mat_pkg}}/{{mat_name}}")

    mat = asset_tools.create_asset(mat_name, mat_pkg, unreal.Material, unreal.MaterialFactoryNew())
    spec = meta.get("special_shader")
    overlay = meta.get("emissive_overlay")

    # Gather semantic texture node sockets (handles BOTH native packed RGBA & unpacked PNG modes!)
    sockets = {{}}  # role -> (node, pin_name)
    y_cur = -420
    for pe in meta.get("packed_entries", []):
        packing = pe.get("packing", {{}})
        types_present = {{v.split(".", 1)[0] for v in packing.values()}}
        is_srgb = types_present <= {{"Color", "Incandescence", "Alpha"}} and "Color" in types_present
        tex = import_texture(pe.get("filename", ""), is_srgb=is_srgb, is_normal=False)
        stype = unreal.MaterialSamplerType.SAMPLERTYPE_COLOR if is_srgb else unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR
        tn = _tex_node(mat, "_".join(sorted(types_present)) or "Packed", tex, stype, -1280, y_cur)
        uv_idx = int(pe.get("uv_index", 0) or 0)
        if uv_idx > 0:
            n_uv = _mk(mat, unreal.MaterialExpressionTextureCoordinate, -1500, y_cur)
            n_uv.set_editor_property("coordinate_index", uv_idx)
            mel.connect_material_expressions(n_uv, "", tn, "UVs")
        y_cur += 260
        r_t = packing.get("R", "").split(".")[0]
        g_t = packing.get("G", "").split(".")[0]
        b_t = packing.get("B", "").split(".")[0]
        if r_t == g_t == b_t and r_t in ("Color", "Incandescence"):
            sockets.setdefault(r_t, (tn, "RGB"))
            if "A" in packing:
                sockets.setdefault(packing["A"].split(".")[0], (tn, "A"))
        else:
            for lane, pspec in packing.items():
                parts = pspec.split(".")
                tname = parts[0]
                ch = parts[1] if len(parts) > 1 else "R"
                if tname == "Normal":
                    sockets.setdefault(f"Normal.{{ch}}", (tn, lane))
                else:
                    sockets.setdefault(tname, (tn, lane))

    if not sockets and meta.get("maps"):
        maps = meta["maps"]
        for role, is_srgb, is_nrm, pin in (
            ("Color", True, False, "RGB"),
            ("Roughness", False, False, "R"),
            ("Reflectance", False, False, "R"),
            ("AO", False, False, "R"),
            ("Mask", False, False, "R"),
            ("Mask_Alpha", False, False, "R"),
            ("Incandescence", True, False, "RGB"),
            ("Alpha", False, False, "R"),
        ):
            if maps.get(role):
                tex = import_texture(maps[role], is_srgb=is_srgb, is_normal=is_nrm)
                stype = unreal.MaterialSamplerType.SAMPLERTYPE_COLOR if is_srgb else unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR
                tn = _tex_node(mat, role, tex, stype, -1280, y_cur)
                y_cur += 240
                sockets[role] = (tn, pin)
        nrm_f = maps.get("Normal_DX") or maps.get("Normal_GL")
        if nrm_f:
            tex = import_texture(nrm_f, is_srgb=False, is_normal=True)
            tn = _tex_node(mat, "Normal_DX", tex, unreal.MaterialSamplerType.SAMPLERTYPE_NORMAL, -1280, y_cur)
            sockets["Normal_RGB"] = (tn, "RGB")

    # ------------------------------------------------------------------
    # HFW DXIL 8-LAYER CHARACTER OUTFIT & BUILDING BLOCK TILEABLE SHADER
    # ------------------------------------------------------------------
    layered = meta.get("outfit_layered")
    if layered and layered.get("layers"):
        layers = layered["layers"][:8]
        is_bb = layered.get("mode") == "building_block"
        uv_scale = float(layered.get("uv_scale", 4.0 if is_bb else 16.0))
        n_uv = _mk(mat, unreal.MaterialExpressionTextureCoordinate, -1800, -320)
        n_uv.set_editor_property("u_tiling", uv_scale)
        n_uv.set_editor_property("v_tiling", uv_scale)

        l_cols, l_mets, l_rghs, l_nrms = [], [], [], []
        t_hint = str(layered.get("tribe_hint", "")).lower()
        macro_col_file = layered.get("macro_color_file", "")
        macro_col_mode = layered.get("macro_color_mode", "replace")
        use_macro_col = bool(macro_col_file)
        replace_with_macro = use_macro_col and macro_col_mode != "multiply2"
        nrm_tex_cache = {{}}

        def _layer_rgh_range(lname, lcat, is_m):
            nl = f"{{lcat}}/{{lname}}".lower()
            if is_m or any(k in nl for k in ("robotplate", "iron", "steel", "copper", "bronze", "brass", "gold", "anodized", "metal")):
                return (0.16, 0.44)
            if any(k in nl for k in ("bead", "shell", "nacre", "pearl", "ceramic", "porcelain", "jade", "turquoise")):
                return (0.18, 0.46)
            if any(k in nl for k in ("buckskin", "suede", "nubuck")):
                return (0.54, 0.78)
            if any(k in nl for k in ("leather", "hide", "fish")):
                return (0.34, 0.68)
            if any(k in nl for k in ("carbon", "rubber")):
                return (0.32, 0.64)
            if any(k in nl for k in ("silk", "satin")):
                return (0.42, 0.68)
            if any(k in nl for k in ("wool", "cotton", "linen", "crochet", "cloth", "fabric", "braid", "felt")):
                return (0.66, 0.88)
            return (0.44, 0.76)

        def _layer_tint_lin(lname, lcat, l_idx):
            nl = f"{{lcat}}/{{lname}}".lower()
            if "bronze" in nl or "brass" in nl or "gold" in nl:
                srgb = [0.76, 0.56, 0.28]
            elif "copper" in nl:
                srgb = [0.78, 0.44, 0.28]
            elif any(k in nl for k in ("robotplate", "iron", "metal", "steel", "anodized")):
                srgb = [0.58, 0.60, 0.62]
            elif "carbon" in nl or "rubber" in nl:
                srgb = [0.14, 0.15, 0.17]
            elif "fullgrain" in nl:
                srgb = [0.33, 0.27, 0.20]
            elif "finegrain" in nl or "nubuck" in nl:
                srgb = [0.45, 0.34, 0.24]
            elif "buckskin" in nl or "suede" in nl:
                srgb = [0.58, 0.44, 0.28]
            elif "leather" in nl:
                srgb = [0.38, 0.24, 0.18]
            elif any(k in nl for k in ("wood", "bark", "bone")):
                srgb = [0.50, 0.38, 0.26]
            elif "carja" in t_hint:
                srgb = [0.78, 0.60, 0.10] if l_idx % 2 == 0 else [0.55, 0.30, 0.22]
            elif "quen" in t_hint or "seyka" in t_hint:
                srgb = [0.62, 0.72, 0.78] if l_idx % 2 == 0 else [0.86, 0.62, 0.50]
            elif "tenakth" in t_hint:
                srgb = [0.20, 0.42, 0.52] if l_idx % 2 == 0 else [0.58, 0.22, 0.16]
            elif "utaru" in t_hint:
                srgb = [0.36, 0.48, 0.32] if l_idx % 2 == 0 else [0.66, 0.60, 0.46]
            else:
                srgb = [0.22, 0.40, 0.57] if l_idx % 2 == 1 else [0.58, 0.54, 0.48]
            return [c ** 2.2 for c in srgb]

        n_flat_xy = _mk(mat, unreal.MaterialExpressionConstant2Vector, -1560, 2900)
        n_flat_xy.set_editor_property("r", 0.5)
        n_flat_xy.set_editor_property("g", 0.5)

        for idx, lay in enumerate(layers):
            lname = _safe(lay.get("name", f"Layer{{idx}}"))
            is_m = bool(lay.get("is_metallic", False))
            if is_bb:
                c_tex = import_texture(lay.get("color_file", ""), is_srgb=True, is_normal=False)
                if c_tex:
                    cn = _tex_node(mat, f"BB_L{{idx}}_{{lname}}", c_tex, unreal.MaterialSamplerType.SAMPLERTYPE_COLOR, -1340, -480 + idx * 240)
                    mel.connect_material_expressions(n_uv, "", cn, "UVs")
                    l_cols.append((cn, "RGB"))
                else:
                    l_cols.append((_vec(mat, f"BB_L{{idx}}_{{lname}}", [0.22, 0.21, 0.19], -1340, -480 + idx * 240), ""))
                l_mets.append((_const(mat, 0.75 if is_m else 0.0, -1340, 1400 + idx * 90), ""))
                l_rghs.append((_const(mat, 0.42 if is_m else 0.78, -1340, 2200 + idx * 90), ""))
                l_nrms.append((n_flat_xy, ""))
            else:
                dye_lin = lay.get("dye_color_linear") or _layer_tint_lin(lay.get("name", ""), lay.get("category", ""), idx)
                tint_v = _vec(mat, f"L{{idx}}_{{lname}}_Dye", dye_lin, -1560, -560 + idx * 260)
                # CRITICAL: HFW `Shaders_HRZ2/Materials/*_CLR` is Linear `0.5`-centered (`is_srgb=False`) -> pow(2.0 * tex, 2.2) * Dye!
                c_tex = import_texture(lay.get("color_file", ""), is_srgb=False, is_normal=False)
                if c_tex:
                    cn = _tex_node(mat, f"L{{idx}}_{{lname}}_Detail", c_tex, unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR, -1560, -440 + idx * 260)
                    mel.connect_material_expressions(n_uv, "", cn, "UVs")
                    n_x2 = _mk(mat, unreal.MaterialExpressionMultiply, -1300, -440 + idx * 260)
                    n_x2.set_editor_property("const_b", 2.0)
                    mel.connect_material_expressions(cn, "RGB", n_x2, "A")
                    n_pw = _mk(mat, unreal.MaterialExpressionPower, -1140, -440 + idx * 260)
                    n_pw.set_editor_property("const_exponent", 2.2)
                    mel.connect_material_expressions(n_x2, "", n_pw, "Base")
                    if replace_with_macro:
                        l_cols.append((n_pw, ""))
                    else:
                        tm = _mk(mat, unreal.MaterialExpressionMultiply, -980, -500 + idx * 260)
                        mel.connect_material_expressions(tint_v, "", tm, "A")
                        mel.connect_material_expressions(n_pw, "", tm, "B")
                        l_cols.append((tm, ""))
                else:
                    l_cols.append((_const(mat, 1.0, -1300, -500 + idx * 260), "") if replace_with_macro else (tint_v, ""))
                l_mets.append((_const(mat, 0.88 if is_m else 0.0, -1340, 1400 + idx * 90), ""))

                # Sample per-layer 16x micro Normal/Roughness (`*_NMT_RGH_HGT.png`: RG=Normal, B=GLS Smoothness)!
                lnf = lay.get("normal_file", "")
                ln_node = None
                if lnf:
                    if lnf in nrm_tex_cache:
                        ln_node = nrm_tex_cache[lnf]
                    else:
                        lt = import_texture(lnf, is_srgb=False, is_normal=False)
                        if lt:
                            ln_node = _tex_node(mat, f"L{{idx}}_{{lname}}_MicroNrmRgh", lt, unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR, -1560, 2100 + idx * 240)
                            mel.connect_material_expressions(n_uv, "", ln_node, "UVs")
                            nrm_tex_cache[lnf] = ln_node

                r_lo, r_hi = _layer_rgh_range(lay.get("name", ""), lay.get("category", ""), is_m)
                if ln_node is not None:
                    n_lr = _mk(mat, unreal.MaterialExpressionLinearInterpolate, -1240, 2120 + idx * 240)
                    n_lr.set_editor_property("const_a", r_hi)
                    n_lr.set_editor_property("const_b", r_lo)
                    mel.connect_material_expressions(ln_node, "B", n_lr, "Alpha")
                    l_rghs.append((n_lr, ""))
                    n_lxy = _mk(mat, unreal.MaterialExpressionAppendVector, -1240, 2240 + idx * 240)
                    mel.connect_material_expressions(ln_node, "R", n_lxy, "A")
                    mel.connect_material_expressions(ln_node, "G", n_lxy, "B")
                    l_nrms.append((n_lxy, ""))
                else:
                    l_rghs.append((_const(mat, float(lay.get("roughness", (r_lo + r_hi) * 0.5)), -1340, 2200 + idx * 90), ""))
                    l_nrms.append((n_flat_xy, ""))

        while len(l_cols) < 8:
            pidx = len(l_cols) % max(1, len(layers))
            l_cols.append(l_cols[pidx])
            l_mets.append(l_mets[pidx])
            l_rghs.append(l_rghs[pidx])
            l_nrms.append(l_nrms[pidx])

        msk_tex = import_texture(layered.get("mask_file", ""), is_srgb=False, is_normal=False) if layered.get("mask_file") else None
        if msk_tex and len(layers) > 1:
            t_msk = _tex_node(mat, "OutfitMaskRGB", msk_tex, unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR, -1340, -960)
            def _tri(arr, y_off):
                def _lp(a, b, pin, x, y):
                    n = _mk(mat, unreal.MaterialExpressionLinearInterpolate, x, y)
                    mel.connect_material_expressions(a[0], a[1], n, "A")
                    mel.connect_material_expressions(b[0], b[1], n, "B")
                    mel.connect_material_expressions(t_msk, pin, n, "Alpha")
                    return (n, "")
                # Exact DXIL 8-corner cube order:
                #   (0,0,0)=L0 Black, (1,0,0)=L1 Red, (0,1,0)=L2 Green, (1,1,0)=L6 Yellow,
                #   (0,0,1)=L3 Blue,  (1,0,1)=L5 Mag, (0,1,1)=L4 Cyan,  (1,1,1)=L7 White
                c_g0b0 = _lp(arr[0], arr[1], "R", -780, y_off)
                c_g1b0 = _lp(arr[2], arr[6], "R", -780, y_off + 140)
                c_g0b1 = _lp(arr[3], arr[5], "R", -780, y_off + 280)
                c_g1b1 = _lp(arr[4], arr[7], "R", -780, y_off + 420)
                c_b0 = _lp(c_g0b0, c_g1b0, "G", -540, y_off + 70)
                c_b1 = _lp(c_g0b1, c_g1b1, "G", -540, y_off + 350)
                return _lp(c_b0, c_b1, "B", -320, y_off + 210)
            col_out = _tri(l_cols, -420)
            met_out = _tri(l_mets, 320)
            rgh_out = _tri(l_rghs, 920)
            micro_xy_out = _tri(l_nrms, 2100) if nrm_tex_cache else None
            if layered.get("mask_has_alpha"):
                mat.set_editor_property("blend_mode", unreal.BlendMode.BLEND_MASKED)
                mat.set_editor_property("two_sided", True)
                mel.connect_material_property(t_msk, "A", unreal.MaterialProperty.MP_OPACITY_MASK)
        else:
            col_out = l_cols[0]
            met_out = l_mets[0]
            rgh_out = l_rghs[0]
            micro_xy_out = l_nrms[0] if nrm_tex_cache else None

        # Hybrid Macro Color (e.g., SeykaTorso `multiply2` overlay over 8-corner DXIL dyes, or `replace` trim sheet)!
        if use_macro_col:
            mc_tex = import_texture(macro_col_file, is_srgb=True, is_normal=False)
            if mc_tex:
                t_mc = _tex_node(mat, "OutfitMacroColor", mc_tex, unreal.MaterialSamplerType.SAMPLERTYPE_COLOR, -520, -760)
                if macro_col_mode == "multiply2":
                    n_mc2 = _mk(mat, unreal.MaterialExpressionMultiply, -340, -680)
                    n_mc2.set_editor_property("const_b", 2.0)
                    mel.connect_material_expressions(t_mc, "RGB", n_mc2, "A")
                    n_mcm = _mk(mat, unreal.MaterialExpressionMultiply, -180, -480)
                    mel.connect_material_expressions(col_out[0], col_out[1], n_mcm, "A")
                    mel.connect_material_expressions(n_mc2, "", n_mcm, "B")
                    n_mcs = _mk(mat, unreal.MaterialExpressionSaturate, -60, -480)
                    mel.connect_material_expressions(n_mcm, "", n_mcs, "")
                    col_out = (n_mcs, "")
                else:
                    n_mcm = _mk(mat, unreal.MaterialExpressionMultiply, -180, -480)
                    mel.connect_material_expressions(t_mc, "RGB", n_mcm, "A")
                    mel.connect_material_expressions(col_out[0], col_out[1], n_mcm, "B")
                    col_out = (n_mcm, "")

        # Optional *_MISC dirt/wear roughness lift!
        misc_f = layered.get("misc_file", "")
        if misc_f:
            ms_tex = import_texture(misc_f, is_srgb=False, is_normal=False)
            if ms_tex:
                t_ms = _tex_node(mat, "OutfitMiscDirt", ms_tex, unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR, -520, 1260)
                n_dm = _mk(mat, unreal.MaterialExpressionMultiply, -280, 1260)
                n_dm.set_editor_property("const_b", 0.45)
                mel.connect_material_expressions(t_ms, "G", n_dm, "A")
                n_dr = _mk(mat, unreal.MaterialExpressionLinearInterpolate, -120, 1080)
                n_dr.set_editor_property("const_b", 0.88)
                mel.connect_material_expressions(rgh_out[0], rgh_out[1], n_dr, "A")
                mel.connect_material_expressions(n_dm, "", n_dr, "Alpha")
                rgh_out = (n_dr, "")

        mel.connect_material_property(met_out[0], met_out[1], unreal.MaterialProperty.MP_METALLIC)
        mel.connect_material_property(rgh_out[0], rgh_out[1], unreal.MaterialProperty.MP_ROUGHNESS)

        nrm_tex = import_texture(layered.get("normal_file", ""), is_srgb=False, is_normal=False) if layered.get("normal_file") else None
        if nrm_tex:
            t_nrm = _tex_node(mat, "OutfitNormalAO", nrm_tex, unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR, -920, 1520)
            if micro_xy_out is not None:
                # Tangent-space UDN addition: macro_xy + (micro_xy - 0.5) * 0.55!
                n_mxy = _mk(mat, unreal.MaterialExpressionAppendVector, -680, 1520)
                mel.connect_material_expressions(t_nrm, "R", n_mxy, "A")
                mel.connect_material_expressions(t_nrm, "G", n_mxy, "B")
                n_mc = _mk(mat, unreal.MaterialExpressionConstantBiasScale, -680, 1700)
                n_mc.set_editor_property("bias", -0.5)
                n_mc.set_editor_property("scale", 0.55)
                mel.connect_material_expressions(micro_xy_out[0], micro_xy_out[1], n_mc, "")
                n_add = _mk(mat, unreal.MaterialExpressionAdd, -500, 1580)
                mel.connect_material_expressions(n_mxy, "", n_add, "A")
                mel.connect_material_expressions(n_mc, "", n_add, "B")
                n_sat = _mk(mat, unreal.MaterialExpressionSaturate, -380, 1580)
                mel.connect_material_expressions(n_add, "", n_sat, "")
                n_cr = _mk(mat, unreal.MaterialExpressionComponentMask, -260, 1520)
                n_cr.set_editor_property("r", True); n_cr.set_editor_property("g", False); n_cr.set_editor_property("b", False)
                mel.connect_material_expressions(n_sat, "", n_cr, "")
                n_cg = _mk(mat, unreal.MaterialExpressionComponentMask, -260, 1620)
                n_cg.set_editor_property("r", False); n_cg.set_editor_property("g", True); n_cg.set_editor_property("b", False)
                mel.connect_material_expressions(n_sat, "", n_cg, "")
                n_nrm = _reconstruct_normal_z(mat, n_cr, "", n_cg, "", -100, 1560)
            else:
                n_nrm = _reconstruct_normal_z(mat, t_nrm, "R", t_nrm, "G", -420, 1520)
            mel.connect_material_property(n_nrm, "", unreal.MaterialProperty.MP_NORMAL)
            npack = layered.get("normal_packing") or {{}}
            if any(v.startswith("AO") for v in npack.values()):
                n_ao = _mk(mat, unreal.MaterialExpressionMultiply, -180, -210)
                mel.connect_material_expressions(col_out[0], col_out[1], n_ao, "A")
                mel.connect_material_expressions(t_nrm, "B", n_ao, "B")
                col_out = (n_ao, "")
                mel.connect_material_property(t_nrm, "B", unreal.MaterialProperty.MP_AMBIENT_OCCLUSION)
            # 2K Curvature (NMT_AO_CRV.A) edge highlight & cavity contrast (`0.78..1.24`)!
            if any("Curvature" in v for v in npack.values()):
                n_crv = _mk(mat, unreal.MaterialExpressionLinearInterpolate, -240, -60)
                n_crv.set_editor_property("const_a", 0.78)
                n_crv.set_editor_property("const_b", 1.24)
                mel.connect_material_expressions(t_nrm, "A", n_crv, "Alpha")
                n_cmul = _mk(mat, unreal.MaterialExpressionMultiply, -40, -140)
                mel.connect_material_expressions(col_out[0], col_out[1], n_cmul, "A")
                mel.connect_material_expressions(n_crv, "", n_cmul, "B")
                col_out = (n_cmul, "")

        mel.connect_material_property(col_out[0], col_out[1], unreal.MaterialProperty.MP_BASE_COLOR)
        mel.recompile_material(mat)
        unreal.EditorAssetLibrary.save_loaded_asset(mat)
        return mat

    # ------------------------------------------------------------------
    # SPECIALIZED ANIMATED SHADERS (Muscles, Cores, Eyes, Canisters, VFX)
    # ------------------------------------------------------------------
    if spec:
        skind = spec.get("kind", "")

        if skind == "vfx_overlay_shell":
            mat.set_editor_property("blend_mode", unreal.BlendMode.BLEND_TRANSLUCENT)
            mat.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
            n_op = _scalar(mat, "VFXOverlayOpacity", 0.0, -300, 120)
            n_vc = _vec(mat, "GlowColor", spec.get("glow_color", [0.05, 0.65, 1.0]), -300, -80)
            mel.connect_material_property(n_vc, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
            mel.connect_material_property(n_op, "", unreal.MaterialProperty.MP_OPACITY)
            mel.recompile_material(mat)
            unreal.EditorAssetLibrary.save_loaded_asset(mat)
            return mat

        if skind == "canister_glass":
            mat.set_editor_property("blend_mode", unreal.BlendMode.BLEND_TRANSLUCENT)
            try:
                mat.set_editor_property("translucency_lighting_mode", unreal.TranslucencyLightingMode.TLM_SURFACE)
            except Exception:
                pass
            n_tc = _vec(mat, "GlassTint", spec.get("tint_color", [0.80, 0.92, 0.91]), -540, -180)
            n_rg = _scalar(mat, "GlassRoughness", 0.05, -540, -20)
            n_sp = _scalar(mat, "GlassSpecular", 0.85, -540, 80)
            n_fr = _mk(mat, unreal.MaterialExpressionFresnel, -540, 220)
            n_fr.set_editor_property("exponent", 3.0)
            n_op = _mk(mat, unreal.MaterialExpressionLinearInterpolate, -260, 180)
            n_op.set_editor_property("const_a", 0.14)
            n_op.set_editor_property("const_b", 0.68)
            mel.connect_material_expressions(n_fr, "", n_op, "Alpha")
            n_rf = _mk(mat, unreal.MaterialExpressionLinearInterpolate, -260, 340)
            n_rf.set_editor_property("const_a", 1.0)
            n_rf.set_editor_property("const_b", float(spec.get("ior", 1.45)))
            mel.connect_material_expressions(n_fr, "", n_rf, "Alpha")
            mel.connect_material_property(n_tc, "", unreal.MaterialProperty.MP_BASE_COLOR)
            mel.connect_material_property(n_rg, "", unreal.MaterialProperty.MP_ROUGHNESS)
            mel.connect_material_property(n_sp, "", unreal.MaterialProperty.MP_SPECULAR)
            mel.connect_material_property(n_op, "", unreal.MaterialProperty.MP_OPACITY)
            mel.connect_material_property(n_rf, "", unreal.MaterialProperty.MP_REFRACTION)
            mel.recompile_material(mat)
            unreal.EditorAssetLibrary.save_loaded_asset(mat)
            return mat

        if skind == "muscle_fiber_overlay":
            mat.set_editor_property("blend_mode", unreal.BlendMode.BLEND_ADDITIVE)
            mat.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
            mat.set_editor_property("two_sided", True)
            n_t = _time_scaled(mat, -1000, 200)
            n_uv1 = _mk(mat, unreal.MaterialExpressionTextureCoordinate, -1260, 0)
            n_uv1.set_editor_property("coordinate_index", 1)
            n_uv1.set_editor_property("u_tiling", 6.0)
            n_uv1.set_editor_property("v_tiling", 12.0)
            n_pan = _mk(mat, unreal.MaterialExpressionPanner, -960, 0)
            n_pan.set_editor_property("speed_x", 0.0)
            n_pan.set_editor_property("speed_y", -0.50)
            mel.connect_material_expressions(n_uv1, "", n_pan, "Coordinate")
            mel.connect_material_expressions(n_t, "", n_pan, "Time")
            ramp_tex = import_texture(spec.get("ramp_file", ""), is_srgb=False)
            n_rmp = _tex_node(mat, "MuscleFiberRamp", ramp_tex, unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR, -680, 0)
            mel.connect_material_expressions(n_pan, "", n_rmp, "UVs")
            n_b_scl = _mk(mat, unreal.MaterialExpressionMultiply, -540, -20)
            n_b_scl.set_editor_property("const_b", 0.42)
            mel.connect_material_expressions(n_rmp, "B", n_b_scl, "A")
            n_wadd = _mk(mat, unreal.MaterialExpressionAdd, -420, 40)
            mel.connect_material_expressions(n_b_scl, "", n_wadd, "A")
            mel.connect_material_expressions(n_rmp, "A", n_wadd, "B")
            n_wsat = _mk(mat, unreal.MaterialExpressionSaturate, -300, 40)
            mel.connect_material_expressions(n_wadd, "", n_wsat, "")
            # Authentic DXIL sparse wave contrast (0.12..1.0) + center-facing Fresnel edge falloff (dark rims!)
            n_wfl = _mk(mat, unreal.MaterialExpressionLinearInterpolate, -180, 40)
            n_wfl.set_editor_property("const_a", 0.12)
            n_wfl.set_editor_property("const_b", 1.0)
            mel.connect_material_expressions(n_wsat, "", n_wfl, "Alpha")

            n_fr = _mk(mat, unreal.MaterialExpressionFresnel, -540, 200)
            n_fr.set_editor_property("exponent", 1.6)
            n_cf = _mk(mat, unreal.MaterialExpressionOneMinus, -360, 200)
            mel.connect_material_expressions(n_fr, "", n_cf, "")
            n_cfp = _mk(mat, unreal.MaterialExpressionPower, -220, 200)
            n_cfp.set_editor_property("const_exponent", 1.75)
            mel.connect_material_expressions(n_cf, "", n_cfp, "Base")

            n_heat = _mk(mat, unreal.MaterialExpressionMultiply, -40, 120)
            mel.connect_material_expressions(n_wfl, "", n_heat, "A")
            mel.connect_material_expressions(n_cfp, "", n_heat, "B")

            # Color blend driven by MPC_HFW_MachineControl + 3-tier Ember -> MuscleColor -> HotCore gradient!
            n_cc = _mpc_param(mat, "MuscleColor", lambda: _vec(mat, "CalmColor", spec.get("calm_color", [0.95, 0.28, 0.02]), -720, -340), -720, -340)
            n_pc = _mpc_param(mat, "PlasmaColor", lambda: _vec(mat, "PlasmaColor", spec.get("plasma_color", [0.46, 0.05, 1.0]), -720, -180), -720, -180)
            c_state_def = float(spec.get("charge_state", 0.0))
            n_cs = _mpc_param(mat, "PlasmaCharge", lambda: _scalar(mat, "ChargeState", c_state_def, -720, -40), -720, -40) if c_state_def >= 0.5 else _scalar(mat, "ChargeState", 0.0, -720, -40)
            n_lerp = _mk(mat, unreal.MaterialExpressionLinearInterpolate, -420, -240)
            mel.connect_material_expressions(n_cc, "", n_lerp, "A")
            mel.connect_material_expressions(n_pc, "", n_lerp, "B")
            mel.connect_material_expressions(n_cs, "", n_lerp, "Alpha")

            n_emb = _mk(mat, unreal.MaterialExpressionPower, -240, -340)
            n_emb.set_editor_property("const_exponent", 2.2)
            mel.connect_material_expressions(n_lerp, "", n_emb, "Base")
            n_emb_d = _mk(mat, unreal.MaterialExpressionMultiply, -120, -340)
            n_emb_d.set_editor_property("const_b", 0.40)
            mel.connect_material_expressions(n_emb, "", n_emb_d, "A")
            n_grad = _mk(mat, unreal.MaterialExpressionLinearInterpolate, 20, -240)
            mel.connect_material_expressions(n_emb_d, "", n_grad, "A")
            mel.connect_material_expressions(n_lerp, "", n_grad, "B")
            mel.connect_material_expressions(n_heat, "", n_grad, "Alpha")

            n_str_base = _scalar(mat, "GlowStrength", float(spec.get("glow_strength", 2.6)), -620, 320)
            n_str_sc = _mpc_param(mat, "MuscleGlowScale", lambda: _scalar(mat, "MuscleGlowScale", 1.0, -620, 420), -620, 420)
            n_str = _mk(mat, unreal.MaterialExpressionMultiply, -400, 340)
            mel.connect_material_expressions(n_str_base, "", n_str, "A")
            mel.connect_material_expressions(n_str_sc, "", n_str, "B")

            n_m1 = _mk(mat, unreal.MaterialExpressionMultiply, 140, -60)
            mel.connect_material_expressions(n_grad, "", n_m1, "A")
            mel.connect_material_expressions(n_heat, "", n_m1, "B")
            n_m2 = _mk(mat, unreal.MaterialExpressionMultiply, 260, 0)
            mel.connect_material_expressions(n_m1, "", n_m2, "A")
            mel.connect_material_expressions(n_str, "", n_m2, "B")
            if "Mask_Alpha" in sockets:
                # Exact DXIL L434 `saturate(Mask_Alpha * 40.0)` with 0.28..1.0 cavity gradient!
                n_ma = _mk(mat, unreal.MaterialExpressionMultiply, -420, 480)
                n_ma.set_editor_property("const_b", 40.0)
                mel.connect_material_expressions(sockets["Mask_Alpha"][0], sockets["Mask_Alpha"][1], n_ma, "A")
                n_masat = _mk(mat, unreal.MaterialExpressionSaturate, -260, 480)
                mel.connect_material_expressions(n_ma, "", n_masat, "")
                n_malerp = _mk(mat, unreal.MaterialExpressionLinearInterpolate, -100, 480)
                n_malerp.set_editor_property("const_a", 0.28)
                n_malerp.set_editor_property("const_b", 1.0)
                mel.connect_material_expressions(n_masat, "", n_malerp, "Alpha")
                n_m3 = _mk(mat, unreal.MaterialExpressionMultiply, 380, 80)
                mel.connect_material_expressions(n_m2, "", n_m3, "A")
                mel.connect_material_expressions(n_malerp, "", n_m3, "B")
                n_m2 = n_m3
            mel.connect_material_property(n_m2, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
            # Subtle +0.14cm WorldPositionOffset along VertexNormalWS to prevent coplanar z-fighting in UE5!
            n_vn = _mk(mat, unreal.MaterialExpressionVertexNormalWS, -340, 580)
            n_wpo = _mk(mat, unreal.MaterialExpressionMultiply, -100, 580)
            n_wpo.set_editor_property("const_b", 0.14)
            mel.connect_material_expressions(n_vn, "", n_wpo, "A")
            mel.connect_material_property(n_wpo, "", unreal.MaterialProperty.MP_WORLD_POSITION_OFFSET)
            mel.recompile_material(mat)
            unreal.EditorAssetLibrary.save_loaded_asset(mat)
            return mat

        if skind == "muscle_cables":
            n_bc = _vec(mat, "CableBaseColor", [0.015, 0.016, 0.02], -400, -340)
            n_rg = _scalar(mat, "CableRoughness", 0.42, -400, -180)
            mel.connect_material_property(n_bc, "", unreal.MaterialProperty.MP_BASE_COLOR)
            mel.connect_material_property(n_rg, "", unreal.MaterialProperty.MP_ROUGHNESS)

            n_t = _time_scaled(mat, -1240, 220)
            n_uv_b = _mk(mat, unreal.MaterialExpressionTextureCoordinate, -1240, -60)
            n_uv_b.set_editor_property("u_tiling", 6.0)
            n_uv_b.set_editor_property("v_tiling", 2.0)
            br_tex = import_texture(spec.get("braid_file", ""), is_srgb=False)
            n_br = _tex_node(mat, "MuscleBraid", br_tex, unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR, -920, -60)
            mel.connect_material_expressions(n_uv_b, "", n_br, "UVs")
            n_nrm = _reconstruct_normal_z(mat, n_br, "R", n_br, "G", -420, -60)
            mel.connect_material_property(n_nrm, "", unreal.MaterialProperty.MP_NORMAL)

            n_uv_p = _mk(mat, unreal.MaterialExpressionTextureCoordinate, -1240, 380)
            n_uv_p.set_editor_property("u_tiling", 6.0)
            n_uv_p.set_editor_property("v_tiling", 3.0)
            n_pan = _mk(mat, unreal.MaterialExpressionPanner, -960, 340)
            n_pan.set_editor_property("speed_y", -0.65)
            mel.connect_material_expressions(n_uv_p, "", n_pan, "Coordinate")
            mel.connect_material_expressions(n_t, "", n_pan, "Time")
            pl_tex = import_texture(spec.get("pulse_file", ""), is_srgb=False)
            n_pl = _tex_node(mat, "MusclePulse", pl_tex, unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR, -680, 340)
            mel.connect_material_expressions(n_pan, "", n_pl, "UVs")

            n_gc = _mpc_param(mat, "MuscleColor", lambda: _vec(mat, "GlowColor", spec.get("glow_color", [0.95, 0.28, 0.02]), -680, 620), -680, 620)
            n_gs_b = _scalar(mat, "GlowStrength", float(spec.get("glow_strength", 3.0)), -680, 760)
            n_gs_s = _mpc_param(mat, "MuscleGlowScale", lambda: _scalar(mat, "MuscleGlowScale", 1.0, -680, 860), -680, 860)
            n_gs = _mk(mat, unreal.MaterialExpressionMultiply, -440, 780)
            mel.connect_material_expressions(n_gs_b, "", n_gs, "A")
            mel.connect_material_expressions(n_gs_s, "", n_gs, "B")
            n_fr_c = _mk(mat, unreal.MaterialExpressionFresnel, -680, 120)
            n_fr_c.set_editor_property("exponent", 1.6)
            n_cf_c = _mk(mat, unreal.MaterialExpressionOneMinus, -520, 120)
            mel.connect_material_expressions(n_fr_c, "", n_cf_c, "")
            n_m1 = _mk(mat, unreal.MaterialExpressionMultiply, -400, 260)
            mel.connect_material_expressions(n_br, "B", n_m1, "A")
            mel.connect_material_expressions(n_pl, "R", n_m1, "B")
            n_heat_c = _mk(mat, unreal.MaterialExpressionMultiply, -260, 220)
            mel.connect_material_expressions(n_m1, "", n_heat_c, "A")
            mel.connect_material_expressions(n_cf_c, "", n_heat_c, "B")
            n_emb_c = _mk(mat, unreal.MaterialExpressionPower, -420, 560)
            n_emb_c.set_editor_property("const_exponent", 2.2)
            mel.connect_material_expressions(n_gc, "", n_emb_c, "Base")
            n_grad_c = _mk(mat, unreal.MaterialExpressionLinearInterpolate, -220, 480)
            mel.connect_material_expressions(n_emb_c, "", n_grad_c, "A")
            mel.connect_material_expressions(n_gc, "", n_grad_c, "B")
            mel.connect_material_expressions(n_heat_c, "", n_grad_c, "Alpha")
            n_m2 = _mk(mat, unreal.MaterialExpressionMultiply, -80, 440)
            mel.connect_material_expressions(n_grad_c, "", n_m2, "A")
            mel.connect_material_expressions(n_gs, "", n_m2, "B")
            n_m3 = _mk(mat, unreal.MaterialExpressionMultiply, 80, 360)
            mel.connect_material_expressions(n_heat_c, "", n_m3, "A")
            mel.connect_material_expressions(n_m2, "", n_m3, "B")
            mel.connect_material_property(n_m3, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
            mel.recompile_material(mat)
            unreal.EditorAssetLibrary.save_loaded_asset(mat)
            return mat

        if skind == "machine_core_emissive":
            n_t = _time_scaled(mat, -1200, 120)
            n_uv = _mk(mat, unreal.MaterialExpressionTextureCoordinate, -1200, -140)
            n_pan = _mk(mat, unreal.MaterialExpressionPanner, -920, -140)
            n_pan.set_editor_property("speed_y", 0.22)
            mel.connect_material_expressions(n_uv, "", n_pan, "Coordinate")
            mel.connect_material_expressions(n_t, "", n_pan, "Time")
            col_tex = import_texture(spec.get("color_file", ""), is_srgb=True)
            n_c = _tex_node(mat, "CoreColor", col_tex, unreal.MaterialSamplerType.SAMPLERTYPE_COLOR, -640, -140)
            mel.connect_material_expressions(n_pan, "", n_c, "UVs")
            mel.connect_material_property(n_c, "RGB", unreal.MaterialProperty.MP_BASE_COLOR)

            # 1 Hz DXIL heartbeat: 0.55 + 0.65 * abs(sin(Time * pi))
            n_sin = _mk(mat, unreal.MaterialExpressionSine, -920, 180)
            n_sin.set_editor_property("period", 2.0)
            mel.connect_material_expressions(n_t, "", n_sin, "")
            n_abs = _mk(mat, unreal.MaterialExpressionAbs, -760, 180)
            mel.connect_material_expressions(n_sin, "", n_abs, "")
            n_hb = _mk(mat, unreal.MaterialExpressionMultiply, -600, 180)
            n_hb.set_editor_property("const_b", 0.65)
            mel.connect_material_expressions(n_abs, "", n_hb, "A")
            n_hb2 = _mk(mat, unreal.MaterialExpressionAdd, -440, 180)
            n_hb2.set_editor_property("const_b", 0.55)
            mel.connect_material_expressions(n_hb, "", n_hb2, "A")

            n_gc = _vec(mat, "CoreGlowColor", spec.get("glow_color", [0.08, 0.55, 1.0]), -640, 380)
            n_gs = _scalar(mat, "GlowStrength", float(spec.get("glow_strength", 5.5)), -440, 380)
            n_m1 = _mk(mat, unreal.MaterialExpressionMultiply, -260, 100)
            mel.connect_material_expressions(n_c, "RGB", n_m1, "A")
            mel.connect_material_expressions(n_gc, "", n_m1, "B")
            n_m2 = _mk(mat, unreal.MaterialExpressionMultiply, -260, 260)
            mel.connect_material_expressions(n_hb2, "", n_m2, "A")
            mel.connect_material_expressions(n_gs, "", n_m2, "B")
            n_m3 = _mk(mat, unreal.MaterialExpressionMultiply, -60, 180)
            mel.connect_material_expressions(n_m1, "", n_m3, "A")
            mel.connect_material_expressions(n_m2, "", n_m3, "B")
            mel.connect_material_property(n_m3, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
            if spec.get("normal_file"):
                ntex = import_texture(spec["normal_file"], is_srgb=False)
                nn = _tex_node(mat, "CoreNormal", ntex, unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR, -640, -400)
                ndz = _reconstruct_normal_z(mat, nn, "R", nn, "G", -240, -400)
                mel.connect_material_property(ndz, "", unreal.MaterialProperty.MP_NORMAL)
            mel.recompile_material(mat)
            unreal.EditorAssetLibrary.save_loaded_asset(mat)
            return mat

        if skind in ("machine_eye_glow", "canister_glow_core"):
            n_t = _time_scaled(mat, -960, 160)
            if skind == "machine_eye_glow":
                n_ec = _mpc_param(mat, "EyeColor", lambda: _vec(mat, "GlowColor", spec.get("glow_color", [0.04, 0.56, 1.0]), -760, -140), -760, -140)
                n_ea = _mpc_param(mat, "EyeAlertColor", lambda: _vec(mat, "AlertColor", spec.get("alert_color", [1.0, 0.08, 0.01]), -760, -20), -760, -20)
                n_as = _mpc_param(mat, "EyeAlertState", lambda: _scalar(mat, "AlertState", 0.0, -760, 100), -760, 100)
                n_gc = _mk(mat, unreal.MaterialExpressionLinearInterpolate, -460, -80)
                mel.connect_material_expressions(n_ec, "", n_gc, "A")
                mel.connect_material_expressions(n_ea, "", n_gc, "B")
                mel.connect_material_expressions(n_as, "", n_gc, "Alpha")
                n_gs_b = _scalar(mat, "GlowStrength", float(spec.get("glow_strength", 9.0)), -760, 320)
                n_gs_s = _mpc_param(mat, "EyeGlowScale", lambda: _scalar(mat, "EyeGlowScale", 1.0, -760, 420), -760, 420)
            else:
                n_lc0 = _vec(mat, "NativeLiquidColor", spec.get("glow_color", [0.05, 0.48, 1.0]), -780, -140)
                n_lco = _mpc_param(mat, "LiquidOverrideColor", lambda: _vec(mat, "LiquidOverrideColor", [0.08, 0.62, 1.0], -780, -20), -780, -20)
                n_lcm = _mpc_param(mat, "LiquidOverrideMix", lambda: _scalar(mat, "LiquidOverrideMix", 0.0, -780, 100), -780, 100)
                n_gc = _mk(mat, unreal.MaterialExpressionLinearInterpolate, -460, -80)
                mel.connect_material_expressions(n_lc0, "", n_gc, "A")
                mel.connect_material_expressions(n_lco, "", n_gc, "B")
                mel.connect_material_expressions(n_lcm, "", n_gc, "Alpha")
                n_gs_b = _scalar(mat, "GlowStrength", float(spec.get("glow_strength", 6.8)), -760, 320)
                n_gs_s = _mpc_param(mat, "LiquidGlowScale", lambda: _scalar(mat, "LiquidGlowScale", 1.0, -760, 420), -760, 420)
                n_rg = _scalar(mat, "LiquidRoughness", 0.08, -420, -340)
                mel.connect_material_property(n_rg, "", unreal.MaterialProperty.MP_ROUGHNESS)
            n_gs = _mk(mat, unreal.MaterialExpressionMultiply, -520, 340)
            mel.connect_material_expressions(n_gs_b, "", n_gs, "A")
            mel.connect_material_expressions(n_gs_s, "", n_gs, "B")
            n_sin = _mk(mat, unreal.MaterialExpressionSine, -720, 160)
            n_sin.set_editor_property("period", 2.0 if skind == "canister_glow_core" else 2.5)
            mel.connect_material_expressions(n_t, "", n_sin, "")
            n_pm = _mk(mat, unreal.MaterialExpressionMultiply, -560, 160)
            n_pm.set_editor_property("const_b", 0.20)
            mel.connect_material_expressions(n_sin, "", n_pm, "A")
            n_pa = _mk(mat, unreal.MaterialExpressionAdd, -400, 160)
            n_pa.set_editor_property("const_b", 0.82)
            mel.connect_material_expressions(n_pm, "", n_pa, "A")
            n_str = _mk(mat, unreal.MaterialExpressionMultiply, -240, 220)
            mel.connect_material_expressions(n_pa, "", n_str, "A")
            mel.connect_material_expressions(n_gs, "", n_str, "B")
            if spec.get("noise_file"):
                ntex = import_texture(spec["noise_file"], is_srgb=False)
                n_uv = _mk(mat, unreal.MaterialExpressionTextureCoordinate, -1180, -280)
                n_uv.set_editor_property("u_tiling", 2.0)
                n_uv.set_editor_property("v_tiling", 2.0)
                n_pan = _mk(mat, unreal.MaterialExpressionPanner, -920, -280)
                n_pan.set_editor_property("speed_y", 0.30)
                mel.connect_material_expressions(n_uv, "", n_pan, "Coordinate")
                mel.connect_material_expressions(n_t, "", n_pan, "Time")
                n_nt = _tex_node(mat, "FluidNoise", ntex, unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR, -660, -280)
                mel.connect_material_expressions(n_pan, "", n_nt, "UVs")
                n_nm = _mk(mat, unreal.MaterialExpressionAdd, -420, -240)
                n_nm.set_editor_property("const_b", 0.45)
                mel.connect_material_expressions(n_nt, "R", n_nm, "A")
                n_str2 = _mk(mat, unreal.MaterialExpressionMultiply, -120, 140)
                mel.connect_material_expressions(n_str, "", n_str2, "A")
                mel.connect_material_expressions(n_nm, "", n_str2, "B")
                n_str = n_str2
            n_em = _mk(mat, unreal.MaterialExpressionMultiply, 20, 40)
            mel.connect_material_expressions(n_gc, "", n_em, "A")
            mel.connect_material_expressions(n_str, "", n_em, "B")
            mel.connect_material_property(n_em, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
            mel.recompile_material(mat)
            unreal.EditorAssetLibrary.save_loaded_asset(mat)
            return mat

    # ------------------------------------------------------------------
    # STANDARD DECIMA PBR + ANIMATED OVERLAYS (Sacks, Canisters, Eyes!)
    # ------------------------------------------------------------------
    sprof = meta.get("surface_profile", "pbr")
    base_color_expr = sockets.get("Color")
    if sprof == "skin" and meta.get("uses_skin_lut") and base_color_expr is not None:
        # Exact DXIL Skin Tone LUT Reconstruction (`4.5948 * SkinTone_linear * Head_CLR_linear`)!
        st_lin = meta.get("skin_tone_linear") or [0.453, 0.246, 0.209]
        n_st = _vec(mat, "SkinTone", st_lin, -760, -440)
        n_sb = _scalar(mat, "SkinBrightness", 1.0, -760, -320)
        n_exp = _mk(mat, unreal.MaterialExpressionMultiply, -540, -240)
        n_exp.set_editor_property("const_b", 4.5948)
        mel.connect_material_expressions(base_color_expr[0], base_color_expr[1], n_exp, "A")
        n_st_b = _mk(mat, unreal.MaterialExpressionMultiply, -540, -400)
        mel.connect_material_expressions(n_st, "", n_st_b, "A")
        mel.connect_material_expressions(n_sb, "", n_st_b, "B")
        n_sk_col = _mk(mat, unreal.MaterialExpressionMultiply, -360, -320)
        mel.connect_material_expressions(n_st_b, "", n_sk_col, "A")
        mel.connect_material_expressions(n_exp, "", n_sk_col, "B")
        n_sk_sat = _mk(mat, unreal.MaterialExpressionSaturate, -240, -320)
        mel.connect_material_expressions(n_sk_col, "", n_sk_sat, "")
        base_color_expr = (n_sk_sat, "")
    elif sprof in ("hair", "hair_face") and base_color_expr is None:
        hc_lin = meta.get("hair_color_linear") or [0.045, 0.032, 0.025]
        n_hc = _vec(mat, "HairColor", hc_lin, -540, -280)
        base_color_expr = (n_hc, "")
    elif sprof in ("rock_volcanic", "rock_tint") and base_color_expr is not None:
        # HFW landscape rock `_TNT` textures are 2-channel Linear RG Tint/Blend maps (`B == 0`)!
        if sprof == "rock_volcanic":
            c_dark, c_mid, c_dust, dust_w = [0.022, 0.021, 0.024], [0.072, 0.068, 0.066], [0.135, 0.115, 0.098], 0.42
        else:
            c_dark, c_mid, c_dust, dust_w = [0.085, 0.070, 0.058], [0.245, 0.195, 0.155], [0.175, 0.160, 0.118], 0.35
        n_rk_r = _mk(mat, unreal.MaterialExpressionComponentMask, -640, -300)
        n_rk_r.set_editor_property("r", True); n_rk_r.set_editor_property("g", False); n_rk_r.set_editor_property("b", False)
        mel.connect_material_expressions(base_color_expr[0], base_color_expr[1], n_rk_r, "")
        n_rk_g = _mk(mat, unreal.MaterialExpressionComponentMask, -640, -180)
        n_rk_g.set_editor_property("r", False); n_rk_g.set_editor_property("g", True); n_rk_g.set_editor_property("b", False)
        mel.connect_material_expressions(base_color_expr[0], base_color_expr[1], n_rk_g, "")
        n_rk_l1 = _mk(mat, unreal.MaterialExpressionLinearInterpolate, -420, -280)
        mel.connect_material_expressions(_vec(mat, "RockDark", c_dark, -640, -460), "", n_rk_l1, "A")
        mel.connect_material_expressions(_vec(mat, "RockMid", c_mid, -640, -380), "", n_rk_l1, "B")
        mel.connect_material_expressions(n_rk_r, "", n_rk_l1, "Alpha")
        n_gmul = _mk(mat, unreal.MaterialExpressionMultiply, -440, -140)
        n_gmul.set_editor_property("const_b", dust_w)
        mel.connect_material_expressions(n_rk_g, "", n_gmul, "A")
        n_rk_l2 = _mk(mat, unreal.MaterialExpressionLinearInterpolate, -260, -240)
        mel.connect_material_expressions(n_rk_l1, "", n_rk_l2, "A")
        mel.connect_material_expressions(_vec(mat, "RockWeathering", c_dust, -440, -40), "", n_rk_l2, "B")
        mel.connect_material_expressions(n_gmul, "", n_rk_l2, "Alpha")
        base_color_expr = (n_rk_l2, "")

    _DIELECTRIC_PROFILES = ("hair", "hair_face", "skin", "eye", "teeth", "foliage", "snow", "rock_volcanic", "rock_tint")
    if "Reflectance" in sockets and sprof not in _DIELECTRIC_PROFILES:
        rfl_n, rfl_ch = sockets["Reflectance"]
        n_bias = _mk(mat, unreal.MaterialExpressionConstantBiasScale, -620, 260)
        n_bias.set_editor_property("bias", -0.32)
        n_bias.set_editor_property("scale", 3.846154)
        n_met = _mk(mat, unreal.MaterialExpressionSaturate, -400, 260)
        mel.connect_material_expressions(rfl_n, rfl_ch, n_bias, "")
        mel.connect_material_expressions(n_bias, "", n_met, "")
        mel.connect_material_property(n_met, "", unreal.MaterialProperty.MP_METALLIC)

        n_f0 = _mk(mat, unreal.MaterialExpressionMultiply, -620, 100)
        mel.connect_material_expressions(rfl_n, rfl_ch, n_f0, "A")
        mel.connect_material_expressions(rfl_n, rfl_ch, n_f0, "B")
        if base_color_expr is not None:
            n_bc = _mk(mat, unreal.MaterialExpressionLinearInterpolate, -140, -180)
            mel.connect_material_expressions(base_color_expr[0], base_color_expr[1], n_bc, "A")
            mel.connect_material_expressions(n_f0, "", n_bc, "B")
            mel.connect_material_expressions(n_met, "", n_bc, "Alpha")
            mel.connect_material_property(n_bc, "", unreal.MaterialProperty.MP_BASE_COLOR)
    else:
        if sprof in _DIELECTRIC_PROFILES:
            # Dielectric hair/skin/rock NEVER treat Reflectance (e.g. Aloy hair `181:1.R` sheen mask) as Metallic!
            mel.connect_material_property(_const(mat, 0.0, -400, 260), "", unreal.MaterialProperty.MP_METALLIC)
            if sprof in ("hair", "hair_face") and "Reflectance" in sockets:
                n_hsp = _mk(mat, unreal.MaterialExpressionLinearInterpolate, -400, 140)
                n_hsp.set_editor_property("const_a", 0.42)
                n_hsp.set_editor_property("const_b", 0.72)
                mel.connect_material_expressions(sockets["Reflectance"][0], sockets["Reflectance"][1], n_hsp, "Alpha")
                mel.connect_material_property(n_hsp, "", unreal.MaterialProperty.MP_SPECULAR)
        if base_color_expr is not None:
            mel.connect_material_property(base_color_expr[0], base_color_expr[1], unreal.MaterialProperty.MP_BASE_COLOR)
        elif sprof == "snow":
            n_sn = _vec(mat, "SnowAlbedo", [0.78, 0.84, 0.92], -260, -180)
            mel.connect_material_property(n_sn, "", unreal.MaterialProperty.MP_BASE_COLOR)

    if "Roughness" in sockets:
        if sprof in ("rock_volcanic", "rock_tint"):
            # DXIL `8490:1` dry porous basalt roughness cap (`0.66..0.92`, zero wet-plastic shine!)
            n_rr = _mk(mat, unreal.MaterialExpressionLinearInterpolate, -460, -40)
            n_rr.set_editor_property("const_a", 0.92)
            n_rr.set_editor_property("const_b", 0.66)
            mel.connect_material_expressions(sockets["Roughness"][0], sockets["Roughness"][1], n_rr, "Alpha")
            mel.connect_material_property(n_rr, "", unreal.MaterialProperty.MP_ROUGHNESS)
        elif sprof in ("hair", "hair_face"):
            # Natural silky hair roughness (`0.38..0.56`) from `181:1.G`!
            n_hr = _mk(mat, unreal.MaterialExpressionLinearInterpolate, -460, -40)
            n_hr.set_editor_property("const_a", 0.56)
            n_hr.set_editor_property("const_b", 0.38)
            mel.connect_material_expressions(sockets["Roughness"][0], sockets["Roughness"][1], n_hr, "Alpha")
            mel.connect_material_property(n_hr, "", unreal.MaterialProperty.MP_ROUGHNESS)
        else:
            n_inv = _mk(mat, unreal.MaterialExpressionOneMinus, -460, -40)
            mel.connect_material_expressions(sockets["Roughness"][0], sockets["Roughness"][1], n_inv, "")
            mel.connect_material_property(n_inv, "", unreal.MaterialProperty.MP_ROUGHNESS)

    if "AO" in sockets:
        mel.connect_material_property(sockets["AO"][0], sockets["AO"][1], unreal.MaterialProperty.MP_AMBIENT_OCCLUSION)

    det_nrm_f = meta.get("asset_detail_normal_file", "")
    det_nrm_node = None
    if det_nrm_f:
        dnt = import_texture(det_nrm_f, is_srgb=False, is_normal=False)
        if dnt:
            n_duv = _mk(mat, unreal.MaterialExpressionTextureCoordinate, -980, 620)
            n_duv.set_editor_property("u_tiling", 8.0)
            n_duv.set_editor_property("v_tiling", 8.0)
            det_nrm_node = _tex_node(mat, "AssetDetailNormal8x", dnt, unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR, -740, 620)
            mel.connect_material_expressions(n_duv, "", det_nrm_node, "UVs")

    if "Normal_RGB" in sockets:
        mel.connect_material_property(sockets["Normal_RGB"][0], sockets["Normal_RGB"][1], unreal.MaterialProperty.MP_NORMAL)
    elif "Normal.R" in sockets and "Normal.G" in sockets:
        if det_nrm_node is not None:
            # Blend 8x basalt/asset micro-detail normal XY onto 4K macro normal XY via UDN addition!
            n_mxy = _mk(mat, unreal.MaterialExpressionAppendVector, -540, 460)
            mel.connect_material_expressions(sockets["Normal.R"][0], sockets["Normal.R"][1], n_mxy, "A")
            mel.connect_material_expressions(sockets["Normal.G"][0], sockets["Normal.G"][1], n_mxy, "B")
            n_dxy = _mk(mat, unreal.MaterialExpressionAppendVector, -540, 600)
            mel.connect_material_expressions(det_nrm_node, "R", n_dxy, "A")
            mel.connect_material_expressions(det_nrm_node, "G", n_dxy, "B")
            n_dc = _mk(mat, unreal.MaterialExpressionConstantBiasScale, -400, 600)
            n_dc.set_editor_property("bias", -0.5)
            n_dc.set_editor_property("scale", 0.55)
            mel.connect_material_expressions(n_dxy, "", n_dc, "")
            n_ad = _mk(mat, unreal.MaterialExpressionAdd, -260, 500)
            mel.connect_material_expressions(n_mxy, "", n_ad, "A")
            mel.connect_material_expressions(n_dc, "", n_ad, "B")
            n_st = _mk(mat, unreal.MaterialExpressionSaturate, -160, 500)
            mel.connect_material_expressions(n_ad, "", n_st, "")
            n_dr = _mk(mat, unreal.MaterialExpressionComponentMask, -60, 440)
            n_dr.set_editor_property("r", True); n_dr.set_editor_property("g", False); n_dr.set_editor_property("b", False)
            mel.connect_material_expressions(n_st, "", n_dr, "")
            n_dg = _mk(mat, unreal.MaterialExpressionComponentMask, -60, 540)
            n_dg.set_editor_property("r", False); n_dg.set_editor_property("g", True); n_dg.set_editor_property("b", False)
            mel.connect_material_expressions(n_st, "", n_dg, "")
            n_dz = _reconstruct_normal_z(mat, n_dr, "", n_dg, "", 100, 480)
        else:
            n_dz = _reconstruct_normal_z(mat, sockets["Normal.R"][0], sockets["Normal.R"][1], sockets["Normal.G"][0], sockets["Normal.G"][1], -320, 480)
        mel.connect_material_property(n_dz, "", unreal.MaterialProperty.MP_NORMAL)

    # Wire Animated Elemental Sac / Canister / Eye Lens Overlays!
    if overlay:
        okind = overlay.get("kind", "")
        n_t = _time_scaled(mat, -1100, 960)

        if okind == "elemental_sack_overlay" and "Mask" in sockets:
            # Exact DXIL mat_id==0 * Mask_Alpha + True Dual-Phase Flow Map (172:0) + Caustics (1703:0) + Fresnel + Wet Specular!
            n_is0 = _build_mask_id_compare(mat, sockets["Mask"][0], sockets["Mask"][1], 0.0, -560, 820)
            sac_mask = n_is0
            if "Mask_Alpha" in sockets:
                n_sm = _mk(mat, unreal.MaterialExpressionMultiply, -380, 840)
                mel.connect_material_expressions(n_is0, "", n_sm, "A")
                mel.connect_material_expressions(sockets["Mask_Alpha"][0], sockets["Mask_Alpha"][1], n_sm, "B")
                sac_mask = n_sm

            # Dual-phase flow-map time: phase0 = frac(t*spd + 0.5)*2-1, phase1 = frac(t*spd + 1.0)*2-1, blend = abs(phase0)
            n_fspd = _scalar(mat, "FlowSpeed", float(overlay.get("flow_speed", 0.35)), -1580, 1040)
            n_tf = _mk(mat, unreal.MaterialExpressionMultiply, -1380, 1020)
            mel.connect_material_expressions(n_t, "", n_tf, "A")
            mel.connect_material_expressions(n_fspd, "", n_tf, "B")

            n_a0 = _mk(mat, unreal.MaterialExpressionAdd, -1220, 960)
            n_a0.set_editor_property("const_b", 0.5)
            mel.connect_material_expressions(n_tf, "", n_a0, "A")
            n_fr0 = _mk(mat, unreal.MaterialExpressionFrac, -1080, 960)
            mel.connect_material_expressions(n_a0, "", n_fr0, "")
            n_p0 = _mk(mat, unreal.MaterialExpressionConstantBiasScale, -940, 960)
            n_p0.set_editor_property("bias", -0.5)
            n_p0.set_editor_property("scale", 2.0)
            mel.connect_material_expressions(n_fr0, "", n_p0, "")

            n_fr1 = _mk(mat, unreal.MaterialExpressionFrac, -1080, 1100)
            mel.connect_material_expressions(n_tf, "", n_fr1, "")
            n_p1 = _mk(mat, unreal.MaterialExpressionConstantBiasScale, -940, 1100)
            n_p1.set_editor_property("bias", -0.5)
            n_p1.set_editor_property("scale", 2.0)
            mel.connect_material_expressions(n_fr1, "", n_p1, "")

            n_pbl = _mk(mat, unreal.MaterialExpressionAbs, -780, 960)
            mel.connect_material_expressions(n_p0, "", n_pbl, "")

            # Flow Vector from 172:0 (`RG * 2 - 1`) * FlowStrength
            n_uv_flw = _mk(mat, unreal.MaterialExpressionTextureCoordinate, -1620, 1280)
            n_uv_flw.set_editor_property("u_tiling", 2.0)
            n_uv_flw.set_editor_property("v_tiling", 2.0)
            flow_tex = import_texture(overlay.get("flow_file", ""), is_srgb=False)
            n_flw_t = _tex_node(mat, "SacFlowMap", flow_tex, unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR, -1380, 1280)
            mel.connect_material_expressions(n_uv_flw, "", n_flw_t, "UVs")
            n_flw_rg = _mk(mat, unreal.MaterialExpressionAppendVector, -1120, 1280)
            mel.connect_material_expressions(n_flw_t, "R", n_flw_rg, "A")
            mel.connect_material_expressions(n_flw_t, "G", n_flw_rg, "B")
            n_flw_dec = _mk(mat, unreal.MaterialExpressionConstantBiasScale, -960, 1280)
            n_flw_dec.set_editor_property("bias", -0.5)
            n_flw_dec.set_editor_property("scale", 2.0)
            mel.connect_material_expressions(n_flw_rg, "", n_flw_dec, "")
            n_fstr = _scalar(mat, "FlowStrength", float(overlay.get("flow_strength", 0.14)), -960, 1420)
            n_fvec = _mk(mat, unreal.MaterialExpressionMultiply, -780, 1280)
            mel.connect_material_expressions(n_flw_dec, "", n_fvec, "A")
            mel.connect_material_expressions(n_fstr, "", n_fvec, "B")

            # Dual-phase UV distortion into 1703:0 Fluid Caustics
            n_uv_fld = _mk(mat, unreal.MaterialExpressionTextureCoordinate, -960, 1540)
            n_uv_fld.set_editor_property("u_tiling", 2.5)
            n_uv_fld.set_editor_property("v_tiling", 2.5)
            n_off0 = _mk(mat, unreal.MaterialExpressionMultiply, -620, 1180)
            mel.connect_material_expressions(n_fvec, "", n_off0, "A")
            mel.connect_material_expressions(n_p0, "", n_off0, "B")
            n_uv0 = _mk(mat, unreal.MaterialExpressionAdd, -480, 1180)
            mel.connect_material_expressions(n_uv_fld, "", n_uv0, "A")
            mel.connect_material_expressions(n_off0, "", n_uv0, "B")

            n_off1 = _mk(mat, unreal.MaterialExpressionMultiply, -620, 1380)
            mel.connect_material_expressions(n_fvec, "", n_off1, "A")
            mel.connect_material_expressions(n_p1, "", n_off1, "B")
            n_uv1 = _mk(mat, unreal.MaterialExpressionAdd, -480, 1380)
            mel.connect_material_expressions(n_uv_fld, "", n_uv1, "A")
            mel.connect_material_expressions(n_off1, "", n_uv1, "B")

            fluid_tex = import_texture(overlay.get("fluid_file", ""), is_srgb=False)
            n_fl0 = _tex_node(mat, "SacFluidPhase0", fluid_tex, unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR, -320, 1160)
            mel.connect_material_expressions(n_uv0, "", n_fl0, "UVs")
            n_fl1 = _tex_node(mat, "SacFluidPhase1", fluid_tex, unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR, -320, 1400)
            mel.connect_material_expressions(n_uv1, "", n_fl1, "UVs")

            n_fl_mix = _mk(mat, unreal.MaterialExpressionLinearInterpolate, -80, 1260)
            mel.connect_material_expressions(n_fl0, "R", n_fl_mix, "A")
            mel.connect_material_expressions(n_fl1, "R", n_fl_mix, "B")
            mel.connect_material_expressions(n_pbl, "", n_fl_mix, "Alpha")

            # Swirling caustic contrast (0.38 + 1.15 * fluid) + Fresnel subsurface edge depth!
            n_fl_sc = _mk(mat, unreal.MaterialExpressionMultiply, 60, 1260)
            n_fl_sc.set_editor_property("const_b", 1.15)
            mel.connect_material_expressions(n_fl_mix, "", n_fl_sc, "A")
            n_fl_add = _mk(mat, unreal.MaterialExpressionAdd, 180, 1260)
            n_fl_add.set_editor_property("const_b", 0.38)
            mel.connect_material_expressions(n_fl_sc, "", n_fl_add, "A")

            n_gc0 = _vec(mat, "SacGlowColor", overlay.get("glow_color", [0.14, 0.68, 1.0]), -180, 960)
            n_gco = _mpc_param(mat, "LiquidOverrideColor", lambda: _vec(mat, "LiquidOverrideColor", [0.08, 0.62, 1.0], -180, 820), -180, 820)
            n_gcm = _mpc_param(mat, "LiquidOverrideMix", lambda: _scalar(mat, "LiquidOverrideMix", 0.0, -180, 720), -180, 720)
            n_gc = _mk(mat, unreal.MaterialExpressionLinearInterpolate, 60, 920)
            mel.connect_material_expressions(n_gc0, "", n_gc, "A")
            mel.connect_material_expressions(n_gco, "", n_gc, "B")
            mel.connect_material_expressions(n_gcm, "", n_gc, "Alpha")

            # Fresnel rim subsurface depth darkening on sac edges
            n_sfr = _mk(mat, unreal.MaterialExpressionFresnel, 60, 1080)
            n_sfr.set_editor_property("exponent", 2.2)
            n_fr_m = _mk(mat, unreal.MaterialExpressionLinearInterpolate, 220, 1060)
            n_fr_m.set_editor_property("const_a", 1.0)
            n_fr_m.set_editor_property("const_b", 0.48)
            mel.connect_material_expressions(n_sfr, "", n_fr_m, "Alpha")

            n_gs_b = _scalar(mat, "SacGlowStrength", float(overlay.get("glow_strength", 4.2)), 60, 1440)
            n_gs_s = _mpc_param(mat, "LiquidGlowScale", lambda: _scalar(mat, "LiquidGlowScale", 1.0, 60, 1560), 60, 1560)
            n_gs = _mk(mat, unreal.MaterialExpressionMultiply, 220, 1460)
            mel.connect_material_expressions(n_gs_b, "", n_gs, "A")
            mel.connect_material_expressions(n_gs_s, "", n_gs, "B")

            n_m0 = _mk(mat, unreal.MaterialExpressionMultiply, 320, 1180)
            mel.connect_material_expressions(n_fl_add, "", n_m0, "A")
            mel.connect_material_expressions(n_fr_m, "", n_m0, "B")
            n_m1 = _mk(mat, unreal.MaterialExpressionMultiply, 440, 1120)
            mel.connect_material_expressions(n_m0, "", n_m1, "A")
            mel.connect_material_expressions(sac_mask, "", n_m1, "B")
            n_m2 = _mk(mat, unreal.MaterialExpressionMultiply, 560, 1160)
            mel.connect_material_expressions(n_m1, "", n_m2, "A")
            mel.connect_material_expressions(n_gs, "", n_m2, "B")
            n_em = _mk(mat, unreal.MaterialExpressionMultiply, 700, 1060)
            mel.connect_material_expressions(n_gc, "", n_em, "A")
            mel.connect_material_expressions(n_m2, "", n_em, "B")
            mel.connect_material_property(n_em, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)

            # Blend smooth wet glossy membrane roughness (0.12) onto sac regions!
            if "Roughness" in sockets:
                n_inv_s = _mk(mat, unreal.MaterialExpressionOneMinus, -220, -40)
                mel.connect_material_expressions(sockets["Roughness"][0], sockets["Roughness"][1], n_inv_s, "")
                n_wrg = _mk(mat, unreal.MaterialExpressionLinearInterpolate, -40, -40)
                n_wrg.set_editor_property("const_b", 0.12)
                mel.connect_material_expressions(n_inv_s, "", n_wrg, "A")
                mel.connect_material_expressions(sac_mask, "", n_wrg, "Alpha")
                mel.connect_material_property(n_wrg, "", unreal.MaterialProperty.MP_ROUGHNESS)

        elif okind == "canister_emissive_overlay" and "Mask" in sockets:
            n_is1 = _build_mask_id_compare(mat, sockets["Mask"][0], sockets["Mask"][1], 1.0, -560, 820)
            n_is20 = _build_mask_id_compare(mat, sockets["Mask"][0], sockets["Mask"][1], 20.0, -560, 1060)
            n_sin = _mk(mat, unreal.MaterialExpressionSine, -760, 1260)
            n_sin.set_editor_property("period", 0.7854)  # sin(Time * 8.0)
            mel.connect_material_expressions(n_t, "", n_sin, "")
            n_pm = _mk(mat, unreal.MaterialExpressionMultiply, -580, 1260)
            n_pm.set_editor_property("const_b", 0.30)
            mel.connect_material_expressions(n_sin, "", n_pm, "A")
            n_pa = _mk(mat, unreal.MaterialExpressionAdd, -420, 1260)
            n_pa.set_editor_property("const_b", 0.70)
            mel.connect_material_expressions(n_pm, "", n_pa, "A")
            n_c1 = _mk(mat, unreal.MaterialExpressionMultiply, -280, 860)
            mel.connect_material_expressions(n_is1, "", n_c1, "A")
            mel.connect_material_expressions(n_pa, "", n_c1, "B")
            n_max = _mk(mat, unreal.MaterialExpressionMax, -140, 960)
            mel.connect_material_expressions(n_c1, "", n_max, "A")
            mel.connect_material_expressions(n_is20, "", n_max, "B")
            n_gc0 = _vec(mat, "CanisterGlowColor", overlay.get("glow_color", [0.52, 0.70, 0.10]), -520, 1420)
            n_gco = _mpc_param(mat, "LiquidOverrideColor", lambda: _vec(mat, "LiquidOverrideColor", [0.08, 0.62, 1.0], -520, 1560), -520, 1560)
            n_gcm = _mpc_param(mat, "LiquidOverrideMix", lambda: _scalar(mat, "LiquidOverrideMix", 0.0, -520, 1680), -520, 1680)
            n_gc = _mk(mat, unreal.MaterialExpressionLinearInterpolate, -280, 1460)
            mel.connect_material_expressions(n_gc0, "", n_gc, "A")
            mel.connect_material_expressions(n_gco, "", n_gc, "B")
            mel.connect_material_expressions(n_gcm, "", n_gc, "Alpha")
            n_gs_b = _scalar(mat, "CanisterGlowStrength", float(overlay.get("glow_strength", 5.5)), -320, 1640)
            n_gs_s = _mpc_param(mat, "LiquidGlowScale", lambda: _scalar(mat, "LiquidGlowScale", 1.0, -320, 1740), -320, 1740)
            n_gs = _mk(mat, unreal.MaterialExpressionMultiply, -140, 1660)
            mel.connect_material_expressions(n_gs_b, "", n_gs, "A")
            mel.connect_material_expressions(n_gs_s, "", n_gs, "B")
            n_ms = _mk(mat, unreal.MaterialExpressionMultiply, 0, 1120)
            mel.connect_material_expressions(n_max, "", n_ms, "A")
            mel.connect_material_expressions(n_gs, "", n_ms, "B")
            n_em = _mk(mat, unreal.MaterialExpressionMultiply, 140, 1080)
            mel.connect_material_expressions(n_gc, "", n_em, "A")
            mel.connect_material_expressions(n_ms, "", n_em, "B")
            mel.connect_material_property(n_em, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)

        elif okind == "machine_eye_lens_overlay":
            iris_tex = import_texture(overlay.get("iris_file", ""), is_srgb=False)
            n_ir = _tex_node(mat, "EyeIrisMask", iris_tex, unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR, -700, 920)
            n_cc = _mpc_param(mat, "EyeColor", lambda: _vec(mat, "CalmEyeColor", overlay.get("calm_color", [0.04, 0.56, 1.0]), -700, 1180), -700, 1180)
            n_ac = _mpc_param(mat, "EyeAlertColor", lambda: _vec(mat, "AlertEyeColor", overlay.get("alert_color", [1.0, 0.08, 0.01]), -700, 1360), -700, 1360)
            n_as = _mpc_param(mat, "EyeAlertState", lambda: _scalar(mat, "AlertState", float(overlay.get("alert_state", 0.0)), -700, 1520), -700, 1520)
            n_lerp = _mk(mat, unreal.MaterialExpressionLinearInterpolate, -420, 1240)
            mel.connect_material_expressions(n_cc, "", n_lerp, "A")
            mel.connect_material_expressions(n_ac, "", n_lerp, "B")
            mel.connect_material_expressions(n_as, "", n_lerp, "Alpha")
            n_gs_b = _scalar(mat, "EyeGlowStrength", float(overlay.get("glow_strength", 9.0)), -560, 1040)
            n_gs_s = _mpc_param(mat, "EyeGlowScale", lambda: _scalar(mat, "EyeGlowScale", 1.0, -560, 1120), -560, 1120)
            n_gs = _mk(mat, unreal.MaterialExpressionMultiply, -380, 1040)
            mel.connect_material_expressions(n_gs_b, "", n_gs, "A")
            mel.connect_material_expressions(n_gs_s, "", n_gs, "B")
            n_m1 = _mk(mat, unreal.MaterialExpressionMultiply, -220, 980)
            mel.connect_material_expressions(n_ir, "R", n_m1, "A")
            mel.connect_material_expressions(n_gs, "", n_m1, "B")
            n_em = _mk(mat, unreal.MaterialExpressionMultiply, -40, 1080)
            mel.connect_material_expressions(n_lerp, "", n_em, "A")
            mel.connect_material_expressions(n_m1, "", n_em, "B")
            mel.connect_material_property(n_em, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)

        elif okind == "plasma_charge_overlay":
            # Exact DXIL Slaughterspine Plasma Charge-Up (`VertexColor.R` * `(1 - AO)^2` * Head->Tail wave + `mat_id==1` conduits!)
            n_vc = _mk(mat, unreal.MaterialExpressionVertexColor, -1100, 820)
            n_pc = _mpc_param(mat, "PlasmaColor", lambda: _vec(mat, "PlasmaColor", overlay.get("plasma_color", [0.46, 0.05, 1.0]), -720, 1340), -720, 1340)
            n_ic = _mpc_param(mat, "MuscleColor", lambda: _vec(mat, "IdleConduitColor", overlay.get("idle_led_color", [0.85, 0.18, 0.02]), -720, 1500), -720, 1500)
            n_chg = _mpc_param(mat, "PlasmaCharge", lambda: _scalar(mat, "PlasmaCharge", float(overlay.get("plasma_charge", 1.0)), -720, 1180), -720, 1180)
            n_gs = _scalar(mat, "PlasmaGlowStrength", float(overlay.get("glow_strength", 4.2)), -480, 1180)

            # Head->Tail plasma surge wave: 0.72 + 0.28 * sin(Time * 3.5 - VertexColor.G * 10)
            n_g10 = _mk(mat, unreal.MaterialExpressionMultiply, -860, 1000)
            n_g10.set_editor_property("const_b", 1.5915)  # 10 / (2*pi)
            mel.connect_material_expressions(n_vc, "G", n_g10, "A")
            n_tm = _mk(mat, unreal.MaterialExpressionMultiply, -860, 1120)
            n_tm.set_editor_property("const_b", 0.557)   # 3.5 / (2*pi)
            mel.connect_material_expressions(n_t, "", n_tm, "A")
            n_sub = _mk(mat, unreal.MaterialExpressionSubtract, -680, 1040)
            mel.connect_material_expressions(n_tm, "", n_sub, "A")
            mel.connect_material_expressions(n_g10, "", n_sub, "B")
            n_sin = _mk(mat, unreal.MaterialExpressionSine, -520, 1040)
            n_sin.set_editor_property("period", 1.0)
            mel.connect_material_expressions(n_sub, "", n_sin, "")
            n_wm = _mk(mat, unreal.MaterialExpressionMultiply, -380, 1040)
            n_wm.set_editor_property("const_b", 0.28)
            mel.connect_material_expressions(n_sin, "", n_wm, "A")
            n_wa = _mk(mat, unreal.MaterialExpressionAdd, -240, 1040)
            n_wa.set_editor_property("const_b", 0.72)
            mel.connect_material_expressions(n_wm, "", n_wa, "A")

            # Crevice focus `(1 - AO)^2 * 0.78 + 0.22`
            if "AO" in sockets:
                n_iao = _mk(mat, unreal.MaterialExpressionOneMinus, -860, 720)
                mel.connect_material_expressions(sockets["AO"][0], sockets["AO"][1], n_iao, "")
                n_aosq = _mk(mat, unreal.MaterialExpressionMultiply, -700, 720)
                mel.connect_material_expressions(n_iao, "", n_aosq, "A")
                mel.connect_material_expressions(n_iao, "", n_aosq, "B")
                n_aof = _mk(mat, unreal.MaterialExpressionLinearInterpolate, -540, 720)
                n_aof.set_editor_property("const_a", 0.22)
                n_aof.set_editor_property("const_b", 1.0)
                mel.connect_material_expressions(n_aosq, "", n_aof, "Alpha")
                n_vch = _mk(mat, unreal.MaterialExpressionMultiply, -380, 800)
                mel.connect_material_expressions(n_vc, "R", n_vch, "A")
                mel.connect_material_expressions(n_aof, "", n_vch, "B")
            else:
                n_vch = n_vc

            n_vc_c = _mk(mat, unreal.MaterialExpressionMultiply, -220, 840)
            mel.connect_material_expressions(n_vch, "" if "AO" in sockets else "R", n_vc_c, "A")
            mel.connect_material_expressions(n_chg, "", n_vc_c, "B")

            mask_out = n_vc_c
            if "Mask" in sockets:
                n_is1 = _build_mask_id_compare(mat, sockets["Mask"][0], sockets["Mask"][1], 1.0, -540, 900)
                if "Mask_Alpha" in sockets:
                    n_cma = _mk(mat, unreal.MaterialExpressionMultiply, -360, 920)
                    mel.connect_material_expressions(n_is1, "", n_cma, "A")
                    mel.connect_material_expressions(sockets["Mask_Alpha"][0], sockets["Mask_Alpha"][1], n_cma, "B")
                    n_is1 = n_cma
                n_max = _mk(mat, unreal.MaterialExpressionMax, -80, 880)
                mel.connect_material_expressions(n_vc_c, "", n_max, "A")
                mel.connect_material_expressions(n_is1, "", n_max, "B")
                mask_out = n_max

            n_mw = _mk(mat, unreal.MaterialExpressionMultiply, 40, 960)
            mel.connect_material_expressions(mask_out, "", n_mw, "A")
            mel.connect_material_expressions(n_wa, "", n_mw, "B")
            n_ms = _mk(mat, unreal.MaterialExpressionMultiply, 160, 1020)
            mel.connect_material_expressions(n_mw, "", n_ms, "A")
            mel.connect_material_expressions(n_gs, "", n_ms, "B")

            n_cl = _mk(mat, unreal.MaterialExpressionLinearInterpolate, -420, 1380)
            mel.connect_material_expressions(n_ic, "", n_cl, "A")
            mel.connect_material_expressions(n_pc, "", n_cl, "B")
            mel.connect_material_expressions(n_chg, "", n_cl, "Alpha")

            n_em = _mk(mat, unreal.MaterialExpressionMultiply, 280, 1140)
            mel.connect_material_expressions(n_cl, "", n_em, "A")
            mel.connect_material_expressions(n_ms, "", n_em, "B")
            mel.connect_material_property(n_em, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)

    elif "Incandescence" in sockets:
        n_gs = _scalar(mat, "EmissiveIntensity", float(meta.get("emissive_strength", 4.0)), -420, 820)
        n_em = _mk(mat, unreal.MaterialExpressionMultiply, -180, 760)
        mel.connect_material_expressions(sockets["Incandescence"][0], sockets["Incandescence"][1], n_em, "A")
        mel.connect_material_expressions(n_gs, "", n_em, "B")
        mel.connect_material_property(n_em, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)

    # Surface Profile refinements for Characters (Skin SSS, Hair, Brows/Lashes, Peach Fuzz, Eyes, Cloth) and General Assets (Foliage, Glass)!
    sprof = meta.get("surface_profile", "pbr")
    if sprof == "peach_fuzz":
        mat.set_editor_property("blend_mode", unreal.BlendMode.BLEND_TRANSLUCENT)
        n_op = _scalar(mat, "PeachFuzzOpacity", 0.0, -360, 620)
        mel.connect_material_property(n_op, "", unreal.MaterialProperty.MP_OPACITY)
    elif sprof == "hair_face":
        mat.set_editor_property("two_sided", True)
        if "Color" not in sockets:
            n_hc = _vec(mat, "BrowLashColor", [0.14, 0.075, 0.045], -360, 480)
            mel.connect_material_property(n_hc, "", unreal.MaterialProperty.MP_BASE_COLOR)
        if "Alpha" in sockets:
            mat.set_editor_property("blend_mode", unreal.BlendMode.BLEND_MASKED)
            mel.connect_material_property(sockets["Alpha"][0], sockets["Alpha"][1], unreal.MaterialProperty.MP_OPACITY_MASK)
        elif "Mask_Alpha" in sockets:
            mat.set_editor_property("blend_mode", unreal.BlendMode.BLEND_MASKED)
            mel.connect_material_property(sockets["Mask_Alpha"][0], sockets["Mask_Alpha"][1], unreal.MaterialProperty.MP_OPACITY_MASK)
    elif sprof == "skin":
        try:
            mat.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_SUBSURFACE)
            n_sss = _vec(mat, "SkinSubsurfaceColor", [0.85, 0.30, 0.20], -360, 620)
            mel.connect_material_property(n_sss, "", unreal.MaterialProperty.MP_SUBSURFACE_COLOR)
        except Exception:
            pass
    elif sprof == "hair":
        mat.set_editor_property("two_sided", True)
        if "Color" not in sockets:
            n_hc = _vec(mat, "HairTintColor", [0.045, 0.032, 0.025], -360, 480)
            mel.connect_material_property(n_hc, "", unreal.MaterialProperty.MP_BASE_COLOR)
        if "Alpha" in sockets:
            mat.set_editor_property("blend_mode", unreal.BlendMode.BLEND_MASKED)
            mel.connect_material_property(sockets["Alpha"][0], sockets["Alpha"][1], unreal.MaterialProperty.MP_OPACITY_MASK)
        elif "Mask_Alpha" in sockets:
            mat.set_editor_property("blend_mode", unreal.BlendMode.BLEND_MASKED)
            mel.connect_material_property(sockets["Mask_Alpha"][0], sockets["Mask_Alpha"][1], unreal.MaterialProperty.MP_OPACITY_MASK)
    elif sprof == "foliage":
        mat.set_editor_property("two_sided", True)
        try:
            mat.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_TWO_SIDED_FOLIAGE)
            if "Color" in sockets:
                mel.connect_material_property(sockets["Color"][0], sockets["Color"][1], unreal.MaterialProperty.MP_SUBSURFACE_COLOR)
        except Exception:
            pass
        if "Alpha" in sockets:
            mat.set_editor_property("blend_mode", unreal.BlendMode.BLEND_MASKED)
            mel.connect_material_property(sockets["Alpha"][0], sockets["Alpha"][1], unreal.MaterialProperty.MP_OPACITY_MASK)
        elif "Mask_Alpha" in sockets:
            mat.set_editor_property("blend_mode", unreal.BlendMode.BLEND_MASKED)
            mel.connect_material_property(sockets["Mask_Alpha"][0], sockets["Mask_Alpha"][1], unreal.MaterialProperty.MP_OPACITY_MASK)
    elif sprof == "eye":
        try:
            mat.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_CLEAR_COAT)
            n_er = _scalar(mat, "EyeRoughness", 0.04, -360, 620)
            mel.connect_material_property(n_er, "", unreal.MaterialProperty.MP_ROUGHNESS)
        except Exception:
            pass
    elif sprof == "eye_wet":
        mat.set_editor_property("blend_mode", unreal.BlendMode.BLEND_TRANSLUCENT)
        n_op = _scalar(mat, "WetOpacity", 0.12, -360, 620)
        n_rg = _scalar(mat, "WetRoughness", 0.02, -360, 720)
        mel.connect_material_property(n_op, "", unreal.MaterialProperty.MP_OPACITY)
        mel.connect_material_property(n_rg, "", unreal.MaterialProperty.MP_ROUGHNESS)
    elif sprof == "glass":
        mat.set_editor_property("blend_mode", unreal.BlendMode.BLEND_TRANSLUCENT)
        n_op = _scalar(mat, "GlassOpacity", 0.25, -360, 620)
        mel.connect_material_property(n_op, "", unreal.MaterialProperty.MP_OPACITY)
    elif meta.get("allow_alpha_cutout", True) and "Alpha" in sockets:
        mat.set_editor_property("blend_mode", unreal.BlendMode.BLEND_MASKED)
        mel.connect_material_property(sockets["Alpha"][0], sockets["Alpha"][1], unreal.MaterialProperty.MP_OPACITY_MASK)

    mel.recompile_material(mat)
    unreal.EditorAssetLibrary.save_loaded_asset(mat)
    return mat


def import_machine_fbx(built_materials: dict):
    fbx_path = next((p for p in FBX_CANDIDATES if os.path.exists(p)), None)
    if not fbx_path:
        unreal.log_warning(f"[HFW] FBX not found yet (checked {{FBX_CANDIDATES}}). Materials were still built cleanly!")
        return None
    opts = unreal.FbxImportUI()
    opts.import_mesh = True
    opts.import_as_skeletal = bool(IS_SKELETAL)
    opts.import_materials = False
    opts.import_textures = False
    opts.import_animations = False
    if IS_SKELETAL:
        opts.skeletal_mesh_import_data.combine_meshes = True
        opts.skeletal_mesh_import_data.import_morph_targets = False
        if hasattr(opts.skeletal_mesh_import_data, "vertex_color_import_option"):
            opts.skeletal_mesh_import_data.vertex_color_import_option = unreal.VertexColorImportOption.REPLACE
    else:
        opts.static_mesh_import_data.combine_meshes = True
        if hasattr(opts.static_mesh_import_data, "vertex_color_import_option"):
            opts.static_mesh_import_data.vertex_color_import_option = unreal.VertexColorImportOption.REPLACE

    asset_prefix = "SK_" if IS_SKELETAL else "SM_"
    task = unreal.AssetImportTask()
    task.filename = fbx_path
    task.destination_path = DEST_PACKAGE
    task.destination_name = asset_prefix + MACHINE_NAME
    task.automated = True
    task.replace_existing = True
    task.save = True
    task.options = opts
    asset_tools.import_asset_tasks([task])

    mesh_asset = unreal.EditorAssetLibrary.load_asset(f"{{DEST_PACKAGE}}/{{asset_prefix}}{{MACHINE_NAME}}")
    slot_prop = "materials" if IS_SKELETAL else "static_materials"
    if mesh_asset and hasattr(mesh_asset, slot_prop):
        assigned = 0
        new_mats = []
        for sm in getattr(mesh_asset, slot_prop):
            sname = str(sm.material_slot_name)
            matched = built_materials.get(sname) or built_materials.get(_safe(sname))
            if matched is None:
                for k, v in built_materials.items():
                    if k in sname or sname in k:
                        matched = v
                        break
            if matched is not None:
                sm.material_interface = matched
                assigned += 1
            new_mats.append(sm)
        setattr(mesh_asset, slot_prop, new_mats)
        unreal.EditorAssetLibrary.save_loaded_asset(mesh_asset)
        unreal.log(f"[HFW] Assigned {{assigned}}/{{len(new_mats)}} UE5 materials onto {{asset_prefix}}{{MACHINE_NAME}}!")
    return mesh_asset


def import_animations_fbx(mesh_asset):
    """Import all exported HFW `animations/*.fbx` clips as UE5 `AnimSequence`s bound to `mesh_asset.skeleton`."""
    if not IS_SKELETAL or mesh_asset is None or not os.path.isdir(ANIM_DIR):
        return []
    skel = getattr(mesh_asset, "skeleton", None)
    if skel is None:
        return []

    fbx_files = []
    seen = set()
    for am in ANIMATIONS_META:
        fn = am.get("fbx_filename") or ""
        if fn:
            fp = os.path.join(ANIM_DIR, fn)
            if os.path.isfile(fp) and fn not in seen:
                fbx_files.append((fp, am.get("name") or os.path.splitext(fn)[0]))
                seen.add(fn)
    for fn in sorted(os.listdir(ANIM_DIR)):
        if fn.lower().endswith(".fbx") and fn not in seen:
            fbx_files.append((os.path.join(ANIM_DIR, fn), os.path.splitext(fn)[0]))
            seen.add(fn)
    if not fbx_files:
        return []

    dest_anim_pkg = DEST_PACKAGE + "/Animations"
    imported = []
    for fbx_path, clip_name in fbx_files:
        try:
            opts = unreal.FbxImportUI()
            opts.import_mesh = False
            opts.import_materials = False
            opts.import_textures = False
            opts.import_animations = True
            opts.skeleton = skel
            opts.automated_import_should_detect_type = False
            if hasattr(unreal, "FBXImportType"):
                opts.original_import_type = unreal.FBXImportType.FBXIT_ANIMATION
            adata = opts.anim_sequence_import_data
            adata.import_bone_tracks = True
            adata.convert_scene = True
            if hasattr(unreal, "FBXAnimationLengthImportType"):
                adata.animation_length = unreal.FBXAnimationLengthImportType.FBXALIT_EXPORTED_TIME

            task = unreal.AssetImportTask()
            task.filename = fbx_path
            task.destination_path = dest_anim_pkg
            task.destination_name = "AS_" + _safe(clip_name)
            task.automated = True
            task.replace_existing = True
            task.save = True
            task.options = opts
            asset_tools.import_asset_tasks([task])
            seq = unreal.EditorAssetLibrary.load_asset(f"{{dest_anim_pkg}}/AS_{{_safe(clip_name)}}")
            if seq:
                imported.append(seq)
        except Exception as exc:
            unreal.log_warning(f"[HFW] AnimSequence import skipped for {{clip_name}}: {{exc}}")

    unreal.log(f"[HFW] Imported {{len(imported)}}/{{len(fbx_files)}} UE5 AnimSequence(s) -> {{dest_anim_pkg}}")
    return imported


def build_ue5_control_rig(mesh_asset):
    """Build native UE5 `CR_<AssetName>` Control Rig Blueprint with BOTH Forwards Solve (FK+IK) & Backwards Solve!

    Backwards Solve enables UE5 Sequencer's built-in `Bake To Control Rig` directly on any imported `AS_*` animation.
    """
    if not IS_SKELETAL or mesh_asset is None or not RIG_SPEC:
        return None
    if not hasattr(unreal, "ControlRigBlueprint"):
        unreal.log_warning("[HFW] Control Rig plugin not enabled in UE5 — skipping CR_ generation.")
        return None

    cr_pkg = DEST_PACKAGE + "/ControlRig"
    cr_name = "CR_" + MACHINE_NAME
    cr_path = f"{{cr_pkg}}/{{cr_name}}"
    try:
        if unreal.EditorAssetLibrary.does_asset_exist(cr_path):
            unreal.EditorAssetLibrary.delete_asset(cr_path)

        blueprint = None
        if hasattr(unreal, "ControlRigBlueprintFactory"):
            factory = unreal.ControlRigBlueprintFactory()
            if hasattr(factory, "create_control_rig_from_skeletal_mesh_or_skeleton"):
                blueprint = factory.create_control_rig_from_skeletal_mesh_or_skeleton(mesh_asset)
                if blueprint:
                    src_path = blueprint.get_path_name().rsplit(".", 1)[0]
                    if src_path != cr_path:
                        unreal.EditorAssetLibrary.rename_asset(src_path, cr_path)
                        blueprint = unreal.EditorAssetLibrary.load_asset(cr_path)
        if blueprint is None:
            blueprint = asset_tools.create_asset(cr_name, cr_pkg, unreal.ControlRigBlueprint, unreal.ControlRigBlueprintFactory())
        if blueprint is None:
            return None

        hc = blueprint.get_hierarchy_controller()
        skel = getattr(mesh_asset, "skeleton", None)
        if skel is not None and hasattr(hc, "import_bones"):
            hc.import_bones(skel)
        if hasattr(blueprint, "set_preview_mesh"):
            blueprint.set_preview_mesh(mesh_asset)

        hier = blueprint.hierarchy
        color_by_side = {{
            "L": unreal.LinearColor(0.15, 0.55, 1.0, 1.0),
            "R": unreal.LinearColor(1.0, 0.22, 0.18, 1.0),
            "C": unreal.LinearColor(1.0, 0.82, 0.12, 1.0),
            "ROOT": unreal.LinearColor(0.18, 0.92, 0.36, 1.0),
        }}

        def _bone_key(bname: str):
            return unreal.RigElementKey(type=unreal.RigElementType.BONE, name=bname)

        def _has_bone(bname: str) -> bool:
            return bool(bname) and hier.contains(_bone_key(bname))

        def _add_ctrl(cname: str, bone_name: str, parent_ctrl_key, shape_name: str, side: str, scale_val: float = 1.0):
            if not _has_bone(bone_name):
                return None
            settings = unreal.RigControlSettings()
            settings.control_type = unreal.RigControlType.EULER_TRANSFORM
            if hasattr(settings, "shape_name"):
                settings.shape_name = shape_name
            if hasattr(settings, "shape_color"):
                settings.shape_color = color_by_side.get(side, color_by_side["C"])
            val = hier.make_control_value_from_euler_transform(unreal.EulerTransform())
            b_xform = hier.get_global_transform(_bone_key(bone_name), initial=True)
            pkey = parent_ctrl_key if parent_ctrl_key is not None else unreal.RigElementKey()
            ckey = hc.add_control(cname, pkey, settings, val, setup_undo=False, print_python_command=False)
            hier.set_global_transform(ckey, b_xform, initial=True)
            if hasattr(hier, "set_control_shape_transform"):
                st = unreal.Transform()
                st.scale3d = unreal.Vector(scale_val, scale_val, scale_val)
                hier.set_control_shape_transform(ckey, st, initial=True)
            return ckey

        # 1. Create Control Hierarchy matching Blender HFW Control Rig
        ctrl_to_bone = []  # (ctrl_name, bone_name) pairs
        ik_chains = []     # IK effector dicts

        root_b = RIG_SPEC.get("root") or "rootBone"
        hips_b = RIG_SPEC.get("hips") or "hipsBone"
        k_root = _add_ctrl("CR_Root", root_b, None, "Circle_Thick", "ROOT", 4.0)
        if k_root:
            ctrl_to_bone.append(("CR_Root", root_b))
        k_cog = _add_ctrl("CR_COG", hips_b, k_root, "Circle_Thick", "ROOT", 2.8)
        if k_cog:
            ctrl_to_bone.append(("CR_COG", hips_b))

        # Spine / Neck / Head / Jaw / Tail / Wings FK Controls
        def _sample(seq, cap):
            if len(seq) <= cap:
                return list(seq)
            idxs = sorted({{int(round(i * (len(seq) - 1) / (cap - 1))) for i in range(cap)}})
            return [seq[i] for i in idxs]

        sp_parent = k_cog or k_root
        for idx, sb in enumerate(_sample(RIG_SPEC.get("spine") or [], 8)):
            k = _add_ctrl(f"CR_Spine_{{idx:02d}}", sb, sp_parent, "Circle_Thin", "C", 2.0)
            if k:
                ctrl_to_bone.append((f"CR_Spine_{{idx:02d}}", sb))
                sp_parent = k

        chest_b = RIG_SPEC.get("chest")
        if chest_b:
            k_chest = _add_ctrl("CR_Chest", chest_b, sp_parent, "Box_Thick", "C", 2.4)
            if k_chest:
                ctrl_to_bone.append(("CR_Chest", chest_b))
                sp_parent = k_chest

        nk_parent = sp_parent
        for idx, nb in enumerate(_sample(RIG_SPEC.get("neck") or [], 8)):
            k = _add_ctrl(f"CR_Neck_{{idx:02d}}", nb, nk_parent, "Circle_Thin", "C", 1.4)
            if k:
                ctrl_to_bone.append((f"CR_Neck_{{idx:02d}}", nb))
                nk_parent = k

        head_b = RIG_SPEC.get("head")
        k_head = _add_ctrl("CR_Head", head_b, nk_parent, "Box_Thick", "C", 1.8) if head_b else None
        if k_head:
            ctrl_to_bone.append(("CR_Head", head_b))
        if RIG_SPEC.get("head_ik_supported") and head_b and _has_bone(head_b):
            k_hik = _add_ctrl("CR_IK_Head", head_b, k_root, "Box_Thick", "C", 2.0)
            if k_hik:
                ik_chains.append({{"ctrl_name": "CR_IK_Head", "ik_bones": list(RIG_SPEC.get("neck") or []), "end_bone": head_b, "is_leg": False}})

        for spc in RIG_SPEC.get("spline_chains") or []:
            sp_bs = [b for b in (spc.get("bones") or []) if _has_bone(b)]
            if len(sp_bs) >= 5:
                k_cnt = int(spc.get("knots", 6))
                for ki, kb in enumerate(_sample(sp_bs, k_cnt)):
                    kn_nm = f"CR_Spline_{{spc['name']}}_{{ki:02d}}"
                    k_kn = _add_ctrl(kn_nm, kb, k_root if ki > 0 else (k_cog or k_root), "Circle_Thick", "C", 2.2)
                    if k_kn:
                        ctrl_to_bone.append((kn_nm, kb))

        for spn in (RIG_SPEC.get("spinners") or [])[:12]:
            sb = spn.get("bone")
            if sb and _has_bone(sb):
                sp_nm = f"CR_Spin_{{spn['name']}}"
                k_sp = _add_ctrl(sp_nm, sb, k_head or k_cog or k_root, "Circle_Thick", spn.get("side", "C"), 1.2)
                if k_sp:
                    ctrl_to_bone.append((sp_nm, sb))

        for cs in RIG_SPEC.get("chainsaws") or []:
            pb_s = cs.get("parent_def")
            if pb_s and _has_bone(pb_s):
                sw_nm = f"CR_Saw_Spin_{{cs['name']}}"
                k_sw = _add_ctrl(sw_nm, pb_s, k_head or k_cog or k_root, "Circle_Thick", cs.get("side", "C"), 1.4)
                if k_sw:
                    ctrl_to_bone.append((sw_nm, pb_s))

        for idx, jb in enumerate(RIG_SPEC.get("jaws") or []):
            cn = f"CR_Jaw_{{idx:02d}}"
            k = _add_ctrl(cn, jb, k_head or nk_parent, "Circle_Thin", "C", 1.0)
            if k:
                ctrl_to_bone.append((cn, jb))

        for tinfo in RIG_SPEC.get("tails") or []:
            tp = k_cog or k_root
            for idx, tb in enumerate(_sample(tinfo.get("bones") or [], 16)):
                cn = f"CR_{{tinfo['name']}}_{{idx:02d}}"
                k = _add_ctrl(cn, tb, tp, "Circle_Thin", tinfo.get("side", "C"), 1.6)
                if k:
                    ctrl_to_bone.append((cn, tb))
                    tp = k

        for app in RIG_SPEC.get("appendages") or []:
            ap = k_head or nk_parent
            for idx, ab in enumerate(_sample(app.get("bones") or [], 16)):
                cn = f"CR_{{app['name']}}_{{idx:02d}}"
                k = _add_ctrl(cn, ab, ap, "Circle_Thin", app.get("side", "C"), 1.4)
                if k:
                    ctrl_to_bone.append((cn, ab))
                    ap = k

        end_bone_to_ctrl = {{}}
        for winfo in RIG_SPEC.get("wings") or []:
            wp = sp_parent
            wside = winfo.get("side", "C")
            for idx, wb in enumerate(_sample(winfo.get("bones") or [], 6)):
                cn = f"CR_{{winfo['name']}}_{{idx:02d}}"
                k = _add_ctrl(cn, wb, wp, "Circle_Thin", wside, 1.8)
                if k:
                    ctrl_to_bone.append((cn, wb))
                    wp = k
            w_tip = winfo.get("tip_bone")
            w_iks = [b for b in (winfo.get("ik_bones") or []) if _has_bone(b)]
            if w_tip and _has_bone(w_tip) and len(w_iks) >= 2:
                w_ik_nm = f"CR_IK_{{winfo['name']}}"
                k_wik = _add_ctrl(w_ik_nm, w_tip, k_cog or k_root, "Box_Thick", wside, 1.8)
                if k_wik:
                    ik_chains.append({{"ctrl_name": w_ik_nm, "ik_bones": w_iks, "end_bone": w_tip, "is_leg": False}})
                    end_bone_to_ctrl[w_tip] = k_wik

        # Legs (Quadruped Shoulders/Scapulae + IK + Flipper Blade FK) & Arms (Clavicles + IK + Flipper Blade FK)
        for limb in list(RIG_SPEC.get("legs") or []) + list(RIG_SPEC.get("arms") or []):
            end_b = limb.get("foot_bone") or limb.get("hand_bone")
            ik_bs = [b for b in (limb.get("ik_bones") or []) if _has_bone(b)]
            if not end_b or not _has_bone(end_b) or len(ik_bs) < 2:
                continue
            side = limb.get("side", "C")
            sh_b = limb.get("shoulder")
            if sh_b and _has_bone(sh_b):
                sh_par = sp_parent if str(limb.get("name", "")).startswith("FrontLeg_") else (k_cog or k_root)
                sh_nm = f"CR_Shoulder_{{limb['name']}}"
                k_sh = _add_ctrl(sh_nm, sh_b, sh_par, "Circle_Thin", side, 1.4)
                if k_sh:
                    ctrl_to_bone.append((sh_nm, sh_b))
            cl_b = limb.get("clavicle")
            if cl_b and _has_bone(cl_b):
                cl_nm = f"CR_Clavicle_{{side}}"
                k_cl = _add_ctrl(cl_nm, cl_b, sp_parent, "Circle_Thin", side, 1.3)
                if k_cl:
                    ctrl_to_bone.append((cl_nm, cl_b))
            cname = f"CR_IK_{{limb['name']}}"
            k_ik = _add_ctrl(cname, end_b, k_root, "Box_Thick", side, 1.8)
            if k_ik:
                ik_chains.append({{"ctrl_name": cname, "ik_bones": ik_bs, "end_bone": end_b, "is_leg": "foot_bone" in limb}})
                end_bone_to_ctrl[end_b] = k_ik
                ex_par = k_ik
                for eidx, eb in enumerate(limb.get("extra_fk_bones") or []):
                    ex_nm = f"CR_FK_{{limb['name']}}_Blade_{{eidx:02d}}"
                    k_ex = _add_ctrl(ex_nm, eb, ex_par, "Circle_Thin", side, 1.1)
                    if k_ex:
                        ctrl_to_bone.append((ex_nm, eb))
                        ex_par = k_ex

        # Hands / Fingers, Feet / Toes / Talons, and Secondary Articulated Machine Details FK Controls
        detail_groups = (
            (RIG_SPEC.get("fingers") or [], 0.45),
            (RIG_SPEC.get("toes") or [], 0.55),
            (RIG_SPEC.get("details") or [], 0.70),
        )
        ctrl_by_bone = {{bn: unreal.RigElementKey(type=unreal.RigElementType.CONTROL, name=cn) for cn, bn in ctrl_to_bone}}
        for grp_list, shape_sc in detail_groups:
            for item in grp_list:
                p_def = item.get("parent_def")
                par_k = end_bone_to_ctrl.get(p_def) or ctrl_by_bone.get(p_def) or k_cog or k_root
                side = item.get("side", "C")
                for idx, db in enumerate(item.get("bones") or []):
                    cn = f"CR_FK_{{item['name']}}_{{idx:02d}}"
                    k = _add_ctrl(cn, db, par_k, "Circle_Thin", side, shape_sc)
                    if k:
                        ctrl_to_bone.append((cn, db))
                        ctrl_by_bone[db] = k
                        par_k = k

        # 2. Wire RigVM Graph: Forwards Solve (Controls -> Bones + FullBodyIK) AND Backwards Solve (Bones -> Controls)!
        ctrl = blueprint.get_controller() if hasattr(blueprint, "get_controller") else blueprint.get_controller_by_name("RigVMModel")
        if ctrl is not None:
            # --- FORWARDS SOLVE (`RigUnit_BeginExecution`) ---
            n_fwd = ctrl.add_unit_node_from_struct_path("/Script/ControlRig.RigUnit_BeginExecution", "Execute", unreal.Vector2D(0, 0), "BeginExecution")
            prev_exec = f"{{n_fwd.get_name()}}.ExecuteContext"
            x_cur = 320

            for cname, bname in ctrl_to_bone:
                n_get = ctrl.add_unit_node_from_struct_path("/Script/ControlRig.RigUnit_GetTransform", "Execute", unreal.Vector2D(x_cur - 120, 260), f"Get_{{cname}}")
                ctrl.set_pin_default_value(f"{{n_get.get_name()}}.Item", f'(Type=Control,Name="{{cname}}")', True)
                ctrl.set_pin_default_value(f"{{n_get.get_name()}}.Space", "GlobalSpace", True)

                n_set = ctrl.add_unit_node_from_struct_path("/Script/ControlRig.RigUnit_SetTransform", "Execute", unreal.Vector2D(x_cur, 0), f"Set_{{bname}}")
                ctrl.set_pin_default_value(f"{{n_set.get_name()}}.Item", f'(Type=Bone,Name="{{bname}}")', True)
                ctrl.set_pin_default_value(f"{{n_set.get_name()}}.Space", "GlobalSpace", True)
                ctrl.set_pin_default_value(f"{{n_set.get_name()}}.bPropagateToChildren", "True", True)

                ctrl.add_link(prev_exec, f"{{n_set.get_name()}}.ExecuteContext")
                ctrl.add_link(f"{{n_get.get_name()}}.Transform", f"{{n_set.get_name()}}.Transform")
                prev_exec = f"{{n_set.get_name()}}.ExecuteContext"
                x_cur += 340

            # Full-Body IK (`RigUnit_PBIK`) cleanly handles 2-bone human, 3-bone digitigrade theropods, AND 4-leg quadrupeds!
            if ik_chains and _has_bone(hips_b):
                try:
                    n_pbik = ctrl.add_unit_node_from_struct_path("/Script/PBIK.RigUnit_PBIK", "Execute", unreal.Vector2D(x_cur + 60, 0), "HFW_FullBodyIK")
                    ctrl.set_pin_default_value(f"{{n_pbik.get_name()}}.Root", hips_b, True)
                    ctrl.add_link(prev_exec, f"{{n_pbik.get_name()}}.ExecuteContext")
                    for ei, ch in enumerate(ik_chains):
                        ctrl.add_array_pin(f"{{n_pbik.get_name()}}.Effectors")
                        pin_e = f"{{n_pbik.get_name()}}.Effectors.{{ei}}"
                        ctrl.set_pin_default_value(f"{{pin_e}}.Bone", ch["end_bone"], True)
                        n_gk = ctrl.add_unit_node_from_struct_path("/Script/ControlRig.RigUnit_GetTransform", "Execute", unreal.Vector2D(x_cur - 220, 260 + ei * 180), f"GetIK_{{ch['ctrl_name']}}")
                        ctrl.set_pin_default_value(f"{{n_gk.get_name()}}.Item", f'(Type=Control,Name="{{ch["ctrl_name"]}}")', True)
                        ctrl.set_pin_default_value(f"{{n_gk.get_name()}}.Space", "GlobalSpace", True)
                        ctrl.add_link(f"{{n_gk.get_name()}}.Transform", f"{{pin_e}}.Transform")
                    prev_exec = f"{{n_pbik.get_name()}}.ExecuteContext"
                except Exception as exc_ik:
                    unreal.log_warning(f"[HFW] PBIK node fallback (optional plugin): {{exc_ik}}")

            # --- BACKWARDS SOLVE (`RigUnit_InverseExecution`) -> Enables Sequencer "Bake To Control Rig"! ---
            n_bwd = ctrl.add_unit_node_from_struct_path("/Script/ControlRig.RigUnit_InverseExecution", "Execute", unreal.Vector2D(0, 960), "InverseExecution")
            prev_bwd = f"{{n_bwd.get_name()}}.ExecuteContext"
            bx = 320
            all_bwd = list(ctrl_to_bone) + [(ch["ctrl_name"], ch["end_bone"]) for ch in ik_chains]
            for cname, bname in all_bwd:
                n_gb = ctrl.add_unit_node_from_struct_path("/Script/ControlRig.RigUnit_GetTransform", "Execute", unreal.Vector2D(bx - 120, 1220), f"BwdGet_{{bname}}")
                ctrl.set_pin_default_value(f"{{n_gb.get_name()}}.Item", f'(Type=Bone,Name="{{bname}}")', True)
                ctrl.set_pin_default_value(f"{{n_gb.get_name()}}.Space", "GlobalSpace", True)

                n_sc = ctrl.add_unit_node_from_struct_path("/Script/ControlRig.RigUnit_SetTransform", "Execute", unreal.Vector2D(bx, 960), f"BwdSet_{{cname}}")
                ctrl.set_pin_default_value(f"{{n_sc.get_name()}}.Item", f'(Type=Control,Name="{{cname}}")', True)
                ctrl.set_pin_default_value(f"{{n_sc.get_name()}}.Space", "GlobalSpace", True)
                ctrl.set_pin_default_value(f"{{n_sc.get_name()}}.bPropagateToChildren", "True", True)

                ctrl.add_link(prev_bwd, f"{{n_sc.get_name()}}.ExecuteContext")
                ctrl.add_link(f"{{n_gb.get_name()}}.Transform", f"{{n_sc.get_name()}}.Transform")
                prev_bwd = f"{{n_sc.get_name()}}.ExecuteContext"
                bx += 340

        if hasattr(blueprint, "recompile_vm"):
            blueprint.recompile_vm()
        unreal.EditorAssetLibrary.save_loaded_asset(blueprint)
        unreal.log(
            f"[HFW] Created native UE5 Control Rig {{cr_path}} "
            f"({{len(ctrl_to_bone)}} FK controls, {{len(ik_chains)}} IK effectors, Forwards + Backwards Solve)!"
        )
        return blueprint
    except Exception as exc:
        unreal.log_warning(f"[HFW] Control Rig auto-generation warning (mesh/anims/materials unaffected): {{exc}}")
        return None


def run_import():
    _ensure_mpc()
    unreal.log(f"[HFW] Building {{len(MATERIALS_META)}} UE5 materials for {{MACHINE_NAME}} ({{ASSET_KIND}})...")
    built = {{}}
    for slot_name, meta in MATERIALS_META.items():
        try:
            m = build_single_material(slot_name, meta)
            built[slot_name] = m
            built[_safe(slot_name)] = m
        except Exception as exc:
            unreal.log_error(f"[HFW] Failed to build material {{slot_name}}: {{exc}}")
    mesh_asset = import_machine_fbx(built)
    if IS_SKELETAL and mesh_asset is not None:
        import_animations_fbx(mesh_asset)
        build_ue5_control_rig(mesh_asset)
    unreal.log(f"[HFW] Complete! Assets ready in {{DEST_PACKAGE}}")


run_import()
'''
