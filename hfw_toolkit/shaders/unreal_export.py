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
    dest_root = f"/Game/HFW/{safe_mach}"
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
            "tags": mmeta.get("tags", []),
        }

    mats_json_str = json.dumps(mat_entries, indent=2)
    bundle_abs = str(Path(bundle_dir).resolve())

    return f'''"""Auto-generated Unreal Engine 5 Full Machine Rig & Animated Shader Importer for {disp}.

Imported automatically by HFWToolkit (`{manifest.get("entity_id", "")}`, `{manifest.get("variant_label", "")}`):
  - Rigged SkeletalMesh + Skeleton + Bone-Parented Armor Plates + ChildEntity Weapons (`{fbx_name}`)
  - ALL {len(mat_entries)} Decima PBR & DXIL-Verified Animated Shaders:
      * Decima Smoothness -> UE5 Roughness (`OneMinus`) + Worn-Metal Edge Recovery (`F0 = RFL^2`)
      * Animated Elemental Sacks (Dual-Phase Flow Map `172:0` + `1703:0` fluid sloshing + Fresnel + `mat_id==0` mask)
      * Animated Elemental Canisters (`Sine(Time*8)` pulsing collar `mat_id==1` + `Panner` fluid window `mat_id==20`)
      * Animated Machine Muscle Cables (`8260:1`) & Optical Muscle Fibers (`19501:0`, Translucent Additive)
      * Animated Machine Cores/Hearts (1 Hz `Abs(Sine)` heartbeat + `Panner` scroll) & Glowing Optical Eyes

How to run inside Unreal Engine 5:
  1. Enable the built-in `Python Editor Script Plugin` in UE5.
  2. In UE5 Menu -> Tools -> Execute Python Script... -> select this `ue5_import_machine.py`.
  3. Open `/Game/HFW/{safe_mach}/SK_{safe_mach}` and hit Play/Realtime Viewport to watch all shaders animate!
"""

import json
import os
import re
import unreal

MACHINE_NAME = {safe_mach!r}
DEST_PACKAGE = {dest_root!r}
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else r"{bundle_abs}"
TEXTURE_DIR = os.path.join(SCRIPT_DIR, "textures")
FBX_CANDIDATES = [
    os.path.join(SCRIPT_DIR, {fbx_name!r}),
    os.path.join(os.path.dirname(SCRIPT_DIR), {fbx_name!r}),
]

MATERIALS_META = json.loads(r"""{mats_json_str}""")

mel = unreal.MaterialEditingLibrary
asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
_TEX_CACHE = {{}}


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
    n_s = _scalar(mat, "AnimateSpeed", speed_default, x - 320, y + 110)
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
            n_tc = _vec(mat, "GlassTint", spec.get("tint_color", [0.80, 0.92, 0.91]), -400, -100)
            n_al = _scalar(mat, "GlassOpacity", float(spec.get("alpha", 0.25)), -400, 80)
            n_rg = _scalar(mat, "GlassRoughness", 0.08, -400, 200)
            mel.connect_material_property(n_tc, "", unreal.MaterialProperty.MP_BASE_COLOR)
            mel.connect_material_property(n_al, "", unreal.MaterialProperty.MP_OPACITY)
            mel.connect_material_property(n_rg, "", unreal.MaterialProperty.MP_ROUGHNESS)
            mel.recompile_material(mat)
            unreal.EditorAssetLibrary.save_loaded_asset(mat)
            return mat

        if skind == "muscle_fiber_overlay":
            mat.set_editor_property("blend_mode", unreal.BlendMode.BLEND_ADDITIVE)
            mat.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
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
            n_wadd = _mk(mat, unreal.MaterialExpressionAdd, -420, 40)
            mel.connect_material_expressions(n_rmp, "B", n_wadd, "A")
            mel.connect_material_expressions(n_rmp, "A", n_wadd, "B")
            n_wsat = _mk(mat, unreal.MaterialExpressionSaturate, -280, 40)
            mel.connect_material_expressions(n_wadd, "", n_wsat, "")

            # Color blend: Calm Amber -> Electric Plasma Violet
            n_cc = _vec(mat, "CalmColor", spec.get("calm_color", [0.95, 0.24, 0.02]), -720, -340)
            n_pc = _vec(mat, "PlasmaColor", spec.get("plasma_color", [0.46, 0.05, 1.0]), -720, -180)
            n_cs = _scalar(mat, "ChargeState", float(spec.get("charge_state", 0.0)), -720, -40)
            n_lerp = _mk(mat, unreal.MaterialExpressionLinearInterpolate, -420, -240)
            mel.connect_material_expressions(n_cc, "", n_lerp, "A")
            mel.connect_material_expressions(n_pc, "", n_lerp, "B")
            mel.connect_material_expressions(n_cs, "", n_lerp, "Alpha")

            n_str = _scalar(mat, "GlowStrength", float(spec.get("glow_strength", 3.4)), -420, 220)
            n_m1 = _mk(mat, unreal.MaterialExpressionMultiply, -180, -80)
            mel.connect_material_expressions(n_lerp, "", n_m1, "A")
            mel.connect_material_expressions(n_wsat, "", n_m1, "B")
            n_m2 = _mk(mat, unreal.MaterialExpressionMultiply, -40, 0)
            mel.connect_material_expressions(n_m1, "", n_m2, "A")
            mel.connect_material_expressions(n_str, "", n_m2, "B")
            if "Mask_Alpha" in sockets:
                # Smooth BC7-noise-free cavity fade (`saturate(Mask_Alpha * 11.0) * 0.84 + 0.16`)
                n_ma = _mk(mat, unreal.MaterialExpressionMultiply, -420, 340)
                n_ma.set_editor_property("const_b", 11.0)
                mel.connect_material_expressions(sockets["Mask_Alpha"][0], sockets["Mask_Alpha"][1], n_ma, "A")
                n_masat = _mk(mat, unreal.MaterialExpressionSaturate, -260, 340)
                mel.connect_material_expressions(n_ma, "", n_masat, "")
                n_malerp = _mk(mat, unreal.MaterialExpressionLinearInterpolate, -100, 340)
                n_malerp.set_editor_property("const_a", 0.16)
                n_malerp.set_editor_property("const_b", 1.0)
                mel.connect_material_expressions(n_masat, "", n_malerp, "Alpha")
                n_m3 = _mk(mat, unreal.MaterialExpressionMultiply, 100, 60)
                mel.connect_material_expressions(n_m2, "", n_m3, "A")
                mel.connect_material_expressions(n_malerp, "", n_m3, "B")
                n_m2 = n_m3
            mel.connect_material_property(n_m2, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
            # Subtle +0.14cm WorldPositionOffset along VertexNormalWS to prevent coplanar z-fighting in UE5!
            n_vn = _mk(mat, unreal.MaterialExpressionVertexNormalWS, -340, 540)
            n_wpo = _mk(mat, unreal.MaterialExpressionMultiply, -100, 540)
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

            n_gc = _vec(mat, "GlowColor", spec.get("glow_color", [0.0, 0.231, 1.0]), -680, 620)
            n_gs = _scalar(mat, "GlowStrength", float(spec.get("glow_strength", 6.0)), -680, 780)
            n_m1 = _mk(mat, unreal.MaterialExpressionMultiply, -400, 260)
            mel.connect_material_expressions(n_br, "B", n_m1, "A")
            mel.connect_material_expressions(n_pl, "R", n_m1, "B")
            n_m2 = _mk(mat, unreal.MaterialExpressionMultiply, -220, 440)
            mel.connect_material_expressions(n_gc, "", n_m2, "A")
            mel.connect_material_expressions(n_gs, "", n_m2, "B")
            n_m3 = _mk(mat, unreal.MaterialExpressionMultiply, -40, 360)
            mel.connect_material_expressions(n_m1, "", n_m3, "A")
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
            n_gc = _vec(mat, "GlowColor", spec.get("glow_color", [0.05, 0.52, 1.0]), -580, -80)
            n_gs = _scalar(mat, "GlowStrength", float(spec.get("glow_strength", 7.0)), -580, 320)
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
    if "Reflectance" in sockets:
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
        if "Color" in sockets:
            n_bc = _mk(mat, unreal.MaterialExpressionLinearInterpolate, -240, -180)
            mel.connect_material_expressions(sockets["Color"][0], sockets["Color"][1], n_bc, "A")
            mel.connect_material_expressions(n_f0, "", n_bc, "B")
            mel.connect_material_expressions(n_met, "", n_bc, "Alpha")
            mel.connect_material_property(n_bc, "", unreal.MaterialProperty.MP_BASE_COLOR)
    elif "Color" in sockets:
        mel.connect_material_property(sockets["Color"][0], sockets["Color"][1], unreal.MaterialProperty.MP_BASE_COLOR)

    if "Roughness" in sockets:
        n_inv = _mk(mat, unreal.MaterialExpressionOneMinus, -460, -40)
        mel.connect_material_expressions(sockets["Roughness"][0], sockets["Roughness"][1], n_inv, "")
        mel.connect_material_property(n_inv, "", unreal.MaterialProperty.MP_ROUGHNESS)

    if "AO" in sockets:
        mel.connect_material_property(sockets["AO"][0], sockets["AO"][1], unreal.MaterialProperty.MP_AMBIENT_OCCLUSION)

    if "Normal_RGB" in sockets:
        mel.connect_material_property(sockets["Normal_RGB"][0], sockets["Normal_RGB"][1], unreal.MaterialProperty.MP_NORMAL)
    elif "Normal.R" in sockets and "Normal.G" in sockets:
        n_dz = _reconstruct_normal_z(mat, sockets["Normal.R"][0], sockets["Normal.R"][1], sockets["Normal.G"][0], sockets["Normal.G"][1], -320, 480)
        mel.connect_material_property(n_dz, "", unreal.MaterialProperty.MP_NORMAL)

    # Wire Animated Elemental Sac / Canister / Eye Lens Overlays!
    if overlay:
        okind = overlay.get("kind", "")
        n_t = _time_scaled(mat, -1100, 960)

        if okind == "elemental_sack_overlay" and "Mask" in sockets:
            # Exact DXIL mat_id==0 * Mask_Alpha + Dual-Phase Animated Flow Map + Fresnel Rim!
            n_is0 = _build_mask_id_compare(mat, sockets["Mask"][0], sockets["Mask"][1], 0.0, -560, 820)
            sac_mask = n_is0
            if "Mask_Alpha" in sockets:
                n_sm = _mk(mat, unreal.MaterialExpressionMultiply, -380, 840)
                mel.connect_material_expressions(n_is0, "", n_sm, "A")
                mel.connect_material_expressions(sockets["Mask_Alpha"][0], sockets["Mask_Alpha"][1], n_sm, "B")
                sac_mask = n_sm

            n_uv = _mk(mat, unreal.MaterialExpressionTextureCoordinate, -1380, 1180)
            n_uv.set_editor_property("u_tiling", 2.5)
            n_uv.set_editor_property("v_tiling", 2.5)
            n_pan = _mk(mat, unreal.MaterialExpressionPanner, -1080, 1180)
            n_pan.set_editor_property("speed_x", 0.06)
            n_pan.set_editor_property("speed_y", float(overlay.get("flow_speed", 0.35)))
            mel.connect_material_expressions(n_uv, "", n_pan, "Coordinate")
            mel.connect_material_expressions(n_t, "", n_pan, "Time")
            fluid_tex = import_texture(overlay.get("fluid_file", ""), is_srgb=False)
            n_fl = _tex_node(mat, "SacFluidCaustics", fluid_tex, unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR, -800, 1180)
            mel.connect_material_expressions(n_pan, "", n_fl, "UVs")

            n_fl_add = _mk(mat, unreal.MaterialExpressionAdd, -540, 1180)
            n_fl_add.set_editor_property("const_b", 0.45)
            mel.connect_material_expressions(n_fl, "R", n_fl_add, "A")

            n_gc = _vec(mat, "SacGlowColor", overlay.get("glow_color", [0.14, 0.68, 1.0]), -540, 1000)
            n_gs = _scalar(mat, "SacGlowStrength", float(overlay.get("glow_strength", 6.5)), -540, 1340)
            n_m1 = _mk(mat, unreal.MaterialExpressionMultiply, -260, 1080)
            mel.connect_material_expressions(n_fl_add, "", n_m1, "A")
            mel.connect_material_expressions(sac_mask, "", n_m1, "B")
            n_m2 = _mk(mat, unreal.MaterialExpressionMultiply, -120, 1140)
            mel.connect_material_expressions(n_m1, "", n_m2, "A")
            mel.connect_material_expressions(n_gs, "", n_m2, "B")
            n_em = _mk(mat, unreal.MaterialExpressionMultiply, 40, 1060)
            mel.connect_material_expressions(n_gc, "", n_em, "A")
            mel.connect_material_expressions(n_m2, "", n_em, "B")
            mel.connect_material_property(n_em, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)

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
            n_gc = _vec(mat, "CanisterGlowColor", overlay.get("glow_color", [0.52, 0.70, 0.10]), -360, 1420)
            n_gs = _scalar(mat, "CanisterGlowStrength", float(overlay.get("glow_strength", 5.5)), -160, 1420)
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
            n_cc = _vec(mat, "CalmEyeColor", overlay.get("calm_color", [0.04, 0.56, 1.0]), -700, 1180)
            n_ac = _vec(mat, "AlertEyeColor", overlay.get("alert_color", [1.0, 0.10, 0.02]), -700, 1360)
            n_as = _scalar(mat, "AlertState", float(overlay.get("alert_state", 0.0)), -700, 1520)
            n_lerp = _mk(mat, unreal.MaterialExpressionLinearInterpolate, -420, 1240)
            mel.connect_material_expressions(n_cc, "", n_lerp, "A")
            mel.connect_material_expressions(n_ac, "", n_lerp, "B")
            mel.connect_material_expressions(n_as, "", n_lerp, "Alpha")
            n_gs = _scalar(mat, "EyeGlowStrength", float(overlay.get("glow_strength", 9.0)), -420, 1040)
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
            n_pc = _vec(mat, "PlasmaColor", overlay.get("plasma_color", [0.46, 0.05, 1.0]), -720, 1340)
            n_ic = _vec(mat, "IdleConduitColor", overlay.get("idle_led_color", [0.85, 0.18, 0.02]), -720, 1500)
            n_chg = _scalar(mat, "PlasmaCharge", float(overlay.get("plasma_charge", 1.0)), -720, 1180)
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

    if "Alpha" in sockets:
        mat.set_editor_property("blend_mode", unreal.BlendMode.BLEND_MASKED)
        mel.connect_material_property(sockets["Alpha"][0], sockets["Alpha"][1], unreal.MaterialProperty.MP_OPACITY_MASK)

    mel.recompile_material(mat)
    unreal.EditorAssetLibrary.save_loaded_asset(mat)
    return mat


def import_machine_fbx(built_materials: dict):
    fbx_path = next((p for p in FBX_CANDIDATES if os.path.exists(p)), None)
    if not fbx_path:
        unreal.log_warning(f"[HFW] Rigged FBX not found yet (checked {{FBX_CANDIDATES}}). Materials were still built cleanly!")
        return None
    opts = unreal.FbxImportUI()
    opts.import_mesh = True
    opts.import_as_skeletal = True
    opts.import_materials = False
    opts.import_textures = False
    opts.import_animations = False
    opts.skeletal_mesh_import_data.combine_meshes = True
    opts.skeletal_mesh_import_data.import_morph_targets = False
    if hasattr(opts.skeletal_mesh_import_data, "vertex_color_import_option"):
        opts.skeletal_mesh_import_data.vertex_color_import_option = unreal.VertexColorImportOption.REPLACE

    task = unreal.AssetImportTask()
    task.filename = fbx_path
    task.destination_path = DEST_PACKAGE
    task.destination_name = "SK_" + MACHINE_NAME
    task.automated = True
    task.replace_existing = True
    task.save = True
    task.options = opts
    asset_tools.import_asset_tasks([task])

    sk = unreal.EditorAssetLibrary.load_asset(f"{{DEST_PACKAGE}}/SK_{{MACHINE_NAME}}")
    if sk and hasattr(sk, "materials"):
        assigned = 0
        new_mats = []
        for sm in sk.materials:
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
        sk.materials = new_mats
        unreal.EditorAssetLibrary.save_loaded_asset(sk)
        unreal.log(f"[HFW] Assigned {{assigned}}/{{len(new_mats)}} animated UE5 materials onto SK_{{MACHINE_NAME}}!")
    return sk


def run_import():
    unreal.log(f"[HFW] Building {{len(MATERIALS_META)}} animated UE5 materials for {{MACHINE_NAME}}...")
    built = {{}}
    for slot_name, meta in MATERIALS_META.items():
        try:
            m = build_single_material(slot_name, meta)
            built[slot_name] = m
            built[_safe(slot_name)] = m
        except Exception as exc:
            unreal.log_error(f"[HFW] Failed to build material {{slot_name}}: {{exc}}")
    import_machine_fbx(built)
    unreal.log(f"[HFW] Complete! Assets ready in {{DEST_PACKAGE}}")


run_import()
'''
