"""Canonical Assembly Bundle exporter (decouples HFW Python 3.14 reader from Blender 5.3 Python 3.13 & UE5)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from hfw_toolkit.decima.reader import ObjectId
from hfw_toolkit.decima.streaming_graph import HFWGame
from hfw_toolkit.extract.machine_assembler import MachineAssembly
from hfw_toolkit.extract.materials import MaterialInfo
from hfw_toolkit.extract.textures import (
    LANE_TO_IDX,
    PackingChannel,
    decode_texture,
    export_packed_texture,
    export_packed_texture_set,
    export_standalone_texture,
    export_texture_array_atlas,
    reconstruct_normal_map,
    save_unpacked_texture_set,
    unpack_texture_set,
)
from hfw_toolkit.shaders.character_dxil import (
    extract_dxil_outfit_corners,
    infer_character_hair_color,
    infer_character_skin_tone,
)

# Known Decima binding hashes for machine detail texture arrays (e.g. Tex 20:0 256x256x23 BC7)
_DETAIL_ARRAY_BINDING_HASHES: frozenset[int] = frozenset({0x342B11E0})

# Known Decima binding hashes for specialized machine muscle / core / eye / canister / VFX shaders
# (verified directly from disassembled HFW PixelProgram DXIL: 8260:1, 19501:0, 4628:0, 32906:0, 41249:0, 37840:0)
_HASH_MUSCLE_BRAID = 0x1B7B57EB      # 4019:1 (64x1024 BC7): R,G = braided muscle normal, B = glow crevice mask
_HASH_MUSCLE_VEINS = 0x6DEF9DB3      # 4019:0 (128x512 BC5U): R = strand alpha, G = glowing circuit veins
_HASH_MUSCLE_PULSE = 0x3A91D95B      # 5645:0 (32x512 BC4U): scrolling energy pulse streaks
_HASH_MUSCLE_RAMP_A = 0x49853A2A     # 5147:0 (64x16 RGBA8): muscle cable energy ramp
_HASH_MUSCLE_RAMP_B = 0x313B7FE9     # 5147:0 (64x16 RGBA8): ForwardBackground body muscle fiber energy bands
_HASH_MUSCLE_DISSOLVE = 0x7912A47E   # 107:0 (512x512 BC4U): noise / dissolve mask
_HASH_CORE_COLOR = 0x6687326C        # 19456:0 (1024x1024 BC7): Thunderjaw Brain/Heart & Slitherfang neck core
_HASH_CORE_NORMAL = 0x09299177       # 19456:1 (512x512 BC5U): Brain/Heart/Core normal map
_HASH_VFX_SHELL_LUT = 0x4754E88E     # 487:0 (8x8 R8): ForwardForeground full-body Focus/VFX shell
_HASH_VFX_SHELL_ARR = 0x1263CD7F     # 1:8 (256x256x60 BC1): ForwardForeground VFX shell array
_HASH_EYE_VB = 0x2A15D368            # 4628:0 inner machine eye glow shader variable
_HASH_GLASS_TINT_VB = 0x01943DD3     # 41249:0 canister glass shell RGBA tint
_HASH_CORE_TINT_VB = 0x5D2FABB8      # 32906:0 / 1348:90 core emissive tint RGB

# DXIL-verified elemental sack, canister & outer eye lens bindings (from scratch/dxil_sacks/*.txt)
_HASH_SAC_FLOW_MAP = 0x7C771B98      # 172:0 (256x256 BC7): RG dual-phase elemental sac flow vectors
_HASH_SAC_FLUID_TEX = 0x23FE87DB     # 1703:0 (256x256 BC4U): elemental fluid caustics / bubbles
_HASH_SAC_COLOR_VB = 0x249F1E74      # PerInstance elemental sac glow RGB
_HASH_EMISSIVE_TINT_VB = 0x0E470C22  # Canister / elemental window glow RGB (sh=7564:6 / 7036:2)
_HASH_EYE_IRIS_MASK = 0x5E969132     # 8837:0 / 20273:0 / 15787:3 outer eye lens POM iris glow mask

# Shaders verified from DXIL to emit animated canister fluid/collar glow
_CANISTER_GLOW_SHADERS: frozenset[str] = frozenset({"7564:6", "7036:2"})


# Authentic HFW linear RGB elemental palettes (calibrated for Blender AgX & UE5 ACES)
_ELEMENT_RGB: dict[str, list[float]] = {
    "fire": [1.0, 0.20, 0.01],         # Blaze / Fire (Molten Orange)
    "acid": [0.36, 0.92, 0.03],        # Acid / Metalbite (Corrosive Lime-Green)
    "adhesive": [0.96, 0.66, 0.02],    # Adhesive / Glue (Amber-Gold)
    "fertilizer": [0.12, 0.80, 0.14],  # Plowhorn Bio-Fertilizer (Emerald Green)
    "cryo": [0.08, 0.62, 1.0],         # Cryo / Chillwater / Frost (Glacial Cyan-Blue)
    "purgewater": [0.04, 0.44, 0.96],  # Purgewater (Deep Aqua-Blue)
    "plasma": [0.46, 0.05, 1.0],       # Plasma / Glowblast (Electric Violet)
    "shock": [0.04, 0.46, 1.0],        # Shock / Sparker (Electric Blue)
    "amplify": [1.0, 0.72, 0.04],      # Spikesnout Amplifying Sac (Golden Yellow)
    "stamina": [0.92, 0.06, 0.24],     # Spikesnout Stamina Drain Sac (Crimson-Magenta)
    "dampener": [0.08, 0.86, 0.48],    # Spikesnout Damage Dampener Sac (Mint-Emerald)
    "rage": [0.92, 0.12, 0.04],        # Tremortusk Rage Sac (Crimson Red)
    "echo": [0.24, 0.76, 0.96],        # Burrower Sonic/Echo Sac (Sonic Cyan)
}


def _match_element_from_text(text: str) -> str | None:
    """Match an HFW elemental type keyword from a part name, catalog part name, material name, or variant description."""
    low = text.lower()
    if not low.strip():
        return None
    # Specific HFW compound/sac terms first (highest specificity)
    if any(k in low for k in ("metalbite", "acid", "corros")):
        return "acid"
    if any(k in low for k in ("adhesive", "glue")):
        return "adhesive"
    if any(k in low for k in ("fertiliz",)):
        return "fertilizer"
    if any(k in low for k in ("amplif", "buffing")):
        return "amplify"
    if any(k in low for k in ("stamina", "debuff")):
        return "stamina"
    if any(k in low for k in ("dampener", "tailsac")):
        return "dampener"
    if any(k in low for k in ("blaze", "inferno", "fireclaw", "fire")):
        return "fire"
    if any(k in low for k in ("chillwater", "cryo", "frost", "freeze", "ice")):
        return "cryo"
    if any(k in low for k in ("purgewater", "purge", "water")):
        return "purgewater"
    if any(k in low for k in ("glowblast", "plasma")):
        return "plasma"
    if any(k in low for k in ("sparker", "shock", "volt", "electr")):
        return "shock"
    if any(k in low for k in ("rage",)):
        return "rage"
    if any(k in low for k in ("echo",)):
        return "echo"
    return None


def _is_placeholder_sac_color(f4: tuple[float, ...] | list[float]) -> bool:
    """Detect Decima's runtime-overridden default placeholder values on `0x249F1E74`."""
    r, g, b = float(f4[0]), float(f4[1]), float(f4[2])
    if max(r, g, b) <= 0.02:
        return True
    # Debug bright green [0.0, 1.0, 0.0] (e.g. Fireclaw shoulders)
    if abs(r) < 0.015 and abs(g - 1.0) < 0.015 and abs(b) < 0.015:
        return True
    # Ubiquitous default Cryo placeholder blue [0.135, 0.27, 0.54] (Slaughterspine, Bellowback, Clawstrider, Bilegut, Plowhorn, Spikesnout, Slitherfang, Snapmaw, Widemaw, Tideripper!)
    if abs(r - 0.135) < 0.015 and abs(g - 0.27) < 0.015 and abs(b - 0.54) < 0.015:
        return True
    # Neutral dim gray placeholder [0.057, 0.057, 0.057] (Burrower Echo sac)
    if abs(r - 0.057) < 0.015 and abs(g - 0.057) < 0.015 and abs(b - 0.057) < 0.015:
        return True
    # Slitherfang default body blue placeholder [0.0, 0.146, 0.614]
    if abs(r) < 0.015 and abs(g - 0.146) < 0.015 and abs(b - 0.614) < 0.015:
        return True
    return False


def _infer_elemental_glow_color(
    primary_hint: str,
    fallback: list[float] | None = None,
    *,
    mesh_hint: str = "",
    mat_hint: str = "",
    variant_hint: str = "",
    machine_hint: str = "",
) -> list[float]:
    """Infer authentic linear elemental RGB glow color in strict priority order:
    1. Official HFW Focus Catalog part name (e.g. `Metalbite Sac` on Standard Slaughterspine vs `Purgewater Sac` on Apex Slaughterspine!)
    2. Mesh/Helper part name (e.g. `AcidSac_L`, `BlazeSac_C_01`, `L_glueSacMesh`)
    3. Material name (e.g. `AcidCanister_NMT`, `BlazeCanister_NMT`, `CryoCanister_NMT`)
    4. Machine Variant official name & description (e.g. `Fire Bellowback`, `Acid Bellowback`, `Frost Bellowback`, `Apex Bellowback` -> Adhesive)
    5. Machine species fallback (e.g. Bilegut -> Acid, Fireclaw -> Fire, Frostclaw/Snapmaw -> Cryo, Tideripper/Widemaw -> Purgewater)
    """
    for ctx in (primary_hint, mesh_hint, mat_hint, variant_hint):
        elem = _match_element_from_text(ctx)
        if elem is not None:
            return list(_ELEMENT_RGB[elem])

    # Species-level fallback when neither part nor variant specifies an element
    low = f"{primary_hint} {mesh_hint} {mat_hint} {variant_hint} {machine_hint}".lower()
    if any(k in low for k in ("inferno", "fireclaw")):
        return list(_ELEMENT_RGB["fire"])
    if any(k in low for k in ("bilegut", "frog", "snake", "slitherfang")):
        return list(_ELEMENT_RGB["acid"])
    if any(k in low for k in ("seabeast", "tideripper", "widemaw", "hippo")):
        return list(_ELEMENT_RGB["purgewater"])
    if any(k in low for k in ("cryobear", "frostclaw", "snapmaw", "beachlizard")):
        return list(_ELEMENT_RGB["cryo"])
    if any(k in low for k in ("plasmaspine", "slaughterspine")):
        return list(_ELEMENT_RGB["plasma"])
    return list(fallback) if fallback is not None else [0.08, 0.56, 1.0]


# Authentic HFW linear RGB machine variant palettes (Muscle Fibers, Cables & Optical Sensors)
MACHINE_VARIANT_PROFILES: dict[str, dict[str, Any]] = {
    "standard": {
        "profile": "standard",
        "label": "Standard",
        "is_apex": False,
        "muscle_color": [0.95, 0.28, 0.02],    # Warm HFW Amber-Gold optical muscle fibers
        "muscle_glow": 2.6,
        "cable_color": [0.95, 0.28, 0.02],     # Matched Warm Amber braided cables
        "cable_glow": 3.0,
        "eye_color": [0.04, 0.56, 1.0],        # Patrol Cyan-Blue optics
        "eye_alert_color": [1.0, 0.08, 0.01],  # Hostile Red optics
        "eye_glow": 9.0,
        "core_color": [0.08, 0.55, 1.0],
        "plasma_color": [0.46, 0.05, 1.0],     # Electric Plasma Violet
    },
    "apex": {
        "profile": "apex",
        "label": "Apex",
        "is_apex": True,
        "muscle_color": [0.54, 0.05, 1.0],     # Signature HFW Apex Violet-Purple muscles
        "muscle_glow": 2.8,
        "cable_color": [0.54, 0.05, 1.0],      # Matched Apex Violet-Purple braided cables
        "cable_glow": 3.2,
        "eye_color": [0.64, 0.05, 1.0],        # Signature Apex Violet-Magenta optics
        "eye_alert_color": [1.0, 0.04, 0.01],  # Apex Hostile Crimson optics
        "eye_glow": 9.5,
        "core_color": [0.58, 0.08, 1.0],
        "plasma_color": [0.58, 0.04, 1.0],
    },
    "corrupted": {
        "profile": "corrupted",
        "label": "Corrupted / Daemonic",
        "is_apex": False,
        "muscle_color": [1.0, 0.05, 0.01],     # Corrupted Crimson-Red muscles
        "muscle_glow": 2.8,
        "cable_color": [1.0, 0.05, 0.01],
        "cable_glow": 3.2,
        "eye_color": [1.0, 0.05, 0.01],        # Hostile Red optics
        "eye_alert_color": [1.0, 0.14, 0.01],
        "eye_glow": 9.5,
        "core_color": [1.0, 0.08, 0.02],
        "plasma_color": [1.0, 0.04, 0.08],
    },
    "overridden": {
        "profile": "overridden",
        "label": "Overridden",
        "is_apex": False,
        "muscle_color": [0.04, 0.52, 1.0],     # Overridden Electric Cyan-Blue muscles
        "muscle_glow": 2.7,
        "cable_color": [0.04, 0.52, 1.0],
        "cable_glow": 3.1,
        "eye_color": [0.04, 0.62, 1.0],        # Overridden Cyan-Blue optics
        "eye_alert_color": [1.0, 0.08, 0.01],
        "eye_glow": 9.5,
        "core_color": [0.05, 0.58, 1.0],
        "plasma_color": [0.06, 0.56, 1.0],
    },
}


def infer_machine_variant_profile(
    machine_hint: str = "",
    variant_hint: str = "",
    is_plasma_machine: bool = False,
) -> dict[str, Any]:
    """Infer authentic HFW muscle, cable, core, and eye color profile from variant metadata (Standard / Apex / Corrupted / Overridden)."""
    vlow = (variant_hint or "").lower()
    if "apex" in vlow:
        base = dict(MACHINE_VARIANT_PROFILES["apex"])
    elif any(k in vlow for k in ("corrupt", "daemonic", "rebel")):
        base = dict(MACHINE_VARIANT_PROFILES["corrupted"])
    elif "overrid" in vlow:
        base = dict(MACHINE_VARIANT_PROFILES["overridden"])
    else:
        base = dict(MACHINE_VARIANT_PROFILES["standard"])

    # Copy lists so callers can't mutate module globals
    for k in ("muscle_color", "cable_color", "eye_color", "eye_alert_color", "core_color", "plasma_color"):
        base[k] = list(base[k])
    base["is_plasma"] = bool(is_plasma_machine)
    base["plasma_charge"] = 1.0 if is_plasma_machine else 0.0
    base["animation_speed"] = 1.0
    base["liquid_glow_scale"] = 1.0
    base["liquid_override_fac"] = 0.0
    base["liquid_override_color"] = [0.08, 0.62, 1.0]
    return base


def _safe_name(text: str, max_len: int = 64) -> str:
    s = re.sub(r"[^A-Za-z0-9_\-.]+", "_", text).strip("_")
    return (s or "Asset")[:max_len]


def _target_channels_to_packing(target_channels: dict[str, str]) -> dict[str, PackingChannel]:
    out: dict[str, PackingChannel] = {}
    for lane, spec in target_channels.items():
        parts = spec.split(".")
        stype = parts[0]
        sch = parts[1] if len(parts) > 1 else "R"
        out[lane] = PackingChannel(tex_type=stype, source_channel=sch)
    return out


def _extract_material_textures(
    game: HFWGame,
    mat: MaterialInfo,
    tex_dir: Path,
    ts_packed_cache: dict[str, list[dict[str, Any]]],
    ts_unpacked_cache: dict[str, dict[str, str]],
    standalone_packed_cache: dict[str, dict[str, Any] | None],
    standalone_unpacked_cache: dict[str, dict[str, str]],
    array_atlas_cache: dict[str, dict[str, Any] | None],
    standalone_raw_cache: dict[str, dict[str, Any] | None],
    max_texture_size: int | None = None,
    unpack_textures: bool = False,
    is_plasma_machine: bool = False,
    machine_hint: str = "",
    variant_hint: str = "",
    catalog_part_hint: str = "",
    mesh_part_hint: str = "",
    variant_profile: dict[str, Any] | None = None,
    asset_kind: str = "machine",
    dxc: Any = None,
    skin_profile: dict[str, Any] | None = None,
    hair_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Export textures referenced by `mat`'s primary technique.

    By default (`unpack_textures=False`), exports the exact native game-packed RGBA textures
    (`packed_entries`) so Blender unpacks channels directly in shader nodes.
    When `unpack_textures=True`, exports individual unpacked semantic PNGs (`maps`).
    Always extracts Guerrilla's 23-slice `2DArray` machine detail atlas (`detail_array`) when bound,
    detects specialized machine muscle / glowing fiber / eye / core / canister / VFX shaders,
    and NEVER misclassifies status-effect BC5 maps (e.g. acid bubbles 241:0 / frost cracks 243:0) as detail normals.
    """
    packed_entries: list[dict[str, Any]] = []
    seen_packed_ids: set[str] = set()
    maps: dict[str, str] = {}
    detail_maps: dict[str, str] = {}
    detail_array: dict[str, Any] | None = None
    detail_tile_x = 12.0
    detail_tile_y = 12.0
    emissive_color: list[float] | None = None
    emissive_strength = 0.0
    special_shader: dict[str, Any] | None = None
    emissive_overlay: dict[str, Any] | None = None

    pt = mat.primary_technique
    if pt is None:
        return {
            "texture_mode": "unpacked" if unpack_textures else "packed",
            "packed_entries": packed_entries,
            "maps": maps,
            "detail_maps": detail_maps,
            "detail_array": detail_array,
            "detail_tile": [detail_tile_x, detail_tile_y],
            "emissive_color": emissive_color,
            "emissive_strength": emissive_strength,
            "special_shader": special_shader,
            "emissive_overlay": emissive_overlay,
        }

    # Inspect VariableBindings for Decima detail UV tiling scale & special shader parameters
    vb_by_hash = {vb.binding_name_hash: vb for vb in pt.variable_bindings}
    if 0x3806EAE3 in vb_by_hash:
        vb_tile = vb_by_hash[0x3806EAE3]
        if 0.5 < vb_tile.float4[0] <= 128.0:
            detail_tile_x = float(vb_tile.float4[0])
            detail_tile_y = float(vb_tile.float4[1]) if 0.5 < vb_tile.float4[1] <= 128.0 else detail_tile_x

    def _export_raw_tex(rid: str, label: str, color_space: str | None = None) -> dict[str, Any] | None:
        ckey = f"{rid}|{color_space or 'auto'}"
        if ckey not in standalone_raw_cache:
            try:
                tobj = game.read_object(ObjectId.parse(rid), read_subgroups=True)
                standalone_raw_cache[ckey] = export_standalone_texture(
                    game, tobj, label, tex_dir, color_space=color_space, max_dimension=max_texture_size
                )
            except Exception:
                standalone_raw_cache[ckey] = None
        return standalone_raw_cache[ckey]

    def _add_packed(entry: dict[str, Any] | None, src_path: str = "") -> None:
        if not entry:
            return
        tid = entry.get("texture_id") or entry.get("filename", "")
        if tid in seen_packed_ids:
            return
        seen_packed_ids.add(tid)
        e_copy = dict(entry)
        if src_path and not e_copy.get("source_path"):
            e_copy["source_path"] = src_path
        fn_low = f"{e_copy.get('filename', '')} {src_path}".lower()
        # Only explicit `UvSet02` textures (e.g. `AloyHair_Cards_UvSet02_CLR`) sample UV1;
        # texture sets named `*UV2_NMT` (like `SeykaTrimsUV2_NMT`) store UVs on UV0 while UV1 is [0,0]!
        if "uvset02" in fn_low:
            e_copy["uv_index"] = 1
        else:
            e_copy["uv_index"] = 0
        packed_entries.append(e_copy)

    tb_by_hash: dict[int, Any] = {}
    for tb in pt.texture_bindings:
        if tb.resource_id:
            tb_by_hash[tb.binding_name_hash] = tb

    is_machine_asset = asset_kind == "machine"

    # Helper to identify shared placeholder / weather / fuzz-atlas junk textures on HFW characters & assets
    _CHAR_JUNK_TOKENS = (
        "placeholder_nmt",
        "placeholder_clr",
        "placeholder",
        "black_2k",
        "idealmale_msk",
        "outfit_fuzz_atlas",
        "shaders_hrz2/materials/shared/effects/frost",
        "shaders_hrz2/materials/shared/effects/dirt",
        "shaders_hrz2/materials/shared/dirt",
        "shaders_hrz2/materials/shared/paint",
        "shaders_hrz2/materials/shared/filler",
        "paint_dry_base",
        "paint_thick",
        "paint_tar",
        "paint_01_",
        "frost_nmt",
        "dirt_nmt",
        "filler_nmt",
        "_wm1_",
        "_wm2_",
        "_wm3_",
        "wrinklemap",
    )

    def _tb_source_str(tb_obj: Any) -> str:
        vals = list(tb_obj.source_paths.values()) if tb_obj.source_paths else []
        return (" ".join(vals) + " " + (tb_obj.name or "")).replace("\\", "/")

    def _is_char_junk(s: str) -> bool:
        sl = s.lower()
        return any(tok in sl for tok in _CHAR_JUNK_TOKENS)

    def _is_layered_submaterial(s: str) -> bool:
        sl = s.lower()
        return ("shaders_hrz2/materials/" in sl) and ("/shared/" not in sl) and ("/effects/" not in sl)

    def _is_bb_tileable(s: str) -> bool:
        sl = s.lower()
        return ("building_block_tileables/" in sl) or ("/tileables/" in sl)

    def _is_world_moss_overlay(s: str) -> bool:
        sl = s.lower()
        return any(
            k in sl
            for k in (
                "shaders_hrz2/rock/moss/",
                "starmoss_clr",
                "snow_clumps_",
                "snow_heavyfresh_",
            )
        )

    # Pre-check whether this material has a unique RGB Mask (`MSK.RGB`) or unique `models/...` Color map
    has_rgb_mask_binding = any(
        (
            "Mask.R" in tb.target_channels.values()
            and "Mask.G" in tb.target_channels.values()
            and "Mask.B" in tb.target_channels.values()
        )
        or (
            tb.resource_type == "TextureSet"
            and "Mask.R" in tb.target_channels.values()
            and not _is_char_junk(_tb_source_str(tb))
        )
        for tb in pt.texture_bindings
    )
    has_unique_model_color_binding = any(
        any(v.startswith("Color.") for v in tb.target_channels.values())
        and (
            "models/" in _tb_source_str(tb).lower()
            or "dlc/models/" in _tb_source_str(tb).lower()
        )
        and not _is_char_junk(_tb_source_str(tb))
        and not _is_bb_tileable(_tb_source_str(tb))
        for tb in pt.texture_bindings
    )
    uses_skin_lut = (not is_machine_asset) and any(tb.resource_id == "570:0" for tb in pt.texture_bindings)
    uses_hair_lut = (not is_machine_asset) and any(
        tb.resource_id == "382:0" for t in mat.techniques for tb in t.texture_bindings
    )

    # Track ordered HFW Character Outfit Sub-Material Layers (`Shaders_HRZ2/Materials/<Cat>/<MatName>/...`)
    # AND Building Block Tileable Layers (`textures/building_block_tileables/<Cat>/<MatName>/...`)!
    outfit_layer_order: list[str] = []
    outfit_layer_map: dict[str, dict[str, Any]] = {}
    bb_layer_order: list[str] = []
    bb_layer_map: dict[str, dict[str, Any]] = {}

    def _record_bb_layer(entry: dict[str, Any], src_low: str) -> None:
        m_bb = re.search(r"(?:building_block_tileables|tileables)/([^/]+)/([^/]+)/", src_low)
        if m_bb:
            lkey = f"{m_bb.group(1)}/{m_bb.group(2)}"
            lcat, lname = m_bb.group(1), m_bb.group(2)
        else:
            lkey = Path(entry.get("filename", "tile")).stem.lower()
            lcat, lname = "tileable", lkey
        if lkey not in bb_layer_map:
            bb_layer_order.append(lkey)
            bb_layer_map[lkey] = {
                "name": lname,
                "category": lcat,
                "color_file": "",
                "normal_file": "",
                "rfl_file": "",
                "is_metallic": any(k in lkey for k in ("metal", "iron", "steel", "copper", "bronze", "brass", "rust", "chrome", "aluminum")),
            }
        pk = entry.get("packing", {})
        tset_t = {v.split(".", 1)[0] for v in pk.values()}
        if "Color" in tset_t and not bb_layer_map[lkey]["color_file"]:
            bb_layer_map[lkey]["color_file"] = entry["filename"]
        if "Normal" in tset_t and not bb_layer_map[lkey]["normal_file"]:
            bb_layer_map[lkey]["normal_file"] = entry["filename"]
        if "Reflectance" in tset_t and not bb_layer_map[lkey]["rfl_file"]:
            bb_layer_map[lkey]["rfl_file"] = entry["filename"]

    for tb in pt.texture_bindings:
        if not tb.resource_id:
            continue
        rid = tb.resource_id
        src_str = _tb_source_str(tb)
        src_low = src_str.lower()
        # On characters & general assets, skip placeholder UV checkerboards, weather frost/dirt, and fuzz-atlas overlays!
        if not is_machine_asset and _is_char_junk(src_str):
            continue
        # On general assets that ALREADY have their own unique `models/...` color map, skip world-space StarMoss overlay!
        if asset_kind == "asset" and has_unique_model_color_binding and _is_world_moss_overlay(src_str):
            continue
        # Capture `Shaders_HRZ2/Materials/<Cat>/<Mat>/...` as layered sub-materials ONLY when a character or masked asset uses them!
        # (On non-masked single-material props/weapons, keep them as direct packed textures so the prop isn't left blank!)
        is_submat = (not is_machine_asset) and _is_layered_submaterial(src_str) and (asset_kind == "character" or has_rgb_mask_binding)
        is_bb_tile = (asset_kind == "asset") and _is_bb_tileable(src_str)
        target_dict = detail_maps if tb.is_detail_texture else maps

        if tb.resource_type == "TextureSet":
            if not unpack_textures:
                if rid not in ts_packed_cache:
                    try:
                        ts_obj = game.read_object(ObjectId.parse(rid), read_subgroups=True)
                        ts_packed_cache[rid] = export_packed_texture_set(
                            game, ts_obj, tex_dir, max_dimension=max_texture_size
                        )
                    except Exception:
                        ts_packed_cache[rid] = []
                for entry in ts_packed_cache[rid]:
                    if is_submat:
                        # Record Reflectance/Metallic or Color/Normal on this tiling sub-material layer
                        m_sub = re.search(r"shaders_hrz2/materials/([^/]+)/([^/]+)/", src_low)
                        if m_sub:
                            lkey = f"{m_sub.group(1)}/{m_sub.group(2)}"
                            if lkey not in outfit_layer_map:
                                outfit_layer_order.append(lkey)
                                outfit_layer_map[lkey] = {
                                    "name": m_sub.group(2),
                                    "category": m_sub.group(1),
                                    "color_file": "",
                                    "normal_file": "",
                                    "rfl_file": "",
                                    "is_metallic": m_sub.group(1) in ("metal", "metals") or any(k in m_sub.group(2) for k in ("metal", "iron", "bronze", "copper", "gold", "silver", "steel", "brass")),
                                }
                            pk = entry.get("packing", {})
                            tset_t = {v.split(".", 1)[0] for v in pk.values()}
                            is_dmg = "damaged" in entry.get("filename", "").lower() or "damaged" in src_low
                            if "Reflectance" in tset_t and (not outfit_layer_map[lkey]["rfl_file"] or not is_dmg):
                                outfit_layer_map[lkey]["rfl_file"] = entry["filename"]
                                outfit_layer_map[lkey]["is_metallic"] = True
                            if "Color" in tset_t and (not outfit_layer_map[lkey]["color_file"] or not is_dmg):
                                outfit_layer_map[lkey]["color_file"] = entry["filename"]
                            if "Normal" in tset_t and (not outfit_layer_map[lkey]["normal_file"] or not is_dmg):
                                outfit_layer_map[lkey]["normal_file"] = entry["filename"]
                    elif is_bb_tile:
                        _record_bb_layer(entry, src_low)
                    elif (
                        (not tb.is_detail_texture)
                        or (asset_kind == "asset" and (_is_layered_submaterial(src_str) or "models/" in src_low or "dlc/models/" in src_low))
                    ):
                        _add_packed(entry, src_str)
            else:
                if rid not in ts_unpacked_cache:
                    try:
                        ts_obj = game.read_object(ObjectId.parse(rid), read_subgroups=True)
                        uts = unpack_texture_set(game, ts_obj, max_dimension=max_texture_size)
                        ts_unpacked_cache[rid] = save_unpacked_texture_set(uts, tex_dir)
                    except Exception:
                        ts_unpacked_cache[rid] = {}
                if not is_submat:
                    for mtype, fname in ts_unpacked_cache[rid].items():
                        target_dict.setdefault(mtype, fname)

        elif tb.resource_type == "Texture":
            # Case A: Decima's mask-driven Machine Detail Texture2DArray (e.g. 20:0 256x256x23 BC7)!
            if tb.binding_name_hash in _DETAIL_ARRAY_BINDING_HASHES:
                if rid not in array_atlas_cache:
                    try:
                        tex_obj = game.read_object(ObjectId.parse(rid), read_subgroups=True)
                        array_atlas_cache[rid] = export_texture_array_atlas(
                            game, tex_obj, tex_dir, cols=5, max_slice_dim=256
                        )
                    except Exception:
                        array_atlas_cache[rid] = None
                if array_atlas_cache[rid] is not None and detail_array is None:
                    detail_array = dict(array_atlas_cache[rid])
                    detail_array["tile_x"] = detail_tile_x
                    detail_array["tile_y"] = detail_tile_y
                continue

            # Case B: Direct Texture binding that belongs to a TextureSetParent (has explicit target_channels!)
            if tb.target_channels:
                if not unpack_textures:
                    if rid not in standalone_packed_cache:
                        try:
                            tex_obj = game.read_object(ObjectId.parse(rid), read_subgroups=True)
                            packing = _target_channels_to_packing(tb.target_channels)
                            standalone_packed_cache[rid] = export_packed_texture(
                                game, tex_obj, packing, tb.name, tex_dir, max_dimension=max_texture_size
                            )
                        except Exception:
                            standalone_packed_cache[rid] = None
                    entry = standalone_packed_cache[rid]
                    if is_submat and entry is not None:
                        m_sub = re.search(r"shaders_hrz2/materials/([^/]+)/([^/]+)/", src_low)
                        if m_sub:
                            lkey = f"{m_sub.group(1)}/{m_sub.group(2)}"
                            if lkey not in outfit_layer_map:
                                outfit_layer_order.append(lkey)
                                outfit_layer_map[lkey] = {
                                    "name": m_sub.group(2),
                                    "category": m_sub.group(1),
                                    "color_file": "",
                                    "normal_file": "",
                                    "rfl_file": "",
                                    "is_metallic": m_sub.group(1) in ("metal", "metals") or any(k in m_sub.group(2) for k in ("metal", "iron", "bronze", "copper", "gold", "silver", "steel", "brass")),
                                }
                            pk = entry.get("packing", {})
                            tset_t = {v.split(".", 1)[0] for v in pk.values()}
                            is_dmg = "damaged" in entry.get("filename", "").lower() or "damaged" in src_low
                            if "Color" in tset_t and (not outfit_layer_map[lkey]["color_file"] or not is_dmg):
                                outfit_layer_map[lkey]["color_file"] = entry["filename"]
                            if "Normal" in tset_t and (not outfit_layer_map[lkey]["normal_file"] or not is_dmg):
                                outfit_layer_map[lkey]["normal_file"] = entry["filename"]
                            if "Reflectance" in tset_t and (not outfit_layer_map[lkey]["rfl_file"] or not is_dmg):
                                outfit_layer_map[lkey]["rfl_file"] = entry["filename"]
                                outfit_layer_map[lkey]["is_metallic"] = True
                    elif is_bb_tile and entry is not None:
                        _record_bb_layer(entry, src_low)
                    elif (not tb.is_detail_texture) or (asset_kind == "asset" and _is_layered_submaterial(src_str)):
                        _add_packed(entry, src_str)
                else:
                    if rid not in standalone_unpacked_cache:
                        saved: dict[str, str] = {}
                        try:
                            tex_obj = game.read_object(ObjectId.parse(rid), read_subgroups=True)
                            dec = decode_texture(game, tex_obj, max_dimension=max_texture_size)
                            if dec is not None:
                                safe = _safe_name(tb.name, 50)
                                uid = rid.replace(":", "_")
                                by_type: dict[str, dict[int, np.ndarray]] = {}
                                for lane, spec in tb.target_channels.items():
                                    parts = spec.split(".")
                                    stype = parts[0]
                                    sch = parts[1] if len(parts) > 1 else "R"
                                    src_idx = LANE_TO_IDX.get(lane, 0)
                                    dst_idx = LANE_TO_IDX.get(sch, 0)
                                    by_type.setdefault(stype, {})[dst_idx] = dec.rgba[:, :, src_idx]
                                for stype, ch_map in by_type.items():
                                    if stype == "Normal" and 0 in ch_map and 1 in ch_map:
                                        for suffix, flip in (("Normal_DX", False), ("Normal_GL", True)):
                                            fn = f"{safe}_{uid}_{suffix}.png"
                                            fp = tex_dir / fn
                                            if not fp.exists():
                                                Image.fromarray(
                                                    reconstruct_normal_map(ch_map[0], ch_map[1], flip), mode="RGB"
                                                ).save(fp)
                                            saved[suffix] = fn
                                    elif stype in ("Color", "Incandescence"):
                                        h, w = next(iter(ch_map.values())).shape
                                        z = np.zeros((h, w), dtype=np.uint8)
                                        rgb = np.stack([ch_map.get(0, z), ch_map.get(1, z), ch_map.get(2, z)], axis=-1)
                                        fn = f"{safe}_{uid}_{stype}.png"
                                        fp = tex_dir / fn
                                        if not fp.exists():
                                            Image.fromarray(rgb, mode="RGB").save(fp)
                                        saved[stype] = fn
                                    else:
                                        arr = next(iter(ch_map.values()))
                                        fn = f"{safe}_{uid}_{stype}.png"
                                        fp = tex_dir / fn
                                        if not fp.exists():
                                            Image.fromarray(arr, mode="L").save(fp)
                                        saved[stype] = fn
                        except Exception:
                            pass
                        standalone_unpacked_cache[rid] = saved
                    if not is_submat:
                        for mtype, fname in standalone_unpacked_cache[rid].items():
                            target_dict.setdefault(mtype, fname)
            # Case C: Standalone textures WITHOUT TextureSetParent (status-effect maps 241:0/243:0, weather 16:0, etc.)
            # are skipped here; specialized muscle/core/VFX shaders below explicitly export the ones they need.

    # Standalone fallback for General Assets whose TextureBindings lack a TextureSetParent (no target_channels)
    _GLOBAL_ENGINE_TEX_IDS = frozenset({
        "1:0", "2:2", "3:1", "3:2", "8:0", "10:0", "11:0", "13:0", "13:1", "13:2",
        "14:0", "14:1", "15:0", "16:0", "24:0", "31:0", "32:0", "50:0", "56:0",
        "74:0", "191:0", "297:0", "570:0", "869:0",
    })
    if asset_kind == "asset" and not packed_entries and not maps and not outfit_layer_order and not bb_layer_order:
        for tb in pt.texture_bindings:
            rid = tb.resource_id
            if not rid or tb.resource_type != "Texture" or rid in _GLOBAL_ENGINE_TEX_IDS:
                continue
            try:
                tobj = game.read_object(ObjectId.parse(rid), read_subgroups=True)
                dec = decode_texture(game, tobj, max_dimension=max_texture_size)
                if dec is None or dec.width < 64 or dec.height < 64:
                    continue
                pf = (dec.pixel_format or "").upper()
                cs = (dec.color_space or "").lower()
                if "BC5" in pf:
                    pk = {"R": "Normal.R", "G": "Normal.G"}
                    cspace = "Non-Color"
                elif "srgb" in cs or "BC1" in pf or "BC7" in pf:
                    pk = {"R": "Color.R", "G": "Color.G", "B": "Color.B"}
                    cspace = "sRGB"
                else:
                    continue
                raw = _export_raw_tex(rid, tb.name or f"AssetTex_{rid.replace(':', '_')}", cspace)
                if raw is not None:
                    raw_e = dict(raw)
                    raw_e["packing"] = pk
                    _add_packed(raw_e, _tb_source_str(tb))
            except Exception:
                continue

    # On non-machine assets, sort packed_entries so unique `models/...` asset textures ALWAYS beat shared/shader overlays!
    if not is_machine_asset and len(packed_entries) > 1:
        def _pe_priority(e: dict[str, Any]) -> tuple[int, int, int]:
            sp = f"{e.get('source_path', '')} {e.get('filename', '')}".replace("\\", "/").lower()
            is_shader_overlay = 1 if ("shaders_hrz2/" in sp or "shaders/" in sp or "starmoss" in sp) else 0
            is_shared = 1 if "shared" in sp else 0
            area = int(e.get("width", 0)) * int(e.get("height", 0))
            return (is_shader_overlay, is_shared, -area)
        packed_entries.sort(key=_pe_priority)

    # Detect specialized DXIL-verified HFW machine shaders (braided muscles, glowing body muscle fibers,
    # inner eye optics, canister cores/glass, Thunderjaw Brain/Heart & Slitherfang coils, VFX overlay shells).
    vprof = variant_profile or infer_machine_variant_profile(machine_hint, variant_hint, is_plasma_machine=is_plasma_machine)
    passes_set = set(pt.available_passes)
    has_tset = bool(mat.texture_set_ids)
    has_mapped = bool(packed_entries or maps)

    if is_machine_asset and (_HASH_MUSCLE_BRAID in tb_by_hash or (_HASH_MUSCLE_VEINS in tb_by_hash and _HASH_MUSCLE_PULSE in tb_by_hash)):
        # Class A: Braided mechanical muscle cables (Guerrilla shader 8260:1, e.g. Clawstrider 5540:23 / 5540:35)
        # Uses authentic variant muscle color (Standard Amber / Apex Purple / Corrupted Red / Overridden Cyan) + shared control!
        braid = _export_raw_tex(tb_by_hash[_HASH_MUSCLE_BRAID].resource_id, "HFW_MuscleBraid", "Non-Color") if _HASH_MUSCLE_BRAID in tb_by_hash else None
        veins = _export_raw_tex(tb_by_hash[_HASH_MUSCLE_VEINS].resource_id, "HFW_MuscleVeins", "Non-Color") if _HASH_MUSCLE_VEINS in tb_by_hash else None
        pulse = _export_raw_tex(tb_by_hash[_HASH_MUSCLE_PULSE].resource_id, "HFW_MusclePulse", "Non-Color") if _HASH_MUSCLE_PULSE in tb_by_hash else None
        ramp = _export_raw_tex(tb_by_hash[_HASH_MUSCLE_RAMP_A].resource_id, "HFW_MuscleRamp", "Non-Color") if _HASH_MUSCLE_RAMP_A in tb_by_hash else None
        special_shader = {
            "kind": "muscle_cables",
            "braid_file": braid["filename"] if braid else "",
            "veins_file": veins["filename"] if veins else "",
            "pulse_file": pulse["filename"] if pulse else "",
            "ramp_file": ramp["filename"] if ramp else "",
            "glow_color": list(vprof["cable_color"]),
            "alert_color": list(vprof["eye_alert_color"]),
            "glow_strength": float(vprof["cable_glow"]),
            "variant_profile": vprof["profile"],
        }
    elif is_machine_asset and (not has_tset) and ("ForwardBackground" in passes_set) and (_HASH_MUSCLE_RAMP_B in tb_by_hash):
        # Class B: Translucent glowing body muscle fiber overlay (Guerrilla shaders 19501:0 / 21395:0 / 20640:0 / 26679:0)
        # Present on Slaughterspine, Clawstrider, Slitherfang, Thunderjaw, Tremortusk, Bilegut, Dreadwing, etc.!
        ramp = _export_raw_tex(tb_by_hash[_HASH_MUSCLE_RAMP_B].resource_id, "HFW_MuscleFiberRamp", "Non-Color")
        dissolve = _export_raw_tex(tb_by_hash[_HASH_MUSCLE_DISSOLVE].resource_id, "HFW_MuscleDissolve", "Non-Color") if _HASH_MUSCLE_DISSOLVE in tb_by_hash else None
        special_shader = {
            "kind": "muscle_fiber_overlay",
            "ramp_file": ramp["filename"] if ramp else "",
            "dissolve_file": dissolve["filename"] if dissolve else "",
            # Authentic variant muscle colors (Standard Warm Amber vs Apex Purple vs Corrupted Red vs Slaughterspine Plasma!)
            "calm_color": list(vprof["muscle_color"]),
            "plasma_color": list(vprof["plasma_color"]) if is_plasma_machine else list(vprof["muscle_color"]),
            "alert_color": list(vprof["eye_alert_color"]),
            "charge_state": 1.0 if is_plasma_machine else 0.0,
            "alert_state": 0.0,
            "glow_strength": float(vprof["muscle_glow"]),
            "fiber_density": 96.0,
            "normal_offset": 0.0014,
            "variant_profile": vprof["profile"],
        }
    elif is_machine_asset and _HASH_CORE_COLOR in tb_by_hash and _HASH_CORE_NORMAL in tb_by_hash:
        # Class C: Machine glowing energy core / Brain / Heart / Neck coils (32906:0, 1348:90)
        core_col = _export_raw_tex(tb_by_hash[_HASH_CORE_COLOR].resource_id, "HFW_CoreColor", "sRGB")
        core_nrm = _export_raw_tex(tb_by_hash[_HASH_CORE_NORMAL].resource_id, "HFW_CoreNormal", "Non-Color")
        tint = list(vprof["core_color"])
        if not vprof["is_apex"] and _HASH_CORE_TINT_VB in vb_by_hash:
            f4 = vb_by_hash[_HASH_CORE_TINT_VB].float4
            if max(f4[:3]) > 0.01:
                tint = [float(f4[0]), float(f4[1]), float(f4[2])]
        special_shader = {
            "kind": "machine_core_emissive",
            "color_file": core_col["filename"] if core_col else "",
            "normal_file": core_nrm["filename"] if core_nrm else "",
            "glow_color": tint,
            "glow_strength": 5.0,
        }
    elif is_machine_asset and (not has_tset) and (not has_mapped) and ("ForwardForeground" in passes_set) and (_HASH_VFX_SHELL_LUT in tb_by_hash):
        # Class D: Full-body UV-less Focus/Stealth/VFX overlay shell (37840:0, 38845:0) -> transparent & hidden in Focus_Overlay!
        special_shader = {
            "kind": "vfx_overlay_shell",
            "glow_color": [0.25, 0.10, 1.0] if is_plasma_machine else [0.05, 0.65, 1.0],
            "opacity": 0.0,
        }
    elif is_machine_asset and (not has_tset) and (not has_mapped) and ("ForwardForeground" in passes_set) and (_HASH_GLASS_TINT_VB in vb_by_hash):
        # Class E: Translucent canister outer glass shell (41249:0, e.g. 14705:34)
        f4 = vb_by_hash[_HASH_GLASS_TINT_VB].float4
        special_shader = {
            "kind": "canister_glass",
            "tint_color": [float(f4[0]), float(f4[1]), float(f4[2])],
            "alpha": 0.22,
            "ior": 1.45,
        }
    elif is_machine_asset and (not has_tset) and (not has_mapped) and ("DeferredEmissive" in passes_set) and (_HASH_MUSCLE_DISSOLVE in tb_by_hash):
        if _HASH_EYE_VB in vb_by_hash:
            # Class F: Inner glowing machine eye optical sensor (4628:0)
            special_shader = {
                "kind": "machine_eye_glow",
                "glow_color": list(vprof["eye_color"]),
                "alert_color": list(vprof["eye_alert_color"]),
                "glow_strength": float(vprof["eye_glow"]),
                "variant_profile": vprof["profile"],
            }
        else:
            # Class G: Elemental canister glowing fluid core (21080:1, e.g. Shock/Plasma/Cryo/Acid/Blaze canister core)
            noise_tex = _export_raw_tex(tb_by_hash[_HASH_MUSCLE_DISSOLVE].resource_id, "HFW_CanisterCoreNoise", "Non-Color")
            if mat.object_id.startswith("13567:"):
                c_col = [0.48, 0.06, 1.0]
            elif mat.object_id.startswith("10036:"):
                c_col = [0.05, 0.50, 1.0]
            elif mat.object_id.startswith("14705:"):
                c_col = [0.16, 0.72, 1.0]
            elif mat.object_id.startswith("13451:"):
                c_col = [0.46, 0.90, 0.06]
            elif mat.object_id.startswith("13323:"):
                c_col = [1.0, 0.45, 0.02]
            elif mat.object_id.startswith("15271:"):
                c_col = [0.08, 0.52, 0.96]
            else:
                c_col = _infer_elemental_glow_color(
                    catalog_part_hint,
                    [0.42, 0.05, 1.0] if is_plasma_machine else [0.05, 0.48, 1.0],
                    mesh_hint=mesh_part_hint,
                    mat_hint=mat.name,
                    variant_hint=variant_hint,
                    machine_hint=machine_hint,
                )
            special_shader = {
                "kind": "canister_glow_core",
                "noise_file": noise_tex["filename"] if noise_tex else "",
                "glow_color": c_col,
                "glow_strength": 6.8,
            }

    # Detect DXIL-verified Emissive/Animated Overlays on standard PBR materials (sacks, canisters, outer eyes)!
    if is_machine_asset and special_shader is None:
        shader_id = pt.shader_id or ""

        # Overlay 1: Elemental Sacks (Cryo, Acid/Metalbite, Blaze/Fire, Purgewater, Adhesive, Fertilizer, Spikesnout!)
        # Exact DXIL signature: binds 0x7c771b98 (172:0 Flow Map) + 0x23fe87db (1703:0 Fluid Caustics) + mat_id==0 * Mask_Alpha!
        if _HASH_SAC_FLOW_MAP in tb_by_hash and _HASH_SAC_FLUID_TEX in tb_by_hash:
            flow_tex = _export_raw_tex(tb_by_hash[_HASH_SAC_FLOW_MAP].resource_id, "HFW_SacFlowMap", "Non-Color")
            fluid_tex = _export_raw_tex(tb_by_hash[_HASH_SAC_FLUID_TEX].resource_id, "HFW_SacFluidCaustics", "Non-Color")
            sac_col = None
            # Only trust PerInstance 0x249F1E74 when it is NOT one of Decima's runtime-overridden placeholders
            # (such as [0.135, 0.27, 0.54] default blue, [0.0, 1.0, 0.0] debug green, or [0.057, 0.057, 0.057] gray!).
            if _HASH_SAC_COLOR_VB in vb_by_hash:
                f4 = vb_by_hash[_HASH_SAC_COLOR_VB].float4
                if not _is_placeholder_sac_color(f4):
                    sac_col = [float(f4[0]), float(f4[1]), float(f4[2])]
            if sac_col is None:
                sac_col = _infer_elemental_glow_color(
                    catalog_part_hint,
                    mesh_hint=mesh_part_hint,
                    mat_hint=mat.name,
                    variant_hint=variant_hint,
                    machine_hint=machine_hint,
                )
            emissive_overlay = {
                "kind": "elemental_sack_overlay",
                "flow_file": flow_tex["filename"] if flow_tex else "",
                "fluid_file": fluid_tex["filename"] if fluid_tex else "",
                "glow_color": sac_col,
                "glow_strength": 3.4,
                "flow_speed": 0.35,
                "flow_strength": 0.14,
                "mask_id": 0.0,  # Exact DXIL L476: round(Mask * 20) == 0 * Mask_Alpha
            }

        # Overlay 2: Elemental Canisters (Acid 13451:37/69, Water 15271:21/61, Blaze 13323:33/56, Plasma 13567:44, Shock 10036:34)
        elif (
            shader_id in _CANISTER_GLOW_SHADERS
            or "canister" in mat.name.lower()
            or (
                _HASH_EMISSIVE_TINT_VB in vb_by_hash
                and max(vb_by_hash[_HASH_EMISSIVE_TINT_VB].float4[:3]) > 0.25
                and not (
                    # Exclude Decima's ubiquitous default debug green [0.1, 0.7, 0.05] on ordinary armor plates
                    abs(vb_by_hash[_HASH_EMISSIVE_TINT_VB].float4[0] - 0.10) < 0.02
                    and abs(vb_by_hash[_HASH_EMISSIVE_TINT_VB].float4[1] - 0.70) < 0.02
                    and abs(vb_by_hash[_HASH_EMISSIVE_TINT_VB].float4[2] - 0.05) < 0.02
                )
            )
        ):
            can_col = None
            if _HASH_EMISSIVE_TINT_VB in vb_by_hash:
                f4 = vb_by_hash[_HASH_EMISSIVE_TINT_VB].float4
                is_default_gray = max(f4[:3]) <= 0.12
                is_default_green = abs(f4[0] - 0.10) < 0.02 and abs(f4[1] - 0.70) < 0.02 and abs(f4[2] - 0.05) < 0.02
                if not is_default_gray and not is_default_green:
                    can_col = [float(f4[0]), float(f4[1]), float(f4[2])]
            if can_col is None:
                can_col = _infer_elemental_glow_color(
                    catalog_part_hint,
                    mesh_hint=mesh_part_hint,
                    mat_hint=mat.name,
                    variant_hint=variant_hint,
                    machine_hint=machine_hint,
                )
            noise_tex = (
                _export_raw_tex(tb_by_hash[_HASH_MUSCLE_DISSOLVE].resource_id, "HFW_CanisterNoise", "Non-Color")
                if _HASH_MUSCLE_DISSOLVE in tb_by_hash
                else None
            )
            emissive_overlay = {
                "kind": "canister_emissive_overlay",
                "noise_file": noise_tex["filename"] if noise_tex else "",
                "glow_color": can_col,
                "glow_strength": 6.0 if shader_id == "7564:6" else 4.5,
                "pulse_speed": 8.0,  # Exact DXIL acid_canister_shell L994: sin(Time * 8.0) * 0.3 + 0.4
                "is_fluid_body": shader_id == "7564:6",
            }

        # Overlay 3: Outer Machine Eye Lens with POM Iris Glow Mask (`0x5e969132` -> 8837:0 / 20273:0 / 15787:3)
        elif _HASH_EYE_IRIS_MASK in tb_by_hash:
            iris_tex = _export_raw_tex(tb_by_hash[_HASH_EYE_IRIS_MASK].resource_id, "HFW_EyeIrisMask", "Non-Color")
            emissive_overlay = {
                "kind": "machine_eye_lens_overlay",
                "iris_file": iris_tex["filename"] if iris_tex else "",
                "calm_color": list(vprof["eye_color"]),
                "alert_color": list(vprof["eye_alert_color"]),
                "alert_state": 0.0,
                "glow_strength": float(vprof["eye_glow"]),
                "variant_profile": vprof["profile"],
            }

        # Overlay 4: Slaughterspine Plasma Charge-Up Shader (`dxil_ss_plasmaspike.ll` L623-768 & L725-732)!
        # Uses Vertex Color `Color0.R` (PlasmaSpikes, PlasmaChargers, PlasmaCore, TailBlasters, Jaw/Rib Vents),
        # `Color0.G` (Head-to-Tail plasma surge wave), `(1 - AO_remap)^2` crevice coil focus, `mat_id == 1` body conduits,
        # and animated `107:0` plasma turbulence noise! Ordinary non-plasma armor plates stay 0 emission automatically.
        elif is_plasma_machine and ("DeferredEmissive" in passes_set) and (has_mapped or has_tset):
            noise_tex = (
                _export_raw_tex(tb_by_hash[_HASH_MUSCLE_DISSOLVE].resource_id, "HFW_PlasmaNoise", "Non-Color")
                if _HASH_MUSCLE_DISSOLVE in tb_by_hash
                else _export_raw_tex("107:0", "HFW_PlasmaNoise", "Non-Color")
            )
            emissive_overlay = {
                "kind": "plasma_charge_overlay",
                "noise_file": noise_tex["filename"] if noise_tex else "",
                "plasma_color": list(vprof["plasma_color"]),
                "rim_color": [0.18, 0.03, 0.95],
                "idle_led_color": list(vprof["muscle_color"]),
                "plasma_charge": 1.0,
                "glow_strength": 4.2,
            }

    # Build HFW 8-Layer Character Outfit Shader OR Building Block Tileable Layered Shader!
    outfit_layered: dict[str, Any] | None = None
    if not is_machine_asset and (outfit_layer_order or bb_layer_order):
        unique_clr_entry = next(
            (
                e
                for e in packed_entries
                if "Color.R" in e.get("packing", {}).values()
                and "Color.G" in e.get("packing", {}).values()
                and "Color.B" in e.get("packing", {}).values()
                and "shared" not in e.get("filename", "").lower()
                and "starmoss" not in e.get("filename", "").lower()
            ),
            None,
        )
        has_unique_clr = unique_clr_entry is not None
        msk_entry = next(
            (
                e
                for e in packed_entries
                if "Mask.R" in e.get("packing", {}).values()
                and "Mask.G" in e.get("packing", {}).values()
                and "Mask.B" in e.get("packing", {}).values()
                and "shared" not in e.get("filename", "").lower()
            ),
            None,
        )
        msk_prefix = ""
        if msk_entry is not None:
            m_fn = msk_entry.get("filename", "")
            msk_prefix = re.sub(r"_\d+_\d+_[A-Z_]+\.png$", "_", m_fn).lower()
        nrm_candidates = [
            e
            for e in packed_entries
            if "Normal.R" in e.get("packing", {}).values()
            and "Normal.G" in e.get("packing", {}).values()
        ]
        nrm_entry = (
            next((e for e in nrm_candidates if msk_prefix and e.get("filename", "").lower().startswith(msk_prefix)), None)
            or (nrm_candidates[0] if nrm_candidates else None)
        )
        misc_candidates = [
            e
            for e in packed_entries
            if any(v.startswith("Misc_01") for v in e.get("packing", {}).values())
            and "shared" not in e.get("filename", "").lower()
        ]
        misc_entry = (
            next((e for e in misc_candidates if msk_prefix and e.get("filename", "").lower().startswith(msk_prefix)), None)
            or (misc_candidates[0] if misc_candidates else None)
        )

        # Case 1: Character (or masked prop) 8-Layer Outfit Shader (`Shaders_HRZ2/Materials/*` + `MSK.RGB`)
        # Also runs on hybrid character outfits that combine 8-layer DXIL sub-materials WITH a macro painted
        # `_CLR` map (e.g. `SeykaTorso_NMT` coral scarf + crochet cotton, `SeykaTrimsUV2_NMT` trims/Focus!)
        if outfit_layer_order and msk_entry is not None and (not has_unique_clr or asset_kind == "character"):
            has_msk_alpha = any(v.startswith("Alpha") for v in msk_entry.get("packing", {}).values())
            dxil_corners = extract_dxil_outfit_corners(game, mat, dxc=dxc)
            if dxil_corners:
                # If a macro painted `_CLR` map exists alongside `outfit_dxil`, check if it is chromatic
                # AND whether its neutral background is mid-grey linear ~0.50 (`sRGB ~186`, e.g. `SeykaTorso_NMT`,
                # meaning `2.0 * macro * dye` overlay!) vs direct painted albedo (`SeykaTrimsUV2_NMT` trim sheet!).
                macro_clr_file = ""
                macro_clr_mode = "replace"
                macro_clr_is_chromatic = False
                if unique_clr_entry is not None:
                    mc_path = tex_dir / unique_clr_entry["filename"]
                    if mc_path.exists():
                        try:
                            im_arr = np.asarray(Image.open(mc_path).convert("RGB"), dtype=np.float32)
                            rg_diff = float(np.percentile(np.abs(im_arr[:, :, 0] - im_arr[:, :, 1]), 92))
                            rb_diff = float(np.percentile(np.abs(im_arr[:, :, 0] - im_arr[:, :, 2]), 92))
                            macro_clr_is_chromatic = max(rg_diff, rb_diff) > 18.0
                            if macro_clr_is_chromatic:
                                macro_clr_file = unique_clr_entry["filename"]
                                med_rgb = np.median(im_arr.reshape(-1, 3), axis=0)
                                if 165.0 <= float(med_rgb.mean()) <= 205.0 and float(med_rgb.max() - med_rgb.min()) < 10.0:
                                    macro_clr_mode = "multiply2"
                        except Exception:
                            pass

                # Exact 8-corner DXIL mapping: corner 0..7 -> exact authored dye_color_linear, metallic, roughness, & submat texture!
                fallback_sub = outfit_layer_map[outfit_layer_order[0]]
                layers_8: list[dict[str, Any]] = []
                for dc in dxil_corners:
                    skey = dc.get("submat_key", "")
                    sub = outfit_layer_map.get(skey)
                    if sub is None and skey:
                        # Fuzzy match leaf name (e.g. `silk` vs `weave/silk`)
                        leaf = skey.split("/")[-1]
                        sub = next((v for k, v in outfit_layer_map.items() if k.endswith("/" + leaf) or v.get("name") == leaf), None)
                    if sub is None:
                        sub = fallback_sub if dc.get("active") else {"name": dc["corner_name"], "category": "", "color_file": "", "normal_file": "", "rfl_file": ""}
                    layers_8.append(
                        {
                            "corner": dc["corner"],
                            "corner_name": dc["corner_name"],
                            "name": sub.get("name") or dc["corner_name"],
                            "category": sub.get("category", ""),
                            "color_file": sub.get("color_file", ""),
                            "normal_file": sub.get("normal_file", ""),
                            "rfl_file": sub.get("rfl_file", ""),
                            "dye_color_linear": list(dc["dye_color_linear"]),
                            "dye_color_srgb": list(dc["dye_color_srgb"]),
                            "f0_rgb": list(dc["f0_rgb"]),
                            "is_metallic": bool(dc["is_metallic"]),
                            "roughness": float(dc["roughness"]),
                            "active": bool(dc["active"]),
                        }
                    )
                outfit_layered = {
                    "mode": "outfit_dxil",
                    "corner_order": "dxil_exact",
                    "mask_file": msk_entry["filename"],
                    "mask_has_alpha": has_msk_alpha,
                    "mask_has_roughness": any(v.startswith("Roughness") for v in msk_entry.get("packing", {}).values()),
                    "normal_file": nrm_entry["filename"] if nrm_entry else "",
                    "normal_packing": nrm_entry.get("packing", {}) if nrm_entry else {},
                    "misc_file": misc_entry["filename"] if misc_entry else "",
                    "macro_color_file": macro_clr_file,
                    "macro_color_mode": macro_clr_mode,
                    "uv_scale": 16.0,
                    "tribe_hint": f"{machine_hint} {variant_hint} {mat.name}".lower(),
                    "layers": layers_8,
                }
            elif not has_unique_clr:
                layers_list = [
                    outfit_layer_map[k]
                    for k in outfit_layer_order[:8]
                    if outfit_layer_map[k].get("color_file")
                ]
                if layers_list:
                    outfit_layered = {
                        "mode": "outfit_fallback",
                        "corner_order": "dxil_exact",
                        "mask_file": msk_entry["filename"],
                        "mask_has_alpha": has_msk_alpha,
                        "normal_file": nrm_entry["filename"] if nrm_entry else "",
                        "normal_packing": nrm_entry.get("packing", {}) if nrm_entry else {},
                        "misc_file": misc_entry["filename"] if misc_entry else "",
                        "macro_color_file": "",
                        "uv_scale": 16.0,
                        "tribe_hint": f"{machine_hint} {variant_hint} {mat.name}".lower(),
                        "layers": layers_list,
                    }
        # Case 2: Building Blocks / Ruins Multi-Layer Tileable Shader (`textures/building_block_tileables/*` + `MSK.RGB`)!
        elif bb_layer_order and not has_unique_clr:
            bb_list = [
                bb_layer_map[k]
                for k in bb_layer_order[:8]
                if bb_layer_map[k].get("color_file") or bb_layer_map[k].get("normal_file")
            ]
            if bb_list:
                outfit_layered = {
                    "mode": "building_block",
                    "corner_order": "dxil_exact",
                    "mask_file": msk_entry["filename"] if msk_entry else "",
                    "mask_has_alpha": False,
                    "mask_has_roughness": bool(msk_entry and any(v.startswith("Roughness") for v in msk_entry.get("packing", {}).values())),
                    "normal_file": nrm_entry["filename"] if nrm_entry else (bb_list[0].get("normal_file", "")),
                    "normal_packing": nrm_entry.get("packing", {}) if nrm_entry else {"R": "Normal.R", "G": "Normal.G", "B": "Roughness.R"},
                    "uv_scale": 4.0,
                    "tribe_hint": f"{machine_hint} {mat.name}".lower(),
                    "layers": bb_list,
                }
        elif outfit_layer_order and not has_unique_clr and not packed_entries:
            # Pure tiling sub-material fallback (if an outfit/prop piece has no unique texture at all)
            first_l = outfit_layer_map[outfit_layer_order[0]]
            if first_l.get("color_file"):
                outfit_layered = {
                    "mode": "outfit_fallback",
                    "corner_order": "dxil_exact",
                    "mask_file": "",
                    "mask_has_alpha": False,
                    "normal_file": first_l.get("normal_file", ""),
                    "normal_packing": {"R": "Normal.R", "G": "Normal.G", "B": "Roughness.R"},
                    "uv_scale": 16.0,
                    "tribe_hint": f"{machine_hint} {variant_hint} {mat.name}".lower(),
                    "layers": [first_l],
                }

    # Detect Character / General Asset material archetype hints (skin, hair, hair_face, peach_fuzz, eye, foliage, glass)
    surface_profile = "pbr"
    allow_alpha_cutout = True
    if not is_machine_asset:
        pe_names = " ".join(e.get("filename", "") for e in packed_entries)
        ml = f"{mat.name} {pe_names} {mesh_part_hint} {' '.join(mat.material_tags)}".lower()
        if not packed_entries and not maps and outfit_layered is None and mat.name.lower() in ("black_2k", "placeholder_clr", "placeholder_nmt"):
            surface_profile = "peach_fuzz"
        elif any(k in ml for k in ("hair_face", "eyelash", "lash", "brow", "beard", "eyebrow")):
            surface_profile = "hair_face"
        elif any(k in ml for k in ("hair_accs", "aloyhair_accs")):
            surface_profile = "cloth"
        elif any(k in ml for k in ("aloyhair", "quenhair", "npchair", "mediumhair", "shorthair", "longhair", "hair_cards", "hair_alpha", "cards_fur", "strand", "ponytail")) or uses_hair_lut:
            surface_profile = "hair"
        elif any(k in ml for k in ("cornea", "eyewet", "eye_wet", "tearline", "lacrimal", "eyes_shell", "eye_shell")):
            surface_profile = "eye_wet"
        elif any(k in ml for k in ("eyes_nmt", "eye_nmt", "iris", "sclera", "eyeball")):
            surface_profile = "eye"
        elif (
            outfit_layered is None
            and any(k in ml for k in ("head_nmt", "head_shared", "female_aloy", "basebody", "body_female", "body_male", "hands", "arm_skin", "neck"))
            or (outfit_layered is None and "skin" in ml and "buckskin" not in ml and "snakeskin" not in ml and "deerskin" not in ml)
        ):
            surface_profile = "skin"
        elif any(k in ml for k in ("teeth", "tongue", "mouth", "gum")):
            surface_profile = "teeth"
        elif any(k in ml for k in ("leaf", "leaves", "foliage", "frond", "fern", "grass", "moss", "pine", "needle", "canopy", "shrub", "bush", "vine", "petal", "flower", "branch")):
            surface_profile = "foliage"
        elif any(k in ml for k in ("snow", "snowcover", "glacier")) and not any("Color.B" in e.get("packing", {}).values() for e in packed_entries):
            surface_profile = "snow"
        elif any(
            "_tnt" in (e.get("filename", "") + " " + e.get("source_path", "")).lower()
            and e.get("color_space") == "Non-Color"
            and any(v.startswith("Color.") for v in e.get("packing", {}).values())
            for e in packed_entries
        ):
            # HFW landscape rock `_TNT` textures are 2-channel Linear RG Tint/Blend maps (B == 0!)
            surface_profile = (
                "rock_volcanic"
                if any(k in ml for k in ("volcanic", "lava", "basalt", "obsidian", "ash", "magma"))
                else "rock_tint"
            )
        elif any(k in ml for k in ("glass", "window", "crystal", "lens", "water", "ice")):
            surface_profile = "glass"
        elif outfit_layered is not None or any(k in ml for k in ("cloth", "fabric", "linen", "leather", "fur", "wool", "braid", "cord", "rope", "weave", "cape", "tunic")):
            surface_profile = "cloth"

        # If `570:0` Skin Tone LUT is bound on a character head/body, guarantee `surface_profile = "skin"`!
        if uses_skin_lut and surface_profile not in ("eye", "eye_wet", "teeth", "hair", "hair_face"):
            surface_profile = "skin"

        # NEVER allow opaque skin/teeth/eyes to get alpha-cutout holes; only allow alpha on explicit alpha/hair/foliage/trim cards!
        if surface_profile in ("skin", "eye", "teeth"):
            allow_alpha_cutout = False
        elif surface_profile in ("hair", "hair_face", "peach_fuzz", "foliage", "eye_wet", "glass"):
            allow_alpha_cutout = True
        else:
            allow_alpha_cutout = any(
                k in pe_names.lower() for k in ("alpha", "fur", "feather", "lace", "fringe", "trim", "leaf", "foliage", "cards")
            )

    # On landscape rocks (`rock_volcanic` / `rock_tint`), export the 2K standalone tiling detail normal map
    # (e.g. `5211:1` 2048x2048 BC7 Linear volcanic basalt detail) so rocks get crisp close-up porous relief!
    asset_detail_normal_file = ""
    if asset_kind == "asset" and surface_profile in ("rock_volcanic", "rock_tint") and not unpack_textures:
        for tb in pt.texture_bindings:
            if tb.resource_type == "Texture" and tb.resource_id and not tb.source_paths:
                try:
                    tobj = game.read_object(ObjectId.parse(tb.resource_id), read_subgroups=True)
                    hdr = (tobj.get("_texture") or {}).get("header") or {}
                    w = int(hdr.get("width", 0))
                    h = int(hdr.get("height", 0))
                    cs = int(hdr.get("color_space", 0))
                    if w >= 512 and h >= 512 and cs == 0 and int(hdr.get("type", 0)) == 0:
                        d_ent = _export_raw_tex(tb.resource_id, f"HFW_RockDetail_{tb.resource_id.replace(':', '_')}", "Non-Color")
                        if d_ent and d_ent.get("filename"):
                            asset_detail_normal_file = d_ent["filename"]
                            break
                except Exception:
                    continue

    has_emissive = (
        "Incandescence" in maps
        or any("Incandescence" in v for e in packed_entries for v in e.get("packing", {}).values())
        or (is_machine_asset and "DeferredEmissive" in mat.material_tags)
    )
    if has_emissive:
        emissive_strength = 4.0

    return {
        "texture_mode": "unpacked" if unpack_textures else "packed",
        "packed_entries": packed_entries,
        "maps": maps,
        "detail_maps": detail_maps,
        "detail_array": detail_array,
        "detail_tile": [detail_tile_x, detail_tile_y],
        "emissive_color": emissive_color,
        "emissive_strength": emissive_strength,
        "special_shader": special_shader,
        "emissive_overlay": emissive_overlay,
        "surface_profile": surface_profile,
        "allow_alpha_cutout": allow_alpha_cutout,
        "outfit_layered": outfit_layered,
        "uses_skin_lut": uses_skin_lut,
        "skin_tone_linear": list(skin_profile["skin_tone_linear"]) if (uses_skin_lut and skin_profile) else None,
        "uses_hair_lut": uses_hair_lut,
        "hair_color_linear": list(hair_profile["hair_color_linear"]) if (hair_profile and surface_profile in ("hair", "hair_face")) else None,
        "asset_detail_normal_file": asset_detail_normal_file,
    }


def export_assembly_bundle(
    game: HFWGame,
    assembly: MachineAssembly,
    bundle_dir: Path,
    max_texture_size: int | None = None,
    unpack_textures: bool = False,
    asset_kind: str = "machine",
    extra_metadata: dict[str, Any] | None = None,
    animations: int | list[str] | str | None = None,
) -> Path:
    """Write a complete self-contained Canonical Assembly Bundle (`assembly.json` + `geometry.npz` + `textures/`).

    Works uniformly for HFW Machines (`asset_kind="machine"`), Characters (`asset_kind="character"`),
    and General Dev-Catalog Assets (`asset_kind="asset"`).

    Parameters
    ----------
    unpack_textures:
        When False (default), exports the game's native packed RGBA textures and lets Blender
        separate channels in shader nodes. When True, unpacks each TextureSet into individual
        semantic PNGs (`_Color`, `_Normal_GL`, `_Roughness`, `_AO`, `_Reflectance`, `_Mask`, etc.).
    """
    bundle_dir = Path(bundle_dir)
    tex_dir = bundle_dir / "textures"
    tex_dir.mkdir(parents=True, exist_ok=True)

    ts_packed_cache: dict[str, list[dict[str, Any]]] = {}
    ts_unpacked_cache: dict[str, dict[str, str]] = {}
    standalone_packed_cache: dict[str, dict[str, Any] | None] = {}
    standalone_unpacked_cache: dict[str, dict[str, str]] = {}
    array_atlas_cache: dict[str, dict[str, Any] | None] = {}
    standalone_raw_cache: dict[str, dict[str, Any] | None] = {}

    machine_hint = f"{assembly.codename} {assembly.display_name}"
    variant_hint = f"{assembly.official_variant_name} {assembly.variant_label} {assembly.official_variant_description}"
    is_plasma_machine = (asset_kind == "machine") and any(
        kw in f"{machine_hint} {variant_hint}".lower()
        for kw in ("plasma", "slaughterspine")
    )
    variant_profile = infer_machine_variant_profile(
        machine_hint=machine_hint,
        variant_hint=variant_hint,
        is_plasma_machine=is_plasma_machine,
    )
    skin_profile = (
        infer_character_skin_tone(character_hint=machine_hint, variant_hint=variant_hint)
        if asset_kind == "character"
        else None
    )
    hair_profile = (
        infer_character_hair_color(character_hint=machine_hint, variant_hint=variant_hint)
        if asset_kind == "character"
        else None
    )
    dxc_tool = None
    if asset_kind == "character":
        try:
            from hfw_toolkit.shaders.dxc import DXCDisassembler

            dxc_tool = DXCDisassembler(game.config)
        except Exception:
            dxc_tool = None

    # Map each material_id -> official HFW Focus Catalog part names (Priority 1) + mesh part names (Priority 2)
    # (so Slaughterspine's Metalbite Sac = Acid Green, Apex Slaughterspine's Purgewater Sac = Aqua Blue,
    # Fire vs Acid Clawstrider, Fire vs Frost vs Acid vs Apex Bellowback, Bilegut Acid vs Adhesive sacs, etc.!)
    mat_cat_hints: dict[str, list[str]] = {}
    mat_mesh_hints: dict[str, list[str]] = {}
    for part in assembly.parts:
        if part.state_index > 0:
            continue
        for pr in part.mesh.primitives:
            if not pr.material_id:
                continue
            if part.catalog_part_name:
                mat_cat_hints.setdefault(pr.material_id, []).append(part.catalog_part_name)
            if part.name:
                mat_mesh_hints.setdefault(pr.material_id, []).append(part.name)

    materials_manifest: dict[str, Any] = {}
    for mid, mat in assembly.materials.items():
        c_hint = " ".join(dict.fromkeys(mat_cat_hints.get(mid, ())))
        m_hint = " ".join(dict.fromkeys(mat_mesh_hints.get(mid, ())))
        tex_info = _extract_material_textures(
            game,
            mat,
            tex_dir,
            ts_packed_cache,
            ts_unpacked_cache,
            standalone_packed_cache,
            standalone_unpacked_cache,
            array_atlas_cache,
            standalone_raw_cache,
            max_texture_size=max_texture_size,
            unpack_textures=unpack_textures,
            is_plasma_machine=is_plasma_machine,
            machine_hint=machine_hint,
            variant_hint=variant_hint,
            catalog_part_hint=c_hint,
            mesh_part_hint=m_hint,
            variant_profile=variant_profile,
            asset_kind=asset_kind,
            dxc=dxc_tool,
            skin_profile=skin_profile,
            hair_profile=hair_profile,
        )
        pt = mat.primary_technique
        eff_mat_name = mat.name
        # On characters/assets, prefer the unique asset texture stem over generic shared names like Frost_NMT / Paint_Dry_Base_NMT / StarMoss_clr
        if asset_kind != "machine" and tex_info["packed_entries"]:
            mlow = eff_mat_name.lower()
            if any(g in mlow for g in ("frost_", "dirt_", "paint_", "filler_", "placeholder", "leather_base", "leather_fullgrain", "metal_robotplate", "starmoss")):
                first_fn = tex_info["packed_entries"][0].get("filename", "")
                if first_fn:
                    eff_mat_name = re.sub(r"_\d+_\d+_[A-Z_]+\.png$", "", first_fn) or eff_mat_name
        materials_manifest[mid] = {
            "material_id": mid,
            "name": eff_mat_name,
            "tags": mat.material_tags,
            "shader_id": pt.shader_id if pt else None,
            "shader_hash": pt.shader_hash if pt else "",
            "passes": pt.available_passes if pt else [],
            "texture_sets": mat.texture_set_ids,
            "texture_mode": tex_info["texture_mode"],
            "packed_entries": tex_info["packed_entries"],
            "maps": tex_info["maps"],
            "detail_maps": tex_info["detail_maps"],
            "detail_array": tex_info["detail_array"],
            "detail_tile": tex_info["detail_tile"],
            "emissive_color": tex_info["emissive_color"],
            "emissive_strength": tex_info["emissive_strength"],
            "special_shader": tex_info["special_shader"],
            "emissive_overlay": tex_info["emissive_overlay"],
            "surface_profile": tex_info.get("surface_profile", "pbr"),
            "allow_alpha_cutout": bool(tex_info.get("allow_alpha_cutout", True)),
            "outfit_layered": tex_info.get("outfit_layered"),
            "uses_skin_lut": bool(tex_info.get("uses_skin_lut", False)),
            "skin_tone_linear": tex_info.get("skin_tone_linear"),
            "uses_hair_lut": bool(tex_info.get("uses_hair_lut", False)),
            "hair_color_linear": tex_info.get("hair_color_linear"),
            "asset_detail_normal_file": tex_info.get("asset_detail_normal_file", ""),
            "variables": [
                {
                    "hash": f"{vb.binding_name_hash:#010x}",
                    "type": vb.variable_type,
                    "float4": list(vb.float4),
                }
                for vb in (pt.variable_bindings if pt else ())
            ],
        }

    # Pack all geometry arrays into a single compressed .npz archive
    arrays: dict[str, np.ndarray] = {}
    parts_manifest: list[dict[str, Any]] = []

    for p_idx, part in enumerate(assembly.parts):
        prims_meta: list[dict[str, Any]] = []
        for pr_idx, prim in enumerate(part.mesh.primitives):
            prefix = f"p{p_idx}_m{pr_idx}"
            arrays[f"{prefix}_pos"] = prim.positions.astype(np.float32, copy=False)
            arrays[f"{prefix}_idx"] = prim.indices.astype(np.uint32, copy=False)
            if prim.normals is not None:
                arrays[f"{prefix}_nrm"] = prim.normals.astype(np.float32, copy=False)
            for u_i, uv in enumerate(prim.uvs):
                arrays[f"{prefix}_uv{u_i}"] = uv.astype(np.float32, copy=False)
            for c_i, col in enumerate(prim.colors):
                arrays[f"{prefix}_col{c_i}"] = col.astype(np.float32, copy=False)
            if prim.bone_indices is not None and prim.bone_weights is not None:
                arrays[f"{prefix}_bi"] = prim.bone_indices.astype(np.uint16, copy=False)
                arrays[f"{prefix}_bw"] = prim.bone_weights.astype(np.float32, copy=False)

            prims_meta.append(
                {
                    "key": prefix,
                    "material_id": prim.material_id,
                    "vertex_count": int(len(prim.positions)),
                    "index_count": int(len(prim.indices)),
                    "has_normals": prim.normals is not None,
                    "uv_count": len(prim.uvs),
                    "color_count": len(prim.colors),
                    "has_weights": prim.bone_indices is not None and prim.bone_weights is not None,
                }
            )

        if not prims_meta:
            continue

        # Route pure UV-less Focus/VFX overlay shell meshes cleanly into the hidden `Focus_Overlay` collection!
        part_coll = part.collection
        part_enabled = part.enabled_by_default
        if all(
            (materials_manifest.get(pm["material_id"], {}).get("special_shader") or {}).get("kind") == "vfx_overlay_shell"
            for pm in prims_meta
            if pm["material_id"]
        ) and any(pm["material_id"] for pm in prims_meta):
            part_coll = "Focus_Overlay"
            part_enabled = False

        parts_manifest.append(
            {
                "part_id": part.part_id,
                "name": part.name,
                "collection": part_coll,
                "source_kind": part.source_kind,
                "is_skinned": part.is_skinned,
                "parent_joint_index": part.parent_joint_index,
                "parent_joint_name": part.parent_joint_name,
                "helper_name": part.helper_name,
                "local_offset_matrix": part.local_offset_matrix.tolist(),
                "world_matrix": part.world_matrix.tolist(),
                "enabled_by_default": part_enabled,
                "state_index": part.state_index,
                "primitives": prims_meta,
            }
        )

    # Include helpers referenced by parts so armature stays clean and organized
    used_helper_names = {p["helper_name"] for p in parts_manifest if p["helper_name"]}
    helpers_manifest = []
    seen_helper_names: set[str] = set()
    for h in assembly.skeleton.helpers:
        if not h.name or h.name in seen_helper_names:
            continue
        if h.name in used_helper_names:
            seen_helper_names.add(h.name)
            helpers_manifest.append(
                {
                    "name": h.name,
                    "parent_joint_index": h.parent_joint_index,
                    "offset_matrix": h.offset_matrix.tolist(),
                    "world_matrix": h.world_matrix.tolist(),
                }
            )

    joints_manifest = [
        {
            "index": j.index,
            "name": j.name,
            "parent_index": j.parent_index,
            "local_matrix": j.local_matrix.tolist(),
            "world_matrix": j.world_matrix.tolist(),
        }
        for j in assembly.skeleton.joints
    ]

    # Analyze universal HFW Control Rig chains (shared identically by Blender & UE5!)
    rig_spec: dict[str, Any] | None = None
    if len(assembly.skeleton.joints) >= 8:
        try:
            from hfw_toolkit.blender.control_rig import analyze_rig_chains

            joint_specs = [
                {
                    "name": j.name,
                    "parent_name": assembly.skeleton.joints[j.parent_index].name
                    if 0 <= j.parent_index < len(assembly.skeleton.joints)
                    else None,
                    "head": j.world_matrix[:3, 3].tolist(),
                }
                for j in assembly.skeleton.joints
            ]
            rig_spec = analyze_rig_chains(joint_specs)
        except Exception:
            rig_spec = None

    # Optional decoded HFW in-game animations (`animations.npz` + `manifest["animations"]`)
    animations_manifest: list[dict[str, Any]] = []
    if animations and assembly.skeleton.joints:
        try:
            from hfw_toolkit.extract.animations import resolve_animations_for_export

            ent_gid: int | None = None
            if assembly.entity_id and ":" in assembly.entity_id:
                try:
                    ent_gid = ObjectId.parse(assembly.entity_id).group_id
                except Exception:
                    ent_gid = None
            gender_hint = "male" if "male" in f"{assembly.display_name} {assembly.variant_label}".lower() and "female" not in f"{assembly.display_name} {assembly.variant_label}".lower() else "female"
            decoded_clips = resolve_animations_for_export(
                game,
                assembly.skeleton,
                entity_group_id=ent_gid,
                asset_kind=asset_kind,
                animations_spec=animations,
                gender=gender_hint,
            )
            if decoded_clips:
                anim_arrays: dict[str, np.ndarray] = {}
                for ai, clip in enumerate(decoded_clips):
                    akey = f"a{ai}"
                    anim_arrays[f"{akey}_loc"] = clip["loc"]
                    anim_arrays[f"{akey}_rot"] = clip["rot_wxyz"]
                    anim_arrays[f"{akey}_scale"] = clip["scale"]
                    anim_arrays[f"{akey}_has_loc"] = clip["has_loc"]
                    anim_arrays[f"{akey}_has_rot"] = clip["has_rot"]
                    anim_arrays[f"{akey}_has_scale"] = clip["has_scale"]
                    animations_manifest.append(
                        {
                            "key": akey,
                            "anim_id": clip["anim_id"],
                            "name": clip["name"],
                            "category": clip["category"],
                            "description": clip.get("description", ""),
                            "skeleton_id": clip["skeleton_id"],
                            "duration": round(float(clip["duration"]), 4),
                            "sample_rate": round(float(clip["sample_rate"]), 3),
                            "num_frames": int(clip["num_frames"]),
                            "joint_names": clip["joint_names"],
                            "locomotion_delta_t": clip["locomotion_delta_t"],
                            "events": clip["events"],
                            "fbx_filename": f"HFW_{_safe_name(clip['name'], 52)}.fbx",
                        }
                    )
                np.savez_compressed(bundle_dir / "animations.npz", **anim_arrays)
        except Exception:
            animations_manifest = []

    manifest = {
        "schema_version": 4,
        "asset_kind": asset_kind,
        "texture_mode": "unpacked" if unpack_textures else "packed",
        "entity_id": assembly.entity_id,
        "codename": assembly.codename,
        "display_name": assembly.display_name,
        "variant_label": assembly.variant_label,
        "official_variant_name": assembly.official_variant_name,
        "shader_controls": (
            variant_profile
            if asset_kind == "machine"
            else (
                {
                    "skin_tone_linear": skin_profile["skin_tone_linear"],
                    "skin_preset": skin_profile["preset"],
                    "hair_color_linear": hair_profile["hair_color_linear"] if hair_profile else [0.045, 0.032, 0.025],
                    "hair_preset": hair_profile["preset"] if hair_profile else "dark_brown",
                }
                if skin_profile
                else {}
            )
        ),
        "skeleton_id": assembly.skeleton.object_id,
        "joints": joints_manifest,
        "helpers": helpers_manifest,
        "rig_spec": rig_spec,
        "animations": animations_manifest,
        "parts": parts_manifest,
        "materials": materials_manifest,
    }
    if extra_metadata:
        manifest["metadata"] = dict(extra_metadata)

    np.savez_compressed(bundle_dir / "geometry.npz", **arrays)
    (bundle_dir / "assembly.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Write ready-to-run Unreal Engine 5 Rig/Mesh + Animated Materials importer script!
    try:
        from hfw_toolkit.shaders.unreal_export import generate_ue5_machine_script

        fbx_name = f"{_safe_name(assembly.display_name)}.fbx"
        ue5_py = generate_ue5_machine_script(manifest, bundle_dir, fbx_filename=fbx_name)
        script_name = {
            "character": "ue5_import_character.py",
            "asset": "ue5_import_asset.py",
        }.get(asset_kind, "ue5_import_machine.py")
        (bundle_dir / script_name).write_text(ue5_py, encoding="utf-8")
        if script_name != "ue5_import_machine.py":
            # Also provide canonical alias so any automated UE5 pipeline finds it seamlessly
            (bundle_dir / "ue5_import_machine.py").write_text(ue5_py, encoding="utf-8")
    except Exception as exc:
        print(f"[HFWToolkit] Warning: UE5 script generation failed: {exc}")

    return bundle_dir
