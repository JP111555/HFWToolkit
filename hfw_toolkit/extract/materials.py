"""RenderEffectResource / RenderTechnique / TextureBinding / VariableBinding material parser."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hfw_toolkit.decima.streaming_graph import HFWGame
from hfw_toolkit.extract.textures import ETEXTURE_SET_TYPE, decode_packing_info, decode_texture_binding_packing


@dataclass(slots=True)
class TextureBindingInfo:
    index: int
    binding_name_hash: int
    swizzle_name_hash: int
    sampler_name_hash: int
    binding_kind: str  # "Texture" | "TextureSet"
    target_channels: dict[str, str]  # e.g. {"R": "Color.R", "G": "Color.G", "B": "Color.B"}
    resource_id: str | None
    resource_type: str | None
    name: str
    source_paths: dict[str, str] = field(default_factory=dict)  # ETextureSetType -> work path
    packing_entries: list[dict[str, Any]] = field(default_factory=list)
    semantic_role: str = "Unknown"  # BaseColor, Normal, Roughness, DetailNormal, Mask, Incandescence, etc.
    is_detail_texture: bool = False


@dataclass(slots=True)
class VariableBindingInfo:
    index: int
    binding_name_hash: int
    variable_id_hash: int
    variable_type: str
    raw_u32: tuple[int, int, int, int]
    float4: tuple[float, float, float, float]
    int4: tuple[int, int, int, int]
    inferred_name: str = ""


@dataclass(slots=True)
class TechniqueInfo:
    technique_set_index: int
    technique_index: int
    technique_set_type: str
    available_passes: list[str]
    material_layer_id: int
    shader_id: str | None
    shader_hash: str
    cull_mode: int
    write_depth: bool
    texture_bindings: list[TextureBindingInfo] = field(default_factory=list)
    variable_bindings: list[VariableBindingInfo] = field(default_factory=list)
    sampler_count: int = 0


@dataclass(slots=True)
class MaterialInfo:
    object_id: str
    name: str
    primary_technique: TechniqueInfo | None
    techniques: list[TechniqueInfo] = field(default_factory=list)
    # Deduplicated semantic maps referenced across the primary technique (for Blender/UE reconstruction)
    texture_set_ids: list[str] = field(default_factory=list)
    standalone_texture_ids: list[str] = field(default_factory=list)
    roles: dict[str, list[TextureBindingInfo]] = field(default_factory=dict)
    material_tags: list[str] = field(default_factory=list)  # e.g. ["DetailNormal", "Emissive", "MultiLayer", "MuscleFiber"]


def _decode_available_passes(game: HFWGame, mask: int) -> list[str]:
    passes = []
    for bit in range(32):
        if mask & (1 << bit):
            passes.append(game.schema.enum_name("ERenderTechniqueType", bit))
    return passes


def _classify_texture_binding(
    binding_kind: str,
    target_channels: dict[str, str],
    source_paths: dict[str, str],
    name: str,
) -> tuple[str, bool]:
    """Infer high-level semantic role and whether this binding is a detail/tileable micro-surface map."""
    all_paths = " ".join(source_paths.values()).lower() + " " + name.lower()
    cleaned = all_paths.replace("nomuscle", "").replace("no_muscle", "").replace("noplate", "").replace("no_plate", "")
    is_muscle = any(kw in cleaned for kw in ("muscle", "fiber", "fibre", "carbonweave"))
    target_types = {v.split(".", 1)[0] for v in target_channels.values()}
    if not target_types and source_paths:
        target_types = set(source_paths.keys())

    # A machine's or character's own unique macro UDIM tile (e.g. PlasmaSpine_Body_02, Raptor_Head_01_NMT,
    # AN_AloyNoraShieldweaverHat_NMT — note "ShieldWEAVEr" contains "weave"!, or any map packing baked
    # AO / Mask / Mask_Alpha / Curvature / Misc_01) is ALWAYS a PRIMARY macro texture, NEVER a tiled detail map!
    all_types_hint = target_types | set(source_paths.keys())
    has_baked_macro_channel = bool(all_types_hint & {"AO", "Mask", "Mask_Alpha", "Curvature", "Misc_01"})
    is_shared_or_shader_tex = any(
        kw in all_paths for kw in ("/shared/", "shaders_hrz2/", "tileable", "micro", "shared_detail")
    )
    is_machine_macro_tile = (
        has_baked_macro_channel
        or (
            (
                "models/characters/robots/" in all_paths
                or "models/characters/humans/" in all_paths
                or "_body_" in all_paths
                or "_plates_" in all_paths
                or "_head_" in all_paths
                or "_tail_" in all_paths
                or "_des_" in all_paths
                or "_dyn_" in all_paths
            )
            and not is_shared_or_shader_tex
        )
    )
    is_detail = (not is_machine_macro_tile) and any(
        kw in cleaned
        for kw in ("detail", "tile", "micro", "carbon", "weave", "grain", "scratch", "noise", "brushed", "procedural", "fiber", "fibre")
    )

    if "Normal" in target_types:
        if is_detail and is_muscle:
            return ("MuscleFiber_Normal", True)
        return ("Detail_Normal" if is_detail else "Normal", is_detail)
    if "Incandescence" in target_types or "Luminance" in target_types:
        return ("Emissive", is_detail)
    if "Color" in target_types:
        if is_detail and is_muscle:
            return ("MuscleFiber_Color", True)
        return ("Detail_Color" if is_detail else "Color", is_detail)
    if "Roughness" in target_types or "Reflectance" in target_types or "AO" in target_types:
        return ("Detail_Surface" if is_detail else "Surface_RMA", is_detail)
    if "Mask" in target_types or "Mask_Alpha" in target_types:
        return ("Mask", is_detail)
    if "Alpha" in target_types:
        return ("Alpha", is_detail)
    if "Height" in target_types:
        return ("Height", is_detail)
    if "Curvature" in target_types:
        return ("Curvature", is_detail)
    if "Translucency_Diffusion" in target_types or "Translucency_Amount" in target_types:
        return ("Translucency", is_detail)
    if "Misc_01" in target_types:
        return ("Misc_01", is_detail)

    # Filename-based fallback for standalone Textures
    if "_nmt" in cleaned or "normal" in cleaned:
        return ("Detail_Normal" if is_detail else "Normal", is_detail)
    if "_clr" in cleaned or "albedo" in cleaned or "diff" in cleaned:
        return ("Detail_Color" if is_detail else "Color", is_detail)
    if "_msk" in cleaned or "mask" in cleaned:
        return ("Mask", is_detail)
    if "_rgh" in cleaned or "rough" in cleaned:
        return ("Roughness", is_detail)
    if "_ems" in cleaned or "emissive" in cleaned or "glow" in cleaned or "plasma" in cleaned:
        return ("Emissive", is_detail)
    return ("Texture" if binding_kind == "Texture" else "Other", is_detail)


def parse_material(game: HFWGame, render_effect_obj: dict[str, Any]) -> MaterialInfo:
    """Parse a `RenderEffectResource` into a rich structured `MaterialInfo`."""
    oid = str(render_effect_obj.get("_id", ""))
    techniques: list[TechniqueInfo] = []

    for ts_idx, tset in enumerate(render_effect_obj.get("TechniqueSets", ())):
        ts_type = game.schema.enum_name("ERenderTechniqueSetType", tset.get("Type", 0))
        avail_passes = _decode_available_passes(game, int(tset.get("AvailableTechniquesMask", 0)))

        for rt_idx, rt in enumerate(tset.get("RenderTechniques", ())):
            shader_obj = game.resolve_ref(rt.get("Shader"))
            shader_id = str(shader_obj["_id"]) if shader_obj is not None else None
            shader_hash = shader_obj.get("_shader", {}).get("hash", "") if shader_obj else ""

            state = rt.get("RenderTechniqueState") or {}
            packed_state = int(state.get("PackedData", 0))
            cull_mode = (packed_state >> 2) & 3
            write_depth = bool((packed_state >> 4) & 1)

            tex_bindings: list[TextureBindingInfo] = []
            for tb_idx, tb in enumerate(rt.get("TextureBindings", ())):
                bkind, tpack = decode_texture_binding_packing(int(tb.get("PackedData", 0)))
                target_ch = {lane: f"{pc.tex_type}.{pc.source_channel}" for lane, pc in tpack.items()}
                res = game.resolve_ref(tb.get("TextureResource"))
                res_id = str(res["_id"]) if res is not None else None
                res_type = res["_type"] if res is not None else None

                src_paths: dict[str, str] = {}
                pack_entries: list[dict[str, Any]] = []
                bname = f"Tex_{res_id.replace(':', '_')}" if res_id else f"TexBinding_{tb_idx}"

                if res is not None and res_type == "TextureSet":
                    for desc in res.get("TextureDesc", ()):
                        tname = ETEXTURE_SET_TYPE.get(desc.get("TextureType", 0), "Invalid")
                        raw_p = desc.get("Path", "")
                        if raw_p:
                            clean = raw_p[5:] if raw_p.lower().startswith("work:") else raw_p
                            src_paths[tname] = clean
                            if bname.startswith("Tex_"):
                                bname = Path(clean).stem
                    for e_i, e in enumerate(res.get("Entries", ())):
                        etex = game.resolve_ref(e.get("Texture"))
                        epack = decode_packing_info(int(e.get("PackingInfo", 0)))
                        pack_entries.append(
                            {
                                "entry_index": e_i,
                                "texture_id": str(etex["_id"]) if etex else None,
                                "packing": {k: f"{v.tex_type}.{v.source_channel}" for k, v in epack.items()},
                            }
                        )
                elif res is not None and res_type == "Texture":
                    ts_parent = game.resolve_ref(res.get("TextureSetParent"))
                    if ts_parent:
                        for desc in ts_parent.get("TextureDesc", ()):
                            tname = ETEXTURE_SET_TYPE.get(desc.get("TextureType", 0), "Invalid")
                            raw_p = desc.get("Path", "")
                            if raw_p:
                                clean = raw_p[5:] if raw_p.lower().startswith("work:") else raw_p
                                src_paths[tname] = clean
                                if bname.startswith("Tex_"):
                                    bname = Path(clean).stem
                        # Also match which entry in parent TextureSet this Texture is!
                        for e_i, e in enumerate(ts_parent.get("Entries", ())):
                            etex = game.resolve_ref(e.get("Texture"))
                            if etex is not None and etex.get("_id") == res.get("_id"):
                                epack = decode_packing_info(int(e.get("PackingInfo", 0)))
                                if not target_ch:
                                    target_ch = {k: f"{v.tex_type}.{v.source_channel}" for k, v in epack.items()}
                                pack_entries.append(
                                    {
                                        "entry_index": e_i,
                                        "texture_id": res_id,
                                        "packing": {k: f"{v.tex_type}.{v.source_channel}" for k, v in epack.items()},
                                    }
                                )
                                break

                role, is_detail = _classify_texture_binding(bkind, target_ch, src_paths, bname)
                tex_bindings.append(
                    TextureBindingInfo(
                        index=tb_idx,
                        binding_name_hash=int(tb.get("BindingNameHash", 0)),
                        swizzle_name_hash=int(tb.get("BindingSwizzleNameHash", 0)),
                        sampler_name_hash=int(tb.get("SamplerNameHash", 0)),
                        binding_kind=bkind,
                        target_channels=target_ch,
                        resource_id=res_id,
                        resource_type=res_type,
                        name=bname,
                        source_paths=src_paths,
                        packing_entries=pack_entries,
                        semantic_role=role,
                        is_detail_texture=is_detail,
                    )
                )

            var_bindings: list[VariableBindingInfo] = []
            for vb_idx, vb in enumerate(rt.get("VariableBindings", ())):
                raw_u = (
                    int(vb.get("VariableData0", 0)) & 0xFFFFFFFF,
                    int(vb.get("VariableData1", 0)) & 0xFFFFFFFF,
                    int(vb.get("VariableData2", 0)) & 0xFFFFFFFF,
                    int(vb.get("VariableData3", 0)) & 0xFFFFFFFF,
                )
                packed_b = struct.pack("<4I", *raw_u)
                f4 = struct.unpack("<4f", packed_b)
                i4 = struct.unpack("<4i", packed_b)
                vtype = game.schema.enum_name("EShaderVariableType", int(vb.get("VariableType", 0)))
                var_bindings.append(
                    VariableBindingInfo(
                        index=vb_idx,
                        binding_name_hash=int(vb.get("BindingNameHash", 0)),
                        variable_id_hash=int(vb.get("VariableIDHash", 0)),
                        variable_type=vtype,
                        raw_u32=raw_u,
                        float4=tuple(round(x, 5) if abs(x) < 1e9 else 0.0 for x in f4),
                        int4=i4,
                    )
                )

            techniques.append(
                TechniqueInfo(
                    technique_set_index=ts_idx,
                    technique_index=rt_idx,
                    technique_set_type=ts_type,
                    available_passes=avail_passes,
                    material_layer_id=int(rt.get("MaterialLayerID", 0)),
                    shader_id=shader_id,
                    shader_hash=shader_hash,
                    cull_mode=cull_mode,
                    write_depth=write_depth,
                    texture_bindings=tex_bindings,
                    variable_bindings=var_bindings,
                    sampler_count=len(rt.get("SamplerBindings", ())),
                )
            )

    # Pick the richest main shading technique (NEVER prefer a 0-texture DepthOnly/Shadow technique over a textured one!)
    primary: TechniqueInfo | None = None
    if techniques:
        def tech_score(t: TechniqueInfo) -> tuple[int, int, int, int]:
            has_tbs = 1 if len(t.texture_bindings) > 0 else 0
            is_normal_set = 1 if "Normal" in t.technique_set_type else 0
            is_color_pass = 1 if any(p in ("Deferred", "DeferredEmissive", "DeferredForeground", "CustomDeferred", "Transparency", "CustomForward", "Unlit") for p in t.available_passes) else 0
            return (has_tbs, is_normal_set, is_color_pass, len(t.texture_bindings))

        primary = max(techniques, key=tech_score)

    # Derive readable material name & semantic role index from primary technique
    roles: dict[str, list[TextureBindingInfo]] = {}
    ts_ids: list[str] = []
    tex_ids: list[str] = []
    tags: set[str] = set()
    mat_name = f"Mat_{oid.replace(':', '_')}"

    if primary:
        for tb in primary.texture_bindings:
            roles.setdefault(tb.semantic_role, []).append(tb)
            if tb.resource_id:
                if tb.resource_type == "TextureSet" and tb.resource_id not in ts_ids:
                    ts_ids.append(tb.resource_id)
                elif tb.resource_type == "Texture" and tb.resource_id not in tex_ids:
                    tex_ids.append(tb.resource_id)
            if tb.is_detail_texture:
                tags.add("DetailTexture")
            if "MuscleFiber" in tb.semantic_role:
                tags.add("MuscleFiber")
            if tb.semantic_role == "Emissive":
                tags.add("Emissive")
            if tb.semantic_role == "Mask":
                tags.add("MultiLayerMask")
            if tb.semantic_role == "Alpha":
                tags.add("AlphaCutout")
            if tb.semantic_role == "Translucency":
                tags.add("Translucency")

        # Choose most descriptive primary texture name: ALWAYS prefer a unique `models/` / `dlc/models/`
        # asset texture (e.g. `Titan_Body_Bottom_01_NMT`, `Carapa_b001_Trunk_c006_NMT`) over shared
        # `Shaders_HRZ2/` weathering overlays (`StarMoss_clr`, `Rust_CLR`, `Grunge_CLR`)!
        def _is_shared_overlay_tb(tb_obj: TextureBindingInfo) -> bool:
            sp = (" ".join(tb_obj.source_paths.values()) + " " + (tb_obj.name or "")).replace("\\", "/").lower()
            return any(
                k in sp
                for k in (
                    "shaders_hrz2/",
                    "starmoss",
                    "rust_clr",
                    "rust_nmt",
                    "rust_hgt",
                    "grunge_clr",
                    "grunge_nmt",
                    "placeholder",
                    "detail_jungle_rough",
                    "detail_albedo_noise",
                )
            )

        def _is_unique_model_tb(tb_obj: TextureBindingInfo) -> bool:
            sp = (" ".join(tb_obj.source_paths.values())).replace("\\", "/").lower()
            return ("models/" in sp) and not _is_shared_overlay_tb(tb_obj)

        for preferred in ("Color", "Normal", "Surface_RMA", "Mask", "Emissive", "MuscleFiber_Color"):
            cands = roles.get(preferred) or []
            unique_hit = next((tb.name for tb in cands if tb.name and not tb.name.startswith("Tex_") and _is_unique_model_tb(tb)), None)
            if unique_hit:
                mat_name = unique_hit
                break
        if mat_name.startswith("Mat_"):
            for preferred in ("Color", "Normal", "Surface_RMA", "Emissive", "MuscleFiber_Color"):
                cands = roles.get(preferred) or []
                non_ov = next((tb.name for tb in cands if tb.name and not tb.name.startswith("Tex_") and not _is_shared_overlay_tb(tb)), None)
                if non_ov:
                    mat_name = non_ov
                    break
        if mat_name.startswith("Mat_"):
            for preferred in ("Color", "Normal", "Surface_RMA", "Emissive", "MuscleFiber_Color"):
                if roles.get(preferred):
                    cand = roles[preferred][0].name
                    if cand and not cand.startswith("Tex_"):
                        mat_name = cand
                        break
        if mat_name.startswith("Mat_") and primary.texture_bindings:
            for tb in primary.texture_bindings:
                if tb.name and not tb.name.startswith("Tex_"):
                    mat_name = tb.name
                    break

        if any("Emissive" in p for p in primary.available_passes):
            tags.add("DeferredEmissive")
        if any("Transparency" in p for p in primary.available_passes):
            tags.add("Transparent")

    if len(techniques) > 4:
        tags.add("MultiPass")

    return MaterialInfo(
        object_id=oid,
        name=mat_name,
        primary_technique=primary,
        techniques=techniques,
        texture_set_ids=ts_ids,
        standalone_texture_ids=tex_ids,
        roles=roles,
        material_tags=sorted(tags),
    )


def apply_shader_overrides(
    game: HFWGame,
    mat: MaterialInfo,
    texture_overrides: dict[int, Any] | None = None,
    variable_overrides: dict[int, dict[str, Any]] | None = None,
) -> int:
    """Apply Decima `RenderEffectShaderOverridesResource` texture & variable overrides onto a parsed `MaterialInfo`.

    HFW Apex & Variant machines store their dark obsidian/gold `*_BVar_CLR/MSK/RGH/RFL` `TextureSet` and `Texture`
    swaps inside `RenderEffectShaderOverridesResource.ShaderOverrides.ShaderTextureOverrides`, keyed 1-to-1 by
    `BindingNameHash`! Applying them directly on `MaterialInfo` updates both Blender and Unreal Engine 5 outputs.

    Returns the number of bindings modified on `mat`.
    """
    if not texture_overrides and not variable_overrides:
        return 0
    tex_ov = texture_overrides or {}
    var_ov = variable_overrides or {}
    changes = 0

    for tech in mat.techniques:
        for tb in tech.texture_bindings:
            ov_res = tex_ov.get(tb.binding_name_hash)
            if ov_res is None:
                continue
            res_obj = ov_res if isinstance(ov_res, dict) else None
            if res_obj is None:
                try:
                    res_obj = game.read_object(ov_res, read_subgroups=True)
                except Exception:
                    continue
            if not res_obj:
                continue

            res_id = str(res_obj.get("_id", ""))
            res_type = str(res_obj.get("_type", ""))
            if not res_id or res_type not in ("TextureSet", "Texture"):
                continue

            src_paths: dict[str, str] = {}
            pack_entries: list[dict[str, Any]] = []
            target_ch = dict(tb.target_channels)
            bname = f"Tex_{res_id.replace(':', '_')}"

            if res_type == "TextureSet":
                for desc in res_obj.get("TextureDesc", ()):
                    tname = ETEXTURE_SET_TYPE.get(desc.get("TextureType", 0), "Invalid")
                    raw_p = desc.get("Path", "")
                    if raw_p:
                        clean = raw_p[5:] if raw_p.lower().startswith("work:") else raw_p
                        src_paths[tname] = clean
                        if bname.startswith("Tex_"):
                            bname = Path(clean).stem
                for e_i, e in enumerate(res_obj.get("Entries", ())):
                    etex = game.resolve_ref(e.get("Texture"))
                    epack = decode_packing_info(int(e.get("PackingInfo", 0)))
                    pack_entries.append(
                        {
                            "entry_index": e_i,
                            "texture_id": str(etex["_id"]) if etex else None,
                            "packing": {k: f"{v.tex_type}.{v.source_channel}" for k, v in epack.items()},
                        }
                    )
            elif res_type == "Texture":
                ts_parent = game.resolve_ref(res_obj.get("TextureSetParent"))
                if ts_parent:
                    for desc in ts_parent.get("TextureDesc", ()):
                        tname = ETEXTURE_SET_TYPE.get(desc.get("TextureType", 0), "Invalid")
                        raw_p = desc.get("Path", "")
                        if raw_p:
                            clean = raw_p[5:] if raw_p.lower().startswith("work:") else raw_p
                            src_paths[tname] = clean
                            if bname.startswith("Tex_"):
                                bname = Path(clean).stem
                    for e_i, e in enumerate(ts_parent.get("Entries", ())):
                        etex = game.resolve_ref(e.get("Texture"))
                        if etex is not None and etex.get("_id") == res_obj.get("_id"):
                            epack = decode_packing_info(int(e.get("PackingInfo", 0)))
                            if epack:
                                target_ch = {k: f"{v.tex_type}.{v.source_channel}" for k, v in epack.items()}
                            pack_entries.append(
                                {
                                    "entry_index": e_i,
                                    "texture_id": res_id,
                                    "packing": {k: f"{v.tex_type}.{v.source_channel}" for k, v in epack.items()},
                                }
                            )
                            break

            tb.resource_id = res_id
            tb.resource_type = res_type
            tb.binding_kind = res_type
            tb.name = bname
            tb.source_paths = src_paths
            tb.packing_entries = pack_entries
            tb.target_channels = target_ch
            tb.semantic_role, tb.is_detail_texture = _classify_texture_binding(
                tb.binding_kind, tb.target_channels, tb.source_paths, tb.name
            )
            changes += 1

        for vb in tech.variable_bindings:
            ov_v = var_ov.get(vb.binding_name_hash) or var_ov.get(vb.variable_id_hash)
            if ov_v is None:
                continue
            raw_u = (
                int(ov_v.get("VariableData0", 0)) & 0xFFFFFFFF,
                int(ov_v.get("VariableData1", 0)) & 0xFFFFFFFF,
                int(ov_v.get("VariableData2", 0)) & 0xFFFFFFFF,
                int(ov_v.get("VariableData3", 0)) & 0xFFFFFFFF,
            )
            packed_b = struct.pack("<4I", *raw_u)
            f4 = struct.unpack("<4f", packed_b)
            i4 = struct.unpack("<4i", packed_b)
            vb.raw_u32 = raw_u
            vb.float4 = tuple(round(x, 5) if abs(x) < 1e9 else 0.0 for x in f4)
            vb.int4 = i4
            changes += 1

    if changes > 0 and mat.primary_technique is not None:
        # Refresh derived primary metadata (texture_set_ids, standalone_texture_ids, roles, name, tags)
        roles: dict[str, list[TextureBindingInfo]] = {}
        ts_ids: list[str] = []
        tex_ids: list[str] = []
        tags = set(mat.material_tags)
        tags.add("VariantOverride")
        for tb in mat.primary_technique.texture_bindings:
            roles.setdefault(tb.semantic_role, []).append(tb)
            if tb.resource_id:
                if tb.resource_type == "TextureSet" and tb.resource_id not in ts_ids:
                    ts_ids.append(tb.resource_id)
                elif tb.resource_type == "Texture" and tb.resource_id not in tex_ids:
                    tex_ids.append(tb.resource_id)
        for preferred in ("Color", "Normal", "Surface_RMA", "Emissive"):
            if roles.get(preferred):
                cand = roles[preferred][0].name
                if cand and not cand.startswith("Tex_"):
                    mat.name = cand
                    break
        mat.roles = roles
        mat.texture_set_ids = ts_ids
        mat.standalone_texture_ids = tex_ids
        mat.material_tags = sorted(tags)

    return changes
