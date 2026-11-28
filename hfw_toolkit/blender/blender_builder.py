"""Self-contained Blender 5.3 scene, armature, bone-parented armor plate & shader-node builder.

Uses ONLY Blender's built-in Python stdlib + `bpy` + `mathutils` + `numpy`.
Can be executed both:
1. Headlessly via `blender --background --python blender_builder.py -- <bundle_dir> <out.blend> [--pack]`
2. Directly imported inside the live Blender 5.3 Add-on via `import_bundle_into_scene(bundle_dir, ...)`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import bpy
import numpy as np
from mathutils import Matrix, Vector


def _mat4(rows: list[list[float]]) -> Matrix:
    return Matrix((rows[0], rows[1], rows[2], rows[3]))


def _get_or_create_collection(name: str, parent: bpy.types.Collection) -> bpy.types.Collection:
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        parent.children.link(coll)
    return coll


def _ensure_decima_pbr_node_group() -> bpy.types.ShaderNodeTree:
    """Create reusable `HFW_DecimaPBR` node group implementing exact DXIL-verified Decima PBR conversion.

    Directly derived from disassembled HFW PixelProgram DXIL (`16236:2` lines 592-782):
      - Decima `MSK_RGH_RFL.G` (`Roughness`) is actually **Smoothness** (`Blender Roughness = 1 - Smoothness`).
      - Decima `CLR` + `RFL` is a Specular/Diffuse workflow where worn bare metal edges have `CLR ≈ (0,0,0)`
        and high `Reflectance` (`0.60..0.90`). The GPU recovers metallic F0 via:
          * `F0_raw   = Reflectance^2`
          * `m_decima = clamp(F0_raw * 1.11 - 0.111, 0.0, 0.98)`
          * `chroma   = max(CLR, 0.001) / max_comp(max(CLR, 0.001))`
          * `F0_diel  = clamp(F0_raw, 0.0, 0.10)`
          * `F0_rgb   = lerp(F0_diel, chroma * F0_raw, m_decima)`
          * `Diff_rgb = (1 - m_decima) * CLR * AO`
      - Maps `(Diff_rgb, F0_rgb)` cleanly into Blender `Principled BSDF` (`Base Color`, `Metallic`,
        `Roughness`, `Specular IOR Level`) so worn bare metal edges render as gleaming reflective steel/silver
        instead of black lines, and matte painted plates stay authentically matte instead of plastic-shiny!
    """
    ng_name = "HFW_DecimaPBR"
    existing = bpy.data.node_groups.get(ng_name)
    if existing is not None:
        return existing

    ng = bpy.data.node_groups.new(ng_name, "ShaderNodeTree")
    s_col = ng.interface.new_socket(name="Color", in_out="INPUT", socket_type="NodeSocketColor")
    s_col.default_value = (0.35, 0.35, 0.35, 1.0)
    s_sm = ng.interface.new_socket(name="Smoothness", in_out="INPUT", socket_type="NodeSocketFloat")
    s_sm.default_value = 0.40
    s_rfl = ng.interface.new_socket(name="Reflectance", in_out="INPUT", socket_type="NodeSocketFloat")
    s_rfl.default_value = 0.20
    s_ao = ng.interface.new_socket(name="AO", in_out="INPUT", socket_type="NodeSocketFloat")
    s_ao.default_value = 1.0
    s_aos = ng.interface.new_socket(name="AO Strength", in_out="INPUT", socket_type="NodeSocketFloat")
    s_aos.default_value = 0.65

    ng.interface.new_socket(name="Base Color", in_out="OUTPUT", socket_type="NodeSocketColor")
    ng.interface.new_socket(name="Metallic", in_out="OUTPUT", socket_type="NodeSocketFloat")
    ng.interface.new_socket(name="Roughness", in_out="OUTPUT", socket_type="NodeSocketFloat")
    ng.interface.new_socket(name="Specular IOR Level", in_out="OUTPUT", socket_type="NodeSocketFloat")

    n_in = ng.nodes.new("NodeGroupInput")
    n_in.location = (-1300, 0)
    n_out = ng.nodes.new("NodeGroupOutput")
    n_out.location = (960, 0)

    # 1. F0_raw = Reflectance^2
    f0_raw = ng.nodes.new("ShaderNodeMath")
    f0_raw.operation = "MULTIPLY"
    f0_raw.label = "F0 = RFL^2"
    f0_raw.location = (-1040, -120)
    ng.links.new(n_in.outputs["Reflectance"], f0_raw.inputs[0])
    ng.links.new(n_in.outputs["Reflectance"], f0_raw.inputs[1])

    # 2. m_decima = clamp(F0_raw * 1.11 - 0.111, 0.0, 0.98)
    m_dec_mad = ng.nodes.new("ShaderNodeMath")
    m_dec_mad.operation = "MULTIPLY_ADD"
    m_dec_mad.inputs[1].default_value = 1.11
    m_dec_mad.inputs[2].default_value = -0.111
    m_dec_mad.use_clamp = True
    m_dec_mad.location = (-840, -180)
    ng.links.new(f0_raw.outputs["Value"], m_dec_mad.inputs[0])

    m_dec = ng.nodes.new("ShaderNodeMath")
    m_dec.operation = "MINIMUM"
    m_dec.inputs[1].default_value = 0.98
    m_dec.location = (-660, -180)
    ng.links.new(m_dec_mad.outputs["Value"], m_dec.inputs[0])

    # 3. F0_diel = clamp(F0_raw, 0.0, 0.10)
    f0_diel = ng.nodes.new("ShaderNodeMath")
    f0_diel.operation = "MINIMUM"
    f0_diel.inputs[1].default_value = 0.10
    f0_diel.location = (-840, -360)
    ng.links.new(f0_raw.outputs["Value"], f0_diel.inputs[0])

    # 4. Normalized chromaticity: chroma = max(CLR, 0.001) / max_comp(max(CLR, 0.001))
    sep_c = ng.nodes.new("ShaderNodeSeparateColor")
    sep_c.location = (-1040, 260)
    ng.links.new(n_in.outputs["Color"], sep_c.inputs["Color"])

    def _max_eps(sock, y_pos: float):
        n = ng.nodes.new("ShaderNodeMath")
        n.operation = "MAXIMUM"
        n.inputs[1].default_value = 0.001
        n.location = (-840, y_pos)
        ng.links.new(sock, n.inputs[0])
        return n.outputs["Value"]

    rc = _max_eps(sep_c.outputs["Red"], 380)
    gc = _max_eps(sep_c.outputs["Green"], 260)
    bc = _max_eps(sep_c.outputs["Blue"], 140)

    max_rg = ng.nodes.new("ShaderNodeMath")
    max_rg.operation = "MAXIMUM"
    max_rg.location = (-660, 320)
    ng.links.new(rc, max_rg.inputs[0])
    ng.links.new(gc, max_rg.inputs[1])

    cmax = ng.nodes.new("ShaderNodeMath")
    cmax.operation = "MAXIMUM"
    cmax.location = (-500, 240)
    ng.links.new(max_rg.outputs["Value"], cmax.inputs[0])
    ng.links.new(bc, cmax.inputs[1])

    # F0_metal = (clr_c / cmax) * F0_raw
    scale_f0 = ng.nodes.new("ShaderNodeMath")
    scale_f0.operation = "DIVIDE"
    scale_f0.location = (-320, 140)
    ng.links.new(f0_raw.outputs["Value"], scale_f0.inputs[0])
    ng.links.new(cmax.outputs["Value"], scale_f0.inputs[1])

    def _mul_ch(ch_sock, y_pos: float):
        m = ng.nodes.new("ShaderNodeMath")
        m.operation = "MULTIPLY"
        m.location = (-120, y_pos)
        ng.links.new(ch_sock, m.inputs[0])
        ng.links.new(scale_f0.outputs["Value"], m.inputs[1])
        return m.outputs["Value"]

    comb_f0m = ng.nodes.new("ShaderNodeCombineColor")
    comb_f0m.location = (80, 220)
    ng.links.new(_mul_ch(rc, 360), comb_f0m.inputs["Red"])
    ng.links.new(_mul_ch(gc, 220), comb_f0m.inputs["Green"])
    ng.links.new(_mul_ch(bc, 80), comb_f0m.inputs["Blue"])

    # 5. F0_rgb = lerp(F0_diel, F0_metal, m_decima)
    comb_f0d = ng.nodes.new("ShaderNodeCombineColor")
    comb_f0d.location = (-120, -240)
    ng.links.new(f0_diel.outputs["Value"], comb_f0d.inputs["Red"])
    ng.links.new(f0_diel.outputs["Value"], comb_f0d.inputs["Green"])
    ng.links.new(f0_diel.outputs["Value"], comb_f0d.inputs["Blue"])

    mix_f0 = ng.nodes.new("ShaderNodeMix")
    mix_f0.data_type = "RGBA"
    mix_f0.label = "Decima Specular F0"
    mix_f0.location = (280, 120)
    ng.links.new(m_dec.outputs["Value"], mix_f0.inputs["Factor"])
    ng.links.new(comb_f0d.outputs["Color"], mix_f0.inputs[6])
    ng.links.new(comb_f0m.outputs["Color"], mix_f0.inputs[7])

    # 6. Diffuse Albedo = (1 - m_decima) * CLR * lerp(1, AO, AO_Strength)
    one_m = ng.nodes.new("ShaderNodeMath")
    one_m.operation = "SUBTRACT"
    one_m.inputs[0].default_value = 1.0
    one_m.location = (-440, -140)
    ng.links.new(m_dec.outputs["Value"], one_m.inputs[1])

    ao_diff = ng.nodes.new("ShaderNodeMix")
    ao_diff.data_type = "FLOAT"
    ao_diff.inputs[2].default_value = 1.0
    ao_diff.location = (-440, 480)
    ng.links.new(n_in.outputs["AO Strength"], ao_diff.inputs["Factor"])
    ng.links.new(n_in.outputs["AO"], ao_diff.inputs[3])

    diff_w = ng.nodes.new("ShaderNodeMath")
    diff_w.operation = "MULTIPLY"
    diff_w.location = (-180, 480)
    ng.links.new(one_m.outputs["Value"], diff_w.inputs[0])
    ng.links.new(ao_diff.outputs[0], diff_w.inputs[1])

    diff_col = ng.nodes.new("ShaderNodeMix")
    diff_col.data_type = "RGBA"
    diff_col.blend_type = "MULTIPLY"
    diff_col.inputs["Factor"].default_value = 1.0
    diff_col.location = (280, 420)
    ng.links.new(n_in.outputs["Color"], diff_col.inputs[6])
    ng.links.new(diff_w.outputs["Value"], diff_col.inputs[7])

    # Subtle AO on metallic cavities (50% of AO Strength) so exposed worn edges stay bright!
    ao_m_str = ng.nodes.new("ShaderNodeMath")
    ao_m_str.operation = "MULTIPLY"
    ao_m_str.inputs[1].default_value = 0.50
    ao_m_str.location = (-180, -40)
    ng.links.new(n_in.outputs["AO Strength"], ao_m_str.inputs[0])

    ao_metal = ng.nodes.new("ShaderNodeMix")
    ao_metal.data_type = "FLOAT"
    ao_metal.inputs[2].default_value = 1.0
    ao_metal.location = (80, -40)
    ng.links.new(ao_m_str.outputs["Value"], ao_metal.inputs["Factor"])
    ng.links.new(n_in.outputs["AO"], ao_metal.inputs[3])

    metal_col = ng.nodes.new("ShaderNodeMix")
    metal_col.data_type = "RGBA"
    metal_col.blend_type = "MULTIPLY"
    metal_col.inputs["Factor"].default_value = 1.0
    metal_col.location = (480, 120)
    ng.links.new(mix_f0.outputs[2], metal_col.inputs[6])
    ng.links.new(ao_metal.outputs[0], metal_col.inputs[7])

    # 7. Blender Metallic mask: dielectric (RFL <= 0.32, F0 <= 0.10) -> 0.0, bare metal (RFL >= 0.58) -> 1.0
    mr_metal = ng.nodes.new("ShaderNodeMapRange")
    mr_metal.label = "Metallic Mask"
    mr_metal.clamp = True
    mr_metal.inputs["From Min"].default_value = 0.32
    mr_metal.inputs["From Max"].default_value = 0.58
    mr_metal.inputs["To Min"].default_value = 0.0
    mr_metal.inputs["To Max"].default_value = 1.0
    mr_metal.location = (280, -160)
    ng.links.new(n_in.outputs["Reflectance"], mr_metal.inputs["Value"])

    # Final Base Color = lerp(Diff_rgb, Metal_rgb, Metallic)
    final_bc = ng.nodes.new("ShaderNodeMix")
    final_bc.data_type = "RGBA"
    final_bc.label = "Base Color (Diff/Metal)"
    final_bc.location = (700, 280)
    ng.links.new(mr_metal.outputs["Result"], final_bc.inputs["Factor"])
    ng.links.new(diff_col.outputs[2], final_bc.inputs[6])
    ng.links.new(metal_col.outputs[2], final_bc.inputs[7])

    # 8. Blender Roughness = clamp(1.0 - Decima_Smoothness, 0.04, 1.0)
    inv_sm = ng.nodes.new("ShaderNodeMath")
    inv_sm.operation = "SUBTRACT"
    inv_sm.inputs[0].default_value = 1.0
    inv_sm.use_clamp = True
    inv_sm.label = "1 - Smoothness"
    inv_sm.location = (480, -340)
    ng.links.new(n_in.outputs["Smoothness"], inv_sm.inputs[1])

    rgh_cl = ng.nodes.new("ShaderNodeMath")
    rgh_cl.operation = "MAXIMUM"
    rgh_cl.inputs[1].default_value = 0.04
    rgh_cl.location = (700, -340)
    ng.links.new(inv_sm.outputs["Value"], rgh_cl.inputs[0])

    # 9. Dielectric Specular IOR Level = F0_diel / 0.08 (Blender Principled F0 = 0.08 * SpecIOR)
    mr_spec = ng.nodes.new("ShaderNodeMapRange")
    mr_spec.clamp = True
    mr_spec.inputs["From Min"].default_value = 0.0
    mr_spec.inputs["From Max"].default_value = 0.10
    mr_spec.inputs["To Min"].default_value = 0.0
    mr_spec.inputs["To Max"].default_value = 1.25
    mr_spec.location = (700, -520)
    ng.links.new(f0_diel.outputs["Value"], mr_spec.inputs["Value"])

    ng.links.new(final_bc.outputs[2], n_out.inputs["Base Color"])
    ng.links.new(mr_metal.outputs["Result"], n_out.inputs["Metallic"])
    ng.links.new(rgh_cl.outputs["Value"], n_out.inputs["Roughness"])
    ng.links.new(mr_spec.outputs["Result"], n_out.inputs["Specular IOR Level"])
    return ng


def _ensure_normal_reconstruct_z_group() -> bpy.types.ShaderNodeTree:
    """Create reusable `HFW_NormalReconstructZ` node group reconstructing tangent-space XYZ from packed RG channels."""
    ng_name = "HFW_NormalReconstructZ"
    existing = bpy.data.node_groups.get(ng_name)
    if existing is not None:
        return existing

    ng = bpy.data.node_groups.new(ng_name, "ShaderNodeTree")
    s_x = ng.interface.new_socket(name="Normal X", in_out="INPUT", socket_type="NodeSocketFloat")
    s_x.default_value = 0.5
    s_y = ng.interface.new_socket(name="Normal Y", in_out="INPUT", socket_type="NodeSocketFloat")
    s_y.default_value = 0.5
    s_fg = ng.interface.new_socket(name="Flip Green", in_out="INPUT", socket_type="NodeSocketFloat")
    s_fg.default_value = 0.0
    ng.interface.new_socket(name="Normal RGB", in_out="OUTPUT", socket_type="NodeSocketColor")

    n_in = ng.nodes.new("NodeGroupInput")
    n_in.location = (-800, 0)
    n_out = ng.nodes.new("NodeGroupOutput")
    n_out.location = (520, 0)

    # nx = X * 2 - 1, ny = Y * 2 - 1
    nx_m = ng.nodes.new("ShaderNodeMath")
    nx_m.operation = "MULTIPLY_ADD"
    nx_m.inputs[1].default_value = 2.0
    nx_m.inputs[2].default_value = -1.0
    nx_m.location = (-540, 140)

    ny_m = ng.nodes.new("ShaderNodeMath")
    ny_m.operation = "MULTIPLY_ADD"
    ny_m.inputs[1].default_value = 2.0
    ny_m.inputs[2].default_value = -1.0
    ny_m.location = (-540, -60)

    # Optional green flip: ny_out = mix(ny, -ny, FlipGreen) -> Y_out = mix(Y, 1 - Y, FlipGreen)
    inv_y = ng.nodes.new("ShaderNodeMath")
    inv_y.operation = "SUBTRACT"
    inv_y.inputs[0].default_value = 1.0
    inv_y.location = (-540, -260)

    mix_y = ng.nodes.new("ShaderNodeMix")
    mix_y.data_type = "FLOAT"
    mix_y.location = (-280, -240)

    # nx^2 + ny^2
    nx2 = ng.nodes.new("ShaderNodeMath")
    nx2.operation = "MULTIPLY"
    nx2.location = (-320, 160)

    ny2 = ng.nodes.new("ShaderNodeMath")
    ny2.operation = "MULTIPLY"
    ny2.location = (-320, 0)

    add_sq = ng.nodes.new("ShaderNodeMath")
    add_sq.operation = "ADD"
    add_sq.location = (-120, 80)

    sub_one = ng.nodes.new("ShaderNodeMath")
    sub_one.operation = "SUBTRACT"
    sub_one.use_clamp = True
    sub_one.inputs[0].default_value = 1.0
    sub_one.location = (60, 80)

    nz = ng.nodes.new("ShaderNodeMath")
    nz.operation = "SQRT"
    nz.location = (220, 80)

    # Map nz (0..1) -> (0.5..1.0)
    nz_enc = ng.nodes.new("ShaderNodeMath")
    nz_enc.operation = "MULTIPLY_ADD"
    nz_enc.inputs[1].default_value = 0.5
    nz_enc.inputs[2].default_value = 0.5
    nz_enc.location = (340, 80)

    comb = ng.nodes.new("ShaderNodeCombineColor")
    comb.location = (380, -100)

    ng.links.new(n_in.outputs["Normal X"], nx_m.inputs[0])
    ng.links.new(n_in.outputs["Normal Y"], ny_m.inputs[0])
    ng.links.new(n_in.outputs["Normal Y"], inv_y.inputs[1])
    ng.links.new(n_in.outputs["Flip Green"], mix_y.inputs["Factor"])
    ng.links.new(n_in.outputs["Normal Y"], mix_y.inputs[2])
    ng.links.new(inv_y.outputs["Value"], mix_y.inputs[3])

    ng.links.new(nx_m.outputs["Value"], nx2.inputs[0])
    ng.links.new(nx_m.outputs["Value"], nx2.inputs[1])
    ng.links.new(ny_m.outputs["Value"], ny2.inputs[0])
    ng.links.new(ny_m.outputs["Value"], ny2.inputs[1])
    ng.links.new(nx2.outputs["Value"], add_sq.inputs[0])
    ng.links.new(ny2.outputs["Value"], add_sq.inputs[1])
    ng.links.new(add_sq.outputs["Value"], sub_one.inputs[1])
    ng.links.new(sub_one.outputs["Value"], nz.inputs[0])
    ng.links.new(nz.outputs["Value"], nz_enc.inputs[0])

    ng.links.new(n_in.outputs["Normal X"], comb.inputs["Red"])
    ng.links.new(mix_y.outputs[0], comb.inputs["Green"])
    ng.links.new(nz_enc.outputs["Value"], comb.inputs["Blue"])
    ng.links.new(comb.outputs["Color"], n_out.inputs["Normal RGB"])
    return ng


def _ensure_detail_normal_group() -> bpy.types.ShaderNodeTree:
    """Create reusable `HFW_NormalDetailBlend` node group (tangent-space detail normal combine)."""
    ng_name = "HFW_NormalDetailBlend"
    existing = bpy.data.node_groups.get(ng_name)
    if existing is not None:
        return existing

    ng = bpy.data.node_groups.new(ng_name, "ShaderNodeTree")
    ng.interface.new_socket(name="Base Normal Map", in_out="INPUT", socket_type="NodeSocketColor")
    ng.interface.new_socket(name="Detail Normal Map", in_out="INPUT", socket_type="NodeSocketColor")
    s_str = ng.interface.new_socket(name="Detail Strength", in_out="INPUT", socket_type="NodeSocketFloat")
    s_str.default_value = 0.5
    ng.interface.new_socket(name="Combined Normal Map", in_out="OUTPUT", socket_type="NodeSocketColor")

    n_in = ng.nodes.new("NodeGroupInput")
    n_in.location = (-550, 0)
    n_out = ng.nodes.new("NodeGroupOutput")
    n_out.location = (450, 0)

    sep_base = ng.nodes.new("ShaderNodeSeparateColor")
    sep_base.location = (-300, 140)
    sep_det = ng.nodes.new("ShaderNodeSeparateColor")
    sep_det.location = (-300, -120)

    sub_r = ng.nodes.new("ShaderNodeMath")
    sub_r.operation = "SUBTRACT"
    sub_r.inputs[1].default_value = 0.5
    sub_r.location = (-100, -40)

    mul_r = ng.nodes.new("ShaderNodeMath")
    mul_r.operation = "MULTIPLY"
    mul_r.location = (60, -40)

    add_r = ng.nodes.new("ShaderNodeMath")
    add_r.operation = "ADD"
    add_r.use_clamp = True
    add_r.location = (180, 80)

    sub_g = ng.nodes.new("ShaderNodeMath")
    sub_g.operation = "SUBTRACT"
    sub_g.inputs[1].default_value = 0.5
    sub_g.location = (-100, -220)

    mul_g = ng.nodes.new("ShaderNodeMath")
    mul_g.operation = "MULTIPLY"
    mul_g.location = (60, -220)

    add_g = ng.nodes.new("ShaderNodeMath")
    add_g.operation = "ADD"
    add_g.use_clamp = True
    add_g.location = (180, -100)

    comb = ng.nodes.new("ShaderNodeCombineColor")
    comb.location = (310, 0)

    ng.links.new(n_in.outputs["Base Normal Map"], sep_base.inputs["Color"])
    ng.links.new(n_in.outputs["Detail Normal Map"], sep_det.inputs["Color"])
    ng.links.new(sep_det.outputs["Red"], sub_r.inputs[0])
    ng.links.new(sub_r.outputs["Value"], mul_r.inputs[0])
    ng.links.new(n_in.outputs["Detail Strength"], mul_r.inputs[1])
    ng.links.new(sep_base.outputs["Red"], add_r.inputs[0])
    ng.links.new(mul_r.outputs["Value"], add_r.inputs[1])

    ng.links.new(sep_det.outputs["Green"], sub_g.inputs[0])
    ng.links.new(sub_g.outputs["Value"], mul_g.inputs[0])
    ng.links.new(n_in.outputs["Detail Strength"], mul_g.inputs[1])
    ng.links.new(sep_base.outputs["Green"], add_g.inputs[0])
    ng.links.new(mul_g.outputs["Value"], add_g.inputs[1])

    ng.links.new(add_r.outputs["Value"], comb.inputs["Red"])
    ng.links.new(add_g.outputs["Value"], comb.inputs["Green"])
    ng.links.new(sep_base.outputs["Blue"], comb.inputs["Blue"])
    ng.links.new(comb.outputs["Color"], n_out.inputs["Combined Normal Map"])
    return ng


def _ensure_machine_detail_array_group(
    atlas_img: bpy.types.Image, cols: int = 5, rows: int = 5
) -> bpy.types.ShaderNodeTree:
    """Create `HFW_MachineDetailArray` implementing Guerrilla's exact DXIL 23-slice detail array shader.

    Verified directly from HFW PS DXIL (`16236:2` lines 355-565 & 775-782):
      - `mat_id = round(Mask * 20.0)`
      - Switch table: `0 -> 16`, `1..10 -> 1..10`, `11 -> 21`, `12 -> 22`, `13..20 -> 13..20`
      - Samples `20:0` (`256x256x23 BC7`) packed as a `cols x rows` tiled atlas PNG:
          * `R * 2.0` = Detail Color modulation
          * `A` = Detail Normal X, `G` = Detail Normal Y (AG tangent-space normal packing)
          * `B * 2.0` = Detail Smoothness/Roughness modulation (`lerp(Sm, Sm * 2 * B, weight)`)
      - Weighted per-pixel by `Mask_Alpha` (`A` of `NMT_AO_MSKA`) * `Detail Strength`.
    """
    ng_name = f"HFW_MachineDetailArray_{atlas_img.name[:32]}"
    existing = bpy.data.node_groups.get(ng_name)
    if existing is not None:
        return existing

    ng = bpy.data.node_groups.new(ng_name, "ShaderNodeTree")
    ng.interface.new_socket(name="Base Color", in_out="INPUT", socket_type="NodeSocketColor")
    s_sm = ng.interface.new_socket(name="Base Smoothness", in_out="INPUT", socket_type="NodeSocketFloat")
    s_sm.default_value = 0.45
    ng.interface.new_socket(name="Base Normal RGB", in_out="INPUT", socket_type="NodeSocketColor")
    s_msk = ng.interface.new_socket(name="Mask", in_out="INPUT", socket_type="NodeSocketFloat")
    s_msk.default_value = 0.0
    s_ma = ng.interface.new_socket(name="Mask Alpha", in_out="INPUT", socket_type="NodeSocketFloat")
    s_ma.default_value = 1.0
    s_tx = ng.interface.new_socket(name="Tile X", in_out="INPUT", socket_type="NodeSocketFloat")
    s_tx.default_value = 12.0
    s_ty = ng.interface.new_socket(name="Tile Y", in_out="INPUT", socket_type="NodeSocketFloat")
    s_ty.default_value = 12.0
    s_str = ng.interface.new_socket(name="Detail Strength", in_out="INPUT", socket_type="NodeSocketFloat")
    s_str.default_value = 0.65

    ng.interface.new_socket(name="Color", in_out="OUTPUT", socket_type="NodeSocketColor")
    ng.interface.new_socket(name="Smoothness", in_out="OUTPUT", socket_type="NodeSocketFloat")
    ng.interface.new_socket(name="Normal RGB", in_out="OUTPUT", socket_type="NodeSocketColor")

    n_in = ng.nodes.new("NodeGroupInput")
    n_in.location = (-1450, 0)
    n_out = ng.nodes.new("NodeGroupOutput")
    n_out.location = (860, 0)

    # 1. mat_id = round(Mask * 20.0)
    mul20 = ng.nodes.new("ShaderNodeMath")
    mul20.operation = "MULTIPLY"
    mul20.inputs[1].default_value = 20.0
    mul20.location = (-1220, -100)

    rnd = ng.nodes.new("ShaderNodeMath")
    rnd.operation = "ROUND"
    rnd.location = (-1040, -100)

    # 2. Exact Decima switch remapping: 0 -> 16, 11 -> 21, 12 -> 22, else mat_id
    def _remap_case(val_socket, match_val: float, target_slice: float, x_pos: float, y_pos: float):
        cmp = ng.nodes.new("ShaderNodeMath")
        cmp.operation = "COMPARE"
        cmp.inputs[1].default_value = match_val
        cmp.inputs[2].default_value = 0.25
        cmp.location = (x_pos, y_pos + 120)
        ng.links.new(rnd.outputs["Value"], cmp.inputs[0])

        mx = ng.nodes.new("ShaderNodeMix")
        mx.data_type = "FLOAT"
        mx.inputs[3].default_value = target_slice
        mx.location = (x_pos + 180, y_pos)
        ng.links.new(cmp.outputs["Value"], mx.inputs["Factor"])
        ng.links.new(val_socket, mx.inputs[2])
        return mx.outputs[0]

    s_out = _remap_case(rnd.outputs["Value"], 0.0, 16.0, -860, -100)
    s_out = _remap_case(s_out, 11.0, 21.0, -560, -100)
    s_out = _remap_case(s_out, 12.0, 22.0, -260, -100)

    # 3. Atlas col = slice % cols, row = floor(slice / cols)
    col_n = ng.nodes.new("ShaderNodeMath")
    col_n.operation = "MODULO"
    col_n.inputs[1].default_value = float(cols)
    col_n.location = (40, -40)
    ng.links.new(s_out, col_n.inputs[0])

    div_r = ng.nodes.new("ShaderNodeMath")
    div_r.operation = "DIVIDE"
    div_r.inputs[1].default_value = float(cols)
    div_r.location = (40, -220)
    ng.links.new(s_out, div_r.inputs[0])

    row_n = ng.nodes.new("ShaderNodeMath")
    row_n.operation = "FLOOR"
    row_n.location = (200, -220)
    ng.links.new(div_r.outputs["Value"], row_n.inputs[0])

    # 4. Tiled UV inside the chosen tile with 1% edge inset to prevent cross-tile bleed
    uv_tc = ng.nodes.new("ShaderNodeTexCoord")
    uv_tc.location = (-1220, -460)
    sep_uv = ng.nodes.new("ShaderNodeSeparateXYZ")
    sep_uv.location = (-1040, -460)
    ng.links.new(uv_tc.outputs["UV"], sep_uv.inputs["Vector"])

    def _tile_axis(uv_ch, tile_in, tile_idx_out, count_f: float, y_pos: float):
        m = ng.nodes.new("ShaderNodeMath")
        m.operation = "MULTIPLY"
        m.location = (-820, y_pos)
        ng.links.new(uv_ch, m.inputs[0])
        ng.links.new(tile_in, m.inputs[1])

        fr = ng.nodes.new("ShaderNodeMath")
        fr.operation = "FRACT"
        fr.location = (-640, y_pos)
        ng.links.new(m.outputs["Value"], fr.inputs[0])

        # Inset 0.01..0.99 inside each tile
        ins = ng.nodes.new("ShaderNodeMath")
        ins.operation = "MULTIPLY_ADD"
        ins.inputs[1].default_value = 0.98
        ins.inputs[2].default_value = 0.01
        ins.location = (-460, y_pos)
        ng.links.new(fr.outputs["Value"], ins.inputs[0])

        ad = ng.nodes.new("ShaderNodeMath")
        ad.operation = "ADD"
        ad.location = (360, y_pos)
        ng.links.new(tile_idx_out, ad.inputs[0])
        ng.links.new(ins.outputs["Value"], ad.inputs[1])

        dv = ng.nodes.new("ShaderNodeMath")
        dv.operation = "DIVIDE"
        dv.inputs[1].default_value = count_f
        dv.location = (520, y_pos)
        ng.links.new(ad.outputs["Value"], dv.inputs[0])
        return dv.outputs["Value"]

    atlas_u = _tile_axis(sep_uv.outputs["X"], n_in.outputs["Tile X"], col_n.outputs["Value"], float(cols), -420)
    atlas_v = _tile_axis(sep_uv.outputs["Y"], n_in.outputs["Tile Y"], row_n.outputs["Value"], float(rows), -620)

    comb_uv = ng.nodes.new("ShaderNodeCombineXYZ")
    comb_uv.location = (680, -520)
    ng.links.new(atlas_u, comb_uv.inputs["X"])
    ng.links.new(atlas_v, comb_uv.inputs["Y"])

    # 5. Sample the 5x5 detail array atlas PNG!
    tex_at = ng.nodes.new("ShaderNodeTexImage")
    tex_at.image = atlas_img
    tex_at.extension = "CLIP"
    tex_at.interpolation = "Linear"
    tex_at.label = "HFW Detail Array Atlas (23 Slices)"
    tex_at.location = (-200, 260)
    ng.links.new(comb_uv.outputs["Vector"], tex_at.inputs["Vector"])

    sep_at = ng.nodes.new("ShaderNodeSeparateColor")
    sep_at.location = (80, 320)
    ng.links.new(tex_at.outputs["Color"], sep_at.inputs["Color"])

    # Effective per-pixel detail weight = Mask_Alpha * Detail Strength
    eff_w = ng.nodes.new("ShaderNodeMath")
    eff_w.operation = "MULTIPLY"
    eff_w.use_clamp = True
    eff_w.location = (100, 120)
    ng.links.new(n_in.outputs["Mask Alpha"], eff_w.inputs[0])
    ng.links.new(n_in.outputs["Detail Strength"], eff_w.inputs[1])

    # Detail Normal packed in (Alpha = X, Green = Y) per HFW PS DXIL L475-482!
    det_nrm_comb = ng.nodes.new("ShaderNodeCombineColor")
    det_nrm_comb.inputs["Blue"].default_value = 1.0
    det_nrm_comb.location = (280, 140)
    ng.links.new(tex_at.outputs["Alpha"], det_nrm_comb.inputs["Red"])
    ng.links.new(sep_at.outputs["Green"], det_nrm_comb.inputs["Green"])

    det_blend = ng.nodes.new("ShaderNodeGroup")
    det_blend.node_tree = _ensure_detail_normal_group()
    det_blend.location = (560, -80)
    ng.links.new(n_in.outputs["Base Normal RGB"], det_blend.inputs["Base Normal Map"])
    ng.links.new(det_nrm_comb.outputs["Color"], det_blend.inputs["Detail Normal Map"])
    ng.links.new(eff_w.outputs["Value"], det_blend.inputs["Detail Strength"])
    ng.links.new(det_blend.outputs["Combined Normal Map"], n_out.inputs["Normal RGB"])

    # Detail Color modulation: R * 2.0 (neutral 0.5 -> 1.0) subtly multiplied into Base Color
    r_x2 = ng.nodes.new("ShaderNodeMath")
    r_x2.operation = "MULTIPLY"
    r_x2.inputs[1].default_value = 2.0
    r_x2.location = (280, 480)
    ng.links.new(sep_at.outputs["Red"], r_x2.inputs[0])

    col_w = ng.nodes.new("ShaderNodeMath")
    col_w.operation = "MULTIPLY"
    col_w.inputs[1].default_value = 0.45
    col_w.location = (280, 340)
    ng.links.new(eff_w.outputs["Value"], col_w.inputs[0])

    mix_col = ng.nodes.new("ShaderNodeMix")
    mix_col.data_type = "RGBA"
    mix_col.blend_type = "MULTIPLY"
    mix_col.location = (560, 400)
    ng.links.new(col_w.outputs["Value"], mix_col.inputs["Factor"])
    ng.links.new(n_in.outputs["Base Color"], mix_col.inputs[6])
    ng.links.new(r_x2.outputs["Value"], mix_col.inputs[7])
    ng.links.new(mix_col.outputs[2], n_out.inputs["Color"])

    # Detail Smoothness modulation (exact DXIL L775-782): lerp(Sm, Sm * (2 * detail.B), eff_w)
    b_x2 = ng.nodes.new("ShaderNodeMath")
    b_x2.operation = "MULTIPLY"
    b_x2.inputs[1].default_value = 2.0
    b_x2.location = (280, 220)
    ng.links.new(sep_at.outputs["Blue"], b_x2.inputs[0])

    sm_mod = ng.nodes.new("ShaderNodeMath")
    sm_mod.operation = "MULTIPLY"
    sm_mod.use_clamp = True
    sm_mod.location = (440, 220)
    ng.links.new(n_in.outputs["Base Smoothness"], sm_mod.inputs[0])
    ng.links.new(b_x2.outputs["Value"], sm_mod.inputs[1])

    mix_sm = ng.nodes.new("ShaderNodeMix")
    mix_sm.data_type = "FLOAT"
    mix_sm.clamp_result = True
    mix_sm.location = (620, 180)
    ng.links.new(eff_w.outputs["Value"], mix_sm.inputs["Factor"])
    ng.links.new(n_in.outputs["Base Smoothness"], mix_sm.inputs[2])
    ng.links.new(sm_mod.outputs["Value"], mix_sm.inputs[3])
    ng.links.new(mix_sm.outputs[0], n_out.inputs["Smoothness"])

    ng.links.new(n_in.outputs["Mask"], mul20.inputs[0])
    ng.links.new(mul20.outputs["Value"], rnd.inputs[0])
    return ng


def _configure_blended_material(mat: bpy.types.Material) -> None:
    """Enable smooth alpha/additive transparency cleanly in Blender 4.2-5.3 (Eevee Next & Cycles)."""
    if hasattr(mat, "surface_render_method"):
        try:
            mat.surface_render_method = "BLENDED"
        except Exception:
            pass
    if hasattr(mat, "blend_method"):
        try:
            mat.blend_method = "BLEND"
        except Exception:
            pass
    if hasattr(mat, "shadow_method"):
        try:
            mat.shadow_method = "NONE"
        except Exception:
            pass


def _add_time_driver_node(
    nt: bpy.types.NodeTree,
    location: tuple[float, float],
    label: str = "HFW GameTime (sec)",
) -> bpy.types.Node:
    """Create a `ShaderNodeValue` driven by Blender's safe Simple Expression `frame / 24.0`.

    Matches Decima's global `Scratch_PerFrame[8].w` GameTime in seconds and animates live in
    Eevee/Cycles viewport playback & animation renders without requiring external Python scripts!
    """
    val = nt.nodes.new("ShaderNodeValue")
    val.label = label
    val.location = location
    val.outputs[0].default_value = 0.0
    try:
        fc = val.outputs[0].driver_add("default_value")
        drv = fc.driver
        drv.type = "SCRIPTED"
        drv.use_self = False
        drv.expression = "frame / 24.0"
    except Exception:
        pass
    return val


def _ensure_elemental_sack_group(
    flow_img: bpy.types.Image | None,
    fluid_img: bpy.types.Image | None,
) -> bpy.types.ShaderNodeTree:
    """Create `HFW_ElementalSack` replicating Guerrilla's exact DXIL animated elemental sack shader.

    Verified from `frostclaw_belly_sac_3488_1219.txt` & `fireclaw_shoulder_sac_3701_3335.txt`:
      - Active ONLY where `round(Mask * 20.0) == 0` weighted by `Mask_Alpha` (never bleeds onto armor!).
      - Dual-phase flow-map fluid sloshing:
          * `flow   = (tex_172(UV * 2.0).RG * 2 - 1) * FlowStrength`
          * `phase0 = frac(t + 0.5) * 2 - 1`, `phase1 = frac(t + 1.0) * 2 - 1`, `blend = abs(phase0)`
          * `fluid  = lerp(tex_1703(UV * 2.5 + flow * phase0), tex_1703(UV * 2.5 + flow * phase1), blend)`
      - Subsurface Fresnel depth color shift (`Hue +0.03`, darker edge -> glowing translucent fluid core)
      - Emits vivid HDR glow + subtle rubbery translucent surface tint on Cryo, Acid, Blaze, Purgewater & Plasma sacs!
    """
    tag = f"{flow_img.name[:16] if flow_img else 'nf'}_{fluid_img.name[:16] if fluid_img else 'nl'}"
    ng_name = f"HFW_ElementalSack_{tag}"
    existing = bpy.data.node_groups.get(ng_name)
    if existing is not None:
        return existing

    ng = bpy.data.node_groups.new(ng_name, "ShaderNodeTree")
    ng.interface.new_socket(name="Base Color", in_out="INPUT", socket_type="NodeSocketColor")
    ng.interface.new_socket(name="Base Smoothness", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 0.45
    ng.interface.new_socket(name="Mask", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 0.0
    ng.interface.new_socket(name="Mask Alpha", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 1.0
    ng.interface.new_socket(name="Mask ID", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 0.0
    ng.interface.new_socket(name="Time", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 0.0
    s_gc = ng.interface.new_socket(name="Sac Glow Color", in_out="INPUT", socket_type="NodeSocketColor")
    s_gc.default_value = (0.14, 0.68, 1.0, 1.0)
    ng.interface.new_socket(name="Glow Strength", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 4.6
    ng.interface.new_socket(name="Flow Speed", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 0.35
    ng.interface.new_socket(name="Flow Strength", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 0.14
    ng.interface.new_socket(name="Animate Speed", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 1.0

    ng.interface.new_socket(name="Color", in_out="OUTPUT", socket_type="NodeSocketColor")
    ng.interface.new_socket(name="Smoothness", in_out="OUTPUT", socket_type="NodeSocketFloat")
    ng.interface.new_socket(name="Emission Color", in_out="OUTPUT", socket_type="NodeSocketColor")
    ng.interface.new_socket(name="Emission Strength", in_out="OUTPUT", socket_type="NodeSocketFloat")
    ng.interface.new_socket(name="Sac Mask", in_out="OUTPUT", socket_type="NodeSocketFloat")

    n_in = ng.nodes.new("NodeGroupInput")
    n_in.location = (-1650, 0)
    n_out = ng.nodes.new("NodeGroupOutput")
    n_out.location = (960, 0)

    # 1. Exact DXIL sac isolation mask supporting mat_id==0 (Bellowback), mat_id==20 (Bilegut/Frostclaw/Fireclaw/Tideripper/Snapmaw),
    #    AND Mask ID < 0.0 (pure liquid bubble/bag meshes driven directly by Mask_Alpha so sacs NEVER turn pure black!)
    m20 = ng.nodes.new("ShaderNodeMath")
    m20.operation = "MULTIPLY"
    m20.inputs[1].default_value = 20.0
    m20.location = (-1380, 420)
    ng.links.new(n_in.outputs["Mask"], m20.inputs[0])

    rnd = ng.nodes.new("ShaderNodeMath")
    rnd.operation = "ROUND"
    rnd.location = (-1200, 420)
    ng.links.new(m20.outputs["Value"], rnd.inputs[0])

    is_id = ng.nodes.new("ShaderNodeMath")
    is_id.operation = "COMPARE"
    is_id.inputs[2].default_value = 0.60
    is_id.location = (-1020, 420)
    ng.links.new(rnd.outputs["Value"], is_id.inputs[0])
    ng.links.new(n_in.outputs["Mask ID"], is_id.inputs[1])

    use_full = ng.nodes.new("ShaderNodeMath")
    use_full.operation = "LESS_THAN"
    use_full.inputs[1].default_value = 0.0
    use_full.location = (-1020, 580)
    ng.links.new(n_in.outputs["Mask ID"], use_full.inputs[0])

    id_or_full = ng.nodes.new("ShaderNodeMath")
    id_or_full.operation = "MAXIMUM"
    id_or_full.use_clamp = True
    id_or_full.location = (-880, 480)
    ng.links.new(is_id.outputs["Value"], id_or_full.inputs[0])
    ng.links.new(use_full.outputs["Value"], id_or_full.inputs[1])

    sac_mask = ng.nodes.new("ShaderNodeMath")
    sac_mask.operation = "MULTIPLY"
    sac_mask.use_clamp = True
    sac_mask.label = "Sac Mask (mat_id==MaskID * MaskAlpha)"
    sac_mask.location = (-720, 400)
    ng.links.new(id_or_full.outputs["Value"], sac_mask.inputs[0])
    ng.links.new(n_in.outputs["Mask Alpha"], sac_mask.inputs[1])

    # 2. Animated effective time: t_eff = Time * FlowSpeed * AnimateSpeed
    t_spd = ng.nodes.new("ShaderNodeMath")
    t_spd.operation = "MULTIPLY"
    t_spd.location = (-1380, -140)
    ng.links.new(n_in.outputs["Time"], t_spd.inputs[0])
    ng.links.new(n_in.outputs["Animate Speed"], t_spd.inputs[1])

    t_flow = ng.nodes.new("ShaderNodeMath")
    t_flow.operation = "MULTIPLY"
    t_flow.location = (-1180, -140)
    ng.links.new(t_spd.outputs["Value"], t_flow.inputs[0])
    ng.links.new(n_in.outputs["Flow Speed"], t_flow.inputs[1])

    # Dual phases: p0 = frac(t + 0.5) * 2 - 1, p1 = frac(t + 1.0) * 2 - 1, blend = abs(p0)
    def _make_phase(offset_val: float, y_pos: float):
        a = ng.nodes.new("ShaderNodeMath")
        a.operation = "ADD"
        a.inputs[1].default_value = offset_val
        a.location = (-980, y_pos)
        ng.links.new(t_flow.outputs["Value"], a.inputs[0])

        fr = ng.nodes.new("ShaderNodeMath")
        fr.operation = "FRACT"
        fr.location = (-800, y_pos)
        ng.links.new(a.outputs["Value"], fr.inputs[0])

        sc = ng.nodes.new("ShaderNodeMath")
        sc.operation = "MULTIPLY_ADD"
        sc.inputs[1].default_value = 2.0
        sc.inputs[2].default_value = -1.0
        sc.location = (-620, y_pos)
        ng.links.new(fr.outputs["Value"], sc.inputs[0])
        return sc.outputs["Value"]

    p0 = _make_phase(0.5, -80)
    p1 = _make_phase(1.0, -260)

    p_blend = ng.nodes.new("ShaderNodeMath")
    p_blend.operation = "ABSOLUTE"
    p_blend.location = (-440, -80)
    ng.links.new(p0, p_blend.inputs[0])

    # 3. Flow Vector from 172:0 (`UV * 2.0`, RG * 2 - 1, * FlowStrength)
    tc = ng.nodes.new("ShaderNodeTexCoord")
    tc.location = (-1620, -480)

    uv_flow_sc = ng.nodes.new("ShaderNodeVectorMath")
    uv_flow_sc.operation = "SCALE"
    uv_flow_sc.inputs["Scale"].default_value = 2.0
    uv_flow_sc.location = (-1380, -440)
    ng.links.new(tc.outputs["UV"], uv_flow_sc.inputs[0])

    uv_fluid_sc = ng.nodes.new("ShaderNodeVectorMath")
    uv_fluid_sc.operation = "SCALE"
    uv_fluid_sc.inputs["Scale"].default_value = 2.5
    uv_fluid_sc.location = (-1380, -640)
    ng.links.new(tc.outputs["UV"], uv_fluid_sc.inputs[0])

    if flow_img is not None:
        t_flw = ng.nodes.new("ShaderNodeTexImage")
        t_flw.image = flow_img
        t_flw.label = "HFW Flow Map (172:0)"
        t_flw.location = (-1140, -440)
        ng.links.new(uv_flow_sc.outputs["Vector"], t_flw.inputs["Vector"])

        flw_dec = ng.nodes.new("ShaderNodeVectorMath")
        flw_dec.operation = "MULTIPLY_ADD"
        flw_dec.inputs[1].default_value = (2.0, 2.0, 0.0)
        flw_dec.inputs[2].default_value = (-1.0, -1.0, 0.0)
        flw_dec.location = (-860, -440)
        ng.links.new(t_flw.outputs["Color"], flw_dec.inputs[0])

        flw_vec = ng.nodes.new("ShaderNodeVectorMath")
        flw_vec.operation = "SCALE"
        flw_vec.location = (-660, -440)
        ng.links.new(flw_dec.outputs["Vector"], flw_vec.inputs[0])
        ng.links.new(n_in.outputs["Flow Strength"], flw_vec.inputs["Scale"])
        flow_sock = flw_vec.outputs["Vector"]
    else:
        flw_c = ng.nodes.new("ShaderNodeCombineXYZ")
        flw_c.inputs["X"].default_value = 0.0
        flw_c.inputs["Y"].default_value = 0.12
        flw_c.location = (-660, -440)
        flow_sock = flw_c.outputs["Vector"]

    # 4. Dual-phase fluid texture samples (1703:0) + linear blend!
    if fluid_img is not None:
        def _sample_phase(phase_sock, y_pos: float, lbl: str):
            off = ng.nodes.new("ShaderNodeVectorMath")
            off.operation = "SCALE"
            off.location = (-440, y_pos)
            ng.links.new(flow_sock, off.inputs[0])
            ng.links.new(phase_sock, off.inputs["Scale"])

            uv_p = ng.nodes.new("ShaderNodeVectorMath")
            uv_p.operation = "ADD"
            uv_p.location = (-260, y_pos)
            ng.links.new(uv_fluid_sc.outputs["Vector"], uv_p.inputs[0])
            ng.links.new(off.outputs["Vector"], uv_p.inputs[1])

            ts = ng.nodes.new("ShaderNodeTexImage")
            ts.image = fluid_img
            ts.label = lbl
            ts.location = (-60, y_pos)
            ng.links.new(uv_p.outputs["Vector"], ts.inputs["Vector"])
            return ts.outputs["Color"]

        f0 = _sample_phase(p0, -340, "Fluid Phase 0 (1703:0)")
        f1 = _sample_phase(p1, -620, "Fluid Phase 1 (1703:0)")

        mix_fl = ng.nodes.new("ShaderNodeMix")
        mix_fl.data_type = "FLOAT"
        mix_fl.label = "Dual-Phase Fluid Blend"
        mix_fl.location = (220, -420)
        ng.links.new(p_blend.outputs["Value"], mix_fl.inputs["Factor"])
        ng.links.new(f0, mix_fl.inputs[2])
        ng.links.new(f1, mix_fl.inputs[3])
        fluid_val_sock = mix_fl.outputs[0]
    else:
        fluid_val_sock = p_blend.outputs["Value"]

    # 5. Subsurface Fresnel Rim Shift (DXIL RGB->HSL Hue+0.03, Lightness-0.25)
    rim_hsv = ng.nodes.new("ShaderNodeHueSaturation")
    rim_hsv.label = "DXIL Subsurface Rim Shift"
    rim_hsv.inputs["Hue"].default_value = 0.53
    rim_hsv.inputs["Saturation"].default_value = 1.25
    rim_hsv.inputs["Value"].default_value = 0.42
    rim_hsv.location = (220, 120)
    ng.links.new(n_in.outputs["Sac Glow Color"], rim_hsv.inputs["Color"])

    lw = ng.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.45
    lw.location = (20, -40)

    # Blend deep shifted rim at grazing edge with vibrant Sac Glow Color at facing core
    mix_rim = ng.nodes.new("ShaderNodeMix")
    mix_rim.data_type = "RGBA"
    mix_rim.label = "Sac Core <-> Rim Color"
    mix_rim.location = (460, 80)
    ng.links.new(lw.outputs["Facing"], mix_rim.inputs["Factor"])
    ng.links.new(n_in.outputs["Sac Glow Color"], mix_rim.inputs[6])
    ng.links.new(rim_hsv.outputs["Color"], mix_rim.inputs[7])

    # 6. Fluid & Breathing Pulse Modulation: rich swirling caustic contrast (0.38 + 1.15 * fluid) * breathing pulse
    fl_mod = ng.nodes.new("ShaderNodeMath")
    fl_mod.operation = "MULTIPLY_ADD"
    fl_mod.inputs[1].default_value = 1.15
    fl_mod.inputs[2].default_value = 0.38
    fl_mod.location = (400, -340)
    ng.links.new(fluid_val_sock, fl_mod.inputs[0])

    t_w = ng.nodes.new("ShaderNodeMath")
    t_w.operation = "MULTIPLY"
    t_w.inputs[1].default_value = 2.4
    t_w.location = (-100, -140)
    ng.links.new(t_spd.outputs["Value"], t_w.inputs[0])

    sin_w = ng.nodes.new("ShaderNodeMath")
    sin_w.operation = "SINE"
    sin_w.location = (80, -140)
    ng.links.new(t_w.outputs["Value"], sin_w.inputs[0])

    pulse_w = ng.nodes.new("ShaderNodeMath")
    pulse_w.operation = "MULTIPLY_ADD"
    pulse_w.inputs[1].default_value = 0.12
    pulse_w.inputs[2].default_value = 0.90
    pulse_w.location = (260, -140)
    ng.links.new(sin_w.outputs["Value"], pulse_w.inputs[0])

    m_fl_p = ng.nodes.new("ShaderNodeMath")
    m_fl_p.operation = "MULTIPLY"
    m_fl_p.location = (560, -240)
    ng.links.new(fl_mod.outputs["Value"], m_fl_p.inputs[0])
    ng.links.new(pulse_w.outputs["Value"], m_fl_p.inputs[1])

    m_mask = ng.nodes.new("ShaderNodeMath")
    m_mask.operation = "MULTIPLY"
    m_mask.location = (720, -160)
    ng.links.new(sac_mask.outputs["Value"], m_mask.inputs[0])
    ng.links.new(m_fl_p.outputs["Value"], m_mask.inputs[1])

    em_str = ng.nodes.new("ShaderNodeMath")
    em_str.operation = "MULTIPLY"
    em_str.location = (840, -160)
    ng.links.new(m_mask.outputs["Value"], em_str.inputs[0])
    ng.links.new(n_in.outputs["Glow Strength"], em_str.inputs[1])

    # 7. Deep saturated translucent liquid diffuse tint (replaces dark rubbery base albedo so sacs NEVER render black!)
    #    + high-gloss wet membrane Smoothness (0.90 -> Roughness 0.10) ONLY where sac_mask > 0
    tint_w = ng.nodes.new("ShaderNodeMath")
    tint_w.operation = "MULTIPLY"
    tint_w.inputs[1].default_value = 0.86
    tint_w.use_clamp = True
    tint_w.location = (440, 340)
    ng.links.new(sac_mask.outputs["Value"], tint_w.inputs[0])

    liq_albedo = ng.nodes.new("ShaderNodeVectorMath")
    liq_albedo.operation = "SCALE"
    liq_albedo.inputs["Scale"].default_value = 0.68
    liq_albedo.location = (440, 480)
    ng.links.new(n_in.outputs["Sac Glow Color"], liq_albedo.inputs[0])

    mix_bc = ng.nodes.new("ShaderNodeMix")
    mix_bc.data_type = "RGBA"
    mix_bc.location = (680, 340)
    ng.links.new(tint_w.outputs["Value"], mix_bc.inputs["Factor"])
    ng.links.new(n_in.outputs["Base Color"], mix_bc.inputs[6])
    ng.links.new(liq_albedo.outputs["Vector"], mix_bc.inputs[7])

    mix_sm = ng.nodes.new("ShaderNodeMix")
    mix_sm.data_type = "FLOAT"
    mix_sm.inputs[3].default_value = 0.90
    mix_sm.location = (680, 180)
    ng.links.new(sac_mask.outputs["Value"], mix_sm.inputs["Factor"])
    ng.links.new(n_in.outputs["Base Smoothness"], mix_sm.inputs[2])

    ng.links.new(mix_bc.outputs[2], n_out.inputs["Color"])
    ng.links.new(mix_sm.outputs[0], n_out.inputs["Smoothness"])
    ng.links.new(mix_rim.outputs[2], n_out.inputs["Emission Color"])
    ng.links.new(em_str.outputs["Value"], n_out.inputs["Emission Strength"])
    ng.links.new(sac_mask.outputs["Value"], n_out.inputs["Sac Mask"])
    return ng


def _ensure_canister_glow_group(noise_img: bpy.types.Image | None) -> bpy.types.ShaderNodeTree:
    """Create `HFW_CanisterGlow` replicating Guerrilla's `7564:6` & `7036:2` animated canister glow shaders.

    Verified from `acid_canister_shell_13451_37.txt` L987-1082 & `acid_canister_glow_13451_69.txt`:
      - `mat_id == 1` (`round(Mask * 20) == 1`): pulsing indicator collar via `sin(Time * 8.0) * 0.3 + 0.7`
      - `mat_id == 20` (`round(Mask * 20) == 20`): scrolling `107:0` fluid noise window glowing with `0x0e470c22` RGB!
    """
    tag = noise_img.name[:24] if noise_img else "no_noise"
    ng_name = f"HFW_CanisterGlow_{tag}"
    existing = bpy.data.node_groups.get(ng_name)
    if existing is not None:
        return existing

    ng = bpy.data.node_groups.new(ng_name, "ShaderNodeTree")
    ng.interface.new_socket(name="Base Color", in_out="INPUT", socket_type="NodeSocketColor")
    ng.interface.new_socket(name="Mask", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 0.0
    ng.interface.new_socket(name="Time", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 0.0
    s_gc = ng.interface.new_socket(name="Glow Color", in_out="INPUT", socket_type="NodeSocketColor")
    s_gc.default_value = (0.52, 0.70, 0.10, 1.0)
    ng.interface.new_socket(name="Glow Strength", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 5.5
    ng.interface.new_socket(name="Pulse Speed", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 8.0
    ng.interface.new_socket(name="Animate Speed", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 1.0
    ng.interface.new_socket(name="Is Fluid Body", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 1.0

    ng.interface.new_socket(name="Color", in_out="OUTPUT", socket_type="NodeSocketColor")
    ng.interface.new_socket(name="Emission Color", in_out="OUTPUT", socket_type="NodeSocketColor")
    ng.interface.new_socket(name="Emission Strength", in_out="OUTPUT", socket_type="NodeSocketFloat")
    ng.interface.new_socket(name="Fluid Mask", in_out="OUTPUT", socket_type="NodeSocketFloat")

    n_in = ng.nodes.new("NodeGroupInput")
    n_in.location = (-1320, 0)
    n_out = ng.nodes.new("NodeGroupOutput")
    n_out.location = (760, 0)

    # mat_id = round(Mask * 20.0)
    m20 = ng.nodes.new("ShaderNodeMath")
    m20.operation = "MULTIPLY"
    m20.inputs[1].default_value = 20.0
    m20.location = (-1060, 260)
    ng.links.new(n_in.outputs["Mask"], m20.inputs[0])

    rnd = ng.nodes.new("ShaderNodeMath")
    rnd.operation = "ROUND"
    rnd.location = (-880, 260)
    ng.links.new(m20.outputs["Value"], rnd.inputs[0])

    is_m1 = ng.nodes.new("ShaderNodeMath")
    is_m1.operation = "COMPARE"
    is_m1.inputs[1].default_value = 1.0
    is_m1.inputs[2].default_value = 0.25
    is_m1.label = "mat_id == 1 (Collar LED)"
    is_m1.location = (-660, 320)
    ng.links.new(rnd.outputs["Value"], is_m1.inputs[0])

    is_m20 = ng.nodes.new("ShaderNodeMath")
    is_m20.operation = "COMPARE"
    is_m20.inputs[1].default_value = 20.0
    is_m20.inputs[2].default_value = 0.25
    is_m20.label = "mat_id == 20 (Fluid Window)"
    is_m20.location = (-660, 120)
    ng.links.new(rnd.outputs["Value"], is_m20.inputs[0])

    # Time * AnimateSpeed
    t_eff = ng.nodes.new("ShaderNodeMath")
    t_eff.operation = "MULTIPLY"
    t_eff.location = (-1060, -180)
    ng.links.new(n_in.outputs["Time"], t_eff.inputs[0])
    ng.links.new(n_in.outputs["Animate Speed"], t_eff.inputs[1])

    # Collar pulse: sin(t_eff * PulseSpeed) * 0.30 + 0.70 (DXIL L994/1048!)
    t_p = ng.nodes.new("ShaderNodeMath")
    t_p.operation = "MULTIPLY"
    t_p.location = (-840, -60)
    ng.links.new(t_eff.outputs["Value"], t_p.inputs[0])
    ng.links.new(n_in.outputs["Pulse Speed"], t_p.inputs[1])

    sin_p = ng.nodes.new("ShaderNodeMath")
    sin_p.operation = "SINE"
    sin_p.location = (-640, -60)
    ng.links.new(t_p.outputs["Value"], sin_p.inputs[0])

    p_val = ng.nodes.new("ShaderNodeMath")
    p_val.operation = "MULTIPLY_ADD"
    p_val.inputs[1].default_value = 0.30
    p_val.inputs[2].default_value = 0.70
    p_val.location = (-440, -60)
    ng.links.new(sin_p.outputs["Value"], p_val.inputs[0])

    collar_glow = ng.nodes.new("ShaderNodeMath")
    collar_glow.operation = "MULTIPLY"
    collar_glow.location = (-200, 200)
    ng.links.new(is_m1.outputs["Value"], collar_glow.inputs[0])
    ng.links.new(p_val.outputs["Value"], collar_glow.inputs[1])

    # Fluid window scrolling 107:0 noise
    if noise_img is not None:
        tc = ng.nodes.new("ShaderNodeTexCoord")
        tc.location = (-1060, -440)
        v_scr = ng.nodes.new("ShaderNodeMath")
        v_scr.operation = "MULTIPLY"
        v_scr.inputs[1].default_value = 0.28
        v_scr.location = (-840, -280)
        ng.links.new(t_eff.outputs["Value"], v_scr.inputs[0])

        comb_off = ng.nodes.new("ShaderNodeCombineXYZ")
        comb_off.location = (-660, -320)
        ng.links.new(v_scr.outputs["Value"], comb_off.inputs["Y"])

        uv_sc = ng.nodes.new("ShaderNodeVectorMath")
        uv_sc.operation = "SCALE"
        uv_sc.inputs["Scale"].default_value = 2.0
        uv_sc.location = (-840, -460)
        ng.links.new(tc.outputs["UV"], uv_sc.inputs[0])

        uv_add = ng.nodes.new("ShaderNodeVectorMath")
        uv_add.operation = "ADD"
        uv_add.location = (-480, -420)
        ng.links.new(uv_sc.outputs["Vector"], uv_add.inputs[0])
        ng.links.new(comb_off.outputs["Vector"], uv_add.inputs[1])

        t_n = ng.nodes.new("ShaderNodeTexImage")
        t_n.image = noise_img
        t_n.label = "HFW Canister Noise (107:0)"
        t_n.location = (-260, -420)
        ng.links.new(uv_add.outputs["Vector"], t_n.inputs["Vector"])

        n_mod = ng.nodes.new("ShaderNodeMath")
        n_mod.operation = "MULTIPLY_ADD"
        n_mod.inputs[1].default_value = 0.65
        n_mod.inputs[2].default_value = 0.55
        n_mod.location = (20, -340)
        ng.links.new(t_n.outputs["Color"], n_mod.inputs[0])
        fluid_mod = n_mod.outputs["Value"]
    else:
        fluid_mod = p_val.outputs["Value"]

    win_base = ng.nodes.new("ShaderNodeMix")
    win_base.data_type = "FLOAT"
    win_base.inputs[2].default_value = 0.70
    win_base.inputs[3].default_value = 1.00
    win_base.location = (-100, -140)
    ng.links.new(n_in.outputs["Is Fluid Body"], win_base.inputs["Factor"])

    win_m1 = ng.nodes.new("ShaderNodeMath")
    win_m1.operation = "MULTIPLY"
    win_m1.location = (160, -80)
    ng.links.new(is_m20.outputs["Value"], win_m1.inputs[0])
    ng.links.new(fluid_mod, win_m1.inputs[1])

    win_glow = ng.nodes.new("ShaderNodeMath")
    win_glow.operation = "MULTIPLY"
    win_glow.location = (320, -80)
    ng.links.new(win_m1.outputs["Value"], win_glow.inputs[0])
    ng.links.new(win_base.outputs[0], win_glow.inputs[1])

    tot_m = ng.nodes.new("ShaderNodeMath")
    tot_m.operation = "MAXIMUM"
    tot_m.location = (460, 40)
    ng.links.new(collar_glow.outputs["Value"], tot_m.inputs[0])
    ng.links.new(win_glow.outputs["Value"], tot_m.inputs[1])

    em_str = ng.nodes.new("ShaderNodeMath")
    em_str.operation = "MULTIPLY"
    em_str.location = (600, 40)
    ng.links.new(tot_m.outputs["Value"], em_str.inputs[0])
    ng.links.new(n_in.outputs["Glow Strength"], em_str.inputs[1])

    # Tint Base Color on mat_id==20 window so canister liquid reads richly in diffuse lighting too
    bc_mix = ng.nodes.new("ShaderNodeMix")
    bc_mix.data_type = "RGBA"
    bc_mix.location = (460, 260)
    ng.links.new(is_m20.outputs["Value"], bc_mix.inputs["Factor"])
    ng.links.new(n_in.outputs["Base Color"], bc_mix.inputs[6])
    ng.links.new(n_in.outputs["Glow Color"], bc_mix.inputs[7])

    ng.links.new(bc_mix.outputs[2], n_out.inputs["Color"])
    ng.links.new(n_in.outputs["Glow Color"], n_out.inputs["Emission Color"])
    ng.links.new(em_str.outputs["Value"], n_out.inputs["Emission Strength"])
    ng.links.new(is_m20.outputs["Value"], n_out.inputs["Fluid Mask"])
    return ng


def _ensure_eye_lens_group(iris_img: bpy.types.Image | None) -> bpy.types.ShaderNodeTree:
    """Create `HFW_MachineEyeLens` replicating Guerrilla's `10380:0`/`17093:0` glowing outer machine eye optics."""
    tag = iris_img.name[:24] if iris_img else "no_iris"
    ng_name = f"HFW_MachineEyeLens_{tag}"
    existing = bpy.data.node_groups.get(ng_name)
    if existing is not None:
        return existing

    ng = bpy.data.node_groups.new(ng_name, "ShaderNodeTree")
    ng.interface.new_socket(name="Time", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 0.0
    s_cc = ng.interface.new_socket(name="Calm Color", in_out="INPUT", socket_type="NodeSocketColor")
    s_cc.default_value = (0.04, 0.56, 1.0, 1.0)
    s_ac = ng.interface.new_socket(name="Alert Color", in_out="INPUT", socket_type="NodeSocketColor")
    s_ac.default_value = (1.0, 0.10, 0.02, 1.0)
    ng.interface.new_socket(name="Alert State", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 0.0
    ng.interface.new_socket(name="Glow Strength", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 9.0
    ng.interface.new_socket(name="Animate Speed", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 1.0

    ng.interface.new_socket(name="Emission Color", in_out="OUTPUT", socket_type="NodeSocketColor")
    ng.interface.new_socket(name="Emission Strength", in_out="OUTPUT", socket_type="NodeSocketFloat")

    n_in = ng.nodes.new("NodeGroupInput")
    n_in.location = (-860, 0)
    n_out = ng.nodes.new("NodeGroupOutput")
    n_out.location = (560, 0)

    # Animated optical scan pulse: 0.88 + 0.12 * sin(Time * AnimateSpeed * 2.5)
    t_m = ng.nodes.new("ShaderNodeMath")
    t_m.operation = "MULTIPLY"
    t_m.location = (-560, -240)
    ng.links.new(n_in.outputs["Time"], t_m.inputs[0])
    ng.links.new(n_in.outputs["Animate Speed"], t_m.inputs[1])

    t_f = ng.nodes.new("ShaderNodeMath")
    t_f.operation = "MULTIPLY"
    t_f.inputs[1].default_value = 2.5
    t_f.location = (-380, -240)
    ng.links.new(t_m.outputs["Value"], t_f.inputs[0])

    s_w = ng.nodes.new("ShaderNodeMath")
    s_w.operation = "SINE"
    s_w.location = (-200, -240)
    ng.links.new(t_f.outputs["Value"], s_w.inputs[0])

    pulse = ng.nodes.new("ShaderNodeMath")
    pulse.operation = "MULTIPLY_ADD"
    pulse.inputs[1].default_value = 0.12
    pulse.inputs[2].default_value = 0.88
    pulse.location = (-20, -240)
    ng.links.new(s_w.outputs["Value"], pulse.inputs[0])

    if iris_img is not None:
        tc = ng.nodes.new("ShaderNodeTexCoord")
        tc.location = (-660, 220)
        ti = ng.nodes.new("ShaderNodeTexImage")
        ti.image = iris_img
        ti.label = "HFW Eye Iris Mask (0x5e969132)"
        ti.location = (-400, 220)
        ng.links.new(tc.outputs["UV"], ti.inputs["Vector"])

        sep = ng.nodes.new("ShaderNodeSeparateColor")
        sep.location = (-100, 220)
        ng.links.new(ti.outputs["Color"], sep.inputs["Color"])

        iris_m = ng.nodes.new("ShaderNodeMath")
        iris_m.operation = "MULTIPLY"
        iris_m.location = (160, 120)
        ng.links.new(sep.outputs["Red"], iris_m.inputs[0])
        ng.links.new(pulse.outputs["Value"], iris_m.inputs[1])
        mask_sock = iris_m.outputs["Value"]
    else:
        mask_sock = pulse.outputs["Value"]

    str_m = ng.nodes.new("ShaderNodeMath")
    str_m.operation = "MULTIPLY"
    str_m.location = (360, 80)
    ng.links.new(mask_sock, str_m.inputs[0])
    ng.links.new(n_in.outputs["Glow Strength"], str_m.inputs[1])

    mix_c = ng.nodes.new("ShaderNodeMix")
    mix_c.data_type = "RGBA"
    mix_c.location = (260, -120)
    ng.links.new(n_in.outputs["Alert State"], mix_c.inputs["Factor"])
    ng.links.new(n_in.outputs["Calm Color"], mix_c.inputs[6])
    ng.links.new(n_in.outputs["Alert Color"], mix_c.inputs[7])

    ng.links.new(mix_c.outputs[2], n_out.inputs["Emission Color"])
    ng.links.new(str_m.outputs["Value"], n_out.inputs["Emission Strength"])
    return ng


def _ensure_muscle_cables_group() -> bpy.types.ShaderNodeTree:
    """Create `HFW_MachineMuscleCables` replicating Guerrilla's `8260:1` braided glowing muscle cable shader.

    Verified from `dxil_muscle_5540_23.txt`:
      - Base obsidian synthetic cable (`Base Color = dark charcoal`, `Roughness = 0.45`, `Specular = 0.5`)
      - `4019:1` (`UV * (6, 2)`): `R,G` = interlocking braided muscle normal, `B` = glowing braid crevices
      - `5645:0` (`UV * (6, 3) + (0, PulsePhase)`): scrolling energy pulse streaks
      - `4019:0` (`UV * (3, 1)`): `G` = glowing circuit veins
      - `Glow Color` (default DXIL Electric Cyan-Blue `(0.0, 0.231, 1.0)`), `Alert Color`, `Alert State`, `Glow Strength`.
    """
    ng_name = "HFW_MachineMuscleCables"
    existing = bpy.data.node_groups.get(ng_name)
    if existing is not None:
        return existing

    ng = bpy.data.node_groups.new(ng_name, "ShaderNodeTree")
    ng.interface.new_socket(name="Braid Mask", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 0.5
    ng.interface.new_socket(name="Pulse Mask", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 0.3
    ng.interface.new_socket(name="Vein Mask", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 0.2
    s_gc = ng.interface.new_socket(name="Glow Color", in_out="INPUT", socket_type="NodeSocketColor")
    s_gc.default_value = (0.0, 0.231, 1.0, 1.0)
    s_ac = ng.interface.new_socket(name="Alert Color", in_out="INPUT", socket_type="NodeSocketColor")
    s_ac.default_value = (0.95, 0.08, 0.02, 1.0)
    ng.interface.new_socket(name="Alert State", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 0.0
    ng.interface.new_socket(name="Glow Strength", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 6.0

    ng.interface.new_socket(name="Emission Color", in_out="OUTPUT", socket_type="NodeSocketColor")
    ng.interface.new_socket(name="Emission Strength", in_out="OUTPUT", socket_type="NodeSocketFloat")

    n_in = ng.nodes.new("NodeGroupInput")
    n_in.location = (-780, 0)
    n_out = ng.nodes.new("NodeGroupOutput")
    n_out.location = (520, 0)

    # Exact DXIL 8260:1 L325-386: dark carbon braided cable with glowing crevices, scrolling pulse streaks & veins,
    # fading to pure dark metallic cable on grazing silhouettes (`saturate((-0.25 - dot(N,V)) * 4.0)`)
    # and transitioning through a rich Dark Ember -> Vibrant Muscle Color -> Hot Bright Core gradient!
    br_adj = ng.nodes.new("ShaderNodeMath")
    br_adj.operation = "MULTIPLY_ADD"
    br_adj.inputs[1].default_value = 1.053
    br_adj.inputs[2].default_value = -0.053
    br_adj.use_clamp = True
    br_adj.location = (-520, 220)
    ng.links.new(n_in.outputs["Braid Mask"], br_adj.inputs[0])

    pl_scale = ng.nodes.new("ShaderNodeMath")
    pl_scale.operation = "MULTIPLY_ADD"
    pl_scale.inputs[1].default_value = 1.10
    pl_scale.inputs[2].default_value = 0.10
    pl_scale.location = (-520, 40)
    ng.links.new(n_in.outputs["Pulse Mask"], pl_scale.inputs[0])

    br_pl = ng.nodes.new("ShaderNodeMath")
    br_pl.operation = "MULTIPLY"
    br_pl.location = (-280, 140)
    ng.links.new(br_adj.outputs["Value"], br_pl.inputs[0])
    ng.links.new(pl_scale.outputs["Value"], br_pl.inputs[1])

    vn_scl = ng.nodes.new("ShaderNodeMath")
    vn_scl.operation = "MULTIPLY"
    vn_scl.inputs[1].default_value = 0.65
    vn_scl.location = (-280, -20)
    ng.links.new(n_in.outputs["Vein Mask"], vn_scl.inputs[0])

    vn_add = ng.nodes.new("ShaderNodeMath")
    vn_add.operation = "ADD"
    vn_add.use_clamp = True
    vn_add.location = (-80, 100)
    ng.links.new(br_pl.outputs["Value"], vn_add.inputs[0])
    ng.links.new(vn_scl.outputs["Value"], vn_add.inputs[1])

    # DXIL 8260:1 L377-379: `LayerWeight.Facing` = 1 - |N.V| (0 at center, 1 at grazing edge)!
    # Map 0.0 (center) -> 1.0 glow, 0.72+ (grazing edge) -> 0.0 dark carbon cable!
    lw = ng.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.50
    lw.location = (-520, -180)

    center_mr = ng.nodes.new("ShaderNodeMapRange")
    center_mr.interpolation_type = "SMOOTHSTEP"
    center_mr.clamp = True
    center_mr.label = "DXIL Center-Facing Falloff (Dark Edges)"
    center_mr.inputs["From Min"].default_value = 0.08
    center_mr.inputs["From Max"].default_value = 0.74
    center_mr.inputs["To Min"].default_value = 1.0
    center_mr.inputs["To Max"].default_value = 0.0
    center_mr.location = (-280, -180)
    ng.links.new(lw.outputs["Facing"], center_mr.inputs["Value"])

    center_pow = ng.nodes.new("ShaderNodeMath")
    center_pow.operation = "POWER"
    center_pow.inputs[1].default_value = 1.45
    center_pow.location = (-80, -180)
    ng.links.new(center_mr.outputs["Result"], center_pow.inputs[0])

    tot_m = ng.nodes.new("ShaderNodeMath")
    tot_m.operation = "MULTIPLY"
    tot_m.use_clamp = True
    tot_m.location = (100, 60)
    ng.links.new(vn_add.outputs["Value"], tot_m.inputs[0])
    ng.links.new(center_pow.outputs["Value"], tot_m.inputs[1])

    tot_str = ng.nodes.new("ShaderNodeMath")
    tot_str.operation = "MULTIPLY"
    tot_str.location = (320, 60)
    ng.links.new(tot_m.outputs["Value"], tot_str.inputs[0])
    ng.links.new(n_in.outputs["Glow Strength"], tot_str.inputs[1])

    # Dynamic 3-Tier Thermal Gradient: Deep Burnt Ember -> Vibrant Muscle Color -> Hot Bright Core!
    mix_c = ng.nodes.new("ShaderNodeMix")
    mix_c.data_type = "RGBA"
    mix_c.location = (-80, -400)
    ng.links.new(n_in.outputs["Alert State"], mix_c.inputs["Factor"])
    ng.links.new(n_in.outputs["Glow Color"], mix_c.inputs[6])
    ng.links.new(n_in.outputs["Alert Color"], mix_c.inputs[7])

    ember_gam = ng.nodes.new("ShaderNodeGamma")
    ember_gam.inputs["Gamma"].default_value = 2.2
    ember_gam.label = "Deep Ember Color"
    ember_gam.location = (100, -320)
    ng.links.new(mix_c.outputs[2], ember_gam.inputs["Color"])

    ember_dim = ng.nodes.new("ShaderNodeVectorMath")
    ember_dim.operation = "SCALE"
    ember_dim.inputs["Scale"].default_value = 0.42
    ember_dim.location = (260, -320)
    ng.links.new(ember_gam.outputs["Color"], ember_dim.inputs[0])

    hot_mix = ng.nodes.new("ShaderNodeMix")
    hot_mix.data_type = "RGBA"
    hot_mix.inputs["Factor"].default_value = 0.45
    hot_mix.inputs[7].default_value = (1.0, 0.90, 0.72, 1.0)
    hot_mix.label = "Hot Core Color"
    hot_mix.location = (100, -520)
    ng.links.new(mix_c.outputs[2], hot_mix.inputs[6])

    t_low = ng.nodes.new("ShaderNodeMapRange")
    t_low.interpolation_type = "SMOOTHSTEP"
    t_low.clamp = True
    t_low.inputs["From Min"].default_value = 0.04
    t_low.inputs["From Max"].default_value = 0.48
    t_low.location = (100, -140)
    ng.links.new(tot_m.outputs["Value"], t_low.inputs["Value"])

    t_hi = ng.nodes.new("ShaderNodeMapRange")
    t_hi.interpolation_type = "SMOOTHSTEP"
    t_hi.clamp = True
    t_hi.inputs["From Min"].default_value = 0.56
    t_hi.inputs["From Max"].default_value = 0.96
    t_hi.location = (260, -140)
    ng.links.new(tot_m.outputs["Value"], t_hi.inputs["Value"])

    g_mid = ng.nodes.new("ShaderNodeMix")
    g_mid.data_type = "RGBA"
    g_mid.label = "Ember -> Muscle Color"
    g_mid.location = (440, -280)
    ng.links.new(t_low.outputs["Result"], g_mid.inputs["Factor"])
    ng.links.new(ember_dim.outputs["Vector"], g_mid.inputs[6])
    ng.links.new(mix_c.outputs[2], g_mid.inputs[7])

    g_hot = ng.nodes.new("ShaderNodeMix")
    g_hot.data_type = "RGBA"
    g_hot.label = "Muscle Color -> Hot Core"
    g_hot.location = (620, -280)
    ng.links.new(t_hi.outputs["Result"], g_hot.inputs["Factor"])
    ng.links.new(g_mid.outputs[2], g_hot.inputs[6])
    ng.links.new(hot_mix.outputs[2], g_hot.inputs[7])

    ng.links.new(g_hot.outputs[2], n_out.inputs["Emission Color"])
    ng.links.new(tot_str.outputs["Value"], n_out.inputs["Emission Strength"])
    return ng


def _ensure_machine_shader_control_group(defaults: dict[str, Any] | None = None) -> bpy.types.ShaderNodeTree:
    """Create/update the unified `HFW_MachineShaderControl` shared controller node group.

    Exposes intuitive, artist-friendly single-point controls synced with the Blender N-Panel across
    ALL muscles, braided cables, optical eye sensors, elemental sacs, canisters, and plasma charge parts:
      - Muscle Color / Muscle Glow Scale
      - Eye Color / Eye Alert Color / Eye Alert State / Eye Glow Scale
      - Liquid Override Color / Liquid Override Mix / Liquid Glow Scale
      - Plasma Color / Plasma Charge
      - Animation Speed
    """
    ng_name = "HFW_MachineShaderControl"
    d = defaults or {}
    mc = d.get("muscle_color", [0.95, 0.28, 0.02])
    ec = d.get("eye_color", [0.04, 0.56, 1.0])
    eac = d.get("eye_alert_color", [1.0, 0.08, 0.01])
    pc = d.get("plasma_color", [0.46, 0.05, 1.0])
    lc = d.get("liquid_override_color", [0.08, 0.62, 1.0])

    existing = bpy.data.node_groups.get(ng_name)
    if existing is not None:
        if defaults:
            # Update defaults when importing a fresh machine/variant
            val_updates = {
                "Muscle Glow Scale": 1.0,
                "Eye Alert State": 0.0,
                "Eye Glow Scale": 1.0,
                "Liquid Override Mix": float(d.get("liquid_override_fac", 0.0)),
                "Liquid Glow Scale": float(d.get("liquid_glow_scale", 1.0)),
                "Plasma Charge": float(d.get("plasma_charge", 1.0 if d.get("is_plasma") else 0.0)),
                "Animation Speed": float(d.get("animation_speed", 1.0)),
                "Detail Tiling Scale": float(d.get("detail_tiling_scale", 1.0)),
                "Detail Strength": float(d.get("detail_strength", 0.72)),
            }
            col_updates = {
                "Muscle Color": (mc[0], mc[1], mc[2], 1.0),
                "Eye Color": (ec[0], ec[1], ec[2], 1.0),
                "Eye Alert Color": (eac[0], eac[1], eac[2], 1.0),
                "Liquid Override Color": (lc[0], lc[1], lc[2], 1.0),
                "Plasma Color": (pc[0], pc[1], pc[2], 1.0),
            }
            for n in existing.nodes:
                if n.type == "VALUE" and n.label in val_updates:
                    n.outputs[0].default_value = val_updates[n.label]
                elif n.type == "RGB" and n.label in col_updates:
                    n.outputs[0].default_value = col_updates[n.label]
        return existing

    ng = bpy.data.node_groups.new(ng_name, "ShaderNodeTree")
    n_out = ng.nodes.new("NodeGroupOutput")
    n_out.location = (320, 0)

    specs: list[tuple[str, str, Any]] = [
        ("Muscle Color", "NodeSocketColor", (mc[0], mc[1], mc[2], 1.0)),
        ("Muscle Glow Scale", "NodeSocketFloat", 1.0),
        ("Eye Color", "NodeSocketColor", (ec[0], ec[1], ec[2], 1.0)),
        ("Eye Alert Color", "NodeSocketColor", (eac[0], eac[1], eac[2], 1.0)),
        ("Eye Alert State", "NodeSocketFloat", 0.0),
        ("Eye Glow Scale", "NodeSocketFloat", 1.0),
        ("Liquid Override Color", "NodeSocketColor", (lc[0], lc[1], lc[2], 1.0)),
        ("Liquid Override Mix", "NodeSocketFloat", float(d.get("liquid_override_fac", 0.0))),
        ("Liquid Glow Scale", "NodeSocketFloat", float(d.get("liquid_glow_scale", 1.0))),
        ("Plasma Color", "NodeSocketColor", (pc[0], pc[1], pc[2], 1.0)),
        ("Plasma Charge", "NodeSocketFloat", float(d.get("plasma_charge", 1.0 if d.get("is_plasma") else 0.0))),
        ("Animation Speed", "NodeSocketFloat", float(d.get("animation_speed", 1.0))),
        ("Detail Tiling Scale", "NodeSocketFloat", float(d.get("detail_tiling_scale", 1.0))),
        ("Detail Strength", "NodeSocketFloat", float(d.get("detail_strength", 0.72))),
    ]

    y = 480
    for name, stype, default_val in specs:
        ng.interface.new_socket(name=name, in_out="OUTPUT", socket_type=stype)
        if stype == "NodeSocketColor":
            node = ng.nodes.new("ShaderNodeRGB")
            node.outputs[0].default_value = default_val
        else:
            node = ng.nodes.new("ShaderNodeValue")
            node.outputs[0].default_value = default_val
        node.label = name
        node.location = (-220, y)
        ng.links.new(node.outputs[0], n_out.inputs[name])
        y -= 110

    return ng


def _ensure_plasma_charge_control_group() -> bpy.types.ShaderNodeTree:
    """Shared single-slider controller (`HFW_PlasmaChargeControl`) driving Slaughterspine's plasma charge-up across all materials."""
    ng_name = "HFW_PlasmaChargeControl"
    existing = bpy.data.node_groups.get(ng_name)
    if existing is not None:
        return existing

    ng = bpy.data.node_groups.new(ng_name, "ShaderNodeTree")
    ng.interface.new_socket(name="Plasma Charge", in_out="OUTPUT", socket_type="NodeSocketFloat")
    val = ng.nodes.new("ShaderNodeValue")
    val.label = "Plasma Charge"
    val.outputs[0].default_value = 1.0
    val.location = (-200, 0)
    n_out = ng.nodes.new("NodeGroupOutput")
    n_out.location = (120, 0)
    ng.links.new(val.outputs[0], n_out.inputs["Plasma Charge"])
    return ng


def _ensure_plasma_charge_group(noise_img: bpy.types.Image | None) -> bpy.types.ShaderNodeTree:
    """Create `HFW_PlasmaCharge` replicating Guerrilla's exact Slaughterspine plasma charge-up DXIL (`1742:0` / `1811:4` / `16236:2`).

    Verified directly from disassembled `dxil_ss_plasmaspike.ll`:
      - `Color0.R` (`STAGE_IO3.x`, L184 & L725): Guerrilla's vertex-painted Plasma Energizer / Dorsal Spine / Core / Jaw / Tail mask!
      - `Color0.G` (`STAGE_IO3.y`): Head-to-Tail coordinate (`0.0 = Head`, `0.35..0.50 = Spines/Chargers`, `0.94 = Tail`) driving the surge wave!
      - `(1 - AO_remap)^2` crevice coil focus (L598-602, L636): deep recessed plasma vents burn hottest while outer metal stays crisp!
      - `round(Mask * 20) == 1 * Mask_Alpha` (L623-768): Slaughterspine Body conduits shift smoothly Idle Amber -> Electric Plasma Purple!
      - Animated `107:0` scrolling plasma turbulence + Fresnel electric violet aura (`VB[10] = [0.023, 0.0065, 0.7446]`, L702-713)!
    """
    tag = noise_img.name[:24] if noise_img else "no_noise"
    ng_name = f"HFW_PlasmaCharge_{tag}"
    existing = bpy.data.node_groups.get(ng_name)
    if existing is not None:
        return existing

    ng = bpy.data.node_groups.new(ng_name, "ShaderNodeTree")
    ng.interface.new_socket(name="Mask", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 0.0
    ng.interface.new_socket(name="Mask Alpha", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 1.0
    ng.interface.new_socket(name="AO", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 0.5
    ng.interface.new_socket(name="Time", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 0.0
    ng.interface.new_socket(name="Plasma Charge", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 1.0
    s_pc = ng.interface.new_socket(name="Plasma Color", in_out="INPUT", socket_type="NodeSocketColor")
    s_pc.default_value = (0.46, 0.05, 1.0, 1.0)
    s_ic = ng.interface.new_socket(name="Idle Conduit Color", in_out="INPUT", socket_type="NodeSocketColor")
    s_ic.default_value = (0.85, 0.18, 0.02, 1.0)
    ng.interface.new_socket(name="Glow Strength", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 4.2
    ng.interface.new_socket(name="Animate Speed", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 1.0

    ng.interface.new_socket(name="Emission Color", in_out="OUTPUT", socket_type="NodeSocketColor")
    ng.interface.new_socket(name="Emission Strength", in_out="OUTPUT", socket_type="NodeSocketFloat")

    n_in = ng.nodes.new("NodeGroupInput")
    n_in.location = (-1680, 0)
    n_out = ng.nodes.new("NodeGroupOutput")
    n_out.location = (980, 0)

    # 1. Read Vertex Color `Color0` (.R = plasma emitter mask, .G = head-to-tail surge gradient!)
    vcol = ng.nodes.new("ShaderNodeAttribute")
    vcol.attribute_name = "Color0"
    vcol.label = "HFW Vertex Color0 (R=PlasmaMask, G=Wave)"
    vcol.location = (-1660, 420)

    sep_vc = ng.nodes.new("ShaderNodeSeparateColor")
    sep_vc.location = (-1420, 420)
    ng.links.new(vcol.outputs["Color"], sep_vc.inputs["Color"])

    # 2. Exact DXIL crevice heat focus (L598-602, L636): (1 - saturate((AO - 0.15) * 1.1765) * 0.88)^2 * 0.78 + 0.22
    ao_mr = ng.nodes.new("ShaderNodeMapRange")
    ao_mr.clamp = True
    ao_mr.inputs["From Min"].default_value = 0.15
    ao_mr.inputs["From Max"].default_value = 1.0
    ao_mr.inputs["To Min"].default_value = 1.0
    ao_mr.inputs["To Max"].default_value = 0.14
    ao_mr.location = (-1420, 160)
    ng.links.new(n_in.outputs["AO"], ao_mr.inputs["Value"])

    ao_sq = ng.nodes.new("ShaderNodeMath")
    ao_sq.operation = "MULTIPLY"
    ao_sq.label = "Crevice Heat (1-AO)^2"
    ao_sq.location = (-1200, 160)
    ng.links.new(ao_mr.outputs["Result"], ao_sq.inputs[0])
    ng.links.new(ao_mr.outputs["Result"], ao_sq.inputs[1])

    ao_floor = ng.nodes.new("ShaderNodeMath")
    ao_floor.operation = "MULTIPLY_ADD"
    ao_floor.inputs[1].default_value = 0.78
    ao_floor.inputs[2].default_value = 0.22
    ao_floor.location = (-1000, 160)
    ng.links.new(ao_sq.outputs["Value"], ao_floor.inputs[0])

    # 3. Animated Head->Tail Plasma Surge Wave: 0.72 + 0.28 * sin(Time * 3.5 * AnimateSpeed - Color0.G * 10.0)
    t_eff = ng.nodes.new("ShaderNodeMath")
    t_eff.operation = "MULTIPLY"
    t_eff.location = (-1420, -180)
    ng.links.new(n_in.outputs["Time"], t_eff.inputs[0])
    ng.links.new(n_in.outputs["Animate Speed"], t_eff.inputs[1])

    t_w = ng.nodes.new("ShaderNodeMath")
    t_w.operation = "MULTIPLY"
    t_w.inputs[1].default_value = 3.5
    t_w.location = (-1200, -100)
    ng.links.new(t_eff.outputs["Value"], t_w.inputs[0])

    g_ph = ng.nodes.new("ShaderNodeMath")
    g_ph.operation = "MULTIPLY"
    g_ph.inputs[1].default_value = 10.0
    g_ph.location = (-1200, 20)
    ng.links.new(sep_vc.outputs["Green"], g_ph.inputs[0])

    w_arg = ng.nodes.new("ShaderNodeMath")
    w_arg.operation = "SUBTRACT"
    w_arg.location = (-1000, -40)
    ng.links.new(t_w.outputs["Value"], w_arg.inputs[0])
    ng.links.new(g_ph.outputs["Value"], w_arg.inputs[1])

    w_sin = ng.nodes.new("ShaderNodeMath")
    w_sin.operation = "SINE"
    w_sin.location = (-820, -40)
    ng.links.new(w_arg.outputs["Value"], w_sin.inputs[0])

    w_mod = ng.nodes.new("ShaderNodeMath")
    w_mod.operation = "MULTIPLY_ADD"
    w_mod.inputs[1].default_value = 0.28
    w_mod.inputs[2].default_value = 0.72
    w_mod.location = (-640, -40)
    ng.links.new(w_sin.outputs["Value"], w_mod.inputs[0])

    # 4. Scrolling 107:0 Plasma Energy Noise (`UV * 1.5 - (0, Time * 0.30)`)
    if noise_img is not None:
        tc = ng.nodes.new("ShaderNodeTexCoord")
        tc.location = (-1420, -480)
        t_scr = ng.nodes.new("ShaderNodeMath")
        t_scr.operation = "MULTIPLY"
        t_scr.inputs[1].default_value = -0.30
        t_scr.location = (-1200, -320)
        ng.links.new(t_eff.outputs["Value"], t_scr.inputs[0])
        c_off = ng.nodes.new("ShaderNodeCombineXYZ")
        c_off.location = (-1000, -340)
        ng.links.new(t_scr.outputs["Value"], c_off.inputs["Y"])
        mp = ng.nodes.new("ShaderNodeMapping")
        mp.inputs["Scale"].default_value = (1.5, 1.5, 1.0)
        mp.location = (-820, -440)
        ng.links.new(tc.outputs["UV"], mp.inputs["Vector"])
        ng.links.new(c_off.outputs["Vector"], mp.inputs["Location"])
        tn = ng.nodes.new("ShaderNodeTexImage")
        tn.image = noise_img
        tn.label = "HFW Plasma Turbulence (107:0)"
        tn.location = (-580, -440)
        ng.links.new(mp.outputs["Vector"], tn.inputs["Vector"])
        n_mod = ng.nodes.new("ShaderNodeMath")
        n_mod.operation = "MULTIPLY_ADD"
        n_mod.inputs[1].default_value = 0.65
        n_mod.inputs[2].default_value = 0.55
        n_mod.location = (-280, -360)
        ng.links.new(tn.outputs["Color"], n_mod.inputs[0])
        turb_sock = n_mod.outputs["Value"]
    else:
        turb_sock = w_mod.outputs["Value"]

    # Combine wave * noise
    wn_mul = ng.nodes.new("ShaderNodeMath")
    wn_mul.operation = "MULTIPLY"
    wn_mul.location = (-120, -140)
    ng.links.new(w_mod.outputs["Value"], wn_mul.inputs[0])
    ng.links.new(turb_sock, wn_mul.inputs[1])

    # 5. Vertex-painted plasma emitter intensity = Color0.R * (crevice + subtle Fresnel rim) * PlasmaCharge * wave * noise
    lw = ng.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.35
    lw.location = (-1000, 320)

    rim_boost = ng.nodes.new("ShaderNodeMath")
    rim_boost.operation = "MULTIPLY_ADD"
    rim_boost.inputs[1].default_value = 0.35
    rim_boost.inputs[2].default_value = 0.0
    rim_boost.location = (-780, 320)
    ng.links.new(lw.outputs["Fresnel"], rim_boost.inputs[0])

    heat_plus_rim = ng.nodes.new("ShaderNodeMath")
    heat_plus_rim.operation = "ADD"
    heat_plus_rim.location = (-580, 260)
    ng.links.new(ao_floor.outputs["Value"], heat_plus_rim.inputs[0])
    ng.links.new(rim_boost.outputs["Value"], heat_plus_rim.inputs[1])

    vc_heat = ng.nodes.new("ShaderNodeMath")
    vc_heat.operation = "MULTIPLY"
    vc_heat.label = "VertexColor.R * CreviceHeat"
    vc_heat.location = (-360, 340)
    ng.links.new(sep_vc.outputs["Red"], vc_heat.inputs[0])
    ng.links.new(heat_plus_rim.outputs["Value"], vc_heat.inputs[1])

    vc_chg = ng.nodes.new("ShaderNodeMath")
    vc_chg.operation = "MULTIPLY"
    vc_chg.location = (-140, 340)
    ng.links.new(vc_heat.outputs["Value"], vc_chg.inputs[0])
    ng.links.new(n_in.outputs["Plasma Charge"], vc_chg.inputs[1])

    vc_final = ng.nodes.new("ShaderNodeMath")
    vc_final.operation = "MULTIPLY"
    vc_final.location = (80, 240)
    ng.links.new(vc_chg.outputs["Value"], vc_final.inputs[0])
    ng.links.new(wn_mul.outputs["Value"], vc_final.inputs[1])

    # 6. Exact DXIL `mat_id == 1` (`round(Mask * 20) == 1 * Mask_Alpha`) body plasma conduits (Body_01..05!)
    m20 = ng.nodes.new("ShaderNodeMath")
    m20.operation = "MULTIPLY"
    m20.inputs[1].default_value = 20.0
    m20.location = (-1200, 620)
    ng.links.new(n_in.outputs["Mask"], m20.inputs[0])

    rnd = ng.nodes.new("ShaderNodeMath")
    rnd.operation = "ROUND"
    rnd.location = (-1000, 620)
    ng.links.new(m20.outputs["Value"], rnd.inputs[0])

    is_m1 = ng.nodes.new("ShaderNodeMath")
    is_m1.operation = "COMPARE"
    is_m1.inputs[1].default_value = 1.0
    is_m1.inputs[2].default_value = 0.25
    is_m1.label = "mat_id == 1 (Body Plasma Conduits)"
    is_m1.location = (-800, 620)
    ng.links.new(rnd.outputs["Value"], is_m1.inputs[0])

    cond_m = ng.nodes.new("ShaderNodeMath")
    cond_m.operation = "MULTIPLY"
    cond_m.use_clamp = True
    cond_m.location = (-580, 600)
    ng.links.new(is_m1.outputs["Value"], cond_m.inputs[0])
    ng.links.new(n_in.outputs["Mask Alpha"], cond_m.inputs[1])

    # Conduits glow subtly (0.28) when idle and brighten to 0.90 when plasma-charged
    cond_lvl = ng.nodes.new("ShaderNodeMath")
    cond_lvl.operation = "MULTIPLY_ADD"
    cond_lvl.inputs[1].default_value = 0.62
    cond_lvl.inputs[2].default_value = 0.28
    cond_lvl.location = (-360, 560)
    ng.links.new(n_in.outputs["Plasma Charge"], cond_lvl.inputs[0])

    cond_w = ng.nodes.new("ShaderNodeMath")
    cond_w.operation = "MULTIPLY"
    cond_w.location = (-140, 560)
    ng.links.new(cond_m.outputs["Value"], cond_w.inputs[0])
    ng.links.new(cond_lvl.outputs["Value"], cond_w.inputs[1])

    cond_final = ng.nodes.new("ShaderNodeMath")
    cond_final.operation = "MULTIPLY"
    cond_final.location = (80, 520)
    ng.links.new(cond_w.outputs["Value"], cond_final.inputs[0])
    ng.links.new(wn_mul.outputs["Value"], cond_final.inputs[1])

    # Total emission intensity = max(vc_final, cond_final) * GlowStrength
    tot_m = ng.nodes.new("ShaderNodeMath")
    tot_m.operation = "MAXIMUM"
    tot_m.location = (320, 320)
    ng.links.new(vc_final.outputs["Value"], tot_m.inputs[0])
    ng.links.new(cond_final.outputs["Value"], tot_m.inputs[1])

    em_str = ng.nodes.new("ShaderNodeMath")
    em_str.operation = "MULTIPLY"
    em_str.location = (560, 260)
    ng.links.new(tot_m.outputs["Value"], em_str.inputs[0])
    ng.links.new(n_in.outputs["Glow Strength"], em_str.inputs[1])

    # Color transition: Idle Conduit Amber -> Electric Plasma Purple (plus hotter lavender core on peak crevice coils!)
    mix_col = ng.nodes.new("ShaderNodeMix")
    mix_col.data_type = "RGBA"
    mix_col.label = "Idle Amber -> Plasma Violet"
    mix_col.location = (320, 40)
    ng.links.new(n_in.outputs["Plasma Charge"], mix_col.inputs["Factor"])
    ng.links.new(n_in.outputs["Idle Conduit Color"], mix_col.inputs[6])
    ng.links.new(n_in.outputs["Plasma Color"], mix_col.inputs[7])

    # Hot lavender-magenta plasma center highlight inside deepest vents (`ao_sq`)
    hot_fac = ng.nodes.new("ShaderNodeMath")
    hot_fac.operation = "MULTIPLY"
    hot_fac.use_clamp = True
    hot_fac.location = (320, -180)
    ng.links.new(ao_sq.outputs["Value"], hot_fac.inputs[0])
    ng.links.new(n_in.outputs["Plasma Charge"], hot_fac.inputs[1])

    mix_hot = ng.nodes.new("ShaderNodeMix")
    mix_hot.data_type = "RGBA"
    mix_hot.label = "Plasma Core Hot Violet"
    mix_hot.inputs[7].default_value = (0.76, 0.22, 1.0, 1.0)
    mix_hot.location = (600, 0)
    ng.links.new(hot_fac.outputs["Value"], mix_hot.inputs["Factor"])
    ng.links.new(mix_col.outputs[2], mix_hot.inputs[6])

    ng.links.new(mix_hot.outputs[2], n_out.inputs["Emission Color"])
    ng.links.new(em_str.outputs["Value"], n_out.inputs["Emission Strength"])
    return ng


def _ensure_muscle_fiber_group() -> bpy.types.ShaderNodeTree:
    """Create `HFW_MachineMuscleFiber` replicating Guerrilla's `19501:0` translucent optical muscle fiber shader.

    Verified from `dxil_tj_muscle_fiber.ll` (`ForwardBackground`):
      - Smooth BC7-artifact-free `Mask_Alpha` ramp (prevents harsh black noise dots from 1-bit BC7 quantization!)
      - Authentic 96-strand fine glowing optical filaments along `UV1.x` (`smoothstep(0.35, 0.92, abs(frac(u*96+0.5)-0.5)*2)`)
        so sleek dark synthetic muscle cables remain clearly visible between the luminous fibers!
      - DXIL view-facing fade (`dot(N, V)`) so curved muscle edges wrap cleanly without blowing out.
      - Cycles & Eevee ray hygiene: skips `Is Shadow Ray` and `Backfacing` via pure `Transparent BSDF` (zero Cycles
        shadow artifacts or backface transparent-bounce blackouts; combined with +1.4mm normal offset -> 0 z-fighting!).
    """
    ng_name = "HFW_MachineMuscleFiber"
    existing = bpy.data.node_groups.get(ng_name)
    if existing is not None:
        return existing

    ng = bpy.data.node_groups.new(ng_name, "ShaderNodeTree")
    ng.interface.new_socket(name="Mask Alpha (UV0)", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 1.0
    ng.interface.new_socket(name="Energy Wave (5147)", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 0.7
    s_cc = ng.interface.new_socket(name="Calm Color", in_out="INPUT", socket_type="NodeSocketColor")
    s_cc.default_value = (0.95, 0.24, 0.02, 1.0)
    s_pc = ng.interface.new_socket(name="Plasma Color", in_out="INPUT", socket_type="NodeSocketColor")
    s_pc.default_value = (0.46, 0.05, 1.0, 1.0)
    s_ac = ng.interface.new_socket(name="Alert Color", in_out="INPUT", socket_type="NodeSocketColor")
    s_ac.default_value = (1.0, 0.04, 0.01, 1.0)
    ng.interface.new_socket(name="Charge State", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 0.0
    ng.interface.new_socket(name="Alert State", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 0.0
    ng.interface.new_socket(name="Glow Strength", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 3.4
    ng.interface.new_socket(name="Fiber Density", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 96.0

    ng.interface.new_socket(name="Shader", in_out="OUTPUT", socket_type="NodeSocketShader")

    n_in = ng.nodes.new("NodeGroupInput")
    n_in.location = (-1180, 0)
    n_out = ng.nodes.new("NodeGroupOutput")
    n_out.location = (980, 0)

    # 1. Exact DXIL L434 `saturate(Mask_Alpha * 40.0)` with subtle anti-blackout floor (0.28..1.0) so ALL
    #    muscle bundles glow without turning grazing/cavity edges into a flat neon slab!
    ma_mr = ng.nodes.new("ShaderNodeMapRange")
    ma_mr.interpolation_type = "SMOOTHSTEP"
    ma_mr.clamp = True
    ma_mr.label = "DXIL Mask_Alpha * 40"
    ma_mr.inputs["From Min"].default_value = 0.0
    ma_mr.inputs["From Max"].default_value = 0.025
    ma_mr.inputs["To Min"].default_value = 0.28
    ma_mr.inputs["To Max"].default_value = 1.0
    ma_mr.location = (-820, 260)
    ng.links.new(n_in.outputs["Mask Alpha (UV0)"], ma_mr.inputs["Value"])

    # 2. Authentic DXIL 96-strand crisp optical fiber filaments along UV1.x (L339-350)
    uv1 = ng.nodes.new("ShaderNodeUVMap")
    uv1.uv_map = "UV1"
    uv1.location = (-1140, -240)
    sep_uv1 = ng.nodes.new("ShaderNodeSeparateXYZ")
    sep_uv1.location = (-920, -240)
    ng.links.new(uv1.outputs["UV"], sep_uv1.inputs["Vector"])

    f_mul = ng.nodes.new("ShaderNodeMath")
    f_mul.operation = "MULTIPLY"
    f_mul.location = (-740, -200)
    ng.links.new(sep_uv1.outputs["X"], f_mul.inputs[0])
    ng.links.new(n_in.outputs["Fiber Density"], f_mul.inputs[1])

    f_add = ng.nodes.new("ShaderNodeMath")
    f_add.operation = "ADD"
    f_add.inputs[1].default_value = 0.5
    f_add.location = (-560, -200)
    ng.links.new(f_mul.outputs["Value"], f_add.inputs[0])

    f_frc = ng.nodes.new("ShaderNodeMath")
    f_frc.operation = "FRACT"
    f_frc.location = (-400, -200)
    ng.links.new(f_add.outputs["Value"], f_frc.inputs[0])

    f_sub = ng.nodes.new("ShaderNodeMath")
    f_sub.operation = "SUBTRACT"
    f_sub.inputs[1].default_value = 0.5
    f_sub.location = (-240, -200)
    ng.links.new(f_frc.outputs["Value"], f_sub.inputs[0])

    f_abs = ng.nodes.new("ShaderNodeMath")
    f_abs.operation = "ABSOLUTE"
    f_abs.location = (-80, -200)
    ng.links.new(f_sub.outputs["Value"], f_abs.inputs[0])

    # Crisp distinct glowing optical filaments (`To Min = 0.0` so dark synthetic carbon cables show cleanly between strands!)
    f_x2 = ng.nodes.new("ShaderNodeMapRange")
    f_x2.interpolation_type = "SMOOTHSTEP"
    f_x2.clamp = True
    f_x2.label = "96-Strand Optical Filaments"
    f_x2.inputs["From Min"].default_value = 0.18
    f_x2.inputs["From Max"].default_value = 0.46
    f_x2.inputs["To Min"].default_value = 0.0
    f_x2.inputs["To Max"].default_value = 1.0
    f_x2.location = (100, -200)
    ng.links.new(f_abs.outputs["Value"], f_x2.inputs["Value"])

    # 3. Exact DXIL `19501:0` L312-313 + L444 View-Facing Falloff (`dot(N, V)^2`!):
    #    Curved muscle bundle silhouettes fade cleanly to dark black metallic cables while camera-facing centers glow!
    lw = ng.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.50
    lw.location = (-560, 40)

    face_mr = ng.nodes.new("ShaderNodeMapRange")
    face_mr.interpolation_type = "SMOOTHSTEP"
    face_mr.clamp = True
    face_mr.label = "DXIL View-Facing Falloff (Dark Rims)"
    face_mr.inputs["From Min"].default_value = 0.06
    face_mr.inputs["From Max"].default_value = 0.78
    face_mr.inputs["To Min"].default_value = 1.0
    face_mr.inputs["To Max"].default_value = 0.03
    face_mr.location = (-360, 40)
    ng.links.new(lw.outputs["Facing"], face_mr.inputs["Value"])

    face_pow = ng.nodes.new("ShaderNodeMath")
    face_pow.operation = "POWER"
    face_pow.inputs[1].default_value = 1.75
    face_pow.label = "DXIL N.V Gradient Curve"
    face_pow.location = (-160, 40)
    ng.links.new(face_mr.outputs["Result"], face_pow.inputs[0])

    # 4. Combine cavity * energy wave * optical strands * view-facing -> `heat` (0..1) * Glow Strength
    m1 = ng.nodes.new("ShaderNodeMath")
    m1.operation = "MULTIPLY"
    m1.location = (-100, 200)
    ng.links.new(ma_mr.outputs["Result"], m1.inputs[0])
    ng.links.new(n_in.outputs["Energy Wave (5147)"], m1.inputs[1])

    m2 = ng.nodes.new("ShaderNodeMath")
    m2.operation = "MULTIPLY"
    m2.location = (100, 140)
    ng.links.new(m1.outputs["Value"], m2.inputs[0])
    ng.links.new(f_x2.outputs["Result"], m2.inputs[1])

    m3 = ng.nodes.new("ShaderNodeMath")
    m3.operation = "MULTIPLY"
    m3.use_clamp = True
    m3.location = (260, 120)
    ng.links.new(m2.outputs["Value"], m3.inputs[0])
    ng.links.new(face_pow.outputs["Value"], m3.inputs[1])

    tot_str = ng.nodes.new("ShaderNodeMath")
    tot_str.operation = "MULTIPLY"
    tot_str.location = (440, 120)
    ng.links.new(m3.outputs["Value"], tot_str.inputs[0])
    ng.links.new(n_in.outputs["Glow Strength"], tot_str.inputs[1])

    # 5. Color state blend + Dynamic 3-Tier In-Game Gradient (Deep Dark Ember -> Muscle Color -> Hot Bright Core!)
    mix_cp = ng.nodes.new("ShaderNodeMix")
    mix_cp.data_type = "RGBA"
    mix_cp.location = (-120, 460)
    ng.links.new(n_in.outputs["Charge State"], mix_cp.inputs["Factor"])
    ng.links.new(n_in.outputs["Calm Color"], mix_cp.inputs[6])
    ng.links.new(n_in.outputs["Plasma Color"], mix_cp.inputs[7])

    mix_al = ng.nodes.new("ShaderNodeMix")
    mix_al.data_type = "RGBA"
    mix_al.location = (60, 460)
    ng.links.new(n_in.outputs["Alert State"], mix_al.inputs["Factor"])
    ng.links.new(mix_cp.outputs[2], mix_al.inputs[6])
    ng.links.new(n_in.outputs["Alert Color"], mix_al.inputs[7])

    ember_gam = ng.nodes.new("ShaderNodeGamma")
    ember_gam.inputs["Gamma"].default_value = 2.2
    ember_gam.label = "Deep Ember Rim"
    ember_gam.location = (240, 560)
    ng.links.new(mix_al.outputs[2], ember_gam.inputs["Color"])

    ember_dim = ng.nodes.new("ShaderNodeVectorMath")
    ember_dim.operation = "SCALE"
    ember_dim.inputs["Scale"].default_value = 0.38
    ember_dim.location = (400, 560)
    ng.links.new(ember_gam.outputs["Color"], ember_dim.inputs[0])

    hot_mix = ng.nodes.new("ShaderNodeMix")
    hot_mix.data_type = "RGBA"
    hot_mix.inputs["Factor"].default_value = 0.48
    hot_mix.inputs[7].default_value = (1.0, 0.90, 0.72, 1.0)
    hot_mix.label = "Hot Spark Core"
    hot_mix.location = (240, 340)
    ng.links.new(mix_al.outputs[2], hot_mix.inputs[6])

    t_low = ng.nodes.new("ShaderNodeMapRange")
    t_low.interpolation_type = "SMOOTHSTEP"
    t_low.clamp = True
    t_low.inputs["From Min"].default_value = 0.03
    t_low.inputs["From Max"].default_value = 0.42
    t_low.location = (240, 220)
    ng.links.new(m3.outputs["Value"], t_low.inputs["Value"])

    t_hi = ng.nodes.new("ShaderNodeMapRange")
    t_hi.interpolation_type = "SMOOTHSTEP"
    t_hi.clamp = True
    t_hi.inputs["From Min"].default_value = 0.52
    t_hi.inputs["From Max"].default_value = 0.94
    t_hi.location = (400, 220)
    ng.links.new(m3.outputs["Value"], t_hi.inputs["Value"])

    g_mid = ng.nodes.new("ShaderNodeMix")
    g_mid.data_type = "RGBA"
    g_mid.label = "Ember -> Muscle Color"
    g_mid.location = (560, 440)
    ng.links.new(t_low.outputs["Result"], g_mid.inputs["Factor"])
    ng.links.new(ember_dim.outputs["Vector"], g_mid.inputs[6])
    ng.links.new(mix_al.outputs[2], g_mid.inputs[7])

    g_hot = ng.nodes.new("ShaderNodeMix")
    g_hot.data_type = "RGBA"
    g_hot.label = "Muscle Color -> Hot Core"
    g_hot.location = (720, 440)
    ng.links.new(t_hi.outputs["Result"], g_hot.inputs["Factor"])
    ng.links.new(g_mid.outputs[2], g_hot.inputs[6])
    ng.links.new(hot_mix.outputs[2], g_hot.inputs[7])

    # 6. Pure additive optical glow + Cycles/Eevee Shadow-Ray Bypass (ZERO Cycles z-fight/shadow/bounce glitches!)
    tr_bsdf = ng.nodes.new("ShaderNodeBsdfTransparent")
    tr_bsdf.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    tr_bsdf.location = (440, -100)

    em_bsdf = ng.nodes.new("ShaderNodeEmission")
    em_bsdf.location = (440, 260)
    ng.links.new(g_hot.outputs[2], em_bsdf.inputs["Color"])
    ng.links.new(tot_str.outputs["Value"], em_bsdf.inputs["Strength"])

    add_sh = ng.nodes.new("ShaderNodeAddShader")
    add_sh.location = (620, 100)
    ng.links.new(tr_bsdf.outputs["BSDF"], add_sh.inputs[0])
    ng.links.new(em_bsdf.outputs["Emission"], add_sh.inputs[1])

    lp = ng.nodes.new("ShaderNodeLightPath")
    lp.location = (380, -340)

    # Bypass ONLY Shadow Rays (NOT Backfacing!) so mirrored/inner-cavity muscle fibers NEVER go invisible!
    mix_safe = ng.nodes.new("ShaderNodeMixShader")
    mix_safe.label = "Cycles/Eevee Safe Additive Overlay"
    mix_safe.location = (800, 20)
    ng.links.new(lp.outputs["Is Shadow Ray"], mix_safe.inputs["Fac"])
    ng.links.new(add_sh.outputs["Shader"], mix_safe.inputs[1])
    ng.links.new(tr_bsdf.outputs["BSDF"], mix_safe.inputs[2])
    ng.links.new(mix_safe.outputs["Shader"], n_out.inputs["Shader"])
    return ng


def _load_image(tex_dir: Path, filename: str, colorspace: str, img_cache: dict[str, bpy.types.Image]) -> bpy.types.Image | None:
    if not filename:
        return None
    cache_key = f"{filename}|{colorspace}"
    if cache_key in img_cache:
        return img_cache[cache_key]
    fpath = tex_dir / filename
    if not fpath.exists():
        return None
    img = bpy.data.images.load(str(fpath), check_existing=True)
    try:
        img.colorspace_settings.name = colorspace
    except Exception:
        pass
    img_cache[cache_key] = img
    return img


def _build_special_material(
    mat: bpy.types.Material,
    spec: dict[str, Any],
    sockets: dict[str, Any],
    tex_dir: Path,
    img_cache: dict[str, bpy.types.Image],
) -> bool:
    """Build specialized DXIL-verified HFW machine shaders (braided muscles, glowing fibers, eyes, cores, glass, VFX)."""
    kind = spec.get("kind", "")
    nt = mat.node_tree
    out_node = nt.nodes.new("ShaderNodeOutputMaterial")
    out_node.location = (720, 0)

    if kind == "muscle_cables":
        # Class A: Braided mechanical synthetic muscles (Guerrilla shader 8260:1, e.g. Clawstrider 5540:23 / 5540:35)
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.location = (380, 0)
        bsdf.inputs["Base Color"].default_value = (0.014, 0.015, 0.018, 1.0)
        bsdf.inputs["Metallic"].default_value = 0.25
        bsdf.inputs["Roughness"].default_value = 0.42
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.55
        nt.links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])

        uv0 = nt.nodes.new("ShaderNodeTexCoord")
        uv0.location = (-1380, 0)

        def _make_scaled_uv(sx: float, sy: float, y_pos: float, lbl: str):
            mp = nt.nodes.new("ShaderNodeMapping")
            mp.label = lbl
            mp.inputs["Scale"].default_value = (sx, sy, 1.0)
            mp.location = (-1140, y_pos)
            nt.links.new(uv0.outputs["UV"], mp.inputs["Vector"])
            return mp.outputs["Vector"]

        mc_grp = nt.nodes.new("ShaderNodeGroup")
        mc_grp.node_tree = _ensure_muscle_cables_group()
        mc_grp.label = "HFW Braided Muscle Glow (8260:1)"
        mc_grp.location = (20, 220)
        gc = spec.get("glow_color", [0.95, 0.28, 0.02])
        ac = spec.get("alert_color", [0.95, 0.08, 0.02])
        mc_grp.inputs["Glow Color"].default_value = (gc[0], gc[1], gc[2], 1.0)
        mc_grp.inputs["Alert Color"].default_value = (ac[0], ac[1], ac[2], 1.0)
        base_str = float(spec.get("glow_strength", 6.0))
        mc_grp.inputs["Glow Strength"].default_value = base_str

        # Link to unified HFW_MachineShaderControl so Muscle Color & Glow Scale update ALL muscles in unison!
        mctrl = nt.nodes.new("ShaderNodeGroup")
        mctrl.node_tree = _ensure_machine_shader_control_group()
        mctrl.label = "HFW Machine Shader Control"
        mctrl.location = (-360, 560)
        nt.links.new(mctrl.outputs["Muscle Color"], mc_grp.inputs["Glow Color"])
        nt.links.new(mctrl.outputs["Eye Alert Color"], mc_grp.inputs["Alert Color"])
        str_mul = nt.nodes.new("ShaderNodeMath")
        str_mul.operation = "MULTIPLY"
        str_mul.inputs[0].default_value = base_str
        str_mul.location = (-120, 420)
        nt.links.new(mctrl.outputs["Muscle Glow Scale"], str_mul.inputs[1])
        nt.links.new(str_mul.outputs["Value"], mc_grp.inputs["Glow Strength"])

        # 4019:1 (braided interlocking muscle normal R,G + crevice glow B) at UV * (6, 2)
        braid_img = _load_image(tex_dir, spec.get("braid_file", ""), "Non-Color", img_cache)
        if braid_img is not None:
            t_br = nt.nodes.new("ShaderNodeTexImage")
            t_br.image = braid_img
            t_br.label = "HFW Muscle Braid 4019:1"
            t_br.location = (-860, 220)
            nt.links.new(_make_scaled_uv(6.0, 2.0, 220, "UV * (6,2)"), t_br.inputs["Vector"])

            sep_br = nt.nodes.new("ShaderNodeSeparateColor")
            sep_br.location = (-540, 220)
            nt.links.new(t_br.outputs["Color"], sep_br.inputs["Color"])
            nt.links.new(sep_br.outputs["Blue"], mc_grp.inputs["Braid Mask"])

            nrm_rec = nt.nodes.new("ShaderNodeGroup")
            nrm_rec.node_tree = _ensure_normal_reconstruct_z_group()
            nrm_rec.location = (-240, -180)
            nt.links.new(sep_br.outputs["Red"], nrm_rec.inputs["Normal X"])
            nt.links.new(sep_br.outputs["Green"], nrm_rec.inputs["Normal Y"])

            nrm_map = nt.nodes.new("ShaderNodeNormalMap")
            nrm_map.location = (60, -180)
            nrm_map.inputs["Strength"].default_value = 1.0
            nt.links.new(nrm_rec.outputs["Normal RGB"], nrm_map.inputs["Color"])
            nt.links.new(nrm_map.outputs["Normal"], bsdf.inputs["Normal"])

        # 5645:0 (scrolling spark/pulse streaks) at UV * (6, 3) + animated (0, Time * AnimSpeed * 0.65)!
        pulse_img = _load_image(tex_dir, spec.get("pulse_file", ""), "Non-Color", img_cache)
        if pulse_img is not None:
            t_drv = _add_time_driver_node(nt, (-1720, -140), "HFW Muscle Pulse Time")
            t_spd = nt.nodes.new("ShaderNodeMath")
            t_spd.operation = "MULTIPLY"
            t_spd.location = (-1540, -140)
            nt.links.new(t_drv.outputs[0], t_spd.inputs[0])
            nt.links.new(mctrl.outputs["Animation Speed"], t_spd.inputs[1])
            t_scr = nt.nodes.new("ShaderNodeMath")
            t_scr.operation = "MULTIPLY"
            t_scr.inputs[1].default_value = -0.65
            t_scr.location = (-1340, -140)
            nt.links.new(t_spd.outputs["Value"], t_scr.inputs[0])

            comb_off = nt.nodes.new("ShaderNodeCombineXYZ")
            comb_off.location = (-1180, -160)
            nt.links.new(t_scr.outputs["Value"], comb_off.inputs["Y"])

            mp_p = nt.nodes.new("ShaderNodeMapping")
            mp_p.label = "UV * (6,3) + Time Scroll"
            mp_p.inputs["Scale"].default_value = (6.0, 3.0, 1.0)
            mp_p.location = (-1000, -80)
            nt.links.new(uv0.outputs["UV"], mp_p.inputs["Vector"])
            nt.links.new(comb_off.outputs["Vector"], mp_p.inputs["Location"])

            t_pl = nt.nodes.new("ShaderNodeTexImage")
            t_pl.image = pulse_img
            t_pl.label = "HFW Muscle Pulse 5645:0"
            t_pl.location = (-740, -80)
            nt.links.new(mp_p.outputs["Vector"], t_pl.inputs["Vector"])
            nt.links.new(t_pl.outputs["Color"], mc_grp.inputs["Pulse Mask"])

        # 4019:0 (circuit/energy veins in G) at UV * (3, 1)
        vein_img = _load_image(tex_dir, spec.get("veins_file", ""), "Non-Color", img_cache)
        if vein_img is not None:
            t_vn = nt.nodes.new("ShaderNodeTexImage")
            t_vn.image = vein_img
            t_vn.label = "HFW Muscle Veins 4019:0"
            t_vn.location = (-860, -380)
            nt.links.new(_make_scaled_uv(3.0, 1.0, -380, "UV * (3,1) Veins"), t_vn.inputs["Vector"])
            sep_vn = nt.nodes.new("ShaderNodeSeparateColor")
            sep_vn.location = (-540, -380)
            nt.links.new(t_vn.outputs["Color"], sep_vn.inputs["Color"])
            nt.links.new(sep_vn.outputs["Green"], mc_grp.inputs["Vein Mask"])

        if "Emission Color" in bsdf.inputs:
            nt.links.new(mc_grp.outputs["Emission Color"], bsdf.inputs["Emission Color"])
        if "Emission Strength" in bsdf.inputs:
            nt.links.new(mc_grp.outputs["Emission Strength"], bsdf.inputs["Emission Strength"])
        return True

    if kind == "muscle_fiber_overlay":
        # Class B: Translucent glowing body muscle fiber shell (19501:0 / 21395:0 / 20640:0 / 26679:0)
        _configure_blended_material(mat)
        mf_grp = nt.nodes.new("ShaderNodeGroup")
        mf_grp.node_tree = _ensure_muscle_fiber_group()
        mf_grp.label = "HFW Glowing Muscle Fibers (19501:0)"
        mf_grp.location = (220, 0)

        cc = spec.get("calm_color", [0.95, 0.28, 0.02])
        pc = spec.get("plasma_color", [0.46, 0.05, 1.0])
        ac = spec.get("alert_color", [1.0, 0.04, 0.01])
        mf_grp.inputs["Calm Color"].default_value = (cc[0], cc[1], cc[2], 1.0)
        mf_grp.inputs["Plasma Color"].default_value = (pc[0], pc[1], pc[2], 1.0)
        mf_grp.inputs["Alert Color"].default_value = (ac[0], ac[1], ac[2], 1.0)
        c_state = float(spec.get("charge_state", 0.0))
        mf_grp.inputs["Charge State"].default_value = c_state
        base_str = float(spec.get("glow_strength", 4.5))
        mf_grp.inputs["Glow Strength"].default_value = base_str
        mf_grp.inputs["Alert State"].default_value = float(spec.get("alert_state", 0.0))
        mf_grp.inputs["Fiber Density"].default_value = float(spec.get("fiber_density", 96.0))

        # Link to unified HFW_MachineShaderControl for 1-click Muscle Color, Glow Scale & Animation Speed!
        mctrl = nt.nodes.new("ShaderNodeGroup")
        mctrl.node_tree = _ensure_machine_shader_control_group()
        mctrl.label = "HFW Machine Shader Control"
        mctrl.location = (-360, 560)
        nt.links.new(mctrl.outputs["Muscle Color"], mf_grp.inputs["Calm Color"])
        nt.links.new(mctrl.outputs["Plasma Color"], mf_grp.inputs["Plasma Color"])
        nt.links.new(mctrl.outputs["Eye Alert Color"], mf_grp.inputs["Alert Color"])
        str_mul = nt.nodes.new("ShaderNodeMath")
        str_mul.operation = "MULTIPLY"
        str_mul.inputs[0].default_value = base_str
        str_mul.location = (-60, 340)
        nt.links.new(mctrl.outputs["Muscle Glow Scale"], str_mul.inputs[1])
        nt.links.new(str_mul.outputs["Value"], mf_grp.inputs["Glow Strength"])
        if c_state >= 0.5:
            # Link Slaughterspine's optical muscle fibers to `HFW_PlasmaChargeControl`!
            pctrl = nt.nodes.new("ShaderNodeGroup")
            pctrl.node_tree = _ensure_plasma_charge_control_group()
            pctrl.label = "Slaughterspine Plasma Charge"
            pctrl.location = (-120, 480)
            nt.links.new(pctrl.outputs["Plasma Charge"], mf_grp.inputs["Charge State"])

        # Bind UV0 to existing NMT_AO_MSKA texture node(s) so Mask_Alpha uses UV0!
        if "Mask_Alpha" in sockets:
            uv0_n = nt.nodes.new("ShaderNodeUVMap")
            uv0_n.uv_map = "UV0"
            uv0_n.location = (-1180, 320)
            for node in nt.nodes:
                if node.type == "TEX_IMAGE":
                    nt.links.new(uv0_n.outputs["UV"], node.inputs["Vector"])
            nt.links.new(sockets["Mask_Alpha"], mf_grp.inputs["Mask Alpha (UV0)"])

        # Sample 5147:0 energy wave ramp on UV1 * (6, 12) + animated (0, Time * AnimSpeed * 0.50) (DXIL L236-251, L290, L318)
        ramp_img = _load_image(tex_dir, spec.get("ramp_file", ""), "Non-Color", img_cache)
        if ramp_img is not None:
            t_drv = _add_time_driver_node(nt, (-1720, -340), "HFW Fiber Wave Time")
            t_spd = nt.nodes.new("ShaderNodeMath")
            t_spd.operation = "MULTIPLY"
            t_spd.location = (-1520, -340)
            nt.links.new(t_drv.outputs[0], t_spd.inputs[0])
            nt.links.new(mctrl.outputs["Animation Speed"], t_spd.inputs[1])
            t_scr = nt.nodes.new("ShaderNodeMath")
            t_scr.operation = "MULTIPLY"
            t_scr.inputs[1].default_value = -0.50
            t_scr.location = (-1320, -340)
            nt.links.new(t_spd.outputs["Value"], t_scr.inputs[0])

            comb_off = nt.nodes.new("ShaderNodeCombineXYZ")
            comb_off.location = (-1140, -340)
            nt.links.new(t_scr.outputs["Value"], comb_off.inputs["Y"])

            uv1_n = nt.nodes.new("ShaderNodeUVMap")
            uv1_n.uv_map = "UV1"
            uv1_n.location = (-1180, -140)
            mp = nt.nodes.new("ShaderNodeMapping")
            mp.label = "Fiber Wave UV1 * (6,12) + Time"
            mp.inputs["Scale"].default_value = (6.0, 12.0, 1.0)
            mp.location = (-940, -180)
            nt.links.new(uv1_n.outputs["UV"], mp.inputs["Vector"])
            nt.links.new(comb_off.outputs["Vector"], mp.inputs["Location"])

            t_rmp = nt.nodes.new("ShaderNodeTexImage")
            t_rmp.image = ramp_img
            t_rmp.label = "HFW Muscle Energy Ramp 5147:0"
            t_rmp.location = (-660, -180)
            nt.links.new(mp.outputs["Vector"], t_rmp.inputs["Vector"])

            sep_r = nt.nodes.new("ShaderNodeSeparateColor")
            sep_r.location = (-360, -180)
            nt.links.new(t_rmp.outputs["Color"], sep_r.inputs["Color"])

            # Exact DXIL L323-338: combine subtle traveling wave (`0.38 * B`) + crisp bright spark pulses (`A`)
            # mapped across authentic dark-to-bright dynamic range (`0.08..1.0`, NOT solid 0.55 floor!)
            b_scl = nt.nodes.new("ShaderNodeMath")
            b_scl.operation = "MULTIPLY"
            b_scl.inputs[1].default_value = 0.42
            b_scl.location = (-260, -100)
            nt.links.new(sep_r.outputs["Blue"], b_scl.inputs[0])

            wave_add = nt.nodes.new("ShaderNodeMath")
            wave_add.operation = "ADD"
            wave_add.use_clamp = True
            wave_add.location = (-100, -160)
            nt.links.new(b_scl.outputs["Value"], wave_add.inputs[0])
            nt.links.new(t_rmp.outputs["Alpha"], wave_add.inputs[1])

            wave_map = nt.nodes.new("ShaderNodeMapRange")
            wave_map.clamp = True
            wave_map.inputs["From Min"].default_value = 0.0
            wave_map.inputs["From Max"].default_value = 1.0
            wave_map.inputs["To Min"].default_value = 0.12
            wave_map.inputs["To Max"].default_value = 1.0
            wave_map.location = (60, -160)
            nt.links.new(wave_add.outputs["Value"], wave_map.inputs["Value"])
            nt.links.new(wave_map.outputs["Result"], mf_grp.inputs["Energy Wave (5147)"])

        nt.links.new(mf_grp.outputs["Shader"], out_node.inputs["Surface"])
        return True

    if kind == "machine_core_emissive":
        # Class C: Machine glowing Brain / Heart / Neck coils (32906:0 / 1348:90: 1 Hz heartbeat + scrolling energy wave!)
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.location = (360, 0)
        bsdf.inputs["Roughness"].default_value = 0.22
        bsdf.inputs["Metallic"].default_value = 0.35
        nt.links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])

        t_drv = _add_time_driver_node(nt, (-1180, 320), "HFW Core Heartbeat Time")
        # 1 Hz DXIL heartbeat: 0.55 + 0.65 * abs(sin(Time * pi))
        t_pi = nt.nodes.new("ShaderNodeMath")
        t_pi.operation = "MULTIPLY"
        t_pi.inputs[1].default_value = 3.14159265
        t_pi.location = (-960, 420)
        nt.links.new(t_drv.outputs[0], t_pi.inputs[0])

        s_hb = nt.nodes.new("ShaderNodeMath")
        s_hb.operation = "SINE"
        s_hb.location = (-780, 420)
        nt.links.new(t_pi.outputs["Value"], s_hb.inputs[0])

        a_hb = nt.nodes.new("ShaderNodeMath")
        a_hb.operation = "ABSOLUTE"
        a_hb.location = (-600, 420)
        nt.links.new(s_hb.outputs["Value"], a_hb.inputs[0])

        hb_mod = nt.nodes.new("ShaderNodeMath")
        hb_mod.operation = "MULTIPLY_ADD"
        hb_mod.inputs[1].default_value = 0.65
        hb_mod.inputs[2].default_value = 0.55
        hb_mod.location = (-400, 420)
        nt.links.new(a_hb.outputs["Value"], hb_mod.inputs[0])

        hb_str = nt.nodes.new("ShaderNodeMath")
        hb_str.operation = "MULTIPLY"
        hb_str.inputs[1].default_value = float(spec.get("glow_strength", 5.5))
        hb_str.location = (120, 340)
        nt.links.new(hb_mod.outputs["Value"], hb_str.inputs[0])

        # Scrolling UV for inner energy coils
        tc_c = nt.nodes.new("ShaderNodeTexCoord")
        tc_c.location = (-1180, 120)
        t_sc = nt.nodes.new("ShaderNodeMath")
        t_sc.operation = "MULTIPLY"
        t_sc.inputs[1].default_value = 0.22
        t_sc.location = (-960, 220)
        nt.links.new(t_drv.outputs[0], t_sc.inputs[0])
        c_off = nt.nodes.new("ShaderNodeCombineXYZ")
        c_off.location = (-780, 220)
        nt.links.new(t_sc.outputs["Value"], c_off.inputs["Y"])
        mp_c = nt.nodes.new("ShaderNodeMapping")
        mp_c.label = "Core Energy Scroll"
        mp_c.location = (-600, 140)
        nt.links.new(tc_c.outputs["UV"], mp_c.inputs["Vector"])
        nt.links.new(c_off.outputs["Vector"], mp_c.inputs["Location"])

        gc = spec.get("glow_color", [0.08, 0.55, 1.0])
        col_img = _load_image(tex_dir, spec.get("color_file", ""), "sRGB", img_cache)
        if col_img is not None:
            t_col = nt.nodes.new("ShaderNodeTexImage")
            t_col.image = col_img
            t_col.label = "HFW Core Color 19456:0"
            t_col.location = (-360, 160)
            nt.links.new(mp_c.outputs["Vector"], t_col.inputs["Vector"])
            nt.links.new(t_col.outputs["Color"], bsdf.inputs["Base Color"])

            mix_em = nt.nodes.new("ShaderNodeMix")
            mix_em.data_type = "RGBA"
            mix_em.blend_type = "MULTIPLY"
            mix_em.inputs["Factor"].default_value = 1.0
            mix_em.inputs[7].default_value = (gc[0], gc[1], gc[2], 1.0)
            mix_em.location = (-40, 120)
            nt.links.new(t_col.outputs["Color"], mix_em.inputs[6])
            if "Emission Color" in bsdf.inputs:
                nt.links.new(mix_em.outputs[2], bsdf.inputs["Emission Color"])
            if "Emission Strength" in bsdf.inputs:
                nt.links.new(hb_str.outputs["Value"], bsdf.inputs["Emission Strength"])

        nrm_img = _load_image(tex_dir, spec.get("normal_file", ""), "Non-Color", img_cache)
        if nrm_img is not None:
            t_nrm = nt.nodes.new("ShaderNodeTexImage")
            t_nrm.image = nrm_img
            t_nrm.label = "HFW Core Normal 19456:1"
            t_nrm.location = (-520, -220)
            sep_n = nt.nodes.new("ShaderNodeSeparateColor")
            sep_n.location = (-240, -220)
            nt.links.new(t_nrm.outputs["Color"], sep_n.inputs["Color"])
            nrec = nt.nodes.new("ShaderNodeGroup")
            nrec.node_tree = _ensure_normal_reconstruct_z_group()
            nrec.location = (-40, -220)
            nt.links.new(sep_n.outputs["Red"], nrec.inputs["Normal X"])
            nt.links.new(sep_n.outputs["Green"], nrec.inputs["Normal Y"])
            nmap = nt.nodes.new("ShaderNodeNormalMap")
            nmap.location = (160, -220)
            nt.links.new(nrec.outputs["Normal RGB"], nmap.inputs["Color"])
            nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
        return True

    if kind == "machine_eye_glow":
        # Class F: Inner glowing machine eye optical sensor (4628:0) synced to HFW_MachineShaderControl!
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.location = (360, 0)
        bsdf.inputs["Base Color"].default_value = (0.01, 0.02, 0.04, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.04
        if "Coat Weight" in bsdf.inputs:
            bsdf.inputs["Coat Weight"].default_value = 1.0
            bsdf.inputs["Coat Roughness"].default_value = 0.02
        t_drv = _add_time_driver_node(nt, (-680, -140), "HFW Eye Pulse Time")
        mctrl = nt.nodes.new("ShaderNodeGroup")
        mctrl.node_tree = _ensure_machine_shader_control_group()
        mctrl.label = "HFW Machine Shader Control"
        mctrl.location = (-680, 260)
        el_grp = nt.nodes.new("ShaderNodeGroup")
        el_grp.node_tree = _ensure_eye_lens_group(None)
        el_grp.label = "HFW Machine Eye Glow (4628:0)"
        el_grp.location = (-80, -40)
        base_eye_str = float(spec.get("glow_strength", 9.0))
        nt.links.new(t_drv.outputs[0], el_grp.inputs["Time"])
        nt.links.new(mctrl.outputs["Eye Color"], el_grp.inputs["Calm Color"])
        nt.links.new(mctrl.outputs["Eye Alert Color"], el_grp.inputs["Alert Color"])
        nt.links.new(mctrl.outputs["Eye Alert State"], el_grp.inputs["Alert State"])
        nt.links.new(mctrl.outputs["Animation Speed"], el_grp.inputs["Animate Speed"])
        es_mul = nt.nodes.new("ShaderNodeMath")
        es_mul.operation = "MULTIPLY"
        es_mul.inputs[0].default_value = base_eye_str
        es_mul.location = (-320, -40)
        nt.links.new(mctrl.outputs["Eye Glow Scale"], es_mul.inputs[1])
        nt.links.new(es_mul.outputs["Value"], el_grp.inputs["Glow Strength"])
        if "Emission Color" in bsdf.inputs:
            nt.links.new(el_grp.outputs["Emission Color"], bsdf.inputs["Emission Color"])
        if "Emission Strength" in bsdf.inputs:
            nt.links.new(el_grp.outputs["Emission Strength"], bsdf.inputs["Emission Strength"])
        nt.links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])
        return True

    if kind == "canister_glow_core":
        # Class G: Elemental canister glowing pressurized liquid core (21080:1) with wet glossy depth,
        # Fresnel subsurface edge darkening, dual-scrolling 107:0 fluid noise & HFW_MachineShaderControl!
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.location = (440, 0)
        gc = spec.get("glow_color", [0.05, 0.48, 1.0])
        bsdf.inputs["Roughness"].default_value = 0.08
        if "Coat Weight" in bsdf.inputs:
            bsdf.inputs["Coat Weight"].default_value = 0.90
            bsdf.inputs["Coat Roughness"].default_value = 0.04
        if "Subsurface Weight" in bsdf.inputs:
            bsdf.inputs["Subsurface Weight"].default_value = 0.40

        mctrl = nt.nodes.new("ShaderNodeGroup")
        mctrl.node_tree = _ensure_machine_shader_control_group()
        mctrl.label = "HFW Machine Shader Control"
        mctrl.location = (-1080, 480)

        # Native elemental color (with optional global Liquid Override mix!)
        liq_col = nt.nodes.new("ShaderNodeMix")
        liq_col.data_type = "RGBA"
        liq_col.label = "Native Liquid <-> Override"
        liq_col.inputs[6].default_value = (gc[0], gc[1], gc[2], 1.0)
        liq_col.location = (-440, 440)
        nt.links.new(mctrl.outputs["Liquid Override Mix"], liq_col.inputs["Factor"])
        nt.links.new(mctrl.outputs["Liquid Override Color"], liq_col.inputs[7])

        # Fresnel liquid depth rim (deeper saturated rim -> luminous glowing fluid core!)
        rim_hsv = nt.nodes.new("ShaderNodeHueSaturation")
        rim_hsv.inputs["Hue"].default_value = 0.52
        rim_hsv.inputs["Saturation"].default_value = 1.25
        rim_hsv.inputs["Value"].default_value = 0.40
        rim_hsv.location = (-200, 320)
        nt.links.new(liq_col.outputs[2], rim_hsv.inputs["Color"])

        lw = nt.nodes.new("ShaderNodeLayerWeight")
        lw.inputs["Blend"].default_value = 0.40
        lw.location = (-200, 560)

        mix_depth = nt.nodes.new("ShaderNodeMix")
        mix_depth.data_type = "RGBA"
        mix_depth.label = "Liquid Core <-> Rim Depth"
        mix_depth.location = (60, 420)
        nt.links.new(lw.outputs["Facing"], mix_depth.inputs["Factor"])
        nt.links.new(liq_col.outputs[2], mix_depth.inputs[6])
        nt.links.new(rim_hsv.outputs["Color"], mix_depth.inputs[7])

        bc_dark = nt.nodes.new("ShaderNodeHueSaturation")
        bc_dark.inputs["Value"].default_value = 0.30
        bc_dark.location = (180, 220)
        nt.links.new(liq_col.outputs[2], bc_dark.inputs["Color"])
        nt.links.new(bc_dark.outputs["Color"], bsdf.inputs["Base Color"])

        t_drv = _add_time_driver_node(nt, (-1260, -140), "HFW Canister Core Time")
        t_spd = nt.nodes.new("ShaderNodeMath")
        t_spd.operation = "MULTIPLY"
        t_spd.location = (-1040, -140)
        nt.links.new(t_drv.outputs[0], t_spd.inputs[0])
        nt.links.new(mctrl.outputs["Animation Speed"], t_spd.inputs[1])

        t_f = nt.nodes.new("ShaderNodeMath")
        t_f.operation = "MULTIPLY"
        t_f.inputs[1].default_value = 3.2
        t_f.location = (-820, -140)
        nt.links.new(t_spd.outputs["Value"], t_f.inputs[0])

        s_p = nt.nodes.new("ShaderNodeMath")
        s_p.operation = "SINE"
        s_p.location = (-620, -140)
        nt.links.new(t_f.outputs["Value"], s_p.inputs[0])

        pulse = nt.nodes.new("ShaderNodeMath")
        pulse.operation = "MULTIPLY_ADD"
        pulse.inputs[1].default_value = 0.18
        pulse.inputs[2].default_value = 0.85
        pulse.location = (-420, -140)
        nt.links.new(s_p.outputs["Value"], pulse.inputs[0])

        noise_img = _load_image(tex_dir, spec.get("noise_file", ""), "Non-Color", img_cache)
        if noise_img is not None:
            tc_n = nt.nodes.new("ShaderNodeTexCoord")
            tc_n.location = (-1060, 120)
            t_scr = nt.nodes.new("ShaderNodeMath")
            t_scr.operation = "MULTIPLY"
            t_scr.inputs[1].default_value = 0.30
            t_scr.location = (-820, 40)
            nt.links.new(t_spd.outputs["Value"], t_scr.inputs[0])
            c_off = nt.nodes.new("ShaderNodeCombineXYZ")
            c_off.location = (-620, 40)
            nt.links.new(t_scr.outputs["Value"], c_off.inputs["Y"])
            mp_n = nt.nodes.new("ShaderNodeMapping")
            mp_n.inputs["Scale"].default_value = (2.2, 2.2, 1.0)
            mp_n.location = (-440, 120)
            nt.links.new(tc_n.outputs["UV"], mp_n.inputs["Vector"])
            nt.links.new(c_off.outputs["Vector"], mp_n.inputs["Location"])
            t_n = nt.nodes.new("ShaderNodeTexImage")
            t_n.image = noise_img
            t_n.label = "HFW Canister Core Fluid Noise (107:0)"
            t_n.location = (-200, 120)
            nt.links.new(mp_n.outputs["Vector"], t_n.inputs["Vector"])
            n_mod = nt.nodes.new("ShaderNodeMath")
            n_mod.operation = "MULTIPLY_ADD"
            n_mod.inputs[1].default_value = 0.72
            n_mod.inputs[2].default_value = 0.52
            n_mod.location = (60, 80)
            nt.links.new(t_n.outputs["Color"], n_mod.inputs[0])
            comb_m = nt.nodes.new("ShaderNodeMath")
            comb_m.operation = "MULTIPLY"
            comb_m.location = (200, -40)
            nt.links.new(n_mod.outputs["Value"], comb_m.inputs[0])
            nt.links.new(pulse.outputs["Value"], comb_m.inputs[1])
            em_base = comb_m.outputs["Value"]
        else:
            em_base = pulse.outputs["Value"]

        em_str0 = nt.nodes.new("ShaderNodeMath")
        em_str0.operation = "MULTIPLY"
        em_str0.inputs[1].default_value = float(spec.get("glow_strength", 6.8))
        em_str0.location = (140, -220)
        nt.links.new(em_base, em_str0.inputs[0])

        em_str = nt.nodes.new("ShaderNodeMath")
        em_str.operation = "MULTIPLY"
        em_str.location = (280, -220)
        nt.links.new(em_str0.outputs["Value"], em_str.inputs[0])
        nt.links.new(mctrl.outputs["Liquid Glow Scale"], em_str.inputs[1])

        if "Emission Color" in bsdf.inputs:
            nt.links.new(mix_depth.outputs[2], bsdf.inputs["Emission Color"])
        if "Emission Strength" in bsdf.inputs:
            nt.links.new(em_str.outputs["Value"], bsdf.inputs["Emission Strength"])
        nt.links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])
        return True

    if kind == "canister_glass":
        # Class E: Physical Fresnel Glass Canister Shell (41249:0) with Cycles/Eevee Shadow-Ray Bypass!
        # Center stays crystal-clear so inner glowing liquid shines vibrantly, while curved edges catch glossy glass rims!
        _configure_blended_material(mat)
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.location = (180, 120)
        tc = spec.get("tint_color", [0.80, 0.92, 0.91])
        bsdf.inputs["Base Color"].default_value = (tc[0], tc[1], tc[2], 1.0)
        bsdf.inputs["Roughness"].default_value = 0.05
        if "IOR" in bsdf.inputs:
            bsdf.inputs["IOR"].default_value = float(spec.get("ior", 1.45))
        if "Transmission Weight" in bsdf.inputs:
            bsdf.inputs["Transmission Weight"].default_value = 0.90
        if "Coat Weight" in bsdf.inputs:
            bsdf.inputs["Coat Weight"].default_value = 1.0
            bsdf.inputs["Coat Roughness"].default_value = 0.03

        lw = nt.nodes.new("ShaderNodeLayerWeight")
        lw.inputs["Blend"].default_value = 0.30
        lw.location = (-360, 40)
        al_mr = nt.nodes.new("ShaderNodeMapRange")
        al_mr.clamp = True
        al_mr.label = "Fresnel Glass Rim Alpha"
        al_mr.inputs["From Min"].default_value = 0.0
        al_mr.inputs["From Max"].default_value = 1.0
        al_mr.inputs["To Min"].default_value = 0.14
        al_mr.inputs["To Max"].default_value = 0.68
        al_mr.location = (-120, 40)
        nt.links.new(lw.outputs["Fresnel"], al_mr.inputs["Value"])
        nt.links.new(al_mr.outputs["Result"], bsdf.inputs["Alpha"])

        # Shadow-Ray transparent bypass so Cycles never casts dark opaque shadows onto inner liquid core!
        tr_bsdf = nt.nodes.new("ShaderNodeBsdfTransparent")
        tr_bsdf.inputs["Color"].default_value = (0.95, 0.98, 1.0, 1.0)
        tr_bsdf.location = (180, -240)
        lp = nt.nodes.new("ShaderNodeLightPath")
        lp.location = (180, -400)
        mix_sh = nt.nodes.new("ShaderNodeMixShader")
        mix_sh.label = "Glass Shadow Bypass"
        mix_sh.location = (480, 0)
        nt.links.new(lp.outputs["Is Shadow Ray"], mix_sh.inputs["Fac"])
        nt.links.new(bsdf.outputs["BSDF"], mix_sh.inputs[1])
        nt.links.new(tr_bsdf.outputs["BSDF"], mix_sh.inputs[2])
        nt.links.new(mix_sh.outputs["Shader"], out_node.inputs["Surface"])
        return True

    if kind == "vfx_overlay_shell":
        # Class D: Full-body UV-less Focus/Stealth VFX overlay shell (37840:0 / 38845:0) -> transparent by default!
        _configure_blended_material(mat)
        tr_bsdf = nt.nodes.new("ShaderNodeBsdfTransparent")
        tr_bsdf.location = (120, 120)
        em_bsdf = nt.nodes.new("ShaderNodeEmission")
        gc = spec.get("glow_color", [0.05, 0.65, 1.0])
        em_bsdf.inputs["Color"].default_value = (gc[0], gc[1], gc[2], 1.0)
        em_bsdf.inputs["Strength"].default_value = 3.0
        em_bsdf.location = (120, -120)
        mix_sh = nt.nodes.new("ShaderNodeMixShader")
        mix_sh.label = "VFX Overlay Opacity"
        mix_sh.inputs["Fac"].default_value = float(spec.get("opacity", 0.0))
        mix_sh.location = (420, 0)
        nt.links.new(tr_bsdf.outputs["BSDF"], mix_sh.inputs[1])
        nt.links.new(em_bsdf.outputs["Emission"], mix_sh.inputs[2])
        nt.links.new(mix_sh.outputs["Shader"], out_node.inputs["Surface"])
        return True

    return False


_ACTIVE_CHAR_CTRL_NAME: str = "HFW_CharacterControl"


def _ensure_character_control_group(
    controls: dict[str, Any] | None = None,
    ng_name: str | None = None,
) -> bpy.types.ShaderNodeTree:
    """Create/update `HFW_CharacterControl` node group driving Head + Body skin tone, SSS, Peach Fuzz, Iris & Hair color."""
    ng_name = ng_name or _ACTIVE_CHAR_CTRL_NAME
    c = controls or {}
    st = c.get("skin_tone_linear") or [0.453, 0.246, 0.209]
    hc = c.get("hair_color_linear") or [0.045, 0.032, 0.025]
    ic = c.get("iris_color_linear") or [0.106, 0.199, 0.061]
    pf = float(c.get("peach_fuzz_intensity", 0.65))

    existing = bpy.data.node_groups.get(ng_name)
    if existing is not None:
        if controls:
            for n in existing.nodes:
                if n.label == "Skin Tone" and n.type == "RGB" and c.get("skin_tone_linear"):
                    n.outputs[0].default_value = (float(st[0]), float(st[1]), float(st[2]), 1.0)
                elif n.label == "Hair Color" and n.type == "RGB" and c.get("hair_color_linear"):
                    n.outputs[0].default_value = (float(hc[0]), float(hc[1]), float(hc[2]), 1.0)
                elif n.label == "Iris Color" and n.type == "RGB" and c.get("iris_color_linear"):
                    n.outputs[0].default_value = (float(ic[0]), float(ic[1]), float(ic[2]), 1.0)
                elif n.label == "Peach Fuzz Intensity" and n.type == "VALUE" and "peach_fuzz_intensity" in c:
                    n.outputs[0].default_value = pf
        return existing

    ng = bpy.data.node_groups.new(ng_name, "ShaderNodeTree")
    ng.interface.new_socket(name="Skin Tone", in_out="OUTPUT", socket_type="NodeSocketColor")
    ng.interface.new_socket(name="Skin Brightness", in_out="OUTPUT", socket_type="NodeSocketFloat")
    ng.interface.new_socket(name="Subsurface Weight", in_out="OUTPUT", socket_type="NodeSocketFloat")
    ng.interface.new_socket(name="Hair Color", in_out="OUTPUT", socket_type="NodeSocketColor")
    ng.interface.new_socket(name="Iris Color", in_out="OUTPUT", socket_type="NodeSocketColor")
    ng.interface.new_socket(name="Peach Fuzz Intensity", in_out="OUTPUT", socket_type="NodeSocketFloat")

    n_out = ng.nodes.new("NodeGroupOutput")
    n_out.location = (220, 0)

    rgb = ng.nodes.new("ShaderNodeRGB")
    rgb.label = "Skin Tone"
    rgb.outputs[0].default_value = (float(st[0]), float(st[1]), float(st[2]), 1.0)
    rgb.location = (-200, 180)
    ng.links.new(rgb.outputs[0], n_out.inputs["Skin Tone"])

    br = ng.nodes.new("ShaderNodeValue")
    br.label = "Skin Brightness"
    br.outputs[0].default_value = 1.0
    br.location = (-200, 20)
    ng.links.new(br.outputs[0], n_out.inputs["Skin Brightness"])

    sss = ng.nodes.new("ShaderNodeValue")
    sss.label = "Subsurface Weight"
    sss.outputs[0].default_value = 0.32
    sss.location = (-200, -100)
    ng.links.new(sss.outputs[0], n_out.inputs["Subsurface Weight"])

    hrgb = ng.nodes.new("ShaderNodeRGB")
    hrgb.label = "Hair Color"
    hrgb.outputs[0].default_value = (float(hc[0]), float(hc[1]), float(hc[2]), 1.0)
    hrgb.location = (-200, -260)
    ng.links.new(hrgb.outputs[0], n_out.inputs["Hair Color"])

    irgb = ng.nodes.new("ShaderNodeRGB")
    irgb.label = "Iris Color"
    irgb.outputs[0].default_value = (float(ic[0]), float(ic[1]), float(ic[2]), 1.0)
    irgb.location = (-200, -440)
    ng.links.new(irgb.outputs[0], n_out.inputs["Iris Color"])

    pfv = ng.nodes.new("ShaderNodeValue")
    pfv.label = "Peach Fuzz Intensity"
    pfv.outputs[0].default_value = pf
    pfv.location = (-200, -600)
    ng.links.new(pfv.outputs[0], n_out.inputs["Peach Fuzz Intensity"])

    return ng


def _build_layered_outfit_material(
    mat: bpy.types.Material,
    layered: dict[str, Any],
    tex_dir: Path,
    img_cache: dict[str, bpy.types.Image],
) -> bool:
    """Build Guerrilla's exact DXIL 8-Layer Character Outfit Shader OR Building Block Tileable Shader.

    DXIL-verified exact 8-corner `MSK.RGB` mapping (`75185:637` L1809..1874 & `74:0` LUT):
      0: Black   (0,0,0)    4: Cyan    (0,1,1)
      1: Red     (1,0,0)    5: Magenta (1,0,1)
      2: Green   (0,1,0)    6: Yellow  (1,1,0)
      3: Blue    (0,0,1)    7: White   (1,1,1)

    In `outfit_dxil` / `outfit_fallback` mode:
      - `Shaders_HRZ2/Materials/*_Base_CLR` maps are `0.5`-centered **Linear** micro-detail maps (`Non-Color`),
        combined with Guerrilla's exact DXIL formula `pow(2.0 * DetailTex, 2.2) * DyeColor_linear` so `0.50`
        neutral texels reproduce the exact authored dye color without darkening!
    In `building_block` mode:
      - `textures/building_block_tileables/*` maps are direct full-color `sRGB` concrete/wood/metal/moss maps
        blended directly by `MSK.RGB` at 4x tiling!
    """
    layers = layered.get("layers") or []
    if not layers:
        return False

    mode = str(layered.get("mode", "outfit_dxil"))
    is_bb = mode == "building_block"

    nt = mat.node_tree
    nt.nodes.clear()
    mat["hfw_shader_kind"] = "building_block_layered" if is_bb else "character_outfit_layered"

    out_node = nt.nodes.new("ShaderNodeOutputMaterial")
    out_node.location = (1060, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (680, 0)
    nt.links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])

    # UV tiling (16x on Character Outfits per DXIL L294-295; 4x on Building Block Tileables!)
    uv_scale = float(layered.get("uv_scale", 4.0 if is_bb else 16.0))
    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.location = (-1860, 260)
    uv_tile = nt.nodes.new("ShaderNodeVectorMath")
    uv_tile.operation = "SCALE"
    uv_tile.inputs["Scale"].default_value = uv_scale
    uv_tile.label = f"HFW Micro-Tile ({uv_scale:.0f}x)"
    uv_tile.location = (-1620, 260)
    nt.links.new(tc.outputs["UV"], uv_tile.inputs[0])

    tribe_hint = str(layered.get("tribe_hint", "")).lower()

    def _layer_fallback_tint_linear(lname: str, lcat: str, l_idx: int) -> tuple[float, float, float]:
        nl = f"{lcat}/{lname}".lower()
        if "bronze" in nl or "brass" in nl or "gold" in nl:
            srgb = (0.76, 0.56, 0.28)
        elif "copper" in nl:
            srgb = (0.78, 0.44, 0.28)
        elif "robotplate" in nl or "iron" in nl or "metal" in nl or "steel" in nl or "anodized" in nl:
            srgb = (0.58, 0.60, 0.62)
        elif "carbon" in nl or "rubber" in nl:
            srgb = (0.14, 0.15, 0.17)
        elif "fullgrain" in nl:
            srgb = (0.33, 0.27, 0.20)
        elif "finegrain" in nl or "nubuck" in nl:
            srgb = (0.45, 0.34, 0.24)
        elif "buckskin" in nl or "suede" in nl:
            srgb = (0.58, 0.44, 0.28)
        elif "leather" in nl:
            srgb = (0.38, 0.24, 0.18)
        elif "wood" in nl or "bark" in nl or "bone" in nl:
            srgb = (0.50, 0.38, 0.26)
        elif "shell" in nl or "nacre" in nl or "pearl" in nl:
            srgb = (0.74, 0.72, 0.70)
        elif "carja" in tribe_hint:
            srgb = (0.78, 0.60, 0.10) if l_idx % 2 == 0 else (0.55, 0.30, 0.22)
        elif "quen" in tribe_hint or "seyka" in tribe_hint:
            srgb = (0.62, 0.72, 0.78) if l_idx % 2 == 0 else (0.86, 0.62, 0.50)
        elif "tenakth" in tribe_hint:
            srgb = (0.20, 0.42, 0.52) if l_idx % 2 == 0 else (0.58, 0.22, 0.16)
        elif "utaru" in tribe_hint:
            srgb = (0.36, 0.48, 0.32) if l_idx % 2 == 0 else (0.66, 0.60, 0.46)
        elif "oseram" in tribe_hint or "erend" in tribe_hint:
            srgb = (0.58, 0.30, 0.16) if l_idx % 2 == 0 else (0.36, 0.24, 0.18)
        else:
            srgb = (0.22, 0.40, 0.57) if l_idx % 2 == 1 else (0.58, 0.54, 0.48)
        return (srgb[0] ** 2.2, srgb[1] ** 2.2, srgb[2] ** 2.2)

    # Build all 8 corner slots (0=Black, 1=Red, 2=Green, 3=Blue, 4=Cyan, 5=Magenta, 6=Yellow, 7=White)
    # Cache per-file micro-normal/smoothness nodes so shared sub-materials (e.g. Metal_Robotplate on 4 corners)
    # are sampled once instead of duplicating TextureImage nodes!
    layer_cols: list[Any] = []
    layer_micro_tex: list[Any] = []  # Neutral-0.5 expanded micro-detail (used when hybrid macro_color_file is present!)
    layer_rghs: list[Any] = []
    layer_mets: list[Any] = []
    layer_dnx: list[Any] = []
    layer_dny: list[Any] = []
    nrm_node_cache: dict[str, tuple[Any, Any, Any]] = {}  # filename -> (sep_dn.R, sep_dn.G, sep_dn.B)
    rfl_node_cache: dict[str, Any] = {}

    def _get_submat_nrm_rgh(nfile: str, y_slot: float) -> tuple[Any, Any, Any] | None:
        if not nfile:
            return None
        if nfile in nrm_node_cache:
            return nrm_node_cache[nfile]
        dn_img = _load_image(tex_dir, nfile, "Non-Color", img_cache)
        if dn_img is None:
            return None
        t_dn = nt.nodes.new("ShaderNodeTexImage")
        t_dn.image = dn_img
        t_dn.label = f"Micro NMT/GLS ({Path(nfile).stem[:22]})"
        t_dn.location = (-1380, -2100 - len(nrm_node_cache) * 260)
        nt.links.new(uv_tile.outputs["Vector"], t_dn.inputs["Vector"])
        sep_dn = nt.nodes.new("ShaderNodeSeparateColor")
        sep_dn.location = (-1100, -2100 - len(nrm_node_cache) * 260)
        nt.links.new(t_dn.outputs["Color"], sep_dn.inputs["Color"])
        res = (sep_dn.outputs["Red"], sep_dn.outputs["Green"], sep_dn.outputs["Blue"])
        nrm_node_cache[nfile] = res
        return res

    for idx, lay in enumerate(layers[:8]):
        cname = lay.get("name", f"Layer{idx}")
        ccat = lay.get("category", "")
        c_tag = lay.get("corner_name", f"L{idx}")
        clow = f"{ccat}/{cname}".lower()

        if is_bb:
            # Building block tileables ALREADY hold authentic full-color sRGB concrete/wood/moss albedo!
            c_img = _load_image(tex_dir, lay.get("color_file", ""), "sRGB", img_cache)
            if c_img is not None:
                tn = nt.nodes.new("ShaderNodeTexImage")
                tn.image = c_img
                tn.label = f"{c_tag} {cname} Albedo"
                tn.location = (-1320, 680 - idx * 280)
                nt.links.new(uv_tile.outputs["Vector"], tn.inputs["Vector"])
                layer_cols.append(tn.outputs["Color"])
                layer_micro_tex.append(tn.outputs["Color"])
            else:
                tint_n = nt.nodes.new("ShaderNodeRGB")
                tint_n.label = f"{c_tag} {cname} Color"
                tint_n.outputs[0].default_value = (0.22, 0.21, 0.19, 1.0)
                tint_n.location = (-1320, 680 - idx * 280)
                layer_cols.append(tint_n.outputs[0])
                layer_micro_tex.append(tint_n.outputs[0])
            is_met = bool(lay.get("is_metallic", False))
            layer_mets.append(0.75 if is_met else 0.0)
            layer_rghs.append(0.42 if is_met else 0.78)
            layer_dnx.append(0.5)
            layer_dny.append(0.5)
        else:
            # Character Outfit: exact DXIL authored linear RGB Dye Color * pow(2.0 * LinearDetailTex, 2.2)!
            # CRITICAL: Decima metallic armor corners store a dark diffuse (`0.04..0.09`) in `dye_color_linear`
            # while their actual metal reflectance color lives in `f0_rgb`! Using `f0_rgb` fixes pitch-black armor plates.
            is_met = bool(lay.get("is_metallic", False))
            dye_lin = lay.get("dye_color_linear")
            f0_lin = lay.get("f0_rgb")
            if is_met:
                d_mx = max(float(c) for c in dye_lin[:3]) if (dye_lin and len(dye_lin) >= 3) else 0.0
                f_mx = max(float(c) for c in f0_lin[:3]) if (f0_lin and len(f0_lin) >= 3) else 0.0
                if f_mx > max(0.18, d_mx * 1.30):
                    tint_rgb = tuple(max(0.14, min(0.96, float(c))) for c in f0_lin[:3])
                elif d_mx >= 0.16:
                    tint_rgb = (float(dye_lin[0]), float(dye_lin[1]), float(dye_lin[2]))
                else:
                    tint_rgb = _layer_fallback_tint_linear(cname, ccat, idx)
            elif dye_lin and len(dye_lin) >= 3:
                tint_rgb = (float(dye_lin[0]), float(dye_lin[1]), float(dye_lin[2]))
            else:
                tint_rgb = _layer_fallback_tint_linear(cname, ccat, idx)

            tint_n = nt.nodes.new("ShaderNodeRGB")
            tint_n.label = f"{c_tag} ({cname}) {'Metal F0' if is_met else 'Dye'}"
            tint_n.outputs[0].default_value = (tint_rgb[0], tint_rgb[1], tint_rgb[2], 1.0)
            tint_n.location = (-1380, 760 - idx * 300)

            # CRITICAL: `Shaders_HRZ2/Materials/*_CLR` has `ETexColorSpace = Linear` (`mean = 0.502`) -> load as `Non-Color`!
            c_img = _load_image(tex_dir, lay.get("color_file", ""), "Non-Color", img_cache)
            if c_img is not None:
                tn = nt.nodes.new("ShaderNodeTexImage")
                tn.image = c_img
                tn.label = f"{c_tag} {cname} Micro-Detail"
                tn.location = (-1380, 600 - idx * 300)
                nt.links.new(uv_tile.outputs["Vector"], tn.inputs["Vector"])

                # Exact DXIL neutral-0.5 micro-detail expansion: pow(2.0 * tex, 2.2) so tex=0.5 -> 1.0!
                sc2 = nt.nodes.new("ShaderNodeVectorMath")
                sc2.operation = "SCALE"
                sc2.inputs["Scale"].default_value = 2.0
                sc2.location = (-1120, 600 - idx * 300)
                nt.links.new(tn.outputs["Color"], sc2.inputs[0])

                gam = nt.nodes.new("ShaderNodeGamma")
                gam.inputs["Gamma"].default_value = 2.2
                gam.location = (-940, 600 - idx * 300)
                nt.links.new(sc2.outputs["Vector"], gam.inputs["Color"])

                # Fold micro-AO (`*_CLR_AO.A`) subtly (0.72..1.00) into detail variation for crisp weave/grain depth!
                mao_mr = nt.nodes.new("ShaderNodeMapRange")
                mao_mr.clamp = True
                mao_mr.inputs["From Min"].default_value = 0.0
                mao_mr.inputs["From Max"].default_value = 1.0
                mao_mr.inputs["To Min"].default_value = 0.72
                mao_mr.inputs["To Max"].default_value = 1.0
                mao_mr.location = (-1120, 480 - idx * 300)
                nt.links.new(tn.outputs["Alpha"], mao_mr.inputs["Value"])

                det_ao = nt.nodes.new("ShaderNodeVectorMath")
                det_ao.operation = "SCALE"
                det_ao.location = (-860, 560 - idx * 300)
                nt.links.new(gam.outputs["Color"], det_ao.inputs[0])
                nt.links.new(mao_mr.outputs["Result"], det_ao.inputs["Scale"])
                layer_micro_tex.append(det_ao.outputs["Vector"])

                tmul = nt.nodes.new("ShaderNodeMix")
                tmul.data_type = "RGBA"
                tmul.blend_type = "MULTIPLY"
                tmul.clamp_result = True
                tmul.inputs["Factor"].default_value = 1.0
                tmul.label = f"{c_tag} Final Color"
                tmul.location = (-720, 660 - idx * 300)
                nt.links.new(tint_n.outputs[0], tmul.inputs[6])
                nt.links.new(det_ao.outputs["Vector"], tmul.inputs[7])
                layer_cols.append(tmul.outputs[2])
            else:
                layer_cols.append(tint_n.outputs[0])
                white_n = nt.nodes.new("ShaderNodeRGB")
                white_n.outputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
                white_n.location = (-940, 600 - idx * 300)
                layer_micro_tex.append(white_n.outputs[0])

            rfl_img = _load_image(tex_dir, lay.get("rfl_file", ""), "Non-Color", img_cache) if is_met else None
            if rfl_img is not None:
                trfl = nt.nodes.new("ShaderNodeTexImage")
                trfl.image = rfl_img
                trfl.label = f"{c_tag} {cname} Metal Mask"
                trfl.location = (-1380, 430 - idx * 300)
                nt.links.new(uv_tile.outputs["Vector"], trfl.inputs["Vector"])
                rm_mr = nt.nodes.new("ShaderNodeMapRange")
                rm_mr.clamp = True
                rm_mr.inputs["From Min"].default_value = 0.25
                rm_mr.inputs["From Max"].default_value = 0.65
                rm_mr.inputs["To Min"].default_value = 0.20
                rm_mr.inputs["To Max"].default_value = 0.95
                rm_mr.location = (-1100, 430 - idx * 300)
                nt.links.new(trfl.outputs["Color"], rm_mr.inputs["Value"])
                layer_mets.append(rm_mr.outputs["Result"])
            else:
                layer_mets.append(0.90 if is_met else 0.0)

            # Per-corner 16x Micro-Normal (R, G) AND Per-Pixel Micro-Roughness from `*_NMT_RGH_HGT.B` (Decima GLS)!
            dn_socks = _get_submat_nrm_rgh(lay.get("normal_file", ""), -2100 - idx * 240)
            if dn_socks is not None:
                dn_x, dn_y, dn_gls = dn_socks
                layer_dnx.append(dn_x)
                layer_dny.append(dn_y)

                # Convert per-pixel Decima Smoothness (`GLS` in `NMT_RGH_HGT.B`) into physically calibrated PBR Roughness!
                # Verified pixel stats: Metal_Robotplate GLS=0.86..0.91, Leather_FullGrain GLS=0.21..0.38,
                # Carbon_Fiber GLS=0.47..0.52, Wool GLS=0.16..0.22, Silk GLS=0.00..0.02!
                r_mr = nt.nodes.new("ShaderNodeMapRange")
                r_mr.clamp = True
                r_mr.label = f"{c_tag} {cname} Roughness"
                if is_met:
                    r_mr.inputs["From Min"].default_value = 0.0
                    r_mr.inputs["From Max"].default_value = 1.0
                    r_mr.inputs["To Min"].default_value = 0.44
                    r_mr.inputs["To Max"].default_value = 0.16
                elif any(k in clow for k in ("shell", "pearl", "bead", "turquoise", "lacquer", "glass", "ceramic")):
                    r_mr.inputs["From Min"].default_value = 0.0
                    r_mr.inputs["From Max"].default_value = 1.0
                    r_mr.inputs["To Min"].default_value = 0.46
                    r_mr.inputs["To Max"].default_value = 0.18
                elif any(k in clow for k in ("fullgrain", "finegrain", "fish", "leather")) and "buckskin" not in clow and "suede" not in clow:
                    r_mr.inputs["From Min"].default_value = 0.12
                    r_mr.inputs["From Max"].default_value = 0.55
                    r_mr.inputs["To Min"].default_value = 0.68
                    r_mr.inputs["To Max"].default_value = 0.34
                elif any(k in clow for k in ("carbon", "pvc", "rubber", "plastic", "synthetic")):
                    r_mr.inputs["From Min"].default_value = 0.20
                    r_mr.inputs["From Max"].default_value = 0.75
                    r_mr.inputs["To Min"].default_value = 0.64
                    r_mr.inputs["To Max"].default_value = 0.32
                elif any(k in clow for k in ("silk", "satin", "buckskin", "suede", "bone", "wood")):
                    r_mr.inputs["From Min"].default_value = 0.0
                    r_mr.inputs["From Max"].default_value = 0.50
                    r_mr.inputs["To Min"].default_value = 0.68
                    r_mr.inputs["To Max"].default_value = 0.42
                else:  # wool, cotton, crochet, linen, tassels, rope
                    r_mr.inputs["From Min"].default_value = 0.0
                    r_mr.inputs["From Max"].default_value = 0.50
                    r_mr.inputs["To Min"].default_value = 0.88
                    r_mr.inputs["To Max"].default_value = 0.66
                r_mr.location = (-860, -2100 - idx * 140)
                nt.links.new(dn_gls, r_mr.inputs["Value"])
                layer_rghs.append(r_mr.outputs["Result"])
            else:
                layer_dnx.append(0.5)
                layer_dny.append(0.5)
                base_rgh = float(lay.get("roughness", 0.30 if is_met else 0.58))
                if is_met:
                    eff_rgh = min(0.36, max(0.18, base_rgh * 0.60))
                elif any(k in clow for k in ("shell", "pearl", "bead", "turquoise", "lacquer")):
                    eff_rgh = 0.28
                elif any(k in clow for k in ("wool", "cotton", "linen", "rope", "tassel")):
                    eff_rgh = 0.82
                elif "silk" in clow:
                    eff_rgh = 0.52
                elif "leather" in clow and "buckskin" not in clow:
                    eff_rgh = 0.48
                else:
                    eff_rgh = max(0.46, min(0.76, base_rgh))
                layer_rghs.append(eff_rgh)

    # Pad to 8 corners if fewer than 8 layers were supplied (fallback / building blocks)
    while len(layer_cols) < 8:
        src_i = len(layer_cols) % max(1, len(layers))
        layer_cols.append(layer_cols[src_i])
        layer_micro_tex.append(layer_micro_tex[src_i])
        layer_mets.append(layer_mets[src_i])
        layer_rghs.append(layer_rghs[src_i])
        layer_dnx.append(layer_dnx[src_i])
        layer_dny.append(layer_dny[src_i])

    msk_img = _load_image(tex_dir, layered.get("mask_file", ""), "Non-Color", img_cache)
    t_msk = None
    if msk_img is not None and len(layers) > 1:
        t_msk = nt.nodes.new("ShaderNodeTexImage")
        t_msk.image = msk_img
        t_msk.label = "HFW 8-Layer MSK (RGB)"
        t_msk.location = (-1380, -1720)
        sep_m = nt.nodes.new("ShaderNodeSeparateColor")
        sep_m.label = "Unpack MSK (R, G, B)"
        sep_m.location = (-1100, -1720)
        nt.links.new(t_msk.outputs["Color"], sep_m.inputs["Color"])

        # Sharpen R, G, B cleanly around 0.5 (matching DXIL `fcmp ogt 0.5`) so BC-compressed seams never rainbow-bleed!
        if not is_bb:
            def _sharpen_bit(sock, lbl: str, y: float):
                mr = nt.nodes.new("ShaderNodeMapRange")
                mr.clamp = True
                mr.label = f"Sharpen {lbl}"
                mr.inputs["From Min"].default_value = 0.42
                mr.inputs["From Max"].default_value = 0.58
                mr.inputs["To Min"].default_value = 0.0
                mr.inputs["To Max"].default_value = 1.0
                mr.location = (-880, y)
                nt.links.new(sock, mr.inputs["Value"])
                return mr.outputs["Result"]
            r_bit = _sharpen_bit(sep_m.outputs["Red"], "MSK.R", -1540)
            g_bit = _sharpen_bit(sep_m.outputs["Green"], "MSK.G", -1720)
            b_bit = _sharpen_bit(sep_m.outputs["Blue"], "MSK.B", -1900)
        else:
            r_bit = sep_m.outputs["Red"]
            g_bit = sep_m.outputs["Green"]
            b_bit = sep_m.outputs["Blue"]

        # Exact DXIL RGB cube corner indexing:
        #   B=0 plane: (0,0,0)=L0 Black, (1,0,0)=L1 Red,  (0,1,0)=L2 Green, (1,1,0)=L6 Yellow!!
        #   B=1 plane: (0,0,1)=L3 Blue,  (1,0,1)=L5 Mag,  (0,1,1)=L4 Cyan,  (1,1,1)=L7 White!!
        def _trilinear_rgba(vals: list[Any], x0: float, y0: float):
            def _mx(a, b, fac, x, y):
                m = nt.nodes.new("ShaderNodeMix")
                m.data_type = "RGBA"
                m.location = (x, y)
                nt.links.new(fac, m.inputs["Factor"])
                nt.links.new(a, m.inputs[6])
                nt.links.new(b, m.inputs[7])
                return m.outputs[2]
            c_g0_b0 = _mx(vals[0], vals[1], r_bit, x0, y0 + 240)  # Black -> Red
            c_g1_b0 = _mx(vals[2], vals[6], r_bit, x0, y0 + 80)   # Green -> Yellow (L6!)
            c_g0_b1 = _mx(vals[3], vals[5], r_bit, x0, y0 - 80)   # Blue (L3!) -> Magenta (L5)
            c_g1_b1 = _mx(vals[4], vals[7], r_bit, x0, y0 - 240)  # Cyan (L4!) -> White (L7)
            c_b0 = _mx(c_g0_b0, c_g1_b0, g_bit, x0 + 200, y0 + 140)
            c_b1 = _mx(c_g0_b1, c_g1_b1, g_bit, x0 + 200, y0 - 140)
            return _mx(c_b0, c_b1, b_bit, x0 + 400, y0)

        def _trilinear_float(vals: list[float], x0: float, y0: float):
            def _mxf(a, b, fac, x, y):
                m = nt.nodes.new("ShaderNodeMix")
                m.data_type = "FLOAT"
                m.location = (x, y)
                nt.links.new(fac, m.inputs["Factor"])
                if isinstance(a, (int, float)):
                    m.inputs[2].default_value = float(a)
                else:
                    nt.links.new(a, m.inputs[2])
                if isinstance(b, (int, float)):
                    m.inputs[3].default_value = float(b)
                else:
                    nt.links.new(b, m.inputs[3])
                return m.outputs[0]
            f_g0_b0 = _mxf(vals[0], vals[1], r_bit, x0, y0 + 180)
            f_g1_b0 = _mxf(vals[2], vals[6], r_bit, x0, y0 + 60)
            f_g0_b1 = _mxf(vals[3], vals[5], r_bit, x0, y0 - 60)
            f_g1_b1 = _mxf(vals[4], vals[7], r_bit, x0, y0 - 180)
            f_b0 = _mxf(f_g0_b0, f_g1_b0, g_bit, x0 + 180, y0 + 100)
            f_b1 = _mxf(f_g0_b1, f_g1_b1, g_bit, x0 + 180, y0 - 100)
            return _mxf(f_b0, f_b1, b_bit, x0 + 360, y0)

        col_sock = _trilinear_rgba(layer_cols, -500, 320)
        met_sock = _trilinear_float(layer_mets, -500, -340)
        rgh_sock = _trilinear_float(layer_rghs, -500, -820)
        micro_nx_sock = _trilinear_float(layer_dnx, -500, -1960) if (not is_bb and nrm_node_cache) else None
        micro_ny_sock = _trilinear_float(layer_dny, -500, -2420) if (not is_bb and nrm_node_cache) else None

        # Hybrid outfits (e.g. SeykaTorso_NMT painted coral-to-slate scarf + 8-layer crochet cotton / leather / metal):
        # - `multiply2`: macro `_CLR` has neutral 0.50 linear background (sRGB ~188) -> `clamp(2.0 * macro * dye_detail)`
        #   preserving dark fish leather / aqua wool / steel clasps while painting the coral scarf gradient on top!
        # - `replace`: macro `_CLR` is a direct painted albedo trim sheet (`SeykaTrimsUV2_NMT`) -> `macro * micro_detail`!
        macro_clr_file = layered.get("macro_color_file", "")
        macro_clr_mode = layered.get("macro_color_mode", "replace")
        if (not is_bb) and macro_clr_file:
            mc_img = _load_image(tex_dir, macro_clr_file, "sRGB", img_cache)
            if mc_img is not None:
                t_mc = nt.nodes.new("ShaderNodeTexImage")
                t_mc.image = mc_img
                t_mc.label = "HFW Painted Macro Albedo"
                t_mc.location = (-440, 740)
                if macro_clr_mode == "multiply2":
                    mc_x2 = nt.nodes.new("ShaderNodeVectorMath")
                    mc_x2.operation = "SCALE"
                    mc_x2.inputs["Scale"].default_value = 2.0
                    mc_x2.label = "Macro Overlay * 2 (0.5 -> 1.0)"
                    mc_x2.location = (-160, 740)
                    nt.links.new(t_mc.outputs["Color"], mc_x2.inputs[0])
                    mc_mul = nt.nodes.new("ShaderNodeMix")
                    mc_mul.data_type = "RGBA"
                    mc_mul.blend_type = "MULTIPLY"
                    mc_mul.clamp_result = True
                    mc_mul.inputs["Factor"].default_value = 1.0
                    mc_mul.label = "DXIL 8-Corner Dye * (2x Macro Overlay)"
                    mc_mul.location = (80, 620)
                    nt.links.new(col_sock, mc_mul.inputs[6])
                    nt.links.new(mc_x2.outputs["Vector"], mc_mul.inputs[7])
                    col_sock = mc_mul.outputs[2]
                else:
                    micro_det_sock = _trilinear_rgba(layer_micro_tex, -500, 820)
                    mc_mul = nt.nodes.new("ShaderNodeMix")
                    mc_mul.data_type = "RGBA"
                    mc_mul.blend_type = "MULTIPLY"
                    mc_mul.clamp_result = True
                    mc_mul.inputs["Factor"].default_value = 1.0
                    mc_mul.label = "Macro Painted Albedo * 16x Micro-Detail"
                    mc_mul.location = (80, 620)
                    nt.links.new(t_mc.outputs["Color"], mc_mul.inputs[6])
                    nt.links.new(micro_det_sock, mc_mul.inputs[7])
                    col_sock = mc_mul.outputs[2]

        nt.links.new(met_sock, bsdf.inputs["Metallic"])

        # Add subtle MISC.G dirt roughness variation (clean worn edges stay sleek while dusty seams roughen!)
        misc_img = _load_image(tex_dir, layered.get("misc_file", ""), "Non-Color", img_cache) if not is_bb else None
        if misc_img is not None:
            t_misc = nt.nodes.new("ShaderNodeTexImage")
            t_misc.image = misc_img
            t_misc.label = "HFW 2K MISC (Wear/Dirt/Fuzz)"
            t_misc.location = (-440, -1180)
            sep_ms = nt.nodes.new("ShaderNodeSeparateColor")
            sep_ms.location = (-160, -1180)
            nt.links.new(t_misc.outputs["Color"], sep_ms.inputs["Color"])
            dirt_r = nt.nodes.new("ShaderNodeMath")
            dirt_r.operation = "MULTIPLY_ADD"
            dirt_r.inputs[1].default_value = 0.16
            dirt_r.use_clamp = True
            dirt_r.label = "Roughness + Dirt Contrast"
            dirt_r.location = (120, -960)
            nt.links.new(sep_ms.outputs["Green"], dirt_r.inputs[0])
            nt.links.new(rgh_sock, dirt_r.inputs[2])
            rgh_sock = dirt_r.outputs["Value"]

        # Subtle cloth velvet/fiber Sheen on non-metallic areas (`(1 - Metallic) * 0.25`)
        if (not is_bb) and "Sheen Weight" in bsdf.inputs:
            inv_met = nt.nodes.new("ShaderNodeMath")
            inv_met.operation = "MULTIPLY_ADD"
            inv_met.inputs[1].default_value = -0.26
            inv_met.inputs[2].default_value = 0.26
            inv_met.use_clamp = True
            inv_met.label = "Non-Metal Cloth Sheen"
            inv_met.location = (260, -340)
            nt.links.new(met_sock, inv_met.inputs[0])
            nt.links.new(inv_met.outputs["Value"], bsdf.inputs["Sheen Weight"])
            if "Sheen Roughness" in bsdf.inputs:
                bsdf.inputs["Sheen Roughness"].default_value = 0.55

        # On building blocks whose MSK Alpha packs Smoothness (Roughness.R), use `1 - Smoothness`!
        if is_bb and layered.get("mask_has_roughness"):
            inv_sm = nt.nodes.new("ShaderNodeMath")
            inv_sm.operation = "SUBTRACT"
            inv_sm.use_clamp = True
            inv_sm.inputs[0].default_value = 1.0
            inv_sm.label = "1 - Smoothness -> Roughness"
            inv_sm.location = (220, -640)
            nt.links.new(t_msk.outputs["Alpha"], inv_sm.inputs[1])
            nt.links.new(inv_sm.outputs["Value"], bsdf.inputs["Roughness"])
        else:
            nt.links.new(rgh_sock, bsdf.inputs["Roughness"])

        # Wire Alpha cutout ONLY when this outfit's own unique MSK texture carries an explicit Alpha lane (e.g. TrimsAlpha!)
        if layered.get("mask_has_alpha"):
            _configure_blended_material(mat)
            if hasattr(mat, "blend_method"):
                try:
                    mat.blend_method = "HASHED"
                except Exception:
                    pass
            nt.links.new(t_msk.outputs["Alpha"], bsdf.inputs["Alpha"])
    else:
        col_sock = layer_cols[0]
        micro_nx_sock = layer_dnx[0] if (not is_bb and nrm_node_cache and not isinstance(layer_dnx[0], (int, float))) else None
        micro_ny_sock = layer_dny[0] if (not is_bb and nrm_node_cache and not isinstance(layer_dny[0], (int, float))) else None
        if isinstance(layer_mets[0], (int, float)):
            bsdf.inputs["Metallic"].default_value = layer_mets[0]
        else:
            nt.links.new(layer_mets[0], bsdf.inputs["Metallic"])
        if isinstance(layer_rghs[0], (int, float)):
            bsdf.inputs["Roughness"].default_value = layer_rghs[0]
        else:
            nt.links.new(layer_rghs[0], bsdf.inputs["Roughness"])

    # Unique 2K Sculpted Normal + Baked AO + Baked Curvature (`NMT_AO_CRV`: R=NrmX, G=NrmY, B=AO, A=Curvature)
    nrm_img = _load_image(tex_dir, layered.get("normal_file", ""), "Non-Color", img_cache)
    if nrm_img is not None:
        t_nrm = nt.nodes.new("ShaderNodeTexImage")
        t_nrm.image = nrm_img
        t_nrm.label = "HFW 2K Sculpted Normal/AO/CRV"
        t_nrm.location = (-560, -1480)
        if is_bb and not layered.get("mask_file"):
            nt.links.new(uv_tile.outputs["Vector"], t_nrm.inputs["Vector"])
        sep_n = nt.nodes.new("ShaderNodeSeparateColor")
        sep_n.location = (-280, -1480)
        nt.links.new(t_nrm.outputs["Color"], sep_n.inputs["Color"])

        nx_in = sep_n.outputs["Red"]
        ny_in = sep_n.outputs["Green"]

        # Blend per-corner 16x micro-detail normal XY cleanly onto 2K macro normal XY (tangent-space additive UDN!)
        if micro_nx_sock is not None and micro_ny_sock is not None:
            def _udn_add(macro_s, micro_s, lbl: str, y: float):
                off = nt.nodes.new("ShaderNodeMath")
                off.operation = "MULTIPLY_ADD"
                off.inputs[1].default_value = 0.85
                off.inputs[2].default_value = -0.425
                off.location = (-80, y)
                nt.links.new(micro_s, off.inputs[0])
                sm = nt.nodes.new("ShaderNodeMath")
                sm.operation = "ADD"
                sm.use_clamp = True
                sm.label = lbl
                sm.location = (80, y)
                nt.links.new(macro_s, sm.inputs[0])
                nt.links.new(off.outputs["Value"], sm.inputs[1])
                return sm.outputs["Value"]
            nx_in = _udn_add(nx_in, micro_nx_sock, "Macro+Micro NrmX", -1420)
            ny_in = _udn_add(ny_in, micro_ny_sock, "Macro+Micro NrmY", -1620)

        nrm_rec = nt.nodes.new("ShaderNodeGroup")
        nrm_rec.node_tree = _ensure_normal_reconstruct_z_group()
        nrm_rec.inputs["Flip Green"].default_value = 0.0
        nrm_rec.location = (240, -1480)
        nt.links.new(nx_in, nrm_rec.inputs["Normal X"])
        nt.links.new(ny_in, nrm_rec.inputs["Normal Y"])

        nrm_map = nt.nodes.new("ShaderNodeNormalMap")
        nrm_map.location = (440, -1480)
        nt.links.new(nrm_rec.outputs["Normal RGB"], nrm_map.inputs["Color"])
        nt.links.new(nrm_map.outputs["Normal"], bsdf.inputs["Normal"])

        npack = layered.get("normal_packing") or {}
        # Apply 2K Baked Curvature (`NMT_AO_CRV.A`, centered at 0.50!) for crisp convex edge highlights & fold depth!
        if any(v.startswith("Curvature") for v in npack.values()):
            crv_mr = nt.nodes.new("ShaderNodeMapRange")
            crv_mr.clamp = True
            crv_mr.label = "2K Curvature Edge Contrast"
            crv_mr.inputs["From Min"].default_value = 0.25
            crv_mr.inputs["From Max"].default_value = 0.75
            crv_mr.inputs["To Min"].default_value = 0.78
            crv_mr.inputs["To Max"].default_value = 1.24
            crv_mr.location = (120, 120)
            nt.links.new(t_nrm.outputs["Alpha"], crv_mr.inputs["Value"])

            crv_sc = nt.nodes.new("ShaderNodeVectorMath")
            crv_sc.operation = "SCALE"
            crv_sc.label = "Apply 2K Curvature"
            crv_sc.location = (280, 180)
            nt.links.new(col_sock, crv_sc.inputs[0])
            nt.links.new(crv_mr.outputs["Result"], crv_sc.inputs["Scale"])
            col_sock = crv_sc.outputs["Vector"]

        # Multiply baked 2K AO (Blue lane when packed as NMT_AO_CRV) into blended Base Color!
        if any(v.startswith("AO") for v in npack.values()):
            ao_mul = nt.nodes.new("ShaderNodeMix")
            ao_mul.data_type = "RGBA"
            ao_mul.blend_type = "MULTIPLY"
            ao_mul.inputs["Factor"].default_value = 0.88
            ao_mul.label = "Apply 2K Baked AO"
            ao_mul.location = (440, 260)
            nt.links.new(col_sock, ao_mul.inputs[6])
            nt.links.new(sep_n.outputs["Blue"], ao_mul.inputs[7])
            col_sock = ao_mul.outputs[2]

    nt.links.new(col_sock, bsdf.inputs["Base Color"])
    if (not is_bb) and "Sheen Weight" in bsdf.inputs:
        bsdf.inputs["Sheen Weight"].default_value = 0.20
        bsdf.inputs["Sheen Roughness"].default_value = 0.55
    return True


def _build_titan_layered_material(
    mat: bpy.types.Material,
    titan: dict[str, Any],
    tex_dir: Path,
    img_cache: dict[str, bpy.types.Image],
) -> bool:
    """Build Guerrilla's exact DXIL Horus / FAS-BOR7 Titan 26-TB Group-132 Multi-Layer Hull Shader.

    Reconstructs `ps_1256_1_p1.ll` faithfully in Cycles & EEVEE:
      - Macro 4K `Titan_*_NMT` (`R,G=Normal`, `B=Baked AO`, `A=Curvature`),
        `Titan_*_MSK` (3-bit RGB 8-corner mask), and `Titan_*_MISC` (`R=Damage`, `G=Rust`, `B=Grunge/Moss`).
      - Active-corner-only sampling of Clean (`132:2/3/8`) <-> Damaged (`132:9/6/7`) Group-132 slices
        at each corner's exact DXIL `tile_scale` (`6x..32x`), tinted by `2.0 * PaintLUT_132_5` via `PaintMask`!
      - Corroded iron-oxide Rust (`64:0/1/3`) & oil Grunge (`132:0/1`) layered BEFORE exact DXIL `Gamma 2.2`!
      - Baked Macro Curvature + Macro/Micro AO applied in linear PBR space.
      - Lush forest `StarMoss` (`1:0/1/2`) blended AFTER `Gamma 2.2` ONLY on world-upward-facing surfaces!
      - Exact `HFW_DecimaPBR` conversion (`Smoothness -> Roughness`, `Reflectance -> Metallic + F0 = RFL^2`).
      - Optional DXIL deep-crevice `(1 - AO)^4` molten orange-red heat glow on Cooling Vents / Drill Weakspots!
    """
    corners = titan.get("corners") or []
    if len(corners) < 8:
        return False

    nt = mat.node_tree
    nt.nodes.clear()
    mat["hfw_shader_kind"] = "titan_hull_layered"

    out_node = nt.nodes.new("ShaderNodeOutputMaterial")
    out_node.location = (1860, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (1480, 0)
    nt.links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])

    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.location = (-2720, 260)

    # 1. Sample 4K Macro NMT/AO/CRV, MSK, and MISC
    nmt_img = _load_image(tex_dir, titan.get("macro_nmt_file", ""), "Non-Color", img_cache)
    msk_img = _load_image(tex_dir, titan.get("macro_msk_file", ""), "Non-Color", img_cache)
    misc_img = _load_image(tex_dir, titan.get("macro_misc_file", ""), "Non-Color", img_cache)

    macro_nx = None
    macro_ny = None
    macro_ao = None
    macro_crv = None
    if nmt_img is not None:
        t_mn = nt.nodes.new("ShaderNodeTexImage")
        t_mn.image = nmt_img
        t_mn.label = "HFW Titan Macro NMT_AO_CRV"
        t_mn.location = (-2360, -1420)
        sep_mn = nt.nodes.new("ShaderNodeSeparateColor")
        sep_mn.label = "Unpack Macro NMT/AO"
        sep_mn.location = (-2060, -1420)
        nt.links.new(t_mn.outputs["Color"], sep_mn.inputs["Color"])
        macro_nx = sep_mn.outputs["Red"]
        macro_ny = sep_mn.outputs["Green"]
        macro_ao = sep_mn.outputs["Blue"]
        macro_crv = t_mn.outputs["Alpha"]

    misc_dmg = None
    misc_rust = None
    misc_grunge = None
    if misc_img is not None:
        t_ms = nt.nodes.new("ShaderNodeTexImage")
        t_ms.image = misc_img
        t_ms.label = "HFW Titan Macro MISC (R=Dmg, G=Rust, B=Grunge)"
        t_ms.location = (-2360, -960)
        sep_ms = nt.nodes.new("ShaderNodeSeparateColor")
        sep_ms.location = (-2060, -960)
        nt.links.new(t_ms.outputs["Color"], sep_ms.inputs["Color"])
        misc_dmg = sep_ms.outputs["Red"]
        misc_rust = sep_ms.outputs["Green"]
        misc_grunge = sep_ms.outputs["Blue"]

    # 2. Sample ONLY active corners, deduplicating TextureImage nodes shared across corners with identical (slice, tile)!
    dom_corner = int(titan.get("dominant_corner", 0))
    active_ids = [int(c["corner"]) for c in corners if c.get("active")]
    if not active_ids:
        active_ids = [dom_corner]

    uv_scale_cache: dict[float, Any] = {}
    slice_tex_cache: dict[tuple[str, float], Any] = {}
    slice_sep_cache: dict[tuple[str, float], Any] = {}
    # Cache Clean<->Damaged per (slice_idx, tile_scale) so identical slices with different paint LUTs share textures!
    pair_cache: dict[tuple[int, float], dict[str, Any]] = {}

    def _get_scaled_uv(scale: float) -> Any:
        key = round(float(scale), 2)
        if key not in uv_scale_cache:
            vm = nt.nodes.new("ShaderNodeVectorMath")
            vm.operation = "SCALE"
            vm.inputs["Scale"].default_value = key
            vm.label = f"Titan Tile UV ({key:.0f}x)"
            vm.location = (-2440, 620 - len(uv_scale_cache) * 140)
            nt.links.new(tc.outputs["UV"], vm.inputs[0])
            uv_scale_cache[key] = vm.outputs["Vector"]
        return uv_scale_cache[key]

    def _get_slice_tex(fname: str, scale: float, lbl: str, y_hint: float) -> Any | None:
        if not fname:
            return None
        key = (fname, round(float(scale), 2))
        if key not in slice_tex_cache:
            img = _load_image(tex_dir, fname, "Non-Color", img_cache)
            if img is None:
                slice_tex_cache[key] = None
                return None
            tn = nt.nodes.new("ShaderNodeTexImage")
            tn.image = img
            tn.label = lbl
            tn.location = (-2100, y_hint)
            nt.links.new(_get_scaled_uv(scale), tn.inputs["Vector"])
            slice_tex_cache[key] = tn
        return slice_tex_cache[key]

    def _get_slice_sep(fname: str, scale: float, lbl: str, y_hint: float) -> Any | None:
        key = (fname, round(float(scale), 2))
        if key not in slice_sep_cache:
            tn = _get_slice_tex(fname, scale, lbl, y_hint)
            if tn is None:
                slice_sep_cache[key] = None
                return None
            sp = nt.nodes.new("ShaderNodeSeparateColor")
            sp.location = (-1820, y_hint)
            nt.links.new(tn.outputs["Color"], sp.inputs["Color"])
            slice_sep_cache[key] = sp
        return slice_sep_cache[key]

    def _lerp_float(a: Any, b: Any, fac: Any, x: float, y: float, lbl: str = "") -> Any:
        if fac is None or a == b:
            return a
        m = nt.nodes.new("ShaderNodeMix")
        m.data_type = "FLOAT"
        if lbl:
            m.label = lbl
        m.location = (x, y)
        nt.links.new(fac, m.inputs["Factor"])
        if isinstance(a, (int, float)):
            m.inputs[2].default_value = float(a)
        else:
            nt.links.new(a, m.inputs[2])
        if isinstance(b, (int, float)):
            m.inputs[3].default_value = float(b)
        else:
            nt.links.new(b, m.inputs[3])
        return m.outputs[0]

    def _lerp_rgba(a: Any, b: Any, fac: Any, x: float, y: float, lbl: str = "") -> Any:
        if fac is None or a == b:
            return a
        m = nt.nodes.new("ShaderNodeMix")
        m.data_type = "RGBA"
        if lbl:
            m.label = lbl
        m.location = (x, y)
        nt.links.new(fac, m.inputs["Factor"])
        nt.links.new(a, m.inputs[6])
        nt.links.new(b, m.inputs[7])
        return m.outputs[2]

    corner_data: dict[int, dict[str, Any]] = {}
    for slot_i, ci in enumerate(active_ids):
        cinfo = corners[ci]
        s_idx = int(cinfo.get("slice_idx", 11))
        t_scale = float(cinfo.get("tile_scale", 16.0))
        y_base = 1100 - slot_i * 420
        pkey = (s_idx, round(t_scale, 2))

        if pkey not in pair_cache:
            c_clr_n = _get_slice_tex(cinfo.get("clean_clr_file", ""), t_scale, f"CleanCLR s{s_idx:02d}", y_base)
            d_clr_n = _get_slice_tex(cinfo.get("dmg_clr_file", ""), t_scale, f"DmgCLR s{s_idx:02d}", y_base - 120)
            c_nmt_s = _get_slice_sep(cinfo.get("clean_nmt_file", ""), t_scale, f"CleanNMT s{s_idx:02d}", y_base - 240)
            d_nmt_s = _get_slice_sep(cinfo.get("dmg_nmt_file", ""), t_scale, f"DmgNMT s{s_idx:02d}", y_base - 320)
            c_rfl_s = _get_slice_sep(cinfo.get("clean_rfl_file", ""), t_scale, f"CleanRFL s{s_idx:02d}", y_base - 400)
            d_rfl_s = _get_slice_sep(cinfo.get("dmg_rfl_file", ""), t_scale, f"DmgRFL s{s_idx:02d}", y_base - 480)

            if c_clr_n is not None and d_clr_n is not None:
                raw_clr = _lerp_rgba(c_clr_n.outputs["Color"], d_clr_n.outputs["Color"], misc_dmg, -1520, y_base, f"s{s_idx:02d} Clean<->Dmg CLR")
            elif c_clr_n is not None:
                raw_clr = c_clr_n.outputs["Color"]
            else:
                fb = nt.nodes.new("ShaderNodeRGB")
                fb.outputs[0].default_value = (0.50, 0.50, 0.50, 1.0)
                fb.location = (-1520, y_base)
                raw_clr = fb.outputs[0]

            if c_rfl_s is not None and d_rfl_s is not None:
                rfl_s = _lerp_float(c_rfl_s.outputs["Red"], d_rfl_s.outputs["Red"], misc_dmg, -1520, y_base - 140)
                mao_s = _lerp_float(c_rfl_s.outputs["Green"], d_rfl_s.outputs["Green"], misc_dmg, -1520, y_base - 210)
                pm_s = _lerp_float(c_rfl_s.outputs["Blue"], d_rfl_s.outputs["Blue"], misc_dmg, -1520, y_base - 280)
            elif c_rfl_s is not None:
                rfl_s, mao_s, pm_s = c_rfl_s.outputs["Red"], c_rfl_s.outputs["Green"], c_rfl_s.outputs["Blue"]
            else:
                rfl_s, mao_s, pm_s = 0.22, 1.0, 1.0

            if c_nmt_s is not None and d_nmt_s is not None:
                nx_s = _lerp_float(c_nmt_s.outputs["Red"], d_nmt_s.outputs["Red"], misc_dmg, -1520, y_base - 350)
                ny_s = _lerp_float(c_nmt_s.outputs["Green"], d_nmt_s.outputs["Green"], misc_dmg, -1520, y_base - 420)
                sm_s = _lerp_float(c_nmt_s.outputs["Blue"], d_nmt_s.outputs["Blue"], misc_dmg, -1520, y_base - 490)
            elif c_nmt_s is not None:
                nx_s, ny_s, sm_s = c_nmt_s.outputs["Red"], c_nmt_s.outputs["Green"], c_nmt_s.outputs["Blue"]
            else:
                nx_s, ny_s, sm_s = 0.5, 0.5, 0.45

            pair_cache[pkey] = {
                "raw_clr": raw_clr,
                "rfl": rfl_s,
                "mao": mao_s,
                "paint_mask": pm_s,
                "nx": nx_s,
                "ny": ny_s,
                "smooth": sm_s,
            }

        pd = pair_cache[pkey]
        # Exact DXIL Paint LUT Tint: `lerp(raw_clr, raw_clr * (2.0 * PaintLUT_132_5), PaintMask)`
        p_srgb = cinfo.get("paint_lut_srgb") or [0.502, 0.502, 0.502]
        lut_idx = int(cinfo.get("color_lut_idx", 0))
        if lut_idx == 0 or (abs(p_srgb[0] - 0.502) < 0.01 and abs(p_srgb[1] - 0.502) < 0.01 and abs(p_srgb[2] - 0.502) < 0.01):
            c_col = pd["raw_clr"]
        else:
            # 2.0 * paint_lut_srgb (clamped cleanly after multiplying raw_clr)
            t_mul = nt.nodes.new("ShaderNodeVectorMath")
            t_mul.operation = "MULTIPLY"
            t_mul.inputs[1].default_value = (float(p_srgb[0]) * 2.0, float(p_srgb[1]) * 2.0, float(p_srgb[2]) * 2.0)
            t_mul.label = f"C{ci} PaintLUT[{lut_idx}] * 2"
            t_mul.location = (-1260, y_base + 40)
            nt.links.new(pd["raw_clr"], t_mul.inputs[0])

            pm = pd["paint_mask"]
            if isinstance(pm, (int, float)):
                c_col = t_mul.outputs["Vector"] if pm >= 0.5 else pd["raw_clr"]
            else:
                c_col = _lerp_rgba(pd["raw_clr"], t_mul.outputs["Vector"], pm, -1040, y_base + 40, f"C{ci} PaintMask Blend")

        corner_data[ci] = {
            "clr": c_col,
            "rfl": pd["rfl"],
            "mao": pd["mao"],
            "nx": pd["nx"],
            "ny": pd["ny"],
            "smooth": pd["smooth"],
            "rust_mult": float(cinfo.get("rust_mult", 1.0)),
        }

    # Fill inactive corners from dominant corner (zero extra texture nodes!)
    dom_d = corner_data.get(dom_corner) or next(iter(corner_data.values()))
    all_c = [corner_data.get(i, dom_d) for i in range(8)]

    # 3. Blend active corners via exact DXIL 8-corner `MSK.RGB` trilinear tree (or direct if 1 corner!)
    if msk_img is not None and len(active_ids) > 1:
        t_msk = nt.nodes.new("ShaderNodeTexImage")
        t_msk.image = msk_img
        t_msk.label = "HFW Titan Macro MSK (8-Corner RGB)"
        t_msk.location = (-1680, -1720)
        sep_mk = nt.nodes.new("ShaderNodeSeparateColor")
        sep_mk.location = (-1400, -1720)
        nt.links.new(t_msk.outputs["Color"], sep_mk.inputs["Color"])

        def _sharp(sock, lbl: str, y: float):
            mr = nt.nodes.new("ShaderNodeMapRange")
            mr.clamp = True
            mr.label = f"Sharpen {lbl}"
            mr.inputs["From Min"].default_value = 0.42
            mr.inputs["From Max"].default_value = 0.58
            mr.inputs["To Min"].default_value = 0.0
            mr.inputs["To Max"].default_value = 1.0
            mr.location = (-1180, y)
            nt.links.new(sock, mr.inputs["Value"])
            return mr.outputs["Result"]

        r_bit = _sharp(sep_mk.outputs["Red"], "MSK.R", -1560)
        g_bit = _sharp(sep_mk.outputs["Green"], "MSK.G", -1720)
        b_bit = _sharp(sep_mk.outputs["Blue"], "MSK.B", -1880)

        def _tri_rgba(vals: list[Any], x0: float, y0: float):
            # Exact DXIL corner order: (0,0,0)=0, (1,0,0)=1, (0,1,0)=2, (1,1,0)=6, (0,0,1)=3, (1,0,1)=5, (0,1,1)=4, (1,1,1)=7
            c00 = _lerp_rgba(vals[0], vals[1], r_bit, x0, y0 + 210)
            c10 = _lerp_rgba(vals[2], vals[6], r_bit, x0, y0 + 70)
            c01 = _lerp_rgba(vals[3], vals[5], r_bit, x0, y0 - 70)
            c11 = _lerp_rgba(vals[4], vals[7], r_bit, x0, y0 - 210)
            cb0 = _lerp_rgba(c00, c10, g_bit, x0 + 180, y0 + 120)
            cb1 = _lerp_rgba(c01, c11, g_bit, x0 + 180, y0 - 120)
            return _lerp_rgba(cb0, cb1, b_bit, x0 + 360, y0)

        def _tri_float(vals: list[Any], x0: float, y0: float):
            f00 = _lerp_float(vals[0], vals[1], r_bit, x0, y0 + 150)
            f10 = _lerp_float(vals[2], vals[6], r_bit, x0, y0 + 50)
            f01 = _lerp_float(vals[3], vals[5], r_bit, x0, y0 - 50)
            f11 = _lerp_float(vals[4], vals[7], r_bit, x0, y0 - 150)
            fb0 = _lerp_float(f00, f10, g_bit, x0 + 160, y0 + 90)
            fb1 = _lerp_float(f01, f11, g_bit, x0 + 160, y0 - 90)
            return _lerp_float(fb0, fb1, b_bit, x0 + 320, y0)

        metal_srgb = _tri_rgba([d["clr"] for d in all_c], -760, 420)
        metal_rfl = _tri_float([d["rfl"] for d in all_c], -760, -140)
        metal_smooth = _tri_float([d["smooth"] for d in all_c], -760, -560)
        metal_mao = _tri_float([d["mao"] for d in all_c], -760, -960)
        metal_nx = _tri_float([d["nx"] for d in all_c], -760, -1380)
        metal_ny = _tri_float([d["ny"] for d in all_c], -760, -1800)
        rust_mult = _tri_float([d["rust_mult"] for d in all_c], -760, -2180)
    else:
        metal_srgb = dom_d["clr"]
        metal_rfl = dom_d["rfl"]
        metal_smooth = dom_d["smooth"]
        metal_mao = dom_d["mao"]
        metal_nx = dom_d["nx"]
        metal_ny = dom_d["ny"]
        rust_mult = dom_d["rust_mult"]

    # 4. Exact DXIL Corroded Iron-Oxide Rust Layer (`64:0/1/3`, BEFORE Gamma 2.2!)
    srgb_cur = metal_srgb
    rfl_cur = metal_rfl
    smooth_cur = metal_smooth
    micro_nx_cur = metal_nx
    micro_ny_cur = metal_ny

    rust_clr_img = _load_image(tex_dir, titan.get("rust_clr_file", ""), "Non-Color", img_cache)
    rust_nmt_img = _load_image(tex_dir, titan.get("rust_nmt_file", ""), "Non-Color", img_cache)
    rust_hgt_img = _load_image(tex_dir, titan.get("rust_hgt_file", ""), "Non-Color", img_cache)
    if rust_clr_img is not None and misc_rust is not None:
        uv_rust = _get_scaled_uv(24.0)
        t_rc = nt.nodes.new("ShaderNodeTexImage")
        t_rc.image = rust_clr_img
        t_rc.label = "HFW Rust_CLR (64:0)"
        t_rc.location = (-280, 920)
        nt.links.new(uv_rust, t_rc.inputs["Vector"])

        rlut = titan.get("rust_lut_srgb") or [0.435, 0.248, 0.172]
        r_col = nt.nodes.new("ShaderNodeVectorMath")
        r_col.operation = "MULTIPLY"
        r_col.inputs[1].default_value = (float(rlut[0]) * 1.45, float(rlut[1]) * 1.40, float(rlut[2]) * 1.38)
        r_col.label = "Rust_CLR * RustLUT (64:3)"
        r_col.location = (20, 920)
        nt.links.new(t_rc.outputs["Color"], r_col.inputs[0])

        # Height-broken rust mask (matching DXIL ps_1256_1_p1.ll): preserves dominant dark charcoal-black Faro
        # gunmetal armor plates while pitting seams, crevices, and corrosion streaks with weathered dark rust!
        rm_base = nt.nodes.new("ShaderNodeMapRange")
        rm_base.interpolation_type = "SMOOTHSTEP"
        rm_base.clamp = True
        rm_base.label = "Rust Mask (MISC.G)"
        rm_base.inputs["From Min"].default_value = 0.36
        rm_base.inputs["From Max"].default_value = 0.94
        rm_base.inputs["To Min"].default_value = 0.0
        rm_base.inputs["To Max"].default_value = 0.72
        rm_base.location = (-260, 680)
        nt.links.new(misc_rust, rm_base.inputs["Value"])
        rust_m = rm_base.outputs["Result"]

        if macro_ao is not None:
            # Subtle crevice rust boost `(0.75 - AO) * 0.22`
            ao_inv = nt.nodes.new("ShaderNodeMapRange")
            ao_inv.clamp = True
            ao_inv.inputs["From Min"].default_value = 0.10
            ao_inv.inputs["From Max"].default_value = 0.72
            ao_inv.inputs["To Min"].default_value = 0.22
            ao_inv.inputs["To Max"].default_value = 0.0
            ao_inv.location = (-260, 500)
            nt.links.new(macro_ao, ao_inv.inputs["Value"])
            r_add = nt.nodes.new("ShaderNodeMath")
            r_add.operation = "ADD"
            r_add.use_clamp = True
            r_add.location = (-60, 620)
            nt.links.new(rust_m, r_add.inputs[0])
            nt.links.new(ao_inv.outputs["Result"], r_add.inputs[1])
            rust_m = r_add.outputs["Value"]

        # Height-map corrosion breakup (`Rust_HGT` 64:0 -> crisp jagged pitted rust edges!)
        if rust_hgt_img is not None:
            t_rh = nt.nodes.new("ShaderNodeTexImage")
            t_rh.image = rust_hgt_img
            t_rh.label = "HFW Rust_HGT (64:0)"
            t_rh.location = (-480, 1140)
            nt.links.new(uv_rust, t_rh.inputs["Vector"])
            rh_mr = nt.nodes.new("ShaderNodeMapRange")
            rh_mr.clamp = True
            rh_mr.inputs["From Min"].default_value = 0.20
            rh_mr.inputs["From Max"].default_value = 0.85
            rh_mr.inputs["To Min"].default_value = 0.18
            rh_mr.inputs["To Max"].default_value = 1.0
            rh_mr.location = (-180, 1140)
            nt.links.new(t_rh.outputs["Color"], rh_mr.inputs["Value"])
            rh_mul = nt.nodes.new("ShaderNodeMath")
            rh_mul.operation = "MULTIPLY"
            rh_mul.use_clamp = True
            rh_mul.location = (20, 720)
            nt.links.new(rust_m, rh_mul.inputs[0])
            nt.links.new(rh_mr.outputs["Result"], rh_mul.inputs[1])
            rust_m = rh_mul.outputs["Value"]

        # Multiply by corner rust_mult (e.g. 0.3 on bare steel corner 3)
        r_mul_c = nt.nodes.new("ShaderNodeMath")
        r_mul_c.operation = "MULTIPLY"
        r_mul_c.use_clamp = True
        r_mul_c.label = "Rust * CornerRustMult"
        r_mul_c.location = (160, 620)
        nt.links.new(rust_m, r_mul_c.inputs[0])
        if isinstance(rust_mult, (int, float)):
            r_mul_c.inputs[1].default_value = float(rust_mult)
        else:
            nt.links.new(rust_mult, r_mul_c.inputs[1])
        rust_mask = r_mul_c.outputs["Value"]

        srgb_cur = _lerp_rgba(srgb_cur, r_col.outputs["Vector"], rust_mask, 320, 580, "Metal <-> Rust Albedo")
        rfl_cur = _lerp_float(rfl_cur, 0.16, rust_mask, 320, -140, "Metal <-> Rust RFL")
        smooth_cur = _lerp_float(smooth_cur, 0.16, rust_mask, 320, -480, "Metal <-> Rust Smoothness")

        if rust_nmt_img is not None:
            t_rn = nt.nodes.new("ShaderNodeTexImage")
            t_rn.image = rust_nmt_img
            t_rn.label = "HFW Rust_NMT (64:1)"
            t_rn.location = (-280, -1240)
            nt.links.new(uv_rust, t_rn.inputs["Vector"])
            sep_rn = nt.nodes.new("ShaderNodeSeparateColor")
            sep_rn.location = (0, -1240)
            nt.links.new(t_rn.outputs["Color"], sep_rn.inputs["Color"])
            micro_nx_cur = _lerp_float(micro_nx_cur, sep_rn.outputs["Red"], rust_mask, 320, -1240)
            micro_ny_cur = _lerp_float(micro_ny_cur, sep_rn.outputs["Green"], rust_mask, 320, -1420)

    # 5. Subtle DXIL Grunge / Oil Grime Layer (`132:1`, 0.5-centered, BEFORE Gamma 2.2!)
    grunge_clr_img = _load_image(tex_dir, titan.get("grunge_clr_file", ""), "Non-Color", img_cache)
    if grunge_clr_img is not None and misc_grunge is not None:
        t_gc = nt.nodes.new("ShaderNodeTexImage")
        t_gc.image = grunge_clr_img
        t_gc.label = "HFW Grunge_CLR (132:1)"
        t_gc.location = (60, 1180)
        nt.links.new(_get_scaled_uv(24.0), t_gc.inputs["Vector"])

        g_x2 = nt.nodes.new("ShaderNodeVectorMath")
        g_x2.operation = "SCALE"
        g_x2.inputs["Scale"].default_value = 1.85
        g_x2.location = (320, 1180)
        nt.links.new(t_gc.outputs["Color"], g_x2.inputs[0])

        g_w = nt.nodes.new("ShaderNodeMath")
        g_w.operation = "MULTIPLY"
        g_w.inputs[1].default_value = 0.40
        g_w.use_clamp = True
        g_w.location = (320, 1020)
        nt.links.new(misc_grunge, g_w.inputs[0])

        g_mul = nt.nodes.new("ShaderNodeMix")
        g_mul.data_type = "RGBA"
        g_mul.blend_type = "MULTIPLY"
        g_mul.clamp_result = True
        g_mul.label = "Apply Grunge (MISC.B)"
        g_mul.location = (520, 580)
        nt.links.new(g_w.outputs["Value"], g_mul.inputs["Factor"])
        nt.links.new(srgb_cur, g_mul.inputs[6])
        nt.links.new(g_x2.outputs["Vector"], g_mul.inputs[7])
        srgb_cur = g_mul.outputs[2]

    # 6. CRITICAL EXACT DXIL `pow(rgb, 2.2)` (ps_1256_1_p1.ll L1953-1961):
    #    Converts Linear-header sRGB-stored Group-132 Paint/Rust into authentic deep gunmetal linear PBR!
    gam_22 = nt.nodes.new("ShaderNodeGamma")
    gam_22.inputs["Gamma"].default_value = 2.2
    gam_22.label = "DXIL pow(rgb, 2.2) -> Linear PBR"
    gam_22.location = (700, 560)
    nt.links.new(srgb_cur, gam_22.inputs["Color"])
    lin_cur = gam_22.outputs["Color"]

    # 7. Apply Baked 4K Curvature (`NMT.A`) & Baked AO (`MacroAO * MicroAO`) in Linear Space
    if macro_crv is not None:
        crv_mr = nt.nodes.new("ShaderNodeMapRange")
        crv_mr.clamp = True
        crv_mr.label = "4K Curvature Edge Highlight"
        crv_mr.inputs["From Min"].default_value = 0.25
        crv_mr.inputs["From Max"].default_value = 0.75
        crv_mr.inputs["To Min"].default_value = 0.80
        crv_mr.inputs["To Max"].default_value = 1.28
        crv_mr.location = (560, 320)
        nt.links.new(macro_crv, crv_mr.inputs["Value"])

        crv_sc = nt.nodes.new("ShaderNodeVectorMath")
        crv_sc.operation = "SCALE"
        crv_sc.location = (740, 360)
        nt.links.new(lin_cur, crv_sc.inputs[0])
        nt.links.new(crv_mr.outputs["Result"], crv_sc.inputs["Scale"])
        lin_cur = crv_sc.outputs["Vector"]

    if macro_ao is not None:
        ao_comb = macro_ao
        if not isinstance(metal_mao, (int, float)):
            ao_m = nt.nodes.new("ShaderNodeMath")
            ao_m.operation = "MULTIPLY"
            ao_m.use_clamp = True
            ao_m.location = (560, 140)
            nt.links.new(macro_ao, ao_m.inputs[0])
            nt.links.new(metal_mao, ao_m.inputs[1])
            ao_comb = ao_m.outputs["Value"]
        ao_mul = nt.nodes.new("ShaderNodeMix")
        ao_mul.data_type = "RGBA"
        ao_mul.blend_type = "MULTIPLY"
        ao_mul.inputs["Factor"].default_value = 0.86
        ao_mul.label = "Apply 4K Macro+Micro AO"
        ao_mul.location = (880, 440)
        nt.links.new(lin_cur, ao_mul.inputs[6])
        nt.links.new(ao_comb, ao_mul.inputs[7])
        lin_cur = ao_mul.outputs[2]

    # 8. Exact DXIL World-Up StarMoss Overgrowth (`1:0/1/2`, AFTER Gamma 2.2 — ONLY on upward-facing surfaces!)
    moss_str = float(titan.get("moss_strength", 0.0))
    moss_clr_img = _load_image(tex_dir, titan.get("moss_clr_file", ""), "Non-Color", img_cache) if moss_str > 0.01 else None
    moss_nmt_img = _load_image(tex_dir, titan.get("moss_nmt_file", ""), "Non-Color", img_cache) if moss_str > 0.01 else None
    if moss_clr_img is not None:
        geom = nt.nodes.new("ShaderNodeNewGeometry")
        geom.location = (180, 1620)
        sep_gn = nt.nodes.new("ShaderNodeSeparateXYZ")
        sep_gn.location = (380, 1620)
        nt.links.new(geom.outputs["Normal"], sep_gn.inputs["Vector"])

        # Only upward-facing carapace/leg tops (`worldNormal.Z > 0.42`) grow jungle moss!
        up_mr = nt.nodes.new("ShaderNodeMapRange")
        up_mr.interpolation_type = "SMOOTHSTEP"
        up_mr.clamp = True
        up_mr.label = "World-Up Moss Slope (Normal.Z)"
        up_mr.inputs["From Min"].default_value = 0.42
        up_mr.inputs["From Max"].default_value = 0.86
        up_mr.inputs["To Min"].default_value = 0.0
        up_mr.inputs["To Max"].default_value = 1.0
        up_mr.location = (560, 1620)
        nt.links.new(sep_gn.outputs["Z"], up_mr.inputs["Value"])

        # Macro world-space 3D patch breakup (replaces DXIL `15:0` 3D noise!)
        ns_m = nt.nodes.new("ShaderNodeTexNoise")
        ns_m.inputs["Scale"].default_value = 0.08
        ns_m.inputs["Detail"].default_value = 5.0
        ns_m.inputs["Roughness"].default_value = 0.62
        ns_m.label = "World Moss Patch Noise (15:0)"
        ns_m.location = (380, 1400)
        nt.links.new(geom.outputs["Position"], ns_m.inputs["Vector"])

        ns_mr = nt.nodes.new("ShaderNodeMapRange")
        ns_mr.interpolation_type = "SMOOTHSTEP"
        ns_mr.clamp = True
        ns_mr.inputs["From Min"].default_value = 0.50
        ns_mr.inputs["From Max"].default_value = 0.78
        ns_mr.inputs["To Min"].default_value = 0.0
        ns_mr.inputs["To Max"].default_value = 1.0
        ns_mr.location = (560, 1400)
        nt.links.new(ns_m.outputs["Fac"], ns_mr.inputs["Value"])

        m_msk = nt.nodes.new("ShaderNodeMath")
        m_msk.operation = "MULTIPLY"
        m_msk.use_clamp = True
        m_msk.location = (740, 1520)
        nt.links.new(up_mr.outputs["Result"], m_msk.inputs[0])
        nt.links.new(ns_mr.outputs["Result"], m_msk.inputs[1])

        m_scl = nt.nodes.new("ShaderNodeMath")
        m_scl.operation = "MULTIPLY"
        m_scl.inputs[1].default_value = moss_str
        m_scl.use_clamp = True
        m_scl.label = "Final Top-Surface Moss Mask"
        m_scl.location = (900, 1520)
        nt.links.new(m_msk.outputs["Value"], m_scl.inputs[0])
        moss_mask = m_scl.outputs["Value"]

        # Lush dark rainforest StarMoss (`MossLUT * StarMoss_clr`, gamma 2.2)
        uv_moss = _get_scaled_uv(18.0)
        t_mc = nt.nodes.new("ShaderNodeTexImage")
        t_mc.image = moss_clr_img
        t_mc.label = "HFW StarMoss_clr (1:1)"
        t_mc.location = (380, 1140)
        nt.links.new(uv_moss, t_mc.inputs["Vector"])

        mlut = titan.get("moss_lut_srgb") or [0.255, 0.355, 0.078]
        m_tint = nt.nodes.new("ShaderNodeVectorMath")
        m_tint.operation = "MULTIPLY"
        m_tint.inputs[1].default_value = (float(mlut[0]) * 1.65, float(mlut[1]) * 1.80, float(mlut[2]) * 1.55)
        m_tint.label = "StarMoss * MossLUT (1:0)"
        m_tint.location = (640, 1140)
        nt.links.new(t_mc.outputs["Color"], m_tint.inputs[0])

        m_gam = nt.nodes.new("ShaderNodeGamma")
        m_gam.inputs["Gamma"].default_value = 2.2
        m_gam.location = (820, 1140)
        nt.links.new(m_tint.outputs["Vector"], m_gam.inputs["Color"])

        lin_cur = _lerp_rgba(lin_cur, m_gam.outputs["Color"], moss_mask, 1040, 520, "Hull <-> Top-Up StarMoss")
        rfl_cur = _lerp_float(rfl_cur, 0.12, moss_mask, 780, -140, "Hull <-> Moss RFL")
        smooth_cur = _lerp_float(smooth_cur, 0.12, moss_mask, 780, -480, "Hull <-> Moss Smoothness")

        if moss_nmt_img is not None:
            t_msn = nt.nodes.new("ShaderNodeTexImage")
            t_msn.image = moss_nmt_img
            t_msn.label = "HFW StarMoss_nmt (1:2)"
            t_msn.location = (380, -1520)
            nt.links.new(uv_moss, t_msn.inputs["Vector"])
            sep_msn = nt.nodes.new("ShaderNodeSeparateColor")
            sep_msn.location = (640, -1520)
            nt.links.new(t_msn.outputs["Color"], sep_msn.inputs["Color"])
            micro_nx_cur = _lerp_float(micro_nx_cur, sep_msn.outputs["Red"], moss_mask, 820, -1420)
            micro_ny_cur = _lerp_float(micro_ny_cur, sep_msn.outputs["Green"], moss_mask, 820, -1600)

    # 9. Exact HFW Decima PBR Conversion (`Smoothness -> Roughness`, `Reflectance -> Metallic & F0 = RFL^2`)
    pbr_grp = nt.nodes.new("ShaderNodeGroup")
    pbr_grp.node_tree = _ensure_decima_pbr_node_group()
    pbr_grp.label = "HFW Decima PBR (Titan Steel/Paint)"
    pbr_grp.location = (1180, 200)
    nt.links.new(lin_cur, pbr_grp.inputs["Color"])
    if isinstance(smooth_cur, (int, float)):
        pbr_grp.inputs["Smoothness"].default_value = float(smooth_cur)
    else:
        nt.links.new(smooth_cur, pbr_grp.inputs["Smoothness"])
    if isinstance(rfl_cur, (int, float)):
        pbr_grp.inputs["Reflectance"].default_value = float(rfl_cur)
    else:
        nt.links.new(rfl_cur, pbr_grp.inputs["Reflectance"])
    pbr_grp.inputs["AO"].default_value = 1.0  # Already folded Macro+Micro AO in linear space!

    nt.links.new(pbr_grp.outputs["Base Color"], bsdf.inputs["Base Color"])
    nt.links.new(pbr_grp.outputs["Metallic"], bsdf.inputs["Metallic"])
    nt.links.new(pbr_grp.outputs["Roughness"], bsdf.inputs["Roughness"])
    if "Specular IOR Level" in bsdf.inputs:
        nt.links.new(pbr_grp.outputs["Specular IOR Level"], bsdf.inputs["Specular IOR Level"])

    # 10. Combine Macro 4K Normal + Micro/Rust/Moss Normal via Tangent-Space UDN Addition
    if macro_nx is not None and macro_ny is not None:
        def _udn_comb(mac_s, mic_s, y: float):
            if isinstance(mic_s, (int, float)):
                return mac_s
            off = nt.nodes.new("ShaderNodeMath")
            off.operation = "MULTIPLY_ADD"
            off.inputs[1].default_value = 0.72
            off.inputs[2].default_value = -0.36
            off.location = (960, y)
            nt.links.new(mic_s, off.inputs[0])
            sm = nt.nodes.new("ShaderNodeMath")
            sm.operation = "ADD"
            sm.use_clamp = True
            sm.location = (1120, y)
            nt.links.new(mac_s, sm.inputs[0])
            nt.links.new(off.outputs["Value"], sm.inputs[1])
            return sm.outputs["Value"]

        nx_fin = _udn_comb(macro_nx, micro_nx_cur, -1180)
        ny_fin = _udn_comb(macro_ny, micro_ny_cur, -1380)
        nrm_rec = nt.nodes.new("ShaderNodeGroup")
        nrm_rec.node_tree = _ensure_normal_reconstruct_z_group()
        nrm_rec.inputs["Flip Green"].default_value = 0.0
        nrm_rec.location = (1260, -1240)
        nt.links.new(nx_fin, nrm_rec.inputs["Normal X"])
        nt.links.new(ny_fin, nrm_rec.inputs["Normal Y"])
        nrm_map = nt.nodes.new("ShaderNodeNormalMap")
        nrm_map.location = (1440, -1240)
        nt.links.new(nrm_rec.outputs["Normal RGB"], nrm_map.inputs["Color"])
        nt.links.new(nrm_map.outputs["Normal"], bsdf.inputs["Normal"])

    # 11. Optional DXIL Deep-Crevice Molten Heat Glow on Horus Heat-Sinks / InteractivePlates / Drill Weakspots!
    heat_def = float(titan.get("heat_glow_default", 0.0))
    if heat_def > 0.01 and macro_ao is not None and "Emission Color" in bsdf.inputs:
        mctrl = nt.nodes.new("ShaderNodeGroup")
        mctrl.node_tree = _ensure_machine_shader_control_group()
        mctrl.label = "HFW Machine Shader Control"
        mctrl.location = (740, -760)

        # Exact DXIL `(1 - AO)^4` deep crevice glow: cooling vents & drill canisters burn hot inside recesses!
        h_inv = nt.nodes.new("ShaderNodeMapRange")
        h_inv.clamp = True
        h_inv.label = "Crevice Heat (1 - MacroAO)"
        h_inv.inputs["From Min"].default_value = 0.04
        h_inv.inputs["From Max"].default_value = 0.68
        h_inv.inputs["To Min"].default_value = 1.0
        h_inv.inputs["To Max"].default_value = 0.0
        h_inv.location = (780, -960)
        nt.links.new(macro_ao, h_inv.inputs["Value"])

        h_pow = nt.nodes.new("ShaderNodeMath")
        h_pow.operation = "POWER"
        h_pow.inputs[1].default_value = 3.2
        h_pow.use_clamp = True
        h_pow.location = (960, -960)
        nt.links.new(h_inv.outputs["Result"], h_pow.inputs[0])

        t_drv = _add_time_driver_node(nt, (540, -1080), "HFW Titan Heat Pulse Time")
        t_f = nt.nodes.new("ShaderNodeMath")
        t_f.operation = "MULTIPLY"
        t_f.inputs[1].default_value = 2.4
        t_f.location = (720, -1080)
        nt.links.new(t_drv.outputs[0], t_f.inputs[0])
        s_p = nt.nodes.new("ShaderNodeMath")
        s_p.operation = "SINE"
        s_p.location = (860, -1080)
        nt.links.new(t_f.outputs["Value"], s_p.inputs[0])
        p_mod = nt.nodes.new("ShaderNodeMath")
        p_mod.operation = "MULTIPLY_ADD"
        p_mod.inputs[1].default_value = 0.16
        p_mod.inputs[2].default_value = 0.88
        p_mod.location = (1000, -1080)
        nt.links.new(s_p.outputs["Value"], p_mod.inputs[0])

        h_m1 = nt.nodes.new("ShaderNodeMath")
        h_m1.operation = "MULTIPLY"
        h_m1.location = (1140, -960)
        nt.links.new(h_pow.outputs["Value"], h_m1.inputs[0])
        nt.links.new(p_mod.outputs["Value"], h_m1.inputs[1])

        h_str = nt.nodes.new("ShaderNodeMath")
        h_str.operation = "MULTIPLY"
        h_str.inputs[1].default_value = heat_def * 9.0
        h_str.location = (1280, -960)
        nt.links.new(h_m1.outputs["Value"], h_str.inputs[0])

        h_fin = nt.nodes.new("ShaderNodeMath")
        h_fin.operation = "MULTIPLY"
        h_fin.location = (1400, -920)
        nt.links.new(h_str.outputs["Value"], h_fin.inputs[0])
        nt.links.new(mctrl.outputs["Muscle Glow Scale"], h_fin.inputs[1])

        hc_deep = titan.get("heat_color_deep") or [0.75, 0.03, 0.002]
        h_col = nt.nodes.new("ShaderNodeMix")
        h_col.data_type = "RGBA"
        h_col.label = "Deep Crimson -> Molten Orange"
        h_col.inputs[6].default_value = (hc_deep[0], hc_deep[1], hc_deep[2], 1.0)
        h_col.location = (1140, -760)
        nt.links.new(h_pow.outputs["Value"], h_col.inputs["Factor"])
        nt.links.new(mctrl.outputs["Muscle Color"], h_col.inputs[7])

        nt.links.new(h_col.outputs[2], bsdf.inputs["Emission Color"])
        if "Emission Strength" in bsdf.inputs:
            nt.links.new(h_fin.outputs["Value"], bsdf.inputs["Emission Strength"])

    return True


def _build_material(
    mat_meta: dict[str, Any],
    tex_dir: Path,
    img_cache: dict[str, bpy.types.Image],
    enable_detail_nodes: bool = True,
) -> bpy.types.Material:
    mid = mat_meta["material_id"]
    mname = f"{mat_meta['name']}_{mid.replace(':', '_')}"
    existing = bpy.data.materials.get(mname)
    if existing is not None:
        return existing

    mat = bpy.data.materials.new(name=mname)
    mat.use_nodes = True
    mat["hfw_material_id"] = mid
    if mat_meta.get("shader_id"):
        mat["hfw_shader_id"] = mat_meta["shader_id"]
    mat["hfw_tags"] = ",".join(mat_meta.get("tags", ()))
    mat["hfw_texture_sets"] = ",".join(mat_meta.get("texture_sets", ()))
    spec_shader = mat_meta.get("special_shader")
    if spec_shader:
        mat["hfw_special_shader"] = spec_shader.get("kind", "")

    nt = mat.node_tree
    nt.nodes.clear()

    # Gather semantic sockets from EITHER native packed game textures (default) OR unpacked PNGs!
    sockets: dict[str, Any] = {}  # semantic key -> Blender NodeSocket
    packed_entries: list[dict[str, Any]] = mat_meta.get("packed_entries", [])
    maps: dict[str, str] = mat_meta.get("maps", {})

    if packed_entries:
        # Native game-packed textures: separate RGBA channels directly inside Blender shader nodes!
        y_cursor = 480
        lane_out_name = {"R": "Red", "G": "Green", "B": "Blue"}

        for p_idx, pe in enumerate(packed_entries):
            packing: dict[str, str] = pe.get("packing", {})
            cspace = pe.get("color_space", "Non-Color")
            # Force sRGB for pure Color/Incandescence textures, Non-Color for data/packed textures
            types_present = {v.split(".", 1)[0] for v in packing.values()}
            if types_present <= {"Color", "Incandescence", "Alpha"} and "Color" in types_present:
                cspace = "sRGB"
            else:
                cspace = "Non-Color"

            img = _load_image(tex_dir, pe.get("filename", ""), cspace, img_cache)
            if img is None:
                continue

            tex_node = nt.nodes.new("ShaderNodeTexImage")
            tex_node.image = img
            tag_lbl = "/".join(sorted(types_present)) or f"Packed_{p_idx}"
            tex_node.label = f"HFW {tag_lbl}"
            tex_node.location = (-960, y_cursor)
            uv_idx = int(pe.get("uv_index", 0) or 0)
            if uv_idx > 0:
                uv_n = nt.nodes.new("ShaderNodeUVMap")
                uv_n.uv_map = f"UV{uv_idx}"
                uv_n.location = (-1180, y_cursor)
                nt.links.new(uv_n.outputs["UV"], tex_node.inputs["Vector"])

            # Check if RGB is a direct full Color / Incandescence triplet
            r_t = packing.get("R", "").split(".")[0]
            g_t = packing.get("G", "").split(".")[0]
            b_t = packing.get("B", "").split(".")[0]
            if r_t == g_t == b_t and r_t in ("Color", "Incandescence"):
                sockets.setdefault(r_t, tex_node.outputs["Color"])
                if "A" in packing:
                    a_t = packing["A"].split(".")[0]
                    sockets.setdefault(a_t, tex_node.outputs["Alpha"])
            else:
                sep = nt.nodes.new("ShaderNodeSeparateColor")
                sep.label = f"Unpack {tag_lbl}"
                sep.location = (-660, y_cursor)
                nt.links.new(tex_node.outputs["Color"], sep.inputs["Color"])
                # Handle Decima 2-channel Rock/Landscape Color packing (`R=Color.R, G=Color.G, B=AO.R, A=Curvature.R`)
                if r_t == "Color" and g_t == "Color" and b_t != "Color" and "Color" not in sockets:
                    comb_rk = nt.nodes.new("ShaderNodeCombineColor")
                    comb_rk.label = "Reconstruct Rock Albedo (RG)"
                    comb_rk.location = (-440, y_cursor + 60)
                    avg_rg = nt.nodes.new("ShaderNodeMath")
                    avg_rg.operation = "MULTIPLY_ADD"
                    avg_rg.inputs[1].default_value = 0.5
                    avg_rg.location = (-560, y_cursor - 20)
                    nt.links.new(sep.outputs["Red"], avg_rg.inputs[0])
                    nt.links.new(sep.outputs["Green"], avg_rg.inputs[2])
                    avg_rg2 = nt.nodes.new("ShaderNodeMath")
                    avg_rg2.operation = "MULTIPLY"
                    avg_rg2.inputs[1].default_value = 0.48
                    avg_rg2.location = (-460, y_cursor - 20)
                    nt.links.new(avg_rg.outputs["Value"], avg_rg2.inputs[0])
                    nt.links.new(sep.outputs["Red"], comb_rk.inputs["Red"])
                    nt.links.new(sep.outputs["Green"], comb_rk.inputs["Green"])
                    nt.links.new(avg_rg2.outputs["Value"], comb_rk.inputs["Blue"])
                    # Tone rock from bright linear mask into natural stone albedo
                    rk_tint = nt.nodes.new("ShaderNodeMix")
                    rk_tint.data_type = "RGBA"
                    rk_tint.blend_type = "MULTIPLY"
                    rk_tint.inputs["Factor"].default_value = 1.0
                    rk_tint.inputs[7].default_value = (0.34, 0.30, 0.27, 1.0)
                    rk_tint.location = (-260, y_cursor + 60)
                    nt.links.new(comb_rk.outputs["Color"], rk_tint.inputs[6])
                    sockets["Color"] = rk_tint.outputs[2]
                for lane, spec in packing.items():
                    parts = spec.split(".")
                    stype = parts[0]
                    sch = parts[1] if len(parts) > 1 else "R"
                    out_sock = tex_node.outputs["Alpha"] if lane == "A" else sep.outputs[lane_out_name[lane]]
                    if stype == "Normal":
                        sockets.setdefault(f"Normal.{sch}", out_sock)
                    else:
                        sockets.setdefault(stype, out_sock)
                        sockets.setdefault(f"{stype}.{sch}", out_sock)

            y_cursor -= 320
    else:
        # Optional unpacked mode (`--unpack-textures`): direct single-purpose semantic PNGs
        if (img := _load_image(tex_dir, maps.get("Color", ""), "sRGB", img_cache)) is not None:
            n = nt.nodes.new("ShaderNodeTexImage")
            n.image = img
            n.label = "HFW Color"
            n.location = (-960, 420)
            sockets["Color"] = n.outputs["Color"]
        if (img := _load_image(tex_dir, maps.get("AO", ""), "Non-Color", img_cache)) is not None:
            n = nt.nodes.new("ShaderNodeTexImage")
            n.image = img
            n.label = "HFW AO"
            n.location = (-960, 140)
            sockets["AO"] = n.outputs["Color"]
        if (img := _load_image(tex_dir, maps.get("Roughness", ""), "Non-Color", img_cache)) is not None:
            n = nt.nodes.new("ShaderNodeTexImage")
            n.image = img
            n.label = "HFW Smoothness (RGH)"
            n.location = (-960, -140)
            sockets["Roughness"] = n.outputs["Color"]
        if (img := _load_image(tex_dir, maps.get("Reflectance", ""), "Non-Color", img_cache)) is not None:
            n = nt.nodes.new("ShaderNodeTexImage")
            n.image = img
            n.label = "HFW Reflectance"
            n.location = (-960, -420)
            sockets["Reflectance"] = n.outputs["Color"]
        if (img := _load_image(tex_dir, maps.get("Mask", ""), "Non-Color", img_cache)) is not None:
            n = nt.nodes.new("ShaderNodeTexImage")
            n.image = img
            n.label = "HFW Mask"
            n.location = (-960, -700)
            sockets["Mask"] = n.outputs["Color"]
        if (img := _load_image(tex_dir, maps.get("Mask_Alpha", ""), "Non-Color", img_cache)) is not None:
            n = nt.nodes.new("ShaderNodeTexImage")
            n.image = img
            n.label = "HFW Mask Alpha"
            n.location = (-960, -980)
            sockets["Mask_Alpha"] = n.outputs["Color"]
        # Prefer Normal_DX (unflipped Y) because Decima UVs are already V-up matching Blender!
        nrm_file = maps.get("Normal_DX") or maps.get("Normal_GL") or ""
        if (img := _load_image(tex_dir, nrm_file, "Non-Color", img_cache)) is not None:
            n = nt.nodes.new("ShaderNodeTexImage")
            n.image = img
            n.label = "HFW Normal"
            n.location = (-960, -1260)
            sockets["Normal_RGB"] = n.outputs["Color"]
        if (img := _load_image(tex_dir, maps.get("Incandescence", ""), "sRGB", img_cache)) is not None:
            n = nt.nodes.new("ShaderNodeTexImage")
            n.image = img
            n.label = "HFW Incandescence"
            n.location = (-960, 700)
            sockets["Incandescence"] = n.outputs["Color"]
        if (img := _load_image(tex_dir, maps.get("Alpha", ""), "Non-Color", img_cache)) is not None:
            n = nt.nodes.new("ShaderNodeTexImage")
            n.image = img
            n.label = "HFW Alpha"
            n.location = (-960, 980)
            sockets["Alpha"] = n.outputs["Color"]

    # Check if this material uses one of Guerrilla's specialized muscle / glow / core / eye / glass / VFX shaders!
    if spec_shader and _build_special_material(mat, spec_shader, sockets, tex_dir, img_cache):
        return mat

    # Check if this is an HFW 8-Layer Character Outfit Material (`MSK.RGB` 3-bit corner mask + tiling Leather/Metal/Silk/Carbon)!
    outfit_layered = mat_meta.get("outfit_layered")
    if outfit_layered and outfit_layered.get("layers"):
        if _build_layered_outfit_material(mat, outfit_layered, tex_dir, img_cache):
            return mat

    # Check if this is a Horus / FAS-BOR7 Titan 26-TB Group-132 Layered Hull Material!
    titan_layered = mat_meta.get("titan_layered")
    if titan_layered and titan_layered.get("corners"):
        if _build_titan_layered_material(mat, titan_layered, tex_dir, img_cache):
            return mat

    # Standard Decima Machine/Armor Deferred PBR Shader (`HFW_MachineDetailArray` -> `HFW_DecimaPBR` -> `Principled BSDF`)
    out_node = nt.nodes.new("ShaderNodeOutputMaterial")
    out_node.location = (820, 0)

    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (480, 0)
    nt.links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])

    # 1. Normal Map RGB (reconstruct Z from packed RG if in packed mode)
    normal_rgb_sock = sockets.get("Normal_RGB")
    if normal_rgb_sock is None and "Normal.R" in sockets and "Normal.G" in sockets:
        nrm_rec = nt.nodes.new("ShaderNodeGroup")
        nrm_rec.node_tree = _ensure_normal_reconstruct_z_group()
        nrm_rec.label = "Reconstruct Normal Z"
        nrm_rec.inputs["Flip Green"].default_value = 0.0
        nrm_rec.location = (-380, -480)
        nt.links.new(sockets["Normal.R"], nrm_rec.inputs["Normal X"])
        nt.links.new(sockets["Normal.G"], nrm_rec.inputs["Normal Y"])
        normal_rgb_sock = nrm_rec.outputs["Normal RGB"]

    color_sock = sockets.get("Color")
    smooth_sock = sockets.get("Roughness")  # Decima `Roughness` channel IS Smoothness (1 - PBR Roughness)!
    ao_sock = sockets.get("AO")
    rfl_sock = sockets.get("Reflectance")

    # 2. Mask-Driven 23-Slice Machine Detail Array (Guerrilla `20:0` Texture2DArray!)
    detail_arr = mat_meta.get("detail_array")
    if (
        enable_detail_nodes
        and detail_arr
        and normal_rgb_sock is not None
        and "Mask" in sockets
    ):
        atlas_img = _load_image(tex_dir, detail_arr.get("filename", ""), "Non-Color", img_cache)
        if atlas_img is not None:
            da_grp = nt.nodes.new("ShaderNodeGroup")
            da_grp.node_tree = _ensure_machine_detail_array_group(
                atlas_img,
                cols=int(detail_arr.get("cols", 5)),
                rows=int(detail_arr.get("rows", 5)),
            )
            da_grp.label = "HFW Machine Detail Array (Mask -> 23 Slices)"
            da_grp.location = (-160, -140)
            base_tx = float(detail_arr.get("tile_x", 28.0))
            base_ty = float(detail_arr.get("tile_y", 28.0))
            # Wire live Detail Tiling Scale & Detail Strength from HFW_MachineShaderControl!
            dm_ctrl = nt.nodes.new("ShaderNodeGroup")
            dm_ctrl.node_tree = _ensure_machine_shader_control_group()
            dm_ctrl.label = "HFW Detail Scale Control"
            dm_ctrl.location = (-660, -260)
            if "Detail Tiling Scale" in dm_ctrl.outputs:
                mul_tx = nt.nodes.new("ShaderNodeMath")
                mul_tx.operation = "MULTIPLY"
                mul_tx.inputs[0].default_value = base_tx
                mul_tx.location = (-400, -180)
                nt.links.new(dm_ctrl.outputs["Detail Tiling Scale"], mul_tx.inputs[1])
                nt.links.new(mul_tx.outputs["Value"], da_grp.inputs["Tile X"])

                mul_ty = nt.nodes.new("ShaderNodeMath")
                mul_ty.operation = "MULTIPLY"
                mul_ty.inputs[0].default_value = base_ty
                mul_ty.location = (-400, -320)
                nt.links.new(dm_ctrl.outputs["Detail Tiling Scale"], mul_ty.inputs[1])
                nt.links.new(mul_ty.outputs["Value"], da_grp.inputs["Tile Y"])
            else:
                da_grp.inputs["Tile X"].default_value = base_tx
                da_grp.inputs["Tile Y"].default_value = base_ty
            if "Detail Strength" in dm_ctrl.outputs:
                nt.links.new(dm_ctrl.outputs["Detail Strength"], da_grp.inputs["Detail Strength"])
            else:
                da_grp.inputs["Detail Strength"].default_value = 0.72
            if color_sock is not None:
                nt.links.new(color_sock, da_grp.inputs["Base Color"])
                color_sock = da_grp.outputs["Color"]
            if smooth_sock is not None:
                nt.links.new(smooth_sock, da_grp.inputs["Base Smoothness"])
                smooth_sock = da_grp.outputs["Smoothness"]
            nt.links.new(normal_rgb_sock, da_grp.inputs["Base Normal RGB"])
            nt.links.new(sockets["Mask"], da_grp.inputs["Mask"])
            if "Mask_Alpha" in sockets:
                nt.links.new(sockets["Mask_Alpha"], da_grp.inputs["Mask Alpha"])
            normal_rgb_sock = da_grp.outputs["Normal RGB"]

    # 2b. DXIL-Verified Animated Elemental Sack / Canister / Eye Lens / Plasma Overlays!
    em_overlay = mat_meta.get("emissive_overlay")
    overlay_em_color = None
    overlay_em_strength = None
    liquid_mask_sock = None

    if em_overlay:
        okind = em_overlay.get("kind", "")
        mat["hfw_emissive_overlay"] = okind

        if okind == "elemental_sack_overlay" and ("Mask" in sockets or float(em_overlay.get("mask_id", 0.0)) < 0.0):
            flow_img = _load_image(tex_dir, em_overlay.get("flow_file", ""), "Non-Color", img_cache)
            fluid_img = _load_image(tex_dir, em_overlay.get("fluid_file", ""), "Non-Color", img_cache)
            t_drv = _add_time_driver_node(nt, (-280, 560), "HFW Sac Flow Time")
            mctrl = nt.nodes.new("ShaderNodeGroup")
            mctrl.node_tree = _ensure_machine_shader_control_group()
            mctrl.label = "HFW Machine Shader Control"
            mctrl.location = (-520, 740)

            sac_grp = nt.nodes.new("ShaderNodeGroup")
            sac_grp.node_tree = _ensure_elemental_sack_group(flow_img, fluid_img)
            sac_grp.label = "HFW Animated Elemental Sac (Flow Map)"
            sac_grp.location = (40, 520)
            gc = em_overlay.get("glow_color", [0.14, 0.68, 1.0])
            base_sac_str = float(em_overlay.get("glow_strength", 4.6))
            sac_grp.inputs["Flow Speed"].default_value = float(em_overlay.get("flow_speed", 0.35))
            sac_grp.inputs["Flow Strength"].default_value = float(em_overlay.get("flow_strength", 0.14))
            if "Mask ID" in sac_grp.inputs:
                sac_grp.inputs["Mask ID"].default_value = float(em_overlay.get("mask_id", 0.0))

            # Preserve per-sac elemental color (e.g. Bilegut Acid + Adhesive) with optional global override!
            liq_col = nt.nodes.new("ShaderNodeMix")
            liq_col.data_type = "RGBA"
            liq_col.label = "Native Sac Color <-> Override"
            liq_col.inputs[6].default_value = (gc[0], gc[1], gc[2], 1.0)
            liq_col.location = (-220, 760)
            nt.links.new(mctrl.outputs["Liquid Override Mix"], liq_col.inputs["Factor"])
            nt.links.new(mctrl.outputs["Liquid Override Color"], liq_col.inputs[7])
            nt.links.new(liq_col.outputs[2], sac_grp.inputs["Sac Glow Color"])

            sac_str = nt.nodes.new("ShaderNodeMath")
            sac_str.operation = "MULTIPLY"
            sac_str.inputs[0].default_value = base_sac_str
            sac_str.location = (-180, 620)
            nt.links.new(mctrl.outputs["Liquid Glow Scale"], sac_str.inputs[1])
            nt.links.new(sac_str.outputs["Value"], sac_grp.inputs["Glow Strength"])
            nt.links.new(mctrl.outputs["Animation Speed"], sac_grp.inputs["Animate Speed"])

            if color_sock is not None:
                nt.links.new(color_sock, sac_grp.inputs["Base Color"])
                color_sock = sac_grp.outputs["Color"]
            if smooth_sock is not None:
                nt.links.new(smooth_sock, sac_grp.inputs["Base Smoothness"])
                smooth_sock = sac_grp.outputs["Smoothness"]
            if "Mask" in sockets:
                nt.links.new(sockets["Mask"], sac_grp.inputs["Mask"])
            if "Mask_Alpha" in sockets:
                nt.links.new(sockets["Mask_Alpha"], sac_grp.inputs["Mask Alpha"])
            nt.links.new(t_drv.outputs[0], sac_grp.inputs["Time"])
            overlay_em_color = sac_grp.outputs["Emission Color"]
            overlay_em_strength = sac_grp.outputs["Emission Strength"]
            liquid_mask_sock = sac_grp.outputs["Sac Mask"]

        elif okind == "canister_emissive_overlay" and "Mask" in sockets:
            noise_img = _load_image(tex_dir, em_overlay.get("noise_file", ""), "Non-Color", img_cache)
            t_drv = _add_time_driver_node(nt, (-280, 560), "HFW Canister Pulse Time")
            mctrl = nt.nodes.new("ShaderNodeGroup")
            mctrl.node_tree = _ensure_machine_shader_control_group()
            mctrl.label = "HFW Machine Shader Control"
            mctrl.location = (-520, 740)

            can_grp = nt.nodes.new("ShaderNodeGroup")
            can_grp.node_tree = _ensure_canister_glow_group(noise_img)
            can_grp.label = "HFW Animated Elemental Canister Glow"
            can_grp.location = (40, 520)
            gc = em_overlay.get("glow_color", [0.52, 0.70, 0.10])
            base_can_str = float(em_overlay.get("glow_strength", 5.5))
            can_grp.inputs["Pulse Speed"].default_value = float(em_overlay.get("pulse_speed", 8.0))
            can_grp.inputs["Is Fluid Body"].default_value = 1.0 if em_overlay.get("is_fluid_body") else 0.0

            liq_col = nt.nodes.new("ShaderNodeMix")
            liq_col.data_type = "RGBA"
            liq_col.inputs[6].default_value = (gc[0], gc[1], gc[2], 1.0)
            liq_col.location = (-220, 760)
            nt.links.new(mctrl.outputs["Liquid Override Mix"], liq_col.inputs["Factor"])
            nt.links.new(mctrl.outputs["Liquid Override Color"], liq_col.inputs[7])
            nt.links.new(liq_col.outputs[2], can_grp.inputs["Glow Color"])

            can_str = nt.nodes.new("ShaderNodeMath")
            can_str.operation = "MULTIPLY"
            can_str.inputs[0].default_value = base_can_str
            can_str.location = (-180, 620)
            nt.links.new(mctrl.outputs["Liquid Glow Scale"], can_str.inputs[1])
            nt.links.new(can_str.outputs["Value"], can_grp.inputs["Glow Strength"])
            nt.links.new(mctrl.outputs["Animation Speed"], can_grp.inputs["Animate Speed"])

            if color_sock is not None:
                nt.links.new(color_sock, can_grp.inputs["Base Color"])
                color_sock = can_grp.outputs["Color"]
            nt.links.new(sockets["Mask"], can_grp.inputs["Mask"])
            nt.links.new(t_drv.outputs[0], can_grp.inputs["Time"])
            overlay_em_color = can_grp.outputs["Emission Color"]
            overlay_em_strength = can_grp.outputs["Emission Strength"]
            liquid_mask_sock = can_grp.outputs["Fluid Mask"]

        elif okind == "machine_eye_lens_overlay":
            iris_img = _load_image(tex_dir, em_overlay.get("iris_file", ""), "Non-Color", img_cache)
            t_drv = _add_time_driver_node(nt, (-280, 560), "HFW Eye Iris Time")
            mctrl = nt.nodes.new("ShaderNodeGroup")
            mctrl.node_tree = _ensure_machine_shader_control_group()
            mctrl.label = "HFW Machine Shader Control"
            mctrl.location = (-520, 740)

            eye_grp = nt.nodes.new("ShaderNodeGroup")
            eye_grp.node_tree = _ensure_eye_lens_group(iris_img)
            eye_grp.label = "HFW Machine Eye Iris Optics"
            eye_grp.location = (40, 520)
            base_eye_str = float(em_overlay.get("glow_strength", 9.0))
            nt.links.new(mctrl.outputs["Eye Color"], eye_grp.inputs["Calm Color"])
            nt.links.new(mctrl.outputs["Eye Alert Color"], eye_grp.inputs["Alert Color"])
            nt.links.new(mctrl.outputs["Eye Alert State"], eye_grp.inputs["Alert State"])
            nt.links.new(mctrl.outputs["Animation Speed"], eye_grp.inputs["Animate Speed"])
            eye_str = nt.nodes.new("ShaderNodeMath")
            eye_str.operation = "MULTIPLY"
            eye_str.inputs[0].default_value = base_eye_str
            eye_str.location = (-180, 620)
            nt.links.new(mctrl.outputs["Eye Glow Scale"], eye_str.inputs[1])
            nt.links.new(eye_str.outputs["Value"], eye_grp.inputs["Glow Strength"])
            nt.links.new(t_drv.outputs[0], eye_grp.inputs["Time"])
            overlay_em_color = eye_grp.outputs["Emission Color"]
            overlay_em_strength = eye_grp.outputs["Emission Strength"]

        elif okind == "plasma_charge_overlay":
            noise_img = _load_image(tex_dir, em_overlay.get("noise_file", ""), "Non-Color", img_cache)
            t_drv = _add_time_driver_node(nt, (-260, 560), "HFW Plasma Charge Time")
            pctrl = nt.nodes.new("ShaderNodeGroup")
            pctrl.node_tree = _ensure_plasma_charge_control_group()
            pctrl.label = "Slaughterspine Plasma Charge"
            pctrl.location = (-260, 420)
            mctrl = nt.nodes.new("ShaderNodeGroup")
            mctrl.node_tree = _ensure_machine_shader_control_group()
            mctrl.label = "HFW Machine Shader Control"
            mctrl.location = (-520, 740)
            pc_grp = nt.nodes.new("ShaderNodeGroup")
            pc_grp.node_tree = _ensure_plasma_charge_group(noise_img)
            pc_grp.label = "HFW Slaughterspine Plasma Charge-Up (DXIL)"
            pc_grp.location = (40, 480)
            pc_grp.inputs["Glow Strength"].default_value = float(em_overlay.get("glow_strength", 4.2))
            nt.links.new(mctrl.outputs["Plasma Color"], pc_grp.inputs["Plasma Color"])
            nt.links.new(mctrl.outputs["Muscle Color"], pc_grp.inputs["Idle Conduit Color"])
            nt.links.new(mctrl.outputs["Animation Speed"], pc_grp.inputs["Animate Speed"])
            if "Mask" in sockets:
                nt.links.new(sockets["Mask"], pc_grp.inputs["Mask"])
            if "Mask_Alpha" in sockets:
                nt.links.new(sockets["Mask_Alpha"], pc_grp.inputs["Mask Alpha"])
            if ao_sock is not None:
                nt.links.new(ao_sock, pc_grp.inputs["AO"])
            nt.links.new(t_drv.outputs[0], pc_grp.inputs["Time"])
            nt.links.new(pctrl.outputs["Plasma Charge"], pc_grp.inputs["Plasma Charge"])
            overlay_em_color = pc_grp.outputs["Emission Color"]
            overlay_em_strength = pc_grp.outputs["Emission Strength"]

    # 2c. Exact DXIL Human Skin Tone LUT Reconstruction (`AloyHead 48346:288` L855-912 + L2456-2464)!
    #     Guerrilla stores human `Head_CLR` / `Female_Aloy_CLR` as `0.5`-centered neutralized variation maps
    #     (`mean_srgb ≈ 0.50`) and multiplies by `2.0 * SkinLUT_srgb` BEFORE `pow(..., 2.2)` -> in linear PBR
    #     space, `pow(2.0 * SkinLUT_srgb * Head_CLR_srgb, 2.2) == 4.5948 * SkinTone_linear * Head_CLR_linear`!
    sprof = mat_meta.get("surface_profile", "pbr")
    char_ctrl_node = None
    if sprof == "skin" and mat_meta.get("uses_skin_lut") and color_sock is not None:
        char_ctrl_node = nt.nodes.new("ShaderNodeGroup")
        char_ctrl_node.node_tree = _ensure_character_control_group(
            {"skin_tone_linear": mat_meta.get("skin_tone_linear")} if mat_meta.get("skin_tone_linear") else None
        )
        char_ctrl_node.label = "HFW Character Skin Control (570:0 LUT)"
        char_ctrl_node.location = (-440, 520)

        # Expand 0.5-centered sRGB skin texture (linear 0.2176 -> 1.000) via exact `2.0^2.2 = 4.5948`!
        sk_exp = nt.nodes.new("ShaderNodeVectorMath")
        sk_exp.operation = "SCALE"
        sk_exp.inputs["Scale"].default_value = 4.5948
        sk_exp.label = "Neutral-0.5 Skin Expand (2^2.2)"
        sk_exp.location = (-220, 340)
        nt.links.new(color_sock, sk_exp.inputs[0])

        sk_br = nt.nodes.new("ShaderNodeVectorMath")
        sk_br.operation = "SCALE"
        sk_br.label = "Apply Skin Brightness"
        sk_br.location = (-60, 420)
        nt.links.new(char_ctrl_node.outputs["Skin Tone"], sk_br.inputs[0])
        nt.links.new(char_ctrl_node.outputs["Skin Brightness"], sk_br.inputs["Scale"])

        sk_mul = nt.nodes.new("ShaderNodeMix")
        sk_mul.data_type = "RGBA"
        sk_mul.blend_type = "MULTIPLY"
        sk_mul.clamp_result = True
        sk_mul.inputs["Factor"].default_value = 1.0
        sk_mul.label = "HFW DXIL Skin Albedo (SkinTone * 4.595 * Tex)"
        sk_mul.location = (20, 360)
        nt.links.new(sk_br.outputs["Vector"], sk_mul.inputs[6])
        nt.links.new(sk_exp.outputs["Vector"], sk_mul.inputs[7])
        color_sock = sk_mul.outputs[2]
    elif sprof in ("hair", "hair_face") and color_sock is None:
        # Guerrilla non-Aloy hair cards (e.g. Seyka `QuenHair_Alpha`, Companion/NPC hair, brows/lashes) bind
        # `382:0` Hair LUT with NO RGB Color map -> drive cleanly from `HFW_CharacterControl.Hair Color` + strand ID!
        char_ctrl_node = nt.nodes.new("ShaderNodeGroup")
        hc_init = {"hair_color_linear": mat_meta.get("hair_color_linear")} if mat_meta.get("hair_color_linear") else None
        char_ctrl_node.node_tree = _ensure_character_control_group(hc_init)
        char_ctrl_node.label = "HFW Character Hair Control (382:0 LUT)"
        char_ctrl_node.location = (-440, 480)
        hair_col_sock = char_ctrl_node.outputs["Hair Color"]
        if "Mask_Alpha" in sockets:
            # Subtle per-strand root/ID luminance variation (0.76..1.24x) from `*_ID.png`
            id_rng = nt.nodes.new("ShaderNodeMapRange")
            id_rng.clamp = True
            id_rng.inputs["From Min"].default_value = 0.0
            id_rng.inputs["From Max"].default_value = 1.0
            id_rng.inputs["To Min"].default_value = 0.76
            id_rng.inputs["To Max"].default_value = 1.24
            id_rng.label = "Hair Strand Variation"
            id_rng.location = (-240, 340)
            nt.links.new(sockets["Mask_Alpha"], id_rng.inputs["Value"])

            h_var = nt.nodes.new("ShaderNodeVectorMath")
            h_var.operation = "SCALE"
            h_var.label = "Modulate Hair Strands"
            h_var.location = (-40, 420)
            nt.links.new(hair_col_sock, h_var.inputs[0])
            nt.links.new(id_rng.outputs["Result"], h_var.inputs["Scale"])
            color_sock = h_var.outputs["Vector"]
        else:
            color_sock = hair_col_sock
    elif sprof in ("rock_volcanic", "rock_tint") and color_sock is not None:
        # HFW landscape rock `_TNT` textures are 2-channel Linear RG Tint/Blend maps (`B == 0`!) —
        # NEVER feed raw RG into Base Color (which renders neon green/orange!). Reconstruct authentic
        # volcanic basalt or weathered sandstone/granite from Red (macro shade) + Green (ash/weathering blend)!
        sep_tnt = nt.nodes.new("ShaderNodeSeparateColor")
        sep_tnt.label = "Unpack Rock _TNT (R=Shade, G=Weathering)"
        sep_tnt.location = (-360, 440)
        nt.links.new(color_sock, sep_tnt.inputs["Color"])

        if sprof == "rock_volcanic":
            c_dark = (0.022, 0.021, 0.024, 1.0)    # Deep cooled volcanic basalt charcoal
            c_mid = (0.072, 0.068, 0.066, 1.0)     # Weathered obsidian / grey lava ridge
            c_dust = (0.135, 0.115, 0.098, 1.0)    # Warm volcanic ash / mineral dust
            dust_w = 0.42
        else:
            c_dark = (0.085, 0.070, 0.058, 1.0)    # Shadowed canyon/cliff stone
            c_mid = (0.245, 0.195, 0.155, 1.0)     # Weathered desert sandstone / granite
            c_dust = (0.175, 0.160, 0.118, 1.0)    # Muted dusty lichen / ochre mineral
            dust_w = 0.35

        m_rock = nt.nodes.new("ShaderNodeMix")
        m_rock.data_type = "RGBA"
        m_rock.label = "Rock Macro Shade (TNT.R)"
        m_rock.inputs[6].default_value = c_dark
        m_rock.inputs[7].default_value = c_mid
        m_rock.location = (-160, 480)
        nt.links.new(sep_tnt.outputs["Red"], m_rock.inputs["Factor"])

        g_scl = nt.nodes.new("ShaderNodeMath")
        g_scl.operation = "MULTIPLY"
        g_scl.inputs[1].default_value = dust_w
        g_scl.use_clamp = True
        g_scl.location = (-160, 300)
        nt.links.new(sep_tnt.outputs["Green"], g_scl.inputs[0])

        m_dust = nt.nodes.new("ShaderNodeMix")
        m_dust.data_type = "RGBA"
        m_dust.label = "Rock Weathering Blend (TNT.G)"
        m_dust.inputs[7].default_value = c_dust
        m_dust.location = (20, 440)
        nt.links.new(g_scl.outputs["Value"], m_dust.inputs["Factor"])
        nt.links.new(m_rock.outputs[2], m_dust.inputs[6])
        color_sock = m_dust.outputs[2]
    elif sprof == "foliage" and mat_meta.get("uses_foliage_lut") and color_sock is not None:
        # Exact HFW Biome Foliage LUT (`1:0`) reconstruction for neutralized 0.5-centered leaf maps
        # (Kapok, Carapa, Mangrove, Strangler Fig, Monstera, Forest Ivy): `4.5948 * FoliageLUT_linear * Leaf_linear`!
        ft_lin = mat_meta.get("foliage_tint_linear") or [0.062, 0.160, 0.018]
        fl_exp = nt.nodes.new("ShaderNodeVectorMath")
        fl_exp.operation = "SCALE"
        fl_exp.inputs["Scale"].default_value = 4.5948
        fl_exp.label = "Neutral-0.5 Leaf Expand (2^2.2)"
        fl_exp.location = (-220, 340)
        nt.links.new(color_sock, fl_exp.inputs[0])

        fl_mul = nt.nodes.new("ShaderNodeMix")
        fl_mul.data_type = "RGBA"
        fl_mul.blend_type = "MULTIPLY"
        fl_mul.clamp_result = True
        fl_mul.inputs["Factor"].default_value = 1.0
        fl_mul.inputs[7].default_value = (float(ft_lin[0]), float(ft_lin[1]), float(ft_lin[2]), 1.0)
        fl_mul.label = "HFW Biome Foliage LUT Tint (1:0)"
        fl_mul.location = (20, 360)
        nt.links.new(fl_exp.outputs["Vector"], fl_mul.inputs[6])
        color_sock = fl_mul.outputs[2]

    # 3. Exact DXIL-verified Decima PBR Conversion (`HFW_DecimaPBR`):
    #    - Inverts Decima Smoothness -> true PBR Roughness (fixes plastic shine on matte paint + matte edges on bare metal!)
    #    - Converts Decima Specular/Reflectance -> Blender Metallic + Base Color ONLY on potentially metallic surfaces
    #      (NEVER on Hair / Skin / Foliage / Landscape Rock, where `Reflectance` is hair sheen/scatter — e.g. Aloy's
    #      `181:1.R` jumps to 0.75-1.0 at the hair tips, which must stay dielectric hair sheen, NOT copper metal!)
    _DIELECTRIC_PROFILES = ("hair", "hair_face", "skin", "eye", "teeth", "foliage", "bark", "snow", "rock_volcanic", "rock_tint")
    if sprof in ("rock_volcanic", "rock_tint"):
        # Per DXIL `8490:1` L1020 (`0x3FD99999A0000000 = 0.40` max rock smoothness): remap `_LUM` / macro Smoothness
        # into realistic dry basalt/stone Smoothness (`0.08..0.34` -> PBR Roughness `0.66..0.92`, ZERO wet plastic shine!)
        if smooth_sock is not None:
            rk_sm = nt.nodes.new("ShaderNodeMapRange")
            rk_sm.clamp = True
            rk_sm.label = "DXIL Dry Rock Smoothness (0.08..0.34)"
            rk_sm.inputs["From Min"].default_value = 0.0
            rk_sm.inputs["From Max"].default_value = 1.0
            rk_sm.inputs["To Min"].default_value = 0.08
            rk_sm.inputs["To Max"].default_value = 0.34 if sprof == "rock_volcanic" else 0.28
            rk_sm.location = (-20, 60)
            nt.links.new(smooth_sock, rk_sm.inputs["Value"])
            smooth_sock = rk_sm.outputs["Result"]
        # Blend 2K standalone tiling volcanic/stone detail normal (`5211:1` etc.) at 8x UV scale onto 4K macro normal!
        ad_nfile = mat_meta.get("asset_detail_normal_file", "")
        if ad_nfile and normal_rgb_sock is not None:
            ad_img = _load_image(tex_dir, ad_nfile, "Non-Color", img_cache)
            if ad_img is not None:
                tc_rk = nt.nodes.new("ShaderNodeTexCoord")
                tc_rk.location = (-960, -820)
                uv_rk = nt.nodes.new("ShaderNodeVectorMath")
                uv_rk.operation = "SCALE"
                uv_rk.inputs["Scale"].default_value = 8.0
                uv_rk.label = "Rock Micro-Detail Tile (8x)"
                uv_rk.location = (-740, -820)
                nt.links.new(tc_rk.outputs["UV"], uv_rk.inputs[0])
                t_ad = nt.nodes.new("ShaderNodeTexImage")
                t_ad.image = ad_img
                t_ad.label = "HFW 2K Rock Detail Normal"
                t_ad.location = (-520, -820)
                nt.links.new(uv_rk.outputs["Vector"], t_ad.inputs["Vector"])
                sep_ad = nt.nodes.new("ShaderNodeSeparateColor")
                sep_ad.location = (-240, -820)
                nt.links.new(t_ad.outputs["Color"], sep_ad.inputs["Color"])
                if "Normal.R" in sockets and "Normal.G" in sockets:
                    # Exact tangent-space UDN detail blend (`macro + (detail - 0.5) * 0.65`)
                    def _rk_udn(mac_s, det_s, y: float):
                        d_off = nt.nodes.new("ShaderNodeMath")
                        d_off.operation = "MULTIPLY_ADD"
                        d_off.inputs[1].default_value = 0.65
                        d_off.inputs[2].default_value = -0.325
                        d_off.location = (-60, y)
                        nt.links.new(det_s, d_off.inputs[0])
                        d_add = nt.nodes.new("ShaderNodeMath")
                        d_add.operation = "ADD"
                        d_add.use_clamp = True
                        d_add.location = (80, y)
                        nt.links.new(mac_s, d_add.inputs[0])
                        nt.links.new(d_off.outputs["Value"], d_add.inputs[1])
                        return d_add.outputs["Value"]
                    rk_nx = _rk_udn(sockets["Normal.R"], sep_ad.outputs["Red"], -740)
                    rk_ny = _rk_udn(sockets["Normal.G"], sep_ad.outputs["Green"], -920)
                    rk_nrec = nt.nodes.new("ShaderNodeGroup")
                    rk_nrec.node_tree = _ensure_normal_reconstruct_z_group()
                    rk_nrec.location = (220, -820)
                    nt.links.new(rk_nx, rk_nrec.inputs["Normal X"])
                    nt.links.new(rk_ny, rk_nrec.inputs["Normal Y"])
                    normal_rgb_sock = rk_nrec.outputs["Normal RGB"]
                # Subtle micro-crevice height contrast (`t_ad.Alpha` 0.80..1.18) on rock albedo
                if color_sock is not None:
                    h_mr = nt.nodes.new("ShaderNodeMapRange")
                    h_mr.clamp = True
                    h_mr.inputs["From Min"].default_value = 0.15
                    h_mr.inputs["From Max"].default_value = 0.85
                    h_mr.inputs["To Min"].default_value = 0.78
                    h_mr.inputs["To Max"].default_value = 1.22
                    h_mr.location = (-60, 260)
                    nt.links.new(t_ad.outputs["Alpha"], h_mr.inputs["Value"])
                    h_sc = nt.nodes.new("ShaderNodeVectorMath")
                    h_sc.operation = "SCALE"
                    h_sc.location = (80, 300)
                    nt.links.new(color_sock, h_sc.inputs[0])
                    nt.links.new(h_mr.outputs["Result"], h_sc.inputs["Scale"])
                    color_sock = h_sc.outputs["Vector"]
    elif sprof == "hair" and smooth_sock is not None:
        # Calibrate Hair smoothness (`181:1.G`) to natural silky hair (`Smoothness 0.44..0.62` -> Roughness `0.38..0.56`)!
        hr_sm = nt.nodes.new("ShaderNodeMapRange")
        hr_sm.clamp = True
        hr_sm.label = "Natural Hair Smoothness"
        hr_sm.inputs["From Min"].default_value = 0.0
        hr_sm.inputs["From Max"].default_value = 1.0
        hr_sm.inputs["To Min"].default_value = 0.44
        hr_sm.inputs["To Max"].default_value = 0.62
        hr_sm.location = (-20, 60)
        nt.links.new(smooth_sock, hr_sm.inputs["Value"])
        smooth_sock = hr_sm.outputs["Result"]

    if color_sock is not None or smooth_sock is not None or rfl_sock is not None:
        pbr_grp = nt.nodes.new("ShaderNodeGroup")
        pbr_grp.node_tree = _ensure_decima_pbr_node_group()
        pbr_grp.label = "HFW Decima PBR (Smoothness & F0 -> Metallic)"
        pbr_grp.location = (180, 220)
        if color_sock is not None:
            nt.links.new(color_sock, pbr_grp.inputs["Color"])
        elif sprof == "snow":
            pbr_grp.inputs["Color"].default_value = (0.78, 0.84, 0.92, 1.0)
        elif "eye" in mname.lower():
            # Outer eye glass lens (`5704:0`) has no Color texture — dark glossy camera glass!
            pbr_grp.inputs["Color"].default_value = (0.015, 0.018, 0.024, 1.0)
        if smooth_sock is not None:
            nt.links.new(smooth_sock, pbr_grp.inputs["Smoothness"])
        if rfl_sock is not None and sprof not in _DIELECTRIC_PROFILES:
            nt.links.new(rfl_sock, pbr_grp.inputs["Reflectance"])
        if ao_sock is not None:
            nt.links.new(ao_sock, pbr_grp.inputs["AO"])

        nt.links.new(pbr_grp.outputs["Base Color"], bsdf.inputs["Base Color"])
        if sprof in _DIELECTRIC_PROFILES:
            bsdf.inputs["Metallic"].default_value = 0.0
        elif liquid_mask_sock is not None:
            # Force translucent liquid sac/fluid membrane pixels to dielectric (Metallic = 0) so liquid NEVER looks dark metallic!
            inv_liq = nt.nodes.new("ShaderNodeMath")
            inv_liq.operation = "SUBTRACT"
            inv_liq.inputs[0].default_value = 1.0
            inv_liq.use_clamp = True
            inv_liq.location = (220, 40)
            nt.links.new(liquid_mask_sock, inv_liq.inputs[1])
            met_liq = nt.nodes.new("ShaderNodeMath")
            met_liq.operation = "MULTIPLY"
            met_liq.use_clamp = True
            met_liq.location = (360, 40)
            nt.links.new(pbr_grp.outputs["Metallic"], met_liq.inputs[0])
            nt.links.new(inv_liq.outputs["Value"], met_liq.inputs[1])
            nt.links.new(met_liq.outputs["Value"], bsdf.inputs["Metallic"])
        else:
            nt.links.new(pbr_grp.outputs["Metallic"], bsdf.inputs["Metallic"])
        nt.links.new(pbr_grp.outputs["Roughness"], bsdf.inputs["Roughness"])
        if "Specular IOR Level" in bsdf.inputs:
            if sprof in _DIELECTRIC_PROFILES:
                bsdf.inputs["Specular IOR Level"].default_value = 0.45 if sprof == "hair" else (0.35 if sprof in ("rock_volcanic", "rock_tint") else 0.50)
            else:
                nt.links.new(pbr_grp.outputs["Specular IOR Level"], bsdf.inputs["Specular IOR Level"])

    # 4. Wire final Normal Map
    if normal_rgb_sock is not None:
        nrm_map = nt.nodes.new("ShaderNodeNormalMap")
        nrm_map.location = (220, -400)
        nrm_map.inputs["Strength"].default_value = 1.0
        nt.links.new(normal_rgb_sock, nrm_map.inputs["Color"])
        nt.links.new(nrm_map.outputs["Normal"], bsdf.inputs["Normal"])

    # 5. Wire Emissive / Elemental Sac / Canister / Eye Iris / Incandescence (all animated!)
    if overlay_em_color is not None and overlay_em_strength is not None:
        if "Incandescence" in sockets:
            # Combine overlay glow + baked Incandescence map cleanly
            em_add = nt.nodes.new("ShaderNodeMix")
            em_add.data_type = "RGBA"
            em_add.blend_type = "ADD"
            em_add.inputs["Factor"].default_value = 1.0
            em_add.location = (260, 460)
            nt.links.new(overlay_em_color, em_add.inputs[6])
            nt.links.new(sockets["Incandescence"], em_add.inputs[7])
            if "Emission Color" in bsdf.inputs:
                nt.links.new(em_add.outputs[2], bsdf.inputs["Emission Color"])
        else:
            if "Emission Color" in bsdf.inputs:
                nt.links.new(overlay_em_color, bsdf.inputs["Emission Color"])
        if "Emission Strength" in bsdf.inputs:
            nt.links.new(overlay_em_strength, bsdf.inputs["Emission Strength"])
    elif "Incandescence" in sockets:
        t_drv = _add_time_driver_node(nt, (-160, 520), "HFW Incandescence Pulse Time")
        t_m = nt.nodes.new("ShaderNodeMath")
        t_m.operation = "MULTIPLY"
        t_m.inputs[1].default_value = 2.2
        t_m.location = (20, 520)
        nt.links.new(t_drv.outputs[0], t_m.inputs[0])
        s_m = nt.nodes.new("ShaderNodeMath")
        s_m.operation = "SINE"
        s_m.location = (160, 520)
        nt.links.new(t_m.outputs["Value"], s_m.inputs[0])
        p_m = nt.nodes.new("ShaderNodeMath")
        p_m.operation = "MULTIPLY_ADD"
        p_m.inputs[1].default_value = 0.12
        p_m.inputs[2].default_value = 0.88
        p_m.location = (280, 520)
        nt.links.new(s_m.outputs["Value"], p_m.inputs[0])
        st_m = nt.nodes.new("ShaderNodeMath")
        st_m.operation = "MULTIPLY"
        st_m.inputs[1].default_value = float(mat_meta.get("emissive_strength", 4.0))
        st_m.location = (380, 460)
        nt.links.new(p_m.outputs["Value"], st_m.inputs[0])
        if "Emission Color" in bsdf.inputs:
            nt.links.new(sockets["Incandescence"], bsdf.inputs["Emission Color"])
        if "Emission Strength" in bsdf.inputs:
            nt.links.new(st_m.outputs["Value"], bsdf.inputs["Emission Strength"])

    # 5b. Physical Wet Membrane Coat & Subsurface Scatter ONLY on Liquid Sac / Canister Fluid Regions!
    if liquid_mask_sock is not None:
        if "Coat Weight" in bsdf.inputs:
            coat_m = nt.nodes.new("ShaderNodeMath")
            coat_m.operation = "MULTIPLY"
            coat_m.inputs[1].default_value = 0.85
            coat_m.location = (280, 120)
            nt.links.new(liquid_mask_sock, coat_m.inputs[0])
            nt.links.new(coat_m.outputs["Value"], bsdf.inputs["Coat Weight"])
            if "Coat Roughness" in bsdf.inputs:
                bsdf.inputs["Coat Roughness"].default_value = 0.05
        if "Subsurface Weight" in bsdf.inputs:
            sss_m = nt.nodes.new("ShaderNodeMath")
            sss_m.operation = "MULTIPLY"
            sss_m.inputs[1].default_value = 0.42
            sss_m.location = (280, -20)
            nt.links.new(liquid_mask_sock, sss_m.inputs[0])
            nt.links.new(sss_m.outputs["Value"], bsdf.inputs["Subsurface Weight"])
            if "Subsurface Scale" in bsdf.inputs:
                bsdf.inputs["Subsurface Scale"].default_value = 0.04

    # 6. Wire Alpha cutout ONLY when material genuinely uses alpha transparency (never on opaque skin/cloth/body!)
    allow_alpha = bool(mat_meta.get("allow_alpha_cutout", True))
    sprof = mat_meta.get("surface_profile", "pbr")
    if allow_alpha and "Alpha" in sockets and sprof not in ("skin", "eye", "teeth"):
        nt.links.new(sockets["Alpha"], bsdf.inputs["Alpha"])

    # 7. Photorealistic Decima Character & General Asset Surface Profile Refinements
    if sprof == "peach_fuzz":
        # Backlit Grazing-Angle Facial Peach Fuzz / Vellus Hair (`3996:0`):
        #   - Grazing silhouette `Facing^3.4` broken up by ultra-fine strand Noise (zero solid white veil face-on!)
        #   - Warm sunlit `Translucent + Sheen BSDF` driven by `HFW_CharacterControl`
        #   - Strictly gated by `Light Path -> Is Shadow Ray -> Transparent BSDF` so it NEVER casts shadows onto skin!
        _configure_blended_material(mat)
        nt.nodes.clear()
        out_pf = nt.nodes.new("ShaderNodeOutputMaterial")
        out_pf.location = (920, 0)

        c_ctrl_pf = nt.nodes.new("ShaderNodeGroup")
        c_ctrl_pf.node_tree = _ensure_character_control_group()
        c_ctrl_pf.label = "HFW Character Control"
        c_ctrl_pf.location = (-760, 260)

        # Warm golden-blonde sunlit vellus tint derived from character's Skin Tone!
        pf_hue = nt.nodes.new("ShaderNodeHueSaturation")
        pf_hue.inputs["Saturation"].default_value = 0.62
        pf_hue.inputs["Value"].default_value = 1.65
        pf_hue.label = "Sunlit Vellus Hair Tint"
        pf_hue.location = (-440, 260)
        nt.links.new(c_ctrl_pf.outputs["Skin Tone"], pf_hue.inputs["Color"])

        # Grazing-angle silhouette falloff: Layer Weight Facing ^ 3.4
        lw_pf = nt.nodes.new("ShaderNodeLayerWeight")
        lw_pf.inputs["Blend"].default_value = 0.48
        lw_pf.location = (-760, -60)
        pw_pf = nt.nodes.new("ShaderNodeMath")
        pw_pf.operation = "POWER"
        pw_pf.inputs[1].default_value = 3.4
        pw_pf.use_clamp = True
        pw_pf.label = "Grazing Rim Falloff (Facing^3.4)"
        pw_pf.location = (-520, -60)
        nt.links.new(lw_pf.outputs["Facing"], pw_pf.inputs[0])

        # Ultra-fine micro-hair strand noise breakup so fuzz catches light as microscopic hairs!
        tc_pf = nt.nodes.new("ShaderNodeTexCoord")
        tc_pf.location = (-980, -280)
        ns_pf = nt.nodes.new("ShaderNodeTexNoise")
        ns_pf.inputs["Scale"].default_value = 420.0
        ns_pf.inputs["Detail"].default_value = 5.0
        ns_pf.inputs["Roughness"].default_value = 0.65
        ns_pf.label = "Micro Vellus Strand Breakup"
        ns_pf.location = (-740, -280)
        nt.links.new(tc_pf.outputs["Object"], ns_pf.inputs["Vector"])

        ns_mr = nt.nodes.new("ShaderNodeMapRange")
        ns_mr.clamp = True
        ns_mr.inputs["From Min"].default_value = 0.34
        ns_mr.inputs["From Max"].default_value = 0.72
        ns_mr.inputs["To Min"].default_value = 0.25
        ns_mr.inputs["To Max"].default_value = 1.0
        ns_mr.location = (-500, -280)
        nt.links.new(ns_pf.outputs["Fac"], ns_mr.inputs["Value"])

        m_fn = nt.nodes.new("ShaderNodeMath")
        m_fn.operation = "MULTIPLY"
        m_fn.location = (-280, -140)
        nt.links.new(pw_pf.outputs["Value"], m_fn.inputs[0])
        nt.links.new(ns_mr.outputs["Result"], m_fn.inputs[1])

        m_int = nt.nodes.new("ShaderNodeMath")
        m_int.operation = "MULTIPLY"
        m_int.location = (-100, -100)
        nt.links.new(m_fn.outputs["Value"], m_int.inputs[0])
        if "Peach Fuzz Intensity" in c_ctrl_pf.outputs:
            nt.links.new(c_ctrl_pf.outputs["Peach Fuzz Intensity"], m_int.inputs[1])
        else:
            m_int.inputs[1].default_value = 0.65

        m_op = nt.nodes.new("ShaderNodeMath")
        m_op.operation = "MULTIPLY"
        m_op.inputs[1].default_value = 0.52
        m_op.use_clamp = True
        m_op.label = "Final Grazing Opacity"
        m_op.location = (80, -60)
        nt.links.new(m_int.outputs["Value"], m_op.inputs[0])

        # Soft backlit Translucent + velvety Sheen micro-hair shader
        tr_pf = nt.nodes.new("ShaderNodeBsdfTransparent")
        tr_pf.location = (120, -240)
        tl_pf = nt.nodes.new("ShaderNodeBsdfTranslucent")
        tl_pf.location = (-120, 260)
        nt.links.new(pf_hue.outputs["Color"], tl_pf.inputs["Color"])
        sh_pf = nt.nodes.new("ShaderNodeBsdfSheen") if hasattr(bpy.types, "ShaderNodeBsdfSheen") else nt.nodes.new("ShaderNodeBsdfDiffuse")
        sh_pf.location = (-120, 120)
        nt.links.new(pf_hue.outputs["Color"], sh_pf.inputs["Color"])
        if "Roughness" in sh_pf.inputs:
            sh_pf.inputs["Roughness"].default_value = 0.45

        mx_fuzz = nt.nodes.new("ShaderNodeMixShader")
        mx_fuzz.inputs["Fac"].default_value = 0.55
        mx_fuzz.label = "Backlit Translucent + Sheen"
        mx_fuzz.location = (120, 200)
        nt.links.new(tl_pf.outputs["BSDF"], mx_fuzz.inputs[1])
        nt.links.new(sh_pf.outputs["BSDF"], mx_fuzz.inputs[2])

        mx_vis = nt.nodes.new("ShaderNodeMixShader")
        mx_vis.label = "Transparent <-> Grazing Vellus Hair"
        mx_vis.location = (420, 60)
        nt.links.new(m_op.outputs["Value"], mx_vis.inputs["Fac"])
        nt.links.new(tr_pf.outputs["BSDF"], mx_vis.inputs[1])
        nt.links.new(mx_fuzz.outputs["Shader"], mx_vis.inputs[2])

        # Mandatory Light Path Shadow-Ray Bypass -> ZERO shadow cast onto face skin!
        lp_pf = nt.nodes.new("ShaderNodeLightPath")
        lp_pf.location = (420, 280)
        mx_shd = nt.nodes.new("ShaderNodeMixShader")
        mx_shd.label = "Shadow-Ray Bypass (Zero Skin Shadow!)"
        mx_shd.location = (680, 40)
        nt.links.new(lp_pf.outputs["Is Shadow Ray"], mx_shd.inputs["Fac"])
        nt.links.new(mx_vis.outputs["Shader"], mx_shd.inputs[1])
        nt.links.new(tr_pf.outputs["BSDF"], mx_shd.inputs[2])
        nt.links.new(mx_shd.outputs["Shader"], out_pf.inputs["Surface"])
        return mat
    elif sprof == "hair_face":
        # Eyebrow, Eyelash & Facial Hair cards: hashed alpha cutout + character hair tint!
        _configure_blended_material(mat)
        if hasattr(mat, "blend_method"):
            try:
                mat.blend_method = "HASHED"
            except Exception:
                pass
        if color_sock is None:
            bsdf.inputs["Base Color"].default_value = (0.14, 0.075, 0.045, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.50
        if "Alpha" in sockets:
            nt.links.new(sockets["Alpha"], bsdf.inputs["Alpha"])
        elif "Mask_Alpha" in sockets:
            nt.links.new(sockets["Mask_Alpha"], bsdf.inputs["Alpha"])
    elif sprof == "skin":
        # Photorealistic Decima Human Skin: Blender 5.3 RANDOM_WALK_SKIN + TRA Translucency map + Micro-Pore Normal + Dual-Lobe Specular!
        if hasattr(bsdf, "subsurface_method"):
            try:
                bsdf.subsurface_method = "RANDOM_WALK_SKIN"
            except Exception:
                try:
                    bsdf.subsurface_method = "RANDOM_WALK"
                except Exception:
                    pass
        if "Subsurface Weight" in bsdf.inputs:
            tra_sock = sockets.get("Translucency_Amount") or sockets.get("Mask_Alpha")
            base_sss_sock = char_ctrl_node.outputs["Subsurface Weight"] if char_ctrl_node is not None else None
            if tra_sock is not None:
                # Wire Decima's per-pixel `Translucency_Amount` (`Head_Shared_RGH.A`) -> ears/nostrils/lips scatter warmly!
                tra_mr = nt.nodes.new("ShaderNodeMapRange")
                tra_mr.clamp = True
                tra_mr.inputs["From Min"].default_value = 0.0
                tra_mr.inputs["From Max"].default_value = 1.0
                tra_mr.inputs["To Min"].default_value = 0.45
                tra_mr.inputs["To Max"].default_value = 1.40
                tra_mr.label = "Decima TRA Translucency Remap"
                tra_mr.location = (180, -140)
                nt.links.new(tra_sock, tra_mr.inputs["Value"])
                sss_mul = nt.nodes.new("ShaderNodeMath")
                sss_mul.operation = "MULTIPLY"
                sss_mul.use_clamp = True
                sss_mul.location = (360, -140)
                if base_sss_sock is not None:
                    nt.links.new(base_sss_sock, sss_mul.inputs[0])
                else:
                    sss_mul.inputs[0].default_value = 0.32
                nt.links.new(tra_mr.outputs["Result"], sss_mul.inputs[1])
                nt.links.new(sss_mul.outputs["Value"], bsdf.inputs["Subsurface Weight"])
            elif base_sss_sock is not None:
                nt.links.new(base_sss_sock, bsdf.inputs["Subsurface Weight"])
            else:
                bsdf.inputs["Subsurface Weight"].default_value = 0.32
        if "Subsurface Scale" in bsdf.inputs:
            bsdf.inputs["Subsurface Scale"].default_value = 0.016
        if "Subsurface Radius" in bsdf.inputs:
            bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.32, 0.15)
        if "Subsurface IOR" in bsdf.inputs:
            bsdf.inputs["Subsurface IOR"].default_value = 1.40
        if "Subsurface Anisotropy" in bsdf.inputs:
            bsdf.inputs["Subsurface Anisotropy"].default_value = 0.25
        # Dual-lobe skin specular (natural sebaceous sheen on cheeks/nose/forehead!)
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.46
        if "Coat Weight" in bsdf.inputs:
            bsdf.inputs["Coat Weight"].default_value = 0.14
            bsdf.inputs["Coat Roughness"].default_value = 0.28
        if "Coat IOR" in bsdf.inputs:
            bsdf.inputs["Coat IOR"].default_value = 1.38
        # Fine close-up micro-pore skin normal detail layered gently on top of 4K Head Normal map!
        if "Normal" in bsdf.inputs:
            tc_pore = nt.nodes.new("ShaderNodeTexCoord")
            tc_pore.location = (-280, -680)
            ns_pore = nt.nodes.new("ShaderNodeTexNoise")
            ns_pore.inputs["Scale"].default_value = 480.0
            ns_pore.inputs["Detail"].default_value = 6.0
            ns_pore.inputs["Roughness"].default_value = 0.58
            ns_pore.label = "Skin Micro-Pore Grain"
            ns_pore.location = (-60, -680)
            nt.links.new(tc_pore.outputs["Object"], ns_pore.inputs["Vector"])
            bmp_pore = nt.nodes.new("ShaderNodeBump")
            bmp_pore.inputs["Strength"].default_value = 0.06
            bmp_pore.inputs["Distance"].default_value = 0.0004
            bmp_pore.label = "Micro-Pore Relief"
            bmp_pore.location = (360, -560)
            nt.links.new(ns_pore.outputs["Fac"], bmp_pore.inputs["Height"])
            for lnk in list(bsdf.inputs["Normal"].links):
                nt.links.new(lnk.from_socket, bmp_pore.inputs["Normal"])
                nt.links.remove(lnk)
            nt.links.new(bmp_pore.outputs["Normal"], bsdf.inputs["Normal"])
    elif sprof == "snow":
        if "Subsurface Weight" in bsdf.inputs:
            bsdf.inputs["Subsurface Weight"].default_value = 0.25
        if "Subsurface Scale" in bsdf.inputs:
            bsdf.inputs["Subsurface Scale"].default_value = 0.03
        if "Coat Weight" in bsdf.inputs:
            bsdf.inputs["Coat Weight"].default_value = 0.15
            bsdf.inputs["Coat Roughness"].default_value = 0.22
    elif sprof == "hair":
        _configure_blended_material(mat)
        if hasattr(mat, "blend_method"):
            try:
                mat.blend_method = "HASHED"
            except Exception:
                pass
        if color_sock is None:
            # NPC hair cards whose tint lives in a shader constant (e.g. Seyka's dark ponytail)
            bsdf.inputs["Base Color"].default_value = (0.045, 0.032, 0.025, 1.0)
        if "Alpha" not in sockets and "Mask_Alpha" in sockets:
            nt.links.new(sockets["Mask_Alpha"], bsdf.inputs["Alpha"])
        if "Anisotropic" in bsdf.inputs:
            bsdf.inputs["Anisotropic"].default_value = 0.68
        if "Sheen Weight" in bsdf.inputs:
            bsdf.inputs["Sheen Weight"].default_value = 0.28
            if "Sheen Roughness" in bsdf.inputs:
                bsdf.inputs["Sheen Roughness"].default_value = 0.42
        if "Coat Weight" in bsdf.inputs:
            # Subtle dielectric hair specular sheen driven by `181:1.R` (NOT Metallic copper!)
            if rfl_sock is not None:
                hc_m = nt.nodes.new("ShaderNodeMath")
                hc_m.operation = "MULTIPLY"
                hc_m.inputs[1].default_value = 0.18
                hc_m.use_clamp = True
                hc_m.location = (260, -140)
                nt.links.new(rfl_sock, hc_m.inputs[0])
                nt.links.new(hc_m.outputs["Value"], bsdf.inputs["Coat Weight"])
            else:
                bsdf.inputs["Coat Weight"].default_value = 0.12
            if "Coat Roughness" in bsdf.inputs:
                bsdf.inputs["Coat Roughness"].default_value = 0.34
        if smooth_sock is None:
            bsdf.inputs["Roughness"].default_value = 0.44
    elif sprof == "eye":
        # Photorealistic Human Eyeball (`Eyes_NMT_40_4_MSK_AO.png`: G = Full Iris Disc, R = Inner Heterochromia Ring, B = Sclera)!
        bsdf.inputs["Roughness"].default_value = 0.03
        if "Coat Weight" in bsdf.inputs:
            bsdf.inputs["Coat Weight"].default_value = 1.0
            bsdf.inputs["Coat Roughness"].default_value = 0.015
        if "IOR" in bsdf.inputs:
            bsdf.inputs["IOR"].default_value = 1.40
        eye_ctrl = nt.nodes.new("ShaderNodeGroup")
        ic_init = {"iris_color_linear": mat_meta.get("iris_color_linear")} if mat_meta.get("iris_color_linear") else None
        eye_ctrl.node_tree = _ensure_character_control_group(ic_init)
        eye_ctrl.label = "HFW Character Iris Control"
        eye_ctrl.location = (-80, 460)
        iris_mask_sock = sockets.get("Mask.G") or sockets.get("Mask")
        if color_sock is not None and iris_mask_sock is not None:
            # Boost neutral-0.45 `Eyes_CLR` iris luminance by 2.2x * IrisColor inside the full Green iris disc mask!
            iris_boost = nt.nodes.new("ShaderNodeVectorMath")
            iris_boost.operation = "SCALE"
            iris_boost.inputs["Scale"].default_value = 2.25
            iris_boost.label = "Iris Albedo Scale (2.25x)"
            iris_boost.location = (140, 440)
            if "Iris Color" in eye_ctrl.outputs:
                nt.links.new(eye_ctrl.outputs["Iris Color"], iris_boost.inputs[0])
            else:
                ic_d = mat_meta.get("iris_color_linear") or [0.106, 0.199, 0.061]
                iris_boost.inputs[0].default_value = (ic_d[0], ic_d[1], ic_d[2])

            iris_mx = nt.nodes.new("ShaderNodeMix")
            iris_mx.data_type = "RGBA"
            iris_mx.blend_type = "MULTIPLY"
            iris_mx.clamp_result = True
            iris_mx.label = "Character Iris Tint (Full Mask.G Disc)"
            iris_mx.location = (360, 320)
            nt.links.new(iris_mask_sock, iris_mx.inputs["Factor"])
            for lnk in list(bsdf.inputs["Base Color"].links):
                nt.links.new(lnk.from_socket, iris_mx.inputs[6])
                nt.links.remove(lnk)
            nt.links.new(iris_boost.outputs["Vector"], iris_mx.inputs[7])
            nt.links.new(iris_mx.outputs[2], bsdf.inputs["Base Color"])
            # Subtle warm pinkish Sclera Subsurface Scattering ONLY outside the full iris disc (`1 - Mask.G`)!
            if "Subsurface Weight" in bsdf.inputs:
                scl_inv = nt.nodes.new("ShaderNodeMath")
                scl_inv.operation = "MULTIPLY_ADD"
                scl_inv.inputs[1].default_value = -0.14
                scl_inv.inputs[2].default_value = 0.14
                scl_inv.use_clamp = True
                scl_inv.label = "Sclera SSS Mask"
                scl_inv.location = (220, -60)
                nt.links.new(iris_mask_sock, scl_inv.inputs[0])
                nt.links.new(scl_inv.outputs["Value"], bsdf.inputs["Subsurface Weight"])
                if "Subsurface Scale" in bsdf.inputs:
                    bsdf.inputs["Subsurface Scale"].default_value = 0.008
                if "Subsurface Radius" in bsdf.inputs:
                    bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.35, 0.24)
    elif sprof == "eye_wet":
        # Crystal-clear physical wet cornea / tearline shell: pure Transparent + Fresnel Glossy catchlights (ZERO milky veil!)
        _configure_blended_material(mat)
        nt.nodes.clear()
        out_ew = nt.nodes.new("ShaderNodeOutputMaterial")
        out_ew.location = (560, 0)
        tr_ew = nt.nodes.new("ShaderNodeBsdfTransparent")
        tr_ew.location = (80, 120)
        gl_ew = nt.nodes.new("ShaderNodeBsdfGlossy")
        gl_ew.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        gl_ew.inputs["Roughness"].default_value = 0.02
        gl_ew.location = (80, -80)
        fr_ew = nt.nodes.new("ShaderNodeFresnel")
        fr_ew.inputs["IOR"].default_value = 1.38
        fr_ew.location = (-160, 220)
        sc_ew = nt.nodes.new("ShaderNodeMath")
        sc_ew.operation = "MULTIPLY"
        sc_ew.inputs[1].default_value = 0.22
        sc_ew.location = (80, 240)
        nt.links.new(fr_ew.outputs["Fac"], sc_ew.inputs[0])
        mx_ew = nt.nodes.new("ShaderNodeMixShader")
        mx_ew.location = (340, 20)
        nt.links.new(sc_ew.outputs["Value"], mx_ew.inputs["Fac"])
        nt.links.new(tr_ew.outputs["BSDF"], mx_ew.inputs[1])
        nt.links.new(gl_ew.outputs["BSDF"], mx_ew.inputs[2])
        nt.links.new(mx_ew.outputs["Shader"], out_ew.inputs["Surface"])
        return mat
    elif sprof == "teeth":
        bsdf.inputs["Roughness"].default_value = 0.18
        if "Coat Weight" in bsdf.inputs:
            bsdf.inputs["Coat Weight"].default_value = 0.65
            bsdf.inputs["Coat Roughness"].default_value = 0.08
    elif sprof == "foliage":
        if "Alpha" in sockets or "Mask_Alpha" in sockets:
            _configure_blended_material(mat)
            if hasattr(mat, "blend_method"):
                try:
                    mat.blend_method = "HASHED"
                except Exception:
                    pass
            if hasattr(mat, "use_backface_culling"):
                try:
                    mat.use_backface_culling = False
                except Exception:
                    pass
            if "Alpha" not in sockets and "Mask_Alpha" in sockets:
                nt.links.new(sockets["Mask_Alpha"], bsdf.inputs["Alpha"])
        if "Subsurface Weight" in bsdf.inputs:
            bsdf.inputs["Subsurface Weight"].default_value = 0.35
        if "Subsurface Scale" in bsdf.inputs:
            bsdf.inputs["Subsurface Scale"].default_value = 0.02
        if "Subsurface Radius" in bsdf.inputs:
            bsdf.inputs["Subsurface Radius"].default_value = (0.45, 0.88, 0.16)
    elif sprof == "bark":
        if smooth_sock is None:
            bsdf.inputs["Roughness"].default_value = 0.84
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.30
    elif sprof == "cloth":
        if "Sheen Weight" in bsdf.inputs:
            bsdf.inputs["Sheen Weight"].default_value = 0.45
            bsdf.inputs["Sheen Roughness"].default_value = 0.60
    elif sprof == "glass":
        _configure_blended_material(mat)
        if "Transmission Weight" in bsdf.inputs:
            bsdf.inputs["Transmission Weight"].default_value = 0.85
        if "IOR" in bsdf.inputs:
            bsdf.inputs["IOR"].default_value = 1.45
        if "Alpha" not in sockets:
            bsdf.inputs["Alpha"].default_value = 0.25

    return mat


def _build_armature(
    name: str,
    joints: list[dict[str, Any]],
    helpers: list[dict[str, Any]],
    root_coll: bpy.types.Collection,
) -> tuple[bpy.types.Object, list[str], dict[str, str]]:
    """Build a unified Blender Armature (or Root Empty for pure static assets with 0 joints) with clean Bone Collections."""
    if not joints and not helpers:
        root_empty = bpy.data.objects.new(f"{name}_Root", None)
        root_empty.empty_display_type = "PLAIN_AXES"
        root_empty.empty_display_size = 0.25
        root_coll.objects.link(root_empty)
        return root_empty, [], {}

    arm_data = bpy.data.armatures.new(f"{name}_Armature")
    arm_obj = bpy.data.objects.new(f"{name}_Armature", arm_data)
    root_coll.objects.link(arm_obj)

    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")
    ebones = arm_data.edit_bones

    children_of: dict[int, list[int]] = {}
    for j in joints:
        p = j["parent_index"]
        if 0 <= p < len(joints):
            children_of.setdefault(p, []).append(j["index"])

    joint_names: list[str] = []
    edit_joint_bones: list[bpy.types.EditBone] = []
    joint_world_mats: list[Matrix] = [_mat4(j["world_matrix"]) for j in joints]

    has_facial_bones = any(j["name"].startswith("FACIAL_") for j in joints)
    has_cloth_bones = any(
        any(k in j["name"].lower() for k in ("_pbd_", "_jbd", "cloth", "skirt", "rope", "strap", "hair", "braid", "tassel", "flap"))
        for j in joints
    )
    main_bcoll = arm_data.collections.new("Body" if has_facial_bones else "Skeleton")
    face_bcoll = arm_data.collections.new("Face") if has_facial_bones else None
    if face_bcoll is not None:
        face_bcoll.is_visible = False  # Collapsible clean Face rig collection!
    cloth_bcoll = arm_data.collections.new("Cloth_Hair") if has_cloth_bones else None
    if cloth_bcoll is not None:
        cloth_bcoll.is_visible = False
    helper_bcoll = arm_data.collections.new("Helpers")
    helper_bcoll.is_visible = False

    for j in joints:
        idx = j["index"]
        wmat = joint_world_mats[idx]
        head = wmat.to_translation()
        eb = ebones.new(j["name"])
        eb.head = head

        kids = children_of.get(idx, [])
        is_face = j["name"].startswith("FACIAL_")
        bone_len = 0.025 if is_face else 0.12
        if kids:
            dists = [(joint_world_mats[k].to_translation() - head).length for k in kids]
            valid = [d for d in dists if d > 0.005]
            if valid:
                min_l = 0.015 if is_face else 0.04
                max_l = 0.06 if is_face else 0.45
                bone_len = max(min_l, min(max_l, sum(valid) / len(valid)))

        y_axis = wmat.to_3x3() @ Vector((0.0, 1.0, 0.0))
        if y_axis.length < 1e-6:
            y_axis = Vector((0.0, 1.0, 0.0))
        else:
            y_axis.normalize()
        eb.tail = head + y_axis * bone_len
        eb.matrix = Matrix.Translation(head) @ wmat.to_3x3().normalized().to_4x4()
        eb.length = bone_len

        jlow = j["name"].lower()
        if face_bcoll is not None and is_face:
            face_bcoll.assign(eb)
        elif cloth_bcoll is not None and any(k in jlow for k in ("_pbd_", "_jbd", "cloth", "skirt", "rope", "strap", "hair", "braid", "tassel", "flap")):
            cloth_bcoll.assign(eb)
        else:
            main_bcoll.assign(eb)
        joint_names.append(eb.name)
        edit_joint_bones.append(eb)

    for j in joints:
        p = j["parent_index"]
        if 0 <= p < len(edit_joint_bones) and p != j["index"]:
            edit_joint_bones[j["index"]].parent = edit_joint_bones[p]

    helper_bone_map: dict[str, str] = {}
    for hp in helpers:
        hname = hp["name"]
        pidx = hp["parent_joint_index"]
        wmat = _mat4(hp["world_matrix"])
        head = wmat.to_translation()
        eb = ebones.new(hname)
        eb.head = head
        eb.tail = head + Vector((0.0, 0.12, 0.0))
        eb.matrix = Matrix.Translation(head) @ wmat.to_3x3().normalized().to_4x4()
        eb.length = 0.12
        if pidx is not None and 0 <= pidx < len(edit_joint_bones):
            eb.parent = edit_joint_bones[pidx]
        helper_bcoll.assign(eb)
        helper_bone_map[hname] = eb.name

    bpy.ops.object.mode_set(mode="OBJECT")
    arm_data.display_type = "STICK"
    return arm_obj, joint_names, helper_bone_map


def _build_part_mesh_object(
    part_meta: dict[str, Any],
    npz: Any,
    mat_map: dict[str, bpy.types.Material],
    joint_names: list[str],
    arm_obj: bpy.types.Object,
    helper_bone_map: dict[str, str],
    target_coll: bpy.types.Collection,
    entity_id: str,
    shared_mesh_cache: dict[str, bpy.types.Mesh] | None = None,
) -> bpy.types.Object | None:
    """Create a Blender mesh object combining the part's primitives, with skinning OR exact bone parenting."""
    prims = part_meta["primitives"]
    if not prims:
        return None

    pname = part_meta["name"]
    shared_mid = (
        str(part_meta.get("shared_mesh_id") or "")
        if (not part_meta["is_skinned"] and shared_mesh_cache is not None)
        else ""
    )
    mesh: bpy.types.Mesh | None = shared_mesh_cache.get(shared_mid) if (shared_mid and shared_mesh_cache is not None) else None
    has_any_weights = False
    all_bi: list[np.ndarray] = []
    all_bw: list[np.ndarray] = []

    if mesh is None:
        all_pos = []
        all_nrm = []
        all_col0 = []
        all_faces = []
        face_mat_indices = []
        all_uvs: list[list[np.ndarray]] = []

        max_uvs = max(p["uv_count"] for p in prims)
        has_any_col0 = any(p.get("color_count", 0) > 0 for p in prims)
        has_all_nrm = all(p["has_normals"] for p in prims)
        has_any_weights = part_meta["is_skinned"] and any(p["has_weights"] for p in prims)
        for _ in range(max_uvs):
            all_uvs.append([])

        used_mats: list[bpy.types.Material | None] = []
        mat_slot_lookup: dict[str | None, int] = {}
        vert_offset = 0

        for prim in prims:
            key = prim["key"]
            pos = npz[f"{key}_pos"].astype(np.float32, copy=True)
            idx = npz[f"{key}_idx"]
            n_verts = len(pos)
            if n_verts == 0 or len(idx) < 3:
                continue

            mid = prim["material_id"]
            mat_obj = mat_map.get(mid) if mid else None
            if mid not in mat_slot_lookup:
                mat_slot_lookup[mid] = len(used_mats)
                used_mats.append(mat_obj)
            mslot = mat_slot_lookup[mid]

            nrm_arr = npz[f"{key}_nrm"] if (f"{key}_nrm" in npz) else None

            # CRITICAL CYCLES & EEVEE FIX:
            # In Decima, `muscle_fiber_overlay` (`19501:0`/`21395:0`/`20640:0`/`26679:0`) shares BIT-FOR-BIT IDENTICAL
            # (`0.000000m`) coplanar vertices with the opaque body muscle mesh underneath (rendered in DX12 with LEQUAL depth-bias).
            # Offset these overlay vertices outward by +1.4mm along their smooth vertex normals so Cycles raytracing & Eevee
            # NEVER suffer coplanar z-fighting speckles or dark triangles while keeping skinning 100% locked!
            if mat_obj is not None and mat_obj.get("hfw_special_shader") == "muscle_fiber_overlay":
                if nrm_arr is not None and len(nrm_arr) == n_verts:
                    v_nrm = nrm_arr.astype(np.float32, copy=False)
                else:
                    v_nrm = np.zeros((n_verts, 3), dtype=np.float32)
                    t_local = idx[: (len(idx) // 3) * 3].reshape(-1, 3)
                    e1 = pos[t_local[:, 1]] - pos[t_local[:, 0]]
                    e2 = pos[t_local[:, 2]] - pos[t_local[:, 0]]
                    fn = np.cross(e1, e2)
                    np.add.at(v_nrm, t_local[:, 0], fn)
                    np.add.at(v_nrm, t_local[:, 1], fn)
                    np.add.at(v_nrm, t_local[:, 2], fn)
                n_lens = np.linalg.norm(v_nrm, axis=1, keepdims=True)
                unit_n = np.where(n_lens > 1e-6, v_nrm / np.maximum(n_lens, 1e-6), 0.0)
                pos = pos + unit_n * np.float32(0.0014)

            all_pos.append(pos)
            if has_all_nrm:
                all_nrm.append(nrm_arr)

            if has_any_col0:
                ckey = f"{key}_col0"
                if ckey in npz:
                    c0 = npz[ckey].astype(np.float32, copy=False)
                    if c0.shape[1] == 3:
                        c0 = np.concatenate([c0, np.ones((n_verts, 1), dtype=np.float32)], axis=1)
                    all_col0.append(c0[:, :4])
                else:
                    c_def = np.zeros((n_verts, 4), dtype=np.float32)
                    c_def[:, 3] = 1.0
                    all_col0.append(c_def)

            for u_i in range(max_uvs):
                ukey = f"{key}_uv{u_i}"
                if ukey in npz:
                    # Decima UVs are ALREADY V-up (0 = bottom, 1 = top), matching Blender UV space directly!
                    uv = npz[ukey].copy()
                    # Guard against dead all-zero UV1 slots (e.g. SeykaTrimsUV2 primitives): mirror UV0 cleanly!
                    if u_i >= 1 and float(np.ptp(uv)) < 1e-5 and f"{key}_uv0" in npz:
                        uv = npz[f"{key}_uv0"].copy()
                    all_uvs[u_i].append(uv)
                else:
                    if u_i >= 1 and f"{key}_uv0" in npz:
                        all_uvs[u_i].append(npz[f"{key}_uv0"].copy())
                    else:
                        all_uvs[u_i].append(np.zeros((n_verts, 2), dtype=np.float32))

            if has_any_weights:
                bikey = f"{key}_bi"
                bwkey = f"{key}_bw"
                if bikey in npz and bwkey in npz:
                    all_bi.append(npz[bikey])
                    all_bw.append(npz[bwkey])
                else:
                    all_bi.append(np.zeros((n_verts, 1), dtype=np.uint16))
                    all_bw.append(np.zeros((n_verts, 1), dtype=np.float32))

            tris = idx[: (len(idx) // 3) * 3].reshape(-1, 3) + vert_offset
            valid = (tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])
            tris = tris[valid]
            all_faces.append(tris)
            face_mat_indices.append(np.full(len(tris), mslot, dtype=np.int32))
            vert_offset += n_verts

        if not all_pos or not all_faces:
            return None

        positions = np.concatenate(all_pos, axis=0)
        faces = np.concatenate(all_faces, axis=0)
        if len(faces) == 0:
            return None
        mat_indices = np.concatenate(face_mat_indices, axis=0)

        mesh = bpy.data.meshes.new(pname)
        n_verts_total = len(positions)
        n_tris_total = len(faces)

        mesh.vertices.add(n_verts_total)
        mesh.vertices.foreach_set("co", positions.ravel())

        mesh.loops.add(n_tris_total * 3)
        mesh.loops.foreach_set("vertex_index", faces.ravel())

        mesh.polygons.add(n_tris_total)
        mesh.polygons.foreach_set("loop_start", np.arange(0, n_tris_total * 3, 3, dtype=np.int32))
        mesh.polygons.foreach_set("loop_total", np.full(n_tris_total, 3, dtype=np.int32))
        mesh.polygons.foreach_set("use_smooth", np.ones(n_tris_total, dtype=bool))
        mesh.polygons.foreach_set("material_index", mat_indices)

        mesh.update(calc_edges=True)

        loop_vidxs = faces.ravel()
        for u_i in range(max_uvs):
            uv_verts = np.concatenate(all_uvs[u_i], axis=0)
            uv_layer = mesh.uv_layers.new(name=f"UV{u_i}")
            uv_loops = uv_verts[loop_vidxs]
            uv_layer.data.foreach_set("uv", uv_loops.ravel())

        if has_any_col0 and all_col0:
            col_verts = np.concatenate(all_col0, axis=0)
            col_attr = mesh.color_attributes.new(name="Color0", type="FLOAT_COLOR", domain="CORNER")
            col_loops = np.ascontiguousarray(col_verts[loop_vidxs], dtype=np.float32)
            col_attr.data.foreach_set("color", col_loops.ravel())

        if has_all_nrm and all_nrm:
            nrm_verts = np.concatenate(all_nrm, axis=0)
            lens = np.linalg.norm(nrm_verts, axis=1, keepdims=True)
            nrm_verts = np.where(lens > 1e-6, nrm_verts / np.maximum(lens, 1e-6), np.array([0.0, 0.0, 1.0], dtype=np.float32))
            try:
                mesh.normals_split_custom_set_from_vertices(nrm_verts)
            except Exception:
                pass

        for m in used_mats:
            mesh.materials.append(m)

        if shared_mid and shared_mesh_cache is not None:
            shared_mesh_cache[shared_mid] = mesh

    obj = bpy.data.objects.new(pname, mesh)
    obj["hfw_entity_id"] = entity_id
    obj["hfw_part_id"] = part_meta["part_id"]
    obj["hfw_source_kind"] = part_meta["source_kind"]
    target_coll.objects.link(obj)

    if part_meta["is_skinned"] and arm_obj.type == "ARMATURE":
        obj.parent = arm_obj
        mod = obj.modifiers.new(name="Armature", type="ARMATURE")
        mod.object = arm_obj
        mod.use_vertex_groups = True
        mod.use_bone_envelopes = False
        mod.use_deform_preserve_volume = False

        if has_any_weights and all_bi:
            max_k = max(b.shape[1] for b in all_bi)
            padded_bi = [
                b if b.shape[1] == max_k else np.pad(b, ((0, 0), (0, max_k - b.shape[1]))) for b in all_bi
            ]
            padded_bw = [
                w if w.shape[1] == max_k else np.pad(w, ((0, 0), (0, max_k - w.shape[1]))) for w in all_bw
            ]
            bi = np.concatenate(padded_bi, axis=0)
            bw = np.concatenate(padded_bw, axis=0)

            unique_bones = np.unique(bi[bw > 1e-5])
            vgroups: dict[int, bpy.types.VertexGroup] = {}
            for bidx in unique_bones:
                bi_int = int(bidx)
                if 0 <= bi_int < len(joint_names):
                    vgroups[bi_int] = obj.vertex_groups.new(name=joint_names[bi_int])

            for bi_int, vg in vgroups.items():
                mask = (bi == bi_int) & (bw > 1e-5)
                v_idxs, k_idxs = np.nonzero(mask)
                weights = bw[v_idxs, k_idxs]
                acc: dict[int, float] = {}
                for vi, wt in zip(v_idxs.tolist(), weights.tolist()):
                    acc[vi] = acc.get(vi, 0.0) + wt
                for vi, wt in acc.items():
                    vg.add([vi], min(1.0, wt), "REPLACE")
    else:
        # Static armor plate / weapon: position at exact bind-pose world_matrix & BONE-parent!
        world_mat = _mat4(part_meta["world_matrix"])
        target_bone_name: str | None = None
        hname = part_meta.get("helper_name")
        if hname and hname in helper_bone_map:
            target_bone_name = helper_bone_map[hname]
        elif part_meta.get("parent_joint_index") is not None:
            pidx = int(part_meta["parent_joint_index"])
            if 0 <= pidx < len(joint_names):
                target_bone_name = joint_names[pidx]

        if target_bone_name is not None and arm_obj.type == "ARMATURE":
            obj.parent = arm_obj
            obj.parent_type = "BONE"
            obj.parent_bone = target_bone_name
            pbone = arm_obj.data.bones[target_bone_name]
            bone_tail_mat = arm_obj.matrix_world @ Matrix.Translation(pbone.tail_local) @ pbone.matrix_local.to_3x3().to_4x4()
            obj.matrix_parent_inverse = bone_tail_mat.inverted() @ world_mat
            obj.matrix_basis = Matrix.Identity(4)
        else:
            obj.parent = arm_obj
            obj.matrix_world = world_mat

    if not part_meta.get("enabled_by_default", True):
        obj.hide_viewport = True
        obj.hide_render = True

    return obj


def _load_control_rig_module():
    """Load sibling `control_rig.py` cleanly inside Blender 5.3 with zero external dependencies."""
    import importlib.util

    cr_path = Path(__file__).resolve().parent / "control_rig.py"
    if not cr_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("hfw_control_rig", str(cr_path))
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bind_native_shader_drivers(
    arm_obj: Any,
    asset_kind: str,
    shader_controls: dict[str, Any] | None,
    has_plasma_charge: bool = False,
) -> None:
    """Expose native keyframeable shader Custom Properties on `arm_obj` & wire pure-C AVERAGE drivers
    directly into `HFW_MachineShaderControl` / `HFW_PlasmaChargeControl` / `HFW_CharacterControl`!

    Works 100% natively in vanilla Blender WITHOUT the HFWToolkit add-on AND even with Python disabled!
    """
    if arm_obj is None:
        return
    sc = shader_controls or {}

    def _set_col_prop(key: str, rgb: Any, desc: str) -> None:
        vals = (float(rgb[0]), float(rgb[1]), float(rgb[2]))
        arm_obj[key] = vals
        try:
            arm_obj.id_properties_ui(key).update(
                min=0.0, max=1.0, soft_min=0.0, soft_max=1.0, subtype="COLOR", description=desc
            )
        except Exception:
            pass

    def _set_flt_prop(key: str, val: float, mn: float, mx: float, desc: str, *, factor: bool = False) -> None:
        arm_obj[key] = float(val)
        try:
            kw: dict[str, Any] = {"min": mn, "max": mx, "soft_min": mn, "soft_max": mx, "description": desc}
            if factor:
                kw["subtype"] = "FACTOR"
            arm_obj.id_properties_ui(key).update(**kw)
        except Exception:
            pass

    def _drive_scalar(sock: Any, prop_key: str) -> None:
        try:
            sock.driver_remove("default_value")
        except Exception:
            pass
        try:
            fc = sock.driver_add("default_value")
            drv = fc.driver
            drv.type = "AVERAGE"
            v = drv.variables.new()
            v.name = "val"
            v.type = "SINGLE_PROP"
            v.targets[0].id_type = "OBJECT"
            v.targets[0].id = arm_obj
            v.targets[0].data_path = f'["{prop_key}"]'
        except Exception:
            pass

    def _drive_color(sock: Any, prop_key: str) -> None:
        for ch in range(3):
            try:
                sock.driver_remove("default_value", ch)
            except Exception:
                pass
            try:
                fc = sock.driver_add("default_value", ch)
                drv = fc.driver
                drv.type = "AVERAGE"
                v = drv.variables.new()
                v.name = "c"
                v.type = "SINGLE_PROP"
                v.targets[0].id_type = "OBJECT"
                v.targets[0].id = arm_obj
                v.targets[0].data_path = f'["{prop_key}"][{ch}]'
            except Exception:
                pass

    if asset_kind == "machine":
        ctrl_ng = bpy.data.node_groups.get("HFW_MachineShaderControl")
        plasma_ng = bpy.data.node_groups.get("HFW_PlasmaChargeControl")
        if ctrl_ng is None and plasma_ng is None:
            return
        arm_obj["hfw_shader_kind"] = "machine"
        mc = sc.get("muscle_color", [0.95, 0.28, 0.02])
        ec = sc.get("eye_color", [0.04, 0.56, 1.0])
        eac = sc.get("eye_alert_color", [1.0, 0.08, 0.01])
        lc = sc.get("liquid_override_color", [0.08, 0.62, 1.0])
        pc = sc.get("plasma_color", [0.46, 0.05, 1.0])
        p_def = float(sc.get("plasma_charge", 1.0 if (has_plasma_charge or sc.get("is_plasma")) else 0.0))

        _set_col_prop("hfw_sh_muscle_color", mc, "Machine optical muscle fiber glow color")
        _set_flt_prop("hfw_sh_muscle_glow", 1.0, 0.0, 4.0, "Machine muscle & cable emission strength")
        _set_col_prop("hfw_sh_eye_color", ec, "Calm / patrol eye optics color")
        _set_col_prop("hfw_sh_eye_alert_color", eac, "Hostile / combat alert eye optics color")
        _set_flt_prop("hfw_sh_eye_alert", 0.0, 0.0, 1.0, "Eye Alert blend (0 = Patrol, 1 = Alert)", factor=True)
        _set_flt_prop("hfw_sh_eye_glow", 1.0, 0.0, 4.0, "Eye optics emission strength")
        _set_col_prop("hfw_sh_liquid_color", lc, "Override color for elemental sacs & canisters")
        _set_flt_prop("hfw_sh_liquid_mix", float(sc.get("liquid_override_fac", 0.0)), 0.0, 1.0, "Override native sac/canister liquid colors", factor=True)
        _set_flt_prop("hfw_sh_liquid_glow", float(sc.get("liquid_glow_scale", 1.0)), 0.0, 4.0, "Elemental liquid & canister glow strength")
        _set_flt_prop("hfw_sh_anim_speed", float(sc.get("animation_speed", 1.0)), 0.0, 4.0, "Animated muscle/liquid/scan shader speed")
        _set_flt_prop("hfw_sh_detail_scale", float(sc.get("detail_tiling_scale", 1.0)), 0.25, 4.0, "Machine micro-detail normal/roughness UV tiling multiplier")
        _set_flt_prop("hfw_sh_detail_strength", float(sc.get("detail_strength", 0.72)), 0.0, 1.5, "Machine micro-detail surface relief strength")
        if has_plasma_charge or sc.get("is_plasma"):
            _set_col_prop("hfw_sh_plasma_color", pc, "Plasma charge-up emission color")
            _set_flt_prop("hfw_sh_plasma_charge", p_def, 0.0, 1.0, "Plasma Charge-Up (0 = Idle, 1 = Full Charge)", factor=True)
            arm_obj["hfw_plasma_charge"] = p_def

        col_map = {
            "Muscle Color": "hfw_sh_muscle_color",
            "Eye Color": "hfw_sh_eye_color",
            "Eye Alert Color": "hfw_sh_eye_alert_color",
            "Liquid Override Color": "hfw_sh_liquid_color",
            "Plasma Color": "hfw_sh_plasma_color",
        }
        val_map = {
            "Muscle Glow Scale": "hfw_sh_muscle_glow",
            "Eye Alert State": "hfw_sh_eye_alert",
            "Eye Glow Scale": "hfw_sh_eye_glow",
            "Liquid Override Mix": "hfw_sh_liquid_mix",
            "Liquid Glow Scale": "hfw_sh_liquid_glow",
            "Animation Speed": "hfw_sh_anim_speed",
            "Detail Tiling Scale": "hfw_sh_detail_scale",
            "Detail Strength": "hfw_sh_detail_strength",
            "Plasma Charge": "hfw_sh_plasma_charge",
        }
        if ctrl_ng is not None:
            for n in ctrl_ng.nodes:
                if n.type == "RGB" and n.label in col_map and col_map[n.label] in arm_obj:
                    _drive_color(n.outputs[0], col_map[n.label])
                elif n.type == "VALUE" and n.label in val_map and val_map[n.label] in arm_obj:
                    _drive_scalar(n.outputs[0], val_map[n.label])
        if plasma_ng is not None and "hfw_sh_plasma_charge" in arm_obj:
            for n in plasma_ng.nodes:
                if n.type == "VALUE" and n.label == "Plasma Charge":
                    _drive_scalar(n.outputs[0], "hfw_sh_plasma_charge")

    elif asset_kind == "character":
        char_ng = bpy.data.node_groups.get(_ACTIVE_CHAR_CTRL_NAME) or bpy.data.node_groups.get("HFW_CharacterControl")
        if char_ng is None:
            return
        arm_obj["hfw_shader_kind"] = "character"
        arm_obj["hfw_char_ctrl_ng"] = char_ng.name
        st = sc.get("skin_tone_linear") or [0.453, 0.246, 0.209]
        hc = sc.get("hair_color_linear") or [0.045, 0.032, 0.025]
        ic = sc.get("iris_color_linear") or [0.106, 0.199, 0.061]
        pf = float(sc.get("peach_fuzz_intensity", 0.65))
        _set_col_prop("hfw_sh_skin_tone", st, "Shared DXIL 570:0 Skin LUT tone across Head & Body")
        _set_flt_prop("hfw_sh_skin_brightness", 1.0, 0.25, 2.0, "Skin albedo brightness multiplier")
        _set_flt_prop("hfw_sh_sss_weight", 0.32, 0.0, 1.0, "RANDOM_WALK_SKIN subsurface scattering weight")
        _set_flt_prop("hfw_sh_peach_fuzz", pf, 0.0, 2.0, "Backlit grazing facial peach fuzz / vellus hair intensity")
        _set_col_prop("hfw_sh_iris_color", ic, "Character iris color tint")
        _set_col_prop("hfw_sh_hair_color", hc, "Shared DXIL 382:0 Hair LUT color")
        for n in char_ng.nodes:
            if n.type == "RGB" and n.label == "Skin Tone":
                _drive_color(n.outputs[0], "hfw_sh_skin_tone")
            elif n.type == "RGB" and n.label == "Hair Color":
                _drive_color(n.outputs[0], "hfw_sh_hair_color")
            elif n.type == "RGB" and n.label == "Iris Color":
                _drive_color(n.outputs[0], "hfw_sh_iris_color")
            elif n.type == "VALUE" and n.label == "Skin Brightness":
                _drive_scalar(n.outputs[0], "hfw_sh_skin_brightness")
            elif n.type == "VALUE" and n.label == "Subsurface Weight":
                _drive_scalar(n.outputs[0], "hfw_sh_sss_weight")
            elif n.type == "VALUE" and n.label == "Peach Fuzz Intensity":
                _drive_scalar(n.outputs[0], "hfw_sh_peach_fuzz")


def _embed_native_rig_ui_in_blend() -> None:
    """Embed self-contained `hfw_rig_ui.py` (`use_module = True`) into the .blend so Rig & Shader UI works WITHOUT the add-on!"""
    ui_src_path = Path(__file__).resolve().with_name("rig_ui_embedded.py")
    if not ui_src_path.is_file():
        return
    try:
        src_text = ui_src_path.read_text(encoding="utf-8")
        txt = bpy.data.texts.get("hfw_rig_ui.py")
        if txt is None:
            txt = bpy.data.texts.new("hfw_rig_ui.py")
        txt.clear()
        txt.write(src_text)
        txt.use_module = True
        # Register immediately in the active Blender session too
        ns: dict[str, Any] = {"__name__": "hfw_rig_ui"}
        exec(compile(src_text, "hfw_rig_ui.py", "exec"), ns)
        reg_fn = ns.get("register")
        if callable(reg_fn):
            reg_fn()
    except Exception as exc:
        print(f"[HFWToolkit] Note: Embedded native Rig UI skipped ({exc})")


def import_bundle_into_scene(
    bundle_dir: str | Path,
    enable_detail_nodes: bool = True,
    pack_textures: bool = False,
    build_control_rig_layer: bool = True,
    import_animations: bool = True,
) -> bpy.types.Object:
    """Import a Canonical Assembly Bundle directly into the active Blender scene."""
    bundle_path = Path(bundle_dir)
    manifest = json.loads((bundle_path / "assembly.json").read_text(encoding="utf-8"))
    npz = np.load(bundle_path / "geometry.npz")
    tex_dir = bundle_path / "textures"

    disp = manifest["display_name"]
    variant = manifest.get("variant_label", "")
    asset_kind = manifest.get("asset_kind", "machine")
    shader_controls = manifest.get("shader_controls") or {}
    root_name = f"HFW_{disp}"
    root_coll = _get_or_create_collection(root_name, bpy.context.scene.collection)

    # Create standard sub-collections dynamically for machines, characters, and general assets!
    sub_colls: dict[str, bpy.types.Collection] = {}
    default_coll_keys = (
        ("Body", "Plates_Intact", "Weapons_And_Components", "Plates_Broken_States", "Focus_Overlay", "Unattached_Misc")
        if asset_kind == "machine"
        else ("Body", "Head_Face", "Hair", "Outfit", "Accessories")
        if asset_kind == "character"
        else ("Assets",)
    )
    for ckey in default_coll_keys:
        sub_colls[ckey] = _get_or_create_collection(f"{disp}_{ckey}", root_coll)
    for pmeta in manifest["parts"]:
        ckey = pmeta.get("collection", "Body")
        if ckey and ckey not in sub_colls:
            sub_colls[ckey] = _get_or_create_collection(f"{disp}_{ckey}", root_coll)

    # Pre-initialize shared shader control node groups BEFORE building materials!
    global _ACTIVE_CHAR_CTRL_NAME
    if asset_kind == "machine":
        _ensure_machine_shader_control_group(shader_controls)
    elif asset_kind == "character":
        # Keep canonical `HFW_CharacterControl` for solo character scenes; auto-namespace when importing multiple characters!
        if bpy.data.node_groups.get("HFW_CharacterControl") is None:
            _ACTIVE_CHAR_CTRL_NAME = "HFW_CharacterControl"
        else:
            safe_tag = "".join(ch if ch.isalnum() else "_" for ch in disp).strip("_")[:24] or "Char"
            cand = f"HFW_CharacterControl_{safe_tag}"
            idx_c = 1
            while bpy.data.node_groups.get(cand) is not None:
                idx_c += 1
                cand = f"HFW_CharacterControl_{safe_tag}_{idx_c}"
            _ACTIVE_CHAR_CTRL_NAME = cand
        _ensure_character_control_group(shader_controls, ng_name=_ACTIVE_CHAR_CTRL_NAME)

    img_cache: dict[str, bpy.types.Image] = {}
    mat_map: dict[str, bpy.types.Material] = {}
    for mid, mmeta in manifest["materials"].items():
        mat_map[mid] = _build_material(mmeta, tex_dir, img_cache, enable_detail_nodes=enable_detail_nodes)

    arm_obj, joint_names, helper_bone_map = _build_armature(
        disp, manifest["joints"], manifest["helpers"], root_coll
    )
    arm_obj["hfw_entity_id"] = manifest["entity_id"]
    arm_obj["hfw_codename"] = manifest["codename"]
    arm_obj["hfw_variant"] = variant
    arm_obj["hfw_asset_kind"] = asset_kind
    if shader_controls:
        arm_obj["hfw_variant_profile"] = str(shader_controls.get("profile", "standard"))

    # Wire native keyframeable Armature shader drivers & embed add-on-free `hfw_rig_ui.py`!
    has_plasma_charge = any(
        (mmeta.get("emissive_overlay") or {}).get("kind") == "plasma_charge_overlay"
        for mmeta in manifest["materials"].values()
    )
    _bind_native_shader_drivers(arm_obj, asset_kind, shader_controls, has_plasma_charge=has_plasma_charge)

    created = 0
    shared_mesh_cache: dict[str, bpy.types.Mesh] = {}
    for pmeta in manifest["parts"]:
        coll = sub_colls.get(pmeta["collection"], root_coll)
        obj = _build_part_mesh_object(
            pmeta,
            npz,
            mat_map,
            joint_names,
            arm_obj,
            helper_bone_map,
            coll,
            manifest["entity_id"],
            shared_mesh_cache=shared_mesh_cache,
        )
        if obj is not None:
            created += 1

    # Auto-bump camera & 3D viewport clip_end for colossal machines (e.g. 289m FAS-BOR7 Horus!)
    try:
        jpos_arr = np.array([[j["world_matrix"][0][3], j["world_matrix"][1][3], j["world_matrix"][2][3]] for j in manifest.get("joints", ())], dtype=np.float64)
        if len(jpos_arr) >= 2:
            span = float(np.linalg.norm(jpos_arr.max(axis=0) - jpos_arr.min(axis=0)))
            if span > 60.0:
                clip_target = max(2500.0, span * 8.0)
                if bpy.context.scene.camera and bpy.context.scene.camera.data:
                    bpy.context.scene.camera.data.clip_end = max(bpy.context.scene.camera.data.clip_end, clip_target)
                for scr in bpy.data.screens:
                    for area in scr.areas:
                        if area.type == "VIEW_3D":
                            for space in area.spaces:
                                if space.type == "VIEW_3D":
                                    space.clip_end = max(space.clip_end, clip_target)
    except Exception:
        pass

    def _hide_layer_collection(layer_coll: bpy.types.LayerCollection, target_name: str) -> None:
        if layer_coll.collection.name == target_name:
            layer_coll.hide_viewport = True
            return
        for child in layer_coll.children:
            _hide_layer_collection(child, target_name)

    for hidden_key in ("Plates_Broken_States", "Focus_Overlay", "Unattached_Misc"):
        if hidden_key in sub_colls:
            _hide_layer_collection(bpy.context.view_layer.layer_collection, sub_colls[hidden_key].name)
            sub_colls[hidden_key].hide_render = True

    # Ensure Cycles never hits a transparent-bounce blackout when viewing overlapping optical muscle fibers!
    if hasattr(bpy.context.scene, "cycles") and hasattr(bpy.context.scene.cycles, "transparent_max_bounces"):
        try:
            if bpy.context.scene.cycles.transparent_max_bounces < 16:
                bpy.context.scene.cycles.transparent_max_bounces = 16
        except Exception:
            pass

    if pack_textures:
        try:
            bpy.ops.file.pack_all()
        except Exception:
            pass

    npz.close()

    # Build non-destructive `DEF + MCH + CTRL` Control Rig + import any extracted HFW animations!
    has_bundle_anims = bool(manifest.get("animations")) and (bundle_path / "animations.npz").exists()
    rig_summary_str = ""
    anim_count = 0
    if arm_obj.type == "ARMATURE" and (build_control_rig_layer or (import_animations and has_bundle_anims)):
        cr_mod = _load_control_rig_module()
        if cr_mod is not None:
            if build_control_rig_layer and len(joint_names) >= 8 and asset_kind in ("machine", "character"):
                try:
                    # Default to Control Rig ON (1.0) when posing, or Game Anim ON (0.0) when clips were loaded!
                    def_master = 0.0 if (import_animations and has_bundle_anims) else 1.0
                    rinfo = cr_mod.build_control_rig(arm_obj, default_master=def_master, rebuild=True)
                    rig_summary_str = (
                        f", ControlRig({rinfo['archetype']}: {rinfo['control_count']} ctrls, "
                        f"{rinfo['ik_limb_count']} IK limbs)"
                    )
                except Exception as exc:
                    print(f"[HFWToolkit] Warning: Control Rig build skipped ({exc})")
            if import_animations and has_bundle_anims:
                try:
                    acts = cr_mod.import_animations_into_armature(
                        arm_obj,
                        bundle_path,
                        activate_first=True,
                        switch_to_anim_mode=True,
                    )
                    anim_count = len(acts)
                except Exception as exc:
                    print(f"[HFWToolkit] Warning: Animation import skipped ({exc})")

    # Always embed the add-on-free `hfw_rig_ui.py` so both Rig Controls & Shader Controls work natively in `Item`!
    _embed_native_rig_ui_in_blend()

    anim_summary_str = f", {anim_count} HFW animations" if anim_count else ""
    print(
        f"[HFWToolkit] Imported '{disp}' ({variant}, textures={manifest.get('texture_mode', 'packed')}): "
        f"{len(joint_names)} bones, {len(helper_bone_map)} helpers, {created} parts, {len(mat_map)} materials"
        f"{rig_summary_str}{anim_summary_str}."
    )
    return arm_obj


def _cli_main() -> None:
    if "--" not in sys.argv:
        print("Usage: blender --background --python blender_builder.py -- <bundle_dir> <out.blend> [--pack] [--no-detail] [--no-control-rig] [--export-fbx <out.fbx>]")
        sys.exit(2)
    args = sys.argv[sys.argv.index("--") + 1 :]
    bundle_dir = Path(args[0])
    out_blend = Path(args[1])
    pack = "--pack" in args
    enable_detail = "--no-detail" not in args
    build_ctrl = "--no-control-rig" not in args
    export_fbx: Path | None = None
    if "--export-fbx" in args:
        idx = args.index("--export-fbx")
        if idx + 1 < len(args):
            export_fbx = Path(args[idx + 1])

    bpy.ops.wm.read_factory_settings(use_empty=True)
    arm_obj = import_bundle_into_scene(
        bundle_dir,
        enable_detail_nodes=enable_detail,
        pack_textures=pack,
        build_control_rig_layer=build_ctrl,
        import_animations=True,
    )

    out_blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out_blend.resolve()))
    print(f"[HFWToolkit] Saved Blender file: {out_blend}")

    if export_fbx is not None:
        export_fbx.parent.mkdir(parents=True, exist_ok=True)
        # Export base SkeletalMesh FBX in clean bind pose using ONLY deform (`DEF`) bones (no CTRL_/MCH_ pollution!)
        prev_action = arm_obj.animation_data.action if (arm_obj and arm_obj.animation_data) else None
        prev_master = float(arm_obj.get("hfw_rig_master", 0.0)) if arm_obj else 0.0
        if arm_obj and arm_obj.type == "ARMATURE":
            if "hfw_rig_master" in arm_obj:
                arm_obj["hfw_rig_master"] = 0.0
            if arm_obj.animation_data:
                arm_obj.animation_data.action = None
            for pb in arm_obj.pose.bones:
                pb.matrix_basis = Matrix.Identity(4)
            arm_obj.update_tag()
            bpy.context.view_layer.update()

        bpy.ops.object.select_all(action="DESELECT")
        for obj in bpy.context.scene.objects:
            if not obj.hide_viewport and not obj.hide_render:
                obj.select_set(True)
        try:
            fbx_kwargs: dict[str, Any] = dict(
                filepath=str(export_fbx.resolve()),
                use_selection=True,
                object_types={"ARMATURE", "MESH"},
                use_armature_deform_only=True,
                add_leaf_bones=False,
                bake_anim=False,
                mesh_smooth_type="FACE",
                path_mode="COPY",
                embed_textures=False,
            )
            try:
                bpy.ops.export_scene.fbx(**fbx_kwargs, colors_type="LINEAR")
            except TypeError:
                bpy.ops.export_scene.fbx(**fbx_kwargs)
            print(f"[HFWToolkit] Saved UE5-ready rigged FBX: {export_fbx}")
        except Exception as exc:
            print(f"[HFWToolkit] Warning: FBX export failed ({exc})")

        # Export any imported HFW game animations as UE5-ready skeletal `.fbx` clips in `animations/`!
        if arm_obj and arm_obj.type == "ARMATURE":
            game_acts = [a for a in bpy.data.actions if a.get("hfw_anim_kind") in ("game_def", "baked_def")]
            if game_acts:
                cr_mod = _load_control_rig_module()
                anim_dir = export_fbx.parent / "animations"
                anim_dir.mkdir(parents=True, exist_ok=True)
                manifest_path = bundle_dir / "assembly.json"
                fbx_name_by_aid: dict[str, str] = {}
                if manifest_path.exists():
                    try:
                        mjson = json.loads(manifest_path.read_text(encoding="utf-8"))
                        for am in mjson.get("animations") or ():
                            fbx_name_by_aid[str(am.get("anim_id", ""))] = str(am.get("fbx_filename", ""))
                    except Exception:
                        pass

                bpy.ops.object.select_all(action="DESELECT")
                arm_obj.select_set(True)
                bpy.context.view_layer.objects.active = arm_obj
                exported_clips = 0
                for act in game_acts:
                    try:
                        if cr_mod is not None:
                            cr_mod.assign_action_to_armature(arm_obj, act)
                        else:
                            arm_obj.animation_data.action = act
                        f0, f1 = act.frame_range
                        bpy.context.scene.frame_start = int(round(f0))
                        bpy.context.scene.frame_end = max(int(round(f0)) + 1, int(round(f1)))
                        bpy.context.scene.frame_set(bpy.context.scene.frame_start)
                        bpy.context.view_layer.update()

                        aid = str(act.get("hfw_anim_id", ""))
                        fname = fbx_name_by_aid.get(aid) or f"{act.name}.fbx"
                        if not fname.lower().endswith(".fbx"):
                            fname += ".fbx"
                        afbx = anim_dir / fname
                        bpy.ops.export_scene.fbx(
                            filepath=str(afbx.resolve()),
                            use_selection=True,
                            object_types={"ARMATURE"},
                            use_armature_deform_only=True,
                            add_leaf_bones=False,
                            bake_anim=True,
                            bake_anim_use_all_bones=True,
                            bake_anim_use_nla_strips=False,
                            bake_anim_use_all_actions=False,
                            bake_anim_force_startend_keying=True,
                            bake_anim_simplify_factor=0.0,
                        )
                        exported_clips += 1
                    except Exception as exc:
                        print(f"[HFWToolkit] Warning: Animation FBX export failed for {act.name} ({exc})")
                print(f"[HFWToolkit] Exported {exported_clips} UE5 AnimSequence FBX clip(s) -> {anim_dir}")

        if arm_obj and prev_action is not None and arm_obj.animation_data:
            arm_obj.animation_data.action = prev_action
        if arm_obj and "hfw_rig_master" in arm_obj:
            arm_obj["hfw_rig_master"] = prev_master


if __name__ == "__main__":
    _cli_main()
