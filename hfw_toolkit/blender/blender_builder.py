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
    ng.interface.new_socket(name="Time", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 0.0
    s_gc = ng.interface.new_socket(name="Sac Glow Color", in_out="INPUT", socket_type="NodeSocketColor")
    s_gc.default_value = (0.14, 0.68, 1.0, 1.0)
    ng.interface.new_socket(name="Glow Strength", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 6.5
    ng.interface.new_socket(name="Flow Speed", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 0.35
    ng.interface.new_socket(name="Flow Strength", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 0.14
    ng.interface.new_socket(name="Animate Speed", in_out="INPUT", socket_type="NodeSocketFloat").default_value = 1.0

    ng.interface.new_socket(name="Color", in_out="OUTPUT", socket_type="NodeSocketColor")
    ng.interface.new_socket(name="Smoothness", in_out="OUTPUT", socket_type="NodeSocketFloat")
    ng.interface.new_socket(name="Emission Color", in_out="OUTPUT", socket_type="NodeSocketColor")
    ng.interface.new_socket(name="Emission Strength", in_out="OUTPUT", socket_type="NodeSocketFloat")

    n_in = ng.nodes.new("NodeGroupInput")
    n_in.location = (-1650, 0)
    n_out = ng.nodes.new("NodeGroupOutput")
    n_out.location = (960, 0)

    # 1. Exact DXIL sac isolation mask: (round(Mask * 20.0) == 0) * clamp(Mask_Alpha, 0, 1)
    m20 = ng.nodes.new("ShaderNodeMath")
    m20.operation = "MULTIPLY"
    m20.inputs[1].default_value = 20.0
    m20.location = (-1380, 420)
    ng.links.new(n_in.outputs["Mask"], m20.inputs[0])

    rnd = ng.nodes.new("ShaderNodeMath")
    rnd.operation = "ROUND"
    rnd.location = (-1200, 420)
    ng.links.new(m20.outputs["Value"], rnd.inputs[0])

    is_zero = ng.nodes.new("ShaderNodeMath")
    is_zero.operation = "COMPARE"
    is_zero.inputs[1].default_value = 0.0
    is_zero.inputs[2].default_value = 0.25
    is_zero.location = (-1020, 420)
    ng.links.new(rnd.outputs["Value"], is_zero.inputs[0])

    sac_mask = ng.nodes.new("ShaderNodeMath")
    sac_mask.operation = "MULTIPLY"
    sac_mask.use_clamp = True
    sac_mask.label = "Sac Mask (mat_id==0 * MaskAlpha)"
    sac_mask.location = (-820, 380)
    ng.links.new(is_zero.outputs["Value"], sac_mask.inputs[0])
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

    # 6. Fluid & Breathing Pulse Modulation: rich swirling caustic contrast (0.28 + 1.25 * fluid) * breathing pulse
    fl_mod = ng.nodes.new("ShaderNodeMath")
    fl_mod.operation = "MULTIPLY_ADD"
    fl_mod.inputs[1].default_value = 1.25
    fl_mod.inputs[2].default_value = 0.28
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

    # 7. Subtle translucent tint on Base Color & rubbery sheen Smoothness ONLY where sac_mask > 0
    tint_w = ng.nodes.new("ShaderNodeMath")
    tint_w.operation = "MULTIPLY"
    tint_w.inputs[1].default_value = 0.45
    tint_w.location = (440, 340)
    ng.links.new(sac_mask.outputs["Value"], tint_w.inputs[0])

    mix_bc = ng.nodes.new("ShaderNodeMix")
    mix_bc.data_type = "RGBA"
    mix_bc.location = (680, 340)
    ng.links.new(tint_w.outputs["Value"], mix_bc.inputs["Factor"])
    ng.links.new(n_in.outputs["Base Color"], mix_bc.inputs[6])
    ng.links.new(n_in.outputs["Sac Glow Color"], mix_bc.inputs[7])

    mix_sm = ng.nodes.new("ShaderNodeMix")
    mix_sm.data_type = "FLOAT"
    mix_sm.inputs[3].default_value = 0.72
    mix_sm.location = (680, 180)
    ng.links.new(sac_mask.outputs["Value"], mix_sm.inputs["Factor"])
    ng.links.new(n_in.outputs["Base Smoothness"], mix_sm.inputs[2])

    ng.links.new(mix_bc.outputs[2], n_out.inputs["Color"])
    ng.links.new(mix_sm.outputs[0], n_out.inputs["Smoothness"])
    ng.links.new(mix_rim.outputs[2], n_out.inputs["Emission Color"])
    ng.links.new(em_str.outputs["Value"], n_out.inputs["Emission Strength"])
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

    # braid_glow = saturate((Braid - 0.05) * 1.053) * (0.45 + Pulse * 1.2)
    br_adj = ng.nodes.new("ShaderNodeMath")
    br_adj.operation = "MULTIPLY_ADD"
    br_adj.inputs[1].default_value = 1.053
    br_adj.inputs[2].default_value = -0.053
    br_adj.use_clamp = True
    br_adj.location = (-480, 220)
    ng.links.new(n_in.outputs["Braid Mask"], br_adj.inputs[0])

    pl_scale = ng.nodes.new("ShaderNodeMath")
    pl_scale.operation = "MULTIPLY_ADD"
    pl_scale.inputs[1].default_value = 1.2
    pl_scale.inputs[2].default_value = 0.45
    pl_scale.location = (-480, 40)
    ng.links.new(n_in.outputs["Pulse Mask"], pl_scale.inputs[0])

    br_pl = ng.nodes.new("ShaderNodeMath")
    br_pl.operation = "MULTIPLY"
    br_pl.location = (-240, 140)
    ng.links.new(br_adj.outputs["Value"], br_pl.inputs[0])
    ng.links.new(pl_scale.outputs["Value"], br_pl.inputs[1])

    # Add circuit veins + Fresnel view-angle edge boost
    vn_add = ng.nodes.new("ShaderNodeMath")
    vn_add.operation = "ADD"
    vn_add.use_clamp = True
    vn_add.location = (-40, 100)
    ng.links.new(br_pl.outputs["Value"], vn_add.inputs[0])
    ng.links.new(n_in.outputs["Vein Mask"], vn_add.inputs[1])

    lw = ng.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.35
    lw.location = (-240, -80)

    fr_boost = ng.nodes.new("ShaderNodeMath")
    fr_boost.operation = "MULTIPLY_ADD"
    fr_boost.inputs[1].default_value = 0.65
    fr_boost.inputs[2].default_value = 0.65
    fr_boost.location = (-40, -80)
    ng.links.new(lw.outputs["Facing"], fr_boost.inputs[0])

    tot_m = ng.nodes.new("ShaderNodeMath")
    tot_m.operation = "MULTIPLY"
    tot_m.location = (140, 60)
    ng.links.new(vn_add.outputs["Value"], tot_m.inputs[0])
    ng.links.new(fr_boost.outputs["Value"], tot_m.inputs[1])

    tot_str = ng.nodes.new("ShaderNodeMath")
    tot_str.operation = "MULTIPLY"
    tot_str.location = (320, 40)
    ng.links.new(tot_m.outputs["Value"], tot_str.inputs[0])
    ng.links.new(n_in.outputs["Glow Strength"], tot_str.inputs[1])

    mix_c = ng.nodes.new("ShaderNodeMix")
    mix_c.data_type = "RGBA"
    mix_c.location = (220, -200)
    ng.links.new(n_in.outputs["Alert State"], mix_c.inputs["Factor"])
    ng.links.new(n_in.outputs["Glow Color"], mix_c.inputs[6])
    ng.links.new(n_in.outputs["Alert Color"], mix_c.inputs[7])

    ng.links.new(mix_c.outputs[2], n_out.inputs["Emission Color"])
    ng.links.new(tot_str.outputs["Value"], n_out.inputs["Emission Strength"])
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

    # 1. Smooth BC7-noise-free Mask_Alpha modulation (eliminates harsh dark speckles inside muscle grooves!)
    ma_mr = ng.nodes.new("ShaderNodeMapRange")
    ma_mr.interpolation_type = "SMOOTHSTEP"
    ma_mr.clamp = True
    ma_mr.label = "Smooth Cavity Mask (No BC7 Noise)"
    ma_mr.inputs["From Min"].default_value = 0.004
    ma_mr.inputs["From Max"].default_value = 0.09
    ma_mr.inputs["To Min"].default_value = 0.16
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

    # Crisp thin optical fiber strands (`abs` is 0..0.5; smoothstep 0.18..0.46 -> 0.03..1.0 leaves dark synthetic cable visible!)
    f_x2 = ng.nodes.new("ShaderNodeMapRange")
    f_x2.interpolation_type = "SMOOTHSTEP"
    f_x2.clamp = True
    f_x2.label = "96-Strand Optical Filaments"
    f_x2.inputs["From Min"].default_value = 0.18
    f_x2.inputs["From Max"].default_value = 0.46
    f_x2.inputs["To Min"].default_value = 0.03
    f_x2.inputs["To Max"].default_value = 1.0
    f_x2.location = (100, -200)
    ng.links.new(f_abs.outputs["Value"], f_x2.inputs["Value"])

    # 3. DXIL View-Facing Mask (`saturate(dot(N, V) * 2.5)` — softens grazing muscle curves!)
    lw = ng.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.45
    lw.location = (-560, 40)

    face_mr = ng.nodes.new("ShaderNodeMapRange")
    face_mr.interpolation_type = "SMOOTHSTEP"
    face_mr.clamp = True
    face_mr.label = "View-Facing Softening"
    face_mr.inputs["From Min"].default_value = 0.0
    face_mr.inputs["From Max"].default_value = 0.92
    face_mr.inputs["To Min"].default_value = 1.0
    face_mr.inputs["To Max"].default_value = 0.15
    face_mr.location = (-320, 40)
    ng.links.new(lw.outputs["Facing"], face_mr.inputs["Value"])

    # 4. Combine smooth cavity * energy wave * optical strands * view-facing * Glow Strength
    m1 = ng.nodes.new("ShaderNodeMath")
    m1.operation = "MULTIPLY"
    m1.location = (-100, 180)
    ng.links.new(ma_mr.outputs["Result"], m1.inputs[0])
    ng.links.new(n_in.outputs["Energy Wave (5147)"], m1.inputs[1])

    m2 = ng.nodes.new("ShaderNodeMath")
    m2.operation = "MULTIPLY"
    m2.location = (100, 140)
    ng.links.new(m1.outputs["Value"], m2.inputs[0])
    ng.links.new(f_x2.outputs["Result"], m2.inputs[1])

    m3 = ng.nodes.new("ShaderNodeMath")
    m3.operation = "MULTIPLY"
    m3.location = (260, 120)
    ng.links.new(m2.outputs["Value"], m3.inputs[0])
    ng.links.new(face_mr.outputs["Result"], m3.inputs[1])

    tot_str = ng.nodes.new("ShaderNodeMath")
    tot_str.operation = "MULTIPLY"
    tot_str.location = (420, 120)
    ng.links.new(m3.outputs["Value"], tot_str.inputs[0])
    ng.links.new(n_in.outputs["Glow Strength"], tot_str.inputs[1])

    # 5. Color state blend: Calm Amber -> Plasma Violet -> Alert Red
    mix_cp = ng.nodes.new("ShaderNodeMix")
    mix_cp.data_type = "RGBA"
    mix_cp.location = (-40, 380)
    ng.links.new(n_in.outputs["Charge State"], mix_cp.inputs["Factor"])
    ng.links.new(n_in.outputs["Calm Color"], mix_cp.inputs[6])
    ng.links.new(n_in.outputs["Plasma Color"], mix_cp.inputs[7])

    mix_al = ng.nodes.new("ShaderNodeMix")
    mix_al.data_type = "RGBA"
    mix_al.location = (200, 380)
    ng.links.new(n_in.outputs["Alert State"], mix_al.inputs["Factor"])
    ng.links.new(mix_cp.outputs[2], mix_al.inputs[6])
    ng.links.new(n_in.outputs["Alert Color"], mix_al.inputs[7])

    # 6. Pure additive optical glow + Cycles/Eevee Shadow-Ray & Backface Bypass (ZERO Cycles z-fight/shadow/bounce glitches!)
    tr_bsdf = ng.nodes.new("ShaderNodeBsdfTransparent")
    tr_bsdf.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    tr_bsdf.location = (420, -100)

    em_bsdf = ng.nodes.new("ShaderNodeEmission")
    em_bsdf.location = (420, 260)
    ng.links.new(mix_al.outputs[2], em_bsdf.inputs["Color"])
    ng.links.new(tot_str.outputs["Value"], em_bsdf.inputs["Strength"])

    add_sh = ng.nodes.new("ShaderNodeAddShader")
    add_sh.location = (620, 100)
    ng.links.new(tr_bsdf.outputs["BSDF"], add_sh.inputs[0])
    ng.links.new(em_bsdf.outputs["Emission"], add_sh.inputs[1])

    lp = ng.nodes.new("ShaderNodeLightPath")
    lp.location = (220, -340)
    geo = ng.nodes.new("ShaderNodeNewGeometry")
    geo.location = (220, -520)

    bypass = ng.nodes.new("ShaderNodeMath")
    bypass.operation = "MAXIMUM"
    bypass.label = "Skip Shadow & Backface Rays"
    bypass.location = (460, -380)
    ng.links.new(lp.outputs["Is Shadow Ray"], bypass.inputs[0])
    ng.links.new(geo.outputs["Backfacing"], bypass.inputs[1])

    mix_safe = ng.nodes.new("ShaderNodeMixShader")
    mix_safe.label = "Cycles/Eevee Safe Additive Overlay"
    mix_safe.location = (800, 20)
    ng.links.new(bypass.outputs["Value"], mix_safe.inputs["Fac"])
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
        gc = spec.get("glow_color", [0.0, 0.231, 1.0])
        ac = spec.get("alert_color", [0.95, 0.08, 0.02])
        mc_grp.inputs["Glow Color"].default_value = (gc[0], gc[1], gc[2], 1.0)
        mc_grp.inputs["Alert Color"].default_value = (ac[0], ac[1], ac[2], 1.0)
        mc_grp.inputs["Glow Strength"].default_value = float(spec.get("glow_strength", 6.0))

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

        # 5645:0 (scrolling spark/pulse streaks) at UV * (6, 3) + animated (0, Time * 0.65)!
        pulse_img = _load_image(tex_dir, spec.get("pulse_file", ""), "Non-Color", img_cache)
        if pulse_img is not None:
            t_drv = _add_time_driver_node(nt, (-1540, -140), "HFW Muscle Pulse Time")
            t_scr = nt.nodes.new("ShaderNodeMath")
            t_scr.operation = "MULTIPLY"
            t_scr.inputs[1].default_value = -0.65
            t_scr.location = (-1340, -140)
            nt.links.new(t_drv.outputs[0], t_scr.inputs[0])

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

        cc = spec.get("calm_color", [0.95, 0.24, 0.02])
        pc = spec.get("plasma_color", [0.46, 0.05, 1.0])
        ac = spec.get("alert_color", [1.0, 0.04, 0.01])
        mf_grp.inputs["Calm Color"].default_value = (cc[0], cc[1], cc[2], 1.0)
        mf_grp.inputs["Plasma Color"].default_value = (pc[0], pc[1], pc[2], 1.0)
        mf_grp.inputs["Alert Color"].default_value = (ac[0], ac[1], ac[2], 1.0)
        c_state = float(spec.get("charge_state", 0.0))
        mf_grp.inputs["Charge State"].default_value = c_state
        if c_state >= 0.5:
            # Link Slaughterspine's optical muscle fibers to the shared single-slider `HFW_PlasmaChargeControl`!
            pctrl = nt.nodes.new("ShaderNodeGroup")
            pctrl.node_tree = _ensure_plasma_charge_control_group()
            pctrl.label = "Slaughterspine Plasma Charge"
            pctrl.location = (-120, 420)
            nt.links.new(pctrl.outputs["Plasma Charge"], mf_grp.inputs["Charge State"])
        mf_grp.inputs["Alert State"].default_value = float(spec.get("alert_state", 0.0))
        mf_grp.inputs["Glow Strength"].default_value = float(spec.get("glow_strength", 3.4))
        mf_grp.inputs["Fiber Density"].default_value = float(spec.get("fiber_density", 96.0))

        # Bind UV0 to existing NMT_AO_MSKA texture node(s) so Mask_Alpha uses UV0!
        if "Mask_Alpha" in sockets:
            uv0_n = nt.nodes.new("ShaderNodeUVMap")
            uv0_n.uv_map = "UV0"
            uv0_n.location = (-1180, 320)
            for node in nt.nodes:
                if node.type == "TEX_IMAGE":
                    nt.links.new(uv0_n.outputs["UV"], node.inputs["Vector"])
            nt.links.new(sockets["Mask_Alpha"], mf_grp.inputs["Mask Alpha (UV0)"])

        # Sample 5147:0 energy wave ramp on UV1 * (6, 12) + animated (0, Time * 0.50) (DXIL L236-251, L290, L318)
        ramp_img = _load_image(tex_dir, spec.get("ramp_file", ""), "Non-Color", img_cache)
        if ramp_img is not None:
            t_drv = _add_time_driver_node(nt, (-1520, -340), "HFW Fiber Wave Time")
            t_scr = nt.nodes.new("ShaderNodeMath")
            t_scr.operation = "MULTIPLY"
            t_scr.inputs[1].default_value = -0.50
            t_scr.location = (-1320, -340)
            nt.links.new(t_drv.outputs[0], t_scr.inputs[0])

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

            # Combine Blue + Alpha energy wave bands (DXIL L291 + L319 -> L338)
            wave_add = nt.nodes.new("ShaderNodeMath")
            wave_add.operation = "ADD"
            wave_add.use_clamp = True
            wave_add.location = (-140, -160)
            nt.links.new(sep_r.outputs["Blue"], wave_add.inputs[0])
            nt.links.new(t_rmp.outputs["Alpha"], wave_add.inputs[1])

            wave_map = nt.nodes.new("ShaderNodeMapRange")
            wave_map.clamp = True
            wave_map.inputs["From Min"].default_value = 0.0
            wave_map.inputs["From Max"].default_value = 1.0
            wave_map.inputs["To Min"].default_value = 0.35
            wave_map.inputs["To Max"].default_value = 1.0
            wave_map.location = (20, -160)
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
        # Class F: Inner glowing machine eye optical sensor (4628:0) with animated scan pulse!
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.location = (360, 0)
        bsdf.inputs["Base Color"].default_value = (0.01, 0.02, 0.04, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.05
        t_drv = _add_time_driver_node(nt, (-560, -140), "HFW Eye Pulse Time")
        el_grp = nt.nodes.new("ShaderNodeGroup")
        el_grp.node_tree = _ensure_eye_lens_group(None)
        el_grp.label = "HFW Machine Eye Glow (4628:0)"
        el_grp.location = (-120, -80)
        gc = spec.get("glow_color", [0.04, 0.52, 1.0])
        ac = spec.get("alert_color", [1.0, 0.12, 0.02])
        el_grp.inputs["Calm Color"].default_value = (gc[0], gc[1], gc[2], 1.0)
        el_grp.inputs["Alert Color"].default_value = (ac[0], ac[1], ac[2], 1.0)
        el_grp.inputs["Glow Strength"].default_value = float(spec.get("glow_strength", 9.0))
        nt.links.new(t_drv.outputs[0], el_grp.inputs["Time"])
        if "Emission Color" in bsdf.inputs:
            nt.links.new(el_grp.outputs["Emission Color"], bsdf.inputs["Emission Color"])
        if "Emission Strength" in bsdf.inputs:
            nt.links.new(el_grp.outputs["Emission Strength"], bsdf.inputs["Emission Strength"])
        nt.links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])
        return True

    if kind == "canister_glow_core":
        # Class G: Elemental canister glowing liquid core (21080:1) with scrolling 107:0 fluid noise & pulse!
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.location = (380, 0)
        gc = spec.get("glow_color", [0.05, 0.48, 1.0])
        bsdf.inputs["Base Color"].default_value = (gc[0] * 0.25, gc[1] * 0.25, gc[2] * 0.25, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.12

        t_drv = _add_time_driver_node(nt, (-980, -140), "HFW Canister Core Time")
        t_f = nt.nodes.new("ShaderNodeMath")
        t_f.operation = "MULTIPLY"
        t_f.inputs[1].default_value = 3.5
        t_f.location = (-740, -140)
        nt.links.new(t_drv.outputs[0], t_f.inputs[0])

        s_p = nt.nodes.new("ShaderNodeMath")
        s_p.operation = "SINE"
        s_p.location = (-560, -140)
        nt.links.new(t_f.outputs["Value"], s_p.inputs[0])

        pulse = nt.nodes.new("ShaderNodeMath")
        pulse.operation = "MULTIPLY_ADD"
        pulse.inputs[1].default_value = 0.22
        pulse.inputs[2].default_value = 0.80
        pulse.location = (-360, -140)
        nt.links.new(s_p.outputs["Value"], pulse.inputs[0])

        noise_img = _load_image(tex_dir, spec.get("noise_file", ""), "Non-Color", img_cache)
        if noise_img is not None:
            tc_n = nt.nodes.new("ShaderNodeTexCoord")
            tc_n.location = (-980, 180)
            t_scr = nt.nodes.new("ShaderNodeMath")
            t_scr.operation = "MULTIPLY"
            t_scr.inputs[1].default_value = 0.30
            t_scr.location = (-740, 80)
            nt.links.new(t_drv.outputs[0], t_scr.inputs[0])
            c_off = nt.nodes.new("ShaderNodeCombineXYZ")
            c_off.location = (-560, 80)
            nt.links.new(t_scr.outputs["Value"], c_off.inputs["Y"])
            mp_n = nt.nodes.new("ShaderNodeMapping")
            mp_n.inputs["Scale"].default_value = (2.0, 2.0, 1.0)
            mp_n.location = (-380, 180)
            nt.links.new(tc_n.outputs["UV"], mp_n.inputs["Vector"])
            nt.links.new(c_off.outputs["Vector"], mp_n.inputs["Location"])
            t_n = nt.nodes.new("ShaderNodeTexImage")
            t_n.image = noise_img
            t_n.label = "HFW Canister Core Noise (107:0)"
            t_n.location = (-160, 180)
            nt.links.new(mp_n.outputs["Vector"], t_n.inputs["Vector"])
            n_mod = nt.nodes.new("ShaderNodeMath")
            n_mod.operation = "MULTIPLY_ADD"
            n_mod.inputs[1].default_value = 0.65
            n_mod.inputs[2].default_value = 0.55
            n_mod.location = (80, 140)
            nt.links.new(t_n.outputs["Color"], n_mod.inputs[0])
            comb_m = nt.nodes.new("ShaderNodeMath")
            comb_m.operation = "MULTIPLY"
            comb_m.location = (200, -40)
            nt.links.new(n_mod.outputs["Value"], comb_m.inputs[0])
            nt.links.new(pulse.outputs["Value"], comb_m.inputs[1])
            em_base = comb_m.outputs["Value"]
        else:
            em_base = pulse.outputs["Value"]

        em_str = nt.nodes.new("ShaderNodeMath")
        em_str.operation = "MULTIPLY"
        em_str.inputs[1].default_value = float(spec.get("glow_strength", 6.5))
        em_str.location = (240, -180)
        nt.links.new(em_base, em_str.inputs[0])

        if "Emission Color" in bsdf.inputs:
            bsdf.inputs["Emission Color"].default_value = (gc[0], gc[1], gc[2], 1.0)
        if "Emission Strength" in bsdf.inputs:
            nt.links.new(em_str.outputs["Value"], bsdf.inputs["Emission Strength"])
        nt.links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])
        return True

    if kind == "canister_glass":
        # Class E: Translucent canister outer glass shell (41249:0)
        _configure_blended_material(mat)
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.location = (360, 0)
        tc = spec.get("tint_color", [0.80, 0.92, 0.91])
        bsdf.inputs["Base Color"].default_value = (tc[0], tc[1], tc[2], 1.0)
        bsdf.inputs["Roughness"].default_value = 0.08
        bsdf.inputs["Alpha"].default_value = float(spec.get("alpha", 0.25))
        if "Transmission Weight" in bsdf.inputs:
            bsdf.inputs["Transmission Weight"].default_value = 0.85
        nt.links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])
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
                for lane, spec in packing.items():
                    parts = spec.split(".")
                    stype = parts[0]
                    sch = parts[1] if len(parts) > 1 else "R"
                    out_sock = tex_node.outputs["Alpha"] if lane == "A" else sep.outputs[lane_out_name[lane]]
                    if stype == "Normal":
                        sockets.setdefault(f"Normal.{sch}", out_sock)
                    else:
                        sockets.setdefault(stype, out_sock)

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
            da_grp.inputs["Tile X"].default_value = float(detail_arr.get("tile_x", 12.0))
            da_grp.inputs["Tile Y"].default_value = float(detail_arr.get("tile_y", 12.0))
            da_grp.inputs["Detail Strength"].default_value = 0.65
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

    # 2b. DXIL-Verified Animated Elemental Sack / Canister / Eye Lens Overlays!
    em_overlay = mat_meta.get("emissive_overlay")
    overlay_em_color = None
    overlay_em_strength = None

    if em_overlay:
        okind = em_overlay.get("kind", "")
        mat["hfw_emissive_overlay"] = okind

        if okind == "elemental_sack_overlay" and "Mask" in sockets:
            flow_img = _load_image(tex_dir, em_overlay.get("flow_file", ""), "Non-Color", img_cache)
            fluid_img = _load_image(tex_dir, em_overlay.get("fluid_file", ""), "Non-Color", img_cache)
            t_drv = _add_time_driver_node(nt, (-160, 520), "HFW Sac Flow Time")
            sac_grp = nt.nodes.new("ShaderNodeGroup")
            sac_grp.node_tree = _ensure_elemental_sack_group(flow_img, fluid_img)
            sac_grp.label = "HFW Animated Elemental Sac (Flow Map)"
            sac_grp.location = (40, 480)
            gc = em_overlay.get("glow_color", [0.14, 0.68, 1.0])
            sac_grp.inputs["Sac Glow Color"].default_value = (gc[0], gc[1], gc[2], 1.0)
            sac_grp.inputs["Glow Strength"].default_value = float(em_overlay.get("glow_strength", 6.5))
            sac_grp.inputs["Flow Speed"].default_value = float(em_overlay.get("flow_speed", 0.35))
            sac_grp.inputs["Flow Strength"].default_value = float(em_overlay.get("flow_strength", 0.14))
            if color_sock is not None:
                nt.links.new(color_sock, sac_grp.inputs["Base Color"])
                color_sock = sac_grp.outputs["Color"]
            if smooth_sock is not None:
                nt.links.new(smooth_sock, sac_grp.inputs["Base Smoothness"])
                smooth_sock = sac_grp.outputs["Smoothness"]
            nt.links.new(sockets["Mask"], sac_grp.inputs["Mask"])
            if "Mask_Alpha" in sockets:
                nt.links.new(sockets["Mask_Alpha"], sac_grp.inputs["Mask Alpha"])
            nt.links.new(t_drv.outputs[0], sac_grp.inputs["Time"])
            overlay_em_color = sac_grp.outputs["Emission Color"]
            overlay_em_strength = sac_grp.outputs["Emission Strength"]

        elif okind == "canister_emissive_overlay" and "Mask" in sockets:
            noise_img = _load_image(tex_dir, em_overlay.get("noise_file", ""), "Non-Color", img_cache)
            t_drv = _add_time_driver_node(nt, (-160, 520), "HFW Canister Pulse Time")
            can_grp = nt.nodes.new("ShaderNodeGroup")
            can_grp.node_tree = _ensure_canister_glow_group(noise_img)
            can_grp.label = "HFW Animated Elemental Canister Glow"
            can_grp.location = (40, 480)
            gc = em_overlay.get("glow_color", [0.52, 0.70, 0.10])
            can_grp.inputs["Glow Color"].default_value = (gc[0], gc[1], gc[2], 1.0)
            can_grp.inputs["Glow Strength"].default_value = float(em_overlay.get("glow_strength", 5.5))
            can_grp.inputs["Pulse Speed"].default_value = float(em_overlay.get("pulse_speed", 8.0))
            can_grp.inputs["Is Fluid Body"].default_value = 1.0 if em_overlay.get("is_fluid_body") else 0.0
            if color_sock is not None:
                nt.links.new(color_sock, can_grp.inputs["Base Color"])
                color_sock = can_grp.outputs["Color"]
            nt.links.new(sockets["Mask"], can_grp.inputs["Mask"])
            nt.links.new(t_drv.outputs[0], can_grp.inputs["Time"])
            overlay_em_color = can_grp.outputs["Emission Color"]
            overlay_em_strength = can_grp.outputs["Emission Strength"]

        elif okind == "machine_eye_lens_overlay":
            iris_img = _load_image(tex_dir, em_overlay.get("iris_file", ""), "Non-Color", img_cache)
            t_drv = _add_time_driver_node(nt, (-160, 520), "HFW Eye Iris Time")
            eye_grp = nt.nodes.new("ShaderNodeGroup")
            eye_grp.node_tree = _ensure_eye_lens_group(iris_img)
            eye_grp.label = "HFW Machine Eye Iris Optics"
            eye_grp.location = (40, 480)
            cc = em_overlay.get("calm_color", [0.04, 0.56, 1.0])
            ac = em_overlay.get("alert_color", [1.0, 0.10, 0.02])
            eye_grp.inputs["Calm Color"].default_value = (cc[0], cc[1], cc[2], 1.0)
            eye_grp.inputs["Alert Color"].default_value = (ac[0], ac[1], ac[2], 1.0)
            eye_grp.inputs["Alert State"].default_value = float(em_overlay.get("alert_state", 0.0))
            eye_grp.inputs["Glow Strength"].default_value = float(em_overlay.get("glow_strength", 9.0))
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
            pc_grp = nt.nodes.new("ShaderNodeGroup")
            pc_grp.node_tree = _ensure_plasma_charge_group(noise_img)
            pc_grp.label = "HFW Slaughterspine Plasma Charge-Up (DXIL)"
            pc_grp.location = (40, 480)
            pc = em_overlay.get("plasma_color", [0.46, 0.05, 1.0])
            ic = em_overlay.get("idle_led_color", [0.85, 0.18, 0.02])
            pc_grp.inputs["Plasma Color"].default_value = (pc[0], pc[1], pc[2], 1.0)
            pc_grp.inputs["Idle Conduit Color"].default_value = (ic[0], ic[1], ic[2], 1.0)
            pc_grp.inputs["Glow Strength"].default_value = float(em_overlay.get("glow_strength", 4.2))
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

    # 3. Exact DXIL-verified Decima PBR Conversion (`HFW_DecimaPBR`):
    #    - Inverts Decima Smoothness -> true PBR Roughness (fixes plastic shine on matte paint + matte edges on bare metal!)
    #    - Converts Decima Specular/Reflectance (CLR + RFL^2 + normalized chroma) -> Blender Metallic + Base Color
    #      so worn bare metal edges (CLR=black, RFL=0.84) render as gleaming reflective silver/steel instead of black lines!
    if color_sock is not None or smooth_sock is not None or rfl_sock is not None:
        pbr_grp = nt.nodes.new("ShaderNodeGroup")
        pbr_grp.node_tree = _ensure_decima_pbr_node_group()
        pbr_grp.label = "HFW Decima PBR (Smoothness & F0 -> Metallic)"
        pbr_grp.location = (160, 220)
        if color_sock is not None:
            nt.links.new(color_sock, pbr_grp.inputs["Color"])
        elif "eye" in mname.lower():
            # Outer eye glass lens (`5704:0`) has no Color texture — dark glossy camera glass!
            pbr_grp.inputs["Color"].default_value = (0.015, 0.018, 0.024, 1.0)
        if smooth_sock is not None:
            nt.links.new(smooth_sock, pbr_grp.inputs["Smoothness"])
        if rfl_sock is not None:
            nt.links.new(rfl_sock, pbr_grp.inputs["Reflectance"])
        if ao_sock is not None:
            nt.links.new(ao_sock, pbr_grp.inputs["AO"])

        nt.links.new(pbr_grp.outputs["Base Color"], bsdf.inputs["Base Color"])
        nt.links.new(pbr_grp.outputs["Metallic"], bsdf.inputs["Metallic"])
        nt.links.new(pbr_grp.outputs["Roughness"], bsdf.inputs["Roughness"])
        if "Specular IOR Level" in bsdf.inputs:
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

    # 6. Wire Alpha cutout (only explicit Alpha, never Mask_Alpha)
    if "Alpha" in sockets:
        nt.links.new(sockets["Alpha"], bsdf.inputs["Alpha"])

    return mat


def _build_armature(
    name: str,
    joints: list[dict[str, Any]],
    helpers: list[dict[str, Any]],
    root_coll: bpy.types.Collection,
) -> tuple[bpy.types.Object, list[str], dict[str, str]]:
    """Build a unified Blender Armature with Skeleton joints + Helper bones in a dedicated Bone Collection."""
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

    main_bcoll = arm_data.collections.new("Skeleton")
    helper_bcoll = arm_data.collections.new("Helpers")
    helper_bcoll.is_visible = False

    for j in joints:
        idx = j["index"]
        wmat = joint_world_mats[idx]
        head = wmat.to_translation()
        eb = ebones.new(j["name"])
        eb.head = head

        kids = children_of.get(idx, [])
        bone_len = 0.15
        if kids:
            dists = [(joint_world_mats[k].to_translation() - head).length for k in kids]
            valid = [d for d in dists if d > 0.01]
            if valid:
                bone_len = max(0.05, min(0.5, sum(valid) / len(valid)))

        y_axis = wmat.to_3x3() @ Vector((0.0, 1.0, 0.0))
        if y_axis.length < 1e-6:
            y_axis = Vector((0.0, 1.0, 0.0))
        else:
            y_axis.normalize()
        eb.tail = head + y_axis * bone_len
        eb.matrix = Matrix.Translation(head) @ wmat.to_3x3().normalized().to_4x4()
        eb.length = bone_len

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
) -> bpy.types.Object | None:
    """Create a Blender mesh object combining the part's primitives, with skinning OR exact bone parenting."""
    prims = part_meta["primitives"]
    if not prims:
        return None

    all_pos = []
    all_nrm = []
    all_col0 = []
    all_faces = []
    face_mat_indices = []
    all_uvs: list[list[np.ndarray]] = []
    all_bi = []
    all_bw = []

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
                all_uvs[u_i].append(uv)
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

    pname = part_meta["name"]
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

    obj = bpy.data.objects.new(pname, mesh)
    obj["hfw_entity_id"] = entity_id
    obj["hfw_part_id"] = part_meta["part_id"]
    obj["hfw_source_kind"] = part_meta["source_kind"]
    target_coll.objects.link(obj)

    if part_meta["is_skinned"]:
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

        if target_bone_name is not None:
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


def import_bundle_into_scene(
    bundle_dir: str | Path,
    enable_detail_nodes: bool = True,
    pack_textures: bool = False,
) -> bpy.types.Object:
    """Import a Canonical Assembly Bundle directly into the active Blender scene."""
    bundle_path = Path(bundle_dir)
    manifest = json.loads((bundle_path / "assembly.json").read_text(encoding="utf-8"))
    npz = np.load(bundle_path / "geometry.npz")
    tex_dir = bundle_path / "textures"

    disp = manifest["display_name"]
    variant = manifest.get("variant_label", "")
    root_name = f"HFW_{disp}"
    root_coll = _get_or_create_collection(root_name, bpy.context.scene.collection)

    sub_colls = {
        "Body": _get_or_create_collection(f"{disp}_Body", root_coll),
        "Plates_Intact": _get_or_create_collection(f"{disp}_Plates_Intact", root_coll),
        "Weapons_And_Components": _get_or_create_collection(f"{disp}_Weapons_And_Components", root_coll),
        "Plates_Broken_States": _get_or_create_collection(f"{disp}_Plates_Broken_States", root_coll),
        "Focus_Overlay": _get_or_create_collection(f"{disp}_Focus_Overlay", root_coll),
        "Unattached_Misc": _get_or_create_collection(f"{disp}_Unattached_Misc", root_coll),
    }

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

    # If this is a Plasma-charging machine (Slaughterspine!), expose a single live `hfw_plasma_charge` slider (0..1)
    # on the Armature driving `HFW_PlasmaChargeControl` across every PlasmaSpike, PlasmaCharger, PlasmaCore, Body & Muscle!
    has_plasma_charge = any(
        (mmeta.get("emissive_overlay") or {}).get("kind") == "plasma_charge_overlay"
        for mmeta in manifest["materials"].values()
    )
    if has_plasma_charge:
        arm_obj["hfw_plasma_charge"] = 1.0
        try:
            ui_m = arm_obj.id_properties_ui("hfw_plasma_charge")
            ui_m.update(
                min=0.0,
                max=1.0,
                soft_min=0.0,
                soft_max=1.0,
                description="Slaughterspine Plasma Charge-Up (0.0 = Uncharged Idle, 1.0 = Full Electric Purple Plasma Charge)",
            )
        except Exception:
            pass
        ctrl_ng = bpy.data.node_groups.get("HFW_PlasmaChargeControl")
        if ctrl_ng is not None:
            for n in ctrl_ng.nodes:
                if n.type == "VALUE" and n.label == "Plasma Charge":
                    try:
                        fc = n.outputs[0].driver_add("default_value")
                        drv = fc.driver
                        drv.type = "AVERAGE"
                        var = drv.variables.new()
                        var.name = "chg"
                        var.type = "SINGLE_PROP"
                        var.targets[0].id_type = "OBJECT"
                        var.targets[0].id = arm_obj
                        var.targets[0].data_path = '["hfw_plasma_charge"]'
                    except Exception:
                        pass

    created = 0
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
        )
        if obj is not None:
            created += 1

    def _hide_layer_collection(layer_coll: bpy.types.LayerCollection, target_name: str) -> None:
        if layer_coll.collection.name == target_name:
            layer_coll.hide_viewport = True
            return
        for child in layer_coll.children:
            _hide_layer_collection(child, target_name)

    for hidden_key in ("Plates_Broken_States", "Focus_Overlay", "Unattached_Misc"):
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
    print(
        f"[HFWToolkit] Imported '{disp}' ({variant}, textures={manifest.get('texture_mode', 'packed')}): "
        f"{len(joint_names)} bones, {len(helper_bone_map)} helpers, {created} parts, {len(mat_map)} materials."
    )
    return arm_obj


def _cli_main() -> None:
    if "--" not in sys.argv:
        print("Usage: blender --background --python blender_builder.py -- <bundle_dir> <out.blend> [--pack] [--no-detail] [--export-fbx <out.fbx>]")
        sys.exit(2)
    args = sys.argv[sys.argv.index("--") + 1 :]
    bundle_dir = Path(args[0])
    out_blend = Path(args[1])
    pack = "--pack" in args
    enable_detail = "--no-detail" not in args
    export_fbx: Path | None = None
    if "--export-fbx" in args:
        idx = args.index("--export-fbx")
        if idx + 1 < len(args):
            export_fbx = Path(args[idx + 1])

    bpy.ops.wm.read_factory_settings(use_empty=True)
    import_bundle_into_scene(bundle_dir, enable_detail_nodes=enable_detail, pack_textures=pack)

    out_blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out_blend.resolve()))
    print(f"[HFWToolkit] Saved Blender file: {out_blend}")

    if export_fbx is not None:
        export_fbx.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.object.select_all(action="DESELECT")
        for obj in bpy.context.scene.objects:
            if not obj.hide_viewport and not obj.hide_render:
                obj.select_set(True)
        try:
            fbx_kwargs: dict[str, Any] = dict(
                filepath=str(export_fbx.resolve()),
                use_selection=True,
                object_types={"ARMATURE", "MESH"},
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


if __name__ == "__main__":
    _cli_main()
