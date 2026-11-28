"""Symbolic DXIL parser for Horus / FAS-BOR7 Titan 26-TB Layered Hull Shaders.

All 259 Horus hull / leg / tail / tentacle / drill / interactive-plate / climbing-spot
materials share Guerrilla's 26-TextureBinding Titan Layered Shader architecture:
  - 3 unique 4K macro maps per mesh tile:
      `0x5C9D17BB` (`Titan_*_NMT`):  R=Normal.X, G=Normal.Y, B=Baked AO, A=Curvature
      `0x58794D54` (`Titan_*_MSK`):  R,G,B 3-bit 8-corner layer mask
      `0x70FA2EF0` (`Titan_*_MISC`): R=Clean/Damaged blend, G=Rust, B=Grunge/Moss
  - 6 shared 16-slice Group-132 `Texture2DArray`s (Clean & Damaged PBR sets):
      Clean:   `132:2` Albedo, `132:3` NMT/Smoothness/Height, `132:8` RFL/AO/PaintMask
      Damaged: `132:9` Albedo, `132:6` NMT/Smoothness/Height, `132:7` RFL/AO/PaintMask
  - Shared LUTs & weathering overlays:
      `132:5` (32x1 Titan Paint Palette LUT, linear-header sRGB-stored -> `pow(rgb, 2.2)` in PS!)
      `64:2`  (16x1 Metallic Reflectance Tint LUT)
      `64:0/1/3` (`Rust_CLR` / `Rust_NMT` / `Rust_HGT` / `Rust_LUT`)
      `132:0/1` (`Grunge_NMT` / `Grunge_CLR`)
      `1:0/1/2` (`Moss_LUT` / `StarMoss_clr` / `StarMoss_nmt`, upward-facing world normal only!)
  - Per-shader compiled DXIL 8-corner `switch i32` table assigning each `MSK.RGB` corner:
      0=Black(0,0,0), 1=Red(1,0,0), 2=Green(0,1,0), 3=Blue(0,0,1),
      4=Cyan(0,1,1),  5=Magenta(1,0,1), 6=Yellow(1,1,0), 7=White(1,1,1)
    its exact `(slice_idx, color_lut_idx, tile_scale, rust_mult, rfl_lut_idx)`.
"""

from __future__ import annotations

import re
import struct
from typing import Any

import numpy as np

from hfw_toolkit.decima.reader import ObjectId
from hfw_toolkit.decima.streaming_graph import HFWGame
from hfw_toolkit.extract.materials import MaterialInfo
from hfw_toolkit.extract.textures import decode_texture
from hfw_toolkit.shaders.character_dxil import CORNER_RGB_LABELS
from hfw_toolkit.shaders.dxc import DXCDisassembler
from hfw_toolkit.shaders.hw_shader import parse_hw_shader

# Exact Decima BindingNameHashes verified from `ps_1256_1_p1.ll` (`StaticPerBatch_Constant` %8)
HASH_TITAN_MACRO_NMT = 0x5C9D17BB      # TB[2]: Titan_*_NMT (R=NrmX, G=NrmY, B=AO, A=Curvature)
HASH_TITAN_MACRO_MSK = 0x58794D54      # TB[3]: Titan_*_MSK (R,G,B 3-bit 8-corner mask)
HASH_TITAN_MACRO_MISC = 0x70FA2EF0     # TB[6]: Titan_*_MISC (R=Damage, G=Rust, B=Grunge/Moss)

HASH_TITAN_CLEAN_NMT_ARR = 0x17D0FCFB  # TB[8]  132:3 (16-slice Clean NMT.RG + Smoothness.B + Height.A)
HASH_TITAN_DMG_NMT_ARR = 0x711C04E8    # TB[9]  132:6 (16-slice Damaged NMT.RG + Smoothness.B + Height.A)
HASH_TITAN_DMG_RFL_ARR = 0x5729C871    # TB[10] 132:7 (16-slice Damaged RFL.R + AO.G + PaintMask.B)
HASH_TITAN_CLEAN_RFL_ARR = 0x294FC783  # TB[11] 132:8 (16-slice Clean RFL.R + AO.G + PaintMask.B)
HASH_TITAN_CLEAN_CLR_ARR = 0x3DDAEDCF  # TB[17] 132:2 (16-slice Clean Albedo)
HASH_TITAN_DMG_CLR_ARR = 0x3C2049DF    # TB[18] 132:9 (16-slice Damaged Albedo)

HASH_TITAN_PAINT_LUT = 0x0DA47D50      # TB[16] 132:5 (32x1 Paint LUT, sRGB values stored in Linear header)
HASH_TITAN_RFL_LUT = 0x2FBCCA7D        # TB[7]  64:2  (16x1 Metallic Reflectance Tint LUT)
HASH_TITAN_RUST_HGT = 0x79D3CB91       # TB[5]  64:0  Rust_HGT
HASH_TITAN_RUST_NMT = 0x62DF89B5       # TB[13] 64:1  Rust_NMT
HASH_TITAN_RUST_LUT = 0x4F51D9E2       # TB[14] 64:3  Rust Color Palette LUT
HASH_TITAN_RUST_CLR = 0x005F5F67       # TB[15] 64:0  Rust_CLR
HASH_TITAN_GRUNGE_NMT = 0x115A96DC     # TB[20] 132:0 Grunge_NMT
HASH_TITAN_GRUNGE_CLR = 0x5B9E180E     # TB[24] 132:1 Grunge_CLR
HASH_TITAN_MOSS_HGT = 0x604A7E18       # TB[0]  1:1   StarMoss_hgt
HASH_TITAN_MOSS_NMT = 0x63A32771       # TB[21] 1:2   StarMoss_nmt
HASH_TITAN_MOSS_LUT = 0x60E9A35A       # TB[22] 1:0   Moss Color Palette LUT
HASH_TITAN_MOSS_CLR = 0x6DA2780E       # TB[23] 1:1   StarMoss_clr

# Verified canonical fallback LUTs sampled directly from HFW `132:5`, `64:2`, `64:3`, and `1:0`
_FALLBACK_PAINT_LUT_132_5: tuple[tuple[float, float, float], ...] = (
    (0.502, 0.502, 0.502),  # 0: Neutral (2x = 1.0, untinted dark metal)
    (0.200, 0.208, 0.133),  # 1: Faro dark military olive-charcoal armor
    (0.698, 0.686, 0.639),  # 2: Weathered light titanium/metal accent
    (0.337, 0.384, 0.396),  # 3: Cool slate-blue weathered steel
    (0.173, 0.184, 0.192),  # 4: Iconic Faro deep gunmetal charcoal-black hull
    (0.698, 0.690, 0.682),  # 5: Light grey climbing grip / brushed steel
    (0.310, 0.416, 0.294),  # 6: Muted oxidized green-grey
    (0.196, 0.522, 0.416),  # 7: Patina teal
    (0.451, 0.427, 0.224),  # 8: Weathered brass/olive-gold accent
    (0.565, 0.588, 0.600),  # 9: Steel grey
    (0.588, 0.427, 0.282),  # 10: Weathered bronze
    (0.098, 0.224, 0.396),  # 11: Deep industrial blue
    (0.220, 0.263, 0.314),  # 12: Dark steel blue-grey
    (0.149, 0.184, 0.220),  # 13: Midnight gunmetal
    (0.294, 0.341, 0.400),  # 14: Industrial slate
    (0.337, 0.325, 0.310),  # 15: Warm medium gunmetal
    (0.443, 0.373, 0.255),  # 16: Khaki metal
    (0.212, 0.235, 0.071),  # 17: Olive drab
    (0.698, 0.486, 0.000),  # 18: Industrial hazard yellow
    (0.231, 0.235, 0.243),  # 19: Charcoal steel
    (0.325, 0.329, 0.341),  # 20: Medium charcoal
    (0.620, 0.588, 0.588),  # 21: Light ash steel
    (0.227, 0.286, 0.286),  # 22: Dark teal-slate
    (0.698, 0.667, 0.400),  # 23: Warm pale gold
    (0.682, 0.698, 0.682),  # 24: Pale silver
    (0.447, 0.565, 0.698),  # 25: Steel sky blue
    (0.310, 0.357, 0.278),  # 26: Olive slate
    (0.141, 0.518, 0.698),  # 27: Industrial cyan
    (0.533, 0.557, 0.518),  # 28: Light sage steel
    (0.557, 0.651, 0.698),  # 29: Pale steel blue
    (0.216, 0.251, 0.314),  # 30: Navy charcoal
    (0.161, 0.200, 0.275),  # 31: Deep navy gunmetal
)

_FALLBACK_RFL_LUT_64_2: tuple[tuple[float, float, float], ...] = (
    (0.502, 0.502, 0.502),  # 0: Neutral (2x = 1.0)
    (0.737, 0.404, 0.204),  # 1: Warm copper-bronze
    (0.502, 0.502, 0.502),
    (0.502, 0.502, 0.502),
    (0.502, 0.502, 0.502),
    (0.502, 0.502, 0.502),
    (0.502, 0.502, 0.502),
    (0.518, 0.498, 0.475),
    (0.416, 0.541, 0.604),
    (0.039, 0.039, 0.039),
    (0.227, 0.247, 0.282),
    (0.518, 0.506, 0.427),
    (0.788, 0.314, 0.000),
    (0.620, 0.412, 0.345),
    (0.439, 0.533, 0.580),
    (0.271, 0.306, 0.376),
)

_FALLBACK_RUST_LUT_SRGB: tuple[float, float, float] = (0.435, 0.248, 0.172)
_FALLBACK_MOSS_LUT_SRGB: tuple[float, float, float] = (0.255, 0.355, 0.078)

# Canonical fallback corner table matching primary Horus hull shader `1256:1`
_FALLBACK_CORNER_TABLE: dict[str, list[Any]] = {
    "slice_idx": [11, 12, 7, 3, 13, 12, 0, 0],
    "color_lut_idx": [4, 1, 2, 0, 4, 1, 0, 0],
    "tile_scale": [16.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0],
    "rust_mult": [1.0, 1.0, 1.0, 0.3, 1.0, 1.0, 1.0, 1.0],
    "rfl_lut_idx": [0, 0, 0, 0, 0, 0, 0, 0],
}


def is_titan_layered_material(mat: MaterialInfo) -> bool:
    """Return True if `mat` uses Horus / Titan's 26-TB Group-132 layered hull shader."""
    pt = mat.primary_technique
    if pt is None or not pt.texture_bindings:
        return False
    hashes = {tb.binding_name_hash for tb in pt.texture_bindings if tb.resource_id}
    if (
        HASH_TITAN_MACRO_NMT in hashes
        and HASH_TITAN_MACRO_MSK in hashes
        and HASH_TITAN_MACRO_MISC in hashes
        and (
            HASH_TITAN_CLEAN_CLR_ARR in hashes
            or any(tb.resource_id == "132:2" for tb in pt.texture_bindings)
        )
    ):
        return True
    return False


def _parse_dxil_float(tok: str) -> float:
    tok = tok.strip().rstrip(",")
    if tok.startswith("0x") and len(tok) == 18:
        try:
            return float(struct.unpack(">d", bytes.fromhex(tok[2:]))[0])
        except Exception:
            return 0.0
    try:
        return float(tok)
    except Exception:
        return 0.0


def _load_titan_luts(game: HFWGame, lut_cache: dict[str, Any]) -> dict[str, Any]:
    """Decode & cache Horus's shared Paint (`132:5`), Reflectance (`64:2`), Rust (`64:3`), and Moss (`1:0`) LUTs."""
    if "paint_lut" in lut_cache:
        return lut_cache

    paint_lut = [list(c) for c in _FALLBACK_PAINT_LUT_132_5]
    rfl_lut = [list(c) for c in _FALLBACK_RFL_LUT_64_2]
    rust_lut_srgb = list(_FALLBACK_RUST_LUT_SRGB)
    moss_lut_srgb = list(_FALLBACK_MOSS_LUT_SRGB)

    try:
        dec = decode_texture(game, game.read_object(ObjectId.parse("132:5")))
        if dec is not None and dec.rgba.shape[1] >= 32:
            arr = dec.rgba[0, :32, :3].astype(np.float32) / 255.0
            paint_lut = [[round(float(c), 4) for c in arr[i]] for i in range(32)]
    except Exception:
        pass

    try:
        dec = decode_texture(game, game.read_object(ObjectId.parse("64:2")))
        if dec is not None and dec.rgba.shape[1] >= 16:
            arr = dec.rgba[0, :16, :3].astype(np.float32) / 255.0
            rfl_lut = [[round(float(c), 4) for c in arr[i]] for i in range(16)]
    except Exception:
        pass

    try:
        dec = decode_texture(game, game.read_object(ObjectId.parse("64:3")))
        if dec is not None and dec.rgba.size > 0:
            arr = dec.rgba.astype(np.float32) / 255.0
            # Sample mid-row representative warm corroded iron-oxide rust
            mid = arr[0, arr.shape[1] // 2, :3]
            mean = arr[:, :, :3].mean(axis=(0, 1))
            blend = 0.55 * mid + 0.45 * mean
            rust_lut_srgb = [round(float(c), 4) for c in blend]
    except Exception:
        pass

    try:
        dec = decode_texture(game, game.read_object(ObjectId.parse("1:0")))
        if dec is not None and dec.rgba.size > 0:
            arr = dec.rgba.astype(np.float32) / 255.0
            mid = arr[0, arr.shape[1] // 2, :3]
            moss_lut_srgb = [round(float(c), 4) for c in mid]
    except Exception:
        pass

    lut_cache["paint_lut"] = paint_lut
    lut_cache["rfl_lut"] = rfl_lut
    lut_cache["rust_lut_srgb"] = rust_lut_srgb
    lut_cache["moss_lut_srgb"] = moss_lut_srgb
    return lut_cache


def _parse_titan_ps_dxil(dis: str, tbs: list[Any]) -> dict[str, list[Any]] | None:
    """Parse exact 8-corner `(slice_idx, color_lut_idx, tile_scale, rust_mult, rfl_lut_idx)` from a Horus PixelProgram."""
    lines = dis.splitlines()

    # Map TB indices dynamically by binding_name_hash so even shifted TB layouts resolve cleanly
    tb_clean_clr = next((tb.index for tb in tbs if tb.binding_name_hash == HASH_TITAN_CLEAN_CLR_ARR or tb.resource_id == "132:2"), 17)
    tb_paint_lut = next((tb.index for tb in tbs if tb.binding_name_hash == HASH_TITAN_PAINT_LUT or tb.resource_id == "132:5"), 16)
    tb_rfl_lut = next((tb.index for tb in tbs if tb.binding_name_hash == HASH_TITAN_RFL_LUT or tb.resource_id == "64:2"), 7)

    # 1. Track %8 StaticPerBatch_Constant -> TB index -> texture sample coordinates
    cb_reg: dict[str, int] = {}
    handle_to_tb: dict[str, int] = {}
    tb_samples: dict[int, tuple[str, str, str]] = {}
    def_line: dict[str, str] = {}

    for ln in lines:
        md = re.match(r"\s*(%\d+)\s*=\s*(.+)", ln)
        if md:
            def_line[md.group(1)] = md.group(2).strip()

        m = re.search(r"(%\d+) = call %dx\.types\.CBufRet\.i32 @dx\.op\.cbufferLoadLegacy\.i32\(i32 59, %dx\.types\.Handle %8, i32 (\d+)\)", ln)
        if m:
            cb_reg[m.group(1)] = int(m.group(2))
            continue
        m2 = re.search(r"(%\d+) = extractvalue %dx\.types\.CBufRet\.i32 (%\d+), (\d+)", ln)
        if m2 and m2.group(2) in cb_reg:
            handle_to_tb[m2.group(1)] = cb_reg[m2.group(2)] * 4 + int(m2.group(3))
            continue
        m3 = re.search(r"(%\d+) = call %dx\.types\.Handle @dx\.op\.createHandleFromHeap\(i32 218, i32 (%\d+),", ln)
        if m3 and m3.group(2) in handle_to_tb:
            handle_to_tb[m3.group(1)] = handle_to_tb[m3.group(2)]
            continue
        m4 = re.search(r"(%\d+) = call %dx\.types\.Handle @dx\.op\.annotateHandle\(i32 216, %dx\.types\.Handle (%\d+),", ln)
        if m4 and m4.group(2) in handle_to_tb:
            handle_to_tb[m4.group(1)] = handle_to_tb[m4.group(2)]
            continue
        m5 = re.search(
            r"dx\.op\.sample\.f32\(i32 60, %dx\.types\.Handle (%\d+), %dx\.types\.Handle %\d+, float ([^,]+), float ([^,]+), float ([^,]+),",
            ln,
        )
        if m5 and m5.group(1) in handle_to_tb:
            tb_i = handle_to_tb[m5.group(1)]
            tb_samples.setdefault(tb_i, (m5.group(2).strip(), m5.group(3).strip(), m5.group(4).strip()))

    # 2. Locate the 8-case switch on MSK.RGB
    sw_idx = -1
    case_labels: dict[int, str] = {}
    merge_lbl = ""
    sw_val_reg = ""
    for i, ln in enumerate(lines):
        m = re.match(r"\s*switch i32 (%\d+), label (%\d+) \[", ln)
        if m:
            sw_val_reg = m.group(1)
            merge_lbl = m.group(2)
            j = i + 1
            cases: dict[int, str] = {}
            while j < len(lines) and "]" not in lines[j]:
                mc = re.match(r"\s*i32 (\d+), label (%\d+)", lines[j])
                if mc:
                    cases[int(mc.group(1))] = mc.group(2)
                j += 1
            if len(cases) == 8:
                sw_idx = i
                case_labels = cases
                break

    if sw_idx < 0:
        return None

    m_lbl_num = merge_lbl.lstrip("%")
    m_idx = -1
    for i in range(sw_idx, len(lines)):
        if lines[i].startswith(f"; <label>:{m_lbl_num}"):
            m_idx = i
            break
    if m_idx < 0:
        return None

    phis: dict[str, tuple[str, list[Any]]] = {}
    for i in range(m_idx + 1, min(len(lines), m_idx + 20)):
        ln = lines[i].strip()
        mp = re.match(r"(%\d+)\s*=\s*phi\s+(float|i32)\s+(.*)", ln)
        if not mp:
            break
        reg, typ, body = mp.group(1), mp.group(2), mp.group(3)
        lbl_to_val = {lbl: val.strip() for val, lbl in re.findall(r"\[\s*([^,\]]+?)\s*,\s*(%\d+)\s*\]", body)}
        vals: list[Any] = []
        for c in range(8):
            raw_v = lbl_to_val.get(case_labels[c], "0")
            if raw_v == sw_val_reg:
                val: Any = c
            elif typ == "i32":
                val = int(raw_v) if raw_v.lstrip("-").isdigit() else 0
            else:
                val = round(_parse_dxil_float(raw_v), 4)
            vals.append(val)
        phis[reg] = (typ, vals)

    if not phis:
        return None

    def _trace_to_phi(expr: str, depth: int = 0) -> str | None:
        expr = expr.strip()
        if expr in phis:
            return expr
        if not expr.startswith("%") or depth > 6:
            return None
        dl = def_line.get(expr, "")
        if not dl:
            return None
        for r in re.findall(r"%\d+", dl):
            if r in phis:
                return r
        for r in re.findall(r"%\d+", dl):
            hit = _trace_to_phi(r, depth + 1)
            if hit:
                return hit
        return None

    def _extract_const_scale(expr: str) -> float | None:
        dl = def_line.get(expr.strip(), "")
        m_mul = re.search(r"fmul\s+(?:fast\s+)?float\s+%\d+,\s*([^\s,]+)", dl)
        if m_mul and not m_mul.group(1).startswith("%"):
            v = _parse_dxil_float(m_mul.group(1))
            if 1.0 <= v <= 128.0:
                return round(v, 4)
        return None

    used_phis: set[str] = set()

    # Slice index (w coord of Clean CLR array `132:2`)
    slice_vals = list(_FALLBACK_CORNER_TABLE["slice_idx"])
    tile_vals = [32.0] * 8
    if tb_clean_clr in tb_samples:
        u_expr, _v_expr, w_expr = tb_samples[tb_clean_clr]
        p_w = _trace_to_phi(w_expr)
        if p_w and p_w in phis:
            slice_vals = [max(0, min(15, int(v))) for v in phis[p_w][1]]
            used_phis.add(p_w)
        elif not w_expr.startswith("%"):
            sv = max(0, min(15, int(round(_parse_dxil_float(w_expr)))))
            slice_vals = [sv] * 8

        p_u = _trace_to_phi(u_expr)
        if p_u and p_u in phis:
            tile_vals = [max(1.0, min(128.0, float(v))) for v in phis[p_u][1]]
            used_phis.add(p_u)
        else:
            cs = _extract_const_scale(u_expr)
            if cs is not None:
                tile_vals = [cs] * 8

    # Paint LUT index (u coord of `132:5` 32x1 LUT)
    color_lut_vals = [0] * 8
    if tb_paint_lut in tb_samples:
        u_expr = tb_samples[tb_paint_lut][0]
        p_c = _trace_to_phi(u_expr)
        if p_c and p_c in phis:
            color_lut_vals = [max(0, min(31, int(v))) for v in phis[p_c][1]]
            used_phis.add(p_c)
        elif not u_expr.startswith("%"):
            ci = max(0, min(31, int(round(_parse_dxil_float(u_expr) * 32.0 - 0.5))))
            color_lut_vals = [ci] * 8

    # Metallic Reflectance Tint LUT index (u coord of `64:2` 16x1 LUT)
    rfl_lut_vals = [0] * 8
    if tb_rfl_lut in tb_samples:
        u_expr = tb_samples[tb_rfl_lut][0]
        p_r = _trace_to_phi(u_expr)
        if p_r and p_r in phis:
            rfl_lut_vals = [max(0, min(15, int(v))) for v in phis[p_r][1]]
            used_phis.add(p_r)
        elif not u_expr.startswith("%"):
            ri = max(0, min(15, int(round(_parse_dxil_float(u_expr) * 16.0 - 0.5))))
            rfl_lut_vals = [ri] * 8

    # Rust multiplier (remaining float phi in [0..2], or constant 1.0 when folded out)
    rust_vals = [1.0] * 8
    for reg, (typ, vals) in phis.items():
        if reg not in used_phis and typ == "float":
            if all(0.0 <= float(v) <= 2.5 for v in vals):
                rust_vals = [round(float(v), 4) for v in vals]
                used_phis.add(reg)
                break

    return {
        "slice_idx": slice_vals,
        "color_lut_idx": color_lut_vals,
        "tile_scale": tile_vals,
        "rust_mult": rust_vals,
        "rfl_lut_idx": rfl_lut_vals,
    }


def extract_titan_hull_spec(
    game: HFWGame,
    mat: MaterialInfo,
    dxc: DXCDisassembler | None = None,
    shader_cache: dict[str, dict[str, list[Any]] | None] | None = None,
    lut_cache: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Extract the complete DXIL-verified 8-corner specification for a Horus/Titan 26-TB layered material."""
    if not is_titan_layered_material(mat):
        return None
    pt = mat.primary_technique
    if pt is None:
        return None

    lcache = lut_cache if lut_cache is not None else {}
    luts = _load_titan_luts(game, lcache)
    paint_lut = luts["paint_lut"]
    rfl_lut = luts["rfl_lut"]

    sid = pt.shader_id or ""
    parsed: dict[str, list[Any]] | None = None
    if sid and shader_cache is not None and sid in shader_cache:
        parsed = shader_cache[sid]
    elif sid:
        try:
            disasm_tool = dxc or DXCDisassembler(game.config)
            sobj = game.read_object(ObjectId.parse(sid), read_subgroups=True)
            hw = parse_hw_shader(sobj)
            ps = hw.get_program("PixelProgram")
            if ps is not None:
                dis = disasm_tool.disassemble_blob(ps.blob)
                parsed = _parse_titan_ps_dxil(dis, pt.texture_bindings)
        except Exception:
            parsed = None
        if shader_cache is not None:
            shader_cache[sid] = parsed

    tbl = parsed or _FALLBACK_CORNER_TABLE
    corners: list[dict[str, Any]] = []
    for c in range(8):
        s_idx = int(tbl["slice_idx"][c])
        c_idx = int(tbl["color_lut_idx"][c])
        r_idx = int(tbl["rfl_lut_idx"][c])
        p_srgb = list(paint_lut[c_idx]) if 0 <= c_idx < len(paint_lut) else [0.502, 0.502, 0.502]
        r_rgb = list(rfl_lut[r_idx]) if 0 <= r_idx < len(rfl_lut) else [0.502, 0.502, 0.502]
        corners.append(
            {
                "corner": c,
                "corner_name": CORNER_RGB_LABELS[c],
                "slice_idx": s_idx,
                "tile_scale": float(tbl["tile_scale"][c]),
                "color_lut_idx": c_idx,
                "paint_lut_srgb": p_srgb,
                "rfl_lut_idx": r_idx,
                "rfl_lut_rgb": r_rgb,
                "rust_mult": float(tbl["rust_mult"][c]),
                "active": True,
            }
        )

    return {
        "shader_id": sid,
        "dxil_parsed": parsed is not None,
        "rust_lut_srgb": list(luts["rust_lut_srgb"]),
        "moss_lut_srgb": list(luts["moss_lut_srgb"]),
        "corners": corners,
    }
