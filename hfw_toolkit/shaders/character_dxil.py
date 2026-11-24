"""Symbolic DXIL evaluator for HFW Character Layered Outfit & Human Skin shaders.

Horizion Forbidden West renders nearly all character outfits (Aloy's 80+ armors,
Companions, Tribal NPCs) using an 8-layer `MSK.RGB` 3-bit corner shader where:
  - Each of the 8 RGB cube corners maps to an exact Layer ID (`0..7`):
      0 = Black   (0,0,0)    4 = Cyan    (0,1,1)
      1 = Red     (1,0,0)    5 = Magenta (1,0,1)
      2 = Green   (0,1,0)    6 = Yellow  (1,1,0)
      3 = Blue    (0,0,1)    7 = White   (1,1,1)
  - The tiling sub-material textures (`Shaders_HRZ2/Materials/<Cat>/<Mat>/*_CLR`)
    are stored as `0.5`-centered **Linear** micro-detail maps (`mean ≈ 0.502`), NOT
    pre-colored albedos!
  - Every outfit piece's exact authored per-layer dye color, metallic reflectance `F0`,
    smoothness, and layer-to-texture assignment are compiled directly into that
    material's DXIL `PixelProgram` (evaluated at `CustomDye = false`, `Wear = 0`).

Similarly, HFW human heads & bodies bind Guerrilla's 16x16 Skin Tone LUT (`570:0`)
and store `Head_CLR` / `Female_Aloy_CLR` as `0.5`-centered neutralized variation maps
combined via `pow(2.0 * SkinLUT_srgb * Head_CLR_srgb, 2.2)` = `4.5948 * SkinLinear * TexLinear`.
"""

from __future__ import annotations

import math
import re
import struct
from typing import Any

from hfw_toolkit.decima.reader import ObjectId
from hfw_toolkit.decima.streaming_graph import HFWGame
from hfw_toolkit.extract.materials import MaterialInfo
from hfw_toolkit.shaders.dxc import DXCDisassembler
from hfw_toolkit.shaders.hw_shader import parse_hw_shader

# Exact Guerrilla HFW DXIL 8-corner mapping: (R>0.5, G>0.5, B>0.5) -> Layer index 0..7
# Verified directly from `75185:637` L1809..1874 and `74:0` 8x1 palette LUT!
CORNER_RGB_LABELS: tuple[str, ...] = (
    "Black",    # 0: (0,0,0)
    "Red",      # 1: (1,0,0)
    "Green",    # 2: (0,1,0)
    "Blue",     # 3: (0,0,1)
    "Cyan",     # 4: (0,1,1)
    "Magenta",  # 5: (1,0,1)
    "Yellow",   # 6: (1,1,0)
    "White",    # 7: (1,1,1)
)

# Canonical sRGB skin tones sampled directly from Guerrilla's 16x16 Skin LUT (`570:0`, rows 0..7)
# Converted to linear RGB (`c^2.2`) so `SkinLinear * (Head_CLR_linear * 4.5948)` reproduces
# Guerrilla's exact `pow(2.0 * SkinLUT_srgb * Head_CLR_srgb, 2.2)` pixel-for-pixel!
_SKIN_LUT_PRESETS_SRGB: dict[str, tuple[float, float, float]] = {
    "fair_rosy":     (178 / 255.0, 135 / 255.0, 125 / 255.0),  # 570:0 [0,0] Aloy, Beta, Alva, Nora/Oseram fair
    "fair_warm":     (176 / 255.0, 133 / 255.0, 108 / 255.0),  # 570:0 [0,9] Warm fair / sun-kissed
    "light_golden":  (164 / 255.0, 118 / 255.0,  96 / 255.0),  # 570:0 [3,10] Seyka, Quen light-olive/golden
    "medium_tan":    (150 / 255.0, 104 / 255.0,  86 / 255.0),  # 570:0 [4,3] Kotallo, Tenakth/Carja weathered tan
    "medium_brown":  (124 / 255.0,  82 / 255.0,  66 / 255.0),  # 570:0 [5,4] Zo, Utaru warm brown
    "deep_brown":    ( 88 / 255.0,  54 / 255.0,  42 / 255.0),  # 570:0 [6,5] Varl, Sylens rich deep brown
    "dark_ebony":    ( 66 / 255.0,  38 / 255.0,  28 / 255.0),  # 570:0 [6,7] Deepest tone
}


def srgb_to_linear_rgb(srgb: tuple[float, float, float]) -> list[float]:
    return [round(max(0.0, min(1.0, float(c))) ** 2.2, 5) for c in srgb]


def infer_character_skin_tone(
    character_hint: str = "",
    variant_hint: str = "",
    texture_hint: str = "",
) -> dict[str, Any]:
    """Select canonical HFW `570:0` Skin LUT tone (in linear RGB + sRGB) for any character."""
    blob = f"{character_hint} {variant_hint} {texture_hint}".lower()

    if any(k in blob for k in ("aloy", "beta", "alva", "elisabet", "sobeck", "erend", "petra", "tilda", "gerard", "erik", "regalla_pale", "child_aloy")):
        preset = "fair_rosy"
    elif any(k in blob for k in ("seyka", "ceo", "bohai", "quen")):
        preset = "light_golden"
    elif any(k in blob for k in ("sylens", "varl", "hekkaro", "regalla", "dekka", "teb", "rost")):
        preset = "deep_brown" if any(k in blob for k in ("sylens", "varl", "hekkaro", "regalla")) else "medium_tan"
    elif any(k in blob for k in ("zo", "kue", "fane", "utaru")):
        preset = "medium_brown"
    elif any(k in blob for k in ("kotallo", "avad", "nil", "talanah", "tenakth", "carja")):
        preset = "medium_tan"
    else:
        preset = "fair_warm"

    srgb = _SKIN_LUT_PRESETS_SRGB[preset]
    return {
        "preset": preset,
        "skin_tone_srgb": [round(c, 4) for c in srgb],
        "skin_tone_linear": srgb_to_linear_rgb(srgb),
    }


# Canonical HFW Character Hair Colors (sampled from Guerrilla's `382:0` 256x64 Hair LUT)
_HAIR_PRESETS_SRGB: dict[str, tuple[float, float, float]] = {
    "aloy_auburn":    (148 / 255.0,  62 / 255.0,  32 / 255.0),  # Aloy, Beta, Elisabet fiery copper-auburn
    "espresso_black": ( 44 / 255.0,  36 / 255.0,  34 / 255.0),  # Seyka, Quen, Kotallo, Zo, Varl, Regalla, Sylens
    "dark_brown":     ( 68 / 255.0,  50 / 255.0,  38 / 255.0),  # Erend, Oseram, Carja, Tenakth, Utaru default
    "warm_chestnut":  (108 / 255.0,  78 / 255.0,  56 / 255.0),  # Alva, warm brown/sandy
    "golden_blonde":  (168 / 255.0, 138 / 255.0,  96 / 255.0),  # Tilda, Zenith blonde
    "silver_grey":    (145 / 255.0, 142 / 255.0, 138 / 255.0),  # Elders (Dekka, Hekkaro, Rost grey)
}


def infer_character_hair_color(
    character_hint: str = "",
    variant_hint: str = "",
    texture_hint: str = "",
) -> dict[str, Any]:
    """Select canonical HFW `382:0` Hair LUT tone (linear + sRGB) for hair/lash/brow cards without baked RGB."""
    blob = f"{character_hint} {variant_hint} {texture_hint}".lower()

    if any(k in blob for k in ("aloy", "beta", "elisabet", "sobeck")):
        preset = "aloy_auburn"
    elif any(k in blob for k in ("seyka", "ceo", "bohai", "quen", "kotallo", "zo", "varl", "regalla", "sylens", "talanah", "nil")):
        preset = "espresso_black"
    elif any(k in blob for k in ("tilda", "gerard", "erik")):
        preset = "golden_blonde"
    elif any(k in blob for k in ("alva", "petra")):
        preset = "warm_chestnut"
    elif any(k in blob for k in ("dekka", "rost", "elder")):
        preset = "silver_grey"
    else:
        preset = "dark_brown"

    srgb = _HAIR_PRESETS_SRGB[preset]
    return {
        "preset": preset,
        "hair_color_srgb": [round(c, 4) for c in srgb],
        "hair_color_linear": srgb_to_linear_rgb(srgb),
    }


def _parse_dxil_float(tok: str) -> float | None:
    tok = tok.strip().rstrip(",")
    if tok.startswith("0x") and len(tok) == 18:
        try:
            return struct.unpack(">d", bytes.fromhex(tok[2:]))[0]
        except Exception:
            return None
    if tok in ("undef", "false"):
        return 0.0
    if tok == "true":
        return 1.0
    try:
        return float(tok)
    except Exception:
        return None


_DEF_RE = re.compile(r"^(%\d+)\s*=\s*(.*)")
_CBUF_I32_RE = re.compile(r"cbufferLoadLegacy\.i32\(i32 59, %dx\.types\.Handle %\d+, i32 (\d+)\)")
_EXT_I32_RE = re.compile(r"^extractvalue %dx\.types\.CBufRet\.i32 (%\d+), (\d+)")
_EXT_F32_RE = re.compile(r"^extractvalue %dx\.types\.(?:CBufRet|ResRet)\.f32 (%\d+), (\d+)")
_HEAP_RE = re.compile(r"createHandleFromHeap\(i32 218, i32 (%\d+), i1 false")
_ANNOT_RE = re.compile(r"annotateHandle\(i32 216, %dx\.types\.Handle (%\d+),")
_SAMPLE_RE = re.compile(
    r"dx\.op\.sample(?:\w*)\.f32\(i32 \d+, %dx\.types\.Handle (%\d+), %dx\.types\.Handle %\d+, float ([^,]+), float ([^,]+),"
)
_BINF_RE = re.compile(r"^(fmul|fadd|fsub|fdiv)\s+(?:fast\s+)?float\s+([^,]+),\s*(\S+)")
_FMAD_RE = re.compile(r"dx\.op\.tertiary\.f32\(i32 46, float ([^,]+), float ([^,]+), float ([^)]+)\)")
_UNARY_RE = re.compile(r"dx\.op\.unary\.f32\(i32 (\d+), float ([^)]+)\)")
_BINARY_RE = re.compile(r"dx\.op\.binary\.f32\(i32 (\d+), float ([^,]+), float ([^)]+)\)")
_ICMP_RE = re.compile(r"^icmp\s+(\w+)\s+i32\s+([^,]+),\s*(\S+)")
_FCMP_RE = re.compile(r"^fcmp\s+(?:fast\s+)?(\w+)\s+float\s+([^,]+),\s*(\S+)")
_SEL_RE = re.compile(r"^select\s+i1\s+(%\d+),\s*float\s+([^,]+),\s*float\s+(\S+)")
_SUBMAT_RE = re.compile(r"shaders_hrz2/materials/([^/]+)/([^/]+)/")
_PHI_PAIR_RE = re.compile(r"\[\s*([^,\]]+),\s*(%\d+)\s*\]")


def extract_dxil_outfit_corners(
    game: HFWGame,
    mat: MaterialInfo,
    dxc: DXCDisassembler | None = None,
) -> list[dict[str, Any]] | None:
    """Evaluate an HFW layered outfit shader's DXIL `PixelProgram` to extract all 8 `MSK.RGB` corners.

    Returns an 8-element list (indexed `0..7`: Black, Red, Green, Blue, Cyan, Magenta, Yellow, White),
    each dict containing:
      - `corner`: int `0..7`
      - `corner_name`: `"Black"`..`"White"`
      - `dye_color_srgb`: `[r, g, b]` (0..1)
      - `dye_color_linear`: `[r, g, b]` (0..1, exact linear PBR Base Color)
      - `f0_rgb`: `[r, g, b]` (Decima Reflectance/Specular output)
      - `is_metallic`: bool (`max(f0_rgb) > 0.35`)
      - `roughness`: float (`1.0 - smoothness`)
      - `submat_key`: `"category/matname"` (e.g. `"weave/silk"`, `"metal/metal_robotplate"`) or `""`
      - `active`: bool (whether this corner has authored shading vs unused neutral fallback)

    Returns `None` if DXC is unavailable or this material is not an 8-corner `switch` outfit shader.
    """
    # Collect candidate techniques with distinct shader_ids (a TechniqueSet holds DepthOnly, Shadowmap,
    # AND Deferred techniques that often share the exact same TextureBinding count; the Deferred GBuffer
    # technique is the one whose 8-case switch merges >= 10 PBR float phis!).
    cand_techs = []
    seen_sids: set[str] = set()
    if mat.primary_technique is not None and mat.primary_technique.shader_id:
        cand_techs.append(mat.primary_technique)
        seen_sids.add(mat.primary_technique.shader_id)
    for t in mat.techniques:
        if t.shader_id and t.shader_id not in seen_sids and t.texture_bindings:
            cand_techs.append(t)
            seen_sids.add(t.shader_id)
    if not cand_techs:
        return None

    try:
        disasm_tool = dxc or DXCDisassembler(game.config)
    except Exception:
        return None

    for pt in cand_techs:
        res = _eval_technique_dxil_outfit_corners(game, pt, disasm_tool)
        if res is not None:
            return res
    return None


def _eval_technique_dxil_outfit_corners(
    game: HFWGame,
    pt: Any,
    disasm_tool: DXCDisassembler,
) -> list[dict[str, Any]] | None:
    try:
        sobj = game.read_object(ObjectId.parse(pt.shader_id), read_subgroups=True)
        hw = parse_hw_shader(sobj)
        ps = hw.get_program("PixelProgram")
        if ps is None:
            return None
        dis = disasm_tool.disassemble_blob(ps.blob)
    except Exception:
        return None

    lines = dis.splitlines()

    # Locate the 8-case switch on MSK.RGB whose merge block has >= 10 float phi nodes (PBR outputs!)
    target_cases: dict[int, str] | None = None
    target_merge_lbl = ""
    target_phis: list[tuple[str, dict[str, str]]] = []
    target_merge_line = -1

    label_lines: dict[str, int] = {}
    for idx, ln in enumerate(lines):
        if ln.startswith("; <label>:"):
            lid = ln[10:].split()[0].strip()
            label_lines[f"%{lid}"] = idx

    for idx, ln in enumerate(lines):
        if "switch i32 " not in ln or "label %" not in ln:
            continue
        m_lbl = re.search(r"label (%\d+)", ln)
        if not m_lbl:
            continue
        cand_cases: dict[int, str] = {}
        for k in range(idx + 1, min(idx + 14, len(lines))):
            mc = re.match(r"\s*i32 (\d+), label (%\d+)", lines[k])
            if mc:
                cand_cases[int(mc.group(1))] = mc.group(2)
        if len(cand_cases) < 4:
            continue
        mlbl = m_lbl.group(1)
        mline = label_lines.get(mlbl, -1)
        if mline < 0:
            continue
        phis: list[tuple[str, dict[str, str]]] = []
        for k in range(mline + 1, min(mline + 32, len(lines))):
            if " = phi float " in lines[k]:
                pairs = {lbl: val.strip() for val, lbl in _PHI_PAIR_RE.findall(lines[k])}
                phis.append((lines[k].split("=")[0].strip(), pairs))
            elif phis and not lines[k].strip().startswith("%"):
                break
        if len(phis) >= 10:
            target_cases = cand_cases
            target_merge_lbl = mlbl
            target_phis = phis
            target_merge_line = mline
            break

    if target_cases is None or len(target_phis) < 10:
        return None

    # Map TextureBindings -> sub-material keys (`shaders_hrz2/materials/<cat>/<name>`)
    tb_map = {tb.index: tb for tb in pt.texture_bindings}
    submat_tb_key: dict[int, str] = {}
    for tb in pt.texture_bindings:
        p = (" ".join(tb.source_paths.values()) + " " + tb.name).replace("\\", "/").lower()
        if "/shared/" in p or "/effects/" in p:
            continue
        ms = _SUBMAT_RE.search(p)
        if ms:
            submat_tb_key[tb.index] = f"{ms.group(1)}/{ms.group(2)}"

    # Single-pass SSA forward evaluator up to `target_merge_line`!
    vals: dict[str, float] = {}
    tvecs: dict[str, tuple[float, float, float, float]] = {}
    bools: dict[str, bool] = {}
    cb_reg: dict[str, int] = {}
    tb_of: dict[str, int] = {}
    var_tbs: dict[str, set[int]] = {}
    # Also track per-block TB samples so even indirect/normal-driven cases know their sub-material!
    block_tbs: dict[str, set[int]] = {}
    cur_block = "%0"

    def getf(tok: str) -> float:
        tok = tok.strip()
        if tok.startswith("%"):
            return vals.get(tok, 0.0)
        v = _parse_dxil_float(tok)
        return 0.0 if v is None else v

    def get_tbs(*toks: str) -> set[int]:
        out: set[int] = set()
        for t in toks:
            t = t.strip()
            if t in var_tbs:
                out |= var_tbs[t]
        return out

    for idx in range(target_merge_line):
        ln = lines[idx]
        if ln.startswith("; <label>:"):
            cur_block = "%" + ln[10:].split()[0].strip()
            continue
        s = ln.strip()
        if not s or s.startswith(";") or s.startswith("}"):
            continue
        m = _DEF_RE.match(s)
        if not m:
            continue
        var, rhs = m.group(1), m.group(2)

        if "cbufferLoadLegacy.i32" in rhs:
            mc = _CBUF_I32_RE.search(rhs)
            if mc:
                cb_reg[var] = int(mc.group(1))
            tvecs[var] = (0.0, 0.0, 0.0, 0.0)  # CustomDye = 0 -> evaluate authored default colors!
            continue
        if "cbufferLoadLegacy.f32" in rhs:
            tvecs[var] = (0.0, 0.0, 0.0, 0.0)
            continue

        me_i = _EXT_I32_RE.match(rhs)
        if me_i:
            if me_i.group(1) in cb_reg:
                tb_of[var] = cb_reg[me_i.group(1)] * 4 + int(me_i.group(2))
            vals[var] = 0.0
            continue

        me_f = _EXT_F32_RE.match(rhs)
        if me_f:
            src, lane = me_f.group(1), int(me_f.group(2))
            vals[var] = tvecs.get(src, (0.0, 0.0, 0.0, 0.0))[lane]
            if src in var_tbs:
                var_tbs[var] = set(var_tbs[src])
            continue

        if "createHandleFromHeap" in rhs:
            mh = _HEAP_RE.search(rhs)
            if mh and mh.group(1) in tb_of:
                tb_of[var] = tb_of[mh.group(1)]
            continue

        if "annotateHandle" in rhs:
            ma = _ANNOT_RE.search(rhs)
            if ma and ma.group(1) in tb_of:
                tb_of[var] = tb_of[ma.group(1)]
            continue

        if "dx.op.sample" in rhs:
            ms = _SAMPLE_RE.search(rhs)
            if ms:
                tb_i = tb_of.get(ms.group(1), -1)
                tb = tb_map.get(tb_i)
                if tb is not None and tb.semantic_role == "Misc_01":
                    tvecs[var] = (0.0, 0.0, 0.0, 0.0)  # 0% wear/damage -> clean pristine Base layer!
                elif tb is not None and ("Normal.R" in tb.target_channels.values()):
                    tvecs[var] = (0.5, 0.5, 0.5, 0.5)
                elif tb is not None and any("Reflectance" in v for v in tb.target_channels.values()):
                    tvecs[var] = (0.85, 0.85, 0.85, 1.0)
                else:
                    tvecs[var] = (0.5, 0.5, 0.5, 1.0)  # Neutral 0.5 linear detail color!
                if tb_i in submat_tb_key:
                    var_tbs[var] = {tb_i}
                    block_tbs.setdefault(cur_block, set()).add(tb_i)
            continue

        mb = _BINF_RE.match(rhs)
        if mb:
            op, a, b = mb.group(1), mb.group(2).strip(), mb.group(3).strip()
            fa, fb = getf(a), getf(b)
            if op == "fmul":
                vals[var] = fa * fb
            elif op == "fadd":
                vals[var] = fa + fb
            elif op == "fsub":
                vals[var] = fa - fb
            elif op == "fdiv":
                vals[var] = (fa / fb) if abs(fb) > 1e-9 else 0.0
            tbs = get_tbs(a, b)
            if tbs:
                var_tbs[var] = tbs
            continue

        if "dx.op.tertiary" in rhs:
            mfmad = _FMAD_RE.search(rhs)
            if mfmad:
                a, b, c = mfmad.group(1), mfmad.group(2), mfmad.group(3)
                vals[var] = getf(a) * getf(b) + getf(c)
                tbs = get_tbs(a, b, c)
                if tbs:
                    var_tbs[var] = tbs
            continue

        if "dx.op.unary" in rhs:
            mun = _UNARY_RE.search(rhs)
            if mun:
                code, a = int(mun.group(1)), mun.group(2).strip()
                fa = getf(a)
                if code == 6:
                    vals[var] = abs(fa)
                elif code == 7:
                    vals[var] = max(0.0, min(1.0, fa))
                elif code == 21:
                    vals[var] = 2.0 ** max(-20.0, min(20.0, fa))
                elif code == 23:
                    vals[var] = math.log2(max(1e-7, fa))
                elif code == 24:
                    vals[var] = math.sqrt(max(0.0, fa))
                elif code == 25:
                    vals[var] = 1.0 / math.sqrt(max(1e-9, fa))
                else:
                    vals[var] = fa
                tbs = get_tbs(a)
                if tbs:
                    var_tbs[var] = tbs
            continue

        if "dx.op.binary" in rhs:
            mbin = _BINARY_RE.search(rhs)
            if mbin:
                code, a, b = int(mbin.group(1)), mbin.group(2).strip(), mbin.group(3).strip()
                fa, fb = getf(a), getf(b)
                if code == 35:
                    vals[var] = max(fa, fb)
                elif code == 36:
                    vals[var] = min(fa, fb)
                tbs = get_tbs(a, b)
                if tbs:
                    var_tbs[var] = tbs
            continue

        micmp = _ICMP_RE.match(rhs)
        if micmp:
            pred, a, b = micmp.group(1), micmp.group(2).strip(), micmp.group(3).strip()
            ia, ib = int(round(getf(a))), int(round(getf(b)))
            bools[var] = (ia != ib) if pred == "ne" else ((ia == ib) if pred == "eq" else False)
            continue

        mfcmp = _FCMP_RE.match(rhs)
        if mfcmp:
            pred, a, b = mfcmp.group(1), mfcmp.group(2).strip(), mfcmp.group(3).strip()
            fa, fb = getf(a), getf(b)
            if pred in ("ogt", "ugt"):
                bools[var] = fa > fb
            elif pred in ("olt", "ult"):
                bools[var] = fa < fb
            elif pred in ("oge", "uge"):
                bools[var] = fa >= fb
            elif pred in ("ole", "ule"):
                bools[var] = fa <= fb
            else:
                bools[var] = False
            continue

        msel = _SEL_RE.match(rhs)
        if msel:
            cond = bools.get(msel.group(1), False)
            chosen = msel.group(2).strip() if cond else msel.group(3).strip()
            vals[var] = getf(chosen)
            tbs = get_tbs(chosen)
            if tbs:
                var_tbs[var] = tbs
            continue

    # Identify merge block phi roles dynamically from the DXIL post-merge GBuffer math!
    # Standard tail layout in Guerrilla's 8-corner outfit shaders:
    #   [-3:]    = Normal XYZ (added to macro normal)
    #   [-6:-3]  = Albedo RGB (sRGB-like, pre-pow2.2!)
    #   F0 / Specular immediately precedes Albedo, BUT DXC CSE collapses monochrome F0 into 1 phi
    #   or eliminates F0 completely (0 phis) when all 8 corners are pure dielectric (`F0 = 0.2`)!
    #   Every F0 phi is unambiguously marked post-merge by `fsub 0x3FC99999A0000000, %phi` (`0.2 - F0`)!
    albedo_phis = target_phis[-6:-3]
    post_block = "\n".join(lines[target_merge_line + 1 : min(len(lines), target_merge_line + 200)])
    n_phis = len(target_phis)
    f0_abs_idxs: list[int] = []
    for pi in range(max(0, n_phis - 10), max(0, n_phis - 6)):
        pv = target_phis[pi][0]
        if re.search(rf"0x3FC99999A0000000,\s*{re.escape(pv)}\b", post_block):
            f0_abs_idxs.append(pi)

    if len(f0_abs_idxs) == 3:
        spec_phis: list[tuple[str, dict[str, str]]] = [target_phis[i] for i in f0_abs_idxs]
        smooth_idx = f0_abs_idxs[0] - 1
    elif len(f0_abs_idxs) == 1:
        # Monochrome / scalar F0 (e.g. Aloy Nora Arms/Pants/Pouch: only silver/iron/dielectric -> 1 phi!)
        spec_phis = [target_phis[f0_abs_idxs[0]]] * 3
        smooth_idx = f0_abs_idxs[0] - 1
    elif len(f0_abs_idxs) == 0:
        # Pure dielectric outfit shader (e.g. Seyka Tassels: constant F0 = 0.2 everywhere -> 0 F0 phis!)
        spec_phis = []
        smooth_idx = n_phis - 7
    else:
        spec_phis = [target_phis[f0_abs_idxs[-1]]] * 3
        smooth_idx = f0_abs_idxs[0] - 1

    smooth_phi = target_phis[smooth_idx] if 0 <= smooth_idx < n_phis else None

    corners: list[dict[str, Any]] = []
    any_real_data = False

    for c_idx in range(8):
        clbl = target_cases.get(c_idx, "")
        raw_a = [max(0.0, min(1.0, getf(p[1].get(clbl, "0.5")))) for p in albedo_phis]
        lin_a = [round(c ** 2.2, 5) for c in raw_a]
        if spec_phis:
            raw_s = [round(max(0.0, min(1.0, getf(p[1].get(clbl, "0.2")))), 4) for p in spec_phis]
        else:
            raw_s = [0.2, 0.2, 0.2]
        sm = max(0.05, min(0.95, getf(smooth_phi[1].get(clbl, "0.5")))) if smooth_phi is not None else 0.5
        rgh = round(max(0.14, min(0.90, 1.0 - sm)), 3)

        # Find which sub-material TB(s) fed this corner's Albedo, Specular, or Normal
        tbs_used: set[int] = set()
        for p in albedo_phis + spec_phis + target_phis[-3:]:
            tok = p[1].get(clbl, "").strip()
            tbs_used |= get_tbs(tok)
        if not tbs_used and clbl in block_tbs:
            tbs_used |= block_tbs[clbl]

        # Pick the primary sub-material key (prefer one with Color binding!)
        sub_keys: list[str] = []
        for ti in sorted(tbs_used):
            sk = submat_tb_key.get(ti, "")
            if sk and sk not in sub_keys:
                sub_keys.append(sk)
        primary_key = sub_keys[0] if sub_keys else ""

        is_met = (max(raw_s) > 0.35) or any(
            k in primary_key for k in ("metal/", "bronze", "brass", "copper", "iron", "gold", "silver", "steel")
        )
        # Check if this corner was genuinely authored vs unused fallback (0.50, 0.50, 0.50 with no submat)
        is_unused = (not primary_key) and all(abs(c - 0.5) < 0.005 for c in raw_a)
        if not is_unused:
            any_real_data = True

        # Cap non-metallic diffuse slightly below 1.0 (0.92 sRGB = 0.83 linear) so pure white
        # fabrics (e.g. Seyka Quen cotton crochet) stay crisp without clipping AgX/ACES highlights!
        if not is_met:
            raw_a = [min(0.92, c) for c in raw_a]
            lin_a = [round(c ** 2.2, 5) for c in raw_a]

        # For pure metallic layers whose diffuse Albedo was darkened for specular workflow, ensure
        # the Blender Metallic BaseColor carries bright metal reflectance tint (`max(lin_a, f0^2)`!)
        if is_met and max(raw_s) > 0.35:
            f0_sq = [round(c * c, 5) for c in raw_s]
            # Blend authored diffuse tint with metallic F0 brightness cleanly
            eff_lin = [round(max(lin_a[k], f0_sq[k] * 0.85), 5) for k in range(3)]
        else:
            eff_lin = lin_a

        corners.append(
            {
                "corner": c_idx,
                "corner_name": CORNER_RGB_LABELS[c_idx],
                "submat_key": primary_key,
                "dye_color_srgb": [round(c, 4) for c in raw_a],
                "dye_color_linear": eff_lin,
                "f0_rgb": raw_s,
                "is_metallic": bool(is_met),
                "roughness": rgh,
                "active": not is_unused,
            }
        )

    return corners if any_real_data else None
