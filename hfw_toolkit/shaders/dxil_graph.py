"""Detailed DXIL SSA Expression Lifter & Decima Engine Specialty Node Graph Builder.

Parses compiled SM 6.x DXIL LLVM-IR (`DisassembledProgram.disassembly`) to lift real GPU instructions
into an interactive, readable node graph showing:
  - Genuine math nodes (`Multiply`, `Add`, `Subtract`, `Divide`, `FMad`, `Lerp`, `Saturate`, `Clamp`,
    `Square / Pow`, `Sin / Cos`, `Frac`, `Abs`, `Dot3`, `Normalize`, `Compare / Select`) with decoded
    IEEE-754 float constants and exact SSA register provenance.
  - Decima Engine Specialty Nodes detected directly from live instruction signatures (`23-Slice Detail Array`,
    `Dual-Phase Flow Map`, `Normal Z Reconstruct`, `Parallax Occlusion Raymarch`, `RGB <-> HSL Rim Shift`,
    `Worn-Metal F0 (RFL^2)`, `Smoothness -> Roughness`, `Octahedral Normal Encode`, `TBN Tangent->World`,
    `Fresnel (N·V)`, `GameTime Animator`, `Weather/Wetness Hook`, `Alpha-Test Cutout`).
  - Exact `SV_Target0..6` G-Buffer or Forward outputs with per-target filter metadata.
"""

from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass, field
from typing import Any

from hfw_toolkit.extract.materials import MaterialInfo
from hfw_toolkit.shaders.dxc import DisassembledProgram
from hfw_toolkit.shaders.dxil_ir import DXILAnalysis

# ---------------------------------------------------------------------------
# Low-level LLVM-IR literal & instruction parsing helpers
# ---------------------------------------------------------------------------

_SSA_DEF_RE = re.compile(r"^\s*(%\d+)\s*=\s*(.+)$")
_SSA_REF_RE = re.compile(r"%\d+")
_HEX_FLOAT_RE = re.compile(r"^0x([0-9A-Fa-f]{16})$")
_HANDLE_BIND_RE = re.compile(
    r"dx\.op\.createHandleFromBinding\([^,]+,\s*%dx\.types\.ResBind\s*\{i32\s*(\d+),\s*i32\s*(\d+),\s*i32\s*(\d+),\s*i8\s*(\d+)\}"
)
_HANDLE_LEGACY_RE = re.compile(r"dx\.op\.createHandle\(i32\s*\d+,\s*i8\s*(\d+),\s*i32\s*(\d+),")
_HANDLE_HEAP_RE = re.compile(r"dx\.op\.createHandleFromHeap\(i32\s*\d+,\s*i32\s*([^,]+),\s*i1\s*(true|false)")
_ANNOTATE_RE = re.compile(r"dx\.op\.annotateHandle\(i32\s*\d+,\s*%dx\.types\.Handle\s*(%\d+).*?resource:\s*([^\r\n;]+)")
_LOAD_INPUT_RE = re.compile(r"dx\.op\.loadInput\.\w+\(i32\s*\d+,\s*i32\s*(\d+),\s*i32\s*(\d+),\s*i8\s*(\d+)")
_CBUF_LOAD_RE = re.compile(r"dx\.op\.cbufferLoadLegacy\.\w+\(i32\s*\d+,\s*%dx\.types\.Handle\s*(%\d+),\s*i32\s*([^)\s]+)")
_EXTRACT_RE = re.compile(r"extractvalue\s+%dx\.types\.(?:ResRet|CBufRet)\.\S+\s+(%\d+),\s*(\d+)")
_SAMPLE_RE = re.compile(
    r"dx\.op\.(sample|sampleLevel|sampleBias|sampleGrad|textureLoad)\.[^(]+\("
    r"i32\s*\d+,\s*%dx\.types\.Handle\s*(%\d+),\s*(?:%dx\.types\.Handle\s*%\d+,\s*)?"
    r"(?:float|i32)\s*([^,]+),\s*(?:float|i32)\s*([^,]+)(?:,\s*(?:float|i32)\s*([^,]+))?"
)
_STORE_OUT_RE = re.compile(
    r"dx\.op\.storeOutput\.[^(]+\(i32\s*\d+,\s*i32\s*(\d+),\s*i32\s*(\d+),\s*i8\s*(\d+),\s*\S+\s*([^)\s]+)"
)
_BIN_FOP_RE = re.compile(r"^(fmul|fadd|fsub|fdiv)\s+(?:fast\s+)?float\s+([^,]+),\s*([^\s;]+)")
_UNARY_DX_RE = re.compile(r"dx\.op\.unary\.f32\(i32\s*(\d+),\s*float\s*([^)\s]+)\)")
_BINARY_DX_RE = re.compile(r"dx\.op\.binary\.f32\(i32\s*(\d+),\s*float\s*([^,]+),\s*float\s*([^)\s]+)\)")
_TERT_DX_RE = re.compile(r"dx\.op\.tertiary\.f32\(i32\s*(\d+),\s*float\s*([^,]+),\s*float\s*([^,]+),\s*float\s*([^)\s]+)\)")
_DOT2_RE = re.compile(r"dx\.op\.dot2\.f32\(i32\s*\d+,\s*float\s*([^,]+),\s*float\s*([^,]+),\s*float\s*([^,]+),\s*float\s*([^)\s]+)\)")
_DOT3_RE = re.compile(
    r"dx\.op\.dot3\.f32\(i32\s*\d+,\s*float\s*([^,]+),\s*float\s*([^,]+),\s*float\s*([^,]+),\s*float\s*([^,]+),\s*float\s*([^,]+),\s*float\s*([^)\s]+)\)"
)
_SELECT_RE = re.compile(r"^select\s+i1\s+(%\d+),\s*\S+\s+([^,]+),\s*\S+\s+([^\s;]+)")
_FCMP_RE = re.compile(r"^fcmp\s+(?:fast\s+)?(\w+)\s+float\s+([^,]+),\s*([^\s;]+)")
_PHI_RE = re.compile(r"^phi\s+(?:float|i32)\s+(.+)$")

_DX_UNARY_NAMES: dict[int, str] = {
    6: "Abs",
    7: "Saturate",
    12: "Cos",
    13: "Sin",
    14: "Tan",
    15: "Acos",
    16: "Asin",
    17: "Atan",
    21: "Exp2",
    22: "Frac",
    23: "Log2",
    24: "Sqrt",
    25: "Rsqrt",
    26: "Round",
    27: "Floor",
    28: "Ceil",
    29: "Trunc",
    85: "dFdx",
    86: "dFdy",
}

_DX_BINARY_NAMES: dict[int, str] = {
    35: "Max",
    36: "Min",
}

_SPACE_NAMES: dict[int, str] = {
    1: "Scratch_PerFrame",
    2: "Scratch_PerPass",
    3: "Scratch_PerView",
    5: "Scratch_PerBatch",
    7: "ShaderInstance_PerBatch",
    8: "ShaderInstance_PerInstance",
}

_DEFERRED_TARGET_META: dict[int, tuple[str, str, str]] = {
    0: ("SV_Target0 — Albedo + AO", "Deferred G-Buffer RT0 (Base Color RGB, AO in A)", "albedo"),
    1: ("SV_Target1 — HDR Emissive", "Deferred G-Buffer RT1 (Glow / Emissive RGB)", "emissive"),
    2: ("SV_Target2 — Octahedral Normal", "Deferred G-Buffer RT2 (Packed World Normal XY)", "normal"),
    3: ("SV_Target3 — Smoothness & Flags", "Deferred G-Buffer RT3 (Smoothness + Material Flags)", "roughness"),
    4: ("SV_Target4 — Specular F0", "Deferred G-Buffer RT4 (Reflectance F0 RGB)", "specular"),
    5: ("SV_Target5 — Motion Vectors", "Deferred G-Buffer RT5 (Velocity XY)", "motion"),
    6: ("SV_Target6 — Subsurface / Misc", "Deferred G-Buffer RT6 (Misc Flags)", "misc"),
}

_FORWARD_TARGET_META: dict[int, tuple[str, str, str]] = {
    0: ("SV_Target0 — Final Color + Alpha", "Forward Pass RGBA Output", "albedo"),
    1: ("SV_Target1 — Aux / Specular", "Forward Pass Secondary Target", "emissive"),
    2: ("SV_Target2 — Velocity / Flags", "Forward Pass Motion / Misc", "motion"),
}


def decode_llvm_float(tok: str) -> float | None:
    """Decode either a decimal float literal or an LLVM 64-bit hex double literal (`0x3FF1C28F60000000`)."""
    t = tok.strip()
    if not t or t.startswith("%") or t in ("undef", "poison"):
        return None
    mh = _HEX_FLOAT_RE.match(t)
    if mh:
        try:
            val = struct.unpack(">d", bytes.fromhex(mh.group(1)))[0]
            if math.isfinite(val):
                return val
        except Exception:
            return None
        return None
    try:
        val = float(t)
        return val if math.isfinite(val) else None
    except ValueError:
        return None


def fmt_float(val: float) -> str:
    if abs(val) < 1e-6:
        return "0"
    if abs(val - round(val)) < 1e-4 and abs(val) < 1e6:
        return f"{int(round(val))}.0"
    if abs(val) >= 1000:
        return f"{val:.1f}"
    return f"{val:.4g}"


def fmt_operand(tok: str) -> str:
    v = decode_llvm_float(tok)
    if v is not None:
        return fmt_float(v)
    return tok.strip()


# ---------------------------------------------------------------------------
# Parsed SSA IR Representation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SSAInst:
    var: str
    line_no: int
    raw: str
    op: str  # loadInput | cbufLoad | extract | sample | fmul | fadd | fsub | fdiv | fmad | unary | binary | dot2 | dot3 | select | fcmp | phi | other
    args: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ParsedPS:
    insts: dict[str, SSAInst]
    order: list[str]
    stores: list[tuple[int, int, str, int]]  # (target_id, col_0..3, val_tok, line_no)
    handles: dict[str, dict[str, Any]]
    switch_count: int
    switch_max_cases: int
    has_discard: bool
    has_calc_lod: bool
    sample_level_count: int
    sample_vars: list[str]
    input_sig: dict[int, str]
    output_count: int


def _parse_pixel_program(prog: DisassembledProgram) -> ParsedPS:
    by_class_id: dict[tuple[int, int], tuple[int, int, str]] = {}
    for b in prog.bindings:
        cls = {"texture": 0, "UAV": 1, "cbuffer": 2, "sampler": 3}.get(b.res_type, -1)
        m_id = re.search(r"(\d+)$", b.res_id)
        if cls >= 0 and m_id:
            by_class_id[(cls, int(m_id.group(1)))] = (b.space, b.register, b.dim)

    input_sig: dict[int, str] = {}
    for inp in prog.inputs:
        input_sig[inp.register] = f"{inp.name}{inp.index}" if inp.name != "SV_Position" else "SV_Position"

    insts: dict[str, SSAInst] = {}
    order: list[str] = []
    stores: list[tuple[int, int, str, int]] = []
    handles: dict[str, dict[str, Any]] = {}
    sample_vars: list[str] = []
    switch_count = 0
    switch_max_cases = 0
    has_discard = False
    has_calc_lod = False
    sample_level_count = 0

    in_main = False
    in_switch_cases = 0
    for line_no, line in enumerate(prog.disassembly.splitlines(), start=1):
        if line.startswith("define void @main("):
            in_main = True
            continue
        if in_main and line.startswith("}"):
            break
        if not in_main:
            continue
        s = line.strip()
        if not s or s.startswith(";") or s.endswith(":"):
            continue

        if s.startswith("switch "):
            switch_count += 1
            in_switch_cases = 1
            continue
        if in_switch_cases > 0:
            if s.startswith("]"):
                switch_max_cases = max(switch_max_cases, in_switch_cases - 1)
                in_switch_cases = 0
            elif "label %" in s:
                in_switch_cases += 1
            continue

        if "@dx.op.discard(" in s:
            has_discard = True
            continue

        ms = _STORE_OUT_RE.search(s)
        if ms:
            stores.append((int(ms.group(1)), int(ms.group(3)), ms.group(4), line_no))
            continue

        mdef = _SSA_DEF_RE.match(s)
        if not mdef:
            continue
        var, rhs = mdef.group(1), mdef.group(2)
        order.append(var)

        if "dx.op.calculateLOD" in rhs:
            has_calc_lod = True

        # Handle creation / annotation
        mhb = _HANDLE_BIND_RE.search(rhs)
        if mhb:
            handles[var] = {
                "space": int(mhb.group(3)),
                "reg": int(mhb.group(1)),
                "class": int(mhb.group(4)),
                "bindless": False,
                "resource_desc": "",
            }
            insts[var] = SSAInst(var, line_no, s, "handle")
            continue
        mhl = _HANDLE_LEGACY_RE.search(rhs)
        if mhl:
            cls, rid = int(mhl.group(1)), int(mhl.group(2))
            sp, rg, dim = by_class_id.get((cls, rid), (None, None, ""))
            handles[var] = {"space": sp, "reg": rg, "class": cls, "bindless": False, "resource_desc": dim}
            insts[var] = SSAInst(var, line_no, s, "handle")
            continue
        mhh = _HANDLE_HEAP_RE.search(rhs)
        if mhh:
            handles[var] = {
                "space": None,
                "reg": None,
                "bindless": True,
                "heap_idx_var": mhh.group(1).strip(),
                "is_sampler": mhh.group(2) == "true",
                "resource_desc": "",
            }
            insts[var] = SSAInst(var, line_no, s, "handle", [mhh.group(1).strip()])
            continue
        man = _ANNOTATE_RE.search(rhs)
        if man:
            base = handles.get(man.group(1), {})
            info = dict(base)
            info["resource_desc"] = man.group(2).strip()
            handles[var] = info
            insts[var] = SSAInst(var, line_no, s, "handle", [man.group(1)])
            continue

        # Load input
        mli = _LOAD_INPUT_RE.search(rhs)
        if mli:
            insts[var] = SSAInst(
                var,
                line_no,
                s,
                "loadInput",
                extra={"sig_id": int(mli.group(1)), "col": int(mli.group(3))},
            )
            continue

        # CBuffer load
        mcb = _CBUF_LOAD_RE.search(rhs)
        if mcb:
            hinfo = handles.get(mcb.group(1), {})
            insts[var] = SSAInst(
                var,
                line_no,
                s,
                "cbufLoad",
                [mcb.group(2)],
                extra={"space": hinfo.get("space"), "reg": hinfo.get("reg"), "reg_index": mcb.group(2)},
            )
            continue

        # Extractvalue
        mext = _EXTRACT_RE.search(rhs)
        if mext:
            insts[var] = SSAInst(var, line_no, s, "extract", [mext.group(1)], extra={"lane": int(mext.group(2))})
            continue

        # Texture sample
        msamp = _SAMPLE_RE.search(rhs)
        if msamp:
            op_kind = msamp.group(1)
            if op_kind == "sampleLevel":
                sample_level_count += 1
            hvar = msamp.group(2)
            hinfo = handles.get(hvar, {})
            u_tok = msamp.group(3).strip()
            v_tok = msamp.group(4).strip()
            w_tok = (msamp.group(5) or "").strip()
            args = [u_tok, v_tok] + ([w_tok] if w_tok and w_tok != "undef" else [])
            insts[var] = SSAInst(
                var,
                line_no,
                s,
                "sample",
                args,
                extra={"op_kind": op_kind, "handle_var": hvar, "handle": hinfo},
            )
            sample_vars.append(var)
            continue

        # Binary float ops
        mbf = _BIN_FOP_RE.match(rhs)
        if mbf:
            insts[var] = SSAInst(var, line_no, s, mbf.group(1), [mbf.group(2).strip(), mbf.group(3).strip()])
            continue

        # DXIL unary
        mun = _UNARY_DX_RE.search(rhs)
        if mun:
            code = int(mun.group(1))
            insts[var] = SSAInst(
                var,
                line_no,
                s,
                "unary",
                [mun.group(2).strip()],
                extra={"code": code, "name": _DX_UNARY_NAMES.get(code, f"Unary_{code}")},
            )
            continue

        # DXIL binary (FMax / FMin)
        mbi = _BINARY_DX_RE.search(rhs)
        if mbi:
            code = int(mbi.group(1))
            insts[var] = SSAInst(
                var,
                line_no,
                s,
                "binary",
                [mbi.group(2).strip(), mbi.group(3).strip()],
                extra={"code": code, "name": _DX_BINARY_NAMES.get(code, f"Bin_{code}")},
            )
            continue

        # DXIL tertiary (FMad = A*B + C)
        mte = _TERT_DX_RE.search(rhs)
        if mte:
            insts[var] = SSAInst(
                var,
                line_no,
                s,
                "fmad",
                [mte.group(2).strip(), mte.group(3).strip(), mte.group(4).strip()],
                extra={"code": int(mte.group(1))},
            )
            continue

        # Dot2 / Dot3
        md2 = _DOT2_RE.search(rhs)
        if md2:
            insts[var] = SSAInst(var, line_no, s, "dot2", [md2.group(i).strip() for i in range(1, 5)])
            continue
        md3 = _DOT3_RE.search(rhs)
        if md3:
            insts[var] = SSAInst(var, line_no, s, "dot3", [md3.group(i).strip() for i in range(1, 7)])
            continue

        # Select / FCmp / Phi
        msel = _SELECT_RE.match(rhs)
        if msel:
            insts[var] = SSAInst(
                var,
                line_no,
                s,
                "select",
                [msel.group(1), msel.group(2).strip(), msel.group(3).strip()],
            )
            continue
        mfc = _FCMP_RE.match(rhs)
        if mfc:
            insts[var] = SSAInst(
                var,
                line_no,
                s,
                "fcmp",
                [mfc.group(2).strip(), mfc.group(3).strip()],
                extra={"pred": mfc.group(1)},
            )
            continue
        mphi = _PHI_RE.match(rhs)
        if mphi:
            refs = _SSA_REF_RE.findall(mphi.group(1))
            # Filter out basic block label refs (every second % in `[ val, %bb ]` pairs)
            val_refs = [m.group(1).strip() for m in re.finditer(r"\[\s*([^,\]]+),", mphi.group(1))]
            insts[var] = SSAInst(var, line_no, s, "phi", val_refs, extra={"all_refs": refs})
            continue

        insts[var] = SSAInst(var, line_no, s, "other", _SSA_REF_RE.findall(rhs))

    return ParsedPS(
        insts=insts,
        order=order,
        stores=stores,
        handles=handles,
        switch_count=switch_count,
        switch_max_cases=switch_max_cases,
        has_discard=has_discard,
        has_calc_lod=has_calc_lod,
        sample_level_count=sample_level_count,
        sample_vars=sample_vars,
        input_sig=input_sig,
        output_count=len(prog.outputs),
    )


# ---------------------------------------------------------------------------
# Texture & Variable Correlation + Decima Specialty Pattern Detection
# ---------------------------------------------------------------------------


def _describe_texture_sample(
    samp: SSAInst,
    parsed: ParsedPS,
    mat: MaterialInfo,
    binding_map: dict[str, dict[str, Any]],
    bindless_slot_names: dict[tuple[int, int], tuple[str, str, dict[str, Any]]],
) -> tuple[str, str, str, dict[str, Any]]:
    """Return `(node_key, title, subtitle, meta)` for a texture sample instruction."""
    h = samp.extra.get("handle") or {}
    rdesc = h.get("resource_desc", "Texture2D")
    dim_tag = "2DArray" if "2DArray" in rdesc else ("3D" if "3D" in rdesc else ("Cube" if "Cube" in rdesc else "2D"))

    if h.get("bindless"):
        idx_var = h.get("heap_idx_var", "")
        idx_inst = parsed.insts.get(idx_var)
        if idx_inst and idx_inst.op == "extract":
            cb_var = idx_inst.args[0]
            lane = idx_inst.extra.get("lane", 0)
            cb_inst = parsed.insts.get(cb_var)
            if cb_inst and cb_inst.op == "cbufLoad":
                sp = cb_inst.extra.get("space")
                ridx_s = str(cb_inst.extra.get("reg_index", "0"))
                ridx = int(ridx_s) if ridx_s.isdigit() else 0
                slot_idx = ridx * 4 + lane
                key = f"tex_heap_{ridx}_{lane}"
                if (ridx, lane) in bindless_slot_names:
                    tname, tsub, tmeta = bindless_slot_names[(ridx, lane)]
                    return key, tname, f"{tsub} • {dim_tag}", {**tmeta, "heap_slot": slot_idx, "dim": dim_tag}
                return (
                    key,
                    f"Bindless Tex[{slot_idx}] ({dim_tag})",
                    f"cb0,space{sp or 7}[{ridx}].{'xyzw'[lane]} • {rdesc}",
                    {"heap_slot": slot_idx, "dim": dim_tag, "resource_desc": rdesc},
                )
        return (
            f"tex_bl_{samp.var}",
            f"Bindless Sample ({dim_tag})",
            f"{samp.extra.get('op_kind', 'sample')} • {rdesc}",
            {"dim": dim_tag, "resource_desc": rdesc},
        )

    sp = h.get("space")
    rg = h.get("reg")
    key = f"tex_s{sp}_t{rg}"
    corr = binding_map.get(f"tex:{sp}:{rg}")
    if corr:
        return (
            key,
            corr.get("name", f"t{rg},space{sp}"),
            f"{corr.get('semantic_role', 'Texture')} • t{rg},space{sp} ({dim_tag})",
            corr,
        )
    if sp == 2:
        return (
            key,
            f"Global World LUT (t{rg},space2)",
            f"Decima Engine Weather / Shadow / AO ({dim_tag})",
            {"space": sp, "reg": rg, "dim": dim_tag, "is_global": True},
        )
    return (
        key,
        f"Texture t{rg},space{sp} ({dim_tag})",
        f"{samp.extra.get('op_kind', 'sample')} • {rdesc}",
        {"space": sp, "reg": rg, "dim": dim_tag},
    )


def _build_bindless_slot_hints(mat: MaterialInfo, parsed: ParsedPS) -> dict[tuple[int, int], tuple[str, str, dict[str, Any]]]:
    """Correlate SM 6.6 PerBatch bindless texture heap slots (`cb0,space7[r].lane`) with known HFW textures."""
    hints: dict[tuple[int, int], tuple[str, str, dict[str, Any]]] = {}
    pt = mat.primary_technique
    if not pt:
        return hints

    # Gather unique texture samples by (ridx, lane, resource_desc)
    slots_seen: list[tuple[int, int, str]] = []
    seen_set: set[tuple[int, int]] = set()
    for sv in parsed.sample_vars:
        sinst = parsed.insts[sv]
        h = sinst.extra.get("handle") or {}
        if not h.get("bindless"):
            continue
        rdesc = h.get("resource_desc", "")
        idx_inst = parsed.insts.get(h.get("heap_idx_var", ""))
        if not (idx_inst and idx_inst.op == "extract"):
            continue
        cb_inst = parsed.insts.get(idx_inst.args[0])
        if not (cb_inst and cb_inst.op == "cbufLoad"):
            continue
        ridx_s = str(cb_inst.extra.get("reg_index", "0"))
        if not ridx_s.isdigit():
            continue
        pair = (int(ridx_s), int(idx_inst.extra.get("lane", 0)))
        if pair not in seen_set:
            seen_set.add(pair)
            slots_seen.append((pair[0], pair[1], rdesc))

    # Match obvious structural types first (Texture2DArray -> Detail Array, Flow Map -> 2xF32, Fluid -> 1xF32, etc.)
    by_hash = {tb.binding_name_hash: tb for tb in pt.texture_bindings if tb.resource_id}
    nrm_tb = next((tb for tb in pt.texture_bindings if "Normal" in " ".join(tb.target_channels.values()) and not tb.is_detail_texture), None)
    rma_tb = next((tb for tb in pt.texture_bindings if ("Roughness" in " ".join(tb.target_channels.values()) or "Reflectance" in " ".join(tb.target_channels.values()))), None)
    clr_tb = next((tb for tb in pt.texture_bindings if "Color" in " ".join(tb.target_channels.values()) and not tb.is_detail_texture), None)
    ems_tb = next((tb for tb in pt.texture_bindings if "Incandescence" in " ".join(tb.target_channels.values())), None)
    det_arr_tb = by_hash.get(0x342B11E0)
    flow_tb = by_hash.get(0x7C771B98)
    fluid_tb = by_hash.get(0x23FE87DB)
    eye_iris_tb = by_hash.get(0x5E969132)
    noise_tb = by_hash.get(0x7912A47E)

    used_tbs: set[int] = set()

    def _tb_entry(tb: Any, role_override: str = "") -> tuple[str, str, dict[str, Any]]:
        used_tbs.add(tb.index)
        ch = ", ".join(f"{k}={v}" for k, v in tb.target_channels.items())
        sub = f"{role_override or tb.semantic_role} • {tb.resource_id}" + (f" [{ch}]" if ch else "")
        return tb.name, sub, {
            "resource_id": tb.resource_id,
            "role": role_override or tb.semantic_role,
            "target_channels": tb.target_channels,
            "binding_hash": f"{tb.binding_name_hash:#010x}",
        }

    # First pass: match exact resource dimensions/roles
    for ridx, lane, rdesc in sorted(slots_seen):
        if "2DArray" in rdesc and det_arr_tb is not None and det_arr_tb.index not in used_tbs:
            hints[(ridx, lane)] = _tb_entry(det_arr_tb, "23-Slice Detail Array")
        elif "2xF32" in rdesc and flow_tb is not None and flow_tb.index not in used_tbs and ridx >= 1:
            hints[(ridx, lane)] = _tb_entry(flow_tb, "Elemental Flow Map (RG)")
        elif "<F32>" in rdesc and fluid_tb is not None and fluid_tb.index not in used_tbs and ridx >= 1:
            hints[(ridx, lane)] = _tb_entry(fluid_tb, "Elemental Fluid Caustics")
        elif "<F32>" in rdesc and noise_tb is not None and noise_tb.index not in used_tbs and ridx >= 1:
            hints[(ridx, lane)] = _tb_entry(noise_tb, "Noise / Dissolve Mask")
        elif "4xF32" in rdesc and eye_iris_tb is not None and eye_iris_tb.index not in used_tbs:
            hints[(ridx, lane)] = _tb_entry(eye_iris_tb, "Machine Eye Iris / POM Depth")
        elif "4xF32" in rdesc and nrm_tb is not None and nrm_tb.index not in used_tbs and ridx == 0:
            hints[(ridx, lane)] = _tb_entry(nrm_tb, "Normal + AO + Mask_Alpha")
        elif "3xF32" in rdesc and rma_tb is not None and rma_tb.index not in used_tbs and ridx == 0:
            hints[(ridx, lane)] = _tb_entry(rma_tb, "Mask + Smoothness + Reflectance")
        elif "3xF32" in rdesc and clr_tb is not None and clr_tb.index not in used_tbs:
            hints[(ridx, lane)] = _tb_entry(clr_tb, "Base Color (Albedo)")
        elif "3xF32" in rdesc and ems_tb is not None and ems_tb.index not in used_tbs:
            hints[(ridx, lane)] = _tb_entry(ems_tb, "Incandescence (Emissive)")

    # Second pass: pair remaining material TextureBindings in order
    remaining_tbs = [tb for tb in pt.texture_bindings if tb.index not in used_tbs and tb.resource_id]
    for ridx, lane, _rdesc in sorted(slots_seen):
        if (ridx, lane) not in hints and remaining_tbs:
            tb = remaining_tbs.pop(0)
            hints[(ridx, lane)] = _tb_entry(tb)

    return hints


def _detect_specialty_nodes(
    parsed: ParsedPS,
    mat: MaterialInfo,
    ps_ir: DXILAnalysis | None,
) -> list[dict[str, Any]]:
    """Detect real Decima Engine Specialty Nodes directly from compiled DXIL SSA instruction patterns!"""
    specs: list[dict[str, Any]] = []
    pt = mat.primary_technique
    tb_hashes = {tb.binding_name_hash for tb in (pt.texture_bindings if pt else ())}
    vb_by_hash = {vb.binding_name_hash: vb for vb in (pt.variable_bindings if pt else ())}

    # Helper checks across parsed instructions
    has_2d_array = any("2DArray" in (parsed.insts[v].extra.get("handle", {}).get("resource_desc", "")) for v in parsed.sample_vars)
    has_dot2 = any(i.op == "dot2" for i in parsed.insts.values())
    has_sqrt = any(i.op == "unary" and i.extra.get("name") == "Sqrt" for i in parsed.insts.values())
    has_sin = [i for i in parsed.insts.values() if i.op == "unary" and i.extra.get("name") in ("Sin", "Cos")]
    has_frc = [i for i in parsed.insts.values() if i.op == "unary" and i.extra.get("name") == "Frac"]
    has_fabs = [i for i in parsed.insts.values() if i.op == "unary" and i.extra.get("name") == "Abs"]
    has_rsqrt = any(i.op == "unary" and i.extra.get("name") == "Rsqrt" for i in parsed.insts.values())

    # Detect GameTime reads: Scratch_PerFrame (cb0,space1) reg 8 lane 3 (.w)
    gametime_vars: list[str] = []
    for inst in parsed.insts.values():
        if inst.op == "extract" and inst.extra.get("lane") == 3:
            cb = parsed.insts.get(inst.args[0])
            if cb and cb.op == "cbufLoad" and cb.extra.get("space") == 1 and str(cb.extra.get("reg_index")) == "8":
                gametime_vars.append(inst.var)

    # Detect Decima F0 = RFL^2 & metallic transition constant (1.11 / -0.111 -> 0x3FF1C28F6 / 0xBFBC6A7F)
    f0_lines: list[str] = []
    for inst in parsed.insts.values():
        if inst.op in ("fmul", "fadd"):
            for a in inst.args:
                fv = decode_llvm_float(a)
                if fv is not None and (abs(fv - 1.11) < 0.015 or abs(fv + 0.111) < 0.015):
                    f0_lines.append(f"L{inst.line_no}: {inst.raw}")
    has_worn_f0 = len(f0_lines) >= 2

    # Detect Octahedral Normal Encoding (`* 8.0` + `+ 8.0` + `Sqrt` + `+ 0.5`)
    oct_lines: list[str] = []
    for inst in parsed.insts.values():
        if inst.op == "fadd":
            for a in inst.args:
                fv = decode_llvm_float(a)
                if fv is not None and abs(fv - 8.0001) < 0.01:
                    oct_lines.append(f"L{inst.line_no}: {inst.raw}")
    has_oct_normal = bool(oct_lines) or (parsed.output_count >= 5 and has_rsqrt)

    # Detect RGB <-> HSL Hue Shift (sac subsurface rim: 6-sector `* 6.0`, `+ 2.0`, `+ 4.0` + `Frc`)
    hsl_lines: list[str] = []
    has_mul6 = False
    has_add2 = False
    has_add4 = False
    for inst in parsed.insts.values():
        if inst.op == "fmul" and any(decode_llvm_float(a) == 6.0 for a in inst.args):
            has_mul6 = True
            hsl_lines.append(f"L{inst.line_no}: {inst.raw}")
        elif inst.op == "fadd" and any(decode_llvm_float(a) == 2.0 for a in inst.args):
            has_add2 = True
        elif inst.op == "fadd" and any(decode_llvm_float(a) == 4.0 for a in inst.args):
            has_add4 = True
    has_hsl_shift = has_mul6 and has_add2 and has_add4 and len(has_frc) >= 1

    # Detect Dual-Phase Flow Map (two half-offset Frc phases + FAbs blend + Flow texture)
    has_dual_flow = (
        (0x7C771B98 in tb_hashes and 0x23FE87DB in tb_hashes)
        or (bool(gametime_vars) and len(has_frc) >= 2 and len(has_fabs) >= 1 and len(parsed.sample_vars) >= 5)
    )

    # Detect 23-Slice Detail Array Switch
    if (parsed.switch_max_cases >= 10 and has_2d_array) or 0x342B11E0 in tb_hashes:
        tile_val = 12.0
        if 0x3806EAE3 in vb_by_hash and 0.5 < vb_by_hash[0x3806EAE3].float4[0] <= 128.0:
            tile_val = vb_by_hash[0x3806EAE3].float4[0]
        specs.append({
            "id": "spec_detail_array",
            "kind": "DecimaSpecialty",
            "title": "Decima 23-Slice Detail Array",
            "subtitle": f"round(Mask×20) switch ({parsed.switch_max_cases or 21} cases) → 2DArray @ {tile_val:.0f}×",
            "stage": "detail",
            "inputs": ["UV0", "Mask", "Mask_Alpha", "DetailArray23"],
            "outputs": ["DetailColor", "DetailNormal", "DetailSmooth"],
            "targets": ["albedo", "normal", "roughness"],
            "formula": "slice = switch(round(Mask * 20)): 0→16, 11→21, 12→22, k→k; Detail = Sample(20:0, UV*Tile, slice)",
            "dxil_snippet": [
                "mat_id = Round_ne(Mask * 20.0)",
                "slice = switch(mat_id) [0->16, 1..10->1..10, 11->21, 12->22, 13..20->13..20]",
                "tex = Sample(Texture2DArray, UV * DetailScale, slice) -> R*2=Col, (A,G)=Normal, B*2=Smooth",
            ],
        })

    # Normal Z Reconstruct
    if has_dot2 and has_sqrt:
        nrm_snips = [f"L{i.line_no}: {i.raw}" for i in parsed.insts.values() if i.op == "dot2"][:3]
        specs.append({
            "id": "spec_nrm_z",
            "kind": "DecimaSpecialty",
            "title": "Decima Normal Z Reconstruct",
            "subtitle": "z = sqrt(sat(1 - dot2(xy*2-1)))",
            "stage": "normal_decode",
            "inputs": ["Normal.XY"],
            "outputs": ["TangentNormal"],
            "targets": ["normal"],
            "formula": "n.xy = tex.rg * 2.0 - 1.0; n.z = sqrt(saturate(1.0 - dot2(n.xy, n.xy)))",
            "dxil_snippet": nrm_snips,
        })

    # TBN Tangent->World Transform
    if any(i.op == "fmad" for i in parsed.insts.values()) and has_rsqrt:
        specs.append({
            "id": "spec_tbn",
            "kind": "DecimaSpecialty",
            "title": "Decima TBN Tangent → World",
            "subtitle": "Bitangent = cross(N, T) * sign; FMad basis",
            "stage": "normal_world",
            "inputs": ["TangentNormal", "DetailNormal", "VertexTBN"],
            "outputs": ["WorldNormal"],
            "targets": ["normal"],
            "formula": "B = cross(N, T) * (T.w > 0.25 ? -1 : 1); WorldN = normalize(T*n.x + B*n.y + N*n.z)",
            "dxil_snippet": [f"L{i.line_no}: {i.raw}" for i in parsed.insts.values() if i.op == "fmad"][:4],
        })

    # Dual-Phase Flow Map (Elemental Sacks!)
    if has_dual_flow:
        sac_col = vb_by_hash[0x249F1E74].float4[:3] if 0x249F1E74 in vb_by_hash else None
        col_txt = f"RGB({sac_col[0]:.2f}, {sac_col[1]:.2f}, {sac_col[2]:.2f})" if sac_col else "PerInstance GlowRGB"
        specs.append({
            "id": "spec_dual_flow",
            "kind": "DecimaSpecialty",
            "title": "Decima Dual-Phase Flow Map",
            "subtitle": f"Animated fluid sloshing (p0/p1 Frac + Abs Lerp) • {col_txt}",
            "stage": "emissive_fx",
            "inputs": ["UV0", "GameTime", "FlowMap", "FluidTex"],
            "outputs": ["FluidPattern", "SacMask"],
            "targets": ["emissive", "albedo"],
            "formula": "flow=(Flow(UV*2)*2-1)*str; p0=frac(t+0.5)*2-1; p1=frac(t+1)*2-1; fluid=lerp(Tex(UV+flow*p0), Tex(UV+flow*p1), |p0|)",
            "dxil_snippet": [f"L{i.line_no}: {i.raw}" for i in has_frc[:4]] + [f"L{i.line_no}: {i.raw}" for i in has_fabs[:2]],
        })

    # RGB <-> HSL Subsurface Rim Shift
    if has_hsl_shift:
        specs.append({
            "id": "spec_hsl_shift",
            "kind": "DecimaSpecialty",
            "title": "Decima RGB ↔ HSL Rim Shift",
            "subtitle": "Hue +0.03, Lightness -0.25 subsurface edge depth",
            "stage": "emissive_post",
            "inputs": ["SacColor", "FluidPattern", "Fresnel"],
            "outputs": ["SubsurfaceGlow"],
            "targets": ["emissive"],
            "formula": "HSL = rgb2hsl(SacColor); RimRGB = hsl2rgb(frac(H+0.03), S, sat(L-0.25)); Glow = lerp(SacColor, RimRGB, Fresnel*Fluid)",
            "dxil_snippet": hsl_lines[:5],
        })

    # Parallax Occlusion Mapping (POM) Raymarch (Machine Eyes & Cores)
    if (parsed.has_calc_lod and parsed.sample_level_count >= 3) or (0x5E969132 in tb_hashes):
        specs.append({
            "id": "spec_pom",
            "kind": "DecimaSpecialty",
            "title": "Decima Parallax Occlusion (POM)",
            "subtitle": f"{parsed.sample_level_count or 6}-step secant depth raymarch (Eye Optics)",
            "stage": "emissive_fx",
            "inputs": ["UV0", "ViewDir", "IrisDepthMap"],
            "outputs": ["ParallaxUV", "IrisMask"],
            "targets": ["emissive", "albedo"],
            "formula": "LOD = CalculateLOD(UV)+0.5; raymarch SampleLevel(UV + step*ViewTS.xy/(-ViewTS.z)) with secant refinement",
            "dxil_snippet": [
                f"L{parsed.insts[v].line_no}: {parsed.insts[v].raw}"
                for v in parsed.sample_vars
                if parsed.insts[v].extra.get("op_kind") == "sampleLevel"
            ][:4],
        })

    # GameTime Animator (Heartbeat / Pulsing LED / Scrolling Muscle Wave)
    if gametime_vars or (has_sin and len(has_sin) >= 1):
        sin_lines = [f"L{i.line_no}: {i.raw}" for i in has_sin[:3]]
        gt_note = f"Scratch_PerFrame[8].w ({', '.join(gametime_vars[:2])})" if gametime_vars else "Periodic Sin/Cos Wave"
        specs.append({
            "id": "spec_gametime",
            "kind": "DecimaSpecialty",
            "title": "Decima GameTime Animator",
            "subtitle": f"{gt_note} → {'Sin Pulse + ' if has_sin else ''}{'Frac Scroll' if has_frc else 'Wave'}",
            "stage": "anim",
            "inputs": ["GameTime"],
            "outputs": ["PulseWave", "ScrollPhase"],
            "targets": ["emissive"],
            "formula": "pulse = abs(sin(Time * freq)) or sin(Time * 8.0) * 0.3 + 0.4; scroll = frac(UV + Time * speed)",
            "dxil_snippet": sin_lines or [f"L{i.line_no}: {i.raw}" for i in has_frc[:3]],
        })

    # Decima Worn-Metal F0 = RFL^2 + Chroma Recovery
    if has_worn_f0:
        specs.append({
            "id": "spec_worn_f0",
            "kind": "DecimaSpecialty",
            "title": "Decima Worn-Metal F0 (RFL²)",
            "subtitle": "m = clamp(RFL²×1.11 - 0.111, 0, 0.98) • Chroma F0",
            "stage": "pbr_eval",
            "inputs": ["Color", "Reflectance"],
            "outputs": ["DiffuseAlbedo", "SpecularF0", "MetalMask"],
            "targets": ["albedo", "specular"],
            "formula": "F0_raw=RFL²; m=clamp(F0_raw*1.11-0.111,0,0.98); chroma=CLR/max(CLR); F0=lerp(min(F0_raw,0.1), chroma*F0_raw, m); Diff=(1-m)*CLR",
            "dxil_snippet": f0_lines[:4],
        })

    # Decima Smoothness -> Roughness & Detail Smoothness Modulation
    if any("Roughness" in " ".join(tb.target_channels.values()) for tb in (pt.texture_bindings if pt else ())):
        specs.append({
            "id": "spec_smoothness",
            "kind": "DecimaSpecialty",
            "title": "Decima Smoothness Convention",
            "subtitle": "RGH channel stores Smoothness (1 - Roughness)",
            "stage": "pbr_eval",
            "inputs": ["Smoothness", "DetailSmooth", "Wetness"],
            "outputs": ["FinalSmoothness"],
            "targets": ["roughness"],
            "formula": "SmoothOut = lerp(Smooth, Smooth * 2 * Detail.B, MaskAlpha); Wetness boosts Smoothness toward 0.92",
            "dxil_snippet": ["Decima G-Buffer SV_Target3.x stores Smoothness directly (inverted to 1-Smooth for Blender/UE5)."],
        })

    # Fresnel View-Facing (N·V)
    dot3_insts = [i for i in parsed.insts.values() if i.op == "dot3"]
    if len(dot3_insts) >= 2:
        specs.append({
            "id": "spec_fresnel",
            "kind": "DecimaSpecialty",
            "title": "Decima View Fresnel (N·V)",
            "subtitle": "Grazing rim boost & Specular horizon factor",
            "stage": "pbr_eval",
            "inputs": ["WorldNormal", "ViewDir"],
            "outputs": ["NdotV", "RimFactor"],
            "targets": ["emissive", "specular", "albedo"],
            "formula": "NdotV = saturate(-dot3(WorldNormal, ViewDir)); Rim = (1 - NdotV * 0.75)^2",
            "dxil_snippet": [f"L{i.line_no}: {i.raw}" for i in dot3_insts[:3]],
        })

    # Octahedral Normal Packing for G-Buffer RT2
    if has_oct_normal and parsed.output_count >= 5:
        specs.append({
            "id": "spec_oct_nrm",
            "kind": "DecimaSpecialty",
            "title": "Decima Octahedral Normal Pack",
            "subtitle": "Compresses 3D WorldNormal → 2D SV_Target2.xy",
            "stage": "normal_pack",
            "inputs": ["WorldNormal"],
            "outputs": ["PackedOctXY"],
            "targets": ["normal"],
            "formula": "denom = sqrt(WorldN.z * 8.0 + 8.0); SV_Target2.xy = WorldN.xy / denom + 0.5",
            "dxil_snippet": oct_lines[:3] or ["SV_Target2 stores stereographic/octahedral 2-channel normal to save G-Buffer bandwidth."],
        })

    # Alpha Cutout Discard
    if parsed.has_discard:
        specs.append({
            "id": "spec_discard",
            "kind": "DecimaSpecialty",
            "title": "Decima Alpha-Test Cutout",
            "subtitle": "@dx.op.discard (masked/dithered clip)",
            "stage": "pbr_eval",
            "inputs": ["Mask_Alpha"],
            "outputs": ["ClipMask"],
            "targets": ["albedo"],
            "formula": "if (alpha < threshold) discard;",
            "dxil_snippet": ["call void @dx.op.discard(i32 82, i1 true)"],
        })

    # Global Engine Weather / Status Overlay Hook
    if ps_ir and ps_ir.uses_global_space2_textures:
        specs.append({
            "id": "spec_weather",
            "kind": "DecimaSpecialty",
            "title": "Decima World Weather & Status",
            "subtitle": f"{len(ps_ir.uses_global_space2_textures)} global space2 LUTs (Rain/Snow/Focus)",
            "stage": "pbr_eval",
            "inputs": ["GlobalWeather", "WorldNormal"],
            "outputs": ["WetDarken", "WetSmooth"],
            "targets": ["albedo", "roughness"],
            "formula": "Reads global rain wetness, snow accumulation on +Z normals, and Aloy Focus scan highlight from space2.",
            "dxil_snippet": [f"Global space2 bindings: {', '.join(ps_ir.uses_global_space2_textures[:5])}"],
        })

    return specs


# ---------------------------------------------------------------------------
# True SSA Math Node Extraction (per-output backward trace with vector folding)
# ---------------------------------------------------------------------------

_OP_DISPLAY: dict[str, tuple[str, str, str]] = {
    "fmul": ("Multiply", "math", "A × B"),
    "fadd": ("Add", "math", "A + B"),
    "fsub": ("Subtract", "math", "A − B"),
    "fdiv": ("Divide", "math", "A ÷ B"),
    "fmad": ("FMad (A×B+C)", "math", "A × B + C"),
    "dot2": ("Dot2", "math", "dot(A.xy, B.xy)"),
    "dot3": ("Dot3", "math", "dot(A.xyz, B.xyz)"),
    "select": ("Select / Branch", "blend", "Cond ? A : B"),
    "fcmp": ("Compare", "math", "A cmp B"),
}


def _classify_math_node(inst: SSAInst, parsed: ParsedPS) -> tuple[str, str, str, list[float]]:
    """Return `(title, category, formula, decoded_constants)` for an SSA instruction."""
    consts: list[float] = []
    for a in inst.args:
        fv = decode_llvm_float(a)
        if fv is not None:
            consts.append(fv)

    if inst.op == "unary":
        uname = inst.extra.get("name", "Unary")
        return f"{uname}", "math", f"{uname}({fmt_operand(inst.args[0])})", consts
    if inst.op == "binary":
        bname = inst.extra.get("name", "Bin")
        return f"{bname}", "math", f"{bname}({fmt_operand(inst.args[0])}, {fmt_operand(inst.args[1])})", consts
    if inst.op == "fmul":
        a0, a1 = inst.args[0], inst.args[1]
        if a0 == a1:
            return "Square (x²)", "math", f"{fmt_operand(a0)}²", consts
        if consts:
            return f"Multiply (×{fmt_float(consts[0])})", "math", f"{fmt_operand(a0)} × {fmt_operand(a1)}", consts
        return "Multiply", "math", f"{fmt_operand(a0)} × {fmt_operand(a1)}", consts
    if inst.op == "fadd":
        a0, a1 = inst.args[0], inst.args[1]
        # Detect Lerp pattern: `a + (b - a) * t` -> `fadd(fmul(fsub(b, a), t), a)`!
        for mul_side, base_side in ((a0, a1), (a1, a0)):
            mi = parsed.insts.get(mul_side)
            if mi and mi.op == "fmul":
                for sub_cand in mi.args:
                    si = parsed.insts.get(sub_cand)
                    if si and si.op == "fsub" and base_side in si.args:
                        other_t = mi.args[1] if mi.args[0] == sub_cand else mi.args[0]
                        return "Lerp / Mix", "blend", f"lerp({fmt_operand(base_side)}, {fmt_operand(si.args[0])}, {fmt_operand(other_t)})", consts
        if consts:
            c = consts[0]
            sign = "+" if c >= 0 else "−"
            return f"Add ({sign}{fmt_float(abs(c))})", "math", f"{fmt_operand(a0)} + {fmt_operand(a1)}", consts
        return "Add", "math", f"{fmt_operand(a0)} + {fmt_operand(a1)}", consts
    if inst.op == "fsub":
        a0, a1 = inst.args[0], inst.args[1]
        if decode_llvm_float(a0) == 1.0:
            return "OneMinus (1 − x)", "math", f"1.0 − {fmt_operand(a1)}", consts
        if decode_llvm_float(a0) == 0.0:
            return "Negate (−x)", "math", f"−{fmt_operand(a1)}", consts
        return "Subtract", "math", f"{fmt_operand(a0)} − {fmt_operand(a1)}", consts
    if inst.op == "fdiv":
        a0, a1 = inst.args[0], inst.args[1]
        if decode_llvm_float(a0) == 1.0:
            return "Reciprocal (1 / x)", "math", f"1.0 / {fmt_operand(a1)}", consts
        return "Divide", "math", f"{fmt_operand(a0)} ÷ {fmt_operand(a1)}", consts
    if inst.op == "fmad":
        return "FMad (A×B+C)", "math", f"({fmt_operand(inst.args[0])} × {fmt_operand(inst.args[1])}) + {fmt_operand(inst.args[2])}", consts
    if inst.op == "select":
        return "Select / Branch", "blend", f"({inst.args[0]}) ? {fmt_operand(inst.args[1])} : {fmt_operand(inst.args[2])}", consts
    if inst.op == "fcmp":
        pred = inst.extra.get("pred", "cmp")
        return f"Compare ({pred})", "math", f"{fmt_operand(inst.args[0])} {pred} {fmt_operand(inst.args[1])}", consts

    title, cat, form = _OP_DISPLAY.get(inst.op, (inst.op, "math", inst.raw[:48]))
    return title, cat, form, consts


def _extract_top_math_nodes_per_target(
    parsed: ParsedPS,
    max_per_target: int = 8,
) -> dict[str, list[dict[str, Any]]]:
    """Walk backwards from each SV_Target output to extract genuine, representative DXIL math nodes (deduped RGB triplets!)."""
    by_target_group: dict[str, list[dict[str, Any]]] = {}
    is_deferred = parsed.output_count >= 4
    tmeta = _DEFERRED_TARGET_META if is_deferred else _FORWARD_TARGET_META

    stores_by_tid: dict[int, list[tuple[int, str, int]]] = {}
    for tid, col, vtok, lno in parsed.stores:
        stores_by_tid.setdefault(tid, []).append((col, vtok, lno))

    for tid, cols in sorted(stores_by_tid.items()):
        _, _, tgroup = tmeta.get(tid, (f"SV_Target{tid}", "", "misc"))
        if tgroup == "motion":
            continue  # Skip boilerplate screen-space velocity matrix math to keep graph focused on shading!

        # BFS backwards from output lanes, preferring non-boilerplate shading math closest to output & texture samples
        visited: set[str] = set()
        frontier: list[tuple[str, int]] = [(vtok, 0) for _c, vtok, _l in cols if vtok.startswith("%")]
        candidates: list[tuple[int, SSAInst, str, str, str, list[float]]] = []
        seen_signatures: set[tuple[str, str]] = set()

        while frontier and len(visited) < 280:
            v, depth = frontier.pop(0)
            if v in visited or depth > 14:
                continue
            visited.add(v)
            inst = parsed.insts.get(v)
            if inst is None:
                continue

            if inst.op in ("fmul", "fadd", "fsub", "fdiv", "fmad", "unary", "binary", "dot2", "dot3", "select"):
                title, cat, formula, consts = _classify_math_node(inst, parsed)
                # Fold identical RGB/XYZ lane repeats (e.g. 3× Multiply by same constant/exposure -> 1 clean RGB node)
                const_sig = ",".join(fmt_float(c) for c in consts[:2])
                sig = (title, const_sig)
                # Skip trivial identity / projection-matrix boilerplate
                is_trivial = (
                    (inst.op == "fadd" and consts and abs(consts[0]) < 1e-6)
                    or (inst.op == "fmul" and consts and abs(consts[0] - 1.0) < 1e-6)
                )
                if not is_trivial and sig not in seen_signatures:
                    seen_signatures.add(sig)
                    # Score interesting shading ops higher (Lerp, Sin, Frac, Saturate, Square, Dot3, Select, const Multiply/Add)
                    prio = 0
                    if title == "Lerp / Mix":
                        prio = 100
                    elif inst.op == "unary" and inst.extra.get("name") in ("Sin", "Cos", "Frac", "Saturate", "Sqrt", "Pow", "Exp2"):
                        prio = 92
                    elif inst.op == "select":
                        prio = 85
                    elif title in ("Square (x²)", "OneMinus (1 − x)"):
                        prio = 82
                    elif inst.op in ("dot3", "dot2", "binary"):
                        prio = 78
                    elif consts:
                        prio = 70
                    elif inst.op in ("fmul", "fadd", "fsub", "fdiv"):
                        prio = 55
                    else:
                        prio = 40
                    # Prefer nodes near the output or mid-pipeline
                    prio -= depth * 2
                    candidates.append((prio, inst, title, cat, formula, consts))

            for a in inst.args:
                if a.startswith("%") and a not in visited:
                    frontier.append((a, depth + 1))

        candidates.sort(key=lambda x: (-x[0], x[1].line_no))
        chosen: list[dict[str, Any]] = []
        for _prio, inst, title, cat, formula, consts in candidates[:max_per_target]:
            # Sort in pipeline order (by line_no) below
            chosen.append({
                "var": inst.var,
                "line_no": inst.line_no,
                "op": inst.op,
                "title": title,
                "category": cat,
                "formula": formula,
                "constants": [round(c, 5) for c in consts],
                "dxil": f"L{inst.line_no}: {inst.raw}",
                "tgroup": tgroup,
                "tid": tid,
            })
        chosen.sort(key=lambda x: x["line_no"])
        if chosen:
            by_target_group.setdefault(tgroup, []).extend(chosen)

    return by_target_group


# ---------------------------------------------------------------------------
# Main Public Graph Builder
# ---------------------------------------------------------------------------


def build_detailed_dxil_graph(
    mat: MaterialInfo,
    ps_prog: DisassembledProgram | None,
    ps_ir: DXILAnalysis | None,
    binding_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build a detailed, interactive DXIL SSA Math & Decima Specialty Node Graph from live Pixel Shader IR."""
    if ps_prog is None or not ps_prog.disassembly:
        return {
            "mode": "decima_detailed",
            "title": f"Detailed DXIL Math & Specialty Graph — {mat.name} (No PS DXIL)",
            "nodes": [],
            "edges": [],
            "stats": {"specialty_nodes": 0, "math_nodes": 0, "texture_samplers": 0, "cbuffer_params": 0},
        }

    parsed = _parse_pixel_program(ps_prog)
    pt = mat.primary_technique
    bindless_hints = _build_bindless_slot_hints(mat, parsed)
    specialties = _detect_specialty_nodes(parsed, mat, ps_ir)
    math_by_target = _extract_top_math_nodes_per_target(parsed, max_per_target=6)

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    node_ids: set[str] = set()

    def add_node(
        nid: str,
        kind: str,
        title: str,
        col: int,
        row: int,
        category: str,
        subtitle: str = "",
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        if nid in node_ids:
            return
        node_ids.add(nid)
        nodes.append({
            "id": nid,
            "kind": kind,
            "title": title,
            "subtitle": subtitle,
            "col": col,
            "row": row,
            "category": category,  # texture | param | math | blend | specialty | global | output
            "inputs": inputs or [],
            "outputs": outputs or [],
            "meta": meta or {},
        })

    def add_edge(src: str, src_port: str, dst: str, dst_port: str, label: str = "", color: str = "#7dd3fc") -> None:
        if src in node_ids and dst in node_ids:
            edges.append({
                "from": src,
                "from_port": src_port,
                "to": dst,
                "to_port": dst_port,
                "label": label,
                "color": color,
            })

    # -----------------------------------------------------------------------
    # Column 0: Mesh Stage Inputs + GameTime + CBuffer Material Parameters
    # -----------------------------------------------------------------------
    c0_row = 0
    add_node(
        "in_uv0",
        "VertexInput",
        "UV0 / Vertex TBN",
        col=0,
        row=c0_row,
        category="param",
        subtitle=f"PS Inputs: {len(ps_prog.inputs)} STAGE_IO streams",
        outputs=["UV0", "VertexTBN", "ViewDir"],
        meta={
            "targets": ["albedo", "normal", "roughness", "specular", "emissive"],
            "formula": "Interpolated UV0, World Normal, Tangent (+handness), and Camera View Vector from Vertex Shader.",
            "dxil_snippet": [f"{i.name}{i.index} ({i.mask}, reg {i.register})" for i in ps_prog.inputs[:8]],
        },
    )
    c0_row += 1

    has_gametime = any(s["id"] in ("spec_gametime", "spec_dual_flow") for s in specialties)
    if has_gametime:
        add_node(
            "in_gametime",
            "GlobalEngine",
            "Global GameTime (sec)",
            col=0,
            row=c0_row,
            category="global",
            subtitle="Scratch_PerFrame cb0,space1[8].w",
            outputs=["GameTime"],
            meta={
                "targets": ["emissive", "albedo"],
                "formula": "Engine monotonic time in seconds (drives flow maps, muscle pulses, canister blinks & heartbeats).",
                "dxil_snippet": ["%t = cbufferLoadLegacy(Scratch_PerFrame cb0,space1, reg=8).w"],
            },
        )
        c0_row += 1

    # Add up to 5 most meaningful non-trivial Material CBuffer Variables (colors, detail tile, intensities)
    cbuf_param_ids: list[tuple[str, str]] = []
    if pt:
        interesting_vars = []
        for vb in pt.variable_bindings:
            h = vb.binding_name_hash
            f4 = vb.float4
            if h == 0x3806EAE3 and 0.5 < f4[0] <= 128.0:
                interesting_vars.append((0, f"Detail UV Scale ({f4[0]:.0f}×)", f"0x{h:08x} = ({f4[0]:.1f}, {f4[1]:.1f})", ["normal", "albedo"], "detail_scale", f4))
            elif h == 0x249F1E74:
                interesting_vars.append((1, "Elemental Sac Glow RGB", f"0x{h:08x} = ({f4[0]:.2f}, {f4[1]:.2f}, {f4[2]:.2f})", ["emissive"], "sac_color", f4))
            elif h == 0x0E470C22 and max(f4[:3]) > 0.12:
                interesting_vars.append((1, "Emissive Tint RGB", f"0x{h:08x} = ({f4[0]:.2f}, {f4[1]:.2f}, {f4[2]:.2f})", ["emissive"], "ems_tint", f4))
            elif h == 0x5D2FABB8 and max(f4[:3]) > 0.05:
                interesting_vars.append((1, "Core Glow Tint RGB", f"0x{h:08x} = ({f4[0]:.2f}, {f4[1]:.2f}, {f4[2]:.2f})", ["emissive"], "core_tint", f4))
            elif h == 0x01943DD3:
                interesting_vars.append((2, "Glass Shell Tint RGBA", f"0x{h:08x} = ({f4[0]:.2f}, {f4[1]:.2f}, {f4[2]:.2f})", ["albedo"], "glass_tint", f4))
            elif vb.inferred_name and max(abs(x) for x in f4) > 0.001:
                interesting_vars.append((3, vb.inferred_name[:28], f"0x{h:08x} = ({f4[0]:.2f}, {f4[1]:.2f}, {f4[2]:.2f})", ["albedo", "emissive"], "var", f4))
        interesting_vars.sort(key=lambda x: x[0])
        for vi, (_pr, vtitle, vsub, vtargs, vrole, f4) in enumerate(interesting_vars[:5]):
            vid = f"param_v{vi}"
            # Compute optional inline RGB color swatch for color-like Float4 parameters
            swatch_css = ""
            r_v, g_v, b_v = float(f4[0]), float(f4[1]), float(f4[2])
            if 0.0 <= min(r_v, g_v, b_v) and max(r_v, g_v, b_v) <= 2.0 and (
                "color" in vrole.lower()
                or "tint" in vrole.lower()
                or "glow" in vrole.lower()
                or (max(r_v, g_v, b_v) > 0.02 and (abs(r_v - g_v) > 0.03 or abs(g_v - b_v) > 0.03))
            ):
                sr = min(255, max(0, int(round((min(1.0, max(0.0, r_v)) ** (1.0 / 2.2)) * 255))))
                sg = min(255, max(0, int(round((min(1.0, max(0.0, g_v)) ** (1.0 / 2.2)) * 255))))
                sb = min(255, max(0, int(round((min(1.0, max(0.0, b_v)) ** (1.0 / 2.2)) * 255))))
                swatch_css = f"rgb({sr},{sg},{sb})"
            add_node(
                vid,
                "CBufferParam",
                vtitle,
                col=0,
                row=c0_row,
                category="param",
                subtitle=vsub,
                outputs=["Val"],
                meta={
                    "targets": vtargs,
                    "role": vrole,
                    "float4": [round(float(x), 4) for x in f4],
                    "color_swatch": swatch_css,
                    "formula": f"PerInstance CBuffer constant: ({f4[0]:.3f}, {f4[1]:.3f}, {f4[2]:.3f}, {f4[3]:.3f})",
                    "dxil_snippet": [f"ShaderInstance_PerInstance cb0,space8 -> {vsub}"],
                },
            )
            cbuf_param_ids.append((vid, vrole))
            c0_row += 1

    if ps_ir and ps_ir.uses_global_space2_textures:
        add_node(
            "in_weather",
            "GlobalEngine",
            "Global Weather / Space2",
            col=0,
            row=c0_row,
            category="global",
            subtitle=f"{len(ps_ir.uses_global_space2_textures)} global LUTs",
            outputs=["GlobalWeather"],
            meta={
                "targets": ["albedo", "roughness"],
                "formula": "Engine global space2 SRVs (Wetness, Snow, Precipitation Occlusion, Focus Scan).",
                "dxil_snippet": ps_ir.uses_global_space2_textures[:8],
            },
        )
        c0_row += 1

    # -----------------------------------------------------------------------
    # Column 1: Actual Disassembled DXIL Texture Samplers
    # -----------------------------------------------------------------------
    tex_nodes_by_role: dict[str, tuple[str, str]] = {}
    seen_tex_keys: dict[str, str] = {}
    c1_row = 0

    for sv in parsed.sample_vars:
        sinst = parsed.insts[sv]
        tkey, ttitle, tsub, tmeta = _describe_texture_sample(sinst, parsed, mat, binding_map, bindless_hints)
        if tmeta.get("is_global"):
            continue  # Global space2 LUTs represented cleanly by `in_weather`
        if tkey in seen_tex_keys:
            continue
        nid = f"samp_{len(seen_tex_keys)}"
        seen_tex_keys[tkey] = nid

        ch_dict = tmeta.get("target_channels") or {}
        if ch_dict:
            out_ports = [f"{lane}:{spec}" for lane, spec in ch_dict.items()]
        elif "2DArray" in tsub or "23-Slice" in tsub:
            out_ports = ["R:Col", "AG:Nrm", "B:Smooth"]
        elif "Flow Map" in tsub:
            out_ports = ["RG:Flow"]
        elif "Fluid" in tsub or "Noise" in tsub:
            out_ports = ["R:Pattern"]
        elif "Iris" in tsub:
            out_ports = ["R:IrisDepth"]
        else:
            out_ports = ["RGBA"]

        # Determine which outputs this sampler feeds
        t_targets = []
        low_meta = f"{ttitle} {tsub} {' '.join(ch_dict.values())}".lower()
        if "color" in low_meta or "albedo" in low_meta or "23-slice" in low_meta:
            t_targets.append("albedo")
        if "normal" in low_meta or "23-slice" in low_meta:
            t_targets.append("normal")
        if "roughness" in low_meta or "smooth" in low_meta or "23-slice" in low_meta:
            t_targets.append("roughness")
        if "reflectance" in low_meta:
            t_targets.append("specular")
        if any(k in low_meta for k in ("flow", "fluid", "incandescence", "emissive", "iris", "noise", "mask", "ramp", "pulse", "braid", "vein", "core")):
            t_targets.append("emissive")
        if not t_targets:
            t_targets = ["albedo", "emissive"]

        add_node(
            nid,
            "TextureSample",
            ttitle,
            col=1,
            row=c1_row,
            category="texture",
            subtitle=tsub,
            inputs=["UV"],
            outputs=out_ports,
            meta={
                **tmeta,
                "targets": sorted(set(t_targets)),
                "ssa_var": sv,
                "preview_url": f"/api/texture/{tmeta['resource_id']}/image?max_dim=96" if tmeta.get("resource_id") else "",
                "dxil_snippet": [f"L{sinst.line_no}: {sinst.raw}"],
                "formula": f"GPU {sinst.extra.get('op_kind', 'sample')} on {ttitle}",
            },
        )
        add_edge("in_uv0", "UV0", nid, "UV", color="#a78bfa")
        c1_row += 1

        # Track semantic handles for connecting downstream specialty & math nodes
        if "23-slice" in low_meta or "2darray" in low_meta:
            tex_nodes_by_role.setdefault("detail_array", (nid, out_ports[0]))
        elif "flow" in low_meta:
            tex_nodes_by_role.setdefault("flow", (nid, out_ports[0]))
        elif "fluid" in low_meta or "noise" in low_meta:
            tex_nodes_by_role.setdefault("fluid", (nid, out_ports[0]))
        elif "iris" in low_meta:
            tex_nodes_by_role.setdefault("iris", (nid, out_ports[0]))
        if "color" in low_meta and "23-slice" not in low_meta:
            tex_nodes_by_role.setdefault("color", (nid, out_ports[0]))
        if "normal" in low_meta and "23-slice" not in low_meta:
            p_nrm = next((p for p in out_ports if "Normal" in p), out_ports[0])
            tex_nodes_by_role.setdefault("normal", (nid, p_nrm))
        if "mask_alpha" in low_meta:
            p_ma = next((p for p in out_ports if "Mask_Alpha" in p), out_ports[-1])
            tex_nodes_by_role.setdefault("mask_alpha", (nid, p_ma))
        if "mask" in low_meta and "mask_alpha" not in low_meta:
            p_m = next((p for p in out_ports if "Mask" in p), out_ports[0])
            tex_nodes_by_role.setdefault("mask", (nid, p_m))
        if "roughness" in low_meta:
            p_r = next((p for p in out_ports if "Roughness" in p), out_ports[0])
            tex_nodes_by_role.setdefault("smoothness", (nid, p_r))
        if "reflectance" in low_meta:
            p_rf = next((p for p in out_ports if "Reflectance" in p), out_ports[0])
            tex_nodes_by_role.setdefault("reflectance", (nid, p_rf))
        if "incandescence" in low_meta or "emissive" in low_meta:
            tex_nodes_by_role.setdefault("emissive", (nid, out_ports[0]))
        if c1_row >= 12:
            break

    # -----------------------------------------------------------------------
    # Columns 2 & 3: Decima Specialty Nodes (placed by pipeline stage!)
    # -----------------------------------------------------------------------
    stage_col_map = {
        "anim": 2,
        "detail": 2,
        "normal_decode": 2,
        "emissive_fx": 2,
        "normal_world": 3,
        "pbr_eval": 3,
        "emissive_post": 3,
        "normal_pack": 4,
    }
    col_rows: dict[int, int] = {2: 0, 3: 0, 4: 0, 5: 0}

    spec_by_id: dict[str, dict[str, Any]] = {}
    for sp in specialties:
        c = stage_col_map.get(sp["stage"], 2)
        r = col_rows.get(c, 0)
        col_rows[c] = r + 1
        spec_by_id[sp["id"]] = sp
        add_node(
            sp["id"],
            "DecimaSpecialty",
            sp["title"],
            col=c,
            row=r,
            category="specialty",
            subtitle=sp["subtitle"],
            inputs=sp["inputs"],
            outputs=sp["outputs"],
            meta={
                "targets": sp["targets"],
                "formula": sp["formula"],
                "dxil_snippet": sp["dxil_snippet"],
                "is_specialty": True,
            },
        )

    # Wire Column 0/1 -> Specialty Nodes
    if "spec_detail_array" in spec_by_id:
        add_edge("in_uv0", "UV0", "spec_detail_array", "UV0", color="#a78bfa")
        if "mask" in tex_nodes_by_role:
            add_edge(*tex_nodes_by_role["mask"], "spec_detail_array", "Mask", color="#facc15")
        if "mask_alpha" in tex_nodes_by_role:
            add_edge(*tex_nodes_by_role["mask_alpha"], "spec_detail_array", "Mask_Alpha", color="#facc15")
        if "detail_array" in tex_nodes_by_role:
            add_edge(*tex_nodes_by_role["detail_array"], "spec_detail_array", "DetailArray23", color="#22d3ee")

    if "spec_nrm_z" in spec_by_id and "normal" in tex_nodes_by_role:
        add_edge(*tex_nodes_by_role["normal"], "spec_nrm_z", "Normal.XY", color="#818cf8")

    if "spec_tbn" in spec_by_id:
        if "spec_nrm_z" in spec_by_id:
            add_edge("spec_nrm_z", "TangentNormal", "spec_tbn", "TangentNormal", color="#818cf8")
        if "spec_detail_array" in spec_by_id:
            add_edge("spec_detail_array", "DetailNormal", "spec_tbn", "DetailNormal", color="#e879f9")
        add_edge("in_uv0", "VertexTBN", "spec_tbn", "VertexTBN", color="#a78bfa")

    if "spec_oct_nrm" in spec_by_id and "spec_tbn" in spec_by_id:
        add_edge("spec_tbn", "WorldNormal", "spec_oct_nrm", "WorldNormal", color="#818cf8")

    if "spec_fresnel" in spec_by_id:
        if "spec_tbn" in spec_by_id:
            add_edge("spec_tbn", "WorldNormal", "spec_fresnel", "WorldNormal", color="#818cf8")
        add_edge("in_uv0", "ViewDir", "spec_fresnel", "ViewDir", color="#a78bfa")

    if "spec_gametime" in spec_by_id and "in_gametime" in node_ids:
        add_edge("in_gametime", "GameTime", "spec_gametime", "GameTime", color="#2dd4bf")

    if "spec_dual_flow" in spec_by_id:
        add_edge("in_uv0", "UV0", "spec_dual_flow", "UV0", color="#a78bfa")
        if "in_gametime" in node_ids:
            add_edge("in_gametime", "GameTime", "spec_dual_flow", "GameTime", color="#2dd4bf")
        if "flow" in tex_nodes_by_role:
            add_edge(*tex_nodes_by_role["flow"], "spec_dual_flow", "FlowMap", color="#22d3ee")
        if "fluid" in tex_nodes_by_role:
            add_edge(*tex_nodes_by_role["fluid"], "spec_dual_flow", "FluidTex", color="#22d3ee")

    if "spec_hsl_shift" in spec_by_id:
        for vid, vrole in cbuf_param_ids:
            if vrole == "sac_color":
                add_edge(vid, "Val", "spec_hsl_shift", "SacColor", color="#f59e0b")
        if "spec_dual_flow" in spec_by_id:
            add_edge("spec_dual_flow", "FluidPattern", "spec_hsl_shift", "FluidPattern", color="#e879f9")
        if "spec_fresnel" in spec_by_id:
            add_edge("spec_fresnel", "RimFactor", "spec_hsl_shift", "Fresnel", color="#e879f9")

    if "spec_pom" in spec_by_id:
        add_edge("in_uv0", "UV0", "spec_pom", "UV0", color="#a78bfa")
        add_edge("in_uv0", "ViewDir", "spec_pom", "ViewDir", color="#a78bfa")
        if "iris" in tex_nodes_by_role:
            add_edge(*tex_nodes_by_role["iris"], "spec_pom", "IrisDepthMap", color="#22d3ee")

    if "spec_worn_f0" in spec_by_id:
        if "color" in tex_nodes_by_role:
            add_edge(*tex_nodes_by_role["color"], "spec_worn_f0", "Color", color="#f87171")
        if "reflectance" in tex_nodes_by_role:
            add_edge(*tex_nodes_by_role["reflectance"], "spec_worn_f0", "Reflectance", color="#60a5fa")

    if "spec_smoothness" in spec_by_id:
        if "smoothness" in tex_nodes_by_role:
            add_edge(*tex_nodes_by_role["smoothness"], "spec_smoothness", "Smoothness", color="#34d399")
        if "spec_detail_array" in spec_by_id:
            add_edge("spec_detail_array", "DetailSmooth", "spec_smoothness", "DetailSmooth", color="#e879f9")
        if "spec_weather" in spec_by_id:
            add_edge("spec_weather", "WetSmooth", "spec_smoothness", "Wetness", color="#2dd4bf")

    if "spec_weather" in spec_by_id:
        if "in_weather" in node_ids:
            add_edge("in_weather", "GlobalWeather", "spec_weather", "GlobalWeather", color="#2dd4bf")
        if "spec_tbn" in spec_by_id:
            add_edge("spec_tbn", "WorldNormal", "spec_weather", "WorldNormal", color="#818cf8")

    # -----------------------------------------------------------------------
    # Columns 4 & 5: Lifted DXIL SSA Math Chains (Multiply, Add, Lerp, Saturate, Sin, etc.)
    # -----------------------------------------------------------------------
    # Group math chains cleanly by output target (albedo, emissive, normal, roughness, specular)
    target_anchor_src: dict[str, tuple[str, str, str]] = {}
    if "spec_worn_f0" in spec_by_id:
        target_anchor_src["albedo"] = ("spec_worn_f0", "DiffuseAlbedo", "#f87171")
        target_anchor_src["specular"] = ("spec_worn_f0", "SpecularF0", "#60a5fa")
    elif "color" in tex_nodes_by_role:
        target_anchor_src["albedo"] = (*tex_nodes_by_role["color"], "#f87171")
    if "spec_hsl_shift" in spec_by_id:
        target_anchor_src["emissive"] = ("spec_hsl_shift", "SubsurfaceGlow", "#fb923c")
    elif "spec_dual_flow" in spec_by_id:
        target_anchor_src["emissive"] = ("spec_dual_flow", "FluidPattern", "#fb923c")
    elif "spec_pom" in spec_by_id:
        target_anchor_src["emissive"] = ("spec_pom", "IrisMask", "#fb923c")
    elif "spec_gametime" in spec_by_id:
        target_anchor_src["emissive"] = ("spec_gametime", "PulseWave", "#fb923c")
    elif "emissive" in tex_nodes_by_role:
        target_anchor_src["emissive"] = (*tex_nodes_by_role["emissive"], "#fb923c")
    if "spec_oct_nrm" in spec_by_id:
        target_anchor_src["normal"] = ("spec_oct_nrm", "PackedOctXY", "#818cf8")
    elif "spec_tbn" in spec_by_id:
        target_anchor_src["normal"] = ("spec_tbn", "WorldNormal", "#818cf8")
    if "spec_smoothness" in spec_by_id:
        target_anchor_src["roughness"] = ("spec_smoothness", "FinalSmoothness", "#34d399")
    elif "smoothness" in tex_nodes_by_role:
        target_anchor_src["roughness"] = (*tex_nodes_by_role["smoothness"], "#34d399")

    tgroup_colors = {
        "albedo": "#f87171",
        "emissive": "#fb923c",
        "normal": "#818cf8",
        "roughness": "#34d399",
        "specular": "#60a5fa",
        "misc": "#94a3b8",
    }

    math_nodes_count = 0
    final_chain_out: dict[str, tuple[str, str]] = {}
    for tg, items in math_by_target.items():
        # Split each target's top lifted DXIL math steps across Columns 4 and 5 (max 2 columns deep so layout stays wide & clean!)
        chain = items[:6]
        if not chain:
            continue
        prev = target_anchor_src.get(tg)
        col_color = tgroup_colors.get(tg, "#7dd3fc")
        mid_idx = max(1, len(chain) // 2)
        for idx_m, m in enumerate(chain):
            mcol = 4 if idx_m < mid_idx else 5
            mrow = col_rows.get(mcol, 0)
            col_rows[mcol] = mrow + 1
            mnid = f"math_{tg}_{idx_m}"
            in_ports = ["A", "B"] if m["op"] in ("fmul", "fadd", "fsub", "fdiv", "binary", "dot2", "dot3") else (["A", "B", "C"] if m["op"] in ("fmad", "select") else ["In"])
            add_node(
                mnid,
                f"DXIL_{m['op'].upper()}",
                m["title"],
                col=mcol,
                row=mrow,
                category=m["category"],
                subtitle=f"{m['var']} • {m['formula'][:36]}",
                inputs=in_ports,
                outputs=["Out"],
                meta={
                    "targets": [tg],
                    "ssa_var": m["var"],
                    "line_no": m["line_no"],
                    "formula": m["formula"],
                    "constants": m["constants"],
                    "dxil_snippet": [m["dxil"]],
                },
            )
            math_nodes_count += 1
            if prev is not None:
                add_edge(prev[0], prev[1], mnid, in_ports[0], color=col_color)
            prev = (mnid, "Out", col_color)
        if prev is not None:
            final_chain_out[tg] = (prev[0], prev[1])

    # -----------------------------------------------------------------------
    # Column 6: Final SV_Target0..6 G-Buffer / Forward Outputs
    # -----------------------------------------------------------------------
    out_col = 6
    is_deferred = parsed.output_count >= 4
    tmeta = _DEFERRED_TARGET_META if is_deferred else _FORWARD_TARGET_META
    store_tids = sorted({tid for tid, _col, _tok, _lno in parsed.stores})
    if not store_tids:
        store_tids = [0, 1, 2]

    for row_o, tid in enumerate(store_tids):
        otitle, osub, tgroup = tmeta.get(tid, (f"SV_Target{tid}", f"Render Target {tid}", "misc"))
        lanes = sorted({col for t, col, _tok, _lno in parsed.stores if t == tid})
        lane_str = "".join("xyzw"[c] for c in lanes if 0 <= c <= 3) or "xyzw"
        out_id = f"out_rt{tid}"
        add_node(
            out_id,
            "GBufferOutput" if is_deferred else "ForwardOutput",
            otitle,
            col=out_col,
            row=row_o,
            category="output",
            subtitle=f"{osub} (.{lane_str})",
            inputs=[tgroup.capitalize()],
            meta={
                "targets": [tgroup],
                "target_id": tid,
                "lanes": lane_str,
                "formula": f"Writes final shaded channels (.{lane_str}) to GPU Render Target SV_Target{tid}.",
                "dxil_snippet": [
                    f"L{lno}: storeOutput(SV_Target{tid}.{'xyzw'[c]}, {vtok})"
                    for t, c, vtok, lno in parsed.stores
                    if t == tid
                ][:4],
            },
        )
        src_pair = final_chain_out.get(tgroup) or (
            (target_anchor_src[tgroup][0], target_anchor_src[tgroup][1]) if tgroup in target_anchor_src else None
        )
        if src_pair is not None:
            add_edge(src_pair[0], src_pair[1], out_id, tgroup.capitalize(), color=tgroup_colors.get(tgroup, "#f87171"))

    return {
        "mode": "decima_detailed",
        "title": f"Detailed DXIL Math & Decima Specialty Graph — {mat.name} ({ps_prog.shader_model}, {ps_ir.total_instructions if ps_ir else len(parsed.order)} IR ops)",
        "nodes": nodes,
        "edges": edges,
        "stage_frames": [
            {"cols": [0], "title": "Inputs, UVs & CBuffer Params", "color": "#f59e0b"},
            {"cols": [1], "title": "DXIL Texture Samplers", "color": "#22d3ee"},
            {"cols": [2, 3], "title": "Decima Engine Specialty Nodes", "color": "#e879f9"},
            {"cols": [4, 5], "title": "Lifted DXIL SSA Math Ops", "color": "#60a5fa"},
            {"cols": [6], "title": "G-Buffer / Forward Targets", "color": "#f87171"},
        ],
        "stats": {
            "specialty_nodes": len(specialties),
            "math_nodes": math_nodes_count,
            "texture_samplers": len(seen_tex_keys),
            "cbuffer_params": len(cbuf_param_ids),
            "outputs": len(store_tids),
        },
        "available_targets": sorted({tg for n in nodes for tg in n.get("meta", {}).get("targets", [])}),
    }
