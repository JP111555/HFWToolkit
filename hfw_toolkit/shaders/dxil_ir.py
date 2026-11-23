"""DXIL LLVM-IR static analyzer: parses `@main()` instructions, resource handles, samples, cbuffer loads, and outputs."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from hfw_toolkit.shaders.dxc import DisassembledProgram


@dataclass(slots=True)
class TextureSampleOp:
    ssa_var: str
    op_kind: str  # sample | sampleLevel | sampleBias | sampleGrad | textureLoad
    handle_var: str
    res_space: int | None
    res_reg: int | None
    is_bindless: bool
    uv_expr_vars: tuple[str, str]
    uv_scaled: bool = False
    uv_scale_hint: float | None = None
    channels_used: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CBufferLoadOp:
    ssa_var: str
    handle_var: str
    res_space: int | None
    res_reg: int | None
    reg_index: str  # literal int or dynamic SSA var


@dataclass(slots=True)
class OutputStoreOp:
    output_id: int  # SV_Target index (0, 1, 2...)
    row: int
    col: int  # 0=x, 1=y, 2=z, 3=w
    val_var: str


@dataclass(slots=True)
class DXILAnalysis:
    stage: str
    shader_model: str
    total_instructions: int
    alu_ops: int
    branch_count: int
    texture_samples: list[TextureSampleOp] = field(default_factory=list)
    cbuffer_loads: list[CBufferLoadOp] = field(default_factory=list)
    output_stores: list[OutputStoreOp] = field(default_factory=list)
    has_discard: bool = False
    has_derivatives: bool = False
    has_pow_or_exp: bool = False
    has_dot3: bool = False
    has_rsqrt: bool = False
    uses_global_space2_textures: list[str] = field(default_factory=list)
    uses_material_space5_textures: list[int] = field(default_factory=list)
    uses_vertex_color: bool = False
    uses_bindless_heap: bool = False
    # Maps `SV_Target{k}` -> list of `(space, reg)` textures that transitively feed into it!
    target_texture_deps: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    # Maps `SV_Target{k}` -> list of `(space, cbuf_idx)` cbuffer loads feeding into it!
    target_cbuf_deps: dict[int, list[tuple[int, str]]] = field(default_factory=dict)


_SSA_DEF_RE = re.compile(r"^\s*(%\d+)\s*=\s*(.+)$")
_SSA_REF_RE = re.compile(r"%\d+")
_HANDLE_BIND_RE = re.compile(
    r"dx\.op\.createHandleFromBinding\([^,]+,\s*%dx\.types\.ResBind\s*\{i32\s*(\d+),\s*i32\s*(\d+),\s*i32\s*(\d+),\s*i8\s*(\d+)\}"
)
_HANDLE_LEGACY_RE = re.compile(r"dx\.op\.createHandle\(i32\s*\d+,\s*i8\s*(\d+),\s*i32\s*(\d+),")
_HANDLE_HEAP_RE = re.compile(r"dx\.op\.createHandleFromHeap\(")
_SAMPLE_RE = re.compile(
    r"dx\.op\.(sample|sampleLevel|sampleBias|sampleGrad|textureLoad)\.[^(]+\("
    r"i32\s*\d+,\s*%dx\.types\.Handle\s*(%\d+),\s*(?:%dx\.types\.Handle\s*%\d+,\s*)?"
    r"(?:float|i32)\s*([^,]+),\s*(?:float|i32)\s*([^,]+)"
)
_CBUF_RE = re.compile(
    r"dx\.op\.cbufferLoadLegacy\.[^(]+\(i32\s*\d+,\s*%dx\.types\.Handle\s*(%\d+),\s*i32\s*([^)\s]+)"
)
_STORE_OUT_RE = re.compile(
    r"dx\.op\.storeOutput\.[^(]+\(i32\s*\d+,\s*i32\s*(\d+),\s*i32\s*(\d+),\s*i8\s*(\d+),\s*\S+\s*([^)\s]+)"
)
_EXTRACT_RE = re.compile(r"extractvalue\s+%dx\.types\.(?:ResRet|CBufRet)\.\S+\s+(%\d+),\s*(\d+)")
_FMUL_CONST_RE = re.compile(r"fmul\s+(?:fast\s+)?float\s+(%\d+),\s*([0-9.eE+\-xA-Fa-f]+)")


def analyze_dxil(program: DisassembledProgram) -> DXILAnalysis:
    """Parse disassembled DXIL LLVM-IR into a structured static analysis + backward dependency map."""
    text = program.disassembly
    in_main = False

    # Maps resource class+id (from header bindings) to (space, reg)
    # ResClass: 0=SRV(texture), 1=UAV, 2=CBV(cbuffer), 3=Sampler
    by_class_id: dict[tuple[int, int], tuple[int, int]] = {}
    for b in program.bindings:
        cls = {"texture": 0, "UAV": 1, "cbuffer": 2, "sampler": 3}.get(b.res_type, -1)
        m_id = re.search(r"(\d+)$", b.res_id)
        if cls >= 0 and m_id:
            by_class_id[(cls, int(m_id.group(1)))] = (b.space, b.register)

    # SSA graph tables
    ssa_deps: dict[str, list[str]] = {}
    ssa_expr: dict[str, str] = {}
    handle_info: dict[str, tuple[int | None, int | None, bool]] = {}  # var -> (space, reg, is_bindless)
    sample_by_var: dict[str, TextureSampleOp] = {}
    cbuf_by_var: dict[str, CBufferLoadOp] = {}

    samples: list[TextureSampleOp] = []
    cbufs: list[CBufferLoadOp] = []
    stores: list[OutputStoreOp] = []

    total_insts = 0
    alu_ops = 0
    branches = 0
    has_discard = False
    has_deriv = False
    has_pow = False
    has_dot3 = False
    has_rsqrt = False

    for line in text.splitlines():
        if line.startswith("define void @main("):
            in_main = True
            continue
        if in_main and line.startswith("}"):
            in_main = False
            break
        if not in_main:
            continue
        s = line.strip()
        if not s or s.startswith(";") or s.endswith(":"):
            continue

        total_insts += 1
        if s.startswith("br ") or s.startswith("switch "):
            branches += 1
        if "@dx.op.discard(" in s:
            has_discard = True
        if "@dx.op.deriv" in s:
            has_deriv = True
        if "@dx.op.unary.f32(i32 21" in s or "@dx.op.unary.f32(i32 23" in s:  # exp2 / log2
            has_pow = True
        if "@dx.op.dot3." in s:
            has_dot3 = True
        if "@dx.op.unary.f32(i32 25" in s:  # rsqrt
            has_rsqrt = True

        # Output stores (void calls)
        ms = _STORE_OUT_RE.search(s)
        if ms:
            stores.append(
                OutputStoreOp(
                    output_id=int(ms.group(1)),
                    row=int(ms.group(2)),
                    col=int(ms.group(3)),
                    val_var=ms.group(4),
                )
            )
            continue

        mdef = _SSA_DEF_RE.match(s)
        if not mdef:
            continue
        var, rhs = mdef.group(1), mdef.group(2)
        ssa_expr[var] = rhs
        ssa_deps[var] = _SSA_REF_RE.findall(rhs)

        if any(op in rhs for op in ("fmul ", "fadd ", "fsub ", "fdiv ", "select ", "fcmp ", "dx.op.binary", "dx.op.tertiary", "dx.op.dot")):
            alu_ops += 1

        # Handle creation
        mhb = _HANDLE_BIND_RE.search(rhs)
        if mhb:
            reg_lo, _reg_hi, space, _cls = int(mhb.group(1)), int(mhb.group(2)), int(mhb.group(3)), int(mhb.group(4))
            handle_info[var] = (space, reg_lo, False)
            continue
        mhl = _HANDLE_LEGACY_RE.search(rhs)
        if mhl:
            cls, rid = int(mhl.group(1)), int(mhl.group(2))
            sp, rg = by_class_id.get((cls, rid), (None, None))
            handle_info[var] = (sp, rg, False)
            continue
        if _HANDLE_HEAP_RE.search(rhs):
            handle_info[var] = (None, None, True)
            continue
        if "dx.op.annotateHandle(" in rhs:
            # Propagate underlying handle info!
            refs = ssa_deps[var]
            if refs and refs[0] in handle_info:
                handle_info[var] = handle_info[refs[0]]
            continue

        # Texture samples
        msamp = _SAMPLE_RE.search(rhs)
        if msamp:
            op_kind, hvar, uvar, vvar = msamp.group(1), msamp.group(2), msamp.group(3).strip(), msamp.group(4).strip()
            sp, rg, is_bl = handle_info.get(hvar, (None, None, False))
            # Check if U or V was scaled by a constant or cbuffer (UV tiling detection!)
            uv_scaled = False
            scale_hint: float | None = None
            for coord_v in (uvar, vvar):
                expr_c = ssa_expr.get(coord_v, "")
                if "fmul " in expr_c:
                    uv_scaled = True
                    mfc = _FMUL_CONST_RE.search(expr_c)
                    if mfc:
                        try:
                            val = float(mfc.group(2))
                            if 1.5 <= abs(val) <= 256.0:
                                scale_hint = round(val, 3)
                        except ValueError:
                            pass
            op = TextureSampleOp(
                ssa_var=var,
                op_kind=op_kind,
                handle_var=hvar,
                res_space=sp,
                res_reg=rg,
                is_bindless=is_bl,
                uv_expr_vars=(uvar, vvar),
                uv_scaled=uv_scaled,
                uv_scale_hint=scale_hint,
            )
            samples.append(op)
            sample_by_var[var] = op
            continue

        # CBuffer loads
        mcb = _CBUF_RE.search(rhs)
        if mcb:
            hvar, ridx = mcb.group(1), mcb.group(2)
            sp, rg, _ = handle_info.get(hvar, (None, None, False))
            cb = CBufferLoadOp(ssa_var=var, handle_var=hvar, res_space=sp, res_reg=rg, reg_index=ridx)
            cbufs.append(cb)
            cbuf_by_var[var] = cb
            continue

        # Track which channels (.x/.y/.z/.w) are extracted from each texture sample
        mext = _EXTRACT_RE.search(rhs)
        if mext:
            src_var, ch_idx = mext.group(1), int(mext.group(2))
            if src_var in sample_by_var and 0 <= ch_idx <= 3:
                lane = ("R", "G", "B", "A")[ch_idx]
                if lane not in sample_by_var[src_var].channels_used:
                    sample_by_var[src_var].channels_used.append(lane)

    # Backward dataflow trace from each SV_Target output to see which textures & cbuffers feed each target!
    target_tex_deps: dict[int, list[tuple[int, int]]] = {}
    target_cbuf_deps: dict[int, list[tuple[int, str]]] = {}

    stores_by_target: dict[int, list[str]] = {}
    for st in stores:
        stores_by_target.setdefault(st.output_id, []).append(st.val_var)

    for tid, root_vars in stores_by_target.items():
        visited: set[str] = set()
        stack = list(root_vars)
        tex_set: set[tuple[int, int]] = set()
        cb_set: set[tuple[int, str]] = set()
        while stack:
            v = stack.pop()
            if v in visited or not v.startswith("%"):
                continue
            visited.add(v)
            if v in sample_by_var:
                op = sample_by_var[v]
                if op.res_space is not None and op.res_reg is not None:
                    tex_set.add((op.res_space, op.res_reg))
            if v in cbuf_by_var:
                cb = cbuf_by_var[v]
                if cb.res_space is not None:
                    cb_set.add((cb.res_space, cb.reg_index))
            # Bounded walk (avoid blowing up on massive control-flow loops)
            if len(visited) < 2500:
                stack.extend(ssa_deps.get(v, ()))
        target_tex_deps[tid] = sorted(tex_set)
        target_cbuf_deps[tid] = sorted(cb_set)[:40]

    # Summarize bound space2 (global) and space5 (material) textures
    space2 = sorted({f"{b.hlsl_bind} ({b.dim})" for b in program.bindings if b.res_type == "texture" and b.space == 2})
    space5 = sorted({b.register for b in program.bindings if b.res_type == "texture" and b.space == 5})
    uses_vcol = any(inp.name == "COLOR" for inp in program.inputs)

    return DXILAnalysis(
        stage=program.stage,
        shader_model=program.shader_model,
        total_instructions=total_insts,
        alu_ops=alu_ops,
        branch_count=branches,
        texture_samples=samples,
        cbuffer_loads=cbufs,
        output_stores=stores,
        has_discard=has_discard,
        has_derivatives=has_deriv,
        has_pow_or_exp=has_pow,
        has_dot3=has_dot3,
        has_rsqrt=has_rsqrt,
        uses_global_space2_textures=space2,
        uses_material_space5_textures=space5,
        uses_vertex_color=uses_vcol,
        uses_bindless_heap=program.uses_bindless_heap,
        target_texture_deps=target_tex_deps,
        target_cbuf_deps=target_cbuf_deps,
    )
