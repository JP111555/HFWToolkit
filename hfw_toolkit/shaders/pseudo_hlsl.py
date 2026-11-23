"""Annotated Readable Pseudo-HLSL generator + optional `dxil-spirv` / `spirv-cross` full HLSL decompiler + Beginner Glossary."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from hfw_toolkit.config import ToolkitConfig
from hfw_toolkit.extract.materials import MaterialInfo
from hfw_toolkit.shaders.dxil_ir import DXILAnalysis
from hfw_toolkit.shaders.dxc import DisassembledProgram

# Beginner-friendly shader glossary for hover tooltips in the Web Explorer!
SHADER_GLOSSARY: dict[str, str] = {
    "SV_Target0": "First G-Buffer render target written by the Pixel Shader — typically stores Linear Albedo / Base Color.",
    "SV_Target1": "Second G-Buffer render target — stores encoded World-Space Normal vector and shading model flags.",
    "SV_Target2": "Third G-Buffer render target — stores PBR surface properties: Roughness, Reflectance/Metalness, and Ambient Occlusion.",
    "Texture2D": "A 2D image resource sampled on the GPU at UV coordinates.",
    "Texture3D": "A 3D volume texture (often used by Decima for 3D noise, fog, or color-grading LUTs).",
    "Sample": "Reads filtered pixel colors from a texture at (U, V) coordinates using bilinear/anisotropic filtering and mipmaps.",
    "cbuffer": "Constant Buffer — a fast GPU block of uniform material/scene parameters (tiling scales, tints, plasma intensity, camera matrices).",
    "space5": "Decima's per-material register space: textures (`t0..tN, space5`) and variables (`cb0, space5`) bound specifically by this RenderTechnique.",
    "space2": "Decima's global per-view/world register space: shared engine LUTs, weather/wetness/snow maps, and lighting buffers.",
    "lerp": "Linear Interpolation: `lerp(A, B, t) = A * (1 - t) + B * t`. Used everywhere to blend between two layers using a mask!",
    "saturate": "Clamps a value cleanly into the `[0.0, 1.0]` range (free instruction on GPUs).",
    "normalize": "Scales a 3D vector to unit length 1.0 — essential after blending or transforming normal vectors.",
    "sqrt": "Square root — used to reconstruct Tangent-Space Normal Z from packed XY: `z = sqrt(saturate(1 - x*x - y*y))`.",
    "dot": "Dot product between two vectors — measures alignment (e.g. surface normal vs view direction for Fresnel rim lighting).",
    "pow": "Raises a value to an exponent — shapes specular falloff, Fresnel rim sharpness, or contrast curves.",
    "discard": "Immediately kills the current pixel so nothing is drawn (alpha-cutout masking for grates, foliage, or cables).",
    "ddx": "Screen-space horizontal derivative — measures how fast UVs/positions change between neighboring pixels (for mip selection & normals).",
    "BindlessHeap": "DirectX 12 Shader Model 6.6 `ResourceDescriptorHeap` — lets the shader dynamically pick any texture in VRAM using an integer index!",
}


def generate_annotated_pseudo_hlsl(
    mat: MaterialInfo,
    ps_prog: DisassembledProgram | None,
    ps_ir: DXILAnalysis | None,
    binding_map: dict[str, dict[str, Any]],
) -> str:
    """Produce human-readable, richly commented Pseudo-HLSL that ties DXIL slots directly to real HFW textures & values."""
    pt = mat.primary_technique
    lines: list[str] = [
        f"// ============================================================================",
        f"// HFW Annotated Pseudo-HLSL — Material: {mat.name} ({mat.object_id})",
        f"// Shader Model: {ps_prog.shader_model if ps_prog else 'N/A'} | Passes: {', '.join(pt.available_passes) if pt else 'N/A'}",
        f"// Note: HLSL source names are stripped in retail DXIL; TextureSet filenames,",
        f"//       channel packing, and parameter values below are recovered from Decima RTTI & SRT.",
        f"// ============================================================================",
        "",
        "// --- Per-Material Textures (register space5) ---",
    ]

    # List correlated space5 textures with original Guerrilla paths & channel roles!
    mat_tex_entries = sorted(
        [(k, v) for k, v in binding_map.items() if k.startswith("tex:5:")],
        key=lambda kv: int(kv[0].split(":")[2]),
    )
    for k, info in mat_tex_entries:
        reg = k.split(":")[2]
        chs = ", ".join(f"{l}={s}" for l, s in info.get("target_channels", {}).items()) or info.get("semantic_role", "Texture")
        lines.append(
            f"Texture2D tex_{reg}_{_ident(info['name'])} : register(t{reg}, space5); "
            f"// [{chs}] res={info.get('resource_id')} hash={info.get('hash')}"
        )

    if ps_ir and ps_ir.uses_global_space2_textures:
        lines.append("")
        lines.append("// --- Global Engine / Weather Textures (register space2) ---")
        for gtex in ps_ir.uses_global_space2_textures:
            lines.append(f"// Global engine resource: {gtex}  (weather / wetness / LUT / world data)")

    lines.append("")
    lines.append("// --- Per-Material Parameters (cb0, space5) ---")
    lines.append("cbuffer MaterialConstants : register(b0, space5) {")
    var_entries = sorted(
        [(k, v) for k, v in binding_map.items() if k.startswith("var:")],
        key=lambda kv: int(kv[0].split(":")[1]),
    )
    detail_tile = 12.0
    for k, v in var_entries[:24]:
        slot = k.split(":")[1]
        vname = _ident(v.get("inferred_name") or f"param_{slot}")
        f4 = v.get("float4", [0, 0, 0, 0])
        if v.get("hash") == "0x3806eae3" and 1.5 <= f4[0] <= 128.0:
            detail_tile = f4[0]
        lines.append(f"    float4 {vname}; // type={v['type']} default=({f4[0]:.3g}, {f4[1]:.3g}, {f4[2]:.3g}, {f4[3]:.3g}) hash={v['hash']}")
    lines.append("};")

    lines.extend(
        [
            "",
            "struct GBufferOut {",
            "    float4 SV_Target0 : SV_Target0; // Base Color / Albedo (Linear RGB)",
            "    float4 SV_Target1 : SV_Target1; // Encoded World-Space Normal + Flags",
            "    float4 SV_Target2 : SV_Target2; // Roughness, Reflectance/Metallic, AO",
            "};",
            "",
            "GBufferOut main(float2 uv0 : STAGE_IO0, float3 worldNormal : STAGE_IO2, float4 worldTangent : STAGE_IO3) {",
            "    GBufferOut o = (GBufferOut)0;",
            "",
            "    // 1. Sample primary packed TextureSets at UV0",
        ]
    )

    color_var = "float3(0.5, 0.5, 0.5)"
    nrm_var = "worldNormal"
    rough_var = "0.5"
    refl_var = "0.08"
    ao_var = "1.0"
    mask_var = None
    has_detail = False

    for k, info in mat_tex_entries:
        reg = k.split(":")[2]
        tident = f"tex_{reg}_{_ident(info['name'])}"
        tchs = info.get("target_channels", {})
        vals = " ".join(tchs.values())
        if info.get("is_detail"):
            has_detail = True
            lines.append(f"    float2 detailUV_{reg} = uv0 * {detail_tile:.2f}; // High-frequency detail repeat")
            lines.append(f"    float4 samp_det_{reg} = {tident}.Sample(anisoSampler, detailUV_{reg}); // {info.get('semantic_role')}")
            continue
        if not tchs:
            continue
        svar = f"samp_{reg}"
        lines.append(f"    float4 {svar} = {tident}.Sample(anisoSampler, uv0); // {', '.join(f'{l}={s}' for l, s in tchs.items())}")
        if "Color" in vals:
            color_var = f"{svar}.rgb"
        if "Normal" in vals:
            lines.append(f"    float2 nrmXY = {svar}.rg * 2.0 - 1.0;")
            lines.append(f"    float3 macroNormal = float3(nrmXY, sqrt(saturate(1.0 - dot(nrmXY, nrmXY)))); // Reconstruct Z")
            nrm_var = "macroNormal"
        for lane, spec in tchs.items():
            lc = lane.lower()
            if spec.startswith("AO"):
                ao_var = f"{svar}.{lc}"
            elif spec.startswith("Roughness"):
                rough_var = f"{svar}.{lc}"
            elif spec.startswith("Reflectance"):
                refl_var = f"{svar}.{lc}"
            elif spec.startswith("Mask."):
                mask_var = f"{svar}.{lc}"

    if has_detail:
        lines.extend(
            [
                "",
                "    // 2. Blend high-frequency micro-detail normal (brushed alloy / muscle fibers / wear)",
                f"    float3 finalNormal = normalize({nrm_var} + float3(detailDelta.xy, 0.0));",
            ]
        )
        nrm_var = "finalNormal"

    if mask_var:
        lines.extend(
            [
                "",
                f"    // 3. Apply multi-layer surface mask ({mask_var}) for paint / alloy / wear blending",
                f"    float layerMask = {mask_var};",
            ]
        )

    if ps_ir and ps_ir.uses_global_space2_textures:
        lines.extend(
            [
                "",
                "    // 4. Dynamic World & Weather modulation (from global space2 engine buffers)",
                f"    float3 finalAlbedo = lerp({color_var}, {color_var} * 0.45, globalWetness);",
                f"    float  finalRough  = lerp({rough_var}, 0.08, globalWetness);",
            ]
        )
        color_var = "finalAlbedo"
        rough_var = "finalRough"

    if ps_ir and ps_ir.has_discard:
        lines.append("    if (alphaMask < alphaCutoff) discard; // Alpha-tested cutout")

    lines.extend(
        [
            "",
            "    // 5. Pack final surface into Decima G-Buffer outputs",
            f"    o.SV_Target0 = float4({color_var}, 1.0); // Albedo",
            f"    o.SV_Target1 = encodeWorldNormal({nrm_var}, worldNormal, worldTangent);",
            f"    o.SV_Target2 = float4({rough_var}, {refl_var}, {ao_var}, 1.0);",
            "    return o;",
            "}",
        ]
    )

    if ps_ir:
        lines.extend(
            [
                "",
                f"// --- Live DXIL Telemetry ({ps_ir.shader_model}) ---",
                f"// Total IR instructions: {ps_ir.total_instructions} | ALU ops: {ps_ir.alu_ops} | Branches: {ps_ir.branch_count}",
                f"// Texture samples: {len(ps_ir.texture_samples)} | CBuffer loads: {len(ps_ir.cbuffer_loads)} | Bindless Heap: {ps_ir.uses_bindless_heap}",
            ]
        )
    return "\n".join(lines)


def _ident(name: str) -> str:
    clean = "".join(c if (c.isalnum() or c == "_") else "_" for c in name)
    return clean[:40] or "unnamed"


def find_external_decompilers(config: ToolkitConfig) -> tuple[Path | None, Path | None]:
    """Auto-detect optional `dxil-spirv` and `spirv-cross` binaries in `HFWToolkit/bin/` or PATH."""
    def _find(exe_names: tuple[str, ...]) -> Path | None:
        for name in exe_names:
            cand = config.bin_dir / name
            if cand.exists():
                return cand
            found = shutil.which(name)
            if found:
                return Path(found)
        return None

    dxil_spirv = _find(("dxil-spirv.exe", "dxil-spirv"))
    spirv_cross = _find(("spirv-cross.exe", "spirv-cross"))
    return dxil_spirv, spirv_cross


def try_decompile_hlsl_external(blob: bytes, config: ToolkitConfig) -> dict[str, Any]:
    """If `dxil-spirv` and `spirv-cross` are installed in `bin/` or PATH, decompile DXIL -> SPIR-V -> HLSL."""
    dxil_spirv, spirv_cross = find_external_decompilers(config)
    if not dxil_spirv or not spirv_cross:
        return {
            "available": False,
            "dxil_spirv": str(dxil_spirv) if dxil_spirv else None,
            "spirv_cross": str(spirv_cross) if spirv_cross else None,
            "hlsl": "",
            "note": (
                "Optional full decompiler not installed. Place `dxil-spirv.exe` and `spirv-cross.exe` in "
                f"`{config.bin_dir}` (or on PATH) to unlock full structured HLSL decompilation alongside "
                "the built-in Annotated Pseudo-HLSL, Visual Node Graph, and Raw DXIL Disassembly."
            ),
        }

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        in_dxil = tdp / "shader.dxil"
        mid_spv = tdp / "shader.spv"
        in_dxil.write_bytes(blob)

        p1 = subprocess.run(
            [str(dxil_spirv), str(in_dxil), "--output", str(mid_spv)],
            capture_output=True,
            text=True,
            check=False,
        )
        if p1.returncode != 0 or not mid_spv.exists():
            return {"available": True, "error": f"dxil-spirv failed: {p1.stderr[:500]}", "hlsl": ""}

        p2 = subprocess.run(
            [str(spirv_cross), str(mid_spv), "--hlsl", "--shader-model", "60"],
            capture_output=True,
            text=True,
            check=False,
        )
        if p2.returncode != 0:
            return {"available": True, "error": f"spirv-cross failed: {p2.stderr[:500]}", "hlsl": ""}

        return {"available": True, "hlsl": p2.stdout, "error": ""}
