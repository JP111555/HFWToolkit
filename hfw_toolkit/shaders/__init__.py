"""Unified multi-layer shader & material deep-dive analyzer."""

from __future__ import annotations

from typing import Any

from hfw_toolkit.decima.reader import ObjectId
from hfw_toolkit.decima.streaming_graph import HFWGame
from hfw_toolkit.extract.materials import MaterialInfo, parse_material
from hfw_toolkit.shaders.dxc import DXCDisassembler, DisassembledProgram
from hfw_toolkit.shaders.dxil_graph import build_detailed_dxil_graph
from hfw_toolkit.shaders.dxil_ir import DXILAnalysis, analyze_dxil
from hfw_toolkit.shaders.explainer import build_plain_english_explanation
from hfw_toolkit.shaders.hash_dict import ShaderHashDictionary
from hfw_toolkit.shaders.hw_shader import HwShaderData, correlate_bindings_to_registers, parse_hw_shader
from hfw_toolkit.shaders.node_graph import build_decima_node_graph, build_unreal_material_graph
from hfw_toolkit.shaders.pseudo_hlsl import SHADER_GLOSSARY, generate_annotated_pseudo_hlsl, try_decompile_hlsl_external
from hfw_toolkit.shaders.unreal_export import (
    detect_ue5_animation_profile,
    generate_ue5_clipboard_t3d,
    generate_ue5_machine_script,
    generate_ue5_python_script,
)


def analyze_material_deep(
    game: HFWGame,
    material_id: str,
    dxc: DXCDisassembler | None = None,
    hash_dict: ShaderHashDictionary | None = None,
    include_raw_dxil: bool = True,
    try_external_decompiler: bool = False,
) -> dict[str, Any]:
    """Perform complete 5-layer deep analysis on a `RenderEffectResource` by ObjectId (`'group:index'`)."""
    dxc = dxc or DXCDisassembler(game.config)
    hash_dict = hash_dict or ShaderHashDictionary(game.config)

    re_obj = game.read_object(ObjectId.parse(material_id), read_subgroups=True)
    mat: MaterialInfo = parse_material(game, re_obj)

    # Enrich variable binding names from hash dictionary
    for tech in mat.techniques:
        for vb in tech.variable_bindings:
            vb.inferred_name = hash_dict.resolve(vb.binding_name_hash) or hash_dict.resolve(vb.variable_id_hash)

    pt = mat.primary_technique
    hw_shader: HwShaderData | None = None
    ps_prog: DisassembledProgram | None = None
    vs_prog: DisassembledProgram | None = None
    ps_ir: DXILAnalysis | None = None
    vs_ir: DXILAnalysis | None = None
    all_stages: list[dict[str, Any]] = []
    binding_map: dict[str, dict[str, Any]] = {}
    external_hlsl: dict[str, Any] = {"available": False, "hlsl": ""}

    if pt and pt.shader_id:
        sh_obj = game.read_object(ObjectId.parse(pt.shader_id), read_subgroups=True)
        hw_shader = parse_hw_shader(sh_obj)

        for prog in hw_shader.programs:
            try:
                dis = dxc.disassemble_program(prog.blob, prog.stage)
                ir = analyze_dxil(dis)
            except Exception as exc:
                all_stages.append({"stage": prog.stage, "error": str(exc)})
                continue

            if prog.stage == "PixelProgram" and ps_prog is None:
                ps_prog, ps_ir = dis, ir
                if try_external_decompiler:
                    external_hlsl = try_decompile_hlsl_external(prog.blob, game.config)
            elif prog.stage == "VertexProgram" and vs_prog is None:
                vs_prog, vs_ir = dis, ir

            all_stages.append(
                {
                    "stage": dis.stage,
                    "shader_model": dis.shader_model,
                    "debug_name": dis.debug_name,
                    "shader_hash": dis.shader_hash,
                    "dxbc_chunks": prog.dxbc_chunks,
                    "blob_size": len(prog.blob),
                    "instructions": ir.total_instructions,
                    "alu_ops": ir.alu_ops,
                    "texture_samples": len(ir.texture_samples),
                    "branches": ir.branch_count,
                    "uses_bindless_heap": ir.uses_bindless_heap,
                    "inputs": [
                        {"name": i.name, "index": i.index, "mask": i.mask, "register": i.register, "fmt": i.fmt}
                        for i in dis.inputs
                    ],
                    "outputs": [
                        {"name": o.name, "index": o.index, "mask": o.mask, "register": o.register, "fmt": o.fmt}
                        for o in dis.outputs
                    ],
                    "bindings": [
                        {
                            "type": b.res_type,
                            "dim": b.dim,
                            "id": b.res_id,
                            "hlsl_bind": b.hlsl_bind,
                            "space": b.space,
                            "register": b.register,
                            "count": b.count,
                        }
                        for b in dis.bindings
                    ],
                    "disassembly": dis.disassembly if include_raw_dxil else "",
                }
            )

        pixel_tex_regs = (
            [(b.space, b.register) for b in ps_prog.bindings if b.res_type == "texture"] if ps_prog else []
        )
        binding_map = correlate_bindings_to_registers(hw_shader, pt, pixel_tex_regs)

    explainer = build_plain_english_explanation(mat, ps_prog, ps_ir, vs_prog, vs_ir, binding_map)
    decima_graph = build_decima_node_graph(mat, ps_ir, binding_map)
    decima_detailed_graph = build_detailed_dxil_graph(mat, ps_prog, ps_ir, binding_map)
    unreal_graph = build_unreal_material_graph(mat, ps_ir)
    pseudo_hlsl = generate_annotated_pseudo_hlsl(mat, ps_prog, ps_ir, binding_map)
    has_weather = bool(ps_ir and ps_ir.uses_global_space2_textures)
    ue5_clipboard = generate_ue5_clipboard_t3d(mat, has_weather=has_weather)
    ue5_python = generate_ue5_python_script(mat, has_weather=has_weather)

    return {
        "material_id": mat.object_id,
        "name": mat.name,
        "tags": mat.material_tags,
        "texture_set_ids": mat.texture_set_ids,
        "standalone_texture_ids": mat.standalone_texture_ids,
        "techniques_summary": [
            {
                "set_index": t.technique_set_index,
                "tech_index": t.technique_index,
                "set_type": t.technique_set_type,
                "passes": t.available_passes,
                "layer_id": t.material_layer_id,
                "shader_id": t.shader_id,
                "tex_count": len(t.texture_bindings),
                "var_count": len(t.variable_bindings),
                "is_primary": (t is pt),
            }
            for t in mat.techniques
        ],
        "primary_textures": [
            {
                "index": tb.index,
                "name": tb.name,
                "role": tb.semantic_role,
                "is_detail": tb.is_detail_texture,
                "binding_kind": tb.binding_kind,
                "resource_id": tb.resource_id,
                "resource_type": tb.resource_type,
                "target_channels": tb.target_channels,
                "source_paths": tb.source_paths,
                "packing_entries": tb.packing_entries,
                "binding_hash": f"{tb.binding_name_hash:#010x}",
            }
            for tb in (pt.texture_bindings if pt else ())
        ],
        "primary_variables": [
            {
                "index": vb.index,
                "binding_hash": f"{vb.binding_name_hash:#010x}",
                "variable_id_hash": f"{vb.variable_id_hash:#010x}",
                "name": vb.inferred_name or f"var_{vb.index}",
                "type": vb.variable_type,
                "float4": list(vb.float4),
                "int4": list(vb.int4),
            }
            for vb in (pt.variable_bindings if pt else ())
        ],
        "binding_map": binding_map,
        "explainer": explainer,
        "decima_graph": decima_graph,
        "decima_detailed_graph": decima_detailed_graph,
        "unreal_graph": unreal_graph,
        "pseudo_hlsl": pseudo_hlsl,
        "external_hlsl": external_hlsl,
        "ue5_clipboard_t3d": ue5_clipboard,
        "ue5_python_script": ue5_python,
        "stages": all_stages,
        "glossary": SHADER_GLOSSARY,
    }


__all__ = [
    "DXCDisassembler",
    "DXILAnalysis",
    "DisassembledProgram",
    "HwShaderData",
    "SHADER_GLOSSARY",
    "ShaderHashDictionary",
    "analyze_dxil",
    "analyze_material_deep",
    "build_decima_node_graph",
    "build_detailed_dxil_graph",
    "build_plain_english_explanation",
    "build_unreal_material_graph",
    "correlate_bindings_to_registers",
    "detect_ue5_animation_profile",
    "generate_annotated_pseudo_hlsl",
    "generate_ue5_clipboard_t3d",
    "generate_ue5_machine_script",
    "generate_ue5_python_script",
    "parse_hw_shader",
]
