"""Phase 4 tests: Multi-layer shader deep-dive analysis, DXIL disassembly, Decima & UE5 node graphs, and UE5 export."""

from __future__ import annotations

import pytest

from hfw_toolkit.config import ToolkitConfig
from hfw_toolkit.decima import AssetCatalog, HFWGame
from hfw_toolkit.extract import assemble_machine, resolve_machine_entity_id
from hfw_toolkit.shaders import DXCDisassembler, ShaderHashDictionary, analyze_material_deep


@pytest.fixture(scope="module")
def env():
    cfg = ToolkitConfig.load()
    game = HFWGame(cfg)
    catalog = AssetCatalog(cfg.index_db_path)
    dxc = DXCDisassembler(cfg)
    hdict = ShaderHashDictionary(cfg)
    yield game, catalog, dxc, hdict
    catalog.close()
    game.close()


@pytest.mark.parametrize("machine_name", ["Slitherfang", "Slaughterspine"])
def test_deep_shader_analysis(env, machine_name: str) -> None:
    game, catalog, dxc, hdict = env
    oid, code, disp = resolve_machine_entity_id(game, catalog, machine_name)
    asm = assemble_machine(game, oid, code, disp, include_broken_states=False)

    # Pick the richest material (most texture bindings in primary technique)
    rich_mid, rich_mat = max(
        asm.materials.items(),
        key=lambda kv: len(kv[1].primary_technique.texture_bindings) if kv[1].primary_technique else 0,
    )
    report = analyze_material_deep(game, rich_mid, dxc=dxc, hash_dict=hdict, include_raw_dxil=True)

    print(f"\n[{disp} Richest Material: {report['name']} ({rich_mid})]")
    print(f"  Headline: {report['explainer']['headline']}")
    print(f"  GPU Profile: {report['explainer']['gpu_profile']}")
    print(f"  Explainer Cards ({len(report['explainer']['cards'])}): {[c['title'] for c in report['explainer']['cards']]}")
    print(f"  Decima Graph: {len(report['decima_graph']['nodes'])} nodes, {len(report['decima_graph']['edges'])} edges")
    print(f"  Unreal Graph: {len(report['unreal_graph']['nodes'])} nodes, {len(report['unreal_graph']['edges'])} edges, fidelity={report['unreal_graph']['fidelity']['score_percent']}%")
    print(f"  Stages disassembled: {[(s['stage'], s.get('shader_model'), s.get('instructions')) for s in report['stages']]}")

    assert len(report["stages"]) >= 1
    assert any(s.get("instructions", 0) > 10 for s in report["stages"])
    assert len(report["explainer"]["cards"]) >= 3
    assert len(report["decima_graph"]["nodes"]) >= 5
    assert len(report["decima_graph"]["edges"]) >= 3
    assert len(report["unreal_graph"]["nodes"]) >= 5
    assert len(report["unreal_graph"]["edges"]) >= 3
    assert "Begin Object Class=/Script/Engine.MaterialExpressionTextureSampleParameter2D" in report["ue5_clipboard_t3d"]
    assert "unreal.MaterialEditingLibrary" in report["ue5_python_script"]
    assert "SV_Target0" in report["pseudo_hlsl"]
    # Detailed DXIL SSA Math & Decima Specialty Node Graph
    dg = report["decima_detailed_graph"]
    assert len(dg["nodes"]) >= 12
    assert len(dg["edges"]) >= 8
    assert dg["stats"]["specialty_nodes"] >= 3
    assert dg["stats"]["math_nodes"] >= 3


def test_detailed_dxil_specialty_nodes_and_animated_ue5(env) -> None:
    game, _catalog, dxc, hdict = env

    # 1. Frostclaw Belly Sac (3488:1219): Dual-Phase Flow Map + RGB<->HSL Rim Shift + 23-Slice Detail + Worn-Metal F0 + Animated UE5 Time/Panner!
    sac = analyze_material_deep(game, "3488:1219", dxc=dxc, hash_dict=hdict, include_raw_dxil=False)
    sac_specs = {n["title"] for n in sac["decima_detailed_graph"]["nodes"] if n["category"] == "specialty"}
    assert "Decima Dual-Phase Flow Map" in sac_specs
    assert "Decima RGB ↔ HSL Rim Shift" in sac_specs
    assert "Decima 23-Slice Detail Array" in sac_specs
    assert "Decima Worn-Metal F0 (RFL²)" in sac_specs
    assert "MaterialExpressionTime" in sac["ue5_clipboard_t3d"]
    assert "MaterialExpressionPanner" in sac["ue5_clipboard_t3d"]
    assert "MaterialExpressionTime" in sac["ue5_python_script"]

    # 2. Thunderjaw Outer Eye (755:179): 6-step Parallax Occlusion Mapping (POM) Raymarch!
    eye = analyze_material_deep(game, "755:179", dxc=dxc, hash_dict=hdict, include_raw_dxil=False)
    eye_specs = {n["title"] for n in eye["decima_detailed_graph"]["nodes"] if n["category"] == "specialty"}
    assert "Decima Parallax Occlusion (POM)" in eye_specs
    assert "MaterialExpressionTime" in eye["ue5_clipboard_t3d"]
