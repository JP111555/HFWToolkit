"""Plain-English Shader & Material Explainer (accessible without knowing HLSL)."""

from __future__ import annotations

from typing import Any

from hfw_toolkit.extract.materials import MaterialInfo
from hfw_toolkit.shaders.dxil_ir import DXILAnalysis
from hfw_toolkit.shaders.dxc import DisassembledProgram


def build_plain_english_explanation(
    mat: MaterialInfo,
    ps_prog: DisassembledProgram | None,
    ps_ir: DXILAnalysis | None,
    vs_prog: DisassembledProgram | None,
    vs_ir: DXILAnalysis | None,
    binding_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Generate structured, beginner-friendly explanation cards for how this HFW material works in-game."""
    pt = mat.primary_technique
    passes = pt.available_passes if pt else []
    cards: list[dict[str, Any]] = []

    # 1. Pipeline & G-Buffer Architecture Card
    is_deferred = any("Deferred" in p for p in passes)
    is_emissive = any("Emissive" in p for p in passes) or ("Emissive" in mat.material_tags)
    is_transparent = any("Transparency" in p or "Forward" in p for p in passes) and not is_deferred
    n_targets = len(ps_prog.outputs) if ps_prog else 0

    if is_deferred:
        pipeline_title = "Deferred PBR Material (Decima G-Buffer)"
        pipeline_body = (
            f"Rather than computing sunlight and point lights directly inside this pixel shader, "
            f"Decima writes this surface's physical properties into {n_targets or 3} screen-space G-Buffer textures "
            f"(SV_Target0: Base Color / Albedo, SV_Target1: World-Space Normal + Shading Flags, "
            f"SV_Target2: Roughness, Reflectance/Metalness & Ambient Occlusion). "
            f"A later full-screen lighting pass lights every pixel on screen at once."
        )
        if is_emissive:
            pipeline_body += (
                " This material ALSO supports the DeferredEmissive pass — allowing glowing elements "
                "(like machine optics, plasma vents, or heated canisters) to emit light independently of scene lighting."
            )
    elif is_transparent:
        pipeline_title = "Forward / Transparent Material"
        pipeline_body = (
            "Because transparent or foreground effects cannot easily use the opaque G-Buffer, "
            "this shader runs in a Forward pass and computes final blended color directly."
        )
    else:
        pipeline_title = f"Specialized Pass ({', '.join(passes[:3]) or 'Standard'})"
        pipeline_body = "This shader renders in a specialized engine pass (such as depth, shadow, or custom overlay)."

    cards.append(
        {
            "id": "pipeline",
            "category": "Rendering Pipeline",
            "title": pipeline_title,
            "summary": pipeline_body,
            "why_it_matters": (
                "Deferred rendering lets Horizon Forbidden West put hundreds of complex machine plates and lights "
                "on screen while keeping lighting cost proportional to screen pixels instead of mesh count."
            ),
            "badges": passes[:5] + ([ps_prog.shader_model] if ps_prog else []),
        }
    )

    # 2. Texture Channel Packing Card
    packed_desc: list[str] = []
    has_normal = False
    has_rma = False
    has_mask = False
    has_detail_tex = False
    detail_tile = None

    if pt:
        for vb in pt.variable_bindings:
            if vb.binding_name_hash == 0x3806EAE3 and 1.5 <= vb.float4[0] <= 128.0:
                detail_tile = vb.float4[0]
        for tb in pt.texture_bindings:
            if tb.target_channels:
                ch_str = ", ".join(f"{lane}={spec}" for lane, spec in tb.target_channels.items())
                packed_desc.append(f"{tb.name}: [{ch_str}]")
                vals = " ".join(tb.target_channels.values())
                if "Normal" in vals:
                    has_normal = True
                if "Roughness" in vals or "Reflectance" in vals:
                    has_rma = True
                if "Mask" in vals:
                    has_mask = True
            if tb.is_detail_texture:
                has_detail_tex = True

    cards.append(
        {
            "id": "channel_packing",
            "category": "Texture Engineering",
            "title": "Aggressive RGBA Channel Packing (TextureSets)",
            "summary": (
                "Guerrilla authors separate maps in Substance Painter/Designer (_CLR, _NMT, _RGH, _RFL, _OCC, _MSK), "
                "then Decima's TextureSet compiler packs unrelated grayscale maps into R, G, B, and A channels of fewer GPU textures "
                "(for example: Normal.XY + AO in Blue + Mask_Alpha in Alpha, and Mask + Roughness + Reflectance in a second texture). "
                "Normal Z is never stored — the GPU reconstructs Z = sqrt(1 - X² - Y²) on the fly!"
            ),
            "why_it_matters": (
                "Cuts VRAM usage and memory bandwidth nearly in half compared to loading separate images for Roughness, AO, Metallic, and Masks."
            ),
            "items": packed_desc[:8],
            "badges": [b for b, cond in (("Normal+AO Packed", has_normal), ("Mask+Rough+Reflect Packed", has_rma), ("Layer Masks", has_mask)) if cond],
        }
    )

    # 3. Micro-Detail Texture2DArray (Mask-Driven 23-Slice Atlas!) & Shared Material Library Card
    scaled_samples = [s for s in (ps_ir.texture_samples if ps_ir else []) if s.uv_scaled]
    shared_tex_count = sum(
        1 for tb in (pt.texture_bindings if pt else []) if tb.resource_id and int(tb.resource_id.split(":")[0]) < 300
    )
    has_detail_array = bool(pt and any(tb.binding_name_hash == 0x342B11E0 for tb in pt.texture_bindings))
    if has_detail_array:
        tile_txt = f"{detail_tile:.0f}×" if detail_tile else "high-frequency"
        cards.append(
            {
                "id": "detail_micro",
                "category": "Close-Up Surface Detail",
                "title": f"Mask-Driven 23-Slice Detail Texture2DArray ({tile_txt} UV Repeat)",
                "summary": (
                    f"Instead of a single detail map, this machine shader binds Guerrilla's global 23-slice "
                    f"Texture2DArray (`20:0`, 256×256×23 BC7 containing carbon-fiber weave, hex honeycomb, brushed alloy, "
                    f"knurled grip, diagonal ribs, and micro-scratches). At every pixel, the shader reads the `Mask` "
                    f"channel (`round(Mask * 20)`), remaps it via a switch table to slice 0..22, tiles UVs by {tile_txt}, "
                    f"unpacks the detail normal from `(Alpha, Green)` and color tint from `Red * 2.0`, and blends it "
                    f"onto the base normal weighted by `Mask_Alpha`!"
                ),
                "why_it_matters": (
                    "Lets a single draw call render 20+ completely distinct micro-surface materials (painted ceramic, "
                    "carbon weave, brushed steel, rubber, hex vents) at razor-sharp close-up resolution using only one 8-bit Mask channel."
                ),
                "badges": ["23-Slice Texture2DArray", "Mask -> Slice Index", "Mask_Alpha Weighted", f"UV Scale {tile_txt}"],
            }
        )
    elif detail_tile is not None or scaled_samples or has_detail_tex or shared_tex_count >= 3:
        tile_txt = f"{detail_tile:.0f}×" if detail_tile else "high-frequency"
        cards.append(
            {
                "id": "detail_micro",
                "category": "Close-Up Surface Detail",
                "title": f"Tiled Micro-Detail & Shared Material Library ({tile_txt} UV Repeat)",
                "summary": (
                    f"This material combines unique baked macro textures with {shared_tex_count} shared library textures "
                    f"(such as brushed alloy grain, micro-scratches, carbon-weave machine muscle fibers, or grime). "
                    f"By multiplying UV coordinates by {tile_txt} inside the pixel shader, tiny surface details stay razor-sharp "
                    f"even when the camera zooms inches away from a huge machine like a Slaughterspine or Slitherfang!"
                ),
                "why_it_matters": (
                    "Without tiled detail maps, a 20-meter machine would need 16K+ textures to avoid looking blurry up close."
                ),
                "badges": [f"UV Scale {tile_txt}" if detail_tile else "UV Scaled", f"{shared_tex_count} Shared Textures"],
            }
        )

    # 4. Decima PBR Reflectance & Smoothness Model Card (DXIL-verified!)
    if has_rma:
        cards.append(
            {
                "id": "reflectance_pbr",
                "category": "Physically Based Shading",
                "title": "Decima Smoothness & Specular-Reflectance (`F0 = RFL²`) vs. Metallic/Roughness",
                "summary": (
                    "Disassembled HFW DXIL (`16236:2` lines 592–782) reveals two critical Decima PBR conventions: "
                    "(1) The channel labeled `Roughness` in `MSK_RGH_RFL.G` is actually **Smoothness** (`Roughness = 1 - Smoothness`), "
                    "where matte ceramic paint stores ~0.32–0.49 and polished bare metal edges / glossy eye lenses store ~0.85–1.00. "
                    "(2) Decima uses a Specular/Diffuse workflow where worn bare-metal paint chips store near-black diffuse `CLR ≈ (0,0,0)` "
                    "and high `Reflectance` (`0.60–0.90`). The pixel shader computes `F0 = RFL²`, `m = clamp(F0 * 1.11 - 0.111, 0, 0.98)`, "
                    "and recovers bright metallic tint via `chroma = max(CLR, 0.001) / max_comp(CLR)` so bare worn metal reflects bright silver/steel instead of black!"
                ),
                "why_it_matters": (
                    "Plugging Decima textures straight into Blender or UE5 without inverting Smoothness and recovering `F0 = chroma * RFL²` "
                    "makes matte paint look weirdly plastic-shiny up close and turns shiny metallic edge wear into dead black lines."
                ),
                "badges": ["Smoothness -> 1 - Roughness", "F0 = norm(CLR) * RFL²", "Worn Edge Recovery"],
            }
        )

    # 5. Global Engine Hooks (Weather / Wetness / Snow / Focus Pulse)
    if ps_ir and ps_ir.uses_global_space2_textures:
        cards.append(
            {
                "id": "global_effects",
                "category": "Dynamic World & Weather Effects",
                "title": "Runtime Weather, Wetness & Global Engine Overlays (space2)",
                "summary": (
                    f"This pixel shader reads {len(ps_ir.uses_global_space2_textures)} global engine textures from register `space2` "
                    f"({', '.join(ps_ir.uses_global_space2_textures[:5])}) alongside global constant buffers (`space1–space3`). "
                    f"Decima uses these shared GPU bindings to inject rain wetness (darkening albedo & lowering roughness), "
                    f"snow/dust accumulation on upward-facing normals, elemental status coatings (acid/frost/plasma), "
                    f"and Aloy's Focus scan highlight — without needing separate shaders for every weather state!"
                ),
                "why_it_matters": (
                    "In offline viewers (Blender/UE5), these dynamic gameplay inputs aren't baked into static textures — "
                    "so HFWToolkit exposes them as editable parameters in our Unreal Engine Material Graph!"
                ),
                "badges": ["Global space2 LUTs", "Dynamic Weather/Focus"],
            }
        )

    # 6. Emissive / Plasma / Elemental Glow Card
    if is_emissive:
        cards.append(
            {
                "id": "emissive",
                "category": "Glow & VFX",
                "title": "Dynamic Emissive / Incandescence Output",
                "summary": (
                    "This material participates in the `DeferredEmissive` pass. Machines like Slaughterspine and Slitherfang "
                    "drive emissive intensity dynamically via shader variables (e.g., ramping purple plasma glow along spine fins "
                    "during charge attacks, or dimming glows when a component is shot off)."
                ),
                "why_it_matters": (
                    "In Unreal Engine 5 and Blender, wiring `Incandescence` × `EmissiveIntensity` parameter lets you animate "
                    "calm vs. overcharged plasma states cleanly."
                ),
                "badges": ["DeferredEmissive", "Animatable Glow"],
            }
        )

    # 7. Alpha Cutout / Vertex Effects Card
    if ps_ir and ps_ir.has_discard:
        cards.append(
            {
                "id": "alpha_cutout",
                "category": "Transparency",
                "title": "Alpha-Tested Cutout (`discard`)",
                "summary": (
                    "The pixel shader uses `dx.op.discard` to clip pixels whose mask/alpha value falls below a threshold — "
                    "commonly used for fine cables, grates, foliage, or damaged jagged edges."
                ),
                "why_it_matters": "Maps directly to `Alpha Clip` in Blender and `Masked` (`Opacity Mask`) blend mode in Unreal Engine 5.",
                "badges": ["Alpha Clip / Masked"],
            }
        )

    if vs_prog:
        in_names = [i.name for i in vs_prog.inputs]
        vs_notes = []
        if any("BLEND" in n for n in in_names):
            vs_notes.append("GPU skeletal skinning")
        if "COLOR" in in_names:
            vs_notes.append("vertex color masking (COLOR0)")
        if "PIVOT_POINT" in in_names:
            vs_notes.append("pivot-point procedural deformation/animation")
        if vs_notes:
            cards.append(
                {
                    "id": "vertex_stage",
                    "category": "Vertex Shader Stage",
                    "title": "Geometry Inputs & Vertex Deformation",
                    "summary": (
                        f"The Vertex Shader consumes `{', '.join(in_names)}` and passes interpolated data to the Pixel Shader, "
                        f"enabling {', '.join(vs_notes)}."
                    ),
                    "why_it_matters": "Shows whether mesh animation/masking relies on vertex attributes beyond standard UV0.",
                    "badges": in_names[:6],
                }
            )

    # 8. GPU Performance & Complexity Profile
    ps_insts = ps_ir.total_instructions if ps_ir else 0
    vs_insts = vs_ir.total_instructions if vs_ir else 0
    tex_samps = len(ps_ir.texture_samples) if ps_ir else 0
    if ps_insts > 1500:
        tier = "Heavy Uber-Shader (Rich Multi-Layer PBR + Global Effects)"
    elif ps_insts > 400:
        tier = "Medium-High PBR Shader"
    else:
        tier = "Lightweight / Utility Shader"

    gpu_profile = {
        "complexity_tier": tier,
        "shader_model": ps_prog.shader_model if ps_prog else (vs_prog.shader_model if vs_prog else "N/A"),
        "ps_instructions": ps_insts,
        "vs_instructions": vs_insts,
        "ps_alu_ops": ps_ir.alu_ops if ps_ir else 0,
        "ps_texture_samples": tex_samps,
        "ps_branches": ps_ir.branch_count if ps_ir else 0,
        "uses_bindless_heap": bool(ps_ir and ps_ir.uses_bindless_heap),
        "gbuffer_targets": n_targets,
        "bound_material_textures": len(pt.texture_bindings) if pt else 0,
        "bound_material_variables": len(pt.variable_bindings) if pt else 0,
    }

    return {
        "material_id": mat.object_id,
        "material_name": mat.name,
        "headline": f"{pipeline_title} — {len(pt.texture_bindings) if pt else 0} textures, {len(pt.variable_bindings) if pt else 0} parameters",
        "cards": cards,
        "gpu_profile": gpu_profile,
    }
