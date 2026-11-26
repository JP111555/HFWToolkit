# HFWToolkit — AI Agent Handoff & Engineering Knowledge Base

> **Audience:** Autonomous AI coding agents and engineers continuing development of **HFWToolkit**.
> **Purpose:** Everything you need to understand the project's mission, architecture, reverse-engineered Decima Engine & SM 6.6 DXIL internals, hard-won shader fixes (so you never regress them), environment gotchas, test/verification workflows, and the next high-value roadmap.

---

## 1. Project Mission, Philosophy & Non-Negotiable Constraints

**HFWToolkit** is a self-contained, read-only extraction, assembly, shader-research, and audio toolkit for **Horizon Forbidden West Complete Edition (PC)** (*Guerrilla Games / Nixxes — Decima Engine*).

Its five core pillars are:
1. **Rigged Machine Assembler (Blender 4.2+/5.x & UE5):** Assembles complete machines (body, destructible intact/broken armor plates, child turrets/weapons, `Skeleton` + `PhysicsSkeleton` + `SkeletonHelpers`) with DXIL-verified animated shaders (glowing optical muscle fibers & cables, dual-phase flow-map elemental liquid sacs, translucent Fresnel glass canisters, glowing eyes, Slaughterspine plasma charge-up) and unified live controls (`HFW_MachineShaderControl` in Blender, `MPC_HFW_MachineControl` in UE5).
2. **Full Character & Outfit Assembler (Blender & UE5):** Assembles Aloy (35+ Base & Burning Shores DLC outfits, headgear toggle, cinematic face & eyes, braided `HairResource`), story Companions (`Seyka`, `Erend`, `Kotallo`, `Zo`, `Alva`, `Beta`, `Varl`, `Sylens`, `Tilda`, `Regalla`, `Walter Londra`, `GAIA`, etc.), and 1,100+ modular tribal NPCs + 211 swappable heads via structured UI or a **Natural-Language Prompt Resolver** (e.g. `"Aloy Carja Blazon no headgear"`, `"Tenakth female warrior"`), producing **one unified poseable humanoid Armature** and exact DXIL Skin LUT / Hair LUT / 8-corner layered outfit shaders.
3. **In-Game Animation Extractor (`EA08`) & Non-Destructive Auto Control Rig (Blender & UE5):** Decodes HFW's Sony Edge Anim `EA08` (`SkeletonAnimationResource`) clips directly onto native Decima skeletons (including remapping 82-joint humanoid clips onto 1,881-joint unified character rigs) with **unlimited (`all`) and per-category (`combat`, `movement`, `idle`, `jump`, `emote`, `mounted`) imports, plain-English event-derived descriptions, and fast `export-pack` imports onto active rigs**, AND generates a **non-destructive `DEF + MCH + CTRL` Control Rig** across all machine archetypes and humanoids with live `Anim (0.0) ↔ Control Rig (1.0)` blending, live **`FK (0.0) ↔ IK (1.0)` limb switching**, full FK controls for **hands/fingers/claws, feet/toes/talons/hooves & machine details** on `MCH_Follow_*` dual-space sockets, **zero ghost-animation rest-lock** (`hfw_rig_lock_secondary`), 1-click bidirectional baking, and native UE5 `ControlRigBlueprint` (`Forwards Solve` + `Backwards Solve`) + `AnimSequence` FBX export.
4. **Guerrilla Developer-Directory General Asset Catalog (`1,454+` Assets):** Indexes and exports Weapons, Wildlife/Animals, Building Blocks/Ruins, Eco Trees/Foliage, Landscape rocks/snow, and Props organized in Guerrilla's original `work:Models/...` and `DLC/models/...` folder structure as Blender `.blend` or UE5 Static/Skeletal meshes.
5. **Material / SM 6.6 DXIL Shader Studio & Audio Extractor:** Interactive FastAPI Web Studio (`http://127.0.0.1:8420`) with a live Decima/UE5 shader graph viewer, texture lab, material diff tool, and a native PyAV-powered **Music / SFX / Voicelines** extractor & in-browser player (Sony ATRAC9, beat-synced multi-stem song mixdowns, subtitles CSV, optional developer directory tree preservation).

### Non-Negotiable Constraints (NEVER Violate)
- **Strictly Read-Only on Game Files:** NEVER write, modify, or delete anything inside the user's Horizon Forbidden West game installation directory (`LocalCacheWinGame/...`, `HorizonForbiddenWest.exe`, etc.).
- **Zero-Setup Portability & Multi-Patch Compatibility:** NEVER hardcode machine-specific user paths (e.g. `E:\...`, `C:\Users\...`) in tracked source files. All paths resolve through [`ToolkitConfig`](hfw_toolkit/config.py) (`HFW_GAME_DIR` env var, `hfw_toolkit_config.json`, Steam Registry AppID `2420110`, Steam `libraryfolders.vdf`, Epic manifests, drive scan). RTTI is extracted dynamically from `HorizonForbiddenWest.exe` on disk (with a bundled compressed fallback schema in `hfw_toolkit/data/hfw_rtti_builtin.json.gz`) so different HFW patches work out of the box.
- **Native Game-Packed Textures by Default:** Always keep `unpack_textures=False` as the default; Blender and UE5 unpack channels directly in shader nodes (`--unpack-textures` remains an opt-in export mode).
- **Blender + UE5 Shader Parity:** Any material, shader, or profile fix implemented in `blender_builder.py` MUST be mirrored in `hfw_toolkit/shaders/unreal_export.py` (and generated UE5 scripts MUST pass `ast.parse()`).
- **No Copyrighted Assets in Git:** `.gitignore` excludes `cache/`, `exports/`, `hfw_toolkit_config.json`, `.venv/`, and `bin/dxcompiler.dll`. Never commit extracted game meshes, textures, audio, or `.sqlite` caches.
- **Visual + Automated Verification Standard:** After any shader, animation, rigging, or geometry change, ALWAYS render headless Cycles verification previews in Blender AND visually inspect the rendered PNGs, AND run the full `pytest -v` regression suite (**currently 46 / 46 passing**).

---

## 2. Repository Architecture & End-to-End Data Flow

### Canonical Bundle Pipeline
All three asset kinds (`"machine"`, `"character"`, `"asset"`) share **one unified pipeline**:

```text
HFW Game Files (read-only DSAR v3.1 LZ4 archives + streaming_graph.core)
  │
  ▼
hfw_toolkit/decima/       (BinaryStorageFile, RTTISchema, StreamingGraph, HFWGame, AssetCatalog)
  │
  ▼
hfw_toolkit/extract/      (machine_assembler.py | character_assembler.py | asset_assembler.py)
  │  └─ Produces MachineAssembly (skeleton, parts, materials, variant/official metadata)
  ▼
hfw_toolkit/blender/bundle.py :: export_assembly_bundle(..., asset_kind=...)
  │  └─ Writes a self-contained Canonical Assembly Bundle directory:
  │       ├── assembly.json          (schema_version=3: joints, helpers, parts, materials, shader_controls)
  │       ├── geometry.npz           (NumPy float32/uint32 vertex positions, normals, UVs, colors, skin weights)
  │       ├── textures/*.png         (native packed BCn-decoded PNGs + sub-material/LUT/detail maps)
  │       └── ue5_import_{machine|character|asset}.py   (self-contained UE5 Python importer)
  │
  ├──▶ hfw_toolkit/blender/blender_builder.py   (Pure bpy + numpy + stdlib; zero external deps!)
  │       ├── Headless CLI (`build_blend_headless`) -> `.blend` + optional rigged `.fbx` for UE5
  │       └── Live Blender Add-on (`import_bundle_into_scene`) -> imports directly into active scene
  │
  └──▶ Unreal Engine 5 (`Tools -> Execute Python Script...` -> `ue5_import_*.py`)
          └─ Imports FBX (`SK_` or `SM_`), textures, `MPC_HFW_MachineControl`, and full Material graphs
```

### Complete Module Map

| Path | Role & Key Symbols |
|---|---|
| `hfw_toolkit/config.py` | `ToolkitConfig`, `auto_detect_game_dir()`, `auto_detect_blender_exe()`, `is_valid_hfw_dir()`. Loads/saves `hfw_toolkit_config.json` (gitignored). |
| `hfw_toolkit/cli.py` | Entry point `hfw-toolkit`. Subcommands: `doctor`, `setup`, `dump-rtti`, `index`, `machines`, `animations {list,export-pack}` (`--category`, `--query`, `--limit 0`, `--json`), `export-bundle`, `export-blend`, `export-ue5` (all supporting `--animations all|category:<cat>|N|ids` and `--no-control-rig`), `inspect-material`, `install-blender-addon`, `web`, `audio {stats,index,list,extract}`, `characters {list,outfits,resolve,export-bundle,export-blend,export-ue5}`, `assets {tree,list,export-bundle,export-blend,export-ue5}`. |
| `hfw_toolkit/decima/dsar.py` | `BinaryStorageFile`, `StreamingStorage` — thread-safe random-access **DSAR v3.1 LZ4** chunked archive reader (`lz4.block`) with LRU chunk cache. |
| `hfw_toolkit/decima/exe_rtti.py` | `extract_rtti_from_exe()`, `get_exe_version_string()` — static PE64 `.rdata`/`.data` scanner extracting 12,173+ Decima RTTI descriptors in ~0.12s (cached to `cache/rtti_exe_v*.json.gz`). |
| `hfw_toolkit/decima/rtti.py` | `RTTISchema` — resolves Decima type hierarchies, MurmurHash3 type hashes, compound fields, enums, containers. Falls back to `hfw_toolkit/data/hfw_rtti_builtin.json.gz`. |
| `hfw_toolkit/decima/reader.py` | `BinaryReader`, `TypeReader`, `ObjectId("group:index")`, `ObjectRef`, `UUIDRef`, `StreamingRef`. |
| `hfw_toolkit/decima/callbacks.py` | Decima custom binary deserializers (e.g., `Texture`, `Mesh`, `VertexArrayResource`, `IndexArrayResource`, `RenderEffectResource`, `EdgeAnimSkeleton`, `WaveResource`, etc.). |
| `hfw_toolkit/decima/streaming_graph.py` | `StreamingGraph`, `HFWGame` — parses `streaming_graph.core` & `streaming_links.stream`; `read_object(oid)`, `read_group(gid)`, `resolve_uuid(uuid)`, `read_stream_data(sref)`. |
| `hfw_toolkit/decima/catalog.py` | `AssetCatalog` backed by `cache/hfw_index.sqlite`. Tables: `meta`, `entities`, `texture_sets`, `shaders`, `entity_texture_sets`, `entity_variants`, `audio_music_cues`, `audio_groups`, `audio_voices`, `audio_assets`, `characters`, `character_outfits`, `character_heads`, `dev_assets`. |
| `hfw_toolkit/extract/skeleton.py` | `SkeletonData`, `extract_skeleton()` (`EdgeAnimSkeleton` tag `ES06`), `extract_physics_skeleton_world_matrices()`, `add_skeleton_helpers()`, `decima_mat34_to_np()`, `decima_mat44_to_np()`. |
| `hfw_toolkit/extract/animations.py` | Sony Edge Anim `EA08` binary decoder (`decode_edge_anim_ea08()`, `decode_animation_resource()`), unlimited & category-filtered discovery (`discover_entity_animations()`, `summarize_animation_categories()`, `resolve_animations_for_export()` supporting `all` / `category:<cat>` / `N` / IDs, `export_animation_pack()`), event-derived plain-English clip descriptions, and exact Blender `matrix_basis` skeleton remapping (`remap_animation_to_blender_basis()`, including 82j → 1,881j character rigs). |
| `hfw_toolkit/extract/meshes.py` | Vectorized NumPy vertex/index/skinning decoder (`decode_mesh_hierarchy()`), handling `StaticMeshResource`, `RegularSkinnedMeshResource`, `MultiMeshResource`, bone-palette remapping, multiple UV/color channels. |
| `hfw_toolkit/extract/textures.py` | `decode_texture()` via `texture2ddecoder` (BC1–BC7, ASTC, RGBA, R8, etc.), `TextureSet` unpacking & channel-packing metadata decoder. |
| `hfw_toolkit/extract/materials.py` | `parse_material(game, render_effect_obj)` -> `MaterialInfo`, `TechniqueInfo`, `TextureBindingInfo`, `decode_texture_binding_packing()`. |
| `hfw_toolkit/extract/machine_assembler.py` | `assemble_machine()`, `resolve_machine_entity_id()`, `list_machine_variants()` — deduplicated machine catalog, destructible intact/broken parts, child turrets/weapons, Focus Catalog official part names. |
| `hfw_toolkit/extract/character_assembler.py` | `assemble_character()`, `resolve_character_prompt()` — merges Body + Outfit + Headgear + Facial (`Head`) + `HairSkinnedMesh` onto ONE unified skeleton (`C_Head_sjnt_0`), Bone Collections `Body` / `Face` / `Cloth_Hair`. Accepts both `facial_id` and legacy alias `facial_override`. |
| `hfw_toolkit/extract/asset_assembler.py` | `assemble_general_asset()` — resolves arbitrary Guerrilla dev-directory assets, upgrades DepthOnly occluder proxies to textured `MultiMeshResource`, fixes `HasBindPose=False` weapon skeletons. |
| `hfw_toolkit/extract/audio.py` | Native PyAV (`av`) Sony ATRAC9 / PCM / ADPCM / XWMA / MP3 / AAC decoder; `MusicProject` cue & beat-synced mixdown exporter; SFX `GraphSoundResource`; Voicelines `LocalizedSimpleSoundResource` (12 spoken languages + subtitles CSV); directory-preserving or `--flat` extraction. |
| `hfw_toolkit/shaders/hw_shader.py` | `parse_hw_shader()` — extracts SM 6.6 DXIL bytecode from Decima `ShaderResource` (`HwShader`) blobs. |
| `hfw_toolkit/shaders/dxc.py` | `DXCDisassembler` — wraps `dxcompiler.dll` (`bin/dxcompiler.dll` or Windows SDK) via `ctypes` COM to disassemble DXIL to LLVM IR (`cache/dxc/*.dxil.ll`). |
| `hfw_toolkit/shaders/character_dxil.py` | SSA symbolic DXIL evaluator (`extract_dxil_outfit_corners()`), `infer_character_skin_tone()` (`570:0` Skin LUT), `infer_character_hair_color()` (`382:0` Hair LUT). |
| `hfw_toolkit/shaders/dxil_ir.py`, `dxil_graph.py`, `node_graph.py`, `pseudo_hlsl.py`, `explainer.py`, `hash_dict.py` | Deep DXIL instruction analyzer, specialty node detector (Fresnel, FlowMap, Panner, Sine, Smoothstep, Lerp, Saturate, Pow), visual Decima/UE5 graph builder, pseudo-HLSL emitter, human-readable shader explainer. |
| `hfw_toolkit/shaders/unreal_export.py` | `generate_ue5_machine_script()` (generates `ue5_import_{machine|character|asset}.py`), `generate_ue5_python_script()`, `generate_ue5_clipboard_t3d()`. |
| `hfw_toolkit/blender/bundle.py` | `export_assembly_bundle()`, `infer_machine_variant_profile()`, material classification, special shader & overlay detection, texture/LUT/detail export. |
| `hfw_toolkit/blender/blender_builder.py` | **Zero-external-dependency** Blender scene, armature, mesh & shader-node builder (`import_bundle_into_scene()`, `_build_material()`, `_build_layered_outfit_material()`, `_ensure_decima_pbr_node_group()`, `_ensure_muscle_fiber_group()`, `_ensure_muscle_cables_group()`, `_ensure_machine_shader_control()`, `_ensure_character_control()`, plus deform-only FBX & per-clip `animations/*.fbx` export). |
| `hfw_toolkit/blender/control_rig.py` | **Zero-external-dependency** universal HFW skeleton analyzer (`analyze_rig_chains()`, covering **all 43/43 HFW machine species** with zero `creature` fallbacks across all 8 archetypes: `theropod`, `quadruped`, `flyer`, `hexapod`, `aquatic`, `serpentine`, `humanoid`; detecting chest anchors, Head IK neck chains, long-chain B-Bone Spline IK candidates, closed-loop jaw/horn chainsaws, rotary drills/grinders/rotors/Tidal Discs, trunks/tongues, bat-wings, glider ribs, 6-leg crab rigs, flippers, and wolf/cat scapulae/shoulders) + **`_analyze_armature_mesh_geometry()`** (single-pass per-bone anatomical cross-section radii + 3D PCA mechanical spin-axis detector so every widget hugs its body part snugly) + **`compute_dynamic_collection_labels()`** + non-destructive `DEF + MCH + CTRL` Control Rig builder (**v1.0.0**: cycle-free single-armature **B-Bone Spline IK** (`hfw_ik_fk_spline`, `pose_serpent_wave("wave"|"coil")`, rigid scale `1.0`) for Slitherfang's 40-bone body, long tails, and Tremortusk's 11-bone trunk; **pole-free Toggleable Head & Neck IK** (`hfw_ik_fk_head`) with **Chest ↔ World space switch** (`hfw_head_ik_space`); Rigify `super_finger`-style **Master Finger/Toe/Talon Curl Levers** (`CTRL_Curl_*` + additive `MCH_Curl_*` + tucked `hf_tweaks` phalanx rings); **2-stage Simple-Expression Closed-Loop Stadium Chainsaw Rig** (`CTRL_Saw_Spin` + `hfw_saw_speed`, `0.01 mm` rest drift, 0 dependency cycles); **Universal PCA-Axis Rotary Drill/Grinder/Rotor/Tidal-Disc Rig** (`CTRL_Spin_*` + `hfw_drill_speed`); multi-bone Chest/Back & Cobra Hood IK↔FK, independent Quadruped Front/Rear Leg IK↔FK, Flyer Dual Wing IK↔FK, 2-pass exact SO(3) pose mirroring, zero-pop `switch_limb_ik_fk()`, `HFW_RestLock_*`), Blender 5.3 Layered Action importer, and 1-click bidirectional baker. |
| `hfw_toolkit/blender/headless.py` | `build_blend_headless()` — invokes `blender.exe --background --python blender_builder.py -- ...`. |
| `hfw_toolkit/blender/addon/__init__.py` | Blender N-Panel Add-on (**v1.0.0**, `HFW Toolkit` sidebar tab): Machines, Characters (prompt + browser), Asset Catalog tabs, **Live Animation Browser (`HFW_UL_AnimationList`)** with category counts, search, plain-English descriptions, All/Category/Selected modes & fast `Import Animations Onto Active Rig`, **Intuitive Control Rig & Animations** panel (`Game Anim` / `Control Rig` switch, auto-snapping `Torso/Hood`, **`Head [Head IK / Neck FK]` + `Head Space: Chest ↔ World`**, **`Spline [Spline IK / Chain FK]` + `[S-Wave Pose]` / `[Coil Pose]`**, `Front Legs`, `Rear Legs / Legs`, `Arms / Forelimbs / Front Flippers`, and `Wings` IK/FK switches, **Chainsaws & Rotary Drills** auto-spin sliders + 1-click `[Rev]`/`[Spin]`/`[Stop]` buttons, 2-column dynamically named Control Group toggles incl. `Head & Neck IK`, `Spline IK`, `Chainsaws & Drills`, and `Tweaks`, `In Front` toggle, 1-click `Chest`/`Hood`, `Head`, `Spline`, `Curls`, `Neck`, `Tail`, `Wings`, `Trunk`/`Tongue`, and `Spine` selectors, **`Mirror L->R` / `Mirror R->L`**, `Lock Uncontrolled Bones`, collapsible sliders, 1-click Bake/Rebuild/Reset), Live Machine & Character Shader Controls. Installed to Blender 4.2–5.3 via `hfw-toolkit install-blender-addon`. |
| `hfw_toolkit/web/app.py` & `web/static/*` | FastAPI backend (`app.py`, 41 REST endpoints including `GET /api/animations/{kind}/{target}` with category counts & descriptions) + Vanilla JS SPA (`index.html`, `app.js`, `assets_lab.js`, `audio_lab.js`, `shader_graph.js`, `texture_lab.js`, `styles.css`). |
| `tests/` | 8 pytest modules (`test_animations_and_rigging.py`, `test_audio.py`, `test_blender_headless.py`, `test_decima_core.py`, `test_extract_machine.py`, `test_shaders.py`, `test_v9_shaders_chars_assets.py`, `test_web_and_cli.py`) — **46 integration & unit tests**. |

---

## 3. Decima Engine (HFW PC) Internals & Hard-Won Lessons

### Storage, Addressing & Streaming
- **Archives:** Game data lives in `<GameDir>/LocalCacheWinGame/package/*.core` and `*.core.stream`, compressed with **DirectStorage Archive (DSAR v3.1)** wrapping **LZ4 block** chunks (`dsar.py`).
- **Graph Index:** `streaming_graph.core` maps ~85,000+ `GroupMeta` entries (`GroupID`), each containing objects indexed `0..NumObjects-1` and referencing dependency `SubGroups`.
- **Object Addressing:** Objects are uniquely addressed by `ObjectId(group_id, object_index)` formatted `"group:index"` (e.g. `"21350:2823"`). Cross-group references use `UUIDRef` (resolved via `RootUUIDs` / `streaming_links.stream`) or `ObjectRef`. Bulk vertex/index/texture/audio payloads live in `*.core.stream` addressed via `StreamingRef` + `LocatorTable` (24-bit file index, 40-bit byte offset).
- **Group Reading API:** On `HFWGame`, read a single object with `game.read_object(ObjectId.parse("g:i"))` (pass `read_subgroups=True` when you need external refs/texture headers resolved) and read a full group with `game.read_group(group_id)` (there is NO `iter_group_objects`).
- **Textures vs TextureSets:** A `TextureSet` object (e.g. `181:0`) is a *container* whose entries reference individual `Texture` objects (`181:5`, `181:1`). Calling `decode_texture` on a `TextureSet` returns `None` — always decode the child `Texture` objects!

### Coordinate System, Skeletons & Geometry
- **Coordinate System:** Decima is ALREADY **right-handed Z-up in meters**, matching Blender 1:1 with **no axis swap**!
  - Both Characters and Machines face **`+Y`** (so front-facing cameras belong at `+Y` looking toward `-Y`; character right temple = `-X`, character left temple = `+X`).
- **Skeletons (`ES06` + `PhysicsSkeleton` + `Helpers`):**
  - `EdgeAnimSkeleton` (`ES06`, magic `0x45533036`) stores local bind-pose joints. Some parts (like Thunderjaw weapons/armor) require `PhysicsSkeleton` world matrices and `SkeletonHelpers` (`OrientationHelper` + `Mat44` offset) so bone-parented static armor/weapons land on their bones instead of world origin `(0, 0, 0)`.
  - **`HasBindPose=False` Weapon Trap:** On rigged weapons like `DLC/models/weapons/Bow_Heavy_Quen_SE`, `SkinnedModelResource.Skeleton` (`85387:13`) has `HasBindPose=False` (all 19 joints at `(0,0,0)`!), while `RegularSkinnedMeshResource.Skeleton` (`85387:1`) and `InitialRepresentationSkeleton` contain the real 35-joint bind pose. `asset_assembler.py` explicitly prioritizes skeletons with non-zero joint translations and remaps skin weights onto the chosen master skeleton.
- **Machine Completeness & Deduplication:**
  - Many machines have multiple entity variants in the archives (some incomplete/part-only). `catalog.py` + `machine_assembler.py` deduplicate machines into one canonical entry per machine with selectable variants (e.g. Clawstrider Standard/Acid/Fire/Apex), pull base `SkinnedModelResource` body parts cleanly (fixing missing Redeye Watcher legs/body), and put damaged/broken states into the hidden `Plates_Broken_States` collection.
- **Dead `UV1` Guard & `uv_index` Selection:**
  - In `bundle.py`, ONLY `"uvset02"` in a texture/material name maps to `uv_index = 1` (e.g. `AloyHair_Cards_UvSet02_CLR`). NEVER match generic `"uv2"` — e.g., `SeykaTrimsUV2_NMT` has an empty `[0, 0]` `UV1` buffer and actually renders from `UV0`!
  - In `blender_builder.py` (`_build_part_mesh_object`), any UV layer `u_i >= 1` with `np.ptp(uv) < 1e-5` automatically mirrors `UV0` as a safety net.

---

## 4. DXIL Shader & Texture Knowledge Base (DO NOT REGRESS!)

Every fix below was verified directly by disassembling HFW's SM 6.6 DXIL shaders and visually inspecting Cycles renders.

### 4.1 Core Decima PBR (`HFW_DecimaPBR`, DXIL `16236:2` L592–782)
- **Decima `Roughness` / `GLS` Channel is SMOOTHNESS:** Always invert to PBR roughness (`Roughness = 1 - Smoothness`).
- **Worn Bare Metal Edge Recovery:** Decima uses a Specular/Diffuse workflow where chipped bare metal edges have `CLR ≈ (0, 0, 0)` and high `Reflectance` (`0.60..0.90`):
  - `F0_raw = Reflectance^2`
  - `m_decima = clamp(F0_raw * 1.11 - 0.111, 0.0, 0.98)`
  - `F0_rgb = lerp(clamp(F0_raw, 0, 0.10), chroma(CLR) * F0_raw, m_decima)`
  - Mapped cleanly to `Principled BSDF` (`Base Color`, `Metallic`, `Roughness`, `Specular IOR Level`) in Blender and equivalent `ConstantBiasScale` + `Lerp` in UE5.
- **`_DIELECTRIC_PROFILES` Guard (CRITICAL!):**
  - `_DIELECTRIC_PROFILES = ("hair", "hair_face", "skin", "eye", "teeth", "foliage", "snow", "rock_volcanic", "rock_tint")`
  - **NEVER** pipe `Reflectance` into `HFW_DecimaPBR` / `Metallic` for these profiles!
  - **Why:** Aloy's hair (`181:5` CLR + `181:1` RFL/RGH) has consistent natural auburn RGB root-to-tip in `181:5`, while `181:1.R` (`Reflectance`) jumps from `0.31` to **`0.74–1.00`** at the hair tips (it is a hair sheen/trans-scatter mask, NOT metal). Piping `181:1.R` into `Metallic` turned the bottom 30% of Aloy's hair into glowing neon-orange copper metal!
  - With the guard, hair forces `Metallic = 0.0`, maps `181:1.G` to silky roughness (`0.38–0.56`), and routes `181:1.R` to subtle dielectric sheen (`Coat Weight = rfl * 0.16`, `Anisotropic = 0.65` in Blender; `Specular 0.42–0.72` in UE5).

### 4.2 Machine Shaders (Blender + UE5 `MPC_HFW_MachineControl`)
- **Optical Muscle Fibers (`19501:0`) & Muscle Cables (`8260:1`):**
  - `19501:0` renders additive fiber strands over `8260:1` dark cables, multiplying `Mask_Alpha * 40` and sampling `5147:0` (a 64×16 Linear ramp LUT that is **~85% zero** with sparse moving waves in `B` and spark flecks in `A`).
  - **View-Angle Darkening (CRITICAL!):** Both `19501:0` (`saturate(-5.0 * dot(N, V))` applied twice!) and `8260:1` (`saturate((-0.25 - dot(N, V)) * 4.0)`) fade curved outer silhouettes to **dark metallic carbon cables**! Do NOT invert `LayerWeight.Facing` (which makes edges glow) and do NOT add a high wave floor (like `0.55`, which destroys all contrast and makes muscles a blazing solid color).
  - **3-Tier Thermal Ramp:** Driven live from `HFW_MachineShaderControl.Muscle Color` (`pow(c, 2.2) * 0.38` deep burnt ember → vibrant muscle color → hot bright spark core), calibrated at `muscle_glow = 2.6..2.8` and `cable_glow = 3.0..3.2`.
  - **Cycles Transparent Bounces:** Always ensure `scene.cycles.transparent_max_bounces >= 16` so overlapping fiber cards never hit blackouts in Cycles!
- **Variant Profiles (`infer_machine_variant_profile`):**
  - `standard`: Warm amber muscles `[0.95, 0.28, 0.02]`, patrol blue eyes `[0.04, 0.56, 1.0]`.
  - `apex`: Purple-violet muscles `[0.54, 0.05, 1.0]`, apex purple eyes `[0.64, 0.05, 1.0]`.
  - `corrupted`: Crimson red muscles & eyes.
  - `overridden`: Electric cyan muscles & eyes.
- **Elemental Liquid Sacs & Canisters:**
  - `elemental_sack_overlay`: Exact DXIL `mat_id == 0 * Mask_Alpha` + true **Dual-Phase Flow Map** (`172:0` flow vector) distorting swirling fluid caustics (`1703:0`) + Fresnel rim depth darkening + glossy wet membrane roughness (`0.12`).
  - Element color inference prioritizes official **Focus Catalog part names** (e.g., `"Metalbite Sac"` -> Acid Green, `"Purgewater Sac"` -> Aqua Blue, `"Blaze Sac"` -> Fire Orange, `"Chillwater Sac"` -> Cryo Cyan, `"Plasma"` -> Violet, `"Adhesive"` -> Yellow) before mesh/variant hints so sacs match each machine variant's true element instead of defaulting blue.
  - `canister_emissive_overlay`: `mat_id == 1` (`sin(Time * 8)` pulsing collar) + `mat_id == 20` (fluid window).
  - `canister_glass`: `BLEND_Translucent` / `Transmission`, IOR `1.45`, Fresnel rim opacity.
- **Slaughterspine Plasma Charge-Up (`plasma_charge_overlay`):**
  - Exact DXIL `VertexColor.R * ((1 - AO)^2 * 0.78 + 0.22)` × Head→Tail surge wave (`sin(Time * 3.5 - VertexColor.G * 10)`) + `mat_id == 1` body conduits, driven by live `hfw_plasma_charge` (`0..1`) property on the Armature (`HFW_PlasmaChargeControl` in Blender, `MPC_HFW_MachineControl.PlasmaCharge` in UE5).

### 4.3 Character Shaders (`character_dxil.py`, `bundle.py`, `blender_builder.py`, `unreal_export.py`)
- **Human Skin Tone LUT (`570:0`, DXIL `48346:288`):**
  - `Head_CLR` and body `*_CLR` textures are `0.5`-centered neutralized variation maps (`mean_srgb ≈ 0.50`).
  - Exact formula: `pow(2.0 * SkinLUT_srgb * Head_CLR_srgb, 2.2)` = **`4.5948 * SkinTone_linear * Head_CLR_linear`**.
  - Driven globally across Head + Body by `HFW_CharacterControl` (`Skin Tone`, `Skin Brightness`, `Subsurface Weight`, `Hair Color`) so there is zero neck seam.
  - Skin/eyes/teeth always set `allow_alpha_cutout = False`.
- **Character Hair LUT (`382:0`):**
  - Non-Aloy companion/NPC hair cards (e.g., Seyka's `QuenHair_Alpha`) have no baked RGB albedo; they tint via `HFW_CharacterControl.Hair Color` modulated by strand ID (`*_ID.png`) and AO (`*_AO.png`).
- **8-Corner Layered Outfits (`outfit_dxil` in `character_dxil.py`):**
  - **Exact DXIL 8-Corner `MSK.RGB` Cube Order:**
    `0=(0,0,0) Black`, `1=(1,0,0) Red`, `2=(0,1,0) Green`, `3=(0,0,1) Blue`, `4=(0,1,1) Cyan`, `5=(1,0,1) Magenta`, `6=(1,1,0) Yellow`, `7=(1,1,1) White`.
    *(Note: in trilinear `R, G, B` cube interpolation, `(1,1,0)` is **L6 Yellow** and `(0,1,1)` is **L4 Cyan** — do NOT swap L3/L4/L5/L6!)*
  - **Multi-Technique Scanning:** Burning Shores DLC characters (e.g. Seyka) often store the Deferred GBuffer 8-case `switch` shader on a non-primary technique. `extract_dxil_outfit_corners()` scans all candidate techniques.
  - **Dynamic Metallic F0 Detection:** At the post-switch phi merge, `character_dxil.py` locates the exact DXIL `0.2 - F0` (`0x3FC99999A0000000`) instruction to distinguish:
    - 3-phi RGB F0 (`[-9, -8, -7]`) — standard metal + dielectric mixed outfits
    - 1-phi scalar F0 (`[-7]`) — e.g. Aloy Nora Anointed sleeves/pants (prevents leather from falsely turning metallic!)
    - 0-phi pure dielectric (`[]`) — e.g. Seyka tassels (prevents cloth tassels from falsely turning metallic!)
  - **Linear `0.5`-Centered Sub-Materials:** `Shaders_HRZ2/Materials/*_CLR` textures MUST be loaded as **`Non-Color` / Linear** and expanded via `pow(2.0 * tex, 2.2) * DyeColor_linear` (with `*_CLR_AO.A` folded subtly as micro-AO `0.72..1.00`).
  - **Full-Map PBR Micro-Detail (v11):** Every active corner also samples its `16×` tiled `*_NMT_RGH_HGT.png` (cached per filename):
    - `R, G` -> trilinearly blended across all 8 corners and UDN-added onto the 2K macro normal (`macro_xy + (micro_xy - 0.5) * 0.55`).
    - `B` (`GLS` smoothness) -> mapped into calibrated per-material-class roughness:
      - Metal: `0.16..0.44` | Shell/Bead/Stone: `0.18..0.46` | Leather: `0.34..0.68` (Buckskin/Suede: `0.54..0.78`) | Carbon/Rubber: `0.32..0.64` | Silk/Satin: `0.42..0.68` | Wool/Cotton/Crochet: `0.66..0.88`.
    - 2K `Curvature` (`NMT_AO_CRV.A`) -> `0.78..1.24×` edge highlight / cavity contrast.
    - 2K `*_MISC.G` -> dirt roughness lift (`+0.16` in Blender / `Lerp(..., 0.88, misc.G * 0.45)` in UE5).
  - **Stem-Matched Normal & MISC Selection (CRITICAL!):** A single material like `SeykaTrimsUV2_NMT` can bind BOTH `SeykaTrims_NMT_1631_6_NMT_AO_CRV` (blank at UV2 trim coordinates!) AND `SeykaTrimsUV2_NMT_1631_1_NMT_AO_CRV`. `bundle.py` matches `nrm_entry` and `misc_entry` to the **same filename stem as `msk_entry`** first!
  - **Hybrid `outfit_dxil` + Painted Macro `_CLR` (`macro_color_mode`):**
    - `"multiply2"`: When the macro `_CLR` has a neutral mid-grey background (`median RGB` in `165..205`, spread `< 10`, i.e., `linear ≈ 0.50`, like `SeykaTorso_NMT_1631_33_CLR.png`), compute `clamp(2.0 * macro_color * (8-corner dye * micro_detail))` so dark fish-leather collars (`L1`), aqua wool (`L3`), and steel clasps (`L4`) keep their DXIL colors while the painted coral scarf overlays cleanly!
    - `"replace"`: When the macro `_CLR` is a direct full-color painted trim sheet (e.g. `SeykaTrimsUV2_NMT_1631_17_CLR.png`, median `~84`), compute `macro_color * micro_detail`.

### 4.4 General Dev-Directory Assets
- **DepthOnly Proxy Upgrading (`asset_assembler.py` & `materials.py`):** 0-binding DepthOnly/Shadow techniques never win over textured passes; occluder proxy models automatically upgrade to the richest textured `MultiMeshResource` in the group.
- **Unique `Detail_*` Texture Retention (`bundle.py`):** Weapons like `Models/weapons/Axe_Nora` (`Axe_Nora_Tile_CLR` `61749:92`) bind their actual Color/Normal/RFL/RGH with `tb.is_detail_texture = True`. Always retain `is_detail_texture` bindings whose source path lives under `models/...` or `dlc/models/...`!
- **Weather Overlay Suppression:** Suppress shared world overlays (`StarMoss_clr`, `snow_clumps_`, `snow_heavyfresh_`) when a non-snow asset already has unique `models/...` textures.
- **Volcanic & Sandstone Landscape Rocks (`rock_volcanic`, `rock_tint`, DXIL `8490:1`):**
  - `_TNT` textures are 2-channel **Linear RG** tint/weathering blend maps (`B == 0`) — never display raw RG! Reconstruct charcoal basalt (`rock_volcanic`) or sandstone (`rock_tint`) albedo from `R` + `G`.
  - `_NMT_RGH_AO.B` packs `_LUM` (`mean ≈ 0.525`); DXIL `8490:1` L1020 caps dry rock smoothness at `0.40`. Remap smoothness to dry porous rock roughness **`0.66..0.92`** (eliminating wet-plastic shine!).
  - Export the 2K standalone tiling detail normal (`asset_detail_normal_file`, e.g. `5211:1`) and UDN-blend it at `8×` UV onto the 4K macro normal in both Blender and UE5 (`AssetDetailNormal8x`).

### 4.5 Audio / Music / SFX / Voicelines (`hfw_toolkit/extract/audio.py`)
- **In-Process Decoding:** Uses PyAV (`av>=14.0.0`) to decode Sony **ATRAC9** (`.at9`), PCM, PCM_FLOAT, ADPCM, XWMA, MP3, and AAC directly in Python (zero external FFmpeg/vgmstream executables).
- **Music (`_MUSIC_GROUPS = (3, 1, 78260, 84481)`):** Walks `MusicProject` trees (1,346 cues / 4,913 stems across Base Game, Burning Shores DLC, Photomode, Tilda's Music Player, Plainsong Chorus) with optional beat-synced full-song mixdowns (`--no-stems` for single combined track per cue).
- **SFX (`70,000` waves) & Voicelines (`95,392` lines, 12 spoken languages):** Links `WaveResource` via `GraphSoundResource -> GraphProgramResource` and `LocalizedSimpleSoundResource` via `SentenceResource -> VoiceResource + LocalizedTextResource` (exporting `voicelines_transcript.csv`). Supports `--flat` or preserved developer directory trees (default).

### 4.6 Sony Edge Anim (`EA08`) Animations & Non-Destructive Control Rig (`animations.py`, `control_rig.py`, `unreal_export.py`)
- **`EA08` Binary Specification (`SkeletonAnimationResource.AnimationData`):**
  - Tag `0x45413038` (`b"80AE"`), MSB-first bitstream, 32-bit `PackedSpec` storing `(recomp_slot = spec & 3)` and three 10-bit component specs **read in MSB-to-LSB order `(c2, c1, c0)`** (bits `22..31`, `12..21`, `2..11`; each 10-bit field = `[sg:1][eb:4][mb:5]`).
  - Universal decode formula for BOTH rotations (`qscale = 1.0`, smallest-three quaternion reconstruction at slot `recomp`) and translations/scales: `eb == 0` -> two's-complement fixed-point `raw / 2^mb`; `eb > 0` -> truncated IEEE-754 float with `bias = 2^(eb-1) - 1`.
  - Each FrameSet starts with 8 `u32` byte sizes (`baseR/T/S/U`, `intraR/T/S`, `pad`), followed by BaseFrame channels, a single **bit-contiguous 1-bit keyframe presence bitmask** (`ceil((aR+aT+aS+aU)*n_intra/8)` bytes), and channel-major IntraFrame absolute values interpolated via Slerp/Lerp across unkeyed frames (including FrameSet boundaries).
  - Always use `game.read_group_local(oid.group_id)` when inspecting `SkeletonAnimationResource` so external `ObjectRef.target_id` links (`Skeleton`, `Events`) are preserved without loading recursive subgroups!
- **Why Rigging NEVER Breaks Game Animations (`DEF + MCH + CTRL`):**
  - Original Decima deform bones (`DEF`) are NEVER renamed, reparented, or re-oriented; imported `EA08` Actions write directly to `DEF` pose bones via Blender 5.3's Layered Action Channelbag API (`action.slots` + `action.layers` + `strip.channelbag(slot, ensure=True)` + `foreach_set("co", ...)`).
  - Hidden `MCH_IK_*` chains solve 2-bone (human/quadruped) and 3-bone digitigrade (theropod/flyer) IK cleanly along exact limb chords, with `MCH_Bind_*` children mirroring each `DEF` bone's exact Decima rest matrix and auto-calibrated `pole_angle` (`< 0.02 mm` rest drift on Clawstrider, Behemoth, Aloy; `< 3 mm` on 6.4m Slaughterspine).
  - Every `DEF` bone constraint's `.influence` is driven by `hfw_rig_master * hfw_rig_<limb>` (`0.0` = raw HFW game anim, `1.0` = Control Rig IK/FK).
  - Bidirectional 1-click baking transfers raw `DEF` animations onto editable `CTRL` handles (`bake_action_to_control_rig`, `0.00 mm` hips/foot round-trip error) and bakes edits back onto `DEF` (`bake_control_rig_to_deform`) for UE5/FBX export (`use_armature_deform_only=True`).
  - UE5 Parity (`unreal_export.py`): `ue5_import_*.py` imports all exported `animations/*.fbx` clips as UE5 `AnimSequence`s AND creates a native `CR_<AssetName>` `ControlRigBlueprint` with **Forwards Solve** (FK + `RigUnit_PBIK` Full-Body IK) and **Backwards Solve** (`RigUnit_InverseExecution`, enabling UE5 Sequencer's built-in `"Bake To Control Rig"`).

---

## 5. Critical Environment & Tooling Gotchas (Save Hours!)

1. **Blender Python Does NOT Have `lz4`!**
   - `hfw_toolkit/blender/__init__.py` imports `bundle.py` (which imports `lz4`, `PIL`, etc.). Blender 5.3's bundled Python 3.13 only has `bpy`, `mathutils`, `numpy`, and stdlib.
   - Therefore, any script running *inside* `blender.exe` MUST load `blender_builder.py` via `importlib.util.spec_from_file_location("hfw_blender_builder", ".../blender_builder.py")`, NEVER via `from hfw_toolkit.blender.blender_builder import ...`!
   - Keep `blender_builder.py` and `addon/__init__.py` strictly free of third-party non-Blender imports!
2. **Always Pass `--factory-startup` in Headless Blender Scripts:**
   - Avoids loading slow user add-ons and guarantees clean, deterministic execution (`blender.exe --factory-startup --background --python-exit-code 1 --python script.py`).
3. **NumPy 2.x Removed `ndarray.ptp()`:**
   - The `.venv` runs Python 3.14 + NumPy 2.x. Always write `np.ptp(arr)` instead of `arr.ptp()`.
4. **PowerShell Inline Python Quoting:**
   - On Windows PowerShell, inline `-c "..."` mangles nested quotes/backslashes. Always pipe a single-quoted here-string:
     ```powershell
     @'
     import json
     print("clean quoting!")
     '@ | & ".\.venv\Scripts\python.exe" -
     ```
5. **Syncing the Blender Add-on:**
   - The installed add-on is a *copy* at `%APPDATA%\Blender Foundation\Blender\<ver>\scripts\addons\hfw_toolkit_addon.py`. After editing `hfw_toolkit/blender/addon/__init__.py`, ALWAYS run:
     ```powershell
     .\.venv\Scripts\hfw-toolkit.exe install-blender-addon
     ```
6. **DXC Disassembler (`dxcompiler.dll`):**
   - `hfw_toolkit/shaders/dxc.py` searches `bin/dxcompiler.dll` first, then Windows SDK installs. Disassembled LLVM IR is cached under `cache/dxc/<hash>.dxil.ll` (fast subsequent runs). If `dxcompiler.dll` is absent, character outfit extraction falls back gracefully to `outfit_fallback`.
7. **Blender 6.0 Deprecation Watch:**
   - `world.use_nodes = True` emits a deprecation warning in Blender 5.3 (removed in Blender 6.0). New render/builder helpers should check/avoid setting it unnecessarily.
8. **Current Git Branch Note:**
   - Active development lives on the **`Development`** branch (tracking `origin/Development`). Always verify `git status -s` and `git branch` before committing or pushing.

---

## 6. Essential Developer Workflows & Copy-Paste Commands

### Run Diagnostics & Full Regression Suite (`46` tests, ~7 min)
```powershell
cd <HFWToolkit>
.\.venv\Scripts\hfw-toolkit.exe doctor
.\.venv\Scripts\pytest.exe -v
```
For fast targeted iteration during shader/character/asset work:
```powershell
.\.venv\Scripts\pytest.exe tests/test_v9_shaders_chars_assets.py -v
```

### Disassemble Any Material's SM 6.6 DXIL Pixel Shader in 10 Lines
```powershell
@'
from hfw_toolkit.config import ToolkitConfig
from hfw_toolkit.decima import HFWGame, ObjectId
from hfw_toolkit.extract.materials import parse_material
from hfw_toolkit.shaders.dxc import DXCDisassembler
from hfw_toolkit.shaders.hw_shader import parse_hw_shader

cfg = ToolkitConfig.load()
g = HFWGame(cfg)
dxc = DXCDisassembler(cfg)
try:
    re_obj = g.read_object(ObjectId.parse("21350:2823"))  # Replace with any RenderEffectResource ObjectId
    mat = parse_material(g, re_obj)
    print(mat.name, "techniques:", [(t.name, t.shader_id) for t in mat.techniques])
    pt = mat.primary_technique
    hw = parse_hw_shader(g, ObjectId.parse(pt.shader_id))
    ps = next(p for p in hw.programs if "pixel" in str(p.stage).lower())
    ir = dxc.disassemble(ps.dxil)
    print("\n".join(ir.splitlines()[:80]))
finally:
    g.close()
'@ | & ".\.venv\Scripts\python.exe" -
```

### Headless Cycles Verification Pattern (Inside Blender 5.3)
```python
import importlib.util, bpy
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "hfw_blender_builder",
    str(Path("<HFWToolkit>") / "hfw_toolkit" / "blender" / "blender_builder.py"),
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

bpy.ops.wm.read_factory_settings(use_empty=True)
arm = builder.import_bundle_into_scene(Path(r"<bundle_dir>"), enable_detail_nodes=True, pack_textures=False)
# Characters & Machines face +Y (Z-up meters). Render Cycles GPU/CPU denoised with AgX and inspect via image viewer!
```

### Launch Web Studio
```powershell
.\.venv\Scripts\hfw-toolkit.exe web
# Opens FastAPI studio at http://127.0.0.1:8420
```

---

## 7. High-Value Next Roadmap for Future Development

The core pipeline is rock-solid (`46/46` tests passing, visual Cycles fidelity verified across machines, characters, general assets, `EA08` in-game animations, and Control Rig IK/FK/detail/rest-lock/baking). The highest-leverage next milestones are:

1. **Facial Expression / Rig Controls & Blendshapes:**
   - Characters already merge the full `FACIAL_*` bone hierarchy (`Face` bone collection). Next step: map Decima's facial rig solver / pose library (or FACS-style control drivers) so animators can pose expressions cleanly without touching 500 raw facial bones.
2. **Interactive Outfit Dye Picker (In-Game Tribal Dyes):**
   - `character_dxil.py` currently evaluates each outfit at `CustomDye = false`. Extracting the `CustomDye = true` branch and Guerrilla's tribal dye palettes (`OutfitDyeResource`) would let users switch any Aloy armor between all official in-game dye schemes live in Blender's `HFW_CharacterControl` N-panel and UE5!
3. **Full Scene / World Prefab & Settlement Assembly (`StreamingTile` / `PrefabResource`):**
   - Extend `asset_assembler.py` to reconstruct multi-object `PrefabResource` instances (full buildings, Cauldron rooms, camp sites) with instanced transforms and terrain/foliage placement.
4. **glTF 2.0 / OpenUSD Export:**
   - Add direct `.glb` / `.usd` export from the Canonical Assembly Bundle (`assembly.json` + `geometry.npz` + `textures/`) for non-Blender/non-UE5 DCCs (Maya, Houdini, Unity, Godot, Three.js).
5. **Broader Decima Engine Support:**
   - Because `exe_rtti.py` statically parses Decima PE64 RTTI and `dsar.py` handles DSAR/LZ4, adding support for *Horizon Zero Dawn Remastered* (PC) or *Death Stranding Director's Cut* mainly requires validating RTTI struct/callback differences in `decima/callbacks.py`.
6. **Mocked / Offline CI Unit Tests:**
   - Currently all 45 tests run against a live HFW PC install. Adding a pure-unit test module with tiny synthetic RTTI/DSAR/DXIL fixtures would allow GitHub Actions CI to verify regressions without requiring the 120 GB game install.
