# HFWToolkit — Horizon Forbidden West Asset Assembler, Shader Studio & Audio Extractor

An automated, read-only extraction and research toolkit for **Horizon Forbidden West Complete Edition (PC)** (`Decima Engine`).

- **Rigged Machine Assembler & Live Shader Controls (Blender 4.2+ / 5.x & Unreal Engine 5):** One-command extraction of complete machine rigs (`Skeleton` + `PhysicsSkeleton` + `ModelHelpers`), skinned bodies, bone-parented destructible armor plates, child turrets/weapons, authentic per-variant muscle & eye color profiles (**Standard Amber**, **Apex Purple**, **Corrupted Red**, **Overridden Cyan**), Cycles/Eevee-safe glowing optical muscles (exact DXIL `Mask_Alpha * 40`), animated wet elemental liquid sacs & translucent glass canisters, Slaughterspine plasma charge-up shaders, and unified live shader controls (`HFW_MachineShaderControl` in Blender + `MPC_HFW_MachineControl` in UE5).
- **Full HFW Character & Outfit Assembler (Blender & UE5):** Import **Aloy** (all 35 Base + Burning Shores DLC outfits, headgear toggle, cinematic face & eyes, braided `HairResource` cards), all main story **Companions** (`Erend`, `Seyka`, `Kotallo`, `Zo`, `Alva`, `Beta`, `Varl`, `Sylens`, `Tilda`, `Regalla`, `Walter Londra`, `GAIA`, etc.), and **1,100+ Modular Tribal NPCs** + **211 swappable heads** — via a **Natural-Language Prompt Resolver** (`"Aloy Carja Blazon no headgear"`, `"Seyka"`, `"Tenakth female warrior"`) or structured browser, automatically assembling body, outfit, head, hair, Guerrilla's 8-layer `MSK.RGB` leather/metal/cloth shaders, and **ONE unified poseable humanoid Armature**.
- **Guerrilla Developer-Directory Asset Catalog (`1,454+` Assets):** Browse and export Weapons (`146`), Wildlife/Animals (`44`), Building Blocks/Ruins (`453`), Eco Trees/Foliage (`135`), Landscape (`60`), and Props (`12`) organized in Guerrilla's original `work:Models/...` & `DLC/models/...` folder structure for both Blender and Unreal Engine 5 (`SM_` StaticMesh or `SK_` SkeletalMesh).
- **Material & DXIL Shader Deep-Dive Studio (Web UI):** Interactive local web app explaining HFW's packed multi-channel `TextureSet`s, micro-detail texture arrays, SM 6.6 DXIL shader instructions, visual Decima & Unreal Engine 5 node graphs, and side-by-side Material Diffs.
- **Music, SFX & Voicelines Audio Extractor:** Native in-process Sony **ATRAC9** & PCM decoder (via bundled PyAV — no external FFmpeg/vgmstream needed) with live in-browser audio playback, beat-synced full-song mixdowns of multi-stem interactive cues, spoken dialogue subtitles (`voicelines_transcript.csv`), and an option to **preserve Guerrilla's reconstructed developer directory tree** or export flat.

---

## Highlights

- **Zero External Dumps or Injectors:** Reads directly from your Horizon Forbidden West installation (`LocalCacheWinGame\package\*.core(.stream)` — strictly read-only) and statically extracts all 12,173+ Decima RTTI type descriptors directly from `HorizonForbiddenWest.exe` on disk in ~0.12s (with a bundled compressed fallback schema for archive-only setups).
- **Zero-Config Path Auto-Detection:** Automatically locates your HFW install via Steam Registry (AppID `2420110`), Steam `libraryfolders.vdf` across drives `C:`–`Z:`, Epic Games Launcher manifests, and common install folders, plus Blender 4.2+ / 5.x.
- **Native Game-Packed Textures by Default:** Exports full-resolution native packed textures (`CLR_OP`, `NRM`, `MSK_RGH_RFL`, `AO`, `LGT`) by default, with an optional `--unpack-textures` flag for individual semantic PNG maps.
- **Blender N-Panel Add-on (v0.4.0):** Built-in one-click installer (`hfw-toolkit install-blender-addon`) adds an **HFW Toolkit** sidebar panel in Blender 3D View with **Machines / Characters / Asset Catalog** tabs, a natural-language character prompt importer, and **Live Machine Shader Controls** (Muscle & Eye color presets/pickers, Alert blend, Liquid color override, Plasma Charge, and Animation Speed).
- **Unreal Engine 5 Ready:** Exports rigged `.fbx`, textures, and automated UE5 Python importers (`ue5_import_machine.py`, `ue5_import_character.py`, `ue5_import_asset.py`) recreating HFW's animated machine shaders (`MPC_HFW_MachineControl`, dual-phase FlowMap sacs, translucent glass canisters, DXIL muscles), 8-layer character outfits, and Static/Skeletal assets inside UE5.
- **Complete Audio Coverage:**
  - **Music (`1,346` cues / `4,913` stems):** Base Game MainScore, Burning Shores DLC (including the 31-minute `H2DLC-EndCredits` medley), Photomode, Tilda's Music Player, Plainsong Chorus (`AsBeforeWeAre`), and optional `DigitalExtras/OST`.
  - **SFX (`70,000` waves across `2,348` folders):** Named via `GraphSoundResource → GraphProgramResource` under Machines, Weapons, Quests, UI, and Environment.
  - **Voicelines (`95,392` lines across `1,935` folders, `590+` speakers):** Paired with speaker names (`Aloy`, `Erend`, `Sylens`, `Zo`, `Varl`, `Kotallo`, `Alva`, etc.), English subtitles, and all installed spoken language packs.

---

## Requirements

- **OS:** Windows 10/11 (64-bit)
- **Game:** A legally purchased PC installation of **Horizon Forbidden West Complete Edition**
- **Python:** Python 3.11+ on `PATH`
- **Optional:**
  - **Blender 4.2+ / 5.x** (for `.blend` & rigged `.fbx` builds and the N-Panel add-on)
  - **Windows 10/11 SDK** or `dxcompiler.dll` placed in `bin/` (for SM 6.6 DXIL shader disassembly in the Shader Deep-Dive Studio — see [`bin/README.md`](bin/README.md))

---

## Quick Start (Windows)

### Option A: Double-Click Launchers (Easiest)
1. Clone or download this repository.
2. Double-click **`Setup_And_Doctor.bat`** once — creates `.venv`, installs dependencies, auto-detects your HFW & Blender paths, builds the SQLite catalog, and installs the Blender N-Panel add-on.
3. Double-click **`Launch_HFW_Studio.bat`** to open the **Material, Shader & Audio Web Studio** at `http://127.0.0.1:8420`.

### Option B: Command Line (`hfw-toolkit`)

```powershell
# Clone & install in a virtual environment
git clone <your-repo-url> HFWToolkit
cd HFWToolkit
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .

# 1. Verify paths, HFW patch version, RTTI schema & audio codec status
hfw-toolkit doctor

# 2. One-step setup (auto-detects Steam/Epic HFW install, or pass --game-dir explicitly)
hfw-toolkit setup
# Custom path example:
# hfw-toolkit setup --game-dir "D:\SteamLibrary\steamapps\common\Horizon Forbidden West Complete Edition"

# 3. Launch the interactive Web Studio (Machines, Material/Shader Lab & Audio/Music Player)
hfw-toolkit web
```

### Exporting Machines to Blender & Unreal Engine 5

```powershell
# List all indexed machines
hfw-toolkit machines

# Export ready-to-pose .blend files (native packed textures by default; add --unpack-textures for split maps)
hfw-toolkit export-blend --machine Slaughterspine --out exports/Slaughterspine.blend
hfw-toolkit export-blend --machine Clawstrider --variant 1 --out exports/Clawstrider_Acid.blend
hfw-toolkit export-blend --machine Bellowback --variant 1 --out exports/Bellowback_Fire.blend
hfw-toolkit export-blend --machine "Redeye Watcher" --out exports/Redeye_Watcher.blend

# Export rigged SkeletalMesh FBX + textures + UE5 animated shader builder script
hfw-toolkit export-ue5 --machine Slaughterspine --out exports/Slaughterspine_UE5
```

### Exporting Characters, Outfits & Unified Rigs (Natural-Language Prompt or Browser)

```powershell
# Resolve a natural-language prompt to preview match & options
hfw-toolkit characters resolve "Aloy Carja Blazon no headgear"

# Export ready-to-pose Blender .blend (unified Body + Outfit + Face + Braided Hair rig!)
hfw-toolkit characters export-blend --prompt "Aloy"
hfw-toolkit characters export-blend --prompt "Aloy Carja Blazon no headgear"
hfw-toolkit characters export-blend --prompt "Seyka"
hfw-toolkit characters export-blend --prompt "Erend"
hfw-toolkit characters export-blend --prompt "Tenakth female warrior"

# Export rigged SkeletalMesh FBX + textures + ue5_import_character.py for Unreal Engine 5
hfw-toolkit characters export-ue5 --prompt "Aloy Quen Commander"

# Browse all 35 Aloy outfits or list characters by tribe/kind
hfw-toolkit characters outfits --char aloy
hfw-toolkit characters list --kind companion
```

### Browsing & Exporting General Assets (Guerrilla Developer Directory Catalog)

```powershell
# Walk Guerrilla's original work:Models/... & DLC/models/... folder hierarchy
hfw-toolkit assets tree
hfw-toolkit assets tree models/weapons
hfw-toolkit assets tree models/Eco_Assets_HRZ2

# Search across all 1,454 indexed developer-directory assets
hfw-toolkit assets list --category weapons -q Bow
hfw-toolkit assets list --category animals
hfw-toolkit assets list --category building_blocks -q ruins

# Export any asset to Blender or Unreal Engine 5 (preserves Guerrilla folder layout under exports/Assets/)
hfw-toolkit assets export-blend --asset DLC/models/weapons/Bow_Heavy_Quen_SE
hfw-toolkit assets export-blend --asset models/eco_assets_hrz2/trees/ponderosa_pine
hfw-toolkit assets export-ue5   --asset DLC/models/weapons/Bow_Heavy_Quen_SE
```

### Extracting Music, SFX & Voicelines

By default, `audio extract` **preserves Guerrilla's reconstructed developer folder hierarchy** (`Music/...`, `SFX/Robots/<Machine>/<SoundName>/...`, `Voicelines/<lang>/<Quest>/<Scene>/<Speaker>/...`). Pass `--flat` to output into a single flat folder. Supported formats: `wav` (16-bit lossless), `flac`, `mp3` (320k), `ogg` (Opus), and `at9` (raw game ATRAC9).

```powershell
# Inspect audio catalog counts & installed spoken languages
hfw-toolkit audio stats

# Search songs, SFX folders, or voicelines
hfw-toolkit audio list --category music -q plasmaspine
hfw-toolkit audio list --category sfx -q slaughterspine

# Extract ALL in-game music (full beat-synced song mixes + individual stems)
hfw-toolkit audio extract --category music --format flac --out exports/Audio/AllMusic
# Or 1 combined mixed song per cue (no individual stems):
hfw-toolkit audio extract --category music --no-stems --format mp3 --out exports/Audio/AllSongs_MP3

# Extract a specific machine's SFX
hfw-toolkit audio extract --category sfx --query slaughterspine --format wav --out exports/Audio/SlaughterspineSFX

# Extract character voicelines with automatic transcript CSV (group by quest or speaker)
hfw-toolkit audio extract --category voicelines --query MQ18 --speaker Aloy --lang en --format wav --out exports/Audio/Aloy_MQ18

# Optional: deep-index all ~168k SFX sound graph names & spoken dialogue subtitles into SQLite
hfw-toolkit audio index --deep
```

---

## Running Tests

```powershell
pip install -e ".[dev]"
pytest -v
```
*(Integration tests require a local Horizon Forbidden West PC installation; Blender tests require Blender 4.2+/5.x.)*

---

## Disclaimer

This repository contains **source code only** — no copyrighted game files, textures, meshes, shaders, or audio from *Horizon Forbidden West* are included. All *Horizon Forbidden West* assets and the Decima Engine belong to **Guerrilla Games** and **Sony Interactive Entertainment**. This tool reads your local game installation strictly in read-only mode.
