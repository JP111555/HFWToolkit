"""Unified CLI for HFWToolkit (`hfw-toolkit`)."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import threading
import time
import webbrowser
from pathlib import Path

from hfw_toolkit.blender import build_blend_headless, export_assembly_bundle
from hfw_toolkit.config import ToolkitConfig
from hfw_toolkit.decima import AssetCatalog, HFWGame, RTTISchema
from hfw_toolkit.decima.exe_rtti import extract_rtti_from_exe, get_exe_version_string
from hfw_toolkit.extract import (
    ANIMATION_CATEGORIES,
    SUPPORTED_OUTPUT_FORMATS,
    AudioItemInfo,
    assemble_character,
    assemble_general_asset,
    assemble_machine,
    decode_audio_object,
    detect_digital_extras_ost,
    detect_installed_languages,
    discover_entity_animations,
    export_animation_pack,
    extract_audio_items_to_dir,
    extract_music_cues_to_dir,
    has_pyav,
    normalize_animation_category,
    resolve_character_prompt,
    resolve_machine_entity_id,
    summarize_animation_categories,
    write_audio_file,
)
from hfw_toolkit.shaders import DXCDisassembler, ShaderHashDictionary, analyze_material_deep


def detect_blender_versions(cfg: ToolkitConfig, explicit_version: str | None = None) -> list[str]:
    """Detect installed Blender major.minor version strings (e.g. ['5.3']) from blender_exe and APPDATA."""
    if explicit_version:
        return [explicit_version]
    versions: list[str] = []
    # Check blender_exe path components for '5.3', '4.3', 'Blender 5.3', etc.
    for part in reversed(cfg.blender_exe.parts):
        m = re.search(r"(\d+\.\d+)", part)
        if m:
            ver = m.group(1)
            if ver not in versions:
                versions.append(ver)
            break
    appdata = os.environ.get("APPDATA")
    if appdata:
        bl_root = Path(appdata) / "Blender Foundation" / "Blender"
        if bl_root.exists():
            for child in sorted(bl_root.iterdir(), reverse=True):
                if child.is_dir() and re.match(r"^\d+\.\d+$", child.name):
                    if child.name not in versions:
                        versions.append(child.name)
    if not versions:
        versions.append("5.3")
    return versions


def install_blender_addon_files(cfg: ToolkitConfig, blender_version: str | None = None) -> list[Path]:
    """Install the HFW Toolkit Blender N-Panel add-on with the local toolkit path pre-configured."""
    src = Path(__file__).resolve().parent / "blender" / "addon" / "__init__.py"
    appdata = os.environ.get("APPDATA")
    if not appdata:
        raise RuntimeError("APPDATA environment variable not found.")
    raw_code = src.read_text(encoding="utf-8")
    stamped_dir = str(cfg.toolkit_dir.resolve()).replace("\\", "\\\\")
    stamped_code = raw_code.replace(
        "DEFAULT_TOOLKIT_DIR = _resolve_default_toolkit_dir()",
        f'DEFAULT_TOOLKIT_DIR = "{stamped_dir}"',
    )
    installed: list[Path] = []
    for ver in detect_blender_versions(cfg, blender_version):
        dest_dir = Path(appdata) / "Blender Foundation" / "Blender" / ver / "scripts" / "addons"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / "hfw_toolkit_addon.py"
        dest_file.write_text(stamped_code, encoding="utf-8")
        installed.append(dest_file)
    return installed


def _cmd_doctor(args: argparse.Namespace, cfg: ToolkitConfig) -> int:
    t0 = time.perf_counter()
    diag = cfg.diagnose()
    dt = time.perf_counter() - t0
    cat = AssetCatalog(cfg.index_db_path)
    try:
        cat_built = cat.is_built()
        cat_stats = cat.stats() if cat_built else {}
    finally:
        cat.close()

    print("=" * 72)
    print(" HFWToolkit — System & Game Compatibility Doctor")
    print("=" * 72)
    print(f"  Toolkit Directory : {diag['toolkit_dir']}")
    print(f"  Config File       : {diag['config_file']} ({'Saved' if diag['config_file_exists'] else 'Using Auto-Detection'})")
    print(f"  Game Directory    : {diag['game_dir']} [{'OK' if diag['game_dir_valid'] else 'NOT FOUND'}]")
    print(
        f"  Game Executable   : {'OK' if diag['game_exe_found'] else 'NOT FOUND'}"
        + (f" (HFW Version {diag['game_version']})" if diag.get("game_version") else "")
    )
    print(f"  Decima Archives   : {diag['core_archive_count']} .core archives in {diag['package_dir']}")
    print(f"  RTTI Schema Mode  : {diag['rtti_source']}")
    print(f"  RTTI Types Loaded : {diag['rtti_type_count']:,} Decima types ready ({dt:.2f}s)")
    print(f"  DXCompiler DLL    : {diag['dxc_path']} [{'OK' if diag['dxc_found'] else 'MISSING'}]")
    print(f"  Audio Codec (PyAV): {'OK (Sony ATRAC9 + WAV/FLAC/MP3/OGG)' if has_pyav() else 'MISSING (run: pip install av)'}")
    print(f"  Blender Exe       : {diag['blender_exe']} [{'OK' if diag['blender_found'] else 'NOT FOUND'}]")
    try:
        from hfw_toolkit.updater import detect_system_pythons, get_local_version_info

        py_info = detect_system_pythons(cfg.toolkit_dir)
        newest = py_info.get("newest_system_python")
        py_note = (
            f"v{py_info['current_version']} ({'venv' if py_info['in_venv'] else 'system'})"
            + (
                f" — newer system Python v{newest['version']} available (run Setup_And_Doctor.bat --upgrade-python)"
                if py_info.get("upgrade_available") and newest
                else " [NEWEST]"
            )
        )
        print(f"  Python Runtime    : {py_note}")
        loc_v = get_local_version_info(cfg.toolkit_dir)
        print(
            f"  Toolkit Version   : {loc_v['short_sha']} ({loc_v['install_mode']} / {loc_v['branch']})"
            + f" | Auto-Update ({cfg.update_branch}): {'ON' if cfg.auto_update else 'OFF (opt-in)'}"
        )
    except Exception:
        pass
    if cat_built:
        print(
            f"  Asset Catalog DB  : READY ({cat_stats.get('machines', 0)} machines, "
            f"{cat_stats.get('characters', 0):,} characters/{cat_stats.get('character_outfits', 0)} outfits, "
            f"{cat_stats.get('dev_assets', 0):,} dev-dir assets, "
            f"{cat_stats.get('texture_sets', 0):,} TextureSets, {cat_stats.get('shaders', 0):,} shaders, "
            f"{cat_stats.get('music_cues', 0):,} music cues)"
        )
    else:
        print("  Asset Catalog DB  : Not built yet (run `hfw-toolkit setup` or `hfw-toolkit index`)")
    print("=" * 72)

    if not diag["ready"]:
        print("\n[!] Game installation was not found at the path above.")
        print("    Pass your HFW install folder via:")
        print('      hfw-toolkit setup --game-dir "C:\\Path\\To\\Horizon Forbidden West Complete Edition"')
        return 1

    print("\n[+] Ready! Works directly with your HFW install (no external RTTI dump or injector needed).")
    return 0


def _cmd_setup(args: argparse.Namespace, cfg: ToolkitConfig) -> int:
    print("=" * 72)
    print(" HFWToolkit — One-Step Setup & Configuration Wizard")
    print("=" * 72)
    cfg.ensure_dirs()
    saved_path = cfg.save()
    print(f"[1/4] Saved configuration -> {saved_path}")
    print(f"      Game Dir    : {cfg.game_dir}")
    print(f"      Blender Exe : {cfg.blender_exe}")

    diag = cfg.diagnose()
    if not diag["ready"]:
        print("\n[!] Could not locate Horizon Forbidden West archives at:", file=sys.stderr)
        print(f"    {cfg.game_dir}", file=sys.stderr)
        print('    Re-run with: hfw-toolkit setup --game-dir "C:\\Path\\To\\Horizon Forbidden West Complete Edition"', file=sys.stderr)
        return 1

    print(f"[2/4] Verifying Decima RTTI schema for HFW v{diag['game_version'] or 'unknown'}...")
    schema = RTTISchema.load_for_game(cfg)
    print(f"      -> Loaded {len(schema.types):,} RTTI types via {schema.source_label}")

    if not args.no_index:
        print("[3/4] Checking SQLite Asset Catalog (`cache/hfw_index.sqlite`)...")
        _cmd_index(argparse.Namespace(rebuild=args.rebuild_index), cfg)
    else:
        print("[3/4] Skipping catalog index build (--no-index).")

    if not args.no_addon:
        try:
            installed = install_blender_addon_files(cfg)
            print(f"[4/4] Installed Blender N-Panel Add-on -> {', '.join(str(p) for p in installed)}")
        except Exception as exc:
            print(f"[4/4] Note: Could not auto-install Blender add-on ({exc})")
    else:
        print("[4/4] Skipping Blender add-on installation (--no-addon).")

    print("\nSetup complete! You can now:")
    print("  • Double-click `Launch_HFW_Studio.bat` (or run `hfw-toolkit web`) to open the Web Studio")
    print("  • Or run `hfw-toolkit export-blend --machine Slaughterspine` to build a ready .blend file")
    return 0


def _cmd_dump_rtti(args: argparse.Namespace, cfg: ToolkitConfig) -> int:
    if not cfg.game_exe.exists():
        print(f"HorizonForbiddenWest.exe not found at: {cfg.game_exe}", file=sys.stderr)
        return 1
    out_path = Path(args.out) if args.out else (cfg.toolkit_dir / "dump" / "hfw.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ver = get_exe_version_string(cfg.game_exe) or "unknown"
    print(f"Scanning {cfg.game_exe} (HFW v{ver}) statically on disk...")
    t0 = time.perf_counter()
    data = extract_rtti_from_exe(cfg.game_exe)
    dt = time.perf_counter() - t0
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Extracted {len(data):,} Decima RTTI types in {dt:.2f}s -> {out_path}")
    return 0


def _cmd_index(args: argparse.Namespace, cfg: ToolkitConfig) -> int:
    cfg.ensure_dirs()
    game = HFWGame(cfg)
    cat = AssetCatalog(cfg.index_db_path)
    try:
        if cat.is_built() and not args.rebuild:
            print(f"Index already built at {cfg.index_db_path}: {cat.stats()} (pass --rebuild to refresh)")
            return 0
        stats = cat.build(game, progress_cb=print)
        print(f"Done: {stats}")
        return 0
    finally:
        cat.close()
        game.close()


def _cmd_machines(args: argparse.Namespace, cfg: ToolkitConfig) -> int:
    cat = AssetCatalog(cfg.index_db_path)
    try:
        if not cat.is_built():
            print("Index not built yet. Run `hfw-toolkit index` first.", file=sys.stderr)
            return 1
        rows = cat.list_machines(args.query or "")
        print(f"Found {len(rows)} machines:")
        for r in rows:
            print(
                f"  {r['group_id']:6d}:{r['object_index']:<4d} | "
                f"{r['display_name']:20s} ({r['codename']:18s}) | parts={r['part_count']}"
            )
        return 0
    finally:
        cat.close()


def _cmd_export_bundle(args: argparse.Namespace, cfg: ToolkitConfig) -> int:
    cfg.ensure_dirs()
    game = HFWGame(cfg)
    cat = AssetCatalog(cfg.index_db_path)
    try:
        oid, code, disp = resolve_machine_entity_id(game, cat, args.machine)
        asm = assemble_machine(
            game,
            oid,
            codename=code,
            display_name=disp,
            variant_index=args.variant,
            lod_index=args.lod,
            include_broken_states=not args.no_broken_states,
        )
        out_dir = Path(args.out) if args.out else (cfg.exports_dir / f"{disp}_v{args.variant}_bundle")
        export_assembly_bundle(
            game,
            asm,
            out_dir,
            max_texture_size=args.max_tex_size,
            unpack_textures=args.unpack_textures,
            animations=getattr(args, "animations", None),
        )
        print(
            f"Exported bundle for {disp} ({asm.variant_label}, textures={'unpacked' if args.unpack_textures else 'packed'}): "
            f"{len(asm.skeleton.joints)} joints, {len(asm.parts)} parts, {len(asm.materials)} materials -> {out_dir}"
        )
        return 0
    finally:
        cat.close()
        game.close()


def _cmd_export_blend(args: argparse.Namespace, cfg: ToolkitConfig) -> int:
    cfg.ensure_dirs()
    game = HFWGame(cfg)
    cat = AssetCatalog(cfg.index_db_path)
    try:
        oid, code, disp = resolve_machine_entity_id(game, cat, args.machine)
        print(f"[1/3] Assembling {disp} ({oid}, variant {args.variant}, LOD {args.lod})...")
        asm = assemble_machine(
            game,
            oid,
            codename=code,
            display_name=disp,
            variant_index=args.variant,
            lod_index=args.lod,
            include_broken_states=not args.no_broken_states,
        )
        print(
            f"      -> {len(asm.skeleton.joints)} joints, {len(asm.skeleton.helpers)} helpers, "
            f"{len(asm.parts)} parts, {len(asm.materials)} materials"
        )
        out_blend = Path(args.out) if args.out else (cfg.exports_dir / f"{disp}_v{args.variant}.blend")
        bundle_dir = out_blend.parent / f"{out_blend.stem}_bundle"
        tex_mode_lbl = "Unpacking textures" if args.unpack_textures else "Exporting native packed game textures"
        print(f"[2/3] {tex_mode_lbl} & canonical bundle -> {bundle_dir}")
        export_assembly_bundle(
            game,
            asm,
            bundle_dir,
            max_texture_size=args.max_tex_size,
            unpack_textures=args.unpack_textures,
            animations=getattr(args, "animations", None),
        )
        fbx_out = (bundle_dir / f"{disp.replace('/', '_').replace(' ', '_')}.fbx") if getattr(args, "export_fbx", False) else None
        print(f"[3/3] Building ready-to-use Blender 5.3 scene -> {out_blend}")
        build_blend_headless(
            bundle_dir,
            out_blend,
            config=cfg,
            pack_textures=args.pack,
            enable_detail_nodes=not args.no_detail,
            export_fbx=fbx_out,
            build_control_rig=not getattr(args, "no_control_rig", False),
        )
        print(f"Success! Open directly in Blender 5.3: {out_blend}")
        if fbx_out and fbx_out.exists():
            print(f"UE5 Rigged FBX + Animated Importer Script: {bundle_dir / 'ue5_import_machine.py'}")
        return 0
    finally:
        cat.close()
        game.close()


def _cmd_export_ue5(args: argparse.Namespace, cfg: ToolkitConfig) -> int:
    """Export a complete UE5-ready rigged FBX + textures + animated UE5 Python material/rig importer."""
    cfg.ensure_dirs()
    game = HFWGame(cfg)
    cat = AssetCatalog(cfg.index_db_path)
    try:
        oid, code, disp = resolve_machine_entity_id(game, cat, args.machine)
        safe_disp = re.sub(r"[^A-Za-z0-9_\-.]+", "_", disp).strip("_") or "Machine"
        print(f"[1/3] Assembling {disp} ({oid}, variant {args.variant}, LOD {args.lod})...")
        asm = assemble_machine(
            game,
            oid,
            codename=code,
            display_name=disp,
            variant_index=args.variant,
            lod_index=args.lod,
            include_broken_states=not args.no_broken_states,
        )
        bundle_dir = Path(args.out) if args.out else (cfg.exports_dir / f"{safe_disp}_v{args.variant}_ue5_bundle")
        print(f"[2/3] Exporting textures & UE5 animated shader/Control Rig script -> {bundle_dir}")
        export_assembly_bundle(
            game,
            asm,
            bundle_dir,
            max_texture_size=args.max_tex_size,
            unpack_textures=args.unpack_textures,
            animations=getattr(args, "animations", None),
        )
        fbx_path = bundle_dir / f"{safe_disp}.fbx"
        tmp_blend = bundle_dir / f"{safe_disp}.blend"
        print(f"[3/3] Exporting rigged SkeletalMesh FBX via Blender -> {fbx_path}")
        build_blend_headless(
            bundle_dir,
            tmp_blend,
            config=cfg,
            pack_textures=False,
            enable_detail_nodes=True,
            export_fbx=fbx_path,
            build_control_rig=not getattr(args, "no_control_rig", False),
        )
        print(f"Success! Run inside Unreal Engine 5 (Tools -> Execute Python Script):")
        print(f"  {bundle_dir / 'ue5_import_machine.py'}")
        return 0
    finally:
        cat.close()
        game.close()


def _cmd_inspect_material(args: argparse.Namespace, cfg: ToolkitConfig) -> int:
    game = HFWGame(cfg)
    try:
        report = analyze_material_deep(
            game,
            args.material_id,
            dxc=DXCDisassembler(cfg),
            hash_dict=ShaderHashDictionary(cfg),
            include_raw_dxil=bool(args.out_dir),
            try_external_decompiler=args.decompile,
        )
        print(f"Material: {report['name']} ({report['material_id']})")
        print(f"Headline: {report['explainer']['headline']}")
        print(f"GPU Profile: {json.dumps(report['explainer']['gpu_profile'], indent=2)}")
        for card in report["explainer"]["cards"]:
            print(f"\n  [{card['category']}] {card['title']}\n    {card['summary']}")
        if args.out_dir:
            od = Path(args.out_dir)
            od.mkdir(parents=True, exist_ok=True)
            (od / "material_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            (od / "pseudo_shader.hlsl").write_text(report["pseudo_hlsl"], encoding="utf-8")
            (od / "ue5_material_nodes.t3d").write_text(report["ue5_clipboard_t3d"], encoding="utf-8")
            (od / "ue5_import_material.py").write_text(report["ue5_python_script"], encoding="utf-8")
            print(f"\nWrote full report, Pseudo-HLSL, and UE5 exports to {od}")
        return 0
    finally:
        game.close()


def _cmd_install_addon(args: argparse.Namespace, cfg: ToolkitConfig) -> int:
    try:
        installed = install_blender_addon_files(cfg, getattr(args, "blender_version", None))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    for p in installed:
        print(f"Installed HFW Toolkit Add-on to: {p}")
    print("Enable it in Blender -> Edit -> Preferences -> Add-ons -> search 'HFW Toolkit'.")
    return 0


def _row_to_audio_item(r: dict) -> AudioItemInfo:
    return AudioItemInfo(
        object_id=r["object_id"],
        kind=r["kind"],
        name=r["name"],
        dev_folder=r["dev_folder"],
        filename_stem=r["filename_stem"],
        group_id=int(r["group_id"]),
        sample_rate=int(r["sample_rate"]),
        channels=int(r["channels"]),
        sample_count=int(r["sample_count"]),
        duration_sec=float(r["duration_sec"]),
        encoding=int(r["encoding"]),
        encoding_name=str(r["encoding_name"]),
        data_size=int(r["data_size"]),
        is_streaming=bool(r["is_streaming"]),
        sound_graph_name=str(r.get("sound_graph_name", "")),
        owner_name=str(r.get("owner_name", "")),
        speaker=str(r.get("speaker", "")),
        subtitle=str(r.get("subtitle", "")),
        languages=tuple(r.get("languages") or ()),
    )


def _cmd_audio(args: argparse.Namespace, cfg: ToolkitConfig) -> int:
    cfg.ensure_dirs()
    action = args.audio_cmd
    game: HFWGame | None = None

    def get_game() -> HFWGame:
        nonlocal game
        if game is None:
            game = HFWGame(cfg)
        return game

    cat = AssetCatalog(cfg.index_db_path)
    try:
        if action == "index":
            if not cat.is_audio_built() or args.rebuild:
                cat.build_audio_index(get_game(), progress_cb=print)
            else:
                print(f"Tier-1 audio catalog already indexed: {cat.audio_stats()}")
            if args.deep:
                print("Starting full deep index of all SFX & Voiceline clips + subtitles...")
                st = cat.build_deep_audio_index(
                    get_game(),
                    progress_cb=lambda i, n, msg: print(f"  [{i}/{n}] {msg}"),
                )
                print(f"Deep audio index complete: {st}")
            return 0

        if not cat.is_audio_built():
            print("Audio index not built yet — building Tier-1 audio index (Music + Voices + Developer Directories)...")
            cat.build_audio_index(get_game(), progress_cb=print)

        if action == "stats":
            st = cat.audio_stats()
            # Detect installed languages & OST directly from filesystem (zero HFWGame startup cost!)
            from hfw_toolkit.extract.audio import SPOKEN_LANGUAGES

            langs = [
                lang
                for lang in SPOKEN_LANGUAGES
                if (cfg.package_dir / lang).is_dir() and any((cfg.package_dir / lang).glob("*.stream"))
            ]
            ost_dir = cfg.game_dir / "DigitalExtras" / "OST"
            ost = (
                [p for p in ost_dir.iterdir() if p.is_file() and p.suffix.lower() in (".mp3", ".flac", ".wav", ".ogg")]
                if ost_dir.is_dir()
                else []
            )
            print("=" * 72)
            print(" HFWToolkit — Audio, Music, SFX & Voicelines Summary")
            print("=" * 72)
            print(f"  PyAV ATRAC9 Codec   : {'READY' if has_pyav() else 'MISSING'}")
            print(f"  Installed Languages : {', '.join(langs) if langs else 'None'}")
            print(f"  Music Cues / Stems  : {st['music_cues']:,} cues ({st['music_stems']:,} stems)")
            print(f"  DigitalExtras OST   : {len(ost)} bonus MP3 tracks")
            print(f"  SFX Directories     : {st['sfx_groups']:,} groups ({st['sfx_waves_total']:,} total waves; {st['sfx_assets_indexed']:,} deep-indexed)")
            print(f"  Voiceline Dirs      : {st['voiceline_groups']:,} groups ({st['voicelines_total']:,} total lines; {st['voiceline_assets_indexed']:,} deep-indexed)")
            print(f"  Character Voices    : {st['voices']:,} registered speakers (Aloy, Sylens, Zo, Varl, ...)")
            print("=" * 72)
            return 0

        if action == "list":
            cat_kind = args.category
            q = args.query or ""
            limit = args.limit
            if cat_kind == "music":
                cues = cat.list_music_cues(query=q, category=args.subcat or "", limit=limit)
                print(f"Music Cues ({len(cues)} shown):")
                for c in cues:
                    print(
                        f"  {c['cue_id']:10s} | {c['duration_sec']:6.1f}s | {c['stem_count']:2d} stems | "
                        f"{c['dev_folder']}"
                    )
            elif cat_kind in ("sfx", "voicelines"):
                k = "sfx" if cat_kind == "sfx" else "voiceline"
                if args.group is not None:
                    items = cat.ensure_audio_group_indexed(get_game(), args.group, k)
                    if q:
                        ql = q.lower()
                        items = [
                            it
                            for it in items
                            if ql in it["name"].lower()
                            or ql in it["dev_folder"].lower()
                            or ql in it["subtitle"].lower()
                            or ql in it["speaker"].lower()
                        ]
                    if args.speaker:
                        sl = args.speaker.lower()
                        items = [it for it in items if sl in it["speaker"].lower()]
                    print(f"Group {args.group} {cat_kind.upper()} ({len(items[:limit])}/{len(items)} shown):")
                    for it in items[:limit]:
                        extra = f' | "{it["subtitle"][:60]}"' if it["subtitle"] else ""
                        print(f"  {it['object_id']:11s} | {it['duration_sec']:5.2f}s | {it['dev_folder']}/{it['filename_stem']}{extra}")
                else:
                    # Show matching indexed items first if speaker/query hits assets, plus matching directories
                    groups = cat.list_audio_groups(kind=k, query=q, category=args.subcat or "", limit=limit)
                    print(f"{cat_kind.upper()} Developer Directories ({len(groups)} shown — pass --group <gid> to list clips):")
                    for g in groups:
                        idx_tag = "[indexed]" if g["indexed_items"] else ""
                        print(
                            f"  group {g['group_id']:6d} | {g['item_count']:4d} clips | {g['dev_folder']} {idx_tag}"
                        )
                    if args.speaker or q:
                        assets = cat.search_audio_assets(kind=k, query=q, speaker=args.speaker or "", limit=min(limit, 30))
                        if assets:
                            print(f"\nIndexed matching {cat_kind} clips ({len(assets)} shown):")
                            for it in assets:
                                extra = f' | "{it["subtitle"][:60]}"' if it["subtitle"] else ""
                                print(f"  {it['object_id']:11s} | {it['duration_sec']:5.2f}s | {it['dev_folder']}/{it['filename_stem']}{extra}")
            return 0

        if action == "extract":
            out_dir = Path(args.out) if args.out else (cfg.exports_dir / "Audio")
            fmt = args.format.lower()
            preserve = not args.flat
            langs = [x.strip() for x in (args.lang or "en").split(",") if x.strip()]

            # 1. Single explicit ObjectId extraction
            if args.id:
                dec = decode_audio_object(get_game(), args.id, lang=langs[0])
                out_dir.mkdir(parents=True, exist_ok=True)
                ext = "." + fmt
                out_path = out_dir / f"audio_{args.id.replace(':', '_')}{ext}"
                nbytes = write_audio_file(dec, out_path, fmt=fmt)
                print(f"Extracted {args.id} ({dec.duration_sec:.2f}s, {dec.sample_rate}Hz, {dec.channels}ch, {nbytes:,} bytes) -> {out_path}")
                return 0

            # 2. Single explicit MusicCue extraction
            if args.cue:
                cue_info = cat.get_music_cue_info(args.cue)
                if cue_info is None:
                    print(f"MusicCue '{args.cue}' not found in catalog.", file=sys.stderr)
                    return 1
                res = extract_music_cues_to_dir(
                    get_game(),
                    [cue_info],
                    out_dir,
                    fmt=fmt,
                    preserve_structure=preserve,
                    mix_cues=not args.no_mix_cues,
                    export_stems=not args.no_stems,
                    include_ost=args.include_ost,
                    progress_cb=lambda i, n, m: print(f"  [{i}/{n}] {m}"),
                )
                print(f"Extracted {res['exported_count']} file(s) ({res['total_bytes']/1e6:.2f} MB) in {res['elapsed_sec']}s -> {res['out_dir']}")
                return 0

            cat_kind = args.category
            q = args.query or ""

            # 3. Music extraction (all songs or filtered!)
            if cat_kind in ("music", "all"):
                cue_rows = cat.list_music_cues(query=q, category=args.subcat or "", limit=args.limit or 10000)
                cues = [ci for r in cue_rows if (ci := cat.get_music_cue_info(r["cue_id"])) is not None]
                print(f"Extracting {len(cues)} music cue(s) to {out_dir} (format={fmt}, preserve_structure={preserve}, mix={not args.no_mix_cues}, stems={not args.no_stems})...")
                res = extract_music_cues_to_dir(
                    get_game(),
                    cues,
                    out_dir,
                    fmt=fmt,
                    preserve_structure=preserve,
                    mix_cues=not args.no_mix_cues,
                    export_stems=not args.no_stems,
                    include_ost=args.include_ost,
                    limit=args.limit,
                    progress_cb=lambda i, n, m: print(f"  [{i}/{n}] {m}") if (i == 1 or i % 10 == 0 or i == n) else None,
                )
                print(
                    f"Music export done: {res['exported_count']} files ({res['total_bytes']/1e6:.2f} MB) "
                    f"in {res['elapsed_sec']}s -> {res['out_dir']}"
                    + (f" ({len(res['errors'])} warnings)" if res["errors"] else "")
                )

            # 4. SFX and/or Voicelines extraction
            for kind_name, db_kind in (("sfx", "sfx"), ("voicelines", "voiceline")):
                if cat_kind not in (kind_name, "all"):
                    continue
                items_rows: list[dict] = []
                if args.group is not None:
                    items_rows = cat.ensure_audio_group_indexed(get_game(), args.group, db_kind)
                else:
                    # Index matching groups on-demand so even Tier-1 catalogs extract seamlessly!
                    matched_groups = cat.list_audio_groups(
                        kind=db_kind, query=q, category=args.subcat or "", limit=2000
                    )
                    if not matched_groups and (q or args.speaker):
                        # Fall back to searching already-deep-indexed individual clips (e.g. subtitle/speaker/SFX graph name)
                        items_rows = cat.search_audio_assets(
                            kind=db_kind, query=q, speaker=args.speaker or "", limit=args.limit or 5000
                        )
                    else:
                        cap = args.limit
                        for gi, gr in enumerate(matched_groups, 1):
                            if gi == 1 or gi % 15 == 0 or gi == len(matched_groups):
                                print(f"  Loading {kind_name} directory [{gi}/{len(matched_groups)}]: {gr['dev_folder']}...")
                            g_items = cat.ensure_audio_group_indexed(get_game(), int(gr["group_id"]), db_kind)
                            items_rows.extend(g_items)
                            if cap and not args.speaker and len(items_rows) >= cap:
                                break

                # Apply speaker & sub-filter if specified
                if args.speaker:
                    sl = args.speaker.lower()
                    items_rows = [r for r in items_rows if sl in r["speaker"].lower()]
                if q and args.group is not None:
                    ql = q.lower()
                    items_rows = [
                        r
                        for r in items_rows
                        if ql in r["name"].lower()
                        or ql in r["dev_folder"].lower()
                        or ql in r["sound_graph_name"].lower()
                        or ql in r["subtitle"].lower()
                        or ql in r["speaker"].lower()
                    ]

                items = [_row_to_audio_item(r) for r in items_rows]
                if not items:
                    print(f"No matching {kind_name} items found (tip: run `hfw-toolkit audio index --deep` to search inside all subtitles & SFX graph names).")
                    continue
                print(f"Extracting {min(len(items), args.limit or len(items))} {kind_name} clip(s) to {out_dir} (format={fmt}, preserve_structure={preserve})...")
                res = extract_audio_items_to_dir(
                    get_game(),
                    items,
                    out_dir,
                    fmt=fmt,
                    preserve_structure=preserve,
                    languages=langs,
                    group_by=args.group_by,
                    write_transcript=not args.no_transcript,
                    limit=args.limit,
                    progress_cb=lambda i, n, m: print(f"  [{i}/{n}] {m}") if (i == 1 or i % 25 == 0 or i == n) else None,
                )
                print(
                    f"{kind_name.upper()} export done: {res['exported_count']} files ({res['total_bytes']/1e6:.2f} MB) "
                    f"in {res['elapsed_sec']}s -> {res['out_dir']}"
                    + (f" | transcript: {res['transcript_path']}" if res.get("transcript_path") else "")
                    + (f" ({len(res['errors'])} warnings)" if res["errors"] else "")
                )
            return 0

        return 1
    finally:
        cat.close()
        if game is not None:
            game.close()


def _cmd_characters(args: argparse.Namespace, cfg: ToolkitConfig) -> int:
    cfg.ensure_dirs()
    cat = AssetCatalog(cfg.index_db_path)
    game: HFWGame | None = None

    def get_game() -> HFWGame:
        nonlocal game
        if game is None:
            game = HFWGame(cfg)
        return game

    def ensure_indexed() -> None:
        if not cat.is_characters_assets_built():
            print("Character & dev-asset catalog not indexed yet — indexing now...")
            cat.index_characters_and_assets(get_game(), progress_cb=print)

    try:
        sub = args.char_cmd
        if sub == "list":
            ensure_indexed()
            rows = cat.list_characters(
                query=args.query or "",
                kind=args.kind or "",
                tribe=args.tribe or "",
                gender=args.gender or "",
                limit=args.limit,
            )
            print(f"Found {len(rows)} character(s):")
            for r in rows:
                print(
                    f"  [{r['character_kind']:9s}] {r['char_id']:18s} | {r['display_name']:28s} | "
                    f"{r['tribe']:10s} | {r['gender']:7s} | outfits={r['outfit_count']:2d} | {r['dev_folder']}"
                )
            return 0

        if sub == "outfits":
            ensure_indexed()
            rows = cat.list_character_outfits(char_id=args.char or "", query=args.query or "", limit=args.limit)
            print(f"Found {len(rows)} outfit variation(s):")
            for r in rows:
                dlc = " [DLC]" if r["is_dlc"] else ""
                hg = " (HG toggle)" if r["has_headgear_toggle"] else ""
                print(
                    f"  {r['outfit_id']:14s} | {r['char_id']:12s} | {r['display_name']:32s}{dlc}{hg} | {r['dev_folder']}"
                )
            return 0

        if sub == "resolve":
            ensure_indexed()
            res = resolve_character_prompt(cat, args.prompt)
            print(f"Prompt      : {res['prompt']!r}")
            print(f"Matched     : {res['summary']} (confidence={res['confidence']})")
            print(f"Character   : {res['character']['char_id'] if res['character'] else None}")
            print(f"Outfit      : {res['outfit']['outfit_id'] if res['outfit'] else None}")
            print(f"Options     : headgear={res['headgear']}, include_hair={res['include_hair']}, hair_lod={res['hair_lod']}")
            if res["alternatives"]:
                print("Alternatives:")
                for a in res["alternatives"][:5]:
                    print(f"  - {a['display_name']} ({a['outfit_name'] or a['char_id']}) [score={a['score']}]")
            return 0

        if sub in ("export-bundle", "export-blend", "export-ue5"):
            ensure_indexed()
            g = get_game()
            print(f"[1/3] Assembling HFW character ({args.prompt or args.char})...")
            asm = assemble_character(
                g,
                catalog=cat,
                char_id=args.char or None,
                outfit_id=args.outfit or None,
                prompt=args.prompt or None,
                headgear=None if (args.prompt and not args.no_headgear) else (not args.no_headgear),
                include_hair=None if (args.prompt and not args.no_hair) else (not args.no_hair),
                hair_lod=args.hair_lod,
                facial_id=args.facial or None,
                lod_index=args.lod,
            )
            safe_name = re.sub(r"[^A-Za-z0-9_\-.]+", "_", f"{asm.display_name}_{asm.variant_label}").strip("_")[:64] or "Character"
            print(
                f"      -> {asm.display_name} ({asm.variant_label}) | "
                f"{len(asm.skeleton.joints)} unified joints, {len(asm.parts)} mesh parts, {len(asm.materials)} materials"
            )
            if sub == "export-bundle":
                out_dir = Path(args.out) if args.out else (cfg.exports_dir / "Characters" / f"{safe_name}_bundle")
                export_assembly_bundle(
                    g,
                    asm,
                    out_dir,
                    max_texture_size=args.max_tex_size,
                    unpack_textures=args.unpack_textures,
                    asset_kind="character",
                    animations=getattr(args, "animations", None),
                )
                print(f"Exported character bundle -> {out_dir}")
                return 0

            if sub == "export-ue5":
                out_dir = Path(args.out) if args.out else (cfg.exports_dir / "Characters" / f"{safe_name}_UE5")
                out_dir.mkdir(parents=True, exist_ok=True)
                export_assembly_bundle(
                    g,
                    asm,
                    out_dir,
                    max_texture_size=args.max_tex_size,
                    unpack_textures=args.unpack_textures,
                    asset_kind="character",
                    animations=getattr(args, "animations", None),
                )
                fbx_out = out_dir / f"{safe_name}.fbx"
                blend_tmp = out_dir / f"{safe_name}.blend"
                print(f"[3/3] Building UE5 Rigged Character FBX -> {fbx_out}")
                build_blend_headless(
                    out_dir,
                    blend_tmp,
                    config=cfg,
                    export_fbx=fbx_out,
                    build_control_rig=not getattr(args, "no_control_rig", False),
                )
                print(f"Success! UE5 Character Importer: {out_dir / 'ue5_import_character.py'}")
                return 0

            # export-blend
            out_blend = Path(args.out) if args.out else (cfg.exports_dir / "Characters" / f"{safe_name}.blend")
            out_blend.parent.mkdir(parents=True, exist_ok=True)
            bdir = out_blend.parent / f"{out_blend.stem}_bundle"
            print(f"[2/3] Exporting character bundle & textures -> {bdir}")
            export_assembly_bundle(
                g,
                asm,
                bdir,
                max_texture_size=args.max_tex_size,
                unpack_textures=args.unpack_textures,
                asset_kind="character",
                animations=getattr(args, "animations", None),
            )
            fbx_out = (bdir / f"{safe_name}.fbx") if args.export_fbx else None
            print(f"[3/3] Building ready-to-pose Blender 5.3 character rig -> {out_blend}")
            build_blend_headless(
                bdir,
                out_blend,
                config=cfg,
                pack_textures=args.pack,
                export_fbx=fbx_out,
                build_control_rig=not getattr(args, "no_control_rig", False),
            )
            print(f"Success! Open directly in Blender 5.3: {out_blend}")
            return 0
        return 1
    finally:
        cat.close()
        if game is not None:
            game.close()


def _cmd_assets(args: argparse.Namespace, cfg: ToolkitConfig) -> int:
    cfg.ensure_dirs()
    cat = AssetCatalog(cfg.index_db_path)
    game: HFWGame | None = None

    def get_game() -> HFWGame:
        nonlocal game
        if game is None:
            game = HFWGame(cfg)
        return game

    def ensure_indexed() -> None:
        if not cat.is_characters_assets_built():
            print("Developer-directory asset catalog not indexed yet — indexing now...")
            cat.index_characters_and_assets(get_game(), progress_cb=print)

    try:
        sub = args.asset_cmd
        if sub == "tree":
            ensure_indexed()
            tree = cat.list_dev_tree_children(args.folder or "")
            print(f"Guerrilla Developer Directory: /{tree['parent_dir'] or ''}")
            for f in tree["folders"]:
                cats = ",".join(f["categories"])
                print(f"  [DIR]   {f['path']:54s} ({f['asset_count']:4d} assets | {cats})")
            for a in tree["assets"]:
                rig = "RIGGED" if a["is_skinned"] else "STATIC"
                print(f"  [{rig}] {a['dev_path']:54s} -> {a['resource_type']} ({a['resource_id']})")
            return 0

        if sub == "list":
            ensure_indexed()
            rows = cat.search_dev_assets(
                query=args.query or "",
                category=args.category or "",
                dir_prefix=args.dir or "",
                limit=args.limit,
            )
            print(f"Found {len(rows)} developer-directory asset(s):")
            for r in rows:
                rig = "RIGGED" if r["is_skinned"] else "STATIC"
                print(
                    f"  [{r['category']:15s}|{rig}] {r['dev_path']:52s} | {r['resource_type']:20s} | {r['resource_id']}"
                )
            return 0

        if sub in ("export-bundle", "export-blend", "export-ue5"):
            ensure_indexed()
            g = get_game()
            row = cat.get_dev_asset(args.asset)
            if row is not None:
                res_id = row["resource_id"]
                disp = row["display_name"]
                dev_p = row["dev_path"]
                cat_name = row["category"]
            else:
                res_id = args.asset
                disp = ""
                dev_p = ""
                cat_name = "props"

            print(f"[1/3] Assembling HFW asset {dev_p or res_id} ({res_id})...")
            asm = assemble_general_asset(
                g,
                res_id,
                display_name=disp,
                dev_path=dev_p,
                category=cat_name,
                lod_index=args.lod,
            )
            safe_name = re.sub(r"[^A-Za-z0-9_\-.]+", "_", asm.display_name).strip("_")[:64] or "Asset"
            # Preserve developer directory layout under exports/Assets/<dev_path> by default!
            rel_folder = Path(*[p for p in dev_p.replace("\\", "/").split("/") if p]) if dev_p else Path(safe_name)
            print(
                f"      -> {asm.display_name} | {len(asm.skeleton.joints)} joints, "
                f"{len(asm.parts)} parts, {len(asm.materials)} materials"
            )

            if sub == "export-bundle":
                out_dir = Path(args.out) if args.out else (cfg.exports_dir / "Assets" / rel_folder)
                export_assembly_bundle(
                    g,
                    asm,
                    out_dir,
                    max_texture_size=args.max_tex_size,
                    unpack_textures=args.unpack_textures,
                    asset_kind="asset",
                )
                print(f"Exported asset bundle -> {out_dir}")
                return 0

            if sub == "export-ue5":
                out_dir = Path(args.out) if args.out else (cfg.exports_dir / "Assets" / rel_folder)
                out_dir.mkdir(parents=True, exist_ok=True)
                export_assembly_bundle(
                    g,
                    asm,
                    out_dir,
                    max_texture_size=args.max_tex_size,
                    unpack_textures=args.unpack_textures,
                    asset_kind="asset",
                )
                fbx_out = out_dir / f"{safe_name}.fbx"
                blend_tmp = out_dir / f"{safe_name}.blend"
                build_blend_headless(out_dir, blend_tmp, config=cfg, export_fbx=fbx_out)
                print(f"Success! UE5 Asset Importer: {out_dir / 'ue5_import_asset.py'}")
                return 0

            out_blend = Path(args.out) if args.out else (cfg.exports_dir / "Assets" / rel_folder / f"{safe_name}.blend")
            out_blend.parent.mkdir(parents=True, exist_ok=True)
            bdir = out_blend.parent / f"{out_blend.stem}_bundle"
            export_assembly_bundle(
                g,
                asm,
                bdir,
                max_texture_size=args.max_tex_size,
                unpack_textures=args.unpack_textures,
                asset_kind="asset",
            )
            fbx_out = (bdir / f"{safe_name}.fbx") if args.export_fbx else None
            build_blend_headless(bdir, out_blend, config=cfg, pack_textures=args.pack, export_fbx=fbx_out)
            print(f"Success! Open directly in Blender 5.3: {out_blend}")
            return 0
        return 1
    finally:
        cat.close()
        if game is not None:
            game.close()


def _cmd_animations(args: argparse.Namespace, cfg: ToolkitConfig) -> int:
    cfg.ensure_dirs()
    game = HFWGame(cfg)
    cat = AssetCatalog(cfg.index_db_path)
    try:
        sub = args.anim_cmd
        if sub == "list":
            if args.machine:
                oid, _code, disp = resolve_machine_entity_id(game, cat, args.machine)
                all_clips = discover_entity_animations(
                    game,
                    entity_group_id=oid.group_id,
                    asset_kind="machine",
                    limit=None,
                )
                kind_lbl = "machine"
                target_lbl = f"{disp} ({oid})"
            else:
                gender = (
                    "male"
                    if (args.character and "male" in args.character.lower() and "female" not in args.character.lower())
                    else "female"
                )
                all_clips = discover_entity_animations(
                    game,
                    entity_group_id=None,
                    asset_kind="character",
                    gender=gender,
                    limit=None,
                )
                kind_lbl = "character"
                target_lbl = f"{args.character or 'Aloy'} ({gender})"

            cat_counts = summarize_animation_categories(all_clips)
            filtered = all_clips
            cat_arg = (getattr(args, "category", "") or "").strip()
            if cat_arg and cat_arg.lower() != "all":
                wanted = {normalize_animation_category(tok) for tok in cat_arg.split(",") if tok.strip()}
                filtered = [c for c in filtered if c.category in wanted]
            q = (getattr(args, "query", "") or "").strip().lower()
            if q:
                filtered = [
                    c
                    for c in filtered
                    if q in c.name.lower()
                    or q in c.description.lower()
                    or q in c.anim_id.lower()
                    or any(q in ev.lower() for ev in c.events)
                ]
            lim = int(getattr(args, "limit", 0) or 0)
            shown = filtered[:lim] if lim > 0 else filtered

            if getattr(args, "json", False):
                payload = {
                    "kind": kind_lbl,
                    "target": target_lbl,
                    "total": len(all_clips),
                    "filtered_total": len(filtered),
                    "count": len(shown),
                    "category_counts": cat_counts,
                    "categories_meta": [{"key": k, "label": lbl} for k, lbl in ANIMATION_CATEGORIES],
                    "items": [
                        {
                            "anim_id": c.anim_id,
                            "name": c.name,
                            "category": c.category,
                            "description": c.description,
                            "skeleton_id": c.skeleton_id,
                            "duration": round(c.duration, 3),
                            "sample_rate": round(c.sample_rate, 2),
                            "num_frames": c.num_frames,
                            "num_joints": c.num_joints,
                            "yaw_delta_deg": round(c.yaw_delta_deg, 1),
                            "events": list(c.events[:8]),
                        }
                        for c in shown
                    ],
                }
                print(json.dumps(payload))
                return 0

            summary_parts = [f"all={cat_counts['all']}"] + [
                f"{k}={cat_counts.get(k, 0)}" for k, _ in ANIMATION_CATEGORIES if cat_counts.get(k, 0) > 0
            ]
            print(f"Discovered {len(all_clips)} HFW animation clip(s) for {kind_lbl} '{target_lbl}' (showing {len(shown)}):")
            print(f"  Categories: {' | '.join(summary_parts)}")
            for c in shown:
                print(
                    f"  {c.anim_id:10s} | {c.category:8s} | {c.duration:5.2f}s ({c.num_frames:3d}f) | "
                    f"{c.name:36s} | {c.description}"
                )
            return 0

        if sub == "export-pack":
            from hfw_toolkit.decima import ObjectId
            from hfw_toolkit.extract import FEMALE_HUMAN_SKELETON_ID, MALE_HUMAN_SKELETON_ID, extract_skeleton

            spec = (args.animations or "all").strip()
            if args.machine:
                oid, code, disp = resolve_machine_entity_id(game, cat, args.machine)
                asm = assemble_machine(
                    game,
                    oid,
                    codename=code,
                    display_name=disp,
                    variant_index=0,
                    lod_index=0,
                    include_broken_states=False,
                )
                skel = asm.skeleton
                ent_gid = oid.group_id
                akind = "machine"
                gender = "female"
                default_stem = re.sub(r"[^A-Za-z0-9_]+", "_", disp).strip("_") or "machine"
            else:
                ch_txt = (args.character or "female").strip()
                gender = "male" if ("male" in ch_txt.lower() and "female" not in ch_txt.lower()) else "female"
                skel_id = MALE_HUMAN_SKELETON_ID if gender == "male" else FEMALE_HUMAN_SKELETON_ID
                skel_obj = game.read_object(ObjectId.parse(skel_id), read_subgroups=True)
                skel = extract_skeleton(game, skel_obj)
                ent_gid = None
                akind = "character"
                default_stem = f"humanoid_{gender}"

            out_dir = Path(args.out) if args.out else (cfg.exports_dir / "anim_packs" / f"{default_stem}_pack")
            export_animation_pack(
                game,
                skel,
                out_dir,
                entity_group_id=ent_gid,
                asset_kind=akind,
                animations_spec=spec,
                gender=gender,
            )
            manifest = json.loads((out_dir / "assembly.json").read_text(encoding="utf-8"))
            n_exp = len(manifest.get("animations", ()))
            print(f"Exported {n_exp} animation clip(s) (spec={spec!r}) -> {out_dir}")
            return 0

        return 1
    finally:
        cat.close()
        game.close()


def _cmd_update(args: argparse.Namespace, cfg: ToolkitConfig) -> int:
    from hfw_toolkit.updater import apply_update, check_for_updates

    branch = args.branch or cfg.update_branch or "main"
    if args.enable_auto is not None:
        cfg.auto_update = bool(args.enable_auto)
        cfg.update_branch = branch
        cfg.save()
        print(f"[HFWToolkit] Auto-updates ({branch}) set to: {'ENABLED' if cfg.auto_update else 'DISABLED'}")
        if not args.apply and not args.check:
            return 0

    if args.apply:
        print(f"[HFWToolkit] Updating from GitHub ({branch} branch)...")
        res = apply_update(cfg, branch=branch, force=bool(args.force), reinstall_deps=not args.no_deps)
        print(f"  Status  : {res['status'].upper()}")
        print(f"  Message : {res['message']}")
        for step in res.get("steps") or []:
            print(f"    • {step}")
        return 0 if res["status"] in ("updated", "up_to_date") else 1

    st = check_for_updates(cfg, branch=branch)
    loc = st["local"]
    rem = st.get("remote")
    print("=" * 72)
    print(" HFWToolkit — GitHub Update Check")
    print("=" * 72)
    print(f"  Repository    : {st['repo']} (target branch: {st['target_branch']})")
    print(f"  Install Mode  : {loc['install_mode']} (branch: {loc['branch']}, commit: {loc['short_sha']})")
    if st.get("error"):
        print(f"  Remote Check  : {st['error']}")
        return 1
    if rem:
        print(f"  Remote Latest : {rem['short_sha']} ({rem['commit_date']}) — {rem['commit_message']}")
    if st["update_available"]:
        print("  Result        : UPDATE AVAILABLE — run `hfw-toolkit update --apply` to update.")
    elif st.get("ahead_of_remote"):
        print("  Result        : Local checkout already includes the latest main commit (ahead of main).")
    else:
        print("  Result        : Up to date!")
    print(f"  Auto-Update   : {'ENABLED' if cfg.auto_update else 'DISABLED (opt-in via `hfw-toolkit update --auto-on`)'}")
    print("=" * 72)
    return 0


def _cmd_web(args: argparse.Namespace, cfg: ToolkitConfig) -> int:
    import uvicorn
    from hfw_toolkit.web import create_app

    cfg.ensure_dirs()
    if cfg.auto_update:
        try:
            from hfw_toolkit.updater import apply_update, check_for_updates

            st = check_for_updates(cfg)
            if st.get("update_available"):
                print(f"[HFWToolkit] Auto-update enabled — updating from GitHub ({st['target_branch']})...")
                res = apply_update(cfg)
                print(f"[HFWToolkit] {res['message']}")
        except Exception as exc:
            print(f"[HFWToolkit] Startup auto-update skipped: {exc}")
    cat = AssetCatalog(cfg.index_db_path)
    built = cat.is_built()
    cat.close()
    if not built and cfg.package_dir.exists():
        print("Catalog index not built yet — building `cache/hfw_index.sqlite` first...")
        _cmd_index(argparse.Namespace(rebuild=False), cfg)

    url = f"http://{args.host}:{args.port}"
    print(f"Starting HFWToolkit Material & Shader Deep-Dive Studio at {url}")
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    app = create_app(cfg)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hfw-toolkit",
        description="Horizon Forbidden West PC — Automated Machine Blender/UE5 Assembler & Shader Deep-Dive Explorer",
    )
    parser.add_argument("--game-dir", default=None, help="Override path to Horizon Forbidden West Complete Edition directory")
    parser.add_argument("--blender-exe", default=None, help="Override path to Blender 4.2+/5.x executable")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="Check HFW install path, game version, zero-setup RTTI schema, Blender & catalog status")

    p_setup = sub.add_parser("setup", help="Auto-detect or configure HFW & Blender paths, build catalog, and install Blender add-on")
    p_setup.add_argument("--game-dir", dest="setup_game_dir", default=None, help="Path to Horizon Forbidden West Complete Edition folder")
    p_setup.add_argument("--blender-exe", dest="setup_blender_exe", default=None, help="Path to blender.exe (Blender 4.2+ / 5.x)")
    p_setup.add_argument("--rebuild-index", action="store_true", help="Force rebuild of SQLite catalog index")
    p_setup.add_argument("--no-index", action="store_true", help="Skip building SQLite catalog index")
    p_setup.add_argument("--no-addon", action="store_true", help="Skip installing Blender N-Panel add-on")

    p_rtti = sub.add_parser("dump-rtti", help="Statically extract Decima RTTI schema from HorizonForbiddenWest.exe on disk (no injector needed)")
    p_rtti.add_argument("--out", default=None, help="Output JSON path (default: dump/hfw.json)")

    p_idx = sub.add_parser("index", help="Build or rebuild the SQLite asset catalog")
    p_idx.add_argument("--rebuild", action="store_true", help="Force full catalog rebuild")

    p_m = sub.add_parser("machines", help="List or search indexed machines")
    p_m.add_argument("query", nargs="?", default="", help="Optional name/codename filter")

    p_b = sub.add_parser("export-bundle", help="Export canonical Assembly Bundle (JSON + NPZ + native packed textures)")
    p_b.add_argument("--machine", required=True, help="Machine name, codename, or 'group:idx'")
    p_b.add_argument("--variant", type=int, default=0)
    p_b.add_argument("--lod", type=int, default=0)
    p_b.add_argument("--no-broken-states", action="store_true")
    p_b.add_argument("--unpack-textures", action="store_true", help="Unpack TextureSets into individual semantic PNG maps instead of native game-packed textures")
    p_b.add_argument("--animations", default=None, help="Include HFW animations: 'all', 'category:combat', 'category:movement,jump', N diverse clips (e.g. '10'), or comma-separated ObjectIds")
    p_b.add_argument("--max-tex-size", type=int, default=None)
    p_b.add_argument("--out", default=None)

    p_bl = sub.add_parser("export-blend", help="One-command extract, assemble & build ready .blend in Blender 5.3")
    p_bl.add_argument("--machine", required=True, help="Machine name, codename, or 'group:idx'")
    p_bl.add_argument("--variant", type=int, default=0)
    p_bl.add_argument("--lod", type=int, default=0)
    p_bl.add_argument("--no-broken-states", action="store_true")
    p_bl.add_argument("--no-detail", action="store_true")
    p_bl.add_argument("--no-control-rig", action="store_true", help="Skip automatic non-destructive IK/FK Control Rig generation")
    p_bl.add_argument("--animations", default=None, help="Include HFW animations: 'all', 'category:combat', 'category:movement', N diverse clips, or comma-separated ObjectIds")
    p_bl.add_argument("--unpack-textures", action="store_true", help="Unpack TextureSets into individual semantic PNG maps instead of native game-packed textures")
    p_bl.add_argument("--pack", action="store_true", help="Pack textures inside .blend")
    p_bl.add_argument("--export-fbx", action="store_true", help="Also export a UE5-ready rigged .fbx alongside ue5_import_machine.py")
    p_bl.add_argument("--max-tex-size", type=int, default=None)
    p_bl.add_argument("--out", default=None)

    p_ue = sub.add_parser("export-ue5", help="Export rigged FBX + Control Rig + animations + textures + Unreal Engine 5 Python import script")
    p_ue.add_argument("--machine", required=True, help="Machine name, codename, or 'group:idx'")
    p_ue.add_argument("--variant", type=int, default=0)
    p_ue.add_argument("--lod", type=int, default=0)
    p_ue.add_argument("--no-broken-states", action="store_true")
    p_ue.add_argument("--no-control-rig", action="store_true")
    p_ue.add_argument("--animations", default=None, help="Include HFW animations: 'all', 'category:combat', 'category:movement', N diverse clips, or comma-separated ObjectIds")
    p_ue.add_argument("--unpack-textures", action="store_true", help="Unpack semantic PNG maps instead of native packed textures")
    p_ue.add_argument("--max-tex-size", type=int, default=None)
    p_ue.add_argument("--out", default=None)

    p_an = sub.add_parser("animations", help="Discover, filter & export HFW in-game SkeletonAnimationResource clips for machines or characters")
    an_sub = p_an.add_subparsers(dest="anim_cmd", required=True)
    p_an_ls = an_sub.add_parser("list", help="List available EA08 animation clips with categories & plain-English descriptions")
    p_an_ls.add_argument("--machine", default="", help="Machine name, codename, or 'group:idx'")
    p_an_ls.add_argument("--character", default="", help="Character name/gender (e.g. Aloy, Erend, male)")
    p_an_ls.add_argument("--category", default="", help="Filter by category (combat, movement, idle, jump, emote, mounted; comma-separated or 'all')")
    p_an_ls.add_argument("--query", "-q", default="", help="Search text across clip names, descriptions & event names")
    p_an_ls.add_argument("--limit", type=int, default=0, help="Max rows to show (0 = show ALL clips)")
    p_an_ls.add_argument("--json", action="store_true", help="Emit structured JSON (used by Blender N-Panel Live Browser)")

    p_an_pk = an_sub.add_parser("export-pack", help="Fast animation-only pack export (assembly.json + animations.npz) for adding clips onto an existing Blender rig")
    p_an_pk.add_argument("--machine", default="", help="Machine name, codename, or 'group:idx'")
    p_an_pk.add_argument("--character", default="", help="Character name/gender (e.g. Aloy, female, male)")
    p_an_pk.add_argument("--animations", default="all", help="Animation selection: 'all', 'category:combat', 'category:movement,jump', N clips, or comma-separated ObjectIds")
    p_an_pk.add_argument("--out", default=None, help="Output directory for assembly.json + animations.npz")

    p_im = sub.add_parser("inspect-material", help="Deep-analyze a RenderEffectResource & export UE5/HLSL reports")
    p_im.add_argument("material_id", help="RenderEffectResource ObjectId ('group:idx')")
    p_im.add_argument("--decompile", action="store_true", help="Also invoke optional dxil-spirv + spirv-cross")
    p_im.add_argument("--out-dir", default=None, help="Directory to save JSON, Pseudo-HLSL, and UE5 files")

    p_ad = sub.add_parser("install-blender-addon", help="Install HFW Toolkit N-Panel Add-on into Blender")
    p_ad.add_argument("--blender-version", default=None, help="Optional specific Blender version (e.g. '5.3'); auto-detected if omitted")

    p_w = sub.add_parser("web", help="Launch the local Material, Shader & Audio Studio Web Explorer")
    p_w.add_argument("--host", default="127.0.0.1")
    p_w.add_argument("--port", type=int, default=8420)
    p_w.add_argument("--no-browser", action="store_true")

    p_up = sub.add_parser("update", help="Check for or apply updates from GitHub main branch (git or ZIP install)")
    p_up.add_argument("--check", action="store_true", help="Check GitHub main branch for newer commits (default)")
    p_up.add_argument("--apply", action="store_true", help="Download & apply latest update from GitHub")
    p_up.add_argument("--branch", default=None, help="Override target GitHub branch (default: main)")
    p_up.add_argument("--force", action="store_true", help="Allow updating even with local git changes / non-main branch")
    p_up.add_argument("--no-deps", action="store_true", help="Skip refreshing pip dependencies after update")
    p_up.add_argument("--auto-on", dest="enable_auto", action="store_true", default=None, help="Enable opt-in auto-updates on startup")
    p_up.add_argument("--auto-off", dest="enable_auto", action="store_false", help="Disable auto-updates on startup")

    # Audio / Music / SFX / Voicelines subcommand group
    p_au = sub.add_parser("audio", help="Browse, search, index & extract HFW Music, SFX, and Voicelines")
    au_sub = p_au.add_subparsers(dest="audio_cmd", required=True)

    au_sub.add_parser("stats", help="Show audio catalog counts, installed languages & ATRAC9 decoder status")

    p_au_idx = au_sub.add_parser("index", help="Build fast Tier-1 audio index (or --deep for all 168k SFX & subtitles)")
    p_au_idx.add_argument("--rebuild", action="store_true", help="Force rebuild of Tier-1 Music/Voices/Directory index")
    p_au_idx.add_argument("--deep", action="store_true", help="Deep-index all SFX clips & spoken voiceline subtitles into SQLite")

    p_au_ls = au_sub.add_parser("list", help="Search & list Music cues, SFX directories, or Voicelines")
    p_au_ls.add_argument("--category", choices=("music", "sfx", "voicelines"), default="music")
    p_au_ls.add_argument("--subcat", default=None, help="Filter by developer subcategory (e.g. Boss, MainQuests, Robots, SideQuests)")
    p_au_ls.add_argument("--query", "-q", default="", help="Search text (song name, machine, quest, SFX name, or spoken quote)")
    p_au_ls.add_argument("--speaker", default=None, help="Filter voicelines by character speaker (e.g. Aloy, Sylens, Zo)")
    p_au_ls.add_argument("--group", type=int, default=None, help="List individual clips inside a specific Decima group ID")
    p_au_ls.add_argument("--limit", type=int, default=50)

    p_au_ex = au_sub.add_parser("extract", help="Extract Music, SFX, or Voicelines (preserving developer directory structure or flat)")
    p_au_ex.add_argument("--category", choices=("music", "sfx", "voicelines", "all"), default="music")
    p_au_ex.add_argument("--subcat", default=None, help="Filter by developer subcategory (e.g. Boss, Exploration, Robots, MainQuests)")
    p_au_ex.add_argument("--query", "-q", default="", help="Filter by cue name, machine/SFX folder, quest, or subtitle")
    p_au_ex.add_argument("--speaker", default=None, help="Filter voicelines by speaker name (e.g. Aloy, Sylens, Zo, Jaxx)")
    p_au_ex.add_argument("--cue", default=None, help="Extract one specific MusicCue by ID (e.g. 3:3688)")
    p_au_ex.add_argument("--group", type=int, default=None, help="Extract all SFX or voicelines from a specific Decima group ID")
    p_au_ex.add_argument("--id", default=None, help="Extract one specific WaveResource or LocalizedSimpleSoundResource ('group:idx')")
    p_au_ex.add_argument("--lang", default="en", help="Spoken language code(s) comma-separated, or 'all' installed (default: en)")
    p_au_ex.add_argument("--format", choices=SUPPORTED_OUTPUT_FORMATS, default="wav", help="Output audio format (default: wav)")
    p_au_ex.add_argument("--preserve-structure", action="store_true", default=True, help="Preserve Guerrilla developer directory hierarchy (default)")
    p_au_ex.add_argument("--flat", action="store_true", help="Output all files flat into --out without subdirectories")
    p_au_ex.add_argument("--no-mix-cues", action="store_true", help="Skip rendering beat-synced full-song mixdowns for multi-stem MusicCues")
    p_au_ex.add_argument("--no-stems", action="store_true", help="For multi-stem MusicCues, export only the combined mixed song (skip individual stems)")
    p_au_ex.add_argument("--group-by", choices=("quest", "speaker"), default="quest", help="Voicelines folder grouping when preserving structure")
    p_au_ex.add_argument("--include-ost", action="store_true", help="Also copy bonus DigitalExtras/OST MP3 tracks")
    p_au_ex.add_argument("--no-transcript", action="store_true", help="Do not write voicelines_transcript.csv alongside extracted dialogue")
    p_au_ex.add_argument("--limit", type=int, default=None, help="Optional maximum number of cues/items to extract")
    p_au_ex.add_argument("--out", default=None, help="Output directory (default: exports/Audio)")

    # Characters & Outfits subcommand group
    p_ch = sub.add_parser("characters", help="Search, resolve via natural-language prompt & export HFW characters, outfits & rigs")
    ch_sub = p_ch.add_subparsers(dest="char_cmd", required=True)

    p_ch_ls = ch_sub.add_parser("list", help="List indexed HFW characters (Aloy, Companions, Tribal NPCs)")
    p_ch_ls.add_argument("--kind", choices=("hero", "companion", "npc"), default="")
    p_ch_ls.add_argument("--tribe", default="")
    p_ch_ls.add_argument("--gender", default="")
    p_ch_ls.add_argument("--query", "-q", default="")
    p_ch_ls.add_argument("--limit", type=int, default=60)

    p_ch_of = ch_sub.add_parser("outfits", help="List outfit variations (e.g. all 35 Aloy base + DLC outfits)")
    p_ch_of.add_argument("--char", default="aloy")
    p_ch_of.add_argument("--query", "-q", default="")
    p_ch_of.add_argument("--limit", type=int, default=80)

    p_ch_res = ch_sub.add_parser("resolve", help="Test natural-language prompt resolution (e.g. 'Aloy Carja Blazon no headgear')")
    p_ch_res.add_argument("prompt", help="Natural-language character description")

    for sub_name, sub_help in (
        ("export-bundle", "Export rigged character bundle (JSON + NPZ + textures + ue5_import_character.py)"),
        ("export-blend", "One-command assemble & build ready-to-pose Blender 5.3 character .blend"),
        ("export-ue5", "Export UE5-ready rigged character FBX + textures + ue5_import_character.py"),
    ):
        p_cx = ch_sub.add_parser(sub_name, help=sub_help)
        p_cx.add_argument("--prompt", "-p", default="", help="Natural-language prompt (e.g. 'Aloy Carja Blazon no headgear', 'Seyka', 'Erend', 'Tenakth female warrior')")
        p_cx.add_argument("--char", default="", help="Character ID (e.g. aloy, erend, seyka, bv:10366:18)")
        p_cx.add_argument("--outfit", default="", help="Outfit ID (e.g. osd:72479:0)")
        p_cx.add_argument("--facial", default="", help="Optional modular head swap FacialAnimationComponentResource ID")
        p_cx.add_argument("--no-headgear", action="store_true", help="Hide outfit headgear when supported")
        p_cx.add_argument("--no-hair", action="store_true", help="Skip external braided HairResource")
        p_cx.add_argument("--hair-lod", type=int, default=2, help="HairResource LOD (0=ultra, 2=balanced default, 3=light)")
        p_cx.add_argument("--lod", type=int, default=0)
        p_cx.add_argument("--animations", default=None, help="Include HFW humanoid animations: 'all', 'category:combat', 'category:movement', N diverse clips, or comma-separated ObjectIds")
        p_cx.add_argument("--no-control-rig", action="store_true", help="Skip automatic non-destructive IK/FK Control Rig generation")
        p_cx.add_argument("--unpack-textures", action="store_true")
        p_cx.add_argument("--max-tex-size", type=int, default=None)
        p_cx.add_argument("--out", default=None)
        if sub_name == "export-blend":
            p_cx.add_argument("--pack", action="store_true")
            p_cx.add_argument("--export-fbx", action="store_true")

    # General Assets (Guerrilla Developer Directory Catalog) subcommand group
    p_as = sub.add_parser("assets", help="Browse Guerrilla's original developer directory tree & export any HFW asset to Blender / UE5")
    as_sub = p_as.add_subparsers(dest="asset_cmd", required=True)

    p_as_tr = as_sub.add_parser("tree", help="Browse Guerrilla's original models/... developer folder tree")
    p_as_tr.add_argument("folder", nargs="?", default="", help="Optional developer folder path (e.g. models/weapons, models/Eco_Assets_HRZ2)")

    p_as_ls = as_sub.add_parser("list", help="Search indexed general assets across all developer folders")
    p_as_ls.add_argument("--category", choices=("weapons", "animals", "building_blocks", "eco_assets", "landscape", "props", "characters", "machines"), default="")
    p_as_ls.add_argument("--dir", default="", help="Developer folder prefix filter")
    p_as_ls.add_argument("--query", "-q", default="")
    p_as_ls.add_argument("--limit", type=int, default=60)

    for sub_name, sub_help in (
        ("export-bundle", "Export general asset bundle preserving developer folder structure"),
        ("export-blend", "Export any HFW asset as a ready-to-use Blender 5.3 .blend"),
        ("export-ue5", "Export any HFW asset as UE5 FBX + materials + ue5_import_asset.py"),
    ):
        p_ax = as_sub.add_parser(sub_name, help=sub_help)
        p_ax.add_argument("--asset", "-a", required=True, help="Developer folder path (e.g. DLC/models/weapons/Bow_Heavy_Quen_SE) or resource ID ('85387:14')")
        p_ax.add_argument("--lod", type=int, default=0)
        p_ax.add_argument("--unpack-textures", action="store_true")
        p_ax.add_argument("--max-tex-size", type=int, default=None)
        p_ax.add_argument("--out", default=None)
        if sub_name == "export-blend":
            p_ax.add_argument("--pack", action="store_true")
            p_ax.add_argument("--export-fbx", action="store_true")

    args = parser.parse_args(argv)
    cfg = ToolkitConfig.load()
    game_override = args.game_dir or getattr(args, "setup_game_dir", None)
    blender_override = args.blender_exe or getattr(args, "setup_blender_exe", None)
    if game_override:
        cfg.game_dir = Path(game_override)
    if blender_override:
        cfg.blender_exe = Path(blender_override)

    dispatch = {
        "doctor": _cmd_doctor,
        "setup": _cmd_setup,
        "dump-rtti": _cmd_dump_rtti,
        "index": _cmd_index,
        "machines": _cmd_machines,
        "export-bundle": _cmd_export_bundle,
        "export-blend": _cmd_export_blend,
        "export-ue5": _cmd_export_ue5,
        "inspect-material": _cmd_inspect_material,
        "install-blender-addon": _cmd_install_addon,
        "audio": _cmd_audio,
        "animations": _cmd_animations,
        "characters": _cmd_characters,
        "assets": _cmd_assets,
        "update": _cmd_update,
        "web": _cmd_web,
    }
    return dispatch[args.cmd](args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
