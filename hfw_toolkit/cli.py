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
    SUPPORTED_OUTPUT_FORMATS,
    AudioItemInfo,
    assemble_machine,
    decode_audio_object,
    detect_digital_extras_ost,
    detect_installed_languages,
    extract_audio_items_to_dir,
    extract_music_cues_to_dir,
    has_pyav,
    resolve_machine_entity_id,
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
    if cat_built:
        print(
            f"  Asset Catalog DB  : READY ({cat_stats.get('machines', 0)} machines, "
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
        print(f"[2/3] Exporting textures & UE5 animated shader script -> {bundle_dir}")
        export_assembly_bundle(
            game,
            asm,
            bundle_dir,
            max_texture_size=args.max_tex_size,
            unpack_textures=args.unpack_textures,
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


def _cmd_web(args: argparse.Namespace, cfg: ToolkitConfig) -> int:
    import uvicorn
    from hfw_toolkit.web import create_app

    cfg.ensure_dirs()
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
    p_b.add_argument("--max-tex-size", type=int, default=None)
    p_b.add_argument("--out", default=None)

    p_bl = sub.add_parser("export-blend", help="One-command extract, assemble & build ready .blend in Blender 5.3")
    p_bl.add_argument("--machine", required=True, help="Machine name, codename, or 'group:idx'")
    p_bl.add_argument("--variant", type=int, default=0)
    p_bl.add_argument("--lod", type=int, default=0)
    p_bl.add_argument("--no-broken-states", action="store_true")
    p_bl.add_argument("--no-detail", action="store_true")
    p_bl.add_argument("--unpack-textures", action="store_true", help="Unpack TextureSets into individual semantic PNG maps instead of native game-packed textures")
    p_bl.add_argument("--pack", action="store_true", help="Pack textures inside .blend")
    p_bl.add_argument("--export-fbx", action="store_true", help="Also export a UE5-ready rigged .fbx alongside ue5_import_machine.py")
    p_bl.add_argument("--max-tex-size", type=int, default=None)
    p_bl.add_argument("--out", default=None)

    p_ue = sub.add_parser("export-ue5", help="Export rigged FBX + textures + animated Unreal Engine 5 Python import script")
    p_ue.add_argument("--machine", required=True, help="Machine name, codename, or 'group:idx'")
    p_ue.add_argument("--variant", type=int, default=0)
    p_ue.add_argument("--lod", type=int, default=0)
    p_ue.add_argument("--no-broken-states", action="store_true")
    p_ue.add_argument("--unpack-textures", action="store_true", help="Unpack semantic PNG maps instead of native packed textures")
    p_ue.add_argument("--max-tex-size", type=int, default=None)
    p_ue.add_argument("--out", default=None)

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
        "web": _cmd_web,
    }
    return dispatch[args.cmd](args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
