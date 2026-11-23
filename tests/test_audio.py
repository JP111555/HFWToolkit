"""Integration & regression tests for HFW Audio, Music, SFX & Voicelines extraction, CLI & Web API."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import uvicorn

from hfw_toolkit.config import ToolkitConfig
from hfw_toolkit.decima import AssetCatalog, HFWGame
from hfw_toolkit.extract.audio import (
    SUPPORTED_OUTPUT_FORMATS,
    decode_audio_object,
    detect_installed_languages,
    encode_audio_bytes,
    extract_audio_items_to_dir,
    extract_music_cues_to_dir,
    extract_sfx_group_items,
    extract_voiceline_group_items,
    has_pyav,
    render_music_cue_mix,
)
from hfw_toolkit.web import create_app


@pytest.fixture(scope="module")
def game() -> HFWGame:
    cfg = ToolkitConfig.load()
    g = HFWGame(cfg)
    yield g
    g.close()


@pytest.fixture(scope="module")
def catalog(game: HFWGame) -> AssetCatalog:
    cat = AssetCatalog(game.config.index_db_path)
    if not cat.is_audio_built():
        cat.build_audio_index(game)
    yield cat
    cat.close()


def test_audio_codecs_and_all_output_formats(game: HFWGame) -> None:
    """Verify native PyAV Sony ATRAC9 & raw PCM decoding plus WAV/FLAC/MP3/OGG/AT9 encoding."""
    assert has_pyav(), "PyAV with atrac9 decoder must be available"
    langs = detect_installed_languages(game)
    assert "en" in langs

    # 1. Streamed stereo ATRAC9 music wave (Group 3)
    dec_music = decode_audio_object(game, "3:1895")
    assert dec_music.sample_rate == 48000
    assert dec_music.channels == 2
    assert dec_music.duration_sec > 50.0
    assert float(np.sqrt(np.mean(dec_music.samples.astype(np.float64) ** 2))) > 100.0

    # 2. Raw uncompressed 16-bit PCM SFX wave (Group 15628)
    dec_pcm = decode_audio_object(game, "15628:0")
    assert dec_pcm.encoding == 0
    assert dec_pcm.sample_rate == 48000
    assert len(dec_pcm.samples) == 20719
    assert int(np.max(np.abs(dec_pcm.samples))) > 1000

    # 3. Streamed English voiceline ATRAC9 (Group 65570) -> verify ALL 5 output formats
    dec_vo = decode_audio_object(game, "65570:2", lang="en")
    assert dec_vo.sample_rate == 48000
    assert dec_vo.channels == 1
    assert 2.0 < dec_vo.duration_sec < 3.0
    assert float(np.sqrt(np.mean(dec_vo.samples.astype(np.float64) ** 2))) > 500.0

    expected_magics = {
        "wav": b"RIFF",
        "flac": b"fLaC",
        "mp3": b"ID3",
        "ogg": b"OggS",
        "at9": b"RIFF",
    }
    for fmt in SUPPORTED_OUTPUT_FORMATS:
        blob = encode_audio_bytes(dec_vo, fmt=fmt)
        assert len(blob) > 1000
        assert blob.startswith(expected_magics[fmt]), f"Unexpected magic for {fmt}: {blob[:8]!r}"


def test_music_catalog_mixdown_and_directory_modes(
    game: HFWGame, catalog: AssetCatalog, tmp_path: Path
) -> None:
    """Verify full MusicProject catalog, beat-synced multi-stem cue mixdown, and structured vs flat export."""
    st = catalog.audio_stats()
    assert st["music_cues"] >= 1300
    assert st["music_stems"] >= 4800

    # Verify Burning Shores End Credits + Slaughterspine Boss Cue exist in catalog
    credits_hits = catalog.list_music_cues(query="EndCredits", limit=10)
    assert any("H2DLC-EndCredits" in c["cue_name"] for c in credits_hits)

    # Test short 2-stem Slaughterspine charge cue (3:5546, ~30s) in BOTH structured & flat modes
    cue = catalog.get_music_cue_info("3:5546")
    assert cue is not None
    assert "PLASMASPINE" in cue.cue_name.upper()
    assert cue.dev_folder.startswith("Music/MainScore/Boss/Robots/")
    assert len(cue.stems) >= 2

    mixed = render_music_cue_mix(game, cue)
    assert mixed.channels == 2
    assert mixed.sample_rate == 48000
    assert mixed.duration_sec > 20.0
    assert float(np.sqrt(np.mean(mixed.samples.astype(np.float64) ** 2))) > 100.0

    # Preserved developer directory structure
    out_struct = tmp_path / "music_structured"
    res_s = extract_music_cues_to_dir(
        game, [cue], out_struct, fmt="flac", preserve_structure=True, mix_cues=True, export_stems=True
    )
    assert res_s["exported_count"] == 1 + len(cue.stems)
    assert not res_s["errors"]
    rel_paths = [it["rel_path"] for it in res_s["items"]]
    assert any(p.startswith("Music/MainScore/Boss/Robots/") and "_FULL_MIX.flac" in p for p in rel_paths)
    assert any("/Stems/" in p and p.endswith(".flac") for p in rel_paths)

    # Flat output mode
    out_flat = tmp_path / "music_flat"
    res_f = extract_music_cues_to_dir(
        game, [cue], out_flat, fmt="mp3", preserve_structure=False, mix_cues=True, export_stems=False
    )
    assert res_f["exported_count"] == 1
    assert "/" not in res_f["items"][0]["rel_path"]
    assert res_f["items"][0]["rel_path"].endswith(".mp3")


def test_sfx_and_voicelines_structure_vs_flat(
    game: HFWGame, catalog: AssetCatalog, tmp_path: Path
) -> None:
    """Verify SFX GraphSoundResource developer paths & Voiceline speaker/subtitle/transcript extraction."""
    # 1. SFX: Group 2880 (Bilegut / Frog)
    sfx_items = extract_sfx_group_items(game, 2880)
    assert len(sfx_items) >= 700
    acid_drip = [it for it in sfx_items if "JumpingSlop_AcidDrip" in it.sound_graph_name]
    assert len(acid_drip) >= 3
    assert "SFX/Robots/Bilegut" in acid_drip[0].dev_folder

    res_sfx_s = extract_audio_items_to_dir(
        game, acid_drip[:2], tmp_path / "sfx_struct", fmt="wav", preserve_structure=True
    )
    assert res_sfx_s["exported_count"] == 2
    assert all(
        "SFX/Robots/Bilegut" in it["rel_path"] and "JumpingSlop_AcidDrip" in it["rel_path"]
        for it in res_sfx_s["items"]
    )

    res_sfx_f = extract_audio_items_to_dir(
        game, acid_drip[:2], tmp_path / "sfx_flat", fmt="wav", preserve_structure=False
    )
    assert res_sfx_f["exported_count"] == 2
    assert all("/" not in it["rel_path"] for it in res_sfx_f["items"])

    # 2. Voicelines: Group 65570 (Jaxx — SQ07 BossMachineFight)
    voices = catalog.get_voice_names(game)
    assert voices.get("1:9") == "Aloy"
    assert voices.get("1:64") == "Sylens"

    vo_items = extract_voiceline_group_items(game, 65570, voice_names=voices)
    assert len(vo_items) == 1
    assert vo_items[0].speaker == "Jaxx"
    assert "fun to us" in vo_items[0].subtitle
    assert "SideQuests/SQ07/" in vo_items[0].dev_folder

    res_vo = extract_audio_items_to_dir(
        game,
        vo_items,
        tmp_path / "vo_struct",
        fmt="wav",
        preserve_structure=True,
        languages=["en"],
        group_by="quest",
        write_transcript=True,
    )
    assert res_vo["exported_count"] == 1
    assert res_vo["items"][0]["rel_path"].startswith("Voicelines/en/SideQuests/SQ07/")
    assert "/Jaxx/" in res_vo["items"][0]["rel_path"]
    tpath = Path(res_vo["transcript_path"])
    assert tpath.is_file()
    rows = list(csv.DictReader(tpath.open(encoding="utf-8-sig")))
    assert len(rows) == 1
    assert rows[0]["speaker"] == "Jaxx"
    assert "fun to us" in rows[0]["subtitle"]


def test_audio_cli_and_web_api(tmp_path: Path) -> None:
    """Verify `hfw-toolkit audio` CLI commands AND FastAPI `/api/audio/...` streaming/browse endpoints."""
    # CLI stats & list
    cp_stats = subprocess.run(
        [sys.executable, "-m", "hfw_toolkit.cli", "audio", "stats"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert cp_stats.returncode == 0, cp_stats.stderr
    assert "Music Cues / Stems" in cp_stats.stdout
    assert "PyAV ATRAC9 Codec" in cp_stats.stdout

    # CLI extract single voiceline group cleanly
    cli_out = tmp_path / "cli_vo"
    cp_ex = subprocess.run(
        [
            sys.executable,
            "-m",
            "hfw_toolkit.cli",
            "audio",
            "extract",
            "--category",
            "voicelines",
            "--group",
            "65570",
            "--format",
            "flac",
            "--out",
            str(cli_out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert cp_ex.returncode == 0, cp_ex.stderr
    assert (cli_out / "audio_manifest.json").is_file()
    manifest = json.loads((cli_out / "audio_manifest.json").read_text(encoding="utf-8"))
    assert manifest["count"] == 1
    assert manifest["items"][0]["speaker"] == "Jaxx"

    # Web Studio Audio API endpoints over real HTTP
    def _get_json(url: str):
        with urllib.request.urlopen(url, timeout=60) as resp:
            assert resp.status == 200
            return json.loads(resp.read().decode("utf-8"))

    def _get_bytes(url: str) -> tuple[bytes, str]:
        with urllib.request.urlopen(url, timeout=60) as resp:
            assert resp.status == 200
            return resp.read(), resp.headers.get("Content-Type", "")

    app = create_app(ToolkitConfig.load())
    port = 18422
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(40):
            try:
                _get_json(f"{base}/api/audio/stats")
                break
            except Exception:
                time.sleep(0.25)

        body = _get_json(f"{base}/api/audio/stats")
        assert body["pyav_available"] is True
        assert body["stats"]["music_cues"] >= 1300

        mu = _get_json(f"{base}/api/audio/music?q=PLASMASPINE&limit=10")
        assert len(mu["items"]) >= 3

        grp = _get_json(f"{base}/api/audio/group/65570?kind=voiceline")
        assert len(grp["items"]) == 1
        assert grp["items"][0]["speaker"] == "Jaxx"

        # Live WAV streaming for browser `<audio>` player!
        wav_bytes, ctype = _get_bytes(f"{base}/api/audio/stream/object/65570:2?lang=en&fmt=wav")
        assert "audio/wav" in ctype
        assert wav_bytes[:4] == b"RIFF" and wav_bytes[8:12] == b"WAVE"

        # Live beat-synced MusicCue mixdown stream (3:5606 — 20s Slaughterspine Outro)
        cue_bytes, cue_ct = _get_bytes(f"{base}/api/audio/stream/cue/3:5606?fmt=wav")
        assert "audio/wav" in cue_ct
        assert cue_bytes[:4] == b"RIFF" and len(cue_bytes) > 100_000
    finally:
        server.should_exit = True
        th.join(timeout=10)
