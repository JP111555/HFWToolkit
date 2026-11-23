"""Horizon Forbidden West PC Audio, Music, SFX & Voiceline Decoder and Extractor.

Reads directly from HFW's Decima `package.*.core(.stream)` archives (strictly read-only):
- **Music**: All `MusicProject` trees (Base Game MainScore, Burning Shores DLC & End Credits,
  Photomode, Tilda's Music Player, Plainsong Chorus) with exact Guerrilla cue names,
  container hierarchy, individual stems, AND optional beat-synced full-cue mixdowns!
- **SFX**: `WaveResource` clips linked via `GraphSoundResource` -> `GraphProgramResource`
  developer names + group authoring paths (`work:` / `source:` / machine codenames / quests).
- **Voicelines**: `LocalizedSimpleSoundResource` linked via `SentenceResource` -> `VoiceResource`
  (exact speaker name: Aloy, Sylens, Zo, Varl, Erend, Kotallo, Alva, GAIA, Tilda, ...) +
  `LocalizedTextResource` (spoken subtitle transcript) across all installed spoken languages.

Decodes Sony ATRAC9 (`.at9` RIFF WAVE), PCM, PCM_FLOAT, ADPCM, XWMA, MP3 & AAC natively
in-process via PyAV (`av`) with zero external executables required.
"""

from __future__ import annotations

import csv
import io
import json
import re
import shutil
import struct
import time
import wave
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from hfw_toolkit.decima.catalog import MACHINE_CODENAMES
from hfw_toolkit.decima.reader import ObjectId, ObjectRef, StreamingRef
from hfw_toolkit.decima.streaming_graph import HFWGame

# EWaveDataEncoding enum from HFW PC RTTI
WAVE_ENCODING_NAMES: dict[int, str] = {
    0: "PCM",
    1: "PCM_FLOAT",
    2: "XWMA",
    3: "ATRAC9",
    4: "MP3",
    5: "ADPCM",
    6: "AAC",
}

# Exact 12 spoken audio language slots in `LocalizedSimpleSoundResource.LocalizedDataSources`
SPOKEN_LANGUAGES: tuple[str, ...] = (
    "en",
    "fr",
    "es",
    "de",
    "it",
    "pt",
    "ru",
    "pl",
    "ja",
    "es-LA",
    "pt-BR",
    "ar",
)

SPOKEN_LANGUAGE_LABELS: dict[str, str] = {
    "en": "English",
    "fr": "French",
    "es": "Spanish (Spain)",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "pl": "Polish",
    "ja": "Japanese",
    "es-LA": "Spanish (Latin America)",
    "pt-BR": "Portuguese (Brazil)",
    "ar": "Arabic",
}

SUPPORTED_OUTPUT_FORMATS: tuple[str, ...] = ("wav", "flac", "mp3", "ogg", "at9")

_MUSIC_GROUPS: tuple[int, ...] = (3, 1, 78260, 84481)

_INVALID_PATH_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')
_WORK_PATH_RE = re.compile(rb"(?:work|source):([A-Za-z0-9_./ -]{4,160})", re.I)
_ROBOT_DIR_RE = re.compile(r"models/characters/robots/([A-Za-z0-9_]+)/", re.I)
_CHAR_DIR_RE = re.compile(r"models/characters/([A-Za-z0-9_]+)/([A-Za-z0-9_]+)/", re.I)
_WEAPON_DIR_RE = re.compile(r"models/weapons/([A-Za-z0-9_]+)(?:/([A-Za-z0-9_]+))?", re.I)
_DOMAIN_DIR_RE = re.compile(r"AI/Domains/(?:DLC/)?([A-Za-z0-9_]+(?:/[A-Za-z0-9_]+)?)", re.I)
_EP_SPAN_RE = re.compile(rb"EntryPoint_([A-Za-z0-9_]{3,96})_[0-9a-f]{32}_")
_QUEST_CODE_RE = re.compile(r"(?<![A-Za-z0-9])((?:MQ|SQ|RF|CR|TN|CA|EQ|DLC)\d{1,3}|MQHUB)(?![A-Za-z0-9])", re.I)
_WINDOWS_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)

_GENERIC_ROBOT_CODES = frozenset(
    {
        "shared",
        "common",
        "generic",
        "shared_textures",
        "detail",
        "weapons",
        "components",
        "generic_components",
        "accessories",
        "canisters",
        "canister",
        "ammo",
    }
)


@dataclass(slots=True)
class DecodedAudio:
    """Decoded 16-bit signed PCM audio waveform alongside original raw bytes."""

    samples: np.ndarray  # shape: (num_frames, channels), dtype=int16
    sample_rate: int
    channels: int
    encoding: int
    encoding_name: str
    duration_sec: float
    raw_bytes: bytes = b""


@dataclass(slots=True)
class MusicStemInfo:
    """One `MusicClipEvent` -> `WaveResource` stem inside a `MusicCue`."""

    wave_id: str
    track_index: int
    start_beats: float
    duration_sec: float
    gain: float
    fade_in_sec: float
    fade_out_sec: float
    sample_rate: int
    channels: int
    sample_count: int
    encoding: int
    is_streaming: bool
    data_size: int


@dataclass(slots=True)
class MusicCueInfo:
    """A full interactive `MusicCue` containing one or more synchronized stems."""

    cue_id: str
    project_name: str
    cue_name: str
    category: str
    dev_folder: str
    branch_path: str
    initial_tempo: float
    play_as_one_shot: bool
    is_transition: bool
    tempo_events: list[tuple[float, float]] = field(default_factory=list)
    stems: list[MusicStemInfo] = field(default_factory=list)

    @property
    def duration_sec(self) -> float:
        if not self.stems:
            return 0.0
        # Approximate total span across stems using initial tempo
        bpm = self.initial_tempo if self.initial_tempo > 10.0 else 120.0
        sec_per_beat = 60.0 / bpm
        end_t = 0.0
        for st in self.stems:
            t0 = st.start_beats * sec_per_beat
            dur = st.duration_sec or (st.sample_count / max(st.sample_rate, 1))
            if t0 + dur > end_t:
                end_t = t0 + dur
        return end_t


@dataclass(slots=True)
class AudioItemInfo:
    """Unified descriptor for any extractable HFW audio clip (music stem, SFX, or voiceline)."""

    object_id: str  # "group:idx"
    kind: str  # "music" | "sfx" | "voiceline"
    name: str
    dev_folder: str  # Reconstructed Guerrilla developer folder path (POSIX slashes)
    filename_stem: str  # Clean collision-free file stem
    group_id: int
    sample_rate: int
    channels: int
    sample_count: int
    duration_sec: float
    encoding: int
    encoding_name: str
    data_size: int
    is_streaming: bool
    # Optional rich metadata
    cue_id: str = ""
    track_index: int = -1
    sound_graph_name: str = ""
    owner_name: str = ""
    speaker: str = ""
    subtitle: str = ""
    languages: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# String / Path Helpers
# ---------------------------------------------------------------------------


def clean_entrypoint_name(ep: str) -> str:
    """Decode a Decima `GraphProgramResource` EntryPoint symbol into its original Guerrilla name."""
    if not ep:
        return ""
    s = ep
    if s.startswith("EntryPoint_"):
        s = s[len("EntryPoint_") :]
    # Strip trailing _<32hex>_<index>_<Action>
    s = re.sub(r"_[0-9a-f]{32}_\d+_[A-Za-z0-9_]+$", "", s)
    # Decode Decima _xHH_ hex escapes (e.g. _x2d_ -> '-', _x20_ -> ' ')
    s = re.sub(r"_x([0-9a-fA-F]{2})_", lambda m: chr(int(m.group(1), 16)), s)
    # Strip trailing Graph / Script / GraphResource boilerplate
    s = re.sub(r"[\s_]+(?:Graph|GraphResource|ConditionGraph)$", "", s)
    s = re.sub(r"^_+", "", s).strip()
    return s


def sanitize_path_segment(name: str, max_len: int = 72, fallback: str = "unnamed") -> str:
    """Sanitize a string into a safe Windows/POSIX directory or filename segment."""
    if not name:
        return fallback
    s = _INVALID_PATH_CHARS_RE.sub("_", name.strip())
    s = re.sub(r"\s+", " ", s).strip(" ._-")
    if not s:
        return fallback
    if len(s) > max_len:
        s = s[:max_len].rstrip(" ._-")
    if s.upper() in _WINDOWS_RESERVED:
        s = f"_{s}"
    return s or fallback


def sanitize_rel_folder(folder: str) -> Path:
    """Convert a POSIX-style developer folder string into a safe relative `Path`."""
    parts = [sanitize_path_segment(p, max_len=64) for p in folder.replace("\\", "/").split("/") if p.strip()]
    return Path(*parts) if parts else Path("Misc")


def _slugify_subtitle(text: str, max_len: int = 44) -> str:
    if not text:
        return ""
    s = re.sub(r"[\r\n\t]+", " ", text)
    s = re.sub(r"[^A-Za-z0-9 ]+", "", s)
    s = re.sub(r"\s+", "_", s.strip())
    if len(s) > max_len:
        s = s[:max_len].rstrip("_")
    return s


def detect_installed_languages(game: HFWGame) -> list[str]:
    """Return spoken language codes whose `.core.stream` folders exist in this HFW install."""
    pkg = game.config.package_dir
    installed: list[str] = []
    for lang in SPOKEN_LANGUAGES:
        ldir = pkg / lang
        if ldir.is_dir() and any(ldir.glob("*.stream")):
            installed.append(lang)
    return installed


def detect_digital_extras_ost(game: HFWGame) -> list[Path]:
    """Return any bonus MP3 soundtrack tracks shipped in `DigitalExtras/OST`."""
    ost_dir = game.config.game_dir / "DigitalExtras" / "OST"
    if not ost_dir.is_dir():
        return []
    return sorted(p for p in ost_dir.iterdir() if p.is_file() and p.suffix.lower() in (".mp3", ".flac", ".wav", ".ogg"))


def has_pyav() -> bool:
    """Return True if PyAV (`av`) with `atrac9` decoder is available."""
    try:
        import av  # noqa: F401

        return "atrac9" in av.codecs_available
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Low-Level Decima Audio Reading & PyAV Codec Engine
# ---------------------------------------------------------------------------


def _parse_at9_fact_chunk(raw_bytes: bytes) -> tuple[int | None, int]:
    """Parse Sony AT9 RIFF WAVE `fact` chunk -> `(total_samples, encoder_delay)`."""
    if len(raw_bytes) < 44 or raw_bytes[:4] != b"RIFF" or raw_bytes[8:12] != b"WAVE":
        return None, 0
    pos = 12
    n = len(raw_bytes)
    while pos + 8 <= n:
        cid = raw_bytes[pos : pos + 4]
        csz = struct.unpack_from("<I", raw_bytes, pos + 4)[0]
        if cid == b"fact" and csz >= 4 and pos + 8 + csz <= n:
            if csz >= 12:
                total_samples, _overlap, enc_delay = struct.unpack_from("<III", raw_bytes, pos + 8)
                return int(total_samples), int(enc_delay)
            total_samples = struct.unpack_from("<I", raw_bytes, pos + 8)[0]
            return int(total_samples), 0
        if cid == b"data":
            break
        pos += 8 + csz + (csz & 1)
    return None, 0


def read_wave_raw_bytes(game: HFWGame, wave_obj: dict[str, Any]) -> tuple[bytes, int, int, int, int, int]:
    """Read raw audio bytes & metadata from a `WaveResource` object."""
    enc = int(wave_obj.get("Encoding", 3))
    sr = int(wave_obj.get("SampleRate", 48000)) or 48000
    ch = int(wave_obj.get("ChannelCount", 1)) or 1
    bps = int(wave_obj.get("BitsPerSample", 16)) or 16
    sample_count = int(wave_obj.get("SampleCount", 0))

    wd = wave_obj.get("WaveData")
    if wd and len(wd) > 0:
        raw = bytes(wd)
    else:
        ds = wave_obj.get("StreamingDataSource")
        if isinstance(ds, dict) and ds.get("Length", 0) > 0:
            raw = game.read_data_source(ds)
        else:
            raw = b""
    return raw, enc, sr, ch, bps, sample_count


def _find_lang_slot_for_localized(game: HFWGame, loc_obj: dict[str, Any], lang: str) -> tuple[int, dict[str, Any]] | None:
    """Locate the `LocalizedDataSources` entry matching `lang` (by file path first, then slot index)."""
    lds = loc_obj.get("LocalizedDataSources") or []
    if not lds:
        return None
    lang_l = lang.lower()
    files = game.graph.files

    # 1. Exact stream file path match (e.g. `cache:package/en/package.01.00.core.stream`)
    needle = f"/{lang_l}/"
    for idx, entry in enumerate(lds):
        ds = entry.get("StreamingDataSource")
        if isinstance(ds, dict) and ds.get("Length", 0) > 0:
            fidx = ds.get("_file_index")
            if fidx is not None and 0 <= fidx < len(files):
                if needle in files[fidx].lower():
                    return idx, entry

    # 2. Canonical slot index fallback
    for idx, code in enumerate(SPOKEN_LANGUAGES):
        if code.lower() == lang_l and idx < len(lds):
            entry = lds[idx]
            ds = entry.get("StreamingDataSource")
            if isinstance(ds, dict) and ds.get("Length", 0) > 0:
                return idx, entry
    return None


def read_localized_sound_raw_bytes(
    game: HFWGame, loc_obj: dict[str, Any], lang: str = "en"
) -> tuple[bytes, int, int, int, int, int]:
    """Read raw audio bytes & metadata for spoken language `lang` from a `LocalizedSimpleSoundResource`."""
    found = _find_lang_slot_for_localized(game, loc_obj, lang)
    if found is None:
        raise ValueError(f"Language '{lang}' has no audio stream in {loc_obj.get('_id')}")
    _slot_idx, entry = found
    ds = entry["StreamingDataSource"]
    fidx = ds.get("_file_index")
    if fidx is None or not (0 <= fidx < len(game.graph.files)):
        raise ValueError(f"Invalid stream file index for {loc_obj.get('_id')} ({lang})")

    dev_path = game.graph.files[fidx]
    fs_path = game.config.resolve_device_path(dev_path)
    if not fs_path.is_file():
        installed = detect_installed_languages(game)
        raise FileNotFoundError(
            f"Spoken language pack '{lang}' is not installed ({fs_path.name} missing). "
            f"Installed languages: {', '.join(installed) or 'none'}"
        )

    raw = game.read_data_source(ds)
    swp = loc_obj.get("SharedWaveProperties") or {}
    enc = int(swp.get("Encoding", 3))
    sr = int(swp.get("SampleRate", 48000)) or 48000
    ch = int(swp.get("ChannelCount", 1)) or 1
    bps = int(swp.get("BitsPerSample", 16)) or 16
    sample_count = int(entry.get("SampleCount", 0))
    return raw, enc, sr, ch, bps, sample_count


def decode_audio_bytes(
    raw_bytes: bytes,
    encoding: int,
    sample_rate: int,
    channels: int,
    bits_per_sample: int = 16,
    expected_samples: int = 0,
) -> DecodedAudio:
    """Decode raw HFW audio bytes (ATRAC9, PCM, PCM_FLOAT, ADPCM, MP3, AAC, XWMA) to 16-bit PCM."""
    enc_name = WAVE_ENCODING_NAMES.get(encoding, f"Encoding_{encoding}")
    sr = max(int(sample_rate or 48000), 1)
    ch = max(int(channels or 1), 1)

    if not raw_bytes:
        return DecodedAudio(
            samples=np.zeros((0, ch), dtype=np.int16),
            sample_rate=sr,
            channels=ch,
            encoding=encoding,
            encoding_name=enc_name,
            duration_sec=0.0,
            raw_bytes=raw_bytes,
        )

    # 1. Uncompressed raw PCM (no RIFF header)
    if encoding == 0 and raw_bytes[:4] != b"RIFF":
        if bits_per_sample == 8:
            u8 = np.frombuffer(raw_bytes, dtype=np.uint8)
            pcm16 = ((u8.astype(np.int16) - 128) << 8)
        elif bits_per_sample == 32:
            i32 = np.frombuffer(raw_bytes[: (len(raw_bytes) // 4) * 4], dtype="<i4")
            pcm16 = (i32 >> 16).astype(np.int16)
        else:
            pcm16 = np.frombuffer(raw_bytes[: (len(raw_bytes) // 2) * 2], dtype="<i2").copy()
        n_frames = len(pcm16) // ch
        pcm = pcm16[: n_frames * ch].reshape(n_frames, ch)
        return DecodedAudio(
            samples=pcm,
            sample_rate=sr,
            channels=ch,
            encoding=encoding,
            encoding_name=enc_name,
            duration_sec=n_frames / float(sr),
            raw_bytes=raw_bytes,
        )

    # 2. Uncompressed raw 32-bit float PCM (no RIFF header)
    if encoding == 1 and raw_bytes[:4] != b"RIFF":
        f32 = np.frombuffer(raw_bytes[: (len(raw_bytes) // 4) * 4], dtype="<f4")
        pcm16 = np.clip(np.round(f32 * 32767.0), -32768, 32767).astype(np.int16)
        n_frames = len(pcm16) // ch
        pcm = pcm16[: n_frames * ch].reshape(n_frames, ch)
        return DecodedAudio(
            samples=pcm,
            sample_rate=sr,
            channels=ch,
            encoding=encoding,
            encoding_name=enc_name,
            duration_sec=n_frames / float(sr),
            raw_bytes=raw_bytes,
        )

    # 3. Container / Compressed audio (Sony ATRAC9 RIFF WAV, MP3, AAC, ADPCM, XWMA) -> PyAV!
    import av

    fact_samples, _enc_delay = _parse_at9_fact_chunk(raw_bytes)
    container = av.open(io.BytesIO(raw_bytes))
    try:
        stream = container.streams.audio[0]
        dec_sr = int(stream.rate or sr)
        dec_ch = int(stream.channels or ch)
        block_align = int(stream.codec_context.block_align or 0)
        resampler = av.AudioResampler(format="s16", layout=stream.layout, rate=dec_sr)
        chunks: list[np.ndarray] = []
        samples_per_block = 1024

        def _append_frames(frames: list[Any]) -> None:
            nonlocal samples_per_block
            for frame in frames:
                if frame.samples > 0:
                    samples_per_block = int(frame.samples)
                for rf in resampler.resample(frame):
                    arr = rf.to_ndarray()
                    # Packed s16 from PyAV has shape (1, frames * channels)
                    if arr.ndim == 2 and arr.shape[0] == 1:
                        chunks.append(arr.reshape(-1, dec_ch))
                    elif arr.ndim == 2 and arr.shape[0] == dec_ch:
                        chunks.append(arr.T)
                    else:
                        chunks.append(arr.reshape(-1, dec_ch))

        for pkt in container.demux(stream):
            if pkt.dts is None and pkt.size == 0:
                continue
            try:
                _append_frames(pkt.decode())
            except av.error.InvalidDataError:
                # Rare FFmpeg `atrac9dec` edge-case on individual v2 superframes inside a
                # multi-superframe WAV packet: split by `block_align` so sibling superframes
                # decode cleanly and preserve exact sample timing with silence for the single bad block.
                raw_pkt = bytes(pkt)
                if block_align > 0 and len(raw_pkt) >= block_align:
                    for off in range(0, len(raw_pkt) - block_align + 1, block_align):
                        sub_pkt = av.Packet(raw_pkt[off : off + block_align])
                        try:
                            _append_frames(stream.codec_context.decode(sub_pkt))
                        except av.error.InvalidDataError:
                            chunks.append(np.zeros((samples_per_block, dec_ch), dtype=np.int16))
                else:
                    chunks.append(np.zeros((samples_per_block, dec_ch), dtype=np.int16))

        for rf in resampler.resample(None):
            arr = rf.to_ndarray()
            if arr.ndim == 2 and arr.shape[0] == 1:
                chunks.append(arr.reshape(-1, dec_ch))
            elif arr.ndim == 2 and arr.shape[0] == dec_ch:
                chunks.append(arr.T)
            else:
                chunks.append(arr.reshape(-1, dec_ch))
    finally:
        container.close()

    if chunks:
        pcm = np.concatenate(chunks, axis=0).astype(np.int16, copy=False)
    else:
        pcm = np.zeros((0, dec_ch), dtype=np.int16)

    # Sample-accurate trimming for ATRAC9 frame padding using `fact` chunk / RTTI `SampleCount`
    target_samples = fact_samples or expected_samples
    if target_samples and 0 < target_samples < len(pcm):
        pcm = pcm[:target_samples]

    return DecodedAudio(
        samples=pcm,
        sample_rate=dec_sr,
        channels=dec_ch,
        encoding=encoding,
        encoding_name=enc_name,
        duration_sec=len(pcm) / float(dec_sr),
        raw_bytes=raw_bytes,
    )


def decode_audio_object(
    game: HFWGame,
    oid: ObjectId | str,
    lang: str = "en",
) -> DecodedAudio:
    """Load and decode either a `WaveResource` or a `LocalizedSimpleSoundResource` by ObjectId."""
    obj_id = ObjectId.parse(oid) if isinstance(oid, str) else oid
    objs = game.read_group_local(obj_id.group_id)
    obj = objs[obj_id.object_index]
    tname = obj["_type"]
    if tname == "WaveResource":
        raw, enc, sr, ch, bps, sc = read_wave_raw_bytes(game, obj)
    elif tname == "LocalizedSimpleSoundResource":
        raw, enc, sr, ch, bps, sc = read_localized_sound_raw_bytes(game, obj, lang=lang)
    else:
        raise ValueError(f"Object {obj_id} is {tname}, expected WaveResource or LocalizedSimpleSoundResource")
    return decode_audio_bytes(raw, enc, sr, ch, bps, expected_samples=sc)


def encode_audio_bytes(decoded: DecodedAudio, fmt: str = "wav", bitrate: int = 320_000) -> bytes:
    """Encode `DecodedAudio` into `.wav`, `.flac`, `.mp3`, `.ogg`, or raw `.at9` bytes."""
    f = fmt.lower().lstrip(".")
    if f == "at9":
        if decoded.raw_bytes:
            return decoded.raw_bytes
        f = "wav"

    if f == "wav":
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(decoded.channels)
            wf.setsampwidth(2)  # 16-bit signed PCM
            wf.setframerate(decoded.sample_rate)
            wf.writeframes(np.ascontiguousarray(decoded.samples, dtype="<i2").tobytes())
        return buf.getvalue()

    import av

    layout = "mono" if decoded.channels == 1 else ("stereo" if decoded.channels == 2 else f"{decoded.channels}c")
    buf = io.BytesIO()
    if f == "flac":
        container = av.open(buf, mode="w", format="flac")
        stream = container.add_stream("flac", rate=decoded.sample_rate, layout=layout)
    elif f == "mp3":
        container = av.open(buf, mode="w", format="mp3")
        stream = container.add_stream("libmp3lame", rate=decoded.sample_rate, layout=layout)
        stream.bit_rate = bitrate
    elif f == "ogg":
        container = av.open(buf, mode="w", format="ogg")
        # HFW audio is 48 kHz which matches Opus natively; resample if another rate ever occurs
        opus_rate = decoded.sample_rate if decoded.sample_rate in (8000, 12000, 16000, 24000, 48000) else 48000
        stream = container.add_stream("libopus", rate=opus_rate, layout=layout)
        stream.bit_rate = min(bitrate, 256_000)
    else:
        raise ValueError(f"Unsupported audio format '{fmt}'. Choose from: {', '.join(SUPPORTED_OUTPUT_FORMATS)}")

    try:
        resampler = None
        if f == "ogg" and stream.rate != decoded.sample_rate:
            resampler = av.AudioResampler(format="s16", layout=layout, rate=stream.rate)
        # Feed in ~1-second chunks so memory stays tiny even on 31-minute credits tracks!
        pcm = np.ascontiguousarray(decoded.samples, dtype=np.int16)
        chunk_frames = max(decoded.sample_rate, 4096)
        pts = 0
        for start in range(0, len(pcm), chunk_frames):
            sub = pcm[start : start + chunk_frames]
            # PyAV packed s16 expects shape (1, frames * channels)
            frame = av.AudioFrame.from_ndarray(sub.reshape(1, -1), format="s16", layout=layout)
            frame.sample_rate = decoded.sample_rate
            if resampler is not None:
                for rf in resampler.resample(frame):
                    rf.pts = pts
                    pts += rf.samples
                    for pkt in stream.encode(rf):
                        container.mux(pkt)
            else:
                frame.pts = pts
                pts += len(sub)
                for pkt in stream.encode(frame):
                    container.mux(pkt)
        if resampler is not None:
            for rf in resampler.resample(None):
                rf.pts = pts
                pts += rf.samples
                for pkt in stream.encode(rf):
                    container.mux(pkt)
        for pkt in stream.encode(None):
            container.mux(pkt)
    finally:
        container.close()
    return buf.getvalue()


def write_audio_file(decoded: DecodedAudio, out_path: Path, fmt: str = "wav", bitrate: int = 320_000) -> int:
    """Encode and write `decoded` to `out_path`, returning bytes written."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = encode_audio_bytes(decoded, fmt=fmt, bitrate=bitrate)
    out_path.write_bytes(data)
    return len(data)


# ---------------------------------------------------------------------------
# Music Hierarchy Reconstruction & Beat-Synced Cue Mixdown
# ---------------------------------------------------------------------------


def _classify_music_folder(project_name: str, cue_name: str) -> tuple[str, str]:
    """Map a Guerrilla music cue name into `(category, dev_folder)` preserving developer hierarchy."""
    cn = cue_name.strip()
    cu = cn.upper()
    safe_cue = sanitize_path_segment(cn, max_len=64)

    if project_name != "MainScore":
        return project_name, f"Music/{project_name}/{safe_cue}"

    # H2-BS-... (Boss Music)
    if cu.startswith("H2-BS-") or cu.startswith("H2-SEQ-BS-"):
        sub = "General"
        if "RBTS" in cu or "PLASMASPINE" in cu or "MAMMOTH" in cu:
            sub = "Robots"
        elif "ZNTH" in cu or "TLDA" in cu or "ERIK" in cu:
            sub = "Zenith"
        elif "REBL" in cu or "RGLA" in cu:
            sub = "Rebels"
        return "Boss", f"Music/MainScore/Boss/{sub}/{safe_cue}"

    # H2-MQxx / MQxx (Main Quests)
    m_mq = re.search(r"(?<![A-Za-z0-9])(MQ\d{1,2}|MQHUB)(?![A-Za-z0-9])", cu)
    if m_mq:
        mq = m_mq.group(1)
        return "MainQuests", f"Music/MainScore/MainQuests/{mq}/{safe_cue}"

    # H2-SQxx / SQxx (Side Quests)
    m_sq = re.search(r"(?<![A-Za-z0-9])(SQ\d{1,2})(?![A-Za-z0-9])", cu)
    if m_sq:
        sq = m_sq.group(1)
        return "SideQuests", f"Music/MainScore/SideQuests/{sq}/{safe_cue}"

    # H2-CRxx (Cauldrons)
    m_cr = re.search(r"(?<![A-Za-z0-9])(CR\d{1,2})(?![A-Za-z0-9])", cu)
    if m_cr:
        cr = m_cr.group(1)
        return "Cauldrons", f"Music/MainScore/Cauldrons/{cr}/{safe_cue}"

    # H2-RFxx (Rebel Facilities / Camps)
    m_rf = re.search(r"(?<![A-Za-z0-9])(RF\d{1,2})(?![A-Za-z0-9])", cu)
    if m_rf:
        rf = m_rf.group(1)
        return "RebelFacilities", f"Music/MainScore/RebelFacilities/{rf}/{safe_cue}"

    # H2-TNxx (Tallnecks)
    m_tn = re.search(r"(?<![A-Za-z0-9])(TN\d{1,2})(?![A-Za-z0-9])", cu)
    if m_tn:
        tn = m_tn.group(1)
        return "Tallnecks", f"Music/MainScore/Tallnecks/{tn}/{safe_cue}"

    # H2-NX-... (Exploration / World Mood Music)
    if cu.startswith("H2-NX-") or "EXPLOR" in cu:
        mood = "Ambient"
        for code, lbl in (
            ("CALM", "Calm"),
            ("SAD", "Sad"),
            ("MYST", "Mystery"),
            ("ANGR", "Combat_Angry"),
            ("TENS", "Tension"),
            ("HOPE", "Hopeful"),
            ("DARK", "Dark"),
            ("TRIB", "Tribal"),
            ("DESR", "Desert"),
            ("SNOW", "Snow"),
            ("JUNG", "Jungle"),
            ("COAS", "Coast"),
        ):
            if code in cu:
                mood = lbl
                break
        return "Exploration", f"Music/MainScore/Exploration/{mood}/{safe_cue}"

    # H2-CA-... (Arena / Challenges)
    if cu.startswith("H2-CA-") or "CHALLENGE" in cu or "ARENA" in cu:
        return "Arena", f"Music/MainScore/Arena_Challenges/{safe_cue}"

    # Conversations / Hub
    if cu.startswith("H2-CN-") or cu.startswith("HUB_CONV") or "CONV" in cu:
        return "Conversations", f"Music/MainScore/Conversations/{safe_cue}"

    # Menus / Title
    if "MAIN-MENU" in cu or "MAIN-OPTIONS" in cu or "NEW_GAME_PLUS" in cu or "MENU" in cu:
        return "Menus", f"Music/MainScore/Menus/{safe_cue}"

    # Burning Shores DLC cues inside MainScore
    if "H2DLC" in cu or "DLC" in cu:
        return "BurningShores", f"Music/BurningShores/{safe_cue}"

    # Locations / Settlements / Story Sequences (H2-LC, H2-SP, H2-QU, H2-ST)
    m_code = re.match(r"^H2-([A-Z]{2})-", cu)
    if m_code:
        tag = m_code.group(1)
        tag_map = {
            "LC": "Locations",
            "SP": "Settlements_And_POIs",
            "QU": "Quests_Misc",
            "ST": "Story_Sequences",
            "SEQ": "Story_Sequences",
        }
        sub = tag_map.get(tag, tag)
        return sub, f"Music/MainScore/{sub}/{safe_cue}"

    return "General", f"Music/MainScore/General/{safe_cue}"


def _extract_envelope_info(env_obj: dict[str, Any] | None) -> tuple[float, float, float]:
    """Extract `(gain, fade_in_sec, fade_out_sec)` from a Decima `EnvelopeResource`."""
    if not isinstance(env_obj, dict):
        return 1.0, 0.0, 0.0
    segs = (env_obj.get("Data") or {}).get("Segments") or []
    if not segs:
        return 1.0, 0.0, 0.0
    levels = [float(s.get("TargetLevel", 1.0)) for s in segs]
    peak = max((lvl for lvl in levels if lvl > 0.0), default=1.0)
    fade_in = float(segs[0].get("Duration", 0.0)) if len(segs) >= 1 else 0.0
    fade_out = 0.0
    if len(segs) >= 2 and float(segs[-1].get("TargetLevel", 0.0)) <= 0.001:
        fade_out = float(segs[-1].get("Duration", 0.0))
    return min(max(peak, 0.05), 2.0), max(0.0, fade_in), max(0.0, fade_out)


def _resolve_local_ref(ref: Any, objects_by_gid: dict[int, list[dict[str, Any]]]) -> dict[str, Any] | None:
    if isinstance(ref, dict) and "_type" in ref:
        return ref
    if isinstance(ref, ObjectRef):
        if isinstance(ref.target, dict):
            return ref.target
        objs = objects_by_gid.get(ref.target_id.group_id)
        if objs is not None and 0 <= ref.target_id.object_index < len(objs):
            return objs[ref.target_id.object_index]
    if isinstance(ref, StreamingRef):
        objs = objects_by_gid.get(ref.target_id.group_id)
        if objs is not None and 0 <= ref.target_id.object_index < len(objs):
            return objs[ref.target_id.object_index]
    return None


def discover_music_cues(game: HFWGame) -> list[MusicCueInfo]:
    """Parse ALL `MusicProject` trees in HFW and return every `MusicCueInfo` with its stems."""
    sg = game.graph
    present_groups = [gid for gid in _MUSIC_GROUPS if gid in sg.groups]

    # Also auto-discover any other group with `MusicProject` (future-proof across HFW versions!)
    if "MusicProject" in sg.type_names_by_hash:
        mp_idx = sg.type_names_by_hash.index("MusicProject")
        tt_idx = sg.type_table_indices
        for gm in sg.group_list:
            if gm.group_id not in present_groups:
                if mp_idx in tt_idx[gm.type_start : gm.type_start + gm.type_count]:
                    present_groups.append(gm.group_id)

    objects_by_gid: dict[int, list[dict[str, Any]]] = {}
    for gid in present_groups:
        objects_by_gid[gid] = game.read_group_local(gid)

    known_project_names = {
        "3:29073": "MainScore",
        "3:31758": "Photomode",
        "3:16316": "TildaMusicPlayer",
        "1:10940": "PlainsongChorus",
        "84481:11": "BurningShores_EndCredits",
        "78260:136": "BurningShores_Interactive",
    }

    cues_out: list[MusicCueInfo] = []
    seen_wave_ids: set[str] = set()
    seen_cue_ids: set[str] = set()

    def _raw_cue_label(cue: dict[str, Any]) -> str:
        tc = _resolve_local_ref(cue.get("TriggerCondition"), objects_by_gid)
        if isinstance(tc, dict):
            gp = _resolve_local_ref(tc.get("GraphProgramResource"), objects_by_gid)
            if isinstance(gp, dict):
                for ep in gp.get("EntryPoints") or []:
                    if ep.get("EntryPoint"):
                        lbl = clean_entrypoint_name(ep["EntryPoint"])
                        if lbl:
                            return lbl
        return ""

    for gid in present_groups:
        objs = objects_by_gid[gid]
        for proj in objs:
            if proj["_type"] != "MusicProject":
                continue
            pid = str(proj["_id"])
            proj_name = known_project_names.get(pid)
            rc = _resolve_local_ref(proj.get("RootCueContainer"), objects_by_gid)
            if not isinstance(rc, dict):
                continue

            tracks = [_resolve_local_ref(r, objects_by_gid) for r in (proj.get("Tracks") or [])]
            track_idx_map = {str(t["_id"]): i for i, t in enumerate(tracks) if isinstance(t, dict)}
            track_gain_map = {
                str(t["_id"]): float(t.get("Gain", 1.0)) for t in tracks if isinstance(t, dict)
            }

            # First pass over this project: collect (branch_tuple, cue_obj, raw_label)
            raw_entries: list[tuple[tuple[int, ...], dict[str, Any], str]] = []

            def _walk(cont: dict[str, Any] | None, branch: tuple[int, ...]) -> None:
                if not isinstance(cont, dict):
                    return
                for cref in cont.get("Cues") or []:
                    cue = _resolve_local_ref(cref, objects_by_gid)
                    if isinstance(cue, dict) and str(cue["_id"]) not in seen_cue_ids:
                        seen_cue_ids.add(str(cue["_id"]))
                        raw_entries.append((branch, cue, _raw_cue_label(cue)))
                for ci, chref in enumerate(cont.get("Children") or []):
                    _walk(_resolve_local_ref(chref, objects_by_gid), branch + (ci,))

            _walk(rc, ())
            if not raw_entries:
                continue

            # Auto-name unknown projects cleanly from their first named cue
            if not proj_name:
                first_lbl = next((lbl for _, _, lbl in raw_entries if lbl), "")
                if "PHOTOMODE" in first_lbl.upper():
                    proj_name = "Photomode"
                elif "TILDA" in first_lbl.upper():
                    proj_name = "TildaMusicPlayer"
                elif "ENDCREDITS" in first_lbl.upper() or "H2DLC" in first_lbl.upper():
                    proj_name = "BurningShores_EndCredits"
                elif "ASBEFOREWEARE" in first_lbl.upper():
                    proj_name = "PlainsongChorus"
                elif len(raw_entries) > 100:
                    proj_name = "MainScore"
                else:
                    proj_name = sanitize_path_segment(first_lbl or f"MusicProject_{gid}_{proj['_id'].object_index}", 48)

            # Build branch -> dominant sibling cue prefix so unnamed BoolFact cues inherit context!
            branch_label_hint: dict[tuple[int, ...], str] = {}
            for branch, _cue, lbl in raw_entries:
                if lbl:
                    for depth in range(len(branch), 0, -1):
                        sub_b = branch[:depth]
                        if sub_b not in branch_label_hint:
                            branch_label_hint[sub_b] = lbl

            # Also track TransitionCue / ExitCue parent relationships for unnamed transitions!
            ref_parent_name: dict[str, str] = {}
            for _branch, cue, lbl in raw_entries:
                if not lbl:
                    continue
                for rkey, suffix in (("TransitionCue", "Transition"), ("ExitCue", "Exit")):
                    tgt = _resolve_local_ref(cue.get(rkey), objects_by_gid)
                    if isinstance(tgt, dict):
                        ref_parent_name.setdefault(str(tgt["_id"]), f"{lbl}_{suffix}")

            unnamed_counter: Counter[str] = Counter()

            for branch, cue, lbl in raw_entries:
                cid = str(cue["_id"])
                final_lbl = lbl
                if not final_lbl and cid in ref_parent_name:
                    final_lbl = ref_parent_name[cid]
                if not final_lbl:
                    hint = ""
                    for depth in range(len(branch), 0, -1):
                        if branch[:depth] in branch_label_hint:
                            hint = branch_label_hint[branch[:depth]]
                            break
                    if hint:
                        # Derive a clean sibling base (first 3 tokens of sibling cue name)
                        parts = re.split(r"[-_ ]+", hint)
                        base = "-".join(parts[: min(len(parts), 3)])
                        unnamed_counter[base] += 1
                        tag = "Transition" if cue.get("IsTransitionCue") else "Cue"
                        final_lbl = f"{base}_{tag}_{unnamed_counter[base]:02d}"
                    else:
                        bstr = "_".join(str(x) for x in branch) or "Root"
                        final_lbl = f"Cue_{bstr}_{cue['_id'].object_index}"

                # Collect MusicClipEvent stems & MusicTempoEvent timeline
                stems: list[MusicStemInfo] = []
                tempo_events: list[tuple[float, float]] = []
                for evref in cue.get("Events") or []:
                    ev = _resolve_local_ref(evref, objects_by_gid)
                    if not isinstance(ev, dict):
                        continue
                    et = ev["_type"]
                    if et == "MusicTempoEvent":
                        st_beats = float((ev.get("StartTime") or {}).get("Value", 0)) / 1048576.0
                        bpm_val = float(ev.get("Tempo", ev.get("BeatsPerMinute", 0.0)) or 0.0)
                        if bpm_val > 10.0:
                            tempo_events.append((st_beats, bpm_val))
                    elif et == "MusicClipEvent":
                        w = _resolve_local_ref(ev.get("Wave"), objects_by_gid)
                        if not isinstance(w, dict) or w.get("_type") != "WaveResource":
                            continue
                        wid = str(w["_id"])
                        seen_wave_ids.add(wid)
                        tr = _resolve_local_ref(ev.get("Track"), objects_by_gid)
                        tr_id = str(tr["_id"]) if isinstance(tr, dict) else ""
                        tidx = track_idx_map.get(tr_id, -1)
                        tr_gain = track_gain_map.get(tr_id, 1.0)
                        env = _resolve_local_ref(ev.get("Envelope"), objects_by_gid)
                        env_gain, fin, fout = _extract_envelope_info(env)

                        sr = int(w.get("SampleRate", 48000)) or 48000
                        ch = int(w.get("ChannelCount", 2)) or 2
                        sc = int(w.get("SampleCount", 0))
                        enc = int(w.get("Encoding", 3))
                        wd = w.get("WaveData")
                        ds = w.get("StreamingDataSource")
                        is_stream = not (wd and len(wd) > 0)
                        data_size = len(wd) if (wd and len(wd) > 0) else int((ds or {}).get("Length", 0))
                        dur_us = int(ev.get("Duration", 0) or 0)
                        dur_sec = (sc / float(sr)) if sc > 0 else (dur_us / 1_000_000.0)
                        st_beats = float((ev.get("StartTime") or {}).get("Value", 0)) / 1048576.0

                        stems.append(
                            MusicStemInfo(
                                wave_id=wid,
                                track_index=tidx,
                                start_beats=st_beats,
                                duration_sec=dur_sec,
                                gain=tr_gain * env_gain,
                                fade_in_sec=fin,
                                fade_out_sec=fout,
                                sample_rate=sr,
                                channels=ch,
                                sample_count=sc,
                                encoding=enc,
                                is_streaming=is_stream,
                                data_size=data_size,
                            )
                        )

                if not stems:
                    continue

                # Sort stems deterministically by (start_beats, track_index, wave_id)
                stems.sort(key=lambda s: (round(s.start_beats, 4), s.track_index, s.wave_id))
                tempo_events.sort(key=lambda x: x[0])

                cat, folder = _classify_music_folder(proj_name, final_lbl)
                cues_out.append(
                    MusicCueInfo(
                        cue_id=cid,
                        project_name=proj_name,
                        cue_name=final_lbl,
                        category=cat,
                        dev_folder=folder,
                        branch_path="/".join(str(x) for x in branch),
                        initial_tempo=float(cue.get("InitialTempo", 120.0) or 120.0),
                        play_as_one_shot=bool(cue.get("PlayAsOneShot", False)),
                        is_transition=bool(cue.get("IsTransitionCue", False)),
                        tempo_events=tempo_events,
                        stems=stems,
                    )
                )

    # Also capture any unassigned WaveResources in Group 3 so 100% of music waves are accessible
    if 3 in objects_by_gid:
        unassigned_stems: list[MusicStemInfo] = []
        for o in objects_by_gid[3]:
            if o["_type"] == "WaveResource" and str(o["_id"]) not in seen_wave_ids:
                sr = int(o.get("SampleRate", 48000)) or 48000
                ch = int(o.get("ChannelCount", 2)) or 2
                sc = int(o.get("SampleCount", 0))
                wd = o.get("WaveData")
                ds = o.get("StreamingDataSource")
                data_size = len(wd) if (wd and len(wd) > 0) else int((ds or {}).get("Length", 0))
                if data_size <= 0:
                    continue
                unassigned_stems.append(
                    MusicStemInfo(
                        wave_id=str(o["_id"]),
                        track_index=-1,
                        start_beats=0.0,
                        duration_sec=sc / float(sr) if sc > 0 else 0.0,
                        gain=1.0,
                        fade_in_sec=0.0,
                        fade_out_sec=0.0,
                        sample_rate=sr,
                        channels=ch,
                        sample_count=sc,
                        encoding=int(o.get("Encoding", 3)),
                        is_streaming=not (wd and len(wd) > 0),
                        data_size=data_size,
                    )
                )
        if unassigned_stems:
            cues_out.append(
                MusicCueInfo(
                    cue_id="3:unassigned",
                    project_name="MainScore",
                    cue_name="Unassigned_Music_Waves",
                    category="Unassigned",
                    dev_folder="Music/MainScore/Unassigned_Music_Waves",
                    branch_path="unassigned",
                    initial_tempo=120.0,
                    play_as_one_shot=False,
                    is_transition=False,
                    stems=unassigned_stems,
                )
            )

    cues_out.sort(key=lambda c: (c.dev_folder.lower(), c.cue_id))
    return cues_out


def _beats_to_seconds(target_beats: float, initial_bpm: float, tempo_events: list[tuple[float, float]]) -> float:
    """Convert a Decima `BeatTime` position to wall-clock seconds using the cue's tempo map."""
    if target_beats <= 0.0:
        return 0.0
    cur_beat = 0.0
    cur_bpm = initial_bpm if initial_bpm > 10.0 else 120.0
    sec = 0.0
    for ev_beat, ev_bpm in tempo_events:
        if ev_beat >= target_beats:
            break
        if ev_beat > cur_beat:
            sec += (ev_beat - cur_beat) * (60.0 / cur_bpm)
            cur_beat = ev_beat
        if ev_bpm > 10.0:
            cur_bpm = ev_bpm
    sec += (target_beats - cur_beat) * (60.0 / cur_bpm)
    return sec


def render_music_cue_mix(game: HFWGame, cue: MusicCueInfo) -> DecodedAudio:
    """Render a full beat-synced stereo mixdown of all stems in `cue`."""
    if not cue.stems:
        raise ValueError(f"MusicCue {cue.cue_id} ({cue.cue_name}) has no audio stems.")

    # Single-stem fast path: return decoded stem directly (zero remixing needed!)
    if len(cue.stems) == 1:
        return decode_audio_object(game, cue.stems[0].wave_id)

    # Decode each unique wave referenced by this cue once
    decoded_by_wid: dict[str, DecodedAudio] = {}
    out_sr = 48000
    for st in cue.stems:
        if st.wave_id not in decoded_by_wid:
            dec = decode_audio_object(game, st.wave_id)
            decoded_by_wid[st.wave_id] = dec
            if dec.sample_rate > 0:
                out_sr = dec.sample_rate

    # Normalize start offsets relative to earliest clip so leading silence isn't wasted
    min_beats = min(st.start_beats for st in cue.stems)
    placements: list[tuple[int, MusicStemInfo, np.ndarray]] = []
    total_frames = 0
    for st in cue.stems:
        dec = decoded_by_wid[st.wave_id]
        if len(dec.samples) == 0:
            continue
        # Convert mono -> stereo cleanly
        pcm = dec.samples.astype(np.float32) / 32768.0
        if pcm.shape[1] == 1:
            pcm = np.repeat(pcm, 2, axis=1)
        elif pcm.shape[1] > 2:
            pcm = pcm[:, :2]

        start_sec = _beats_to_seconds(st.start_beats - min_beats, cue.initial_tempo, cue.tempo_events)
        start_frame = max(0, int(round(start_sec * out_sr)))
        end_frame = start_frame + len(pcm)
        if end_frame > total_frames:
            total_frames = end_frame
        placements.append((start_frame, st, pcm))

    if total_frames <= 0:
        return DecodedAudio(
            samples=np.zeros((0, 2), dtype=np.int16),
            sample_rate=out_sr,
            channels=2,
            encoding=0,
            encoding_name="PCM",
            duration_sec=0.0,
        )

    mix = np.zeros((total_frames, 2), dtype=np.float32)
    for start_frame, st, pcm in placements:
        n = len(pcm)
        gain = st.gain if st.gain > 0.0 else 1.0
        scaled = pcm * gain
        # Apply envelope fade-in / fade-out when short and well-formed
        fi_frames = min(int(st.fade_in_sec * out_sr), n // 2)
        if fi_frames > 16:
            scaled[:fi_frames] *= np.linspace(0.0, 1.0, fi_frames, dtype=np.float32)[:, None]
        fo_frames = min(int(st.fade_out_sec * out_sr), n // 2)
        if fo_frames > 16 and st.fade_out_sec < st.duration_sec * 0.5:
            scaled[-fo_frames:] *= np.linspace(1.0, 0.0, fo_frames, dtype=np.float32)[:, None]
        mix[start_frame : start_frame + n] += scaled

    # Gentle anti-clip normalization only when layered stems exceed 0.98 peak
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0.98:
        mix *= 0.96 / peak

    pcm16 = np.clip(np.round(mix * 32767.0), -32768, 32767).astype(np.int16)
    return DecodedAudio(
        samples=pcm16,
        sample_rate=out_sr,
        channels=2,
        encoding=0,
        encoding_name="PCM",
        duration_sec=len(pcm16) / float(out_sr),
    )


# ---------------------------------------------------------------------------
# Voice Names, Group Developer-Path Classification, SFX & Voiceline Parsing
# ---------------------------------------------------------------------------


def resolve_voice_name(game: HFWGame, voice_oid: ObjectId, voice_names: dict[str, str]) -> str:
    """Resolve a single `VoiceResource` ObjectId -> speaker name, caching the result."""
    key = str(voice_oid)
    cached = voice_names.get(key)
    if cached is not None:
        return cached
    try:
        objs = game.read_group_local(voice_oid.group_id, cache=(voice_oid.group_id in (1, 2, 3)))
        for o in objs:
            if o["_type"] != "VoiceResource":
                continue
            nr = o.get("NameResource")
            name = ""
            if isinstance(nr, ObjectRef) and isinstance(nr.target, dict):
                texts = nr.target.get("_texts") or []
                name = (texts[0] if texts else "").strip()
            voice_names[str(o["_id"])] = name or f"Voice_{voice_oid.group_id}_{o['_id'].object_index}"
    except Exception:
        voice_names[key] = f"Voice_{voice_oid.group_id}_{voice_oid.object_index}"
    return voice_names.get(key, f"Voice_{voice_oid.group_id}_{voice_oid.object_index}")


def load_all_voice_names(game: HFWGame, max_group_objects: int = 3000) -> dict[str, str]:
    """Load `VoiceResource` ObjectId ('group:idx') -> English character/speaker name across HFW.

    Loads Group 1 (`Aloy`, `Sylens`) plus all compact voice groups (`<= max_group_objects`, ~525+
    character voices in ~2-3s). Voices living inside huge world-tile groups resolve lazily on demand
    via `resolve_voice_name()`.
    """
    cached = getattr(game, "_voice_names_cache", None)
    if cached is not None:
        return cached

    sg = game.graph
    if "VoiceResource" not in sg.type_names_by_hash:
        return {}
    v_idx = sg.type_names_by_hash.index("VoiceResource")
    tt_idx = sg.type_table_indices

    voice_groups = [
        gm.group_id
        for gm in sg.group_list
        if (gm.group_id == 1 or gm.num_objects <= max_group_objects)
        and v_idx in tt_idx[gm.type_start : gm.type_start + gm.type_count]
    ]

    voices: dict[str, str] = {}
    for gid in voice_groups:
        objs = game.read_group_local(gid, cache=(gid in (1, 2, 3)))
        for o in objs:
            if o["_type"] != "VoiceResource":
                continue
            nr = o.get("NameResource")
            name = ""
            if isinstance(nr, ObjectRef) and isinstance(nr.target, dict):
                texts = nr.target.get("_texts") or []
                name = (texts[0] if texts else "").strip()
            voices[str(o["_id"])] = name or f"Voice_{gid}_{o['_id'].object_index}"

    game._voice_names_cache = voices  # type: ignore[attr-defined]
    return voices


def _codename_pretty(code: str) -> str:
    cl = code.lower()
    disp = MACHINE_CODENAMES.get(cl)
    if disp:
        if disp.lower() == cl:
            return disp
        return f"{disp} ({code})"
    return code.replace("_", " ").title()


def classify_audio_group_dev_folder(
    game: HFWGame,
    gid: int,
    kind: str,
    entity_group_map: dict[int, tuple[str, str, str]] | None = None,
    max_scan_bytes: int = 2_000_000,
) -> tuple[str, str, str]:
    """Infer authentic Guerrilla developer folder `(category, owner_name, dev_folder)` for a group.

    Reads only the raw span bytes with fast compiled regexes (no full RTTI deserialization) so
    classifying all ~3,800 SFX & Voiceline groups takes only ~1.5 seconds!
    """
    sg = game.graph
    prefix = "SFX" if kind == "sfx" else "Voicelines"

    # 1. Check existing entity catalog mapping (exact machine/character match!)
    if entity_group_map and gid in entity_group_map:
        ecat, code, disp = entity_group_map[gid]
        if ecat == "machine" and code:
            owner = f"{disp} ({code})" if disp.lower() != code.lower() else disp
            return "Robots", owner, f"{prefix}/Robots/{sanitize_path_segment(owner)}"
        if code and ecat != "entity":
            owner = disp or code
            return ecat.title(), owner, f"{prefix}/{sanitize_path_segment(ecat.title())}/{sanitize_path_segment(owner)}"

    # 2. Fast regex scan over group raw span bytes for Guerrilla `work:` / `source:` paths & EntryPoints
    robot_hits: Counter[str] = Counter()
    char_hits: Counter[str] = Counter()
    weapon_hits: Counter[str] = Counter()
    domain_hits: Counter[str] = Counter()
    quest_hits: Counter[str] = Counter()
    ep_names: list[str] = []
    has_ui = False

    def _scan_gid(target_gid: int, max_bytes: int = 262_144) -> None:
        nonlocal has_ui
        read_so_far = 0
        for sp in sg.group_spans(target_gid):
            take = min(sp.length, max_bytes - read_so_far)
            if take <= 0:
                break
            raw = game.storage.read(sg.files[sp.file_index], sp.offset, take)
            read_so_far += take
            for m in _WORK_PATH_RE.finditer(raw):
                p = m.group(1).decode("ascii", errors="ignore")
                pl = p.lower()
                if "interface/" in pl or "menu/" in pl or "/ui/" in pl:
                    has_ui = True
                rm = _ROBOT_DIR_RE.search(p)
                if rm:
                    rc = rm.group(1)
                    if rc.lower() not in _GENERIC_ROBOT_CODES:
                        robot_hits[rc] += 1
                    continue
                cm = _CHAR_DIR_RE.search(p)
                if cm:
                    char_hits[f"{cm.group(1)}/{cm.group(2)}"] += 1
                    continue
                wm = _WEAPON_DIR_RE.search(p)
                if wm:
                    wname = wm.group(2) or wm.group(1)
                    weapon_hits[wname] += 1
                    continue
                dm = _DOMAIN_DIR_RE.search(p)
                if dm:
                    domain_hits[dm.group(1)] += 1
            for m in _EP_SPAN_RE.finditer(raw):
                ep = clean_entrypoint_name("EntryPoint_" + m.group(1).decode("ascii", errors="ignore"))
                if ep and not ep.startswith(("StateMachineNodeResource", "GraphResource", "Particle_Emitter")):
                    ep_names.append(ep)
                    qm = _QUEST_CODE_RE.search(ep)
                    if qm:
                        quest_hits[qm.group(1).upper()] += 1
            if robot_hits or quest_hits or weapon_hits or domain_hits:
                break

    _scan_gid(gid, max_bytes=max_scan_bytes)

    # If small group has no paths/EPs itself (e.g. single-sentence voiceline group 59293),
    # peek at up to 2 small supergroups to recover its Quest/Scene script name!
    if not robot_hits and not quest_hits and not ep_names and not weapon_hits and not domain_hits:
        for sup in sg.supergroups.get(gid, ())[:3]:
            if sup in (1, 2, 3) or sg.group(sup).num_objects > 8000:
                continue
            _scan_gid(sup, max_bytes=131_072)
            if quest_hits or robot_hits or ep_names:
                break

    if robot_hits:
        code = robot_hits.most_common(1)[0][0]
        owner = _codename_pretty(code)
        return "Robots", owner, f"{prefix}/Robots/{sanitize_path_segment(owner)}"

    if weapon_hits:
        wname = weapon_hits.most_common(1)[0][0]
        return "Weapons", wname, f"{prefix}/Weapons/{sanitize_path_segment(wname)}"

    if quest_hits:
        qcode = quest_hits.most_common(1)[0][0]
        # Pick best descriptive EntryPoint containing this quest code
        scene = next((e for e in ep_names if qcode in e.upper()), qcode)
        scene = re.sub(r"^StateMachine_", "", scene)
        scene = re.sub(r"_Script$", "", scene)
        scene_seg = sanitize_path_segment(scene, max_len=56)
        if qcode.startswith("MQ"):
            top = "MainQuests"
        elif qcode.startswith("SQ"):
            top = "SideQuests"
        elif qcode.startswith(("RF", "CR", "TN", "CA")):
            top = "Activities"
        else:
            top = "Quests"
        return top, scene, f"{prefix}/{top}/{qcode}/{scene_seg}"

    if domain_hits:
        dom = domain_hits.most_common(1)[0][0]
        parts = dom.split("/")
        last = parts[-1]
        if last.lower() in MACHINE_CODENAMES:
            owner = _codename_pretty(last)
            return "Robots", owner, f"{prefix}/Robots/{sanitize_path_segment(owner)}"
        return "Characters", dom, f"{prefix}/Characters/{sanitize_path_segment(last)}"

    if char_hits:
        ch_path = char_hits.most_common(1)[0][0]
        sub_cat, sub_name = ch_path.split("/", 1)
        return "Characters", sub_name, f"{prefix}/Characters/{sanitize_path_segment(sub_cat)}/{sanitize_path_segment(sub_name)}"

    if has_ui:
        return "UI", f"UI_Group_{gid}", f"{prefix}/UI/Group_{gid}"

    if ep_names:
        best_ep = ep_names[0]
        best_ep = re.sub(r"^StateMachine_", "", best_ep)
        best_ep = re.sub(r"_Script$", "", best_ep)
        ep_u = best_ep.upper()
        if any(k in ep_u for k in ("WIND", "RAIN", "AMBIENT", "FAUNA", "WATER", "WEATHER", "ENV")):
            cat = "Environment"
        elif any(k in ep_u for k in ("MENU", "UI_", "HUD", "MAP", "INVENTORY")):
            cat = "UI"
        elif any(k in ep_u for k in ("HUB", "CIN_", "SEQ_", "_IC_")):
            cat = "Story_And_Hub"
        else:
            cat = "World_And_General"
        return cat, best_ep, f"{prefix}/{cat}/{sanitize_path_segment(best_ep, max_len=56)}"

    return "General", f"Group_{gid}", f"{prefix}/General/Group_{gid}"


def extract_sfx_group_items(
    game: HFWGame,
    gid: int,
    base_folder: str | None = None,
    owner_name: str = "",
) -> list[AudioItemInfo]:
    """Parse one group and return every SFX `WaveResource` mapped to its `GraphSoundResource` name."""
    if base_folder is None:
        _cat, owner_name, base_folder = classify_audio_group_dev_folder(game, gid, kind="sfx")

    objs = game.read_group_local(gid, cache=(gid in (1, 2, 3)))

    # Map WaveResource ObjectId -> list of GraphSoundResource developer names
    wave_to_graph_name: dict[str, str] = {}
    for o in objs:
        ot = o["_type"]
        if ot == "GraphSoundResource":
            gp_ref = o.get("GraphProgram")
            gp = gp_ref.target if isinstance(gp_ref, ObjectRef) and isinstance(gp_ref.target, dict) else None
            if not isinstance(gp, dict):
                continue
            eps = gp.get("EntryPoints") or []
            raw_ep = eps[0].get("EntryPoint", "") if eps else ""
            gsr_name = clean_entrypoint_name(raw_ep) or f"SoundGraph_{gid}_{o['_id'].object_index}"

            nc_ref = gp.get("ExposedDataResource")
            nc = nc_ref.target if isinstance(nc_ref, ObjectRef) and isinstance(nc_ref.target, dict) else None
            if isinstance(nc, dict):
                params = nc.get("Parameters") or {}
                for list_key in ("DefaultSoftLinkedObjects", "DefaultHardLinkedObjects"):
                    for ref in params.get(list_key) or ():
                        if isinstance(ref, ObjectRef):
                            tgt = ref.target
                            if isinstance(tgt, dict) and tgt.get("_type") == "WaveResource":
                                wave_to_graph_name.setdefault(str(tgt["_id"]), gsr_name)
                            elif ref.target_id.group_id == gid:
                                # Intra-group safety
                                t_obj = objs[ref.target_id.object_index]
                                if t_obj["_type"] == "WaveResource":
                                    wave_to_graph_name.setdefault(str(t_obj["_id"]), gsr_name)
        elif ot == "SimpleSoundResource":
            wref = o.get("Wave")
            if isinstance(wref, ObjectRef) and isinstance(wref.target, dict) and wref.target["_type"] == "WaveResource":
                wave_to_graph_name.setdefault(str(wref.target["_id"]), f"SimpleSound_{gid}_{o['_id'].object_index}")

    # Emit every WaveResource in this group
    items: list[AudioItemInfo] = []
    name_counts: Counter[str] = Counter()
    for o in objs:
        if o["_type"] != "WaveResource":
            continue
        oid_str = str(o["_id"])
        sr = int(o.get("SampleRate", 48000)) or 48000
        ch = int(o.get("ChannelCount", 1)) or 1
        sc = int(o.get("SampleCount", 0))
        enc = int(o.get("Encoding", 3))
        wd = o.get("WaveData")
        ds = o.get("StreamingDataSource")
        is_stream = not (wd and len(wd) > 0)
        data_size = len(wd) if (wd and len(wd) > 0) else int((ds or {}).get("Length", 0))
        if data_size <= 0:
            continue

        gsr_name = wave_to_graph_name.get(oid_str, "")
        if gsr_name:
            safe_g = sanitize_path_segment(gsr_name, max_len=60)
            dev_folder = f"{base_folder}/{safe_g}"
            name_counts[safe_g] += 1
            idx_num = name_counts[safe_g]
            stem = f"{safe_g}_{idx_num:02d}_{gid}_{o['_id'].object_index}"
            disp_name = f"{gsr_name} #{idx_num:02d}"
        else:
            dev_folder = base_folder
            stem = f"Wave_{gid}_{o['_id'].object_index}"
            disp_name = stem

        items.append(
            AudioItemInfo(
                object_id=oid_str,
                kind="sfx",
                name=disp_name,
                dev_folder=dev_folder,
                filename_stem=stem,
                group_id=gid,
                sample_rate=sr,
                channels=ch,
                sample_count=sc,
                duration_sec=(sc / float(sr)) if sc > 0 else 0.0,
                encoding=enc,
                encoding_name=WAVE_ENCODING_NAMES.get(enc, str(enc)),
                data_size=data_size,
                is_streaming=is_stream,
                sound_graph_name=gsr_name,
                owner_name=owner_name,
            )
        )
    return items


def extract_voiceline_group_items(
    game: HFWGame,
    gid: int,
    voice_names: dict[str, str] | None = None,
    base_folder: str | None = None,
    owner_name: str = "",
) -> list[AudioItemInfo]:
    """Parse one group and return every `LocalizedSimpleSoundResource` with speaker & subtitle."""
    if voice_names is None:
        voice_names = load_all_voice_names(game)
    if base_folder is None:
        _cat, owner_name, base_folder = classify_audio_group_dev_folder(game, gid, kind="voiceline")

    objs = game.read_group_local(gid, cache=(gid in (1, 2, 3)))
    files = game.graph.files

    # Link SentenceResource -> (LocalizedSimpleSoundResource, speaker, subtitle)
    loc_meta: dict[str, tuple[str, str]] = {}
    for o in objs:
        if o["_type"] != "SentenceResource":
            continue
        ss_ref = o.get("SimpleSound")
        if not isinstance(ss_ref, ObjectRef):
            continue
        ss_id = str(ss_ref.target_id)

        speaker = ""
        v_ref = o.get("Voice")
        if isinstance(v_ref, ObjectRef):
            speaker = voice_names.get(str(v_ref.target_id), "")
            if not speaker and isinstance(v_ref.target, dict):
                nr = v_ref.target.get("NameResource")
                if isinstance(nr, ObjectRef) and isinstance(nr.target, dict):
                    texts = nr.target.get("_texts") or []
                    speaker = (texts[0] if texts else "").strip()
            if not speaker and v_ref.target_id.group_id >= 0:
                speaker = resolve_voice_name(game, v_ref.target_id, voice_names)

        subtitle = ""
        t_ref = o.get("Text")
        if isinstance(t_ref, ObjectRef) and isinstance(t_ref.target, dict):
            texts = t_ref.target.get("_texts") or []
            subtitle = (texts[0] if texts else "").replace("\r", " ").replace("\n", " ").strip()

        if ss_id not in loc_meta or (subtitle and not loc_meta[ss_id][1]):
            loc_meta[ss_id] = (speaker, subtitle)

    items: list[AudioItemInfo] = []
    for o in objs:
        if o["_type"] != "LocalizedSimpleSoundResource":
            continue
        oid_str = str(o["_id"])
        lds = o.get("LocalizedDataSources") or []
        if not lds:
            continue

        # Detect available language codes and representative sample_count/data_size (prefer English)
        avail_langs: list[str] = []
        rep_samples = 0
        rep_size = 0
        for idx, entry in enumerate(lds):
            ds = entry.get("StreamingDataSource")
            if not isinstance(ds, dict) or ds.get("Length", 0) <= 0:
                continue
            code = SPOKEN_LANGUAGES[idx] if idx < len(SPOKEN_LANGUAGES) else f"lang{idx}"
            fidx = ds.get("_file_index")
            if fidx is not None and 0 <= fidx < len(files):
                m_l = re.search(r"package/([^/]+)/package\.", files[fidx])
                if m_l:
                    code = m_l.group(1)
            avail_langs.append(code)
            if code == "en" or rep_size == 0:
                rep_samples = int(entry.get("SampleCount", 0))
                rep_size = int(ds.get("Length", 0))

        if rep_size <= 0:
            continue

        swp = o.get("SharedWaveProperties") or {}
        sr = int(swp.get("SampleRate", 48000)) or 48000
        ch = int(swp.get("ChannelCount", 1)) or 1
        enc = int(swp.get("Encoding", 3))

        speaker, subtitle = loc_meta.get(oid_str, ("", ""))
        speaker_clean = speaker or "Unknown_Speaker"
        sub_slug = _slugify_subtitle(subtitle, max_len=44)
        spk_seg = sanitize_path_segment(speaker_clean, max_len=40)
        dev_folder = f"{base_folder}/{spk_seg}"

        if sub_slug:
            stem = f"{spk_seg}_{sub_slug}_{gid}_{o['_id'].object_index}"
            disp_name = f"{speaker_clean}: {subtitle[:80]}"
        else:
            stem = f"{spk_seg}_line_{gid}_{o['_id'].object_index}"
            disp_name = f"{speaker_clean} ({oid_str})"

        items.append(
            AudioItemInfo(
                object_id=oid_str,
                kind="voiceline",
                name=disp_name,
                dev_folder=dev_folder,
                filename_stem=stem,
                group_id=gid,
                sample_rate=sr,
                channels=ch,
                sample_count=rep_samples,
                duration_sec=(rep_samples / float(sr)) if rep_samples > 0 else 0.0,
                encoding=enc,
                encoding_name=WAVE_ENCODING_NAMES.get(enc, str(enc)),
                data_size=rep_size,
                is_streaming=True,
                owner_name=owner_name,
                speaker=speaker_clean,
                subtitle=subtitle,
                languages=tuple(avail_langs),
            )
        )
    return items


# ---------------------------------------------------------------------------
# High-Level Batch Extraction (Preserving Developer Directory Structure or Flat)
# ---------------------------------------------------------------------------


def _unique_path(base_path: Path) -> Path:
    if not base_path.exists():
        return base_path
    stem = base_path.stem
    suf = base_path.suffix
    parent = base_path.parent
    for i in range(2, 10000):
        cand = parent / f"{stem}_{i}{suf}"
        if not cand.exists():
            return cand
    return parent / f"{stem}_{int(time.time()*1000)}{suf}"


def extract_music_cues_to_dir(
    game: HFWGame,
    cues: list[MusicCueInfo],
    out_dir: Path,
    *,
    fmt: str = "wav",
    preserve_structure: bool = True,
    mix_cues: bool = True,
    export_stems: bool = True,
    include_ost: bool = False,
    limit: int | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Extract selected `MusicCueInfo` cues (full beat-synced songs and/or individual stems)."""
    t0 = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = "." + fmt.lower().lstrip(".")

    selected = cues[:limit] if limit else cues
    exported: list[dict[str, Any]] = []
    errors: list[str] = []
    total_bytes = 0
    total_steps = len(selected)

    for idx, cue in enumerate(selected, 1):
        if progress_cb:
            progress_cb(idx, total_steps, f"Music: {cue.cue_name}")
        target_dir = (out_dir / sanitize_rel_folder(cue.dev_folder)) if preserve_structure else out_dir
        safe_cue = sanitize_path_segment(cue.cue_name, max_len=64)

        # 1. Full beat-synced song mixdown (or direct single-stem track)
        if mix_cues and cue.cue_id != "3:unassigned":
            try:
                mixed = render_music_cue_mix(game, cue)
                if len(mixed.samples) > 0:
                    suffix = "_FULL_MIX" if (len(cue.stems) > 1 and export_stems) else ""
                    fname = f"{safe_cue}{suffix}{ext}" if preserve_structure else f"{cue.project_name}_{safe_cue}{suffix}_{cue.cue_id.replace(':', '_')}{ext}"
                    out_path = _unique_path(target_dir / fname)
                    nbytes = write_audio_file(mixed, out_path, fmt=fmt)
                    total_bytes += nbytes
                    exported.append(
                        {
                            "path": str(out_path),
                            "rel_path": str(out_path.relative_to(out_dir)).replace("\\", "/"),
                            "kind": "music_mix" if len(cue.stems) > 1 else "music",
                            "cue_id": cue.cue_id,
                            "cue_name": cue.cue_name,
                            "category": cue.category,
                            "duration_sec": round(mixed.duration_sec, 3),
                            "sample_rate": mixed.sample_rate,
                            "channels": mixed.channels,
                            "bytes": nbytes,
                        }
                    )
            except Exception as exc:
                errors.append(f"Mix {cue.cue_id} ({cue.cue_name}): {exc}")

        # 2. Individual stems (always when `export_stems=True` AND either multi-stem or `not mix_cues`)
        if export_stems and (len(cue.stems) > 1 or not mix_cues or cue.cue_id == "3:unassigned"):
            stem_dir = (target_dir / "Stems") if (preserve_structure and mix_cues and len(cue.stems) > 1) else target_dir
            for s_i, st in enumerate(cue.stems, 1):
                try:
                    dec = decode_audio_object(game, st.wave_id)
                    if len(dec.samples) == 0:
                        continue
                    wid_tag = st.wave_id.replace(":", "_")
                    tr_tag = f"track{st.track_index:02d}_" if st.track_index >= 0 else ""
                    if preserve_structure:
                        fname = f"{safe_cue}_stem{s_i:02d}_{tr_tag}{wid_tag}{ext}"
                    else:
                        fname = f"{cue.project_name}_{safe_cue}_stem{s_i:02d}_{tr_tag}{wid_tag}{ext}"
                    out_path = _unique_path(stem_dir / fname)
                    nbytes = write_audio_file(dec, out_path, fmt=fmt)
                    total_bytes += nbytes
                    exported.append(
                        {
                            "path": str(out_path),
                            "rel_path": str(out_path.relative_to(out_dir)).replace("\\", "/"),
                            "kind": "music_stem",
                            "cue_id": cue.cue_id,
                            "wave_id": st.wave_id,
                            "cue_name": cue.cue_name,
                            "track_index": st.track_index,
                            "start_beats": round(st.start_beats, 3),
                            "duration_sec": round(dec.duration_sec, 3),
                            "sample_rate": dec.sample_rate,
                            "channels": dec.channels,
                            "bytes": nbytes,
                        }
                    )
                except Exception as exc:
                    errors.append(f"Stem {st.wave_id} ({cue.cue_name}): {exc}")

    # Optional: copy DigitalExtras/OST bonus tracks cleanly
    if include_ost:
        ost_files = detect_digital_extras_ost(game)
        ost_dest = (out_dir / "Music" / "OfficialSoundtrack_OST") if preserve_structure else out_dir
        for src in ost_files:
            ost_dest.mkdir(parents=True, exist_ok=True)
            dst = _unique_path(ost_dest / src.name)
            shutil.copy2(src, dst)
            nbytes = dst.stat().st_size
            total_bytes += nbytes
            exported.append(
                {
                    "path": str(dst),
                    "rel_path": str(dst.relative_to(out_dir)).replace("\\", "/"),
                    "kind": "ost",
                    "cue_name": src.stem,
                    "bytes": nbytes,
                }
            )

    manifest_path = out_dir / "audio_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "category": "music",
                "format": fmt,
                "preserve_structure": preserve_structure,
                "mix_cues": mix_cues,
                "export_stems": export_stems,
                "count": len(exported),
                "total_bytes": total_bytes,
                "items": exported,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "out_dir": str(out_dir),
        "manifest_path": str(manifest_path),
        "exported_count": len(exported),
        "total_bytes": total_bytes,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "errors": errors,
        "items": exported,
    }


def extract_audio_items_to_dir(
    game: HFWGame,
    items: list[AudioItemInfo],
    out_dir: Path,
    *,
    fmt: str = "wav",
    preserve_structure: bool = True,
    languages: list[str] | None = None,
    group_by: str = "quest",
    write_transcript: bool = True,
    limit: int | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Extract a list of SFX and/or Voiceline `AudioItemInfo` clips cleanly to disk."""
    t0 = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = "." + fmt.lower().lstrip(".")

    installed_langs = detect_installed_languages(game)
    req_langs = languages or (["en"] if "en" in installed_langs else installed_langs[:1] or ["en"])
    if "all" in [l.lower() for l in req_langs]:
        req_langs = installed_langs or ["en"]

    selected = items[:limit] if limit else items
    exported: list[dict[str, Any]] = []
    transcript_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    total_bytes = 0
    total_steps = len(selected)

    for idx, item in enumerate(selected, 1):
        if progress_cb:
            progress_cb(idx, total_steps, f"{item.kind.upper()}: {item.name[:60]}")

        if item.kind == "voiceline":
            langs_to_run = [l for l in req_langs if (not item.languages or l in item.languages)]
            if not langs_to_run:
                langs_to_run = req_langs[:1]
            for lang in langs_to_run:
                try:
                    dec = decode_audio_object(game, item.object_id, lang=lang)
                    if len(dec.samples) == 0:
                        continue
                    if preserve_structure:
                        # Either Voicelines/<lang>/<Quest>/<Scene>/<Speaker> OR Voicelines/<lang>/<Speaker>/<Quest>/<Scene>
                        rel_parts = [p for p in item.dev_folder.split("/") if p]
                        # Strip leading "Voicelines" so we insert <lang> right underneath it
                        if rel_parts and rel_parts[0].lower() == "voicelines":
                            rel_parts = rel_parts[1:]
                        if group_by == "speaker" and len(rel_parts) >= 2:
                            spk = rel_parts[-1]
                            rest = rel_parts[:-1]
                            folder_str = "/".join(["Voicelines", lang, spk, *rest])
                        else:
                            folder_str = "/".join(["Voicelines", lang, *rel_parts])
                        target_dir = out_dir / sanitize_rel_folder(folder_str)
                    else:
                        target_dir = out_dir

                    fname = f"{sanitize_path_segment(item.filename_stem, 76)}{ext}"
                    if not preserve_structure and len(langs_to_run) > 1:
                        fname = f"{lang}_{fname}"
                    out_path = _unique_path(target_dir / fname)
                    nbytes = write_audio_file(dec, out_path, fmt=fmt)
                    total_bytes += nbytes
                    rel_str = str(out_path.relative_to(out_dir)).replace("\\", "/")
                    row = {
                        "path": str(out_path),
                        "rel_path": rel_str,
                        "object_id": item.object_id,
                        "kind": "voiceline",
                        "language": lang,
                        "speaker": item.speaker,
                        "subtitle": item.subtitle,
                        "owner": item.owner_name,
                        "duration_sec": round(dec.duration_sec, 3),
                        "sample_rate": dec.sample_rate,
                        "channels": dec.channels,
                        "bytes": nbytes,
                    }
                    exported.append(row)
                    transcript_rows.append(row)
                except Exception as exc:
                    errors.append(f"Voiceline {item.object_id} ({lang}): {exc}")
        else:
            # SFX or music stem
            try:
                dec = decode_audio_object(game, item.object_id)
                if len(dec.samples) == 0:
                    continue
                if preserve_structure:
                    target_dir = out_dir / sanitize_rel_folder(item.dev_folder)
                    fname = f"{sanitize_path_segment(item.filename_stem, 76)}{ext}"
                else:
                    target_dir = out_dir
                    owner_pref = f"{sanitize_path_segment(item.owner_name, 32)}_" if item.owner_name else ""
                    fname = f"{owner_pref}{sanitize_path_segment(item.filename_stem, 76)}{ext}"
                out_path = _unique_path(target_dir / fname)
                nbytes = write_audio_file(dec, out_path, fmt=fmt)
                total_bytes += nbytes
                exported.append(
                    {
                        "path": str(out_path),
                        "rel_path": str(out_path.relative_to(out_dir)).replace("\\", "/"),
                        "object_id": item.object_id,
                        "kind": item.kind,
                        "name": item.name,
                        "sound_graph": item.sound_graph_name,
                        "owner": item.owner_name,
                        "duration_sec": round(dec.duration_sec, 3),
                        "sample_rate": dec.sample_rate,
                        "channels": dec.channels,
                        "bytes": nbytes,
                    }
                )
            except Exception as exc:
                errors.append(f"{item.kind} {item.object_id}: {exc}")

    transcript_path: Path | None = None
    if write_transcript and transcript_rows:
        transcript_path = out_dir / "voicelines_transcript.csv"
        with open(transcript_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["rel_path", "object_id", "language", "speaker", "subtitle", "owner", "duration_sec"],
            )
            w.writeheader()
            for r in transcript_rows:
                w.writerow({k: r.get(k, "") for k in w.fieldnames})

    manifest_path = out_dir / "audio_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format": fmt,
                "preserve_structure": preserve_structure,
                "count": len(exported),
                "total_bytes": total_bytes,
                "items": exported,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "out_dir": str(out_dir),
        "manifest_path": str(manifest_path),
        "transcript_path": str(transcript_path) if transcript_path else None,
        "exported_count": len(exported),
        "total_bytes": total_bytes,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "errors": errors,
        "items": exported,
    }
