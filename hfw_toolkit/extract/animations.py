"""Sony Edge Anim (`EA08`) decoder, animation discovery, rich naming/categorization & skeleton retargeting.

Decodes Decima `SkeletonAnimationResource` (`AnimationData` binary `EA08` blocks) directly into
per-frame local joint `(translation, rotation_xyzw, scale)` curves with exact FrameSet BaseFrame,
contiguous 1-bit keyframe presence bitmask, MSB-first `(c2, c1, c0)` bit-packed truncated IEEE-754 /
two's-complement fixed-point channels, smallest-three quaternion reconstruction, and Slerp/Lerp
interpolation across unkeyed intra-frames.
"""

from __future__ import annotations

import json
import math
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from hfw_toolkit.decima.reader import ObjectId, ObjectRef, StreamingRef
from hfw_toolkit.decima.streaming_graph import HFWGame
from hfw_toolkit.extract.skeleton import SkeletonData, parse_edgeanim_base_pose

EDGE_ANIM_TAG = 0x45413038  # Little-endian bytes b"80AE" ('EA08')

# Known HFW humanoid skeleton IDs (82-joint base locomotion rigs)
FEMALE_HUMAN_SKELETON_ID = "186:0"
MALE_HUMAN_SKELETON_ID = "122:0"

# Humanoid animation groups ordered by gameplay utility (movement/crouch/climb/gestures first)
DEFAULT_HUMANOID_ANIM_GROUPS: tuple[int, ...] = (528, 108, 34, 13410, 72061, 1630, 71882)

# Clean user-facing animation categories (consistent across CLI, Web Studio, and Blender N-Panel)
ANIMATION_CATEGORIES: tuple[tuple[str, str], ...] = (
    ("combat", "Combat & Attacks"),
    ("movement", "Movement & Locomotion"),
    ("idle", "Idle & Stances"),
    ("jump", "Jumps, Leaps & Vaults"),
    ("emote", "Emotes, Roars & Gestures"),
    ("mounted", "Mount / Rider & Special"),
)
VALID_ANIMATION_CATEGORY_KEYS: frozenset[str] = frozenset(k for k, _ in ANIMATION_CATEGORIES)

# Category aliases users may pass via CLI (`--animations category:attacks` -> `combat`, etc.)
_CATEGORY_ALIASES: dict[str, str] = {
    "combat": "combat",
    "attack": "combat",
    "attacks": "combat",
    "ranged": "combat",
    "melee": "combat",
    "hitreact": "combat",
    "movement": "movement",
    "locomotion": "movement",
    "move": "movement",
    "walk": "movement",
    "run": "movement",
    "idle": "idle",
    "stance": "idle",
    "stances": "idle",
    "jump": "jump",
    "jumps": "jump",
    "leap": "jump",
    "vault": "jump",
    "emote": "emote",
    "emotes": "emote",
    "gesture": "emote",
    "gestures": "emote",
    "roar": "emote",
    "mounted": "mounted",
    "mount": "mounted",
    "rider": "mounted",
    "action": "mounted",
    "other": "mounted",
    "special": "mounted",
}


def normalize_animation_category(cat: str) -> str:
    """Normalize a user-supplied category alias to one of the 6 canonical category keys."""
    c = (cat or "").strip().lower()
    return _CATEGORY_ALIASES.get(c, c)


def _ref_target_id(ref: Any) -> ObjectId | None:
    if isinstance(ref, (ObjectRef, StreamingRef)):
        return ref.target_id
    return None


def _read_obj_fast(game: HFWGame, oid: ObjectId) -> dict[str, Any]:
    """Read a single object cleanly preserving external `target_id` references without loading all subgroups."""
    objs = game.read_group_local(oid.group_id, cache=True)
    return objs[oid.object_index]


@dataclass(slots=True)
class AnimationClipInfo:
    """Rich metadata descriptor for a discovered HFW `SkeletonAnimationResource`."""

    anim_id: str
    name: str
    category: str
    skeleton_id: str
    duration: float
    sample_rate: float
    num_frames: int
    num_joints: int
    locomotion_delta_t: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw_delta_deg: float = 0.0
    events: tuple[str, ...] = ()
    description: str = ""
    group_id: int = 0


@dataclass(slots=True)
class DecodedAnimation:
    """Fully decoded per-frame local joint transforms for one HFW animation clip."""

    anim_id: str
    name: str
    category: str
    skeleton_id: str
    duration: float
    sample_rate: float
    num_frames: int
    joint_names: list[str]
    rotations_xyzw: np.ndarray  # (F, J, 4) float32 unit quaternions (x, y, z, w)
    translations: np.ndarray  # (F, J, 3) float32 local translations (m)
    scales: np.ndarray  # (F, J, 3) float32 local scales
    has_rot_track: np.ndarray  # (J,) bool
    has_trans_track: np.ndarray  # (J,) bool
    has_scale_track: np.ndarray  # (J,) bool
    locomotion_delta_t: tuple[float, float, float] = (0.0, 0.0, 0.0)
    events: list[tuple[float, str]] = field(default_factory=list)
    description: str = ""


class _MSBBitReader:
    """Fast MSB-first bitstream reader backed by a single Python integer."""

    __slots__ = ("_val", "_total_bits", "bitpos")

    def __init__(self, data: bytes, start_byte: int = 0, byte_len: int | None = None) -> None:
        end = len(data) if byte_len is None else min(len(data), start_byte + byte_len)
        chunk = data[start_byte:end]
        self._total_bits = len(chunk) * 8
        self._val = int.from_bytes(chunk, "big") if chunk else 0
        self.bitpos = 0

    def read(self, nbits: int) -> int:
        if nbits <= 0:
            return 0
        shift = self._total_bits - self.bitpos - nbits
        self.bitpos += nbits
        if shift >= 0:
            return (self._val >> shift) & ((1 << nbits) - 1)
        if shift + nbits <= 0:
            return 0
        return (self._val << (-shift)) & ((1 << nbits) - 1)


def _decode_component(raw: int, mb: int, eb: int, sg: int) -> float:
    """Decode one bit-packed EdgeAnim component (truncated IEEE-754 float or two's-complement fixed-point)."""
    w = mb + eb + sg
    if w == 0:
        return 0.0
    if eb == 0:
        if sg and (raw & (1 << mb)):
            raw -= 1 << (mb + 1)
        return raw / float(1 << mb)
    sign = -1.0 if (sg and ((raw >> (eb + mb)) & 1)) else 1.0
    rem = raw & ((1 << (eb + mb)) - 1)
    exp = rem >> mb
    mant = rem & ((1 << mb) - 1)
    if exp == 0 and mant == 0:
        return 0.0
    bias = (1 << (eb - 1)) - 1
    return sign * (1.0 + mant / float(1 << mb)) * math.ldexp(1.0, exp - bias)


def _unpack_spec(spec: int) -> tuple[int, tuple[tuple[int, int, int, int], ...]]:
    """Pre-decode a 32-bit EdgeAnim PackedSpec into `(recomp_slot, ((mb, eb, sg, w) for c2, c1, c0))`."""
    recomp = spec & 3
    comps = []
    # CRITICAL: EdgeAnim packs components in MSB-to-LSB spec order: c2 (bits 22..31), c1 (12..21), c0 (2..11)
    for ci in (2, 1, 0):
        c = (spec >> (2 + ci * 10)) & 0x3FF
        mb = c & 31
        eb = (c >> 5) & 15
        sg = (c >> 9) & 1
        comps.append((mb, eb, sg, mb + eb + sg))
    return recomp, tuple(comps)


def _read_quat(br: _MSBBitReader, unpacked_spec: tuple[int, tuple[tuple[int, int, int, int], ...]]) -> tuple[float, float, float, float]:
    """Read a smallest-three bit-packed unit quaternion `(x, y, z, w)` from `br`."""
    recomp, comps = unpacked_spec
    v0 = _decode_component(br.read(comps[0][3]), comps[0][0], comps[0][1], comps[0][2])
    v1 = _decode_component(br.read(comps[1][3]), comps[1][0], comps[1][1], comps[1][2])
    v2 = _decode_component(br.read(comps[2][3]), comps[2][0], comps[2][1], comps[2][2])
    s2 = v0 * v0 + v1 * v1 + v2 * v2
    rem = math.sqrt(max(0.0, 1.0 - s2))
    if recomp == 0:
        q = (rem, v0, v1, v2)
    elif recomp == 1:
        q = (v0, rem, v1, v2)
    elif recomp == 2:
        q = (v0, v1, rem, v2)
    else:
        q = (v0, v1, v2, rem)
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if n > 1e-8:
        inv = 1.0 / n
        return (q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv)
    return (0.0, 0.0, 0.0, 1.0)


def _read_vec3(br: _MSBBitReader, unpacked_spec: tuple[int, tuple[tuple[int, int, int, int], ...]]) -> tuple[float, float, float]:
    """Read a bit-packed 3D translation or scale vector `(x, y, z)` from `br`."""
    _, comps = unpacked_spec
    vx = _decode_component(br.read(comps[0][3]), comps[0][0], comps[0][1], comps[0][2])
    vy = _decode_component(br.read(comps[1][3]), comps[1][0], comps[1][1], comps[1][2])
    vz = _decode_component(br.read(comps[2][3]), comps[2][0], comps[2][1], comps[2][2])
    return (vx, vy, vz)


def _slerp_quat(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Shortest-arc spherical linear interpolation between unit XYZW quaternions."""
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        n = float(np.linalg.norm(out))
        return (out / n) if n > 1e-8 else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    dot = min(1.0, max(-1.0, dot))
    theta = math.acos(dot)
    sin_t = math.sin(theta)
    w0 = math.sin((1.0 - t) * theta) / sin_t
    w1 = math.sin(t * theta) / sin_t
    out = w0 * q0 + w1 * q1
    n = float(np.linalg.norm(out))
    return (out / n) if n > 1e-8 else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _mat4_to_trs_xyzw(m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose a 4x4 column-vector matrix into `(translation, quat_xyzw, scale)`."""
    t = m[:3, 3].astype(np.float32)
    c0, c1, c2 = m[:3, 0], m[:3, 1], m[:3, 2]
    sx = float(np.linalg.norm(c0))
    sy = float(np.linalg.norm(c1))
    sz = float(np.linalg.norm(c2))
    r = np.eye(3, dtype=np.float64)
    if sx > 1e-8:
        r[:, 0] = c0 / sx
    if sy > 1e-8:
        r[:, 1] = c1 / sy
    if sz > 1e-8:
        r[:, 2] = c2 / sz
    if np.linalg.det(r) < 0.0:
        sx = -sx
        r[:, 0] = -r[:, 0]

    tr = r[0, 0] + r[1, 1] + r[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(max(1e-12, 1.0 + r[0, 0] - r[1, 1] - r[2, 2])) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(max(1e-12, 1.0 + r[1, 1] - r[0, 0] - r[2, 2])) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(max(1e-12, 1.0 + r[2, 2] - r[0, 0] - r[1, 1])) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float32)
    qn = float(np.linalg.norm(q))
    if qn > 1e-8:
        q /= qn
    else:
        q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return t, q, np.array([sx, sy, sz], dtype=np.float32)


def _read_joint_index_lists(
    data: bytes,
    start_off: int,
    end_off: int,
    counts: tuple[int, ...],
    num_joints: int,
) -> list[list[int]]:
    """Read the 6 EdgeAnim joint-index arrays (`constR/T/S`, `animR/T/S`) skipping sentinel padding cleanly."""
    pos = start_off
    lists: list[list[int]] = []
    for cnt in counts:
        if cnt == 0:
            lists.append([])
            continue
        if pos + cnt * 2 > end_off:
            raise ValueError(f"EA08 joint index table truncated at {pos:#x} (need {cnt} u16s, end={end_off:#x})")
        arr = list(struct.unpack_from(f"<{cnt}H", data, pos))
        pos += cnt * 2
        while pos + 2 <= end_off:
            nxt = struct.unpack_from("<H", data, pos)[0]
            if nxt >= num_joints:
                pos += 2
            else:
                break
        lists.append(arr)
    return lists


def parse_ea08_header(data: bytes) -> dict[str, Any] | None:
    """Fast header-only inspection of a `SkeletonAnimationResource.AnimationData` buffer."""
    if not data or len(data) < 0x60:
        return None
    tag, duration, sample_freq = struct.unpack_from("<Iff", data, 0x00)
    if tag != EDGE_ANIM_TAG:
        return None
    (
        size_header,
        num_joints,
        num_frames,
        num_frame_sets,
        _eval_buf,
        cR,
        cT,
        cS,
        cU,
        aR,
        aT,
        aS,
        aU,
    ) = struct.unpack_from("<13H", data, 0x0C)
    if num_joints == 0 or num_frames == 0 or num_frame_sets == 0:
        return None
    return {
        "duration": float(duration),
        "sample_rate": float(sample_freq) if sample_freq > 1.0 else 30.0,
        "size_header": int(size_header),
        "num_joints": int(num_joints),
        "num_frames": int(num_frames),
        "num_frame_sets": int(num_frame_sets),
        "cR": int(cR),
        "cT": int(cT),
        "cS": int(cS),
        "cU": int(cU),
        "aR": int(aR),
        "aT": int(aT),
        "aS": int(aS),
        "aU": int(aU),
    }


def decode_edge_anim_ea08(
    data: bytes,
    *,
    anim_id: str = "",
    name: str = "Animation",
    category: str = "movement",
    skeleton_id: str = "",
    joint_names: list[str] | None = None,
    bind_local_mats: list[np.ndarray] | None = None,
    locomotion_delta_t: tuple[float, float, float] = (0.0, 0.0, 0.0),
    events: list[tuple[float, str]] | None = None,
    description: str = "",
) -> DecodedAnimation:
    """Decode a Sony Edge Anim (`EA08`) binary buffer into per-frame local TRS arrays."""
    hdr = parse_ea08_header(data)
    if hdr is None:
        raise ValueError(f"Invalid or empty EA08 animation buffer ({len(data)} bytes)")

    num_joints = hdr["num_joints"]
    num_frames = hdr["num_frames"]
    num_fs = hdr["num_frame_sets"]
    cR, cT, cS = hdr["cR"], hdr["cT"], hdr["cS"]
    aR, aT, aS, aU = hdr["aR"], hdr["aT"], hdr["aS"], hdr["aU"]

    rel_dma, rel_bf, rel_cR, rel_cT, rel_cS, rel_cU, rel_specs = struct.unpack_from("<7I", data, 0x38)
    abs_dma = 0x38 + rel_dma
    abs_bf = 0x3C + rel_bf
    abs_cR = 0x40 + rel_cR
    abs_cT = 0x44 + rel_cT
    abs_cS = 0x48 + rel_cS
    abs_cU = 0x4C + rel_cU
    abs_specs = 0x50 + rel_specs

    first_data_off = min(
        off for off in (abs_cR, abs_cT, abs_cS, abs_cU, abs_specs, abs_dma, abs_bf, hdr["size_header"]) if off >= 0x60
    )
    cR_j, cT_j, cS_j, aR_j, aT_j, aS_j = _read_joint_index_lists(
        data, 0x60, first_data_off, (cR, cT, cS, aR, aT, aS), num_joints
    )

    next_after_specs = min(
        [off for off in (abs_dma, abs_bf, hdr["size_header"], len(data)) if off > abs_specs],
        default=hdr["size_header"],
    )
    n_avail_specs = (next_after_specs - abs_specs) // 4
    raw_specs = struct.unpack_from(f"<{n_avail_specs}I", data, abs_specs)

    if n_avail_specs >= (3 + aR + aT + aS):
        idx_s = 0
        spec_cR = _unpack_spec(raw_specs[idx_s]); idx_s += 1
        specs_aR = [_unpack_spec(raw_specs[idx_s + k]) for k in range(aR)]; idx_s += aR
        spec_cT = _unpack_spec(raw_specs[idx_s]); idx_s += 1
        specs_aT = [_unpack_spec(raw_specs[idx_s + k]) for k in range(aT)]; idx_s += aT
        spec_cS = _unpack_spec(raw_specs[idx_s]); idx_s += 1
        specs_aS = [_unpack_spec(raw_specs[idx_s + k]) for k in range(aS)]
    else:
        idx_s = 0
        spec_cR = _unpack_spec(raw_specs[idx_s]) if cR > 0 else _unpack_spec(0)
        if cR > 0:
            idx_s += 1
        specs_aR = [_unpack_spec(raw_specs[idx_s + k]) for k in range(aR)]; idx_s += aR
        spec_cT = _unpack_spec(raw_specs[idx_s]) if cT > 0 else _unpack_spec(0)
        if cT > 0:
            idx_s += 1
        specs_aT = [_unpack_spec(raw_specs[idx_s + k]) for k in range(aT)]; idx_s += aT
        spec_cS = _unpack_spec(raw_specs[idx_s]) if cS > 0 else _unpack_spec(0)
        if cS > 0:
            idx_s += 1
        specs_aS = [_unpack_spec(raw_specs[idx_s + k]) for k in range(aS)]

    rot = np.zeros((num_frames, num_joints, 4), dtype=np.float32)
    rot[:, :, 3] = 1.0
    trans = np.zeros((num_frames, num_joints, 3), dtype=np.float32)
    scales = np.ones((num_frames, num_joints, 3), dtype=np.float32)

    if bind_local_mats:
        for j_i in range(min(num_joints, len(bind_local_mats))):
            bt, bq, bs = _mat4_to_trs_xyzw(bind_local_mats[j_i])
            rot[:, j_i, :] = bq
            trans[:, j_i, :] = bt
            scales[:, j_i, :] = bs

    has_rot_track = np.zeros(num_joints, dtype=bool)
    has_trans_track = np.zeros(num_joints, dtype=bool)
    has_scale_track = np.zeros(num_joints, dtype=bool)

    if cR > 0:
        br = _MSBBitReader(data, abs_cR, max(0, abs_cT - abs_cR))
        for j_i in cR_j:
            q = _read_quat(br, spec_cR)
            if 0 <= j_i < num_joints:
                rot[:, j_i, :] = q
                has_rot_track[j_i] = True

    if cT > 0:
        br = _MSBBitReader(data, abs_cT, max(0, abs_cS - abs_cT))
        for j_i in cT_j:
            v = _read_vec3(br, spec_cT)
            if 0 <= j_i < num_joints:
                trans[:, j_i, :] = v
                has_trans_track[j_i] = True

    if cS > 0:
        cS_end = abs_cU if abs_cU > abs_cS else abs_specs
        br = _MSBBitReader(data, abs_cS, max(0, cS_end - abs_cS))
        for j_i in cS_j:
            v = _read_vec3(br, spec_cS)
            if 0 <= j_i < num_joints:
                scales[:, j_i, :] = v
                has_scale_track[j_i] = True

    for j_i in aR_j:
        if 0 <= j_i < num_joints:
            has_rot_track[j_i] = True
    for j_i in aT_j:
        if 0 <= j_i < num_joints:
            has_trans_track[j_i] = True
    for j_i in aS_j:
        if 0 <= j_i < num_joints:
            has_scale_track[j_i] = True

    dma_entries = [struct.unpack_from("<2I", data, abs_dma + i * 8) for i in range(num_fs)]
    bf_entries = [struct.unpack_from("<2H", data, abs_bf + i * 4) for i in range(num_fs)]

    keyed_r = np.zeros((aR, num_frames), dtype=bool)
    keyed_t = np.zeros((aT, num_frames), dtype=bool)
    keyed_s = np.zeros((aS, num_frames), dtype=bool)

    for fs_idx in range(num_fs):
        _fs_size, fs_off = dma_entries[fs_idx]
        base_frame, n_intra = bf_entries[fs_idx]
        if fs_off + 32 > len(data) or base_frame >= num_frames:
            continue

        baseR_B, baseT_B, baseS_B, baseU_B, intraR_B, intraT_B, intraS_B, _pad_B = struct.unpack_from(
            "<8I", data, fs_off
        )
        cur = fs_off + 32

        if aR > 0 and baseR_B > 0:
            br = _MSBBitReader(data, cur, baseR_B)
            for k, j_i in enumerate(aR_j):
                q = _read_quat(br, specs_aR[k])
                if 0 <= j_i < num_joints:
                    rot[base_frame, j_i, :] = q
                keyed_r[k, base_frame] = True
        cur += baseR_B

        if aT > 0 and baseT_B > 0:
            br = _MSBBitReader(data, cur, baseT_B)
            for k, j_i in enumerate(aT_j):
                v = _read_vec3(br, specs_aT[k])
                if 0 <= j_i < num_joints:
                    trans[base_frame, j_i, :] = v
                keyed_t[k, base_frame] = True
        cur += baseT_B

        if aS > 0 and baseS_B > 0:
            br = _MSBBitReader(data, cur, baseS_B)
            for k, j_i in enumerate(aS_j):
                v = _read_vec3(br, specs_aS[k])
                if 0 <= j_i < num_joints:
                    scales[base_frame, j_i, :] = v
                keyed_s[k, base_frame] = True
        cur += baseS_B + baseU_B

        if n_intra <= 0:
            continue

        total_mask_bits = (aR + aT + aS + aU) * n_intra
        mask_bytes = (total_mask_bits + 7) // 8
        br_mask = _MSBBitReader(data, cur, mask_bytes)
        cur += mask_bytes

        r_flags = [[br_mask.read(1) for _ in range(n_intra)] for _ in range(aR)]
        t_flags = [[br_mask.read(1) for _ in range(n_intra)] for _ in range(aT)]
        s_flags = [[br_mask.read(1) for _ in range(n_intra)] for _ in range(aS)]

        if aR > 0 and intraR_B > 0:
            br = _MSBBitReader(data, cur, intraR_B)
            for k, j_i in enumerate(aR_j):
                sp = specs_aR[k]
                for f_off, flag in enumerate(r_flags[k]):
                    if flag:
                        q = _read_quat(br, sp)
                        f_idx = base_frame + 1 + f_off
                        if f_idx < num_frames:
                            if 0 <= j_i < num_joints:
                                rot[f_idx, j_i, :] = q
                            keyed_r[k, f_idx] = True
        cur += intraR_B

        if aT > 0 and intraT_B > 0:
            br = _MSBBitReader(data, cur, intraT_B)
            for k, j_i in enumerate(aT_j):
                sp = specs_aT[k]
                for f_off, flag in enumerate(t_flags[k]):
                    if flag:
                        v = _read_vec3(br, sp)
                        f_idx = base_frame + 1 + f_off
                        if f_idx < num_frames:
                            if 0 <= j_i < num_joints:
                                trans[f_idx, j_i, :] = v
                            keyed_t[k, f_idx] = True
        cur += intraT_B

        if aS > 0 and intraS_B > 0:
            br = _MSBBitReader(data, cur, intraS_B)
            for k, j_i in enumerate(aS_j):
                sp = specs_aS[k]
                for f_off, flag in enumerate(s_flags[k]):
                    if flag:
                        v = _read_vec3(br, sp)
                        f_idx = base_frame + 1 + f_off
                        if f_idx < num_frames:
                            if 0 <= j_i < num_joints:
                                scales[f_idx, j_i, :] = v
                            keyed_s[k, f_idx] = True
        cur += intraS_B

    for k, j_i in enumerate(aR_j):
        if not (0 <= j_i < num_joints):
            continue
        kframes = np.flatnonzero(keyed_r[k])
        if len(kframes) == 0:
            continue
        if kframes[0] > 0:
            rot[: kframes[0], j_i, :] = rot[kframes[0], j_i, :]
        for idx in range(len(kframes) - 1):
            f0, f1 = int(kframes[idx]), int(kframes[idx + 1])
            if f1 - f0 > 1:
                q0, q1 = rot[f0, j_i], rot[f1, j_i]
                inv_span = 1.0 / float(f1 - f0)
                for f in range(f0 + 1, f1):
                    rot[f, j_i, :] = _slerp_quat(q0, q1, (f - f0) * inv_span)
        if kframes[-1] < num_frames - 1:
            rot[kframes[-1] + 1 :, j_i, :] = rot[kframes[-1], j_i, :]

    for k, j_i in enumerate(aT_j):
        if not (0 <= j_i < num_joints):
            continue
        kframes = np.flatnonzero(keyed_t[k])
        if len(kframes) == 0:
            continue
        if kframes[0] > 0:
            trans[: kframes[0], j_i, :] = trans[kframes[0], j_i, :]
        for idx in range(len(kframes) - 1):
            f0, f1 = int(kframes[idx]), int(kframes[idx + 1])
            if f1 - f0 > 1:
                v0, v1 = trans[f0, j_i], trans[f1, j_i]
                inv_span = 1.0 / float(f1 - f0)
                for f in range(f0 + 1, f1):
                    trans[f, j_i, :] = v0 + ((f - f0) * inv_span) * (v1 - v0)
        if kframes[-1] < num_frames - 1:
            trans[kframes[-1] + 1 :, j_i, :] = trans[kframes[-1], j_i, :]

    for k, j_i in enumerate(aS_j):
        if not (0 <= j_i < num_joints):
            continue
        kframes = np.flatnonzero(keyed_s[k])
        if len(kframes) == 0:
            continue
        if kframes[0] > 0:
            scales[: kframes[0], j_i, :] = scales[kframes[0], j_i, :]
        for idx in range(len(kframes) - 1):
            f0, f1 = int(kframes[idx]), int(kframes[idx + 1])
            if f1 - f0 > 1:
                v0, v1 = scales[f0, j_i], scales[f1, j_i]
                inv_span = 1.0 / float(f1 - f0)
                for f in range(f0 + 1, f1):
                    scales[f, j_i, :] = v0 + ((f - f0) * inv_span) * (v1 - v0)
        if kframes[-1] < num_frames - 1:
            scales[kframes[-1] + 1 :, j_i, :] = scales[kframes[-1], j_i, :]

    for j_i in range(num_joints):
        if not has_rot_track[j_i]:
            continue
        for f in range(1, num_frames):
            if float(np.dot(rot[f - 1, j_i], rot[f, j_i])) < 0.0:
                rot[f, j_i] = -rot[f, j_i]

    jnames = (
        list(joint_names[:num_joints])
        if joint_names and len(joint_names) >= num_joints
        else [f"joint_{i}" for i in range(num_joints)]
    )

    return DecodedAnimation(
        anim_id=anim_id,
        name=name,
        category=category,
        skeleton_id=skeleton_id,
        duration=hdr["duration"],
        sample_rate=hdr["sample_rate"],
        num_frames=num_frames,
        joint_names=jnames,
        rotations_xyzw=rot,
        translations=trans,
        scales=scales,
        has_rot_track=has_rot_track,
        has_trans_track=has_trans_track,
        has_scale_track=has_scale_track,
        locomotion_delta_t=locomotion_delta_t,
        events=list(events or ()),
        description=description,
    )


def _extract_events_for_anim(
    game: HFWGame,
    anim_obj: dict[str, Any],
    group_objs: list[dict[str, Any]] | None = None,
    payload_name_cache: dict[str, str] | None = None,
) -> list[tuple[float, str]]:
    """Extract `(time_sec, event_name)` pairs from `SkeletonAnimationResource.Events`."""
    ev_ref = anim_obj.get("Events")
    ev_oid = _ref_target_id(ev_ref)
    ev_obj = ev_ref.target if isinstance(ev_ref, ObjectRef) and ev_ref.target is not None else None
    if ev_obj is None and ev_oid is not None:
        if group_objs is not None and ev_oid.group_id == anim_obj["_id"].group_id:
            if 0 <= ev_oid.object_index < len(group_objs):
                ev_obj = group_objs[ev_oid.object_index]
        else:
            try:
                ev_obj = _read_obj_fast(game, ev_oid)
            except Exception:
                ev_obj = None
    if not ev_obj:
        return []

    def _name_from_payload_obj(po: dict[str, Any]) -> str:
        for k in ("Name", "EventName", "Symbol", "Label", "ID"):
            v = po.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    out: list[tuple[float, str]] = []
    for e in ev_obj.get("Events") or ():
        ename = str(e.get("EventName") or "").strip()
        if not ename:
            p_ref = e.get("Payload")
            p_obj = p_ref.target if isinstance(p_ref, ObjectRef) and p_ref.target is not None else None
            if p_obj is not None:
                ename = _name_from_payload_obj(p_obj)
            else:
                p_oid = _ref_target_id(p_ref)
                if p_oid is not None:
                    pkey = str(p_oid)
                    if payload_name_cache is not None and pkey in payload_name_cache:
                        ename = payload_name_cache[pkey]
                    else:
                        try:
                            po = _read_obj_fast(game, p_oid)
                            ename = _name_from_payload_obj(po)
                        except Exception:
                            ename = ""
                        if payload_name_cache is not None:
                            payload_name_cache[pkey] = ename
        if not ename:
            continue
        t = float(e.get("StartTime", e.get("Time", 0.0)) or 0.0)
        out.append((t, ename))
    return out


def _compute_yaw_delta_deg(rot_dict: dict[str, Any] | None) -> float:
    """Compute signed yaw rotation in degrees around Z from `LocomotionDeltaRotation` quaternion."""
    if not isinstance(rot_dict, dict):
        return 0.0
    x = float(rot_dict.get("X", 0.0) or 0.0)
    y = float(rot_dict.get("Y", 0.0) or 0.0)
    z = float(rot_dict.get("Z", 0.0) or 0.0)
    w = float(rot_dict.get("W", 1.0) or 1.0)
    # ZYX yaw from quaternion
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


def _classify_clip_name_category_desc(
    anim_id: str,
    duration: float,
    num_frames: int,
    delta_t: tuple[float, float, float],
    yaw_deg: float,
    events: list[str],
    index_hint: int = 0,
) -> tuple[str, str, str]:
    """Classify an HFW clip into `(readable_name, canonical_category, plain_english_description)`."""
    ev_joined = " ".join(events).lower()
    horiz_dist = math.hypot(delta_t[0], delta_t[1])
    vert_dist = abs(delta_t[2])
    speed = horiz_dist / max(duration, 0.1)
    id_sfx = anim_id.replace(":", "_")

    # Highlight non-boilerplate events for the user-friendly description
    _NOISE_EV = {"disableheadik", "disablegroundalignment", "disablenavmeshclamping", "disablegrapplestrike", "endofstate", "endwarp", "displacement", "updatetarget", "collision_path", "pose_blend_in"}
    key_evs = [e for e in events if e.lower() not in _NOISE_EV][:4]
    ev_tail = f" — events: {', '.join(key_evs)}" if key_evs else ""
    dur_str = f"{duration:.2f}s, {num_frames}f"

    def _finish(label: str, cat: str, summary: str) -> tuple[str, str, str]:
        cat_lbl = dict(ANIMATION_CATEGORIES).get(cat, cat.title())
        return f"{label}_{id_sfx}", cat, f"{cat_lbl} • {summary} ({dur_str}){ev_tail}"

    # 1. Explicit Combat / Attack / Hit-Reaction events
    for kw, label, desc in (
        ("multistomp", "MultiStomp_Attack", "Multi-stomp ground slam attack"),
        ("stomp", "Stomp_Attack", "Heavy foot stomp attack"),
        ("tailspindamage", "TailSpin_Attack", "360° spinning tail sweep attack"),
        ("chargetail", "TailCharge_Attack", "Charged tail whip attack"),
        ("tail", "Tail_Attack", "Tail strike attack"),
        ("biteburst", "BiteBurst_Attack", "Bite burst / lunge attack"),
        ("bite", "Bite_Attack", "Bite melee attack"),
        ("chargedamage", "Charge_Impact_Attack", "Charging ram attack"),
        ("weaponfiring", "Weapon_Fire", "Ranged weapon firing"),
        ("weaponcharging", "Weapon_Charge", "Charging ranged weapon / plasma"),
        ("attack_start", "Melee_Attack", "Melee strike / combat attack"),
        ("primarydamagepoint", "Combat_Strike", "Primary melee attack strike"),
        ("damagesecondary", "Combo_Attack", "Combo follow-up strike"),
        ("endto_combo", "Combo_Attack", "Combo attack chain"),
        ("endtocombo", "Combo_Attack", "Combo attack chain"),
        ("followupattack", "FollowUp_Attack", "Follow-up melee attack"),
        ("damage", "Combat_Attack", "Combat attack"),
        ("charge", "Charge_Attack", "Aggressive charge attack"),
        ("to_ragdoll", "Death_Knockdown", "Knockdown / death transition"),
        ("stunned_getup", "Stunned_Getup", "Recover / get up from stun"),
        ("tumblehit", "Tumble_HitReact", "Heavy tumble hit reaction"),
        ("endflinch", "Flinch_HitReact", "Combat flinch / stagger reaction"),
        ("breakfree", "BreakFree_Struggle", "Break-free struggle reaction"),
        ("disableshield", "Shield_Break", "Armor / shield break reaction"),
        ("death", "Death", "Death animation"),
        ("knockdown", "Knockdown", "Knockdown reaction"),
        ("stagger", "Stagger", "Stagger hit reaction"),
    ):
        if kw in ev_joined:
            return _finish(label, "combat", desc)

    # 2. Jumps, Leaps, Vaults, Ground Breach
    for kw, label, desc in (
        ("groundbreach", "Ground_Breach", "Burrow / ground breach emergence"),
        ("sfxgearvault", "Vault_Jump", "Obstacle vault / leap"),
        ("takeoffgroundcontact", "Jump_Leap", "Jump / leap takeoff & landing"),
        ("landinggroundcontact", "Jump_Land", "Jump / drop landing"),
        ("pounce", "Pounce_Leap", "Pounce leap"),
    ):
        if kw in ev_joined:
            return _finish(label, "jump", desc)

    # 3. Emotes, Roars, Snarls, Gestures
    for kw, label, desc in (
        ("callforreinforcements", "Call_Reinforcements", "Alert howl / call for reinforcements"),
        ("mouthmovementsnarl", "Snarl_Roar", "Snarl / threat roar"),
        ("mouthmovement", "Vocal_Roar", "Vocalization / roar"),
        ("scream", "Scream_Roar", "Machine scream / roar"),
        ("roar", "Roar", "Threat roar"),
        ("gesture_trigger_eat", "Gesture_Eat", "Eating gesture"),
        ("gesture_trigger_drink", "Gesture_Drink", "Drinking gesture"),
        ("gesture_trigger_chew", "Gesture_Chew", "Chewing gesture"),
        ("sfxgeararmstretch", "Emote_Stretch", "Idle stretch / arm emote"),
        ("gesture", "Gesture", "Expressive conversation gesture"),
    ):
        if kw in ev_joined:
            return _finish(label, "emote", desc)

    # 4. Mount / Rider specific
    if any(k in ev_joined for k in ("rider_start", "rider_footik", "rider_handik", "rider_reins")):
        return _finish("Mount_Rider", "mounted", "Mounted / rider interaction")

    # 5. Movement / Locomotion (explicit SFX/events + kinematics)
    for kw, label, desc in (
        ("sfxgearclimbladder", "Climb_Ladder", "Ladder climbing movement"),
        ("sfxgearsneak", "Sneak_Move", "Stealth crouch sneaking"),
        ("sfxgearstandcrouch", "Stand_To_Crouch", "Transition stand to crouch"),
        ("sfxgearcrouchstand", "Crouch_To_Stand", "Transition crouch to stand"),
        ("sfxgearwalk", "Walk", "Walking cycle"),
        ("sprint", "Sprint", "Full-speed sprint"),
        ("footsteps_sprint", "Sprint", "Full-speed sprint"),
        ("footsteps_run", "Run", "Running cycle"),
        ("footsteps_walk", "Walk", "Walking cycle"),
        ("endtostand", "Stop_To_Stand", "Decelerate / stop to standing"),
        ("endtomove", "Start_Move", "Start moving / transition into walk or run"),
    ):
        if kw in ev_joined:
            extra = f" ({horiz_dist:.1f}m)" if horiz_dist > 0.2 else ""
            return _finish(label, "movement", f"{desc}{extra}")

    # 6. Idle explicit events
    if "grazing" in ev_joined:
        return _finish("Idle_Grazing", "idle", "Ambient grazing / foraging idle")
    if "idle" in ev_joined:
        return _finish("Idle", "idle", "Idle loop / stance")

    # 7. Kinematics fallback (Jump -> Sprint -> Run -> Walk -> Turn -> Footstep Move -> Idle)
    if vert_dist > 0.65:
        return _finish("Jump_Leap", "jump", f"Vertical leap ({vert_dist:.1f}m height)")
    if speed > 4.0:
        return _finish("Sprint_Fwd", "movement", f"High-speed sprint ({horiz_dist:.1f}m @ {speed:.1f} m/s)")
    if speed > 1.7:
        return _finish("Run_Fwd", "movement", f"Running locomotion ({horiz_dist:.1f}m @ {speed:.1f} m/s)")
    if speed > 0.30:
        return _finish("Walk_Fwd", "movement", f"Walking locomotion ({horiz_dist:.1f}m @ {speed:.1f} m/s)")
    if abs(yaw_deg) >= 35.0:
        side = "Left" if yaw_deg > 0 else "Right"
        deg_lbl = 180 if abs(yaw_deg) >= 135 else (90 if abs(yaw_deg) >= 65 else 45)
        return _finish(f"Turn_{side}_{deg_lbl}", "movement", f"In-place turn {side.lower()} ({abs(yaw_deg):.0f}°)")
    if any(k in ev_joined for k in ("lbackfoot", "rbackfoot", "lfrontfoot", "rfrontfoot", "leftfoot", "rightfoot")):
        return _finish("Footstep_Move", "movement", f"Stepping / locomotion adjustment ({horiz_dist:.1f}m)")
    if duration >= 1.5 and horiz_dist < 0.20:
        return _finish("Idle_Stance", "idle", "Stationary idle / alert stance")
    return _finish(f"Action_{index_hint:03d}", "mounted", "General / special animation clip")


def _load_skeleton_joints_and_bind(
    game: HFWGame,
    skeleton_id: str,
    cache: dict[str, tuple[list[str], list[np.ndarray]]] | None = None,
) -> tuple[list[str], list[np.ndarray]]:
    """Read joint names and local bind matrices for a Decima `Skeleton` ObjectId."""
    if cache is not None and skeleton_id in cache:
        return cache[skeleton_id]
    try:
        sobj = _read_obj_fast(game, ObjectId.parse(skeleton_id))
        jnames = [str(j.get("Name", f"joint_{i}")) for i, j in enumerate(sobj.get("Joints") or ())]
        ebytes = sobj.get("EdgeAnimSkeleton", b"")
        bind_mats = parse_edgeanim_base_pose(ebytes) if (sobj.get("HasBindPose") and ebytes) else []
        res = (jnames, bind_mats)
    except Exception:
        res = ([], [])
    if cache is not None:
        cache[skeleton_id] = res
    return res


def summarize_animation_categories(clips: list[AnimationClipInfo]) -> dict[str, int]:
    """Return per-category clip counts including `all` total."""
    counts: dict[str, int] = {"all": len(clips)}
    for k, _ in ANIMATION_CATEGORIES:
        counts[k] = 0
    for c in clips:
        counts[c.category] = counts.get(c.category, 0) + 1
    return counts


def discover_entity_animations(
    game: HFWGame,
    entity_group_id: int | None = None,
    *,
    asset_kind: str = "machine",
    target_skeleton_id: str | None = None,
    gender: str = "female",
    categories: list[str] | tuple[str, ...] | set[str] | None = None,
    query: str = "",
    limit: int | None = None,
) -> list[AnimationClipInfo]:
    """Discover compatible `SkeletonAnimationResource` clips for a machine or character.

    Pass `limit=None` (or `limit=0`) to return **ALL** compatible clips without artificial truncation!
    Pass `categories=["combat"]` or `["combat", "movement"]` to filter by motion type.
    """
    cat_filter: set[str] | None = None
    if categories:
        cat_filter = {normalize_animation_category(c) for c in categories if c and c.strip().lower() != "all"}
        if not cat_filter:
            cat_filter = None

    candidate_groups: list[int] = []
    if asset_kind == "machine" and entity_group_id is not None:
        candidate_groups.append(entity_group_id)
        try:
            sg_ranked: list[tuple[int, int]] = []
            for sg in game.graph.group_subgroups(entity_group_id):
                if sg in candidate_groups or sg == 1:
                    continue
                cnt = sum(1 for t in game.graph.group_types(sg) if t == "SkeletonAnimationResource")
                if cnt > 0:
                    sg_ranked.append((cnt, sg))
            sg_ranked.sort(reverse=True)
            for _cnt, sg in sg_ranked:
                candidate_groups.append(sg)
        except Exception:
            pass
    else:
        desired_skel = target_skeleton_id
        if not desired_skel or desired_skel not in (FEMALE_HUMAN_SKELETON_ID, MALE_HUMAN_SKELETON_ID):
            desired_skel = MALE_HUMAN_SKELETON_ID if gender.lower().startswith("m") else FEMALE_HUMAN_SKELETON_ID
        target_skeleton_id = desired_skel
        if entity_group_id is not None and entity_group_id not in candidate_groups:
            try:
                if "SkeletonAnimationResource" in game.graph.group_types(entity_group_id):
                    candidate_groups.append(entity_group_id)
            except Exception:
                pass
        # For humanoid characters, scan gameplay/gesture hubs (528, 108, 34) first!
        # Avoid dumping 10,000 raw cutscene clips unless specifically requested with a very high limit
        hum_groups = (528, 108, 34) if asset_kind == "character" else DEFAULT_HUMANOID_ANIM_GROUPS
        for gid in hum_groups:
            if gid in game.graph.groups and gid not in candidate_groups:
                candidate_groups.append(gid)

    clips: list[AnimationClipInfo] = []
    seen_ids: set[str] = set()
    payload_name_cache: dict[str, str] = {}

    for gid in candidate_groups:
        try:
            types = game.graph.group_types(gid)
        except Exception:
            continue
        sar_indices = [i for i, t in enumerate(types) if t == "SkeletonAnimationResource"]
        if not sar_indices:
            continue

        try:
            objs = game.read_group_local(gid, cache=False)
        except Exception:
            continue

        for local_i, oidx in enumerate(sar_indices):
            aobj = objs[oidx]
            aid = str(aobj["_id"])
            if aid in seen_ids:
                continue
            ad = aobj.get("AnimationData")
            if not isinstance(ad, (bytes, bytearray)):
                continue
            hdr = parse_ea08_header(bytes(ad))
            if hdr is None or hdr["num_frames"] < 2 or hdr["duration"] < 0.05:
                continue
            if (hdr["cR"] + hdr["aR"] + hdr["cT"] + hdr["aT"]) == 0:
                continue

            sk_oid = _ref_target_id(aobj.get("Skeleton"))
            sk_id = str(sk_oid) if sk_oid is not None else ""
            if asset_kind == "character" and sk_id not in (FEMALE_HUMAN_SKELETON_ID, MALE_HUMAN_SKELETON_ID):
                continue
            if target_skeleton_id and sk_id and sk_id != target_skeleton_id:
                if not (
                    asset_kind == "character"
                    and sk_id in (FEMALE_HUMAN_SKELETON_ID, MALE_HUMAN_SKELETON_ID)
                    and target_skeleton_id in (FEMALE_HUMAN_SKELETON_ID, MALE_HUMAN_SKELETON_ID)
                ):
                    continue

            dt_dict = aobj.get("LocomotionDeltaTranslation") or {}
            delta_t = (
                float(dt_dict.get("X", 0.0) or 0.0),
                float(dt_dict.get("Y", 0.0) or 0.0),
                float(dt_dict.get("Z", 0.0) or 0.0),
            )
            yaw_deg = _compute_yaw_delta_deg(aobj.get("LocomotionDeltaRotation"))
            ev_pairs = _extract_events_for_anim(game, aobj, objs, payload_name_cache)
            ev_names = tuple(dict.fromkeys(nm for _, nm in ev_pairs))
            cname, cat, desc = _classify_clip_name_category_desc(
                aid, hdr["duration"], hdr["num_frames"], delta_t, yaw_deg, list(ev_names), local_i
            )

            seen_ids.add(aid)
            clips.append(
                AnimationClipInfo(
                    anim_id=aid,
                    name=cname,
                    category=cat,
                    skeleton_id=sk_id,
                    duration=hdr["duration"],
                    sample_rate=hdr["sample_rate"],
                    num_frames=hdr["num_frames"],
                    num_joints=hdr["num_joints"],
                    locomotion_delta_t=delta_t,
                    yaw_delta_deg=round(yaw_deg, 1),
                    events=ev_names,
                    description=desc,
                    group_id=gid,
                )
            )

    # Keep clips matching the machine's dominant skeleton (filters out unrelated rider/gadget sub-clips)
    if asset_kind == "machine" and not target_skeleton_id and clips:
        counts: dict[str, int] = {}
        for c in clips:
            if c.skeleton_id:
                counts[c.skeleton_id] = counts.get(c.skeleton_id, 0) + 1
        if counts:
            dominant = max(counts, key=lambda k: (counts[k], max(c.num_joints for c in clips if c.skeleton_id == k)))
            clips = [c for c in clips if c.skeleton_id == dominant]

    if cat_filter is not None:
        clips = [c for c in clips if c.category in cat_filter]

    if query and query.strip():
        ql = query.strip().lower()
        clips = [
            c
            for c in clips
            if ql in c.name.lower()
            or ql in c.category.lower()
            or ql in c.description.lower()
            or ql in c.anim_id.lower()
            or any(ql in ev.lower() for ev in c.events)
        ]

    cat_priority = {"combat": 0, "movement": 1, "idle": 2, "jump": 3, "emote": 4, "mounted": 5}
    clips.sort(
        key=lambda c: (
            cat_priority.get(c.category, 9),
            0 if c.events else 1,
            c.name,
        )
    )
    if limit is not None and limit > 0:
        return clips[:limit]
    return clips


def select_diverse_animations(clips: list[AnimationClipInfo], count: int) -> list[AnimationClipInfo]:
    """Select `count` diverse representative clips spanning categories (Combat, Movement, Idle, Jump, Emote)."""
    if count <= 0 or not clips:
        return []
    if len(clips) <= count:
        return list(clips)

    selected: list[AnimationClipInfo] = []
    used: set[str] = set()
    # Round-robin across categories so larger showcase counts stay balanced!
    by_cat: dict[str, list[AnimationClipInfo]] = {}
    for c in clips:
        by_cat.setdefault(c.category, []).append(c)

    order = ("idle", "movement", "combat", "jump", "emote", "mounted")
    made_progress = True
    while len(selected) < count and made_progress:
        made_progress = False
        for cat in order:
            bucket = by_cat.get(cat)
            while bucket:
                cand = bucket.pop(0)
                if cand.anim_id not in used:
                    selected.append(cand)
                    used.add(cand.anim_id)
                    made_progress = True
                    break
            if len(selected) >= count:
                break
    return selected[:count]


def decode_animation_resource(
    game: HFWGame,
    anim_id: str | ObjectId,
    *,
    name: str = "",
    category: str = "",
    description: str = "",
    skel_cache: dict[str, tuple[list[str], list[np.ndarray]]] | None = None,
    payload_name_cache: dict[str, str] | None = None,
) -> DecodedAnimation:
    """Read and decode a `SkeletonAnimationResource` ObjectId from the HFW archives."""
    oid = anim_id if isinstance(anim_id, ObjectId) else ObjectId.parse(str(anim_id))
    aobj = _read_obj_fast(game, oid)
    if aobj.get("_type") != "SkeletonAnimationResource":
        raise ValueError(f"Object {oid} is {aobj.get('_type')}, expected SkeletonAnimationResource")

    ad = aobj.get("AnimationData")
    if not isinstance(ad, (bytes, bytearray)):
        raise ValueError(f"SkeletonAnimationResource {oid} has no AnimationData")

    sk_oid = _ref_target_id(aobj.get("Skeleton"))
    sk_id = str(sk_oid) if sk_oid is not None else ""
    jnames, bind_mats = _load_skeleton_joints_and_bind(game, sk_id, skel_cache) if sk_id else ([], [])

    dt_dict = aobj.get("LocomotionDeltaTranslation") or {}
    delta_t = (
        float(dt_dict.get("X", 0.0) or 0.0),
        float(dt_dict.get("Y", 0.0) or 0.0),
        float(dt_dict.get("Z", 0.0) or 0.0),
    )
    yaw_deg = _compute_yaw_delta_deg(aobj.get("LocomotionDeltaRotation"))
    events = _extract_events_for_anim(game, aobj, payload_name_cache=payload_name_cache)
    hdr = parse_ea08_header(bytes(ad))
    if not name or not description:
        auto_name, auto_cat, auto_desc = _classify_clip_name_category_desc(
            str(oid),
            float(hdr["duration"]) if hdr else float(aobj.get("ComputedDuration", 1.0) or 1.0),
            int(hdr["num_frames"]) if hdr else 30,
            delta_t,
            yaw_deg,
            [nm for _, nm in events],
        )
        if not name:
            name = auto_name
        if not category:
            category = auto_cat
        if not description:
            description = auto_desc

    return decode_edge_anim_ea08(
        bytes(ad),
        anim_id=str(oid),
        name=name,
        category=category or "movement",
        skeleton_id=sk_id,
        joint_names=jnames,
        bind_local_mats=bind_mats,
        locomotion_delta_t=delta_t,
        events=events,
        description=description,
    )


def _quat_xyzw_to_mat3(q: np.ndarray) -> np.ndarray:
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def remap_animation_to_blender_basis(
    decoded: DecodedAnimation,
    target_skeleton: SkeletonData,
    *,
    preserve_non_root_bone_lengths: bool = False,
) -> dict[str, Any]:
    """Map a decoded HFW clip onto `target_skeleton` by joint name and compute exact Blender `pose_bone.matrix_basis` TRS."""
    F = decoded.num_frames
    matched_pairs: list[tuple[int, int, str]] = []

    lower_lookup = {j.name.lower(): j for j in target_skeleton.joints}
    for src_i, jname in enumerate(decoded.joint_names):
        if not (decoded.has_rot_track[src_i] or decoded.has_trans_track[src_i] or decoded.has_scale_track[src_i]):
            continue
        dst_j = target_skeleton.joint_by_name.get(jname) or lower_lookup.get(jname.lower())
        if dst_j is not None:
            matched_pairs.append((src_i, dst_j.index, dst_j.name))

    K = len(matched_pairs)
    out_loc = np.zeros((F, K, 3), dtype=np.float32)
    out_rot = np.zeros((F, K, 4), dtype=np.float32)
    out_rot[:, :, 0] = 1.0
    out_scale = np.ones((F, K, 3), dtype=np.float32)
    out_has_loc = np.zeros(K, dtype=bool)
    out_has_rot = np.zeros(K, dtype=bool)
    out_has_scale = np.zeros(K, dtype=bool)
    out_names: list[str] = []

    root_like = {"rootbone", "groundbone", "hipsbone", "c_hips_sjnt_0", "pelvis"}

    for k, (src_i, dst_i, bname) in enumerate(matched_pairs):
        out_names.append(bname)
        dst_j = target_skeleton.joints[dst_i]

        wmat = dst_j.world_matrix
        w_rot = wmat[:3, :3].copy()
        for col in range(3):
            n = float(np.linalg.norm(w_rot[:, col]))
            if n > 1e-8:
                w_rot[:, col] /= n
        rest_world = np.eye(4, dtype=np.float64)
        rest_world[:3, :3] = w_rot
        rest_world[:3, 3] = wmat[:3, 3]

        if 0 <= dst_j.parent_index < len(target_skeleton.joints) and dst_j.parent_index != dst_i:
            pwmat = target_skeleton.joints[dst_j.parent_index].world_matrix
            pw_rot = pwmat[:3, :3].copy()
            for col in range(3):
                n = float(np.linalg.norm(pw_rot[:, col]))
                if n > 1e-8:
                    pw_rot[:, col] /= n
            parent_rest = np.eye(4, dtype=np.float64)
            parent_rest[:3, :3] = pw_rot
            parent_rest[:3, 3] = pwmat[:3, 3]
            rest_local = np.linalg.inv(parent_rest) @ rest_world
        else:
            rest_local = rest_world

        inv_rest_local = np.linalg.inv(rest_local)
        rest_t, rest_q_xyzw, rest_s = _mat4_to_trs_xyzw(rest_local)

        use_t = bool(decoded.has_trans_track[src_i])
        if preserve_non_root_bone_lengths and bname.lower() not in root_like:
            use_t = False
        use_r = bool(decoded.has_rot_track[src_i])
        use_s = bool(decoded.has_scale_track[src_i])

        for f in range(F):
            anim_local = np.eye(4, dtype=np.float64)
            q_xyzw = decoded.rotations_xyzw[f, src_i] if use_r else rest_q_xyzw
            t_vec = decoded.translations[f, src_i] if use_t else rest_t
            s_vec = decoded.scales[f, src_i] if use_s else rest_s
            rmat = _quat_xyzw_to_mat3(q_xyzw)
            anim_local[:3, 0] = rmat[:, 0] * float(s_vec[0])
            anim_local[:3, 1] = rmat[:, 1] * float(s_vec[1])
            anim_local[:3, 2] = rmat[:, 2] * float(s_vec[2])
            anim_local[:3, 3] = t_vec.astype(np.float64)

            basis = inv_rest_local @ anim_local
            bt, bq_xyzw, bs = _mat4_to_trs_xyzw(basis)
            out_loc[f, k, :] = bt
            out_rot[f, k, 0] = bq_xyzw[3]
            out_rot[f, k, 1] = bq_xyzw[0]
            out_rot[f, k, 2] = bq_xyzw[1]
            out_rot[f, k, 3] = bq_xyzw[2]
            out_scale[f, k, :] = bs

        for f in range(1, F):
            if float(np.dot(out_rot[f - 1, k], out_rot[f, k])) < 0.0:
                out_rot[f, k] = -out_rot[f, k]

        if use_t and float(np.max(np.abs(out_loc[:, k, :]))) > 1e-4:
            out_has_loc[k] = True
        if use_r:
            out_has_rot[k] = True
        if use_s and float(np.max(np.abs(out_scale[:, k, :] - 1.0))) > 1e-3:
            out_has_scale[k] = True

    return {
        "anim_id": decoded.anim_id,
        "name": decoded.name,
        "category": decoded.category,
        "description": decoded.description,
        "skeleton_id": decoded.skeleton_id,
        "duration": decoded.duration,
        "sample_rate": decoded.sample_rate,
        "num_frames": F,
        "joint_names": out_names,
        "loc": out_loc,
        "rot_wxyz": out_rot,
        "scale": out_scale,
        "has_loc": out_has_loc,
        "has_rot": out_has_rot,
        "has_scale": out_has_scale,
        "locomotion_delta_t": list(decoded.locomotion_delta_t),
        "events": [{"time": t, "name": nm} for t, nm in decoded.events],
    }


def resolve_animations_for_export(
    game: HFWGame,
    skeleton: SkeletonData,
    *,
    entity_group_id: int | None = None,
    asset_kind: str = "machine",
    animations_spec: int | list[str] | str | None = None,
    gender: str = "female",
) -> list[dict[str, Any]]:
    """Resolve `animations_spec` into remapped clip dicts with NO artificial clip cap!

    Supported `animations_spec` formats:
      - `"all"` -> ALL compatible clips for this machine (or gameplay/locomotion set for characters)
      - `"category:combat"` / `"category:movement,jump"` -> ALL clips matching those category(ies)
      - `int` / `"15"` -> N diverse showcase clips spanning all categories
      - `"1587:321,1587:1076"` / `["1587:321", ...]` -> explicit ObjectIds
    """
    if not animations_spec:
        return []

    explicit_ids: list[str] = []
    select_all = False
    category_filter: list[str] | None = None
    count = 0

    if isinstance(animations_spec, int):
        count = max(0, animations_spec)
    elif isinstance(animations_spec, str):
        s = animations_spec.strip()
        sl = s.lower()
        if not s or s == "0" or sl == "none":
            return []
        if sl == "all":
            select_all = True
        elif sl.startswith(("category:", "cat:")):
            raw_cats = s.split(":", 1)[1]
            category_filter = [normalize_animation_category(tok) for tok in raw_cats.split(",") if tok.strip()]
            select_all = True
        elif sl in _CATEGORY_ALIASES:
            category_filter = [_CATEGORY_ALIASES[sl]]
            select_all = True
        elif s.isdigit():
            count = int(s)
        else:
            explicit_ids = [tok.strip() for tok in s.split(",") if tok.strip()]
    elif isinstance(animations_spec, (list, tuple)):
        for item in animations_spec:
            tok = str(item).strip()
            if tok:
                explicit_ids.append(tok)

    skel_cache: dict[str, tuple[list[str], list[np.ndarray]]] = {}
    payload_cache: dict[str, str] = {}
    chosen: list[tuple[str, str, str, str]] = []  # (anim_id, name, category, description)

    if explicit_ids:
        for aid in explicit_ids:
            chosen.append((aid, "", "", ""))
    elif select_all or count > 0:
        discovered = discover_entity_animations(
            game,
            entity_group_id=entity_group_id,
            asset_kind=asset_kind,
            target_skeleton_id=skeleton.object_id if asset_kind == "machine" else None,
            gender=gender,
            categories=category_filter,
            limit=None,
        )
        if not discovered and asset_kind == "machine" and entity_group_id is not None:
            discovered = discover_entity_animations(
                game,
                entity_group_id=entity_group_id,
                asset_kind="machine",
                target_skeleton_id=None,
                gender=gender,
                categories=category_filter,
                limit=None,
            )
        picked = discovered if select_all else select_diverse_animations(discovered, count)
        chosen = [(c.anim_id, c.name, c.category, c.description) for c in picked]

    preserve_lengths = asset_kind == "character"
    results: list[dict[str, Any]] = []
    for aid, cname, ccat, cdesc in chosen:
        try:
            dec = decode_animation_resource(
                game,
                aid,
                name=cname,
                category=ccat,
                description=cdesc,
                skel_cache=skel_cache,
                payload_name_cache=payload_cache,
            )
            remapped = remap_animation_to_blender_basis(
                dec,
                skeleton,
                preserve_non_root_bone_lengths=preserve_lengths,
            )
            if remapped["joint_names"]:
                results.append(remapped)
        except Exception:
            continue
    return results


def export_animation_pack(
    game: HFWGame,
    skeleton: SkeletonData,
    out_dir: Path,
    *,
    entity_group_id: int | None = None,
    asset_kind: str = "machine",
    animations_spec: int | list[str] | str = "all",
    gender: str = "female",
) -> Path:
    """Export ONLY `assembly.json` (animations manifest) + `animations.npz` — fast add-on import without re-exporting meshes/textures!"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    decoded_clips = resolve_animations_for_export(
        game,
        skeleton,
        entity_group_id=entity_group_id,
        asset_kind=asset_kind,
        animations_spec=animations_spec,
        gender=gender,
    )
    anim_arrays: dict[str, np.ndarray] = {}
    anim_metas: list[dict[str, Any]] = []
    for ai, clip in enumerate(decoded_clips):
        akey = f"a{ai}"
        anim_arrays[f"{akey}_loc"] = clip["loc"]
        anim_arrays[f"{akey}_rot"] = clip["rot_wxyz"]
        anim_arrays[f"{akey}_scale"] = clip["scale"]
        anim_arrays[f"{akey}_has_loc"] = clip["has_loc"]
        anim_arrays[f"{akey}_has_rot"] = clip["has_rot"]
        anim_arrays[f"{akey}_has_scale"] = clip["has_scale"]
        safe_fn = re.sub(r"[^A-Za-z0-9_\-.]+", "_", clip["name"]).strip("_")[:52] or f"Anim_{ai}"
        anim_metas.append(
            {
                "key": akey,
                "anim_id": clip["anim_id"],
                "name": clip["name"],
                "category": clip["category"],
                "description": clip.get("description", ""),
                "skeleton_id": clip["skeleton_id"],
                "duration": round(float(clip["duration"]), 4),
                "sample_rate": round(float(clip["sample_rate"]), 3),
                "num_frames": int(clip["num_frames"]),
                "joint_names": clip["joint_names"],
                "locomotion_delta_t": clip["locomotion_delta_t"],
                "events": clip["events"],
                "fbx_filename": f"HFW_{safe_fn}.fbx",
            }
        )
    if anim_arrays:
        np.savez_compressed(out_dir / "animations.npz", **anim_arrays)
    (out_dir / "assembly.json").write_text(
        json.dumps({"schema_version": 4, "animations": anim_metas}, indent=2),
        encoding="utf-8",
    )
    return out_dir
