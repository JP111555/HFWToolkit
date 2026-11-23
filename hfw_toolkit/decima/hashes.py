"""Decima Engine hashing utilities (MurmurHash3_x64_128, CRC32C, FNV-1a 64)."""

from __future__ import annotations

import struct

try:
    import mmh3 as _mmh3
except ImportError:
    _mmh3 = None


def murmur3_x64_128(data: bytes | str, seed: int = 42) -> tuple[int, int]:
    """Return unsigned (h1, h2) 64-bit halves of MurmurHash3_x64_128."""
    raw = data.encode("utf-8") if isinstance(data, str) else data
    if _mmh3 is not None:
        return _mmh3.hash64(raw, seed=seed, signed=False)
    return _murmur3_x64_128_py(raw, seed)


def decima_type_hash(type_name: str) -> int:
    """Compute Decima HFW 64-bit RTTI type ID hash for `type_name`."""
    h1, _ = murmur3_x64_128(("00000001_" + type_name).encode("utf-8"), seed=42)
    return h1


def _murmur3_x64_128_py(data: bytes, seed: int = 42) -> tuple[int, int]:
    m64 = 0xFFFFFFFFFFFFFFFF
    c1 = 0x87C37B91114253D5
    c2 = 0x4CF5AD432745937F

    def rotl64(x: int, r: int) -> int:
        return ((x << r) | (x >> (64 - r))) & m64

    def fmix64(k: int) -> int:
        k ^= k >> 33
        k = (k * 0xFF51AFD7ED558CCD) & m64
        k ^= k >> 33
        k = (k * 0xC4CEB9FE1A85EC53) & m64
        k ^= k >> 33
        return k

    h1 = seed & m64
    h2 = seed & m64
    n = len(data)
    nblocks = n // 16
    for i in range(nblocks):
        k1, k2 = struct.unpack_from("<QQ", data, i * 16)
        k1 = (k1 * c1) & m64
        k1 = rotl64(k1, 31)
        k1 = (k1 * c2) & m64
        h1 ^= k1
        h1 = rotl64(h1, 27)
        h1 = (h1 + h2) & m64
        h1 = (h1 * 5 + 0x52DCE729) & m64

        k2 = (k2 * c2) & m64
        k2 = rotl64(k2, 33)
        k2 = (k2 * c1) & m64
        h2 ^= k2
        h2 = rotl64(h2, 31)
        h2 = (h2 + h1) & m64
        h2 = (h2 * 5 + 0x38495AB5) & m64

    tail = data[nblocks * 16 :]
    k1 = 0
    k2 = 0
    rem = len(tail)
    for idx in range(rem - 1, -1, -1):
        if idx >= 8:
            k2 = (k2 << 8) | tail[idx]
        else:
            k1 = (k1 << 8) | tail[idx]
    if rem > 8:
        k2 = (k2 * c2) & m64
        k2 = rotl64(k2, 33)
        k2 = (k2 * c1) & m64
        h2 ^= k2
    if rem > 0:
        k1 = (k1 * c1) & m64
        k1 = rotl64(k1, 31)
        k1 = (k1 * c2) & m64
        h1 ^= k1

    h1 ^= n
    h2 ^= n
    h1 = (h1 + h2) & m64
    h2 = (h2 + h1) & m64
    h1 = fmix64(h1)
    h2 = fmix64(h2)
    h1 = (h1 + h2) & m64
    h2 = (h2 + h1) & m64
    return h1, h2


# Reflected CRC32C table (polynomial 0x1EDC6F41 -> reflected 0x82F63B78)
# Decima uses init=0, xorout=0, reflected in & out.
_CRC32C_TABLE: tuple[int, ...] = tuple(
    (
        lambda v: [
            v := (v >> 1) ^ 0x82F63B78 if (v & 1) else (v >> 1)
            for _ in range(8)
        ][-1]
    )(i)
    for i in range(256)
)


def decima_crc32c(data: bytes | str) -> int:
    """Decima's CRC32C (init=0, xorout=0, reflected)."""
    raw = data.encode("utf-8") if isinstance(data, str) else data
    crc = 0
    table = _CRC32C_TABLE
    for b in raw:
        crc = table[(crc ^ b) & 0xFF] ^ (crc >> 8)
    return crc & 0xFFFFFFFF


def decima_string_hash(data: bytes | str) -> int:
    """31-bit string checksum used in Decima binary String fields and shader hashes."""
    return decima_crc32c(data) & 0x7FFFFFFF


def fnv1a_64_fold32(name: str) -> int:
    """JoltPhysics Factory type hash: FNV-1a 64-bit XOR-folded to 32-bit."""
    h = 0xCBF29CE484222325
    for b in name.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return ((h ^ (h >> 32)) & 0xFFFFFFFF)
