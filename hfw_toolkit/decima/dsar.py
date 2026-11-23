"""DirectStorage Archive (DSAR v3.1) LZ4 reader and multi-file storage manager."""

from __future__ import annotations

import bisect
import struct
import threading
from collections import OrderedDict
from pathlib import Path

import lz4.block

from hfw_toolkit.config import ToolkitConfig

DSAR_MAGIC = b"DSAR"
CHUNK_STRUCT = struct.Struct("<QQIIB7x")
HEADER_STRUCT = struct.Struct("<4sHHIIQQ")


class BinaryStorageFile:
    """Random-access read-only file backed by either a DSAR v3.1 LZ4 archive or a raw file."""

    def __init__(self, path: Path, cache_chunks: int = 32) -> None:
        self.path = path
        self._fp = open(path, "rb")
        self._lock = threading.Lock()
        self._cache: OrderedDict[int, bytes] = OrderedDict()
        self._cache_limit = cache_chunks

        magic = self._fp.read(4)
        self.is_dsar = magic == DSAR_MAGIC
        self.chunks: list[tuple[int, int, int, int, int]] = []
        self.offsets: list[int] = []

        if self.is_dsar:
            self._fp.seek(0)
            _, vmaj, vmin, chunk_count, _first_off, total_size, _pad = HEADER_STRUCT.unpack(
                self._fp.read(HEADER_STRUCT.size)
            )
            if vmaj != 3 and vmin != 1:
                self._fp.close()
                raise IOError(f"Unsupported DSAR version {vmaj}.{vmin} in {path}")
            self.size = total_size
            raw_table = self._fp.read(chunk_count * CHUNK_STRUCT.size)
            chunks = []
            offsets = []
            unpack_from = CHUNK_STRUCT.unpack_from
            for i in range(chunk_count):
                off, coff, sz, csz, ctype = unpack_from(raw_table, i * 32)
                chunks.append((off, coff, sz, csz, ctype))
                offsets.append(off)
            self.chunks = chunks
            self.offsets = offsets
        else:
            self._fp.seek(0, 2)
            self.size = self._fp.tell()

    def _read_chunk(self, idx: int) -> bytes:
        cached = self._cache.get(idx)
        if cached is not None:
            self._cache.move_to_end(idx)
            return cached
        _off, coff, sz, csz, ctype = self.chunks[idx]
        self._fp.seek(coff)
        comp = self._fp.read(csz)
        if ctype == 0 or (csz == sz and ctype == 0):
            data = comp
        else:
            data = lz4.block.decompress(comp, uncompressed_size=sz)
        self._cache[idx] = data
        if len(self._cache) > self._cache_limit:
            self._cache.popitem(last=False)
        return data

    def read(self, offset: int, length: int) -> bytes:
        if length == 0:
            return b""
        with self._lock:
            if not self.is_dsar:
                self._fp.seek(offset)
                return self._fp.read(length)

            idx = bisect.bisect_right(self.offsets, offset) - 1
            if idx < 0:
                raise EOFError(f"Offset {offset} before first chunk in {self.path}")
            off, _coff, sz, _csz, _ctype = self.chunks[idx]
            rel = offset - off
            if rel + length <= sz:
                # Fast path: single chunk
                chunk_data = self._read_chunk(idx)
                return chunk_data[rel : rel + length]

            out = bytearray(length)
            written = 0
            pos = offset
            rem = length
            while rem > 0:
                idx = bisect.bisect_right(self.offsets, pos) - 1
                off, _coff, sz, _csz, _ctype = self.chunks[idx]
                rel = pos - off
                take = min(sz - rel, rem)
                if take <= 0:
                    raise EOFError(f"Truncated DSAR read at {pos} in {self.path}")
                chunk_data = self._read_chunk(idx)
                out[written : written + take] = chunk_data[rel : rel + take]
                written += take
                pos += take
                rem -= take
            return bytes(out)

    def close(self) -> None:
        with self._lock:
            self._cache.clear()
            if not self._fp.closed:
                self._fp.close()


class StreamingStorage:
    """Manages lazy mounting and cached random-access reads across all HFW package files."""

    def __init__(self, config: ToolkitConfig) -> None:
        self.config = config
        self._files: dict[str, BinaryStorageFile] = {}
        self._lock = threading.Lock()

    def get(self, device_path: str) -> BinaryStorageFile:
        with self._lock:
            sf = self._files.get(device_path)
            if sf is not None:
                return sf
        fs_path = self.config.resolve_device_path(device_path)
        if not fs_path.exists():
            raise FileNotFoundError(f"Decima package file not found: {fs_path} (from {device_path})")
        sf = BinaryStorageFile(fs_path)
        with self._lock:
            existing = self._files.get(device_path)
            if existing is not None:
                sf.close()
                return existing
            self._files[device_path] = sf
            return sf

    def read(self, device_path: str, offset: int, length: int) -> bytes:
        return self.get(device_path).read(offset, length)

    def close(self) -> None:
        with self._lock:
            for sf in self._files.values():
                sf.close()
            self._files.clear()
