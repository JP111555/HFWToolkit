"""Fast binary RTTI deserializer and object reference types for HFW PC."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any, Iterator

from hfw_toolkit.decima.callbacks import read_extra_binary
from hfw_toolkit.decima.rtti import RTTISchema


@dataclass(frozen=True, slots=True)
class ObjectId:
    group_id: int
    object_index: int

    def __str__(self) -> str:
        return f"{self.group_id}:{self.object_index}"

    @classmethod
    def parse(cls, text: str) -> "ObjectId":
        g, i = text.split(":", 1)
        return cls(int(g), int(i))


@dataclass(slots=True)
class ObjectRef:
    """Resolved or lazy intra-/sub-group pointer (`Ref`, `WeakPtr`, `cptr`)."""

    kind: str  # "Ref", "WeakPtr", "cptr"
    target_id: ObjectId
    target: dict[str, Any] | None = None

    def get(self) -> dict[str, Any] | None:
        return self.target


@dataclass(frozen=True, slots=True)
class StreamingRef:
    """Cross-group reference pointing to an object in another streaming group."""

    target_id: ObjectId


@dataclass(frozen=True, slots=True)
class UUIDRef:
    """Reference identified by a 16-byte Decima GGUUID."""

    uuid: str


def gguuid_to_str(raw: dict[str, Any] | bytes) -> str:
    """Format a Decima GGUUID compound or 16 bytes into standard UUID string."""
    if isinstance(raw, bytes):
        b = raw
    else:
        b = bytes(raw[f"Data{i}"] for i in range(16))
    # Decima GGUUID displays big-endian hex chunks matching standard RFC 4122 byte layout
    return (
        f"{b[0:4].hex()}-{b[4:6].hex()}-{b[6:8].hex()}-"
        f"{b[8:10].hex()}-{b[10:16].hex()}"
    )


class BinaryReader:
    """Lightweight little-endian cursor over an in-memory `bytes` or `memoryview` buffer."""

    __slots__ = ("data", "pos", "length")

    def __init__(self, data: bytes | memoryview) -> None:
        self.data = data
        self.pos = 0
        self.length = len(data)

    @property
    def remaining(self) -> int:
        return self.length - self.pos

    def skip(self, n: int) -> None:
        self.pos += n

    def read(self, n: int) -> bytes:
        p = self.pos
        self.pos = p + n
        out = self.data[p : p + n]
        return bytes(out) if isinstance(out, memoryview) else out

    def u8(self) -> int:
        v = self.data[self.pos]
        self.pos += 1
        return v

    def i8(self) -> int:
        v = struct.unpack_from("<b", self.data, self.pos)[0]
        self.pos += 1
        return v

    def u16(self) -> int:
        v = struct.unpack_from("<H", self.data, self.pos)[0]
        self.pos += 2
        return v

    def i16(self) -> int:
        v = struct.unpack_from("<h", self.data, self.pos)[0]
        self.pos += 2
        return v

    def u32(self) -> int:
        v = struct.unpack_from("<I", self.data, self.pos)[0]
        self.pos += 4
        return v

    def i32(self) -> int:
        v = struct.unpack_from("<i", self.data, self.pos)[0]
        self.pos += 4
        return v

    def u64(self) -> int:
        v = struct.unpack_from("<Q", self.data, self.pos)[0]
        self.pos += 8
        return v

    def i64(self) -> int:
        v = struct.unpack_from("<q", self.data, self.pos)[0]
        self.pos += 8
        return v

    def f16(self) -> float:
        v = struct.unpack_from("<e", self.data, self.pos)[0]
        self.pos += 2
        return v

    def f32(self) -> float:
        v = struct.unpack_from("<f", self.data, self.pos)[0]
        self.pos += 4
        return v

    def f64(self) -> float:
        v = struct.unpack_from("<d", self.data, self.pos)[0]
        self.pos += 8
        return v


# Map primitive atom base types to (struct_char, byte_size) for fast array unpacking
_FAST_ATOM_STRUCT: dict[str, tuple[str, int]] = {
    "bool": ("?", 1),
    "uint8": ("B", 1),
    "int8": ("b", 1),
    "uint16": ("H", 2),
    "int16": ("h", 2),
    "wchar": ("H", 2),
    "tchar": ("H", 2),
    "uint": ("I", 4),
    "uint32": ("I", 4),
    "ucs4": ("I", 4),
    "int": ("i", 4),
    "int32": ("i", 4),
    "uint64": ("Q", 8),
    "uintptr": ("Q", 8),
    "int64": ("q", 8),
    "HalfFloat": ("e", 2),
    "float": ("f", 4),
    "double": ("d", 8),
}


class TypeReader:
    """Deserializes Decima RTTI objects from a `BinaryReader` using `RTTISchema`."""

    def __init__(self, schema: RTTISchema) -> None:
        self.schema = schema
        self._plans: dict[str, tuple[list[tuple[str, str]], str | None]] = {}

    def _plan(self, type_name: str) -> tuple[list[tuple[str, str]], str | None]:
        plan = self._plans.get(type_name)
        if plan is not None:
            return plan
        attrs = [(a.name, a.type_name) for a in self.schema.ordered_attrs(type_name)]
        mrb_owner = self.schema.msg_read_binary_owner(type_name)
        plan = (attrs, mrb_owner)
        self._plans[type_name] = plan
        return plan

    def read_standalone_object(self, r: BinaryReader) -> dict[str, Any]:
        """Read a top-level `.core` object (e.g. `streaming_graph.core`) with type hash + size header."""
        type_hash = r.u64()
        expected_size = r.u32()
        type_name = self.schema.hash_to_name.get(type_hash)
        if type_name is None:
            raise KeyError(f"Unknown Decima RTTI type hash: {type_hash:#018x}")
        start = r.pos
        obj = self.read_compound(type_name, r)
        actual_size = r.pos - start
        if actual_size != expected_size:
            raise ValueError(
                f"Size mismatch reading {type_name}: actual={actual_size} != expected={expected_size}"
            )
        num_links = r.u32()
        if num_links != 0:
            raise ValueError(f"Expected 0 trailing links in standalone object {type_name}, got {num_links}")
        return obj

    def read_value(
        self,
        type_name: str,
        r: BinaryReader,
        link_iter: Iterator[tuple[int | None, int]] | None = None,
        loc_iter: Iterator[tuple[int, int]] | None = None,
        cur_group_id: int = -1,
        cur_objects: list[dict[str, Any]] | None = None,
        sub_groups: list[tuple[int, list[dict[str, Any]]]] | None = None,
    ) -> Any:
        tinfo = self.schema.raw[type_name]
        kind = tinfo["kind"]

        if kind == "atom":
            base = self.schema.atom_base(type_name)
            if base == "String":
                ln = r.u32()
                if ln == 0:
                    return ""
                r.skip(4)  # CRC32C checksum
                return r.read(ln).decode("utf-8", errors="replace")
            if base == "WString":
                ln = r.u32()
                return r.read(ln * 2).decode("utf-16-le", errors="replace")
            fmt, sz = _FAST_ATOM_STRUCT[base]
            val = struct.unpack_from("<" + fmt, r.data, r.pos)[0]
            r.pos += sz
            return val

        if kind in ("enum", "enum flags"):
            sz = tinfo["size"]
            if sz == 1:
                return r.i8()
            if sz == 2:
                return r.i16()
            if sz == 4:
                return r.i32()
            raise ValueError(f"Unexpected enum size {sz} for {type_name}")

        if kind == "pointer":
            if not r.u8():
                return None
            ptype = tinfo["type"]
            if ptype == "UUIDRef":
                uuid_bytes = r.read(16)
                return UUIDRef(gguuid_to_str(uuid_bytes))
            if link_iter is None:
                return None
            link_group, link_index = next(link_iter)
            if ptype == "StreamingRef":
                if link_group is not None:
                    return StreamingRef(ObjectId(link_group, link_index))
                return None
            if link_group is not None:
                if sub_groups is None or link_group >= len(sub_groups):
                    return ObjectRef(ptype, ObjectId(-1, link_index), None)
                sg_id, sg_objs = sub_groups[link_group]
                target = sg_objs[link_index] if link_index < len(sg_objs) else None
                return ObjectRef(ptype, ObjectId(sg_id, link_index), target)
            target = cur_objects[link_index] if (cur_objects is not None and link_index < len(cur_objects)) else None
            return ObjectRef(ptype, ObjectId(cur_group_id, link_index), target)

        if kind == "container":
            ctype = tinfo["type"]
            itype = tinfo["item_type"]
            count = r.u32()
            if count == 0:
                if self.schema.raw[itype]["kind"] == "atom" and self.schema.atom_base(itype) in ("uint8", "int8"):
                    return b""
                return []
            if ctype in ("HashMap", "HashSet"):
                out = []
                for _ in range(count):
                    r.skip(4)  # key/value hash
                    out.append(
                        self.read_value(itype, r, link_iter, loc_iter, cur_group_id, cur_objects, sub_groups)
                    )
                return out

            # Fast paths for primitive atom arrays
            it_info = self.schema.raw[itype]
            if it_info["kind"] == "atom":
                ibase = self.schema.atom_base(itype)
                if ibase in ("uint8", "int8"):
                    return r.read(count)
                fast = _FAST_ATOM_STRUCT.get(ibase)
                if fast is not None:
                    fmt, sz = fast
                    nbytes = count * sz
                    vals = struct.unpack_from(f"<{count}{fmt}", r.data, r.pos)
                    r.pos += nbytes
                    return vals
            return [
                self.read_value(itype, r, link_iter, loc_iter, cur_group_id, cur_objects, sub_groups)
                for _ in range(count)
            ]

        if kind == "compound":
            if type_name == "GGUUID":
                return gguuid_to_str(r.read(16))
            return self.read_compound(type_name, r, link_iter, loc_iter, cur_group_id, cur_objects, sub_groups)

        raise ValueError(f"Unsupported RTTI kind {kind!r} for {type_name!r}")

    def read_compound(
        self,
        type_name: str,
        r: BinaryReader,
        link_iter: Iterator[tuple[int | None, int]] | None = None,
        loc_iter: Iterator[tuple[int, int]] | None = None,
        cur_group_id: int = -1,
        cur_objects: list[dict[str, Any]] | None = None,
        sub_groups: list[tuple[int, list[dict[str, Any]]]] | None = None,
        target: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        obj = target if target is not None else {"_type": type_name}
        attrs, mrb_owner = self._plan(type_name)
        for aname, atype in attrs:
            obj[aname] = self.read_value(atype, r, link_iter, loc_iter, cur_group_id, cur_objects, sub_groups)

        if type_name == "StreamingDataSource" and loc_iter is not None:
            # Channel is uint8 (255 = Invalid / -1 signed)
            if obj.get("Channel", 255) != 255 and obj.get("Length", 0) > 0:
                file_idx, file_offset = next(loc_iter)
                obj["_file_index"] = file_idx
                obj["_file_offset"] = file_offset

        if mrb_owner is not None:
            read_extra_binary(mrb_owner, r, obj)

        return obj
