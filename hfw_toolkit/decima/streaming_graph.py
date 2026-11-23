"""HFW PC StreamingGraph, StreamingObjectReader, and high-level HFWGame facade."""

from __future__ import annotations

import struct
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterator

from hfw_toolkit.config import ToolkitConfig
from hfw_toolkit.decima.dsar import StreamingStorage
from hfw_toolkit.decima.reader import BinaryReader, ObjectId, ObjectRef, StreamingRef, TypeReader
from hfw_toolkit.decima.rtti import RTTISchema


@dataclass(frozen=True, slots=True)
class Span:
    file_index: int
    offset: int
    length: int


@dataclass(frozen=True, slots=True)
class GroupMeta:
    group_id: int
    num_objects: int
    group_size: int
    subgroup_start: int
    subgroup_count: int
    root_start: int
    root_count: int
    span_start: int
    span_count: int
    type_start: int
    type_count: int
    link_start: int
    link_size: int
    locator_start: int
    locator_count: int


class StreamingGraph:
    """In-memory index of HFW's `streaming_graph.core` and `streaming_links.stream`."""

    def __init__(self, schema: RTTISchema, storage: StreamingStorage, config: ToolkitConfig) -> None:
        self.schema = schema
        self.storage = storage
        self.config = config

        raw_graph = config.streaming_graph_path.read_bytes()
        reader = TypeReader(schema)
        sg = reader.read_standalone_object(BinaryReader(raw_graph))

        self.files: list[str] = list(sg["Files"])
        self.type_hashes: tuple[int, ...] = tuple(sg["TypeHashes"])
        self.type_names_by_hash: list[str] = [
            schema.hash_to_name.get(h, f"UnknownType_{h:016x}") for h in self.type_hashes
        ]

        # Decode packed TypeTableData (u32 comp, stride, count, count2, unk10 + u16[count])
        tt = BinaryReader(sg["TypeTableData"])
        comp, stride, count, count2, unk10 = struct.unpack("<5I", tt.read(20))
        if comp != 0 or stride != 2 or count != count2 or unk10 != 1:
            raise ValueError(f"Unexpected TypeTableData header: {(comp, stride, count, count2, unk10)}")
        self.type_table_indices: tuple[int, ...] = struct.unpack_from(f"<{count}H", tt.data, tt.pos)

        self.subgroups_flat: tuple[int, ...] = tuple(sg["SubGroups"])
        self.root_indices_flat: tuple[int, ...] = tuple(sg["RootIndices"])
        self.root_uuids_flat: list[str] = list(sg["RootUUIDs"])

        self.spans: list[Span] = [
            Span(
                file_index=sp["FileIndexAndIsPatch"] & 0x7FFFFFFF,
                offset=sp["Offset"],
                length=sp["Length"],
            )
            for sp in sg["SpanTable"]
        ]
        # Each locator packs (file_index in low 24 bits, file_offset in upper 40 bits)
        self.locators: list[tuple[int, int]] = [
            (loc["Data"] & 0xFFFFFF, loc["Data"] >> 24) for loc in sg["LocatorTable"]
        ]

        self.groups: dict[int, GroupMeta] = {}
        self.group_list: list[GroupMeta] = []
        for g in sg["Groups"]:
            gm = GroupMeta(
                group_id=g["GroupID"],
                num_objects=g["NumObjects"],
                group_size=g["GroupSize"],
                subgroup_start=g["SubGroupStart"],
                subgroup_count=g["SubGroupCount"],
                root_start=g["RootStart"],
                root_count=g["RootCount"],
                span_start=g["SpanStart"],
                span_count=g["SpanCount"],
                type_start=g["TypeStart"],
                type_count=g["TypeCount"],
                link_start=g["LinkStart"],
                link_size=g["LinkSize"],
                locator_start=g["LocatorStart"],
                locator_count=g["LocatorCount"],
            )
            self.groups[gm.group_id] = gm
            self.group_list.append(gm)

        # Build reverse super-group map
        self.supergroups: dict[int, list[int]] = {}
        for gm in self.group_list:
            for sg_id in self.group_subgroups(gm.group_id):
                self.supergroups.setdefault(sg_id, []).append(gm.group_id)

        # Load streaming_links.stream
        link_file = self.files[int(sg["LinkTableID"])]
        self.link_table: bytes = storage.read(link_file, 0, int(sg["LinkTableSize"]))

    def group(self, group_id: int) -> GroupMeta:
        return self.groups[group_id]

    def group_subgroups(self, group_id: int) -> tuple[int, ...]:
        g = self.groups[group_id]
        return self.subgroups_flat[g.subgroup_start : g.subgroup_start + g.subgroup_count]

    def group_types(self, group_id: int) -> list[str]:
        g = self.groups[group_id]
        names = self.type_names_by_hash
        idxs = self.type_table_indices
        return [names[idxs[g.type_start + i]] for i in range(g.type_count)]

    def group_spans(self, group_id: int) -> list[Span]:
        g = self.groups[group_id]
        return self.spans[g.span_start : g.span_start + g.span_count]

    def group_locators(self, group_id: int) -> list[tuple[int, int]]:
        g = self.groups[group_id]
        return self.locators[g.locator_start : g.locator_start + g.locator_count]

    def group_roots(self, group_id: int) -> tuple[int, ...]:
        g = self.groups[group_id]
        return self.root_indices_flat[g.root_start : g.root_start + g.root_count]

    def iter_links(self, start_pos: int) -> Iterator[tuple[int | None, int]]:
        buf = self.link_table
        i = start_pos
        n = len(buf)
        while i < n:
            first = buf[i]
            i += 1
            has_group = (first & 0x40) != 0
            # Inline varint decode
            tmp = first & 0xBF
            val = tmp & 0x7F
            while tmp & 0x80:
                tmp = buf[i]
                i += 1
                val = (val << 7) | (tmp & 0x7F)
            if has_group:
                link_group: int | None = val
                tmp = buf[i]
                i += 1
                idx_val = tmp & 0x7F
                while tmp & 0x80:
                    tmp = buf[i]
                    i += 1
                    idx_val = (idx_val << 7) | (tmp & 0x7F)
                yield (link_group, idx_val)
            else:
                yield (None, val)


class HFWGame:
    """Main entry point for reading objects, groups, and streaming binary data from HFW PC."""

    def __init__(self, config: ToolkitConfig | None = None, group_cache_size: int = 2048) -> None:
        self.config = config or ToolkitConfig.load()
        self.schema = RTTISchema.load_for_game(self.config)
        self.storage = StreamingStorage(self.config)
        self.graph = StreamingGraph(self.schema, self.storage, self.config)
        self._reader = TypeReader(self.schema)
        self._group_cache: OrderedDict[tuple[int, bool], list[dict[str, Any]]] = OrderedDict()
        self._local_group_cache: OrderedDict[int, list[dict[str, Any]]] = OrderedDict()
        self._cache_limit = group_cache_size
        self._lock = threading.RLock()

    def read_group_local(self, group_id: int, cache: bool = True) -> list[dict[str, Any]]:
        """Deserialize all objects in `group_id` resolving intra-group links and recording exact
        external `ObjectId` references WITHOUT loading any subgroup archives.

        Much faster than `read_group(gid, read_subgroups=True)` for audio/catalog indexing because
        groups referencing huge common subgroups (Group 1/2/3) don't trigger recursive reads.
        """
        with self._lock:
            rec_cached = self._group_cache.get((group_id, True))
            if rec_cached is not None:
                self._group_cache.move_to_end((group_id, True))
                return rec_cached
            if cache:
                loc_cached = self._local_group_cache.get(group_id)
                if loc_cached is not None:
                    self._local_group_cache.move_to_end(group_id)
                    return loc_cached

            obj_types = self.graph.group_types(group_id)
            objects: list[dict[str, Any]] = [
                {"_type": tname, "_id": ObjectId(group_id, idx)} for idx, tname in enumerate(obj_types)
            ]
            sub_shells: list[tuple[int, list[dict[str, Any]]]] = [
                (sg_id, []) for sg_id in self.graph.group_subgroups(group_id)
            ]
            gmeta = self.graph.group(group_id)
            link_iter = self.graph.iter_links(gmeta.link_start)
            loc_iter = iter(self.graph.group_locators(group_id))

            obj_idx = 0
            for span in self.graph.group_spans(group_id):
                fname = self.graph.files[span.file_index]
                span_bytes = self.storage.read(fname, span.offset, span.length)
                br = BinaryReader(span_bytes)
                while br.remaining > 0:
                    target = objects[obj_idx]
                    tname = obj_types[obj_idx]
                    obj_idx += 1
                    self._reader.read_compound(
                        tname,
                        br,
                        link_iter=link_iter,
                        loc_iter=loc_iter,
                        cur_group_id=group_id,
                        cur_objects=objects,
                        sub_groups=sub_shells,
                        target=target,
                    )

            if obj_idx != len(objects):
                raise ValueError(
                    f"Group {group_id} object count mismatch: read {obj_idx} vs expected {len(objects)}"
                )

            if cache:
                self._local_group_cache[group_id] = objects
                if len(self._local_group_cache) > 512:
                    self._local_group_cache.popitem(last=False)
            return objects

    def read_group(self, group_id: int, read_subgroups: bool = True) -> list[dict[str, Any]]:
        """Deserialize all objects in `group_id`. When `read_subgroups=True`, resolves all Ref/WeakPtr/cptr links."""
        with self._lock:
            return self._read_group_impl(group_id, read_subgroups, set())

    def _read_group_impl(
        self, group_id: int, read_subgroups: bool, loading: set[int]
    ) -> list[dict[str, Any]]:
        key = (group_id, read_subgroups)
        cached = self._group_cache.get(key)
        if cached is not None:
            self._group_cache.move_to_end(key)
            return cached
        # If flat read requested and recursive read already cached, reuse it!
        if not read_subgroups:
            rec_cached = self._group_cache.get((group_id, True))
            if rec_cached is not None:
                self._group_cache.move_to_end((group_id, True))
                return rec_cached

        if group_id in loading:
            raise RecursionError(f"Cyclic subgroup dependency involving group {group_id}")
        loading.add(group_id)
        try:
            sub_results: list[tuple[int, list[dict[str, Any]]]] = []
            if read_subgroups:
                for sg_id in self.graph.group_subgroups(group_id):
                    sub_results.append((sg_id, self._read_group_impl(sg_id, True, loading)))

            obj_types = self.graph.group_types(group_id)
            # Pre-allocate object shells so intra-group forward references resolve cleanly!
            objects: list[dict[str, Any]] = [
                {"_type": tname, "_id": ObjectId(group_id, idx)} for idx, tname in enumerate(obj_types)
            ]

            gmeta = self.graph.group(group_id)
            link_iter = self.graph.iter_links(gmeta.link_start) if read_subgroups else None
            loc_iter = iter(self.graph.group_locators(group_id))

            obj_idx = 0
            for span in self.graph.group_spans(group_id):
                fname = self.graph.files[span.file_index]
                span_bytes = self.storage.read(fname, span.offset, span.length)
                br = BinaryReader(span_bytes)
                while br.remaining > 0:
                    target = objects[obj_idx]
                    tname = obj_types[obj_idx]
                    obj_idx += 1
                    self._reader.read_compound(
                        tname,
                        br,
                        link_iter=link_iter,
                        loc_iter=loc_iter,
                        cur_group_id=group_id,
                        cur_objects=objects,
                        sub_groups=sub_results,
                        target=target,
                    )

            if obj_idx != len(objects):
                raise ValueError(
                    f"Group {group_id} object count mismatch: read {obj_idx} vs expected {len(objects)}"
                )

            self._group_cache[key] = objects
            if len(self._group_cache) > self._cache_limit:
                self._group_cache.popitem(last=False)
            return objects
        finally:
            loading.discard(group_id)

    def read_object(self, oid: ObjectId, read_subgroups: bool = True) -> dict[str, Any]:
        objs = self.read_group(oid.group_id, read_subgroups=read_subgroups)
        return objs[oid.object_index]

    def resolve_ref(self, ref: Any) -> dict[str, Any] | None:
        """Resolve an `ObjectRef` or `StreamingRef` (loading target group on demand for `StreamingRef`)."""
        if ref is None:
            return None
        if isinstance(ref, ObjectRef):
            return ref.target
        if isinstance(ref, StreamingRef):
            return self.read_object(ref.target_id, read_subgroups=True)
        if isinstance(ref, dict) and "_type" in ref:
            return ref
        return None

    def read_data_source(self, ds: dict[str, Any], offset: int | None = None, length: int | None = None) -> bytes:
        """Read streamed binary bytes described by a `StreamingDataSource` compound."""
        if not ds:
            return b""
        if ds.get("Channel", 255) == 255 or ds.get("Length", 0) <= 0:
            return b""
        file_idx = ds.get("_file_index")
        file_off = ds.get("_file_offset")
        if file_idx is None or file_off is None:
            return b""
        rel_off = ds["Offset"] if offset is None else offset
        read_len = ds["Length"] if length is None else length
        if read_len <= 0:
            return b""
        fname = self.graph.files[file_idx]
        return self.storage.read(fname, file_off + rel_off, read_len)

    def close(self) -> None:
        self._group_cache.clear()
        self._local_group_cache.clear()
        self.storage.close()
