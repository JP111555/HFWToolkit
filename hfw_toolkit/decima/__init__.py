"""Decima Engine PC binary reader, RTTI schema, streaming graph & SQLite catalog."""

from hfw_toolkit.decima.catalog import AssetCatalog, MACHINE_CODENAMES, decode_packing_summary
from hfw_toolkit.decima.dsar import BinaryStorageFile, StreamingStorage
from hfw_toolkit.decima.hashes import decima_crc32c, decima_string_hash, decima_type_hash, murmur3_x64_128
from hfw_toolkit.decima.reader import BinaryReader, ObjectId, ObjectRef, StreamingRef, TypeReader, UUIDRef
from hfw_toolkit.decima.rtti import RTTISchema
from hfw_toolkit.decima.streaming_graph import GroupMeta, HFWGame, Span, StreamingGraph

__all__ = [
    "AssetCatalog",
    "BinaryReader",
    "BinaryStorageFile",
    "GroupMeta",
    "HFWGame",
    "MACHINE_CODENAMES",
    "ObjectId",
    "ObjectRef",
    "RTTISchema",
    "Span",
    "StreamingGraph",
    "StreamingRef",
    "StreamingStorage",
    "TypeReader",
    "UUIDRef",
    "decima_crc32c",
    "decima_string_hash",
    "decima_type_hash",
    "decode_packing_summary",
    "murmur3_x64_128",
]
