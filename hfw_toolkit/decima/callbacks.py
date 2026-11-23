"""Binary deserializers for all 20 Decima `MsgReadBinary` resource types in HFW PC."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING, Any

from hfw_toolkit.decima.hashes import fnv1a_64_fold32

if TYPE_CHECKING:
    from hfw_toolkit.decima.reader import BinaryReader

# Written languages in HFW (ELanguage values 1..26)
WRITTEN_LANGUAGE_COUNT = 26

# JoltPhysics Factory type-name hash table (FNV-1a 64 XOR-folded to 32-bit)
_JOLT_TYPE_NAMES = (
    "SkeletalAnimation",
    "Skeleton",
    "CompoundShapeSettings",
    "StaticCompoundShapeSettings",
    "MutableCompoundShapeSettings",
    "TriangleShapeSettings",
    "SphereShapeSettings",
    "BoxShapeSettings",
    "CapsuleShapeSettings",
    "TaperedCapsuleShapeSettings",
    "CylinderShapeSettings",
    "ScaledShapeSettings",
    "MeshShapeSettings",
    "ConvexHullShapeSettings",
    "HeightFieldShapeSettings",
    "RotatedTranslatedShapeSettings",
    "OffsetCenterOfMassShapeSettings",
    "RagdollSettings",
    "PointConstraintSettings",
    "SixDOFConstraintSettings",
    "SliderConstraintSettings",
    "SwingTwistConstraintSettings",
    "DistanceConstraintSettings",
    "HingeConstraintSettings",
    "FixedConstraintSettings",
    "ConeConstraintSettings",
    "PathConstraintSettings",
    "VehicleConstraintSettings",
    "WheeledVehicleControllerSettings",
    "PathConstraintPath",
    "PathConstraintPathHermite",
    "MotorSettings",
    "PhysicsScene",
    "PhysicsMaterial",
    "PhysicsMaterialSimple",
    "GroupFilter",
    "GroupFilterTable",
)
_JOLT_HASH_TO_NAME: dict[int, str] = {fnv1a_64_fold32(n): n for n in _JOLT_TYPE_NAMES}

# EDataBufferFormat enum (from HFW extension definitions)
DATA_BUFFER_FORMAT_NAMES = {
    0: "Invalid",
    4: "R_FLOAT_32",
    5: "RG_FLOAT_32",
    6: "RGB_FLOAT_32",
    7: "RGBA_FLOAT_32",
    10: "R_UINT_32",
    11: "RG_UINT_32",
    12: "RGB_UINT_32",
    13: "RGBA_UINT_32",
    14: "R_INT_32",
    15: "RG_INT_32",
    16: "RGB_INT_32",
    17: "RGBA_INT_32",
    27: "RGBA_UINT_16",
    28: "RGBA_UINT_8",
    32: "Structured",
}

VERTEX_STORAGE_TYPE_NAMES = {
    0: "Undefined",
    1: "SignedShortNormalized",
    2: "Float",
    3: "HalfFloat",
    4: "UnsignedByteNormalized",
    5: "SignedShort",
    6: "X10Y10Z10W2Normalized",
    7: "UnsignedByte",
    8: "UnsignedShort",
    9: "UnsignedShortNormalized",
    10: "UNorm8sRGB",
    11: "X10Y10Z10W2UNorm",
}


def read_extra_binary(owner_type: str, r: BinaryReader, obj: dict[str, Any]) -> None:
    """Dispatch `MsgReadBinary` deserialization for `owner_type`."""
    fn = _CALLBACKS.get(owner_type)
    if fn is None:
        raise NotImplementedError(f"No MsgReadBinary callback registered for {owner_type}")
    fn(r, obj)


def _read_murmur_hash(r: BinaryReader) -> bytes:
    return r.read(16)


def _read_texture_info(r: BinaryReader) -> dict[str, Any]:
    hdr_bytes = r.read(16)
    (
        ttype,
        raw_w,
        raw_h,
        num_surfaces,
        num_mips,
        pixel_format,
        tex_coord_type,
        color_space,
        flags,
        creation_flags,
        streaming_mip_count,
        high_mip_skip_count,
    ) = struct.unpack_from("<HHHHBBBBBBBB", hdr_bytes, 0)
    tex_hash = _read_murmur_hash(r)

    total_size, embedded_size, streamed_size, streamed_mips = struct.unpack("<IIII", r.read(16))
    embedded_data = r.read(total_size - 12)
    return {
        "header": {
            "type": ttype,
            "raw_width": raw_w,
            "raw_height": raw_h,
            "width": raw_w & 0x3FFF,
            "height": raw_h & 0x3FFF,
            "num_surfaces": num_surfaces,
            "num_mips": num_mips,
            "pixel_format": pixel_format,
            "tex_coord_type": tex_coord_type,
            "color_space": color_space,
            "flags": flags,
            "creation_flags": creation_flags,
            "streaming_mip_count": streaming_mip_count,
            "high_mip_skip_count": high_mip_skip_count,
            "hash": tex_hash.hex(),
        },
        "data": {
            "total_size": total_size,
            "embedded_size": embedded_size,
            "streamed_size": streamed_size,
            "streamed_mips": streamed_mips,
            "embedded_data": embedded_data,
        },
    }


def _read_ui_texture_frames_info(r: BinaryReader) -> dict[str, Any]:
    data_len = r.u32()
    data = r.read(data_len)
    span_count = r.u32()
    spans = struct.unpack(f"<{span_count}Q", r.read(span_count * 8)) if span_count else ()
    width, height, pixel_format, frequency, size, scale_w, scale_h = struct.unpack("<IIIIIIf", r.read(28))
    # FSize is two floats (Width, Height) -> wait: check total 4*5 + 8 = 28 bytes?
    # Wait: width(4)+height(4)+pixelFormat(4)+frequency(4)+size(4) = 20 bytes + FSize(Width:f32, Height:f32 = 8 bytes) = 28 bytes!
    # Unpack cleanly:
    r.pos -= 28
    width, height, pixel_format, frequency, size, scale_w, scale_h = struct.unpack("<IIIIIff", r.read(28))
    return {
        "data": data,
        "spans": spans,
        "width": width,
        "height": height,
        "pixel_format": pixel_format,
        "frequency": frequency & 0xFF,
        "size": size,
        "scale": (scale_w, scale_h),
    }


def _cb_texture(r: BinaryReader, obj: dict[str, Any]) -> None:
    obj["_texture"] = _read_texture_info(r)


def _cb_texture_list(r: BinaryReader, obj: dict[str, Any]) -> None:
    count = r.u32()
    entries = []
    for _ in range(count):
        s_off, s_len = struct.unpack("<II", r.read(8))
        tex = _read_texture_info(r)
        entries.append({"streaming_offset": s_off, "streaming_length": s_len, "texture": tex})
    obj["_texture_list"] = entries


def _cb_ui_texture(r: BinaryReader, obj: dict[str, Any]) -> None:
    animated = bool(r.u8())
    small_size, large_size = struct.unpack("<II", r.read(8))
    info: dict[str, Any] = {"animated": animated}
    if small_size > 0:
        info["small"] = _read_ui_texture_frames_info(r) if animated else _read_texture_info(r)
    if large_size > 0:
        info["large"] = _read_ui_texture_frames_info(r) if animated else _read_texture_info(r)
    obj["_ui_texture"] = info


def _cb_ui_texture_frames(r: BinaryReader, obj: dict[str, Any]) -> None:
    obj["_ui_texture_frames"] = _read_ui_texture_frames_info(r)


def _cb_world_map_super_tile(r: BinaryReader, obj: dict[str, Any]) -> None:
    small_sz, large_sz, mask = struct.unpack("<III", r.read(12))
    small = []
    large = []
    if small_sz > 0:
        for i in range(4):
            if mask & (1 << i):
                small.append(_read_texture_info(r))
    if large_sz > 0:
        for i in range(4):
            if mask & (1 << i):
                large.append(_read_texture_info(r))
    obj["_super_tile"] = {"mask": mask, "small": small, "large": large}


def _cb_shader_resource(r: BinaryReader, obj: dict[str, Any]) -> None:
    size = r.u32()
    sh_hash = _read_murmur_hash(r)
    data = r.read(size)
    obj["_shader"] = {"hash": sh_hash.hex(), "data": data}


def _cb_vertex_array_resource(r: BinaryReader, obj: dict[str, Any]) -> None:
    num_vertices, num_streams, is_streaming = struct.unpack("<IIB", r.read(9))
    streaming = bool(is_streaming)
    streams = []
    for _ in range(num_streams):
        flags, stride, num_elements = struct.unpack("<III", r.read(12))
        elements = []
        for _ in range(num_elements):
            offset, storage_type, slots_used, element = struct.unpack("<BBBB", r.read(4))
            elements.append(
                {
                    "offset": offset,
                    "storage_type": storage_type,
                    "storage_type_name": VERTEX_STORAGE_TYPE_NAMES.get(storage_type, str(storage_type)),
                    "slots_used": slots_used,
                    "element": element,
                }
            )
        stream_hash = _read_murmur_hash(r)
        data = None if streaming else r.read(stride * num_vertices)
        streams.append(
            {
                "flags": flags,
                "stride": stride,
                "elements": elements,
                "hash": stream_hash.hex(),
                "data": data,
            }
        )
    obj["_vertex_array"] = {
        "count": num_vertices,
        "is_streaming": streaming,
        "streams": streams,
    }


def _cb_index_array_resource(r: BinaryReader, obj: dict[str, Any]) -> None:
    count, flags, fmt, is_streaming = struct.unpack("<IIII", r.read(16))
    checksum = _read_murmur_hash(r)
    streaming = bool(is_streaming)
    stride = 2 if fmt == 0 else 4
    data = None if streaming else r.read(count * stride)
    obj["_index_array"] = {
        "count": count,
        "flags": flags,
        "format": fmt,
        "stride": stride,
        "is_streaming": streaming,
        "checksum": checksum.hex(),
        "data": data,
    }


def _cb_data_buffer_resource(r: BinaryReader, obj: dict[str, Any]) -> None:
    count = r.u32()
    if count == 0:
        obj["_data_buffer"] = {"count": 0, "is_streaming": False, "flags": 0, "format": 0, "stride": 0, "data": b""}
        return
    is_streaming, flags, fmt, stride = struct.unpack("<IIII", r.read(16))
    streaming = bool(is_streaming)
    data = None if streaming else r.read(stride * count)
    obj["_data_buffer"] = {
        "count": count,
        "is_streaming": streaming,
        "flags": flags,
        "format": fmt,
        "format_name": DATA_BUFFER_FORMAT_NAMES.get(fmt, str(fmt)),
        "stride": stride,
        "data": data,
    }


def _cb_localized_text_resource(r: BinaryReader, obj: dict[str, Any]) -> None:
    texts = []
    for _ in range(WRITTEN_LANGUAGE_COUNT):
        n = r.u16()
        texts.append(r.read(n).decode("utf-8", errors="replace"))
    obj["_texts"] = texts


def _cb_pose(r: BinaryReader, obj: dict[str, Any]) -> None:
    if r.u8():
        c1 = r.u32()
        m1 = [struct.unpack("<12f", r.read(48)) for _ in range(c1)]
        m2 = [struct.unpack("<12f", r.read(48)) for _ in range(c1)]
        c2 = r.u32()
        ids = struct.unpack(f"<{c2}I", r.read(c2 * 4)) if c2 else ()
        obj["_pose"] = {"mats1": m1, "mats2": m2, "ids": ids}


def _cb_debug_mouse_cursor_ps4(r: BinaryReader, obj: dict[str, Any]) -> None:
    stride = r.u32()
    data_len = r.u32()
    obj["_cursor"] = {"stride": stride, "data": r.read(data_len)}


def _cb_morpheme_animation(r: BinaryReader, obj: dict[str, Any]) -> None:
    n = r.u32()
    data = r.read(n)
    h = _read_murmur_hash(r)
    obj["_morpheme_anim"] = {"size": n, "hash": h.hex(), "data": data}


def _cb_morpheme_asset(r: BinaryReader, obj: dict[str, Any]) -> None:
    n = r.u32()
    data = r.read(n)
    u1, u2, u3 = struct.unpack("<IIQ", r.read(16))
    obj["_morpheme_asset"] = {"size": n, "unk01": u1, "unk02": u2, "unk03": u3}


def _cb_morpheme_network_def(r: BinaryReader, obj: dict[str, Any]) -> None:
    pass


def _cb_morpheme_network_preinit(r: BinaryReader, obj: dict[str, Any]) -> None:
    count, unk04 = struct.unpack("<II", r.read(8))
    if unk04 != 4:
        raise ValueError(f"MorphemeNetworkInstancePreInitializedData expected unk04=4, got {unk04}")
    if count > 0:
        r.skip(count)
        n2 = r.u32()
        r.skip(n2 * 4)


def _cb_static_tile(r: BinaryReader, obj: dict[str, Any]) -> None:
    r.skip(r.u32() * 20)
    r.skip(r.u32() * 64)
    r.skip(r.u32() * 12)
    r.skip(r.u32() * 16)
    r.skip(r.u32() * 48)


# --- JoltPhysics binary state skippers (PhysicsShapeResource & PhysicsRagdollResource) ---

def _jolt_skip_shape_body(r: BinaryReader) -> None:
    subtype = r.u8()
    _user_data = r.u32()
    if subtype == 0:  # Sphere (ConvexShape)
        r.skip(4 + 4)  # density + radius
    elif subtype == 1:  # Box (ConvexShape)
        r.skip(4 + 16 + 4)  # density + halfExtent(alignedVec3) + convexRadius
    elif subtype == 3:  # Capsule (ConvexShape)
        r.skip(4 + 4 + 4)  # density + radius + halfHeight
    elif subtype == 4:  # TaperedCapsule (ConvexShape)
        r.skip(4 + 16 + 7 * 4)  # density + com + 7 floats
    elif subtype == 5:  # Cylinder (ConvexShape)
        r.skip(4 + 12)  # density + halfHeight + radius + convexRadius
    elif subtype == 6:  # ConvexHull (ConvexShape)
        r.skip(4 + 16 + 64 + 32)  # density + com + inertia(64) + aabox(32)
        n_pts = r.u64()
        r.skip(n_pts * 32)  # alignedVec3(16) + count(4) + 3 ints(12)
        n_faces = r.u64()
        r.skip(n_faces * 4)
        n_planes = r.u64()
        r.skip(n_planes * 16)
        n_vidx = r.u64()
        r.skip(n_vidx)
        r.skip(12)  # convexRadius, volume, innerRadius
    elif subtype == 7:  # StaticCompound (CompoundShape)
        r.skip(16 + 32 + 4)  # com + aabox + innerRadius
        n_sub = r.u64()
        r.skip(n_sub * 28)  # userData(4) + pos(12) + rot(12)
        n_nodes = r.u64()
        r.skip(n_nodes * 64)  # 6 * 4 shorts (48) + 4 ints (16)
    elif subtype == 9:  # RotatedTranslated (DecoratedShape)
        r.skip(16 + 16)  # com + quat
    elif subtype == 10:  # Scaled (DecoratedShape)
        r.skip(16)  # scale
    elif subtype == 11:  # OffsetCenterOfMass (DecoratedShape)
        r.skip(16)  # offset
    elif subtype == 12:  # Mesh
        n_tree = r.u64()
        r.skip(n_tree)
    elif subtype == 13:  # HeightField
        r.skip(16 + 16 + 4 + 4 + 1 + 2 + 2)
        n_rb = r.u64()
        r.skip(n_rb * 16)
        r.skip(r.u64())  # heightSamples
        r.skip(r.u64())  # activeEdges
        r.skip(r.u64())  # materialIndices
        r.skip(4)  # numBitsPerMaterialIndex
    else:
        raise ValueError(f"Unsupported Jolt ShapeSubType: {subtype}")


def _jolt_restore_shape_tree(r: BinaryReader, shape_map: list[bool]) -> None:
    shape_id = r.u32()
    if shape_id == 0xFFFFFFFF:
        return
    if shape_id < len(shape_map):
        return
    _jolt_skip_shape_body(r)
    while len(shape_map) < shape_id:
        shape_map.append(False)
    shape_map.append(True)
    n_children = r.u32()
    for _ in range(n_children):
        _jolt_restore_shape_tree(r, shape_map)
    n_materials = r.u32()
    r.skip(n_materials * 4)


def _cb_physics_shape_resource(r: BinaryReader, obj: dict[str, Any]) -> None:
    shape_map = [False]  # 1-indexed (index 0 reserved for null)
    _jolt_restore_shape_tree(r, shape_map)


def _jolt_skip_constraint(r: BinaryReader) -> None:
    h = r.u32()
    name = _JOLT_HASH_TO_NAME.get(h)
    if name is None:
        raise ValueError(f"Unknown Jolt constraint type hash: {h:#010x}")
    r.skip(4)  # drawConstraintSize
    if name == "HingeConstraintSettings":
        r.skip(6 * 16 + 3 * 4 + 6 * 4)  # 6 alignedVec3 + 3 floats + MotorSettings(6 floats)
    elif name == "PointConstraintSettings":
        r.skip(16)
    elif name == "SliderConstraintSettings":
        r.skip(16 + 3 * 4 + 6 * 4)
    elif name == "SwingTwistConstraintSettings":
        r.skip(6 * 16 + 5 * 4 + 2 * 6 * 4)
    else:
        raise ValueError(f"Unsupported Jolt constraint: {name}")


def _cb_physics_ragdoll_resource(r: BinaryReader, obj: dict[str, Any]) -> None:
    # Jolt Skeleton
    n_joints = r.u32()
    for _ in range(n_joints):
        r.skip(r.u64())  # name
        r.skip(4)  # parentJointIndex
        r.skip(r.u64())  # parentName
    shape_map: list[bool] = []
    group_filter_count = 0
    n_parts = r.u32()
    for _ in range(n_parts):
        # BodyCreationSettings.restoreBinaryState:
        # pos(16) + rot(16) + collisionGroup(8) + objectLayer(2) + motionType(1) + allowDyn(1)
        # + motionQual(1) + allowSleep(1) + 7 floats(28) + overrideMassProps(1) + inertiaMult(4)
        # + MassProperties(mass:4 + mat44:64) = 147 bytes
        r.skip(16 + 16 + 8 + 2 + 1 + 1 + 1 + 1 + 28 + 1 + 4 + 4 + 64)
        # Shape (optional)
        _jolt_restore_shape_tree(r, shape_map)
        # GroupFilter
        gf_id = r.u32()
        if gf_id != 0xFFFFFFFF and gf_id >= group_filter_count:
            gf_hash = r.u32()
            gf_name = _JOLT_HASH_TO_NAME.get(gf_hash)
            if gf_name != "GroupFilterTable":
                raise ValueError(f"Unsupported Jolt GroupFilter: {gf_name} ({gf_hash:#010x})")
            r.skip(4)  # numSubGroups
            r.skip(r.u64())  # table bytes
            group_filter_count += 1
        # Constraint (optional)
        if r.u8():
            _jolt_skip_constraint(r)


# --- RigLogic (FacialAnimationDNAResource) big-endian skipper ---

def _cb_facial_animation_dna(r: BinaryReader, obj: dict[str, Any]) -> None:
    def be_u32() -> int:
        v = struct.unpack_from(">I", r.data, r.pos)[0]
        r.pos += 4
        return v

    def skip_cond_table() -> None:
        r.skip(be_u32() * 2)  # inputIndices
        r.skip(be_u32() * 2)  # outputIndices
        r.skip(be_u32() * 4)  # fromValues
        r.skip(be_u32() * 4)  # toValues
        r.skip(be_u32() * 4)  # slopeValues
        r.skip(be_u32() * 4)  # cutValues
        r.skip(4)  # inputCount(2) + outputCount(2)

    r.skip(4)  # Configuration (calcType)
    r.skip(14)  # RigMetrics (7 shorts)
    # Controls
    skip_cond_table()
    # PSDMatrix
    r.skip(2)  # distinctPSDs
    r.skip(be_u32() * 2)  # rowIndices
    r.skip(be_u32() * 2)  # columnIndices
    r.skip(be_u32() * 4)  # values
    # Joints -> JointStorage
    r.skip(be_u32() * 2)  # values
    r.skip(be_u32() * 2)  # inputIndices
    r.skip(be_u32() * 2)  # outputIndices
    r.skip(be_u32() * 12)  # lodRegions
    r.skip(be_u32() * 32)  # jointGroups
    r.skip(be_u32() * 4)  # neutralValues
    n_var_attr = be_u32()
    for _ in range(n_var_attr):
        r.skip(be_u32() * 2)
    r.skip(2)  # jointGroupCount
    # BlendShapes
    r.skip(be_u32() * 2)
    r.skip(be_u32() * 2)
    r.skip(be_u32() * 2)
    # AnimatedMaps
    r.skip(be_u32() * 2)
    skip_cond_table()


_CALLBACKS = {
    "Texture": _cb_texture,
    "TextureList": _cb_texture_list,
    "UITexture": _cb_ui_texture,
    "UITextureFrames": _cb_ui_texture_frames,
    "WorldMapSuperTile": _cb_world_map_super_tile,
    "ShaderResource": _cb_shader_resource,
    "VertexArrayResource": _cb_vertex_array_resource,
    "IndexArrayResource": _cb_index_array_resource,
    "DataBufferResource": _cb_data_buffer_resource,
    "LocalizedTextResource": _cb_localized_text_resource,
    "Pose": _cb_pose,
    "DebugMouseCursorPS4": _cb_debug_mouse_cursor_ps4,
    "MorphemeAnimation": _cb_morpheme_animation,
    "MorphemeAsset": _cb_morpheme_asset,
    "MorphemeNetworkDefResource": _cb_morpheme_network_def,
    "MorphemeNetworkInstancePreInitializedData": _cb_morpheme_network_preinit,
    "StaticTile": _cb_static_tile,
    "PhysicsShapeResource": _cb_physics_shape_resource,
    "PhysicsRagdollResource": _cb_physics_ragdoll_resource,
    "FacialAnimationDNAResource": _cb_facial_animation_dna,
}
