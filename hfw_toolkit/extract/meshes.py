"""Vectorized NumPy vertex/index/skinning decoder for HFW PC meshes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hfw_toolkit.decima.reader import UUIDRef
from hfw_toolkit.decima.streaming_graph import HFWGame
from hfw_toolkit.extract.skeleton import decima_mat34_to_np


@dataclass(slots=True)
class DecodedPrimitive:
    primitive_index: int
    material_id: str | None  # RenderEffectResource ObjectId ("group:index")
    positions: np.ndarray  # (N, 3) float32
    indices: np.ndarray  # (M,) uint32 (triangle list)
    normals: np.ndarray | None = None  # (N, 3) float32
    tangents: np.ndarray | None = None  # (N, 4) float32
    uvs: list[np.ndarray] = field(default_factory=list)  # list of (N, 2) float32
    colors: list[np.ndarray] = field(default_factory=list)  # list of (N, 4) float32 (0..1)
    bone_indices: np.ndarray | None = None  # (N, K) uint16 (remapped to Skeleton joint indices!)
    bone_weights: np.ndarray | None = None  # (N, K) float32
    is_shadow_caster_only: bool = False


@dataclass(slots=True)
class DecodedMesh:
    object_id: str
    mesh_type: str  # "StaticMeshResource" | "RegularSkinnedMeshResource"
    is_skinned: bool
    primitives: list[DecodedPrimitive] = field(default_factory=list)
    local_transform: np.ndarray | None = None  # Optional 4x4 transform from MultiMeshResource
    skeleton_id: str | None = None
    skeleton_joint_uuids: list[str | None] = field(default_factory=list)
    skeleton_joint_names: list[str] = field(default_factory=list)


def _align(val: int, alignment: int) -> int:
    if alignment <= 1:
        return val
    return ((val + alignment - 1) // alignment) * alignment


def _unpack_element(
    buf: bytes,
    count: int,
    stride: int,
    offset: int,
    storage_type: str,
    slots: int,
) -> np.ndarray | None:
    """Vectorized extraction of a vertex attribute from an interleaved buffer."""
    if count == 0 or slots == 0:
        return None
    raw = np.frombuffer(buf, dtype=np.uint8, count=count * stride).reshape(count, stride)

    if storage_type == "Float":
        sl = np.ascontiguousarray(raw[:, offset : offset + slots * 4])
        return sl.view(np.float32).reshape(count, slots).astype(np.float32, copy=False)

    if storage_type == "HalfFloat":
        sl = np.ascontiguousarray(raw[:, offset : offset + slots * 2])
        return sl.view(np.float16).reshape(count, slots).astype(np.float32)

    if storage_type == "UnsignedByte":
        return raw[:, offset : offset + slots].astype(np.uint16)

    if storage_type in ("UnsignedByteNormalized", "UNorm8sRGB"):
        return raw[:, offset : offset + slots].astype(np.float32) * (1.0 / 255.0)

    if storage_type == "SignedShort":
        sl = np.ascontiguousarray(raw[:, offset : offset + slots * 2])
        return sl.view(np.int16).reshape(count, slots).astype(np.int32)

    if storage_type == "SignedShortNormalized":
        sl = np.ascontiguousarray(raw[:, offset : offset + slots * 2])
        arr = sl.view(np.int16).reshape(count, slots).astype(np.float32) / 32767.0
        return np.clip(arr, -1.0, 1.0)

    if storage_type == "UnsignedShort":
        sl = np.ascontiguousarray(raw[:, offset : offset + slots * 2])
        return sl.view(np.uint16).reshape(count, slots)

    if storage_type == "UnsignedShortNormalized":
        sl = np.ascontiguousarray(raw[:, offset : offset + slots * 2])
        return sl.view(np.uint16).reshape(count, slots).astype(np.float32) * (1.0 / 65535.0)

    if storage_type in ("X10Y10Z10W2Normalized", "X10Y10Z10W2UNorm"):
        sl = np.ascontiguousarray(raw[:, offset : offset + 4])
        packed = sl.view(np.uint32).reshape(count)
        if storage_type == "X10Y10Z10W2UNorm":
            x = (packed & 0x3FF).astype(np.float32) / 1023.0
            y = ((packed >> 10) & 0x3FF).astype(np.float32) / 1023.0
            z = ((packed >> 20) & 0x3FF).astype(np.float32) / 1023.0
            w = ((packed >> 30) & 0x3).astype(np.float32) / 3.0
        else:
            # 10-bit signed two's complement
            def s10(v: np.ndarray) -> np.ndarray:
                v_i = v.astype(np.int32)
                v_i = np.where(v_i >= 512, v_i - 1024, v_i)
                return np.clip(v_i.astype(np.float32) / 511.0, -1.0, 1.0)

            def s2(v: np.ndarray) -> np.ndarray:
                v_i = v.astype(np.int32)
                v_i = np.where(v_i >= 2, v_i - 4, v_i)
                return np.clip(v_i.astype(np.float32), -1.0, 1.0)

            x = s10(packed & 0x3FF)
            y = s10((packed >> 10) & 0x3FF)
            z = s10((packed >> 20) & 0x3FF)
            w = s2((packed >> 30) & 0x3)
        full = np.stack([x, y, z, w], axis=-1)
        return full[:, :slots].astype(np.float32)

    return None


def _is_shadow_caster_only(mesh_obj: dict[str, Any]) -> bool:
    df = mesh_obj.get("DrawFlags")
    if not isinstance(df, dict):
        return False
    # Exact HFW DrawFlags bitfield (verified from Odradek DrawFlagsExtension.java & EDrawPartType RTTI):
    #   bits 0..1: CastShadows, bits 2..3: ShadowCullMode,
    #   bits 4..5: RenderType (0=Normal, 1=ShadowCasterOnly, 2=ReflectionOnly),
    #   bits 6..7: ViewLayer, bits 8..23: ShadowBiasMultiplier (fp16)
    data = df.get("Data", 0)
    render_type = (data >> 4) & 3
    return render_type == 1


def _decode_primitives_from_mesh(
    game: HFWGame,
    mesh_obj: dict[str, Any],
    is_skinned: bool,
) -> DecodedMesh:
    oid = str(mesh_obj.get("_id", ""))
    mtype = mesh_obj["_type"]
    shading_groups = [game.resolve_ref(r) for r in mesh_obj.get("ShadingGroups", ())]
    primitives_raw = [game.resolve_ref(r) for r in mesh_obj.get("Primitives", ())]
    ds = mesh_obj.get("StreamingDataSource")
    stream_buf = game.read_data_source(ds) if ds else b""
    stream_pos = 0

    # Record the skinned mesh's own Skeleton joint table (so machine_assembler can align
    # bone indices when a mesh references a different Skeleton object than SMR.Skeleton).
    # NOTE: Vertex BlendIndices ALREADY contain direct Skeleton joint indices — do NOT remap
    # through SkinnedMeshIndexedJointBindings.JointIndexList (that list only indexes IBMs)!
    mesh_skel_id: str | None = None
    mesh_skel_uuids: list[str | None] = []
    mesh_skel_names: list[str] = []
    if is_skinned:
        mesh_skel = game.resolve_ref(mesh_obj.get("Skeleton"))
        if mesh_skel is not None:
            mesh_skel_id = str(mesh_skel.get("_id", ""))
            for j in mesh_skel.get("Joints", ()):
                jid = j.get("JointID")
                mesh_skel_uuids.append(jid.uuid if isinstance(jid, UUIDRef) else None)
                mesh_skel_names.append(j.get("Name", ""))

    pos_scale = mesh_obj.get("PositionBoundsScale") if is_skinned else None
    pos_offset = mesh_obj.get("PositionBoundsOffset") if is_skinned else None
    shadow_only = _is_shadow_caster_only(mesh_obj)

    decoded_prims: list[DecodedPrimitive] = []
    for pidx, prim in enumerate(primitives_raw):
        if prim is None:
            continue
        va_obj = game.resolve_ref(prim.get("VertexArray"))
        ia_obj = game.resolve_ref(prim.get("IndexArray"))
        if not va_obj or not ia_obj:
            continue
        va = va_obj.get("_vertex_array")
        ia = ia_obj.get("_index_array")
        if not va or not ia:
            continue

        sg = shading_groups[pidx] if pidx < len(shading_groups) else None
        mat_ref = sg.get("RenderEffect") if sg else None
        mat_obj = game.resolve_ref(mat_ref) if mat_ref else None
        mat_id = str(mat_obj["_id"]) if mat_obj is not None else None

        vcount = va["count"]
        positions = None
        pos_was_quantized = False
        normals = None
        tangents = None
        uvs_by_slot: dict[int, np.ndarray] = {}
        colors: list[np.ndarray] = []
        weights_list: list[np.ndarray] = []
        bones_list: list[np.ndarray] = []

        for stream in va["streams"]:
            stride = stream["stride"]
            if va["is_streaming"]:
                stream_pos = _align(stream_pos, stride)
                nbytes = vcount * stride
                view = stream_buf[stream_pos : stream_pos + nbytes]
                stream_pos += nbytes
            else:
                view = stream["data"] or b""

            for elem in stream["elements"]:
                sem_name = game.schema.enum_name("EVertexElement", elem["element"])
                st_name = elem["storage_type_name"]
                arr = _unpack_element(view, vcount, stride, elem["offset"], st_name, elem["slots_used"])
                if arr is None:
                    continue

                if sem_name == "Pos":
                    positions = arr[:, :3].astype(np.float32, copy=False)
                    if st_name not in ("Float", "HalfFloat"):
                        pos_was_quantized = True
                elif sem_name == "Normal":
                    normals = arr[:, :3].astype(np.float32, copy=False)
                elif sem_name == "Tangent":
                    tangents = arr.astype(np.float32, copy=False)
                elif sem_name == "Color":
                    if arr.shape[1] == 3:
                        ones = np.ones((vcount, 1), dtype=np.float32)
                        arr = np.concatenate([arr.astype(np.float32), ones], axis=1)
                    colors.append(arr[:, :4].astype(np.float32, copy=False))
                elif sem_name.startswith("UV") and sem_name[2:].isdigit():
                    uv_slot = int(sem_name[2:])
                    uvs_by_slot[uv_slot] = arr[:, :2].astype(np.float32, copy=False)
                elif sem_name in ("BlendWeights", "BlendWeights2", "BlendWeights3"):
                    weights_list.append(arr.astype(np.float32, copy=False))
                elif sem_name in ("BlendIndices", "BlendIndices2", "BlendIndices3"):
                    bones_list.append(arr.astype(np.uint16, copy=False))

        # Index buffer
        icount = ia["count"]
        istride = ia["stride"]
        if ia["is_streaming"]:
            stream_pos = _align(stream_pos, istride)
            ibytes = stream_buf[stream_pos : stream_pos + icount * istride]
            stream_pos += icount * istride
        else:
            ibytes = ia["data"] or b""

        idtype = np.uint16 if ia["format"] == 0 else np.uint32
        all_indices = np.frombuffer(ibytes, dtype=idtype, count=icount)
        s_idx = max(0, int(prim.get("StartIndex", 0)))
        e_idx = int(prim.get("EndIndex", icount))
        if e_idx <= s_idx or e_idx > icount:
            e_idx = icount
        indices = all_indices[s_idx:e_idx].astype(np.uint32, copy=True)

        if positions is None or len(indices) == 0:
            continue

        # Dequantize normalized skinned positions if PositionBoundsScale is non-zero
        if pos_was_quantized and pos_scale and pos_offset:
            sx, sy, sz = pos_scale["X"], pos_scale["Y"], pos_scale["Z"]
            ox, oy, oz = pos_offset["X"], pos_offset["Y"], pos_offset["Z"]
            if abs(sx) > 1e-8 or abs(sy) > 1e-8 or abs(sz) > 1e-8:
                positions = positions * np.array([sx, sy, sz], dtype=np.float32) + np.array([ox, oy, oz], dtype=np.float32)

        # Combine multi-stream bone indices/weights (up to 12 influences)
        # Vertex BlendIndices already directly index mesh_skel.Joints; normalize weights per vertex
        # to match Decima's vertex shader (and Odradek's Accessor.toFloatsNormalized()).
        bone_indices: np.ndarray | None = None
        bone_weights: np.ndarray | None = None
        if bones_list:
            bone_indices = np.concatenate(bones_list, axis=1) if len(bones_list) > 1 else bones_list[0]
            if weights_list:
                bone_weights = np.concatenate(weights_list, axis=1) if len(weights_list) > 1 else weights_list[0]
                wsum = bone_weights.sum(axis=1, keepdims=True)
                bone_weights = np.where(wsum > 1e-6, bone_weights / np.maximum(wsum, 1e-6), 0.0).astype(
                    np.float32, copy=False
                )
            elif bone_indices.shape[1] == 1:
                bone_weights = np.ones((vcount, 1), dtype=np.float32)

        uvs = [uvs_by_slot[k] for k in sorted(uvs_by_slot)]

        decoded_prims.append(
            DecodedPrimitive(
                primitive_index=pidx,
                material_id=mat_id,
                positions=positions,
                indices=indices,
                normals=normals,
                tangents=tangents,
                uvs=uvs,
                colors=colors,
                bone_indices=bone_indices,
                bone_weights=bone_weights,
                is_shadow_caster_only=shadow_only,
            )
        )

    return DecodedMesh(
        object_id=oid,
        mesh_type=mtype,
        is_skinned=is_skinned,
        primitives=decoded_prims,
        skeleton_id=mesh_skel_id,
        skeleton_joint_uuids=mesh_skel_uuids,
        skeleton_joint_names=mesh_skel_names,
    )


def decode_mesh_hierarchy(
    game: HFWGame,
    mesh_obj: dict[str, Any] | None,
    lod_index: int = 0,
    include_shadow_only: bool = False,
) -> list[DecodedMesh]:
    """Recursively decode any `MeshResourceBase` (`LodMeshResource`, `MultiMeshResource`, `StaticMeshResource`, `RegularSkinnedMeshResource`)."""
    if mesh_obj is None:
        return []
    mtype = mesh_obj["_type"]

    if mtype == "LodMeshResource":
        parts = mesh_obj.get("RuntimeMeshes", ())
        if not parts:
            return []
        pick = min(lod_index, len(parts) - 1)
        sub = game.resolve_ref(parts[pick].get("Mesh"))
        return decode_mesh_hierarchy(game, sub, lod_index=lod_index, include_shadow_only=include_shadow_only)

    if mtype == "MultiMeshResource":
        out: list[DecodedMesh] = []
        meshes = mesh_obj.get("Meshes", ())
        transforms = mesh_obj.get("Transforms", ())
        for i, mref in enumerate(meshes):
            sub = game.resolve_ref(mref)
            sub_decoded = decode_mesh_hierarchy(game, sub, lod_index=lod_index, include_shadow_only=include_shadow_only)
            mat = decima_mat34_to_np(transforms[i]) if i < len(transforms) else None
            for dm in sub_decoded:
                if mat is not None:
                    dm.local_transform = mat if dm.local_transform is None else (mat @ dm.local_transform)
                out.append(dm)
        return out

    if mtype == "StaticMeshResource":
        dm = _decode_primitives_from_mesh(game, mesh_obj, is_skinned=False)
        if not include_shadow_only:
            dm.primitives = [p for p in dm.primitives if not p.is_shadow_caster_only]
        return [dm] if dm.primitives else []

    if mtype == "RegularSkinnedMeshResource":
        dm = _decode_primitives_from_mesh(game, mesh_obj, is_skinned=True)
        if not include_shadow_only:
            dm.primitives = [p for p in dm.primitives if not p.is_shadow_caster_only]
        return [dm] if dm.primitives else []

    return []
