"""Parser for Decima `ShaderResource` (`HwShader`, `HwShaderProgram`, and trailing SRT binding table)."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any

from hfw_toolkit.extract.materials import TechniqueInfo

EPROGRAM_TYPE = {
    0: "VertexProgram",
    1: "GeometryProgram",
    2: "VertexProgram",  # Verify dynamically from DXIL header/debug name too
    3: "PixelProgram",
    4: "ComputeProgram",
}


@dataclass(slots=True)
class HwShaderProgramData:
    index: int
    raw_type: int
    stage: str
    shader_model_raw: int
    blob: bytes
    dxbc_chunks: list[str] = field(default_factory=list)


@dataclass(slots=True)
class HwShaderData:
    object_id: str
    shader_hash: str
    program_type_mask: int
    programs: list[HwShaderProgramData] = field(default_factory=list)
    root_signature_blob: bytes = b""
    srt_hashes: list[int] = field(default_factory=list)

    def get_program(self, stage: str) -> HwShaderProgramData | None:
        for p in self.programs:
            if p.stage == stage:
                return p
        return None


def _inspect_dxbc_chunks(blob: bytes) -> tuple[list[str], str | None]:
    if len(blob) < 32 or blob[:4] != b"DXBC":
        return ([], None)
    chunk_count = struct.unpack_from("<I", blob, 28)[0]
    chunks: list[str] = []
    stage_hint: str | None = None
    for i in range(min(chunk_count, 32)):
        off = struct.unpack_from("<I", blob, 32 + i * 4)[0]
        if off + 8 > len(blob):
            break
        fourcc = blob[off : off + 4].decode("ascii", errors="replace")
        csz = struct.unpack_from("<I", blob, off + 4)[0]
        chunks.append(f"{fourcc}({csz})")
        if fourcc == "ILDN" and off + 12 < len(blob):
            # ILDN contains debug PDB name like "PixelProgram_....pdb" or "VertexProgram_....pdb"
            name_chunk = blob[off + 8 : min(len(blob), off + 8 + csz)]
            if b"PixelProgram" in name_chunk:
                stage_hint = "PixelProgram"
            elif b"VertexProgram" in name_chunk:
                stage_hint = "VertexProgram"
            elif b"ComputeProgram" in name_chunk:
                stage_hint = "ComputeProgram"
            elif b"GeometryProgram" in name_chunk:
                stage_hint = "GeometryProgram"
    return (chunks, stage_hint)


def parse_hw_shader(shader_obj: dict[str, Any]) -> HwShaderData:
    """Parse `ShaderResource._shader` binary payload into per-stage DXIL blobs + SRT binding table."""
    oid = str(shader_obj.get("_id", ""))
    sh = shader_obj.get("_shader") or {}
    sh_hash = sh.get("hash", "")
    data: bytes = sh.get("data", b"")
    if len(data) < 24:
        return HwShaderData(object_id=oid, shader_hash=sh_hash, program_type_mask=0)

    pos = 0
    _u0, _u4, prog_mask, _u0c, _u10, num_progs = struct.unpack_from("<6I", data, pos)
    pos += 24

    programs: list[HwShaderProgramData] = []
    for i in range(num_progs):
        if pos + 56 > len(data):
            break
        hdr = struct.unpack_from("<14I", data, pos)
        pos += 56
        raw_type = hdr[5]
        smodel_raw = hdr[7]
        blob_len = hdr[13]
        blob = data[pos : pos + blob_len]
        pos += blob_len
        chunks, stage_hint = _inspect_dxbc_chunks(blob)
        stage = stage_hint or EPROGRAM_TYPE.get(raw_type, f"ProgramType{raw_type}")
        programs.append(
            HwShaderProgramData(
                index=i,
                raw_type=raw_type,
                stage=stage,
                shader_model_raw=smodel_raw,
                blob=blob,
                dxbc_chunks=chunks,
            )
        )

    rs_blob = b""
    if pos + 4 <= len(data):
        rs_len = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        rs_blob = data[pos : pos + rs_len]

    # Extract trailing uint32 hash table (Decima places BindingNameHash u32s at the end of the SRT blob)
    srt_hashes: list[int] = []
    if len(rs_blob) >= 4:
        # Store all aligned u32 values in the last 1024 bytes for fast hash correlation
        tail_start = max(0, len(rs_blob) - 1024)
        tail_start -= tail_start % 4
        n_u32 = (len(rs_blob) - tail_start) // 4
        srt_hashes = list(struct.unpack_from(f"<{n_u32}I", rs_blob, tail_start))

    return HwShaderData(
        object_id=oid,
        shader_hash=sh_hash,
        program_type_mask=prog_mask,
        programs=programs,
        root_signature_blob=rs_blob,
        srt_hashes=srt_hashes,
    )


def correlate_bindings_to_registers(
    hw_shader: HwShaderData,
    technique: TechniqueInfo,
    pixel_tex_registers: list[tuple[int, int]],  # [(space, reg), ...] from DXIL
) -> dict[str, dict[str, Any]]:
    """Correlate `RenderTechnique` TextureBindings & VariableBindings to DXIL registers using `HwShader` SRT order.

    Returns mapping:
      `"tex:<space>:<reg>" -> {binding_index, name, role, resource_id, target_channels, source_paths, hash}`
      `"var:<index>"       -> {binding_index, hash, inferred_name, type, float4}`
    """
    result: dict[str, dict[str, Any]] = {}

    # Collect material-space (space 5) texture registers in register order
    mat_regs = sorted([r for (sp, r) in pixel_tex_registers if sp == 5])
    rs = hw_shader.root_signature_blob

    # Find each TextureBinding's byte offset in `hw_shader.root_signature_blob`!
    # Decima stores texture binding hashes in exact SRT slot order at the tail of the SRT blob!
    ordered_tex: list[tuple[int, Any]] = []
    for tb in technique.texture_bindings:
        pat = struct.pack("<I", tb.binding_name_hash & 0xFFFFFFFF)
        off = rs.rfind(pat)
        ordered_tex.append((off if off != -1 else 1_000_000 + tb.index, tb))
    ordered_tex.sort(key=lambda x: (x[0], x[1].index))

    for slot_i, (srt_off, tb) in enumerate(ordered_tex):
        reg = mat_regs[slot_i] if slot_i < len(mat_regs) else slot_i
        result[f"tex:5:{reg}"] = {
            "binding_index": tb.index,
            "srt_offset": srt_off if srt_off < 1_000_000 else None,
            "hash": f"{tb.binding_name_hash:#010x}",
            "name": tb.name,
            "semantic_role": tb.semantic_role,
            "is_detail": tb.is_detail_texture,
            "resource_id": tb.resource_id,
            "resource_type": tb.resource_type,
            "target_channels": tb.target_channels,
            "source_paths": tb.source_paths,
        }

    # Same ordering for VariableBindings into cb0,space5
    ordered_var: list[tuple[int, Any]] = []
    for vb in technique.variable_bindings:
        pat = struct.pack("<I", vb.binding_name_hash & 0xFFFFFFFF)
        off = rs.rfind(pat)
        ordered_var.append((off if off != -1 else 1_000_000 + vb.index, vb))
    ordered_var.sort(key=lambda x: (x[0], x[1].index))

    for slot_i, (srt_off, vb) in enumerate(ordered_var):
        result[f"var:{slot_i}"] = {
            "binding_index": vb.index,
            "srt_offset": srt_off if srt_off < 1_000_000 else None,
            "hash": f"{vb.binding_name_hash:#010x}",
            "variable_id_hash": f"{vb.variable_id_hash:#010x}",
            "inferred_name": vb.inferred_name,
            "type": vb.variable_type,
            "float4": list(vb.float4),
        }

    return result
