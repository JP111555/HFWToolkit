"""Texture & TextureSet decoding (BC1-BC7, uncompressed, streamed mips, RGBA channel unpacking)."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import texture2ddecoder

from hfw_toolkit.decima.streaming_graph import HFWGame

# ETextureSetType enum
ETEXTURE_SET_TYPE: dict[int, str] = {
    0: "Invalid",
    1: "Color",
    2: "Alpha",
    3: "Normal",
    4: "Reflectance",
    5: "AO",
    6: "Roughness",
    7: "Height",
    8: "Mask",
    9: "Mask_Alpha",
    10: "Incandescence",
    11: "Translucency_Diffusion",
    12: "Translucency_Amount",
    13: "Misc_01",
    14: "Curvature",
    15: "Luminance",
}

ECHANNEL_IDX: dict[int, str] = {0: "R", 1: "G", 2: "B", 3: "A"}
LANE_TO_IDX: dict[str, int] = {"R": 0, "G": 1, "B": 2, "A": 3}

# EPixelFormat enum values (verified against dump/hfw.json)
# We resolve dynamically from schema when available, and also map canonical names here:
PIXEL_FORMAT_INFO: dict[str, tuple[int, int, int, int]] = {
    # name -> (block_w, block_h, bytes_per_block, dxgi_format)
    "R_UNORM_8": (1, 1, 1, 61),
    "R_UNORM_16": (1, 1, 2, 56),
    "RGBA_8888": (1, 1, 4, 28),
    "RGBA_8888_REV": (1, 1, 4, 87),
    "RGBA_FLOAT_16": (1, 1, 8, 10),
    "R_FLOAT_16": (1, 1, 2, 54),
    "R_FLOAT_32": (1, 1, 4, 41),
    "BC1": (4, 4, 8, 71),
    "BC2": (4, 4, 16, 74),
    "BC3": (4, 4, 16, 77),
    "BC4U": (4, 4, 8, 80),
    "BC4S": (4, 4, 8, 81),
    "BC5U": (4, 4, 16, 83),
    "BC5S": (4, 4, 16, 84),
    "BC6U": (4, 4, 16, 95),
    "BC6S": (4, 4, 16, 96),
    "BC7": (4, 4, 16, 98),
}


@dataclass(frozen=True, slots=True)
class PackingChannel:
    tex_type: str  # e.g. "Color", "Normal", "Roughness", "AO", "Mask"
    source_channel: str  # "R", "G", "B", "A"


@dataclass(slots=True)
class DecodedTexture:
    object_id: str
    width: int
    height: int
    mip_count: int
    pixel_format: str
    color_space: str  # "Linear" | "sRGB"
    tex_type: str  # "_2D" | "_2DArray" | "_3D" | "CubeMap"
    rgba: np.ndarray  # shape (H, W, 4), uint8
    mip0_raw: bytes  # raw mip0 compressed/uncompressed bytes (for DDS export)
    packing: dict[str, PackingChannel] = field(default_factory=dict)


def decode_packing_info(packing_info: int) -> dict[str, PackingChannel]:
    """Decode `TextureSetEntry.PackingInfo` uint32 into `{R/G/B/A -> PackingChannel}`."""
    result: dict[str, PackingChannel] = {}
    for shift, lane in ((0, "R"), (8, "G"), (16, "B"), (24, "A")):
        byte = (packing_info >> shift) & 0xFF
        ttype = byte & 0x0F
        if ttype == 0:
            continue
        tname = ETEXTURE_SET_TYPE.get(ttype, f"Type{ttype}")
        if (byte & 0x80) == 0:
            ch = ECHANNEL_IDX.get((byte >> 4) & 0x07, "R")
        else:
            ch = "R"
        result[lane] = PackingChannel(tex_type=tname, source_channel=ch)
    return result


def decode_texture_binding_packing(packed_data: int) -> tuple[str, dict[str, PackingChannel]]:
    """Decode `ShaderTextureBinding.PackedData`. Returns `('Texture', {})` or `('TextureSet', packing)`."""
    kind = packed_data & 3
    if kind == 1:
        return ("Texture", {})
    if kind == 2:
        return ("TextureSet", decode_packing_info(packed_data >> 2))
    return (f"Unknown({kind})", {})


def _surface_byte_size(w: int, h: int, bw: int, bh: int, bpb: int) -> int:
    blocks_x = max(1, (w + bw - 1) // bw)
    blocks_y = max(1, (h + bh - 1) // bh)
    return blocks_x * blocks_y * bpb


def _decode_surface_to_rgba(raw: bytes, width: int, height: int, fmt_name: str) -> np.ndarray:
    """Decode raw GPU surface bytes into `(height, width, 4)` uint8 RGBA numpy array."""
    # texture2ddecoder returns BGRA uint8 bytes for BC1/BC3/BC4/BC5/BC6/BC7
    if fmt_name == "BC1":
        bgra = texture2ddecoder.decode_bc1(raw, width, height)
        arr = np.frombuffer(bgra, dtype=np.uint8).reshape(height, width, 4)
        return arr[:, :, [2, 1, 0, 3]].copy()
    if fmt_name == "BC2":
        # Fallback: BC2 rarely used; decode via PIL/manual if needed
        arr = _decode_bc2_fallback(raw, width, height)
        return arr
    if fmt_name == "BC3":
        bgra = texture2ddecoder.decode_bc3(raw, width, height)
        arr = np.frombuffer(bgra, dtype=np.uint8).reshape(height, width, 4)
        return arr[:, :, [2, 1, 0, 3]].copy()
    if fmt_name in ("BC4U", "BC4S"):
        bgra = texture2ddecoder.decode_bc4(raw, width, height)
        arr = np.frombuffer(bgra, dtype=np.uint8).reshape(height, width, 4)
        # Put decoded single channel into R (and G/B for preview), A=255
        r = arr[:, :, 2]
        out = np.empty((height, width, 4), dtype=np.uint8)
        out[:, :, 0] = r
        out[:, :, 1] = r
        out[:, :, 2] = r
        out[:, :, 3] = 255
        return out
    if fmt_name in ("BC5U", "BC5S"):
        bgra = texture2ddecoder.decode_bc5(raw, width, height)
        arr = np.frombuffer(bgra, dtype=np.uint8).reshape(height, width, 4)
        out = arr[:, :, [2, 1, 0, 3]].copy()
        if fmt_name == "BC5S":
            # Signed BC5 (-1..1) -> unsigned 0..255 mapping
            s = out[:, :, :2].astype(np.int8).astype(np.float32) / 127.0
            out[:, :, :2] = np.clip((s * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        return out
    if fmt_name == "BC6U":
        bgra = texture2ddecoder.decode_bc6(raw, width, height, False)
        arr = np.frombuffer(bgra, dtype=np.uint8).reshape(height, width, 4)
        return arr[:, :, [2, 1, 0, 3]].copy()
    if fmt_name == "BC6S":
        bgra = texture2ddecoder.decode_bc6(raw, width, height, True)
        arr = np.frombuffer(bgra, dtype=np.uint8).reshape(height, width, 4)
        return arr[:, :, [2, 1, 0, 3]].copy()
    if fmt_name == "BC7":
        bgra = texture2ddecoder.decode_bc7(raw, width, height)
        arr = np.frombuffer(bgra, dtype=np.uint8).reshape(height, width, 4)
        return arr[:, :, [2, 1, 0, 3]].copy()
    if fmt_name == "RGBA_8888":
        return np.frombuffer(raw[: width * height * 4], dtype=np.uint8).reshape(height, width, 4).copy()
    if fmt_name == "RGBA_8888_REV":
        arr = np.frombuffer(raw[: width * height * 4], dtype=np.uint8).reshape(height, width, 4)
        return arr[:, :, [2, 1, 0, 3]].copy()
    if fmt_name == "R_UNORM_8":
        r = np.frombuffer(raw[: width * height], dtype=np.uint8).reshape(height, width)
        out = np.full((height, width, 4), 255, dtype=np.uint8)
        out[:, :, 0] = r
        out[:, :, 1] = r
        out[:, :, 2] = r
        return out
    if fmt_name == "R_UNORM_16":
        r16 = np.frombuffer(raw[: width * height * 2], dtype=np.uint16).reshape(height, width)
        r = (r16 >> 8).astype(np.uint8)
        out = np.full((height, width, 4), 255, dtype=np.uint8)
        out[:, :, 0] = r
        out[:, :, 1] = r
        out[:, :, 2] = r
        return out
    if fmt_name == "RGBA_FLOAT_16":
        f = np.frombuffer(raw[: width * height * 8], dtype=np.float16).reshape(height, width, 4).astype(np.float32)
        return np.clip(f * 255.0, 0, 255).astype(np.uint8)
    if fmt_name == "R_FLOAT_16":
        f = np.frombuffer(raw[: width * height * 2], dtype=np.float16).reshape(height, width).astype(np.float32)
        u = np.clip(f * 255.0, 0, 255).astype(np.uint8)
        out = np.full((height, width, 4), 255, dtype=np.uint8)
        out[:, :, 0] = out[:, :, 1] = out[:, :, 2] = u
        return out
    if fmt_name == "R_FLOAT_32":
        f = np.frombuffer(raw[: width * height * 4], dtype=np.float32).reshape(height, width)
        u = np.clip(f * 255.0, 0, 255).astype(np.uint8)
        out = np.full((height, width, 4), 255, dtype=np.uint8)
        out[:, :, 0] = out[:, :, 1] = out[:, :, 2] = u
        return out
    raise ValueError(f"Unsupported pixel format for decoding: {fmt_name}")


def _decode_bc2_fallback(raw: bytes, width: int, height: int) -> np.ndarray:
    # BC2: 8 bytes explicit 4-bit alpha + 8 bytes BC1 color per 4x4 block
    bx = max(1, (width + 3) // 4)
    by = max(1, (height + 3) // 4)
    bc1_chunks = bytearray()
    alphas = np.zeros((by * 4, bx * 4), dtype=np.uint8)
    off = 0
    for y in range(by):
        for x in range(bx):
            abits = struct.unpack_from("<Q", raw, off)[0]
            for py in range(4):
                for px in range(4):
                    a4 = (abits >> ((py * 4 + px) * 4)) & 0xF
                    alphas[y * 4 + py, x * 4 + px] = a4 * 17
            bc1_chunks.extend(raw[off + 8 : off + 16])
            off += 16
    rgba = _decode_surface_to_rgba(bytes(bc1_chunks), width, height, "BC1")
    rgba[:, :, 3] = alphas[:height, :width]
    return rgba


def decode_texture(
    game: HFWGame,
    tex_obj: dict[str, Any],
    max_dimension: int | None = None,
    packing: dict[str, PackingChannel] | None = None,
) -> DecodedTexture | None:
    """Decode a Decima `Texture` object into an RGBA numpy array (picking highest mip <= `max_dimension`)."""
    tinfo = tex_obj.get("_texture")
    if not tinfo:
        return None
    hdr = tinfo["header"]
    tdata = tinfo["data"]

    fmt_name = game.schema.enum_name("EPixelFormat", hdr["pixel_format"])
    tex_type = game.schema.enum_name("ETextureType", hdr["type"])
    color_space = game.schema.enum_name("ETexColorSpace", hdr["color_space"])
    if fmt_name not in PIXEL_FORMAT_INFO:
        return None

    bw, bh, bpb, _dxgi = PIXEL_FORMAT_INFO[fmt_name]
    width = hdr["width"]
    height = hdr["height"]
    num_mips = hdr["num_mips"]
    num_surfaces = hdr["num_surfaces"]
    streamed_mips = tdata["streamed_mips"]
    embedded_data: bytes = tdata["embedded_data"]

    # Resolve streaming data source (either parent TextureSet or standalone Texture)
    ts_parent = game.resolve_ref(tex_obj.get("TextureSetParent"))
    ds = ts_parent.get("StreamingDataSource") if ts_parent else tex_obj.get("StreamingDataSource")
    streaming_mip_offsets = tex_obj.get("StreamingMipOffsets") or ()

    # Choose target mip level (0 = full res, or downsampled for quick previews)
    target_mip = 0
    if max_dimension is not None and num_mips > 1:
        while target_mip < num_mips - 1 and max(width >> target_mip, height >> target_mip) > max_dimension:
            target_mip += 1

    # Walk mips to locate exact byte slice for `target_mip` (surface 0)
    streamed_offset = 0
    embedded_offset = 0
    mip_raw = b""
    mip_w = max(width >> target_mip, bw)
    mip_h = max(height >> target_mip, bh)

    # Read streamed bytes lazily only if target_mip < streamed_mips
    streamed_data = b""
    if target_mip < streamed_mips and ds:
        if ts_parent is not None and target_mip < len(streaming_mip_offsets):
            # Read just the slice we need from the TextureSet stream!
            mip_sz = _surface_byte_size(mip_w, mip_h, bw, bh, bpb)
            rel_off = ds["Offset"] + int(streaming_mip_offsets[target_mip])
            mip_raw = game.read_data_source(ds, offset=rel_off, length=mip_sz)
        else:
            streamed_data = game.read_data_source(ds)

    if not mip_raw:
        for mip in range(num_mips):
            mw = max(width >> mip, bw)
            mh = max(height >> mip, bh)
            if tex_type in ("2DArray", "_2DArray"):
                elements = num_surfaces
            elif tex_type in ("3D", "_3D"):
                elements = 1 << max(0, num_surfaces - mip)
            elif tex_type == "CubeMap":
                elements = 6
            else:
                elements = 1
            surf_sz = _surface_byte_size(mw, mh, bw, bh, bpb)

            for elem in range(elements):
                if mip >= streamed_mips:
                    chunk = embedded_data[embedded_offset : embedded_offset + surf_sz]
                    embedded_offset += surf_sz
                elif ts_parent is None:
                    chunk = streamed_data[streamed_offset : streamed_offset + surf_sz]
                    streamed_offset += surf_sz
                else:
                    off = int(streaming_mip_offsets[mip])
                    chunk = streamed_data[off : off + surf_sz]
                if mip == target_mip and elem == 0:
                    mip_raw = chunk
                    mip_w, mip_h = mw, mh
                    break
            if mip == target_mip:
                break

    if not mip_raw:
        return None

    # Actual unpadded pixel dimensions for this mip
    actual_w = max(1, width >> target_mip)
    actual_h = max(1, height >> target_mip)
    rgba = _decode_surface_to_rgba(mip_raw, mip_w, mip_h, fmt_name)
    if mip_w != actual_w or mip_h != actual_h:
        rgba = rgba[:actual_h, :actual_w, :].copy()

    return DecodedTexture(
        object_id=str(tex_obj.get("_id", "")),
        width=actual_w,
        height=actual_h,
        mip_count=num_mips,
        pixel_format=fmt_name,
        color_space=color_space,
        tex_type=tex_type,
        rgba=rgba,
        mip0_raw=mip_raw,
        packing=packing or {},
    )


def reconstruct_normal_map(
    nx_u8: np.ndarray, ny_u8: np.ndarray, flip_green_for_opengl: bool = False
) -> np.ndarray:
    """Reconstruct tangent-space XYZ normal map `(H, W, 3)` uint8 from XY channels.

    HFW stores tangent-space normals in DirectX convention (Y down / green down).
    - Unreal Engine uses DirectX convention (`flip_green_for_opengl=False`).
    - Blender uses OpenGL convention (`flip_green_for_opengl=True`).
    """
    nx = nx_u8.astype(np.float32) * (2.0 / 255.0) - 1.0
    ny = ny_u8.astype(np.float32) * (2.0 / 255.0) - 1.0
    nz_sq = np.clip(1.0 - nx * nx - ny * ny, 0.0, 1.0)
    nz = np.sqrt(nz_sq)
    if flip_green_for_opengl:
        ny = -ny
    out = np.empty((nx.shape[0], nx.shape[1], 3), dtype=np.uint8)
    out[:, :, 0] = np.clip((nx * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    out[:, :, 1] = np.clip((ny * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    out[:, :, 2] = np.clip((nz * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    return out


def render_channel_image(decoded: DecodedTexture, mode: str = "rgba") -> Image.Image:
    """Render an inspection view (`rgba`, `rgb`, `r`, `g`, `b`, `a`, `normal_dx`, `normal_gl`)."""
    arr = decoded.rgba
    m = mode.lower()
    if m == "rgba":
        return Image.fromarray(arr, mode="RGBA")
    if m == "rgb":
        return Image.fromarray(arr[:, :, :3], mode="RGB")
    if m in ("r", "g", "b", "a"):
        ch = arr[:, :, LANE_TO_IDX[m.upper()]]
        return Image.fromarray(ch, mode="L")
    if m == "normal_dx":
        return Image.fromarray(reconstruct_normal_map(arr[:, :, 0], arr[:, :, 1], False), mode="RGB")
    if m == "normal_gl":
        return Image.fromarray(reconstruct_normal_map(arr[:, :, 0], arr[:, :, 1], True), mode="RGB")
    return Image.fromarray(arr, mode="RGBA")


def write_dds(decoded: DecodedTexture, out_path: Path) -> None:
    """Write mip 0 of `decoded` as a standard DX10 DDS file."""
    info = PIXEL_FORMAT_INFO.get(decoded.pixel_format)
    if info is None:
        raise ValueError(f"Cannot export DDS for format {decoded.pixel_format}")
    _bw, _bh, _bpb, dxgi_fmt = info
    if decoded.color_space == "sRGB":
        srgb_map = {71: 72, 74: 75, 77: 78, 98: 99, 28: 29}
        dxgi_fmt = srgb_map.get(dxgi_fmt, dxgi_fmt)

    # DDS_HEADER (124 bytes) + DDS_HEADER_DXT10 (20 bytes)
    flags = 0x1 | 0x2 | 0x4 | 0x1000 | 0x80000  # CAPS | HEIGHT | WIDTH | PIXELFORMAT | LINEARSIZE
    header = struct.pack(
        "<4sIIIIIII44xII4sIIIII5I",
        b"DDS ",
        124,
        flags,
        decoded.height,
        decoded.width,
        len(decoded.mip0_raw),
        1,
        1,
        32,
        0x4,  # DDPF_FOURCC
        b"DX10",
        0,
        0,
        0,
        0,
        0,
        0x1000,  # DDSCAPS_TEXTURE
        0,
        0,
        0,
        0,
    )
    dx10 = struct.pack("<IIIII", dxgi_fmt, 3, 0, 1, 0)  # D3D10_RESOURCE_DIMENSION_TEXTURE2D
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(header + dx10 + decoded.mip0_raw)


@dataclass(slots=True)
class UnpackedTextureSet:
    object_id: str
    name: str
    source_paths: dict[str, str]  # ETextureSetType -> work-relative authoring path
    packed_entries: list[dict[str, Any]]
    # Semantic map name -> path on disk (when exported) or numpy array (in-memory)
    semantic_maps: dict[str, np.ndarray] = field(default_factory=dict)


def unpack_texture_set(
    game: HFWGame,
    ts_obj: dict[str, Any],
    max_dimension: int | None = None,
) -> UnpackedTextureSet:
    """Decode all packed textures in a `TextureSet` and reassemble semantic authoring maps."""
    oid = str(ts_obj.get("_id", ""))
    source_paths: dict[str, str] = {}
    first_stem = f"TextureSet_{oid.replace(':', '_')}"
    for desc in ts_obj.get("TextureDesc", ()):
        tname = ETEXTURE_SET_TYPE.get(desc.get("TextureType", 0), "Invalid")
        raw_p = desc.get("Path", "")
        if raw_p:
            clean = raw_p[5:] if raw_p.lower().startswith("work:") else raw_p
            source_paths[tname] = clean
            if first_stem.startswith("TextureSet_"):
                first_stem = Path(clean).stem

    # Collect per-(semantic_type, channel_idx) planes
    planes: dict[str, dict[int, np.ndarray]] = {}
    packed_entries: list[dict[str, Any]] = []

    for idx, entry in enumerate(ts_obj.get("Entries", ())):
        tex = game.resolve_ref(entry.get("Texture"))
        packing = decode_packing_info(entry.get("PackingInfo", 0))
        entry_meta: dict[str, Any] = {
            "entry_index": idx,
            "texture_id": str(tex.get("_id", "")) if tex else "",
            "packing": {k: f"{v.tex_type}.{v.source_channel}" for k, v in packing.items()},
        }
        if tex is None:
            packed_entries.append(entry_meta)
            continue
        dec = decode_texture(game, tex, max_dimension=max_dimension, packing=packing)
        if dec is None:
            packed_entries.append(entry_meta)
            continue
        entry_meta.update(
            {
                "width": dec.width,
                "height": dec.height,
                "pixel_format": dec.pixel_format,
                "color_space": dec.color_space,
            }
        )
        packed_entries.append(entry_meta)
        for lane, pchan in packing.items():
            src_idx = LANE_TO_IDX[lane]
            dst_idx = LANE_TO_IDX.get(pchan.source_channel, 0)
            planes.setdefault(pchan.tex_type, {})[dst_idx] = dec.rgba[:, :, src_idx]

    # Reassemble semantic maps!
    semantic_maps: dict[str, np.ndarray] = {}
    for stype, ch_map in planes.items():
        if not ch_map:
            continue
        # Match resolution to largest plane (in case mips differ slightly)
        max_h = max(a.shape[0] for a in ch_map.values())
        max_w = max(a.shape[1] for a in ch_map.values())

        def get_plane(ci: int, default: int = 0) -> np.ndarray:
            arr = ch_map.get(ci)
            if arr is None:
                return np.full((max_h, max_w), default, dtype=np.uint8)
            if arr.shape != (max_h, max_w):
                return np.array(Image.fromarray(arr, mode="L").resize((max_w, max_h), Image.BILINEAR))
            return arr

        if stype == "Normal":
            nx = get_plane(0, 128)
            ny = get_plane(1, 128)
            semantic_maps["Normal_DX"] = reconstruct_normal_map(nx, ny, flip_green_for_opengl=False)
            semantic_maps["Normal_GL"] = reconstruct_normal_map(nx, ny, flip_green_for_opengl=True)
        elif stype in ("Color", "Incandescence", "Translucency_Diffusion"):
            rgb = np.stack([get_plane(0), get_plane(1), get_plane(2)], axis=-1)
            semantic_maps[stype] = rgb
        else:
            # Single-channel grayscale map (Roughness, Reflectance, AO, Mask, Alpha, Height, Curvature, Misc_01...)
            if len(ch_map) == 1:
                semantic_maps[stype] = next(iter(ch_map.values()))
            else:
                rgb = np.stack([get_plane(0), get_plane(1), get_plane(2)], axis=-1)
                semantic_maps[stype] = rgb

    return UnpackedTextureSet(
        object_id=oid,
        name=first_stem,
        source_paths=source_paths,
        packed_entries=packed_entries,
        semantic_maps=semantic_maps,
    )


def save_unpacked_texture_set(unpacked: UnpackedTextureSet, out_dir: Path) -> dict[str, str]:
    """Write all semantic maps of `unpacked` to PNG files in `out_dir`. Returns `{map_type: relative_filename}`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_base = "".join(c if (c.isalnum() or c in "-_") else "_" for c in unpacked.name)[:60]
    uid = unpacked.object_id.replace(":", "_")
    saved: dict[str, str] = {}
    for map_type, arr in unpacked.semantic_maps.items():
        fname = f"{safe_base}_{uid}_{map_type}.png"
        fpath = out_dir / fname
        if not fpath.exists():
            mode = "L" if arr.ndim == 2 else ("RGB" if arr.shape[2] == 3 else "RGBA")
            Image.fromarray(arr, mode=mode).save(fpath)
        saved[map_type] = fname
    return saved


def _packing_short_tag(packing: dict[str, PackingChannel]) -> str:
    """Build a readable filename suffix like `CLR`, `MSK_RGH_RFL`, `NMT_AO_MSKA`."""
    abbr = {
        "Color": "CLR",
        "Normal": "NMT",
        "Roughness": "RGH",
        "Reflectance": "RFL",
        "AO": "AO",
        "Mask": "MSK",
        "Mask_Alpha": "MSKA",
        "Alpha": "ALP",
        "Height": "HGT",
        "Incandescence": "EMS",
        "Curvature": "CRV",
        "Translucency_Diffusion": "TRD",
        "Translucency_Amount": "TRA",
    }
    seen: list[str] = []
    for lane in ("R", "G", "B", "A"):
        pc = packing.get(lane)
        if pc is None:
            continue
        t = abbr.get(pc.tex_type, pc.tex_type[:4].upper())
        if t not in seen:
            seen.append(t)
    return "_".join(seen) or "PACKED"


def export_packed_texture_set(
    game: HFWGame,
    ts_obj: dict[str, Any],
    out_dir: Path,
    max_dimension: int | None = None,
) -> list[dict[str, Any]]:
    """Export native game-packed RGBA textures from a `TextureSet` WITHOUT unpacking channels.

    Returns a list of packed entry descriptors:
        [{"filename", "texture_id", "color_space", "has_alpha", "packing": {"R": "Normal.R", ...}}]
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    oid = str(ts_obj.get("_id", ""))
    first_stem = f"TextureSet_{oid.replace(':', '_')}"
    for desc in ts_obj.get("TextureDesc", ()):
        raw_p = desc.get("Path", "")
        if raw_p:
            clean = raw_p[5:] if raw_p.lower().startswith("work:") else raw_p
            first_stem = Path(clean).stem
            break

    safe_base = "".join(c if (c.isalnum() or c in "-_") else "_" for c in first_stem)[:55]
    entries_out: list[dict[str, Any]] = []

    for idx, entry in enumerate(ts_obj.get("Entries", ())):
        tex = game.resolve_ref(entry.get("Texture"))
        if tex is None:
            continue
        packing = decode_packing_info(int(entry.get("PackingInfo", 0)))
        dec = decode_texture(game, tex, max_dimension=max_dimension, packing=packing)
        if dec is None:
            continue
        tex_uid = str(tex.get("_id", f"{oid}_{idx}")).replace(":", "_")
        tag = _packing_short_tag(packing)
        fname = f"{safe_base}_{tex_uid}_{tag}.png"
        fpath = out_dir / fname
        has_alpha = "A" in packing
        if not fpath.exists():
            if has_alpha:
                Image.fromarray(dec.rgba, mode="RGBA").save(fpath)
            else:
                Image.fromarray(dec.rgba[:, :, :3], mode="RGB").save(fpath)
        entries_out.append(
            {
                "filename": fname,
                "texture_id": str(tex.get("_id", "")),
                "color_space": "sRGB" if dec.color_space == "sRGB" else "Non-Color",
                "has_alpha": has_alpha,
                "width": dec.width,
                "height": dec.height,
                "pixel_format": dec.pixel_format,
                "packing": {k: f"{v.tex_type}.{v.source_channel}" for k, v in packing.items()},
            }
        )
    return entries_out


def export_packed_texture(
    game: HFWGame,
    tex_obj: dict[str, Any],
    packing: dict[str, PackingChannel],
    name_hint: str,
    out_dir: Path,
    max_dimension: int | None = None,
) -> dict[str, Any] | None:
    """Export a single native game-packed `Texture` (e.g. direct `Texture` binding into a `TextureSetParent`)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dec = decode_texture(game, tex_obj, max_dimension=max_dimension, packing=packing)
    if dec is None:
        return None
    tex_uid = str(tex_obj.get("_id", "")).replace(":", "_")
    safe_base = "".join(c if (c.isalnum() or c in "-_") else "_" for c in name_hint)[:55] or "Tex"
    tag = _packing_short_tag(packing)
    fname = f"{safe_base}_{tex_uid}_{tag}.png"
    fpath = out_dir / fname
    has_alpha = bool(("A" in packing) or (int(dec.rgba[:, :, 3].min()) < 255))
    if not fpath.exists():
        if has_alpha:
            Image.fromarray(dec.rgba, mode="RGBA").save(fpath)
        else:
            Image.fromarray(dec.rgba[:, :, :3], mode="RGB").save(fpath)
    return {
        "filename": fname,
        "texture_id": str(tex_obj.get("_id", "")),
        "color_space": "sRGB" if dec.color_space == "sRGB" else "Non-Color",
        "has_alpha": has_alpha,
        "width": dec.width,
        "height": dec.height,
        "pixel_format": dec.pixel_format,
        "packing": {k: f"{v.tex_type}.{v.source_channel}" for k, v in packing.items()},
    }


def export_standalone_texture(
    game: HFWGame,
    tex_obj: dict[str, Any],
    name_hint: str,
    out_dir: Path,
    color_space: str | None = None,
    max_dimension: int | None = None,
) -> dict[str, Any] | None:
    """Export a standalone non-TextureSet Decima `Texture` (e.g. machine muscle braid/veins/pulse/ramp/core)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dec = decode_texture(game, tex_obj, max_dimension=max_dimension)
    if dec is None:
        return None
    tex_uid = str(tex_obj.get("_id", "")).replace(":", "_")
    safe_base = "".join(c if (c.isalnum() or c in "-_") else "_" for c in name_hint)[:55] or "Tex"
    fname = f"{safe_base}_{tex_uid}.png"
    fpath = out_dir / fname
    has_alpha = bool(int(dec.rgba[:, :, 3].min()) < 255)
    if not fpath.exists():
        if has_alpha:
            Image.fromarray(dec.rgba, mode="RGBA").save(fpath)
        else:
            Image.fromarray(dec.rgba[:, :, :3], mode="RGB").save(fpath)
    cs = color_space or ("sRGB" if dec.color_space == "sRGB" else "Non-Color")
    return {
        "filename": fname,
        "texture_id": str(tex_obj.get("_id", "")),
        "color_space": cs,
        "has_alpha": has_alpha,
        "width": dec.width,
        "height": dec.height,
        "pixel_format": dec.pixel_format,
    }


def decode_texture_array_slices(
    game: HFWGame,
    tex_obj: dict[str, Any],
    max_slice_dim: int | None = None,
) -> list[np.ndarray]:
    """Decode ALL slices of a Decima `2DArray` Texture (e.g. `20:0` 256x256x23 BC7 machine detail array)."""
    tinfo = tex_obj.get("_texture")
    if not tinfo:
        return []
    hdr = tinfo["header"]
    tdata = tinfo["data"]
    tex_type = game.schema.enum_name("ETextureType", hdr["type"])
    fmt_name = game.schema.enum_name("EPixelFormat", hdr["pixel_format"])
    if tex_type not in ("2DArray", "_2DArray") or fmt_name not in PIXEL_FORMAT_INFO:
        return []
    bw, bh, bpb, _ = PIXEL_FORMAT_INFO[fmt_name]
    width = hdr["width"]
    height = hdr["height"]
    num_mips = hdr["num_mips"]
    num_surfaces = max(1, hdr["num_surfaces"])
    streamed_mips = tdata["streamed_mips"]
    embedded_data: bytes = tdata["embedded_data"]
    ds = tex_obj.get("StreamingDataSource")
    streamed_data = game.read_data_source(ds) if (streamed_mips > 0 and ds) else b""

    target_mip = 0
    if max_slice_dim is not None and num_mips > 1:
        while target_mip < num_mips - 1 and max(width >> target_mip, height >> target_mip) > max_slice_dim:
            target_mip += 1

    streamed_offset = 0
    embedded_offset = 0
    slices: list[np.ndarray] = []
    for mip in range(num_mips):
        mw = max(width >> mip, bw)
        mh = max(height >> mip, bh)
        surf_sz = _surface_byte_size(mw, mh, bw, bh, bpb)
        actual_w = max(1, width >> mip)
        actual_h = max(1, height >> mip)
        for _elem in range(num_surfaces):
            if mip >= streamed_mips:
                chunk = embedded_data[embedded_offset : embedded_offset + surf_sz]
                embedded_offset += surf_sz
            else:
                chunk = streamed_data[streamed_offset : streamed_offset + surf_sz]
                streamed_offset += surf_sz
            if mip == target_mip and len(chunk) >= surf_sz:
                rgba = _decode_surface_to_rgba(chunk, mw, mh, fmt_name)
                if mw != actual_w or mh != actual_h:
                    rgba = rgba[:actual_h, :actual_w, :].copy()
                slices.append(rgba)
        if mip == target_mip:
            break
    return slices


def export_texture_array_atlas(
    game: HFWGame,
    tex_obj: dict[str, Any],
    out_dir: Path,
    cols: int = 5,
    max_slice_dim: int = 256,
) -> dict[str, Any] | None:
    """Export all slices of a `_2DArray` detail texture as a single RGBA tiled atlas PNG (bottom-left slice 0 in Blender UV space!)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    oid = str(tex_obj.get("_id", ""))
    uid = oid.replace(":", "_")
    slices = decode_texture_array_slices(game, tex_obj, max_slice_dim=max_slice_dim)
    if not slices:
        return None
    n_slices = len(slices)
    rows = (n_slices + cols - 1) // cols
    sh, sw = slices[0].shape[:2]
    fname = f"HFW_DetailArray_{uid}_{cols}x{rows}.png"
    fpath = out_dir / fname
    if not fpath.exists():
        # Neutral RGBA fill (128,128,128,128) for unused trailing slots so there's zero black fringe
        atlas = np.full((rows * sh, cols * sw, 4), 128, dtype=np.uint8)
        for idx, sl in enumerate(slices):
            r, c = divmod(idx, cols)
            # Store row 0 at BOTTOM of image so in Blender UV space (V=0 bottom, V=1 top),
            # slice `idx` lives directly at tile `(col = idx % cols, row = idx // cols)`!
            img_r = (rows - 1 - r) * sh
            img_c = c * sw
            atlas[img_r : img_r + sh, img_c : img_c + sw, :] = sl
        Image.fromarray(atlas, mode="RGBA").save(fpath)
    return {
        "filename": fname,
        "texture_id": oid,
        "slice_count": n_slices,
        "cols": cols,
        "rows": rows,
        "slice_width": sw,
        "slice_height": sh,
    }
