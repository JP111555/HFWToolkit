"""DirectX Shader Compiler (`dxcompiler.dll`) Python ctypes COM wrapper for HFW PC DXIL shaders."""

from __future__ import annotations

import ctypes
import hashlib
import re
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from hfw_toolkit.config import ToolkitConfig


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_ubyte * 8),
    ]


def _make_guid(guid_str: str) -> _GUID:
    u = uuid.UUID(guid_str)
    return _GUID(u.time_low, u.time_mid, u.time_hi_version, (ctypes.c_ubyte * 8)(*u.bytes[8:]))


_CLSID_DXC_UTILS = _make_guid("6245d6af-66e0-48fd-80b4-4d271796748c")
_IID_IDXC_UTILS = _make_guid("4605c4cb-2019-492a-ada4-65f20bb7d67f")
_CLSID_DXC_COMPILER = _make_guid("73e22d93-e6ce-47f3-b5bf-f0664f39c1b0")
_IID_IDXC_COMPILER = _make_guid("8c210bf3-011f-4422-8d70-6f9acb8db617")


def _vtable_fn(obj_ptr: ctypes.c_void_p, slot: int, restype, *argtypes):
    vtbl = ctypes.cast(obj_ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(vtbl[slot])


def _com_release(obj_ptr: ctypes.c_void_p) -> None:
    if obj_ptr and obj_ptr.value:
        release_fn = _vtable_fn(obj_ptr, 2, ctypes.c_uint32)
        release_fn(obj_ptr)


@dataclass(slots=True)
class ResourceBindingDecl:
    name: str
    res_type: str  # cbuffer, texture, sampler, UAV
    fmt: str
    dim: str  # NA, 2d, 3d, cube, buf, r/w
    res_id: str  # CB0, T6, S0...
    hlsl_bind: str  # cb0,space5 / t0,space5
    space: int
    register: int
    count: int


@dataclass(slots=True)
class SignatureParam:
    name: str
    index: int
    mask: str
    register: int
    sys_value: str
    fmt: str
    used: str


@dataclass(slots=True)
class DisassembledProgram:
    stage: str  # VertexProgram | PixelProgram | GeometryProgram | ComputeProgram
    shader_model: str
    debug_name: str
    shader_hash: str
    disassembly: str
    inputs: list[SignatureParam] = field(default_factory=list)
    outputs: list[SignatureParam] = field(default_factory=list)
    bindings: list[ResourceBindingDecl] = field(default_factory=list)
    cbuffer_sizes: list[int] = field(default_factory=list)
    uses_bindless_heap: bool = False


class DXCDisassembler:
    """Thread-safe wrapper around HFW's shipped `tools/bin/x64/dxcompiler.dll` with disk caching."""

    def __init__(self, config: ToolkitConfig | None = None) -> None:
        self.config = config or ToolkitConfig.load()
        self.cache_dir = self.config.cache_dir / "dxc"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._initialized = False
        self._dxc = None
        self._p_utils = ctypes.c_void_p()
        self._p_comp = ctypes.c_void_p()
        self._create_blob = None
        self._disassemble = None

    def _ensure_init(self) -> None:
        if self._initialized:
            return
        dll_path = self.config.dxcompiler_dll
        if not dll_path.exists():
            raise FileNotFoundError(f"dxcompiler.dll not found at {dll_path}")
        self._dxc = ctypes.WinDLL(str(dll_path))
        create_inst = self._dxc.DxcCreateInstance
        create_inst.argtypes = [ctypes.POINTER(_GUID), ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p)]
        create_inst.restype = ctypes.c_long

        hr1 = create_inst(ctypes.byref(_CLSID_DXC_UTILS), ctypes.byref(_IID_IDXC_UTILS), ctypes.byref(self._p_utils))
        hr2 = create_inst(ctypes.byref(_CLSID_DXC_COMPILER), ctypes.byref(_IID_IDXC_COMPILER), ctypes.byref(self._p_comp))
        if hr1 != 0 or hr2 != 0:
            raise OSError(f"DxcCreateInstance failed: hr1={hr1:#x}, hr2={hr2:#x}")

        self._create_blob = _vtable_fn(
            self._p_utils, 4, ctypes.c_long, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)
        )
        self._disassemble = _vtable_fn(
            self._p_comp, 5, ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
        )
        self._initialized = True

    def disassemble_blob(self, blob: bytes) -> str:
        if not blob:
            return ""
        key = hashlib.sha256(blob).hexdigest()[:24]
        cache_file = self.cache_dir / f"{key}.dxil.ll"
        if cache_file.exists():
            return cache_file.read_text(encoding="utf-8", errors="replace").replace("\x00", "")

        with self._lock:
            self._ensure_init()
            buf = (ctypes.c_ubyte * len(blob)).from_buffer_copy(blob)
            p_src = ctypes.c_void_p()
            p_dis = ctypes.c_void_p()
            hr = self._create_blob(self._p_utils, buf, len(blob), 0, ctypes.byref(p_src))
            if hr != 0 or not p_src.value:
                raise OSError(f"IDxcUtils::CreateBlobFromPinned failed: {hr:#x}")
            try:
                hr2 = self._disassemble(self._p_comp, p_src, ctypes.byref(p_dis))
                if hr2 != 0 or not p_dis.value:
                    raise OSError(f"IDxcCompiler::Disassemble failed: {hr2:#x}")
                try:
                    ptr = _vtable_fn(p_dis, 3, ctypes.c_void_p)(p_dis)
                    sz = _vtable_fn(p_dis, 4, ctypes.c_size_t)(p_dis)
                    text = ctypes.string_at(ptr, sz).decode("utf-8", errors="replace").replace("\x00", "")
                finally:
                    _com_release(p_dis)
            finally:
                _com_release(p_src)

        cache_file.write_text(text, encoding="utf-8")
        return text

    def disassemble_program(self, blob: bytes, stage: str) -> DisassembledProgram:
        text = self.disassemble_blob(blob)
        return parse_dxil_header(text, stage)


_BIND_RE = re.compile(
    r"^;\s*(?P<name>\S*)\s+(?P<type>cbuffer|texture|sampler|UAV)\s+"
    r"(?P<fmt>\S+)\s+(?P<dim>\S+)\s+(?P<id>[A-Z]+\d+)\s+"
    r"(?P<bind>(?:cb|t|s|u)\d+(?:,space\d+)?)\s+(?P<count>\S+)"
)
_SIG_RE = re.compile(
    r"^;\s+(?P<name>[A-Za-z0-9_]+)\s+(?P<idx>\d+)\s+(?P<mask>[xyzw]+)\s+"
    r"(?P<reg>\d+)\s+(?P<sys>\S+)\s+(?P<fmt>\S+)(?:\s+(?P<used>[xyzw ]+))?\s*$"
)
_REG_SPACE_RE = re.compile(r"^[a-z]+(\d+)(?:,space(\d+))?$")
_CBUF_SZ_RE = re.compile(r"^;\s+\[(\d+)\s+x\s+i8\]")
_SM_RE = re.compile(r'!dx\.shaderModel\s*=\s*!\{!(\d+)\}')


def parse_dxil_header(disasm: str, stage: str) -> DisassembledProgram:
    """Parse structured header comments (signatures, resource bindings, cbuffer sizes) from DXIL text."""
    debug_name = ""
    shader_hash = ""
    shader_model = "6.x"
    inputs: list[SignatureParam] = []
    outputs: list[SignatureParam] = []
    bindings: list[ResourceBindingDecl] = []
    cbuffer_sizes: list[int] = []
    uses_bindless = False

    section = ""
    for line in disasm.splitlines():
        if line.startswith("target datalayout"):
            break
        if "Resource descriptor heap indexing" in line or "Sampler descriptor heap indexing" in line:
            uses_bindless = True
        if line.startswith("; shader debug name:"):
            debug_name = line.split(":", 1)[1].strip()
        elif line.startswith("; shader hash:"):
            shader_hash = line.split(":", 1)[1].strip()
        elif line.startswith("; Input signature:"):
            section = "in" if not inputs else "in2"
        elif line.startswith("; Output signature:"):
            section = "out" if not outputs else "out2"
        elif line.startswith("; Buffer Definitions:"):
            section = "cbuf"
        elif line.startswith("; Resource Bindings:"):
            section = "bind"
        elif line.startswith("; Pipeline Runtime Information:"):
            section = "pipe"
        elif section in ("in", "out"):
            m = _SIG_RE.match(line)
            if m:
                sp = SignatureParam(
                    name=m.group("name"),
                    index=int(m.group("idx")),
                    mask=m.group("mask"),
                    register=int(m.group("reg")),
                    sys_value=m.group("sys"),
                    fmt=m.group("fmt"),
                    used=(m.group("used") or "").strip(),
                )
                if section == "in":
                    inputs.append(sp)
                else:
                    outputs.append(sp)
        elif section == "cbuf":
            mc = _CBUF_SZ_RE.match(line)
            if mc:
                cbuffer_sizes.append(int(mc.group(1)))
        elif section == "bind":
            mb = _BIND_RE.match(line)
            if mb:
                bind_s = mb.group("bind")
                mrs = _REG_SPACE_RE.match(bind_s)
                reg = int(mrs.group(1)) if mrs else 0
                space = int(mrs.group(2)) if (mrs and mrs.group(2)) else 0
                cnt_s = mb.group("count")
                cnt = -1 if cnt_s == "unbounded" else (int(cnt_s) if cnt_s.isdigit() else 1)
                bindings.append(
                    ResourceBindingDecl(
                        name=mb.group("name"),
                        res_type=mb.group("type"),
                        fmt=mb.group("fmt"),
                        dim=mb.group("dim"),
                        res_id=mb.group("id"),
                        hlsl_bind=bind_s,
                        space=space,
                        register=reg,
                        count=cnt,
                    )
                )

    # Also detect exact Shader Model from `!dx.shaderModel` metadata near bottom
    sm_match = re.search(r'!\{!"(vs|ps|gs|cs)",\s*i32\s+(\d+),\s*i32\s+(\d+)\}', disasm)
    if sm_match:
        shader_model = f"{sm_match.group(1)}_{sm_match.group(2)}_{sm_match.group(3)}"

    return DisassembledProgram(
        stage=stage,
        shader_model=shader_model,
        debug_name=debug_name,
        shader_hash=shader_hash,
        disassembly=disasm,
        inputs=inputs,
        outputs=outputs,
        bindings=bindings,
        cbuffer_sizes=cbuffer_sizes,
        uses_bindless_heap=uses_bindless,
    )
