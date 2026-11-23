"""Reverse-lookup dictionary for Decima shader binding & variable hashes (CRC32C & 31-bit)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from hfw_toolkit.config import ToolkitConfig
from hfw_toolkit.decima.hashes import decima_crc32c, decima_string_hash

# Curated known Decima material/shader variable names + common PBR/Horizon identifiers
_KNOWN_NAMES = [
    # Common Decima shader variables & textures
    "BaseColor", "Diffuse", "Albedo", "Color", "ColorTint", "TintColor", "Tint",
    "Normal", "NormalMap", "NormalStrength", "DetailNormal", "DetailNormalMap", "DetailNormalStrength",
    "DetailTile", "DetailTiling", "DetailScale", "DetailUVScale", "DetailStrength", "DetailRoughness",
    "Roughness", "RoughnessScale", "RoughnessBias", "RoughnessMin", "RoughnessMax", "RoughnessMultiplier",
    "Reflectance", "ReflectanceScale", "Metallic", "Metalness", "Specular", "F0",
    "AO", "AmbientOcclusion", "Occlusion", "OcclusionStrength",
    "Mask", "MaskMap", "LayerMask", "WearMask", "DirtMask", "DamageMask", "Damage", "Health",
    "Emissive", "EmissiveColor", "EmissiveIntensity", "EmissiveStrength", "Incandescence", "Glow", "GlowIntensity",
    "PlasmaCharge", "PlasmaIntensity", "PlasmaColor", "ChargeLevel", "Overcharge", "ElementalCharge",
    "Alpha", "AlphaRef", "AlphaCutoff", "AlphaTest", "Opacity", "Fade", "Dissolve",
    "Subsurface", "Translucency", "TranslucencyColor", "TranslucencyAmount", "TranslucencyDiffusion",
    "Curvature", "Height", "Parallax", "Displacement", "Sheen", "Anisotropy", "ClearCoat",
    "Wetness", "Snow", "Frost", "Dust", "Mud", "Sand", "Scorch", "Burn", "Acid", "Corrosion",
    "UVScale", "UVOffset", "Tile", "Tiling", "Scale", "Strength", "Intensity", "Power", "Contrast", "Brightness",
    "MuscleFiber", "CarbonFiber", "BrushedMetal", "PaintColor", "PlateColor", "ApexColor",
    # Empirically identified common HFW machine shader parameters (semantic aliases when hash unknown)
]

# Empirically observed HFW machine parameter hashes (annotated from live Slitherfang / Slaughterspine analysis)
EMPIRICAL_HFW_HASH_HINTS: dict[int, str] = {
    0x3806EAE3: "Detail_UV_Tiling (empirical)",
    0x4CBDF9B1: "Surface_Param_A (0.19 default)",
    0x7FC0520A: "Surface_Roughness_Range (0.17 / 0.38)",
    0x0E470C22: "Layer_Blend_Weights (0.1 / 0.7 / 0.05)",
    0x5F343110: "Dynamic_State_Scalar_0",
    0x72F39F6C: "Dynamic_State_Flag_0",
    0x0B3E6131: "Dynamic_State_Scalar_1",
}


class ShaderHashDictionary:
    """Resolves Decima 32-bit `BindingNameHash` / `VariableIDHash` to readable names when possible."""

    def __init__(self, config: ToolkitConfig | None = None) -> None:
        self.config = config or ToolkitConfig.load()
        self.cache_path = self.config.cache_dir / "shader_hash_dict.json"
        self._map: dict[int, str] = {}
        self._load_or_build()

    def _add_candidate(self, name: str) -> None:
        h31 = decima_string_hash(name)
        h32 = decima_crc32c(name)
        self._map.setdefault(h31, name)
        self._map.setdefault(h32, name)

    def _load_or_build(self) -> None:
        # 1. Always seed curated + empirical hints
        for k, v in EMPIRICAL_HFW_HASH_HINTS.items():
            self._map[k] = v
        for n in _KNOWN_NAMES:
            self._add_candidate(n)
            self._add_candidate(n.lower())
            self._add_candidate("g_" + n)
            self._add_candidate("u_" + n)
            self._add_candidate("m_" + n)
            self._add_candidate("c_" + n)

        if self.cache_path.exists():
            try:
                saved = json.loads(self.cache_path.read_text(encoding="utf-8"))
                for k_hex, name in saved.items():
                    self._map[int(k_hex, 16)] = name
                return
            except Exception:
                pass

        # 2. Harvest shader-like attribute & symbol names from dump/hfw.json (fast, <0.2s)
        if self.config.rtti_json and self.config.rtti_json.exists():
            try:
                raw = json.loads(self.config.rtti_json.read_text(encoding="utf-8"))
                for tname, tinfo in raw.items():
                    if any(k in tname for k in ("Shader", "Material", "Texture", "Render", "Color", "Light")):
                        self._add_candidate(tname)
                    for attr in tinfo.get("attrs", ()):
                        an = attr.get("name")
                        if an:
                            self._add_candidate(an)
            except Exception:
                pass

    def resolve(self, hash_val: int) -> str:
        u = hash_val & 0xFFFFFFFF
        return self._map.get(u) or self._map.get(u & 0x7FFFFFFF) or ""

    def scan_executable_strings(self, target_hashes: set[int]) -> dict[int, str]:
        """Optional fast scan of ASCII strings inside `HorizonForbiddenWest.exe` against `target_hashes`."""
        exe_path = self.config.game_dir / "HorizonForbiddenWest.exe"
        if not exe_path.exists() or not target_hashes:
            return {}
        data = exe_path.read_bytes()
        pat = re.compile(rb"[A-Za-z_][A-Za-z0-9_]{2,60}")
        found: dict[int, str] = {}
        target_31 = {h & 0x7FFFFFFF for h in target_hashes}
        for m in pat.finditer(data):
            s = m.group(0).decode("ascii")
            h31 = decima_string_hash(s)
            if h31 in target_31 and h31 not in found:
                found[h31] = s
                self._map[h31] = s
        if found:
            # Persist newly discovered names
            existing = {}
            if self.cache_path.exists():
                try:
                    existing = json.loads(self.cache_path.read_text(encoding="utf-8"))
                except Exception:
                    existing = {}
            for h, s in found.items():
                existing[f"{h:#010x}"] = s
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        return found
