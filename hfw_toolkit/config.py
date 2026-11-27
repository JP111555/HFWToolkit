"""Centralized path configuration & zero-config auto-detection for HFWToolkit."""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_GAME_DIR = Path(
    r"C:\Program Files (x86)\Steam\steamapps\common\Horizon Forbidden West Complete Edition"
)
DEFAULT_BLENDER_EXE = Path(r"C:\Program Files\Blender Foundation\Blender 5.3\blender.exe")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def is_valid_hfw_dir(path: Path | str | None) -> bool:
    """Return True if `path` contains Horizon Forbidden West's `LocalCacheWinGame/package/streaming_graph.core`."""
    if not path:
        return False
    try:
        p = Path(path)
        return (p / "LocalCacheWinGame" / "package" / "streaming_graph.core").is_file()
    except Exception:
        return False


def auto_detect_game_dir() -> Path:
    """Automatically locate Horizon Forbidden West Complete Edition across Steam, Epic, Registry & mounted drives."""
    if env_dir := os.environ.get("HFW_GAME_DIR"):
        if is_valid_hfw_dir(env_dir):
            return Path(env_dir)

    if is_valid_hfw_dir(DEFAULT_GAME_DIR):
        return DEFAULT_GAME_DIR

    candidates: list[Path] = []

    # 1. Windows Registry (Steam App 2420110 = Horizon Forbidden West Complete Edition, plus Steam base path)
    if os.name == "nt":
        try:
            import winreg

            for root, subkey, val_name in (
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Steam App 2420110", "InstallLocation"),
                (winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam", "SteamPath"),
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam", "InstallPath"),
            ):
                try:
                    with winreg.OpenKey(root, subkey) as k:
                        val, _ = winreg.QueryValueEx(k, val_name)
                        if val:
                            p = Path(str(val))
                            if val_name == "InstallLocation":
                                candidates.append(p)
                            else:
                                candidates.append(p / "steamapps" / "common" / "Horizon Forbidden West Complete Edition")
                                vdf = p / "steamapps" / "libraryfolders.vdf"
                                if vdf.is_file():
                                    for m in re.finditer(r'"path"\s+"([^"]+)"', vdf.read_text(encoding="utf-8", errors="ignore")):
                                        lib = Path(m.group(1).replace("\\\\", "\\"))
                                        candidates.append(lib / "steamapps" / "common" / "Horizon Forbidden West Complete Edition")
                except OSError:
                    pass
        except Exception:
            pass

    # 2. Epic Games Launcher manifests
    epic_manifests = Path(r"C:\ProgramData\Epic\EpicGamesLauncher\Data\Manifests")
    if epic_manifests.is_dir():
        for item_file in epic_manifests.glob("*.item"):
            try:
                data = json.loads(item_file.read_text(encoding="utf-8", errors="ignore"))
                loc = data.get("InstallLocation")
                if loc and "horizon" in str(loc).lower():
                    candidates.append(Path(loc))
            except Exception:
                pass

    # 3. Common Steam / Epic / GOG / Games folders across all mounted Windows drives
    rel_subdirs = (
        r"Program Files (x86)\Steam\steamapps\common\Horizon Forbidden West Complete Edition",
        r"SteamLibrary\steamapps\common\Horizon Forbidden West Complete Edition",
        r"Steam\steamapps\common\Horizon Forbidden West Complete Edition",
        r"Games\Horizon Forbidden West Complete Edition",
        r"Epic Games\HorizonForbiddenWest",
        r"Horizon Forbidden West Complete Edition",
    )
    for drv_letter in "CDEFGHIJKLMNOPQRSTUVWXYZ":
        drv = Path(f"{drv_letter}:\\")
        if not drv.exists():
            continue
        # Also parse Steam libraryfolders.vdf if present
        for vdf_rel in (r"Program Files (x86)\Steam\steamapps\libraryfolders.vdf", r"Steam\steamapps\libraryfolders.vdf"):
            vdf = drv / vdf_rel
            if vdf.is_file():
                try:
                    for m in re.finditer(r'"path"\s+"([^"]+)"', vdf.read_text(encoding="utf-8", errors="ignore")):
                        lib = Path(m.group(1).replace("\\\\", "\\"))
                        candidates.append(lib / "steamapps" / "common" / "Horizon Forbidden West Complete Edition")
                except Exception:
                    pass
        for rel in rel_subdirs:
            candidates.append(drv / rel)

    seen: set[str] = set()
    for cand in candidates:
        key = str(cand).lower()
        if key in seen:
            continue
        seen.add(key)
        if is_valid_hfw_dir(cand):
            return cand

    return DEFAULT_GAME_DIR


def auto_detect_blender_exe() -> Path:
    """Automatically locate the newest installed Blender 4.2+ / 5.x executable on this system."""
    if env_b := os.environ.get("HFW_BLENDER_EXE"):
        p = Path(env_b)
        if p.is_file():
            return p

    if DEFAULT_BLENDER_EXE.is_file():
        return DEFAULT_BLENDER_EXE

    found: list[Path] = []
    for pf_env in ("ProgramFiles", "ProgramFiles(x86)"):
        pf = os.environ.get(pf_env)
        if not pf:
            continue
        bf_dir = Path(pf) / "Blender Foundation"
        if bf_dir.is_dir():
            for sub in sorted(bf_dir.glob("Blender*"), reverse=True):
                exe = sub / "blender.exe"
                if exe.is_file():
                    found.append(exe)
        steam_b = Path(pf) / "Steam" / "steamapps" / "common" / "Blender" / "blender.exe"
        if steam_b.is_file():
            found.append(steam_b)

    if found:
        return found[0]

    if which_b := shutil.which("blender"):
        return Path(which_b)

    return DEFAULT_BLENDER_EXE


@dataclass
class ToolkitConfig:
    game_dir: Path = field(default_factory=auto_detect_game_dir)
    rtti_json: Path | None = None
    blender_exe: Path = field(default_factory=auto_detect_blender_exe)
    cache_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "cache")
    exports_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "exports")
    bin_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "bin")
    platform: str = "WinGame"
    auto_update: bool = False
    update_branch: str = "main"
    check_updates_on_startup: bool = True

    def __post_init__(self) -> None:
        self.game_dir = Path(self.game_dir)
        if not is_valid_hfw_dir(self.game_dir):
            detected = auto_detect_game_dir()
            if is_valid_hfw_dir(detected):
                self.game_dir = detected

        self.blender_exe = Path(self.blender_exe)
        if not self.blender_exe.is_file():
            detected_b = auto_detect_blender_exe()
            if detected_b.is_file():
                self.blender_exe = detected_b

        self.cache_dir = Path(self.cache_dir)
        self.exports_dir = Path(self.exports_dir)
        self.bin_dir = Path(self.bin_dir)
        if self.rtti_json is None:
            self.rtti_json = self.game_dir / "dump" / "hfw.json"
        else:
            self.rtti_json = Path(self.rtti_json)

    @property
    def toolkit_dir(self) -> Path:
        return PROJECT_ROOT

    @property
    def config_file_path(self) -> Path:
        return PROJECT_ROOT / "hfw_toolkit_config.json"

    @property
    def game_exe_path(self) -> Path:
        return self.game_dir / "HorizonForbiddenWest.exe"

    @property
    def game_exe(self) -> Path:
        return self.game_exe_path

    @property
    def local_cache_dir(self) -> Path:
        return self.game_dir / f"LocalCache{self.platform}"

    @property
    def package_dir(self) -> Path:
        return self.local_cache_dir / "package"

    @property
    def streaming_graph_path(self) -> Path:
        return self.package_dir / "streaming_graph.core"

    @property
    def dxcompiler_dll(self) -> Path:
        """Locate `dxcompiler.dll` from portable `HFWToolkit/bin/`, game `tools/`, or Windows 10/11 SDK."""
        candidates = [
            self.bin_dir / "dxcompiler.dll",
            self.game_dir / "tools" / "bin" / "x64" / "dxcompiler.dll",
        ]
        win_kits = Path(r"C:\Program Files (x86)\Windows Kits\10\bin")
        if win_kits.is_dir():
            for sdk_dxc in sorted(win_kits.glob("*/x64/dxcompiler.dll"), reverse=True):
                candidates.append(sdk_dxc)
        for c in candidates:
            if c.is_file():
                return c
        return candidates[0]

    @property
    def index_db_path(self) -> Path:
        return self.cache_dir / "hfw_index.sqlite"

    def resolve_device_path(self, device_path: str) -> Path:
        if ":" not in device_path:
            return self.game_dir / device_path
        device, rel = device_path.split(":", 1)
        rel = rel.lstrip("/\\")
        if device == "cache":
            return self.local_cache_dir / rel
        if device == "source":
            return self.game_dir / rel
        if device == "tools":
            return self.game_dir / "tools" / rel
        raise ValueError(f"Unknown Decima device path: {device_path!r}")

    def ensure_dirs(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        self.bin_dir.mkdir(parents=True, exist_ok=True)

    def diagnose(self) -> dict[str, Any]:
        """Run a complete portability & readiness health check across game files, RTTI, Blender, DXC, and index."""
        from hfw_toolkit.decima.exe_rtti import get_exe_version_string
        from hfw_toolkit.decima.rtti import BUILTIN_RTTI_GZ_PATH, RTTISchema

        sg_ok = self.streaming_graph_path.is_file()
        exe_ok = self.game_exe_path.is_file()
        exe_ver = get_exe_version_string(self.game_exe_path) if exe_ok else "not found"
        core_count = len(list(self.package_dir.glob("*.core"))) if self.package_dir.is_dir() else 0

        rtti_ok = False
        rtti_source = "none"
        rtti_count = 0
        try:
            schema = RTTISchema.load_for_game(self)
            rtti_ok = len(schema.raw) > 1000
            rtti_source = schema.source_label
            rtti_count = len(schema.raw)
        except Exception as exc:
            rtti_source = f"error: {exc}"

        dxc_path = self.dxcompiler_dll
        dxc_ok = dxc_path.is_file()
        blender_ok = self.blender_exe.is_file()
        index_ok = self.index_db_path.is_file()

        return {
            "ready": bool(sg_ok and rtti_ok),
            "toolkit_dir": str(self.toolkit_dir),
            "config_file": str(self.config_file_path),
            "config_file_exists": self.config_file_path.is_file(),
            "game_dir": str(self.game_dir),
            "game_dir_valid": sg_ok,
            "package_dir": str(self.package_dir),
            "game_exe_found": exe_ok,
            "game_version": exe_ver,
            "core_archive_count": core_count,
            "rtti_ok": rtti_ok,
            "rtti_source": rtti_source,
            "rtti_type_count": rtti_count,
            "builtin_rtti_available": BUILTIN_RTTI_GZ_PATH.is_file(),
            "blender_found": blender_ok,
            "blender_exe": str(self.blender_exe),
            "dxc_found": dxc_ok,
            "dxc_path": str(dxc_path),
            "index_db_found": index_ok,
            "index_db_path": str(self.index_db_path),
            "exports_dir": str(self.exports_dir),
            "auto_update": bool(self.auto_update),
            "update_branch": self.update_branch or "main",
            "check_updates_on_startup": bool(self.check_updates_on_startup),
        }

    @classmethod
    def load(cls, config_file: Path | None = None) -> "ToolkitConfig":
        cfg_path = config_file or (PROJECT_ROOT / "hfw_toolkit_config.json")
        kwargs: dict[str, Any] = {}
        if cfg_path.exists():
            try:
                data = json.loads(cfg_path.read_text(encoding="utf-8"))
                for key in ("game_dir", "rtti_json", "blender_exe", "cache_dir", "exports_dir", "bin_dir", "platform", "update_branch"):
                    if key in data and data[key]:
                        kwargs[key] = data[key]
                for bkey in ("auto_update", "check_updates_on_startup"):
                    if bkey in data and isinstance(data[bkey], bool):
                        kwargs[bkey] = data[bkey]
            except Exception:
                pass
        if env_game := os.environ.get("HFW_GAME_DIR"):
            kwargs["game_dir"] = env_game
        if env_blender := os.environ.get("HFW_BLENDER_EXE"):
            kwargs["blender_exe"] = env_blender
        return cls(**kwargs)

    def save(self, config_file: Path | None = None) -> Path:
        cfg_path = config_file or (PROJECT_ROOT / "hfw_toolkit_config.json")
        payload = {
            "game_dir": str(self.game_dir),
            "rtti_json": str(self.rtti_json) if self.rtti_json else "",
            "blender_exe": str(self.blender_exe),
            "cache_dir": str(self.cache_dir),
            "exports_dir": str(self.exports_dir),
            "bin_dir": str(self.bin_dir),
            "platform": self.platform,
            "auto_update": bool(self.auto_update),
            "update_branch": self.update_branch or "main",
            "check_updates_on_startup": bool(self.check_updates_on_startup),
        }
        cfg_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return cfg_path
