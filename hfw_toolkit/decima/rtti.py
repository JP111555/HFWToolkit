"""RTTI schema loader for Horizon Forbidden West PC (`dump/hfw.json`).

Handles:
- MurmurHash3 TypeID computation (`00000001_<TypeName>`, seed 42)
- Recursive base-class attribute collection
- Exact Decima deterministic LCG quicksort by attribute offset
- Post-sort filtering of `ATTR_DONT_SERIALIZE_BINARY` (flag bit 2)
- Recursive base-class inheritance checks (`is_subclass`, `msg_read_binary_owner`)
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hfw_toolkit.decima.exe_rtti import load_or_extract_exe_rtti
from hfw_toolkit.decima.hashes import decima_type_hash

ATTR_DONT_SERIALIZE_BINARY = 2
BUILTIN_RTTI_GZ_PATH = Path(__file__).resolve().parent.parent / "data" / "hfw_rtti_builtin.json.gz"


@dataclass(frozen=True, slots=True)
class AttrSpec:
    name: str
    type_name: str
    offset: int
    flags: int
    category: str | None = None


@dataclass(slots=True)
class EnumSpec:
    name: str
    size: int
    is_flags: bool
    val_to_name: dict[int, str]
    name_to_val: dict[str, int]


class RTTISchema:
    """Compiled Decima RTTI schema (from live `HorizonForbiddenWest.exe`, `dump/hfw.json`, or bundled built-in schema)."""

    def __init__(
        self,
        json_path: Path | str | dict[str, dict[str, Any]] | None = None,
        source_label: str = "",
        cache_dir: Path | None = None,
    ) -> None:
        self.source_label = source_label
        if isinstance(json_path, dict):
            self.json_path = Path("<in-memory>")
            self.raw: dict[str, dict] = json_path
            if not self.source_label:
                self.source_label = "in-memory schema"
        else:
            p = Path(json_path) if json_path is not None else BUILTIN_RTTI_GZ_PATH
            self.json_path = p
            if p.is_file():
                if p.name.endswith(".gz"):
                    self.raw = json.loads(gzip.decompress(p.read_bytes()).decode("utf-8"))
                else:
                    with open(p, "r", encoding="utf-8") as fp:
                        self.raw = json.load(fp)
                if not self.source_label:
                    self.source_label = "built-in HFW schema" if p == BUILTIN_RTTI_GZ_PATH else str(p)
            else:
                # Automatic zero-setup fallback when `dump/hfw.json` doesn't exist on a vanilla HFW install:
                # 1. Try static PE64 extraction from `<game_dir>/HorizonForbiddenWest.exe`
                # 2. Fall back to bundled `hfw_rtti_builtin.json.gz`
                cand_exe = p.parent.parent / "HorizonForbiddenWest.exe"
                if cand_exe.is_file():
                    self.raw, self.source_label = load_or_extract_exe_rtti(cand_exe, cache_dir=cache_dir)
                elif BUILTIN_RTTI_GZ_PATH.is_file():
                    self.raw = json.loads(gzip.decompress(BUILTIN_RTTI_GZ_PATH.read_bytes()).decode("utf-8"))
                    self.source_label = "built-in HFW schema (fallback)"
                else:
                    raise FileNotFoundError(f"RTTI schema not found at {p} and no HorizonForbiddenWest.exe or built-in schema available")

        self.hash_to_name: dict[int, str] = {}
        self.name_to_hash: dict[str, int] = {}
        self.enums: dict[str, EnumSpec] = {}
        self._atom_base_cache: dict[str, str] = {}
        self._ordered_attrs_cache: dict[str, list[AttrSpec]] = {}
        self._mrb_owner_cache: dict[str, str | None] = {}
        self._ancestors_cache: dict[str, frozenset[str]] = {}

        for name, info in self.raw.items():
            h = decima_type_hash(name)
            self.hash_to_name[h] = name
            self.name_to_hash[name] = h
            kind = info["kind"]
            if kind in ("enum", "enum flags"):
                v2n: dict[int, str] = {}
                n2v: dict[str, int] = {}
                for entry in info.get("values", []):
                    val = int(entry["value"])
                    ename = entry["name"]
                    v2n.setdefault(val, ename)
                    n2v[ename] = val
                self.enums[name] = EnumSpec(
                    name=name,
                    size=int(info["size"]),
                    is_flags=(kind == "enum flags"),
                    val_to_name=v2n,
                    name_to_val=n2v,
                )

    @classmethod
    def load_for_game(cls, config: Any) -> "RTTISchema":
        """Load the most accurate RTTI schema for `config.game_dir` without requiring prior setup:
        1. Explicit custom `rtti_json` if user configured a non-default existing file.
        2. Live/cached static extraction from `<game_dir>/HorizonForbiddenWest.exe` (auto-adapts to any HFW patch!).
        3. `<game_dir>/dump/hfw.json` if present.
        4. Bundled `hfw_toolkit/data/hfw_rtti_builtin.json.gz`.
        """
        rtti_path = getattr(config, "rtti_json", None)
        game_dir = Path(getattr(config, "game_dir", "."))
        cache_dir = getattr(config, "cache_dir", None)
        default_dump = game_dir / "dump" / "hfw.json"

        # 1. Explicit custom rtti_json override
        if rtti_path and Path(rtti_path) != default_dump and Path(rtti_path).is_file():
            return cls(Path(rtti_path), source_label=f"custom ({Path(rtti_path).name})", cache_dir=cache_dir)

        # 2. Live/cached static PE64 scan of HorizonForbiddenWest.exe (guarantees version match across patches!)
        exe_path = game_dir / "HorizonForbiddenWest.exe"
        if exe_path.is_file():
            try:
                raw, label = load_or_extract_exe_rtti(exe_path, cache_dir=cache_dir)
                return cls(raw, source_label=label, cache_dir=cache_dir)
            except Exception:
                pass

        # 3. Existing dump/hfw.json in game folder
        if default_dump.is_file():
            return cls(default_dump, source_label="dump/hfw.json", cache_dir=cache_dir)

        # 4. Bundled built-in schema
        return cls(BUILTIN_RTTI_GZ_PATH, source_label="built-in HFW schema", cache_dir=cache_dir)

    @property
    def types(self) -> dict[str, dict[str, Any]]:
        return self.raw

    def kind(self, type_name: str) -> str:
        info = self.raw.get(type_name)
        return info["kind"] if info is not None else "compound"

    def atom_base(self, type_name: str) -> str:
        cached = self._atom_base_cache.get(type_name)
        if cached is not None:
            return cached
        cur = type_name
        while True:
            bt = self.raw[cur].get("base_type")
            if not bt or bt == cur:
                break
            cur = bt
        self._atom_base_cache[type_name] = cur
        return cur

    def ancestors(self, type_name: str) -> frozenset[str]:
        cached = self._ancestors_cache.get(type_name)
        if cached is not None:
            return cached
        seen: set[str] = {type_name}
        stack = [type_name]
        while stack:
            cur = stack.pop()
            for base in self.raw.get(cur, {}).get("bases", []):
                bname = base["type"]
                if bname not in seen:
                    seen.add(bname)
                    stack.append(bname)
        res = frozenset(seen)
        self._ancestors_cache[type_name] = res
        return res

    def is_subclass(self, type_name: str, base_name: str) -> bool:
        return base_name in self.ancestors(type_name)

    def msg_read_binary_owner(self, type_name: str) -> str | None:
        """Return the class (self or ancestor) that defines `MsgReadBinary`, or None."""
        if type_name in self._mrb_owner_cache:
            return self._mrb_owner_cache[type_name]
        info = self.raw.get(type_name)
        if info is None:
            self._mrb_owner_cache[type_name] = None
            return None
        if "MsgReadBinary" in info.get("messages", ()):
            self._mrb_owner_cache[type_name] = type_name
            return type_name
        for base in info.get("bases", ()):
            owner = self.msg_read_binary_owner(base["type"])
            if owner is not None:
                self._mrb_owner_cache[type_name] = owner
                return owner
        self._mrb_owner_cache[type_name] = None
        return None

    def ordered_attrs(self, type_name: str) -> list[AttrSpec]:
        """Return serialized compound attributes in Decima's exact binary read order."""
        cached = self._ordered_attrs_cache.get(type_name)
        if cached is not None:
            return cached

        collected: list[AttrSpec] = []

        def collect(tname: str, base_offset: int) -> None:
            tinfo = self.raw.get(tname)
            if tinfo is None:
                return
            for base in tinfo.get("bases", ()):
                if base["offset"] < 0:
                    continue
                collect(base["type"], base_offset + int(base["offset"]))
            cur_cat: str | None = None
            for attr in tinfo.get("attrs", ()):
                if "category" in attr:
                    cur_cat = attr["category"]
                    continue
                collected.append(
                    AttrSpec(
                        name=attr["name"],
                        type_name=attr["type"],
                        offset=base_offset + int(attr["offset"]),
                        flags=int(attr["flags"]),
                        category=cur_cat,
                    )
                )

        collect(type_name, 0)
        _decima_quicksort(collected)
        result = [a for a in collected if (a.flags & ATTR_DONT_SERIALIZE_BINARY) == 0]
        self._ordered_attrs_cache[type_name] = result
        return result

    def enum_name(self, enum_type: str, value: int) -> str:
        spec = self.enums.get(enum_type)
        if spec is None:
            return str(value)
        if spec.is_flags:
            parts = [n for v, n in spec.val_to_name.items() if v != 0 and (value & v) == v]
            return "|".join(parts) if parts else spec.val_to_name.get(0, str(value))
        return spec.val_to_name.get(value, f"Unknown({value})")


def _decima_quicksort(items: list[AttrSpec]) -> None:
    """Exact replica of Decima's deterministic LCG-pivoted quicksort by attribute offset."""

    def qsort(left: int, right: int, state: int) -> int:
        if left < right:
            state = (0x19660D * state + 0x3C6EF35F) & 0xFFFFFFFF
            pivot = (state >> 8) % (right - left)
            items[left + pivot], items[right] = items[right], items[left + pivot]

            start = left - 1
            end = right
            pivot_off = items[right].offset
            while True:
                start += 1
                while start < end and items[start].offset < pivot_off:
                    start += 1
                end -= 1
                while end > start and pivot_off < items[end].offset:
                    end -= 1
                if start >= end:
                    break
                items[start], items[end] = items[end], items[start]

            items[start], items[right] = items[right], items[start]
            state = qsort(left, start - 1, state)
            state = qsort(start + 1, right, state)
        return state

    if len(items) > 1:
        qsort(0, len(items) - 1, 0)
