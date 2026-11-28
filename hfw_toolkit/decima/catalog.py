"""SQLite Asset Catalog & Cross-Reference Index (`cache/hfw_index.sqlite`).

Indexes:
- All ControlledEntityResources (machines, animals, characters) with friendly names + Guerrilla codenames
- All BodyVariants & SkinnedModelResources
- All TextureSets with original Guerrilla `work:` Substance/PSD paths & decoded RGBA packing
- All Shaders & RenderEffectResources (linked to TextureSets) for cross-reference queries
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable

from hfw_toolkit.decima.streaming_graph import HFWGame

# Curated Guerrilla internal codename -> Horizon Forbidden West display name map
# Verified directly against HFW PC TextureSet authoring paths & skeleton/model inspection
MACHINE_CODENAMES: dict[str, str] = {
    "alligator": "Snapmaw",
    "anteater": "Spikesnout",
    "antelope": "Grazer",
    "armadillo": "Rollerback",
    "baboon": "Clamberjaw",
    "bat": "Dreadwing",
    "beachlizard": "Snapmaw",
    "bear": "Frostclaw",
    "bighorn": "Charger",
    "boar": "Bristleback",
    "buffalo": "Behemoth",
    "bull": "Trampler",
    "camel": "Bellowback",
    "cargohopper": "Leaplasher",
    "cargorhino": "Behemoth",
    "cheetah": "Stalker",
    "cobra": "Slitherfang",
    "coyote": "Scrounger",
    "crab": "Shell-Walker",
    "cryobear": "Frostclaw",
    "deer": "Grazer",
    "dredger": "Plowhorn",
    "dredgerwar": "Grimhorn",
    "dungbeetle": "Widemaw",
    "eagle": "Stormbird",
    "elementalmink": "Burrower",
    "elephant": "Tremortusk",
    "firebear": "Fireclaw",
    "fly": "Stingspawn",
    "flyegg": "Stingspawn",
    "frog": "Bilegut",
    "giraffe": "Tallneck",
    "glider": "Glinthawk",
    "goat": "Lancehorn",
    "grey_wolf": "Ravager",
    "greywolf": "Ravager",
    "hackbot": "Corruptor",
    "hellhound": "Scorcher",
    "hippo": "Widemaw",
    "horse": "Charger",
    "horus": "Horus",
    "hyena": "Scrapper",
    "ibex": "Lancehorn",
    "infernobear": "Fireclaw",
    "jaguar": "Sawtooth",
    "kangaroo": "Leaplasher",
    "laserscout": "Redeye Watcher",
    "lizard": "Sunwing",
    "longlegbird": "Longleg",
    "mammoth": "Tremortusk",
    "mink": "Burrower",
    "mole": "Rockbreaker",
    "moth": "Stingspawn",
    "ostrich": "Longleg",
    "panther": "Scorcher",
    "pelican": "Waterwing",
    "plasmaspine": "Slaughterspine",
    "plesiosaur": "Tideripper",
    "pterosaur": "Sunwing",
    "raider": "Clawstrider",
    "raiderelemental": "Clawstrider",
    "raidermelee": "Clawstrider",
    "raptor": "Thunderjaw",
    "reindeer": "Fanghorn",
    "rhino": "Behemoth",
    "sabertooth": "Sawtooth",
    "scarab": "Corruptor",
    "scorpion": "Corruptor",
    "scout": "Redeye Watcher",
    "seabeast": "Tideripper",
    "servitorhuman": "Specter Prime",
    "servitorone": "Specter",
    "slitherfang": "Slitherfang",
    "snake": "Slitherfang",
    "spider": "Specter",
    "spinosaurus": "Slaughterspine",
    "spraybot": "Bellowback",
    "stag": "Plowhorn",
    "stalker": "Stalker",
    "stallion": "Strider",
    "stingray": "Tideripper",
    "sunwing": "Sunwing",
    "swooper": "Skydrifter",
    "thunderhawk": "Stormbird",
    "titan": "Horus",
    "trex": "Thunderjaw",
    "triceratops": "Grimhorn",
    "tunnelmink": "Burrower",
    "turtle": "Shellsnapper",
    "vulture": "Glinthawk",
    "warrobot": "Corruptor",
    "warthog": "Bristleback",
    "watcher": "Redeye Watcher",
    "waterwing": "Waterwing",
    "weasel": "Burrower",
    "wolf": "Ravager",
    "wolverine": "Scrounger",
}

_GENERIC_ROBOT_CODES: frozenset[str] = frozenset(
    {
        "shared",
        "common",
        "generic",
        "shared_textures",
        "detail",
        "weapons",
        "components",
        "generic_components",
        "accessories",
        "canisters",
        "canister",
        "ammo",
        "servitor_modules",
        "warrobot_modules",
        "flyegg_mine",
        "harvester",
    }
)

# Global common groups (Group 1 holds distant LOD impostor TextureSets for many machines,
# Group 2/3 hold shared foliage/environment textures) that must never determine an entity's identity.
_GLOBAL_SHARED_GROUPS: frozenset[int] = frozenset({1, 2, 3})

_ROBOT_PATH_RE = re.compile(rb"models/characters/robots/([A-Za-z0-9_]+)/", re.I)
_CHAR_PATH_RE = re.compile(rb"models/characters/([A-Za-z0-9_]+)/([A-Za-z0-9_]+)/", re.I)

_ETEXTURE_SET_TYPE = {
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
_ECHANNEL = {0: "R", 1: "G", 2: "B", 3: "A"}


def decode_packing_summary(packing_info: int) -> dict[str, str]:
    """Decode `TextureSetEntry.PackingInfo` uint32 into `{R: 'Color.R', ...}`."""
    out: dict[str, str] = {}
    for shift, lane in ((0, "R"), (8, "G"), (16, "B"), (24, "A")):
        byte = (packing_info >> shift) & 0xFF
        ttype = byte & 0x0F
        if ttype == 0:
            continue
        tname = _ETEXTURE_SET_TYPE.get(ttype, f"Type{ttype}")
        if (byte & 0x80) == 0:
            ch = _ECHANNEL.get((byte >> 4) & 0x7, "R")
            out[lane] = f"{tname}.{ch}"
        else:
            out[lane] = tname
    return out


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entities (
    group_id INTEGER NOT NULL,
    object_index INTEGER NOT NULL,
    entity_type TEXT NOT NULL,
    category TEXT NOT NULL,
    codename TEXT NOT NULL,
    display_name TEXT NOT NULL,
    has_skeleton INTEGER NOT NULL,
    has_destructibility INTEGER NOT NULL,
    part_count INTEGER NOT NULL,
    texture_hints TEXT NOT NULL,
    PRIMARY KEY (group_id, object_index)
);

CREATE TABLE IF NOT EXISTS texture_sets (
    group_id INTEGER NOT NULL,
    object_index INTEGER NOT NULL,
    name TEXT NOT NULL,
    paths_json TEXT NOT NULL,
    packing_json TEXT NOT NULL,
    category TEXT NOT NULL,
    codename TEXT NOT NULL,
    PRIMARY KEY (group_id, object_index)
);

CREATE TABLE IF NOT EXISTS shaders (
    group_id INTEGER NOT NULL,
    object_index INTEGER NOT NULL,
    shader_hash TEXT NOT NULL,
    data_size INTEGER NOT NULL,
    PRIMARY KEY (group_id, object_index)
);

CREATE TABLE IF NOT EXISTS entity_texture_sets (
    entity_group_id INTEGER NOT NULL,
    ts_group_id INTEGER NOT NULL,
    ts_object_index INTEGER NOT NULL,
    PRIMARY KEY (entity_group_id, ts_group_id, ts_object_index)
);

CREATE TABLE IF NOT EXISTS entity_variants (
    entity_group_id INTEGER NOT NULL,
    entity_object_index INTEGER NOT NULL,
    variant_index INTEGER NOT NULL,
    label TEXT NOT NULL,
    destructibility_id TEXT NOT NULL,
    smr_id TEXT NOT NULL,
    part_count INTEGER NOT NULL,
    PRIMARY KEY (entity_group_id, entity_object_index, variant_index)
);

CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(display_name, codename);
CREATE INDEX IF NOT EXISTS idx_entities_cat ON entities(category);
CREATE INDEX IF NOT EXISTS idx_ts_name ON texture_sets(name);
CREATE INDEX IF NOT EXISTS idx_ts_codename ON texture_sets(codename);
CREATE INDEX IF NOT EXISTS idx_ets_ts ON entity_texture_sets(ts_group_id, ts_object_index);
CREATE INDEX IF NOT EXISTS idx_ev_entity ON entity_variants(entity_group_id, entity_object_index);

CREATE TABLE IF NOT EXISTS audio_music_cues (
    cue_id TEXT PRIMARY KEY,
    project_name TEXT NOT NULL,
    cue_name TEXT NOT NULL,
    category TEXT NOT NULL,
    dev_folder TEXT NOT NULL,
    branch_path TEXT NOT NULL,
    initial_tempo REAL NOT NULL,
    duration_sec REAL NOT NULL,
    stem_count INTEGER NOT NULL,
    play_as_one_shot INTEGER NOT NULL,
    is_transition INTEGER NOT NULL,
    tempo_events_json TEXT NOT NULL,
    stems_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audio_groups (
    group_id INTEGER NOT NULL,
    kind TEXT NOT NULL,
    category TEXT NOT NULL,
    owner_name TEXT NOT NULL,
    dev_folder TEXT NOT NULL,
    item_count INTEGER NOT NULL,
    indexed_items INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (group_id, kind)
);

CREATE TABLE IF NOT EXISTS audio_voices (
    voice_id TEXT PRIMARY KEY,
    speaker TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audio_assets (
    object_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    group_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    dev_folder TEXT NOT NULL,
    filename_stem TEXT NOT NULL,
    sample_rate INTEGER NOT NULL,
    channels INTEGER NOT NULL,
    sample_count INTEGER NOT NULL,
    duration_sec REAL NOT NULL,
    encoding INTEGER NOT NULL,
    encoding_name TEXT NOT NULL,
    data_size INTEGER NOT NULL,
    is_streaming INTEGER NOT NULL,
    sound_graph_name TEXT NOT NULL,
    owner_name TEXT NOT NULL,
    speaker TEXT NOT NULL,
    subtitle TEXT NOT NULL,
    languages_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_amc_cat ON audio_music_cues(category, project_name);
CREATE INDEX IF NOT EXISTS idx_amc_name ON audio_music_cues(cue_name);
CREATE INDEX IF NOT EXISTS idx_ag_kind ON audio_groups(kind, category);
CREATE INDEX IF NOT EXISTS idx_ag_folder ON audio_groups(dev_folder);
CREATE INDEX IF NOT EXISTS idx_aa_kind_grp ON audio_assets(kind, group_id);
CREATE INDEX IF NOT EXISTS idx_aa_speaker ON audio_assets(speaker);
CREATE INDEX IF NOT EXISTS idx_aa_name ON audio_assets(name);

CREATE TABLE IF NOT EXISTS characters (
    char_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    character_kind TEXT NOT NULL,
    tribe TEXT NOT NULL,
    gender TEXT NOT NULL,
    dev_folder TEXT NOT NULL,
    body_variant_id TEXT NOT NULL,
    facial_id TEXT NOT NULL,
    hair_id TEXT NOT NULL,
    outfit_count INTEGER NOT NULL,
    part_count INTEGER NOT NULL,
    tags TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS character_outfits (
    outfit_id TEXT PRIMARY KEY,
    char_id TEXT NOT NULL,
    display_name TEXT NOT NULL,
    tribe TEXT NOT NULL,
    dev_folder TEXT NOT NULL,
    resource_type TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    facial_id TEXT NOT NULL,
    has_headgear_toggle INTEGER NOT NULL,
    is_dlc INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS character_heads (
    facial_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    gender TEXT NOT NULL,
    dev_folder TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dev_assets (
    dev_path TEXT PRIMARY KEY,
    asset_id TEXT NOT NULL,
    parent_dir TEXT NOT NULL,
    category TEXT NOT NULL,
    display_name TEXT NOT NULL,
    resource_type TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    group_id INTEGER NOT NULL,
    is_skinned INTEGER NOT NULL,
    tset_count INTEGER NOT NULL,
    shader_count INTEGER NOT NULL,
    is_dlc INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_char_kind ON characters(character_kind, tribe, display_name);
CREATE INDEX IF NOT EXISTS idx_co_char ON character_outfits(char_id, display_name);
CREATE INDEX IF NOT EXISTS idx_ch_gender ON character_heads(gender, display_name);
CREATE INDEX IF NOT EXISTS idx_da_parent ON dev_assets(parent_dir);
CREATE INDEX IF NOT EXISTS idx_da_cat ON dev_assets(category, display_name);
CREATE INDEX IF NOT EXISTS idx_da_aid ON dev_assets(asset_id);
CREATE INDEX IF NOT EXISTS idx_da_rid ON dev_assets(resource_id);
"""


def _refine_robot_codename(code: str, text: str) -> str:
    low = text.lower()
    if code == "dlc1":
        if "inferno" in low or "firebear" in low:
            return "infernobear"
        if "cryo" in low or "frost" in low:
            return "cryobear"
    return code


def _codename_to_display(codename: str) -> str:
    low = codename.lower()
    if low in MACHINE_CODENAMES:
        return MACHINE_CODENAMES[low]
    # Check partial keys (e.g. "plasmaspine" inside "plasmaspine_apex")
    for k, v in MACHINE_CODENAMES.items():
        if k == low or low.startswith(k + "_"):
            return v
    return codename.replace("_", " ").title()


class AssetCatalog:
    """SQLite-backed searchable catalog of HFW machines, entities, TextureSets, and shaders."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA_SQL)
        self._migrate_machine_catalog_v12()
        self._migrate_horus_v13()

    def _migrate_horus_v13(self) -> None:
        """Idempotent fast migration registering Burning Shores FAS-BOR7 Horus (`14110:7066`) when DLC assets exist."""
        try:
            row = self.conn.execute("SELECT value FROM meta WHERE key = 'horus_v13'").fetchone()
            if row is not None and row["value"] == "1":
                return
            # Only inject Horus if Burning Shores Titan TextureSets exist in this indexed game install
            ts_rows = self.conn.execute(
                "SELECT group_id, object_index, name FROM texture_sets WHERE lower(codename) = 'titan' LIMIT 50"
            ).fetchall()
            if not ts_rows:
                return
            sample_hints = [r["name"] for r in ts_rows[:5]]
            with self.conn:
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO entities
                    VALUES (14110, 7066, 'EntityResource', 'machine', 'titan', 'Horus', 1, 1, 1864, ?)
                    """,
                    (json.dumps(sample_hints),),
                )
                self.conn.executemany(
                    "INSERT OR IGNORE INTO entity_texture_sets VALUES (14110, ?, ?)",
                    [(int(r["group_id"]), int(r["object_index"])) for r in ts_rows],
                )
                self.conn.execute(
                    "DELETE FROM entity_variants WHERE entity_group_id = 14110 AND entity_object_index = 7066"
                )
                self.conn.executemany(
                    "INSERT OR REPLACE INTO entity_variants VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [
                        (14110, 7066, 0, "Horus — Complete Boss (1,229 joints, 1,864 hull parts + 102 turrets & heat sinks)", "14110:1000", "1624:64", 1864),
                        (14110, 7066, 1, "Horus — Clean Hull / Cinematic (1,229 joints, 1,812 hull parts)", "", "68784:11", 1812),
                        (14110, 7066, 2, "Horus — Drill Tentacles Boss Phase (254 joints, 258 parts + 18 drills)", "23470:402", "23470:239", 258),
                    ],
                )
                self.conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('horus_v13', '1')")
        except Exception:
            pass

    def _migrate_machine_catalog_v12(self) -> None:
        """Idempotent fast migration fixing swapped Skydrifter/Glinthawk names and stale contaminated variants."""
        try:
            row = self.conn.execute("SELECT value FROM meta WHERE key = 'codename_fix_v12'").fetchone()
            if row is not None and row["value"] == "1":
                return
            with self.conn:
                # Fix swapped glider (Glinthawk, 4360:48) vs swooper (Skydrifter, 3155:76) display names
                self.conn.execute(
                    "UPDATE entities SET display_name = 'Glinthawk' WHERE lower(codename) = 'glider' AND display_name = 'Skydrifter'"
                )
                self.conn.execute(
                    "UPDATE entities SET display_name = 'Skydrifter' WHERE lower(codename) = 'swooper' AND display_name = 'Glinthawk'"
                )
                # Remove known cross-species contaminated cached variant rows (Stingspawn<-Bilegut, Stormbird<-Dreadwing)
                self.conn.execute(
                    "DELETE FROM entity_variants WHERE entity_group_id = 1859 AND (lower(label) LIKE '%bilegut%' OR destructibility_id LIKE '12669:%' OR destructibility_id LIKE '7278:%')"
                )
                self.conn.execute(
                    "DELETE FROM entity_variants WHERE entity_group_id = 2622 AND (lower(label) LIKE '%dreadwing%' OR destructibility_id LIKE '81822:%')"
                )
                self.conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('codename_fix_v12', '1')")
        except Exception:
            pass

    def close(self) -> None:
        self.conn.close()

    def is_built(self) -> bool:
        row = self.conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        return row is not None and row["value"] == "3"

    def stats(self) -> dict[str, int]:
        out = {}
        for tbl in ("entities", "texture_sets", "shaders", "entity_texture_sets"):
            out[tbl] = self.conn.execute(f"SELECT COUNT(*) AS c FROM {tbl}").fetchone()["c"]
        out["machines"] = self.conn.execute(
            "SELECT COUNT(*) AS c FROM entities WHERE category = 'machine'"
        ).fetchone()["c"]
        out["music_cues"] = self.conn.execute("SELECT COUNT(*) AS c FROM audio_music_cues").fetchone()["c"]
        out["sfx_groups"] = self.conn.execute(
            "SELECT COUNT(*) AS c FROM audio_groups WHERE kind = 'sfx'"
        ).fetchone()["c"]
        out["voiceline_groups"] = self.conn.execute(
            "SELECT COUNT(*) AS c FROM audio_groups WHERE kind = 'voiceline'"
        ).fetchone()["c"]
        out["characters"] = self.conn.execute("SELECT COUNT(*) AS c FROM characters").fetchone()["c"]
        out["character_outfits"] = self.conn.execute("SELECT COUNT(*) AS c FROM character_outfits").fetchone()["c"]
        out["character_heads"] = self.conn.execute("SELECT COUNT(*) AS c FROM character_heads").fetchone()["c"]
        out["dev_assets"] = self.conn.execute("SELECT COUNT(*) AS c FROM dev_assets").fetchone()["c"]
        return out

    def build(
        self,
        game: HFWGame,
        progress_cb: Callable[[str], None] | None = None,
    ) -> dict[str, int]:
        """Build the complete SQLite catalog from the live HFW streaming graph."""
        t0 = time.time()
        log = progress_cb or (lambda _msg: None)
        graph = game.graph
        names_by_hash = graph.type_names_by_hash
        tt_idx = graph.type_table_indices

        # Clear existing rows for clean rebuild
        with self.conn:
            for tbl in ("entities", "texture_sets", "shaders", "entity_texture_sets", "entity_variants", "meta"):
                self.conn.execute(f"DELETE FROM {tbl}")

        # Pass 1: Scan streaming graph metadata to locate target groups
        ce_groups: list[tuple[int, list[int]]] = []
        ts_groups: list[int] = []
        shader_groups: list[int] = []

        for gm in graph.group_list:
            ts, tc = gm.type_start, gm.type_count
            ce_indices = []
            has_ts = False
            has_sh = False
            for i in range(tc):
                tname = names_by_hash[tt_idx[ts + i]]
                if tname == "ControlledEntityResource":
                    ce_indices.append(i)
                elif tname == "TextureSet":
                    has_ts = True
                elif tname == "ShaderResource":
                    has_sh = True
            if ce_indices:
                ce_groups.append((gm.group_id, ce_indices))
            if has_ts:
                ts_groups.append(gm.group_id)
            if has_sh:
                shader_groups.append(gm.group_id)

        log(
            f"Graph scan: {len(ce_groups)} ControlledEntity groups, "
            f"{len(ts_groups)} TextureSet groups, {len(shader_groups)} Shader groups"
        )

        # Pass 2: Index all TextureSets (fast flat reads — no recursive subgroup loading needed!)
        ts_rows: list[tuple] = []
        ts_by_group: dict[int, list[tuple[int, str, str]]] = {}  # gid -> [(idx, category, codename)]
        for idx_g, gid in enumerate(ts_groups):
            try:
                objs = game.read_group(gid, read_subgroups=False)
            except Exception:
                continue
            for oidx, obj in enumerate(objs):
                if obj.get("_type") != "TextureSet":
                    continue
                paths = []
                for desc in obj.get("TextureDesc", ()):
                    p = desc.get("Path", "")
                    if p:
                        clean = p[5:] if p.lower().startswith("work:") else p
                        paths.append({"type": _ETEXTURE_SET_TYPE.get(desc.get("TextureType", 0), "?"), "path": clean})
                packings = [decode_packing_summary(e.get("PackingInfo", 0)) for e in obj.get("Entries", ())]
                first_path = paths[0]["path"] if paths else f"TextureSet_{gid}_{oidx}"
                short_name = Path(first_path).stem if paths else first_path

                # Categorize from path
                low = first_path.lower().replace("\\", "/")
                cat = "environment"
                code = ""
                m = re.search(r"models/characters/robots/([^/]+)/", low)
                if m:
                    cat = "machine"
                    code = _refine_robot_codename(m.group(1), low)
                else:
                    m2 = re.search(r"models/characters/([^/]+)/([^/]+)/", low)
                    if m2:
                        cat = m2.group(1)
                        code = m2.group(2)
                    elif "weapons" in low:
                        cat = "weapon"

                ts_rows.append(
                    (gid, oidx, short_name, json.dumps(paths), json.dumps(packings), cat, code)
                )
                ts_by_group.setdefault(gid, []).append((oidx, cat, code))
            if (idx_g + 1) % 500 == 0:
                log(f"Indexed TextureSets: {idx_g + 1}/{len(ts_groups)} groups ({len(ts_rows)} sets)...")

        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO texture_sets VALUES (?, ?, ?, ?, ?, ?, ?)", ts_rows
            )
        log(f"Indexed {len(ts_rows)} TextureSets in {time.time() - t0:.1f}s")

        # Pass 3: Index all ControlledEntityResource groups (machines, wildlife, characters)
        entity_rows, ets_rows = self._classify_entities(game, ce_groups, ts_by_group)

        # Pass 4: Index ShaderResource locations (just from graph metadata — ultra fast!)
        shader_rows: list[tuple] = []
        for gid in shader_groups:
            gm = graph.group(gid)
            for i in range(gm.type_count):
                if names_by_hash[tt_idx[gm.type_start + i]] == "ShaderResource":
                    shader_rows.append((gid, i, "", 0))

        variant_rows = self._index_machine_variants(game, entity_rows)

        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO entities VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", entity_rows
            )
            self.conn.executemany(
                "INSERT OR IGNORE INTO entity_texture_sets VALUES (?, ?, ?)", ets_rows
            )
            self.conn.executemany(
                "INSERT OR REPLACE INTO entity_variants VALUES (?, ?, ?, ?, ?, ?, ?)", variant_rows
            )
            self.conn.executemany(
                "INSERT OR REPLACE INTO shaders VALUES (?, ?, ?, ?)", shader_rows
            )
            self.conn.execute("INSERT OR REPLACE INTO meta VALUES ('schema_version', '3')")
            self.conn.execute(
                "INSERT OR REPLACE INTO meta VALUES ('built_at', ?)", (str(int(time.time())),)
            )

        self.build_audio_index(game, progress_cb=log)
        self.index_characters_and_assets(game, progress_cb=log)
        st = self.stats()
        log(f"Catalog build complete in {time.time() - t0:.1f}s: {st}")
        return st

    def _classify_entities(
        self,
        game: HFWGame,
        ce_groups: list[tuple[int, list[int]]],
        ts_by_group: dict[int, list[tuple[int, str, str]]],
    ) -> tuple[list[tuple], list[tuple]]:
        """Classify ControlledEntityResource groups cleanly without Group 1/2/3 impostor bleed."""
        graph = game.graph
        names_by_hash = graph.type_names_by_hash
        tt_idx = graph.type_table_indices

        entity_rows: list[tuple] = []
        ets_rows: list[tuple] = []

        def collect_tree_bfs(root_gid: int) -> dict[int, int]:
            dist: dict[int, int] = {root_gid: 0}
            queue = [root_gid]
            while queue:
                cur = queue.pop(0)
                d = dist[cur]
                for nxt in graph.group_subgroups(cur):
                    if nxt in _GLOBAL_SHARED_GROUPS:
                        continue
                    if nxt not in dist:
                        dist[nxt] = d + 1
                        queue.append(nxt)
            return dist

        for gid, ce_indices in ce_groups:
            tree_dist = collect_tree_bfs(gid)
            robot_scores: dict[str, float] = {}
            other_scores: dict[str, float] = {}
            has_skel = 0
            has_destr = 0
            part_count = 0

            for sgid, depth in tree_dist.items():
                gm = graph.group(sgid)
                for i in range(gm.type_count):
                    tn = names_by_hash[tt_idx[gm.type_start + i]]
                    if tn == "Skeleton":
                        has_skel = 1
                    elif tn == "DestructibilityResource":
                        has_destr = 1
                    elif tn == "DestructibilityPart":
                        part_count += 1
                weight = 100.0 / (depth + 1)
                for oidx, cat, code in ts_by_group.get(sgid, ()):
                    ets_rows.append((gid, sgid, oidx))
                    if cat == "machine" and code:
                        if code not in _GENERIC_ROBOT_CODES:
                            robot_scores[code] = robot_scores.get(code, 0.0) + weight
                    elif code:
                        key = f"{cat}/{code}"
                        other_scores[key] = other_scores.get(key, 0.0) + weight

            # Check 1-hop & 2-hop non-world supergroups for multi-variant machines (e.g. Clawstrider, Ravager, Charger)
            # AND to record accurate Destructibility part counts!
            super_robot_counts: dict[str, int] = {}
            super_max_parts = part_count
            sup_seen: set[int] = {gid}
            sup_q: list[tuple[int, int]] = [(s, 1) for s in graph.supergroups.get(gid, ())]
            while sup_q:
                cur_s, d_s = sup_q.pop(0)
                if cur_s in sup_seen or d_s > 2:
                    continue
                sup_seen.add(cur_s)
                gm_s = graph.group(cur_s)
                if gm_s.num_objects > 25_000:
                    continue
                s_parts = 0
                for i in range(gm_s.type_count):
                    tn = names_by_hash[tt_idx[gm_s.type_start + i]]
                    if tn == "DestructibilityResource":
                        has_destr = 1
                    elif tn == "DestructibilityPart":
                        s_parts += 1
                if s_parts > super_max_parts:
                    super_max_parts = s_parts
                for oidx, cat, code in ts_by_group.get(cur_s, ()):
                    if cat == "machine" and code and code not in _GENERIC_ROBOT_CODES:
                        super_robot_counts[code] = super_robot_counts.get(code, 0) + 1
                if d_s < 2:
                    for nxt_s in graph.supergroups.get(cur_s, ()):
                        if nxt_s not in sup_seen:
                            sup_q.append((nxt_s, d_s + 1))

            # Also check 1-hop subgroups of immediate supergroups if still empty (e.g. Ravager 5051, Charger 2911)
            if not robot_scores and not other_scores and not super_robot_counts:
                for sup1 in graph.supergroups.get(gid, ()):
                    if graph.group(sup1).num_objects > 25_000:
                        continue
                    for sg in graph.group_subgroups(sup1):
                        if sg in _GLOBAL_SHARED_GROUPS:
                            continue
                        for _oidx, cat, code in ts_by_group.get(sg, ()):
                            if cat == "machine" and code and code not in _GENERIC_ROBOT_CODES:
                                super_robot_counts[code] = super_robot_counts.get(code, 0) + 1

            # Pick primary identity:
            # 1. Non-generic machine TextureSets in the entity's own subgroup tree (ignoring Group 1/2/3!)
            # 2. Non-machine (human/animal) TextureSets in subtree
            # 3. Supergroup machine TextureSets ONLY when a single machine dominates (>= 50% of hits)
            if robot_scores:
                codename = max(robot_scores.items(), key=lambda kv: kv[1])[0]
                category = "machine"
                display_name = _codename_to_display(codename)
                part_count = max(part_count, super_max_parts)
            elif other_scores:
                best = max(other_scores.items(), key=lambda kv: kv[1])[0]
                category, codename = best.split("/", 1)
                display_name = _codename_to_display(codename)
            elif super_robot_counts:
                total_hits = sum(super_robot_counts.values())
                top_code, top_hits = max(super_robot_counts.items(), key=lambda kv: kv[1])
                # Require clear dominance (or shared mount pair horse/goat) so generic templates stay 'entity'
                if top_hits / max(total_hits, 1) >= 0.50 or set(super_robot_counts) <= {"horse", "goat"}:
                    codename = top_code
                    category = "machine"
                    display_name = _codename_to_display(codename)
                    part_count = max(part_count, super_max_parts)
                else:
                    category = "entity"
                    codename = f"group_{gid}"
                    display_name = f"Entity Group {gid}"
            else:
                category = "entity"
                codename = f"group_{gid}"
                display_name = f"Entity Group {gid}"

            cur_rows = self.conn.execute(
                "SELECT name FROM texture_sets WHERE codename = ? LIMIT 5", (codename,)
            ).fetchall()
            sample_paths = [r["name"] for r in cur_rows]

            for ce_idx in ce_indices:
                entity_rows.append(
                    (
                        gid,
                        ce_idx,
                        "ControlledEntityResource",
                        category,
                        codename,
                        display_name,
                        has_skel,
                        has_destr,
                        part_count,
                        json.dumps(sample_paths),
                    )
                )

        # Pass 3b: Validate & deduplicate 'machine' entries using actual ControlledEntityResource components!
        # Eliminates child turrets (3-12 joints), wild animals (no DestructibleArmor), cutscene props, and
        # duplicate partial CEs so every species appears ONCE with its complete rigged model.
        from hfw_toolkit.decima.reader import ObjectId

        _SHARED_SUBSPECIES_NAMES: dict[int, tuple[str, str]] = {
            1587: ("raider", "Clawstrider"),
            4262: ("antelope_reindeer", "Grazer / Fanghorn"),
            2911: ("horse_goat", "Charger / Lancehorn"),
            3385: ("dredger_war", "Plowhorn / Grimhorn"),
        }

        validated_rows: list[tuple] = []
        machine_candidates: list[tuple[int, tuple[int, int, int, int, int], list[Any]]] = []

        for row in entity_rows:
            row_list = list(row)
            gid, ce_idx, _etype, cat, code, disp, has_skel, has_destr, pcnt, hints = row_list
            if cat != "machine":
                validated_rows.append(tuple(row_list))
                continue

            if gid in _SHARED_SUBSPECIES_NAMES:
                code, disp = _SHARED_SUBSPECIES_NAMES[gid]
                row_list[4] = code
                row_list[5] = disp

            # Inspect this candidate ControlledEntityResource's real components & skeleton joint count
            has_destr_armor = 0
            has_ai_nav = 0
            has_weapons = 0
            max_joints = 0
            has_body_smr = 0
            direct_parts = 0
            try:
                ce_obj = game.read_object(ObjectId(gid, ce_idx), read_subgroups=True)
                for cref in ce_obj.get("EntityComponentResources", ()):
                    comp = game.resolve_ref(cref)
                    if not comp:
                        continue
                    ct = comp.get("_type", "")
                    if ct == "DestructibleArmorComponentResource":
                        has_destr_armor = 1
                    elif ct == "AINavMeshComponentResource":
                        has_ai_nav = 1
                    elif ct == "WeaponsContainerComponentResource":
                        has_weapons = 1
                    elif ct == "SkinnedModelResource":
                        if len(comp.get("ModelPartResources", ())) > 0:
                            has_body_smr = 1
                        sk = game.resolve_ref(comp.get("Skeleton"))
                        if sk:
                            max_joints = max(max_joints, len(sk.get("Joints", ())))
                    elif ct == "GraphAnimationManagerResource":
                        sk = game.resolve_ref(comp.get("Skeleton"))
                        if sk:
                            max_joints = max(max_joints, len(sk.get("Joints", ())))
                    elif ct == "DestructibilityResource":
                        direct_parts = max(direct_parts, len(comp.get("ConvertedParts", ())))
            except Exception:
                pass

            # Also check Skeletons inside gid itself if GraphAnimationManager didn't reference one directly
            if max_joints == 0 and has_skel:
                try:
                    for o in game.read_group(gid, read_subgroups=False):
                        if o.get("_type") == "Skeleton":
                            max_joints = max(max_joints, len(o.get("Joints", ())))
                except Exception:
                    pass

            eff_parts = max(pcnt, direct_parts)
            row_list[8] = eff_parts

            # Classify whether this CE is a real complete machine vs a child turret / cutscene prop / wild animal
            is_tiny_species = code in ("fly", "flyegg", "moth")
            is_peaceful_machine = code in ("giraffe",)  # Tallneck
            is_valid_machine = (
                (max_joints >= 15 or (is_tiny_species and max_joints >= 4))
                and (
                    has_destr_armor == 1
                    or (is_peaceful_machine and has_body_smr == 1)
                    or (is_tiny_species and (has_ai_nav == 1 or has_body_smr == 1))
                )
            )
            if not is_valid_machine:
                row_list[3] = "machine_subpart" if (has_destr_armor or has_weapons or max_joints >= 3) else "entity"
                validated_rows.append(tuple(row_list))
                continue

            score = (
                has_destr_armor,
                has_ai_nav,
                1 if max_joints >= 25 else 0,
                eff_parts,
                max_joints,
            )
            idx_pos = len(validated_rows)
            validated_rows.append(tuple(row_list))
            machine_candidates.append((idx_pos, score, row_list))

        # Deduplicate valid machines by display_name — keep the single canonical complete entity per species!
        best_by_species: dict[str, tuple[int, tuple[int, int, int, int, int]]] = {}
        max_parts_by_species: dict[str, int] = {}
        for idx_pos, score, row_list in machine_candidates:
            disp_key = str(row_list[5]).strip().lower()
            max_parts_by_species[disp_key] = max(max_parts_by_species.get(disp_key, 0), int(row_list[8]))
            prev = best_by_species.get(disp_key)
            if prev is None or score > prev[1]:
                best_by_species[disp_key] = (idx_pos, score)

        winners = {v[0] for v in best_by_species.values()}
        for idx_pos, _score, row_list in machine_candidates:
            disp_key = str(row_list[5]).strip().lower()
            if idx_pos in winners:
                # Inherit highest known variant part count for clean UI display (e.g. Slitherfang 4658:59 -> 271 parts!)
                row_list[8] = max(int(row_list[8]), max_parts_by_species.get(disp_key, 0))
                validated_rows[idx_pos] = tuple(row_list)
            else:
                row_list[3] = "machine_subpart"
                validated_rows[idx_pos] = tuple(row_list)

        # Inject Burning Shores colossal FAS-BOR7 Horus (`14110:7066`, EntityResource — not ControlledEntityResource!)
        # whenever Burning Shores archives (`group 14110` + `group 1624`) are present in the live streaming graph.
        try:
            gm_h = graph.group(14110)
            graph.group(1624)
            if 7066 < gm_h.num_objects and names_by_hash[tt_idx[gm_h.type_start + 7066]] == "EntityResource":
                titan_ts_names = [
                    r["name"]
                    for r in self.conn.execute(
                        "SELECT name FROM texture_sets WHERE lower(codename) = 'titan' LIMIT 5"
                    ).fetchall()
                ]
                validated_rows.append(
                    (
                        14110,
                        7066,
                        "EntityResource",
                        "machine",
                        "titan",
                        "Horus",
                        1,
                        1,
                        1864,
                        json.dumps(titan_ts_names),
                    )
                )
                for tgid, tlist in ts_by_group.items():
                    for oidx, _tcat, tcode in tlist:
                        if tcode.lower() == "titan":
                            ets_rows.append((14110, tgid, oidx))
        except Exception:
            pass

        return validated_rows, ets_rows

    def _index_machine_variants(
        self,
        game: HFWGame,
        entity_rows: list[tuple],
    ) -> list[tuple]:
        """Pre-index all named variants for every cataloged machine so Blender Add-on & UI can populate Variant dropdowns instantly."""
        from hfw_toolkit.decima.reader import ObjectId
        from hfw_toolkit.extract.machine_assembler import discover_machine_variants

        variant_rows: list[tuple] = []
        for row in entity_rows:
            gid, ce_idx, _etype, cat = row[0], row[1], row[2], row[3]
            if cat != "machine":
                continue
            try:
                _ce, _smr, variants = discover_machine_variants(game, ObjectId(gid, ce_idx))
            except Exception:
                variants = []
            if not variants:
                variant_rows.append((gid, ce_idx, 0, "Standard (Default)", "", "", int(row[8])))
                continue
            for v in variants:
                variant_rows.append(
                    (
                        gid,
                        ce_idx,
                        int(v.variant_index),
                        str(v.label),
                        str(v.destructibility_id),
                        str(v.smr_id or ""),
                        int(v.part_count),
                    )
                )
        return variant_rows

    def refresh_entities(self, game: HFWGame) -> dict[str, int]:
        """Fast (<5s) re-classification of all entities, machine variants & dlc1 TextureSets without re-scanning all 10,660 TextureSets."""
        graph = game.graph
        names_by_hash = graph.type_names_by_hash
        tt_idx = graph.type_table_indices

        # Refine any legacy 'dlc1' TextureSet codenames into 'cryobear' (Frostclaw) / 'infernobear' (Fireclaw)
        dlc1_rows = self.conn.execute(
            "SELECT group_id, object_index, name, paths_json FROM texture_sets WHERE codename = 'dlc1'"
        ).fetchall()
        updates = []
        for r in dlc1_rows:
            refined = _refine_robot_codename("dlc1", f"{r['name']} {r['paths_json']}")
            if refined != "dlc1":
                updates.append((refined, r["group_id"], r["object_index"]))
        if updates:
            with self.conn:
                self.conn.executemany(
                    "UPDATE texture_sets SET codename = ? WHERE group_id = ? AND object_index = ?",
                    updates,
                )

        ts_by_group: dict[int, list[tuple[int, str, str]]] = {}
        for r in self.conn.execute("SELECT group_id, object_index, category, codename FROM texture_sets"):
            ts_by_group.setdefault(r["group_id"], []).append((r["object_index"], r["category"], r["codename"]))

        ce_groups: list[tuple[int, list[int]]] = []
        for gm in graph.group_list:
            ts, tc = gm.type_start, gm.type_count
            ce_indices = [i for i in range(tc) if names_by_hash[tt_idx[ts + i]] == "ControlledEntityResource"]
            if ce_indices:
                ce_groups.append((gm.group_id, ce_indices))

        entity_rows, ets_rows = self._classify_entities(game, ce_groups, ts_by_group)
        variant_rows = self._index_machine_variants(game, entity_rows)
        with self.conn:
            self.conn.execute("DELETE FROM entities")
            self.conn.execute("DELETE FROM entity_texture_sets")
            self.conn.execute("DELETE FROM entity_variants")
            self.conn.executemany(
                "INSERT OR REPLACE INTO entities VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", entity_rows
            )
            self.conn.executemany(
                "INSERT OR IGNORE INTO entity_texture_sets VALUES (?, ?, ?)", ets_rows
            )
            self.conn.executemany(
                "INSERT OR REPLACE INTO entity_variants VALUES (?, ?, ?, ?, ?, ?, ?)", variant_rows
            )
            self.conn.execute("INSERT OR REPLACE INTO meta VALUES ('schema_version', '3')")
        return self.stats()

    # --- Query helpers ---

    def list_machine_variants(self, group_id: int, object_index: int) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT variant_index, label, destructibility_id, smr_id, part_count
            FROM entity_variants
            WHERE entity_group_id = ? AND entity_object_index = ?
            ORDER BY variant_index
            """,
            (group_id, object_index),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_machines(self, query: str = "") -> list[dict[str, Any]]:
        if query:
            like = f"%{query}%"
            rows = self.conn.execute(
                """
                SELECT * FROM entities
                WHERE category = 'machine'
                  AND (display_name LIKE ? OR codename LIKE ? OR CAST(group_id AS TEXT) = ?)
                ORDER BY display_name, part_count DESC
                """,
                (like, like, query),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM entities WHERE category = 'machine' ORDER BY display_name, part_count DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def search_entities(self, query: str = "", category: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        sql = "SELECT * FROM entities WHERE 1=1"
        params: list[Any] = []
        if category:
            sql += " AND category = ?"
            params.append(category)
        if query:
            sql += " AND (display_name LIKE ? OR codename LIKE ? OR texture_hints LIKE ? OR CAST(group_id AS TEXT) = ?)"
            like = f"%{query}%"
            params.extend([like, like, like, query])
        sql += " ORDER BY CASE WHEN category='machine' THEN 0 ELSE 1 END, part_count DESC, display_name LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def search_texture_sets(self, query: str = "", codename: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        sql = "SELECT * FROM texture_sets WHERE 1=1"
        params: list[Any] = []
        if codename:
            sql += " AND codename = ?"
            params.append(codename)
        if query:
            sql += " AND (name LIKE ? OR paths_json LIKE ? OR codename LIKE ?)"
            like = f"%{query}%"
            params.extend([like, like, like])
        sql += " ORDER BY name LIMIT ?"
        params.append(limit)
        out = []
        for r in self.conn.execute(sql, params).fetchall():
            d = dict(r)
            d["paths"] = json.loads(d.pop("paths_json"))
            d["packing"] = json.loads(d.pop("packing_json"))
            out.append(d)
        return out

    def texture_set_users(self, ts_group_id: int, ts_object_index: int) -> list[dict[str, Any]]:
        """Cross-reference: return all entities/machines that reference this TextureSet."""
        rows = self.conn.execute(
            """
            SELECT DISTINCT e.group_id, e.object_index, e.category, e.codename, e.display_name, e.part_count
            FROM entity_texture_sets ets
            JOIN entities e ON e.group_id = ets.entity_group_id
            WHERE ets.ts_group_id = ? AND ets.ts_object_index = ?
            ORDER BY e.display_name
            """,
            (ts_group_id, ts_object_index),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Audio / Music / SFX / Voicelines Two-Tier SQLite Index
    # ------------------------------------------------------------------

    def is_audio_built(self) -> bool:
        row = self.conn.execute("SELECT value FROM meta WHERE key = 'audio_schema_version'").fetchone()
        return row is not None and row["value"] == "1"

    def audio_stats(self) -> dict[str, Any]:
        music_cues = self.conn.execute("SELECT COUNT(*) AS c FROM audio_music_cues").fetchone()["c"]
        music_stems = self.conn.execute("SELECT COALESCE(SUM(stem_count), 0) AS c FROM audio_music_cues").fetchone()["c"]
        sfx_groups = self.conn.execute("SELECT COUNT(*) AS c FROM audio_groups WHERE kind = 'sfx'").fetchone()["c"]
        sfx_waves_total = self.conn.execute(
            "SELECT COALESCE(SUM(item_count), 0) AS c FROM audio_groups WHERE kind = 'sfx'"
        ).fetchone()["c"]
        sfx_groups_indexed = self.conn.execute(
            "SELECT COUNT(*) AS c FROM audio_groups WHERE kind = 'sfx' AND indexed_items = 1"
        ).fetchone()["c"]
        vo_groups = self.conn.execute(
            "SELECT COUNT(*) AS c FROM audio_groups WHERE kind = 'voiceline'"
        ).fetchone()["c"]
        vo_lines_total = self.conn.execute(
            "SELECT COALESCE(SUM(item_count), 0) AS c FROM audio_groups WHERE kind = 'voiceline'"
        ).fetchone()["c"]
        vo_groups_indexed = self.conn.execute(
            "SELECT COUNT(*) AS c FROM audio_groups WHERE kind = 'voiceline' AND indexed_items = 1"
        ).fetchone()["c"]
        voices = self.conn.execute("SELECT COUNT(*) AS c FROM audio_voices").fetchone()["c"]
        indexed_sfx_assets = self.conn.execute(
            "SELECT COUNT(*) AS c FROM audio_assets WHERE kind = 'sfx'"
        ).fetchone()["c"]
        indexed_vo_assets = self.conn.execute(
            "SELECT COUNT(*) AS c FROM audio_assets WHERE kind = 'voiceline'"
        ).fetchone()["c"]
        return {
            "audio_indexed": self.is_audio_built(),
            "music_cues": music_cues,
            "music_stems": music_stems,
            "sfx_groups": sfx_groups,
            "sfx_waves_total": sfx_waves_total,
            "sfx_groups_indexed": sfx_groups_indexed,
            "sfx_assets_indexed": indexed_sfx_assets,
            "voiceline_groups": vo_groups,
            "voicelines_total": vo_lines_total,
            "voiceline_groups_indexed": vo_groups_indexed,
            "voiceline_assets_indexed": indexed_vo_assets,
            "voices": voices,
            "deep_indexed": (sfx_groups > 0 and sfx_groups_indexed == sfx_groups and vo_groups_indexed == vo_groups),
        }

    def build_audio_index(
        self,
        game: HFWGame,
        progress_cb: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Tier-1 Audio Catalog build (~4-6s):
        - Indexes ALL MusicProjects, MusicCues & MusicStems
        - Indexes ALL compact VoiceResource character speaker names
        - Classifies ALL ~3,800 SFX & Voiceline groups into Guerrilla's developer directory structure
        """
        from hfw_toolkit.extract.audio import (
            classify_audio_group_dev_folder,
            discover_music_cues,
            load_all_voice_names,
        )

        t0 = time.time()
        log = progress_cb or (lambda _msg: None)
        graph = game.graph
        names_by_hash = graph.type_names_by_hash
        tt_idx = graph.type_table_indices

        # 1. Music Cues & Stems (all MusicProjects)
        log("Indexing HFW Music projects, cues & stems...")
        cues = discover_music_cues(game)
        cue_rows = [
            (
                c.cue_id,
                c.project_name,
                c.cue_name,
                c.category,
                c.dev_folder,
                c.branch_path,
                c.initial_tempo,
                round(c.duration_sec, 3),
                len(c.stems),
                1 if c.play_as_one_shot else 0,
                1 if c.is_transition else 0,
                json.dumps(c.tempo_events),
                json.dumps(
                    [
                        {
                            "wave_id": s.wave_id,
                            "track_index": s.track_index,
                            "start_beats": s.start_beats,
                            "duration_sec": round(s.duration_sec, 3),
                            "gain": round(s.gain, 4),
                            "fade_in_sec": round(s.fade_in_sec, 3),
                            "fade_out_sec": round(s.fade_out_sec, 3),
                            "sample_rate": s.sample_rate,
                            "channels": s.channels,
                            "sample_count": s.sample_count,
                            "encoding": s.encoding,
                            "is_streaming": s.is_streaming,
                            "data_size": s.data_size,
                        }
                        for s in c.stems
                    ]
                ),
            )
            for c in cues
        ]

        # 2. VoiceResource speaker names (Group 1 + compact voice groups)
        log("Indexing HFW character voice names...")
        voice_names = load_all_voice_names(game)
        voice_rows = [(vid, spk) for vid, spk in voice_names.items()]

        # 3. Map existing entity catalog group_ids (machines & characters) -> developer owner info
        entity_group_map: dict[int, tuple[str, str, str]] = {}
        try:
            for r in self.conn.execute("SELECT group_id, category, codename, display_name FROM entities"):
                gid = int(r["group_id"])
                if gid not in entity_group_map or r["category"] == "machine":
                    entity_group_map[gid] = (r["category"], r["codename"], r["display_name"])
        except Exception:
            pass

        # 4. Scan streaming graph for all SFX (WaveResource) & Voiceline (LocalizedSimpleSoundResource) groups
        log("Classifying SFX & Voiceline developer directories...")
        wave_tidx = names_by_hash.index("WaveResource") if "WaveResource" in names_by_hash else -1
        loc_tidx = (
            names_by_hash.index("LocalizedSimpleSoundResource")
            if "LocalizedSimpleSoundResource" in names_by_hash
            else -1
        )
        music_gids = {3, 78260, 84481}

        group_rows: list[tuple] = []
        for gm in graph.group_list:
            sl = tt_idx[gm.type_start : gm.type_start + gm.type_count]
            wc = sl.count(wave_tidx) if wave_tidx >= 0 else 0
            lc = sl.count(loc_tidx) if loc_tidx >= 0 else 0
            if wc > 0 and gm.group_id not in music_gids:
                cat, owner, folder = classify_audio_group_dev_folder(
                    game, gm.group_id, kind="sfx", entity_group_map=entity_group_map, max_scan_bytes=524_288
                )
                group_rows.append((gm.group_id, "sfx", cat, owner, folder, wc, 0))
            if lc > 0:
                cat, owner, folder = classify_audio_group_dev_folder(
                    game, gm.group_id, kind="voiceline", entity_group_map=entity_group_map, max_scan_bytes=524_288
                )
                group_rows.append((gm.group_id, "voiceline", cat, owner, folder, lc, 0))

        # Preserve any already deep-indexed `indexed_items = 1` flags for unchanged groups
        existing_indexed: set[tuple[int, str]] = {
            (int(r["group_id"]), str(r["kind"]))
            for r in self.conn.execute("SELECT group_id, kind FROM audio_groups WHERE indexed_items = 1")
        }
        if existing_indexed:
            group_rows = [
                (g, k, c, o, f, cnt, 1 if (g, k) in existing_indexed else 0)
                for (g, k, c, o, f, cnt, _) in group_rows
            ]

        with self.conn:
            self.conn.execute("DELETE FROM audio_music_cues")
            self.conn.execute("DELETE FROM audio_groups")
            self.conn.execute("DELETE FROM audio_voices")
            self.conn.executemany(
                "INSERT OR REPLACE INTO audio_music_cues VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                cue_rows,
            )
            self.conn.executemany(
                "INSERT OR REPLACE INTO audio_groups VALUES (?, ?, ?, ?, ?, ?, ?)",
                group_rows,
            )
            self.conn.executemany(
                "INSERT OR REPLACE INTO audio_voices VALUES (?, ?)",
                voice_rows,
            )
            self.conn.execute("INSERT OR REPLACE INTO meta VALUES ('audio_schema_version', '1')")
            self.conn.execute("INSERT OR REPLACE INTO meta VALUES ('audio_built_at', ?)", (str(int(time.time())),))

        st = self.audio_stats()
        log(
            f"Audio index ready in {time.time() - t0:.1f}s: "
            f"{st['music_cues']} music cues ({st['music_stems']} stems), "
            f"{st['sfx_groups']} SFX dirs ({st['sfx_waves_total']:,} waves), "
            f"{st['voiceline_groups']} voiceline dirs ({st['voicelines_total']:,} lines)"
        )
        return st

    def get_voice_names(self, game: HFWGame | None = None) -> dict[str, str]:
        """Return cached VoiceResource ID -> speaker name dictionary (loads from SQLite in ~2ms)."""
        rows = self.conn.execute("SELECT voice_id, speaker FROM audio_voices").fetchall()
        if rows:
            return {r["voice_id"]: r["speaker"] for r in rows}
        if game is not None:
            from hfw_toolkit.extract.audio import load_all_voice_names

            voices = load_all_voice_names(game)
            with self.conn:
                self.conn.executemany("INSERT OR REPLACE INTO audio_voices VALUES (?, ?)", list(voices.items()))
            return voices
        return {}

    def ensure_audio_group_indexed(self, game: HFWGame, group_id: int, kind: str) -> list[dict[str, Any]]:
        """Ensure individual SFX or Voiceline clips inside `group_id` are parsed & cached in SQLite."""
        row = self.conn.execute(
            "SELECT category, owner_name, dev_folder, indexed_items FROM audio_groups WHERE group_id = ? AND kind = ?",
            (group_id, kind),
        ).fetchone()
        if row and row["indexed_items"] == 1:
            return self.search_audio_assets(kind=kind, group_id=group_id, limit=5000)

        from hfw_toolkit.extract.audio import (
            classify_audio_group_dev_folder,
            extract_sfx_group_items,
            extract_voiceline_group_items,
        )

        if row:
            owner_name = row["owner_name"]
            dev_folder = row["dev_folder"]
            cat = row["category"]
        else:
            cat, owner_name, dev_folder = classify_audio_group_dev_folder(game, group_id, kind=kind)

        if kind == "sfx":
            items = extract_sfx_group_items(game, group_id, base_folder=dev_folder, owner_name=owner_name)
        elif kind == "voiceline":
            voices = self.get_voice_names(game)
            before_n = len(voices)
            items = extract_voiceline_group_items(
                game, group_id, voice_names=voices, base_folder=dev_folder, owner_name=owner_name
            )
            if len(voices) > before_n:
                with self.conn:
                    self.conn.executemany(
                        "INSERT OR REPLACE INTO audio_voices VALUES (?, ?)", list(voices.items())
                    )
        else:
            raise ValueError(f"Unsupported audio group kind: {kind}")

        asset_rows = [
            (
                it.object_id,
                it.kind,
                it.group_id,
                it.name,
                it.dev_folder,
                it.filename_stem,
                it.sample_rate,
                it.channels,
                it.sample_count,
                round(it.duration_sec, 3),
                it.encoding,
                it.encoding_name,
                it.data_size,
                1 if it.is_streaming else 0,
                it.sound_graph_name,
                it.owner_name,
                it.speaker,
                it.subtitle,
                json.dumps(list(it.languages)),
            )
            for it in items
        ]

        with self.conn:
            self.conn.execute("DELETE FROM audio_assets WHERE group_id = ? AND kind = ?", (group_id, kind))
            self.conn.executemany(
                "INSERT OR REPLACE INTO audio_assets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                asset_rows,
            )
            self.conn.execute(
                """
                INSERT INTO audio_groups VALUES (?, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(group_id, kind) DO UPDATE SET indexed_items = 1, item_count = excluded.item_count
                """,
                (group_id, kind, cat, owner_name, dev_folder, len(items)),
            )

        return self.search_audio_assets(kind=kind, group_id=group_id, limit=5000)

    def build_deep_audio_index(
        self,
        game: HFWGame,
        kinds: tuple[str, ...] = ("sfx", "voiceline"),
        progress_cb: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, Any]:
        """Tier-2 Optional Deep Index: parses all unindexed SFX & Voiceline groups into `audio_assets`."""
        if not self.is_audio_built():
            self.build_audio_index(game)

        placeholders = ",".join("?" for _ in kinds)
        pending = self.conn.execute(
            f"""
            SELECT group_id, kind, owner_name
            FROM audio_groups
            WHERE kind IN ({placeholders}) AND indexed_items = 0
            ORDER BY item_count ASC
            """,
            kinds,
        ).fetchall()
        total = len(pending)
        for i, r in enumerate(pending, 1):
            gid = int(r["group_id"])
            k = str(r["kind"])
            if progress_cb and (i == 1 or i % 10 == 0 or i == total):
                progress_cb(i, total, f"Indexing {k} group {gid} ({r['owner_name']}) [{i}/{total}]")
            try:
                self.ensure_audio_group_indexed(game, gid, k)
            except Exception:
                continue
        return self.audio_stats()

    def list_music_cues(
        self,
        query: str = "",
        category: str = "",
        project: str = "",
        limit: int = 250,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM audio_music_cues WHERE 1=1"
        params: list[Any] = []
        if category:
            sql += " AND category = ?"
            params.append(category)
        if project:
            sql += " AND project_name = ?"
            params.append(project)
        if query:
            like = f"%{query}%"
            sql += " AND (cue_name LIKE ? OR dev_folder LIKE ? OR category LIKE ? OR project_name LIKE ? OR cue_id = ?)"
            params.extend([like, like, like, like, query])
        sql += " ORDER BY dev_folder, cue_name LIMIT ?"
        params.append(limit)
        out: list[dict[str, Any]] = []
        for r in self.conn.execute(sql, params).fetchall():
            d = dict(r)
            d["stems"] = json.loads(d.pop("stems_json"))
            d["tempo_events"] = json.loads(d.pop("tempo_events_json"))
            out.append(d)
        return out

    def get_music_cue_info(self, cue_id: str) -> Any | None:
        """Reconstruct a `MusicCueInfo` dataclass directly from SQLite in <1ms."""
        from hfw_toolkit.extract.audio import MusicCueInfo, MusicStemInfo

        r = self.conn.execute("SELECT * FROM audio_music_cues WHERE cue_id = ?", (cue_id,)).fetchone()
        if not r:
            return None
        stems_raw = json.loads(r["stems_json"])
        tempo_raw = [(float(a), float(b)) for a, b in json.loads(r["tempo_events_json"])]
        stems = [
            MusicStemInfo(
                wave_id=s["wave_id"],
                track_index=int(s["track_index"]),
                start_beats=float(s["start_beats"]),
                duration_sec=float(s["duration_sec"]),
                gain=float(s["gain"]),
                fade_in_sec=float(s["fade_in_sec"]),
                fade_out_sec=float(s["fade_out_sec"]),
                sample_rate=int(s["sample_rate"]),
                channels=int(s["channels"]),
                sample_count=int(s["sample_count"]),
                encoding=int(s["encoding"]),
                is_streaming=bool(s["is_streaming"]),
                data_size=int(s["data_size"]),
            )
            for s in stems_raw
        ]
        return MusicCueInfo(
            cue_id=r["cue_id"],
            project_name=r["project_name"],
            cue_name=r["cue_name"],
            category=r["category"],
            dev_folder=r["dev_folder"],
            branch_path=r["branch_path"],
            initial_tempo=float(r["initial_tempo"]),
            play_as_one_shot=bool(r["play_as_one_shot"]),
            is_transition=bool(r["is_transition"]),
            tempo_events=tempo_raw,
            stems=stems,
        )

    def list_audio_groups(
        self,
        kind: str = "sfx",
        query: str = "",
        category: str = "",
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM audio_groups WHERE kind = ?"
        params: list[Any] = [kind]
        if category:
            sql += " AND category = ?"
            params.append(category)
        if query:
            like = f"%{query}%"
            sql += " AND (owner_name LIKE ? OR dev_folder LIKE ? OR category LIKE ? OR CAST(group_id AS TEXT) = ?)"
            params.extend([like, like, like, query])
        sql += " ORDER BY item_count DESC, dev_folder LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def list_audio_categories(self, kind: str = "music") -> list[dict[str, Any]]:
        if kind == "music":
            rows = self.conn.execute(
                """
                SELECT category, COUNT(*) AS cue_count, COALESCE(SUM(stem_count), 0) AS item_count
                FROM audio_music_cues
                GROUP BY category
                ORDER BY cue_count DESC, category
                """
            ).fetchall()
            return [dict(r) for r in rows]
        rows = self.conn.execute(
            """
            SELECT category, COUNT(*) AS group_count, COALESCE(SUM(item_count), 0) AS item_count
            FROM audio_groups
            WHERE kind = ?
            GROUP BY category
            ORDER BY item_count DESC, category
            """,
            (kind,),
        ).fetchall()
        return [dict(r) for r in rows]

    def search_audio_assets(
        self,
        kind: str = "",
        query: str = "",
        speaker: str = "",
        group_id: int | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM audio_assets WHERE 1=1"
        params: list[Any] = []
        if kind:
            sql += " AND kind = ?"
            params.append(kind)
        if group_id is not None:
            sql += " AND group_id = ?"
            params.append(group_id)
        if speaker:
            sql += " AND speaker LIKE ?"
            params.append(f"%{speaker}%")
        if query:
            like = f"%{query}%"
            sql += (
                " AND (name LIKE ? OR dev_folder LIKE ? OR sound_graph_name LIKE ? "
                "OR owner_name LIKE ? OR speaker LIKE ? OR subtitle LIKE ? OR object_id = ?)"
            )
            params.extend([like, like, like, like, like, like, query])
        sql += " ORDER BY dev_folder, filename_stem LIMIT ?"
        params.append(limit)
        out: list[dict[str, Any]] = []
        for r in self.conn.execute(sql, params).fetchall():
            d = dict(r)
            d["languages"] = json.loads(d.pop("languages_json"))
            out.append(d)
        return out

    def list_speakers(self, query: str = "", limit: int = 200) -> list[dict[str, Any]]:
        """List known speakers (combining indexed voiceline counts + VoiceResource registry)."""
        params: list[Any] = []
        where = ""
        if query:
            where = "WHERE speaker LIKE ?"
            params.append(f"%{query}%")
        # Merge counts from already-indexed audio_assets + distinct speakers from audio_voices
        rows = self.conn.execute(
            f"""
            SELECT speaker, COUNT(*) AS indexed_lines
            FROM (
                SELECT speaker FROM audio_assets WHERE kind = 'voiceline' AND speaker != ''
            )
            {where}
            GROUP BY speaker
            ORDER BY indexed_lines DESC, speaker
            LIMIT ?
            """,
            [*params, limit],
        ).fetchall()
        found = {r["speaker"]: int(r["indexed_lines"]) for r in rows}
        # Also include known VoiceResource speakers even before deep indexing
        v_rows = self.conn.execute(
            f"SELECT DISTINCT speaker FROM audio_voices {where} ORDER BY speaker LIMIT ?",
            [*params, limit],
        ).fetchall()
        for vr in v_rows:
            s = vr["speaker"]
            if s and s not in found:
                found[s] = 0
        items = [{"speaker": k, "indexed_lines": v} for k, v in found.items()]
        items.sort(key=lambda x: (-x["indexed_lines"], x["speaker"].lower()))
        return items[:limit]

    # ------------------------------------------------------------------
    # Characters, Outfits, Modular Heads & Guerrilla Dev-Directory Asset Catalog
    # ------------------------------------------------------------------

    def is_characters_assets_built(self) -> bool:
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key = 'chars_assets_schema_version'"
        ).fetchone()
        return row is not None and row["value"] == "1"

    def index_characters_and_assets(
        self,
        game: HFWGame,
        progress_cb: Callable[[str], None] | None = None,
    ) -> dict[str, int]:
        """Fast (~2-4s) indexing of ALL HFW Characters (Aloy + outfits, Companions, Tribal NPCs, 211 Heads)
        and the Guerrilla Developer Directory Asset Catalog (`models/...`, `dlc/models/...`).
        """
        from hfw_toolkit.extract.asset_assembler import discover_dev_directory_assets
        from hfw_toolkit.extract.character_assembler import discover_all_characters

        t0 = time.time()
        log = progress_cb or (lambda _msg: None)

        log("Indexing HFW characters, outfits & modular facial heads...")
        characters, outfits, heads = discover_all_characters(game, self)

        log("Building Guerrilla Developer Directory Asset Catalog...")
        dev_assets = discover_dev_directory_assets(game, self)

        char_rows = [
            (
                c.char_id,
                c.display_name,
                c.character_kind,
                c.tribe,
                c.gender,
                c.dev_folder,
                c.body_variant_id,
                c.facial_id,
                c.hair_id,
                int(c.outfit_count),
                int(c.part_count),
                c.tags,
            )
            for c in characters
        ]
        outfit_rows = [
            (
                o.outfit_id,
                o.char_id,
                o.display_name,
                o.tribe,
                o.dev_folder,
                o.resource_type,
                o.resource_id,
                o.facial_id,
                1 if o.has_headgear_toggle else 0,
                1 if o.is_dlc else 0,
            )
            for o in outfits
        ]
        head_rows = [
            (
                h["facial_id"],
                h["display_name"],
                h["gender"],
                h["dev_folder"],
            )
            for h in heads
        ]
        asset_rows = [
            (
                a.dev_path,
                a.asset_id,
                a.parent_dir,
                a.category,
                a.display_name,
                a.resource_type,
                a.resource_id,
                int(a.group_id),
                1 if a.is_skinned else 0,
                int(a.tset_count),
                int(a.shader_count),
                1 if a.is_dlc else 0,
            )
            for a in dev_assets
        ]

        with self.conn:
            self.conn.execute("DELETE FROM characters")
            self.conn.execute("DELETE FROM character_outfits")
            self.conn.execute("DELETE FROM character_heads")
            self.conn.execute("DELETE FROM dev_assets")
            self.conn.executemany(
                "INSERT OR REPLACE INTO characters VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                char_rows,
            )
            self.conn.executemany(
                "INSERT OR REPLACE INTO character_outfits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                outfit_rows,
            )
            self.conn.executemany(
                "INSERT OR REPLACE INTO character_heads VALUES (?, ?, ?, ?)",
                head_rows,
            )
            self.conn.executemany(
                "INSERT OR REPLACE INTO dev_assets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                asset_rows,
            )
            self.conn.execute("INSERT OR REPLACE INTO meta VALUES ('chars_assets_schema_version', '1')")
            self.conn.execute(
                "INSERT OR REPLACE INTO meta VALUES ('chars_assets_built_at', ?)",
                (str(int(time.time())),),
            )

        res = {
            "characters": len(char_rows),
            "character_outfits": len(outfit_rows),
            "character_heads": len(head_rows),
            "dev_assets": len(asset_rows),
        }
        log(
            f"Indexed {res['characters']} characters ({res['character_outfits']} outfits, "
            f"{res['character_heads']} heads) & {res['dev_assets']} dev-directory assets in {time.time() - t0:.1f}s"
        )
        return res

    def list_characters(
        self,
        query: str = "",
        kind: str = "",
        tribe: str = "",
        gender: str = "",
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM characters WHERE 1=1"
        params: list[Any] = []
        if kind:
            sql += " AND character_kind = ?"
            params.append(kind)
        if tribe:
            sql += " AND LOWER(tribe) = LOWER(?)"
            params.append(tribe)
        if gender:
            sql += " AND LOWER(gender) = LOWER(?)"
            params.append(gender)
        if query:
            like = f"%{query}%"
            sql += (
                " AND (display_name LIKE ? OR char_id LIKE ? OR tribe LIKE ? "
                "OR dev_folder LIKE ? OR tags LIKE ?)"
            )
            params.extend([like, like, like, like, like])
        sql += (
            " ORDER BY CASE character_kind WHEN 'hero' THEN 0 WHEN 'companion' THEN 1 ELSE 2 END, "
            "CASE WHEN facial_id != '' THEN 0 ELSE 1 END, outfit_count DESC, display_name LIMIT ?"
        )
        params.append(limit)
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def get_character(self, char_id: str) -> dict[str, Any] | None:
        r = self.conn.execute("SELECT * FROM characters WHERE char_id = ?", (char_id,)).fetchone()
        if r is None:
            r = self.conn.execute(
                "SELECT * FROM characters WHERE LOWER(display_name) = LOWER(?) LIMIT 1", (char_id,)
            ).fetchone()
        return dict(r) if r is not None else None

    def list_character_outfits(
        self,
        char_id: str = "",
        query: str = "",
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM character_outfits WHERE 1=1"
        params: list[Any] = []
        if char_id:
            sql += " AND char_id = ?"
            params.append(char_id)
        if query:
            like = f"%{query}%"
            sql += " AND (display_name LIKE ? OR tribe LIKE ? OR dev_folder LIKE ? OR outfit_id = ?)"
            params.extend([like, like, like, query])
        sql += " ORDER BY is_dlc ASC, tribe, display_name LIMIT ?"
        params.append(limit)
        out = []
        for r in self.conn.execute(sql, params).fetchall():
            d = dict(r)
            d["has_headgear_toggle"] = bool(d["has_headgear_toggle"])
            d["is_dlc"] = bool(d["is_dlc"])
            out.append(d)
        return out

    def get_character_outfit(self, outfit_id: str) -> dict[str, Any] | None:
        r = self.conn.execute(
            "SELECT * FROM character_outfits WHERE outfit_id = ? OR resource_id = ? LIMIT 1",
            (outfit_id, outfit_id),
        ).fetchone()
        if r is None:
            return None
        d = dict(r)
        d["has_headgear_toggle"] = bool(d["has_headgear_toggle"])
        d["is_dlc"] = bool(d["is_dlc"])
        return d

    def list_character_heads(
        self,
        query: str = "",
        gender: str = "",
        limit: int = 300,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM character_heads WHERE 1=1"
        params: list[Any] = []
        if gender:
            sql += " AND LOWER(gender) = LOWER(?)"
            params.append(gender)
        if query:
            like = f"%{query}%"
            sql += " AND (display_name LIKE ? OR dev_folder LIKE ? OR facial_id = ?)"
            params.extend([like, like, query])
        sql += " ORDER BY display_name LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def list_dev_asset_categories(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT category, COUNT(*) AS asset_count,
                   SUM(CASE WHEN is_skinned = 1 THEN 1 ELSE 0 END) AS rigged_count
            FROM dev_assets
            GROUP BY category
            ORDER BY asset_count DESC, category
            """
        ).fetchall()
        return [dict(r) for r in rows]

    def list_dev_tree_children(self, parent_dir: str = "") -> dict[str, Any]:
        """Browse Guerrilla's original developer directory hierarchy one folder level at a time.

        Returns immediate child folders (with recursive asset counts) and direct assets inside `parent_dir`.
        """
        norm_parent = parent_dir.replace("\\", "/").strip("/")
        norm_parent_low = norm_parent.lower()
        prefix_low = (norm_parent_low + "/") if norm_parent_low else ""

        rows = self.conn.execute("SELECT * FROM dev_assets ORDER BY dev_path").fetchall()
        folder_counts: dict[str, dict[str, Any]] = {}
        direct_assets: list[dict[str, Any]] = []

        for r in rows:
            dp = r["dev_path"]
            dp_low = dp.lower()
            if prefix_low and not dp_low.startswith(prefix_low):
                continue
            rel = dp[len(prefix_low) :] if prefix_low else dp
            parts = [p for p in rel.split("/") if p]
            if not parts:
                continue
            if len(parts) == 1:
                d = dict(r)
                d["is_skinned"] = bool(d["is_skinned"])
                d["is_dlc"] = bool(d["is_dlc"])
                direct_assets.append(d)
            else:
                seg = parts[0]
                child_path = f"{norm_parent}/{seg}" if norm_parent else seg
                ckey = child_path.lower()
                ent = folder_counts.get(ckey)
                if ent is None:
                    ent = {
                        "path": child_path,
                        "name": seg,
                        "asset_count": 0,
                        "categories": set(),
                    }
                    folder_counts[ckey] = ent
                ent["asset_count"] += 1
                ent["categories"].add(r["category"])

        folders = [
            {
                "path": v["path"],
                "name": v["name"],
                "asset_count": v["asset_count"],
                "categories": sorted(v["categories"]),
            }
            for v in sorted(folder_counts.values(), key=lambda x: (-x["asset_count"], x["name"].lower()))
        ]
        return {
            "parent_dir": norm_parent,
            "folders": folders,
            "assets": direct_assets,
        }

    def search_dev_assets(
        self,
        query: str = "",
        category: str = "",
        dir_prefix: str = "",
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM dev_assets WHERE 1=1"
        params: list[Any] = []
        if category:
            sql += " AND category = ?"
            params.append(category)
        if dir_prefix:
            pref = dir_prefix.replace("\\", "/").strip("/")
            sql += " AND (LOWER(dev_path) LIKE LOWER(?) OR LOWER(parent_dir) LIKE LOWER(?))"
            params.extend([f"{pref}%", f"{pref}%"])
        if query:
            like = f"%{query}%"
            sql += (
                " AND (display_name LIKE ? OR dev_path LIKE ? OR resource_type LIKE ? "
                "OR asset_id = ? OR resource_id = ?)"
            )
            params.extend([like, like, like, query, query])
        sql += " ORDER BY is_skinned DESC, tset_count DESC, dev_path LIMIT ?"
        params.append(limit)
        out = []
        for r in self.conn.execute(sql, params).fetchall():
            d = dict(r)
            d["is_skinned"] = bool(d["is_skinned"])
            d["is_dlc"] = bool(d["is_dlc"])
            out.append(d)
        return out

    def get_dev_asset(self, asset_id_or_path: str) -> dict[str, Any] | None:
        key = asset_id_or_path.replace("\\", "/").strip("/")
        r = self.conn.execute(
            """
            SELECT * FROM dev_assets
            WHERE LOWER(dev_path) = LOWER(?) OR asset_id = ? OR resource_id = ?
            LIMIT 1
            """,
            (key, asset_id_or_path, asset_id_or_path),
        ).fetchone()
        if r is None:
            return None
        d = dict(r)
        d["is_skinned"] = bool(d["is_skinned"])
        d["is_dlc"] = bool(d["is_dlc"])
        return d
