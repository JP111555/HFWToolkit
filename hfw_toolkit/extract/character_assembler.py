"""Complete Horizon Forbidden West Character, Outfit, Head, Hair & Unified Rig Assembler.

Supports:
- Aloy (Player Hero) with ALL ~40 base + Burning Shores DLC `OutfitStreamingData` outfits,
  headgear on/off toggle, cinematic `FacialAnimationComponentResource` head (`48346:0`),
  and iconic braided `HairResource` (`43305:37`) `HairSkinnedMeshLod` geometry!
- Main Companions & Story Characters (Erend, Varl, Zo, Kotallo, Alva, Sylens, Beta, Tilda,
  Seyka, Regalla, Hekarro, Avad, Talanah, Petra, etc.) across all story/costume variations!
- Hundreds of Modular Tribal NPCs (`1,275` `BodyVariant`s + `211` `FacialAnimationComponentResource` heads)
  organized by Guerrilla's original `models/characters/humans/...` developer folders!
- Unified Master Humanoid Rig merging Body + Outfit + Facial (`FACIAL_*`) + Hair weights onto
  ONE single usable Armature with zero detachment in Blender Pose Mode and Unreal Engine 5.
- Natural-Language Character Prompt Resolver (e.g. "Aloy Carja Blazon no headgear",
  "Seyka", "Erend", "Tenakth female warrior", "Carja male guard").
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hfw_toolkit.decima.catalog import AssetCatalog
from hfw_toolkit.decima.reader import ObjectId, UUIDRef
from hfw_toolkit.decima.streaming_graph import HFWGame
from hfw_toolkit.extract.machine_assembler import AssembledPart, MachineAssembly, MachineVariantOption
from hfw_toolkit.extract.materials import MaterialInfo, parse_material
from hfw_toolkit.extract.meshes import DecodedMesh, DecodedPrimitive, _unpack_element, decode_mesh_hierarchy
from hfw_toolkit.extract.skeleton import (
    JointInfo,
    SkeletonData,
    add_skeleton_helpers,
    decima_mat44_to_np,
    extract_physics_skeleton_world_matrices,
    extract_skeleton,
)

# Verified Aloy core resources in HFW PC (also discovered dynamically as fallback)
ALOY_FACIAL_COMPONENT_ID = "48346:0"
ALOY_HAIR_RESOURCE_ID = "43305:37"
DEFAULT_ALOY_OUTFIT_KEYWORDS = ("aloynorashieldweaver", "aloynora", "aloynorabrave")

# Curated Guerrilla folder/token -> Official Horizon Forbidden West Aloy Outfit Display Names
ALOY_OUTFIT_FRIENDLY_NAMES: dict[str, tuple[str, str]] = {
    "h2_aloynorashieldweaver": ("Nora Anointed (Starter)", "Nora"),
    "h2_aloynorabrave": ("Nora Anointed", "Nora"),
    "h2_aloynora": ("Nora Huntress", "Nora"),
    "h2_aloynoravarl": ("Nora Champion (Varl)", "Nora"),
    "h2_aloynorarost": ("Nora Legacy (Rost)", "Nora"),
    "h2_aloynorathunderjaw": ("Nora Thunder Warrior", "Nora"),
    "h2_aloynorathunder": ("Nora Thunder Warrior", "Nora"),
    "h2_aloynorasurvivor": ("Nora Valiant", "Nora"),
    "h2_aloynoravalor": ("Nora Valiant", "Nora"),
    "h2_aloynorastealth": ("Nora Tracker", "Nora"),
    "h2_aloynorascout": ("Nora Lookout (DLC)", "Nora"),
    "aloyoutcast": ("Nora Outcast (Classic)", "Nora"),
    "h2_aloycarjablazon": ("Carja Blazon", "Carja"),
    "h2_aloycarjastalker": ("Carja Stalker Elite", "Carja"),
    "h2_aloycarjastorm": ("Carja Stalker Elite", "Carja"),
    "h2_aloycarjastalwart": ("Carja Shadow", "Carja"),
    "h2_aloycarjadesert": ("Carja Wanderer", "Carja"),
    "h2_aloycarjarhino": ("Carja Behemoth Trapper", "Carja"),
    "h2_aloycarjafireresistance": ("Carja Trader (Fire)", "Carja"),
    "aloycarja_stormbringer": ("Carja Stormbringer", "Carja"),
    "h2_aloycarja": ("Carja Stormbringer", "Carja"),
    "h2_aloydeserttenakthpredator": ("Tenakth Recon (Desert)", "Tenakth"),
    "h2_aloydeserttenakthgrunt": ("Tenakth Reaver (Desert)", "Tenakth"),
    "h2_aloydeserttenakth": ("Tenakth Desert Outfit", "Tenakth"),
    "h2_aloyjungletenakthknight": ("Tenakth Vanquisher (Lowland)", "Tenakth"),
    "h2_aloyjungletenakthshaman": ("Tenakth Marshal (Lowland)", "Tenakth"),
    "h2_aloyjungletenakth": ("Tenakth Lowland Outfit", "Tenakth"),
    "h2_aloymountaintenakthspike": ("Tenakth Dragoon (Sky)", "Tenakth"),
    "h2_aloymountaintenakhtmarkswoman": ("Tenakth Tactician (Sky)", "Tenakth"),
    "h2_aloymountaintenakthmarkswoman": ("Tenakth Tactician (Sky)", "Tenakth"),
    "h2_aloymountaintenakth": ("Tenakth Sky Outfit", "Tenakth"),
    "h2_aloyutaruspiritsinger": ("Utaru Winterweave", "Utaru"),
    "h2_aloyutaruplantsinger": ("Utaru Harvester", "Utaru"),
    "h2_aloyutaruweaver": ("Utaru Whisperer", "Utaru"),
    "h2_aloyutaruwarden": ("Utaru Warden", "Utaru"),
    "h2_aloyutaru": ("Utaru Outfit", "Utaru"),
    "h2_aloyoseramdesert": ("Oseram Artificer", "Oseram"),
    "h2_aloyoseramfire": ("Oseram Forester", "Oseram"),
    "h2_aloyoseramarrowbreaker": ("Oseram Vanguard", "Oseram"),
    "h2_aloyoseramagent": ("Oseram Striker", "Oseram"),
    "h2_aloyoseramdervahl": ("Oseram Wayfarer", "Oseram"),
    "h2_aloyoseramtank": ("Oseram Explorer", "Oseram"),
    "h2_aloyoseram": ("Oseram Outfit", "Oseram"),
    "h2_aloyquenmarkswoman": ("Quen Deadeye (DLC)", "Quen"),
    "h2_aloyquenheavyranged": ("Quen Marine (DLC)", "Quen"),
    "h2_aloyquencommander": ("Quen Commander (DLC)", "Quen"),
    "h2_aloyseyka": ("Quen Commander (DLC)", "Quen"),
    "h2_aloyquen": ("Quen Outfit (DLC)", "Quen"),
    "h2_aloysobeck": ("Sobeck's Raiment", "Old World"),
    "h2_aloyblacktide": ("Blacktide Quen (DLC)", "Quen"),
}

# Named HFW Story Characters & Companions (folder keyword -> (Display Name, Tribe/Faction, Gender))
NAMED_CHARACTER_PATTERNS: list[tuple[str, str, str, str]] = [
    ("h2_aloy", "Aloy", "Nora", "female"),
    ("h2_beta", "Beta", "Old World", "female"),
    ("/beta", "Beta", "Old World", "female"),
    ("h2_seyka", "Seyka", "Quen", "female"),
    ("h2_erend", "Erend", "Oseram", "male"),
    ("h2_varl", "Varl", "Nora", "male"),
    ("h2_zo", "Zo", "Utaru", "female"),
    ("h2_kotallo", "Kotallo", "Tenakth", "male"),
    ("h2_chon", "Alva", "Quen", "female"),
    ("h2_alva", "Alva", "Quen", "female"),
    ("/alva", "Alva", "Quen", "female"),
    ("h2_sylens", "Sylens", "Banuk", "male"),
    ("h2_tilda", "Tilda", "Far Zenith", "female"),
    ("/tilda", "Tilda", "Far Zenith", "female"),
    ("h2_regalla", "Regalla", "Tenakth", "female"),
    ("h2_hekarro", "Hekarro", "Tenakth", "male"),
    ("h2_avad", "Avad", "Carja", "male"),
    ("h2_talanah", "Talanah", "Carja", "female"),
    ("/talanah", "Talanah", "Carja", "female"),
    ("h2_petra", "Petra", "Oseram", "female"),
    ("h2_fashav", "Fashav", "Carja", "male"),
    ("h2_gerard", "Gerard", "Far Zenith", "male"),
    ("h2_erik", "Erik", "Far Zenith", "male"),
    ("h2_walterlondra", "Walter Londra", "Far Zenith", "male"),
    ("h2_londra", "Walter Londra", "Far Zenith", "male"),
    ("h2_rost", "Rost", "Nora", "male"),
    ("h2_gaia", "GAIA", "Old World", "female"),
    ("/gaia", "GAIA", "Old World", "female"),
    ("h2_sobeck", "Elisabet Sobeck", "Old World", "female"),
    ("h2_elisabet", "Elisabet Sobeck", "Old World", "female"),
    ("h2_dekka", "Dekka", "Tenakth", "female"),
    ("h2_morlund", "Morlund", "Oseram", "male"),
    ("h2_stemmur", "Stemmur", "Oseram", "male"),
    ("h2_abadund", "Abadund", "Oseram", "male"),
    ("h2_boomer", "Boomer", "Oseram", "female"),
    ("h2_delah", "Delah", "Oseram", "female"),
    ("h2_tekotteh", "Tekotteh", "Tenakth", "male"),
    ("h2_drakka", "Drakka", "Tenakth", "male"),
    ("h2_yarra", "Yarra", "Tenakth", "female"),
    ("h2_ceo", "Ceo", "Quen", "male"),
    ("h2_bohai", "Bohai", "Quen", "male"),
    ("h2_gildun", "Gildun", "Oseram", "male"),
    ("h2_kue", "Kue", "Utaru", "male"),
    ("h2_fane", "Fane", "Utaru", "male"),
    ("h2_marad", "Marad", "Carja", "male"),
    ("h2_vuadis", "Studious Vuadis", "Carja", "male"),
    ("h2_ulvund", "Ulvund", "Oseram", "male"),
    ("h2_javad", "Javad", "Carja", "male"),
    ("h2_kalla", "Kalla", "Nora", "female"),
    ("h2_amilia", "Amilia", "Carja", "female"),
]

TRIBE_KEYWORDS: list[tuple[str, str]] = [
    ("tenakth", "Tenakth"),
    ("carja", "Carja"),
    ("oseram", "Oseram"),
    ("utaru", "Utaru"),
    ("nora", "Nora"),
    ("quen", "Quen"),
    ("zenith", "Far Zenith"),
    ("banuk", "Banuk"),
    ("oldworld", "Old World"),
    ("eclipse", "Eclipse"),
    ("rebel", "Tenakth Rebels"),
]


@dataclass(slots=True)
class CharacterRecord:
    char_id: str                # e.g. "aloy" | "bv:21350:1"
    display_name: str
    character_kind: str         # "hero" | "companion" | "npc"
    tribe: str
    gender: str                 # "female" | "male" | "child" | "unknown"
    dev_folder: str             # Guerrilla developer path e.g. "models/characters/humans/h2_erend"
    body_variant_id: str        # e.g. "21350:1" (or default OSD for Aloy)
    facial_id: str              # e.g. "48346:0" (FacialAnimationComponentResource)
    hair_id: str                # e.g. "43305:37" for Aloy, "" for integrated BodyVariant hair
    outfit_count: int = 1
    part_count: int = 0
    tags: str = ""              # searchable keywords


@dataclass(slots=True)
class CharacterOutfitRecord:
    outfit_id: str              # e.g. "osd:72966:0" | "bv:65337:0"
    char_id: str                # parent character id e.g. "aloy" | "erend"
    display_name: str
    tribe: str
    dev_folder: str
    resource_type: str          # "OutfitStreamingData" | "BodyVariant"
    resource_id: str            # "72966:0" | "65337:0"
    facial_id: str = ""
    has_headgear_toggle: bool = False
    is_dlc: bool = False


def _humanize_camel(token: str) -> str:
    s = re.sub(r"^(?:h2_|hrz2_|aloy_?)", "", token, flags=re.IGNORECASE)
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    s = re.sub(r"[_\-]+", " ", s).strip()
    return " ".join(w.capitalize() for w in s.split()) or token


def _clean_dev_folder(raw_path: str) -> str:
    s = str(raw_path).replace("\\", "/")
    if s.lower().startswith("work:"):
        s = s[5:]
    s = s.lstrip("/")
    # Trim `/textures/...` or `/texture/...` suffix to get the character/asset developer directory
    low = s.lower()
    for marker in ("/textures/", "/texture/"):
        idx = low.find(marker)
        if idx > 0:
            return s[:idx]
    parts = [p for p in s.split("/") if p]
    return "/".join(parts[:-1]) if len(parts) > 1 else s


def _collect_group_tset_paths(
    game: HFWGame,
    catalog: AssetCatalog | None,
    group_id: int,
    tset_by_group: dict[int, list[str]],
    max_depth: int = 2,
) -> list[str]:
    """Collect Guerrilla `work:` TextureSet paths referenced by `group_id` (and up to `max_depth` subgroups)."""
    out: list[str] = []
    seen_g: set[int] = {group_id}
    frontier = [group_id]
    for _depth in range(max_depth + 1):
        nxt: list[int] = []
        for gid in frontier:
            if gid in tset_by_group:
                out.extend(tset_by_group[gid])
            for sub in game.graph.group_subgroups(gid):
                if sub not in seen_g and sub > 5:  # Skip mega-common root groups 1..5
                    seen_g.add(sub)
                    nxt.append(sub)
        if out or not nxt:
            break
        frontier = nxt[:40]
    return out


def _load_tset_paths_by_group(catalog: AssetCatalog) -> dict[int, list[str]]:
    by_g: dict[int, list[str]] = {}
    for gid, pjson in catalog.conn.execute("SELECT group_id, paths_json FROM texture_sets").fetchall():
        try:
            items = json.loads(pjson) if pjson else []
        except Exception:
            continue
        for it in items:
            p = it.get("path", "") if isinstance(it, dict) else str(it)
            if p:
                by_g.setdefault(int(gid), []).append(p)
                break
    return by_g


def _classify_outfit_name(paths: list[str], fallback_gid: int) -> tuple[str, str, str, bool]:
    """Return `(display_name, tribe, dev_folder, is_dlc)` from TextureSet paths of an `OutfitStreamingData` group."""
    dev_folder = f"models/characters/humans/outfit_{fallback_gid}"
    is_dlc = False
    best_path = ""
    # Prefer specific Aloy/character outfit folder paths (skip BaseBody, Heads, Focus, Rebreather, generic shaders)
    for p in paths:
        pl = p.replace("\\", "/").lower()
        if "humans" in pl and not any(k in pl for k in ("basebody", "h2_heads", "baseparts/focus", "h2_rebreather", "softfur")):
            best_path = p
            break
    if not best_path and paths:
        best_path = paths[0]

    if best_path:
        dev_folder = _clean_dev_folder(best_path)
        low_all = " ".join(p.replace("\\", "/").lower() for p in paths)
        low = dev_folder.lower()
        is_dlc = "/dlc/" in low or low.startswith("dlc/")
        # Check curated friendly names against folder first, then full paths (e.g. aloycarja_stormbringer!)
        for key in sorted(ALOY_OUTFIT_FRIENDLY_NAMES.keys(), key=len, reverse=True):
            if key in low or key in low_all:
                dname, tribe = ALOY_OUTFIT_FRIENDLY_NAMES[key]
                return (dname, tribe, dev_folder, is_dlc)
        # Fallback: clean last folder component
        leaf = dev_folder.rstrip("/").split("/")[-1]
        dname = _humanize_camel(leaf)
        tribe = "Nora"
        for kw, tname in TRIBE_KEYWORDS:
            if kw in low:
                tribe = tname
                break
        if is_dlc and "(DLC)" not in dname:
            dname = f"{dname} (DLC)"
        return (dname, tribe, dev_folder, is_dlc)

    return (f"Aloy Outfit {fallback_gid}", "Nora", dev_folder, False)


def _classify_body_variant(
    paths: list[str],
    anim_tags: list[str],
    gid: int,
    oid: int,
    *,
    allow_named_match: bool = True,
    head_folder: str = "",
) -> tuple[str, str, str, str, str, str]:
    """Classify a `BodyVariant` -> `(char_key, display_name, char_kind, tribe, gender, dev_folder)`."""
    dev_folder = f"models/characters/humans/bodyvariant_{gid}_{oid}"
    # Pick most informative human path (prefer named character / tribe outfit over generic BaseBody/Heads)
    ranked = sorted(
        paths,
        key=lambda p: (
            0 if "humans" in p.lower() else 2,
            1 if any(k in p.lower() for k in ("basebody", "h2_heads", "h2_hair", "eyelash", "teeth", "eye", "baseparts")) else 0,
            -len(p),
        ),
    )
    if ranked:
        dev_folder = _clean_dev_folder(ranked[0])
    elif head_folder:
        dev_folder = head_folder

    # Match named characters ONLY against primary outfit dev_folder + this BV's own facial head folder (no group bleed!)
    own_blob = f"{dev_folder} {head_folder}".replace("\\", "/").lower()
    blob = " ".join(p.replace("\\", "/").lower() for p in paths) + " " + head_folder.lower()
    tag_blob = " ".join(str(t).lower() for t in anim_tags)

    gender = "unknown"
    if "female" in tag_blob or "/female/" in blob or "female" in blob:
        gender = "female"
    elif "male" in tag_blob or "/male/" in blob or "male" in blob:
        gender = "male"
    elif "child" in tag_blob or "/child/" in blob:
        gender = "child"

    # 1. Check named story characters / companions strictly against this BodyVariant's own outfit/head folder!
    if allow_named_match:
        for pat, cname, tribe, def_gender in NAMED_CHARACTER_PATTERNS:
            if pat in own_blob:
                kind = "hero" if cname == "Aloy" else "companion"
                # Always trust canonical gender of named story characters!
                return (cname.lower().replace(" ", "_"), cname, kind, tribe, def_gender, dev_folder)

    # 2. Tribe NPC classification
    tribe = "Civilian"
    for kw, tname in TRIBE_KEYWORDS:
        if kw in own_blob or kw in blob:
            tribe = tname
            break

    # Extract informative role / outfit leaf name from dev_folder
    parts = [p for p in dev_folder.split("/") if p]
    leaf = parts[-1] if parts else f"NPC_{gid}"
    role = _humanize_camel(leaf)
    sub_bits: list[str] = []
    if tribe != "Civilian" and tribe.lower() not in role.lower():
        sub_bits.append(tribe)
    if gender in ("female", "male", "child") and gender not in role.lower():
        sub_bits.append(gender.capitalize())
    if role and role.lower() not in ("humans", "characters", "models", "h2 heads", "h2 basebodyfemale", "h2 basebodymale"):
        sub_bits.append(role)
    else:
        sub_bits.append(f"NPC {gid}")
    disp = " ".join(sub_bits)
    ckey = f"bv:{gid}:{oid}"
    return (ckey, disp, "npc", tribe, gender, dev_folder)


def discover_all_characters(
    game: HFWGame,
    catalog: AssetCatalog,
) -> tuple[list[CharacterRecord], list[CharacterOutfitRecord], list[dict[str, Any]]]:
    """Discover Aloy (with all ~40 outfits), all main companions, all NPC BodyVariants, and all 211 Heads.

    Runs in ~1–3 seconds using `read_group_local` + SQLite `texture_sets` lookup!
    """
    tset_by_group = _load_tset_paths_by_group(catalog)
    graph = game.graph

    characters: list[CharacterRecord] = []
    outfits: list[CharacterOutfitRecord] = []
    heads: list[dict[str, Any]] = []

    # Fast type index lookup (avoid rebuilding group_types lists across 100k groups!)
    names_by_hash = graph.type_names_by_hash
    tt_idx = graph.type_table_indices
    fac_tidx = names_by_hash.index("FacialAnimationComponentResource") if "FacialAnimationComponentResource" in names_by_hash else -1
    osd_tidx = names_by_hash.index("OutfitStreamingData") if "OutfitStreamingData" in names_by_hash else -1
    bv_tidx = names_by_hash.index("BodyVariant") if "BodyVariant" in names_by_hash else -1

    # 1. Discover ALL 211 FacialAnimationComponentResources (modular swappable heads!)
    heads_by_id: dict[str, dict[str, Any]] = {}
    for gm in graph.group_list:
        sl = tt_idx[gm.type_start : gm.type_start + gm.type_count]
        if fac_tidx < 0 or fac_tidx not in sl:
            continue
        gid = gm.group_id
        paths = _collect_group_tset_paths(game, catalog, gid, tset_by_group, max_depth=1)
        head_path = next((p for p in paths if "head" in p.lower() or "face" in p.lower() or "humans" in p.lower()), paths[0] if paths else "")
        dev_f = _clean_dev_folder(head_path) if head_path else f"models/characters/humans/heads/group_{gid}"
        blob = " ".join(paths).lower()
        gender = "female" if "/female/" in blob or "female" in blob else ("male" if "/male/" in blob or "male" in blob else ("child" if "/child/" in blob else "unknown"))
        leaf = dev_f.rstrip("/").split("/")[-1]
        hname = _humanize_camel(leaf)
        for pat, cname, _tr, dg in NAMED_CHARACTER_PATTERNS:
            if pat in dev_f.lower():
                hname = f"{cname} Head"
                gender = dg
                break
        for oid, tidx in enumerate(sl):
            if tidx == fac_tidx:
                fid = f"{gid}:{oid}"
                hrec = {
                    "facial_id": fid,
                    "display_name": hname,
                    "gender": gender,
                    "dev_folder": dev_f,
                }
                heads.append(hrec)
                heads_by_id[fid] = hrec

    # 2. Discover ALL Aloy Outfits (`OutfitStreamingData`) — skipping tiny Focus/Rebreather-only overlays!
    aloy_outfits: list[CharacterOutfitRecord] = []
    default_aloy_osd = ""
    seen_aloy_names: dict[str, int] = {}

    for gm in sorted(graph.group_list, key=lambda x: x.group_id):
        sl = tt_idx[gm.type_start : gm.type_start + gm.type_count]
        if osd_tidx < 0 or osd_tidx not in sl:
            continue
        gid = gm.group_id
        oid = sl.index(osd_tidx)
        paths = _collect_group_tset_paths(game, catalog, gid, tset_by_group, max_depth=2)
        dname, tribe, dev_f, is_dlc = _classify_outfit_name(paths, gid)
        low_f = dev_f.lower()
        # Skip accessory-only Focus / Rebreather / Shader-effect OSDs (not full-body outfits!)
        if any(skip in low_f for skip in ("baseparts/focus", "h2_rebreather", "outfit_", "shaders_hrz2", "shaders/")):
            continue

        has_hg_toggle = False
        try:
            osd_obj = game.read_group_local(gid)[oid]
            has_hg_toggle = len(osd_obj.get("ModelParts") or ()) >= 2
        except Exception:
            pass

        # Disambiguate rare duplicate display names (e.g. Base vs Blacktide Quen Marine DLC)
        seen_aloy_names[dname] = seen_aloy_names.get(dname, 0) + 1
        final_dname = dname if seen_aloy_names[dname] == 1 else f"{dname} (Variant {seen_aloy_names[dname]})"

        rec = CharacterOutfitRecord(
            outfit_id=f"osd:{gid}:{oid}",
            char_id="aloy",
            display_name=final_dname,
            tribe=tribe,
            dev_folder=dev_f,
            resource_type="OutfitStreamingData",
            resource_id=f"{gid}:{oid}",
            facial_id=ALOY_FACIAL_COMPONENT_ID,
            has_headgear_toggle=has_hg_toggle,
            is_dlc=is_dlc,
        )
        aloy_outfits.append(rec)
        # Prioritize Nora Anointed (Starter / Shield-Weaver) as Aloy's #1 default outfit!
        if any(kw in low_f for kw in DEFAULT_ALOY_OUTFIT_KEYWORDS):
            if not default_aloy_osd or "shieldweaver" in low_f:
                default_aloy_osd = f"{gid}:{oid}"

    if not default_aloy_osd and aloy_outfits:
        default_aloy_osd = aloy_outfits[0].resource_id

    # Register Aloy as #1 Hero character!
    characters.append(
        CharacterRecord(
            char_id="aloy",
            display_name="Aloy",
            character_kind="hero",
            tribe="Nora",
            gender="female",
            dev_folder="models/characters/humans/H2_AloyNoraShieldWeaver",
            body_variant_id=default_aloy_osd,
            facial_id=ALOY_FACIAL_COMPONENT_ID,
            hair_id=ALOY_HAIR_RESOURCE_ID,
            outfit_count=len(aloy_outfits),
            part_count=3,
            tags="aloy hero player nora protagonist redhead braids outfits dlc",
        )
    )
    outfits.extend(aloy_outfits)

    # 3. Discover ALL Companion & NPC `BodyVariant`s across the entire game!
    companion_groups: dict[str, list[tuple[CharacterRecord, CharacterOutfitRecord]]] = {}
    npc_records: list[CharacterRecord] = []

    for gm in sorted(graph.group_list, key=lambda x: x.group_id):
        sl = tt_idx[gm.type_start : gm.type_start + gm.type_count]
        if bv_tidx < 0 or bv_tidx not in sl:
            continue
        gid = gm.group_id
        try:
            objs = game.read_group_local(gid)
        except Exception:
            continue

        group_paths = _collect_group_tset_paths(game, catalog, gid, tset_by_group, max_depth=2)
        blob_all = " ".join(group_paths).lower()
        if group_paths and ("robots/" in blob_all or "animals/" in blob_all) and "humans/" not in blob_all:
            continue

        for oid, obj in enumerate(objs):
            if obj.get("_type") != "BodyVariant":
                continue
            mps = obj.get("ModelPartResources") or []
            ecs = obj.get("EntityComponentResources") or []
            if not mps and not ecs:
                continue

            # Gather paths ONLY from this BodyVariant's OWN ModelParts & Components (zero cross-NPC bleed!)
            own_paths: list[str] = []
            facial_id = ""
            for ref in list(mps) + list(ecs):
                tid = getattr(ref, "target_id", None)
                if tid is None:
                    continue
                rgid, roid = tid.group_id, tid.object_index
                if rgid in graph.groups:
                    rgm = graph.groups[rgid]
                    if roid < rgm.type_count and tt_idx[rgm.type_start + roid] == fac_tidx:
                        facial_id = f"{rgid}:{roid}"
                # Only pull TextureSets from the referenced group when it's a compact dedicated part/head group
                if rgid in graph.groups and (rgid != gid or len(objs) <= 250):
                    own_paths.extend(_collect_group_tset_paths(game, catalog, rgid, tset_by_group, max_depth=1))

            head_folder = heads_by_id[facial_id]["dev_folder"] if facial_id in heads_by_id else ""
            used_own = bool(own_paths or head_folder)
            bv_paths = own_paths if own_paths else list(group_paths)
            if not bv_paths and not head_folder:
                continue

            if not mps and facial_id == ALOY_FACIAL_COMPONENT_ID:
                continue

            anim_tags = list(obj.get("AnimationSetTags") or ())
            ckey, disp, ckind, tribe, gender, dev_f = _classify_body_variant(
                bv_paths,
                anim_tags,
                gid,
                oid,
                allow_named_match=used_own,
                head_folder=head_folder,
            )

            # Aloy's canonical outfits are already indexed from OutfitStreamingData; skip duplicate BodyVariants
            if ckey == "aloy":
                continue

            if ckind == "companion":
                leaf = _humanize_camel(dev_f.rstrip("/").split("/")[-1])
                var_label = f"{disp} — {leaf}" if leaf.lower() != disp.lower() else f"{disp} (Variant {gid})"
                c_rec = CharacterRecord(
                    char_id=ckey,
                    display_name=disp,
                    character_kind="companion",
                    tribe=tribe,
                    gender=gender,
                    dev_folder=dev_f,
                    body_variant_id=f"{gid}:{oid}",
                    facial_id=facial_id,
                    hair_id="",
                    outfit_count=1,
                    part_count=len(mps) + (1 if facial_id else 0),
                    tags=f"{disp.lower()} {tribe.lower()} {gender} companion main story {dev_f.lower()}",
                )
                o_rec = CharacterOutfitRecord(
                    outfit_id=f"bv:{gid}:{oid}",
                    char_id=ckey,
                    display_name=var_label,
                    tribe=tribe,
                    dev_folder=dev_f,
                    resource_type="BodyVariant",
                    resource_id=f"{gid}:{oid}",
                    facial_id=facial_id,
                    has_headgear_toggle=False,
                    is_dlc="/dlc/" in dev_f.lower() or dev_f.lower().startswith("dlc/"),
                )
                companion_groups.setdefault(ckey, []).append((c_rec, o_rec))
            else:
                npc_records.append(
                    CharacterRecord(
                        char_id=ckey,
                        display_name=disp,
                        character_kind="npc",
                        tribe=tribe,
                        gender=gender,
                        dev_folder=dev_f,
                        body_variant_id=f"{gid}:{oid}",
                        facial_id=facial_id,
                        hair_id="",
                        outfit_count=1,
                        part_count=len(mps) + (1 if facial_id else 0),
                        tags=f"{disp.lower()} {tribe.lower()} {gender} npc {dev_f.lower()}",
                    )
                )

    # Finalize grouped companions (pick the most complete BodyVariant with BOTH body & face as default!)
    for ckey, pairs in sorted(companion_groups.items()):
        pairs.sort(key=lambda pair: (1 if pair[0].facial_id else 0, pair[0].part_count), reverse=True)
        best_c = pairs[0][0]
        seen_oid: set[str] = set()
        comp_outfits: list[CharacterOutfitRecord] = []
        for _c, o in pairs:
            if o.outfit_id not in seen_oid:
                seen_oid.add(o.outfit_id)
                comp_outfits.append(o)
        best_c.outfit_count = len(comp_outfits)
        characters.append(best_c)
        outfits.extend(comp_outfits)

    # Sort NPCs: those with BOTH body + head first, then alphabetically by tribe/name
    npc_records.sort(key=lambda r: (0 if r.facial_id else 1, r.tribe, r.display_name))
    characters.extend(npc_records)
    return characters, outfits, heads


# ==============================================================================
# NATURAL-LANGUAGE CHARACTER PROMPT RESOLVER
# ==============================================================================

_STOP_WORDS = frozenset({
    "a", "an", "the", "with", "and", "in", "of", "from", "hfw", "horizon", "character",
    "model", "rig", "mesh", "import", "export", "wearing", "outfit", "armor", "armour",
    "skin", "variation", "variant", "version", "please", "get", "me",
})


def resolve_character_prompt(
    catalog: AssetCatalog,
    prompt: str,
) -> dict[str, Any]:
    """Resolve an intuitive free-text prompt (e.g. `"Aloy Carja Blazon no headgear"`,
    `"Seyka"`, `"Erend"`, `"Tenakth female warrior"`, `"Carja male guard"`) into the best
    matching indexed character, outfit variation, headgear flag, hair flag, and top alternatives!
    """
    raw = (prompt or "").strip()
    low = raw.lower()

    # Parse natural modifiers directly from prompt
    headgear = True
    if any(p in low for p in ("no headgear", "without headgear", "no helmet", "without helmet", "no hat", "headgear off", "hide headgear", "bare head")):
        headgear = False
    include_hair = True
    if any(p in low for p in ("no hair", "without hair", "bald")):
        include_hair = False
    hair_lod = 2
    if any(p in low for p in ("lod0", "ultra hair", "cinematic hair")):
        hair_lod = 0
    elif any(p in low for p in ("lod1", "high hair")):
        hair_lod = 1
    elif any(p in low for p in ("lod3", "low poly hair", "light hair")):
        hair_lod = 3

    # Direct ID shortcut (`aloy`, `bv:21350:1`, `osd:72966:0`)
    if raw.startswith("osd:") or raw.startswith("bv:"):
        o_row = catalog.get_character_outfit(raw)
        if o_row:
            c_row = catalog.get_character(o_row["char_id"])
            return {
                "prompt": raw,
                "matched": True,
                "character": c_row,
                "outfit": o_row,
                "headgear": headgear,
                "include_hair": include_hair,
                "hair_lod": hair_lod,
                "confidence": 1.0,
                "alternatives": [],
                "summary": f"{c_row['display_name'] if c_row else o_row['char_id']} — {o_row['display_name']}",
            }
        c_row = catalog.get_character(raw)
        if c_row:
            return {
                "prompt": raw,
                "matched": True,
                "character": c_row,
                "outfit": None,
                "headgear": headgear,
                "include_hair": include_hair,
                "hair_lod": hair_lod,
                "confidence": 1.0,
                "alternatives": [],
                "summary": c_row["display_name"],
            }

    # Tokenize query removing modifiers & stop words
    cleaned = re.sub(r"\b(no|without|hide)\s+(headgear|helmet|hat|hair)\b", " ", low)
    cleaned = re.sub(r"\b(headgear\s+off|ultra\s+hair|cinematic\s+hair|lod[0-3])\b", " ", cleaned)
    tokens = [t for t in re.findall(r"[a-z0-9]+", cleaned) if t and t not in _STOP_WORDS]
    if not tokens:
        tokens = ["aloy"]

    wanted_gender: str | None = None
    if "female" in tokens or "woman" in tokens or "girl" in tokens:
        wanted_gender = "female"
    elif "male" in tokens or "man" in tokens or "guy" in tokens:
        wanted_gender = "male"
    elif "child" in tokens or "kid" in tokens or "boy" in tokens:
        wanted_gender = "child"

    all_chars = catalog.list_characters(limit=2000)
    all_outfits = catalog.list_character_outfits(limit=500)
    outfits_by_char: dict[str, list[dict[str, Any]]] = {}
    for o in all_outfits:
        outfits_by_char.setdefault(o["char_id"], []).append(o)

    scored: list[tuple[float, dict[str, Any], dict[str, Any] | None]] = []

    for ch in all_chars:
        ch_name = ch["display_name"].lower()
        ch_id = ch["char_id"].lower()
        ch_tribe = ch["tribe"].lower()
        ch_gender = ch["gender"].lower()
        ch_folder = ch["dev_folder"].lower()
        ch_tags = ch.get("tags", "").lower()
        ch_blob = f"{ch_name} {ch_id} {ch_tribe} {ch_gender} {ch_folder} {ch_tags}"

        base_score = 0.0
        # Exact character name match gets huge boost!
        if ch_name == " ".join(tokens) or ch_id == "_".join(tokens):
            base_score += 120.0
        elif any(t == ch_name or t == ch_id for t in tokens):
            base_score += 80.0
        else:
            for t in tokens:
                if t in ch_name:
                    base_score += 24.0
                elif t in ch_tribe:
                    base_score += 18.0
                elif t in ch_folder:
                    base_score += 14.0
                elif t in ch_blob:
                    base_score += 8.0

        # Gender alignment bonus/penalty
        if wanted_gender:
            if ch_gender == wanted_gender:
                base_score += 20.0
            elif ch_gender != "unknown":
                base_score -= 18.0

        # Prefer complete characters with face + Hero/Companion priority when tied
        if ch.get("facial_id"):
            base_score += 6.0
        if ch["character_kind"] == "hero":
            base_score += 5.0
        elif ch["character_kind"] == "companion":
            base_score += 3.0

        # Score outfit variations under this character if any!
        ch_outfits = outfits_by_char.get(ch["char_id"], [])
        if ch_outfits:
            best_o: dict[str, Any] | None = None
            best_o_score = -1.0
            for o in ch_outfits:
                o_blob = f"{o['display_name']} {o['tribe']} {o['dev_folder']}".lower()
                oscore = 0.0
                for t in tokens:
                    if t == ch_name or t == ch_id:
                        continue
                    if t in o["display_name"].lower():
                        oscore += 32.0
                    elif t in o["tribe"].lower():
                        oscore += 18.0
                    elif t in o_blob:
                        oscore += 14.0
                # Default starter outfit preference when user simply typed "Aloy"
                if oscore == 0.0 and o["resource_id"] == ch.get("body_variant_id"):
                    oscore += 4.0
                if oscore > best_o_score:
                    best_o_score = oscore
                    best_o = o
            total = base_score + max(0.0, best_o_score)
            if total > 10.0:
                scored.append((total, ch, best_o))
        else:
            if base_score > 10.0:
                scored.append((base_score, ch, None))

    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        # Fallback to Aloy default if nothing matched
        aloy = catalog.get_character("aloy")
        return {
            "prompt": raw,
            "matched": False,
            "character": aloy,
            "outfit": outfits_by_char.get("aloy", [None])[0] if aloy else None,
            "headgear": headgear,
            "include_hair": include_hair,
            "hair_lod": hair_lod,
            "confidence": 0.0,
            "alternatives": [],
            "summary": f"No exact match for '{raw}' (fallback: Aloy)",
        }

    top_score, top_ch, top_o = scored[0]
    alts = [
        {
            "char_id": c["char_id"],
            "display_name": c["display_name"],
            "outfit_id": o["outfit_id"] if o else "",
            "outfit_name": o["display_name"] if o else "",
            "tribe": c["tribe"],
            "gender": c["gender"],
            "score": round(sc, 1),
        }
        for sc, c, o in scored[1:8]
    ]
    summary = f"{top_ch['display_name']}" + (f" — {top_o['display_name']}" if top_o else "")
    if not headgear:
        summary += " (No Headgear)"
    return {
        "prompt": raw,
        "matched": True,
        "character": top_ch,
        "outfit": top_o,
        "headgear": headgear,
        "include_hair": include_hair,
        "hair_lod": hair_lod,
        "confidence": min(1.0, round(top_score / 110.0, 2)),
        "alternatives": alts,
        "summary": summary,
    }


# ==============================================================================
# HAIR DECODER (`HairResource` -> `HairSkinnedMeshLod`)
# ==============================================================================

def decode_hair_skinned_meshes(
    game: HFWGame,
    hair_obj: dict[str, Any],
    head_world_matrix: np.ndarray,
    head_joint_index: int,
    neck_joint_index: int | None = None,
    spine_joint_index: int | None = None,
    lod_index: int = 2,
) -> list[DecodedMesh]:
    """Decode Aloy's `HairResource` (`43305:37`) `HairSkinnedMeshLod` hair cards into world-space skinned meshes.

    Hair vertices live in `C_Head_sjnt_0` local space; this transforms them into exact bind-pose
    world space and computes smooth scalp->neck->spine skinning weights so her braids follow her
    head cleanly without stabbing into her shoulders in Blender Pose Mode & UE5!
    """
    lods = hair_obj.get("MeshLods") or []
    if not lods:
        return []
    pick = max(0, min(lod_index, len(lods) - 1))
    lod = lods[pick]
    skinned_meshes = lod.get("SkinnedMeshes") or []
    out_meshes: list[DecodedMesh] = []

    # HairSkinnedMeshLod positions live in WORLD-ORIENTED head-origin space (`pos_local + C_Head_sjnt_0.pos`).
    # Do NOT multiply by C_Head_sjnt_0's local bone-axis rotation (which swaps Y/Z axes for bone orientation)!
    trans = head_world_matrix[:3, 3].astype(np.float32)
    neck_idx = neck_joint_index if neck_joint_index is not None else head_joint_index
    spine_idx = spine_joint_index if spine_joint_index is not None else neck_idx

    for sm_idx, sm in enumerate(skinned_meshes):
        va_obj = game.resolve_ref(sm.get("SkinnedVertexArray"))
        ia_obj = game.resolve_ref(sm.get("SkinnedIndexArray"))
        pos_db = game.resolve_ref(sm.get("SkinnedPositionDataBufferResource"))
        eff_obj = game.resolve_ref(sm.get("SkinnedRenderEffectResource"))
        if not va_obj or not ia_obj or not pos_db:
            continue

        va = va_obj.get("_vertex_array")
        ia = ia_obj.get("_index_array")
        pdb = pos_db.get("_data_buffer")
        if not va or not ia or not pdb:
            continue

        vcount = int(va.get("count", 0))
        p_raw = pdb.get("data") or b""
        if not p_raw and pos_db.get("StreamingDataSource"):
            p_raw = game.read_data_source(pos_db["StreamingDataSource"])
        p_stride = int(pdb.get("stride", 16))
        if vcount <= 0 or len(p_raw) < vcount * p_stride:
            continue

        pos_local = np.frombuffer(p_raw[: vcount * p_stride], dtype=np.float32).reshape(vcount, p_stride // 4)[:, :3].copy()
        # Translate from world-aligned Head-origin space into exact World bind-pose space
        pos_world = pos_local + trans

        # Decode index buffer
        icount = int(ia.get("count", 0))
        i_raw = ia.get("data") or b""
        idtype = np.uint16 if int(ia.get("format", 1)) == 0 else np.uint32
        if icount <= 0 or len(i_raw) < icount * np.dtype(idtype).itemsize:
            continue
        indices = np.frombuffer(i_raw[: icount * np.dtype(idtype).itemsize], dtype=idtype).astype(np.uint32, copy=True)

        # Decode Normals, Tangents, UVs, Colors from SkinnedVertexArray
        normals: np.ndarray | None = None
        tangents: np.ndarray | None = None
        uvs_by_slot: dict[int, np.ndarray] = {}
        colors: list[np.ndarray] = []

        for stream in va.get("streams") or ():
            stride = int(stream.get("stride", 0))
            sdata = stream.get("data") or b""
            if stride <= 0 or len(sdata) < vcount * stride:
                continue
            for elem in stream.get("elements") or ():
                sem_name = game.schema.enum_name("EVertexElement", elem["element"])
                st_name = elem["storage_type_name"]
                arr = _unpack_element(sdata, vcount, stride, elem["offset"], st_name, elem["slots_used"])
                if arr is None:
                    continue
                if sem_name == "Normal":
                    normals = arr[:, :3].astype(np.float32, copy=False)
                elif sem_name == "Tangent":
                    tangents = arr.astype(np.float32, copy=False)
                elif sem_name == "Color":
                    if arr.shape[1] == 3:
                        arr = np.concatenate([arr.astype(np.float32), np.ones((vcount, 1), dtype=np.float32)], axis=1)
                    colors.append(arr[:, :4].astype(np.float32, copy=False))
                elif sem_name.startswith("UV") and sem_name[2:].isdigit():
                    uvs_by_slot[int(sem_name[2:])] = arr[:, :2].astype(np.float32, copy=False)

        # Smooth height-graded humanoid skinning weights in head-local Z:
        #   Crown/Scalp (z >= -0.14): 100% Head
        #   Neck/Shoulders (-0.28 .. -0.14): smooth blend Head -> Neck -> Spine
        #   Lower Back Braids (z < -0.28): 35% Head + 20% Neck + 45% Spine
        lz = pos_local[:, 2]
        t_drape = np.clip((-0.14 - lz) / 0.18, 0.0, 1.0)
        w_head = (1.0 - 0.65 * t_drape).astype(np.float32)
        w_neck = (0.20 * t_drape).astype(np.float32)
        w_spine = (0.45 * t_drape).astype(np.float32)

        bi = np.empty((vcount, 3), dtype=np.uint16)
        bi[:, 0] = head_joint_index
        bi[:, 1] = neck_idx
        bi[:, 2] = spine_idx
        bw = np.stack([w_head, w_neck, w_spine], axis=-1)

        mat_id = str(eff_obj["_id"]) if eff_obj is not None else None
        uvs = [uvs_by_slot[k] for k in sorted(uvs_by_slot)]

        prim = DecodedPrimitive(
            primitive_index=sm_idx,
            material_id=mat_id,
            positions=pos_world,
            indices=indices,
            normals=normals,
            tangents=tangents,
            uvs=uvs,
            colors=colors,
            bone_indices=bi,
            bone_weights=bw,
        )
        out_meshes.append(
            DecodedMesh(
                object_id=f"{hair_obj.get('_id', 'hair')}_lod{pick}_{sm_idx}",
                mesh_type="HairSkinnedMesh",
                is_skinned=True,
                primitives=[prim],
            )
        )

    return out_meshes


# ==============================================================================
# MASTER HUMANOID SKELETON MERGER & CHARACTER ASSEMBLER
# ==============================================================================

def _merge_skeleton_into_master(
    master: SkeletonData,
    secondary: SkeletonData,
) -> dict[int, int]:
    """Merge `secondary` joints (e.g. `FacialSkeleton` `3993:0` or outfit/gear skeleton) into `master`.

    Matches overlapping root/core bones (`C_Spine_sjnt_4`, `C_Neck_sjnt_0..2`, `C_Head_sjnt_0`,
    `L/R_Clavicle`, `L/R_Arm`) by UUID or exact bone name, appends any new child joints (such as
    `FACIAL_*` or outfit cloth bones) parented cleanly under their mapped master parent, and
    returns `{secondary_joint_index: master_joint_index}` for exact skin-weight remapping!
    """
    sec_to_master: dict[int, int] = {}

    # Pass 1: Match all shared bones by UUID or exact Name
    for sj in secondary.joints:
        mj = master.joint_by_uuid.get(sj.uuid) if sj.uuid else None
        if mj is None and sj.name:
            mj = master.joint_by_name.get(sj.name)
        if mj is not None:
            sec_to_master[sj.index] = mj.index

    # Pass 2: Append unmatched child bones in topological order (parents before children)
    unmatched = [sj for sj in secondary.joints if sj.index not in sec_to_master]
    # Multiple passes handle arbitrary parent ordering safely
    for _ in range(8):
        if not unmatched:
            break
        remaining: list[JointInfo] = []
        progressed = False
        for sj in unmatched:
            p_master: int = -1
            if 0 <= sj.parent_index < len(secondary.joints):
                if sj.parent_index not in sec_to_master:
                    remaining.append(sj)
                    continue
                p_master = sec_to_master[sj.parent_index]

            new_idx = len(master.joints)
            wmat = sj.world_matrix.copy()
            lmat = sj.local_matrix.copy()
            # If secondary world_matrix is identity/origin while parent isn't, propagate from parent
            if p_master >= 0 and np.allclose(wmat, np.eye(4)) and not np.allclose(master.joints[p_master].world_matrix, np.eye(4)):
                wmat = master.joints[p_master].world_matrix @ lmat
            elif p_master >= 0:
                try:
                    lmat = np.linalg.inv(master.joints[p_master].world_matrix) @ wmat
                except np.linalg.LinAlgError:
                    pass

            new_j = JointInfo(
                index=new_idx,
                name=sj.name or f"Joint_{new_idx}",
                parent_index=p_master,
                uuid=sj.uuid,
                local_matrix=lmat,
                world_matrix=wmat,
            )
            master.joints.append(new_j)
            if new_j.uuid:
                master.joint_by_uuid[new_j.uuid] = new_j
            if new_j.name and new_j.name not in master.joint_by_name:
                master.joint_by_name[new_j.name] = new_j
            sec_to_master[sj.index] = new_idx
            progressed = True

        unmatched = remaining
        if not progressed:
            # Attach any remaining orphan roots cleanly to C_Head_sjnt_0 or root 0
            fallback_p = master.joint_by_name["C_Head_sjnt_0"].index if "C_Head_sjnt_0" in master.joint_by_name else (0 if master.joints else -1)
            for sj in unmatched:
                new_idx = len(master.joints)
                new_j = JointInfo(
                    index=new_idx,
                    name=sj.name or f"Joint_{new_idx}",
                    parent_index=fallback_p,
                    uuid=sj.uuid,
                    local_matrix=sj.local_matrix.copy(),
                    world_matrix=sj.world_matrix.copy(),
                )
                master.joints.append(new_j)
                if new_j.uuid:
                    master.joint_by_uuid[new_j.uuid] = new_j
                if new_j.name and new_j.name not in master.joint_by_name:
                    master.joint_by_name[new_j.name] = new_j
                sec_to_master[sj.index] = new_idx
            break

    return sec_to_master


def _remap_mesh_weights(
    game: HFWGame,
    mesh: DecodedMesh,
    master: SkeletonData,
    skel_cache: dict[str, SkeletonData],
    merge_maps: dict[str, dict[int, int]],
) -> None:
    """Remap all primitive `bone_indices` of `mesh` onto `master`'s unified joint table."""
    if not mesh.is_skinned or not mesh.skeleton_id or mesh.skeleton_id == master.object_id:
        return

    sid = mesh.skeleton_id
    if sid not in merge_maps:
        # Load & merge this secondary skeleton into master!
        sec_sk = skel_cache.get(sid)
        if sec_sk is None:
            try:
                sk_obj = game.read_object(ObjectId.parse(sid), read_subgroups=True)
                sec_sk = extract_skeleton(game, sk_obj)
                skel_cache[sid] = sec_sk
            except Exception:
                sec_sk = None
        if sec_sk is not None:
            merge_maps[sid] = _merge_skeleton_into_master(master, sec_sk)
        else:
            # Fallback match directly from mesh.skeleton_joint_uuids / names
            m: dict[int, int] = {}
            for idx, (ju, jn) in enumerate(zip(mesh.skeleton_joint_uuids, mesh.skeleton_joint_names)):
                mj = master.joint_by_uuid.get(ju) if ju else None
                if mj is None and jn:
                    mj = master.joint_by_name.get(jn)
                if mj is not None:
                    m[idx] = mj.index
            merge_maps[sid] = m

    lut_map = merge_maps[sid]
    if not lut_map:
        return
    max_src = max(lut_map.keys()) + 1
    lut = np.zeros(max(max_src, 2048), dtype=np.uint16)
    for src_i, dst_i in lut_map.items():
        if src_i < len(lut):
            lut[src_i] = dst_i

    for pr in mesh.primitives:
        if pr.bone_indices is not None:
            clipped = np.clip(pr.bone_indices, 0, len(lut) - 1)
            pr.bone_indices = lut[clipped]


def _classify_character_part_collection(name: str, source_kind: str, mat_names: list[str]) -> str:
    if source_kind == "HairSkinnedMesh":
        return "Hair"
    if source_kind == "FacialMesh":
        return "Head_Face"
    if source_kind == "OutfitAlt":
        return "Outfit_Alt"
    blob = f"{name} {' '.join(mat_names)}".lower()
    if any(k in blob for k in ("hair", "braid", "ponytail", "beard", "eyebrow")):
        return "Hair"
    if any(k in blob for k in ("head", "face", "eye", "teeth", "lash")):
        return "Head_Face"
    if any(k in blob for k in ("quiver", "pouch", "bag", "focus", "necklace", "earring", "weapon", "belt", "cloak", "cape")):
        return "Accessories"
    if any(k in blob for k in ("basebody", "body_female", "body_male")):
        return "Body"
    return "Outfit"


def assemble_character(
    game: HFWGame,
    catalog: AssetCatalog | None = None,
    *,
    prompt: str | None = None,
    char_id: str | None = None,
    outfit_id: str | None = None,
    facial_id: str | None = None,
    facial_override: str | None = None,
    headgear: bool | None = None,
    include_hair: bool | None = None,
    hair_lod: int | None = None,
    lod_index: int = 0,
) -> MachineAssembly:
    """Assemble a complete HFW Character (Aloy, Companion, or Tribal NPC) with ALL mesh segments,
    textures, materials, and ONE unified usable Humanoid Armature (Body + Outfit + Face + Hair)!

    Accepts EITHER a natural-language `prompt` (e.g. `"Aloy Carja Blazon no headgear"`, `"Seyka"`,
    `"Erend"`, `"Tenakth female warrior"`) OR explicit `char_id` / `outfit_id` / `facial_id` IDs!
    """
    resolved_char: dict[str, Any] | None = None
    resolved_outfit: dict[str, Any] | None = None
    eff_headgear = True if headgear is None else bool(headgear)
    eff_hair = True if include_hair is None else bool(include_hair)
    eff_hair_lod = 2 if hair_lod is None else int(hair_lod)

    if prompt and catalog is not None:
        res = resolve_character_prompt(catalog, prompt)
        resolved_char = res.get("character")
        resolved_outfit = res.get("outfit")
        if headgear is None:
            eff_headgear = bool(res.get("headgear", True))
        if include_hair is None:
            eff_hair = bool(res.get("include_hair", True))
        if hair_lod is None:
            eff_hair_lod = int(res.get("hair_lod", 2))
    elif char_id and catalog is not None:
        resolved_char = catalog.get_character(char_id)
        if resolved_char is None:
            res = resolve_character_prompt(catalog, char_id)
            resolved_char = res.get("character")
            resolved_outfit = res.get("outfit")

    if outfit_id and catalog is not None:
        # Explicit outfit override (ID or search keyword!)
        o_exact = catalog.get_character_outfit(outfit_id)
        if o_exact is not None:
            resolved_outfit = o_exact
        else:
            all_o = catalog.list_character_outfits(char_id=resolved_char["char_id"] if resolved_char else None, limit=300)
            ol = outfit_id.lower()
            for o in all_o:
                if ol in o["display_name"].lower() or ol in o["dev_folder"].lower() or ol in o["resource_id"]:
                    resolved_outfit = o
                    break

    # Determine primary resource ID (`OutfitStreamingData` or `BodyVariant`)
    primary_type = "BodyVariant"
    primary_res_id = ""
    eff_facial_id = facial_id or facial_override or ""
    eff_hair_id = ""
    disp_name = "Character"
    codename = "character"
    variant_label = ""
    dev_folder = ""

    if resolved_outfit is not None:
        primary_type = resolved_outfit["resource_type"]
        primary_res_id = resolved_outfit["resource_id"]
        eff_facial_id = eff_facial_id or resolved_outfit.get("facial_id") or (resolved_char.get("facial_id") if resolved_char else "")
        disp_name = resolved_char["display_name"] if resolved_char else resolved_outfit["display_name"]
        codename = resolved_char["char_id"] if resolved_char else resolved_outfit["outfit_id"]
        variant_label = resolved_outfit["display_name"]
        dev_folder = resolved_outfit.get("dev_folder", "")
        if resolved_char and resolved_char.get("hair_id"):
            eff_hair_id = resolved_char["hair_id"]
    elif resolved_char is not None:
        bv_id = resolved_char["body_variant_id"]
        # Check if char_id is Aloy pointing at an OSD group
        g_id_i = int(bv_id.split(":")[0])
        if "OutfitStreamingData" in game.graph.group_types(g_id_i):
            primary_type = "OutfitStreamingData"
        else:
            primary_type = "BodyVariant"
        primary_res_id = bv_id
        eff_facial_id = eff_facial_id or resolved_char.get("facial_id", "")
        eff_hair_id = resolved_char.get("hair_id", "")
        disp_name = resolved_char["display_name"]
        codename = resolved_char["char_id"]
        variant_label = resolved_char.get("tribe", "Default")
        dev_folder = resolved_char.get("dev_folder", "")
    elif outfit_id and ":" in outfit_id:
        clean_id = outfit_id.split(":", 1)[1] if outfit_id.startswith(("osd:", "bv:")) else outfit_id
        primary_type = "OutfitStreamingData" if outfit_id.startswith("osd:") else "BodyVariant"
        primary_res_id = clean_id
        disp_name = f"Character_{clean_id.replace(':', '_')}"
        codename = disp_name.lower()
    elif char_id and ":" in char_id:
        clean_id = char_id.split(":", 1)[1] if char_id.startswith(("osd:", "bv:")) else char_id
        primary_type = "OutfitStreamingData" if char_id.startswith("osd:") else "BodyVariant"
        primary_res_id = clean_id
        disp_name = f"Character_{clean_id.replace(':', '_')}"
        codename = disp_name.lower()
    else:
        raise ValueError(f"Could not resolve character from prompt={prompt!r}, char_id={char_id!r}, outfit_id={outfit_id!r}")

    # 1. Load primary resource (OutfitStreamingData or BodyVariant) and collect ModelPartResources + Components!
    primary_obj = game.read_object(ObjectId.parse(primary_res_id), read_subgroups=True)
    raw_model_parts: list[tuple[str, dict[str, Any], bool]] = []  # (label, ModelPartResource, enabled_by_default)
    candidate_skeletons: list[SkeletonData] = []
    skel_cache: dict[str, SkeletonData] = {}
    helper_objs: list[dict[str, Any]] = []

    def _register_skel_obj(sk_obj: dict[str, Any] | None) -> SkeletonData | None:
        if sk_obj is None:
            return None
        sid = str(sk_obj.get("_id", ""))
        if sid in skel_cache:
            return skel_cache[sid]
        sd = extract_skeleton(game, sk_obj)
        skel_cache[sid] = sd
        candidate_skeletons.append(sd)
        return sd

    if primary_type == "OutfitStreamingData":
        # Check group for body Skeleton & SkeletonHelpers
        pgid = ObjectId.parse(primary_res_id).group_id
        for obj in game.read_group(pgid, read_subgroups=True):
            if obj.get("_type") == "Skeleton":
                _register_skel_obj(obj)
            elif obj.get("_type") == "SkeletonHelpers":
                helper_objs.append(obj)

        osd_mps = primary_obj.get("ModelParts") or []
        # Analyze each OSD ModelPart's Z bounds to separate Headgear-On vs Headgear-Off duplicate bodies!
        decoded_osd_entries: list[tuple[int, dict[str, Any], list[DecodedMesh], float, float]] = []
        for idx, entry in enumerate(osd_mps):
            mpr = game.resolve_ref(entry.get("ModelPartResource"))
            if mpr is None:
                continue
            mres = game.resolve_ref(mpr.get("MeshResource"))
            dmeshes = decode_mesh_hierarchy(game, mres, lod_index=lod_index) if mres else []
            if not dmeshes:
                continue
            pts = np.concatenate([pr.positions for m in dmeshes for pr in m.primitives], axis=0)
            zmin, zmax = float(pts[:, 2].min()), float(pts[:, 2].max())
            decoded_osd_entries.append((idx, mpr, dmeshes, zmin, zmax))

        # Full-body variants span from feet (zmin < 0.25) to shoulders/head (zmax > 1.35)
        full_body_entries = [e for e in decoded_osd_entries if e[3] < 0.25 and e[4] > 1.35]
        other_entries = [e for e in decoded_osd_entries if e not in full_body_entries]

        chosen_body_idx: int | None = None
        if len(full_body_entries) >= 2:
            # Taller zmax = WITH headgear; shorter zmax = WITHOUT headgear!
            full_body_entries.sort(key=lambda e: e[4], reverse=eff_headgear)
            chosen_body_idx = full_body_entries[0][0]
        elif len(full_body_entries) == 1:
            chosen_body_idx = full_body_entries[0][0]

        for idx, mpr, dmeshes, _zmin, _zmax in decoded_osd_entries:
            is_primary_body = (chosen_body_idx is None) or (idx == chosen_body_idx) or (idx in [e[0] for e in other_entries])
            src_kind = "OutfitPrimary" if is_primary_body else "OutfitAlt"
            raw_model_parts.append((src_kind, {"_mpr": mpr, "_decoded": dmeshes, "_idx": idx}, is_primary_body))
    else:
        # Standard BodyVariant (Companions & NPCs!)
        bv_mps = primary_obj.get("ModelPartResources") or []
        init_states = primary_obj.get("InitiallyEnabledStates") or ()
        for idx, mp_ref in enumerate(bv_mps):
            mpr = game.resolve_ref(mp_ref)
            if mpr is None:
                continue
            enabled = bool(init_states[idx]) if idx < len(init_states) else True
            if not enabled:
                continue
            raw_model_parts.append(("BodyVariantPart", mpr, True))

        # Inspect BodyVariant's EntityComponentResources for FacialAnimationComponentResource & SkeletonHelpers
        for ec_ref in primary_obj.get("EntityComponentResources") or ():
            ec = game.resolve_ref(ec_ref)
            if ec is None:
                continue
            et = ec.get("_type", "")
            if et == "FacialAnimationComponentResource" and not eff_facial_id:
                eff_facial_id = str(ec["_id"])
            elif et == "HairModelComponentResource" and not eff_hair_id:
                hr = game.resolve_ref(ec.get("HairResource"))
                if hr is not None:
                    eff_hair_id = str(hr["_id"])
            elif et == "EntityComponentSetResource":
                for sub_ref in ec.get("ComponentResources") or ():
                    sc = game.resolve_ref(sub_ref)
                    if sc is None:
                        continue
                    if sc.get("_type") == "FacialAnimationComponentResource" and not eff_facial_id:
                        eff_facial_id = str(sc["_id"])
                    elif sc.get("_type") == "SkeletonHelpersComponentResource":
                        for sh_ref in sc.get("SkeletonHelpers") or ():
                            sh = game.resolve_ref(sh_ref)
                            if sh is not None:
                                helper_objs.append(sh)

    # 2. Decode all Body / Outfit ModelPartResources & collect their skeletons
    decoded_parts_queue: list[tuple[str, str, list[DecodedMesh], bool]] = []  # (name, source_kind, meshes, enabled)
    for src_kind, entry, enabled in raw_model_parts:
        if "_decoded" in entry:
            mpr = entry["_mpr"]
            dmeshes = entry["_decoded"]
            p_idx = entry["_idx"]
            pname = f"OutfitPart_{p_idx}_{str(mpr.get('_id', '')).replace(':', '_')}"
        else:
            mpr = entry
            mres = game.resolve_ref(mpr.get("MeshResource"))
            dmeshes = decode_mesh_hierarchy(game, mres, lod_index=lod_index) if mres else []
            pname = f"BodyPart_{str(mpr.get('_id', '')).replace(':', '_')}"
        if not dmeshes:
            continue
        for dm in dmeshes:
            if dm.skeleton_id and dm.skeleton_id not in skel_cache:
                try:
                    _register_skel_obj(game.read_object(ObjectId.parse(dm.skeleton_id), read_subgroups=True))
                except Exception:
                    pass
        decoded_parts_queue.append((pname, src_kind, dmeshes, enabled))

    # 3. Decode Head / Face (`FacialAnimationComponentResource` -> `FacialMesh` + `FacialSkeleton`)
    facial_skel: SkeletonData | None = None
    if eff_facial_id:
        try:
            fac_obj = game.read_object(ObjectId.parse(eff_facial_id), read_subgroups=True)
            fsk_obj = game.resolve_ref((fac_obj.get("FacialSkeleton") or {}).get("Value"))
            if fsk_obj is not None:
                facial_skel = _register_skel_obj(fsk_obj)
            fmesh_obj = game.resolve_ref((fac_obj.get("FacialMesh") or {}).get("Value"))
            fmeshes = decode_mesh_hierarchy(game, fmesh_obj, lod_index=lod_index) if fmesh_obj else []
            if fmeshes:
                for fm in fmeshes:
                    if not fm.skeleton_id and facial_skel is not None:
                        fm.skeleton_id = facial_skel.object_id
                decoded_parts_queue.append((f"Head_{eff_facial_id.replace(':', '_')}", "FacialMesh", fmeshes, True))
        except Exception:
            pass

    # 4. Build ONE Unified Master Humanoid Skeleton!
    #    Pick the body skeleton (prefer non-facial skeleton with the most joints, falling back to facial skeleton)
    non_facial_skels = [s for s in candidate_skeletons if facial_skel is None or s.object_id != facial_skel.object_id]
    if non_facial_skels:
        non_facial_skels.sort(key=lambda s: len(s.joints), reverse=True)
        master_skel = non_facial_skels[0]
    elif facial_skel is not None:
        master_skel = facial_skel
    else:
        master_skel = SkeletonData(object_id="none")

    for sh in helper_objs:
        try:
            add_skeleton_helpers(master_skel, sh)
        except Exception:
            pass

    merge_maps: dict[str, dict[int, int]] = {}
    # Pre-merge facial skeleton so facial bones & head joint are guaranteed in master_skel before hair!
    if facial_skel is not None and facial_skel.object_id != master_skel.object_id:
        merge_maps[facial_skel.object_id] = _merge_skeleton_into_master(master_skel, facial_skel)

    # 5. Decode Aloy's Iconic Braided Hair (`HairResource` `43305:37`) aligned & skinned to Head/Neck/Spine!
    if eff_hair and eff_hair_id:
        try:
            hair_obj = game.read_object(ObjectId.parse(eff_hair_id), read_subgroups=True)
            head_j = master_skel.joint_by_name.get("C_Head_sjnt_0")
            neck_j = master_skel.joint_by_name.get("C_Neck_sjnt_2") or master_skel.joint_by_name.get("C_Neck_sjnt_1")
            spine_j = master_skel.joint_by_name.get("C_Spine_sjnt_4") or master_skel.joint_by_name.get("C_Spine_sjnt_3")
            head_wmat = head_j.world_matrix if head_j is not None else np.array([
                [1, 0, 0, 0.0],
                [0, 1, 0, 0.0077],
                [0, 0, 1, 1.5459],
                [0, 0, 0, 1.0],
            ], dtype=np.float64)
            head_idx = head_j.index if head_j is not None else 0
            h_meshes = decode_hair_skinned_meshes(
                game,
                hair_obj,
                head_world_matrix=head_wmat,
                head_joint_index=head_idx,
                neck_joint_index=neck_j.index if neck_j is not None else None,
                spine_joint_index=spine_j.index if spine_j is not None else None,
                lod_index=eff_hair_lod,
            )
            if h_meshes:
                decoded_parts_queue.append(("Aloy_Hair_Braids", "HairSkinnedMesh", h_meshes, True))
        except Exception:
            pass

    # 6. Remap ALL skin weights onto master_skel, parse ALL Materials, and emit clean AssembledParts!
    materials: dict[str, MaterialInfo] = {}
    parts: list[AssembledPart] = []
    ident = np.eye(4, dtype=np.float64)

    for base_name, src_kind, dmeshes, enabled in decoded_parts_queue:
        for m_i, dm in enumerate(dmeshes):
            # Remap weights onto unified master skeleton
            _remap_mesh_weights(game, dm, master_skel, skel_cache, merge_maps)

            # Parse every material referenced by this mesh
            mat_names: list[str] = []
            for pr in dm.primitives:
                if pr.material_id and pr.material_id not in materials:
                    try:
                        m_obj = game.read_object(ObjectId.parse(pr.material_id), read_subgroups=True)
                        materials[pr.material_id] = parse_material(game, m_obj)
                    except Exception:
                        pass
                if pr.material_id and pr.material_id in materials:
                    mat_names.append(materials[pr.material_id].name)

            coll = _classify_character_part_collection(base_name, src_kind, mat_names)
            part_name = base_name if len(dmeshes) == 1 else f"{base_name}_{m_i}"
            wmat = dm.local_transform if (dm.local_transform is not None and not dm.is_skinned) else ident

            parts.append(
                AssembledPart(
                    part_id=f"{dm.object_id}_{m_i}",
                    name=part_name,
                    collection=coll,
                    source_kind=src_kind,
                    is_skinned=dm.is_skinned,
                    parent_joint_index=None,
                    parent_joint_name=None,
                    helper_name=None,
                    local_offset_matrix=wmat.copy(),
                    world_matrix=wmat.copy(),
                    mesh=dm,
                    enabled_by_default=enabled,
                    state_index=0 if enabled else 1,
                )
            )

    avail_vars: list[MachineVariantOption] = []
    if catalog is not None and resolved_char is not None:
        for vi, o in enumerate(catalog.list_character_outfits(char_id=resolved_char["char_id"], limit=100)):
            avail_vars.append(
                MachineVariantOption(
                    variant_index=vi,
                    label=o["display_name"],
                    destructibility_id=o["outfit_id"],
                    group_id=int(o["resource_id"].split(":")[0]) if ":" in o["resource_id"] else 0,
                    part_count=1,
                    official_name=o["display_name"],
                    official_description=o["dev_folder"],
                )
            )

    return MachineAssembly(
        entity_id=f"{primary_type}:{primary_res_id}",
        codename=codename,
        display_name=disp_name,
        variant_label=variant_label or dev_folder,
        available_variants=avail_vars,
        skeleton=master_skel,
        parts=parts,
        materials=materials,
        official_variant_name=variant_label,
        official_variant_description=dev_folder,
    )
