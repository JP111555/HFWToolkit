"""FastAPI backend for HFWToolkit Material & Shader Deep-Dive Studio."""

from __future__ import annotations

import io
import os
import subprocess
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from hfw_toolkit.blender import build_blend_headless, export_assembly_bundle
from hfw_toolkit.cli import install_blender_addon_files
from hfw_toolkit.config import ToolkitConfig, auto_detect_blender_exe, auto_detect_game_dir
from hfw_toolkit.decima import AssetCatalog, HFWGame, ObjectId
from hfw_toolkit.extract import (
    ANIMATION_CATEGORIES,
    SPOKEN_LANGUAGE_LABELS,
    SUPPORTED_OUTPUT_FORMATS,
    AudioItemInfo,
    MachineAssembly,
    assemble_character,
    assemble_general_asset,
    assemble_machine,
    decode_audio_object,
    decode_texture,
    detect_digital_extras_ost,
    detect_installed_languages,
    discover_entity_animations,
    encode_audio_bytes,
    extract_audio_items_to_dir,
    extract_music_cues_to_dir,
    has_pyav,
    normalize_animation_category,
    render_channel_image,
    render_music_cue_mix,
    resolve_character_prompt,
    resolve_machine_entity_id,
    summarize_animation_categories,
    unpack_texture_set,
)
from hfw_toolkit.shaders import DXCDisassembler, ShaderHashDictionary, analyze_material_deep
from hfw_toolkit.shaders.pseudo_hlsl import find_external_decompilers

STATIC_DIR = Path(__file__).resolve().parent / "static"


class ExportBlendRequest(BaseModel):
    machine: str
    variant_index: int = 0
    lod_index: int = 0
    include_broken_states: bool = True
    unpack_textures: bool = False
    pack_textures: bool = False
    enable_detail_nodes: bool = True
    export_fbx: bool = False
    build_control_rig: bool = True
    animations: str | int | list[str] | None = None
    max_texture_size: int | None = None


class UpdateConfigRequest(BaseModel):
    game_dir: str | None = None
    blender_exe: str | None = None
    auto_detect: bool = False
    auto_update: bool | None = None
    update_branch: str | None = None
    check_updates_on_startup: bool | None = None


class UpdaterConfigRequest(BaseModel):
    auto_update: bool | None = None
    update_branch: str | None = None
    check_updates_on_startup: bool | None = None


class UpdaterApplyRequest(BaseModel):
    branch: str | None = None
    force: bool = False
    reinstall_deps: bool = True


class OpenBlendRequest(BaseModel):
    blend_path: str | None = None
    open_folder: bool = False


class AudioExtractRequest(BaseModel):
    category: str = "music"  # "music" | "sfx" | "voicelines" | "all"
    subcat: str = ""
    query: str = ""
    speaker: str = ""
    cue_id: str | None = None
    group_id: int | None = None
    object_ids: list[str] | None = None
    lang: str = "en"
    format: str = "wav"
    preserve_structure: bool = True
    mix_cues: bool = True
    export_stems: bool = True
    group_by: str = "quest"  # "quest" | "speaker"
    include_ost: bool = False
    write_transcript: bool = True
    limit: int | None = None
    out_dir: str | None = None


class AudioIndexRequest(BaseModel):
    rebuild: bool = False
    deep: bool = False


class OpenAudioFolderRequest(BaseModel):
    path: str | None = None


class ExportCharacterRequest(BaseModel):
    prompt: str = ""
    char_id: str = ""
    outfit_id: str = ""
    facial_override: str = ""
    headgear: bool = True
    include_hair: bool = True
    hair_lod: int = 2
    lod_index: int = 0
    unpack_textures: bool = False
    pack_textures: bool = False
    export_fbx: bool = False
    build_control_rig: bool = True
    animations: str | int | list[str] | None = None
    max_texture_size: int | None = None


class ExportDevAssetRequest(BaseModel):
    asset: str  # dev_path or resource_id
    lod_index: int = 0
    unpack_textures: bool = False
    pack_textures: bool = False
    export_fbx: bool = False
    max_texture_size: int | None = None


class ToolkitServerState:
    def __init__(self, config: ToolkitConfig) -> None:
        self.config = config
        self.game = HFWGame(config)
        self.catalog = AssetCatalog(config.index_db_path)
        self.dxc = DXCDisassembler(config)
        self.hash_dict = ShaderHashDictionary(config)
        self._asm_cache: OrderedDict[tuple[str, int], MachineAssembly] = OrderedDict()
        self._extra_asm_cache: OrderedDict[str, MachineAssembly] = OrderedDict()
        self._mat_cache: OrderedDict[tuple[str, bool], dict[str, Any]] = OrderedDict()
        self._tex_png_cache: OrderedDict[tuple[str, str, int], bytes] = OrderedDict()
        self._audio_bytes_cache: OrderedDict[tuple[str, str, str], bytes] = OrderedDict()
        self._lock = threading.Lock()
        self._db_lock = threading.RLock()
        self.export_jobs: dict[str, dict[str, Any]] = {}
        self.audio_jobs: dict[str, dict[str, Any]] = {}
        self._python_info_cache: dict[str, Any] | None = None
        self._update_check_cache: tuple[float, dict[str, Any]] | None = None

    def get_python_info(self, refresh: bool = False) -> dict[str, Any]:
        if self._python_info_cache is None or refresh:
            from hfw_toolkit.updater import detect_system_pythons

            self._python_info_cache = detect_system_pythons(self.config.toolkit_dir)
        return self._python_info_cache

    def get_extra_assembly(self, cache_key: str, builder_fn) -> MachineAssembly:
        with self._lock:
            cached = self._extra_asm_cache.get(cache_key)
            if cached is not None:
                self._extra_asm_cache.move_to_end(cache_key)
                return cached
        asm = builder_fn()
        with self._lock:
            self._extra_asm_cache[cache_key] = asm
            if len(self._extra_asm_cache) > 8:
                self._extra_asm_cache.popitem(last=False)
        return asm

    def get_assembly(self, entity_id: str, variant_index: int = 0) -> MachineAssembly:
        key = (entity_id, variant_index)
        with self._lock:
            cached = self._asm_cache.get(key)
            if cached is not None:
                self._asm_cache.move_to_end(key)
                return cached
        oid, code, disp = resolve_machine_entity_id(self.game, self.catalog, entity_id)
        asm = assemble_machine(
            self.game,
            oid,
            codename=code,
            display_name=disp,
            variant_index=variant_index,
            lod_index=0,
            include_broken_states=True,
        )
        with self._lock:
            self._asm_cache[key] = asm
            if len(self._asm_cache) > 8:
                self._asm_cache.popitem(last=False)
        return asm

    def get_material_report(self, material_id: str, external_decompile: bool = False) -> dict[str, Any]:
        key = (material_id, external_decompile)
        with self._lock:
            cached = self._mat_cache.get(key)
            if cached is not None:
                self._mat_cache.move_to_end(key)
                return cached
        report = analyze_material_deep(
            self.game,
            material_id,
            dxc=self.dxc,
            hash_dict=self.hash_dict,
            include_raw_dxil=True,
            try_external_decompiler=external_decompile,
        )
        with self._lock:
            self._mat_cache[key] = report
            if len(self._mat_cache) > 64:
                self._mat_cache.popitem(last=False)
        return report

    def close(self) -> None:
        self.catalog.close()
        self.game.close()


def create_app(config: ToolkitConfig | None = None) -> FastAPI:
    cfg = config or ToolkitConfig.load()
    state = ToolkitServerState(cfg)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        state.close()

    app = FastAPI(title="HFWToolkit — Material & Shader Deep-Dive Studio", version="0.1.0", lifespan=lifespan)
    app.state.toolkit = state

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/stats")
    def api_stats():
        from hfw_toolkit.updater import get_local_version_info

        ds, sc = find_external_decompilers(state.config)
        diag = state.config.diagnose()
        py_info = state.get_python_info()
        loc_ver = get_local_version_info(state.config.toolkit_dir)
        return {
            "catalog": state.catalog.stats(),
            "game_dir": str(state.config.game_dir),
            "game_version": diag["game_version"],
            "rtti_source": diag["rtti_source"],
            "rtti_type_count": diag["rtti_type_count"],
            "core_archive_count": diag["core_archive_count"],
            "blender_exe": str(state.config.blender_exe),
            "blender_available": state.config.blender_exe.exists(),
            "dxcompiler_available": state.config.dxcompiler_dll.exists(),
            "external_decompiler_available": bool(ds and sc),
            "python_runtime": py_info,
            "version_info": loc_ver,
            "auto_update": bool(state.config.auto_update),
            "update_branch": state.config.update_branch or "main",
            "check_updates_on_startup": bool(state.config.check_updates_on_startup),
            "diagnose": diag,
        }

    @app.get("/api/doctor")
    def api_doctor():
        from hfw_toolkit.updater import get_local_version_info

        diag = state.config.diagnose()
        diag["catalog"] = state.catalog.stats() if state.catalog.is_built() else {}
        diag["python_runtime"] = state.get_python_info()
        diag["version_info"] = get_local_version_info(state.config.toolkit_dir)
        return diag

    @app.get("/api/updater/status")
    def api_updater_status(force: bool = False):
        from hfw_toolkit.updater import check_for_updates

        now = time.time()
        cached = state._update_check_cache
        if not force and cached is not None and (now - cached[0]) < 180.0:
            res = dict(cached[1])
        else:
            res = check_for_updates(state.config)
            state._update_check_cache = (now, dict(res))
        res["auto_update"] = bool(state.config.auto_update)
        res["check_updates_on_startup"] = bool(state.config.check_updates_on_startup)
        res["target_branch"] = state.config.update_branch or "main"
        res["python_runtime"] = state.get_python_info(refresh=force)
        return res

    @app.post("/api/updater/config")
    def api_updater_config(req: UpdaterConfigRequest):
        if req.auto_update is not None:
            state.config.auto_update = bool(req.auto_update)
        if req.update_branch is not None and req.update_branch.strip():
            state.config.update_branch = req.update_branch.strip()
        if req.check_updates_on_startup is not None:
            state.config.check_updates_on_startup = bool(req.check_updates_on_startup)
        state.config.save()
        state._update_check_cache = None
        return {
            "status": "ok",
            "auto_update": bool(state.config.auto_update),
            "update_branch": state.config.update_branch,
            "check_updates_on_startup": bool(state.config.check_updates_on_startup),
        }

    @app.post("/api/updater/apply")
    def api_updater_apply(req: UpdaterApplyRequest):
        from hfw_toolkit.updater import apply_update

        res = apply_update(
            state.config,
            branch=req.branch,
            force=req.force,
            reinstall_deps=req.reinstall_deps,
        )
        state._update_check_cache = None
        return res

    @app.post("/api/config")
    def api_update_config(req: UpdateConfigRequest):
        if req.auto_detect:
            state.config.game_dir = auto_detect_game_dir()
            state.config.blender_exe = auto_detect_blender_exe()
        else:
            if req.game_dir and req.game_dir.strip():
                state.config.game_dir = Path(req.game_dir.strip())
            if req.blender_exe and req.blender_exe.strip():
                state.config.blender_exe = Path(req.blender_exe.strip())
        if req.auto_update is not None:
            state.config.auto_update = bool(req.auto_update)
        if req.update_branch is not None and req.update_branch.strip():
            state.config.update_branch = req.update_branch.strip()
        if req.check_updates_on_startup is not None:
            state.config.check_updates_on_startup = bool(req.check_updates_on_startup)
        state.config.rtti_json = state.config.game_dir / "dump" / "hfw.json"
        state.config.save()
        with state._lock:
            state._asm_cache.clear()
            state._mat_cache.clear()
            state._tex_png_cache.clear()
            try:
                state.game.close()
            except Exception:
                pass
            if state.config.streaming_graph_path.is_file():
                state.game = HFWGame(state.config)
                state.dxc = DXCDisassembler(state.config)
                if not state.catalog.is_built():
                    state.catalog.build(state.game)
        diag = state.config.diagnose()
        diag["catalog"] = state.catalog.stats() if state.catalog.is_built() else {}
        return diag

    @app.post("/api/install-addon")
    def api_install_addon():
        try:
            paths = install_blender_addon_files(state.config)
            return {"status": "ok", "installed_paths": [str(p) for p in paths]}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/open-blend")
    def api_open_blend(req: OpenBlendRequest):
        if req.open_folder or not req.blend_path:
            state.config.exports_dir.mkdir(parents=True, exist_ok=True)
            if hasattr(os, "startfile"):
                os.startfile(str(state.config.exports_dir))
            return {"status": "opened_folder", "path": str(state.config.exports_dir)}
        p = Path(req.blend_path)
        if not p.is_file():
            raise HTTPException(status_code=404, detail=f".blend file not found: {p}")
        if not state.config.blender_exe.is_file():
            raise HTTPException(status_code=400, detail=f"Blender executable not found: {state.config.blender_exe}")
        subprocess.Popen([str(state.config.blender_exe), str(p)])
        return {"status": "launched_blender", "blend_path": str(p)}

    @app.get("/api/machines")
    def api_machines(q: str = "", category: str | None = None):
        if category == "all":
            rows = state.catalog.search_entities(q, category=None, limit=120)
        else:
            rows = state.catalog.list_machines(q)
        return {"items": rows}

    @app.get("/api/machine/{entity_id}")
    def api_machine_detail(entity_id: str, variant: int = 0):
        try:
            asm = state.get_assembly(entity_id, variant_index=variant)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        # Group parts and summarize materials cleanly for the UI
        parts_out = []
        mat_part_counts: dict[str, list[str]] = {}
        for p in asm.parts:
            p_mats = sorted({pr.material_id for pr in p.mesh.primitives if pr.material_id})
            for m in p_mats:
                mat_part_counts.setdefault(m, []).append(p.name)
            parts_out.append(
                {
                    "part_id": p.part_id,
                    "name": p.name,
                    "collection": p.collection,
                    "is_skinned": p.is_skinned,
                    "parent_joint": p.parent_joint_name,
                    "helper_name": p.helper_name,
                    "state_index": p.state_index,
                    "enabled_by_default": p.enabled_by_default,
                    "vertex_count": sum(len(pr.positions) for pr in p.mesh.primitives),
                    "triangle_count": sum(len(pr.indices) // 3 for pr in p.mesh.primitives),
                    "material_ids": p_mats,
                }
            )

        materials_out = []
        for mid, mat in asm.materials.items():
            pt = mat.primary_technique
            materials_out.append(
                {
                    "material_id": mid,
                    "name": mat.name,
                    "tags": mat.material_tags,
                    "passes": pt.available_passes if pt else [],
                    "texture_count": len(pt.texture_bindings) if pt else 0,
                    "variable_count": len(pt.variable_bindings) if pt else 0,
                    "shader_id": pt.shader_id if pt else None,
                    "used_by_parts": mat_part_counts.get(mid, [])[:12],
                    "used_by_count": len(mat_part_counts.get(mid, [])),
                }
            )
        materials_out.sort(key=lambda m: (-m["texture_count"], m["name"]))

        return {
            "entity_id": asm.entity_id,
            "codename": asm.codename,
            "display_name": asm.display_name,
            "variant_label": asm.variant_label,
            "variants": [
                {
                    "variant_index": v.variant_index,
                    "label": v.label,
                    "group_id": v.group_id,
                    "part_count": v.part_count,
                }
                for v in asm.available_variants
            ],
            "skeleton": {
                "joint_count": len(asm.skeleton.joints),
                "helper_count": len(asm.skeleton.helpers),
            },
            "parts": parts_out,
            "materials": materials_out,
        }

    @app.get("/api/material/{material_id}")
    def api_material_detail(material_id: str, external_decompile: bool = False):
        try:
            report = state.get_material_report(material_id, external_decompile=external_decompile)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=f"Failed to analyze material {material_id}: {exc}") from exc

        # Attach cross-reference counts for each bound TextureSet!
        xrefs: dict[str, list[dict[str, Any]]] = {}
        for ts_id in report.get("texture_set_ids", ()):
            try:
                oid = ObjectId.parse(ts_id)
                xrefs[ts_id] = state.catalog.texture_set_users(oid.group_id, oid.object_index)
            except Exception:
                xrefs[ts_id] = []
        report["texture_set_xrefs"] = xrefs
        return report

    @app.get("/api/diff")
    def api_material_diff(a: str = Query(...), b: str = Query(...)):
        """Side-by-side diff comparing two RenderEffectResources (e.g. Standard vs Apex plate!)."""
        try:
            ra = state.get_material_report(a, external_decompile=False)
            rb = state.get_material_report(b, external_decompile=False)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        tex_a = {t["binding_hash"]: t for t in ra["primary_textures"]}
        tex_b = {t["binding_hash"]: t for t in rb["primary_textures"]}
        all_th = sorted(set(tex_a) | set(tex_b))
        tex_diffs = []
        for h in all_th:
            ta, tb = tex_a.get(h), tex_b.get(h)
            status = "same"
            if ta is None:
                status = "added_in_b"
            elif tb is None:
                status = "removed_in_b"
            elif ta["resource_id"] != tb["resource_id"]:
                status = "changed"
            tex_diffs.append({"binding_hash": h, "status": status, "a": ta, "b": tb})

        var_a = {v["binding_hash"]: v for v in ra["primary_variables"]}
        var_b = {v["binding_hash"]: v for v in rb["primary_variables"]}
        all_vh = sorted(set(var_a) | set(var_b))
        var_diffs = []
        for h in all_vh:
            va, vb = var_a.get(h), var_b.get(h)
            status = "same"
            if va is None:
                status = "added_in_b"
            elif vb is None:
                status = "removed_in_b"
            elif va["float4"] != vb["float4"]:
                status = "changed"
            var_diffs.append({"binding_hash": h, "status": status, "a": va, "b": vb})

        return {
            "a": {"material_id": ra["material_id"], "name": ra["name"], "tags": ra["tags"], "gpu": ra["explainer"]["gpu_profile"]},
            "b": {"material_id": rb["material_id"], "name": rb["name"], "tags": rb["tags"], "gpu": rb["explainer"]["gpu_profile"]},
            "textures": tex_diffs,
            "variables": var_diffs,
        }

    @app.get("/api/texturesets")
    def api_search_texturesets(q: str = "", codename: str | None = None, limit: int = 80):
        return {"items": state.catalog.search_texture_sets(q, codename=codename, limit=limit)}

    @app.get("/api/textureset/{ts_id}")
    def api_textureset_detail(ts_id: str):
        try:
            oid = ObjectId.parse(ts_id)
            ts_obj = state.game.read_object(oid, read_subgroups=True)
            uts = unpack_texture_set(state.game, ts_obj, max_dimension=512)
            users = state.catalog.texture_set_users(oid.group_id, oid.object_index)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "object_id": uts.object_id,
            "name": uts.name,
            "source_paths": uts.source_paths,
            "packed_entries": uts.packed_entries,
            "semantic_maps": list(uts.semantic_maps.keys()),
            "used_by": users,
        }

    @app.get("/api/textureset/{ts_id}/map/{map_name}")
    def api_textureset_map_png(ts_id: str, map_name: str, max_dim: int = 512):
        ckey = (f"ts:{ts_id}", map_name, max_dim)
        with state._lock:
            cached = state._tex_png_cache.get(ckey)
            if cached is not None:
                state._tex_png_cache.move_to_end(ckey)
                return Response(content=cached, media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})
        try:
            ts_obj = state.game.read_object(ObjectId.parse(ts_id), read_subgroups=True)
            uts = unpack_texture_set(state.game, ts_obj, max_dimension=max_dim)
            arr = uts.semantic_maps.get(map_name)
            if arr is None:
                raise KeyError(f"Map {map_name} not in TextureSet {ts_id}")
            from PIL import Image

            mode = "L" if arr.ndim == 2 else ("RGB" if arr.shape[2] == 3 else "RGBA")
            buf = io.BytesIO()
            Image.fromarray(arr, mode=mode).save(buf, format="PNG")
            png = buf.getvalue()
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        with state._lock:
            state._tex_png_cache[ckey] = png
            if len(state._tex_png_cache) > 256:
                state._tex_png_cache.popitem(last=False)
        return Response(content=png, media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})

    @app.get("/api/texture/{tex_id}/image")
    def api_texture_png(tex_id: str, mode: str = "rgba", max_dim: int = 512):
        ckey = (f"tex:{tex_id}", mode.lower(), max_dim)
        with state._lock:
            cached = state._tex_png_cache.get(ckey)
            if cached is not None:
                state._tex_png_cache.move_to_end(ckey)
                return Response(content=cached, media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})
        try:
            obj = state.game.read_object(ObjectId.parse(tex_id), read_subgroups=True)
            # If caller passed a TextureSet ID, prefer its Color entry before falling back to entries[0]!
            if obj["_type"] == "TextureSet":
                entries = obj.get("Entries", ())
                if not entries:
                    raise ValueError("Empty TextureSet")
                chosen = entries[0]
                for ent in entries:
                    pk_str = str(ent.get("PackingInfo", ""))
                    if "Color" in pk_str:
                        chosen = ent
                        break
                obj = state.game.resolve_ref(chosen["Texture"])
            dec = decode_texture(state.game, obj, max_dimension=max_dim)
            if dec is None:
                raise ValueError("Could not decode texture")
            img = render_channel_image(dec, mode=mode)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            png = buf.getvalue()
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        with state._lock:
            state._tex_png_cache[ckey] = png
            if len(state._tex_png_cache) > 256:
                state._tex_png_cache.popitem(last=False)
        return Response(content=png, media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})

    @app.post("/api/export/blend")
    def api_export_blend(req: ExportBlendRequest):
        job_id = f"job_{int(time.time() * 1000)}"
        state.export_jobs[job_id] = {"status": "running", "message": "Assembling machine...", "output_blend": None}

        def _worker() -> None:
            try:
                oid, code, disp = resolve_machine_entity_id(state.game, state.catalog, req.machine)
                asm = assemble_machine(
                    state.game,
                    oid,
                    codename=code,
                    display_name=disp,
                    variant_index=req.variant_index,
                    lod_index=req.lod_index,
                    include_broken_states=req.include_broken_states,
                )
                tex_lbl = "Unpacking textures" if req.unpack_textures else "Exporting native packed game textures"
                state.export_jobs[job_id]["message"] = f"{tex_lbl} & writing bundle for {disp}..."
                import re

                safe_disp = re.sub(r"[^A-Za-z0-9_\-.]+", "_", disp).strip("_") or "Machine"
                bdir = state.config.exports_dir / f"{safe_disp}_v{req.variant_index}_bundle"
                out_blend = state.config.exports_dir / f"{safe_disp}_v{req.variant_index}.blend"
                fbx_out = (bdir / f"{safe_disp}.fbx") if req.export_fbx else None
                export_assembly_bundle(
                    state.game,
                    asm,
                    bdir,
                    max_texture_size=req.max_texture_size,
                    unpack_textures=req.unpack_textures,
                    animations=req.animations,
                )
                state.export_jobs[job_id]["message"] = f"Building {out_blend.name}{' + UE5 FBX' if req.export_fbx else ''} in Blender..."
                build_blend_headless(
                    bdir,
                    out_blend,
                    config=state.config,
                    pack_textures=req.pack_textures,
                    enable_detail_nodes=req.enable_detail_nodes,
                    export_fbx=fbx_out,
                    build_control_rig=req.build_control_rig,
                )
                state.export_jobs[job_id] = {
                    "status": "done",
                    "message": (
                        f"Exported UE5 Rig + Animated Shaders -> {bdir / 'ue5_import_machine.py'}"
                        if req.export_fbx
                        else f"Exported {out_blend.name} ({len(asm.parts)} parts, {len(asm.materials)} materials)"
                    ),
                    "output_blend": str(out_blend),
                    "bundle_dir": str(bdir),
                    "ue5_script": str(bdir / "ue5_import_machine.py"),
                    "fbx_path": str(fbx_out) if fbx_out else None,
                }
            except Exception as exc:
                state.export_jobs[job_id] = {"status": "error", "message": str(exc), "output_blend": None}

        threading.Thread(target=_worker, daemon=True).start()
        return {"job_id": job_id}

    @app.get("/api/export/status/{job_id}")
    def api_export_status(job_id: str):
        job = state.export_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Unknown export job")
        return job

    # ------------------------------------------------------------------
    # Audio, Music, SFX & Voicelines Studio Endpoints
    # ------------------------------------------------------------------

    _AUDIO_MIME = {
        "wav": "audio/wav",
        "flac": "audio/flac",
        "mp3": "audio/mpeg",
        "ogg": "audio/ogg",
        "at9": "application/octet-stream",
    }

    def _ensure_audio_tier1() -> None:
        with state._db_lock:
            if not state.catalog.is_audio_built():
                state.catalog.build_audio_index(state.game)

    def _row_to_item(r: dict[str, Any]) -> AudioItemInfo:
        return AudioItemInfo(
            object_id=r["object_id"],
            kind=r["kind"],
            name=r["name"],
            dev_folder=r["dev_folder"],
            filename_stem=r["filename_stem"],
            group_id=int(r["group_id"]),
            sample_rate=int(r["sample_rate"]),
            channels=int(r["channels"]),
            sample_count=int(r["sample_count"]),
            duration_sec=float(r["duration_sec"]),
            encoding=int(r["encoding"]),
            encoding_name=str(r["encoding_name"]),
            data_size=int(r["data_size"]),
            is_streaming=bool(r["is_streaming"]),
            sound_graph_name=str(r.get("sound_graph_name", "")),
            owner_name=str(r.get("owner_name", "")),
            speaker=str(r.get("speaker", "")),
            subtitle=str(r.get("subtitle", "")),
            languages=tuple(r.get("languages") or ()),
        )

    @app.get("/api/audio/stats")
    def api_audio_stats():
        _ensure_audio_tier1()
        with state._db_lock:
            st = state.catalog.audio_stats()
        langs = detect_installed_languages(state.game)
        ost = detect_digital_extras_ost(state.game)
        return {
            "stats": st,
            "pyav_available": has_pyav(),
            "installed_languages": [{"code": c, "label": SPOKEN_LANGUAGE_LABELS.get(c, c)} for c in langs],
            "ost_count": len(ost),
            "supported_formats": list(SUPPORTED_OUTPUT_FORMATS),
            "default_out_dir": str(state.config.exports_dir / "Audio"),
        }

    @app.get("/api/audio/categories")
    def api_audio_categories(kind: str = "music"):
        _ensure_audio_tier1()
        k = "voiceline" if kind == "voicelines" else kind
        with state._db_lock:
            return {"items": state.catalog.list_audio_categories(kind=k)}

    @app.get("/api/audio/music")
    def api_audio_music(q: str = "", category: str = "", project: str = "", limit: int = 250):
        _ensure_audio_tier1()
        with state._db_lock:
            items = state.catalog.list_music_cues(query=q, category=category, project=project, limit=limit)
        return {"items": items}

    @app.get("/api/audio/groups")
    def api_audio_groups(kind: str = "sfx", q: str = "", category: str = "", limit: int = 250):
        _ensure_audio_tier1()
        k = "voiceline" if kind == "voicelines" else kind
        with state._db_lock:
            items = state.catalog.list_audio_groups(kind=k, query=q, category=category, limit=limit)
        return {"items": items}

    @app.get("/api/audio/group/{group_id}")
    def api_audio_group_items(
        group_id: int,
        kind: str = "sfx",
        q: str = "",
        speaker: str = "",
        limit: int = 1000,
    ):
        _ensure_audio_tier1()
        k = "voiceline" if kind == "voicelines" else kind
        try:
            with state._db_lock:
                items = state.catalog.ensure_audio_group_indexed(state.game, group_id, k)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if speaker:
            sl = speaker.lower()
            items = [it for it in items if sl in it["speaker"].lower()]
        if q:
            ql = q.lower()
            items = [
                it
                for it in items
                if ql in it["name"].lower()
                or ql in it["dev_folder"].lower()
                or ql in it["sound_graph_name"].lower()
                or ql in it["subtitle"].lower()
                or ql in it["speaker"].lower()
                or ql in it["object_id"]
            ]
        return {"group_id": group_id, "kind": k, "total": len(items), "items": items[:limit]}

    @app.get("/api/audio/search")
    def api_audio_search(kind: str = "", q: str = "", speaker: str = "", limit: int = 200):
        _ensure_audio_tier1()
        k = "voiceline" if kind == "voicelines" else kind
        with state._db_lock:
            items = state.catalog.search_audio_assets(kind=k, query=q, speaker=speaker, limit=limit)
        return {"items": items}

    @app.get("/api/audio/speakers")
    def api_audio_speakers(q: str = "", limit: int = 200):
        _ensure_audio_tier1()
        with state._db_lock:
            items = state.catalog.list_speakers(query=q, limit=limit)
        return {"items": items}

    @app.get("/api/audio/stream/object/{object_id}")
    def api_audio_stream_object(
        object_id: str,
        lang: str = "en",
        fmt: str = "wav",
        download: bool = False,
        filename: str = "",
    ):
        f = fmt.lower().lstrip(".")
        if f not in SUPPORTED_OUTPUT_FORMATS:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {fmt}")
        ckey = (f"obj:{object_id}", lang, f)
        with state._lock:
            cached = state._audio_bytes_cache.get(ckey)
            if cached is not None:
                state._audio_bytes_cache.move_to_end(ckey)
        if cached is None:
            try:
                dec = decode_audio_object(state.game, object_id, lang=lang)
                cached = encode_audio_bytes(dec, fmt=f)
            except Exception as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            if len(cached) <= 40 * 1024 * 1024:
                with state._lock:
                    state._audio_bytes_cache[ckey] = cached
                    while len(state._audio_bytes_cache) > 24:
                        state._audio_bytes_cache.popitem(last=False)

        headers = {"Cache-Control": "public, max-age=3600"}
        if download:
            dl_name = filename or f"hfw_{object_id.replace(':', '_')}.{f}"
            if not dl_name.lower().endswith(f".{f}"):
                dl_name += f".{f}"
            headers["Content-Disposition"] = f'attachment; filename="{dl_name}"'
        return Response(content=cached, media_type=_AUDIO_MIME.get(f, "application/octet-stream"), headers=headers)

    @app.get("/api/audio/stream/cue/{cue_id}")
    def api_audio_stream_cue(cue_id: str, fmt: str = "wav", download: bool = False):
        _ensure_audio_tier1()
        f = fmt.lower().lstrip(".")
        if f not in SUPPORTED_OUTPUT_FORMATS:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {fmt}")
        with state._db_lock:
            cue = state.catalog.get_music_cue_info(cue_id)
        if cue is None:
            raise HTTPException(status_code=404, detail=f"MusicCue {cue_id} not found")

        ckey = (f"cue:{cue_id}", "mix", f)
        with state._lock:
            cached = state._audio_bytes_cache.get(ckey)
            if cached is not None:
                state._audio_bytes_cache.move_to_end(ckey)
        if cached is None:
            try:
                mixed = render_music_cue_mix(state.game, cue)
                cached = encode_audio_bytes(mixed, fmt=f)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            if len(cached) <= 60 * 1024 * 1024:
                with state._lock:
                    state._audio_bytes_cache[ckey] = cached
                    while len(state._audio_bytes_cache) > 16:
                        state._audio_bytes_cache.popitem(last=False)

        headers = {"Cache-Control": "public, max-age=3600"}
        if download:
            import re as _re

            safe = _re.sub(r"[^A-Za-z0-9_\-.]+", "_", cue.cue_name).strip("_") or f"cue_{cue_id.replace(':', '_')}"
            headers["Content-Disposition"] = f'attachment; filename="{safe}.{f}"'
        return Response(content=cached, media_type=_AUDIO_MIME.get(f, "application/octet-stream"), headers=headers)

    @app.post("/api/audio/extract")
    def api_audio_extract(req: AudioExtractRequest):
        job_id = f"audio_{int(time.time() * 1000)}"
        state.audio_jobs[job_id] = {
            "status": "running",
            "current": 0,
            "total": 0,
            "message": "Preparing audio extraction...",
            "result": None,
        }

        def _worker() -> None:
            try:
                _ensure_audio_tier1()
                out_dir = Path(req.out_dir.strip()) if (req.out_dir and req.out_dir.strip()) else (state.config.exports_dir / "Audio")
                fmt = req.format.lower().lstrip(".")
                langs = [x.strip() for x in (req.lang or "en").split(",") if x.strip()]

                def _cb(cur: int, tot: int, msg: str) -> None:
                    state.audio_jobs[job_id]["current"] = cur
                    state.audio_jobs[job_id]["total"] = tot
                    state.audio_jobs[job_id]["message"] = msg

                # Explicit single MusicCue
                if req.cue_id:
                    with state._db_lock:
                        cue = state.catalog.get_music_cue_info(req.cue_id)
                    if cue is None:
                        raise ValueError(f"MusicCue {req.cue_id} not found")
                    res = extract_music_cues_to_dir(
                        state.game,
                        [cue],
                        out_dir,
                        fmt=fmt,
                        preserve_structure=req.preserve_structure,
                        mix_cues=req.mix_cues,
                        export_stems=req.export_stems,
                        include_ost=req.include_ost,
                        progress_cb=_cb,
                    )
                    state.audio_jobs[job_id] = {
                        "status": "done",
                        "current": 1,
                        "total": 1,
                        "message": f"Extracted {res['exported_count']} file(s) ({res['total_bytes']/1e6:.2f} MB) -> {res['out_dir']}",
                        "result": res,
                    }
                    return

                # Explicit list of object_ids
                if req.object_ids:
                    items: list[AudioItemInfo] = []
                    with state._db_lock:
                        for oid in req.object_ids:
                            rows = state.catalog.search_audio_assets(query=oid, limit=5)
                            match = next((r for r in rows if r["object_id"] == oid), None)
                            if match:
                                items.append(_row_to_item(match))
                    res = extract_audio_items_to_dir(
                        state.game,
                        items,
                        out_dir,
                        fmt=fmt,
                        preserve_structure=req.preserve_structure,
                        languages=langs,
                        group_by=req.group_by,
                        write_transcript=req.write_transcript,
                        progress_cb=_cb,
                    )
                    state.audio_jobs[job_id] = {
                        "status": "done",
                        "current": len(items),
                        "total": len(items),
                        "message": f"Extracted {res['exported_count']} clip(s) ({res['total_bytes']/1e6:.2f} MB) -> {res['out_dir']}",
                        "result": res,
                    }
                    return

                combined_count = 0
                combined_bytes = 0
                last_res: dict[str, Any] = {"out_dir": str(out_dir)}

                if req.category in ("music", "all"):
                    with state._db_lock:
                        cue_rows = state.catalog.list_music_cues(
                            query=req.query, category=req.subcat, limit=req.limit or 10000
                        )
                        cues = [ci for r in cue_rows if (ci := state.catalog.get_music_cue_info(r["cue_id"])) is not None]
                    res = extract_music_cues_to_dir(
                        state.game,
                        cues,
                        out_dir,
                        fmt=fmt,
                        preserve_structure=req.preserve_structure,
                        mix_cues=req.mix_cues,
                        export_stems=req.export_stems,
                        include_ost=req.include_ost,
                        limit=req.limit,
                        progress_cb=_cb,
                    )
                    combined_count += res["exported_count"]
                    combined_bytes += res["total_bytes"]
                    last_res = res

                for req_cat, db_kind in (("sfx", "sfx"), ("voicelines", "voiceline")):
                    if req.category not in (req_cat, "all"):
                        continue
                    items_rows: list[dict[str, Any]] = []
                    if req.group_id is not None:
                        with state._db_lock:
                            items_rows = state.catalog.ensure_audio_group_indexed(state.game, req.group_id, db_kind)
                    else:
                        with state._db_lock:
                            groups = state.catalog.list_audio_groups(
                                kind=db_kind, query=req.query, category=req.subcat, limit=2000
                            )
                        if not groups and (req.query or req.speaker):
                            with state._db_lock:
                                items_rows = state.catalog.search_audio_assets(
                                    kind=db_kind, query=req.query, speaker=req.speaker, limit=req.limit or 5000
                                )
                        else:
                            for gi, gr in enumerate(groups, 1):
                                _cb(gi, len(groups), f"Indexing {req_cat} folder {gr['dev_folder']} ({gi}/{len(groups)})...")
                                with state._db_lock:
                                    items_rows.extend(
                                        state.catalog.ensure_audio_group_indexed(state.game, int(gr["group_id"]), db_kind)
                                    )
                                if req.limit and not req.speaker and len(items_rows) >= req.limit:
                                    break
                    if req.speaker:
                        sl = req.speaker.lower()
                        items_rows = [r for r in items_rows if sl in r["speaker"].lower()]
                    if req.query and req.group_id is not None:
                        ql = req.query.lower()
                        items_rows = [
                            r
                            for r in items_rows
                            if ql in r["name"].lower()
                            or ql in r["dev_folder"].lower()
                            or ql in r["sound_graph_name"].lower()
                            or ql in r["subtitle"].lower()
                            or ql in r["speaker"].lower()
                        ]
                    items = [_row_to_item(r) for r in items_rows]
                    if items:
                        res = extract_audio_items_to_dir(
                            state.game,
                            items,
                            out_dir,
                            fmt=fmt,
                            preserve_structure=req.preserve_structure,
                            languages=langs,
                            group_by=req.group_by,
                            write_transcript=req.write_transcript,
                            limit=req.limit,
                            progress_cb=_cb,
                        )
                        combined_count += res["exported_count"]
                        combined_bytes += res["total_bytes"]
                        last_res = res

                state.audio_jobs[job_id] = {
                    "status": "done",
                    "current": combined_count,
                    "total": combined_count,
                    "message": f"Extracted {combined_count:,} audio file(s) ({combined_bytes/1e6:.2f} MB) -> {out_dir}",
                    "result": last_res,
                }
            except Exception as exc:
                state.audio_jobs[job_id] = {
                    "status": "error",
                    "current": 0,
                    "total": 0,
                    "message": str(exc),
                    "result": None,
                }

        threading.Thread(target=_worker, daemon=True).start()
        return {"job_id": job_id}

    @app.post("/api/audio/index")
    def api_audio_index(req: AudioIndexRequest):
        job_id = f"aidx_{int(time.time() * 1000)}"
        state.audio_jobs[job_id] = {
            "status": "running",
            "current": 0,
            "total": 0,
            "message": "Starting audio indexing...",
            "result": None,
        }

        def _worker() -> None:
            try:
                if req.rebuild or not state.catalog.is_audio_built():
                    state.audio_jobs[job_id]["message"] = "Building Tier-1 audio index (Music + Voices + Directories)..."
                    with state._db_lock:
                        st = state.catalog.build_audio_index(
                            state.game,
                            progress_cb=lambda m: state.audio_jobs[job_id].update({"message": m}),
                        )
                if req.deep:
                    def _cb(cur: int, tot: int, msg: str) -> None:
                        state.audio_jobs[job_id]["current"] = cur
                        state.audio_jobs[job_id]["total"] = tot
                        state.audio_jobs[job_id]["message"] = msg

                    with state._db_lock:
                        st = state.catalog.build_deep_audio_index(state.game, progress_cb=_cb)
                else:
                    with state._db_lock:
                        st = state.catalog.audio_stats()
                state.audio_jobs[job_id] = {
                    "status": "done",
                    "current": 1,
                    "total": 1,
                    "message": (
                        f"Audio index ready: {st['music_cues']:,} music cues, "
                        f"{st['sfx_assets_indexed']:,} deep-indexed SFX, "
                        f"{st['voiceline_assets_indexed']:,} deep-indexed voicelines."
                    ),
                    "result": st,
                }
            except Exception as exc:
                state.audio_jobs[job_id] = {
                    "status": "error",
                    "current": 0,
                    "total": 0,
                    "message": str(exc),
                    "result": None,
                }

        threading.Thread(target=_worker, daemon=True).start()
        return {"job_id": job_id}

    @app.get("/api/audio/job/{job_id}")
    def api_audio_job_status(job_id: str):
        job = state.audio_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Unknown audio job")
        return job

    @app.post("/api/audio/open-folder")
    def api_audio_open_folder(req: OpenAudioFolderRequest):
        target = Path(req.path) if req.path else (state.config.exports_dir / "Audio")
        target.mkdir(parents=True, exist_ok=True)
        if hasattr(os, "startfile"):
            os.startfile(str(target))
        return {"status": "opened_folder", "path": str(target)}

    # ------------------------------------------------------------------
    # Characters, Outfits, Prompt Resolver & Guerrilla Dev-Directory Asset Catalog
    # ------------------------------------------------------------------

    def _ensure_chars_assets_indexed() -> None:
        with state._db_lock:
            if not state.catalog.is_characters_assets_built():
                state.catalog.index_characters_and_assets(state.game)

    @app.post("/api/index/characters-assets")
    def api_index_chars_assets():
        with state._db_lock:
            res = state.catalog.index_characters_and_assets(state.game)
        return {"status": "ok", "counts": res}

    @app.get("/api/characters")
    def api_characters(
        q: str = "",
        kind: str = "",
        tribe: str = "",
        gender: str = "",
        limit: int = 200,
    ):
        _ensure_chars_assets_indexed()
        with state._db_lock:
            items = state.catalog.list_characters(query=q, kind=kind, tribe=tribe, gender=gender, limit=limit)
            st = state.catalog.stats()
        return {
            "items": items,
            "counts": {
                "characters": st.get("characters", 0),
                "outfits": st.get("character_outfits", 0),
                "heads": st.get("character_heads", 0),
            },
        }

    @app.get("/api/characters/resolve")
    def api_characters_resolve(prompt: str = Query(..., min_length=1)):
        _ensure_chars_assets_indexed()
        with state._db_lock:
            res = resolve_character_prompt(state.catalog, prompt)
        return res

    @app.get("/api/characters/{char_id}/outfits")
    def api_character_outfits(char_id: str, q: str = "", limit: int = 120):
        _ensure_chars_assets_indexed()
        with state._db_lock:
            items = state.catalog.list_character_outfits(char_id=char_id, query=q, limit=limit)
        return {"char_id": char_id, "items": items}

    @app.get("/api/characters/heads")
    def api_character_heads(q: str = "", gender: str = "", limit: int = 250):
        _ensure_chars_assets_indexed()
        with state._db_lock:
            items = state.catalog.list_character_heads(query=q, gender=gender, limit=limit)
        return {"items": items}

    @app.post("/api/characters/export")
    def api_export_character(req: ExportCharacterRequest):
        import re as _re

        job_id = f"char_{int(time.time() * 1000)}"
        state.export_jobs[job_id] = {"status": "running", "message": "Assembling character segments & unified rig...", "output_blend": None}

        def _worker() -> None:
            try:
                _ensure_chars_assets_indexed()
                use_prompt = bool(req.prompt and req.prompt.strip() and not req.char_id)
                asm = assemble_character(
                    state.game,
                    catalog=state.catalog,
                    char_id=req.char_id or None,
                    outfit_id=req.outfit_id or None,
                    prompt=req.prompt.strip() if use_prompt else None,
                    headgear=None if use_prompt else req.headgear,
                    include_hair=None if use_prompt else req.include_hair,
                    hair_lod=req.hair_lod,
                    facial_id=req.facial_override or None,
                    lod_index=req.lod_index,
                )
                safe_n = _re.sub(r"[^A-Za-z0-9_\-.]+", "_", f"{asm.display_name}_{asm.variant_label}").strip("_")[:64] or "Character"
                base_dir = state.config.exports_dir / "Characters"
                base_dir.mkdir(parents=True, exist_ok=True)
                out_blend = base_dir / f"{safe_n}.blend"
                bdir = base_dir / (f"{safe_n}_UE5" if req.export_fbx else f"{safe_n}_bundle")
                fbx_out = (bdir / f"{safe_n}.fbx") if req.export_fbx else None

                state.export_jobs[job_id]["message"] = (
                    f"Exporting {asm.display_name} ({asm.variant_label}) textures & {len(asm.materials)} character materials..."
                )
                export_assembly_bundle(
                    state.game,
                    asm,
                    bdir,
                    max_texture_size=req.max_texture_size,
                    unpack_textures=req.unpack_textures,
                    asset_kind="character",
                    animations=req.animations,
                )
                state.export_jobs[job_id]["message"] = f"Building unified rig {out_blend.name}{' + UE5 FBX' if req.export_fbx else ''} in Blender..."
                build_blend_headless(
                    bdir,
                    out_blend,
                    config=state.config,
                    pack_textures=req.pack_textures,
                    enable_detail_nodes=False,
                    export_fbx=fbx_out,
                    build_control_rig=req.build_control_rig,
                )
                state.export_jobs[job_id] = {
                    "status": "done",
                    "message": (
                        f"Exported UE5 Rigged Character -> {bdir / 'ue5_import_character.py'}"
                        if req.export_fbx
                        else f"Exported {out_blend.name} ({len(asm.skeleton.joints)} unified joints, {len(asm.parts)} segments, {len(asm.materials)} materials)"
                    ),
                    "output_blend": str(out_blend),
                    "bundle_dir": str(bdir),
                    "ue5_script": str(bdir / "ue5_import_character.py"),
                    "fbx_path": str(fbx_out) if fbx_out else None,
                }
            except Exception as exc:
                state.export_jobs[job_id] = {"status": "error", "message": str(exc), "output_blend": None}

        threading.Thread(target=_worker, daemon=True).start()
        return {"job_id": job_id}

    def _summarize_assembly_for_ui(asm: MachineAssembly) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        parts_out = []
        mat_part_counts: dict[str, list[str]] = {}
        for p in asm.parts:
            p_mats = sorted({pr.material_id for pr in p.mesh.primitives if pr.material_id})
            for m in p_mats:
                mat_part_counts.setdefault(m, []).append(p.name)
            parts_out.append(
                {
                    "part_id": p.part_id,
                    "name": p.name,
                    "collection": p.collection,
                    "is_skinned": p.is_skinned,
                    "parent_joint": p.parent_joint_name,
                    "helper_name": p.helper_name,
                    "state_index": p.state_index,
                    "enabled_by_default": p.enabled_by_default,
                    "vertex_count": sum(len(pr.positions) for pr in p.mesh.primitives),
                    "triangle_count": sum(len(pr.indices) // 3 for pr in p.mesh.primitives),
                    "material_ids": p_mats,
                }
            )
        materials_out = []
        for mid, mat in asm.materials.items():
            pt = mat.primary_technique
            materials_out.append(
                {
                    "material_id": mid,
                    "name": mat.name,
                    "tags": mat.material_tags,
                    "passes": pt.available_passes if pt else [],
                    "texture_count": len(pt.texture_bindings) if pt else 0,
                    "variable_count": len(pt.variable_bindings) if pt else 0,
                    "shader_id": pt.shader_id if pt else None,
                    "used_by_parts": mat_part_counts.get(mid, [])[:12],
                    "used_by_count": len(mat_part_counts.get(mid, [])),
                }
            )
        materials_out.sort(key=lambda m: (-m["texture_count"], m["name"]))
        return parts_out, materials_out

    @app.get("/api/characters/{char_id}/inspect")
    def api_character_inspect(
        char_id: str,
        outfit_id: str = "",
        facial_override: str = "",
        headgear: bool = True,
        include_hair: bool = True,
        hair_lod: int = 2,
    ):
        _ensure_chars_assets_indexed()
        ckey = f"char:{char_id}:{outfit_id}:{facial_override}:{int(headgear)}:{int(include_hair)}:{hair_lod}"
        try:
            asm = state.get_extra_assembly(
                ckey,
                lambda: assemble_character(
                    state.game,
                    catalog=state.catalog,
                    char_id=char_id,
                    outfit_id=outfit_id or None,
                    headgear=headgear,
                    include_hair=include_hair,
                    hair_lod=hair_lod,
                    facial_id=facial_override or None,
                    lod_index=0,
                ),
            )
        except Exception as exc:
            raise HTTPException(status_code=404, detail=f"Could not inspect character {char_id}: {exc}") from exc
        parts_out, materials_out = _summarize_assembly_for_ui(asm)
        return {
            "entity_kind": "character",
            "char_id": char_id,
            "display_name": asm.display_name,
            "variant_label": asm.variant_label,
            "skeleton": {
                "joint_count": len(asm.skeleton.joints),
                "helper_count": len(asm.skeleton.helpers),
            },
            "parts": parts_out,
            "materials": materials_out,
        }

    @app.get("/api/assets/inspect")
    def api_asset_inspect(asset: str = Query(..., min_length=1)):
        _ensure_chars_assets_indexed()
        with state._db_lock:
            row = state.catalog.get_dev_asset(asset)
        if row is not None:
            res_id, disp, dev_p, cat_name = row["resource_id"], row["display_name"], row["dev_path"], row["category"]
        else:
            res_id, disp, dev_p, cat_name = asset, asset, asset, "props"
        ckey = f"asset:{res_id}:{dev_p}"
        try:
            asm = state.get_extra_assembly(
                ckey,
                lambda: assemble_general_asset(
                    state.game,
                    res_id,
                    display_name=disp,
                    dev_path=dev_p,
                    category=cat_name,
                    lod_index=0,
                ),
            )
        except Exception as exc:
            raise HTTPException(status_code=404, detail=f"Could not inspect asset {asset}: {exc}") from exc
        parts_out, materials_out = _summarize_assembly_for_ui(asm)
        return {
            "entity_kind": "asset",
            "dev_path": dev_p,
            "resource_id": res_id,
            "category": cat_name,
            "display_name": asm.display_name,
            "skeleton": {
                "joint_count": len(asm.skeleton.joints),
                "helper_count": len(asm.skeleton.helpers),
            },
            "parts": parts_out,
            "materials": materials_out,
        }

    @app.get("/api/assets/categories")
    def api_asset_categories():
        _ensure_chars_assets_indexed()
        with state._db_lock:
            cats = state.catalog.list_dev_asset_categories()
            st = state.catalog.stats()
        return {"categories": cats, "total": st.get("dev_assets", 0)}

    @app.get("/api/assets/tree")
    def api_asset_tree(dir: str = ""):
        _ensure_chars_assets_indexed()
        with state._db_lock:
            tree = state.catalog.list_dev_tree_children(dir)
        return tree

    @app.get("/api/assets/search")
    def api_asset_search(q: str = "", category: str = "", dir: str = "", limit: int = 160):
        _ensure_chars_assets_indexed()
        with state._db_lock:
            items = state.catalog.search_dev_assets(query=q, category=category, dir_prefix=dir, limit=limit)
        return {"items": items}

    @app.post("/api/assets/export")
    def api_export_dev_asset(req: ExportDevAssetRequest):
        import re as _re

        job_id = f"asset_{int(time.time() * 1000)}"
        state.export_jobs[job_id] = {"status": "running", "message": f"Assembling asset {req.asset}...", "output_blend": None}

        def _worker() -> None:
            try:
                _ensure_chars_assets_indexed()
                with state._db_lock:
                    row = state.catalog.get_dev_asset(req.asset)
                if row is not None:
                    res_id = row["resource_id"]
                    disp = row["display_name"]
                    dev_p = row["dev_path"]
                    cat_name = row["category"]
                else:
                    res_id, disp, dev_p, cat_name = req.asset, "", "", "props"

                asm = assemble_general_asset(
                    state.game,
                    res_id,
                    display_name=disp,
                    dev_path=dev_p,
                    category=cat_name,
                    lod_index=req.lod_index,
                )
                safe_n = _re.sub(r"[^A-Za-z0-9_\-.]+", "_", asm.display_name).strip("_")[:64] or "Asset"
                # Preserve Guerrilla's original developer folder layout under exports/Assets/<dev_path>!
                rel_parts = [p for p in dev_p.replace("\\", "/").split("/") if p] if dev_p else [safe_n]
                asset_dir = state.config.exports_dir / "Assets" / Path(*rel_parts)
                asset_dir.mkdir(parents=True, exist_ok=True)
                out_blend = asset_dir / f"{safe_n}.blend"
                bdir = asset_dir / (f"{safe_n}_UE5" if req.export_fbx else f"{safe_n}_bundle")
                fbx_out = (bdir / f"{safe_n}.fbx") if req.export_fbx else None

                state.export_jobs[job_id]["message"] = f"Exporting {asm.display_name} ({len(asm.parts)} meshes, {len(asm.materials)} materials)..."
                export_assembly_bundle(
                    state.game,
                    asm,
                    bdir,
                    max_texture_size=req.max_texture_size,
                    unpack_textures=req.unpack_textures,
                    asset_kind="asset",
                )
                state.export_jobs[job_id]["message"] = f"Building {out_blend.name}{' + UE5 FBX' if req.export_fbx else ''} in Blender..."
                build_blend_headless(
                    bdir,
                    out_blend,
                    config=state.config,
                    pack_textures=req.pack_textures,
                    export_fbx=fbx_out,
                )
                state.export_jobs[job_id] = {
                    "status": "done",
                    "message": (
                        f"Exported UE5 Asset -> {bdir / 'ue5_import_asset.py'}"
                        if req.export_fbx
                        else f"Exported {out_blend.name} ({'Rigged' if asm.skeleton.joints else 'Static'}, {len(asm.parts)} parts, {len(asm.materials)} materials)"
                    ),
                    "output_blend": str(out_blend),
                    "bundle_dir": str(bdir),
                    "ue5_script": str(bdir / "ue5_import_asset.py"),
                    "fbx_path": str(fbx_out) if fbx_out else None,
                }
            except Exception as exc:
                state.export_jobs[job_id] = {"status": "error", "message": str(exc), "output_blend": None}

        threading.Thread(target=_worker, daemon=True).start()
        return {"job_id": job_id}

    @app.get("/api/animations/{kind}/{target}")
    def api_animations_list(kind: str, target: str, category: str = "", q: str = "", limit: int = 0):
        """Discover ALL compatible HFW `EA08` animations with categories & plain-English descriptions."""
        try:
            if kind == "machine":
                oid, _code, disp = resolve_machine_entity_id(state.game, state.catalog, target)
                all_clips = discover_entity_animations(
                    state.game,
                    entity_group_id=oid.group_id,
                    asset_kind="machine",
                    limit=None,
                )
                label = disp
            else:
                gender = "male" if ("male" in target.lower() and "female" not in target.lower()) else "female"
                all_clips = discover_entity_animations(
                    state.game,
                    entity_group_id=None,
                    asset_kind="character",
                    gender=gender,
                    limit=None,
                )
                label = target
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        cat_counts = summarize_animation_categories(all_clips)
        filtered = all_clips
        cat_clean = (category or "").strip()
        if cat_clean and cat_clean.lower() != "all":
            wanted = {normalize_animation_category(tok) for tok in cat_clean.split(",") if tok.strip()}
            filtered = [c for c in filtered if c.category in wanted]
        ql = (q or "").strip().lower()
        if ql:
            filtered = [
                c
                for c in filtered
                if ql in c.name.lower()
                or ql in c.description.lower()
                or ql in c.anim_id.lower()
                or any(ql in ev.lower() for ev in c.events)
            ]
        shown = filtered[:limit] if (limit and limit > 0) else filtered
        return {
            "kind": kind,
            "target": label,
            "total": len(all_clips),
            "filtered_total": len(filtered),
            "count": len(shown),
            "category_counts": cat_counts,
            "categories_meta": [{"key": k, "label": lbl} for k, lbl in ANIMATION_CATEGORIES],
            "items": [
                {
                    "anim_id": c.anim_id,
                    "name": c.name,
                    "category": c.category,
                    "description": c.description,
                    "skeleton_id": c.skeleton_id,
                    "duration": round(c.duration, 3),
                    "sample_rate": round(c.sample_rate, 2),
                    "num_frames": c.num_frames,
                    "num_joints": c.num_joints,
                    "yaw_delta_deg": round(c.yaw_delta_deg, 1),
                    "events": list(c.events[:8]),
                }
                for c in shown
            ],
        }

    return app
