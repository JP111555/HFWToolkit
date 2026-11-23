"""Phase 1 tests: Decima core hashes, DSAR/LZ4, RTTI, streaming_graph.core, recursive group reads & catalog."""

from __future__ import annotations

from pathlib import Path
import pytest

from hfw_toolkit.config import ToolkitConfig
from hfw_toolkit.decima import AssetCatalog, HFWGame, RTTISchema, decima_string_hash, decima_type_hash


@pytest.fixture(scope="module")
def game() -> HFWGame:
    cfg = ToolkitConfig.load()
    g = HFWGame(cfg)
    yield g
    g.close()


def test_hashes() -> None:
    assert decima_type_hash("StreamingGraphResource") == 0x929D7AF6A30CD1C5
    # Verify CRC32C string hash is deterministic 31-bit non-negative int
    h = decima_string_hash("Texture0")
    assert 0 <= h <= 0x7FFFFFFF


def test_streaming_graph_loaded(game: HFWGame) -> None:
    assert len(game.graph.groups) > 100_000
    assert len(game.graph.files) >= 100
    assert len(game.graph.type_table_indices) > 10_000_000


def test_recursive_machine_group_deserialization(game: HFWGame) -> None:
    """Verify full recursive deserialization (including Physics/Ragdoll/LocalizedText/UITexture/Shaders) on Tallneck (5756)."""
    objs = game.read_group(5756, read_subgroups=True)
    assert len(objs) == game.graph.group(5756).num_objects
    types = {o["_type"] for o in objs}
    assert "ControlledEntityResource" in types
    assert "DestructibilityResource" in types or "SkinnedModelResource" in types


def test_catalog_build_and_queries(game: HFWGame) -> None:
    catalog = AssetCatalog(game.config.index_db_path)
    try:
        if not catalog.is_built():
            catalog.build(game, progress_cb=print)
        stats = catalog.stats()
        assert stats["machines"] >= 20
        assert stats["texture_sets"] >= 5000
        machines = catalog.list_machines()
        assert len(machines) >= 20
    finally:
        catalog.close()


def test_zero_setup_exe_rtti_and_builtin_fallback(tmp_path: Path) -> None:
    """Verify HFWToolkit works on any vanilla HFW install without dump/hfw.json, and falls back to builtin schema."""
    base_cfg = ToolkitConfig.load()
    # 1. Simulate a vanilla HFW install without dump/hfw.json
    vanilla_cfg = ToolkitConfig(
        game_dir=base_cfg.game_dir,
        rtti_json=tmp_path / "no_dump_exists.json",
        blender_exe=base_cfg.blender_exe,
        cache_dir=tmp_path / "cache",
        exports_dir=tmp_path / "exports",
    )
    schema_exe = RTTISchema.load_for_game(vanilla_cfg)
    assert "HorizonForbiddenWest.exe" in schema_exe.source_label
    assert len(schema_exe.types) >= 12000

    # Verify full group deserialization works using the zero-setup static EXE RTTI schema
    g = HFWGame(vanilla_cfg)
    try:
        objs = g.read_group(5756, read_subgroups=True)
        assert len(objs) == g.graph.group(5756).num_objects
    finally:
        g.close()

    # 2. Simulate an archive-only directory without HorizonForbiddenWest.exe or dump/hfw.json
    archive_only_cfg = ToolkitConfig(
        game_dir=tmp_path / "empty_game_dir",
        rtti_json=tmp_path / "no_dump_exists.json",
        cache_dir=tmp_path / "cache2",
        exports_dir=tmp_path / "exports2",
    )
    # Force game_dir to remain the empty dir (bypass auto_detect fallback)
    archive_only_cfg.game_dir = tmp_path / "empty_game_dir"
    schema_builtin = RTTISchema.load_for_game(archive_only_cfg)
    assert "built-in" in schema_builtin.source_label
    assert len(schema_builtin.types) >= 12000
