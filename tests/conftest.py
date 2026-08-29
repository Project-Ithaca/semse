import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "indexer"))
sys.path.insert(0, str(ROOT / "tests"))

import fixture_index as fx  # noqa: E402


@pytest.fixture
def make_engine(tmp_path, monkeypatch):
    """Build a SearchEngine backed by a throwaway fixture index.

    Repoints api.search's module-level paths at a tmpdir and swaps the real
    sentence-transformers Embedder for the deterministic FakeEmbedder, so no
    model load, no network, and no dependency on the user's real index.
    """
    def _make(chunks, *, images=None, personas=None, summaries=None,
              image_index=False, llm=None):
        from api import search as search_mod

        data_dir = tmp_path / "data"
        fx.build_fixture(
            data_dir, chunks, images=images, personas=personas, summaries=summaries
        )
        if image_index:
            fx.build_image_index(data_dir, images or [])

        monkeypatch.setattr(search_mod, "INDEX_PATH", data_dir / "index.faiss")
        monkeypatch.setattr(search_mod, "META_DB_PATH", data_dir / "metadata.db")
        monkeypatch.setattr(search_mod, "ID_MAP_PATH", data_dir / "id_map.json")
        monkeypatch.setattr(search_mod, "IMAGE_INDEX_PATH", data_dir / "images.faiss")
        monkeypatch.setattr(search_mod, "IMAGE_ID_MAP_PATH", data_dir / "image_id_map.json")
        monkeypatch.setattr(search_mod, "Embedder", fx.FakeEmbedder)

        engine = search_mod.SearchEngine()
        engine._openai = llm if llm is not None else fx.FakeLLM(content="")
        if image_index:
            engine.clip_embedder = fx.FakeClipEmbedder()
        return engine

    return _make
