"""Tests for SQLiteVSS.add_texts with iterable inputs.

Regression coverage for
https://github.com/langchain-ai/langchain-community/issues/651
where generator inputs to add_texts were silently dropped because the
iterable was consumed twice inside the method. SQLiteVSS has the same
bug as SQLiteVec — fix and tests mirror that file.
"""

from __future__ import annotations

from typing import Iterator, List

import pytest
from langchain_core.embeddings import Embeddings

from langchain_community.vectorstores.sqlitevss import SQLiteVSS


class _LengthEmbeddings(Embeddings):
    """Deterministic 1-dim embeddings keyed on text length."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[float(len(text))] for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return [float(len(text))]


def _gen(items: List[str]) -> Iterator[str]:
    """Single-use generator over items, to exercise the iterable path."""
    yield from items


@pytest.mark.requires("sqlite_vss")
class TestSQLiteVSSAddTextsIterable:
    """add_texts is typed Iterable[str]; behavior must match the type."""

    @pytest.fixture
    def store(self, tmp_path) -> SQLiteVSS:
        return SQLiteVSS(
            table="t_vectors",
            connection=None,
            embedding=_LengthEmbeddings(),
            db_file=str(tmp_path / "test.db"),
        )

    def test_list_input_inserts_all_rows(self, store: SQLiteVSS) -> None:
        """Baseline: list inputs have always worked."""
        ids = store.add_texts(["alpha", "beta"])

        assert len(ids) == 2
        hits = store.similarity_search("alpha", k=2)
        assert {d.page_content for d in hits} == {"alpha", "beta"}

    def test_tuple_input_inserts_all_rows(self, store: SQLiteVSS) -> None:
        """Tuples are Iterable and re-iterable; worked before the fix."""
        ids = store.add_texts(("alpha", "beta"))

        assert len(ids) == 2

    def test_generator_input_inserts_all_rows(self, store: SQLiteVSS) -> None:
        """Regression: generator inputs were consumed before insertion."""
        ids = store.add_texts(_gen(["alpha", "beta"]))

        assert len(ids) == 2, "generator input must not be silently dropped"
        hits = store.similarity_search("alpha", k=2)
        assert {d.page_content for d in hits} == {"alpha", "beta"}

    def test_generator_input_with_default_metadatas(self, store: SQLiteVSS) -> None:
        """Second manifestation: metadatas-defaulting also broke on
        already-consumed generators. Covered explicitly here."""
        ids = store.add_texts(_gen(["alpha", "beta", "gamma"]), metadatas=None)

        assert len(ids) == 3
        hits = store.similarity_search("alpha", k=3)
        assert {d.page_content for d in hits} == {"alpha", "beta", "gamma"}

    def test_generator_input_with_explicit_metadatas(self, store: SQLiteVSS) -> None:
        ids = store.add_texts(
            _gen(["alpha", "beta"]),
            metadatas=[{"src": "a"}, {"src": "b"}],
        )

        assert len(ids) == 2
        hits = store.similarity_search("alpha", k=2)
        assert sorted(d.metadata.get("src") for d in hits) == ["a", "b"]
