"""Tests for SQLiteVec.add_texts with iterable inputs.

Regression coverage for
https://github.com/langchain-ai/langchain-community/issues/651
where generator inputs to add_texts were silently dropped because the
iterable was consumed twice inside the method.
"""

from __future__ import annotations

from typing import Iterator, List

import pytest
from langchain_core.embeddings import Embeddings

from langchain_community.vectorstores.sqlitevec import SQLiteVec


class _LengthEmbeddings(Embeddings):
    """Deterministic 1-dim embeddings keyed on text length.

    Kept inline (rather than reusing the shared FakeEmbeddings from
    tests/integration_tests) so this unit test has no cross-boundary
    imports and the embedding output is fully predictable.
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[float(len(text))] for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return [float(len(text))]


def _gen(items: List[str]) -> Iterator[str]:
    """Return a single-use generator over items.

    Crucial that this is a real generator (not a list) — the bug only
    manifests on iterables that exhaust after one pass.
    """
    yield from items


@pytest.mark.requires("sqlite_vec")
class TestSQLiteVecAddTextsIterable:
    """add_texts is typed Iterable[str]; behavior must match the type.

    Before the fix, the method ran ``embed_documents(list(texts))`` then
    iterated over ``texts`` two more times (once to build default
    metadatas, once in the row-building zip). With a generator input the
    first call exhausted the iterable, and the later iterations produced
    empty lists — so the insert payload was empty and rows were silently
    dropped.
    """

    @pytest.fixture
    def store(self, tmp_path) -> SQLiteVec:
        return SQLiteVec(
            table="t_vectors",
            connection=None,
            embedding=_LengthEmbeddings(),
            db_file=str(tmp_path / "test.db"),
        )

    def test_list_input_inserts_all_rows(self, store: SQLiteVec) -> None:
        """Baseline: list inputs have always worked. Lock in behavior."""
        ids = store.add_texts(["alpha", "beta"])

        assert len(ids) == 2
        hits = store.similarity_search("alpha", k=2)
        assert {d.page_content for d in hits} == {"alpha", "beta"}

    def test_tuple_input_inserts_all_rows(self, store: SQLiteVec) -> None:
        """Tuples are Iterable and re-iterable; worked before the fix."""
        ids = store.add_texts(("alpha", "beta"))

        assert len(ids) == 2

    def test_generator_input_inserts_all_rows(self, store: SQLiteVec) -> None:
        """Regression: generator inputs were consumed before row construction.

        The strong assertion is that documents are retrievable after
        insert — checking only ``len(ids)`` would be weaker, since under
        the bug the IDs list was empty too.
        """
        ids = store.add_texts(_gen(["alpha", "beta"]))

        assert len(ids) == 2, "generator input must not be silently dropped"
        hits = store.similarity_search("alpha", k=2)
        assert {d.page_content for d in hits} == {"alpha", "beta"}

    def test_generator_input_with_default_metadatas(self, store: SQLiteVec) -> None:
        """Second manifestation of the same root cause:
        ``metadatas = [{} for _ in texts]`` also produced ``[]`` when
        texts was an already-consumed generator. Covering this path
        explicitly prevents a partial fix that handles the embeds-side
        consumption but leaves metadatas defaulting still broken.
        """
        ids = store.add_texts(_gen(["alpha", "beta", "gamma"]), metadatas=None)

        assert len(ids) == 3
        hits = store.similarity_search("alpha", k=3)
        assert {d.page_content for d in hits} == {"alpha", "beta", "gamma"}

    def test_generator_input_with_explicit_metadatas(self, store: SQLiteVec) -> None:
        """When metadatas is explicit, it must still align with texts
        after the iterable is materialized."""
        ids = store.add_texts(
            _gen(["alpha", "beta"]),
            metadatas=[{"src": "a"}, {"src": "b"}],
        )

        assert len(ids) == 2
        hits = store.similarity_search("alpha", k=2)
        assert sorted(d.metadata.get("src") for d in hits) == ["a", "b"]
