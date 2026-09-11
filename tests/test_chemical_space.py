"""Smoke tests for the chemical-space exploration modules.

These guard the things that would silently produce a wrong *picture* rather than an
error: an embedding that drops rows, a similarity matrix that counts a compound as
its own neighbour, a scaffold decomposition that loses the aromatic ring it was
asked about, and a cliff definition that stops being symmetric in the pair.

The plotting functions are exercised through the Agg backend -- not to check what
they look like, but because a broken colour-mode branch raises rather than
mis-renders, and that is worth catching in CI.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")

from cyp import constants as C  # noqa: E402
from cyp import data, embedding, scaffolds, similarity  # noqa: E402

# A small, fixed set of real compounds from the challenge data. Using real SMILES
# rather than toy ones matters here: the scaffold and MMP code paths depend on ring
# perception behaving on drug-like structures.
N_SAMPLE = 60


def _sample() -> pl.DataFrame:
    return data.load_train_inhibition().head(N_SAMPLE)


# ── embedding ─────────────────────────────────────────────────────────────────


def test_fingerprint_matrix_shape():
    sample = _sample()
    fps = embedding.fingerprint_matrix(sample)
    assert fps.shape[0] == sample.height
    # The ECFP6 default is 4096-bit; a silently different default would change every
    # embedding in the notebook.
    assert fps.shape[1] == 4096


def test_embeddings_preserve_rows_and_order():
    sample = _sample()
    fps = embedding.fingerprint_matrix(sample)

    umapped = embedding.add_umap(sample, fps)
    tsned = embedding.add_tsne(sample, fps)

    for frame, cols in ((umapped, ("UMAP_x", "UMAP_y")), (tsned, ("TSNE_x", "TSNE_y"))):
        assert frame.height == sample.height
        assert frame["Molecule_Name"].to_list() == sample["Molecule_Name"].to_list()
        for col in cols:
            assert col in frame.columns
            assert frame[col].null_count() == 0
            assert np.isfinite(frame[col].to_numpy()).all()


def test_embedding_is_deterministic():
    """Same seed, same coordinates -- otherwise a cached embedding and a fresh one
    would disagree and the notebook's cache would be silently misleading."""
    sample = _sample()
    fps = embedding.fingerprint_matrix(sample)
    first = embedding.add_umap(sample, fps)["UMAP_x"].to_numpy()
    second = embedding.add_umap(sample, fps)["UMAP_x"].to_numpy()
    np.testing.assert_allclose(first, second)


def test_embedding_scatter_colour_modes():
    sample = _sample()
    fps = embedding.fingerprint_matrix(sample)
    frame = embedding.add_umap(sample, fps).with_columns(pl.lit("#4e79a7").alias("literal_color"))

    # All three branches of the colour dispatch.
    assert embedding.embedding_scatter(frame, "UMAP_x", "UMAP_y") is not None
    assert (
        embedding.embedding_scatter(
            frame,
            "UMAP_x",
            "UMAP_y",
            color_col="literal_color",
            color_legend={"#4e79a7": "all"},
        )
        is not None
    )
    assert (
        embedding.embedding_scatter(
            frame,
            "UMAP_x",
            "UMAP_y",
            color_col=C.REGRESSION_ENDPOINTS[3],
            cmap="ryg",
        )
        is not None
    )


def test_endpoint_grid_handles_sparse_endpoints():
    """Every endpoint is null for most rows; the grid must not choke on a panel
    whose endpoint happens to have no labelled compound in the sample."""
    sample = _sample()
    fps = embedding.fingerprint_matrix(sample)
    frame = embedding.add_umap(sample, fps)
    assert embedding.endpoint_grid(frame, C.REGRESSION_ENDPOINTS) is not None


# ── similarity ────────────────────────────────────────────────────────────────


def test_pairwise_similarities_are_unique_pairs():
    sample = _sample()
    sims = similarity.pairwise_similarities(sample, "Molecule_Name", "ecfp4_1k")

    # n*(n-1)/2 unordered pairs, no self-pairs, no duplicates.
    assert sims.height == N_SAMPLE * (N_SAMPLE - 1) // 2
    assert (sims["ID1"] == sims["ID2"]).sum() == 0
    pairs = {frozenset(p) for p in sims.select(["ID1", "ID2"]).rows()}
    assert len(pairs) == sims.height

    values = sims["similarity"].to_numpy()
    assert values.min() >= 0.0 and values.max() <= 1.0


def test_cross_similarities_self_diagonal_is_one():
    """A compound compared to itself is Tanimoto 1.0. This is what makes the
    diagonal masking in the notebook's coherence cell necessary, so pin it."""
    sample = _sample().head(20)
    matrix = similarity.cross_similarities(sample, sample, "ecfp4_1k")
    assert matrix.shape == (20, 20)
    np.testing.assert_allclose(np.diag(matrix), 1.0, atol=1e-6)


def test_nearest_neighbours_finds_self_when_present():
    """The strongest available check that the NN lookup indexes correctly: when the
    query is inside the reference, every compound's nearest neighbour is itself."""
    sample = _sample().head(25)
    result = similarity.nearest_neighbours(sample, sample, carry=[C.REGRESSION_ENDPOINTS[3]])

    assert result.height == sample.height
    assert result["nn_Molecule_Name"].to_list() == sample["Molecule_Name"].to_list()
    np.testing.assert_allclose(result["nn_similarity"].to_numpy(), 1.0, atol=1e-6)
    assert f"nn_{C.REGRESSION_ENDPOINTS[3]}" in result.columns


def test_nearest_neighbours_on_disjoint_sets():
    sample = _sample()
    query, reference = sample.head(20), sample.tail(20)
    result = similarity.nearest_neighbours(query, reference)

    assert result.height == query.height
    # Every neighbour must come from the reference set, never from the query.
    assert set(result["nn_Molecule_Name"].to_list()) <= set(reference["Molecule_Name"].to_list())
    assert result["nn_similarity"].max() <= 1.0


def test_activity_cliffs_are_symmetric_and_above_threshold():
    sample = _sample()
    sims = similarity.pairwise_similarities(sample, "Molecule_Name", "ecfp4_1k")
    endpoint = C.REGRESSION_ENDPOINTS[3]
    cliffs = similarity.activity_cliffs(
        sims, sample, endpoint, sim_threshold=0.2, delta_threshold=1.0
    )

    if cliffs.height:
        assert (cliffs["similarity"] >= 0.2).all()
        assert (cliffs["delta"] >= 1.0).all()
        # delta really is |value_1 - value_2|, not a signed difference.
        recomputed = (cliffs["value_1"] - cliffs["value_2"]).abs().to_numpy()
        np.testing.assert_allclose(recomputed, cliffs["delta"].to_numpy(), atol=1e-6)

    pair_set = similarity.cliff_pair_set(cliffs)
    assert len(pair_set) == cliffs.height


def test_cliff_summary_covers_every_endpoint():
    sample = _sample()
    sims = similarity.pairwise_similarities(sample, "Molecule_Name", "ecfp4_1k")
    summary = similarity.cliff_summary(sims, sample, C.REGRESSION_ENDPOINTS, sim_threshold=0.2)

    assert summary.height == len(C.REGRESSION_ENDPOINTS)
    assert set(summary["endpoint"].to_list()) == set(C.REGRESSION_ENDPOINTS)
    # A rate is only meaningful as a percentage of the similar pairs it was drawn from.
    assert (summary["cliffs"] <= summary["similar_pairs"]).all()
    assert (summary["cliff_rate_pct"] <= 100.0).all()


def test_potency_band_boundaries():
    assert similarity.potency_band(7.0) == "potent (>=6)"
    assert similarity.potency_band(6.0) == "potent (>=6)"
    assert similarity.potency_band(5.9) == "moderate (5-6)"
    assert similarity.potency_band(4.0) == "weak (4-5)"
    assert similarity.potency_band(3.9) == "inactive (<4)"
    assert similarity.potency_band(None) == similarity.NOT_TESTED_BAND


def test_neighbour_label_profile_marks_untested_tracks():
    """Regression test. Polars' map_elements skips null inputs rather than passing
    them to the function, so an unmeasured track silently stayed null instead of
    becoming "not tested" -- which dropped the most interesting band off the plot.
    Endpoints here are sparse by construction, so some band must be "not tested"."""
    sample = _sample()
    query, reference = sample.head(20), sample.tail(30)

    profile = similarity.neighbour_label_profile(
        query,
        reference,
        regression_endpoints=C.REGRESSION_ENDPOINTS,
        classification_endpoints=(),
    )

    assert profile.height == query.height
    for endpoint in C.REGRESSION_ENDPOINTS:
        band = profile[f"band_{endpoint}"]
        assert band.null_count() == 0, f"band_{endpoint} still contains nulls"
        # A null measurement must map to "not tested", never to a potency band.
        untested = profile.filter(pl.col(f"nn_{endpoint}").is_null())
        if untested.height:
            assert (untested[f"band_{endpoint}"] == similarity.NOT_TESTED_BAND).all()


def test_neighbour_label_profile_track_counts():
    sample = _sample()
    query, reference = sample.head(20), sample.tail(30)

    profile = similarity.neighbour_label_profile(
        query,
        reference,
        regression_endpoints=C.REGRESSION_ENDPOINTS,
        classification_endpoints=(),
    )

    # n_tracks_measured must equal the number of non-null nn_* values on that row.
    recomputed = sum(
        profile[f"nn_{e}"].is_not_null().cast(pl.Int32) for e in C.REGRESSION_ENDPOINTS
    )
    assert profile["n_tracks_measured"].to_list() == recomputed.to_list()
    assert profile["n_regression_tracks"].to_list() == recomputed.to_list()
    assert (profile["n_tracks_measured"] <= len(C.REGRESSION_ENDPOINTS)).all()


def test_similarity_percentiles_are_monotonic():
    sample = _sample()
    sims = pl.concat(
        [
            similarity.pairwise_similarities(sample, "Molecule_Name", fp)
            for fp in ("maccs", "ecfp4_1k")
        ]
    )
    stats = similarity.similarity_percentiles(sims)

    assert stats.height == 2
    for row in stats.iter_rows(named=True):
        assert row["median"] <= row["Q75"] <= row["p95"] <= row["p99"] <= row["max"]


def test_unknown_fingerprint_raises():
    sample = _sample().head(5)
    try:
        similarity.bitvects_for(sample, "not_a_fingerprint")
    except ValueError as exc:
        assert "not_a_fingerprint" in str(exc)
    else:
        raise AssertionError("expected ValueError for an unknown fingerprint")


# ── scaffolds ─────────────────────────────────────────────────────────────────


def test_decompose_known_structures():
    """Hand-checkable cases, so a regression in ring perception is unambiguous."""
    result = scaffolds.decompose(
        [
            "c1ccccc1",  # benzene: one ring system, no linker, no full scaffold
            "c1ccc(-c2ccccc2)cc1",  # biphenyl: two systems directly bonded
            "CCCC",  # acyclic: contributes no rows at all
        ]
    )

    benzene = result.filter(pl.col("SMILES") == "c1ccccc1")
    assert benzene.height == 1
    assert benzene["scaffold_type"][0] == "ring_system"

    biphenyl = result.filter(pl.col("SMILES") == "c1ccc(-c2ccccc2)cc1")
    types = set(biphenyl["scaffold_type"].to_list())
    assert "ring_system" in types
    assert "linked_ring_systems" in types

    assert result.filter(pl.col("SMILES") == "CCCC").height == 0


def test_decompose_records_invalid_smiles():
    result = scaffolds.decompose(["not_a_molecule"])
    assert result.height == 1
    assert result["parse_error"][0] is not None
    assert result["scaffold_smiles"][0] is None


def test_decompose_real_compounds():
    sample = _sample()
    result = scaffolds.decompose(sample["SMILES"].to_list())

    assert result.height > 0
    assert result["parse_error"].null_count() == result.height, "all challenge SMILES parse"
    assert set(result["scaffold_type"].unique().to_list()) <= set(scaffolds.SCAFFOLD_TYPES)
    # Every emitted scaffold must be re-parseable -- that is the whole point of the
    # kekulize step in _canonical_fragment.
    assert result["scaffold_heavy_atoms"].null_count() == 0
    assert (result["scaffold_heavy_atoms"] > 0).all()


def test_scaffold_counts_respects_membership():
    sample = _sample().with_columns(
        pl.lit(True).alias("in_inhibition"),
        (pl.int_range(pl.len()) < 10).alias("in_test"),
    )
    long = scaffolds.decompose(sample["SMILES"].to_list())
    counts = scaffolds.scaffold_counts(
        long,
        membership=sample.select(["SMILES", "in_inhibition", "in_test"]),
        flag_columns=["in_inhibition", "in_test"],
    )

    assert {"n_total", "n_in_inhibition", "n_in_test"} <= set(counts.columns)
    # A flag can never be set on more molecules than carry the scaffold at all.
    assert (counts["n_in_inhibition"] <= counts["n_total"]).all()
    assert (counts["n_in_test"] <= counts["n_total"]).all()


def test_scaffold_activity_spread_is_consistent():
    sample = _sample()
    long = scaffolds.decompose(sample["SMILES"].to_list())
    spread = scaffolds.scaffold_activity(
        long, sample, endpoint=C.REGRESSION_ENDPOINTS[3], min_molecules=2
    )

    if spread.height:
        assert (spread["n_molecules"] >= 2).all()
        assert (spread["min_value"] <= spread["mean_value"]).all()
        assert (spread["mean_value"] <= spread["max_value"]).all()
        np.testing.assert_allclose(
            (spread["max_value"] - spread["min_value"]).to_numpy(),
            spread["spread"].to_numpy(),
            atol=1e-6,
        )
