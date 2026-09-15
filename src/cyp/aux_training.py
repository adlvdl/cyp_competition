"""CV harnesses for the auxiliary-data arms of `notebooks/05_auxiliary_data.py`.

`multitask.py` answers "does sharing across the four *scored* endpoints help?".
This module answers the next question: does data that is not a scored endpoint at
all help -- the single-concentration screen, the screen negatives, the selection
rule that produced the sparsity, and public PubChem measurements.

Every runner here returns `cv.oof_frame`'s long schema and takes the same shared
fold assignments as `multitask.run_cv_multitarget`, so each arm is paired against
the 03 winner (`chemprop_multitask`) on identical rows and
`evaluation.paired_bootstrap` between them is legitimate.

## The constraint every arm is shaped by

The blinded test set carries ``Molecule_Name`` and ``SMILES`` and nothing else. No
screen measurement, no Emax, no public assay record. So auxiliary data can never be
a *feature*: whatever a model reads at training time must also exist at prediction
time, and none of this does. The three usable shapes are

1. an auxiliary **target** -- predict it alongside pIC50, so the shared encoder is
   shaped by it, then ignore that head at prediction time (`run_cv_auxiliary_heads`);
2. extra labelled **rows** in the same target (`run_cv_augmented`, fed by
   `auxiliary.censored_labels`);
3. a **pretraining** corpus for the encoder, fine-tuned away afterwards
   (`run_cv_pretrained`).

Anything that looks like a fourth option is usually feature leakage wearing a hat.

## Why each arm carries its own control

PXR's retrospective records auxiliary-assay and multitask training *hurting* there,
while the same sources helped the top teams. That is a strong prior that the
direction of these effects is not predictable from the setup, so no arm is run
without its twin. `notebooks/05_auxiliary_data.py` runs the plain
`multitask.run_cv_multitarget` on the same folds as the reference row for all of them.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from . import auxiliary
from . import constants as C
from .cv import fold_assignment_splits, report_fold, shared_scaffold_folds
from .multitask import _oof_records, endpoint_indicator_frame


def _wide_frame(frames: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """One row per compound, one column per endpoint, null where unmeasured.

    Same construction `multitask.run_cv_multitarget` uses, factored out because
    every runner here needs it and they must agree cell for cell with that
    reference arm -- otherwise a difference in results could be a difference in how
    the training matrix was assembled rather than in the thing being tested.
    """
    wide = None
    for endpoint, frame in frames.items():
        block = frame.select("Molecule_Name", "SMILES", pl.col("y_true").alias(endpoint))
        wide = (
            block
            if wide is None
            else wide.join(block, on=["Molecule_Name", "SMILES"], how="full", coalesce=True)
        )
    return wide.sort("Molecule_Name")


def _split_validation(
    train_wide: pl.DataFrame, seed: int, fold: int, p_val: float
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Carve an early-stopping set out of the training compounds.

    Scaffold-clean by construction, since the fold split it is drawn from already
    was. Identical to `run_cv_multitarget`'s handling so the arms stay comparable.
    """
    rng = np.random.default_rng(seed + fold)
    order = rng.permutation(train_wide.height)
    n_val = max(1, int(p_val * train_wide.height))
    return train_wide[order[n_val:]], train_wide[order[:n_val]]


def _predictions_for(
    test_long: pl.DataFrame,
    test_wide: pl.DataFrame,
    predictions: np.ndarray,
    columns: list[str],
) -> np.ndarray:
    """Map a `(n_compounds, n_targets)` prediction block onto the long scoring table.

    `columns` is the model's full target list, which may be longer than the scored
    endpoints -- auxiliary heads occupy trailing columns and are simply never
    looked up here, which is how they train without ever being scored.
    """
    row_of = {name: i for i, name in enumerate(test_wide["Molecule_Name"].to_list())}
    column_of = {name: i for i, name in enumerate(columns)}
    return np.array(
        [
            predictions[row_of[molecule], column_of[endpoint]]
            for molecule, endpoint in zip(
                test_long["Molecule_Name"].to_list(),
                test_long["endpoint"].to_list(),
                strict=True,
            )
        ]
    )


def run_cv_auxiliary_heads(
    frames: dict[str, pl.DataFrame],
    auxiliary_targets: pl.DataFrame | None = None,
    method_name: str = "chemprop_aux_screen",
    from_foundation: str | None = None,
    n_outer: int = 5,
    n_inner: int = 5,
    seed: int = 42,
    p_val: float = 0.1,
    assignments: pl.DataFrame | None = None,
    folds: list[int] | None = None,
    on_fold=None,
    **chemprop_kwargs,
) -> pl.DataFrame:
    """A1 -- the screen as extra Chemprop output heads, trained but never scored.

    The model gets four scored pIC50 heads plus one head per auxiliary column, all
    sharing one message-passing encoder. The auxiliary heads are dropped at scoring
    time, so nothing that is missing from the test set is ever required to make a
    prediction -- the screen shapes the *encoder*, not the inference path.

    The reason to expect this to work is coverage. The screen is dense over 4,376
    compounds and all four isoforms, where the pIC50 labels are sparse and truncated,
    so the auxiliary heads give the encoder a supervised signal on roughly three
    times as many compound/isoform pairs as the scored heads do. The reason it might
    not is that a dense, easy auxiliary task can dominate the loss and pull the shared
    representation toward predicting the screen rather than potency.

    Args:
        frames: Scored endpoint name -> that endpoint's training frame.
        auxiliary_targets: Wide frame of extra targets keyed on ``Molecule_Name``;
            defaults to `auxiliary.screen_targets`. Compounds absent from it get
            nulls, which chemprop masks out of the loss.
        method_name: Name written into the OOF frame's ``method`` column.
        from_foundation: Backbone to warm-start from, e.g. ``"CHEMELEON"``.
        n_outer: Outer repeats.
        n_inner: Folds per repeat.
        seed: Split seed.
        p_val: Validation fraction per training fold, for early stopping.
        assignments: Fold table from `cv.shared_scaffold_folds`.
        folds: Run only these fold indices. Fold-major notebooks pass one at a time
            so an interrupted sweep leaves every arm complete through the same fold.
        on_fold: `(fold, n_folds) -> None` progress callback.
        **chemprop_kwargs: Forwarded to `ChempropMultitargetModel`.

    Returns:
        Long OOF frame covering only the scored endpoints.
    """
    from .graph_models import ChempropMultitargetModel

    if assignments is None:
        _, assignments = shared_scaffold_folds(frames, n_outer=n_outer, n_inner=n_inner, seed=seed)
    if auxiliary_targets is None:
        auxiliary_targets = auxiliary.screen_targets()

    endpoints = list(frames)
    aux_columns = [c for c in auxiliary_targets.columns if c not in ("Molecule_Name", "SMILES")]
    all_targets = endpoints + aux_columns

    # Left join: every compound with a scored label stays, and picks up auxiliary
    # values where the screen covered it. A right/outer join would add compounds that
    # have only auxiliary data, which is a different experiment (more encoder
    # coverage, no scored labels) and is deliberately not what this arm tests.
    wide = _wide_frame(frames).join(
        auxiliary_targets.select(["Molecule_Name", *aux_columns]),
        on="Molecule_Name",
        how="left",
    )

    stacked, _ = endpoint_indicator_frame(frames)
    n_folds = assignments["fold"].n_unique()

    records: list[dict] = []
    for fold, outer, inner, train_long, _val, test_long in fold_assignment_splits(
        stacked, assignments
    ):
        if folds is not None and fold not in folds:
            continue

        train_names = set(train_long["Molecule_Name"].to_list())
        train_wide = wide.filter(pl.col("Molecule_Name").is_in(train_names))
        fit_wide, val_wide = _split_validation(train_wide, seed, fold, p_val)

        model = ChempropMultitargetModel(
            targets=all_targets, from_foundation=from_foundation, **chemprop_kwargs
        )
        model.fit(
            fit_wide["SMILES"].to_list(),
            fit_wide.select(all_targets).to_numpy(),
            smiles_val=val_wide["SMILES"].to_list(),
            y_val=val_wide.select(all_targets).to_numpy(),
        )

        test_names = set(test_long["Molecule_Name"].to_list())
        test_wide = wide.filter(pl.col("Molecule_Name").is_in(test_names)).sort("Molecule_Name")
        predictions = model.predict(test_wide["SMILES"].to_list())

        y_pred = _predictions_for(test_long, test_wide, predictions, all_targets)
        records.extend(_oof_records(test_long, y_pred, method_name, fold, outer, inner))
        report_fold(on_fold, fold, n_folds)

    return pl.DataFrame(records)


def run_cv_augmented(
    frames: dict[str, pl.DataFrame],
    augmented_frames: dict[str, pl.DataFrame],
    method_name: str = "chemprop_augmented",
    from_foundation: str | None = None,
    n_outer: int = 5,
    n_inner: int = 5,
    seed: int = 42,
    p_val: float = 0.1,
    assignments: pl.DataFrame | None = None,
    folds: list[int] | None = None,
    on_fold=None,
    **chemprop_kwargs,
) -> pl.DataFrame:
    """A2 -- train on real labels plus censored screen negatives, score on real only.

    The split between the two frame arguments is what makes this arm honest, and it
    is the part most easily got wrong. `augmented_frames` supplies the *training*
    matrix and includes the weak censored rows; `frames` supplies the *scoring*
    table and holds only genuine dose-response measurements. A weak label therefore
    never enters a reported metric, whichever fold it lands in.

    Scoring on the weak rows would be doubly wrong: it would reward a model for
    reproducing a label this repo invented, and because those rows carry a wide
    credible interval, ST-RAE would score most of them as zero error and flatter the
    arm by construction.

    Args:
        frames: Scored endpoints -> real-label frames. Drives scoring.
        augmented_frames: Same keys -> frames including the censored rows. Drives
            training. Build with `auxiliary.augmented_training_frame`.
        method_name: Name written into the OOF ``method`` column.
        from_foundation: Backbone to warm-start from.
        n_outer: Outer repeats.
        n_inner: Folds per repeat.
        seed: Split seed.
        p_val: Validation fraction per training fold.
        assignments: Fold table. Must be assigned over the *union* of augmented
            compounds, or a weak row could sit in another arm's test fold -- pass
            `cv.shared_scaffold_folds(augmented_frames)`.
        folds: Run only these fold indices.
        on_fold: Progress callback.
        **chemprop_kwargs: Forwarded to `ChempropMultitargetModel`.

    Returns:
        Long OOF frame over the real labels only.
    """
    from .graph_models import ChempropMultitargetModel

    if set(frames) != set(augmented_frames):
        raise ValueError(
            f"frames and augmented_frames must cover the same endpoints; "
            f"got {sorted(frames)} and {sorted(augmented_frames)}"
        )
    if assignments is None:
        _, assignments = shared_scaffold_folds(
            augmented_frames, n_outer=n_outer, n_inner=n_inner, seed=seed
        )

    endpoints = list(frames)
    train_wide_all = _wide_frame(augmented_frames)
    score_wide = _wide_frame(frames)

    # Scoring runs off the real-label stack; training reads the augmented one. The
    # fold assignment is shared, so a compound is on the same side of the split in
    # both regardless of which frame it came from.
    scored_stack, _ = endpoint_indicator_frame(frames)
    augmented_stack, _ = endpoint_indicator_frame(augmented_frames)
    n_folds = assignments["fold"].n_unique()

    train_splits = {
        fold: train_long
        for fold, _outer, _inner, train_long, _val, _test in fold_assignment_splits(
            augmented_stack, assignments
        )
    }

    records: list[dict] = []
    for fold, outer, inner, _train_long, _val, test_long in fold_assignment_splits(
        scored_stack, assignments
    ):
        if folds is not None and fold not in folds:
            continue

        train_names = set(train_splits[fold]["Molecule_Name"].to_list())
        train_wide = train_wide_all.filter(pl.col("Molecule_Name").is_in(train_names))
        fit_wide, val_wide = _split_validation(train_wide, seed, fold, p_val)

        model = ChempropMultitargetModel(
            targets=endpoints, from_foundation=from_foundation, **chemprop_kwargs
        )
        model.fit(
            fit_wide["SMILES"].to_list(),
            fit_wide.select(endpoints).to_numpy(),
            smiles_val=val_wide["SMILES"].to_list(),
            y_val=val_wide.select(endpoints).to_numpy(),
        )

        test_names = set(test_long["Molecule_Name"].to_list())
        test_wide = score_wide.filter(pl.col("Molecule_Name").is_in(test_names)).sort(
            "Molecule_Name"
        )
        predictions = model.predict(test_wide["SMILES"].to_list())

        y_pred = _predictions_for(test_long, test_wide, predictions, endpoints)
        records.extend(_oof_records(test_long, y_pred, method_name, fold, outer, inner))
        report_fold(on_fold, fold, n_folds)

    return pl.DataFrame(records)


def public_pretraining_matrix(
    endpoints: list[str],
    exclude_smiles: list[str] | None = None,
    snapshot: str | None = None,
    include_inactives: bool = True,
) -> pl.DataFrame:
    """Assemble the PubChem panel into one wide multi-target pretraining frame.

    Args:
        endpoints: Scored endpoint names, in the column order the fine-tuned model
            uses. The pretraining head must line up with the fine-tuning head
            position for position, or the warm-start transfers the wrong weights.
        exclude_smiles: Challenge SMILES to remove from the pretraining set. Pass the
            full challenge set: a public compound that is structurally identical to a
            challenge compound would let a fold's test label reach the encoder
            through the pretraining corpus, which is leakage even though the label
            came from a different lab.
        snapshot: External snapshot date.
        include_inactives: Keep explicitly-inactive public compounds, labelled at the
            assay's weakest measurable potency.

    Returns:
        Wide frame with ``smiles_std`` and one column per endpoint, nulls where that
        isoform never tested the compound.
    """
    from . import external

    excluded: set[str] = set()
    if exclude_smiles:
        for smiles in exclude_smiles:
            standard = external.standardise_smiles(smiles)
            if standard is not None:
                excluded.add(standard)

    wide = None
    for endpoint in endpoints:
        isoform = endpoint.split("_")[0]
        block = external.pretraining_frame(
            isoform, snapshot=snapshot, include_inactives=include_inactives
        ).select("smiles_std", pl.col("y_true").alias(endpoint))
        wide = (
            block if wide is None else wide.join(block, on="smiles_std", how="full", coalesce=True)
        )

    if excluded:
        wide = wide.filter(pl.col("smiles_std").is_in(list(excluded)).not_())
    return wide.sort("smiles_std")


def run_cv_pretrained(
    frames: dict[str, pl.DataFrame],
    pretraining: pl.DataFrame,
    method_name: str = "chemprop_pubchem_pretrained",
    freeze_encoder: bool = False,
    pretrain_epochs: int = 30,
    n_outer: int = 5,
    n_inner: int = 5,
    seed: int = 42,
    p_val: float = 0.1,
    assignments: pl.DataFrame | None = None,
    folds: list[int] | None = None,
    on_fold=None,
    **chemprop_kwargs,
) -> pl.DataFrame:
    """A3 -- pretrain the encoder on public PubChem data, then fine-tune per fold.

    The encoder is pretrained **once**, before the fold loop, and every fold
    fine-tunes from that same checkpoint. That is legitimate here in a way it would
    not be for challenge data: the pretraining corpus contains no challenge label, so
    no fold's test measurement can reach the encoder through it -- provided the
    structural overlap was removed, which `public_pretraining_matrix`'s
    ``exclude_smiles`` does.

    Pretraining once rather than per fold is also the only affordable option: a
    per-fold pretrain would multiply the most expensive step in the repo by 25.

    Args:
        frames: Scored endpoint name -> training frame.
        pretraining: Wide frame from `public_pretraining_matrix`, with a
            ``smiles_std`` column and one column per endpoint in the same order.
        method_name: Name written into the OOF ``method`` column.
        freeze_encoder: Lock the message-passing weights during fine-tuning. Worth
            testing both ways -- freezing protects the pretrained representation from
            a small fine-tuning set but cannot adapt to the assay shift between the
            public panel and the challenge's protocol.
        pretrain_epochs: Epochs for the single pretraining run.
        n_outer: Outer repeats.
        n_inner: Folds per repeat.
        seed: Split seed.
        p_val: Validation fraction per training fold.
        assignments: Fold table from `cv.shared_scaffold_folds`.
        folds: Run only these fold indices. The pretrain runs on the first call and
            the checkpoint is reused, so a fold-major notebook pays for it once.
        on_fold: Progress callback.
        **chemprop_kwargs: Forwarded to `ChempropMultitargetModel`.

    Returns:
        Long OOF frame over the scored endpoints.
    """
    from .graph_models import ChempropMultitargetModel

    if assignments is None:
        _, assignments = shared_scaffold_folds(frames, n_outer=n_outer, n_inner=n_inner, seed=seed)

    endpoints = list(frames)
    missing = [e for e in endpoints if e not in pretraining.columns]
    if missing:
        raise ValueError(f"pretraining frame is missing endpoint columns {missing}")

    wide = _wide_frame(frames)
    stacked, _ = endpoint_indicator_frame(frames)
    n_folds = assignments["fold"].n_unique()

    # One pretrain for the whole run. A dedicated directory keeps this checkpoint
    # away from the default one, so an unpretrained control arm in the same session
    # cannot accidentally warm-start from it -- silent contamination that would make
    # the control look as good as the treatment and hide a real effect.
    pretrain_dir = C.PROJECT_ROOT / ".chemprop_pretrain" / method_name
    template = ChempropMultitargetModel(
        targets=endpoints,
        pretrain_dir=pretrain_dir,
        pretrain_epochs=pretrain_epochs,
        **chemprop_kwargs,
    )
    if not (pretrain_dir / "model_0" / "best.pt").exists():
        rng = np.random.default_rng(seed)
        order = rng.permutation(pretraining.height)
        n_val = max(1, int(p_val * pretraining.height))
        pre_val, pre_fit = pretraining[order[:n_val]], pretraining[order[n_val:]]
        template.pretrain_wide(
            pre_fit["smiles_std"].to_list(),
            pre_fit.select(endpoints).to_numpy(),
            pre_val["smiles_std"].to_list(),
            pre_val.select(endpoints).to_numpy(),
        )

    records: list[dict] = []
    for fold, outer, inner, train_long, _val, test_long in fold_assignment_splits(
        stacked, assignments
    ):
        if folds is not None and fold not in folds:
            continue

        train_names = set(train_long["Molecule_Name"].to_list())
        train_wide = wide.filter(pl.col("Molecule_Name").is_in(train_names))
        fit_wide, val_wide = _split_validation(train_wide, seed, fold, p_val)

        model = ChempropMultitargetModel(
            targets=endpoints,
            pretrain_dir=pretrain_dir,
            freeze_encoder=freeze_encoder,
            **chemprop_kwargs,
        )
        model.fit(
            fit_wide["SMILES"].to_list(),
            fit_wide.select(endpoints).to_numpy(),
            smiles_val=val_wide["SMILES"].to_list(),
            y_val=val_wide.select(endpoints).to_numpy(),
        )

        test_names = set(test_long["Molecule_Name"].to_list())
        test_wide = wide.filter(pl.col("Molecule_Name").is_in(test_names)).sort("Molecule_Name")
        predictions = model.predict(test_wide["SMILES"].to_list())

        y_pred = _predictions_for(test_long, test_wide, predictions, endpoints)
        records.extend(_oof_records(test_long, y_pred, method_name, fold, outer, inner))
        report_fold(on_fold, fold, n_folds)

    return pl.DataFrame(records)


def run_cv_selection_corrected(
    frames: dict[str, pl.DataFrame],
    method_name: str = "chemprop_selection_corrected",
    propensity_method: str = "lgbm",
    fingerprint: str = "ecfp",
    n_bits: int = 2048,
    n_outer: int = 5,
    n_inner: int = 5,
    seed: int = 42,
    p_val: float = 0.1,
    assignments: pl.DataFrame | None = None,
    folds: list[int] | None = None,
    on_fold=None,
    **chemprop_kwargs,
) -> pl.DataFrame:
    """A4 -- correct for the screen-to-curve triage that made the labels a biased sample.

    A compound has a pIC50 only because it hit in the primary screen, so the labelled
    set is a truncated selection rather than a sample of chemical space. On CYP2D6
    the truncation is nearly a hard threshold (`auxiliary.triage_separation`), and
    CYP2D6 is the endpoint that has resisted every method tried so far. That
    coincidence is what this arm tests.

    The correction is inverse-propensity weighting, in two stages:

    1. A classifier predicts, from structure alone, whether a compound was progressed
       to a dose-response curve (`auxiliary.triage_frame`). Fitted inside the fold,
       on training compounds only, so the propensity carries no test information.
    2. The regression fold trains with each compound weighted by ``1 / p``, its
       inverse propensity of being selected. Compounds that were unlikely to be
       measured but were measured anyway stand in for the unmeasured region, which is
       the region the model currently never sees.

    Weights are clipped: an inverse propensity is unbounded as ``p`` approaches zero,
    and a single compound carrying a weight of several hundred would dominate the
    loss and add variance far exceeding the bias it removes.

    Chemprop weights a *datapoint* rather than a (datapoint, target) pair, so the
    per-endpoint propensities are averaged over the endpoints a compound actually
    carries. That is an approximation, and it is the reason this arm uses one weight
    per compound rather than four.

    Args:
        frames: Scored endpoint name -> training frame.
        method_name: Name written into the OOF ``method`` column.
        propensity_method: Key of `models.MODEL_FACTORIES`' classifier twin
            (`models.CLASSIFIER_FACTORIES`) used for stage one.
        fingerprint: Featurizer for the propensity model.
        n_bits: Fingerprint width for the propensity model.
        n_outer: Outer repeats.
        n_inner: Folds per repeat.
        seed: Split seed.
        p_val: Validation fraction per training fold.
        assignments: Fold table from `cv.shared_scaffold_folds`.
        folds: Run only these fold indices.
        on_fold: Progress callback.
        **chemprop_kwargs: Forwarded to `ChempropMultitargetModel`.

    Returns:
        Long OOF frame over the scored endpoints.
    """
    from . import fingerprints, models
    from .graph_models import ChempropMultitargetModel

    if assignments is None:
        _, assignments = shared_scaffold_folds(frames, n_outer=n_outer, n_inner=n_inner, seed=seed)

    endpoints = list(frames)
    wide = _wide_frame(frames)
    stacked, _ = endpoint_indicator_frame(frames)
    n_folds = assignments["fold"].n_unique()

    # The triage tables cover every screened compound, labelled or not -- they are
    # what carries the information about compounds the challenge never measured.
    triage = {endpoint: auxiliary.triage_frame(endpoint) for endpoint in endpoints}
    triage_features = {
        endpoint: fingerprints.compute(frame["SMILES"].to_list(), fingerprint, n_bits=n_bits)
        for endpoint, frame in triage.items()
    }

    records: list[dict] = []
    for fold, outer, inner, train_long, _val, test_long in fold_assignment_splits(
        stacked, assignments
    ):
        if folds is not None and fold not in folds:
            continue

        train_names = set(train_long["Molecule_Name"].to_list())

        # Stage one, per endpoint: how likely was each training compound to have been
        # progressed to a curve? Fitted only on compounds in this fold's training
        # side, so no test compound informs its own weight.
        propensity: dict[str, dict[str, float]] = {}
        for endpoint in endpoints:
            frame, features = triage[endpoint], triage_features[endpoint]
            in_train = np.array(
                [name in train_names for name in frame["Molecule_Name"].to_list()], dtype=bool
            )
            if in_train.sum() < 2 or len(set(frame["y_true"].to_numpy()[in_train])) < 2:
                continue
            classifier = models.CLASSIFIER_FACTORIES[propensity_method]()
            classifier.fit(features[in_train], frame["y_true"].to_numpy()[in_train])
            scores = classifier.predict_proba(features[in_train])[:, 1]
            propensity[endpoint] = dict(
                zip(
                    np.array(frame["Molecule_Name"].to_list())[in_train].tolist(),
                    scores.tolist(),
                    strict=True,
                )
            )

        train_wide = wide.filter(pl.col("Molecule_Name").is_in(train_names))
        weights = _inverse_propensity_weights(train_wide, endpoints, propensity)
        fit_wide, val_wide, fit_weights = _split_validation_weighted(
            train_wide, weights, seed, fold, p_val
        )

        model = ChempropMultitargetModel(targets=endpoints, **chemprop_kwargs)
        model.fit(
            fit_wide["SMILES"].to_list(),
            fit_wide.select(endpoints).to_numpy(),
            smiles_val=val_wide["SMILES"].to_list(),
            y_val=val_wide.select(endpoints).to_numpy(),
            sample_weight=fit_weights,
        )

        test_names = set(test_long["Molecule_Name"].to_list())
        test_wide = wide.filter(pl.col("Molecule_Name").is_in(test_names)).sort("Molecule_Name")
        predictions = model.predict(test_wide["SMILES"].to_list())

        y_pred = _predictions_for(test_long, test_wide, predictions, endpoints)
        records.extend(_oof_records(test_long, y_pred, method_name, fold, outer, inner))
        report_fold(on_fold, fold, n_folds)

    return pl.DataFrame(records)


#: Ceiling on an inverse-propensity weight, as a multiple of the mean weight. A
#: compound with a vanishing propensity would otherwise carry unbounded weight and
#: dominate the loss -- trading a modest bias reduction for a large variance increase.
WEIGHT_CLIP = 10.0


def _inverse_propensity_weights(
    train_wide: pl.DataFrame,
    endpoints: list[str],
    propensity: dict[str, dict[str, float]],
) -> np.ndarray:
    """One weight per training compound, averaged over the endpoints it carries.

    Chemprop weights datapoints rather than (datapoint, target) pairs, so a compound
    measured for three isoforms gets one weight covering all three. Compounds with no
    propensity estimate fall back to 1.0, which leaves them at the unweighted default
    rather than silently dropping them.
    """
    names = train_wide["Molecule_Name"].to_list()
    weights = np.ones(len(names), dtype=float)

    for i, name in enumerate(names):
        estimates = [
            propensity[endpoint][name]
            for endpoint in endpoints
            if endpoint in propensity
            and name in propensity[endpoint]
            and train_wide[endpoint][i] is not None
        ]
        if estimates:
            # Floor the propensity before inverting: a predicted probability can be
            # arbitrarily close to zero, and 1/p would overflow the weight budget
            # before the clip below ever saw it.
            weights[i] = 1.0 / max(float(np.mean(estimates)), 1e-3)

    weights /= weights.mean()
    return np.clip(weights, 1.0 / WEIGHT_CLIP, WEIGHT_CLIP)


def _split_validation_weighted(
    train_wide: pl.DataFrame,
    weights: np.ndarray,
    seed: int,
    fold: int,
    p_val: float,
) -> tuple[pl.DataFrame, pl.DataFrame, np.ndarray]:
    """`_split_validation`, carrying the weight vector through the same permutation."""
    rng = np.random.default_rng(seed + fold)
    order = rng.permutation(train_wide.height)
    n_val = max(1, int(p_val * train_wide.height))
    fit_index, val_index = order[n_val:], order[:n_val]
    return train_wide[fit_index], train_wide[val_index], weights[fit_index]
