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

    # Excluded on InChIKey connectivity block, not canonical SMILES. SMILES
    # distinguishes stereoisomers, tautomers and salt forms that are the *same
    # compound* for leakage purposes -- see `external.inchikey_skeleton`.
    excluded = external.skeleton_keys(exclude_smiles) if exclude_smiles else set()

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
        wide = (
            external.add_skeleton(wide, column="smiles_std")
            .filter(
                pl.col("inchikey_skeleton").is_in(list(excluded)).not_()
                | pl.col("inchikey_skeleton").is_null()
            )
            .drop("inchikey_skeleton")
        )
    return wide.sort("smiles_std")


def public_union_matrix(
    endpoints: list[str],
    exclude_smiles: list[str] | None = None,
    snapshot: str | None = None,
    include_inactives: bool = True,
    include_max_response: bool = True,
) -> tuple[pl.DataFrame, list[str]]:
    """`public_pretraining_matrix` widened with the qHTS efficacy readout.

    The change the SuperCowPowers entry's union FeatureSet makes over what notebook
    05 already does, and the cheapest real gain available: their argument is that
    ``max_response`` -- percent change from control at the top tested concentration --
    is recorded for *every* screened compound, where a pIC50 exists only where a curve
    fitted. Measured on our own snapshot that is 100% coverage against roughly 50%,
    so the efficacy column carries the entire inactive half of the library, which is
    exactly the low-activity region the challenge's hit-enriched labels never reach.

    Each isoform's efficacy is a separate head on its own scale rather than being
    folded into the potency column. The two are different measurements -- efficacy is
    negative for inhibition and is not a potency at all -- so merging them would need
    a cross-assay correction that separate heads make unnecessary.

    Args:
        endpoints: Scored endpoint names, in the fine-tuning model's column order.
            These occupy the first positions so a warm-start lines up head for head.
        exclude_smiles: Challenge SMILES to remove, as in `public_pretraining_matrix`.
        snapshot: External snapshot date.
        include_inactives: Keep explicitly-inactive compounds in the potency columns.
        include_max_response: Add the efficacy heads. False reproduces
            `public_pretraining_matrix` exactly, which is the control arm.

    Returns:
        `(wide, columns)` -- the frame, and the full target order including the
        auxiliary heads. Pass `columns` wherever the head order matters.
    """
    from . import external

    wide = public_pretraining_matrix(
        endpoints,
        exclude_smiles=exclude_smiles,
        snapshot=snapshot,
        include_inactives=include_inactives,
    )
    columns = list(endpoints)
    if not include_max_response:
        return wide, columns

    excluded = external.skeleton_keys(exclude_smiles) if exclude_smiles else set()

    for endpoint in endpoints:
        isoform = endpoint.split("_")[0]
        name = f"{isoform}_max_response"
        block = external.add_standard_smiles(external.load_pubchem(isoform, snapshot))
        # A compound can appear on several plates or as several salts of one parent;
        # the median is the robust summary, matching `pretraining_frame`.
        block = (
            block.filter(pl.col("max_response").is_not_null())
            .group_by("smiles_std")
            .agg(pl.col("max_response").median().alias(name))
        )
        if excluded:
            block = (
                external.add_skeleton(block, column="smiles_std")
                .filter(
                    pl.col("inchikey_skeleton").is_in(list(excluded)).not_()
                    | pl.col("inchikey_skeleton").is_null()
                )
                .drop("inchikey_skeleton")
            )
        wide = wide.join(block, on="smiles_std", how="full", coalesce=True)
        columns.append(name)

    return wide.sort("smiles_std"), columns


def full_union_matrix(
    endpoints: list[str],
    exclude_smiles: list[str] | None = None,
    snapshot: str | None = None,
    include_chembl: bool = True,
    include_tox21: bool = True,
    include_cyp2c19: bool = True,
) -> tuple[pl.DataFrame, list[str]]:
    """Every public source, each as its own head on its own scale.

    The full version of the reference entry's union FeatureSet
    (https://supercowpowers.github.io/workbench/blogs/cyp_challenge/), where
    `public_union_matrix` covers only the PubChem efficacy readout. Sources are
    *unioned as rows* and each keeps its own target column, so a compound measured by
    two labs contributes to both heads without either scale being forced onto the
    other.

    **Why separate heads rather than a merged potency column.** On shared compounds
    the assays correlate only 0.31-0.66 and disagree systematically in level, so
    merging would need a cross-assay affine correction per pair. Separate heads make
    that unnecessary: the encoder learns from every source's chemistry, and each head
    absorbs its own source's scale. This is also what makes ChEMBL usable at all --
    `cyp.external` documents why its heterogeneity made it a poor *pretraining
    target*, and that objection does not apply to a head of its own.

    **What each source contributes**, measured on our own snapshots rather than taken
    from the blog:

    - Veith qHTS pIC50: the base corpus, ~12,900 compounds at 4 heads.
    - Veith ``max_response``: efficacy at the top dose, 100% coverage against ~50%
      for potency, so it carries the inactive half of that library.
    - ChEMBL: ~24,900 new skeletons with almost no challenge overlap. New chemistry,
      not new labels; nothing below pIC50 4.0.
    - Tox21: ~2,000 explicitly *inactive* compounds per isoform. Note the blog's
      claim that its actives are weak (median 4.76) does not reproduce here -- ours
      sit at 4.92-5.07. The confirmed negatives are the real contribution, and no
      other source has any. No CYP1A2 assay exists in this batch.
    - CYP2C19: an unscored fifth isoform every public panel measures anyway, free as
      a correlated auxiliary task.

    Args:
        endpoints: Scored endpoint names, in the fine-tuning column order. These
            occupy the first positions so a warm-start lines up head for head.
        exclude_smiles: Challenge SMILES to remove from every source.
        snapshot: External snapshot date.
        include_chembl: Add the ChEMBL potency heads.
        include_tox21: Add the Tox21 potency heads.
        include_cyp2c19: Add CYP2C19 wherever a source measures it.

    Returns:
        `(wide, columns)` -- the frame, and the full head order.
    """
    from . import chembl as chembl_module
    from . import external
    from . import tox21 as tox21_module

    wide, columns = public_union_matrix(endpoints, exclude_smiles=exclude_smiles, snapshot=snapshot)

    excluded = external.skeleton_keys(exclude_smiles) if exclude_smiles else set()

    def _attach(block: pl.DataFrame, name: str) -> None:
        """Standardise, deduplicate and union one source's column onto the frame."""
        nonlocal wide
        block = external.add_standard_smiles(block).filter(pl.col(name).is_not_null())
        if excluded:
            block = (
                external.add_skeleton(block, column="smiles_std")
                .filter(
                    pl.col("inchikey_skeleton").is_in(list(excluded)).not_()
                    | pl.col("inchikey_skeleton").is_null()
                )
                .drop("inchikey_skeleton")
            )
        # Salts and stereoisomers collapse to one structure, so a median is the
        # robust summary -- matching `external.pretraining_frame`.
        block = block.group_by("smiles_std").agg(pl.col(name).median().alias(name))
        wide = wide.join(block, on="smiles_std", how="full", coalesce=True)
        columns.append(name)

    isoforms = [e.split("_")[0] for e in endpoints]

    if include_chembl:
        targets = list(C.CHEMBL_TARGETS) if include_cyp2c19 else isoforms
        for isoform in targets:
            if isoform not in C.CHEMBL_TARGETS:
                continue
            name = f"{isoform}_pic50_chembl"
            frame = chembl_module.load_chembl(isoform, snapshot)
            if frame.height:
                _attach(frame.rename({"pIC50": name}).select("SMILES", name), name)

    if include_tox21:
        for isoform in C.TOX21_AIDS:
            name = f"{isoform}_pic50_tox21"
            frame = tox21_module.load_tox21(isoform, snapshot)
            if frame.height:
                _attach(frame.rename({"pIC50": name}).select("SMILES", name), name)

    return wide.sort("smiles_std"), columns


def challenge_auxiliary_matrix(
    endpoints: list[str], snapshot: str | None = None
) -> tuple[pl.DataFrame, list[str]]:
    """The challenge's own unused arms, as extra fine-tuning heads.

    Two of the five released files carry no scored endpoint and no notebook before
    this one used them as targets. They need no download and no cross-assay
    correction -- same platform, same chemistry, same lab as the scored labels.

    - **TDI-condition pIC50** lifts CYP3A4 coverage from 2,335 to 3,583 (+53%),
      measured here, and brings 1,240 compounds the direct-inhibition table does not
      contain at all.
    - **Emax against positive control** is the one readout carrying CYP2D6-specific
      signal: its direct-inhibition variant correlates **+0.770** (Spearman) with
      CYP2D6 potency against -0.39 to -0.46 on the other three. For the endpoint
      whose weak ordering caps our macro, that is the strongest auxiliary signal in
      the release.

    Both Emax variants are kept. The direct-inhibition one is stronger on every
    isoform, but the TDI-condition one is measured under different conditions and is
    not redundant with it.

    Returns:
        `(frame, columns)` keyed by ``Molecule_Name`` with ``SMILES``, ready to join
        onto a fine-tuning wide frame.
    """
    from . import data as data_module

    tdi = data_module.load_train_tdi(snapshot)
    emax = data_module.load_train_emax(snapshot)

    isoforms = [e.split("_")[0] for e in endpoints]
    tdi_columns = [f"{i}_pIC50_TDI_condition" for i in isoforms]
    emax_columns = [
        f"{i}_EmaxVsPosCtrl_{condition}"
        for i in isoforms
        for condition in ("direct_inhibition", "TDI_condition")
    ]

    frame = tdi.select(["Molecule_Name", "SMILES", *tdi_columns]).join(
        emax.select(["Molecule_Name", *[c for c in emax_columns if c in emax.columns]]),
        on="Molecule_Name",
        how="left",
    )
    present = [c for c in tdi_columns + emax_columns if c in frame.columns]
    return frame, present


def run_cv_pretrained(
    frames: dict[str, pl.DataFrame],
    pretraining: pl.DataFrame,
    method_name: str = "chemprop_pubchem_pretrained",
    pretrain_targets: list[str] | None = None,
    finetune_targets: pl.DataFrame | None = None,
    freeze_encoder: bool = False,
    pretrain_epochs: int = 30,
    n_outer: int = 5,
    n_inner: int = 5,
    seed: int = 42,
    p_val: float = 0.1,
    assignments: pl.DataFrame | None = None,
    folds: list[int] | None = None,
    on_fold=None,
    uncertainty: bool = False,
    finetune_descriptors: pl.DataFrame | None = None,
    descriptor_timeout_s: float | None = None,
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
        pretrain_targets: Columns of `pretraining` to pretrain on. Defaults to the
            scored endpoints alone, which is 05's behaviour. **Pass the full head
            list** from `public_union_matrix` or `full_union_matrix` to actually use
            the auxiliary sources: without it those columns are built and then
            silently dropped, so arms built on different matrices train on identical
            data and the comparison measures nothing.

            The scored endpoints must come first, in `frames` order, because chemprop
            restores output heads positionally when warm-starting.
        finetune_targets: Extra per-compound targets to fine-tune on alongside the
            scored endpoints, keyed on ``Molecule_Name`` -- see
            `challenge_auxiliary_matrix` for the challenge's own TDI-condition and
            Emax arms. Unmeasured cells stay null and chemprop masks them in the loss.

            Both stages must expose the same head list or the warm-start cannot line
            up, so each frame is padded with null columns for whatever the other has.
            A null means "not measured", which is what a public-assay head is for a
            challenge compound and what a challenge auxiliary is for a public one.
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
        uncertainty: Request `y_unc` alongside `y_pred` in the OOF frame (notebook
            06). Requires `chemprop_kwargs` to include an `uncertainty_method`, since
            that is what actually gives chemprop something to report -- see
            `graph_models.ChempropModel.predict`. `y_unc` is always a variance.
        finetune_descriptors: Molecule-level extra features for the *fine-tuning*
            stage only -- keyed on ``Molecule_Name``, one column per name in
            `chemprop_kwargs["descriptor_columns"]`. Not used for pretraining: the
            pretraining corpus is public data with no pharmacophore descriptors
            precomputed, and the geometry hypothesis this exists to test
            (notebook 09) is about the challenge-scale fine-tuning signal, not the
            encoder's warm start. Every fine-tuning compound (train, val and test in
            every fold) must have a row here when `descriptor_columns` is set, or the
            join below silently drops it and `ChempropMultitargetModel` raises on the
            resulting shape mismatch rather than training on a partial table.
        descriptor_timeout_s: Wall-clock bound per molecule for the pretraining
            corpus's descriptor computation below (only reached when
            `descriptor_columns` is set). Public-library molecules average larger
            and more flexible than the challenge's own curated set, so a handful
            can dominate an otherwise-fast batch; a timed-out molecule gets NaN
            descriptors rather than stalling the whole pretrain. `None` (default)
            disables the bound -- see `pharmacophore.descriptors`. The resulting
            matrices are themselves cached to `pretrain_dir` (one `.npy` per
            fit/val split, keyed on row count), so a crash after this step but
            before the pretrain checkpoint is written does not repeat it on rerun.
        **chemprop_kwargs: Forwarded to `ChempropMultitargetModel`.

    Returns:
        Long OOF frame over the scored endpoints, with a `y_unc` column added when
        `uncertainty=True`.
    """
    if uncertainty and chemprop_kwargs.get("uncertainty_method") is None:
        raise ValueError("uncertainty=True needs an uncertainty_method in chemprop_kwargs")
    from .graph_models import ChempropMultitargetModel

    if assignments is None:
        _, assignments = shared_scaffold_folds(frames, n_outer=n_outer, n_inner=n_inner, seed=seed)

    endpoints = list(frames)
    missing = [e for e in endpoints if e not in pretraining.columns]
    if missing:
        raise ValueError(f"pretraining frame is missing endpoint columns {missing}")

    # Heads to train, scored endpoints first. Chemprop restores output heads by
    # position when warm-starting from a checkpoint, so both stages must agree on
    # this list exactly -- a 16-head pretrain cannot warm-start a 4-head fine-tune.
    targets = list(pretrain_targets) if pretrain_targets else list(endpoints)
    if targets[: len(endpoints)] != endpoints:
        raise ValueError(
            "pretrain_targets must begin with the scored endpoints in `frames` order: "
            f"got {targets[: len(endpoints)]} against {endpoints}"
        )
    absent = [t for t in targets if t not in pretraining.columns]
    if absent:
        raise ValueError(f"pretraining frame is missing target columns {absent}")

    descriptor_columns = chemprop_kwargs.get("descriptor_columns")
    if descriptor_columns and finetune_descriptors is None:
        raise ValueError(
            "chemprop_kwargs sets descriptor_columns but no finetune_descriptors "
            "frame was given -- the fine-tuning model has an FFN sized for these "
            "columns and cannot be fit or predicted from without them"
        )
    if finetune_descriptors is not None and not descriptor_columns:
        raise ValueError(
            "finetune_descriptors was given but chemprop_kwargs has no "
            "descriptor_columns -- name the columns so ChempropMultitargetModel "
            "knows to use them"
        )

    wide = _wide_frame(frames)
    if finetune_descriptors is not None:
        missing_names = set(wide["Molecule_Name"]) - set(finetune_descriptors["Molecule_Name"])
        if missing_names:
            raise ValueError(
                f"finetune_descriptors is missing {len(missing_names)} of "
                f"{wide.height} fine-tuning compounds, e.g. {sorted(missing_names)[:5]}"
            )
        wide = wide.join(
            finetune_descriptors.select(["Molecule_Name", *descriptor_columns]),
            on="Molecule_Name",
            how="left",
        )
    if finetune_targets is not None:
        extra = [c for c in finetune_targets.columns if c not in ("Molecule_Name", "SMILES")]
        wide = wide.join(
            finetune_targets.select(["Molecule_Name", *extra]),
            on="Molecule_Name",
            how="left",
        )
        targets = targets + [c for c in extra if c not in targets]

    # Pad each stage with the other's columns. A head the other stage measures is
    # simply unmeasured here, which is a null like any other unmeasured endpoint and
    # is masked in chemprop's loss rather than read as a real value of zero.
    pretraining = pretraining.with_columns(
        [pl.lit(None, dtype=pl.Float64).alias(t) for t in targets if t not in pretraining.columns]
    )
    wide = wide.with_columns(
        [pl.lit(None, dtype=pl.Float64).alias(t) for t in targets if t not in wide.columns]
    )

    stacked, _ = endpoint_indicator_frame(frames)
    n_folds = assignments["fold"].n_unique()

    # One pretrain for the whole run. A dedicated directory keeps this checkpoint
    # away from the default one, so an unpretrained control arm in the same session
    # cannot accidentally warm-start from it -- silent contamination that would make
    # the control look as good as the treatment and hide a real effect.
    pretrain_dir = C.PROJECT_ROOT / ".chemprop_pretrain" / method_name
    template = ChempropMultitargetModel(
        targets=targets,
        pretrain_dir=pretrain_dir,
        pretrain_epochs=pretrain_epochs,
        **chemprop_kwargs,
    )
    if not (pretrain_dir / "model_0" / "best.pt").exists():
        rng = np.random.default_rng(seed)
        order = rng.permutation(pretraining.height)
        n_val = max(1, int(p_val * pretraining.height))
        pre_val, pre_fit = pretraining[order[:n_val]], pretraining[order[n_val:]]

        pre_fit_descriptors = pre_val_descriptors = None
        if descriptor_columns:
            # The pretraining corpus has no pharmacophore descriptors of its own --
            # they are computed here, once, purely so the pretrain checkpoint's FFN
            # has the same input width the fine-tuning model will need. See
            # `ChempropMultitargetModel.pretrain_wide` for why a width mismatch
            # between the two stages fails the `--checkpoint` load outright.
            #
            # n_jobs=-1: a public pretraining corpus can run to tens of thousands
            # of compounds (measured: 12,719 for the PubChem matrix), and
            # single-threaded 3D descriptor computation on a corpus that size took
            # over an hour with no progress signal in one run -- public-library
            # molecules average larger and more flexible than the challenge's own
            # curated set, so cost does not extrapolate linearly from a small
            # sample. The `on_progress` print is deliberate, not decorative, for
            # the same reason CLAUDE.md's `on_fold` convention exists: a
            # multi-minute step with nothing printed is indistinguishable from a
            # hang, which is exactly what happened here before this was added.
            # Cached to `pretrain_dir` itself rather than a separate cache tree --
            # this descriptor matrix has no purpose beyond warm-starting exactly
            # this method's pretrain, so it belongs next to the checkpoint it
            # feeds, and lives no longer than the checkpoint does. Without this, a
            # crash between finishing the descriptors and `best.pt` being written
            # (the `pretrain_wide` call below can itself run long) throws away
            # every minute of the step above, which measured over an hour
            # single-threaded and ~40 minutes at n_jobs=-1 on the PubChem corpus.
            # The row count is stamped into the filename as a cheap staleness
            # check: `pretraining`/`seed`/`p_val` changing shifts `pre_fit`'s
            # size, so a stale cache from a different corpus or split is a miss
            # rather than a silent wrong-shape load.
            from . import pharmacophore

            def _log_progress(label: str):
                # `done` advances by 1 per compound (joblib streams results as
                # each finishes, not in fixed-size batches), so gate the print on
                # a window rather than an exact multiple -- a step size that
                # doesn't evenly divide `total` would otherwise never print at all.
                def _callback(done: int, total: int) -> None:
                    if done == total or done % (max(1, total // 10)) < 100:
                        print(f"  pharmacophore descriptors ({label}): {done}/{total}")

                return _callback

            pretrain_dir.mkdir(parents=True, exist_ok=True)
            _fit_cache = pretrain_dir / f"pretrain_descriptors_fit_{pre_fit.height}.npy"
            if _fit_cache.exists():
                pre_fit_descriptors = np.load(_fit_cache)
            else:
                pre_fit_descriptors = pharmacophore.matrix(
                    pharmacophore.descriptors(
                        pre_fit["smiles_std"].to_list(),
                        n_jobs=-1,
                        timeout_s=descriptor_timeout_s,
                        on_progress=_log_progress("pretrain fit"),
                    ),
                    descriptor_columns,
                )
                np.save(_fit_cache, pre_fit_descriptors)

            _val_cache = pretrain_dir / f"pretrain_descriptors_val_{pre_val.height}.npy"
            if _val_cache.exists():
                pre_val_descriptors = np.load(_val_cache)
            else:
                pre_val_descriptors = pharmacophore.matrix(
                    pharmacophore.descriptors(
                        pre_val["smiles_std"].to_list(),
                        n_jobs=-1,
                        timeout_s=descriptor_timeout_s,
                        on_progress=_log_progress("pretrain val"),
                    ),
                    descriptor_columns,
                )
                np.save(_val_cache, pre_val_descriptors)

        template.pretrain_wide(
            pre_fit["smiles_std"].to_list(),
            pre_fit.select(targets).to_numpy(),
            pre_val["smiles_std"].to_list(),
            pre_val.select(targets).to_numpy(),
            descriptors_train=pre_fit_descriptors,
            descriptors_val=pre_val_descriptors,
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
            targets=targets,
            pretrain_dir=pretrain_dir,
            freeze_encoder=freeze_encoder,
            **chemprop_kwargs,
        )
        fit_descriptors = (
            fit_wide.select(descriptor_columns).to_numpy() if descriptor_columns else None
        )
        val_descriptors = (
            val_wide.select(descriptor_columns).to_numpy() if descriptor_columns else None
        )
        model.fit(
            fit_wide["SMILES"].to_list(),
            fit_wide.select(targets).to_numpy(),
            smiles_val=val_wide["SMILES"].to_list(),
            y_val=val_wide.select(targets).to_numpy(),
            descriptors=fit_descriptors,
            descriptors_val=val_descriptors,
        )

        test_names = set(test_long["Molecule_Name"].to_list())
        test_wide = wide.filter(pl.col("Molecule_Name").is_in(test_names)).sort("Molecule_Name")
        test_descriptors = (
            test_wide.select(descriptor_columns).to_numpy() if descriptor_columns else None
        )

        if uncertainty:
            predictions, unc = model.predict(
                test_wide["SMILES"].to_list(),
                return_uncertainty=True,
                descriptors=test_descriptors,
            )
            y_pred = _predictions_for(test_long, test_wide, predictions, endpoints)
            y_unc = _predictions_for(test_long, test_wide, unc, endpoints)
            fold_records = _oof_records(test_long, y_pred, method_name, fold, outer, inner)
            for record, unc_value in zip(fold_records, y_unc, strict=True):
                record["y_unc"] = float(unc_value)
            records.extend(fold_records)
        else:
            predictions = model.predict(test_wide["SMILES"].to_list(), descriptors=test_descriptors)
            y_pred = _predictions_for(test_long, test_wide, predictions, endpoints)
            records.extend(_oof_records(test_long, y_pred, method_name, fold, outer, inner))
        report_fold(on_fold, fold, n_folds)

    return pl.DataFrame(records)


def run_cv_ensemble_weighting(
    frames: dict[str, pl.DataFrame],
    pretraining: pl.DataFrame,
    ensemble_size: int = 4,
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
    """Compare uniform vs inverse-variance-weighted ensemble averaging, paired.

    A Chemprop ensemble (`uncertainty_method="ensemble"`) is `ensemble_size`
    checkpoints fit on the same data with different initialisations; chemprop's own
    CLI averages them uniformly and reports their variance separately. This fits
    that same ensemble once per fold and reads out each checkpoint's own prediction
    (`graph_models.ChempropMultitargetModel.predict_ensemble_members`), so a uniform
    average and an inverse-variance-weighted average can be built from *the same
    fit* rather than two separate CV runs -- which is what makes them properly
    paired for `evaluation.paired_bootstrap` (identical folds is not enough; the
    exact same checkpoints removes fit-to-fit noise as an explanation for a
    difference).

    Per-compound inverse-variance weights use each checkpoint's squared deviation
    from the ensemble mean as its variance proxy, floored to avoid a divide-by-zero
    when checkpoints agree exactly. A checkpoint that consistently disagrees with
    its peers on a compound is downweighted there specifically, rather than
    discarded globally -- this is what lets one checkpoint be trustworthy on some
    compounds and not others, which a global model-selection weight cannot express.

    Args:
        frames: Scored endpoint name -> training frame.
        pretraining: Wide frame from `public_pretraining_matrix`.
        ensemble_size: Number of checkpoints Chemprop trains per fold.
        pretrain_epochs: Epochs for the single pretraining run (shared with any
            other `run_cv_pretrained` call using the same `method_name="pubchem"`
            checkpoint directory convention -- pass a `pretrain_dir` via
            `chemprop_kwargs` to point at an existing checkpoint rather than
            retraining one).
        n_outer, n_inner, seed, p_val, assignments, folds, on_fold: As
            `run_cv_pretrained`.
        **chemprop_kwargs: Forwarded to `ChempropMultitargetModel`, minus
            `uncertainty_method`/`ensemble_size`, which this function sets itself.

    Returns:
        Long OOF frame with `method` set to `"ensemble_uniform"` or
        `"ensemble_ivw"` for every row, on identical (fold, compound, endpoint)
        keys -- concatenate and pivot for a paired comparison.
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

    pretrain_dir = chemprop_kwargs.pop("pretrain_dir", None)
    if pretrain_dir is None:
        pretrain_dir = C.PROJECT_ROOT / ".chemprop_pretrain" / "ensemble_weighting"
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
            uncertainty_method="ensemble",
            ensemble_size=ensemble_size,
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
        # (n_compounds, n_targets, ensemble_size) -- every checkpoint's own prediction.
        members = model.predict_ensemble_members(test_wide["SMILES"].to_list())

        uniform = members.mean(axis=-1)
        member_mean = members.mean(axis=-1, keepdims=True)
        member_var = np.clip((members - member_mean) ** 2, 1e-6, None)
        weights = 1.0 / member_var
        weights /= weights.sum(axis=-1, keepdims=True)
        weighted = (members * weights).sum(axis=-1)

        for method_name, predictions in (("ensemble_uniform", uniform), ("ensemble_ivw", weighted)):
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
