import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # 11 — structure-based 3D: docking, Uni-Mol poses and interaction fingerprints

    `09_cyp2d6_representation.py` closed the *ligand-side* 3D question: explicit
    pharmacophore geometry made every endpoint worse when folded into the pretrained
    encoder, and no 3D-only arm beat plain ECFP. Its conclusion was that CYP2D6's gap
    is a chemical-space problem rather than a representation-class one.

    That result says nothing about the **protein**. Every representation this repo has
    tried — ECFP, Mordred, CheMeleon, Uni-Mol-style conformers, pharmacophore
    descriptors — describes a molecule in isolation. A CYP inhibitor's potency is a
    property of a *complex*. `PLAN.md` has carried docking as "promising but a
    multi-week project" since day one, and PXR's post-mortem named 3D/structural
    information as one of the two things that cost the most.

    ## Arms

    All five share folds and compound sets, so every pair is legitimately paired.

    | arm | what it encodes | what a win would mean |
    |:--|:--|:--|
    | `unimol_etkdg` | Uni-Mol CLS, free ETKDG conformer | Uni-Mol is a good *ligand* featurizer |
    | `unimol_docked` | Uni-Mol CLS on the docked pose | — (only vs `unimol_etkdg`) |
    | `docking_score` | smina affinity, 1 column | classical docking alone carries signal |
    | `plif` | ProLIF protein contacts, named residue/interaction bits | **the protein matters** |
    | `pubchem` | 05's incumbent | the control everything is measured against |

    ## The control that makes this readable

    `unimol_docked` cannot be interpreted on its own, and this was measured before
    building anything. Six known CYP ligands docked into CYP1A2, Uni-Mol CLS from the
    docked pose against the same molecule's free ETKDG conformer:

    | compound | cosine |
    |:--|--:|
    | caffeine | 1.0000 |
    | paracetamol | 0.9998 |
    | warfarin | 0.9717 |
    | midazolam | 0.9519 |
    | ketoconazole fragment | 0.8268 |
    | quinidine | 0.8046 |
    | *between different molecules* | *0.725* |

    Uni-Mol was pretrained on isolated-ligand conformers and has never seen a protein.
    For rigid compounds the docked vector **is** the free vector; even flexible ones
    only approach the between-molecule floor. So `unimol_docked` beating `pubchem`
    would be evidence for Uni-Mol as a 3D ligand featurizer, not for docking. Only
    `unimol_docked` − `unimol_etkdg` supports a docking claim.

    This is 09's trap from the other direction: there, `ecfp+pharmacophore_3d` lifted
    every endpoint while lifting CYP2D6 *least*, which is how we learned the lift was
    a better featurizer rather than the hypothesised mechanism.

    ## Why the PLIF arm is the interesting one

    A PLIF has no such ambiguity by construction: every bit *is* a protein contact,
    and a molecule without a protein has no fingerprint. It also re-asks 09's question
    in the only form left. 09 asked "does the ligand have the right geometry" and
    answered no. A PLIF asks "does it make the contact".

    On the CYP2D6 receptor (5TFT chain A) ProLIF recovers **Glu216 and Asp301** — the
    acidic anchors of the literature CYP2D6 pharmacophore that 09's descriptors failed
    to express — alongside the Phe cluster (Phe120, Phe112, Phe483) lining the site.

    ## What could not be used

    **gnina was the first choice and cannot run here.** It is CUDA-only: the sole
    v1.3.3 release asset is a 1.96 GB CUDA-12.8 Linux binary and every documented build
    path requires `nvcc`, against an arm64 Mac with no NVIDIA GPU. **smina** is the Vina
    fork gnina derives from, and supplies the same autobox and scoring interface. What
    is lost is gnina's learned CNN pose score — which is what `plif` and Uni-Mol are
    measuring by other means anyway.
    """
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import sys
    import time
    from pathlib import Path

    import numpy as np
    import polars as pl
    from scipy import stats

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    # `import cyp` first: it claims the OpenMP slot for LightGBM. See CLAUDE.md --
    # this notebook also drives torch (Uni-Mol) and every such call goes through a
    # subprocess for exactly that reason.
    import cyp  # noqa: F401
    from cyp import (
        aux_training,
        cv,
        data,
        docking,
        evaluation,
        external,
        metrics,
        models,
        plif,
        timings,
        unimol,
    )
    from cyp import constants as C

    return (
        C,
        PROJECT_ROOT,
        Path,
        aux_training,
        cv,
        data,
        docking,
        evaluation,
        external,
        metrics,
        models,
        np,
        pl,
        plif,
        stats,
        time,
        timings,
        unimol,
    )


@app.cell
def _(Path, PROJECT_ROOT):
    NOTEBOOK_NAME = Path(__file__).stem
    OUT_DIR = PROJECT_ROOT / "experiments" / NOTEBOOK_NAME
    CACHE_DIR = OUT_DIR / "cache"
    POSE_DIR = OUT_DIR / "poses"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    POSE_DIR.mkdir(parents=True, exist_ok=True)
    TIMINGS = OUT_DIR / "timings.csv"
    return CACHE_DIR, NOTEBOOK_NAME, OUT_DIR, POSE_DIR, TIMINGS


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Settings

    **Exhaustiveness 8, smina's default.** A 4-vs-8 probe on 20 CYP2D6 compounds
    initially suggested 4 was sufficient (mean |Δaffinity| = 0.031 kcal/mol, Pearson
    0.9976), but the per-compound timings behind that probe ranged 4.3-101s across
    repeated runs of identical work -- machine contention (`biomesyncd` at 83% CPU
    during one measurement), not a real exhaustiveness cost, the same failure mode
    `timings.flag_contention` exists to catch. With the timing evidence unreliable,
    kept the literature-default setting rather than trust a shortcut measured on a
    noisy machine. Docking is ~6,500 compound-receptor pairs across the four
    endpoints, so this is the setting that decides whether the run fits in a night.

    **Top pose only** (`num_modes=1`). Keeping several would multiply the
    featurisation work without a principled way to choose among them; pose selection
    is a real question but a separate one, and mixing it in would confound the arm
    comparison.
    """
    )
    return


@app.cell
def _(C):
    SNAPSHOT = "20260917"  # structure snapshot; challenge data uses its own latest
    EXHAUSTIVENESS = 8
    N_OUTER, N_INNER = 5, 5
    SEED = 42
    # Halved from the machine's 8 performance cores after the fan ran continuously
    # at --cpu 8 during the first attempt -- roughly doubles wall time but keeps
    # thermal headroom for a run left unattended overnight.
    DOCK_CPU = 4
    # Molecules per smina call. See the markdown above: chunking exists so a
    # multi-hour docking stage gives real per-chunk timing signal early instead of
    # only at the end, and so a run that is stressing the machine can be stopped
    # between chunks rather than only diagnosed after the fact.
    CHUNK_SIZE = 100

    # Scoped to CYP2D6 alone for this run. CYP1A2's docking got roughly 3 hours in
    # (8 of 15 chunks, cached on disk under experiments/11_docking_3d/poses/) before
    # the true per-chunk cost -- median 941s versus an early lucky-case 390s, no
    # thermal cause found -- put a full four-isoform run at an honest 17-28h rather
    # than the ~7h first estimate. CYP2D6 is the endpoint most worth spending that
    # budget on first: weakest baseline (03/05), and the ProLIF probe already found
    # the literature Glu216/Asp301 anchors on its receptor specifically. Widen back
    # to `C.REGRESSION_ENDPOINTS` to resume the other three -- their partial docking
    # progress (CYP1A2) or full receptor prep (all four) is cached and reused, not
    # lost by narrowing here.
    ENDPOINTS = ["CYP2D6_pIC50_direct_inhibition"]
    ISOFORM_OF = {e: e.split("_")[0] for e in ENDPOINTS}
    return (
        CHUNK_SIZE,
        DOCK_CPU,
        ENDPOINTS,
        EXHAUSTIVENESS,
        ISOFORM_OF,
        N_INNER,
        N_OUTER,
        SEED,
        SNAPSHOT,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Receptors

    One holo structure per isoform, each chosen so its co-crystal ligand marks the
    catalytic site and can define the search box. Resolution is not the only criterion:
    CYP3A4's 1TQN is better resolved (2.05 Å) than 4NY4 (2.95 Å) but is effectively
    ligand-free, and 3A4's cavity is large and plastic enough that a guessed box would
    dock compounds into the wrong subsite. A well-placed box beats a sharp structure.

    The heme stays in the receptor. Stripping HETATM records wholesale is the reflex
    when cleaning a PDB and would remove it, leaving a cavity whose floor is missing so
    ligands dock a few angstrom too deep — poses, not an error.
    """
    )
    return


@app.cell
def _(SNAPSHOT, docking, mo, pl):
    if not docking.smina_available():
        raise RuntimeError(
            "smina not on PATH. Install with `brew install brewsci/bio/smina`. "
            "gnina is not an option on this machine -- it is CUDA-only."
        )

    RECEPTOR_FILES = {}
    _rows = []
    for _isoform, _receptor in docking.RECEPTORS.items():
        _rec, _box, _rec_h = docking.prepare_receptor(_receptor, SNAPSHOT)
        RECEPTOR_FILES[_isoform] = (_rec, _box, _rec_h)
        _rows.append(
            {
                "isoform": _isoform,
                "pdb": _receptor.pdb_id,
                "ligand": _receptor.ligand_code,
                "chain": _receptor.chain,
                "resolution": _receptor.resolution,
                "note": _receptor.note,
            }
        )

    receptor_table = pl.DataFrame(_rows)
    mo.ui.table(receptor_table, selection=None)
    return RECEPTOR_FILES, receptor_table


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Dock every endpoint's compounds into its own receptor

    **Docked in chunks of 100 molecules**, not as one call over the full ~1,400-2,300
    compound set. A single smina call gives no signal until it finishes, and the
    first full-isoform attempt at exhaustiveness 8 gave no way to tell real cost from
    machine contention — this repo's timings for the identical 20-compound probe
    ranged 4.3-101s/compound across runs, purely from contention (CLAUDE.md). 100
    molecules finishes in single-digit minutes even in a bad case, which is short
    enough to check system state between chunks without losing much wall time to
    per-chunk fixed overhead (receptor load, box computation), and short enough to
    interrupt a run that is overheating the machine rather than only be able to
    diagnose it after the fact.

    Fold-major caching does not apply here — docking is not a CV step — but the same
    failure mode does, so poses are written once (now per-chunk) and reused. A killed
    run resumes at the chunk it reached, not the isoform.

    Compounds that fail to embed are dropped from the SDF and reported. They are *not*
    silently written as 2D: smina would accept a flat molecule and return a pose from
    it, which is a correctness failure that looks like a result.
    """
    )
    return


@app.cell
def _(
    CHUNK_SIZE,
    DOCK_CPU,
    ENDPOINTS,
    EXHAUSTIVENESS,
    ISOFORM_OF,
    POSE_DIR,
    RECEPTOR_FILES,
    TIMINGS,
    data,
    docking,
    mo,
    pl,
    timings,
):
    FRAMES = {e: data.training_frame(e) for e in ENDPOINTS}

    POSES = {}
    DOCKED_NAMES = {}
    _rows = []
    for _endpoint in mo.status.progress_bar(ENDPOINTS, title="docking"):
        _isoform = ISOFORM_OF[_endpoint]
        _frame = FRAMES[_endpoint]
        _rec, _box, _rec_h = RECEPTOR_FILES[_isoform]

        _lig_sdf = POSE_DIR / f"{_isoform}_ligands.sdf"
        _pose_sdf = POSE_DIR / f"{_isoform}_poses.sdf"
        _names_path = POSE_DIR / f"{_isoform}_names.json"

        if _pose_sdf.exists() and _names_path.exists():
            import json

            _embedded = json.loads(_names_path.read_text())
        else:
            import json

            _, _embedded = docking.ligand_sdf(
                _frame["SMILES"].to_list(), _frame["Molecule_Name"].to_list(), _lig_sdf
            )
            _names_path.write_text(json.dumps(_embedded))

            def _on_chunk(i, n_chunks, seconds, _pose_chunk, _isoform=_isoform):
                # Real per-chunk cost, recorded as it happens rather than only at
                # the end -- exactly what a single full-isoform smina call could
                # not give us. A cached chunk (seconds == 0.0 from dock_in_chunks)
                # is not written, so the timing log stays a record of genuine work.
                if seconds > 0:
                    timings.record(
                        TIMINGS, "docking", "smina", f"{_isoform}_chunk{i}", "full", seconds
                    )
                print(
                    f"  {_isoform} chunk {i + 1}/{n_chunks}: {seconds:.1f}s",
                    flush=True,
                )

            docking.dock_in_chunks(
                _rec,
                _box,
                _lig_sdf,
                _pose_sdf,
                chunk_size=CHUNK_SIZE,
                exhaustiveness=EXHAUSTIVENESS,
                cpu=DOCK_CPU,
                on_chunk=_on_chunk,
            )

        POSES[_endpoint] = _pose_sdf
        DOCKED_NAMES[_endpoint] = _embedded
        _scores = docking.score_frame(_pose_sdf)
        # Total isoform time is the sum of its chunk rows, not a single scalar --
        # `dock_in_chunks` times per-chunk, and a cached run (poses already on disk)
        # writes no chunk rows at all, which is the correct "None" for a rerun.
        _chunk_log = timings.load(TIMINGS) if TIMINGS.exists() else None
        _total_seconds = None
        if _chunk_log is not None and _chunk_log.height:
            _matched = _chunk_log.filter(
                pl.col("stage") == "docking",
                pl.col("endpoint").str.starts_with(f"{_isoform}_chunk"),
            )
            if _matched.height:
                _total_seconds = round(float(_matched["seconds"].sum()), 1)
        _rows.append(
            {
                "endpoint": _isoform,
                "compounds": _frame.height,
                "embedded": len(_embedded),
                "posed": _scores.height,
                "mean_affinity": round(float(_scores["docking_affinity"].mean()), 3)
                if _scores.height
                else None,
                "seconds": _total_seconds,
            }
        )

    docking_summary = pl.DataFrame(_rows)
    mo.ui.table(docking_summary, selection=None)
    return DOCKED_NAMES, FRAMES, POSES, docking_summary


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Interaction fingerprints

    `VdWContact` is excluded from the interaction set. It fires on nearly every
    residue within range, which turns the fingerprint into a proximity map that cannot
    tell a hydrogen bond from an incidental contact.

    Bits are pruned to those set in 1–99% of compounds. A constant bit carries nothing;
    a bit set in one compound out of 1,500 is closer to an identifier than a feature,
    and under scaffold CV those rare bits concentrate in single folds where a tree can
    memorise them.
    """
    )
    return


@app.cell
def _(
    CACHE_DIR,
    ENDPOINTS,
    ISOFORM_OF,
    POSES,
    RECEPTOR_FILES,
    TIMINGS,
    mo,
    pl,
    plif,
    time,
    timings,
):
    PLIF_FRAMES = {}
    _rows = []
    for _endpoint in mo.status.progress_bar(ENDPOINTS, title="PLIF"):
        _isoform = ISOFORM_OF[_endpoint]
        _cache = CACHE_DIR / f"plif_{_isoform}.parquet"
        if _cache.exists():
            _frame = pl.read_parquet(_cache)
        else:
            _t0 = time.time()
            _frame = plif.compute(POSES[_endpoint], RECEPTOR_FILES[_isoform][2])
            timings.record(TIMINGS, "plif", "prolif", _isoform, "full", time.time() - _t0)
            _frame.write_parquet(_cache)
        _pruned = plif.prune(_frame)
        PLIF_FRAMES[_endpoint] = _pruned
        _rows.append(
            {
                "endpoint": _isoform,
                "raw_bits": _frame.width - 1,
                "kept_bits": _pruned.width - 1,
                "compounds": _frame.height,
            }
        )

    plif_summary = pl.DataFrame(_rows)
    mo.ui.table(plif_summary, selection=None)
    return PLIF_FRAMES, plif_summary


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Which contacts does the CYP2D6 site make?

    The interpretable half. If the PLIF arm helps, these rows say which residues did
    the work — and whether Glu216/Asp301, the anchors 09's ligand-side descriptors
    could not express, are among them.
    """
    )
    return


@app.cell
def _(CACHE_DIR, PLIF_FRAMES, mo, pl, plif):
    _cyp2d6 = [e for e in PLIF_FRAMES if e.startswith("CYP2D6")][0]
    cyp2d6_contacts = plif.summarise(PLIF_FRAMES[_cyp2d6]).head(20)
    cyp2d6_contacts.write_csv(CACHE_DIR.parent / "cyp2d6_contacts.csv")

    _acidic = cyp2d6_contacts.filter(
        pl.col("residue").str.contains("GLU216") | pl.col("residue").str.contains("ASP301")
    )
    mo.md(
        f"Top contacts, CYP2D6. Acidic-anchor bits in the top 20: "
        f"**{_acidic.height}**\n\n{mo.ui.table(cyp2d6_contacts, selection=None)}"
    )
    return (cyp2d6_contacts,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Uni-Mol embeddings — both arms

    Every call runs in a subprocess. Two independent reasons, either sufficient: torch
    in-process with LightGBM segfaults on this machine (CLAUDE.md), and `unimol_tools`
    forks worker processes internally, so a caller without a `__main__` guard
    fork-bombs — many processes at ~100% CPU, with the real error buried. The second
    looks like the first but runs hot rather than sitting at 0% in `kmp_flag_64::wait`.
    """
    )
    return


@app.cell
def _(
    CACHE_DIR,
    DOCKED_NAMES,
    ENDPOINTS,
    FRAMES,
    ISOFORM_OF,
    POSES,
    TIMINGS,
    docking,
    mo,
    np,
    time,
    timings,
    unimol,
):
    UNIMOL_ETKDG = {}
    UNIMOL_DOCKED = {}
    for _endpoint in mo.status.progress_bar(ENDPOINTS, title="Uni-Mol"):
        _isoform = ISOFORM_OF[_endpoint]
        _frame = FRAMES[_endpoint]
        _names = _frame["Molecule_Name"].to_list()

        _etkdg_cache = CACHE_DIR / f"unimol_etkdg_{_isoform}.npy"
        if _etkdg_cache.exists():
            UNIMOL_ETKDG[_endpoint] = np.load(_etkdg_cache)
        else:
            _t0 = time.time()
            _embeddings = unimol.embed_smiles(_frame["SMILES"].to_list())
            timings.record(TIMINGS, "unimol", "etkdg", _isoform, "full", time.time() - _t0)
            np.save(_etkdg_cache, _embeddings)
            UNIMOL_ETKDG[_endpoint] = _embeddings

        _docked_cache = CACHE_DIR / f"unimol_docked_{_isoform}.npy"
        if _docked_cache.exists():
            UNIMOL_DOCKED[_endpoint] = np.load(_docked_cache)
        else:
            _t0 = time.time()
            _pose_names, _atoms, _coords = docking.pose_coordinates(POSES[_endpoint])
            _raw = unimol.embed_poses(_atoms, _coords)
            # Aligned onto the label frame's order, zero-filling compounds that never
            # produced a pose -- every arm must span the same compounds for the paired
            # comparison below to be legitimate.
            _aligned = unimol.align(_raw, _pose_names, _names)
            timings.record(TIMINGS, "unimol", "docked", _isoform, "full", time.time() - _t0)
            np.save(_docked_cache, _aligned)
            UNIMOL_DOCKED[_endpoint] = _aligned
    return UNIMOL_DOCKED, UNIMOL_ETKDG


@app.cell
def _(mo):
    mo.md(
        r"""
    ### How far does docking actually move the Uni-Mol vector?

    The six-ligand probe repeated at full scale. This is the number that decides
    whether `unimol_docked` and `unimol_etkdg` are meaningfully different arms at all.
    """
    )
    return


@app.cell
def _(ENDPOINTS, ISOFORM_OF, OUT_DIR, UNIMOL_DOCKED, UNIMOL_ETKDG, mo, np, pl):
    _rows = []
    for _endpoint in ENDPOINTS:
        _a, _b = UNIMOL_DOCKED[_endpoint], UNIMOL_ETKDG[_endpoint]
        # Zero rows are compounds that never docked; they would read as cosine 0 and
        # make the pose look far more informative than it is.
        _mask = (np.linalg.norm(_a, axis=1) > 0) & (np.linalg.norm(_b, axis=1) > 0)
        _an = _a[_mask] / np.linalg.norm(_a[_mask], axis=1, keepdims=True)
        _bn = _b[_mask] / np.linalg.norm(_b[_mask], axis=1, keepdims=True)
        _paired = (_an * _bn).sum(1)
        # Between-molecule cosine, for scale: how much of the space does the
        # docked-vs-free difference actually cover?
        _idx = np.random.default_rng(0).choice(
            _an.shape[0], size=min(300, _an.shape[0]), replace=False
        )
        _off = _an[_idx] @ _an[_idx].T
        _iu = np.triu_indices(len(_idx), 1)
        _rows.append(
            {
                "endpoint": ISOFORM_OF[_endpoint],
                "n_paired": int(_mask.sum()),
                "median_cosine": round(float(np.median(_paired)), 4),
                "p10_cosine": round(float(np.percentile(_paired, 10)), 4),
                "between_molecule_median": round(float(np.median(_off[_iu])), 4),
            }
        )

    pose_shift = pl.DataFrame(_rows)
    pose_shift.write_csv(OUT_DIR / "pose_shift.csv")
    mo.ui.table(pose_shift, selection=None)
    return (pose_shift,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Feature matrices

    `docking_score` is one column — deliberately. It is the whole of what classical
    docking claims about a compound, so an elaborate pose encoding has to beat it to
    justify itself. It is paired with ECFP4 rather than run alone, because a
    single-column arm would be measuring the absence of a representation, not the
    presence of docking information.
    """
    )
    return


@app.cell
def _(
    CACHE_DIR,
    ENDPOINTS,
    FRAMES,
    ISOFORM_OF,
    PLIF_FRAMES,
    POSES,
    UNIMOL_DOCKED,
    UNIMOL_ETKDG,
    docking,
    np,
    pl,
    plif,
):
    import importlib

    _fingerprints = importlib.import_module("cyp.fingerprints")

    def ecfp_for(endpoint: str) -> np.ndarray:
        _isoform = ISOFORM_OF[endpoint]
        _path = CACHE_DIR / f"ecfp_{_isoform}.npy"
        if _path.exists():
            return np.load(_path)
        _matrix = _fingerprints.compute(FRAMES[endpoint]["SMILES"].to_list(), "ecfp", n_bits=2048)
        np.save(_path, _matrix)
        return _matrix

    FEATURES = {}
    for _endpoint in ENDPOINTS:
        _names = FRAMES[_endpoint]["Molecule_Name"].to_list()
        _ecfp = ecfp_for(_endpoint)

        _scores = docking.score_frame(POSES[_endpoint])
        _lookup = dict(
            zip(
                _scores["Molecule_Name"].to_list(),
                _scores["docking_affinity"].to_list(),
                strict=True,
            )
        )
        # Compounds without a pose get the mean affinity rather than 0.0: a 0 kcal/mol
        # score means "binds not at all", which is a strong and wrong claim, whereas
        # the mean is uninformative -- the honest encoding of "we do not know".
        _observed = np.array([_lookup.get(n, np.nan) for n in _names], dtype=np.float64)
        _filled = np.where(np.isnan(_observed), np.nanmean(_observed), _observed)

        FEATURES[_endpoint] = {
            "unimol_etkdg": UNIMOL_ETKDG[_endpoint],
            "unimol_docked": UNIMOL_DOCKED[_endpoint],
            "docking_score": np.hstack([_ecfp, _filled.reshape(-1, 1)]),
            "plif": np.hstack([_ecfp, plif.matrix(PLIF_FRAMES[_endpoint], _names)]),
            "ecfp": _ecfp,
        }
    return FEATURES, ecfp_for


@app.cell
def _(mo):
    mo.md(
        r"""
    ## CV — LightGBM first

    Trees before chemprop, deliberately. LightGBM on 6,500 compound-endpoint pairs is
    minutes; chemprop is ~90 s per fold, so a 5×5 on four endpoints is hours. `03`
    established that on this data representation moves the numbers and model choice
    largely does not, so a representation that cannot beat its control under LightGBM
    is unlikely to be rescued by a graph model.

    Shared folds across arms (`cv.shared_scaffold_folds`) — required for the paired
    bootstrap, and the correctness issue from CLAUDE.md: 1,309 compounds carry more
    than one endpoint, so per-endpoint splits would leak across them.
    """
    )
    return


@app.cell
def _(
    CACHE_DIR,
    ENDPOINTS,
    FEATURES,
    FRAMES,
    ISOFORM_OF,
    N_INNER,
    N_OUTER,
    SEED,
    TIMINGS,
    cv,
    mo,
    models,
    pl,
    time,
    timings,
):
    ARMS = ("ecfp", "unimol_etkdg", "unimol_docked", "docking_score", "plif")

    _, FOLD_ASSIGNMENTS = cv.shared_scaffold_folds(
        FRAMES, n_outer=N_OUTER, n_inner=N_INNER, seed=SEED
    )

    _parts = []
    _total = len(ENDPOINTS) * len(ARMS)
    with mo.status.progress_bar(total=_total, title="LightGBM CV") as _bar:
        for _endpoint in ENDPOINTS:
            _isoform = ISOFORM_OF[_endpoint]
            for _arm in ARMS:
                _cache = CACHE_DIR / f"oof_lgbm_{_arm}_{_isoform}.parquet"
                if _cache.exists():
                    _parts.append(pl.read_parquet(_cache))
                    _bar.update()
                    continue
                _t0 = time.time()
                _oof = models.run_cv(
                    FRAMES[_endpoint],
                    _endpoint,
                    methods=("lgbm",),
                    features=FEATURES[_endpoint][_arm],
                    n_outer=N_OUTER,
                    n_inner=N_INNER,
                    seed=SEED,
                )
                _oof = _oof.with_columns(pl.lit(_arm).alias("arm"))
                timings.record(TIMINGS, "cv_lgbm", _arm, _isoform, "full", time.time() - _t0)
                _oof.write_parquet(_cache)
                _parts.append(_oof)
                _bar.update()

    oof_lgbm = pl.concat(_parts, how="diagonal_relaxed")
    oof_lgbm.write_parquet(CACHE_DIR.parent / "oof_lgbm.parquet")
    oof_lgbm.head()
    return ARMS, FOLD_ASSIGNMENTS, oof_lgbm


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Per-endpoint ST-RAE by arm

    Scored in ST-RAE, the challenge metric, rather than Spearman: 09 used Spearman
    because it was asking about ordering within a hypothesis test, whereas this
    notebook is asking what to ship.
    """
    )
    return


@app.cell
def _(ARMS, ENDPOINTS, ISOFORM_OF, OUT_DIR, metrics, oof_lgbm, mo, np, pl):
    _metrics = metrics

    _rows = []
    for _endpoint in ENDPOINTS:
        _row = {"endpoint": ISOFORM_OF[_endpoint]}
        for _arm in ARMS:
            _sub = oof_lgbm.filter((pl.col("endpoint") == _endpoint) & (pl.col("arm") == _arm))
            _per_fold = []
            for _fold in _sub["fold"].unique().to_list():
                _f = _sub.filter(pl.col("fold") == _fold)
                _per_fold.append(
                    _metrics.st_rae(
                        _f["y_true"].to_numpy(),
                        _f["y_pred"].to_numpy(),
                        _f["y_lower"].to_numpy(),
                        _f["y_upper"].to_numpy(),
                    )
                )
            _row[_arm] = round(float(np.mean(_per_fold)), 4)
        _rows.append(_row)

    strae_table = pl.DataFrame(_rows)
    _macro = {"endpoint": "macro"}
    for _arm in ARMS:
        _macro[_arm] = round(float(np.mean(strae_table[_arm].to_numpy())), 4)
    strae_table = pl.concat([strae_table, pl.DataFrame([_macro])], how="vertical")
    strae_table.write_csv(OUT_DIR / "strae_by_arm.csv")
    mo.ui.table(strae_table, selection=None)
    return (strae_table,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ### The two comparisons that carry the result

    1. **`unimol_docked` vs `unimol_etkdg`** — the only pair that isolates docking.
    2. **`plif` vs `ecfp`** — does adding protein contacts to the incumbent
       representation help, on the same compounds and folds?

    Paired bootstrap with Holm correction across arms, per CLAUDE.md's first PXR
    lesson: do not rank models on differences the data cannot resolve.
    """
    )
    return


@app.cell
def _(ENDPOINTS, ISOFORM_OF, OUT_DIR, evaluation, mo, oof_lgbm, pl):
    PAIRS = (
        ("unimol_docked", "unimol_etkdg"),
        ("plif", "ecfp"),
        ("docking_score", "ecfp"),
        ("unimol_etkdg", "ecfp"),
    )

    _rows = []
    _pvals = {}
    for _endpoint in ENDPOINTS:
        for _a, _b in PAIRS:
            _fa = oof_lgbm.filter((pl.col("endpoint") == _endpoint) & (pl.col("arm") == _a))
            _fb = oof_lgbm.filter((pl.col("endpoint") == _endpoint) & (pl.col("arm") == _b))
            # Join on (fold, Molecule_Name) rather than sorting each side
            # independently: two arms' OOF frames are not guaranteed to emit rows
            # in the same order, and sorting by y_true (a value, not a key) does
            # not recover row-for-row alignment when ties exist. A mismatched pair
            # would silently score the wrong compound against the wrong prediction.
            _paired = _fa.join(
                _fb.select("fold", "Molecule_Name", pl.col("y_pred").alias("y_pred_b")),
                on=["fold", "Molecule_Name"],
                how="inner",
            )
            _result = evaluation.paired_bootstrap(
                _paired["y_true"].to_numpy(),
                _paired["y_pred"].to_numpy(),
                _paired["y_pred_b"].to_numpy(),
                metric="st_rae",
                y_lower=_paired["y_lower"].to_numpy(),
                y_upper=_paired["y_upper"].to_numpy(),
            )
            _key = f"{ISOFORM_OF[_endpoint]}:{_a}-{_b}"
            _pvals[_key] = _result["p_value"]
            _rows.append(
                {
                    "endpoint": ISOFORM_OF[_endpoint],
                    "comparison": f"{_a} - {_b}",
                    "diff": round(_result["diff"], 4),
                    "ci_low": round(_result["ci_low"], 4),
                    "ci_high": round(_result["ci_high"], 4),
                    "p": round(_result["p_value"], 4),
                }
            )

    comparisons = pl.DataFrame(_rows)
    holm = evaluation.holm_bonferroni(_pvals)
    comparisons.write_csv(OUT_DIR / "paired_comparisons.csv")
    holm.write_csv(OUT_DIR / "holm.csv")
    mo.ui.table(comparisons, selection=None)
    return PAIRS, comparisons, holm


@app.cell
def _(holm, mo):
    mo.ui.table(holm, selection=None)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Chemprop -- molecule-level and atom-level, promoted arms only

    LGBM promoted `plif` and `docking_score` (both beat `ecfp` at Holm-significant
    p). This was structurally blocked at first: `aux_training.run_cv_pretrained`'s
    `descriptor_columns` path computed `pharmacophore.descriptors()` on the
    **pretraining corpus itself**, valid for `09`'s SMILES-derived geometry but not
    for PLIF/`docking_score`, which need a pose docked against the CYP2D6 receptor
    the public PubChem corpus never had. Confirmed directly: this run got 15.6 hours
    into docking, then died with `ColumnNotFoundError` after paying for a full
    ~1-hour pharmacophore computation the descriptor didn't need.

    **Fixed same day.** `pretrain_descriptors_from_smiles=False` zero-pads the
    pretrain-stage descriptor block instead of computing it, purely to keep the
    FFN's `--checkpoint` input width loadable -- the pretrain corpus never sees a
    real PLIF/docking value, only the fine-tuning stage does. Two molecule-level
    arms, both warm-started from a **fresh PubChem pretrain** (not `05`'s pubchem
    checkpoint -- that one has no descriptor columns at all, so its FFN width
    cannot accept these):

    - `pubchem_plif` -- PLIF bits as `--descriptors-columns`
    - `pubchem_docking_score` -- the single smina affinity column

    **Atom-level PLIF is a separate code path.** `--atom-features-path` attaches to
    chemprop's own atom nodes rather than the pooled molecule vector, and chemprop's
    CLI only accepts it against a single combined data file plus a `--splits-file`
    (confirmed against chemprop 2.2.1's source, not assumed) -- incompatible with
    `run_cv_pretrained`'s three-separate-CSV convention and with warm-starting from
    a pretrain checkpoint built without atom features. `pubchem_atom_plif` therefore
    trains from scratch, fold by fold, via `ChempropMultitargetModel` directly.

    All three are compared against `pubchem` (05's incumbent, reused OOF) and
    against each other on the same shared folds.
    """
    )
    return


@app.cell
def _(comparisons, mo, pl):
    # Which arms cleared the pre-registered promotion rule: beats its LGBM control
    # at p < 0.05 with a negative (better) ST-RAE difference.
    _promising = (
        comparisons.filter((pl.col("diff") < 0) & (pl.col("p") < 0.05))
        .get_column("comparison")
        .unique()
        .to_list()
    )
    CHEMPROP_ARMS = sorted({c.split(" - ")[0] for c in _promising})
    mo.md(f"Arms promoted to chemprop: **{CHEMPROP_ARMS or 'none'}**.")
    return (CHEMPROP_ARMS,)


@app.cell
def _(ENDPOINTS, aux_training, data, mo):
    _challenge_smiles = data.load_train_inhibition()["SMILES"].to_list()
    pretraining = aux_training.public_pretraining_matrix(
        ENDPOINTS, exclude_smiles=_challenge_smiles
    )
    mo.md(
        f"Pretraining corpus: **{pretraining.height:,} public compounds**, structural "
        "overlap with the challenge set removed."
    )
    return (pretraining,)


@app.cell
def _(
    CACHE_DIR,
    ENDPOINTS,
    FOLD_ASSIGNMENTS,
    FRAMES,
    N_OUTER,
    PLIF_FRAMES,
    POSES,
    TIMINGS,
    aux_training,
    docking,
    mo,
    pl,
    plif,
    pretraining,
    time,
    timings,
):
    # Molecule-level descriptor arms. Each fine-tuning compound needs a row keyed on
    # Molecule_Name -- built once here and reused by both arms and every fold.
    _endpoint = ENDPOINTS[0]
    _frame = FRAMES[_endpoint]
    _names = _frame["Molecule_Name"].to_list()

    _scores = docking.score_frame(POSES[_endpoint])
    _score_lookup = dict(
        zip(_scores["Molecule_Name"].to_list(), _scores["docking_affinity"].to_list(), strict=True)
    )
    import numpy as _np

    _observed = _np.array([_score_lookup.get(n, _np.nan) for n in _names], dtype=_np.float64)
    _filled_score = _np.where(_np.isnan(_observed), _np.nanmean(_observed), _observed)
    docking_score_descriptors = pl.DataFrame(
        {"Molecule_Name": _names, "docking_score_0": _filled_score}
    )

    _plif_matrix = plif.matrix(PLIF_FRAMES[_endpoint], _names)
    _plif_cols = [c for c in PLIF_FRAMES[_endpoint].columns if c != "Molecule_Name"]
    plif_descriptors = pl.DataFrame(
        {"Molecule_Name": _names} | {c: _plif_matrix[:, i] for i, c in enumerate(_plif_cols)}
    )

    MOLECULE_ARMS = {
        "pubchem_docking_score": {
            "descriptor_columns": ["docking_score_0"],
            "finetune_descriptors": docking_score_descriptors,
        },
        "pubchem_plif": {
            "descriptor_columns": _plif_cols,
            "finetune_descriptors": plif_descriptors,
        },
    }

    _n_folds = FOLD_ASSIGNMENTS["fold"].n_unique()
    _bar = mo.status.progress_bar(
        total=_n_folds * len(MOLECULE_ARMS), title="Chemprop molecule-level CV"
    )
    _parts = []
    with _bar as _pbar:
        for _fold in range(_n_folds):
            for _arm, _kwargs in MOLECULE_ARMS.items():
                _cache = CACHE_DIR / f"chemprop_{_arm}_fold{_fold}.parquet"
                if _cache.exists():
                    _parts.append(pl.read_parquet(_cache))
                    _pbar.update()
                    continue
                _t0 = time.time()
                _result = aux_training.run_cv_pretrained(
                    {_endpoint: _frame},
                    pretraining=pretraining,
                    method_name=_arm,
                    pretrain_epochs=30,
                    n_outer=N_OUTER,
                    pretrain_descriptors_from_smiles=False,
                    assignments=FOLD_ASSIGNMENTS,
                    folds=[_fold],
                    epochs=50,
                    **_kwargs,
                )
                timings.record(
                    TIMINGS, "chemprop_molecule", _arm, f"fold{_fold}", "full", time.time() - _t0
                )
                _result.write_parquet(_cache)
                _parts.append(_result)
                _pbar.update()

    oof_chemprop_molecule = pl.concat(_parts, how="diagonal_relaxed")
    oof_chemprop_molecule.write_parquet(CACHE_DIR.parent / "oof_chemprop_molecule.parquet")
    oof_chemprop_molecule.head()
    return MOLECULE_ARMS, oof_chemprop_molecule


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Atom-level PLIF

    Multi-hot per-heavy-atom interaction vectors, built once over every compound in
    the frame (a pose-less compound gets an all-zero block of the right heavy-atom
    count, same convention as the molecule-level arm). Trained from scratch --
    no PubChem warm-start available for this input shape yet -- so this is a fair
    comparison to a plain chemprop control on the same folds, not to `pubchem`
    directly; the `pubchem` row alongside it is context, not a paired test.
    """
    )
    return


@app.cell
def _(
    CACHE_DIR,
    ENDPOINTS,
    FOLD_ASSIGNMENTS,
    FRAMES,
    N_OUTER,
    POSES,
    RECEPTOR_FILES,
    TIMINGS,
    ISOFORM_OF,
    cv,
    mo,
    pl,
    plif,
    time,
    timings,
):
    from cyp.graph_models import ChempropMultitargetModel as _ChempropMTModel
    from cyp.multitask import _oof_records as _oof_records_fn

    _endpoint = ENDPOINTS[0]
    _isoform = ISOFORM_OF[_endpoint]
    _frame = FRAMES[_endpoint].with_columns(pl.lit(_endpoint).alias("endpoint"))

    _raw_atom_features = plif.atom_features(POSES[_endpoint], RECEPTOR_FILES[_isoform][2])
    _pruned_atom_features, ATOM_INTERACTIONS = plif.prune_atom_interactions(_raw_atom_features)
    _n_interactions = len(ATOM_INTERACTIONS)

    _n_folds = FOLD_ASSIGNMENTS["fold"].n_unique()
    _cache_root = CACHE_DIR / "atom_plif"
    _cache_root.mkdir(exist_ok=True)

    _parts = []
    with mo.status.progress_bar(total=_n_folds, title="Chemprop atom-level PLIF CV") as _bar:
        for _fold, _outer, _inner, _train, _val, _test in cv.fold_assignment_splits(
            _frame, FOLD_ASSIGNMENTS, p_val=0.1, seed=42
        ):
            _cache = _cache_root / f"fold{_fold}.parquet"
            if _cache.exists():
                _parts.append(pl.read_parquet(_cache))
                _bar.update()
                continue

            _smiles_train = _train["SMILES"].to_list()
            _smiles_val = _val["SMILES"].to_list()
            _smiles_test = _test["SMILES"].to_list()
            _names_train = _train["Molecule_Name"].to_list()
            _names_val = _val["Molecule_Name"].to_list()
            _names_test = _test["Molecule_Name"].to_list()

            _train_npz = plif.atom_features_npz(
                _pruned_atom_features,
                _smiles_train + _smiles_val,
                _names_train + _names_val,
                _n_interactions,
                _cache_root / f"train_fold{_fold}.npz",
            )
            _test_npz = plif.atom_features_npz(
                _pruned_atom_features,
                _smiles_test,
                _names_test,
                _n_interactions,
                _cache_root / f"test_fold{_fold}.npz",
            )

            _model = _ChempropMTModel(
                targets=[_endpoint],
                atom_feature_width=_n_interactions,
                model_dir=_cache_root / f"model_fold{_fold}",
                epochs=50,
            )
            _t0 = time.time()
            _model.fit(
                _smiles_train,
                _train["y_true"].to_numpy().reshape(-1, 1),
                smiles_val=_smiles_val,
                y_val=_val["y_true"].to_numpy().reshape(-1, 1),
                atom_features_path=_train_npz,
            )
            _y_pred = _model.predict(_smiles_test, atom_features_path=_test_npz)
            timings.record(
                TIMINGS,
                "chemprop_atom",
                "pubchem_atom_plif",
                f"fold{_fold}",
                "full",
                time.time() - _t0,
            )

            _fold_records = _oof_records_fn(
                _test, _y_pred.reshape(-1), "pubchem_atom_plif", _fold, _outer, _inner
            )
            _fold_result = pl.DataFrame(_fold_records)
            _fold_result.write_parquet(_cache)
            _parts.append(_fold_result)
            _bar.update()

    oof_chemprop_atom = pl.concat(_parts, how="diagonal_relaxed")
    oof_chemprop_atom.write_parquet(CACHE_DIR.parent / "oof_chemprop_atom.parquet")
    oof_chemprop_atom.head()
    return ATOM_INTERACTIONS, oof_chemprop_atom


@app.cell
def _(mo):
    mo.md(
        r"""
    ### `pubchem` incumbent, read from `05`'s committed OOF

    Not re-derived: `05_auxiliary_data.py` already fit and scored this arm, and
    refitting it here would spend the one thing this run cannot afford twice — a
    PubChem-pretrained chemprop fit — on a number already on disk. Read directly from
    `experiments/05_auxiliary_data/oof.parquet`.

    The comparison below is only trustworthy if `05`'s fold assignment is the *same*
    assignment used in this notebook — both use `shared_scaffold_folds(n_outer=5,
    n_inner=5, seed=42)`, but `05` built it over `augmented_frames` (frames widened
    with screen negatives) while this notebook builds it over the plain challenge
    frames, and a scaffold assignment is a function of which compounds are in the
    union. Rather than assume they agree, this checks it directly: for every shared
    `(fold, Molecule_Name)` pair, `y_true` must match. Any mismatch means the two
    runs scored different splits and the comparison is voided rather than silently
    reported.
    """
    )
    return


@app.cell
def _(ENDPOINTS, PROJECT_ROOT, mo, oof_lgbm, pl):
    _pubchem_path = PROJECT_ROOT / "experiments" / "05_auxiliary_data" / "oof.parquet"
    if not _pubchem_path.exists():
        pubchem_oof = None
        pubchem_alignment_ok = False
        _msg = f"**{_pubchem_path} not found** — pubchem comparison skipped."
    else:
        _raw = pl.read_parquet(_pubchem_path).filter(
            (pl.col("method") == "pubchem") & pl.col("endpoint").is_in(ENDPOINTS)
        )
        # The alignment check: join our OOF and 05's on the compounds and folds they
        # share, and require y_true to agree everywhere. y_true is unaffected by
        # anything about a model, so any disagreement is unambiguous evidence the two
        # runs used different fold assignments over different compound sets.
        _joined = oof_lgbm.filter(pl.col("arm") == "ecfp").join(
            _raw.select("fold", "Molecule_Name", "endpoint", pl.col("y_true").alias("y_true_05")),
            on=["fold", "Molecule_Name", "endpoint"],
            how="inner",
        )
        _mismatches = _joined.filter((pl.col("y_true") - pl.col("y_true_05")).abs() > 1e-6).height
        pubchem_alignment_ok = _joined.height > 0 and _mismatches == 0
        pubchem_oof = _raw
        _msg = (
            f"Shared (fold, compound) rows: **{_joined.height}**, mismatches: "
            f"**{_mismatches}**. Alignment {'OK' if pubchem_alignment_ok else 'FAILED'}"
            f"{' — comparison below is void.' if not pubchem_alignment_ok else '.'}"
        )
    mo.md(_msg)
    return pubchem_alignment_ok, pubchem_oof


@app.cell
def _(
    ENDPOINTS,
    ISOFORM_OF,
    OUT_DIR,
    metrics,
    mo,
    np,
    oof_chemprop_atom,
    oof_chemprop_molecule,
    pl,
    pubchem_alignment_ok,
    pubchem_oof,
    strae_table,
):
    # LGBM arms (already measured), the two molecule-level chemprop arms, the
    # from-scratch atom-level chemprop arm, and the pubchem incumbent -- all against
    # the same CYP2D6 compounds and folds. Cross-model-class rows (chemprop vs LGBM)
    # are context, not a paired statistical claim; only same-model-class comparisons
    # (LGBM arm vs LGBM arm, done above) carry a significance test.
    def _strae_by_endpoint(oof: pl.DataFrame, arm_name: str) -> list[dict]:
        rows = []
        for endpoint in ENDPOINTS:
            sub = oof.filter(pl.col("endpoint") == endpoint)
            if sub.height == 0:
                continue
            per_fold = [
                metrics.st_rae(
                    f["y_true"].to_numpy(),
                    f["y_pred"].to_numpy(),
                    f["y_lower"].to_numpy(),
                    f["y_upper"].to_numpy(),
                )
                for _fold, f in sub.group_by("fold")
            ]
            rows.append(
                {
                    "endpoint": ISOFORM_OF[endpoint],
                    "arm": arm_name,
                    "st_rae": round(float(np.mean(per_fold)), 4),
                }
            )
        return rows

    _rows = []
    for _arm in ("ecfp", "unimol_etkdg", "unimol_docked", "docking_score", "plif"):
        for _row in strae_table.filter(pl.col("endpoint") != "macro").iter_rows(named=True):
            _rows.append(
                {"endpoint": _row["endpoint"], "arm": f"lgbm_{_arm}", "st_rae": _row[_arm]}
            )

    for _method, _label in (
        ("pubchem_docking_score", "chemprop_pubchem_docking_score"),
        ("pubchem_plif", "chemprop_pubchem_plif"),
    ):
        _rows.extend(
            _strae_by_endpoint(oof_chemprop_molecule.filter(pl.col("method") == _method), _label)
        )
    _rows.extend(_strae_by_endpoint(oof_chemprop_atom, "chemprop_atom_plif"))

    if pubchem_oof is not None and pubchem_alignment_ok:
        _rows.extend(_strae_by_endpoint(pubchem_oof, "pubchem (05 chemprop, reused OOF)"))

    chemprop_summary = pl.DataFrame(_rows)
    chemprop_summary.write_csv(OUT_DIR / "chemprop_summary.csv")
    mo.ui.table(chemprop_summary.sort("st_rae"), selection=None)
    return (chemprop_summary,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Timings

    Written as each measurement is taken, so a run that dies partway still leaves the
    record of what finished — and so the cached rerun does not report an empty table.
    """
    )
    return


@app.cell
def _(TIMINGS, mo, timings):
    _summary = timings.summary(TIMINGS) if TIMINGS.exists() else None
    mo.ui.table(_summary, selection=None) if _summary is not None else mo.md("no timings yet")
    return


if __name__ == "__main__":
    app.run()
