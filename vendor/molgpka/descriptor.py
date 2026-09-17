"""Adapted from MolGpKa's `utils/descriptor.py` (Xundrug/MolGpKa, MIT).

Builds the 29-dim per-atom feature vector and bond graph `GCNNet` expects. Copied
near-verbatim from upstream's `get_atom_features`/`get_bond_pair` -- the only change
is returning plain tensors instead of a `torch_geometric.data.Data` object, since
`net.GCNNet` here takes `(x, edge_index, batch)` directly rather than a `Data`.

Feature order matters and is fixed by the trained weights: this must stay in exact
correspondence with what the original `mol2vec` produced, since a state_dict trained
under one ordering has no way to signal a silent reordering at load time.
"""

from __future__ import annotations

import torch
from rdkit import Chem
from rdkit.Chem import AllChem

_ATOM_SYMBOLS = ["C", "H", "O", "N", "S", "Cl", "F", "Br", "P", "I"]
_HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

# Upstream's donor/acceptor SMARTS, verbatim (Chem.MolFromSmarts is cached lazily
# on first use -- RDKit compiles these into pattern objects once per interpreter).
_DONOR_SMARTS = (
    "[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]",
    "[!$([#6,H0,-,-2,-3]),$([!H0;#7,#8,#9])]",
)
_ACCEPTOR_SMARTS = (
    "[!$([#1,#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]",
    "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),"
    "$([N;v3;!$(N-*=[O,N,P,S])]),n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]",
)


def _one_hot(value, allowed: list) -> list[bool]:
    if value not in allowed:
        value = allowed[-1]
    return [value == candidate for candidate in allowed]


def _substructure_atoms(mol: Chem.Mol, smarts_patterns: tuple[str, ...]) -> set[int]:
    atoms: set[int] = set()
    for smarts in smarts_patterns:
        pattern = Chem.MolFromSmarts(smarts)
        atoms.update(match[0] for match in mol.GetSubstructMatches(pattern))
    return atoms


def atom_features(mol: Chem.Mol, query_atom: int) -> list[list[float]]:
    """One 29-float feature row per atom, conditioned on `query_atom`.

    The last two features are what make this a per-query-atom rather than a
    per-molecule descriptor: every atom's topological distance to `query_atom`
    (0 for the query atom itself), and a one-hot flag marking which atom is being
    queried. Running the same molecule through this once per candidate ionization
    site, as `predict_pka` does, is how one graph-level model produces a distinct
    pKa per site.
    """
    AllChem.ComputeGasteigerCharges(mol)
    Chem.AssignStereochemistry(mol)
    donor_atoms = _substructure_atoms(mol, _DONOR_SMARTS)
    acceptor_atoms = _substructure_atoms(mol, _ACCEPTOR_SMARTS)
    ring = mol.GetRingInfo()

    rows: list[list[float]] = []
    for idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(idx)
        row: list[float] = []
        row += _one_hot(atom.GetSymbol(), _ATOM_SYMBOLS)
        row.append(atom.GetDegree())
        row += _one_hot(atom.GetHybridization(), _HYBRIDIZATIONS)
        row.append(atom.GetImplicitValence())
        row.append(atom.GetIsAromatic())
        row += [ring.IsAtomInRingOfSize(idx, size) for size in (3, 4, 5, 6, 7, 8)]
        row.append(idx in donor_atoms)
        row.append(idx in acceptor_atoms)
        row.append(atom.GetFormalCharge())
        if idx == query_atom:
            row.append(0)
        else:
            row.append(len(Chem.rdmolops.GetShortestPath(mol, idx, query_atom)))
        row.append(idx == query_atom)
        rows.append([float(v) for v in row])
    return rows


def bond_edge_index(mol: Chem.Mol) -> torch.Tensor:
    """`(2, 2*n_bonds)` directed edge index -- each bond contributes both directions,
    matching `net.GCNConvPT`'s expectation of a symmetric edge list (the original
    `MessagePassing` layer assumes undirected graphs represented this way)."""
    row: list[int] = []
    col: list[int] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [i, j]
        col += [j, i]
    return torch.tensor([row, col], dtype=torch.long)


def mol_to_graph(mol: Chem.Mol, query_atom: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`(x, edge_index, batch)` for one molecule, ready for `net.GCNNet.forward`."""
    x = torch.tensor(atom_features(mol, query_atom), dtype=torch.float32)
    edge_index = bond_edge_index(mol)
    batch = torch.zeros(x.size(0), dtype=torch.long)
    return x, edge_index, batch
