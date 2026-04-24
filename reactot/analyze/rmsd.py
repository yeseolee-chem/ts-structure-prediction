from typing import List
import numpy as np

from pymatgen.core import Molecule
from pymatgen.analysis.molecule_matcher import BruteForceOrderMatcher, GeneticOrderMatcher, HungarianOrderMatcher, KabschMatcher
from pymatgen.io.xyz import XYZ

from torch import Tensor


def xh2pmg(xh):
    mol = Molecule(
        species=xh[:, -1].long().cpu().numpy(),
        coords=xh[:, :3].cpu().numpy(),
    )
    return mol


def xyz2pmg(xyzfile):
    xyz_converter = XYZ(mol=None)
    mol = xyz_converter.from_file(xyzfile).molecule
    return mol


def rmsd_core(mol1, mol2, threshold=0.5, same_order=False):
    _, count = np.unique(mol1.atomic_numbers, return_counts=True)
    if same_order:
        bfm = KabschMatcher(mol1)
        _, rmsd = bfm.fit(mol2)
        return rmsd
    total_permutations = 1
    for c in count:
        total_permutations *= np.math.factorial(c)  # type: ignore
    if total_permutations < 1e4:
        bfm = BruteForceOrderMatcher(mol1)
        _, rmsd = bfm.fit(mol2)
    else:
        bfm = GeneticOrderMatcher(mol1, threshold=threshold)
        pairs = bfm.fit(mol2)
        rmsd = threshold
        for pair in pairs:
            rmsd = min(rmsd, pair[-1])
        if not len(pairs):
            bfm = HungarianOrderMatcher(mol1)
            _, rmsd = bfm.fit(mol2)
    return rmsd


def pymatgen_rmsd(
    mol1,
    mol2,
    ignore_chirality: bool = False,
    threshold: float = 0.5,
    same_order: bool = True,
):
    if isinstance(mol1, str):
        mol1 = xyz2pmg(mol1)
    if isinstance(mol2, str):
        mol2 = xyz2pmg(mol2)
    rmsd = rmsd_core(mol1, mol2, threshold, same_order=same_order)
    if ignore_chirality:
        coords = mol2.cart_coords
        coords[:, -1] = -coords[:, -1]
        mol2_reflect = Molecule(
            species=mol2.species,
            coords=coords,
        )
        rmsd_reflect = rmsd_core(
            mol1, mol2_reflect, threshold, same_order=same_order)
        rmsd = min(rmsd, rmsd_reflect)
    return rmsd


def batch_rmsd(
    fragments_nodes: List[Tensor],
    out_samples: List[Tensor],
    xh: List[Tensor],
    idx: int = 1,
    threshold: float = 0.5,
    same_order: bool = False,
) -> List[float]:
    rmsds = []
    out_samples_use = out_samples[idx]
    xh_use = xh[idx]
    nodes = fragments_nodes[idx].long().cpu().numpy()
    start_ind, end_ind = 0, 0
    for jj, natoms in enumerate(nodes):
        end_ind += natoms
        mol1 = xh2pmg(out_samples_use[start_ind:end_ind])
        mol2 = xh2pmg(xh_use[start_ind:end_ind])
        try:
            rmsd = pymatgen_rmsd(
                mol1,
                mol2,
                ignore_chirality=True,
                threshold=threshold,
                same_order=same_order,
            )
        except:
            rmsd = 1
        rmsds.append(min(rmsd, 1.0))
        start_ind = end_ind
    return rmsds

def batch_rmsd_sb(
    fragments_node: Tensor,
    pred_xh: Tensor,
    target_xh: Tensor,
    threshold: float = 0.5,
    same_order: bool = True,
    atom_weights: "Tensor | None" = None,
) -> List[float]:

    rmsds = []

    end_ind = np.cumsum(fragments_node.long().cpu().numpy())
    start_ind = np.concatenate([np.int64(np.zeros(1)), end_ind[:-1]])

    if atom_weights is not None:
        # Idea 1-AB: Weighted RMSD path. Uses same_order pairing — the weights
        # align to atom indices from Stage 2 training, so pymatgen reordering
        # would break the alignment.
        w = atom_weights.detach().cpu().numpy().astype(np.float64)
        pred = pred_xh[:, : 3].detach().cpu().numpy().astype(np.float64)
        tgt = target_xh[:, : 3].detach().cpu().numpy().astype(np.float64)
        for start, end in zip(start_ind, end_ind):
            rmsds.append(
                float(weighted_rmsd(pred[start:end], tgt[start:end], w[start:end]))
            )
        return rmsds

    for start, end in zip(start_ind, end_ind):
        mol1 = xh2pmg(pred_xh[start : end])
        mol2 = xh2pmg(target_xh[start : end])
        rmsd = pymatgen_rmsd(
            mol1,
            mol2,
            ignore_chirality=True,
            threshold=threshold,
            same_order=same_order,
        )
        rmsds.append(min(rmsd, 1.0))
    return rmsds


def weighted_rmsd(
    pos_pred: np.ndarray,
    pos_true: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Weighted RMSD for Stage 5 / Stage 2 consistency.

        RMSD_w = sqrt( sum(w_i * ||r_pred_i - r_true_i||^2) / sum(w_i) )

    Assumes pos_pred and pos_true are aligned atom-for-atom (same order).
    """
    pos_pred = np.asarray(pos_pred, dtype=np.float64)
    pos_true = np.asarray(pos_true, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    sq_dist = np.sum((pos_pred - pos_true) ** 2, axis=1)
    return float(np.sqrt(np.sum(weights * sq_dist) / (np.sum(weights) + 1e-8)))
