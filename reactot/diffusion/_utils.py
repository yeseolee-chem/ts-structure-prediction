from typing import List, Optional

import math
import torch
from torch import Tensor
from torch_scatter import scatter_add, scatter_mean

import ase
from ase.calculators.emt import EMT
try:
    from ase.neb import NEB
except ImportError:
    from ase.mep import NEB
from ase import Atoms


def remove_mean_batch(x, indices):
    mean = scatter_mean(x, indices, dim=0)
    x = x - mean[indices]
    return x


def assert_mean_zero_with_mask(x, node_mask, eps=1e-10):
    largest_value = x.abs().max().item()
    error = scatter_add(x, node_mask, dim=0).abs().max().item()
    rel_error = error / (largest_value + eps)
    assert rel_error < 1e-2, f"Mean is not zero, relative_error {rel_error}"


def sample_center_gravity_zero_gaussian_batch(
    size: List[int], indices: List[Tensor]
) -> Tensor:
    assert len(size) == 2
    x = torch.randn(size, device=indices[0].device)

    # This projection only works because Gaussian is rotation invariant
    # around zero and samples are independent!
    x_projected = remove_mean_batch(x, torch.cat(indices))
    return x_projected


def sum_except_batch(x, indices, dim_size):
    return scatter_add(x.sum(-1), indices, dim=0, dim_size=dim_size)


def cdf_standard_gaussian(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2)))


def sample_gaussian(size, device):
    x = torch.randn(size, device=device)
    return x


def num_nodes_to_batch_mask(n_samples, num_nodes, device):
    assert isinstance(num_nodes, int) or len(num_nodes) == n_samples

    if isinstance(num_nodes, torch.Tensor):
        num_nodes = num_nodes.to(device)

    sample_inds = torch.arange(n_samples, device=device)

    return torch.repeat_interleave(sample_inds, num_nodes)


def unsqueeze_xdim(z, xdim):
    bc_dim = (...,) + (None,) * len(xdim)
    return z[bc_dim]


def space_indices(num_steps, count):
    assert count <= num_steps

    if count <= 1:
        frac_stride = 1
    else:
        frac_stride = (num_steps - 1) / (count - 1)

    cur_idx = 0.0
    taken_steps = []
    for _ in range(count):
        taken_steps.append(round(cur_idx))
        cur_idx += frac_stride

    return taken_steps


def idpp_guess(r_pos, p_pos, x0_size, x0_other,
               n_images=3,
               interpolate="idpp",
               use_clash_penalty=True,
               learned_x0_checkpoint: Optional[str] = None):
    """Initial-guess generator for x_0 in OT-FM.

    Parameters
    ----------
    r_pos, p_pos : Tensor
        Per-atom positions for reactants / products (concatenated across batch).
    x0_size : Tensor
        Per-sample atom counts (used to split the concatenated tensors).
    x0_other : Tensor
        Per-atom feature block whose last column is atomic number Z.
    n_images : int
        Number of NEB images (used for the IDPP/linear paths).
    interpolate : {"idpp", "linear", "learned"}
        - "idpp"    : IDPP-interpolated NEB midpoint (legacy default)
        - "linear"  : straight-line NEB midpoint
        - "learned" : x_0 from a trained ``X0PredictorEGNN`` checkpoint
                      (Idea 2-F). Requires ``learned_x0_checkpoint``.
    use_clash_penalty : bool
        Reserved for future clash-aware variants; currently a no-op.
    learned_x0_checkpoint : Optional[str]
        Path to a trained x_0 predictor checkpoint. Required when
        ``interpolate == "learned"``.
    """
    if interpolate == "learned":
        if learned_x0_checkpoint is None:
            raise ValueError(
                "interpolate='learned'는 learned_x0_checkpoint 경로가 필요합니다"
            )
        from reactot.utils.initial_guess import compute_learned_x0

        split_indices = torch.cumsum(x0_size, dim=0).cpu().tolist()[:-1]
        _r_pos = torch.tensor_split(r_pos, split_indices)
        _p_pos = torch.tensor_split(p_pos, split_indices)
        z_split = torch.tensor_split(x0_other[:, -1], split_indices)
        z_list = [_z.long().cpu().numpy() for _z in z_split]

        device_str = "cuda" if (
            isinstance(x0_size, torch.Tensor) and x0_size.is_cuda
        ) else "cpu"

        ts_pos = []
        for x_r, x_p, atom_number in zip(_r_pos, _p_pos, z_list):
            x0 = compute_learned_x0(
                x_r.cpu().numpy(), x_p.cpu().numpy(), atom_number,
                model_checkpoint=learned_x0_checkpoint,
                device=device_str,
            )
            ts_pos.append(torch.tensor(x0, dtype=torch.float32))

        return torch.concat(ts_pos).to(x0_size.device)

    _r_pos = torch.tensor_split(
        r_pos,
        torch.cumsum(x0_size, dim=0).to("cpu")[:-1]
    )
    _p_pos = torch.tensor_split(
        p_pos,
        torch.cumsum(x0_size, dim=0).to("cpu")[:-1]
    )
    z = torch.tensor_split(
        x0_other[:, -1],
        torch.cumsum(x0_size, dim=0).to("cpu")[:-1]
    )
    z = [_z.long().cpu().numpy() for _z in z]

    ts_pos = []
    for x_r, x_p, atom_number in zip(_r_pos, _p_pos, z):
        mol_r = Atoms(
            numbers=atom_number,
            positions=x_r.cpu().numpy(),
        )
        mol_p = Atoms(
            numbers=atom_number,
            positions=x_p.cpu().numpy(),
        )

        images = [mol_r.copy()]
        for _ in range(n_images - 2):
            images.append(mol_r.copy())
        images.append(mol_p.copy())

        for image in images:
            image.calc = EMT()

        neb = NEB(images)
        if interpolate == "idpp":
            neb.idpp_interpolate(
                traj=None, log=None, fmax=1000, optimizer=ase.optimize.MDMin, mic=False, steps=0)
        elif interpolate == "linear":
            neb.interpolate('linear')
        else:
            raise ValueError("interpolate can only be idpp, linear, or learned")
        x_ts = torch.tensor(
            neb.images[n_images // 2].arrays["positions"],
            dtype=torch.float32,
        )
        ts_pos.append(x_ts)

    ts_pos = torch.concat(ts_pos).to(x0_size.device)
    return ts_pos