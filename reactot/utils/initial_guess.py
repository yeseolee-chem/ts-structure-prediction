"""Learned x_0 initial-guess utilities for OT-FM.

Contains:
    - process-level model cache (so each forward pass doesn't re-load weights)
    - ``compute_learned_x0`` entry point used by ``reactot/diffusion/_utils.py``
      when ``interpolate="learned"`` is selected.
"""
from typing import Optional

import numpy as np
import torch


# 모델 캐시: (checkpoint_path, device) → (model, mapping)
# v2: 매 호출마다 load_state_dict하지 않도록 process-level 캐시
_X0_PREDICTOR_CACHE: dict = {}


def _get_or_load_x0_predictor(model_checkpoint: str, device: str = 'cpu',
                              n_atom_types: Optional[int] = None):
    key = (model_checkpoint, device)
    if key in _X0_PREDICTOR_CACHE:
        return _X0_PREDICTOR_CACHE[key]

    from reactot.model.x0_predictor import X0PredictorEGNN

    # 프로젝트의 실제 atom-type 개수가 있으면 그걸 우선
    if n_atom_types is None:
        try:
            from reactot.dataset.ff_lmdb import N_HALO_ATOM_TYPES as _N
            n_atom_types = _N
        except Exception:
            n_atom_types = 7

    model = X0PredictorEGNN(n_atom_types=n_atom_types, hidden_dim=64, n_layers=3)
    state = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    _X0_PREDICTOR_CACHE[key] = model
    return model


def compute_learned_x0(
    pos_R: np.ndarray,
    pos_P: np.ndarray,
    atomic_numbers: np.ndarray,
    model_checkpoint: str,
    device: str = 'cpu',
) -> np.ndarray:
    """
    학습된 x_0 predictor로 초기 구조를 예측한다.

    v2: 모델은 process-level 캐시되어 매 호출마다 재로딩되지 않음.
    """
    # 프로젝트의 실제 HALO_ATOM_MAPPING을 import해서 사용 (여기서는 inline 정의)
    HALO_ATOM_MAPPING = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 16: 5, 17: 5, 35: 6}
    try:
        from reactot.dataset.ff_lmdb import HALO_ATOM_MAPPING as _PROJECT_MAPPING
        HALO_ATOM_MAPPING = _PROJECT_MAPPING
    except Exception:
        pass

    model = _get_or_load_x0_predictor(model_checkpoint, device)

    atom_types = np.array(
        [HALO_ATOM_MAPPING.get(int(z), 0) for z in atomic_numbers]
    )

    pos_R_t = torch.tensor(pos_R, dtype=torch.float32, device=device)
    pos_P_t = torch.tensor(pos_P, dtype=torch.float32, device=device)
    atom_types_t = torch.tensor(atom_types, dtype=torch.long, device=device)

    with torch.no_grad():
        x0_pred = model(pos_R_t, pos_P_t, atom_types_t)

    return x0_pred.cpu().numpy()
