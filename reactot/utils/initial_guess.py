"""
OT-FM 초기 구조(x_0) 생성 모듈 — Idea 2-D (GFN2-xTB 반경험적 사전 최적화).

IDPP (또는 (R+P)/2) 결과를 GFN2-xTB로 짧게(10~20 step) 구조 최적화하여
비물리적 구조를 제거한다. Core 원자 간 거리를 constraint로 고정하여
반응 좌표를 보존하면서, 비결합 상호작용(vdW, 정전기, lone pair 반발)만 이완.

References:
- GFN2-xTB: Bannwarth, Ehlert, Grimme, J. Chem. Theory Comput., 2019,
  DOI: 10.1021/acs.jctc.8b01176
- Halo8 dataset (xTB optimization in reactant prep): Lee et al.,
  Sci. Data, 2025, DOI: 10.1038/s41597-025-05944-3
"""
import numpy as np
from typing import Optional, List, Tuple


def compute_xtb_refined_x0(
    x0_initial: np.ndarray,
    atomic_numbers: np.ndarray,
    core_atom_pairs: Optional[List[Tuple[int, int]]] = None,
    max_steps: int = 20,
    fmax: float = 0.5,
    max_step_size: float = 0.05,
    method: str = 'GFN2-xTB',
    charge: int = 0,
    multiplicity: int = 1,
    fallback_on_error: bool = True,
) -> np.ndarray:
    """
    GFN2-xTB로 초기 구조를 짧게 최적화한다.

    Core 원자 간 거리를 constraint로 고정하여 반응 좌표를 보존하면서,
    비결합 상호작용(vdW, 정전기)만 이완시킨다.

    Args:
        x0_initial: (N, 3) 초기 구조 (IDPP 또는 (R+P)/2)
        atomic_numbers: (N,)
        core_atom_pairs: list of (i, j) — constraint로 고정할 core 원자쌍.
                         None이면 constraint 없이 최적화.
        max_steps: 최대 최적화 step 수 (매우 짧게!)
        fmax: force 수렴 기준 (**eV/Å**, ASE 표준; 느슨하게 설정)
        max_step_size: 한 step의 최대 이동량 (Å)
        method: xTB 방법 ('GFN2-xTB', 'GFN1-xTB', 'GFN-FF')
        charge: 분자 전하 (예: Cl⁻ + CH₃Br 시스템 = -1)
        multiplicity: 스핀 다중도 (closed-shell singlet = 1)
        fallback_on_error: xTB 오류 시 원본 반환 여부

    Returns:
        x0_refined: (N, 3) refined structure (center-of-mass at origin)
    """
    try:
        from ase import Atoms
        from ase.optimize import LBFGS
        from ase.constraints import FixBondLengths
    except ImportError:
        raise ImportError("ASE가 필요합니다: pip install ase")

    # xTB calculator import (두 가지 경로 시도)
    XTB = None
    try:
        from xtb.ase.calculator import XTB as _XTB
        XTB = _XTB
    except ImportError:
        try:
            from ase.calculators.xtb import XTB as _XTB
            XTB = _XTB
        except ImportError:
            if fallback_on_error:
                print("WARNING: xTB not available, returning original structure")
                return x0_initial.copy()
            raise ImportError(
                "xtb-python이 필요합니다: conda install -c conda-forge xtb-python"
            )

    # ASE Atoms 객체 생성
    atoms = Atoms(
        numbers=atomic_numbers.astype(int),
        positions=x0_initial.copy(),
    )

    # xTB calculator 설정
    # Note: multiplicity는 일부 xtb 버전에서 'uhf' (unpaired electrons) 인자로 전달
    try:
        calc = XTB(method=method, charge=charge, uhf=multiplicity - 1)
    except TypeError:
        # 구버전 xtb-python은 uhf 인자 미지원
        calc = XTB(method=method, charge=charge)
    atoms.calc = calc

    # Core 원자쌍 constraint (결합 길이 고정)
    if core_atom_pairs is not None and len(core_atom_pairs) > 0:
        # FixBondLengths는 list of [i, j] 형식 요구
        constraints = FixBondLengths([list(pair) for pair in core_atom_pairs])
        atoms.set_constraint(constraints)

    # 짧은 최적화
    try:
        opt = LBFGS(atoms, maxstep=max_step_size, logfile=None)
        opt.run(fmax=fmax, steps=max_steps)
    except Exception as e:
        if fallback_on_error:
            print(f"WARNING: xTB optimization failed ({e}), returning original")
            return x0_initial.copy()
        raise

    # 결과 추출
    x0_refined = atoms.get_positions()

    # Center-of-mass 제거
    x0_refined -= x0_refined.mean(axis=0)

    return x0_refined


def _fallback_core_pairs(pos_R: np.ndarray, pos_P: np.ndarray,
                         atomic_numbers: np.ndarray,
                         distance_change_thr: float = 0.5) -> List[Tuple[int, int]]:
    """
    Idea 1의 reactive core 식별 함수가 아직 구현되지 않은 경우의 fallback.

    R과 P 사이에서 결합 길이가 distance_change_thr (Å) 이상 변하는 원자쌍을
    "reactive core 결합"으로 간주하여 constraint 대상에서 **제외**하고,
    그 외 결합들(공유결합 거리 안)을 core_pairs로 반환.

    이는 "반응에 직접 참여하는 결합은 풀어두고, 나머지 결합은 고정"하는
    보수적 전략. 결과적으로 vdW/정전기만 이완.
    """
    from scipy.spatial.distance import cdist

    covalent_radii = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66,
                      9: 0.57, 16: 1.05, 17: 0.99, 35: 1.20, 53: 1.39}

    N = len(atomic_numbers)
    d_R = cdist(pos_R, pos_R)
    d_P = cdist(pos_P, pos_P)

    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            r_cov = (covalent_radii.get(int(atomic_numbers[i]), 0.77) +
                     covalent_radii.get(int(atomic_numbers[j]), 0.77))
            # R 또는 P에서 공유결합으로 보이고
            is_bond = (d_R[i, j] < r_cov * 1.3) or (d_P[i, j] < r_cov * 1.3)
            if not is_bond:
                continue
            # 결합 길이 변화가 작으면 (반응에 참여하지 않음) → 고정
            if abs(d_R[i, j] - d_P[i, j]) < distance_change_thr:
                pairs.append((i, j))
    return pairs


def compute_xtb_refined_x0_with_idpp(
    pos_R: np.ndarray,
    pos_P: np.ndarray,
    atomic_numbers: np.ndarray,
    idpp_kwargs: Optional[dict] = None,
    xtb_kwargs: Optional[dict] = None,
    use_idea1_core: bool = False,
) -> np.ndarray:
    """
    IDPP → xTB refinement 2단계 파이프라인.

    1. IDPP로 초기 구조 생성 (Idea 2-A의 compute_idpp)
    2. Core 식별:
       - use_idea1_core=True 이고 Idea 1이 구현되어 있으면 그 함수 사용
       - 그렇지 않으면 _fallback_core_pairs 사용 (R↔P 결합 길이 변화 기준)
    3. xTB로 짧은 최적화 (core 결합 길이 고정)
    """
    idpp_kw = idpp_kwargs or {}
    xtb_kw = xtb_kwargs or {}

    # 1. IDPP (Idea 2-A의 함수, 같은 파일에 정의됨)
    x0_idpp = compute_idpp(pos_R, pos_P, atomic_numbers, **idpp_kw)

    # 2. Core 결합쌍
    core_pairs: List[Tuple[int, int]] = []
    if use_idea1_core:
        try:
            # Idea 1이 구현된 경우 (실제 모듈 경로는 프로젝트에 맞게 조정)
            from reactot.utils.reactive_core import (
                find_reactive_core_from_positions,
                build_adjacency_matrix,
            )
            core = find_reactive_core_from_positions(pos_R, pos_P, atomic_numbers)
            core_list = sorted(core)
            adj = build_adjacency_matrix(pos_R, atomic_numbers)
            for i in core_list:
                for j in core_list:
                    if j > i and adj[i, j]:
                        core_pairs.append((i, j))
        except ImportError:
            print("INFO: Idea 1의 reactive_core 모듈이 없어 fallback 사용")
            core_pairs = _fallback_core_pairs(pos_R, pos_P, atomic_numbers)
    else:
        core_pairs = _fallback_core_pairs(pos_R, pos_P, atomic_numbers)

    # 3. xTB refinement
    x0_refined = compute_xtb_refined_x0(
        x0_idpp, atomic_numbers,
        core_atom_pairs=core_pairs if core_pairs else None,
        **xtb_kw,
    )

    return x0_refined
