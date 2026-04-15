# OT-FM 2차 정리: Claude Code 명령문

## 판단 근거

### pl_trainer.py 내부 구조 (3개 클래스)
```
DDPMModule      → EnVariationalDiffusion (en_diffusion.py)  → 삭제 대상
SBModule        → EnSB (en_sb.py)                           → ★ OT-FM 핵심. 절대 보존
ConfidenceModule → Confidence (dynamics/confidence.py)       → 삭제 대상
```

`train_rpsb_ts1x.py`에서 `from ... import SBModule, DDPMModule`로 import하지만,
실제로 인스턴스화하는 것은 `SBModule`뿐. DDPMModule은 dead code.

### 삭제 대상 파일 (4개)
| 파일 | 이유 |
|------|------|
| `reactot/trainer/potential_module.py` | Potential 에너지 학습용. OT-FM(=TS 구조 생성)과 무관 |
| `reactot/trainer/train_halo8.py` | PotentialModule 사용. OT-FM 학습 아님 |
| `reactot/evaluate/generate_confidence_sample.py` | Confidence 모델 전용 |
| `reactot/sampling/sample_datasets.py` | QM9 참조. T1x/Halogen OT-FM과 무관 |

### 편집 대상 파일 (3개) — 삭제된 모듈의 잔여 참조 제거
| 파일 | 편집 내용 |
|------|----------|
| `reactot/dataset/__init__.py` | QM9, Zeolite import 제거 |
| `reactot/diffusion/__init__.py` | EnVariationalDiffusion import 제거 |
| `reactot/trainer/pl_trainer.py` | DDPMModule 클래스 전체, ConfidenceModule 클래스 전체, 관련 import 제거 |

---

## Claude Code 명령문

아래 명령을 정확한 순서로 실행해줘. 
**핵심 규칙: SBModule 클래스와 그 의존성은 절대 수정하지 마.**

```
### Step 1: 독립 파일 삭제
rm -f reactot/trainer/potential_module.py
rm -f reactot/trainer/train_halo8.py
rm -f reactot/evaluate/generate_confidence_sample.py
rm -f reactot/sampling/sample_datasets.py

### Step 2: reactot/dataset/__init__.py 편집
# 아래 두 줄을 찾아서 삭제해줘:
#   from .qm9 import ProcessedQM9, ProcessedDoubleQM9, ProcessedTripleQM9
#   from .zeolite import ProcessedZeolite
# 
# 남겨야 할 줄:
#   from .base_dataset import BaseDataset
#   from .transition1x import ProcessedTS1x
#   from .sampler import DynamicBatchSampler
#   from .ff_lmdb import LmdbDataset  (있다면)

### Step 3: reactot/diffusion/__init__.py 편집
# 아래 줄을 찾아서 삭제해줘:
#   from .en_diffusion import EnVariationalDiffusion
#
# 남겨야 할 줄:
#   from . import _utils as utils
#   from ._schedule import DiffSchedule, PredefinedNoiseSchedule
#   from ._normalizer import Normalizer
#   from .en_sb import EnSB
#
# 참고: DiffSchedule, PredefinedNoiseSchedule은 _schedule.py에 남아있고
# 다른 곳에서 참조될 수 있으므로 __init__.py의 import는 유지해도 됨.

### Step 4: reactot/trainer/pl_trainer.py — 가장 중요한 수술

#### 4-1. import 줄 편집

# (a) 아래 줄을:
#   from reactot.dataset import ProcessedQM9, ProcessedDoubleQM9, ProcessedTripleQM9, ProcessedTS1x, DynamicBatchSampler, ProcessedZeolite
# 이렇게 변경해줘:
#   from reactot.dataset import ProcessedTS1x, DynamicBatchSampler

# (b) 아래 줄을:
#   from reactot.dynamics import EGNNDynamics, Confidence
# 이렇게 변경해줘:
#   from reactot.dynamics import EGNNDynamics

# (c) 아래 줄을:
#   from reactot.diffusion.en_diffusion import EnVariationalDiffusion
# 삭제해줘 (줄 전체 삭제)

# (d) 아래 줄은 그대로 유지:
#   from reactot.diffusion._schedule import DiffSchedule, PredefinedNoiseSchedule, SBSchedule
#   from reactot.diffusion.en_sb import EnSB
# DiffSchedule, PredefinedNoiseSchedule은 삭제된 DDPMModule만 사용하지만,
# _schedule.py 자체는 남아있으므로 import 에러는 안 남. 깔끔하게 제거해도 되지만
# 안전을 위해 유지.

# (e) Confidence 관련 metrics import:
#   from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score, BinaryPrecision, BinaryCohenKappa
# 이 줄은 ConfidenceModule만 사용. 삭제해줘.

#### 4-2. PROCESS_FUNC과 FILE_TYPE dict 편집

# PROCESS_FUNC에서 아래 항목들을 제거해줘:
#   "QM9": ProcessedQM9,
#   "DoubleQM9": ProcessedDoubleQM9,
#   "TripleQM9": ProcessedTripleQM9,
#   "Zeolite": ProcessedZeolite,
#
# 남길 항목:
#   "TS1x": ProcessedTS1x,

# FILE_TYPE에서도 동일하게:
#   "QM9": ".npz",    ← 삭제
#   "DoubleQM9": ".npz",  ← 삭제
#   "TripleQM9": ".npz",  ← 삭제
#   "Zeolite": ".pkl",    ← 삭제
#
# 남길 항목:
#   "TS1x": ".pkl",

#### 4-3. DDPMModule 클래스 전체 삭제

# `class DDPMModule(LightningModule):` 부터
# 그 클래스가 끝나는 지점(다음 class 정의 직전 또는 파일 끝)까지 전체 삭제.
#
# ⚠️ 주의: DDPMModule 바로 뒤에 SBModule이 정의되어 있을 수 있음.
# SBModule의 class 정의 시작점을 절대 삭제하지 마.
# 
# 확인 방법: grep -n "^class " reactot/trainer/pl_trainer.py
# 로 각 클래스의 시작 줄 번호를 확인한 후 삭제 범위를 결정해줘.

#### 4-4. ConfidenceModule 클래스 전체 삭제

# pl_trainer.py 하단에 ConfidenceModule 클래스가 있으면 전체 삭제.
# (BinaryAccuracy 등의 metrics를 사용하는 클래스)

### Step 5: train_rpsb_ts1x.py import 수정

# 아래 줄을:
#   from reactot.trainer.pl_trainer import SBModule, DDPMModule
# 이렇게 변경해줘:
#   from reactot.trainer.pl_trainer import SBModule

### Step 6: dynamics/__init__.py 편집

# Confidence import가 있으면 제거:
#   from .confidence import Confidence  ← 삭제
# EGNNDynamics import는 유지.

### Step 7: import 검증

python -c "from reactot.trainer.pl_trainer import SBModule; print('✅ SBModule OK')"
python -c "from reactot.model import LEFTNet; print('✅ LEFTNet OK')"
python -c "from reactot.diffusion.en_sb import EnSB; print('✅ EnSB OK')"
python -c "from reactot.dynamics import EGNNDynamics; print('✅ EGNNDynamics OK')"
python -c "from reactot.dataset.transition1x import ProcessedTS1x; print('✅ ProcessedTS1x OK')"
python -c "from reactot.dataset.ff_lmdb import LmdbDataset; print('✅ LmdbDataset OK')"

### Step 8: 전체 import 스캔 — 잔여 참조 최종 확인

grep -rn "ProcessedQM9\|ProcessedZeolite\|EnVariationalDiffusion\|Confidence\|PotentialModule\|potential_module\|qm9\|zeolite" \
  --include="*.py" reactot/ \
  | grep -v "__pycache__" \
  | grep -v ".pyc"

# 위 결과가 나오면, 각 줄을 보여주고 내 판단을 받은 후 처리해줘.
# 삭제된 파일 자체(.py)에서의 결과는 무시 가능.

### Step 9: 커밋

git add -A && git commit -m "cleanup round 2: remove DDPMModule, ConfidenceModule, PotentialModule and all QM9/Zeolite references"
```

---

## pl_trainer.py 수술 후 예상 구조

```python
# === imports ===
from reactot.dataset import ProcessedTS1x, DynamicBatchSampler
from reactot.dynamics import EGNNDynamics
from reactot.diffusion._schedule import SBSchedule  # DiffSchedule, PredefinedNoiseSchedule 제거 가능
from reactot.diffusion._normalizer import Normalizer, FEATURE_MAPPING
from reactot.diffusion.en_sb import EnSB
# ... (나머지 torch, lightning imports 유지)

PROCESS_FUNC = {
    "TS1x": ProcessedTS1x,
}
FILE_TYPE = {
    "TS1x": ".pkl",
}

# === DDPMModule 삭제됨 ===
# === ConfidenceModule 삭제됨 ===

class SBModule(LightningModule):
    # ... (완전히 보존)
```

---

## CLAUDE.md 업데이트 사항

CLAUDE.md의 Project Structure 섹션에 아래를 반영:
- `reactot/evaluate/` 디렉토리 추가 (generate_on_example_path.py 등이 있을 수 있음)
- `reactot/analyze/` 디렉토리 추가 (rmsd.py — pl_trainer.py가 import함)
- `reactot/sampling/` 디렉토리는 sample_datasets.py 삭제 후 빈 디렉토리면 삭제

```
아래 내용을 CLAUDE.md의 Project Structure 섹션 끝에 추가해줘:

    ├── evaluate/
    │   └── ...                # Evaluation scripts (RMSD, energy analysis)
    ├── analyze/
    │   └── rmsd.py            # batch_rmsd_sb, batch_rmsd (used by pl_trainer)
```
