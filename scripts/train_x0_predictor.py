"""
x_0 predictor 모델 학습.

Loss: ||x0_pred - ts_approx||² (coordinate MSE)

사용법:
    python scripts/train_x0_predictor.py \
        --data_path data/x0_training_data/x0_training_data.npz \
        --output_dir checkpoints/x0_predictor/ \
        --epochs 100 \
        --lr 1e-3 \
        --batch_size 1
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# Halo8 매핑 (프로젝트의 정의와 일치시켜야 함)
HALO_ATOM_MAPPING = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 16: 5, 17: 5, 35: 6}
# Note: 위 매핑은 예시. 프로젝트의 실제 HALO_ATOM_MAPPING을 사용할 것.
try:
    from reactot.dataset.ff_lmdb import HALO_ATOM_MAPPING as _PROJECT_MAPPING
    from reactot.dataset.ff_lmdb import N_HALO_ATOM_TYPES as _PROJECT_N_TYPES
    HALO_ATOM_MAPPING = _PROJECT_MAPPING
    N_ATOM_TYPES = _PROJECT_N_TYPES
except Exception:
    N_ATOM_TYPES = max(HALO_ATOM_MAPPING.values()) + 1


class X0Dataset(Dataset):
    """
    npz로 저장된 (R, P, TS_approx) 삼중항 데이터셋.

    npz 구조 (prepare_x0_targets.py 출력 형식):
        - pos_R: object array, 각 원소가 (N_i, 3)
        - pos_P: object array
        - ts_approx: object array
        - atomic_numbers: object array
    """

    def __init__(self, data_path: str):
        data = np.load(data_path, allow_pickle=True)
        self.pos_R = data['pos_R']
        self.pos_P = data['pos_P']
        self.ts_approx = data['ts_approx']
        self.atomic_numbers = data['atomic_numbers']

    def __len__(self):
        return len(self.pos_R)

    def __getitem__(self, idx):
        z = np.asarray(self.atomic_numbers[idx])
        atom_types = np.array([HALO_ATOM_MAPPING.get(int(zi), 0) for zi in z])
        # object-dtype array elements can come back as plain Python objects;
        # coerce to a concrete float32 ndarray before handing to torch.tensor.
        pos_R = np.asarray(self.pos_R[idx], dtype=np.float32)
        pos_P = np.asarray(self.pos_P[idx], dtype=np.float32)
        ts = np.asarray(self.ts_approx[idx], dtype=np.float32)
        return {
            'pos_R': torch.from_numpy(pos_R),
            'pos_P': torch.from_numpy(pos_P),
            'ts_approx': torch.from_numpy(ts),
            'atom_types': torch.tensor(atom_types, dtype=torch.long),
        }


def collate_single(batch_list):
    """배치 사이즈 1 collate (분자 크기가 다양하므로 padding 대신 단일 처리)."""
    return batch_list[0]


def train(args):
    from reactot.model.x0_predictor import X0PredictorEGNN

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = X0PredictorEGNN(
        n_atom_types=N_ATOM_TYPES, hidden_dim=64, n_layers=3, cutoff=10.0,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    dataset = X0Dataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, collate_fn=collate_single)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for batch in dataloader:
            pos_R = batch['pos_R'].to(device)
            pos_P = batch['pos_P'].to(device)
            atom_types = batch['atom_types'].to(device)
            ts_target = batch['ts_approx'].to(device)
            # COM 제거 (target과 prediction의 reference frame 일치)
            ts_target = ts_target - ts_target.mean(dim=0, keepdim=True)

            x0_pred = model(pos_R, pos_P, atom_types)
            loss = ((x0_pred - ts_target) ** 2).sum(dim=-1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        scheduler.step()
        avg_loss = total_loss / max(len(dataloader), 1)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.6f}")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(),
                       output_dir / f'x0_pred_epoch{epoch+1}.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--output_dir', default='checkpoints/x0_predictor/')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()
    train(args)
