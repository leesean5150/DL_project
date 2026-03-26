import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from Utils import TransactionDataset
from Models import TransformerFraudModel

ds = TransactionDataset(
    processed_dir="data/processed_fraud",
    seq_len=8,
    split="train",
)

sample = ds[0]

x = sample["x"].unsqueeze(0)  # (1, 8, 112)
attention_mask = sample["attention_mask"].unsqueeze(0)  # (1, 8)

model = TransformerFraudModel(
    feature_dim=111,
    seq_len=8,
    d_model=64,
    nhead=4,
    num_layers=2,
    dim_feedforward=128,
    dropout=0.1,
)

logits = model(x, attention_mask)

print("x shape:", x.shape)
print("attention_mask shape:", attention_mask.shape)
print("logits shape:", logits.shape)
print("logits:", logits)