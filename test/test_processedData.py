import numpy as np
import json


# These are relative paths so run this in root.
print("[UNNORMALIZED]=================================")
features = np.load("data/processed_fraud/features.npy", mmap_mode="r")
labels = np.load("data/processed_fraud/labels.npy", mmap_mode="r")
seq_lengths = np.load("data/processed_fraud/seq_lengths.npy", mmap_mode="r")
with open("data/processed_fraud/feature_keys.json", "r", encoding="utf-8") as f:
    feature_keys = json.load(f)

print(features[:10])                    # just quick inspection of features
print(features.shape)                   # Should be (590540, 98)
print(labels.shape)                     # (590540,)
print(seq_lengths.shape)                # (217850,)
print(seq_lengths[:10])                 # Just for our our own inspection
print(features.min(), features.max())   # Ensuring that no nans

# checking how bad the -999 nan sentinel is
# neg999_cols = []
# for i, key in enumerate(feature_keys):
#     count = int(np.sum(features[:, i] == -999.0))
#     if count > 0:
#         neg999_cols.append((key, count))

# print(f"Columns containing -999: {len(neg999_cols)}")
# for key, count in neg999_cols:
#     print(f"{key}: {count}")

# Columns containing -999: 15
# D1: 1269
# D10: 76022
# D11: 279287
# D12: 525823
# D13: 528588
# D14: 528353
# D15: 89113
# D2: 280797
# D3: 262878
# D4: 168922
# D5: 309841
# D6: 517353
# D7: 551623
# D8: 515614
# D9: 515614

print([k for k in feature_keys if k.endswith("_missing")])
print([k for k in feature_keys if k.endswith("_present")])

try:
    print("[NORMALIZED]=================================")
    features = np.load("data/processed_fraud_normalized/features.npy", mmap_mode="r")
    labels = np.load("data/processed_fraud_normalized/labels.npy", mmap_mode="r")
    seq_lengths = np.load("data/processed_fraud_normalized/seq_lengths.npy", mmap_mode="r")
    with open("data/processed_fraud_normalized/feature_keys.json", "r", encoding="utf-8") as f:
        feature_keys = json.load(f)

    print(features[:10])                    # just quick inspection of features
    print(features.shape)                   # Should be (590540, 98)
    print(labels.shape)                     # (590540,)
    print(seq_lengths.shape)                # (217850,)
    print(seq_lengths[:10])                 # Just for our our own inspection
    print(features.min(), features.max())   # Ensuring that no nans

    print([k for k in feature_keys if k.endswith("_missing")])
    print([k for k in feature_keys if k.endswith("_present")])
    for k in ["ProductCD", "card4", "M1"]:
        idx = feature_keys.index(k)
        print(k, features[:10, idx])
except Exception as e:
    print("Normalised set not created yet.")