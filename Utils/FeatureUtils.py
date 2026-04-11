import torch
import numpy as np

def extract_features(model, loader, device):
    model.eval()

    all_features = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            # Raw input features (original transaction data)
            x = x.to(device)

            # Latent features from autoencoder (compressed representation)
            z = model.encoder(x)
            x_hat = model.decoder(z)

            diff = x - x_hat

            # Mean reconstruction error (overall anomaly score)
            mse = (diff ** 2).mean(dim=1, keepdim=True)
            
            abs_err = torch.abs(diff)

            # Maximum per-feature reconstruction error
            max_err = abs_err.max(dim=1, keepdim=True)[0]

            # Standard deviation of reconstruction errors (spread of anomalies)
            std_err = abs_err.std(dim=1, keepdim=True)

            # Sum of absolute reconstruction errors (L1 anomaly signal)
            l1_err = abs_err.mean(dim=1, keepdim=True)

            # Top-k largest feature reconstruction errors (strongest anomalies)
            topk_err = torch.topk(abs_err, k=5, dim=1)[0].mean(dim=1, keepdim=True)

            # Final features
            features = torch.cat([
                x,
                z,
                mse,
                max_err,
                std_err,
                l1_err,
                topk_err
            ], dim=1)

            all_features.append(features.cpu().numpy())
            all_labels.append(y.numpy())

    return np.concatenate(all_features), np.concatenate(all_labels)
