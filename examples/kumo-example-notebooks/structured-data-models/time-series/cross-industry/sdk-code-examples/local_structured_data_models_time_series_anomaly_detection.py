# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multivariate synthetic time-series anomaly detection

Generate local files first: python synthetic-data/sdm/time_series.py --output synthetic_data/time_series

Preview: this time-series model is not available in the published
structured-data-models package. The commented workflow is not runnable.
Install when the model becomes available: pip install structured-data-models
"""

# import torch
#
# import sdm
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# import pandas as pd
# frame = pd.read_csv("synthetic_data/time_series/anomaly.csv")
# time = torch.tensor(frame.step.to_numpy(), device=device)
# x = torch.tensor(frame[["sensor_a", "sensor_b"]].to_numpy(), dtype=torch.float32, device=device)
#
# model = sdm.models.KumoAnomaly(device=device)
# out = model(x)
#
# # Higher MAE scores indicate more likely anomalies:
# print("Reconstruction Error (MAE):")
# print(out["mae"].numerical[130:160, 0])
#
# # SCS flags anomalies using adaptive segments of the preprocessed series:
# print("Anomaly Labels (SCS):")
# print(out["scs"].categorical.decode()[130:160, 0])
#
# # MACS flags anomalies using variance-weighted score bounds:
# print("Anomaly Labels (MACS):")
# print(out["macs"].categorical.decode()[130:160, 0])
#
# # Batch predictions: reconstruction scores align with the complete input timeline.
# import pandas as pd
# assert out["mae"].size(0) == len(time)
# results = pd.DataFrame({"step": time.cpu().numpy(), "mae": out["mae"].numerical[:, 0].cpu().numpy()})
# print(results.head(10))
# results.to_csv("anomaly_scores.csv", index=False)
