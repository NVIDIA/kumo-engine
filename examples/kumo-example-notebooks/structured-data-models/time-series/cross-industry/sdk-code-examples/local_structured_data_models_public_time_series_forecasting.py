# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hourly forecasts, day-ahead energy prices and weekly retail sales

Generate local files first: python synthetic-data/sdm/time_series.py --output synthetic_data/time_series

Preview: Chronos2 and TimesFM3 are not available in the published
structured-data-models package. The commented workflow is not runnable.
Install when the model becomes available: pip install structured-data-models
"""

# import pandas as pd
# import torch
#
# import sdm
#
# from pathlib import Path
# url = Path("synthetic_data/time_series")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = sdm.models.Chronos2(device=device)
# # Alternative model preview: TimesFM3 is not available in the published package.
# model = sdm.models.TimesFM3(device=device)
#
# print("# Univariate Forecasting")  ############################################
# # 16 items ("item_id"), each with up to 960 observations:
# df = pd.read_csv(f"{url}/hourly.csv")
# df = df.sort_values(["item_id", "timestamp"])  # Left pad with NaNs:
# df["step"] = df.groupby("item_id").cumcount(ascending=False)
# df = df.pivot(index="item_id", columns="step", values="target").iloc[:, ::-1]
#
# tensor = torch.from_numpy(df.to_numpy(dtype="float32", copy=True)).to(device)
# table = sdm.TableTensor.from_tensor(tensor.unsqueeze(-1), columns=["target"])
#
# with torch.amp.autocast(device.type, torch.float16, enabled=table.is_cuda):
#     out = model(
#         x_context=table.drop_columns("target"),
#         y_context=table["target"],
#         # Forecast 24 observations per item:
#         x_query=sdm.TableTensor(size=(table.size(0), 24), device=device),
#         num_estimators=2,
#     )["target__q50"].numerical.squeeze(-1)  # Median prediction
# # Batch predictions: one row per item and forecast hour, in input item order.
# assert out.shape == (len(df), 24)
# results = pd.DataFrame(out.detach().cpu().numpy(), index=df.index).rename_axis("item_id")
# print(results.head(5))
# results.to_csv("hourly_forecasts.csv")
# print(f"24H forecast for item '{df.index[0]}':")
# print(out[0].cpu().numpy(), end="\n\n")
#
# print("# Forecasting with Covariates")  #######################################
# # Hourly energy prices for the next day using historical prices and day-ahead
# # forecasts of load and renewable energy generation
# context_df = pd.read_parquet(f"{url}/energy_train.parquet")
# query_df = pd.read_parquet(f"{url}/energy_test.parquet")
# stypes = {
#     "Ampirion Load Forecast": "numerical",
#     "PV+Wind Forecast": "numerical",
#     "target": "numerical",
# }
# context = sdm.TableTensor.from_pandas(context_df, stypes, device=device)
# query = sdm.TableTensor.from_pandas(query_df, stypes, device=device)
#
# with torch.amp.autocast(device.type, torch.float16, enabled=table.is_cuda):
#     out = model(
#         x_context=context.drop_columns("target"),
#         y_context=context["target"],
#         x_query=query.drop_columns("target"),
#         num_estimators=2,
#     )["target__q50"].numerical.squeeze(-1)  # Median prediction
# # Batch predictions: preserve the source test-row index for every forecast.
# assert len(out) == len(query_df)
# results = pd.DataFrame({"source_row": query_df.index, "median_price": out.detach().cpu().numpy()})
# print(results.head(10))
# results.to_csv("energy_price_forecasts.csv", index=False)
# print("24H energy price forecast for region 'DE':")
# print(out.cpu().numpy(), end="\n\n")
#
# print("# Forecasting with Past-only Covariates")  #############################
# # Forecast next quarter's weekly store sales using historical sales, historical
# # customer footfall, and known covariates indicating store operation, promotion
# # periods, and holidays.
# context_df = pd.read_parquet(f"{url}/retail_train.parquet")
# context_df = context_df.sort_values(["id", "timestamp"])
# query_df = pd.read_parquet(f"{url}/retail_test.parquet")
# query_df = query_df.sort_values(["id", "timestamp"])
# stypes = {
#     "Sales": "numerical",
#     "Open": "numerical",
#     "Promo": "numerical",
#     "SchoolHoliday": "numerical",
#     "StateHoliday": "numerical",
# }
# context = sdm.TableTensor.from_pandas(
#     context_df,
#     stypes=stypes | {"Customers": "numerical"},  # Add past-only covariate.
#     device=device,
# ).view(16, 120, -1)  # 16 stores + 120 weeks
# query = sdm.TableTensor.from_pandas(
#     query_df,
#     stypes,
#     device=device,
# ).view(16, 13, -1)  # 16 stores + 13 weeks
#
# with torch.amp.autocast(device.type, torch.float16, enabled=table.is_cuda):
#     out = model(
#         x_context=context.drop_columns("Sales"),
#         y_context=context["Sales"],  # type: ignore
#         x_query=query.drop_columns("Sales"),
#         num_estimators=2,
#     )["Sales__q50"].numerical.squeeze(-1)  # Median prediction
# # Batch predictions: flatten store-by-week forecasts in the same sorted order.
# assert out.numel() == len(query_df)
# results = query_df[["id", "timestamp"]].copy()
# results["median_sales"] = out.detach().cpu().numpy().reshape(-1)
# assert results.index.equals(query_df.index)
# print(results.head(10))
# results.to_csv("weekly_sales_forecasts.csv", index=False)
# print("Weekly sales forecast for item '1':")
# print(out[0].cpu().numpy())
