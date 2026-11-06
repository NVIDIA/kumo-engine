# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Forecast ice cream and cold drink demand with historical weather and known future prices.

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
# import pandas as pd
# frame = pd.read_csv("synthetic_data/time_series/weather.csv")
# x_context = sdm.TableTensor.from_pandas(frame.iloc[:-24][["observed_weather", "planned_price"]], stypes={"observed_weather": "numerical", "planned_price": "numerical"}, device=device)
# y_context = sdm.TableTensor.from_pandas(frame.iloc[:-24][["ice_cream", "cold_drinks"]], stypes={"ice_cream": "numerical", "cold_drinks": "numerical"}, device=device)
# x_query = sdm.TableTensor.from_pandas(frame.iloc[-24:][["planned_price"]], stypes={"planned_price": "numerical"}, device=device)
#
# # The first run prompts for acceptance of Google's non-commercial license.
# model = sdm.models.TimesFM3(device=device)
# model.fit(x_context, y_context)
# forecast = model.predict(x_query)
#
# print(forecast.columns[sdm.Stype.numerical])
# print(forecast.numerical)
#
# # Batch predictions: forecast every planned future step without future demand labels.
# import pandas as pd
# values = forecast.numerical.detach().cpu().numpy()
# assert values.shape[-2] == x_query.size(-2)
# # This example has one series; each row is its next forecast step.
# results = pd.DataFrame(values.reshape(-1, values.shape[-1]), columns=forecast.columns[sdm.Stype.numerical])
# results.insert(0, "forecast_step", range(1, len(results) + 1))
# print(results.head(10))
# results.to_csv("weather_price_forecasts.csv", index=False)
