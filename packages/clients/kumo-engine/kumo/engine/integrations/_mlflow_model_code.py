# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from kumo.engine.connector._extras import import_provider
from kumo.engine.integrations._mlflow_model import KumoModel

import_provider('mlflow.models', 'mlflow').set_model(KumoModel())
