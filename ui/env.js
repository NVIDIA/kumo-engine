// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// This will be replaced by the update-env script
const runningLocally = __UI_LOCAL__;

const port = runningLocally ? 3000 : 8000;

window._env_ = {
  REACT_APP_BASE_URL_DEV: runningLocally ? "/" : `http://127.0.0.1:${port}/`,
  REACT_APP_BASE_URL_PROD: runningLocally ? "/" : `http://127.0.0.1:${port}/`,
  AUTH_DISABLED: __AUTH_DISABLED__,
};
