// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import DOMPurify from "dompurify";

if (window.trustedTypes && window.trustedTypes.createPolicy) {
    window.trustedTypes.createPolicy("default", {
        createHTML: (string) =>
            DOMPurify.sanitize(string, { RETURN_TRUSTED_TYPE: true }),
        createScriptURL: (string) =>
            DOMPurify.sanitize(string, { RETURN_TRUSTED_TYPE: true }),
        createScript: (string) =>
            DOMPurify.sanitize(string, { RETURN_TRUSTED_TYPE: true })
    });
}
