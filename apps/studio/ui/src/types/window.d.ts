// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Extend window interface for global utility functions
declare global {
    interface Window {
        reopenProjectList?: () => void;
    }
}

export {};
