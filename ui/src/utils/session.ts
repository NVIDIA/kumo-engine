// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Session seam — lets layer-restricted code (services/, utils/) read the
 * current user session without importing the Zustand store directly
 * (services/utils must not depend on stores/).
 *
 * `userStore` mirrors `isLoggedIn` here on every change (see
 * stores/userStore). Readers here are non-reactive snapshots, which is what
 * the network layer needs.
 */
let loggedIn = false;

/** Mirror of userStore.userDetails.isLoggedIn, kept in sync by userStore. */
export const setSessionLoggedIn = (value: boolean): void => {
    loggedIn = value;
};

export const isSessionLoggedIn = (): boolean => loggedIn;
