// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { mockNavigateCall, mockUseNavigate } from "tests/mockRouter";

import { getProjectPath } from "utils/Constants";
import {
    navigateToLandingPageAfterLoginSuccess,
    useCustomNavigation
} from "utils/navigator";

describe("navigateToLandingPageAfterLoginSuccess", () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("should navigate to the selected project's models", () => {
        navigateToLandingPageAfterLoginSuccess(mockUseNavigate);
        expect(mockUseNavigate).toHaveBeenCalledWith(
            getProjectPath("project-tests").ModelJobsList
        );
    });
});

describe("useCustomNavigation - navRule", () => {
    beforeEach(() => {
        vi.clearAllMocks();
        mockNavigateCall();
    });

    it.each([
        {
            description: "should call navigate with path and options",
            path: "/my-path" as string,
            options: { replace: true },
            event: undefined,
            expectedCall: ["/my-path", { replace: true }],
            shouldCallNavigate: true
        },
        {
            description:
                "should call navigate with path only if options are missing",
            path: "/simple-path" as string,
            options: undefined,
            event: undefined,
            expectedCall: ["/simple-path"],
            shouldCallNavigate: true
        },
        {
            description: "should call navigate with -1 to go back",
            path: -1 as -1,
            options: undefined,
            event: undefined,
            expectedCall: [-1],
            shouldCallNavigate: true
        }
    ])(
        "$description",
        ({ path, options, event, expectedCall, shouldCallNavigate }) => {
            const navRule = useCustomNavigation();
            navRule(path as string | -1, options, event);

            if (shouldCallNavigate) {
                expect(mockUseNavigate).toHaveBeenCalledWith(...expectedCall);
            }
        }
    );

    it.each([
        {
            description: "should open in new tab if ctrlKey is pressed",
            path: "/new-tab",
            event: { ctrlKey: true } as MouseEvent
        },
        {
            description: "should open in new tab if metaKey is pressed (mac)",
            path: "/mac-tab",
            event: { metaKey: true } as MouseEvent
        }
    ])("$description", ({ path, event }) => {
        const openSpy = vi.spyOn(window, "open").mockImplementation(() => null);
        const navRule = useCustomNavigation();
        navRule(path, null, event);

        expect(mockUseNavigate).toBeCalledTimes(0);
        expect(openSpy).toHaveBeenCalledWith(path, "_blank");
    });
});
