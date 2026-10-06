// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { validateProjectsName } from "gen/v0/api/projects/projects";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { inputState, kumoResource } from "utils/Constants";
import { validateResourceName } from "utils/validator";

vi.mock("gen/v0/api/projects/projects", () => ({
    validateProjectsName: vi.fn()
}));

describe("validateResourceName", () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("validates a project name without sending group context", async () => {
        vi.mocked(validateProjectsName).mockResolvedValue({
            data: { is_valid: true, error_msg: "" }
        } as never);

        const result = await validateResourceName(
            kumoResource.Project,
            "Fraud Detection",
            "",
            undefined
        );

        expect(validateProjectsName).toHaveBeenCalledWith(
            { name: "Fraud Detection" },
            undefined,
            undefined
        );
        expect(result).toMatchObject({
            validationResponse: { isValid: true },
            inputState: inputState.Valid
        });
    });
});
