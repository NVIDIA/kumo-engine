// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { act, ComponentType } from "react";
import { MemoryRouter } from "react-router";

import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { verifyTextExists } from "tests/testHelpers";

import WithErrorBoundary, {
    getComponentNameFromStack
} from "../WithErrorBoundary";

describe("WithErrorBoundary", () => {
    const renderWithErrorBoundary = async <P extends object>(
        WrappedComponent: ComponentType<P>,
        props: P = {} as P
    ) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <WrappedComponent {...props} />
                </MemoryRouter>
            );
        });
    };

    const createErrorComponent = (errorMessage = "Test error") => {
        return () => {
            throw new Error(errorMessage);
        };
    };

    const verifyErrorFallback = async () => {
        await verifyTextExists(/Something went wrong/i);
        await verifyTextExists(
            /Please contact Kumo support\s*or try refreshing the page if the issue persists/i
        );
    };

    it("Should render the wrapped component when no error is thrown", async () => {
        const TestComponent = () => <div>Test Component</div>;
        const WrappedComponent = WithErrorBoundary(TestComponent);

        await renderWithErrorBoundary(WrappedComponent);

        await verifyTextExists("Test Component");
    });

    it("Should render fallback UI when wrapped component throws an error", async () => {
        const TestComponent = createErrorComponent();
        const WrappedComponent = WithErrorBoundary(TestComponent);

        await renderWithErrorBoundary(WrappedComponent);

        await verifyErrorFallback();

        expect(screen.queryByText(/Stack Trace/i)).toBeNull();
    });

    it("Should trigger page reload when refresh button is clicked", async () => {
        vi.spyOn(window.location, "reload").mockImplementation(() => {});
        const TestComponent = createErrorComponent();
        const WrappedComponent = WithErrorBoundary(TestComponent);

        await renderWithErrorBoundary(WrappedComponent);

        const refreshButton = screen.getByText("Refresh Page");
        await userEvent.click(refreshButton);

        expect(window.location.reload).toHaveBeenCalledTimes(1);
    });
});

describe("getComponentNameFromStack", () => {
    it("Should return Unknown Component when stack is undefined", () => {
        expect(getComponentNameFromStack(undefined)).toBe("Unknown Component");
    });

    it("Should return the component name when stack contains it", () => {
        const fakeStack =
            "Error: something\n    at MyComponent (index.tsx:10:5)";
        expect(getComponentNameFromStack(fakeStack)).toBe("MyComponent");
    });

    it("Should return Unknown Component when stack does not match", () => {
        const fakeStack = "Error: something\n    random line without match";
        expect(getComponentNameFromStack(fakeStack)).toBe("Unknown Component");
    });
});
