// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Meta, StoryObj } from "@storybook/react-vite";
import { expect, userEvent, within } from "storybook/test";
import { useEffect, useMemo, useState } from "react";

const meta: Meta<typeof Alert> = {
    title: "UI/Alert",
    component: Alert,
    tags: ["autodocs"],
    argTypes: {
        variant: {
            control: { type: "select" },
            options: ["default", "destructive", "warn"]
        },
        enableClose: { control: "boolean", options: [true, false] }
    }
};

export default meta;

type Story = StoryObj<typeof Alert>;

export const Default: Story = {
    args: {
        variant: "default",
        onClose: () => {
            console.log("Alert closed");
        },
        children: (
            <>
                <AlertTitle>Information</AlertTitle>
                <AlertDescription>
                    This is a standard alert message.
                </AlertDescription>
            </>
        )
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const alert = canvas.getByRole("alert");
        await expect(alert).toBeVisible();
        await expect(canvas.getByText("Information")).toBeVisible();
        await expect(
            canvas.getByText("This is a standard alert message.")
        ).toBeVisible();

        const closeButton = canvas.getByTestId("alert-close");
        await userEvent.click(closeButton);
        await expect(canvas.queryByRole("alert")).not.toBeInTheDocument();
    }
};

export const Warn: Story = {
    args: {
        variant: "warn",
        children: (
            <>
                <AlertTitle>Warning</AlertTitle>
                <AlertDescription>
                    This is a warning alert message.
                </AlertDescription>
            </>
        )
    }
};

export const Destructive: Story = {
    args: {
        variant: "destructive",
        children: (
            <>
                <AlertTitle>Error</AlertTitle>
                <AlertDescription>
                    Your session has expired. Please log in again.
                </AlertDescription>
            </>
        )
    }
};

export const Title_Only: Story = {
    args: {
        variant: "warn",
        children: (
            <>
                <AlertTitle>Warning Alert with Title only.</AlertTitle>
            </>
        )
    }
};

export const Description_Only: Story = {
    args: {
        variant: "destructive",
        children: (
            <>
                <AlertDescription>
                    Error Alert with Description only.
                </AlertDescription>
            </>
        )
    }
};

export const Alert_without_Close_Button: Story = {
    args: {
        variant: "default",
        enableClose: false,
        children: (
            <>
                <AlertTitle>Alert</AlertTitle>
                <AlertDescription>This is an Alert message.</AlertDescription>
            </>
        )
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByRole("alert")).toBeVisible();
        await expect(
            canvas.queryByTestId("alert-close")
        ).not.toBeInTheDocument();
    }
};

export const Alert_with_Scrollbar: Story = {
    args: {
        variant: "destructive",
        enableClose: false,
        children: (
            <>
                <AlertTitle>Lengthy Alert</AlertTitle>
                <AlertDescription>
                    Traceback (most recent call last): File
                    "/home/user/project/venv/lib/python3.10/site-packages/anyio/streams/memory.py",
                    line 98, in receive return self.receive_nowait() File
                    "/home/user/project/venv/lib/python3.10/site-packages/anyio/streams/memory.py",
                    line 93, in receive_nowait raise WouldBlock anyio.WouldBlock
                    During handling of the above exception, another exception
                    occurred: Traceback (most recent call last): File
                    "/home/user/project/venv/lib/python3.10/site-packages/starlette/middleware/base.py",
                    line 43, in call_next message = await recv_stream.receive()
                    File
                    "/home/user/project/venv/lib/python3.10/site-packages/anyio/streams/memory.py",
                    line 118, in receive raise EndOfStream anyio.EndOfStream
                    During handling of the above exception, another exception
                    occurred: Traceback (most recent call last): File
                    "/home/user/project/venv/lib/python3.10/site-packages/starlette/middleware/errors.py",
                    line 162, in __call__ await self.app(scope, receive, _send)
                    File
                    "/home/user/project/venv/lib/python3.10/site-packages/opentelemetry/instrumentation/asgi/__init__.py",
                    line 606, in __call__ await self.app(scope, otel_receive,
                    otel_send) File
                    "/home/user/project/venv/lib/python3.10/site-packages/starlette/middleware/base.py",
                    line 68, in __call__ response = await
                    self.dispatch_func(request, call_next) File
                    "/home/user/project/rest/utils/security_header_middleware.py",
                    line 10, in dispatch response = await call_next(request)
                    File
                    "/home/user/project/venv/lib/python3.10/site-packages/starlette/middleware/base.py",
                    line 46, in call_next raise app_exc File
                    "/home/user/project/venv/lib/python3.10/site-packages/starlette/middleware/base.py",
                    line 36, in coro await self.app(scope, request.receive,
                    send_stream.send) File
                    "/home/user/project/rest/utils/profiling_middleware.py",
                    line 66, in __call__ return await self._app(scope, receive,
                    send) File
                    "/home/user/project/rest/utils/strip_header_middleware.py",
                    line 43, in __call__ return await self._app(scope, receive,
                    send_wrapper) File
                    "/home/user/project/venv/lib/python3.10/site-packages/starlette/middleware/base.py",
                    line 68, in __call__ response = await
                    self.dispatch_func(request, call_next) File
                    "/home/user/project/rest/utils/request_time_middleware.py",
                    line 56, in dispatch resp = await call_next(request) File
                    "/home/user/project/venv/lib/python3.10/site-packages/starlette/middleware/base.py",
                    line 46, in call_next raise app_exc File
                    "/home/user/project/venv/lib/python3.10/site-packages/starlette/middleware/base.py",
                    line 36, in coro await self.app(scope, request.receive,
                    send_stream.send)
                </AlertDescription>
            </>
        )
    }
};

const LargeWarnings = () => {
    const warnings = useMemo(
        () => [
            {
                title: "Sub-title 1",
                description: "This is description 1"
            },
            {
                title: "Sub-title 2",
                description: "This is description 2"
            },
            {
                title: "Sub-title 3",
                description: "This is description 3"
            },
            {
                title: "Sub-title 4",
                description: "This is description 4"
            },
            {
                title: "Sub-title 5",
                description: "This is description 5"
            }
        ],
        []
    );

    const showButton = warnings?.length > 2 ? true : false;
    const [showMore, setShowMore] = useState<boolean>(false);
    const [slicedWarnings, setWarnings] = useState(
        showMore ? warnings : warnings?.slice(0, 2)
    );

    useEffect(() => {
        setWarnings(showMore ? warnings : warnings?.slice(0, 2));
    }, [showMore, warnings]);

    return (
        <>
            {slicedWarnings.map((w, windx) => (
                <div key={"warn-cta-list" + windx}>
                    {w?.title !== "" && (
                        <span className="font-semibold">{w.title}:</span>
                    )}
                    <span> {w.description} </span>
                </div>
            ))}
            {showButton && (
                <Button
                    className="p-0"
                    onClick={() => {
                        setShowMore(!showMore);
                    }}
                    variant="link">
                    {showMore ? "Show Less" : "Show More"}
                </Button>
            )}
        </>
    );
};

export const Alert_with_Show_More: Story = {
    args: {
        variant: "warn",
        children: (
            <>
                <AlertTitle>Warning</AlertTitle>
                <AlertDescription>
                    <LargeWarnings />
                </AlertDescription>
            </>
        )
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByRole("alert")).toBeVisible();

        await expect(canvas.getByText(/Sub-title 1/)).toBeVisible();
        await expect(canvas.queryByText(/Sub-title 3/)).not.toBeInTheDocument();
        const showMoreButton = canvas.getByRole("button", {
            name: /show more/i
        });
        await expect(showMoreButton).toBeVisible();

        await userEvent.click(showMoreButton);
        await expect(canvas.getByText(/Sub-title 3/)).toBeVisible();
        await expect(canvas.getByText(/Sub-title 4/)).toBeVisible();

        const showLessButton = canvas.getByRole("button", {
            name: /show less/i
        });
        await expect(showLessButton).toBeVisible();

        await userEvent.click(showLessButton);
        await expect(canvas.queryByText(/Sub-title 3/)).not.toBeInTheDocument();
    }
};
