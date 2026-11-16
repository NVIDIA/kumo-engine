// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button } from "@/components/ui/button";
import { Description } from "@/components/ui/description";
import { StartJobPopover } from "@/components/ui/start-job-popover";
import { Meta, StoryObj } from "@storybook/react-vite";
import { useState } from "react";
import { expect, fn, screen, userEvent, waitFor, within } from "storybook/test";

const startTraining = fn();

const meta: Meta<typeof StartJobPopover> = {
    title: "UI/StartJobPopover",
    component: StartJobPopover,
    tags: ["autodocs"],
    argTypes: {}
};

export default meta;
type Story = StoryObj<typeof StartJobPopover>;

export const Training: Story = {
    render: function DefaultFunc() {
        const [open, setOpen] = useState(false);
        const jobs: string[] = [
            "Training Table Generation",
            "Training job",
            "Snapshot"
        ];

        return (
            <div className="relative flex flex-col items-center">
                <StartJobPopover
                    showPopover={open}
                    setShowPopover={setOpen}
                    startButtonTitle={"Start training "}
                    jobsSectionTitle="Training jobs"
                    headerLabel={
                        "The following jobs will be created when you start training."
                    }
                    tagHeader="Tags"
                    tagDescription={
                        "Add key-value tags to organize and filter this training job. Tags will be applied to both jobs."
                    }
                    initialTags={{ bp: "bp" }}
                    jobs={jobs}
                    startTraining={startTraining}>
                    <div role="alert">
                        <Button onClick={() => setOpen(!open)}>
                            Start training
                        </Button>
                    </div>
                </StartJobPopover>
            </div>
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);

        // Open the popover
        await userEvent.click(canvas.getByText("Start training"));

        await waitFor(() => {
            expect(screen.getByText("Training jobs")).toBeVisible();
            expect(
                screen.getByText(
                    "The following jobs will be created when you start training."
                )
            ).toBeVisible();
            expect(screen.getByText("Training Table Generation")).toBeVisible();
            expect(screen.getByText("Training job")).toBeVisible();
            expect(screen.getByText("Snapshot")).toBeVisible();
            expect(screen.getByText("Tags")).toBeVisible();
            expect(screen.getByText("bp=")).toBeVisible();
            expect(screen.getByText("bp")).toBeVisible();
        });

        // Add a tag by clicking "+ Add tag"
        await userEvent.click(screen.getByText("Add tag"));

        const keyInput = screen.getByPlaceholderText("key");
        const valueInput = screen.getByPlaceholderText("value");
        await userEvent.type(keyInput, "env");
        await userEvent.type(valueInput, "production");
        await userEvent.keyboard("{Enter}");

        await waitFor(() => {
            expect(screen.getByText("env=")).toBeVisible();
            expect(screen.getByText("production")).toBeVisible();
        });

        // Add a second tag
        await userEvent.click(screen.getByText("Add tag"));
        await userEvent.type(screen.getByPlaceholderText("key"), "team");
        await userEvent.type(screen.getByPlaceholderText("value"), "ml");
        await userEvent.keyboard("{Enter}");

        await waitFor(() => {
            expect(screen.getByText("team=")).toBeVisible();
            expect(screen.getByText("bp=")).toBeVisible();
        });

        // Remove the first tag
        const removeButtons = screen.getAllByTestId("tag-remove-icon");
        await userEvent.click(removeButtons[0]);
        await waitFor(() => {
            expect(screen.queryByText("bp=")).toBeNull();
        });

        // Submit
        await waitFor(() => {
            expect(document.getElementById("start-job")).toBeTruthy();
        });
        await userEvent.click(document.getElementById("start-job")!);
        await waitFor(() => {
            expect(startTraining).toHaveBeenCalledWith({
                env: "production",
                team: "ml"
            });
            expect(screen.queryByText("Training jobs")).toBeNull();
        });
    }
};

export const Prediction: Story = {
    render: function DefaultFunc() {
        const [open, setOpen] = useState(false);
        const jobs: string[] = ["Prediction Table Generation", "Prediction"];

        return (
            <div className="relative flex flex-col items-center">
                <StartJobPopover
                    showPopover={open}
                    setShowPopover={setOpen}
                    startButtonTitle={"Start predicting "}
                    jobsSectionTitle="Prediction jobs"
                    headerLabel={
                        "The following jobs will be created when you start predicting."
                    }
                    tagHeader="Tags"
                    tagDescription={
                        "Add key-value tags to organize and filter this prediction job. Tags will be applied to both jobs."
                    }
                    jobs={jobs}
                    startTraining={(tags) => {
                        console.log("Tags", tags);
                    }}>
                    <div role="alert">
                        <Button onClick={() => setOpen(!open)}>
                            Start predicting
                        </Button>
                    </div>
                </StartJobPopover>
            </div>
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);

        await userEvent.click(canvas.getByText("Start predicting"));

        await waitFor(() => {
            expect(screen.getByText("Prediction jobs")).toBeVisible();
            expect(
                screen.getByText("Prediction Table Generation")
            ).toBeVisible();
            expect(screen.getByText("Prediction")).toBeVisible();
        });

        // Dismiss with Escape
        await userEvent.keyboard("{Escape}");
        await waitFor(() => {
            expect(
                screen.queryByText("Prediction jobs")
            ).not.toBeInTheDocument();
        });
    }
};

export const Customised: Story = {
    render: function DefaultFunc() {
        const [open, setOpen] = useState(false);
        const jobs: string[] = [
            "TRAINING TABLE GENERATION",
            "TRAINING",
            "SNAPSHOT"
        ];

        return (
            <div className="relative flex flex-col items-center">
                <StartJobPopover
                    showPopover={open}
                    setShowPopover={setOpen}
                    startButtonTitle={"Start Job"}
                    jobsSectionTitle={
                        <Description className="font-bold leading-5 text-purple-500">
                            {"Custom Job Title"}
                        </Description>
                    }
                    headerLabel={
                        <Description className="leading-5 text-purple-400">
                            {"Custom subtitle for the jobs section."}
                        </Description>
                    }
                    tagHeader={
                        <Description className="font-bold leading-5 text-fuchsia-500">
                            {"Tags"}
                        </Description>
                    }
                    tagDescription={
                        <Description className="leading-5 text-orange-500">
                            {
                                "Custom tag description with long text to adjust popover dialog width."
                            }
                        </Description>
                    }
                    jobs={jobs}
                    popoverContentClassName="w-100"
                    startTraining={(tags) => {
                        console.log("Tags", tags);
                    }}>
                    <div role="alert">
                        <Button onClick={() => setOpen(!open)}>
                            Toggle Popover
                        </Button>
                    </div>
                </StartJobPopover>
            </div>
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await userEvent.click(canvas.getByText("Toggle Popover"));
        await waitFor(() => {
            expect(screen.getByText("Custom Job Title")).toBeVisible();
            expect(screen.getByText("TRAINING")).toBeVisible();
            expect(screen.getByText("SNAPSHOT")).toBeVisible();
        });
        await userEvent.keyboard("{Escape}");
        await waitFor(() => {
            expect(
                screen.queryByText("Custom Job Title")
            ).not.toBeInTheDocument();
        });
    }
};
