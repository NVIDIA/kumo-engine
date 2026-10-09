// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Calendar } from "@/components/ui/calendar";
import { Card, CardContent } from "@/components/ui/card";
import {
    Command,
    CommandEmpty,
    CommandGroup,
    CommandInput,
    CommandItem,
    CommandList
} from "@/components/ui/command";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
    Popover,
    PopoverContent,
    PopoverTrigger
} from "@/components/ui/popover";
import { Separator } from "@/components/ui/separator";
import { Check } from "@/icons/MaterialIcons/Check";
import { ChevronDown } from "@/icons/MaterialIcons/ChevronDown";
import { ChevronRight } from "@/icons/MaterialIcons/ChevronRight";
import { CloseSmall } from "@/icons/MaterialIcons/CloseSmall";
import { DeveloperBoard } from "@/icons/MaterialIcons/DeveloperBoard";
import { LabelIcon } from "@/icons/MaterialIcons/LabelIcon";
import { cn } from "@/lib/utils";
import { Meta, StoryObj } from "@storybook/react-vite";
import { expect, userEvent, waitFor, within } from "storybook/test";
import { format } from "date-fns";

const meta = {
    title: "UI/Popover",
    component: Popover,
    parameters: {
        layout: "centered"
    },
    tags: ["autodocs"]
} satisfies Meta<typeof Popover>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
    render: () => (
        <Popover>
            <PopoverTrigger asChild>
                <Button variant="light-outline">Open Popover</Button>
            </PopoverTrigger>
            <PopoverContent>
                <p>This is the popover content.</p>
            </PopoverContent>
        </Popover>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);

        await userEvent.click(
            canvas.getByRole("button", { name: /open popover/i })
        );
        const body = within(document.body);
        await waitFor(() => {
            expect(
                body.getByText("This is the popover content.")
            ).toBeVisible();
        });
        await userEvent.keyboard("{Escape}");
        await waitFor(() => {
            expect(
                body.queryByText("This is the popover content.")
            ).not.toBeInTheDocument();
        });
    }
};

export const WithForm: Story = {
    render: () => (
        <Popover>
            <PopoverTrigger asChild>
                <Button variant="light-outline">Edit Settings</Button>
            </PopoverTrigger>
            <PopoverContent className="w-80">
                <div className="grid gap-4">
                    <div className="space-y-2">
                        <h4 className="font-medium leading-none">Dimensions</h4>
                        <p className="text-muted-foreground text-sm">
                            Set the dimensions for the layer.
                        </p>
                    </div>
                    <div className="grid gap-2">
                        <div className="grid grid-cols-3 items-center gap-4">
                            <Label htmlFor="width">Width</Label>
                            <Input
                                id="width"
                                defaultValue="100%"
                                className="col-span-2 h-8"
                            />
                        </div>
                        <div className="grid grid-cols-3 items-center gap-4">
                            <Label htmlFor="height">Height</Label>
                            <Input
                                id="height"
                                defaultValue="25px"
                                className="col-span-2 h-8"
                            />
                        </div>
                    </div>
                </div>
            </PopoverContent>
        </Popover>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);

        await userEvent.click(
            canvas.getByRole("button", { name: /edit settings/i })
        );
        const body = within(document.body);
        await waitFor(() => {
            expect(body.getByLabelText("Width")).toBeVisible();
            expect(body.getByLabelText("Height")).toBeVisible();
        });
        await userEvent.keyboard("{Escape}");
        await waitFor(() => {
            expect(body.queryByLabelText("Width")).not.toBeInTheDocument();
        });
    }
};

export const DatePicker: Story = {
    render: function DatePickerStory() {
        const [open, setOpen] = useState(false);
        const [date, setDate] = useState<Date | undefined>(new Date());

        return (
            <Popover open={open} onOpenChange={setOpen}>
                <PopoverTrigger asChild>
                    <button
                        type="button"
                        aria-label="Select date"
                        className={cn(
                            "w-[240px] items-center justify-between",
                            "inline-flex cursor-pointer gap-2 whitespace-nowrap rounded-sm font-medium",
                            "shadow-xs border border-slate-200 bg-white hover:bg-slate-100",
                            "h-8 px-3 py-1 text-sm",
                            date ? "text-slate-800" : "text-slate-500"
                        )}
                        onClick={() => setOpen(!open)}>
                        <div className="flex items-center overflow-hidden text-ellipsis">
                            {date
                                ? format(date, "yyyy-MM-dd HH:mm:ss")
                                : "Select date..."}
                        </div>
                        <ChevronDown className="size-5" />
                    </button>
                </PopoverTrigger>
                <PopoverContent align="start" className="w-auto p-0">
                    <Calendar
                        mode="single"
                        selected={date}
                        onSelect={(selectedDate) => {
                            if (selectedDate) {
                                setDate(selectedDate);
                            }
                        }}
                        initialFocus
                    />
                </PopoverContent>
            </Popover>
        );
    }
};

export const ComboboxSelect: Story = {
    render: function ComboboxSelectStory() {
        const [open, setOpen] = useState(false);
        const [value, setValue] = useState("");

        const options = [
            { value: "test1", label: "test1" },
            { value: "test2", label: "test2" },
            { value: "test3", label: "test3" },
            { value: "test4", label: "test4" }
        ];

        const selectedOption = options.find((opt) => opt.value === value);

        return (
            <Popover open={open} onOpenChange={setOpen}>
                <PopoverTrigger asChild>
                    <Button
                        variant="light-outline"
                        role="combobox"
                        aria-expanded={open}
                        aria-label="Select tags"
                        className={cn(
                            "w-[200px] justify-between font-normal",
                            !value && "text-muted-foreground"
                        )}>
                        {selectedOption ? selectedOption.label : "Any Tags"}
                        <ChevronDown className="ml-2 size-4 shrink-0 opacity-50" />
                    </Button>
                </PopoverTrigger>
                <PopoverContent className="w-[200px] p-0" align="start">
                    <Command>
                        <CommandInput placeholder="Search..." />
                        <CommandList>
                            <CommandEmpty>No options found.</CommandEmpty>
                            <CommandGroup>
                                {options.map((option) => (
                                    <CommandItem
                                        key={option.value}
                                        value={option.value}
                                        onSelect={() => {
                                            setValue(option.value);
                                            setOpen(false);
                                        }}>
                                        <Check
                                            className={cn(
                                                "mr-2 size-4",
                                                value === option.value
                                                    ? "opacity-100"
                                                    : "opacity-0"
                                            )}
                                        />
                                        {option.label}
                                    </CommandItem>
                                ))}
                            </CommandGroup>
                        </CommandList>
                    </Command>
                </PopoverContent>
            </Popover>
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);

        await userEvent.click(
            canvas.getByRole("combobox", { name: /select tags/i })
        );
        const body = within(document.body);
        await waitFor(() => {
            expect(body.getByText("test1")).toBeVisible();
        });

        await userEvent.click(body.getByText("test2"));
        await waitFor(() => {
            expect(canvas.getByRole("combobox")).toHaveTextContent("test2");
            expect(body.queryByText("test1")).not.toBeInTheDocument();
        });
    }
};

export const MultiSelect: Story = {
    render: function MultiSelectStory() {
        const [open, setOpen] = useState(false);
        const [selected, setSelected] = useState<string[]>(["done"]);

        const options = [
            { value: "done", label: "Done" },
            { value: "failed", label: "Qailed" },
            { value: "queued", label: "Queued" },
            { value: "canceled", label: "Canceled" }
        ];

        const toggleValue = (value: string) => {
            setSelected((prev) =>
                prev.includes(value)
                    ? prev.filter((v) => v !== value)
                    : [...prev, value]
            );
        };

        return (
            <Popover open={open} onOpenChange={setOpen} modal={false}>
                <PopoverTrigger asChild>
                    <button
                        type="button"
                        aria-label="Select status"
                        className={cn(
                            "inline-flex w-[200px] items-center justify-between",
                            "cursor-pointer gap-2 rounded-sm font-medium",
                            "shadow-xs border border-slate-200 bg-white hover:bg-slate-100",
                            "h-8 px-3 py-1 text-sm",
                            selected.length === 0 && "text-slate-500"
                        )}>
                        <span>
                            {selected.length > 0
                                ? `${selected.length} selected`
                                : "Any Status"}
                        </span>
                        <ChevronDown className="size-5" />
                    </button>
                </PopoverTrigger>
                <PopoverContent className="w-[200px] p-0" align="start">
                    <Command>
                        <CommandList>
                            <CommandEmpty>No results.</CommandEmpty>
                            <CommandGroup>
                                {options.map((option) => {
                                    const isSelected = selected.includes(
                                        option.value
                                    );
                                    return (
                                        <CommandItem
                                            key={option.value}
                                            value={option.value}
                                            onSelect={() =>
                                                toggleValue(option.value)
                                            }
                                            className={cn(
                                                "cursor-pointer justify-between font-medium",
                                                isSelected
                                                    ? "text-slate-800"
                                                    : "text-slate-500"
                                            )}>
                                            <span>{option.label}</span>
                                            <Check
                                                className={cn(
                                                    "ml-2 size-4",
                                                    isSelected
                                                        ? "text-green-500"
                                                        : "opacity-0"
                                                )}
                                            />
                                        </CommandItem>
                                    );
                                })}
                            </CommandGroup>
                        </CommandList>
                    </Command>
                </PopoverContent>
            </Popover>
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);

        await userEvent.click(
            canvas.getByRole("button", { name: /select status/i })
        );
        const body = within(document.body);
        await waitFor(() => {
            expect(body.getByText("Done")).toBeVisible();
        });

        await userEvent.click(body.getByText("Queued"));
        await waitFor(() => {
            expect(
                canvas.getByRole("button", { name: /select status/i })
            ).toHaveTextContent("2 selected");
        });
        await userEvent.keyboard("{Escape}");
        await waitFor(() => {
            expect(body.queryByText("Done")).not.toBeInTheDocument();
        });
    }
};

export const StartPrediction: Story = {
    render: function StartPredictionStory() {
        const [open, setOpen] = useState(false);
        const [tags, setTags] = useState<string[]>([]);
        const [inputValue, setInputValue] = useState("");
        const [validationError, setValidationError] = useState<string | null>(
            null
        );

        const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
            if (
                (e.key === "Enter" || e.key === "," || e.key === "Tab") &&
                inputValue?.trim()
            ) {
                e.preventDefault();
                const trimmedInput = inputValue?.trim().replace(/,$/, "");

                if (validationError || !trimmedInput) return;

                if (!tags.includes(trimmedInput)) {
                    setTags([...tags, trimmedInput]);
                }
                setInputValue("");
                setValidationError(null);
            } else if (
                e.key === "Backspace" &&
                inputValue === "" &&
                tags.length > 0
            ) {
                setTags(tags.slice(0, -1));
            }
        };

        const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
            const value = e.target.value;
            setInputValue(value);
            const trimmedValue = value?.trim();
            if (/^[a-zA-Z0-9_-]*$/.test(trimmedValue)) {
                setValidationError(null);
            } else {
                setValidationError(
                    "Tags can only contain alphanumeric characters, underscores (_), or hyphens (-)"
                );
            }
        };

        const removeTag = (index: number) => {
            setTags(tags.filter((_, i) => i !== index));
        };

        const jobs = ["PREDICTION TABLE GENERATION", "PREDICTION"];

        return (
            <Popover open={open} onOpenChange={setOpen} modal={true}>
                <PopoverTrigger asChild>
                    <Button>
                        Start predicting
                        <ChevronRight className="fill-white" />
                    </Button>
                </PopoverTrigger>
                <PopoverContent className="w-fit p-3" align="end" side="bottom">
                    <p className="pb-2 text-xs font-medium">
                        Following jobs will be created for this prediction
                    </p>

                    <Card className="items-start gap-0 rounded-sm py-2">
                        {jobs.map((job, i) => (
                            <div className="w-full" key={`job-${i}`}>
                                <CardContent>
                                    <div className="flex w-full items-center gap-2">
                                        <DeveloperBoard className="size-4" />
                                        <span className="text-xs font-bold text-slate-600">
                                            {job}
                                        </span>
                                    </div>
                                </CardContent>
                                {i < jobs.length - 1 && (
                                    <Separator className="my-2" />
                                )}
                            </div>
                        ))}
                    </Card>

                    <p className="py-2 text-xs font-medium">Tags</p>

                    <Card className="items-start rounded-sm px-1 py-1">
                        <CardContent className="px-1 py-1">
                            <div className="flex w-[300px] flex-wrap items-center gap-2">
                                {tags.map((tag, index) => (
                                    <Badge
                                        key={`badge-${index}`}
                                        variant="default"
                                        size="sm"
                                        className="flex items-center gap-1">
                                        {tag}
                                        <Button
                                            variant="link"
                                            className="p-0"
                                            size="xs"
                                            aria-label={`Remove ${tag} tag`}
                                            onClick={() => removeTag(index)}>
                                            <CloseSmall className="!size-3.5 !fill-black" />
                                        </Button>
                                    </Badge>
                                ))}
                            </div>

                            <div className="relative">
                                <LabelIcon className="text-muted-foreground absolute left-1 top-1/2 size-4 -translate-y-1/2" />
                                <Input
                                    className="border-none pl-8 shadow-none focus-visible:border-transparent focus-visible:ring-0"
                                    placeholder={tags.length ? "" : "Add tags"}
                                    aria-label="Add tags"
                                    value={inputValue}
                                    onChange={handleInputChange}
                                    onKeyDown={handleKeyDown}
                                />
                            </div>
                        </CardContent>
                    </Card>

                    {validationError && (
                        <p className="w-[320px] pb-0 pt-2 text-xs font-normal text-red-500">
                            {validationError}
                        </p>
                    )}

                    <p className="w-[320px] py-2 text-xs font-medium leading-5 text-slate-500">
                        Tags will be applied to all jobs created for this
                        prediction. Add multiple tags by pressing Enter after
                        each one.
                    </p>

                    <Button
                        onClick={() => {
                            const tmpTags = [...tags];
                            const trimmedInput = inputValue?.trim();

                            if (
                                trimmedInput !== "" &&
                                !tags.includes(trimmedInput)
                            ) {
                                tmpTags.push(trimmedInput);
                            }
                            console.log(
                                "Starting prediction with tags:",
                                tmpTags
                            );
                            setOpen(false);
                        }}
                        disabled={!!validationError}>
                        <span className="text-sm">Start predicting</span>
                        <ChevronRight />
                    </Button>
                </PopoverContent>
            </Popover>
        );
    }
};
