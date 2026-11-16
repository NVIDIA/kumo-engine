// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// src/components/ui/combobox.stories.tsx
import { Button } from "@/components/ui/button";
import { Combobox, ComboboxOption } from "@/components/ui/combobox";
import { Label } from "@/components/ui/label";
import type { Meta, StoryObj } from "@storybook/react-vite";
import { expect, userEvent, waitFor, within } from "storybook/test";
import {
    ChevronsUpDown,
    Globe,
    Laptop,
    Settings,
    Smartphone,
    User
} from "lucide-react";
import { SetStateAction, useState } from "react";

// Helper imports for custom examples
import {
    Command,
    CommandEmpty,
    CommandGroup,
    CommandInput,
    CommandItem,
    CommandList,
    CommandSeparator
} from "@/components/ui/command";
import {
    Popover,
    PopoverContent,
    PopoverTrigger
} from "@/components/ui/popover";
import { Check } from "@/icons";
import { cn } from "@/lib/utils";

const meta: Meta<typeof Combobox> = {
    title: "UI/Combobox",
    component: Combobox,
    tags: ["autodocs"],
    argTypes: {
        value: { control: "text" },
        placeholder: { control: "text" },
        emptyText: { control: "text" },
        disabled: { control: "boolean" },
        searchPlaceholder: { control: "text" }
    }
};

export default meta;
type Story = StoryObj<typeof Combobox>;

// Sample data
const frameworks: ComboboxOption[] = [
    { value: "react", label: "React" },
    { value: "vue", label: "Vue" },
    { value: "svelte", label: "Svelte" },
    { value: "angular", label: "Angular" },
    { value: "solid", label: "Solid" },
    { value: "nextjs", label: "Next.js" },
    { value: "nuxt", label: "Nuxt.js" }
];

const countries: ComboboxOption[] = [
    { value: "us", label: "United States" },
    { value: "ca", label: "Canada" },
    { value: "mx", label: "Mexico" },
    { value: "br", label: "Brazil" },
    { value: "uk", label: "United Kingdom" },
    { value: "fr", label: "France" },
    { value: "de", label: "Germany" },
    { value: "it", label: "Italy" },
    { value: "es", label: "Spain" },
    { value: "cn", label: "China" },
    { value: "jp", label: "Japan" },
    { value: "in", label: "India" },
    { value: "au", label: "Australia" }
];

// Basic Combobox
export const Basic: Story = {
    args: {
        options: frameworks,
        placeholder: "Select a framework"
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const trigger = canvas.getByRole("combobox", {
            name: /select a framework/i
        });

        await userEvent.click(trigger);
        const body = within(document.body);
        await waitFor(() => {
            expect(body.getByText("React")).toBeVisible();
        });

        await userEvent.click(body.getByText("Svelte"));
        await waitFor(() => {
            expect(trigger).toHaveTextContent("Svelte");
            expect(body.queryByText("React")).not.toBeInTheDocument();
        });
    }
};

// With Default Value
export const WithDefaultValue: Story = {
    args: {
        options: frameworks,
        value: "react",
        placeholder: "Select a framework"
    }
};

// Disabled Combobox
export const Disabled: Story = {
    args: {
        options: frameworks,
        value: "react",
        placeholder: "Select a framework",
        disabled: true
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const trigger = canvas.getByRole("combobox", {
            name: /select a framework/i
        });
        await expect(trigger).toBeDisabled();
        await expect(trigger).toHaveTextContent("React");
    }
};

// With Label
export const WithLabel: Story = {
    render: () => (
        <div className="grid w-full max-w-sm items-center gap-1.5">
            <Label htmlFor="framework">Framework</Label>
            <Combobox options={frameworks} placeholder="Select a framework" />
        </div>
    )
};

// With Form Layout
export const WithFormLayout: Story = {
    render: function FormLayoutExample() {
        const [framework, setFramework] = useState("react");

        return (
            <div className="w-full max-w-md space-y-4 rounded-lg border p-6">
                <h3 className="text-lg font-medium">Project Settings</h3>
                <div className="space-y-4">
                    <div className="grid w-full items-center gap-1.5">
                        <Label htmlFor="framework">Framework</Label>
                        <Combobox
                            options={frameworks}
                            value={framework}
                            onValueChange={setFramework}
                            placeholder="Select a framework"
                        />
                        <p className="text-muted-foreground text-sm">
                            The JavaScript framework used in your project.
                        </p>
                    </div>
                    <div className="flex justify-end pt-4">
                        <Button>Save Settings</Button>
                    </div>
                </div>
            </div>
        );
    }
};

// With Disabled Options
export const WithDisabledOptions: Story = {
    render: () => {
        const options = [
            { value: "apple", label: "Apple" },
            { value: "banana", label: "Banana" },
            { value: "orange", label: "Orange", disabled: true },
            { value: "strawberry", label: "Strawberry" },
            { value: "pineapple", label: "Pineapple", disabled: true }
        ];

        return <Combobox options={options} placeholder="Select a fruit" />;
    }
};

// Controlled Combobox
export const Controlled: Story = {
    render: function ControlledExample() {
        const [value, setValue] = useState("");

        return (
            <div className="w-full max-w-sm space-y-4">
                <Combobox
                    options={frameworks}
                    value={value}
                    onValueChange={setValue}
                    placeholder="Select a framework"
                />
                <div className="rounded-md border p-4">
                    <p className="text-sm">
                        Selected value: {value ? value : "none"}
                    </p>
                    <div className="mt-2 flex gap-2">
                        <Button
                            variant="light-outline"
                            size="sm"
                            onClick={() => setValue("react")}>
                            Set to React
                        </Button>
                        <Button
                            variant="light-outline"
                            size="sm"
                            onClick={() => setValue("vue")}>
                            Set to Vue
                        </Button>
                        <Button
                            variant="light-outline"
                            size="sm"
                            onClick={() => setValue("")}>
                            Clear
                        </Button>
                    </div>
                </div>
            </div>
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByText("Selected value: none")).toBeVisible();

        await userEvent.click(
            canvas.getByRole("button", { name: /set to react/i })
        );
        await expect(canvas.getByText("Selected value: react")).toBeVisible();

        await userEvent.click(
            canvas.getByRole("button", { name: /set to vue/i })
        );
        await expect(canvas.getByText("Selected value: vue")).toBeVisible();

        await userEvent.click(canvas.getByRole("button", { name: /clear/i }));
        await expect(canvas.getByText("Selected value: none")).toBeVisible();
    }
};

// Long List
export const LongList: Story = {
    args: {
        options: countries,
        placeholder: "Select a country",
        popoverClassName: "w-[300px]"
    }
};

// Custom Empty State
export const CustomEmptyState: Story = {
    args: {
        options: frameworks,
        placeholder: "Select a framework",
        emptyText: "No frameworks match your search. Try a different term."
    }
};

// Multi-section Combobox
export const MultiSection: Story = {
    render: function MultiSectionExample() {
        // Define our custom implementation for multi-section combobox
        const [open, setOpen] = useState(false);
        const [value, setValue] = useState("");

        const frameworks = [
            { value: "react", label: "React", type: "library" },
            { value: "vue", label: "Vue", type: "library" },
            { value: "svelte", label: "Svelte", type: "library" },
            { value: "angular", label: "Angular", type: "framework" },
            { value: "nextjs", label: "Next.js", type: "framework" },
            { value: "nuxt", label: "Nuxt.js", type: "framework" },
            { value: "remix", label: "Remix", type: "framework" }
        ];

        const libraries = frameworks.filter((item) => item.type === "library");
        const fullFrameworks = frameworks.filter(
            (item) => item.type === "framework"
        );

        const selectedItem = frameworks.find((item) => item.value === value);

        return (
            <div className="w-full max-w-sm">
                <Popover open={open} onOpenChange={setOpen}>
                    <PopoverTrigger asChild>
                        <Button
                            variant="light-outline"
                            role="combobox"
                            aria-expanded={open}
                            aria-label="Select framework"
                            className="w-full justify-between font-normal">
                            {selectedItem
                                ? selectedItem.label
                                : "Select framework"}
                            <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
                        </Button>
                    </PopoverTrigger>
                    <PopoverContent className="w-[250px] p-0">
                        <Command>
                            <CommandInput placeholder="Search frameworks..." />
                            <CommandList>
                                <CommandEmpty>No results found.</CommandEmpty>
                                <CommandGroup heading="Libraries">
                                    {libraries.map((item) => (
                                        <CommandItem
                                            key={item.value}
                                            value={item.value}
                                            onSelect={(
                                                currentValue: SetStateAction<string>
                                            ) => {
                                                setValue(currentValue);
                                                setOpen(false);
                                            }}>
                                            <Check
                                                className={cn(
                                                    "mr-2 size-4",
                                                    value === item.value
                                                        ? "opacity-100"
                                                        : "opacity-0"
                                                )}
                                            />
                                            {item.label}
                                        </CommandItem>
                                    ))}
                                </CommandGroup>
                                <CommandSeparator />
                                <CommandGroup heading="Frameworks">
                                    {fullFrameworks.map((item) => (
                                        <CommandItem
                                            key={item.value}
                                            value={item.value}
                                            onSelect={(
                                                currentValue: SetStateAction<string>
                                            ) => {
                                                setValue(currentValue);
                                                setOpen(false);
                                            }}>
                                            <Check
                                                className={cn(
                                                    "mr-2 size-4",
                                                    value === item.value
                                                        ? "opacity-100"
                                                        : "opacity-0"
                                                )}
                                            />
                                            {item.label}
                                        </CommandItem>
                                    ))}
                                </CommandGroup>
                            </CommandList>
                        </Command>
                    </PopoverContent>
                </Popover>
            </div>
        );
    }
};

// With Custom Styling
export const CustomStyled: Story = {
    render: () => (
        <div className="grid w-full max-w-sm gap-6">
            <div className="space-y-2">
                <Label>Default Style</Label>
                <Combobox
                    options={frameworks}
                    placeholder="Select a framework"
                />
            </div>

            <div className="space-y-2">
                <Label>Primary Style</Label>
                <Combobox
                    options={frameworks}
                    placeholder="Select a framework"
                    triggerClassName="bg-primary/5 border-primary/50 text-primary"
                />
            </div>

            <div className="space-y-2">
                <Label>Success Style</Label>
                <Combobox
                    options={frameworks}
                    placeholder="Select a framework"
                    triggerClassName="bg-green-50 border-green-200 text-green-600"
                />
            </div>

            <div className="space-y-2">
                <Label>Warning Style</Label>
                <Combobox
                    options={frameworks}
                    placeholder="Select a framework"
                    triggerClassName="bg-yellow-50 border-yellow-200 text-yellow-600"
                />
            </div>

            <div className="space-y-2">
                <Label>Danger Style</Label>
                <Combobox
                    options={frameworks}
                    placeholder="Select a framework"
                    triggerClassName="bg-red-50 border-red-200 text-red-600"
                />
            </div>
        </div>
    )
};

// With Icons in Items
export const WithIcons: Story = {
    render: function IconsExample() {
        // Define our custom implementation for icons in items
        const [open, setOpen] = useState(false);
        const [value, setValue] = useState("");

        const devices = [
            { value: "desktop", label: "Desktop", icon: Laptop },
            { value: "mobile", label: "Mobile", icon: Smartphone },
            { value: "server", label: "Server", icon: Globe },
            { value: "account", label: "Account", icon: User },
            { value: "settings", label: "Settings", icon: Settings }
        ];

        const selectedItem = devices.find((item) => item.value === value);

        return (
            <div className="w-full max-w-sm">
                <Popover open={open} onOpenChange={setOpen}>
                    <PopoverTrigger asChild>
                        <Button
                            variant="light-outline"
                            role="combobox"
                            aria-expanded={open}
                            aria-label="Select a device"
                            className="w-full justify-between font-normal">
                            {selectedItem ? (
                                <div className="flex items-center">
                                    {selectedItem.icon && (
                                        <selectedItem.icon className="mr-2 h-4 w-4" />
                                    )}
                                    {selectedItem.label}
                                </div>
                            ) : (
                                "Select a device"
                            )}
                            <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
                        </Button>
                    </PopoverTrigger>
                    <PopoverContent className="w-[250px] p-0">
                        <Command>
                            <CommandInput placeholder="Search devices..." />
                            <CommandList>
                                <CommandEmpty>No results found.</CommandEmpty>
                                <CommandGroup>
                                    {devices.map((item) => (
                                        <CommandItem
                                            key={item.value}
                                            value={item.value}
                                            onSelect={(currentValue) => {
                                                setValue(currentValue);
                                                setOpen(false);
                                            }}>
                                            {item.icon && (
                                                <item.icon className="mr-2 h-4 w-4" />
                                            )}
                                            {item.label}
                                            <Check
                                                className={cn(
                                                    "ml-auto size-4",
                                                    value === item.value
                                                        ? "opacity-100"
                                                        : "opacity-0"
                                                )}
                                            />
                                        </CommandItem>
                                    ))}
                                </CommandGroup>
                            </CommandList>
                        </Command>
                    </PopoverContent>
                </Popover>
            </div>
        );
    }
};

// Dark Mode
export const DarkMode: Story = {
    render: () => (
        <div className="dark rounded-lg bg-slate-950 p-6">
            <div className="grid w-full max-w-sm items-center gap-1.5">
                <Label htmlFor="framework-dark">Framework</Label>
                <Combobox
                    options={frameworks}
                    placeholder="Select a framework"
                />
            </div>
        </div>
    )
};

// Usage in Complex Form
export const ComplexForm: Story = {
    render: function ComplexFormExample() {
        const [formState, setFormState] = useState({
            framework: "react",
            country: "us"
        });

        const handleChange = (field: string, value: string) => {
            setFormState((prev) => ({
                ...prev,
                [field]: value
            }));
        };

        return (
            <div className="w-full max-w-md rounded-lg border p-6">
                <h3 className="mb-4 text-lg font-medium">Developer Profile</h3>
                <div className="space-y-4">
                    <div className="grid w-full items-center gap-1.5">
                        <Label htmlFor="framework">Preferred Framework</Label>
                        <Combobox
                            options={frameworks}
                            value={formState.framework}
                            onValueChange={(value) =>
                                handleChange("framework", value)
                            }
                            placeholder="Select a framework"
                        />
                    </div>

                    <div className="grid w-full items-center gap-1.5">
                        <Label htmlFor="country">Country</Label>
                        <Combobox
                            options={countries}
                            value={formState.country}
                            onValueChange={(value) =>
                                handleChange("country", value)
                            }
                            placeholder="Select a country"
                            popoverClassName="w-[300px]"
                        />
                    </div>

                    <div className="flex justify-end space-x-2 pt-4">
                        <Button
                            variant="light-outline"
                            onClick={() =>
                                setFormState({ framework: "", country: "" })
                            }>
                            Reset
                        </Button>
                        <Button>Save Profile</Button>
                    </div>
                </div>
            </div>
        );
    }
};

// Async Loading Example
export const AsyncLoading: Story = {
    render: function AsyncLoadingExample() {
        const [isLoading, setIsLoading] = useState(false);
        const [options, setOptions] = useState<ComboboxOption[]>([]);
        const [value, setValue] = useState("");

        // Simulate fetching data from an API
        const fetchData = () => {
            setIsLoading(true);

            // Simulate API delay
            setTimeout(() => {
                setOptions([
                    { value: "react", label: "React" },
                    { value: "vue", label: "Vue" },
                    { value: "angular", label: "Angular" },
                    { value: "svelte", label: "Svelte" },
                    { value: "solid", label: "Solid" }
                ]);
                setIsLoading(false);
            }, 1000);
        };

        return (
            <div className="w-full max-w-sm space-y-4">
                <div className="space-y-2">
                    <Label htmlFor="async-combobox">Framework (Async)</Label>

                    <div className="relative">
                        <Combobox
                            options={options}
                            value={value}
                            onValueChange={setValue}
                            placeholder={
                                isLoading
                                    ? "Loading options..."
                                    : "Select a framework"
                            }
                            disabled={isLoading}
                            emptyText={
                                options.length === 0
                                    ? "Click 'Load Options' to fetch data"
                                    : "No results found"
                            }
                        />
                        {isLoading && (
                            <div className="absolute right-10 top-1/2 -translate-y-1/2">
                                <div className="border-primary h-4 w-4 animate-spin rounded-full border-2 border-t-transparent"></div>
                            </div>
                        )}
                    </div>
                </div>

                <Button
                    onClick={fetchData}
                    disabled={isLoading}
                    variant="light-outline">
                    {isLoading ? "Loading..." : "Load Options"}
                </Button>
            </div>
        );
    }
};

// Searchable with Custom Filter Logic
export const CustomSearchFilter: Story = {
    render: function CustomSearchFilterExample() {
        const [open, setOpen] = useState(false);
        const [value, setValue] = useState("");
        const [search, setSearch] = useState("");

        const languages = [
            {
                value: "javascript",
                label: "JavaScript",
                tags: ["frontend", "web"]
            },
            {
                value: "typescript",
                label: "TypeScript",
                tags: ["frontend", "typed"]
            },
            { value: "python", label: "Python", tags: ["backend", "data"] },
            { value: "rust", label: "Rust", tags: ["systems", "performance"] },
            { value: "go", label: "Go", tags: ["backend", "performance"] },
            { value: "java", label: "Java", tags: ["backend", "enterprise"] },
            { value: "csharp", label: "C#", tags: ["backend", "microsoft"] },
            { value: "ruby", label: "Ruby", tags: ["backend", "web"] },
            { value: "php", label: "PHP", tags: ["backend", "web"] },
            { value: "swift", label: "Swift", tags: ["mobile", "apple"] },
            { value: "kotlin", label: "Kotlin", tags: ["mobile", "android"] }
        ];

        // Custom filter function that searches both label and tags
        const filterLanguages = (searchValue: string) => {
            return languages.filter(
                (lang) =>
                    lang.label
                        .toLowerCase()
                        .includes(searchValue.toLowerCase()) ||
                    lang.tags.some((tag) =>
                        tag.toLowerCase().includes(searchValue.toLowerCase())
                    )
            );
        };

        const filteredLanguages = search ? filterLanguages(search) : languages;
        const selectedItem = languages.find((item) => item.value === value);

        return (
            <div className="w-full max-w-sm">
                <Popover open={open} onOpenChange={setOpen}>
                    <PopoverTrigger asChild>
                        <Button
                            variant="light-outline"
                            role="combobox"
                            aria-expanded={open}
                            aria-label="Select programming language"
                            className="w-full justify-between font-normal">
                            {selectedItem
                                ? selectedItem.label
                                : "Select programming language"}
                            <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
                        </Button>
                    </PopoverTrigger>
                    <PopoverContent className="w-[300px] p-0">
                        <Command>
                            <CommandInput
                                placeholder="Search by name or tag..."
                                value={search}
                                onValueChange={setSearch}
                            />
                            <CommandList>
                                <CommandEmpty>No results found.</CommandEmpty>
                                <CommandGroup>
                                    {filteredLanguages.map((item) => (
                                        <CommandItem
                                            key={item.value}
                                            value={item.value}
                                            onSelect={(currentValue) => {
                                                setValue(currentValue);
                                                setOpen(false);
                                                setSearch("");
                                            }}>
                                            <div className="flex flex-col">
                                                <span>{item.label}</span>
                                                <span className="text-muted-foreground text-xs">
                                                    {item.tags
                                                        .map((tag) => `#${tag}`)
                                                        .join(" ")}
                                                </span>
                                            </div>
                                            <Check
                                                className={cn(
                                                    "ml-auto size-4",
                                                    value === item.value
                                                        ? "opacity-100"
                                                        : "opacity-0"
                                                )}
                                            />
                                        </CommandItem>
                                    ))}
                                </CommandGroup>
                            </CommandList>
                        </Command>
                    </PopoverContent>
                </Popover>
            </div>
        );
    }
};

// Combobox with Description
export const WithDescriptions: Story = {
    render: function WithDescriptionsExample() {
        const [open, setOpen] = useState(false);
        const [value, setValue] = useState("");

        const editors = [
            {
                value: "vscode",
                label: "Visual Studio Code",
                description:
                    "Free, open-source code editor by Microsoft with extensive plugin support."
            },
            {
                value: "webstorm",
                label: "WebStorm",
                description:
                    "Commercial JavaScript IDE with powerful code assistance and tooling."
            },
            {
                value: "vim",
                label: "Vim",
                description:
                    "Highly configurable text editor built to enable efficient text editing."
            },
            {
                value: "atom",
                label: "Atom",
                description:
                    "A hackable text editor for the 21st Century, now sunset."
            },
            {
                value: "sublime",
                label: "Sublime Text",
                description:
                    "Sophisticated text editor for code, markup and prose."
            }
        ];

        const selectedItem = editors.find((item) => item.value === value);

        return (
            <div className="w-full max-w-sm">
                <Popover open={open} onOpenChange={setOpen}>
                    <PopoverTrigger asChild>
                        <Button
                            variant="light-outline"
                            role="combobox"
                            aria-expanded={open}
                            aria-label="Select an editor"
                            className="w-full justify-between font-normal">
                            {selectedItem
                                ? selectedItem.label
                                : "Select an editor"}
                            <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
                        </Button>
                    </PopoverTrigger>
                    <PopoverContent className="w-[350px] p-0">
                        <Command>
                            <CommandInput placeholder="Search editors..." />
                            <CommandList>
                                <CommandEmpty>No results found.</CommandEmpty>
                                <CommandGroup>
                                    {editors.map((item) => (
                                        <CommandItem
                                            key={item.value}
                                            value={item.value}
                                            onSelect={(currentValue) => {
                                                setValue(currentValue);
                                                setOpen(false);
                                            }}
                                            className="flex flex-col items-start py-2">
                                            <div className="flex w-full justify-between">
                                                <span className="font-medium">
                                                    {item.label}
                                                </span>
                                                <Check
                                                    className={cn(
                                                        "size-4",
                                                        value === item.value
                                                            ? "opacity-100"
                                                            : "opacity-0"
                                                    )}
                                                />
                                            </div>
                                            <span className="text-muted-foreground text-xs">
                                                {item.description}
                                            </span>
                                        </CommandItem>
                                    ))}
                                </CommandGroup>
                            </CommandList>
                        </Command>
                    </PopoverContent>
                </Popover>
            </div>
        );
    }
};
