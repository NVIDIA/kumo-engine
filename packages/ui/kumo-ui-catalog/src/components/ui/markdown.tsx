// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";
import { CopyToClipBoard } from "./copy-to-clipboard";

// Helper function to generate slug from heading text
const generateSlug = (text: string): string => {
    return text
        .toLowerCase()
        .replace(/[^\w\s-]/g, "")
        .replace(/\s+/g, "-")
        .replace(/-+/g, "-")
        .trim();
};

// Alert types configuration
const alertTypes = {
    NOTE: {
        icon: "ℹ️",
        className:
            "border-blue-500 bg-blue-50 text-blue-900 dark:bg-blue-950 dark:text-blue-100 dark:border-blue-400"
    },
    TIP: {
        icon: "💡",
        className:
            "border-green-500 bg-green-50 text-green-900 dark:bg-green-950 dark:text-green-100 dark:border-green-400"
    },
    IMPORTANT: {
        icon: "⚠️",
        className:
            "border-purple-500 bg-purple-50 text-purple-900 dark:bg-purple-950 dark:text-purple-100 dark:border-purple-400"
    },
    WARNING: {
        icon: "⚠️",
        className:
            "border-yellow-500 bg-yellow-50 text-yellow-900 dark:bg-yellow-950 dark:text-yellow-100 dark:border-yellow-600"
    },
    CAUTION: {
        icon: "🔥",
        className:
            "border-red-500 bg-red-50 text-red-900 dark:bg-red-950 dark:text-red-100 dark:border-red-400"
    }
};

const markdownVariants = cva("prose prose-slate max-w-none", {
    variants: {
        size: {
            sm: "prose-sm",
            base: "prose-base",
            lg: "prose-lg"
        },
        theme: {
            default: "",
            dark: "prose-invert"
        }
    },
    defaultVariants: {
        size: "base",
        theme: "default"
    }
});

export interface MarkdownProps
    extends React.HTMLAttributes<HTMLDivElement>,
        VariantProps<typeof markdownVariants> {
    children: string;
    components?: React.ComponentProps<typeof ReactMarkdown>["components"];
    enableSyntaxHighlighting?: boolean;
    enableCopyButton?: boolean;
    enableHeadingAnchors?: boolean;
    enableLineNumbers?: boolean;
}

function Markdown({
    className,
    size,
    theme,
    children,
    components,
    enableSyntaxHighlighting = false,
    enableCopyButton = true,
    enableHeadingAnchors = false,
    enableLineNumbers = false,
    ...props
}: MarkdownProps) {
    return (
        <div
            data-slot="markdown"
            className={cn(markdownVariants({ size, theme }), className)}
            {...props}>
            <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                rehypePlugins={[rehypeRaw]}
                components={{
                    h1: ({ children, ...props }) => {
                        const text = String(children);
                        const id = enableHeadingAnchors
                            ? generateSlug(text)
                            : undefined;
                        return (
                            <h1
                                id={id}
                                className="group relative mb-4 mt-6 text-2xl font-bold first:mt-0"
                                {...props}>
                                {children}
                                {enableHeadingAnchors && id && (
                                    <a
                                        href={`#${id}`}
                                        className="ml-2 text-slate-400 no-underline opacity-0 transition-opacity group-hover:opacity-100"
                                        aria-label={`Link to ${text}`}>
                                        #
                                    </a>
                                )}
                            </h1>
                        );
                    },
                    h2: ({ children, ...props }) => {
                        const text = String(children);
                        const id = enableHeadingAnchors
                            ? generateSlug(text)
                            : undefined;
                        return (
                            <h2
                                id={id}
                                className="group relative mb-3 mt-5 text-xl font-bold"
                                {...props}>
                                {children}
                                {enableHeadingAnchors && id && (
                                    <a
                                        href={`#${id}`}
                                        className="ml-2 text-slate-400 no-underline opacity-0 transition-opacity group-hover:opacity-100"
                                        aria-label={`Link to ${text}`}>
                                        #
                                    </a>
                                )}
                            </h2>
                        );
                    },
                    h3: ({ children, ...props }) => {
                        const text = String(children);
                        const id = enableHeadingAnchors
                            ? generateSlug(text)
                            : undefined;
                        return (
                            <h3
                                id={id}
                                className="group relative mb-2 mt-4 text-lg font-semibold"
                                {...props}>
                                {children}
                                {enableHeadingAnchors && id && (
                                    <a
                                        href={`#${id}`}
                                        className="ml-2 text-slate-400 no-underline opacity-0 transition-opacity group-hover:opacity-100"
                                        aria-label={`Link to ${text}`}>
                                        #
                                    </a>
                                )}
                            </h3>
                        );
                    },
                    h4: ({ children, ...props }) => {
                        const text = String(children);
                        const id = enableHeadingAnchors
                            ? generateSlug(text)
                            : undefined;
                        return (
                            <h4
                                id={id}
                                className="group relative mb-2 mt-3 text-base font-semibold"
                                {...props}>
                                {children}
                                {enableHeadingAnchors && id && (
                                    <a
                                        href={`#${id}`}
                                        className="ml-2 text-slate-400 no-underline opacity-0 transition-opacity group-hover:opacity-100"
                                        aria-label={`Link to ${text}`}>
                                        #
                                    </a>
                                )}
                            </h4>
                        );
                    },
                    h5: ({ children, ...props }) => {
                        const text = String(children);
                        const id = enableHeadingAnchors
                            ? generateSlug(text)
                            : undefined;
                        return (
                            <h5
                                id={id}
                                className="group relative mb-2 mt-3 text-sm font-semibold"
                                {...props}>
                                {children}
                                {enableHeadingAnchors && id && (
                                    <a
                                        href={`#${id}`}
                                        className="ml-2 text-slate-400 no-underline opacity-0 transition-opacity group-hover:opacity-100"
                                        aria-label={`Link to ${text}`}>
                                        #
                                    </a>
                                )}
                            </h5>
                        );
                    },
                    h6: ({ children, ...props }) => {
                        const text = String(children);
                        const id = enableHeadingAnchors
                            ? generateSlug(text)
                            : undefined;
                        return (
                            <h6
                                id={id}
                                className="group relative mb-2 mt-2 text-xs font-semibold uppercase tracking-wide"
                                {...props}>
                                {children}
                                {enableHeadingAnchors && id && (
                                    <a
                                        href={`#${id}`}
                                        className="ml-2 text-slate-400 no-underline opacity-0 transition-opacity group-hover:opacity-100"
                                        aria-label={`Link to ${text}`}>
                                        #
                                    </a>
                                )}
                            </h6>
                        );
                    },
                    p: ({ ...props }) => (
                        <p className="mb-4 leading-7" {...props} />
                    ),
                    img: ({ ...props }) => (
                        <img
                            className="my-4 h-auto max-w-full rounded"
                            loading="lazy"
                            {...props}
                        />
                    ),
                    ul: ({ ...props }) => (
                        <ul
                            className="mb-4 list-disc space-y-2 pl-6"
                            {...props}
                        />
                    ),
                    ol: ({ ...props }) => (
                        <ol
                            className="mb-4 list-decimal space-y-2 pl-6"
                            {...props}
                        />
                    ),
                    li: ({ children, ...props }) => {
                        // Check if this is a task list item
                        const hasCheckbox = React.Children.toArray(
                            children
                        ).some(
                            (child) =>
                                React.isValidElement(child) &&
                                child.type === "input" &&
                                (child.props as { type?: string }).type ===
                                    "checkbox"
                        );

                        if (hasCheckbox) {
                            return (
                                <li
                                    className="flex list-none items-start gap-2 leading-7"
                                    {...props}>
                                    {children}
                                </li>
                            );
                        }

                        return (
                            <li className="leading-7" {...props}>
                                {children}
                            </li>
                        );
                    },
                    input: ({ ...props }) => {
                        // Style checkboxes in task lists
                        if (props.type === "checkbox") {
                            return (
                                <input
                                    className="mt-1 cursor-pointer"
                                    aria-label={
                                        props.checked
                                            ? "Completed task"
                                            : "Incomplete task"
                                    }
                                    {...props}
                                />
                            );
                        }
                        return <input {...props} />;
                    },
                    blockquote: ({
                        children,
                        // eslint-disable-next-line @typescript-eslint/no-unused-vars
                        node,
                        ...props
                    }) => {
                        // Check if this is a GitHub-style alert
                        const childArray = React.Children.toArray(children);
                        const firstChild = childArray[0];

                        if (
                            React.isValidElement(firstChild) &&
                            firstChild.type === "p"
                        ) {
                            const pChildren = React.Children.toArray(
                                firstChild.props.children
                            );
                            const firstText =
                                typeof pChildren[0] === "string"
                                    ? pChildren[0]
                                    : "";

                            // Check for alert pattern: [!TYPE]
                            const alertMatch = firstText.match(/^\[!(\w+)\]/);
                            if (alertMatch) {
                                const alertType =
                                    alertMatch[1].toUpperCase() as keyof typeof alertTypes;
                                const alert = alertTypes[alertType];

                                if (alert) {
                                    // Remove the [!TYPE] from the content
                                    const restOfText = firstText
                                        .replace(/^\[!\w+\]\s*/, "")
                                        .trim();
                                    const modifiedChildren = [
                                        restOfText,
                                        ...pChildren.slice(1)
                                    ];

                                    return (
                                        <div
                                            className={cn(
                                                "my-4 rounded border-l-4 p-4 not-italic",
                                                alert.className
                                            )}
                                            role="alert">
                                            <div className="flex items-start gap-2">
                                                <span className="text-lg">
                                                    {alert.icon}
                                                </span>
                                                <div className="flex-1">
                                                    <p className="mb-0 font-semibold">
                                                        {alertType}
                                                    </p>
                                                    <div className="mt-1">
                                                        {React.cloneElement(
                                                            firstChild,
                                                            {},
                                                            modifiedChildren
                                                        )}
                                                        {childArray.slice(1)}
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    );
                                }
                            }
                        }

                        // Regular blockquote
                        return (
                            <blockquote
                                className="my-4 border-l-4 border-slate-300 pl-4 italic text-slate-600 dark:border-slate-700 dark:text-slate-400"
                                {...props}>
                                {children}
                            </blockquote>
                        );
                    },
                    code: ({ className, children, ref, ...props }) => {
                        const isInline = !className;
                        const match = /language-(\w+)/.exec(className || "");
                        const language = match ? match[1] : "";
                        const codeString = String(children).replace(/\n$/, "");

                        if (!isInline && enableSyntaxHighlighting && language) {
                            return (
                                <div className="relative my-4">
                                    {enableCopyButton && (
                                        <div className="absolute right-2 top-2 z-10">
                                            <CopyToClipBoard
                                                copiedText={codeString}
                                                size="xs"
                                                variant="light-outline"
                                                iconSize={14}
                                            />
                                        </div>
                                    )}
                                    <SyntaxHighlighter
                                        // eslint-disable-next-line @typescript-eslint/no-explicit-any
                                        style={vscDarkPlus as any}
                                        language={language}
                                        PreTag="div"
                                        showLineNumbers={enableLineNumbers}
                                        className="rounded text-sm">
                                        {codeString}
                                    </SyntaxHighlighter>
                                </div>
                            );
                        }

                        if (!isInline) {
                            return (
                                <div className="relative my-4">
                                    {enableCopyButton && (
                                        <div className="absolute right-2 top-2 z-10">
                                            <CopyToClipBoard
                                                copiedText={codeString}
                                                size="xs"
                                                variant="light-outline"
                                                iconSize={14}
                                            />
                                        </div>
                                    )}
                                    <code
                                        className={cn(
                                            "block overflow-x-auto rounded bg-slate-100 p-4 font-mono text-sm text-slate-800 dark:bg-slate-800 dark:text-slate-200",
                                            className
                                        )}
                                        ref={ref}
                                        {...props}>
                                        {children}
                                    </code>
                                </div>
                            );
                        }

                        return (
                            <code
                                className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-sm text-slate-800 dark:bg-slate-800 dark:text-slate-200"
                                ref={ref}
                                {...props}>
                                {children}
                            </code>
                        );
                    },
                    pre: ({ ...props }) => (
                        <pre
                            className="mb-4 overflow-x-auto rounded bg-slate-100 dark:bg-slate-800"
                            {...props}
                        />
                    ),
                    a: ({ ...props }) => (
                        <a
                            className="text-blue-600 underline hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300"
                            target="_blank"
                            rel="noopener noreferrer"
                            {...props}
                        />
                    ),
                    del: ({ ...props }) => (
                        <del
                            className="text-slate-500 line-through dark:text-slate-400"
                            {...props}
                        />
                    ),
                    hr: ({ ...props }) => (
                        <hr
                            className="my-6 border-slate-300 dark:border-slate-700"
                            {...props}
                        />
                    ),
                    table: ({ ...props }) => (
                        <div className="mb-4 overflow-x-auto">
                            <table
                                className="min-w-full border-collapse border border-slate-300 dark:border-slate-700"
                                {...props}
                            />
                        </div>
                    ),
                    thead: ({ ...props }) => (
                        <thead
                            className="bg-slate-100 dark:bg-slate-700 dark:text-slate-100"
                            {...props}
                        />
                    ),
                    tbody: ({ ...props }) => <tbody {...props} />,
                    tr: ({ ...props }) => <tr {...props} />,
                    th: ({ ...props }) => (
                        <th
                            className="border border-slate-300 px-4 py-2 text-left font-semibold dark:border-slate-700 dark:text-slate-100"
                            {...props}
                        />
                    ),
                    td: ({ ...props }) => (
                        <td
                            className="border border-slate-300 px-4 py-2 dark:border-slate-700 dark:text-slate-200"
                            {...props}
                        />
                    ),
                    strong: ({ ...props }) => (
                        <strong className="font-bold" {...props} />
                    ),
                    em: ({ ...props }) => <em className="italic" {...props} />,
                    sup: ({ ...props }) => (
                        <sup className="align-super text-xs" {...props} />
                    ),
                    sub: ({ ...props }) => (
                        <sub className="align-sub text-xs" {...props} />
                    ),
                    kbd: ({ ...props }) => (
                        <kbd
                            className="rounded border border-slate-300 bg-slate-50 px-1.5 py-0.5 font-mono text-xs shadow-sm dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200"
                            {...props}
                        />
                    ),
                    ...components
                }}>
                {children}
            </ReactMarkdown>
        </div>
    );
}

export { Markdown };
