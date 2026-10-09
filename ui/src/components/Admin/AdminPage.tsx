// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from "react";

import { cn, SearchBar, Title } from "@kumo-ai/kumo-ui-catalog";

interface AdminPageProps {
    title: string;
    description?: ReactNode;
    descriptionClassName?: string;
    children: ReactNode;
}

export const AdminPage = ({
    title,
    description,
    descriptionClassName,
    children
}: AdminPageProps) => (
    <section
        className="flex w-full flex-col gap-6"
        data-testid="admin-page"
        data-admin-page-title={title}>
        <header className="flex flex-col gap-2">
            <Title
                variant="primary-heading"
                className="text-3xl font-bold text-title-primary">
                {title}
            </Title>
            {description && (
                <p
                    className={cn(
                        "max-w-3xl text-sm text-title-secondary",
                        descriptionClassName
                    )}>
                    {description}
                </p>
            )}
        </header>
        {children}
    </section>
);

interface AdminActionRowProps {
    search: string;
    onSearchChange: (value: string) => void;
    searchPlaceholder?: string;
    children?: ReactNode;
}

export const AdminActionRow = ({
    search,
    onSearchChange,
    searchPlaceholder = "Search",
    children
}: AdminActionRowProps) => (
    <div
        className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between"
        data-testid="admin-action-row">
        <div
            className="w-full sm:max-w-74"
            data-testid="admin-search-container">
            <SearchBar
                aria-label={searchPlaceholder}
                placeholder={searchPlaceholder}
                className="h-9 w-full"
                value={search}
                variant="secondary"
                onChange={(event) => onSearchChange(event.target.value)}
            />
        </div>
        {children && (
            <div className="flex shrink-0 flex-wrap items-center gap-2">
                {children}
            </div>
        )}
    </div>
);

export const AdminCollection = ({ children }: { children: ReactNode }) => (
    <div className="flex flex-col gap-4" data-testid="admin-collection">
        {children}
    </div>
);

interface AdminSectionProps {
    title: string;
    children: ReactNode;
    actions?: ReactNode;
}

export const AdminSection = ({
    title,
    children,
    actions
}: AdminSectionProps) => (
    <section className="rounded-lg border border-border-secondary bg-white p-4">
        <div className="mb-4 flex items-center justify-between gap-3">
            <Title variant="section-header" className="font-bold">
                {title}
            </Title>
            {actions}
        </div>
        {children}
    </section>
);
