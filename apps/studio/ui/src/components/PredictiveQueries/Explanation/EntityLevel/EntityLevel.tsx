// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";

import {
    Button,
    ChevronDown,
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger
} from "@kumo-ai/kumo-ui-catalog";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import PaginationComponent from "components/PredictiveQueries/Explanation/EntityLevel/Pagination";

import { getKeys } from "utils/Helpers/objectHelpers";

interface CategoryMap {
    [key: string]: string[];
}

export interface EntitySelectionProps {
    categories: CategoryMap;
    category?: string;
    setCategory?: (category: string) => void;
    selectedEntityIndex: number;
    setSelectedEntityIndex: (index: number) => void;
    entityId: string;
    originalEntityId: string | null;
    showCategories?: boolean;
    entityIdentifier?: string;
}

const EntityLevel: React.FC<EntitySelectionProps> = ({
    categories,
    category,
    setCategory,
    selectedEntityIndex,
    setSelectedEntityIndex,
    entityId,
    originalEntityId,
    showCategories = true,
    entityIdentifier
}) => {
    const handleCategoryChange = (newCategory: string): void => {
        if (setCategory) {
            setCategory(newCategory);
        }
        setSelectedEntityIndex(0);
    };

    const handleEntityChange = (newIndex: number): void => {
        setSelectedEntityIndex(newIndex - 1);
    };

    const formatCategory = (cat: string): string => {
        return cat.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase());
    };

    if (!category || !categories?.[category]?.length) {
        return null;
    }

    const displayEntityId = originalEntityId ?? decodeURIComponent(entityId);

    return (
        <div className="flex justify-between items-center">
            {showCategories ? (
                <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                        <Button
                            variant="light-outline"
                            className="h-9 gap-2 text-title-secondary text-sm font-medium"
                            data-testid="testid-category-dropdown">
                            Sort: {formatCategory(category)}
                            <ChevronDown className="size-4" />
                        </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="start">
                        {getKeys(categories).map((cat) => (
                            <DropdownMenuItem
                                key={cat}
                                onSelect={() => handleCategoryChange(cat)}>
                                {formatCategory(cat)}
                            </DropdownMenuItem>
                        ))}
                    </DropdownMenuContent>
                </DropdownMenu>
            ) : (
                <div />
            )}

            <PaginationComponent
                currentSelection={selectedEntityIndex + 1}
                totalOptions={categories[category].length}
                onSelectChange={handleEntityChange}
                labelText={displayEntityId}
                entityIdentifier={entityIdentifier}
            />

            <span className="text-title-secondary text-sm font-medium">
                Entity {selectedEntityIndex + 1} of{" "}
                {categories[category].length}
            </span>
        </div>
    );
};

export default WithErrorBoundary(EntityLevel);
