// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    Pagination,
    PaginationContent,
    PaginationEllipsis,
    PaginationItem,
    PaginationLink,
    PaginationNext,
    PaginationPrevious
} from "@/components/ui/pagination";

export const DataTablePagination = ({
    canPreviousPage,
    canNextPage,
    previousPage,
    nextPage,
    pageIndex,
    pageCount,
    handlePageChange
}: {
    canPreviousPage: boolean;
    canNextPage: boolean;
    previousPage: () => void;
    nextPage: () => void;
    pageIndex: number;
    pageCount: number;
    handlePageChange: (pageIndex: number) => void;
}) => {
    const currentPage = pageIndex + 1;

    const generatePaginationItems = () => {
        if (pageCount <= 7) {
            return Array.from({ length: pageCount }, (_, i) => i + 1);
        }

        if (currentPage <= 4) {
            return [1, 2, 3, 4, 5, "ellipsis", pageCount];
        }

        if (currentPage > pageCount - 4) {
            return [
                1,
                "ellipsis",
                pageCount - 4,
                pageCount - 3,
                pageCount - 2,
                pageCount - 1,
                pageCount
            ];
        }

        return [
            1,
            "ellipsis",
            currentPage - 1,
            currentPage,
            currentPage + 1,
            "ellipsis",
            pageCount
        ];
    };

    const paginationItems = generatePaginationItems();

    return (
        <Pagination className="border-border-secondary m-0 h-10 w-full rounded-b-sm border border-t-0 bg-white text-base text-slate-800">
            <PaginationContent className="!m-2 flex w-full justify-between !px-1">
                <PaginationItem className="w-24 flex-none justify-start">
                    <PaginationPrevious
                        onClick={previousPage}
                        className={`${!canPreviousPage ? "pointer-events-none opacity-50" : "cursor-pointer"}`}
                        aria-disabled={!canPreviousPage}
                        data-testid={`pagination-prev`}
                    />
                </PaginationItem>
                <li className="flex flex-1 items-center justify-center">
                    {paginationItems.map((pageNumber, index) => {
                        if (pageNumber === "ellipsis") {
                            return (
                                <PaginationEllipsis
                                    key={`${pageNumber}-${index}`}
                                />
                            );
                        }

                        return (
                            <PaginationLink
                                key={pageNumber}
                                onClick={() =>
                                    handlePageChange(Number(pageNumber))
                                }
                                isActive={currentPage === pageNumber}
                                className={
                                    currentPage === pageNumber
                                        ? "pointer-events-none"
                                        : "cursor-pointer opacity-50"
                                }
                                data-testid={`pagination-${Number(pageNumber) + 1}`}>
                                {pageNumber}
                            </PaginationLink>
                        );
                    })}
                </li>

                <PaginationItem className="flex w-24 flex-none justify-end">
                    <PaginationNext
                        onClick={nextPage}
                        className={`${!canNextPage ? "pointer-events-none opacity-50" : "cursor-pointer"}`}
                        aria-disabled={!canNextPage}
                        data-testid={`pagination-next`}
                    />
                </PaginationItem>
            </PaginationContent>
        </Pagination>
    );
};
