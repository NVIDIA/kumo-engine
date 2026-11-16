// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RenderResult, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import {
    defaultGraphLinkGroupsProps,
    setupGraphTests
} from "components/Graphs/__tests__/graphTestUtils";

import { GraphLinkGroupsProps } from "../LinkGroups/GraphLinkGroups";

describe("GraphLinkGroups", () => {
    const { renderGraphLinkGroups } = setupGraphTests();

    const renderComponent = async (
        overrides: Partial<GraphLinkGroupsProps> = {}
    ) => {
        return await renderGraphLinkGroups({
            ...defaultGraphLinkGroupsProps,
            show: true,
            ...overrides
        });
    };

    const openAccordion = async (
        getByTestId: RenderResult["getByTestId"],
        testId: string = "fwdlinks-0"
    ) => {
        const accordionTrigger = getByTestId(testId);
        await userEvent.click(accordionTrigger);
        return accordionTrigger;
    };

    const selectTable = async (
        getByText: RenderResult["getByText"],
        findByRole: RenderResult["findByRole"],
        tableName: string
    ) => {
        const tableContainer = getByText("Table");
        await userEvent.click(tableContainer);
        const listbox = await findByRole("listbox");
        const tableOption = within(listbox).getByRole("option", {
            name: tableName
        });
        await userEvent.click(tableOption);
    };

    const selectColumn = async (
        getByText: RenderResult["getByText"],
        findByRole: RenderResult["findByRole"],
        columnName: string
    ) => {
        const columnContainer = getByText("Column");
        await userEvent.click(columnContainer);
        const listbox = await findByRole("listbox");
        const columnOption = within(listbox).getByRole("option", {
            name: columnName
        });
        await userEvent.click(columnOption);
    };

    afterEach(() => {
        vi.clearAllMocks();
    });

    it("Should display empty state when 'show' is false", async () => {
        const { container } = await renderComponent({ show: false });
        expect(container).toBeEmptyDOMElement();
    });

    it("Should render an empty accordion when both groups and backlinks are empty", async () => {
        const { container } = await renderComponent({
            groups: [],
            backwardLinks: {}
        });
        expect(container.querySelector(".accordion-item")).toBeNull();
    });

    it("Should render only backwardLinks when forward links are empty with disabled inputs and correct values", async () => {
        const { queryAllByTestId, getByTestId } = await renderComponent({
            groups: []
        });

        const fwdLinkHeaders = queryAllByTestId(/fwdlinks-/);
        expect(fwdLinkHeaders).toHaveLength(0);

        const bkLinkHeaders = queryAllByTestId(/bklinks-/);
        expect(bkLinkHeaders).toHaveLength(1);

        const backLinkAccordion = getByTestId("bklinks-0");
        await userEvent.click(backLinkAccordion);

        const backLinkTable = document.querySelector("#backlinks-table-0");
        const backLinkColumn = document.querySelector("#backlinks-columns-0");

        expect(backLinkTable).toBeInTheDocument();
        expect(backLinkColumn).toBeInTheDocument();

        expect(backLinkTable).toHaveAttribute("disabled");
        expect(backLinkColumn).toHaveAttribute("disabled");

        const tableSelectValue = backLinkTable?.textContent;
        const columnSelectValue = backLinkColumn?.textContent;

        expect(tableSelectValue).toContain("transactions");
        expect(columnSelectValue).toContain("article_id");
    });

    it("Should render only forward links when backwardLinks are empty", async () => {
        const { queryAllByTestId } = await renderComponent({
            backwardLinks: {}
        });

        const fwdLinkHeaders = queryAllByTestId(/fwdlinks-/);
        expect(fwdLinkHeaders).toHaveLength(1);

        const bkLinkHeaders = queryAllByTestId(/bklinks-/);
        expect(bkLinkHeaders).toHaveLength(0);
    });

    it("Should toggle accordion expanded state on click", async () => {
        const { getByTestId } = await renderComponent();
        const fwdLinkGroup = getByTestId("fwdlinks-0");

        expect(fwdLinkGroup).toHaveAttribute("data-state", "closed");
        expect(fwdLinkGroup).toHaveAttribute("aria-expanded", "false");

        await userEvent.click(fwdLinkGroup);
        expect(fwdLinkGroup).toHaveAttribute("data-state", "open");
        expect(fwdLinkGroup).toHaveAttribute("aria-expanded", "true");
    });

    it("Should render group headers for forward and backward links", async () => {
        const { getAllByTestId, findByTestId, getByTestId } =
            await renderComponent();

        const fwdLinkHeaders = getAllByTestId(/fwdlinks-/);
        const bkLinkHeaders = getAllByTestId(/bklinks-/);
        expect(fwdLinkHeaders).toHaveLength(1);
        expect(bkLinkHeaders).toHaveLength(1);

        const fwdLinkGroup = await findByTestId("fwdlinks-0");
        expect(fwdLinkGroup).toBeInTheDocument();
        expect(
            await within(fwdLinkGroup).findByText("articles")
        ).toBeInTheDocument();
        expect(
            await within(fwdLinkGroup).findByText("1 Link")
        ).toBeInTheDocument();
        expect(
            await within(fwdLinkGroup).findByText("article_id")
        ).toBeInTheDocument();

        const backLinkGroup = getByTestId("bklinks-0");
        expect(
            within(backLinkGroup).getByText("transactions")
        ).toBeInTheDocument();
        expect(
            within(backLinkGroup).getByText(/1\s+Backlink/i)
        ).toBeInTheDocument();
    });

    it("Should allow selecting a table and column", async () => {
        const { getByTestId, getByText, findByRole } = await renderComponent();

        await openAccordion(getByTestId);

        expect(getByText("Add more links")).toBeInTheDocument();
        expect(getByText("Table")).toBeInTheDocument();
        expect(getByText("Column")).toBeInTheDocument();

        await selectTable(getByText, findByRole, "transactions");

        const selectedTables = within(
            getByTestId("testid-group-0-table-2")
        ).getAllByText("transactions");
        expect(selectedTables).toHaveLength(1);
        expect(selectedTables[0]).toHaveTextContent("transactions");

        getByText("Select a reference column which links the two tables");

        await selectColumn(getByText, findByRole, "invoiceNo");

        const selectedColumn = within(
            getByTestId("testid-group-0-column-2")
        ).getAllByText("invoiceNo");
        expect(selectedColumn).toHaveLength(1);
        expect(selectedColumn[0]).toHaveTextContent("invoiceNo");
    });

    it("Should add a new empty row when a table and column are selected", async () => {
        const { getByTestId, getByText, findByRole } = await renderComponent();

        await openAccordion(getByTestId);
        await selectTable(getByText, findByRole, "transactions");
        await selectColumn(getByText, findByRole, "invoiceNo");

        // Verify new empty row was added
        const emptyTableInput = document.querySelector(
            "#group-0-table-3"
        ) as HTMLInputElement;
        const emptyColumnInput = document.querySelector(
            "#group-0-column-3"
        ) as HTMLInputElement;

        expect(emptyTableInput.value).toBe("");
        expect(emptyColumnInput.value).toBe("");
    });

    it("Should prevent selecting more than one Primary Key and display error message", async () => {
        const handleErrorMock = vi.fn();
        const { getByTestId, getByText, findByRole, queryByText } =
            await renderComponent({ handleError: handleErrorMock });

        await openAccordion(getByTestId);
        await selectTable(getByText, findByRole, "articles");
        await selectColumn(getByText, findByRole, "article_id (PK)");

        const selectedPK = queryByText("article_id (PK)", {
            selector: ".css-1dimb5e-singleValue"
        });
        expect(selectedPK).not.toBeInTheDocument();

        expect(handleErrorMock).toHaveBeenCalledWith(
            expect.objectContaining({
                response: expect.objectContaining({
                    data: expect.stringContaining(
                        "No more than one of the selected columns can be a Primary Key."
                    )
                })
            }),
            "Column group creation failed!"
        );
    });

    it("Should delete row when delete button is clicked", async () => {
        const { getByTestId, queryByTestId } = await renderComponent();

        await openAccordion(getByTestId);

        const selectedTables = within(
            getByTestId("testid-group-0-table-0")
        ).getAllByText("transactions");
        expect(selectedTables).toHaveLength(1);
        expect(selectedTables[0]).toHaveTextContent("transactions");

        const deleteColumn = document.querySelector("#delete-column")!;
        await userEvent.click(deleteColumn);

        const tableAfterDeletion = queryByTestId("testid-group-0-table-0");
        expect(tableAfterDeletion).not.toBeInTheDocument();

        const confirmColumnDeletion = document.querySelector("#delete-column");
        expect(confirmColumnDeletion).not.toBeInTheDocument();
    });
});
