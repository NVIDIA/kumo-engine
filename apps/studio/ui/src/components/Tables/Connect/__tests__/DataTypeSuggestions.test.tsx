// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Column } from "gen/v1/types/column";
import { Dtype as KumoapiTypingDtype } from "gen/v1/types/dtype";
import { Stype as KumoapiTypingStype } from "gen/v1/types/stype";

import DataTypeSuggestions, {
    DataTypeSuggestionsProps
} from "../DataTypeSuggestions";

import { expectedColumnResponse } from "./MockFunction/ColumnResponse.test.mock";

const TEST_FIXTURES = {
    singleCategoricalFloatColumn: [
        {
            name: "TARGET",
            stype: KumoapiTypingStype.ID,
            dtype: KumoapiTypingDtype.FLOAT
        },
        {
            name: "arange",
            stype: KumoapiTypingStype.CATEGORICAL,
            dtype: KumoapiTypingDtype.FLOAT
        }
    ] as Column[],

    multipleCategoricalFloatColumns: [
        {
            name: "userId",
            stype: KumoapiTypingStype.CATEGORICAL,
            dtype: KumoapiTypingDtype.FLOAT
        },
        {
            name: "movieId",
            stype: KumoapiTypingStype.CATEGORICAL,
            dtype: KumoapiTypingDtype.FLOAT
        },
        {
            name: "rating",
            stype: KumoapiTypingStype.ID,
            dtype: KumoapiTypingDtype.FLOAT64
        }
    ] as Column[],

    emptyColumns: [] as Column[]
};

const TEXT_CONTENT = {
    warning: "Warning",
    dataTypeSuggestion: /Data type suggestion/i
};

describe("DataTypeSuggestions", () => {
    const renderDataTypeSuggestions = async (
        props: DataTypeSuggestionsProps
    ) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <DataTypeSuggestions {...props} />
                </MemoryRouter>
            );
        });
    };

    const expectAlertToBeVisible = () => {
        const alert = screen.queryByRole("alert");
        expect(alert).toBeInTheDocument();
        return alert;
    };

    const expectWarningHeader = () => {
        expect(screen.getByText(TEXT_CONTENT.warning)).toBeInTheDocument();
    };

    const expectDataTypeSuggestion = () => {
        expect(
            screen.getByText(TEXT_CONTENT.dataTypeSuggestion)
        ).toBeInTheDocument();
    };

    it("Should display `Something went wrong` for `Empty/Invalid` response", async () => {
        await renderDataTypeSuggestions({
            columns: {} as Column[]
        });

        expectAlertToBeVisible();
        expect(screen.getByText("Something went wrong in")).toBeInTheDocument();
        expect(
            screen.getByText(/Please contact Kumo support/i)
        ).toBeInTheDocument();
    });

    it("Should not render `DataTypeSuggestions` if there are no columns needing suggestions", async () => {
        const colData = expectedColumnResponse() as Column[];
        await renderDataTypeSuggestions({
            columns: colData
        });

        expect(
            screen.queryByText(TEXT_CONTENT.dataTypeSuggestion)
        ).not.toBeInTheDocument();
    });

    it("Should not render `DataTypeSuggestions` for empty columns array", async () => {
        await renderDataTypeSuggestions({
            columns: TEST_FIXTURES.emptyColumns
        });

        expect(screen.queryByRole("alert")).not.toBeInTheDocument();
        expect(
            screen.queryByText(TEXT_CONTENT.dataTypeSuggestion)
        ).not.toBeInTheDocument();
    });

    it("Should render `DataTypeSuggestions` for column with stype categorical and float dtype", async () => {
        await renderDataTypeSuggestions({
            columns: TEST_FIXTURES.singleCategoricalFloatColumn
        });

        expectAlertToBeVisible();
        expectWarningHeader();
        expectDataTypeSuggestion();

        expect(
            screen.getByText(
                'Use of data type <float> for column "arange" with Type "Categorical" may result in data mismatch because of precision issue. Consider converting the data type to integers in your data source.'
            )
        ).toBeInTheDocument();
    });

    it("Should render a single consolidated `DataTypeSuggestions` for multiple columns with stype categorical and float dtype", async () => {
        await renderDataTypeSuggestions({
            columns: TEST_FIXTURES.multipleCategoricalFloatColumns
        });

        expectAlertToBeVisible();
        expectWarningHeader();

        // Only one "Data type suggestion" entry for all columns with the same dtype
        const suggestions = screen.getAllByText(
            TEXT_CONTENT.dataTypeSuggestion
        );
        expect(suggestions.length).toBe(1);

        expect(
            screen.getByText(
                /Use of data type <float> for columns "userId", "movieId"/i
            )
        ).toBeInTheDocument();
    });

    it("Should close the `DataTypeSuggestions` on clicking Close button", async () => {
        await renderDataTypeSuggestions({
            columns: TEST_FIXTURES.singleCategoricalFloatColumn
        });

        await screen.findByRole("alert");
        expectWarningHeader();
        expectDataTypeSuggestion();

        const closeButton = screen.getByTestId("alert-close");
        expect(closeButton).toBeInTheDocument();
        await userEvent.click(closeButton);

        await waitFor(() => {
            expect(screen.queryByRole("alert")).not.toBeInTheDocument();
        });
    });
});
