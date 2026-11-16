// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useMemo, useState } from "react";
import { useParams } from "react-router";
import Spacer from "react-spacer";

import {
    Alert,
    Card,
    ColumnDef,
    DataTable,
    Description,
    Title
} from "@kumo-ai/kumo-ui-catalog";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import { GraphLinkHealthColumns } from "components/@ui/TableColumns/GraphLinkHealth";

import { CustomError } from "types/common";
import { EdgeStats } from "types/graphs";
import { HandleErrorFunc } from "types/validation";

import { isValid } from "utils/Helpers/typeGuards";

const NO_EDGES: string = "No Edges in this Graph";

export interface GraphLinkHealthProps {
    summaryParams?: {
        graphsEdges: Array<EdgeStats>;
        message: "Fetching" | "Success" | string;
    };
    handleError: HandleErrorFunc;
}

const GraphLinkHealth = ({
    summaryParams,
    handleError
}: GraphLinkHealthProps) => {
    const { id } = useParams();
    const [graphEdges, setGraphEdges] = useState<Array<EdgeStats>>([]);
    const [pageLoading, setPageLoading] = useState<boolean>(true);
    const [calculating, setCalculating] = useState<boolean>(false);
    const [emptyMessage, setEmptyMessage] = useState<string>(NO_EDGES);
    useEffect(() => {
        /* Function to process health stats when used from Graph Summary. GraphSummary
           makes the API call to edge_stats, the results of which can be used
           here, instead of making another API call in this component.
        */
        const processGraphHealthStats = () => {
            if (summaryParams?.message !== "Fetching") {
                const calculating: boolean = summaryParams?.message.startsWith(
                    "Calculating"
                )
                    ? true
                    : false;
                let msg: string =
                    summaryParams?.graphsEdges &&
                    summaryParams?.graphsEdges?.length > 0
                        ? ""
                        : NO_EDGES;
                let data: Array<EdgeStats> = [];
                if (summaryParams?.message === "Success") {
                    msg = summaryParams.graphsEdges?.length > 0 ? "" : NO_EDGES;
                    data = summaryParams.graphsEdges;
                } else if (calculating) {
                    msg = summaryParams?.message as string;
                } else if (isValid(summaryParams?.message)) {
                    const alertErr: CustomError = {
                        response: {
                            data: summaryParams?.message
                        },
                        name: "",
                        message: summaryParams?.message as string
                    };
                    handleError(alertErr, "Something went wrong!");
                }
                setGraphEdges(data);
                setEmptyMessage(msg);
                setCalculating(calculating);
                setPageLoading(false);
            }
        };
        if (summaryParams) {
            //From Summary page
            processGraphHealthStats();
        }
    }, [summaryParams, handleError, id]);

    /**
     * Table columns for Primary and Foreign key show 'TABLE.COLUMN' where
     * TABLE is underlined and linked, whereas COLUMN is not. So we create
     * these customized columns here for rendering.
     *
     */
    const columns = useMemo<ColumnDef<EdgeStats>[]>(
        () => GraphLinkHealthColumns(),
        []
    );

    if (graphEdges?.length === 0) return null;
    return (
        <Card className="p-6 gap-0">
            <Title variant="tertiary-heading" children={"Graph Link Health"} />
            <Spacer height={16} />
            <Description
                variant="secondary"
                size="md"
                className="my-0 py-0 leading-5">
                {
                    "Match rates show the % of rows in A that have at least one matching key in B (and vice-versa)."
                }
            </Description>
            <Spacer height={24} />
            {emptyMessage && graphEdges?.length < 1 ? (
                <Alert enableClose={false}>{emptyMessage}</Alert>
            ) : (
                <DataTable
                    disableRowSelectionHover={true}
                    enableGlobalFilter={false}
                    isLoading={pageLoading || calculating}
                    id={"graph-edge"}
                    columns={columns}
                    data={graphEdges}
                />
            )}
        </Card>
    );
};

export default WithErrorBoundary(GraphLinkHealth);
