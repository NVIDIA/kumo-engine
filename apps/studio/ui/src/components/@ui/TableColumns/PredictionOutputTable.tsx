// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    ColumnDef,
    CopyToClipBoard,
    Row,
    Symbol,
    Tooltip,
    TooltipContent,
    TooltipTrigger
} from "@kumo-ai/kumo-ui-catalog";

import { ColumnArgs } from "types/tables";

import { bytesToSize, parseNumbers } from "utils/Helpers/numberHelpers";
import { capitalizeFirstLetter } from "utils/Helpers/stringHelpers";
import { isValid } from "utils/Helpers/typeGuards";
import { isObjectStoreOutputStorage } from "utils/objectStoreTablePath";

export const PredictionOutputTableColumns = (
    storage_type?:
        | "ADLS"
        | "BIGQUERY"
        | "SNOWFLAKE"
        | "DATABRICKS"
        | "S3"
        | "GCS"
        | "LOCAL"
): ColumnDef<ColumnArgs & { component?: React.ReactNode }>[] => {
    return [
        {
            id: "artifact_type",
            header: () => (
                <div className="text-title-secondary font-medium">
                    Output Type
                </div>
            ),
            accessorKey: "",
            cell: ({
                row
            }: {
                row: Row<
                    { artifact_type?: string } & {
                        component?: React.ReactNode;
                    }
                >;
            }) => {
                const value = row.original.artifact_type as string;
                if (value) {
                    return (
                        <div className="text-title-primary whitespace-nowrap">
                            {capitalizeFirstLetter(value?.toLowerCase())}
                        </div>
                    );
                }
                return <Symbol type="emdash" />;
            }
        },
        isObjectStoreOutputStorage(storage_type)
            ? {
                  id: "output_path",
                  header: () => (
                      <div className="text-title-secondary font-medium">
                          Output Path
                      </div>
                  ),
                  accessorKey: "",
                  cell: ({
                      row
                  }: {
                      row: Row<
                          { output_path?: string } & {
                              component?: React.ReactNode;
                          }
                      >;
                  }) => {
                      const value = row.original.output_path as string;
                      if (value && !value?.toUpperCase()?.includes("N/A")) {
                          return <CopyCell value={value} />;
                      }
                      return <Symbol type="emdash" />;
                  }
              }
            : {
                  id: "table_name",
                  header: () => (
                      <div className="text-title-secondary font-medium">
                          Table Name
                      </div>
                  ),
                  accessorKey: "",
                  cell: ({
                      row
                  }: {
                      row: Row<
                          { table_name?: string } & {
                              component?: React.ReactNode;
                          }
                      >;
                  }) => {
                      const value = row.original.table_name as string;
                      if (value && !value?.toUpperCase()?.includes("N/A")) {
                          return <CopyCell value={value} />;
                      }
                      return <Symbol type="emdash" />;
                  }
              },
        {
            id: "max_categories",
            header: () => (
                <div className="text-title-secondary font-medium">
                    Max Categories
                </div>
            ),
            accessorKey: "",
            cell: ({
                row
            }: {
                row: Row<
                    { max_categories?: string } & {
                        component?: React.ReactNode;
                    }
                >;
            }) => {
                const value = row.original.max_categories as string;
                if (value) {
                    return <div className="text-title-primary">{value}</div>;
                }
                return <Symbol type="emdash" />;
            }
        },
        ...(isObjectStoreOutputStorage(storage_type)
            ? [
                  {
                      id: "file_type",
                      header: () => (
                          <div className="text-title-secondary font-medium">
                              File Type
                          </div>
                      ),
                      accessorKey: "",
                      cell: ({
                          row
                      }: {
                          row: Row<
                              { output_path?: string } & {
                                  component?: React.ReactNode;
                              }
                          >;
                      }) => {
                          if (
                              isValid(row?.original?.output_path) &&
                              row?.original?.output_path?.includes(".")
                          ) {
                              const fileType = row?.original?.output_path
                                  ?.split(".")
                                  .pop();
                              if (fileType)
                                  return (
                                      <div className="text-title-primary">
                                          {capitalizeFirstLetter(
                                              fileType
                                                  .replaceAll("/", "")
                                                  .toLowerCase()
                                          )}
                                      </div>
                                  );
                          }
                          return <Symbol type="emdash" />;
                      }
                  }
              ]
            : []),
        {
            id: "configs",
            header: () => (
                <div className="text-title-secondary font-medium">Configs</div>
            ),
            accessorKey: "",
            cell: ({
                row
            }: {
                row: Row<
                    { num_cols?: number; num_rows?: number } & {
                        component?: React.ReactNode;
                    }
                >;
            }) => (
                <div>
                    {isValid(row?.original?.num_cols) ? (
                        <label className="text-title-primary">
                            {`${parseNumbers(row?.original?.num_rows, { millionSystem: true })} rows X ${row?.original?.num_cols} columns`}
                        </label>
                    ) : (
                        <Symbol type="emdash" />
                    )}
                </div>
            )
        },
        {
            id: "size",
            header: () => (
                <div className="text-title-secondary font-medium">Size</div>
            ),
            accessorKey: "",
            enableHiding: true,
            cell: ({
                row
            }: {
                row: Row<
                    { size_bytes?: number } & {
                        component?: React.ReactNode;
                    }
                >;
            }) => (
                <div>
                    {row?.original?.size_bytes ? (
                        <label className="text-title-primary">
                            {`${bytesToSize(row?.original?.size_bytes)}`}
                        </label>
                    ) : (
                        <Symbol type="emdash" />
                    )}
                </div>
            )
        }
    ];
};

const CopyCell = ({ value }: { value: string }) => {
    return (
        <span className="flex items-center">
            {value?.length > 40 ? (
                <Tooltip id={"output_path"}>
                    <TooltipTrigger>
                        <span className="text-title-primary w-75 whitespace-nowrap text-ellipsis">
                            {value}
                        </span>
                    </TooltipTrigger>
                    <TooltipContent arrow={false} sideOffset={5}>
                        {value}
                    </TooltipContent>
                </Tooltip>
            ) : (
                value
            )}
            <CopyToClipBoard
                copiedText={value as string}
                iconSize={16}
                tooltipText={"Copied"}
            />
        </span>
    );
};
