// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isValid } from "utils/Helpers/typeGuards";

export const parsePresignedURLasDatabricks = (url: string): string | null => {
    try {
        const [, , catalog, schema, volume, ...pathParts] = url
            ? url.split("/")
            : [];
        const path = pathParts?.join("/");
        if (
            isValid(catalog) &&
            isValid(schema) &&
            isValid(volume) &&
            isValid(path)
        ) {
            const command =
                `Holdout data is available in your Unity Catalog Volume:\n` +
                `Catalog: ${catalog}\n` +
                `Schema: ${schema}\n` +
                `Volume: ${volume}\n` +
                `File Path: ${path}\n` +
                `Full Unity Catalog Volume Path: /Volumes/${catalog}/${schema}/${volume}/${path}`;
            return command;
        } else {
            console.error(
                "Error parsing databricks url",
                url,
                catalog,
                schema,
                volume,
                path
            );
            return null;
        }
    } catch (error) {
        console.error("Error parsing databricks url", url, error);
        return null;
    }
};
