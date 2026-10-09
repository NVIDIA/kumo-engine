// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const METHOD_PREFIX = {
    get: "get",
    post: "create",
    put: "update",
    delete: "delete",
    patch: "update"
};

/**
 * Simplifies and normalizes an OpenAPI `operationId` based on the HTTP method
 * and path.
 *
 * This is useful for generating consistent client SDK method names or for
 * resolving duplicate API entries when multiple versions (e.g., v0, v1)
 * have similar paths.
 *
 * The function attempts to:
 * - Normalize the HTTP method using a `METHOD_PREFIX` map.
 * - Respect certain prefixes in the existing `operationId` (e.g., `get`, `validate`, `cancel`).
 * - Construct a new identifier using the path segments, method, and version.
 * - Include any path parameters (e.g., `{userId}`) as suffixes.
 *
 * @param {string} pathKey - The API endpoint path, e.g., `/v1/users/{userId}`.
 * @param {string} method - The HTTP method (e.g., `get`, `post`, `put`).
 * @param {string} [operationId] - The existing operationId from the OpenAPI spec (optional).
 *
 * @returns {string} A simplified and normalized operation ID string, e.g., `get_v1_usersByUserId`.
 */
function simplifyOperationId(pathKey, method, operationId) {
    let newMethod = METHOD_PREFIX[method] || method;
    let methodInOpId = operationId ? operationId?.split("_")[0] : "";
    if (methodInOpId) {
        // replace list with get
        methodInOpId = methodInOpId === "list" ? "get" : methodInOpId;
        // Some API's uses post request for get request, in that case
        // generated types/schema uses `create` as a prefix.
        // Similar to above, validate and cancel request uses get/put
        // as method name and uses that for code generation.
        // So using `prefix` from old `operationId` in case `operationId`
        //  includes get, validate or cancel
        if (["get", "validate", "cancel"].includes(methodInOpId)) {
            newMethod = methodInOpId;
        }
    }
    const pathSegments = pathKey.split("/").filter(Boolean);

    // Including V1 version is because avoid duplicate API entries,
    // in case both v0 and v1 has same end points. prefix `_v1` for v1 endpoints
    const version = pathSegments[0] === "v1" ? "_v1" : "";
    // API tag or section
    const section = pathSegments[1];

    // check if any param used and get last string
    const checkParam = pathSegments.filter(
        (seg) => !seg.startsWith("{") && !seg.endsWith("}")
    );
    const lastSegment = checkParam
        ? checkParam.at(-1)
        : pathSegments[pathSegments.length - 1] || "resource";
    let finalMethod =
        section.toUpperCase() === lastSegment?.toUpperCase()
            ? "_" + lastSegment
            : "_" + section + "_" + lastSegment;
    // A nested slug endpoint such as
    // /projects/{project_id}/identities/by-slug/{slug} must retain its nested
    // resource name. Otherwise it collides with /projects/by-slug/{slug} and
    // Orval silently omits one of the two operations.
    const nestedResource = checkParam.at(-2);
    if (
        lastSegment === "by-slug" &&
        nestedResource &&
        nestedResource !== section
    ) {
        finalMethod = `_${section}_${nestedResource}_${lastSegment}`;
    }
    if (
        methodInOpId &&
        newMethod === methodInOpId &&
        finalMethod.toLowerCase().includes(methodInOpId.toLowerCase())
    ) {
        finalMethod = finalMethod.replace(methodInOpId, "");
    }

    // check if last value is param, if yes then add param by using
    // `by_` prefix
    const matchParam = pathKey.match(/\{[^}]+\}/g);
    const lastParam = matchParam ? matchParam.at(-1).replace("/", "") : "";
    const lastStr = lastParam ? trimProps(lastParam) : "";
    const finalOpId = `${newMethod}${version}${finalMethod}${lastStr}`;
    return finalOpId;
}

function trimProps(str) {
    return "_by_" + str.replace("{", "").replace("}", "");
}

function cleanupDescription(str) {
    return str
        .replace(/\n/g, " ")
        .replace(/\*/g, " ")
        .replace(/\//g, "<slash>")
        .replaceAll('"', "`")
        .trim();
}

function cloneSchema(schema) {
    return JSON.parse(JSON.stringify(schema));
}

function addSchemaAlias(schemas, aliasName, targetName) {
    if (!schemas?.[targetName] || schemas[aliasName]) {
        return;
    }
    schemas[aliasName] = {
        ...cloneSchema(schemas[targetName]),
        title: aliasName
    };
}

function addConstEnumSchema(schemas, schemaName, value) {
    if (schemas?.[schemaName]) {
        return;
    }
    schemas[schemaName] = {
        enum: [value],
        type: "string",
        const: value,
        title: schemaName
    };
}

function addCompatibilitySchemas(spec) {
    const schemas = spec?.components?.schemas;
    if (!schemas) {
        return;
    }

    const sharedAliases = {
        KumoapiGraphGraphDefinition: "GraphDefinition",
        KumoapiGraphGraphDefinitionTables: "GraphDefinitionTables",
        RestConfigGraphV2GraphDefinition: "GraphDefinition",
        RestConfigGraphV2GraphDefinitionTables: "GraphDefinitionTables",
        KumoValidationValidationValidationResponse: "ValidationResponse",
        StatedbConfigTypingTaskType: "TaskType"
    };

    for (const [aliasName, targetName] of Object.entries(sharedAliases)) {
        addSchemaAlias(schemas, aliasName, targetName);
    }

    const v0Aliases = {
        KumoapiJobsJobEventLogEntry: "JobEventLogEntry",
        RegressionHeatmapResponseResidualsChart:
            "RegressionHeatmapResponseResidualsChart",
        RestConfigTableColumnStats: "ColumnStats",
        StatedbConfigCommonJobEventLogEntry: "JobEventLogEntry",
        StatedbConfigCommonJobSource: "JobSource",
        StatedbConfigCommonJobStatus: "JobStatus"
    };

    for (const [aliasName, targetName] of Object.entries(v0Aliases)) {
        addSchemaAlias(schemas, aliasName, targetName);
    }

    if (!schemas.GraphSnapshotResourceStages && schemas.JobState) {
        schemas.GraphSnapshotResourceStages = {
            additionalProperties: {
                $ref: "#/components/schemas/JobState"
            },
            type: "object",
            title: "GraphSnapshotResourceStages"
        };
    }

    if (
        !schemas.RegressionHeatmapResponseResidualsChart &&
        schemas.RegressionResiduals
    ) {
        schemas.RegressionHeatmapResponseResidualsChart = {
            additionalProperties: {
                $ref: "#/components/schemas/RegressionResiduals"
            },
            type: "object",
            title: "RegressionHeatmapResponseResidualsChart"
        };
    }

    const connectorTypeSchemas = {
        BigQueryConnectorResourceConfigType: "bigquery",
        DatabricksConnectorResourceConfigType: "databricks",
        FileConnectorResourceConfigType: "file",
        SnowflakeConnectorResourceConfigType: "snowflake"
    };

    for (const [schemaName, value] of Object.entries(connectorTypeSchemas)) {
        addConstEnumSchema(schemas, schemaName, value);
    }

    const predictionOutputAliases = {
        KumoapiJobsBigQueryPredictionOutput: "BigQueryPredictionOutput",
        KumoapiJobsDatabricksPredictionOutput: "DatabricksPredictionOutput",
        RestConfigJobsBigQueryPredictionOutput: "BigQueryPredictionOutput",
        RestConfigJobsDatabricksPredictionOutput: "DatabricksPredictionOutput"
    };

    for (const [aliasName, targetName] of Object.entries(
        predictionOutputAliases
    )) {
        addSchemaAlias(schemas, aliasName, targetName);
    }

    const predictionStorageTypeSchemas = {
        KumoapiJobsADLSPredictionOutputStorageType: "ADLS",
        KumoapiJobsBigQueryPredictionOutputStorageType: "BIGQUERY",
        KumoapiJobsDatabricksPredictionOutputStorageType: "DATABRICKS",
        KumoapiJobsGCSPredictionOutputStorageType: "GCS",
        KumoapiJobsS3PredictionOutputStorageType: "S3",
        KumoapiJobsSnowflakePredictionOutputStorageType: "SNOWFLAKE",
        LocalPredictionOutputStorageType: "LOCAL",
        RestConfigJobsADLSPredictionOutputStorageType: "ADLS",
        RestConfigJobsBigQueryPredictionOutputStorageType: "BIGQUERY",
        RestConfigJobsDatabricksPredictionOutputStorageType: "DATABRICKS",
        RestConfigJobsGCSPredictionOutputStorageType: "GCS",
        RestConfigJobsS3PredictionOutputStorageType: "S3",
        RestConfigJobsSnowflakePredictionOutputStorageType: "SNOWFLAKE"
    };

    for (const [schemaName, value] of Object.entries(
        predictionStorageTypeSchemas
    )) {
        addConstEnumSchema(schemas, schemaName, value);
    }
}

function preProcess(spec) {
    addCompatibilitySchemas(spec);

    for (const [pathKey, pathObj] of Object.entries(spec.paths)) {
        for (const method of Object.keys(pathObj)) {
            if (["get", "post", "put", "delete", "patch"].includes(method)) {
                const op = pathObj[method];
                const newId = simplifyOperationId(
                    pathKey,
                    method,
                    op.operationId ?? ""
                );

                if (spec?.paths[pathKey][method]?.operationId) {
                    spec.paths[pathKey][method].operationId = newId;
                }
                if (spec?.paths[pathKey][method]?.description) {
                    // Remove empty spaces and `\n`
                    spec.paths[pathKey][method].description =
                        cleanupDescription(
                            spec.paths[pathKey][method].description
                        );
                }

                // To remove Under develpoment prefix/sufix from code gen
                if (op.tags) {
                    op.tags?.forEach((tag, i) => {
                        op.tags[i] = tag.replace(
                            "API v1 (Under Development): ",
                            ""
                        );
                    });
                }
            }
        }
    }

    // Some description fields contains \ (backslash) and typescript throws
    // error for linting.
    // Remove unnecessary escape character: \ (backslash) to avoid typescript
    // error
    if (spec?.components?.schemas) {
        for (const [schemaKey, schemaObj] of Object.entries(
            spec?.components?.schemas ?? {}
        )) {
            if (schemaObj.description) {
                let cleaned_backslash = cleanupDescription(
                    schemaObj.description
                );
                spec.components.schemas[schemaKey].description =
                    cleaned_backslash;
            }
            if (schemaObj.enum) {
                const isAllNumeric = schemaObj.enum.every(
                    (value) =>
                        typeof value === "number" && Number.isFinite(value)
                );
                if (!isAllNumeric) {
                    schemaObj["x-enumNames"] = schemaObj.enum.map((e) =>
                        e?.toString().toUpperCase().replaceAll(".", "_")
                    );
                }
            }
        }
    }
    // If the schemas has a `Index` key, Orval creates a file for this inside
    // types folder with name index.ts. once all types are generated, to export
    // all the generated types, Orval uses index.ts. so this conflicts and
    // overrides the previously generated type file, here we renaming Index
    // to KumoIndex to avoid this conflicts

    if (spec?.components?.schemas?.Index) {
        spec.components.schemas.KumoIndex = spec.components.schemas.Index;
        const specStr = JSON.stringify(spec);
        const updatedSpecStr = specStr.replace(
            /"\$ref":\s*"#\/components\/schemas\/Index"/g,
            '"$ref": "#/components/schemas/KumoIndex"'
        );
        return JSON.parse(updatedSpecStr);
    }

    return spec;
}

// eslint-disable-next-line no-undef
module.exports = preProcess;
