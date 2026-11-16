#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/* eslint-disable no-undef */
/* eslint-disable @typescript-eslint/no-require-imports */

const fs = require("fs");
const path = require("path");

function extractResponseVariables(sourceContent) {
    const responseVariables = [];
    const lines = sourceContent.split("\n");

    for (const line of lines) {
        // Match export const/let/var declarations ending with "Response"
        const match = line.match(
            /export\s+(?:const|let|var)\s+(\w+Response)\s*=/
        );
        if (match) {
            responseVariables.push(match[1]);
        }
    }

    return responseVariables;
}

function updateApiClientsWithResponseSchemas(content, responseVariables) {
    let updatedContent = content;

    for (const responseVariable of responseVariables) {
        // Strip "Response" suffix and lowercase first char to match orval v8 camelCase function names
        const withoutSuffix = responseVariable.replace(/Response$/, "");
        const functionName =
            withoutSuffix.charAt(0).toLowerCase() + withoutSuffix.slice(1);
        // Create a regex to match the entire function export and replace options in customClient calls
        const functionRegex = new RegExp(
            `(export const ${functionName} = [\\s\\S]*?customClient[\\s\\S]*?\\([\\s\\S]*?,\\s*)options(\\s*\\)[\\s\\S]*?};)`,
            "g"
        );

        const match = functionRegex.exec(updatedContent);
        if (match) {
            const replacement = `${match[1]}options || ${responseVariable}${match[2]}`;
            updatedContent = updatedContent.replace(match[0], replacement);
            console.log(
                `Updated ${functionName} to use options || ${responseVariable}`
            );
        } else {
            console.log(
                `Warning: Could not find function ${functionName} or customClient call`
            );
        }
    }

    return updatedContent;
}

function addImportsToDestination(
    destContent,
    sourceFilePath,
    responseVariables,
    destFilePath
) {
    if (responseVariables.length === 0) {
        return destContent;
    }

    // Check which variables are already imported
    const alreadyImported = new Set();

    // Use regex to find all import statements, including multiline ones
    const importRegex = /import\s*{([^}]+)}\s*from\s*['"][^'"]+['"]/g;
    let importMatch;

    while ((importMatch = importRegex.exec(destContent)) !== null) {
        // Extract variables from the import statement
        const importedVarsString = importMatch[1];
        const importedVars = importedVarsString
            .split(",")
            .map((v) => v.trim())
            .filter((v) => v.length > 0);

        importedVars.forEach((variable) => alreadyImported.add(variable));
    }

    // Filter out already imported variables
    const variablesToImport = responseVariables.filter(
        (variable) => !alreadyImported.has(variable)
    );

    if (variablesToImport.length === 0) {
        console.log("All Response variables are already imported, skipping...");
        return destContent;
    }

    console.log(
        `Found ${alreadyImported.size} already imported variables: ${Array.from(alreadyImported).join(", ")}`
    );
    console.log(
        `Will import ${variablesToImport.length} new variables: ${variablesToImport.join(", ")}`
    );

    // Create the import statement with variables on separate lines
    const relativePath = path
        .relative(path.dirname(destFilePath), sourceFilePath)
        .replace(/\.ts$/, "");
    const importPath = relativePath.startsWith(".")
        ? relativePath
        : "./" + relativePath;

    const importStatement = `import {\n${variablesToImport.map((variable) => `  ${variable}`).join(",\n")}\n} from '${importPath}';\n`;

    // Split content into lines for processing
    const lines = destContent.split("\n");

    // Find the position to insert the import (after existing imports or at the top)
    let insertIndex = 0;

    // Find the last import statement
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        if (line.startsWith("import ") && line.includes("from ")) {
            insertIndex = i + 1;
        } else if (
            line &&
            !line.startsWith("//") &&
            !line.startsWith("/*") &&
            !line.startsWith("*")
        ) {
            // Stop at the first non-import, non-comment line
            break;
        }
    }

    // Insert the import statement
    lines.splice(insertIndex, 0, importStatement);

    let contentWithImports = lines.join("\n");

    // Update API clients to use response schemas - only process variables that were newly imported
    contentWithImports = updateApiClientsWithResponseSchemas(
        contentWithImports,
        variablesToImport
    );

    return contentWithImports;
}

function processFile(sourcePath, destPath) {
    try {
        // Read source file and extract Response variables
        const sourceContent = fs.readFileSync(sourcePath, "utf8");
        const responseVariables = extractResponseVariables(sourceContent);

        if (responseVariables.length === 0) {
            console.log(
                `No variables ending with "Response" found in ${sourcePath}`
            );
            return;
        }

        console.log(
            `Processing ${path.basename(sourcePath)}: Found ${responseVariables.length} Response variable(s): ${responseVariables.join(", ")}`
        );

        // Read destination file and add imports
        const destContent = fs.readFileSync(destPath, "utf8");
        const updatedContent = addImportsToDestination(
            destContent,
            sourcePath,
            responseVariables,
            destPath
        );

        // Write back to destination file
        fs.writeFileSync(destPath, updatedContent);

        console.log(`Successfully updated ${destPath}`);
    } catch (error) {
        console.error(`Error processing ${sourcePath}: ${error.message}`);
    }
}

function processAllFiles() {
    const versions = ["v0", "v1"];

    for (const version of versions) {
        const schemasDir = `./src/gen/${version}/schemas`;
        const apiBaseDir = `./src/gen/${version}/api`;

        // Check if schemas directory exists
        if (!fs.existsSync(schemasDir)) {
            console.log(
                `Skipping ${version}: schemas directory does not exist`
            );
            continue;
        }

        console.log(`\nProcessing ${version} files...`);

        // Get all .ts files in schemas directory
        const schemaFiles = fs
            .readdirSync(schemasDir)
            .filter((file) => file.endsWith(".ts"));

        if (schemaFiles.length === 0) {
            console.log(`No schema files found in ${schemasDir}`);
            continue;
        }

        for (const schemaFile of schemaFiles) {
            const sourcePath = path.join(schemasDir, schemaFile);
            const baseName = path.basename(schemaFile, ".ts");
            const destPath = path.join(apiBaseDir, baseName, schemaFile);

            // Check if destination file exists
            if (!fs.existsSync(destPath)) {
                console.log(
                    `Warning: Destination file does not exist: ${destPath}`
                );
                continue;
            }

            processFile(sourcePath, destPath);
        }
    }
}

function main() {
    const args = process.argv.slice(2);

    if (args.length === 0) {
        // No arguments - process all files
        console.log("Processing all schema files...");
        processAllFiles();
    } else if (args.length === 2) {
        // Two arguments - process specific files
        const [sourcePath, destPath] = args;

        // Validate file paths
        if (!fs.existsSync(sourcePath)) {
            console.error(`Source file does not exist: ${sourcePath}`);
            process.exit(1);
        }

        if (!fs.existsSync(destPath)) {
            console.error(`Destination file does not exist: ${destPath}`);
            process.exit(1);
        }

        processFile(sourcePath, destPath);
    } else {
        console.error(
            "Usage: node import-responses.cjs [<source.ts> <destination.ts>]"
        );
        console.error(
            "  With no arguments: processes all files in schemas directory"
        );
        console.error(
            "  With two arguments: processes specific source and destination files"
        );
        process.exit(1);
    }
}

main();
