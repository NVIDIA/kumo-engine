// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import crypto from "crypto";
import fs from "fs";
import path from "path";

const BAR = "=".repeat(78);
const RULE = "-".repeat(78);
const LEGAL_FILE =
    /(?:^|[._-])(?:licen[cs]e|notice|copying|authors?|copyright)(?:$|[._-])/i;
const LICENSE_FILE = /(?:^|[._-])(?:licen[cs]e|copying)(?:$|[._-])/i;
const SOURCE_FILE = /\.(?:[cm]?js|jsx|tsx?)$/i;

function fail(message) {
    throw new Error(message);
}

function packageLicense(metadata) {
    const declared = metadata.license ?? metadata.licenses;
    if (typeof declared === "string" && declared.trim()) {
        return declared.trim();
    }
    if (Array.isArray(declared) && declared.length > 0) {
        return declared
            .map((item) => (typeof item === "string" ? item : item.type))
            .filter(Boolean)
            .join(" OR ");
    }
    if (declared && typeof declared.type === "string") {
        return declared.type;
    }
    fail(`${metadata.name}@${metadata.version} has no declared license`);
}

function sha256(contents) {
    return crypto.createHash("sha256").update(contents).digest("hex");
}

function loadOverrides(root) {
    const filename = path.join(root, "third_party/npm-license-overrides.json");
    if (!fs.existsSync(filename)) {
        fail(`npm license override inventory is missing: ${filename}`);
    }
    const overrides = JSON.parse(fs.readFileSync(filename, "utf8"));
    if (!overrides.packages || !overrides.sources) {
        fail(`${filename} must define packages and sources`);
    }
    return overrides;
}

function applyOverrides(root, packages, overrides) {
    const usedPackages = new Set();
    const usedSources = new Set();
    for (const component of packages) {
        if (
            [...component.files.values()].some((file) =>
                LICENSE_FILE.test(file.basename)
            )
        ) {
            continue;
        }
        const identity = `${component.name}@${component.version}`;
        const sourceName = overrides.packages[identity];
        if (!sourceName) {
            fail(`${identity} has no legal text or reviewed override`);
        }
        const source = overrides.sources[sourceName];
        if (!source) {
            fail(`${identity} references unknown legal source ${sourceName}`);
        }
        if (source.license !== component.license) {
            fail(
                `${identity} declares ${component.license}, not reviewed ${source.license}`
            );
        }
        usedPackages.add(identity);
        const filename = path.resolve(root, source.path);
        const rootPrefix = `${path.resolve(root)}${path.sep}`;
        if (!filename.startsWith(rootPrefix) || !fs.existsSync(filename)) {
            fail(`${identity} legal source is missing: ${source.path}`);
        }
        const contents = fs.readFileSync(filename, "utf8").trimEnd();
        const digest = sha256(contents);
        if (!contents || digest !== source.sha256) {
            fail(`${identity} legal source hash differs: ${source.path}`);
        }
        component.files.set(`${path.basename(filename)}\0${digest}`, {
            basename: path.basename(filename),
            contents,
            digest
        });
        usedSources.add(sourceName);
    }
    const stalePackages = Object.keys(overrides.packages).filter(
        (identity) => !usedPackages.has(identity)
    );
    if (stalePackages.length > 0) {
        fail(`stale npm license overrides: ${stalePackages.join(", ")}`);
    }
    const staleSources = Object.keys(overrides.sources).filter(
        (source) => !usedSources.has(source)
    );
    if (staleSources.length > 0) {
        fail(`unused npm legal sources: ${staleSources.join(", ")}`);
    }
}

function bundledDirectories(sourceMap) {
    if (!sourceMap) {
        return null;
    }
    const directories = new Set();
    const maps = fs.statSync(sourceMap).isDirectory()
        ? fs
              .readdirSync(sourceMap)
              .filter((name) => name.endsWith(".map"))
              .map((name) => path.join(sourceMap, name))
        : [sourceMap];
    if (maps.length === 0) {
        fail(`${sourceMap} contains no source maps`);
    }
    for (const filename of maps) {
        const map = JSON.parse(fs.readFileSync(filename, "utf8"));
        if (!Array.isArray(map.sources)) {
            fail(`${filename} does not contain a sources array`);
        }
        for (const source of map.sources) {
            const segments = path
                .resolve(path.dirname(filename), source)
                .split(path.sep);
            const nodeModules = segments.lastIndexOf("node_modules");
            if (nodeModules < 0 || !segments[nodeModules + 1]) {
                continue;
            }
            const length = segments[nodeModules + 1].startsWith("@") ? 3 : 2;
            directories.add(
                segments.slice(0, nodeModules + length).join(path.sep)
            );
        }
    }
    return directories;
}

function packageDirectory(nodeModules, specifier) {
    if (
        !specifier ||
        specifier.startsWith(".") ||
        specifier.startsWith("/") ||
        specifier.includes(":")
    ) {
        return null;
    }
    const segments = specifier.split("/");
    const packageName = specifier.startsWith("@")
        ? segments.slice(0, 2).join("/")
        : segments[0];
    const directory = path.resolve(nodeModules, packageName);
    return directory.startsWith(`${nodeModules}${path.sep}`) &&
        fs.existsSync(directory)
        ? directory
        : null;
}

function cssDirectories(root, cssRoot) {
    if (!cssRoot) {
        return new Set();
    }
    const resolvedRoot = path.resolve(root);
    const sourceRoot = path.resolve(cssRoot);
    if (
        sourceRoot !== resolvedRoot &&
        !sourceRoot.startsWith(`${resolvedRoot}${path.sep}`)
    ) {
        fail(`CSS source root is outside the package root: ${cssRoot}`);
    }
    if (!fs.statSync(sourceRoot).isDirectory()) {
        fail(`CSS source root is not a directory: ${cssRoot}`);
    }
    const nodeModules = path.resolve(root, "node_modules");
    const directories = new Set();
    const pending = [sourceRoot];
    while (pending.length > 0) {
        const directory = pending.pop();
        for (const entry of fs.readdirSync(directory, {
            withFileTypes: true
        })) {
            const filename = path.join(directory, entry.name);
            if (entry.isDirectory()) {
                pending.push(filename);
                continue;
            }
            if (!entry.isFile()) {
                continue;
            }
            const extension = path.extname(entry.name);
            if (
                ![".css", ".js", ".jsx", ".mjs", ".ts", ".tsx"].includes(
                    extension
                )
            ) {
                continue;
            }
            const contents = fs
                .readFileSync(filename, "utf8")
                .replace(/\/\*[\s\S]*?\*\//g, "");
            const directives =
                extension === ".css"
                    ? /@(?:import|plugin)\s+(?:url\(\s*)?["']([^"']+)["']/g
                    : /["']([^"']+\.css(?:\?[^"']*)?)["']/g;
            for (const match of contents.matchAll(directives)) {
                const packagePath = packageDirectory(nodeModules, match[1]);
                if (packagePath) {
                    directories.add(packagePath);
                }
            }
        }
    }
    return directories;
}

function includedDirectories(root, includedPackages) {
    const directories = new Set();
    const nodeModules = path.resolve(root, "node_modules");
    for (const packageName of (includedPackages ?? "").split(",")) {
        if (!packageName) {
            continue;
        }
        const directory = packageDirectory(nodeModules, packageName);
        if (!directory) {
            fail(`invalid explicitly bundled package: ${packageName}`);
        }
        directories.add(directory);
    }
    return directories;
}

function installedPackages(root, sourceMap, includedPackages, cssRoot) {
    const lock = JSON.parse(
        fs.readFileSync(path.join(root, "package-lock.json"), "utf8")
    );
    if (!lock.packages || typeof lock.packages !== "object") {
        fail("package-lock.json does not contain a packages inventory");
    }

    const bundled = bundledDirectories(sourceMap);
    const included = includedDirectories(root, includedPackages);
    const css = cssDirectories(root, cssRoot);
    const selected = new Set([...(bundled ?? []), ...included, ...css]);
    const packages = new Map();
    const matchedDirectories = new Set();
    for (const [relative, locked] of Object.entries(lock.packages)) {
        if (!relative.startsWith("node_modules/") || locked.link === true) {
            if (locked.link === true) {
                selected.delete(path.resolve(root, relative));
            }
            continue;
        }
        const directory = path.join(root, relative);
        const resolvedDirectory = path.resolve(directory);
        if (bundled && !selected.has(resolvedDirectory)) {
            continue;
        }
        if (
            !bundled &&
            locked.dev === true &&
            !included.has(resolvedDirectory)
        ) {
            continue;
        }
        const manifest = path.join(directory, "package.json");
        if (!fs.existsSync(manifest)) {
            if (locked.optional === true) {
                continue;
            }
            fail(`locked production package is not installed: ${relative}`);
        }
        matchedDirectories.add(resolvedDirectory);
        const metadata = JSON.parse(fs.readFileSync(manifest, "utf8"));
        if (!metadata.name || !metadata.version) {
            fail(`${manifest} lacks name or version`);
        }
        const identity = `${metadata.name}@${metadata.version}`;
        const record = packages.get(identity) ?? {
            files: new Map(),
            license: packageLicense(metadata),
            name: metadata.name,
            version: metadata.version
        };
        for (const basename of fs.readdirSync(directory).sort()) {
            const candidate = path.join(directory, basename);
            if (
                !LEGAL_FILE.test(basename) ||
                SOURCE_FILE.test(basename) ||
                !fs.statSync(candidate).isFile()
            ) {
                continue;
            }
            const contents = fs.readFileSync(candidate, "utf8").trimEnd();
            if (contents) {
                const digest = sha256(contents);
                record.files.set(`${basename}\0${digest}`, {
                    basename,
                    contents,
                    digest
                });
            }
        }
        packages.set(identity, record);
    }
    if (selected.size > 0) {
        const unmatched = [...selected].filter(
            (directory) => !matchedDirectories.has(directory)
        );
        if (unmatched.length > 0) {
            fail(
                `source map packages are absent from package-lock.json: ${unmatched.join(", ")}`
            );
        }
    }
    const installed = [...packages.values()].sort((left, right) =>
        `${left.name.toLowerCase()}\0${left.version}`.localeCompare(
            `${right.name.toLowerCase()}\0${right.version}`
        )
    );
    applyOverrides(root, installed, loadOverrides(root));
    return installed;
}

function render(packages) {
    const texts = new Map();
    for (const component of packages) {
        for (const file of component.files.values()) {
            texts.set(file.digest, file.contents);
        }
    }
    const textIds = new Map(
        [...texts.keys()].map((digest) => [digest, digest.slice(0, 12)])
    );
    const emitted = new Set();
    const lines = [BAR, "Kumo UI npm third-party notices", BAR, ""];
    for (const component of packages) {
        lines.push(
            RULE,
            `${component.name} ${component.version}`,
            RULE,
            `  License: ${component.license}`,
            ""
        );
        for (const file of [...component.files.values()].sort((left, right) =>
            left.basename.localeCompare(right.basename)
        )) {
            const textId = textIds.get(file.digest);
            lines.push(`  Legal text ${textId}: ${file.basename}`);
            if (!emitted.has(file.digest)) {
                emitted.add(file.digest);
                lines.push(
                    "",
                    ...file.contents
                        .split("\n")
                        .map((line) => (line ? `    ${line}` : "")),
                    ""
                );
            }
        }
    }
    return `${lines.join("\n").trimEnd()}\n`;
}

function argumentsFrom(argv) {
    const values = {};
    for (let index = 0; index < argv.length; index += 2) {
        const option = argv[index];
        const value = argv[index + 1];
        if (!option?.startsWith("--") || value === undefined) {
            fail(
                "usage: generate-third-party-notice.mjs --root DIR --output FILE [--source-map FILE] [--css-root DIR] [--include-packages NAME,...]"
            );
        }
        values[option.slice(2)] = value;
    }
    if (!values.root || !values.output) {
        fail(
            "usage: generate-third-party-notice.mjs --root DIR --output FILE [--source-map FILE] [--css-root DIR] [--include-packages NAME,...]"
        );
    }
    return values;
}

try {
    const args = argumentsFrom(process.argv.slice(2));
    const output = path.resolve(args.output);
    fs.mkdirSync(path.dirname(output), { recursive: true });
    fs.writeFileSync(
        output,
        render(
            installedPackages(
                path.resolve(args.root),
                args["source-map"] ? path.resolve(args["source-map"]) : null,
                args["include-packages"],
                args["css-root"] ? path.resolve(args["css-root"]) : null
            )
        )
    );
} catch (error) {
    process.stderr.write(`npm third-party notice error: ${error.message}\n`);
    process.exitCode = 1;
}
