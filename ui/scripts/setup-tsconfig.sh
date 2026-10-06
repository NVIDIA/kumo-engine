#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Setup tsconfig.local.json for TypeScript path resolution in your editor
#
# Usage:
#   ./scripts/setup-tsconfig.sh --local  # Enable local catalog paths for editor intellisense
#   ./scripts/setup-tsconfig.sh --reset  # Reset to default (no catalog paths)

set -e

UI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CATALOG_LOCAL_PATH="${UI_CATALOG_PATH:-$UI_DIR/../packages/kumo-ui-catalog}"

show_usage() {
    echo "Usage: $0 [--local|--reset]"
    echo ""
    echo "Options:"
    echo "  --local  Configure editor TypeScript paths for local catalog development"
    echo "  --reset  Reset to default (no catalog paths)"
    exit 1
}

check_catalog_exists() {
    if [ ! -d "$CATALOG_LOCAL_PATH" ]; then
        echo "Error: kumo-ui-catalog repo not found at $CATALOG_LOCAL_PATH"
        echo ""
        echo "The in-tree catalog source must be present:"
        echo "  git status -- packages/kumo-ui-catalog"
        echo ""
        echo "Or set UI_CATALOG_PATH to point to your catalog location."
        exit 1
    fi
}

create_local_config() {
    check_catalog_exists
    node - "$UI_DIR" "$CATALOG_LOCAL_PATH" << 'JS'
const fs = require("fs");
const path = require("path");
const [uiDir, catalogDir] = process.argv.slice(2);
const source = path.join(fs.realpathSync(catalogDir), "src");
const relative = path.relative(path.join(uiDir, "src"), source).split(path.sep).join("/");
const config = {
    compilerOptions: {
        paths: {
            "@kumo-ai/kumo-ui-catalog": [`${relative}/index.ts`],
            "@kumo-ai/kumo-ui-catalog/*": [`${relative}/*`],
            "@/*": [`${relative}/*`]
        }
    }
};
fs.writeFileSync(path.join(uiDir, "tsconfig.local.json"), JSON.stringify(config, null, 4) + "\n");
JS
    echo "Created tsconfig.local.json for local catalog development"
}

create_reset_config() {
    cat > "$UI_DIR/tsconfig.local.json" << 'EOF'
{
    "compilerOptions": {}
}
EOF
    echo "Reset tsconfig.local.json to default"
}

# Main
case "$1" in
    --local)
        create_local_config
        ;;
    --reset)
        create_reset_config
        ;;
    *)
        show_usage
        ;;
esac

# Tell git to ignore local changes to this file
git update-index --skip-worktree "$UI_DIR/tsconfig.local.json" 2>/dev/null || true

echo ""
echo "Done! To pick up changes, open a .ts/.tsx file and restart the TypeScript server:"
echo "  VS Code: Ctrl+Shift+P -> 'TypeScript: Restart TS Server'"
echo "  Or simply close and reopen your editor."
