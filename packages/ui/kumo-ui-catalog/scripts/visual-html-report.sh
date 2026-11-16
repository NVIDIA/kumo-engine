#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# Generate a single self-contained HTML report from visual regression test output.
# All images are base64-encoded inline so the file works when opened via file:// URLs.
# If /tmp/visual-report.txt already exists (pre-populated by CI), skip re-running tests.

if [ ! -f /tmp/visual-report.txt ]; then
    echo "Running visual regression tests..."
    npm run test:visual 2>&1 | tee /tmp/visual-report.txt
else
    echo "Using existing test output from /tmp/visual-report.txt"
fi

# Extract pass/fail counts
passed=$(grep -E "Tests.*passed" /tmp/visual-report.txt | grep -oE "[0-9]+ passed" | head -1 | grep -oE "[0-9]+")
failed=$(grep -E "Tests.*failed" /tmp/visual-report.txt | grep -oE "[0-9]+ failed" | head -1 | grep -oE "[0-9]+")

if [ -z "$passed" ]; then passed=0; fi
if [ -z "$failed" ]; then failed=0; fi

total=$((passed + failed))

# Count total stories from story files (excluding design token stories which are documentation-only)
total_stories=$(grep -roE "export\s+const\s+\w+\s*[=:]" --include="*.stories.tsx" --exclude-dir="designtokens" src/stories/ | grep -vcE "export\s+const\s+(meta|_)")
if [ -z "$total_stories" ]; then total_stories=0; fi

# Count stories that have visual tests (story files imported by .visual.test.tsx files)
with_visual_tests=0
for vtest in $(find src -name "*.visual.test.tsx" -not -path "*__screenshots__*" 2>/dev/null); do
    # Extract the story file path from the import statement
    story_import=$(grep -oE "from\s+[\"']@/stories/[^\"']+" "$vtest" | sed "s|from [\"']@/stories/||")
    if [ -n "$story_import" ]; then
        story_file="src/stories/${story_import}.tsx"
        if [ -f "$story_file" ]; then
            count=$(grep -oE "export\s+const\s+\w+\s*[=:]" "$story_file" | grep -vcE "export\s+const\s+(meta|_)")
            with_visual_tests=$((with_visual_tests + count))
        fi
    fi
done

if [ $total_stories -gt 0 ]; then
    coverage_pct=$(awk "BEGIN {printf \"%.1f\", ($with_visual_tests / $total_stories) * 100}")
else
    coverage_pct="0.0"
fi

# Determine coverage color
coverage_int=$(awk "BEGIN {printf \"%.0f\", $coverage_pct}")
if [ $coverage_int -ge 80 ]; then
    coverage_color="#0cce6b"
    coverage_label="Good"
elif [ $coverage_int -ge 50 ]; then
    coverage_color="#ffa400"
    coverage_label="Partial"
else
    coverage_color="#ff4e42"
    coverage_label="Low"
fi

# Create report directory
mkdir -p coverage/visual

# --- Parse failures from test output ---
sed 's/\x1b\[[0-9;]*m//g' /tmp/visual-report.txt | grep -v '::error' | grep -v '^::' > /tmp/visual-report-clean.txt

SCREENSHOTS_ROOT="src/stories/components/ui/__tests__/VisualRegression/__screenshots__"

# Parse failed tests
cat > /tmp/visual-parse.awk << 'AWKSCRIPT'
BEGIN { component=""; story=""; test_file=""; pixels=""; ratio="" }

/FAIL.*visual\.test\.tsx.*>.*story matches screenshot/ {
    if (component != "" && story != "") {
        prefix = tolower(component)
        printf "%s\t%s\t%s\t%s\t%s\t%s\n", component, story, prefix, test_file, pixels, ratio
    }

    component = ""
    story = ""
    test_file = ""
    pixels = ""
    ratio = ""

    line = $0
    match(line, /src\/[a-zA-Z0-9_\/-]+\.visual\.test\.tsx/)
    if (RSTART > 0) {
        test_file = substr(line, RSTART, RLENGTH)
    }

    idx = index(line, "> ")
    if (idx > 0) {
        rest = substr(line, idx + 2)
        idx2 = index(rest, " > ")
        if (idx2 > 0) {
            comp_desc = substr(rest, 1, idx2 - 1)
            sub(/ Visual Regression$/, "", comp_desc)
            component = comp_desc

            story_rest = substr(rest, idx2 + 3)
            sub(/ story matches screenshot$/, "", story_rest)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", story_rest)
            story = story_rest
        }
    }
    next
}

/pixels \(ratio/ {
    line = $0
    match(line, /[0-9]+ pixels/)
    if (RSTART > 0) {
        pixels = substr(line, RSTART, RLENGTH)
        sub(/ pixels$/, "", pixels)
    }
    match(line, /ratio [0-9.]+/)
    if (RSTART > 0) {
        ratio = substr(line, RSTART + 6, RLENGTH - 6)
    }
}

END {
    if (component != "" && story != "") {
        prefix = tolower(component)
        printf "%s\t%s\t%s\t%s\t%s\t%s\n", component, story, prefix, test_file, pixels, ratio
    }
}
AWKSCRIPT

awk -f /tmp/visual-parse.awk /tmp/visual-report-clean.txt > /tmp/visual-failures.txt

# Parse ALL test lines (both PASS and FAIL)
cat > /tmp/visual-parse-all.awk << 'AWKSCRIPT'
/visual\.test\.tsx.*>.*story matches screenshot/ {
    line = $0

    status = "passed"
    if (index(line, "FAIL") > 0) status = "failed"

    test_file = ""
    match(line, /src\/[a-zA-Z0-9_\/-]+\.visual\.test\.tsx/)
    if (RSTART > 0) {
        test_file = substr(line, RSTART, RLENGTH)
    }

    idx = index(line, "> ")
    if (idx > 0) {
        rest = substr(line, idx + 2)
        idx2 = index(rest, " > ")
        if (idx2 > 0) {
            comp_desc = substr(rest, 1, idx2 - 1)
            sub(/ Visual Regression$/, "", comp_desc)
            component = comp_desc

            story_rest = substr(rest, idx2 + 3)
            sub(/ story matches screenshot$/, "", story_rest)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", story_rest)
            story = story_rest

            prefix = tolower(component)
            printf "%s\t%s\t%s\t%s\t%s\n", component, story, prefix, test_file, status
        }
    }
}
AWKSCRIPT

awk -f /tmp/visual-parse-all.awk /tmp/visual-report-clean.txt > /tmp/visual-all-tests.txt

# --- Helper: base64-encode an image file ---
img_to_base64() {
    local img_path="$1"
    if [ -n "$img_path" ] && [ -f "$img_path" ]; then
        echo "data:image/png;base64,$(base64 -w 0 "$img_path")"
    fi
}

# --- Pre-compute base64 images for each failure ---
declare -A failure_baseline_b64
declare -A failure_actual_b64
declare -A failure_diff_b64
declare -A failure_pixels
declare -A failure_ratio

while IFS=$'\t' read -r comp story prefix test_file pixels ratio; do
    [ -z "$comp" ] && continue

    # test_file is now a full path like src/stories/components/blocks/__tests__/VisualRegression/diverging-bar-chart.visual.test.tsx
    test_file_basename=$(basename "$test_file")
    test_file_dir=$(dirname "$test_file")

    # Derive screenshot prefix from test filename (e.g. "diverging-bar-chart.visual.test.tsx" -> "diverging-bar-chart")
    screenshot_prefix="${test_file_basename%.visual.test.tsx}"
    screenshot_name="${screenshot_prefix}-${story}"
    story_key="${comp}::${story}"

    # __screenshots__ is a sibling directory of the test file
    screenshot_dir="${test_file_dir}/__screenshots__"
    test_screenshot_dir="${screenshot_dir}/${test_file_basename}"

    baseline=""
    actual=""
    diff_img=""

    # Baseline lives in __screenshots__/<test_file>/<name>-chromium-linux.png
    # The exact filename pattern only matches real baselines (vitest failure dumps use a different naming convention)
    if [ -d "$test_screenshot_dir" ]; then
        baseline_file=$(find "$test_screenshot_dir" -maxdepth 1 -name "${screenshot_name}-chromium-linux.png" 2>/dev/null | head -1)
        if [ -n "$baseline_file" ]; then baseline="$baseline_file"; fi
    fi

    # Actual and diff images are stored by vitest in .vitest-attachments/
    # Naming pattern: {name}-actual-chromium-linux.png, {name}-diff-chromium-linux.png
    # When no baseline exists, vitest produces a -reference- image instead of -actual-/-diff-
    # The directory mirrors the test file path
    attachments_dir=".vitest-attachments/${test_file}"
    if [ -d "$attachments_dir" ]; then
        actual_file=$(find "$attachments_dir" -maxdepth 1 -name "${screenshot_name}-actual-chromium-linux.png" 2>/dev/null | head -1)
        if [ -n "$actual_file" ]; then actual="$actual_file"; fi

        # Fallback: when no baseline exists, vitest saves the current screenshot as -reference-
        if [ -z "$actual" ]; then
            ref_file=$(find "$attachments_dir" -maxdepth 1 -name "${screenshot_name}-reference-chromium-linux.png" 2>/dev/null | head -1)
            if [ -n "$ref_file" ]; then actual="$ref_file"; fi
        fi

        diff_file=$(find "$attachments_dir" -maxdepth 1 -name "${screenshot_name}-diff-chromium-linux.png" 2>/dev/null | head -1)
        if [ -n "$diff_file" ]; then diff_img="$diff_file"; fi
    fi

    failure_baseline_b64[$story_key]=$(img_to_base64 "$baseline")
    failure_actual_b64[$story_key]=$(img_to_base64 "$actual")
    failure_diff_b64[$story_key]=$(img_to_base64 "$diff_img")
    failure_pixels[$story_key]="$pixels"
    failure_ratio[$story_key]="$ratio"

done < /tmp/visual-failures.txt

# --- Generate single self-contained index.html ---
cat > coverage/visual/index.html << 'HTMLHEAD'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visual Regression Report</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
            padding: 40px 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            margin-bottom: 30px;
            font-size: 28px;
        }
        .subtitle {
            text-align: center;
            color: #888;
            margin-bottom: 40px;
        }
        .score-section {
            display: flex;
            justify-content: center;
            gap: 40px;
            margin-bottom: 50px;
            flex-wrap: wrap;
        }
        .score-card {
            background: #16213e;
            border-radius: 16px;
            padding: 30px 40px;
            text-align: center;
            min-width: 200px;
        }
        .score-circle {
            width: 120px;
            height: 120px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 15px;
            font-size: 36px;
            font-weight: bold;
        }
        .score-label {
            font-size: 14px;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .score-title {
            font-size: 12px;
            color: #aaa;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .stats-row {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 50px;
            flex-wrap: wrap;
        }
        .stat-box {
            background: #16213e;
            border-radius: 12px;
            padding: 20px 30px;
            text-align: center;
            min-width: 120px;
        }
        .stat-number {
            font-size: 32px;
            font-weight: bold;
        }
        .stat-number.passed { color: #0cce6b; }
        .stat-number.failed { color: #ff4e42; }
        .stat-number.total { color: #ffa400; }
        .stat-number.stories { color: #4da6ff; }
        .stat-label {
            font-size: 12px;
            color: #888;
            text-transform: uppercase;
            margin-top: 5px;
        }
        h2 {
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #333;
        }
        .component-card {
            background: #16213e;
            border-radius: 12px;
            margin-bottom: 15px;
            overflow: hidden;
        }
        .component-card.has-failures {
            border-left: 4px solid #ff4e42;
        }
        .component-card.all-passed {
            border-left: 4px solid #0cce6b;
        }
        .component-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 16px 20px;
            cursor: pointer;
            transition: background 0.2s;
        }
        .component-header:hover {
            background: #1a2744;
        }
        .expand-icon {
            display: inline-block;
            font-size: 12px;
            transition: transform 0.2s;
            color: #888;
        }
        .expand-icon.expanded {
            transform: rotate(90deg);
        }
        .component-name {
            font-weight: 600;
            font-size: 16px;
        }
        .component-badges {
            display: flex;
            gap: 8px;
            align-items: center;
        }
        .badge {
            padding: 3px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }
        .badge-passed {
            background: #0cce6b22;
            color: #0cce6b;
        }
        .badge-failed {
            background: #ff4e4233;
            color: #ff6b6b;
        }
        .stories-container {
            border-top: 1px solid #253350;
        }
        .story-item {
            border-top: 1px solid #253350;
        }
        .story-item:first-child {
            border-top: none;
        }
        .story-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 20px 12px 40px;
            font-size: 14px;
        }
        .story-row.clickable {
            cursor: pointer;
            transition: background 0.2s;
        }
        .story-row.clickable:hover {
            background: #1a2744;
        }
        .story-info {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            flex-shrink: 0;
        }
        .status-dot.pass { background: #0cce6b; }
        .status-dot.fail { background: #ff4e42; }
        .story-name {
            color: #ccc;
        }
        /* Detail panel (inline, below the story row) */
        .detail-panel {
            background: #0f1a2e;
            border-top: 1px solid #253350;
            padding: 24px 20px;
        }
        .detail-breadcrumb {
            color: #888;
            font-size: 13px;
            margin-bottom: 20px;
        }
        .detail-breadcrumb span { color: #ff6b6b; }
        .images-grid {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .image-card {
            background: #16213e;
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid #253350;
        }
        .image-card-header {
            padding: 12px 16px;
            font-weight: 600;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 1px;
            border-bottom: 1px solid #253350;
        }
        .image-card-header.baseline { color: #0cce6b; }
        .image-card-header.actual { color: #ff6b6b; }
        .image-card-header.diff { color: #ffa400; }
        .image-card-body {
            padding: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 150px;
            background: #0a0f1e;
        }
        .image-card-body img {
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }
        .no-image {
            color: #555;
            font-size: 13px;
            text-align: center;
            padding: 30px 10px;
        }
        .timestamp {
            text-align: center;
            color: #666;
            font-size: 12px;
            margin-top: 40px;
        }
        .hidden {
            display: none;
        }
        .search-container {
            margin-bottom: 20px;
        }
        #searchInput {
            width: 100%;
            padding: 12px 16px;
            background: #16213e;
            border: 1px solid #253350;
            border-radius: 8px;
            color: #eee;
            font-size: 14px;
            outline: none;
            transition: border-color 0.2s;
        }
        #searchInput:focus {
            border-color: #4da6ff;
        }
        #searchInput::placeholder {
            color: #666;
        }
        .success-card {
            background: #16213e;
            border-radius: 12px;
            border-left: 4px solid #0cce6b;
            text-align: center;
            padding: 30px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Visual Regression Report</h1>

        <div class="score-section">
            <div class="score-card">
                <div class="score-title">Coverage</div>
HTMLHEAD

# Add dynamic scores
cat >> coverage/visual/index.html << HTMLSCORE
                <div class="score-circle" style="background: ${coverage_color}22; border: 4px solid ${coverage_color}; color: ${coverage_color};">
                    ${coverage_pct}%
                </div>
                <div class="score-label">${coverage_label}</div>
            </div>
        </div>

        <div class="stats-row">
            <div class="stat-box">
                <div class="stat-number passed">${passed}</div>
                <div class="stat-label">Passed</div>
            </div>
            <div class="stat-box">
                <div class="stat-number failed">${failed}</div>
                <div class="stat-label">Failed</div>
            </div>
            <div class="stat-box">
                <div class="stat-number total">${total}</div>
                <div class="stat-label">Total Tests</div>
            </div>
            <div class="stat-box">
                <div class="stat-number stories">${total_stories}</div>
                <div class="stat-label">Total Stories</div>
            </div>
        </div>

HTMLSCORE

# --- Build component/story hierarchy ---
declare -A comp_stories_list
declare -A comp_story_status
comp_order=()

while IFS=$'\t' read -r comp story prefix test_file status; do
    [ -z "$comp" ] && continue

    if [ -z "${comp_stories_list[$comp]+x}" ]; then
        comp_order+=("$comp")
        comp_stories_list[$comp]=""
    fi

    story_key="${comp}::${story}"

    if [[ "${comp_stories_list[$comp]}" != *"${story_key}"* ]]; then
        comp_stories_list[$comp]="${comp_stories_list[$comp]}${story_key}"$'\n'
    fi

    comp_story_status[$story_key]="$status"
done < /tmp/visual-all-tests.txt

# Count failures
has_any_failures=false
for comp in "${comp_order[@]}"; do
    while IFS= read -r sk; do
        [ -z "$sk" ] && continue
        if [ "${comp_story_status[$sk]}" = "failed" ]; then
            has_any_failures=true
            break 2
        fi
    done <<< "${comp_stories_list[$comp]}"
done

# Generate section header with search bar
if [ "$has_any_failures" = true ]; then
    cat >> coverage/visual/index.html << 'FAILHDR'
        <h2>⚠️ Visual Differences Found</h2>
        <div class="search-container">
            <input type="text" id="searchInput" placeholder="Search components or stories..." />
        </div>
        <div class="violations-list">
FAILHDR
else
    cat >> coverage/visual/index.html << 'PASSHDR'
        <h2>Results</h2>
        <div class="search-container">
            <input type="text" id="searchInput" placeholder="Search components or stories..." />
        </div>
PASSHDR
fi

comp_index=0
for comp in "${comp_order[@]}"; do
    comp_passed_count=0
    comp_failed_count=0
    story_keys=()

    while IFS= read -r sk; do
        [ -z "$sk" ] && continue
        story_keys+=("$sk")
        if [ "${comp_story_status[$sk]}" = "failed" ]; then
            ((comp_failed_count++))
        else
            ((comp_passed_count++))
        fi
    done <<< "${comp_stories_list[$comp]}"

    # Only show components that have failures
    if [ $comp_failed_count -eq 0 ]; then
        ((comp_index++))
        continue
    fi

    card_class="has-failures"

    # Build searchable text (component + all story names)
    comp_searchable=$(echo "$comp" | tr '[:upper:]' '[:lower:]')
    for sk in "${story_keys[@]}"; do
        sname="${sk#*::}"
        comp_searchable="${comp_searchable} $(echo "$sname" | tr '[:upper:]' '[:lower:]')"
    done

    cat >> coverage/visual/index.html << COMP_START
            <div class="component-card ${card_class}" data-component-id="comp-${comp_index}" data-searchable="${comp_searchable}">
                <div class="component-header" onclick="toggleComponent('comp-${comp_index}')">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span class="expand-icon" id="icon-comp-${comp_index}">&#9654;</span>
                        <span class="component-name">${comp}</span>
                    </div>
                    <div class="component-badges">
COMP_START

    cat >> coverage/visual/index.html << FBADGE
                        <span class="badge badge-failed">${comp_failed_count} failed</span>
FBADGE

    cat >> coverage/visual/index.html << COMP_MID
                    </div>
                </div>
                <div class="stories-container hidden" id="stories-comp-${comp_index}">
COMP_MID

    # Generate story items
    story_index=0
    for story_key in "${story_keys[@]}"; do
        story_name="${story_key#*::}"
        status="${comp_story_status[$story_key]}"
        detail_id="detail-comp${comp_index}-story${story_index}"

        if [ "$status" = "failed" ]; then
            dot_class="fail"

            # Get pre-computed base64 images and diff info
            b64_baseline="${failure_baseline_b64[$story_key]}"
            b64_actual="${failure_actual_b64[$story_key]}"
            b64_diff="${failure_diff_b64[$story_key]}"
            diff_pixels="${failure_pixels[$story_key]}"
            diff_ratio="${failure_ratio[$story_key]}"

            # Build failure reason text
            if [ -n "$diff_pixels" ] && [ -n "$diff_ratio" ]; then
                diff_pct=$(awk "BEGIN {printf \"%.0f\", $diff_ratio * 100}")
                fail_reason="Screenshot mismatch — ${diff_pixels} pixels (${diff_pct}%) differ"
            else
                fail_reason="Screenshot does not match the stored reference"
            fi

            story_searchable=$(echo "$story_name" | tr '[:upper:]' '[:lower:]')
            cat >> coverage/visual/index.html << STORY_FAIL_ROW
                    <div class="story-item" data-parent="comp-${comp_index}" data-searchable="${story_searchable}">
                        <div class="story-row clickable" onclick="toggleDetail('${detail_id}')">
                            <div class="story-info">
                                <span class="expand-icon" id="icon-${detail_id}">&#9654;</span>
                                <span class="status-dot ${dot_class}"></span>
                                <span class="story-name">${story_name}</span>
                            </div>
                            <span style="color: #4da6ff; font-size: 13px;">View Diff</span>
                        </div>
                        <div class="detail-panel hidden" id="${detail_id}">
                            <p class="detail-breadcrumb"><span>${comp}</span> / ${story_name}</p>
                            <p style="color: #ff6b6b; font-size: 13px; margin-bottom: 16px;">${fail_reason}</p>
                            <div class="images-grid">
                                <div class="image-card">
                                    <div class="image-card-header baseline">Baseline (Expected)</div>
                                    <div class="image-card-body">
STORY_FAIL_ROW

            if [ -n "$b64_baseline" ]; then
                cat >> coverage/visual/index.html << BIMG
                                        <img src="${b64_baseline}" alt="Baseline screenshot" />
BIMG
            else
                cat >> coverage/visual/index.html << 'NOBIMG'
                                        <div class="no-image">No baseline image found</div>
NOBIMG
            fi

            cat >> coverage/visual/index.html << 'MID_ACTUAL'
                                    </div>
                                </div>
                                <div class="image-card">
                                    <div class="image-card-header actual">Actual (Current)</div>
                                    <div class="image-card-body">
MID_ACTUAL

            if [ -n "$b64_actual" ]; then
                cat >> coverage/visual/index.html << AIMG
                                        <img src="${b64_actual}" alt="Actual screenshot" />
AIMG
            else
                cat >> coverage/visual/index.html << 'NOAIMG'
                                        <div class="no-image">No actual image found</div>
NOAIMG
            fi

            cat >> coverage/visual/index.html << 'MID_DIFF'
                                    </div>
                                </div>
                                <div class="image-card">
                                    <div class="image-card-header diff">Diff</div>
                                    <div class="image-card-body">
MID_DIFF

            if [ -n "$b64_diff" ]; then
                cat >> coverage/visual/index.html << DIMG
                                        <img src="${b64_diff}" alt="Diff screenshot" />
DIMG
            else
                cat >> coverage/visual/index.html << 'NODIMG'
                                        <div class="no-image">No diff image found</div>
NODIMG
            fi

            cat >> coverage/visual/index.html << 'STORY_FAIL_END'
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
STORY_FAIL_END

        else
            # Skip passed stories — only show failures in the report
            ((story_index++))
            continue
        fi

        ((story_index++))
    done

    cat >> coverage/visual/index.html << COMP_END
                </div>
            </div>
COMP_END

    ((comp_index++))
done

if [ "$has_any_failures" = true ]; then
    cat >> coverage/visual/index.html << 'FAILFOOT'
        </div>
FAILFOOT
fi

# If no tests found at all
if [ ${#comp_order[@]} -eq 0 ]; then
    if [ $total -eq 0 ]; then
        cat >> coverage/visual/index.html << 'NOTESTS'
        <div class="success-card">
            <span style="font-size: 24px;">&#9888;&#65039;</span>
            <p style="color: #ffa400; font-size: 18px; margin-top: 10px;">No visual regression tests found.</p>
            <p style="color: #888; font-size: 14px; margin-top: 8px;">Add .visual.test.tsx files to start tracking visual regressions.</p>
        </div>
NOTESTS
    else
        cat >> coverage/visual/index.html << 'ALLPASS'
        <h2>Results</h2>
        <div class="success-card">
            <span style="font-size: 24px;">&#127881;</span>
            <p style="color: #0cce6b; font-size: 18px; margin-top: 10px;">All visual regression tests passed!</p>
            <p style="color: #888; font-size: 14px; margin-top: 8px;">No visual differences detected.</p>
        </div>
ALLPASS
    fi
fi

# Cleanup temp files
rm -f /tmp/visual-report-clean.txt /tmp/visual-failures.txt /tmp/visual-all-tests.txt
rm -f /tmp/visual-parse.awk /tmp/visual-parse-all.awk

# Close HTML with JavaScript
cat >> coverage/visual/index.html << 'HTMLFOOT'

        <script>
            function toggleComponent(id) {
                var container = document.getElementById('stories-' + id);
                var icon = document.getElementById('icon-' + id);
                if (container.classList.contains('hidden')) {
                    container.classList.remove('hidden');
                    icon.classList.add('expanded');
                } else {
                    container.classList.add('hidden');
                    icon.classList.remove('expanded');
                }
            }
            function toggleDetail(id) {
                var panel = document.getElementById(id);
                var icon = document.getElementById('icon-' + id);
                if (panel.classList.contains('hidden')) {
                    panel.classList.remove('hidden');
                    if (icon) icon.classList.add('expanded');
                } else {
                    panel.classList.add('hidden');
                    if (icon) icon.classList.remove('expanded');
                }
            }

            document.getElementById('searchInput').addEventListener('input', function(e) {
                var searchTerm = e.target.value.toLowerCase();
                var componentCards = document.querySelectorAll('.component-card[data-component-id]');

                if (searchTerm === '') {
                    componentCards.forEach(function(card) {
                        card.classList.remove('hidden');
                        var compId = card.getAttribute('data-component-id');
                        var storiesContainer = document.getElementById('stories-' + compId);
                        if (storiesContainer) storiesContainer.classList.add('hidden');
                        var icon = document.getElementById('icon-' + compId);
                        if (icon) icon.classList.remove('expanded');
                        var storyItems = card.querySelectorAll('.story-item');
                        storyItems.forEach(function(item) { item.classList.remove('hidden'); });
                    });
                } else {
                    componentCards.forEach(function(card) {
                        var compId = card.getAttribute('data-component-id');
                        var compSearchable = card.getAttribute('data-searchable') || '';
                        var compNameMatch = compSearchable.split(' ')[0].includes(searchTerm);
                        var storyItems = card.querySelectorAll('.story-item[data-parent="' + compId + '"]');
                        var hasMatch = compNameMatch;

                        storyItems.forEach(function(item) {
                            var storySearchable = item.getAttribute('data-searchable');
                            if (compNameMatch || (storySearchable && storySearchable.includes(searchTerm))) {
                                hasMatch = true;
                                item.classList.remove('hidden');
                            } else {
                                item.classList.add('hidden');
                            }
                        });

                        if (hasMatch) {
                            card.classList.remove('hidden');
                            var storiesContainer = document.getElementById('stories-' + compId);
                            if (storiesContainer) storiesContainer.classList.remove('hidden');
                            var icon = document.getElementById('icon-' + compId);
                            if (icon) icon.classList.add('expanded');
                        } else {
                            card.classList.add('hidden');
                        }
                    });
                }
            });
        </script>
HTMLFOOT

cat >> coverage/visual/index.html << HTMLTIMESTAMP
        <p class="timestamp">Generated on $(date '+%Y-%m-%d %H:%M:%S')</p>
    </div>
</body>
</html>
HTMLTIMESTAMP

echo ""
echo "========================================"
echo "Visual Regression HTML Report generated at:"
echo "coverage/visual/index.html"
echo "========================================"
echo ""
echo "Open with: open coverage/visual/index.html"
