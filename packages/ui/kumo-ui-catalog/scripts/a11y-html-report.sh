#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# Generate HTML report from a11y test output
# If /tmp/a11y-report.txt already exists (pre-populated by CI), skip re-running tests.

if [ ! -f /tmp/a11y-report.txt ]; then
    echo "Running accessibility tests..."
    npm run test:storybook -- --run 2>&1 | tee /tmp/a11y-report.txt
else
    echo "Using existing test output from /tmp/a11y-report.txt"
fi

# Extract pass/fail counts
passed=$(grep -E "Tests.*passed" /tmp/a11y-report.txt | grep -oE "[0-9]+ passed" | head -1 | grep -oE "[0-9]+")
failed=$(grep -E "Tests.*failed" /tmp/a11y-report.txt | grep -oE "[0-9]+ failed" | head -1 | grep -oE "[0-9]+")

if [ -z "$passed" ]; then passed=0; fi
if [ -z "$failed" ]; then failed=0; fi

total=$((passed + failed))

if [ $total -gt 0 ]; then
    score=$(awk "BEGIN {printf \"%.1f\", ($passed / $total) * 100}")
    score_int=$(awk "BEGIN {printf \"%.0f\", ($passed / $total) * 100}")
else
    score="0"
    score_int=0
fi

# Determine score color
if [ $score_int -ge 90 ]; then
    score_color="#0cce6b"
    score_label="Excellent"
elif [ $score_int -ge 70 ]; then
    score_color="#ffa400"
    score_label="Needs Improvement"
else
    score_color="#ff4e42"
    score_label="Poor"
fi

# Create report directory
mkdir -p coverage/a11y

# Generate HTML report
cat > coverage/a11y/index.html << 'HTMLHEAD'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Accessibility Report</title>
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
            max-width: 900px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            margin-bottom: 10px;
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
        .score-title {
            font-size: 12px;
            color: #aaa;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .score-label {
            font-size: 14px;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .stats-row {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-bottom: 50px;
            flex-wrap: wrap;
        }
        .stat-box {
            background: #16213e;
            border-radius: 12px;
            padding: 20px 30px;
            text-align: center;
        }
        .stat-number {
            font-size: 32px;
            font-weight: bold;
        }
        .stat-number.passed { color: #0cce6b; }
        .stat-number.failed { color: #ff4e42; }
        .stat-number.total { color: #ffa400; }
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
        .violations-list {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        .violation-card {
            background: #16213e;
            border-radius: 12px;
            padding: 20px;
            border-left: 4px solid #ff4e42;
        }
        .violation-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .violation-name {
            font-weight: 600;
            font-size: 16px;
        }
        .violation-count {
            background: #ff4e42;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
        }
        .violation-description {
            color: #aaa;
            font-size: 14px;
            margin-bottom: 12px;
            line-height: 1.5;
        }
        .violation-link {
            color: #4da6ff;
            text-decoration: none;
            font-size: 13px;
        }
        .violation-link:hover {
            text-decoration: underline;
        }
        .timestamp {
            text-align: center;
            color: #666;
            font-size: 12px;
            margin-top: 40px;
        }
        .impact-critical { border-left-color: #ff4e42; }
        .impact-serious { border-left-color: #ffa400; }
        .impact-moderate { border-left-color: #ffd700; }
        .impact-minor { border-left-color: #4da6ff; }
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
        .stories-container {
            border-top: 1px solid #253350;
        }
        .story-item {
            border-top: 1px solid #253350;
        }
        .story-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 20px 12px 40px;
            cursor: pointer;
            transition: background 0.2s;
            font-size: 14px;
        }
        .story-header:hover {
            background: #1a2744;
        }
        .violation-badge {
            background: #ff4e4233;
            color: #ff6b6b;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 12px;
        }
        .violation-details {
            padding: 0 20px 0 56px;
        }
        .violation-detail-card {
            background: #0f1a2e;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
            border: 1px solid #253350;
        }
        .detail-rule {
            font-weight: 600;
            color: #ff6b6b;
            font-size: 15px;
            margin-bottom: 6px;
        }
        .detail-msg {
            color: #ccc;
            font-size: 14px;
            margin-bottom: 10px;
        }
        .detail-element {
            background: #0a0f1e;
            border-radius: 6px;
            padding: 10px 14px;
            margin-bottom: 10px;
            overflow-x: auto;
        }
        .detail-element code {
            color: #e8a87c;
            font-family: 'SF Mono', 'Fira Code', monospace;
            font-size: 13px;
        }
        .detail-fixes {
            color: #aaa;
            font-size: 13px;
            line-height: 1.8;
            margin-bottom: 10px;
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
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Accessibility Report</h1>
        <p class="subtitle">Storybook Components A11y Analysis</p>

        <div class="score-section">
            <div class="score-card">
                <div class="score-title">Accessibility Score</div>
HTMLHEAD

# Add dynamic score
cat >> coverage/a11y/index.html << HTMLSCORE
                <div class="score-circle" style="background: ${score_color}22; border: 4px solid ${score_color}; color: ${score_color};">
                    ${score}%
                </div>
                <div class="score-label">${score_label}</div>
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
        </div>

        <h2>⚠️ Violations Found</h2>
        <div class="search-container">
            <input type="text" id="searchInput" placeholder="Search components or stories..." />
        </div>
        <div class="violations-list">
HTMLSCORE

# Strip ANSI codes and create clean version for parsing
sed 's/\x1b\[[0-9;]*m//g' /tmp/a11y-report.txt > /tmp/a11y-report-clean.txt

# Parse FAIL blocks into structured data using awk
# Write awk script to temp file to avoid bash escaping issues
# Write awk script to temp file to avoid bash escaping issues
# Uses TAB as field separator for reliable bash parsing
cat > /tmp/a11y-parse.awk << 'AWKSCRIPT'
/^ FAIL .* > / {
    if (component != "") {
        gsub(/\n/, "\\n", fixes)
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n", component, story, rule, msg, element, fixes, help_link
    }
    line = $0
    match(line, /src\/stories\/[^ ]+/)
    filepath = substr(line, RSTART, RLENGTH)
    split(filepath, parts, "/")
    fname = parts[length(parts)]
    sub(/\.stories\.tsx$/, "", fname)
    sub(/\.stories\.ts$/, "", fname)
    component = fname

    idx = index(line, " > ")
    if (idx > 0) {
        story = substr(line, idx + 3)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", story)
    } else {
        story = "Unknown"
    }

    rule = ""
    msg = ""
    element = ""
    fixes = ""
    help_link = ""
    in_fixes = 0
    in_element = 0
    next
}

/^".*\(.*\)"$/ {
    line = $0
    gsub(/^"|"$/, "", line)
    match(line, /\([a-z][-a-z]*\)/)
    if (RSTART > 0) {
        rule = substr(line, RSTART+1, RLENGTH-2)
        msg = substr(line, 1, RSTART-1)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", msg)
    }
    in_fixes = 0
    next
}

/Expected the HTML found at/ {
    in_element = 1
    next
}

in_element == 1 && /^</ {
    element = $0
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", element)
    in_element = 0
    next
}

/^Fix any of the following:/ {
    in_fixes = 1
    next
}

in_fixes == 1 && /^[[:space:]]+/ {
    line = $0
    gsub(/^[[:space:]]+/, "", line)
    if (fixes != "") fixes = fixes "\n"
    fixes = fixes line
    next
}

in_fixes == 1 && !/^[[:space:]]+/ {
    in_fixes = 0
}

/dequeuniversity\.com/ {
    help_link = $0
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", help_link)
}

END {
    if (component != "") {
        gsub(/\n/, "\\n", fixes)
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n", component, story, rule, msg, element, fixes, help_link
    }
}
AWKSCRIPT

awk -f /tmp/a11y-parse.awk /tmp/a11y-report-clean.txt > /tmp/a11y-failures.txt

# Build HTML for hierarchical violations
# First, collect unique components and their stories
declare -A component_stories
declare -A story_data

component_order=()

while IFS=$'\t' read -r comp story_name rule msg element fixes help_link; do
    [ -z "$comp" ] && continue

    # Track component order (first seen)
    if [ -z "${component_stories[$comp]+x}" ]; then
        component_order+=("$comp")
        component_stories[$comp]=""
    fi

    # Build a unique key for this story
    story_key="${comp}::${story_name}"

    # Append this violation to the story (a story can have multiple violations)
    if [ -z "${story_data[$story_key]+x}" ]; then
        component_stories[$comp]="${component_stories[$comp]}${story_key}"$'\n'
        story_data[$story_key]=""
    fi

    # Store violation data (tab-separated)
    story_data[$story_key]="${story_data[$story_key]}${rule}	${msg}	${element}	${fixes}	${help_link}"$'\n'
done < /tmp/a11y-failures.txt

# Generate HTML for each component
comp_index=0
for comp in "${component_order[@]}"; do
    # Count stories for this component
    story_count=0
    story_keys=()
    while IFS= read -r sk; do
        [ -z "$sk" ] && continue
        story_keys+=("$sk")
        ((story_count++))
    done <<< "${component_stories[$comp]}"

    # Build searchable text (component + all story names)
    comp_searchable=$(echo "$comp" | tr '[:upper:]' '[:lower:]')
    for sk in "${story_keys[@]}"; do
        sname="${sk#*::}"
        comp_searchable="${comp_searchable} $(echo "$sname" | tr '[:upper:]' '[:lower:]')"
    done

    cat >> coverage/a11y/index.html << COMP_START
            <div class="violation-card" data-component-id="comp-${comp_index}" data-searchable="${comp_searchable}" style="border-left-color: #ff4e42; padding: 0; overflow: hidden;">
                <div class="component-header" onclick="toggleComponent('comp-${comp_index}')">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span class="expand-icon" id="icon-comp-${comp_index}">▶</span>
                        <span class="violation-name">${comp}</span>
                    </div>
                    <span class="violation-count">${story_count} $([ "$story_count" -eq 1 ] && echo "story" || echo "stories")</span>
                </div>
                <div class="stories-container hidden" id="stories-comp-${comp_index}">
COMP_START

    # Generate HTML for each story in this component
    story_index=0
    for story_key in "${story_keys[@]}"; do
        story_name="${story_key#*::}"
        story_id="comp-${comp_index}-story-${story_index}"

        # Count violations for this story
        violation_count=0
        while IFS= read -r vline; do
            [ -z "$vline" ] && continue
            ((violation_count++))
        done <<< "${story_data[$story_key]}"

        story_searchable=$(echo "$story_name" | tr '[:upper:]' '[:lower:]')
        cat >> coverage/a11y/index.html << STORY_START
                    <div class="story-item" data-parent="comp-${comp_index}" data-searchable="${story_searchable}">
                        <div class="story-header" onclick="toggleStory('${story_id}')">
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span class="expand-icon" id="icon-${story_id}">▶</span>
                                <span>${story_name}</span>
                            </div>
                            <span class="violation-badge">${violation_count} violation(s)</span>
                        </div>
                        <div class="violation-details hidden" id="details-${story_id}">
STORY_START

        # Generate HTML for each violation in this story
        while IFS=$'\t' read -r rule msg element fixes help_link; do
            [ -z "$rule" ] && continue

            # Escape HTML in element
            element_escaped=$(echo "$element" | sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g')

            # Escape HTML in fixes, then convert \n to <br> bullets
            fixes_html=$(echo "$fixes" | sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g; s/\\n/<br>• /g; s/^/• /')

            cat >> coverage/a11y/index.html << VIOLATION
                            <div class="violation-detail-card">
                                <div class="detail-rule">${rule}</div>
                                <p class="detail-msg">${msg}</p>
                                <div class="detail-element"><code>${element_escaped}</code></div>
                                <div class="detail-fixes">
                                    <strong>Fix any of the following:</strong><br>
                                    ${fixes_html}
                                </div>
                                <a href="${help_link}" target="_blank" class="violation-link">
                                    📚 Learn more about this rule →
                                </a>
                            </div>
VIOLATION
        done <<< "${story_data[$story_key]}"

        cat >> coverage/a11y/index.html << STORY_END
                        </div>
                    </div>
STORY_END

        ((story_index++))
    done

    cat >> coverage/a11y/index.html << COMP_END
                </div>
            </div>
COMP_END

    ((comp_index++))
done

# If no failures found, show a success message
if [ ${#component_order[@]} -eq 0 ]; then
    cat >> coverage/a11y/index.html << NOFAILS
            <div class="violation-card" style="border-left-color: #0cce6b; text-align: center; padding: 30px;">
                <span style="font-size: 24px;">🎉</span>
                <p style="color: #0cce6b; font-size: 18px; margin-top: 10px;">All tests passed! No accessibility violations found.</p>
            </div>
NOFAILS
fi

# Cleanup temp files
rm -f /tmp/a11y-report-clean.txt /tmp/a11y-failures.txt /tmp/a11y-parse.awk

# Close HTML with JavaScript
cat >> coverage/a11y/index.html << 'HTMLFOOT'
        </div>

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
            function toggleStory(id) {
                var details = document.getElementById('details-' + id);
                var icon = document.getElementById('icon-' + id);
                if (details.classList.contains('hidden')) {
                    details.classList.remove('hidden');
                    icon.classList.add('expanded');
                } else {
                    details.classList.add('hidden');
                    icon.classList.remove('expanded');
                }
            }

            document.getElementById('searchInput').addEventListener('input', function(e) {
                var searchTerm = e.target.value.toLowerCase();
                var componentCards = document.querySelectorAll('.violation-card[data-component-id]');

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

cat >> coverage/a11y/index.html << HTMLTIMESTAMP
        <p class="timestamp">Generated on $(date '+%Y-%m-%d %H:%M:%S')</p>
    </div>
</body>
</html>
HTMLTIMESTAMP

echo ""
echo "========================================"
echo "HTML Report generated at:"
echo "coverage/a11y/index.html"
echo "========================================"
echo ""
echo "Open with: open coverage/a11y/index.html"
