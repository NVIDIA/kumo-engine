# Notebook widget design

Use this checklist when planning, implementing, or reviewing Engine visual features, including graph visualization and explainability.
The goal is to help users find an answer and act without reading the entire widget.
The [shared widget recipe](../../packages/kumo-engine/README.md#developing-another-notebook-widget) owns transport, theme, isolation, lifecycle, and export mechanics; reuse it rather than creating another shell.
These design checks complement functional validation and require inspection of rendered output.

## Before implementation

Record this short brief in the task's existing work record; do not create a second planning system.

- **Question:** What specific question brings the user to this view?
- **Scan tasks:** Name three things the user must find, compare, or do, with the expected answer or outcome for a representative fixture.
- **Hierarchy:** Identify the first thing to notice, the primary action if any, and the supporting details.
- **Layout:** Identify shared alignment edges, repeated row positions, grouping, and which information must remain adjacent at narrow widths.
- **Meaning:** Define each visual encoding and distinguish default, selected, unavailable, and exceptional states.
- **Reuse and evidence:** Name existing widget helpers and theme tokens to reuse, fixtures and states to inspect, and the rendered captures and functional checks needed.

Inspect the existing graph viewer and theme before choosing styles.
Use the established widget design system; do not import another UI kit or invent a parallel palette for an individual feature.
Resolve data semantics before assigning labels, ranking values, or selecting visual encodings.

## Implementation and review checks

| ID | Required check | Failure to correct |
| --- | --- | --- |
| W1 | Align related headings, labels, controls, and rows to shared boundaries or baselines. | Arbitrary offsets or controls floating independently of their content. |
| W2 | Left-align reading text in left-to-right layouts; right-align comparable numeric values with consistent units and precision. | Centered multiline descriptions or numbers that cannot be compared down a column. |
| W3 | Give compact elements clear horizontal and vertical relationships to neighbors. | Adding filler content, stretched buttons, or decoration solely to occupy empty space. |
| W4 | Keep labels, values, and associated actions near each other at every supported width. | Full-width rows that force the eye across a large empty gap to match a value to its label. |
| W5 | Group long lists by a distinction useful to the user's scan task; keep recurring metadata in predictable positions. | A long undifferentiated list or attributes changing position between rows. |
| W6 | Distinguish titles, metadata, states, and actions before adding vertical space. | Uniformly taller rows that increase scrolling without clarifying relationships. |
| W7 | Use familiar visuals when they reduce reading, with accessible names or text equivalents. | Decorative icons or diagrams that require more interpretation than the text they replace. |
| W8 | Give ambiguous actions visible labels; keep essential information available without hover. | Users must wait for a tooltip to discover what a necessary control does. |
| W9 | Make interactive affordances truthful and preserve useful actions. | Static chips styled as editable controls, or frequent actions hidden only to simplify layout. |
| W10 | Keep ordinary/default values quiet but legible; distinguish selection and exceptions. | Every default value rendered as a saturated badge, leaving no clear emphasis. |
| W11 | Reduce competing emphasis before increasing the focal element's weight or saturation. | Multiple equally dominant controls, headings, or status colors with no clear starting point. |
| W12 | Use consistent color meanings and an additional signal such as text, shape, or position. | Color alone conveys selection, contribution direction, or an error. |
| W13 | Use alignment, sections, and spacing before adding containers; retain cards where the boundary conveys a distinct object. | Nested borders and rounded panels that communicate no additional grouping. |
| W14 | Preserve context through interaction and fallback states. | Selection or hover unexpectedly moves the reading target, or an empty/error/static view conceals what data is unavailable. |

These are review heuristics, not a requirement to attach every element to two literal borders.
Document a specific usability reason for an exception in the review record, with a capture showing its effect.
Do not waive accessibility, data correctness, or functional failures as aesthetic preferences.

## Applying the checks to explainability

This section is a planning contract for future work, not a claim that an explainability widget already implements these views.
Confirm which data the explanation API actually provides before choosing the presentation.

- Identify the explanation scope visibly: a prediction, entity, cohort, or global summary, including the relevant target and context.
  Do not let a local explanation appear to describe the whole model.
- For comparable contribution values, consider aligned bars with adjacent names and numeric values on a common scale.
  Use a zero baseline and explicit signs only when the API returns signed contributions; do not invent direction for unsigned importance.
- State what the values mean and their units or normalization.
  Do not describe attribution as causation, confidence, or probability unless the underlying result supports that interpretation.
- Keep feature or table grouping and the ranking basis visible.
  Preserve a stable order while inspecting a row; make reordering an explicit operation.
- Keep selection styling distinct from contribution magnitude and direction.
  A selected row must not appear to have a larger contribution merely because it gains emphasis.
- Show concise identifying labels and the key result first, with secondary detail available on selection or expansion.
  Keep exact values and long feature names accessible without relying solely on hover.
- Distinguish zero contribution, missing values, unavailable explanations, and failed computation.
  Explain an empty state with the relevant recovery action, when one exists.

Before coding, choose three concrete scan tasks such as identifying the explained prediction, finding its strongest supported contributor, and inspecting that contributor's details.
Replace these examples when the supported API or user workflow calls for different tasks.

## Evidence required before handoff

Use existing fixtures and validation tools where possible.
Extend them only for behaviors or states the feature introduces; do not add tests that merely repeat this checklist's prose.

1. Render representative content through each affected notebook transport, recording fixture, viewport, theme, state, and source revision.
   Inspect actual screenshots in light and dark themes at a normal notebook width and a narrow width, such as 320 CSS pixels.
   Check long labels, many rows, missing values, defaults, selection, and empty/error states where applicable.
2. Exercise the three scan tasks from the brief with expected and observed outcomes.
   Record whether the needed label, value, and action can be located without reading unrelated rows or opening an undisclosed menu.
   Agent inspection is a heuristic review, not evidence of measured human completion times.
3. Exercise keyboard navigation and visible focus, selected and disabled controls, and pointer interactions.
   Check clipping, overlap, unintended horizontal scrolling, contrast, and access to information otherwise shown on hover.
4. Reuse the recipe's isolation and lifecycle checks for multiple outputs and redisplay; verify affected static/export paths separately.
   Record unavailable host validation explicitly; local browser replay does not establish native notebook support.
5. Record W1 through W14 as pass, fail, or not applicable, with capture references and a reason for each exception or non-applicable item.
   Fix failures within scope before marking visual review complete; report blocked or untested cases explicitly.

A compact review record can use the following fields in the existing task log:

```text
Source revision and fixture:
Viewport / theme / transport / state:
Scan task -> expected -> observed:
W1-W14 dispositions and capture references:
Interaction / fallback checks and results:
Exceptions, blockers, and untested hosts:
```
