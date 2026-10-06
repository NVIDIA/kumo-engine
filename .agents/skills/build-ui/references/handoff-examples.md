# Build UI Handoff Examples

Use these illustrative handoff shapes when constructing the `/code-ui` handoff in Phase 4. Resolve file paths, hooks, and dependencies against the current tree before using them for a real task.

## Cross-Surface Workflow Handoff

```text
/code-ui
Task: Implement AC for #27326: Add cancel confirmation dialogs

Accepted AC:
1. New Model page cancel shows "Discard this model?" dialog when form is dirty -> modify src/pages/NewJob/Training/Training.tsx
2. New Prediction page cancel shows "Discard this prediction?" dialog when training job selected -> modify src/pages/NewJob/Prediction/Prediction.tsx
3. All confirmation dialogs use destructive button variant and updated copy -> modify src/components/@ui/Dialog/ConfirmationDialog.tsx + all callers
4. Tests for AC 1-3 -> modify tests for all changed components

Impacted Files:
- modify: src/components/@ui/Dialog/ConfirmationDialog.tsx, src/types/common.ts, src/pages/NewJob/Training/Training.tsx, src/pages/NewJob/Prediction/Prediction.tsx, + all dialog callers
- create: none

Exemplars:
- src/components/Admin/ConnectorSettings/CancelAlert.tsx (existing cancel dialog pattern)
- src/pages/Connectors/ConnectorsList/ConnectorsList.tsx (delete dialog usage)

API Hooks: none (UI-only change)

Design Notes:
- dialog padding: px-[30px] py-6 gap-4
- title: text-title-primary leading-7
- description: text-title-secondary text-base font-medium leading-5
- destructive button: variant="destructive" using --color-title-red token
- no DialogHeader/DialogFooter separators

UI States: dialog open, dialog closed, form dirty vs clean

Limitations: verify the required destructive button variant in the in-tree catalog and coordinate any component change with its callers

Flags And Constraints: none

Validation Scope:
- static checks: type-check, lint-check, format-check, build
- targeted tests: ConfirmationDialog.test.tsx, all dialog caller tests
- related tests: ConnectTable.test.tsx, JobOverview.test.tsx
```

## Trivial Bug Fix Handoff

```text
/code-ui
Task: Fix empty-state copy in connectors list

Accepted AC:
1. Empty connectors page shows the approved copy from Checkpoint 1 -> modify src/pages/Connectors/ConnectorsList/ConnectorsList.tsx
2. Loading and error states remain unchanged -> modify src/pages/Connectors/__tests__/ConnectorsList.test.tsx

Impacted Files:
- modify: src/pages/Connectors/ConnectorsList/ConnectorsList.tsx, src/pages/Connectors/__tests__/ConnectorsList.test.tsx
- create: none

Exemplars:
- src/pages/Models/ModelsList/ModelsList.tsx

API Hooks:
- useGetV1Connectors

Design Notes:
- none

UI States:
- loading
- empty
- error
- success

Limitations:
- no backend contract changes

Flags And Constraints:
- none

Validation Scope:
- static checks: type-check, lint-check
- targeted tests: src/pages/Connectors/__tests__/ConnectorsList.test.tsx
- related tests: skipped unless targeted test reveals broader impact
```

## New Page Handoff

```text
/code-ui
Task: Add project-level model comparison page

Accepted AC:
1. Add a model comparison page under the approved route -> create src/pages/ModelComparison/ModelComparison.tsx
2. Add route, navigation, and loader wiring -> modify route and loader files
3. Support loading, empty, error, and success states -> modify page and tests
4. Add regression coverage for the new route and rendering states -> create src/pages/ModelComparison/__tests__/ModelComparison.test.tsx

Impacted Files:
- modify: src/routes/Routes.tsx, src/routes/Loaders.ts, src/utils/Constants/routes.ts, src/tests/mockAxios.ts
- create: src/pages/ModelComparison/ModelComparison.tsx, src/pages/ModelComparison/index.ts, src/pages/ModelComparison/__tests__/ModelComparison.test.tsx

Exemplars:
- src/pages/Models/Models.tsx
- src/pages/Projects/ProjectOverview/ProjectOverview.tsx

API Hooks:
- useGetV1Models
- useGetV1ProjectsProjectIdComparison

Design Notes:
- title: text-title-primary
- body copy: text-title-secondary
- spacing derived from Phase 3 token mapping

UI States:
- loading
- empty
- error
- success

Limitations:
- browser verification may be partial if backend data is unavailable

Flags And Constraints:
- respect feature-flag gating discovered during recon

Validation Scope:
- static checks: type-check, lint-check, format-check, build
- targeted tests: src/pages/ModelComparison/__tests__/ModelComparison.test.tsx
- related tests: relevant project and model page tests
```
