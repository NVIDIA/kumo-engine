// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useRef, useState } from "react";
import { useBlocker, useLocation } from "react-router";

import {
    Button,
    ChevronRight,
    cn,
    DocumentationLink,
    HoverCard,
    HoverCardContent,
    HoverCardTrigger,
    Info,
    Skeleton,
    StartJobPopover,
    Title,
    useWithDebounce
} from "@kumo-ai/kumo-ui-catalog";
import { useQueryClient } from "@tanstack/react-query";
import { JobType } from "gen/v0/types";
import { ErrorDetail, QueryDialect, RunMode } from "gen/v1/types";
import yaml from "js-yaml";
import { getCurrentUserDisplayName } from "stores/userStore";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import PQCodeEditor from "components/CodeEditor/PQCodeEditor";
import CancelAlert from "components/NewJob/CancelAlert";
import TrainingSetting from "components/NewJob/Training/Settings/TrainingSetting";
import Validation from "components/Validation/Validation";
import ValidationStatus from "components/Validation/ValidationStatus";

import { useGetGraphInfo } from "hooks/graph/graphs";
import { useCreateGraph, useGraphValidation } from "hooks/graph/newGraph";
import { useDefaultModelPlan } from "hooks/jobs/training/modelPlan";
import { useQueryValidation } from "hooks/jobs/training/query";
import { useCreateTrainingJob } from "hooks/jobs/training/trainingJob";
import { useProjectId } from "hooks/project/projects";
import useFluidCardHeight from "hooks/utils/useFluidCardHeight";
import useResizablePanel from "hooks/utils/useResizablePanel";
import { useValidationState } from "hooks/validation/validation";

import { CustomError } from "types/common";
import { ColumnGroup, GraphFlowInfo, JSGraph } from "types/graphs";
import { DropdownOption, InputState } from "types/inputs";
import { CreateTrainingJobArgs } from "types/jobs";

import { getProjectPath } from "utils/Constants";
import { tablesFromGraphResponse } from "utils/Converters/graphsConverter";
import { parseErrorMessage } from "utils/errorHandler";
import { getDefaultGraphGroups, getDefaultGraphInfo } from "utils/graph";
import {
    getKeys,
    getValues,
    isObjectNotEmpty
} from "utils/Helpers/objectHelpers";
import { isValid } from "utils/Helpers/typeGuards";
import { useCustomNavigation } from "utils/navigator";
import {
    invalidateGraphsList,
    invalidateJobsList,
    invalidateTrainingTags
} from "utils/queryInvalidation";

const getInitializedArgs = (): CreateTrainingJobArgs => ({
    name: "",
    graph_id: "",
    query_yaml: "",
    query_dialect: QueryDialect.PQLANG_V2, // Before submitting this will be re-inferred
    model_planner_yaml: "",
    run_mode: RunMode.FAST,
    description: "",
    graph_name: null,
    graph: null,
    user: getCurrentUserDisplayName()
});

const NewTrainingJob = () => {
    const queryClient = useQueryClient();
    // Load parameters if we want to create a query using a specific graph
    const location = useLocation();
    const projectId = useProjectId();
    const projectPaths = getProjectPath(projectId);
    const cancelNavigationUrl = `${projectPaths.ModelJobsList}?page=1&job_type=${JobType.TRAINING_JOB}`;
    const navigate = useCustomNavigation();
    const preselectedGraph = location?.state?.preselectedGraph;
    const copyTrainingTags = location?.state?.copytrainingtags;
    const initialArgs = {
        ...getInitializedArgs(),
        ...(location?.state?.copytrainingargs ?? {}),
        ...(preselectedGraph
            ? {
                  graph_id: preselectedGraph.id,
                  graph_name: preselectedGraph.name
              }
            : {})
    };

    const [loading, setLoading] = useState<boolean>(false);

    const [banner, alert, handleError, clearErrors, dialogRef] =
        useValidationState(setLoading);

    const [hasErrors, setError] = useState<ErrorDetail[]>([]);
    const [nameErrr, setNameError] = useState("");

    const { validateGraph: validateGraphConfig } = useGraphValidation();
    const { createGraph } = useCreateGraph(false);

    const [graphFlowInfo, setGraphFlowInfo] = useState<GraphFlowInfo>(() =>
        getDefaultGraphInfo(location)
    );
    const [groups, setGroups] = useState<ColumnGroup[]>(() =>
        getDefaultGraphGroups(location?.state?.copytrainingargs?.graph)
    );

    const [callCreateGraph, setCallCreateGraph] = useState(false);
    const [saveGraph, setSaveGraph] = useState<boolean>(false);
    const [newGraphId, setNewGraphId] = useState<string | null>(null);
    // Tracks the original graph name when the user edits an existing template.
    const [modifiedFromGraphName, setModifiedFromGraphName] = useState<
        string | null
    >(null);
    const [validatingGraph, setValidatingGraph] = useState<boolean>(false);

    const [validationState, setValidationState] = useState<InputState>(
        InputState.None
    );
    const [createJobArgs, setJobArgs] = useState<CreateTrainingJobArgs>({
        ...initialArgs
    });
    const [copiedTaskType, setCopiedTaskType] = useState(
        location?.state?.taskType ?? ""
    );
    const preserveInitialGraphTaskType = useRef(Boolean(copiedTaskType));
    const isSDKJob = location?.state?.isSDKJob ?? false;

    const [
        showOverwrittenModelPlanWarning,
        setShowOverwrittenModelPlanWarning
    ] = useState<boolean>(false);
    const [openPopover, setOpenPopover] = useState(false);
    const [showCancelDialog, setShowCancelDialog] = useState(false);

    const isDirty =
        createJobArgs.query_yaml !== "" ||
        createJobArgs.model_planner_yaml !== "" ||
        createJobArgs.graph_id !== "" ||
        createJobArgs.name !== "" ||
        callCreateGraph;

    // kumo-ai/kumo#30141: the "Discard this model?" confirmation used to be
    // wired only to the Cancel button, so browser Back, in-app navigation,
    // and page refresh/close silently dropped the dirty form. Block router
    // navigations while dirty (reusing the same CancelAlert below as the
    // confirmation) and arm the browser's native confirmation for
    // refresh/tab-close. Navigations the wizard issues on purpose (a
    // confirmed discard, the redirect to the freshly created training job)
    // set this ref first so they are never re-prompted.
    const intentionalNavigationRef = useRef(false);

    const shouldBlockNavigation = useCallback(
        () => isDirty && !intentionalNavigationRef.current,
        [isDirty]
    );
    const navigationBlocker = useBlocker(shouldBlockNavigation);

    useEffect(() => {
        if (!isDirty) {
            return undefined;
        }
        const promptBeforeUnload = (event: BeforeUnloadEvent) => {
            event.preventDefault();
            // Chromium only shows the confirmation when returnValue is set.
            event.returnValue = "";
        };
        window.addEventListener("beforeunload", promptBeforeUnload);
        return () => {
            window.removeEventListener("beforeunload", promptBeforeUnload);
        };
    }, [isDirty]);

    const onCancelClick = useCallback(() => {
        if (isDirty) {
            setShowCancelDialog(true);
        } else {
            navigate(cancelNavigationUrl, { replace: true });
        }
    }, [cancelNavigationUrl, isDirty, navigate]);

    const { validateQuery } = useQueryValidation();
    const validationVersion = useRef(0);
    const currentGraphId = useRef(createJobArgs.graph_id);

    // State updates
    const setField = useCallback(
        (field: string, value: string | boolean | JSGraph | null) => {
            const invalidatesTaskType =
                ["graph", "graph_id", "query_yaml", "run_mode"].includes(
                    field
                ) &&
                !(field === "graph_id" && value === currentGraphId.current) &&
                !(field === "graph" && preserveInitialGraphTaskType.current);
            if (field === "graph") preserveInitialGraphTaskType.current = false;
            if (field === "graph_id") currentGraphId.current = value as string;
            if (invalidatesTaskType) {
                setCopiedTaskType("");
            }
            if (invalidatesTaskType || field === "model_planner_yaml") {
                validationVersion.current += 1;
                setError([]);
            }
            setJobArgs(
                (args: CreateTrainingJobArgs) =>
                    ({
                        ...args,
                        [field]:
                            field === "graph" && value
                                ? { ...(value as JSGraph) }
                                : value
                    }) as CreateTrainingJobArgs
            );
        },
        [setJobArgs, setCopiedTaskType]
    );

    const graphTables = createJobArgs?.graph
        ? tablesFromGraphResponse(createJobArgs?.graph)
        : [];
    const hasUsableGraph =
        getValues(createJobArgs?.graph?.tables ?? {}).length > 0;

    const triggerError = useCallback((error: CustomError) => {
        // We want to handle errors differently for formatting
        const errorText = `Unexpected validation error: ${parseErrorMessage(
            error as CustomError
        )}. Please file a bug.`;

        setValidationState(InputState.InValid);
        setError([
            {
                title: "Error in Predictive Query / Model Planner",
                description: errorText,
                type: 0,
                cta: null
            }
        ]);
    }, []);

    /**
     * Get default model planner
     */

    const {
        isLoading: modelPlanLoading,
        modelPlanYaml,
        taskType: inferredTaskType,
        fetchModelPlan
    } = useDefaultModelPlan(
        createJobArgs.run_mode,
        createJobArgs.graph,
        createJobArgs.query_yaml,
        triggerError
    );
    const taskType = inferredTaskType ?? copiedTaskType;

    useEffect(() => {
        if (modelPlanYaml) {
            setField("model_planner_yaml", modelPlanYaml);
        }
    }, [modelPlanYaml, setField]);

    const getAndUpdateModelPlan = useCallback(() => {
        if (
            createJobArgs?.graph &&
            createJobArgs?.query_yaml &&
            createJobArgs?.run_mode
        ) {
            fetchModelPlan();
        }
    }, [
        createJobArgs?.graph,
        createJobArgs?.query_yaml,
        createJobArgs?.run_mode,
        fetchModelPlan
    ]);

    const checkModelPlanner = useCallback(() => {
        let error: ErrorDetail[] = [];
        try {
            const modelPlanYaml = yaml.load(createJobArgs?.model_planner_yaml);
            if (modelPlanYaml) {
                const training_table_gen_plan =
                    (
                        modelPlanYaml as unknown as {
                            training_table_generation: unknown;
                        }
                    )?.training_table_generation ?? "";
                if (training_table_gen_plan) return error;
            }
        } catch (err) {
            // We want to handle errors differently for formatting
            const errorText = `Unexpected validation error: ${parseErrorMessage(
                err as CustomError
            )}. Please file a bug.`;
            error = [
                {
                    title: "Error in Predictive Query / Model Planner",
                    description: errorText,
                    type: 0,
                    cta: null
                }
            ];
        }
        return error;
    }, [createJobArgs?.model_planner_yaml]);

    // To check newly created graph already exists
    const { selectedGraphInfo } = useGetGraphInfo(
        newGraphId ?? null,
        handleError,
        setLoading,
        newGraphId !== null
    );

    const onCreateGraph = useCallback(
        async (
            withName = false,
            isCurrentRequest: () => boolean = () => true
        ) => {
            if (!isCurrentRequest()) return;
            setNewGraphId(null);

            const gfc: GraphFlowInfo = { ...graphFlowInfo };
            //Update the name.
            if (withName) {
                gfc.name = createJobArgs.graph_name?.trim() as string;
            } else {
                gfc.name = "";
            }
            let tmpError = "";
            let flow = "Graph Validation";

            try {
                if (getKeys(gfc?.tables).length > 0) {
                    const { errors: validationErrors } =
                        await validateGraphConfig(gfc, groups);
                    if (!isCurrentRequest()) return;
                    if (validationErrors?.length < 1) {
                        flow = "Graph Creation";
                        const { graphId, error } = await createGraph(
                            gfc,
                            groups
                        );
                        if (!isCurrentRequest()) return;
                        if (graphId) {
                            if (!withName) {
                                setField("graph_id", graphId);
                                setCallCreateGraph(false);
                                setGraphFlowInfo(gfc);
                                setField("model_planner_yaml", "");
                                setNewGraphId(graphId);
                            }
                        } else if (error) {
                            tmpError = error;
                        }
                    } else {
                        tmpError = validationErrors?.join(", ");
                    }
                }
            } catch (error) {
                if (!isCurrentRequest()) return;
                tmpError = parseErrorMessage(error as CustomError);
            }
            if (tmpError) {
                const title = `${flow} failed!`;
                setError([
                    {
                        title: title,
                        description: tmpError,
                        type: 0,
                        cta: null
                    }
                ]);
                setValidationState(InputState.InValid);
            }
            setValidatingGraph(false);
        },
        [
            graphFlowInfo,
            createJobArgs.graph_name,
            validateGraphConfig,
            groups,
            createGraph,
            setField
        ]
    );

    // Each validation-version increment also schedules a state update. That
    // render must capture the new version before its debounced validation runs.
    const scheduledValidationVersion = validationVersion.current;
    const validateQueryAndGetModelPlan = useCallback(async () => {
        const requestVersion = scheduledValidationVersion;
        const isCurrentRequest = () =>
            requestVersion === validationVersion.current;
        // Start query validation
        try {
            if (callCreateGraph) {
                if (!isCurrentRequest()) return;
                await onCreateGraph(false, isCurrentRequest);
                return;
            }
            const { error, warnings } = await validateQuery(createJobArgs);
            if (!isCurrentRequest()) return;
            const allIssues = [...(error ?? []), ...(warnings ?? [])];
            if (error?.length > 0) {
                setError(allIssues);
                setValidationState(InputState.InValid);
            } else {
                if (warnings?.length > 0) {
                    setError(allIssues);
                    setValidationState(InputState.Warning);
                } else {
                    setValidationState(InputState.Valid);
                    setError([]);
                }
                if (createJobArgs?.model_planner_yaml === "") {
                    getAndUpdateModelPlan();
                } else {
                    const err = checkModelPlanner();
                    if (err.length > 0) {
                        setError(err);
                        setValidationState(InputState.InValid);
                    }
                }
            }
        } catch (error) {
            if (!isCurrentRequest()) return;
            setValidationState(InputState.InValid);
            triggerError(error as CustomError);
        }
    }, [
        callCreateGraph,
        validateQuery,
        onCreateGraph,
        createJobArgs,
        getAndUpdateModelPlan,
        checkModelPlanner,
        scheduledValidationVersion,
        triggerError
    ]);

    const triggerAlert = useCallback(
        (title: string, description: string) => {
            setLoading(false);
            const error: CustomError = {
                response: {
                    data: description
                },
                name: "",
                message: description
            };
            handleError(error, title);
        },
        [handleError]
    );

    const getAlreadyExistsError = useCallback(
        () =>
            `Graph already exists with name ${selectedGraphInfo?.name}, Please uncheck 'Save graph to reuse later' option and try again!`,
        [selectedGraphInfo?.name]
    );

    const { createTrainingJob } = useCreateTrainingJob(triggerAlert);

    const startTraining = useCallback(
        async (tags: Record<string, string>) => {
            if (saveGraph) {
                if (isValid(selectedGraphInfo?.name)) {
                    const err = getAlreadyExistsError();
                    const error: CustomError = {
                        response: { data: err },
                        name: "",
                        message: err
                    };
                    handleError(error, "Graph already exists!");
                    return;
                } else {
                    setLoading(true);
                    await onCreateGraph(true);
                }
            }

            setLoading(true);
            const { error } = await validateQuery(createJobArgs);

            if (error?.length > 0) {
                setLoading(false);
                setError(error);
                setValidationState(InputState.InValid);
            } else {
                setError([]);
                setValidationState(InputState.Valid);
                const createArgs = { ...createJobArgs };
                // If the user edited an existing graph template without saving it,
                // pass the original name so the snapshot is stored as "<name> (modified)".
                if (!saveGraph && modifiedFromGraphName) {
                    createArgs.base_graph_name = modifiedFromGraphName;
                }

                // Use the hook
                const { jobId } = await createTrainingJob(createArgs, tags);
                if (jobId) {
                    if (saveGraph) {
                        invalidateGraphsList(queryClient);
                    }
                    invalidateJobsList(queryClient);
                    invalidateTrainingTags(queryClient);

                    const trainingJobBasePath = projectPaths.TrainingJob;
                    const path = `${trainingJobBasePath}${jobId}?job_type=${JobType.TRAINING_JOB}`;
                    // The form args are still "dirty" here, but this is the
                    // job-created success redirect, not data loss: skip the
                    // navigation blocker.
                    intentionalNavigationRef.current = true;
                    navigate(path);
                } else {
                    setLoading(false);
                    setValidationState(InputState.InValid);
                }
            }
        },
        [
            saveGraph,
            modifiedFromGraphName,
            validateQuery,
            createJobArgs,
            selectedGraphInfo?.name,
            getAlreadyExistsError,
            handleError,
            onCreateGraph,
            createTrainingJob,
            queryClient,
            projectPaths.TrainingJob,
            navigate
        ]
    );

    const handleSetNewGraph = useCallback(
        (value: boolean) => {
            validationVersion.current += 1;
            setError([]);
            setCallCreateGraph(value);
            setModifiedFromGraphName(value ? createJobArgs.graph_name : null);
        },
        [createJobArgs.graph_name]
    );

    const restartQueryValidation = useWithDebounce(
        validateQueryAndGetModelPlan,
        /* seconds */ process.env.NODE_ENV === "test" ? 0 : 1
    );

    const changeRunMode = (newVal: DropdownOption) => {
        if (isObjectNotEmpty(newVal)) {
            if (createJobArgs?.graph_id && createJobArgs?.query_yaml) {
                const newRunModeEnum =
                    RunMode[newVal?.value as keyof typeof RunMode];
                if (newRunModeEnum !== createJobArgs.run_mode) {
                    setField("run_mode", newRunModeEnum);
                    if (createJobArgs?.model_planner_yaml?.length > 0) {
                        setShowOverwrittenModelPlanWarning(true);
                    }
                    setField("model_planner_yaml", "");
                }
            }
        }
    };

    useEffect(() => {
        // Check if run_mode is undefined or null
        if (!createJobArgs?.run_mode) {
            setJobArgs({
                ...createJobArgs,
                run_mode: RunMode.FAST
            });
        }
    }, [createJobArgs]);

    useEffect(() => {
        if (
            hasUsableGraph &&
            (createJobArgs?.query_yaml || callCreateGraph === true)
        ) {
            if (callCreateGraph === true) {
                setValidatingGraph(true);
            }
            restartQueryValidation();
            setValidationState(InputState.Validating);
        } else {
            setValidatingGraph(false);
            setValidationState(InputState.None);
        }
    }, [
        callCreateGraph,
        createJobArgs?.query_yaml,
        createJobArgs?.model_planner_yaml,
        createJobArgs?.graph,
        createJobArgs?.run_mode,
        hasUsableGraph,
        restartQueryValidation
    ]);

    // Resizable panels
    const { headerRef, cardHeight } = useFluidCardHeight<HTMLDivElement>();

    const { topPx, isDragging, handleProps, minTop, maxTop } =
        useResizablePanel(
            250, // default top height
            80, // min top height
            cardHeight ? cardHeight - 200 : 600 // max top height (leave 200px for bottom)
        );

    return (
        <>
            <CancelAlert
                show={showCancelDialog || navigationBlocker.state === "blocked"}
                title="Discard this model?"
                description="Your query, model settings, and model planner configurations will not be saved."
                handleClose={() => {
                    setShowCancelDialog(false);
                    if (navigationBlocker.state === "blocked") {
                        navigationBlocker.reset();
                    }
                }}
                handleConfirm={() => {
                    if (navigationBlocker.state === "blocked") {
                        // Continue the navigation the blocker intercepted
                        // (browser Back or an in-app route change).
                        navigationBlocker.proceed();
                        return;
                    }
                    intentionalNavigationRef.current = true;
                    navigate(cancelNavigationUrl, { replace: true });
                }}
            />
            <div
                ref={headerRef}
                data-screenshot-ready={
                    loading || modelPlanLoading
                        ? undefined
                        : "new-training-page"
                }>
                <div className="flex items-center justify-between mb-6">
                    <Title
                        data-testid="new-model-page-heading"
                        variant="page-heading"
                        className="text-3xl">
                        New Model
                    </Title>
                    <div className="flex items-center gap-2">
                        <div className="flex items-center gap-1">
                            {taskType ? (
                                <>
                                    <span
                                        className="text-base font-medium text-title-secondary"
                                        data-testid="infer-task-type">
                                        Task type:{" "}
                                        {taskType
                                            ?.toLowerCase()
                                            ?.replaceAll("_", " ")}
                                    </span>
                                    <HoverCard>
                                        <HoverCardTrigger className="cursor-pointer flex items-center">
                                            <Info className="size-3.5 text-title-secondary" />
                                        </HoverCardTrigger>
                                        <HoverCardContent
                                            arrow={false}
                                            className="w-52 text-white text-sm font-medium p-4 rounded space-y-2"
                                            // Allowed per migration plan: tooltip-dark token until catalog class exists
                                            // eslint-disable-next-line no-restricted-syntax
                                            style={{
                                                backgroundColor:
                                                    "var(--tooltip-dark, #3a3a41)"
                                            }}>
                                            <p>Inferred from your query.</p>
                                            <DocumentationLink
                                                id="task-type-docs"
                                                url="https://kumo.ai/docs/task-types"
                                                linkText="Learn more"
                                                className="text-blue-300! hover:text-blue-400!"
                                                showExternalIcon={true}
                                            />
                                        </HoverCardContent>
                                    </HoverCard>
                                </>
                            ) : createJobArgs?.query_yaml !== "" &&
                              (validationState === InputState.Validating ||
                                  modelPlanLoading) ? (
                                <Skeleton
                                    className="w-40 h-6"
                                    data-testid="task-type-pending"
                                />
                            ) : null}
                        </div>
                        <ValidationStatus
                            state={
                                createJobArgs?.query_yaml === ""
                                    ? InputState.None
                                    : validationState
                            }
                            errorInfo={
                                createJobArgs?.query_yaml !== ""
                                    ? hasErrors
                                    : []
                            }
                            emptyStateText={
                                !hasUsableGraph ? "Select Graph" : "Empty Query"
                            }
                        />
                        <Button
                            id="cancel-job"
                            variant="outline"
                            data-testid="cancel-job"
                            onClick={onCancelClick}>
                            Cancel
                        </Button>
                        <StartJobPopover
                            showPopover={openPopover}
                            setShowPopover={setOpenPopover}
                            startButtonTitle={"Start training "}
                            jobsSectionTitle="Training jobs"
                            headerLabel={
                                "The following jobs will be created when you start training."
                            }
                            tagHeader="Tags"
                            tagDescription={
                                "Add key-value tags to organize and filter this training job. Tags will be applied to both jobs."
                            }
                            initialTags={copyTrainingTags}
                            jobs={["Training Job", "Training Table Generation"]}
                            startTraining={startTraining}
                            disablePopover={
                                ![
                                    InputState.Valid,
                                    InputState.Warning
                                ].includes(validationState) ||
                                isValid(nameErrr) ||
                                (saveGraph && createJobArgs?.graph_name === "")
                            }>
                            <div role="alert">
                                <Button
                                    id="start-training"
                                    data-testid="start-training"
                                    onClick={() => {
                                        setOpenPopover(!openPopover);
                                    }}
                                    disabled={
                                        ![
                                            InputState.Valid,
                                            InputState.Warning
                                        ].includes(validationState) ||
                                        isValid(nameErrr) ||
                                        (saveGraph &&
                                            createJobArgs?.graph_name === "")
                                    }>
                                    {"Start training "}
                                    <ChevronRight className="fill-white" />
                                </Button>
                            </div>
                        </StartJobPopover>
                    </div>
                </div>
                <Validation
                    pageLoading={loading}
                    banner={banner}
                    alert={alert}
                    ctaMsgs={null}
                    dialogRef={dialogRef}
                    hideWarnings={() => {}}
                    clearErrors={clearErrors}
                />
            </div>
            <div
                className="overflow-hidden flex flex-col"
                style={cardHeight ? { height: cardHeight } : undefined}>
                {/* Top panel: Query Editor */}
                <div
                    id="pql-editor-wrapper"
                    className="border border-border-secondary rounded mb-2 overflow-hidden shrink-0"
                    // eslint-disable-next-line no-restricted-syntax
                    style={{ height: topPx }}>
                    <style>{`
                        #pql-editor-wrapper > div, #pql-editor-wrapper .cm-theme, #pql-editor-wrapper .cm-editor { height: 100% !important; }
                        #pql-editor-wrapper .cm-scroller { padding: 24px !important; }
                        #pql-editor-wrapper .cm-gutters { border-right: 1px solid var(--border-secondary) !important; margin-right: 16px; padding-right: 12px; background: transparent !important; }
                        #pql-editor-wrapper .cm-activeLineGutter { background: transparent !important; }
                        #pql-editor-wrapper .cm-activeLine { background: transparent !important; }
                    `}</style>
                    <PQCodeEditor
                        value={createJobArgs?.query_yaml}
                        language={"PQLANG_V2"}
                        editable={true}
                        isInvalid={false}
                        errorMsg={""}
                        graphTables={graphTables}
                        hideBorder
                        onChange={async (newval: string) => {
                            setField("query_yaml", newval);
                            if (createJobArgs?.model_planner_yaml?.length > 0) {
                                setShowOverwrittenModelPlanWarning(true);
                            }
                            setField("model_planner_yaml", "");
                        }}
                        height="100%"
                    />
                </div>

                {/* Drag handle */}
                <div
                    role="separator"
                    aria-orientation="horizontal"
                    aria-valuenow={topPx}
                    aria-valuemin={minTop}
                    aria-valuemax={maxTop}
                    aria-label="Resize query editor"
                    tabIndex={0}
                    {...handleProps}
                    className="h-2 shrink-0 cursor-row-resize flex items-center justify-center touch-none">
                    <div className="w-6 h-1 rounded-lg bg-title-secondary pointer-events-none" />
                </div>

                {/* Bottom panel: Configuration + Schema — fills remaining space */}
                <div
                    className={cn(
                        "flex-1 min-h-0 overflow-hidden flex flex-col",
                        isDragging && "select-none"
                    )}>
                    <TrainingSetting
                        createJobArgs={createJobArgs}
                        setField={setField}
                        changeRunMode={changeRunMode}
                        modelPlanLoading={modelPlanLoading}
                        taskType={taskType}
                        showOverwrittenWarning={showOverwrittenModelPlanWarning}
                        closeWarning={() =>
                            setShowOverwrittenModelPlanWarning(false)
                        }
                        setNameError={setNameError}
                        handleError={handleError}
                        groups={groups}
                        setGroups={setGroups}
                        graphFlowInfo={graphFlowInfo}
                        setGraphFlowInfo={setGraphFlowInfo}
                        setNewGraph={handleSetNewGraph}
                        setSaveGraph={setSaveGraph}
                        graphExist={
                            selectedGraphInfo?.name && newGraphId
                                ? selectedGraphInfo?.name
                                : ""
                        }
                        validatingGraph={validatingGraph}
                        hideGraphSelection={isSDKJob}
                        setParentLoading={setLoading}
                        projectId={projectId as string}
                        bottomPanelHeight={
                            cardHeight ? cardHeight - topPx - 8 : undefined
                        }
                    />
                </div>
            </div>
        </>
    );
};

export default WithErrorBoundary(NewTrainingJob);
