// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    createContext,
    useContext,
    useEffect,
    useMemo,
    useRef,
    useState
} from "react";

import {
    Add,
    Button,
    cn,
    DateTimeInput,
    DocumentationLink,
    HoverCard,
    HoverCardContent,
    HoverCardTrigger,
    Info,
    Input,
    Select,
    Switch
} from "@kumo-ai/kumo-ui-catalog";
import { format, isValid, parse } from "date-fns";
import type { ModelPlan } from "gen/v1/types";
import type { ModelArchitecturePlan } from "gen/v1/types/modelArchitecturePlan";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";

import type { ColumnGroup } from "types/graphs";

// Context to provide the scrollable ancestor as collision boundary for Select popovers.
// Without this, Radix checks against the viewport — dropdowns near the bottom of a
// scroll container can overflow the container even though they fit in the viewport.
const CollisionBoundaryContext = createContext<Element | null>(null);

const useCollisionBoundary = () => useContext(CollisionBoundaryContext);
const MODEL_PLANNER_SELECT_POPOVER_CLASS =
    "w-[var(--radix-popover-trigger-width)] max-w-[90vw]";

/** Walk up the DOM to find the nearest scrollable ancestor. */
const findScrollableAncestor = (el: Element | null): Element | null => {
    let current = el?.parentElement;
    while (current) {
        const style = getComputedStyle(current);
        if (
            style.overflowY === "auto" ||
            style.overflowY === "scroll" ||
            style.overflow === "auto" ||
            style.overflow === "scroll"
        ) {
            return current;
        }
        current = current.parentElement;
    }
    return null;
};

import {
    buildEncoderOverrides,
    buildNaStrategy,
    buildSplitString,
    DirectedEdge,
    EncoderRule,
    getActiveHops,
    getDirectedEdges,
    getEncoderOptionsForStype,
    getEncoderRules,
    getHopObj,
    HOP_KEYS,
    HopKey,
    isRatioSplit,
    parseSplitString,
    setHopDirectedEdges,
    setHopValue,
    SPLIT_TYPES,
    SplitParams
} from "utils/modelPlanYaml";

// ─── Helpers ─────────────────────────────────────────────────────────────────

/** Number input that allows free typing and commits on blur. */
const NumericInput = ({
    value,
    onCommit,
    ...props
}: Omit<React.ComponentProps<typeof Input>, "onChange" | "value"> & {
    value: number | string;
    onCommit: (value: string) => void;
}) => {
    const [localValue, setLocalValue] = useState(String(value));

    useEffect(() => {
        setLocalValue(String(value));
    }, [value]);

    return (
        <Input
            {...props}
            value={localValue}
            onChange={(e) => setLocalValue(e.target.value)}
            onBlur={() => onCommit(localValue)}
        />
    );
};

// ─── Constants ────────────────────────────────────────────────────────────────

// Default k values for @k metrics (spec: 1, 10, 100)
const DEFAULT_K_VALUES = [1, 10, 100];

/** Expand metric base names into concrete @k options for each default k value. */
const expandAtKMetrics = (
    baseNames: string[]
): Array<{ value: string; label: string }> =>
    baseNames.flatMap((name) => {
        const label = name
            .split("_")
            .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
            .join(" ");
        return DEFAULT_K_VALUES.map((k) => ({
            value: `${name}@${k}`,
            label: `${label}@${k}`
        }));
    });

// Metrics options keyed by task type, per backend spec (model_plan.py / trainer.py)
const METRICS_BY_TASK: Record<
    string,
    Array<{ value: string; label: string }>
> = {
    binary_classification: [
        { value: "acc", label: "Accuracy" },
        { value: "auroc", label: "AUROC" },
        { value: "auprc", label: "AUPRC" },
        { value: "ap", label: "AP" },
        { value: "f1", label: "F1" },
        { value: "ndcg", label: "NDCG" },
        { value: "precision", label: "Precision" },
        { value: "recall", label: "Recall" }
    ],
    multiclass_classification: [
        { value: "acc", label: "Accuracy" },
        { value: "f1", label: "F1" },
        { value: "precision", label: "Precision" },
        { value: "recall", label: "Recall" }
    ],
    multilabel_classification: [
        { value: "acc", label: "Accuracy" },
        { value: "f1", label: "F1" },
        { value: "precision", label: "Precision" },
        { value: "recall", label: "Recall" },
        { value: "auroc_macro", label: "AUROC Macro" },
        { value: "auroc_micro", label: "AUROC Micro" },
        { value: "auroc_per_label", label: "AUROC Per Label" },
        { value: "auprc_macro", label: "AUPRC Macro" },
        { value: "auprc_micro", label: "AUPRC Micro" },
        { value: "auprc_per_label", label: "AUPRC Per Label" },
        { value: "ap_macro", label: "AP Macro" },
        { value: "ap_micro", label: "AP Micro" },
        { value: "ap_per_label", label: "AP Per Label" }
    ],
    multilabel_ranking: expandAtKMetrics([
        "map",
        "precision",
        "recall",
        "f1",
        "ndcg"
    ]),
    link_prediction: [
        // Non-weighted metrics (default)
        ...expandAtKMetrics(["map", "precision", "recall", "f1"]),
        // Weighted metrics (when weight column is present)
        ...expandAtKMetrics(["weighted_ndcg", "weighted_recall"]),
        // Also in ALLOWED_METRICS and usable for link prediction
        ...expandAtKMetrics(["ndcg"])
    ],
    regression: [
        { value: "mae", label: "MAE" },
        { value: "mse", label: "MSE" },
        { value: "rmse", label: "RMSE" },
        { value: "smape", label: "SMAPE" }
    ],
    forecasting: [
        { value: "mae", label: "MAE" },
        { value: "mse", label: "MSE" },
        { value: "rmse", label: "RMSE" },
        { value: "smape", label: "SMAPE" }
    ]
};

// Also map temporal variants to their base task type for metrics lookup
const TASK_TYPE_ALIASES: Record<string, string> = {
    temporal_binary_classification: "binary_classification",
    temporal_multiclass_classification: "multiclass_classification",
    temporal_multilabel_classification: "multilabel_classification",
    temporal_regression: "regression",
    temporal_link_prediction: "link_prediction",
    static_link_prediction: "link_prediction",
    temporal_forecasting: "forecasting"
};

const getMetricsForTaskType = (
    taskType?: string
): Array<{ value: string; label: string }> => {
    if (!taskType) return Object.values(METRICS_BY_TASK).flat();
    const resolved = TASK_TYPE_ALIASES[taskType] ?? taskType;
    return METRICS_BY_TASK[resolved] ?? Object.values(METRICS_BY_TASK).flat();
};

const EXPERIMENT_OPTIONS = Array.from({ length: 10 }, (_, i) => ({
    value: String(i + 1),
    label: String(i + 1)
}));

const MODEL_TYPE_OPTIONS = [
    { value: "GNN", label: "GNN" },
    { value: "GraphTransformer", label: "GraphTransformer" }
];

// Per-encoder na_strategy options (NumericalNaStrategy enum)
// These stay in local state only — not emitted to column_processing.na_strategy
const NA_STRATEGY_OPTIONS = [
    { value: "zero", label: "Zero" },
    { value: "mean", label: "Mean" },
    { value: "separate", label: "Separate" },
    { value: "raise", label: "Raise" }
];

const CATEGORICAL_NA_STRATEGY_OPTIONS = [
    { value: "'zero'", label: "Zero" },
    { value: "'separate'", label: "Separate" },
    { value: "'most_frequent'", label: "Most Frequent" }
];

const MULTI_CATEGORICAL_NA_STRATEGY_OPTIONS = [
    { value: "'zero'", label: "Zero" },
    { value: "'separate'", label: "Separate" },
    { value: "'most_frequent'", label: "Most Frequent" }
];

const DATETIME_BOOLEAN_OPTIONS = [
    { value: "true", label: "True" },
    { value: "false", label: "False" }
];

// Quoted as the backend writes it (see normalizeEncoderValue).
const DEFAULT_TEXT_EMBEDDING_MODEL = "'glove.6B-50d'";

// Hosted images bake only the default model. A plan that already names
// another model the backend accepts still shows it.
const textEmbeddingModelOptions = (model: string) =>
    [DEFAULT_TEXT_EMBEDDING_MODEL, model]
        .filter((value, index, values) => values.indexOf(value) === index)
        .map((value) => ({ value, label: value.replace(/^'(.*)'$/, "$1") }));

// ─── Layout helpers ───────────────────────────────────────────────────────────

interface SectionInfoPopoverProps {
    description: string;
    linkText: string;
    pageLink: string;
    id: string;
}

const SectionInfoPopover = ({
    description,
    linkText,
    pageLink,
    id
}: SectionInfoPopoverProps) => (
    <HoverCard>
        <HoverCardTrigger className="cursor-pointer flex items-center">
            <Info className="size-3.5 text-icon-secondary" />
        </HoverCardTrigger>
        <HoverCardContent
            arrow={false}
            className="w-72 text-white text-sm font-medium p-4 rounded space-y-2"
            // eslint-disable-next-line no-restricted-syntax -- tooltip-dark token not yet a class
            style={{ backgroundColor: "var(--tooltip-dark, #3a3a41)" }}>
            <p>{description}</p>
            <DocumentationLink
                id={id}
                pageLink={pageLink}
                linkText={linkText}
                isNewDocs={true}
                className="text-blue-300! hover:text-blue-400!"
                showExternalIcon={true}
            />
        </HoverCardContent>
    </HoverCard>
);

interface SectionRowProps {
    title: string;
    infoPopover: React.ReactNode;
    children: React.ReactNode;
    rightAction?: React.ReactNode;
}

const SectionRow = ({
    title,
    infoPopover,
    children,
    rightAction
}: SectionRowProps) => (
    <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-1">
                <span className="text-base font-medium text-title-primary">
                    {title}
                </span>
                {infoPopover}
            </div>
            {rightAction}
        </div>
        <div className="flex flex-col gap-4">{children}</div>
    </div>
);

interface FieldRowProps {
    label: string;
    children: React.ReactNode;
}

const FieldRow = ({ label, children }: FieldRowProps) => (
    <div className="flex flex-col gap-2">
        <span className="text-sm font-medium text-title-secondary">
            {label}
        </span>
        <div>{children}</div>
    </div>
);

interface FieldPairRowProps {
    left: React.ReactNode;
    right: React.ReactNode;
}

const FieldPairRow = ({ left, right }: FieldPairRowProps) => (
    <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
        {left}
        {right}
    </div>
);

const sectionActionButtonClassName =
    "h-6 rounded border border-border-primary bg-white px-2 text-sm font-medium text-title-secondary shadow-none";

const getFormEncoderOptions = (stype: string | undefined) => [
    ...getEncoderOptionsForStype(stype).map((option) => ({
        value: option.value,
        label: option.label
    }))
];

const getEncoderName = (encoder: string) => {
    const trimmed = encoder.trim();
    const match = trimmed.match(/^([A-Za-z_][\w.]*)\(/);
    if (match) {
        return match[1].split(".").pop() ?? match[1];
    }
    return trimmed.replace(/\(\)$/, "");
};

const parseEncoderArgs = (encoder: string) => {
    const match = encoder.match(/^[A-Za-z_][\w.]*\((.*)\)$/);
    if (!match) return {} as Record<string, string>;
    const argsString = match[1].trim();
    if (!argsString) return {};

    return argsString.split(",").reduce<Record<string, string>>((acc, part) => {
        const [key, ...rest] = part.split("=");
        if (!key || rest.length === 0) return acc;
        acc[key.trim()] = rest.join("=").trim();
        return acc;
    }, {});
};

const buildEncoderExpression = (
    encoderName: string,
    args: Record<string, string>
) => {
    const entries = Object.entries(args).filter(([, value]) => value.trim());
    if (entries.length === 0) {
        return `${encoderName}()`;
    }
    return `${encoderName}(${entries
        .map(([key, value]) => `${key}=${value}`)
        .join(", ")})`;
};

const getEncoderSelectValue = (stype: string | undefined, encoder: string) => {
    const encoderName = getEncoderName(encoder);
    const matchedOption = getFormEncoderOptions(stype).find(
        (option) => getEncoderName(option.value) === encoderName
    );
    return matchedOption?.value ?? encoder;
};

const getSelectedEncoderExpression = (
    selectedValue: string,
    currentEncoder: string
) => {
    const selectedName = getEncoderName(selectedValue);
    const currentName = getEncoderName(currentEncoder);
    const currentArgs = parseEncoderArgs(currentEncoder);
    const nextArgs =
        currentName === selectedName
            ? currentArgs
            : getDefaultEncoderArgs(selectedName);
    return buildEncoderExpression(selectedName, nextArgs);
};

const getDefaultEncoderArgs = (encoderName: string): Record<string, string> => {
    switch (encoderName) {
        case "Index":
            return { min_occ: "1", na_strategy: "'separate'" };
        case "Hash":
            return {
                num_components: "1000",
                min_occ: "1",
                na_strategy: "'separate'"
            };
        case "MultiCategorical":
            return { min_occ: "1" };
        case "Numerical":
            return {};
        case "MaxLogNumerical":
        case "MinLogNumerical":
            return {};
        case "NumericalList":
            return { na_strategy: "'zero'" };
        case "Datetime":
            return {
                include_minute: "true",
                include_hour: "true",
                include_day_of_week: "true",
                include_day_of_month: "true",
                include_day_of_year: "true",
                include_year: "true"
            };
        case "TextEmbedding":
            return { model: DEFAULT_TEXT_EMBEDDING_MODEL };
        default:
            return {};
    }
};

const toPickerDate = (value: string): Date | undefined => {
    const trimmed = value.trim();
    if (!trimmed) return undefined;
    const knownFormats = [
        "yyyy-MM-dd HH:mm:ss",
        "yyyy-MM-dd HH:mm",
        "yyyy-MM-dd'T'HH:mm:ss",
        "yyyy-MM-dd'T'HH:mm:ss.SSSX",
        "yyyy-MM-dd'T'HH:mm:ssX",
        "yyyy-MM-dd"
    ];

    for (const dateFormat of knownFormats) {
        const parsedDate = parse(trimmed, dateFormat, new Date());
        if (isValid(parsedDate)) {
            return parsedDate;
        }
    }

    const fallbackDate = new Date(trimmed);
    return isValid(fallbackDate) ? fallbackDate : undefined;
};

// ─── Section 1: Training Table Generation ────────────────────────────────────

interface TrainingTableGenerationSectionProps {
    data: ModelPlan["training_table_generation"];
    onChange: (updated: ModelPlan["training_table_generation"]) => void;
}

const TrainingTableGenerationSection = ({
    data,
    onChange
}: TrainingTableGenerationSectionProps) => {
    const collisionBoundary = useCollisionBoundary();
    const [split, setSplit] = useState<SplitParams>(() =>
        parseSplitString(data?.split)
    );
    const splitRef = useRef<SplitParams>(split);
    const lastEmittedSplitRef = useRef<string | undefined>(data?.split);
    const prevSplitValue = useRef(data?.split);

    useEffect(() => {
        splitRef.current = split;
    }, [split]);

    useEffect(() => {
        if (data?.split !== prevSplitValue.current) {
            prevSplitValue.current = data?.split;
            if (data?.split === lastEmittedSplitRef.current) {
                return;
            }
            const parsedSplit = parseSplitString(data?.split);
            splitRef.current = parsedSplit;
            setSplit(parsedSplit);
        }
    }, [data?.split]);

    const getDefaultSplitParams = (type: string): SplitParams => {
        switch (type) {
            case "RandomSplit":
            case "ApproxDateOffsetSplit":
            case "TemporalSplit":
                return {
                    ...split,
                    type,
                    train: 0.8,
                    val: 0.1,
                    test: 0.1,
                    offsets: "",
                    unit: "",
                    timeframe: "",
                    anchorStartOffset: "",
                    timeframeStep: "",
                    trainStart: "",
                    trainEnd: "",
                    valStart: "",
                    valEnd: "",
                    testStart: "",
                    testEnd: ""
                };
            case "DateOffsetSplit":
                return {
                    ...split,
                    type,
                    train: 0.8,
                    val: 0.1,
                    test: 0.1,
                    offsets: "",
                    unit: "",
                    timeframe: "",
                    anchorStartOffset: "",
                    timeframeStep: "",
                    trainStart: "",
                    trainEnd: "",
                    valStart: "",
                    valEnd: "",
                    testStart: "",
                    testEnd: ""
                };
            case "TimeRangeSplit":
                return {
                    ...split,
                    type,
                    train: 0.8,
                    val: 0.1,
                    test: 0.1,
                    offsets: "",
                    unit: "",
                    timeframe: "",
                    anchorStartOffset: "",
                    timeframeStep: "",
                    trainStart: "",
                    trainEnd: "",
                    valStart: "",
                    valEnd: "",
                    testStart: "",
                    testEnd: ""
                };
            case "inferred":
                return {
                    ...split,
                    type,
                    offsets: "",
                    unit: "",
                    timeframe: "",
                    anchorStartOffset: "",
                    timeframeStep: "",
                    trainStart: "",
                    trainEnd: "",
                    valStart: "",
                    valEnd: "",
                    testStart: "",
                    testEnd: ""
                };
            default:
                return { ...split, type };
        }
    };

    const onSplitFieldChange = (field: keyof SplitParams, value: string) => {
        const numFields = new Set(["train", "val", "test"]);
        const updated = {
            ...splitRef.current,
            [field]: numFields.has(field) ? parseFloat(value) || 0 : value
        };
        splitRef.current = updated;
        const serializedSplit = buildSplitString(updated);
        lastEmittedSplitRef.current = serializedSplit;
        setSplit(updated);
        onChange({ ...data, split: serializedSplit });
    };

    const onBranchChange = (type: string) => {
        const updated = getDefaultSplitParams(type);
        splitRef.current = updated;
        const serializedSplit = buildSplitString(updated);
        lastEmittedSplitRef.current = serializedSplit;
        setSplit(updated);
        onChange({ ...data, split: serializedSplit });
    };

    const ratioSum = split.train + split.val + split.test;
    const hasRatioError =
        isRatioSplit(split.type) && Math.abs(ratioSum - 1.0) > 0.001;

    return (
        <SectionRow
            title="Training Table Generation"
            infoPopover={
                <SectionInfoPopover
                    description="Defines how Kumo splits your data into train, validation, and holdout sets."
                    linkText="Training table generation docs"
                    pageLink="model-planner/"
                    id="mp-training-table-generation-info"
                />
            }>
            <div className="flex flex-col gap-4">
                <FieldRow label="Branch">
                    <Select
                        variant="secondary"
                        popoverTriggerClassName="text-title-primary"
                        popoverClassName={MODEL_PLANNER_SELECT_POPOVER_CLASS}
                        collisionBoundary={collisionBoundary}
                        mode="single"
                        search={false}
                        placeholder="Select split type"
                        selectedValues={[split.type]}
                        options={SPLIT_TYPES.map((t) => ({
                            value: t.value,
                            label: t.label
                        }))}
                        onValuesChange={(vals) =>
                            onBranchChange(vals[0] ?? "ApproxDateOffsetSplit")
                        }
                    />
                </FieldRow>

                {/* Ratio-based: ApproxDateOffsetSplit, RandomSplit */}
                {isRatioSplit(split.type) && (
                    <>
                        <FieldRow label="Split parameters (train / val / test)">
                            <div className="flex items-end gap-2">
                                <div className="flex flex-col gap-2 flex-1">
                                    <span className="text-sm font-medium text-title-secondary">
                                        Train
                                    </span>
                                    <NumericInput
                                        type="number"
                                        step={0.01}
                                        min={0}
                                        max={1}
                                        value={split.train}
                                        onCommit={(v) =>
                                            onSplitFieldChange("train", v)
                                        }
                                    />
                                </div>
                                <div className="flex flex-col gap-2 flex-1">
                                    <span className="text-sm font-medium text-title-secondary">
                                        Val
                                    </span>
                                    <NumericInput
                                        type="number"
                                        step={0.01}
                                        min={0}
                                        max={1}
                                        value={split.val}
                                        onCommit={(v) =>
                                            onSplitFieldChange("val", v)
                                        }
                                    />
                                </div>
                                <div className="flex flex-col gap-2 flex-1">
                                    <span className="text-sm font-medium text-title-secondary">
                                        Test
                                    </span>
                                    <NumericInput
                                        type="number"
                                        step={0.01}
                                        min={0}
                                        max={1}
                                        value={split.test}
                                        onCommit={(v) =>
                                            onSplitFieldChange("test", v)
                                        }
                                    />
                                </div>
                            </div>
                        </FieldRow>
                        {hasRatioError && (
                            <p className="text-sm text-red-500">
                                Ratios define the proportion of data per split
                                and must sum to 1
                            </p>
                        )}
                    </>
                )}

                {/* DateOffsetSplit */}
                {split.type === "DateOffsetSplit" && (
                    <FieldPairRow
                        left={
                            <FieldRow label="Offsets">
                                <Input
                                    placeholder="[val_offset, test_offset]"
                                    value={split.offsets}
                                    onChange={(e) =>
                                        onSplitFieldChange(
                                            "offsets",
                                            e.target.value
                                        )
                                    }
                                />
                            </FieldRow>
                        }
                        right={
                            <FieldRow label="Unit (optional)">
                                <Input
                                    placeholder="days"
                                    value={split.unit}
                                    onChange={(e) =>
                                        onSplitFieldChange(
                                            "unit",
                                            e.target.value
                                        )
                                    }
                                />
                            </FieldRow>
                        }
                    />
                )}

                {/* TimeRangeSplit */}
                {split.type === "TimeRangeSplit" && (
                    <>
                        <div className="flex gap-2">
                            <div className="flex-1">
                                <FieldRow label="Train date start">
                                    <DateTimeInput
                                        key={`train-start-${split.trainStart}`}
                                        placeholder="select date"
                                        defaultDate={toPickerDate(
                                            split.trainStart
                                        )}
                                        onDateTimeChange={(date) =>
                                            onSplitFieldChange(
                                                "trainStart",
                                                format(
                                                    date,
                                                    "yyyy-MM-dd HH:mm:ss"
                                                )
                                            )
                                        }
                                    />
                                </FieldRow>
                            </div>
                            <div className="flex-1">
                                <FieldRow label="Train date end">
                                    <DateTimeInput
                                        key={`train-end-${split.trainEnd}`}
                                        placeholder="select date"
                                        defaultDate={toPickerDate(
                                            split.trainEnd
                                        )}
                                        onDateTimeChange={(date) =>
                                            onSplitFieldChange(
                                                "trainEnd",
                                                format(
                                                    date,
                                                    "yyyy-MM-dd HH:mm:ss"
                                                )
                                            )
                                        }
                                    />
                                </FieldRow>
                            </div>
                        </div>
                        <div className="flex gap-2">
                            <div className="flex-1">
                                <FieldRow label="Val date start">
                                    <DateTimeInput
                                        key={`val-start-${split.valStart}`}
                                        placeholder="select date"
                                        defaultDate={toPickerDate(
                                            split.valStart
                                        )}
                                        onDateTimeChange={(date) =>
                                            onSplitFieldChange(
                                                "valStart",
                                                format(
                                                    date,
                                                    "yyyy-MM-dd HH:mm:ss"
                                                )
                                            )
                                        }
                                    />
                                </FieldRow>
                            </div>
                            <div className="flex-1">
                                <FieldRow label="Val date end">
                                    <DateTimeInput
                                        key={`val-end-${split.valEnd}`}
                                        placeholder="select date"
                                        defaultDate={toPickerDate(split.valEnd)}
                                        onDateTimeChange={(date) =>
                                            onSplitFieldChange(
                                                "valEnd",
                                                format(
                                                    date,
                                                    "yyyy-MM-dd HH:mm:ss"
                                                )
                                            )
                                        }
                                    />
                                </FieldRow>
                            </div>
                        </div>
                        <div className="flex gap-2">
                            <div className="flex-1">
                                <FieldRow label="Test date start">
                                    <DateTimeInput
                                        key={`test-start-${split.testStart}`}
                                        placeholder="select date"
                                        defaultDate={toPickerDate(
                                            split.testStart
                                        )}
                                        onDateTimeChange={(date) =>
                                            onSplitFieldChange(
                                                "testStart",
                                                format(
                                                    date,
                                                    "yyyy-MM-dd HH:mm:ss"
                                                )
                                            )
                                        }
                                    />
                                </FieldRow>
                            </div>
                            <div className="flex-1">
                                <FieldRow label="Test date end">
                                    <DateTimeInput
                                        key={`test-end-${split.testEnd}`}
                                        placeholder="select date"
                                        defaultDate={toPickerDate(
                                            split.testEnd
                                        )}
                                        onDateTimeChange={(date) =>
                                            onSplitFieldChange(
                                                "testEnd",
                                                format(
                                                    date,
                                                    "yyyy-MM-dd HH:mm:ss"
                                                )
                                            )
                                        }
                                    />
                                </FieldRow>
                            </div>
                        </div>
                    </>
                )}
            </div>
        </SectionRow>
    );
};

// ─── Section 2: Training Job ──────────────────────────────────────────────────

interface TrainingJobSectionProps {
    data: ModelPlan["training_job"];
    onChange: (updated: ModelPlan["training_job"]) => void;
    taskType?: string;
}

const TrainingJobSection = ({
    data,
    onChange,
    taskType
}: TrainingJobSectionProps) => {
    const collisionBoundary = useCollisionBoundary();
    const numExperiments = data?.num_experiments;
    const metrics = Array.isArray(data?.metrics)
        ? (data.metrics as string[])
        : [];
    const tuneMetric =
        typeof data?.tune_metric === "string" ? data.tune_metric : "";

    // Get applicable metrics for this task type
    const taskMetrics = getMetricsForTaskType(taskType);
    // Include any extra metrics already in the plan but not in the task-specific list
    const knownValues = new Set(taskMetrics.map((o) => o.value));
    const extraOptions = metrics
        .filter((m) => !knownValues.has(m))
        .map((m) => ({ value: m, label: m.toUpperCase().replace(/_/g, " ") }));
    const allMetricOptions =
        extraOptions.length > 0
            ? [...taskMetrics, ...extraOptions]
            : taskMetrics;

    const metricLabelByValue = new Map(
        allMetricOptions.map((option) => [option.value, option.label])
    );
    const tuneMetricOptions =
        metrics.length > 0
            ? metrics.map((metric) => ({
                  value: metric,
                  label: metricLabelByValue.get(metric) ?? metric
              }))
            : allMetricOptions;

    return (
        <SectionRow
            title="Training Job"
            infoPopover={
                <SectionInfoPopover
                    description="Configures the number of experiments and the metric used to select the best model. You can also customize which evaluation metrics are reported after training."
                    linkText="Training Job"
                    pageLink="reference/metrics/"
                    id="mp-training-job-info"
                />
            }>
            <div className="flex flex-col gap-4">
                <FieldRow label="Number of experiments">
                    <Select
                        variant="secondary"
                        popoverTriggerClassName="text-title-primary"
                        popoverClassName={MODEL_PLANNER_SELECT_POPOVER_CLASS}
                        collisionBoundary={collisionBoundary}
                        mode="single"
                        search={false}
                        sortOptions={false}
                        placeholder="Select"
                        selectedValues={
                            numExperiments !== undefined
                                ? [String(numExperiments)]
                                : []
                        }
                        options={EXPERIMENT_OPTIONS}
                        onValuesChange={(vals) =>
                            onChange({
                                ...data,
                                num_experiments: vals[0]
                                    ? parseInt(vals[0])
                                    : undefined
                            })
                        }
                    />
                </FieldRow>
                <FieldRow label="Eval metrics">
                    <Select
                        variant="secondary"
                        popoverTriggerClassName="text-title-primary"
                        popoverClassName={MODEL_PLANNER_SELECT_POPOVER_CLASS}
                        collisionBoundary={collisionBoundary}
                        maxPopoverHeight={200}
                        mode="multi"
                        search={true}
                        sortOptions={false}
                        placeholder="Select metrics"
                        showSelectedCount={false}
                        multiModeText={
                            <>
                                Selected metrics
                                {metrics.length > 0 && (
                                    <span className="flex items-center justify-center w-6 h-6 bg-surface-secondary rounded">
                                        {metrics.length}
                                    </span>
                                )}
                            </>
                        }
                        selectedValues={metrics}
                        options={allMetricOptions}
                        onValuesChange={(vals) =>
                            onChange({
                                ...data,
                                metrics: vals.length > 0 ? vals : undefined,
                                tune_metric: vals.includes(tuneMetric)
                                    ? tuneMetric
                                    : (vals[0] ?? undefined)
                            })
                        }
                    />
                </FieldRow>
                <FieldRow label="Tune metrics">
                    <Select
                        variant="secondary"
                        popoverTriggerClassName="text-title-primary"
                        popoverClassName={MODEL_PLANNER_SELECT_POPOVER_CLASS}
                        collisionBoundary={collisionBoundary}
                        maxPopoverHeight={200}
                        mode="single"
                        search={true}
                        sortOptions={false}
                        placeholder="Select metric"
                        selectedValues={tuneMetric ? [tuneMetric] : []}
                        options={tuneMetricOptions}
                        onValuesChange={(vals) =>
                            onChange({
                                ...data,
                                tune_metric: vals[0] ?? undefined
                            })
                        }
                    />
                </FieldRow>
            </div>
        </SectionRow>
    );
};

// ─── Section 3: Column Processing ────────────────────────────────────────────

interface ColumnProcessingSectionProps {
    data: ModelPlan["column_processing"];
    onChange: (updated: ModelPlan["column_processing"]) => void;
    graphTables?: Record<
        string,
        { cols: Array<{ name: string; stype?: string }>; pkey?: string | null }
    >;
}

const ColumnProcessingSection = ({
    data,
    onChange,
    graphTables
}: ColumnProcessingSectionProps) => {
    const collisionBoundary = useCollisionBoundary();
    const tableOptions = useMemo(
        () =>
            Object.keys(graphTables ?? {}).map((t) => ({
                value: t,
                label: t
            })),
        [graphTables]
    );

    const getColumnOptions = (tableName: string) => {
        const table = graphTables?.[tableName];
        if (!table) return [];
        return table.cols
            .filter((c) => c.name !== table.pkey)
            .map((c) => ({ value: c.name, label: c.name }));
    };

    const getColumnStype = (
        tableName: string,
        columnName: string
    ): string | undefined => {
        const table = graphTables?.[tableName];
        if (!table) return undefined;
        return table.cols.find((c) => c.name === columnName)?.stype;
    };

    const isPrimaryKeyColumn = (tableName: string, columnName: string) =>
        Boolean(
            tableName &&
            columnName &&
            graphTables?.[tableName]?.pkey === columnName
        );

    const naStrategyMap = (data as Record<string, unknown> | undefined)
        ?.na_strategy as Record<string, unknown> | null;
    const [rules, setRules] = useState<EncoderRule[]>(() =>
        getEncoderRules(data?.encoder_overrides ?? null, naStrategyMap)
    );
    const skipNextSyncRef = useRef(false);
    const prevOverrides = useRef(data?.encoder_overrides);

    useEffect(() => {
        if (data?.encoder_overrides !== prevOverrides.current) {
            prevOverrides.current = data?.encoder_overrides;
            if (skipNextSyncRef.current) {
                skipNextSyncRef.current = false;
                return;
            }
            setRules(
                getEncoderRules(data?.encoder_overrides ?? null, naStrategyMap)
            );
        }
    }, [data?.encoder_overrides, naStrategyMap]);

    const commitRules = (updated: EncoderRule[]) => {
        setRules(updated);
        // Skip the next parent-driven sync so we don't lose empty rows
        // that can't round-trip through YAML serialization.
        skipNextSyncRef.current = true;
        onChange({
            ...data,
            encoder_overrides: buildEncoderOverrides(updated),
            na_strategy: buildNaStrategy(updated)
        } as unknown as ModelPlan["column_processing"]);
    };

    const addRule = () =>
        commitRules([
            ...rules,
            { table: "", column: "", encoder: "Numerical()", naStrategy: "" }
        ]);

    const updateExplicitOverride = (
        index: number,
        field: keyof EncoderRule,
        value: string
    ) => {
        const nextRules = rules.map((rule, ruleIndex) =>
            ruleIndex === index ? { ...rule, [field]: value } : rule
        );
        commitRules(nextRules);
    };

    const updateExplicitOverrideEncoder = (index: number, encoder: string) => {
        const nextRules = rules.map((rule, ruleIndex) =>
            ruleIndex === index
                ? {
                      ...rule,
                      encoder,
                      naStrategy: encoder.startsWith("Numerical")
                          ? rule.naStrategy
                          : ""
                  }
                : rule
        );
        commitRules(nextRules);
    };

    const updateEncoderArg = (
        index: number,
        encoderName: string,
        key: string,
        value: string
    ) => {
        const nextRules = rules.map((rule, ruleIndex) => {
            if (ruleIndex !== index) return rule;
            const nextArgs = {
                ...parseEncoderArgs(rule.encoder),
                [key]: value
            };
            return {
                ...rule,
                encoder: buildEncoderExpression(encoderName, nextArgs)
            };
        });
        commitRules(nextRules);
    };

    const removeExplicitOverride = (index: number) =>
        commitRules(rules.filter((_, ruleIndex) => ruleIndex !== index));

    return (
        <SectionRow
            title="Column Processing"
            infoPopover={
                <SectionInfoPopover
                    description="Controls how each column in your graph is encoded before being passed to the model. Override the defaults to customize encoding or missing value handling for specific columns."
                    linkText="Column Processing"
                    pageLink="reference/column_processing/"
                    id="mp-column-processing-info"
                />
            }
            rightAction={
                <Button
                    variant="outline"
                    size="sm"
                    onClick={addRule}
                    className={sectionActionButtonClassName}>
                    <Add className="size-3.5" />
                    Add rule
                </Button>
            }>
            {rules.length > 0 && (
                <div className="flex flex-col gap-4">
                    {rules.map((rule, index) => (
                        <div
                            key={`${rule.table}-${rule.column}-${index}`}
                            className="flex flex-col gap-2">
                            {(() => {
                                const columnStype = getColumnStype(
                                    rule.table,
                                    rule.column
                                );
                                const selectedEncoderValue =
                                    getEncoderSelectValue(
                                        columnStype,
                                        rule.encoder
                                    );
                                const encoderName = getEncoderName(
                                    rule.encoder
                                );
                                const encoderArgs = parseEncoderArgs(
                                    rule.encoder
                                );

                                return (
                                    <>
                                        <div className="flex items-center justify-between">
                                            <span className="text-sm font-medium text-title-secondary">
                                                Rule {index + 1}
                                            </span>
                                            <Button
                                                variant="outline"
                                                size="sm"
                                                onClick={() =>
                                                    removeExplicitOverride(
                                                        index
                                                    )
                                                }
                                                className={
                                                    sectionActionButtonClassName
                                                }>
                                                Remove
                                            </Button>
                                        </div>
                                        <FieldPairRow
                                            left={
                                                <FieldRow label="Table">
                                                    <Select
                                                        variant="secondary"
                                                        popoverTriggerClassName="text-title-primary"
                                                        popoverClassName={
                                                            MODEL_PLANNER_SELECT_POPOVER_CLASS
                                                        }
                                                        collisionBoundary={
                                                            collisionBoundary
                                                        }
                                                        mode="single"
                                                        search={true}
                                                        placeholder="select"
                                                        selectedValues={
                                                            rule.table
                                                                ? [rule.table]
                                                                : []
                                                        }
                                                        options={tableOptions}
                                                        onValuesChange={(
                                                            vals
                                                        ) =>
                                                            updateExplicitOverride(
                                                                index,
                                                                "table",
                                                                vals[0] ?? ""
                                                            )
                                                        }
                                                    />
                                                </FieldRow>
                                            }
                                            right={
                                                <FieldRow label="Column">
                                                    <Select
                                                        variant="secondary"
                                                        popoverTriggerClassName="text-title-primary"
                                                        popoverClassName={
                                                            MODEL_PLANNER_SELECT_POPOVER_CLASS
                                                        }
                                                        collisionBoundary={
                                                            collisionBoundary
                                                        }
                                                        mode="single"
                                                        search={true}
                                                        placeholder="select"
                                                        selectedValues={
                                                            rule.column
                                                                ? [rule.column]
                                                                : []
                                                        }
                                                        options={getColumnOptions(
                                                            rule.table
                                                        )}
                                                        onValuesChange={(
                                                            vals
                                                        ) =>
                                                            updateExplicitOverride(
                                                                index,
                                                                "column",
                                                                vals[0] ?? ""
                                                            )
                                                        }
                                                    />
                                                </FieldRow>
                                            }
                                        />
                                        {isPrimaryKeyColumn(
                                            rule.table,
                                            rule.column
                                        ) && (
                                            <p className="text-sm text-red-500">
                                                Primary key columns are
                                                structural graph keys and cannot
                                                be configured in Column
                                                Processing. Remove this rule and
                                                choose a non-primary-key column.
                                            </p>
                                        )}
                                        <FieldPairRow
                                            left={
                                                <FieldRow label="Encoder type">
                                                    <Select
                                                        variant="secondary"
                                                        popoverTriggerClassName="text-title-primary"
                                                        popoverClassName={
                                                            MODEL_PLANNER_SELECT_POPOVER_CLASS
                                                        }
                                                        collisionBoundary={
                                                            collisionBoundary
                                                        }
                                                        mode="single"
                                                        search={false}
                                                        placeholder="select"
                                                        selectedValues={[
                                                            selectedEncoderValue
                                                        ]}
                                                        options={getFormEncoderOptions(
                                                            columnStype
                                                        )}
                                                        onValuesChange={(
                                                            vals
                                                        ) =>
                                                            !isPrimaryKeyColumn(
                                                                rule.table,
                                                                rule.column
                                                            ) &&
                                                            updateExplicitOverrideEncoder(
                                                                index,
                                                                getSelectedEncoderExpression(
                                                                    vals[0] ??
                                                                        "Numerical()",
                                                                    rule.encoder
                                                                )
                                                            )
                                                        }
                                                    />
                                                </FieldRow>
                                            }
                                            right={
                                                !isPrimaryKeyColumn(
                                                    rule.table,
                                                    rule.column
                                                ) ? (
                                                    <FieldRow label="Na_strategy">
                                                        <Select
                                                            variant="secondary"
                                                            popoverTriggerClassName="text-title-primary"
                                                            popoverClassName={
                                                                MODEL_PLANNER_SELECT_POPOVER_CLASS
                                                            }
                                                            collisionBoundary={
                                                                collisionBoundary
                                                            }
                                                            mode="single"
                                                            search={false}
                                                            placeholder="select"
                                                            selectedValues={
                                                                encoderName ===
                                                                    "Numerical" ||
                                                                encoderName ===
                                                                    "MaxLogNumerical" ||
                                                                encoderName ===
                                                                    "MinLogNumerical" ||
                                                                encoderName ===
                                                                    "NumericalList"
                                                                    ? rule.naStrategy
                                                                        ? [
                                                                              rule.naStrategy
                                                                          ]
                                                                        : encoderName ===
                                                                            "NumericalList"
                                                                          ? [
                                                                                "zero"
                                                                            ]
                                                                          : []
                                                                    : [
                                                                          encoderArgs.na_strategy ??
                                                                              "'separate'"
                                                                      ]
                                                            }
                                                            options={
                                                                encoderName ===
                                                                "NumericalList"
                                                                    ? [
                                                                          {
                                                                              value: "zero",
                                                                              label: "Zero"
                                                                          }
                                                                      ]
                                                                    : encoderName ===
                                                                            "Numerical" ||
                                                                        encoderName ===
                                                                            "MaxLogNumerical" ||
                                                                        encoderName ===
                                                                            "MinLogNumerical"
                                                                      ? NA_STRATEGY_OPTIONS
                                                                      : encoderName ===
                                                                          "MultiCategorical"
                                                                        ? MULTI_CATEGORICAL_NA_STRATEGY_OPTIONS
                                                                        : CATEGORICAL_NA_STRATEGY_OPTIONS
                                                            }
                                                            onValuesChange={(
                                                                vals
                                                            ) =>
                                                                encoderName ===
                                                                    "Numerical" ||
                                                                encoderName ===
                                                                    "MaxLogNumerical" ||
                                                                encoderName ===
                                                                    "MinLogNumerical" ||
                                                                encoderName ===
                                                                    "NumericalList"
                                                                    ? updateExplicitOverride(
                                                                          index,
                                                                          "naStrategy",
                                                                          vals[0] ??
                                                                              ""
                                                                      )
                                                                    : updateEncoderArg(
                                                                          index,
                                                                          encoderName,
                                                                          "na_strategy",
                                                                          vals[0] ??
                                                                              "'separate'"
                                                                      )
                                                            }
                                                        />
                                                    </FieldRow>
                                                ) : (
                                                    <div />
                                                )
                                            }
                                        />
                                        {!isPrimaryKeyColumn(
                                            rule.table,
                                            rule.column
                                        ) &&
                                            encoderName === "Index" && (
                                                <FieldPairRow
                                                    left={
                                                        <FieldRow label="min_occ">
                                                            <NumericInput
                                                                type="number"
                                                                min={1}
                                                                value={
                                                                    encoderArgs.min_occ ??
                                                                    "1"
                                                                }
                                                                onCommit={(
                                                                    value
                                                                ) =>
                                                                    updateEncoderArg(
                                                                        index,
                                                                        "Index",
                                                                        "min_occ",
                                                                        value
                                                                    )
                                                                }
                                                            />
                                                        </FieldRow>
                                                    }
                                                    right={<div />}
                                                />
                                            )}
                                        {!isPrimaryKeyColumn(
                                            rule.table,
                                            rule.column
                                        ) &&
                                            encoderName === "Hash" && (
                                                <>
                                                    <FieldPairRow
                                                        left={
                                                            <FieldRow label="num_components">
                                                                <NumericInput
                                                                    type="number"
                                                                    min={1}
                                                                    value={
                                                                        encoderArgs.num_components ??
                                                                        "1000"
                                                                    }
                                                                    onCommit={(
                                                                        value
                                                                    ) =>
                                                                        updateEncoderArg(
                                                                            index,
                                                                            "Hash",
                                                                            "num_components",
                                                                            value
                                                                        )
                                                                    }
                                                                />
                                                            </FieldRow>
                                                        }
                                                        right={
                                                            <FieldRow label="min_occ">
                                                                <NumericInput
                                                                    type="number"
                                                                    min={1}
                                                                    value={
                                                                        encoderArgs.min_occ ??
                                                                        "1"
                                                                    }
                                                                    onCommit={(
                                                                        value
                                                                    ) =>
                                                                        updateEncoderArg(
                                                                            index,
                                                                            "Hash",
                                                                            "min_occ",
                                                                            value
                                                                        )
                                                                    }
                                                                />
                                                            </FieldRow>
                                                        }
                                                    />
                                                </>
                                            )}
                                        {!isPrimaryKeyColumn(
                                            rule.table,
                                            rule.column
                                        ) &&
                                            encoderName ===
                                                "MultiCategorical" && (
                                                <FieldPairRow
                                                    left={
                                                        <FieldRow label="min_occ">
                                                            <NumericInput
                                                                type="number"
                                                                min={1}
                                                                value={
                                                                    encoderArgs.min_occ ??
                                                                    "1"
                                                                }
                                                                onCommit={(
                                                                    value
                                                                ) =>
                                                                    updateEncoderArg(
                                                                        index,
                                                                        "MultiCategorical",
                                                                        "min_occ",
                                                                        value
                                                                    )
                                                                }
                                                            />
                                                        </FieldRow>
                                                    }
                                                    right={<div />}
                                                />
                                            )}
                                        {!isPrimaryKeyColumn(
                                            rule.table,
                                            rule.column
                                        ) &&
                                            encoderName === "Datetime" && (
                                                <>
                                                    <FieldPairRow
                                                        left={
                                                            <FieldRow label="num_year_periods">
                                                                <NumericInput
                                                                    type="number"
                                                                    min={1}
                                                                    value={
                                                                        encoderArgs.num_year_periods ??
                                                                        ""
                                                                    }
                                                                    onCommit={(
                                                                        value
                                                                    ) =>
                                                                        updateEncoderArg(
                                                                            index,
                                                                            "Datetime",
                                                                            "num_year_periods",
                                                                            value
                                                                        )
                                                                    }
                                                                />
                                                            </FieldRow>
                                                        }
                                                        right={<div />}
                                                    />
                                                    {[
                                                        "include_minute",
                                                        "include_hour",
                                                        "include_day_of_week",
                                                        "include_day_of_month",
                                                        "include_day_of_year",
                                                        "include_year"
                                                    ]
                                                        .reduce<
                                                            Array<
                                                                [
                                                                    string,
                                                                    (
                                                                        | string
                                                                        | undefined
                                                                    )
                                                                ]
                                                            >
                                                        >(
                                                            (
                                                                pairs,
                                                                fieldName,
                                                                i,
                                                                arr
                                                            ) => {
                                                                if (
                                                                    i % 2 ===
                                                                    0
                                                                ) {
                                                                    pairs.push([
                                                                        fieldName,
                                                                        arr[
                                                                            i +
                                                                                1
                                                                        ]
                                                                    ]);
                                                                }
                                                                return pairs;
                                                            },
                                                            []
                                                        )
                                                        .map(
                                                            ([
                                                                leftField,
                                                                rightField
                                                            ]) => (
                                                                <FieldPairRow
                                                                    key={
                                                                        leftField
                                                                    }
                                                                    left={
                                                                        <FieldRow
                                                                            label={
                                                                                leftField
                                                                            }>
                                                                            <Select
                                                                                variant="secondary"
                                                                                popoverTriggerClassName="text-title-primary"
                                                                                popoverClassName={
                                                                                    MODEL_PLANNER_SELECT_POPOVER_CLASS
                                                                                }
                                                                                collisionBoundary={
                                                                                    collisionBoundary
                                                                                }
                                                                                mode="single"
                                                                                search={
                                                                                    false
                                                                                }
                                                                                placeholder="select"
                                                                                selectedValues={[
                                                                                    encoderArgs[
                                                                                        leftField
                                                                                    ] ??
                                                                                        "true"
                                                                                ]}
                                                                                options={
                                                                                    DATETIME_BOOLEAN_OPTIONS
                                                                                }
                                                                                onValuesChange={(
                                                                                    vals
                                                                                ) =>
                                                                                    updateEncoderArg(
                                                                                        index,
                                                                                        "Datetime",
                                                                                        leftField,
                                                                                        vals[0] ??
                                                                                            "true"
                                                                                    )
                                                                                }
                                                                            />
                                                                        </FieldRow>
                                                                    }
                                                                    right={
                                                                        rightField ? (
                                                                            <FieldRow
                                                                                label={
                                                                                    rightField
                                                                                }>
                                                                                <Select
                                                                                    variant="secondary"
                                                                                    popoverTriggerClassName="text-title-primary"
                                                                                    popoverClassName={
                                                                                        MODEL_PLANNER_SELECT_POPOVER_CLASS
                                                                                    }
                                                                                    collisionBoundary={
                                                                                        collisionBoundary
                                                                                    }
                                                                                    mode="single"
                                                                                    search={
                                                                                        false
                                                                                    }
                                                                                    placeholder="select"
                                                                                    selectedValues={[
                                                                                        encoderArgs[
                                                                                            rightField
                                                                                        ] ??
                                                                                            "true"
                                                                                    ]}
                                                                                    options={
                                                                                        DATETIME_BOOLEAN_OPTIONS
                                                                                    }
                                                                                    onValuesChange={(
                                                                                        vals
                                                                                    ) =>
                                                                                        updateEncoderArg(
                                                                                            index,
                                                                                            "Datetime",
                                                                                            rightField,
                                                                                            vals[0] ??
                                                                                                "true"
                                                                                        )
                                                                                    }
                                                                                />
                                                                            </FieldRow>
                                                                        ) : (
                                                                            <div />
                                                                        )
                                                                    }
                                                                />
                                                            )
                                                        )}
                                                </>
                                            )}
                                        {!isPrimaryKeyColumn(
                                            rule.table,
                                            rule.column
                                        ) &&
                                            encoderName === "TextEmbedding" && (
                                                <FieldPairRow
                                                    left={
                                                        <FieldRow label="model">
                                                            <Select
                                                                variant="secondary"
                                                                popoverTriggerClassName="text-title-primary"
                                                                popoverClassName={
                                                                    MODEL_PLANNER_SELECT_POPOVER_CLASS
                                                                }
                                                                collisionBoundary={
                                                                    collisionBoundary
                                                                }
                                                                mode="single"
                                                                search={false}
                                                                placeholder="select"
                                                                selectedValues={[
                                                                    encoderArgs.model ??
                                                                        DEFAULT_TEXT_EMBEDDING_MODEL
                                                                ]}
                                                                options={textEmbeddingModelOptions(
                                                                    encoderArgs.model ??
                                                                        DEFAULT_TEXT_EMBEDDING_MODEL
                                                                )}
                                                                onValuesChange={(
                                                                    vals
                                                                ) =>
                                                                    updateEncoderArg(
                                                                        index,
                                                                        "TextEmbedding",
                                                                        "model",
                                                                        vals[0] ??
                                                                            DEFAULT_TEXT_EMBEDDING_MODEL
                                                                    )
                                                                }
                                                            />
                                                        </FieldRow>
                                                    }
                                                    right={<div />}
                                                />
                                            )}
                                    </>
                                );
                            })()}
                        </div>
                    ))}
                </div>
            )}
        </SectionRow>
    );
};

// ─── Section 4: Neighbor Sampling ────────────────────────────────────────────

interface NeighborSamplingSectionProps {
    data: ModelPlan["neighbor_sampling"];
    onChange: (updated: ModelPlan["neighbor_sampling"]) => void;
    graphTables?: Record<
        string,
        { cols: Array<{ name: string; stype?: string }>; pkey?: string | null }
    >;
    graphGroups?: ColumnGroup[];
}

const EMPTY_EDGE: DirectedEdge = {
    table1: "",
    column1: "",
    table2: "",
    column2: "",
    budget: ""
};

const NeighborSamplingSection = ({
    data,
    onChange,
    graphTables,
    graphGroups
}: NeighborSamplingSectionProps) => {
    const collisionBoundary = useCollisionBoundary();
    const validConnections = useMemo(() => {
        const connectionMap = new Map<string, Set<string>>();
        for (const group of graphGroups ?? []) {
            const columns = group.columns ?? [];
            if (columns.length < 2) continue;
            for (const source of columns) {
                for (const target of columns) {
                    if (
                        !source ||
                        !target ||
                        (source.table_name === target.table_name &&
                            source.col_name === target.col_name)
                    ) {
                        continue;
                    }
                    const sourceKey = `${source.table_name}.${source.col_name}`;
                    const targetKey = `${target.table_name}.${target.col_name}`;
                    if (!connectionMap.has(sourceKey)) {
                        connectionMap.set(sourceKey, new Set());
                    }
                    connectionMap.get(sourceKey)?.add(targetKey);
                }
            }
        }
        return connectionMap;
    }, [graphGroups]);

    const linkedTableNames = useMemo(() => {
        const names = new Set<string>();
        for (const sourceKey of validConnections.keys()) {
            names.add(sourceKey.split(".")[0]);
        }
        for (const targets of validConnections.values()) {
            for (const targetKey of targets) {
                names.add(targetKey.split(".")[0]);
            }
        }
        return names;
    }, [validConnections]);

    const tableOptions = useMemo(
        () =>
            Array.from(linkedTableNames).map((t) => ({
                value: t,
                label: t
            })),
        [linkedTableNames]
    );

    const getColumnOptions = (
        tableName: string,
        peer?: {
            tableName?: string;
            columnName?: string;
            direction?: "from" | "to";
        }
    ) => {
        const table = graphTables?.[tableName];
        if (!table) return [];

        if (!peer?.tableName || !peer?.columnName || !peer?.direction) {
            return table.cols
                .filter(
                    (column) =>
                        Array.from(validConnections.keys()).some((key) => {
                            const [sourceTable, sourceColumn] = key.split(".");
                            return (
                                sourceTable === tableName &&
                                sourceColumn === column.name
                            );
                        }) ||
                        Array.from(validConnections.values()).some((targets) =>
                            Array.from(targets).some((targetKey) => {
                                const [targetTable, targetColumn] =
                                    targetKey.split(".");
                                return (
                                    targetTable === tableName &&
                                    targetColumn === column.name
                                );
                            })
                        )
                )
                .map((c) => ({ value: c.name, label: c.name }));
        }

        if (peer.direction === "from") {
            const targetKey = `${peer.tableName}.${peer.columnName}`;
            return table.cols
                .filter((column) =>
                    validConnections
                        .get(`${tableName}.${column.name}`)
                        ?.has(targetKey)
                )
                .map((c) => ({ value: c.name, label: c.name }));
        }

        const sourceKey = `${peer.tableName}.${peer.columnName}`;
        return table.cols
            .filter((column) =>
                validConnections
                    .get(sourceKey)
                    ?.has(`${tableName}.${column.name}`)
            )
            .map((c) => ({ value: c.name, label: c.name }));
    };
    const activeHops = getActiveHops(data?.num_neighbors);
    const usedHopKeys = new Set(activeHops.map((h) => h.hopKey));
    const nextAvailableHop = HOP_KEYS.find((k) => !usedHopKeys.has(k));

    // Local state for directed edges per hop
    const [edgesMap, setEdgesMap] = useState<Record<string, DirectedEdge[]>>(
        () => {
            const m: Record<string, DirectedEdge[]> = {};
            for (const { hopKey } of activeHops) {
                const hopObj = getHopObj(data?.num_neighbors, hopKey);
                const edges = getDirectedEdges(hopObj);
                if (edges.length > 0) m[hopKey] = edges;
            }
            return m;
        }
    );
    const numNeighborsRef = useRef(data?.num_neighbors);

    // Sync edges from external YAML changes (e.g. switching from YAML tab)
    // Preserve local in-progress edges that haven't been committed yet
    const prevNumNeighbors = useRef(data?.num_neighbors);
    useEffect(() => {
        numNeighborsRef.current = data?.num_neighbors;
        if (data?.num_neighbors !== prevNumNeighbors.current) {
            prevNumNeighbors.current = data?.num_neighbors;
            setEdgesMap((prev) => {
                const m: Record<string, DirectedEdge[]> = {};
                for (const { hopKey } of getActiveHops(data?.num_neighbors)) {
                    const hopObj = getHopObj(data?.num_neighbors, hopKey);
                    const yamlEdges = getDirectedEdges(hopObj);
                    // Keep local edges if they exist (in-progress), otherwise use YAML
                    m[hopKey] =
                        yamlEdges.length > 0 ? yamlEdges : (prev[hopKey] ?? []);
                }
                return m;
            });
        }
    }, [data?.num_neighbors]);

    const onHopBudgetChange = (hopKey: HopKey, budgetStr: string) => {
        const budget = parseFloat(budgetStr);
        const updated = setHopValue(
            data?.num_neighbors,
            hopKey,
            isNaN(budget) ? null : budget
        );
        onChange({ ...data, num_neighbors: updated });
    };

    const addHop = () => {
        if (!nextAvailableHop) return;
        const updated = setHopValue(data?.num_neighbors, nextAvailableHop, 10);
        onChange({ ...data, num_neighbors: updated });
    };

    const removeHop = (hopKey: HopKey) => {
        const updated = setHopValue(numNeighborsRef.current, hopKey, null);
        numNeighborsRef.current = updated;
        setEdgesMap((prev) => {
            const next = { ...prev };
            delete next[hopKey];
            return next;
        });
        onChange({ ...data, num_neighbors: updated });
    };

    const addDirectedEdge = (hopKey: HopKey) => {
        setEdgesMap((prev) => ({
            ...prev,
            [hopKey]: [...(prev[hopKey] ?? []), { ...EMPTY_EDGE }]
        }));
    };

    const removeDirectedEdge = (hopKey: HopKey, edgeIndex: number) => {
        setEdgesMap((prev) => {
            const nextEdges = [...(prev[hopKey] ?? [])];
            nextEdges.splice(edgeIndex, 1);
            commitDirectedEdges(hopKey, nextEdges);
            return { ...prev, [hopKey]: nextEdges };
        });
    };

    const updateEdgeField = (
        hopKey: HopKey,
        edgeIndex: number,
        field: keyof DirectedEdge,
        value: string
    ) => {
        setEdgesMap((prev) => {
            const edges = [...(prev[hopKey] ?? [])];
            edges[edgeIndex] = { ...edges[edgeIndex], [field]: value };
            return { ...prev, [hopKey]: edges };
        });
    };

    const commitDirectedEdges = (hopKey: HopKey, edges: DirectedEdge[]) => {
        const completeEdges = edges.filter(
            (edge) =>
                edge.table1.trim() &&
                edge.column1.trim() &&
                edge.table2.trim() &&
                edge.column2.trim()
        );
        const updated = setHopDirectedEdges(
            numNeighborsRef.current,
            hopKey,
            completeEdges
        );
        numNeighborsRef.current = updated;
        onChange({ ...data, num_neighbors: updated });
    };

    const updateAndCommitEdge = (
        hopKey: HopKey,
        edgeIndex: number,
        field: keyof DirectedEdge,
        value: string
    ) => {
        const edges = [...(edgesMap[hopKey] ?? [])];
        edges[edgeIndex] = { ...edges[edgeIndex], [field]: value };
        setEdgesMap((prev) => ({ ...prev, [hopKey]: edges }));
        commitDirectedEdges(hopKey, edges);
    };

    return (
        <SectionRow
            title="Neighbor Sampling"
            infoPopover={
                <SectionInfoPopover
                    description="Controls how many neighboring nodes are sampled per hop when building each node's local subgraph. Higher budgets increase graph context at the cost of memory and training time."
                    linkText="Neighbor Sampling"
                    pageLink="reference/num_neighbors/"
                    id="mp-neighbor-sampling-info"
                />
            }
            rightAction={
                nextAvailableHop ? (
                    <Button
                        variant="outline"
                        size="sm"
                        onClick={addHop}
                        className={sectionActionButtonClassName}>
                        <Add className="size-3.5" />
                        Add hop
                    </Button>
                ) : null
            }>
            {activeHops.length > 0 && (
                <div className="flex flex-col gap-4">
                    {activeHops.map(({ hopKey, budget }, index) => {
                        const edges = edgesMap[hopKey] ?? [];
                        return (
                            <div key={hopKey} className="flex flex-col gap-2">
                                {index === 0 && (
                                    <span className="text-sm font-medium text-title-secondary">
                                        Hop budget support
                                    </span>
                                )}
                                <div className="flex items-start gap-2">
                                    <div className="flex-1">
                                        <FieldRow
                                            label={hopKey.replace(
                                                "hop",
                                                "Hop "
                                            )}>
                                            <NumericInput
                                                type="number"
                                                min={-1}
                                                max={128}
                                                value={budget}
                                                onCommit={(v) =>
                                                    onHopBudgetChange(hopKey, v)
                                                }
                                                aria-label={`${hopKey} budget`}
                                            />
                                        </FieldRow>
                                    </div>
                                    <Button
                                        variant="outline"
                                        size="sm"
                                        onClick={() => removeHop(hopKey)}
                                        className={cn(
                                            sectionActionButtonClassName,
                                            "mt-7"
                                        )}>
                                        Remove
                                    </Button>
                                </div>

                                {/* Directed edge fields */}
                                {edges.map((edge, ei) => (
                                    <div
                                        key={ei}
                                        className="flex flex-col gap-2">
                                        <div className="flex justify-end">
                                            <Button
                                                variant="outline"
                                                size="sm"
                                                onClick={() =>
                                                    removeDirectedEdge(
                                                        hopKey,
                                                        ei
                                                    )
                                                }
                                                className={
                                                    sectionActionButtonClassName
                                                }>
                                                Remove
                                            </Button>
                                        </div>
                                        <FieldPairRow
                                            left={
                                                <FieldRow label="Table 1">
                                                    <Select
                                                        variant="secondary"
                                                        popoverTriggerClassName="text-title-primary"
                                                        popoverClassName={
                                                            MODEL_PLANNER_SELECT_POPOVER_CLASS
                                                        }
                                                        collisionBoundary={
                                                            collisionBoundary
                                                        }
                                                        mode="single"
                                                        search={true}
                                                        placeholder="select"
                                                        selectedValues={
                                                            edge.table1
                                                                ? [edge.table1]
                                                                : []
                                                        }
                                                        options={tableOptions}
                                                        onValuesChange={(
                                                            vals
                                                        ) =>
                                                            updateAndCommitEdge(
                                                                hopKey,
                                                                ei,
                                                                "table1",
                                                                vals[0] ?? ""
                                                            )
                                                        }
                                                    />
                                                </FieldRow>
                                            }
                                            right={
                                                <FieldRow label="Column">
                                                    <Select
                                                        variant="secondary"
                                                        popoverTriggerClassName="text-title-primary"
                                                        popoverClassName={
                                                            MODEL_PLANNER_SELECT_POPOVER_CLASS
                                                        }
                                                        collisionBoundary={
                                                            collisionBoundary
                                                        }
                                                        mode="single"
                                                        search={true}
                                                        placeholder="select"
                                                        selectedValues={
                                                            edge.column1
                                                                ? [edge.column1]
                                                                : []
                                                        }
                                                        options={getColumnOptions(
                                                            edge.table1
                                                        )}
                                                        onValuesChange={(
                                                            vals
                                                        ) =>
                                                            updateAndCommitEdge(
                                                                hopKey,
                                                                ei,
                                                                "column1",
                                                                vals[0] ?? ""
                                                            )
                                                        }
                                                    />
                                                </FieldRow>
                                            }
                                        />
                                        <FieldPairRow
                                            left={
                                                <FieldRow label="Table 2">
                                                    <Select
                                                        variant="secondary"
                                                        popoverTriggerClassName="text-title-primary"
                                                        popoverClassName={
                                                            MODEL_PLANNER_SELECT_POPOVER_CLASS
                                                        }
                                                        collisionBoundary={
                                                            collisionBoundary
                                                        }
                                                        mode="single"
                                                        search={true}
                                                        placeholder="select"
                                                        selectedValues={
                                                            edge.table2
                                                                ? [edge.table2]
                                                                : []
                                                        }
                                                        options={Array.from(
                                                            new Set(
                                                                Array.from(
                                                                    validConnections.get(
                                                                        `${edge.table1}.${edge.column1}`
                                                                    ) ?? []
                                                                ).map(
                                                                    (
                                                                        targetKey
                                                                    ) =>
                                                                        targetKey.split(
                                                                            "."
                                                                        )[0]
                                                                )
                                                            )
                                                        ).map((tableName) => ({
                                                            value: tableName,
                                                            label: tableName
                                                        }))}
                                                        onValuesChange={(
                                                            vals
                                                        ) =>
                                                            updateAndCommitEdge(
                                                                hopKey,
                                                                ei,
                                                                "table2",
                                                                vals[0] ?? ""
                                                            )
                                                        }
                                                    />
                                                </FieldRow>
                                            }
                                            right={
                                                <FieldRow label="Column">
                                                    <Select
                                                        variant="secondary"
                                                        popoverTriggerClassName="text-title-primary"
                                                        popoverClassName={
                                                            MODEL_PLANNER_SELECT_POPOVER_CLASS
                                                        }
                                                        collisionBoundary={
                                                            collisionBoundary
                                                        }
                                                        mode="single"
                                                        search={true}
                                                        placeholder="select"
                                                        selectedValues={
                                                            edge.column2
                                                                ? [edge.column2]
                                                                : []
                                                        }
                                                        options={getColumnOptions(
                                                            edge.table2,
                                                            {
                                                                tableName:
                                                                    edge.table1,
                                                                columnName:
                                                                    edge.column1,
                                                                direction: "to"
                                                            }
                                                        )}
                                                        onValuesChange={(
                                                            vals
                                                        ) =>
                                                            updateAndCommitEdge(
                                                                hopKey,
                                                                ei,
                                                                "column2",
                                                                vals[0] ?? ""
                                                            )
                                                        }
                                                    />
                                                </FieldRow>
                                            }
                                        />
                                        <FieldRow label="Budget of directed edge (optional)">
                                            <Input
                                                placeholder="Inferred"
                                                type="text"
                                                value={edge.budget}
                                                onChange={(e) =>
                                                    updateEdgeField(
                                                        hopKey,
                                                        ei,
                                                        "budget",
                                                        e.target.value
                                                    )
                                                }
                                                onBlur={(e) =>
                                                    updateAndCommitEdge(
                                                        hopKey,
                                                        ei,
                                                        "budget",
                                                        e.target.value
                                                    )
                                                }
                                            />
                                        </FieldRow>
                                    </div>
                                ))}

                                {edges.length === 0 && (
                                    <Button
                                        variant="outline"
                                        size="sm"
                                        onClick={() => addDirectedEdge(hopKey)}
                                        className={cn(
                                            sectionActionButtonClassName,
                                            "self-start"
                                        )}>
                                        <Add className="size-3.5" />
                                        Add directed edge
                                    </Button>
                                )}
                            </div>
                        );
                    })}
                </div>
            )}
        </SectionRow>
    );
};

// ─── Section 5: Model Architecture ───────────────────────────────────────────

interface ModelArchitectureSectionProps {
    data: ModelPlan["model_architecture"];
    onChange: (updated: ModelPlan["model_architecture"]) => void;
    taskType?: string;
}

const LINK_PRED_MODULE_OPTIONS = [
    { value: "ranking", label: "Ranking" },
    { value: "embedding", label: "Embedding" }
];

const ModelArchitectureSection = ({
    data,
    onChange,
    taskType
}: ModelArchitectureSectionProps) => {
    const collisionBoundary = useCollisionBoundary();
    const isLinkPrediction =
        taskType === "temporal_link_prediction" ||
        taskType === "static_link_prediction" ||
        taskType === "link_prediction";

    const modelArr = data?.model as Array<Record<string, unknown>> | undefined;
    const modelConfig = useMemo(() => modelArr?.[0] ?? {}, [modelArr]);
    const modelName =
        typeof modelConfig.name === "string" ? modelConfig.name : "";
    const useSeqIdValue = (data as Record<string, unknown> | undefined)
        ?.use_seq_id;
    const useSeqId = Array.isArray(useSeqIdValue)
        ? Boolean(useSeqIdValue[0])
        : Boolean(useSeqIdValue);

    // Link prediction fields
    const dataRecord = data as Record<string, unknown> | undefined;
    const moduleValue = dataRecord?.module;
    const moduleStr = typeof moduleValue === "string" ? moduleValue : "";

    const isGraphTransformer = modelName === "GraphTransformer";
    const modelType = isGraphTransformer ? "GraphTransformer" : "GNN";
    const normalizedModelName: "GNN" | "GraphTransformer" =
        modelName === "GraphTransformer" ? "GraphTransformer" : "GNN";

    useEffect(() => {
        if (
            modelArr?.length &&
            modelName &&
            modelName !== "GNN" &&
            modelName !== "GraphTransformer"
        ) {
            onChange({
                ...data,
                model: [
                    {
                        ...modelConfig,
                        name: "GNN"
                    }
                ] as ModelArchitecturePlan["model"]
            });
        }
    }, [data, modelArr?.length, modelConfig, modelName, onChange]);

    const setModelName = (name: "GNN" | "GraphTransformer") => {
        onChange({
            ...data,
            model: [
                {
                    ...modelConfig,
                    name
                }
            ] as ModelArchitecturePlan["model"]
        });
    };

    const onTypeChange = (type: string) => {
        if (type === "GraphTransformer") {
            setModelName("GraphTransformer");
        } else {
            setModelName("GNN");
        }
    };

    const onUseSeqIdChange = (checked: boolean) => {
        onChange({
            ...data,
            model: modelArr?.length
                ? [
                      {
                          ...modelConfig,
                          name: normalizedModelName
                      }
                  ]
                : data?.model,
            use_seq_id: [
                checked
            ] as unknown as ModelArchitecturePlan["use_seq_id"]
        });
    };

    return (
        <SectionRow
            title="Model Architecture"
            infoPopover={
                <SectionInfoPopover
                    description="Configures the model type and the hyperparameter search space for your experiments. Fix a specific architecture to skip AutoML search, or leave it open to let Kumo find the best configuration."
                    linkText="Model Architecture"
                    pageLink="reference/gnn_model/"
                    id="mp-model-architecture-info"
                />
            }>
            <div className="flex flex-col gap-4">
                <FieldRow label="Model type">
                    <Select
                        variant="secondary"
                        popoverTriggerClassName="text-title-primary"
                        popoverClassName={MODEL_PLANNER_SELECT_POPOVER_CLASS}
                        collisionBoundary={collisionBoundary}
                        mode="single"
                        search={false}
                        placeholder="Select model type"
                        selectedValues={modelType ? [modelType] : []}
                        options={MODEL_TYPE_OPTIONS}
                        onValuesChange={(vals) =>
                            onTypeChange(vals[0] ?? "GNN")
                        }
                    />
                </FieldRow>
                {isLinkPrediction && (
                    <FieldRow label="Model architecture">
                        <Select
                            variant="secondary"
                            popoverTriggerClassName="text-title-primary"
                            popoverClassName={
                                MODEL_PLANNER_SELECT_POPOVER_CLASS
                            }
                            collisionBoundary={collisionBoundary}
                            mode="single"
                            search={false}
                            placeholder="Select architecture"
                            selectedValues={moduleStr ? [moduleStr] : []}
                            options={LINK_PRED_MODULE_OPTIONS}
                            onValuesChange={(vals) =>
                                onChange({
                                    ...data,
                                    module: vals[0] ?? undefined
                                } as ModelArchitecturePlan)
                            }
                        />
                    </FieldRow>
                )}
                <div className="flex items-center justify-between gap-2">
                    <span className="text-sm font-medium text-title-secondary">
                        Use Seq ID
                    </span>
                    <Switch
                        id="mp-use-seq-id"
                        checked={useSeqId}
                        onCheckedChange={onUseSeqIdChange}
                    />
                </div>
            </div>
        </SectionRow>
    );
};

// ─── Main export ──────────────────────────────────────────────────────────────

interface ModelPlannerFormSectionsProps {
    plan: ModelPlan;
    onChange: (updated: ModelPlan) => void;
    taskType?: string;
    graphTables?: Record<
        string,
        { cols: Array<{ name: string; stype?: string }>; pkey?: string | null }
    >;
    graphGroups?: ColumnGroup[];
}

const ModelPlannerFormSections = ({
    plan,
    onChange,
    taskType,
    graphTables,
    graphGroups
}: ModelPlannerFormSectionsProps) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const [boundary, setBoundary] = useState<Element | null>(null);

    useEffect(() => {
        setBoundary(findScrollableAncestor(containerRef.current));
    }, []);

    return (
        <CollisionBoundaryContext.Provider value={boundary}>
            <div ref={containerRef} className="flex flex-col gap-6">
                <TrainingTableGenerationSection
                    data={plan.training_table_generation}
                    onChange={(updated) =>
                        onChange({
                            ...plan,
                            training_table_generation: updated
                        })
                    }
                />
                <TrainingJobSection
                    data={plan.training_job}
                    onChange={(updated) =>
                        onChange({ ...plan, training_job: updated })
                    }
                    taskType={taskType}
                />
                <ColumnProcessingSection
                    data={plan.column_processing}
                    onChange={(updated) =>
                        onChange({ ...plan, column_processing: updated })
                    }
                    graphTables={graphTables}
                />
                <NeighborSamplingSection
                    data={plan.neighbor_sampling}
                    onChange={(updated) =>
                        onChange({ ...plan, neighbor_sampling: updated })
                    }
                    graphTables={graphTables}
                    graphGroups={graphGroups}
                />
                <ModelArchitectureSection
                    data={plan.model_architecture}
                    onChange={(updated) =>
                        onChange({ ...plan, model_architecture: updated })
                    }
                    taskType={taskType}
                />
            </div>
        </CollisionBoundaryContext.Provider>
    );
};

export default WithErrorBoundary(ModelPlannerFormSections);
