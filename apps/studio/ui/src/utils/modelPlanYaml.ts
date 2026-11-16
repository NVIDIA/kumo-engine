// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
    _NumNeighborsConfig,
    ColumnProcessingPlanEncoderOverrides,
    ModelPlan,
    NeighborSamplingPlanNumNeighbors
} from "gen/v1/types";
import { dump as yamlDump, load as yamlLoad } from "js-yaml";

// ─── Core parse / serialize ───────────────────────────────────────────────────

export function parseModelPlanYaml(yamlString: string): ModelPlan | null {
    if (!yamlString?.trim()) return null;
    try {
        const parsed = yamlLoad(yamlString);
        if (parsed && typeof parsed === "object") {
            return parsed as ModelPlan;
        }
        return null;
    } catch {
        return null;
    }
}

export function serializeModelPlanYaml(plan: ModelPlan): string {
    try {
        return yamlDump(plan, {
            noRefs: true,
            lineWidth: -1,
            skipInvalid: true
        });
    } catch {
        return "";
    }
}

// ─── Split string helpers ─────────────────────────────────────────────────────
// Handles "ApproxDateOffsetSplit(ratio=[0.8, 0.1, 0.1])" ↔ { type, train, val, test }

export const SPLIT_TYPES = [
    { value: "ApproxDateOffsetSplit", label: "Approx Date Offset Split" },
    { value: "RandomSplit", label: "Random Split" },
    { value: "TemporalSplit", label: "Temporal Split" },
    { value: "DateOffsetSplit", label: "Date Offset Split" },
    { value: "TimeRangeSplit", label: "Time Range Split" },
    { value: "inferred", label: "Inferred" }
] as const;

export interface SplitParams {
    type: string;
    // Ratio-based (ApproxDateOffsetSplit, RandomSplit)
    train: number;
    val: number;
    test: number;
    // DateOffsetSplit
    offsets: string;
    unit: string;
    timeframe: string;
    anchorStartOffset: string;
    timeframeStep: string;
    // TimeRangeSplit
    trainStart: string;
    trainEnd: string;
    valStart: string;
    valEnd: string;
    testStart: string;
    testEnd: string;
}

const SPLIT_DEFAULTS: SplitParams = {
    type: "ApproxDateOffsetSplit",
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

function stripWrappingQuotes(value: string): string {
    const trimmed = value.trim();
    if (
        (trimmed.startsWith("'") && trimmed.endsWith("'")) ||
        (trimmed.startsWith('"') && trimmed.endsWith('"'))
    ) {
        return trimmed.slice(1, -1);
    }
    return trimmed;
}

function splitTopLevelArgs(args: string): string[] {
    const parts: string[] = [];
    let current = "";
    let squareDepth = 0;
    let parenDepth = 0;
    let braceDepth = 0;
    let quote: "'" | '"' | null = null;

    for (const char of args) {
        if (quote) {
            current += char;
            if (char === quote) {
                quote = null;
            }
            continue;
        }

        if (char === "'" || char === '"') {
            quote = char;
            current += char;
            continue;
        }

        if (char === "[") squareDepth += 1;
        if (char === "]") squareDepth -= 1;
        if (char === "(") parenDepth += 1;
        if (char === ")") parenDepth -= 1;
        if (char === "{") braceDepth += 1;
        if (char === "}") braceDepth -= 1;

        if (
            char === "," &&
            squareDepth === 0 &&
            parenDepth === 0 &&
            braceDepth === 0
        ) {
            if (current.trim()) {
                parts.push(current.trim());
            }
            current = "";
            continue;
        }

        current += char;
    }

    if (current.trim()) {
        parts.push(current.trim());
    }

    return parts;
}

function parseNamedArgs(args: string): Record<string, string> {
    if (!args.trim()) return {};
    return splitTopLevelArgs(args).reduce<Record<string, string>>(
        (acc, part) => {
            const [key, ...rest] = part.split("=");
            if (!key || rest.length === 0) return acc;
            acc[key.trim()] = rest.join("=").trim();
            return acc;
        },
        {}
    );
}

function hasConcreteParam(value: string | undefined): boolean {
    if (!value) return false;
    const normalized = stripWrappingQuotes(value).trim().toLowerCase();
    return normalized !== "" && normalized !== "none" && normalized !== "null";
}

function parseRatioValues(value: string): number[] {
    return value
        .replace(/^\[/, "")
        .replace(/\]$/, "")
        .split(",")
        .map((s) => parseFloat(s.trim()))
        .filter((v) => !isNaN(v));
}

function parseBoundaryPairs(value: string): Array<[string, string]> {
    const trimmed = value.trim();
    if (!trimmed.startsWith("[") || !trimmed.endsWith("]")) {
        return [];
    }

    const inner = trimmed.slice(1, -1).trim();
    if (!inner) return [];

    return splitTopLevelArgs(inner)
        .map((part) => {
            const normalized = part
                .trim()
                .replace(/^[[()]\s*/, "")
                .replace(/\s*[\])]?\s*$/, "");
            const values =
                splitTopLevelArgs(normalized).map(stripWrappingQuotes);
            if (values.length < 2) return null;
            return [values[0], values[1]] as [string, string];
        })
        .filter((pair): pair is [string, string] => pair !== null);
}

type NamedArgs = ReturnType<typeof parseNamedArgs>;
type BoundaryRanges = ReturnType<typeof parseBoundaryPairs>;

/** The 6 train/val/test start/end fields from parsed boundary ranges. */
function rangeBoundaryFields(
    ranges: BoundaryRanges
): Pick<
    SplitParams,
    "trainStart" | "trainEnd" | "valStart" | "valEnd" | "testStart" | "testEnd"
> {
    return {
        trainStart: ranges[0]?.[0] ?? "",
        trainEnd: ranges[0]?.[1] ?? "",
        valStart: ranges[1]?.[0] ?? "",
        valEnd: ranges[1]?.[1] ?? "",
        testStart: ranges[2]?.[0] ?? "",
        testEnd: ranges[2]?.[1] ?? ""
    };
}

// Ratio-based: ApproxDateOffsetSplit([0.8, 0.1, 0.1]), RandomSplit(...),
// TemporalSplit(...). Also accept the previous legacy `ratio=` format.
function parseRatioSplit(split: string): SplitParams | null {
    const ratioMatch = split.match(
        /^(ApproxDateOffsetSplit|RandomSplit|TemporalSplit)\((?:ratio=)?\[([^\]]+)\](?:\s*,[^)]*)?\)$/
    );
    if (!ratioMatch) return null;
    const [, type, ratioStr] = ratioMatch;
    const parts = parseRatioValues(`[${ratioStr}]`);
    if (parts.length >= 2) {
        return {
            ...SPLIT_DEFAULTS,
            type,
            train: parts[0],
            val: parts[1],
            test: parts[2] ?? 0
        };
    }
    return { ...SPLIT_DEFAULTS, type };
}

function timeSplitFractions(params: NamedArgs): SplitParams | null {
    if (!params.fractions) return null;
    const parts = parseRatioValues(params.fractions);
    if (parts.length < 2) return null;
    return {
        ...SPLIT_DEFAULTS,
        type: "TemporalSplit",
        train: parts[0],
        val: parts[1],
        test: parts[2] ?? 0,
        unit: stripWrappingQuotes(params.unit ?? ""),
        timeframe: params.timeframe_size ?? params.timeframe ?? "",
        anchorStartOffset: params.anchor_start_offset ?? ""
    };
}

function timeSplitTimestampBoundaries(params: NamedArgs): SplitParams | null {
    if (!hasConcreteParam(params.timestamp_boundaries)) return null;
    const ranges = parseBoundaryPairs(params.timestamp_boundaries);
    return {
        ...SPLIT_DEFAULTS,
        type: "TimeRangeSplit",
        ...rangeBoundaryFields(ranges),
        unit: stripWrappingQuotes(params.unit ?? ""),
        timeframe: params.timeframe_size ?? params.timeframe ?? "",
        anchorStartOffset: params.anchor_start_offset ?? ""
    };
}

function timeSplitOffsetBoundaries(params: NamedArgs): SplitParams | null {
    if (!hasConcreteParam(params.offset_boundaries)) return null;
    const ranges = parseBoundaryPairs(params.offset_boundaries);
    const offsets = ranges
        .slice(0, -1)
        .map(([, end]) => end)
        .filter((value) => value && value.toUpperCase() !== "INF");
    return {
        ...SPLIT_DEFAULTS,
        type: "DateOffsetSplit",
        offsets: offsets.join(", "),
        unit: stripWrappingQuotes(params.unit ?? ""),
        timeframe: params.timeframe_size ?? params.timeframe ?? "",
        anchorStartOffset: params.anchor_start_offset ?? ""
    };
}

function parseTimeSplit(split: string): SplitParams | null {
    const timeSplitMatch = split.match(/^TimeSplit\(([\s\S]+)\)$/);
    if (!timeSplitMatch) return null;
    const params = parseNamedArgs(timeSplitMatch[1]);
    return (
        timeSplitFractions(params) ??
        timeSplitTimestampBoundaries(params) ??
        timeSplitOffsetBoundaries(params)
    );
}

// DateOffsetSplit([...], unit=day) and legacy DateOffsetSplit(offsets=[...], unit=day)
function parseDateOffsetSplit(split: string): SplitParams | null {
    const dosMatch = split.match(
        /^DateOffsetSplit\((?:offsets=)?\[([^\]]*)\](?:,\s*(.*))?\)$/
    );
    if (!dosMatch) return null;
    const params = parseNamedArgs(dosMatch[2] ?? "");
    return {
        ...SPLIT_DEFAULTS,
        type: "DateOffsetSplit",
        offsets: dosMatch[1] ?? "",
        unit: stripWrappingQuotes(params.unit ?? ""),
        timeframe: params.timeframe ?? "",
        anchorStartOffset: params.anchor_start_offset ?? "",
        timeframeStep: params.timeframe_step ?? ""
    };
}

// TimeRangeSplit([('2024-01-01', '2024-02-01'), ...], unit=days, ...)
function parseTimeRangeSplit(split: string): SplitParams | null {
    const trsMatch = split.match(
        /^TimeRangeSplit\((\[[\s\S]*\])(?:,\s*(.*))?\)$/
    );
    if (!trsMatch) return null;
    const ranges = parseBoundaryPairs(trsMatch[1]);
    const params = parseNamedArgs(trsMatch[2] ?? "");
    return {
        ...SPLIT_DEFAULTS,
        type: "TimeRangeSplit",
        ...rangeBoundaryFields(ranges),
        unit: stripWrappingQuotes(params.unit ?? ""),
        timeframe: params.timeframe ?? "",
        anchorStartOffset: params.anchor_start_offset ?? "",
        timeframeStep: params.timeframe_step ?? ""
    };
}

export function parseSplitString(split: unknown): SplitParams {
    if (!split) return { ...SPLIT_DEFAULTS };
    if (split === "inferred") return { ...SPLIT_DEFAULTS, type: "inferred" };
    if (typeof split !== "string") return { ...SPLIT_DEFAULTS };

    return (
        parseRatioSplit(split) ??
        parseTimeSplit(split) ??
        parseDateOffsetSplit(split) ??
        parseTimeRangeSplit(split) ?? { ...SPLIT_DEFAULTS, type: split }
    );
}

export function buildSplitString(params: SplitParams): string {
    const { type } = params;
    if (type === "inferred") return "inferred";
    if (type === "DateOffsetSplit") {
        const parts = [`[${params.offsets}]`];
        if (params.unit.trim()) parts.push(`unit='${params.unit.trim()}'`);
        if (params.timeframe.trim()) {
            parts.push(`timeframe=${params.timeframe.trim()}`);
        }
        if (params.anchorStartOffset.trim()) {
            parts.push(
                `anchor_start_offset=${params.anchorStartOffset.trim()}`
            );
        }
        if (params.timeframeStep.trim()) {
            parts.push(`timeframe_step=${params.timeframeStep.trim()}`);
        }
        return `DateOffsetSplit(${parts.join(", ")})`;
    }
    if (type === "TimeRangeSplit") {
        const ranges = [
            [params.trainStart, params.trainEnd],
            [params.valStart, params.valEnd],
            [params.testStart, params.testEnd]
        ]
            .filter(([start, end]) => start.trim() || end.trim())
            .map(([start, end]) => `('${start.trim()}', '${end.trim()}')`);
        if (ranges.length === 0) return "TimeRangeSplit([])";
        const parts = [`[${ranges.join(", ")}]`];
        if (params.unit.trim()) parts.push(`unit='${params.unit.trim()}'`);
        if (params.timeframe.trim()) {
            parts.push(`timeframe=${params.timeframe.trim()}`);
        }
        if (params.anchorStartOffset.trim()) {
            parts.push(
                `anchor_start_offset=${params.anchorStartOffset.trim()}`
            );
        }
        if (params.timeframeStep.trim()) {
            parts.push(`timeframe_step=${params.timeframeStep.trim()}`);
        }
        return `TimeRangeSplit(${parts.join(", ")})`;
    }
    return `${type}([${params.train}, ${params.val}, ${params.test}])`;
}

export function isRatioSplit(type: string): boolean {
    return (
        type === "ApproxDateOffsetSplit" ||
        type === "RandomSplit" ||
        type === "TemporalSplit"
    );
}

// ─── Neighbor Sampling helpers ────────────────────────────────────────────────
// num_neighbors format: [{ hop1: { default: N }, hop2: { default: N }, ... }]

export const HOP_KEYS = [
    "hop1",
    "hop2",
    "hop3",
    "hop4",
    "hop5",
    "hop6"
] as const;
export type HopKey = (typeof HOP_KEYS)[number];

function parseLegacyHopArray(
    arr: number[]
): Array<{ hopKey: HopKey; budget: number }> {
    return arr
        .map((val, idx) => {
            const hopKey = HOP_KEYS[idx];
            const budget =
                typeof val === "number" ? val : parseInt(String(val), 10);
            return hopKey !== undefined && !isNaN(budget)
                ? { hopKey, budget }
                : null;
        })
        .filter((h): h is { hopKey: HopKey; budget: number } => h !== null);
}

export function getActiveHops(
    numNeighbors: NeighborSamplingPlanNumNeighbors | undefined
): Array<{ hopKey: HopKey; budget: number }> {
    if (
        !numNeighbors ||
        !Array.isArray(numNeighbors) ||
        numNeighbors.length === 0
    )
        return [];
    const first = numNeighbors[0];
    if (!first) return [];
    if (Array.isArray(first)) return parseLegacyHopArray(first as number[]);
    if (typeof first !== "object") return [];
    const result: Array<{ hopKey: HopKey; budget: number }> = [];
    for (const key of HOP_KEYS) {
        const hop = (first as Record<string, unknown>)[key];
        if (hop && typeof hop === "object") {
            const def = (hop as { default?: number }).default;
            if (def !== undefined) result.push({ hopKey: key, budget: def });
        }
    }
    return result;
}

function getFirstHopObj(
    numNeighbors: NeighborSamplingPlanNumNeighbors | undefined
): Record<string, unknown> {
    if (!Array.isArray(numNeighbors) || numNeighbors.length === 0) return {};
    const first = numNeighbors[0];
    if (!first || typeof first !== "object") return {};
    if (Array.isArray(first)) {
        const obj: Record<string, unknown> = {};
        for (const { hopKey, budget } of parseLegacyHopArray(
            first as number[]
        )) {
            obj[hopKey] = { default: budget };
        }
        return obj;
    }
    return { ...(first as Record<string, unknown>) };
}

export function getHopObj(
    numNeighbors: NeighborSamplingPlanNumNeighbors | undefined,
    hopKey: HopKey
): Record<string, unknown> | undefined {
    const first = getFirstHopObj(numNeighbors);
    const hop = first[hopKey];
    return hop && typeof hop === "object"
        ? (hop as Record<string, unknown>)
        : undefined;
}

export function setHopValue(
    numNeighbors: NeighborSamplingPlanNumNeighbors | undefined,
    hopKey: HopKey,
    value: number | null
): _NumNeighborsConfig[] {
    const existing = getFirstHopObj(numNeighbors);
    if (value !== null) {
        const prev = existing[hopKey];
        // Preserve directed edge keys if the hop already exists
        const base =
            prev && typeof prev === "object"
                ? { ...(prev as Record<string, unknown>) }
                : {};
        base.default = value;
        existing[hopKey] = base;
    } else {
        delete existing[hopKey];
    }
    return [existing as _NumNeighborsConfig];
}

export function setHopDirectedEdges(
    numNeighbors: NeighborSamplingPlanNumNeighbors | undefined,
    hopKey: HopKey,
    edges: DirectedEdge[]
): _NumNeighborsConfig[] {
    const existing = getFirstHopObj(numNeighbors);
    const prev = existing[hopKey];
    if (!prev || typeof prev !== "object")
        return [existing as _NumNeighborsConfig];

    // Strip old directed edge keys, keep only "default"
    const hopObj: Record<string, unknown> = {
        default: (prev as Record<string, unknown>).default
    };
    // Add new directed edge keys
    const entries = buildDirectedEdgeEntries(
        edges,
        typeof hopObj.default === "number" ? hopObj.default : undefined
    );
    for (const [k, v] of Object.entries(entries)) {
        hopObj[k] = v;
    }
    existing[hopKey] = hopObj;
    return [existing as _NumNeighborsConfig];
}

// ─── Encoder rule helpers ─────────────────────────────────────────────────────
// encoder_overrides format: { "column_name": "encoder_type_string" }

export interface EncoderRule {
    table: string;
    column: string;
    encoder: string;
    naStrategy: string;
}

interface GlobalEncoderDefault {
    stype: string;
    encoder: string;
    naStrategy: string;
}

interface ScopedEncoderRule {
    table: string;
    columnPattern: string;
    encoder: string;
    naStrategy: string;
}

export interface ColumnProcessingFormState {
    globalDefaults: GlobalEncoderDefault[];
    scopedRules: ScopedEncoderRule[];
    explicitOverrides: EncoderRule[];
}

// Encoder values must be Python constructor syntax: "ClassName()" for repr_to_object parsing
// Each encoder supports specific stypes (from kumoapi.encoder.*.supported_stypes)
export const ENCODER_OPTIONS = [
    { value: "Numerical()", label: "Numerical", stypes: ["numerical"] },
    { value: "Hash()", label: "Hash", stypes: ["categorical", "ID"] },
    {
        value: "MultiCategorical()",
        label: "Multi-Categorical",
        stypes: ["multicategorical"]
    },
    { value: "Datetime()", label: "Datetime", stypes: ["timestamp"] },
    { value: "NumericalList()", label: "Numerical List", stypes: ["sequence"] },
    {
        value: "MaxLogNumerical()",
        label: "Max Log Numerical",
        stypes: ["numerical"]
    },
    {
        value: "MinLogNumerical()",
        label: "Min Log Numerical",
        stypes: ["numerical"]
    },
    { value: "Index()", label: "Kumo Index", stypes: ["categorical", "ID"] },
    {
        value: "TextEmbedding()",
        label: "Text Embedding",
        stypes: ["text"]
    },
    { value: "Image()", label: "Image", stypes: ["image"] },
    {
        value: "Null()",
        label: "Null (no encoding)",
        stypes: [
            "categorical",
            "ID",
            "text",
            "image",
            "multicategorical",
            "numerical",
            "unsupported",
            "sequence",
            "timestamp"
        ]
    }
] as const;

function normalizeEncoderLookupKey(value: string): string {
    return value.toLowerCase().replace(/[\s_\-()]/g, "");
}

const ENCODER_VALUE_BY_LOOKUP_KEY = (() => {
    const entries = new Map<string, string>();
    for (const option of ENCODER_OPTIONS) {
        entries.set(normalizeEncoderLookupKey(option.value), option.value);
        entries.set(normalizeEncoderLookupKey(option.label), option.value);
        entries.set(
            normalizeEncoderLookupKey(option.value.replace(/\(\)$/, "")),
            option.value
        );
    }
    entries.set(normalizeEncoderLookupKey("KumoIndex"), "Index()");
    entries.set(normalizeEncoderLookupKey("KumoIndex()"), "Index()");
    return entries;
})();

// Written as the backend writes it, TextEmbedding(model='glove.6B-50d'); it
// also accepts double quotes and the model as the first positional argument.
function normalizeTextEmbeddingArgs(rawArgs: string): string {
    const match = rawArgs.trim().match(/^(?:model\s*=\s*)?(['"])([^'"]*)\1$/);
    return match ? `model='${match[2]}'` : rawArgs;
}

export function normalizeEncoderValue(encoder: string): string {
    const trimmed = encoder.trim();
    if (!trimmed) return "Numerical()";

    const qualifiedConstructorMatch = trimmed.match(
        /^([A-Za-z_][\w.]*)\((.*)\)$/
    );
    if (qualifiedConstructorMatch) {
        const [, rawName, rawArgs] = qualifiedConstructorMatch;
        const baseName = rawName.split(".").pop() ?? rawName;
        const normalizedBaseName =
            ENCODER_VALUE_BY_LOOKUP_KEY.get(
                normalizeEncoderLookupKey(baseName)
            ) ?? baseName;
        const args =
            normalizedBaseName === "TextEmbedding()"
                ? normalizeTextEmbeddingArgs(rawArgs)
                : rawArgs;
        return normalizedBaseName.endsWith("()")
            ? `${normalizedBaseName.slice(0, -2)}(${args})`
            : `${normalizedBaseName}(${args})`;
    }

    const qualifiedNameMatch = trimmed.match(/^([A-Za-z_][\w.]*)$/);
    if (qualifiedNameMatch) {
        const baseName = trimmed.split(".").pop() ?? trimmed;
        const normalizedBaseName =
            ENCODER_VALUE_BY_LOOKUP_KEY.get(
                normalizeEncoderLookupKey(baseName)
            ) ?? trimmed;
        return normalizedBaseName.endsWith("()")
            ? normalizedBaseName
            : `${normalizedBaseName}()`;
    }

    return (
        ENCODER_VALUE_BY_LOOKUP_KEY.get(normalizeEncoderLookupKey(trimmed)) ??
        trimmed
    );
}

export function getEncoderOptionsForStype(
    stype: string | undefined
): (typeof ENCODER_OPTIONS)[number][] {
    if (!stype) return [...ENCODER_OPTIONS];
    return ENCODER_OPTIONS.filter((e) =>
        (e.stypes as readonly string[]).includes(stype)
    );
}

export function getEncoderRules(
    overrides: ColumnProcessingPlanEncoderOverrides,
    naStrategy?: Record<string, unknown> | null
): EncoderRule[] {
    if (!overrides) return [];
    return Object.entries(overrides).map(([key, encoder]) => {
        const dotIdx = key.indexOf(".");
        const table = dotIdx >= 0 ? key.slice(0, dotIdx) : "";
        const column = dotIdx >= 0 ? key.slice(dotIdx + 1) : key;
        const na = naStrategy?.[key];
        return {
            table,
            column,
            encoder:
                typeof encoder === "string"
                    ? normalizeEncoderValue(encoder)
                    : "Numerical()",
            naStrategy: typeof na === "string" ? na : ""
        };
    });
}

function ruleKey(rule: EncoderRule): string {
    const table = rule.table.trim();
    const column = rule.column.trim();
    if (!column) return "";
    return table ? `${table}.${column}` : column;
}

export function buildEncoderOverrides(
    rules: EncoderRule[]
): ColumnProcessingPlanEncoderOverrides {
    const result: Record<string, string> = {};
    for (const rule of rules) {
        const key = ruleKey(rule);
        if (key) result[key] = normalizeEncoderValue(rule.encoder);
    }
    return Object.keys(result).length > 0 ? result : null;
}

export function buildNaStrategy(
    rules: EncoderRule[]
): Record<string, string> | undefined {
    const result: Record<string, string> = {};
    for (const rule of rules) {
        if (rule.encoder.startsWith("Numerical") && rule.naStrategy.trim()) {
            const key = ruleKey(rule);
            if (key) result[key] = rule.naStrategy;
        }
    }
    return Object.keys(result).length > 0 ? result : undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === "object" && value !== null;
}

function parseEncoderRuleValue(
    value: unknown
): { encoder: string; naStrategy: string } | null {
    if (typeof value === "string") {
        return { encoder: value, naStrategy: "" };
    }
    if (!isRecord(value) || typeof value.encoder !== "string") {
        return null;
    }
    return {
        encoder: value.encoder,
        naStrategy:
            typeof value.na_strategy === "string" ? value.na_strategy : ""
    };
}

export function getColumnProcessingFormState(
    data: ModelPlan["column_processing"]
): { state: ColumnProcessingFormState; isSupported: boolean } {
    const record = (data ?? {}) as Record<string, unknown>;
    const naStrategyMap = record.na_strategy as Record<string, unknown> | null;
    const explicitOverrides = getEncoderRules(
        data?.encoder_overrides ?? null,
        naStrategyMap
    );

    const globalDefaults: GlobalEncoderDefault[] = [];
    const scopedRules: ScopedEncoderRule[] = [];
    let isSupported = true;

    const defaults = record.encoder_defaults;
    if (defaults !== undefined) {
        if (!isRecord(defaults)) {
            isSupported = false;
        } else {
            for (const [stype, value] of Object.entries(defaults)) {
                const parsedValue = parseEncoderRuleValue(value);
                if (!parsedValue) {
                    isSupported = false;
                    continue;
                }
                globalDefaults.push({
                    stype,
                    encoder: parsedValue.encoder,
                    naStrategy: parsedValue.naStrategy
                });
            }
        }
    }

    const rules = record.encoder_rules;
    if (rules !== undefined) {
        if (!Array.isArray(rules)) {
            isSupported = false;
        } else {
            for (const rule of rules) {
                if (!isRecord(rule) || !isRecord(rule.match)) {
                    isSupported = false;
                    continue;
                }
                const parsedValue = parseEncoderRuleValue(rule);
                const stype = rule.match.stype;
                if (!parsedValue || typeof stype !== "string") {
                    isSupported = false;
                    continue;
                }
                scopedRules.push({
                    table:
                        typeof rule.match.table === "string"
                            ? rule.match.table
                            : "",
                    columnPattern:
                        typeof rule.match.column_pattern === "string"
                            ? rule.match.column_pattern
                            : "",
                    encoder: parsedValue.encoder,
                    naStrategy: parsedValue.naStrategy
                });
            }
        }
    }

    return {
        state: {
            globalDefaults,
            scopedRules,
            explicitOverrides
        },
        isSupported
    };
}

export function buildColumnProcessingPlan(
    state: ColumnProcessingFormState,
    existingData: ModelPlan["column_processing"]
): ModelPlan["column_processing"] {
    const encoderDefaults = state.globalDefaults.reduce<
        Record<
            string,
            | string
            | {
                  encoder: string;
                  na_strategy?: string;
              }
        >
    >((acc, rule) => {
        const stype = rule.stype.trim();
        if (!stype || !rule.encoder.trim()) return acc;
        acc[stype] = rule.naStrategy.trim()
            ? {
                  encoder: rule.encoder,
                  na_strategy: rule.naStrategy.trim()
              }
            : rule.encoder;
        return acc;
    }, {});

    const encoderRules = state.scopedRules.reduce<
        Array<{
            match: { stype: string; table?: string; column_pattern?: string };
            encoder: string;
            na_strategy?: string;
        }>
    >((acc, rule) => {
        if (!rule.encoder.trim()) return acc;
        const nextRule: {
            match: { stype: string; table?: string; column_pattern?: string };
            encoder: string;
            na_strategy?: string;
        } = {
            match: { stype: "numerical" },
            encoder: rule.encoder
        };
        if (rule.table.trim()) {
            nextRule.match.table = rule.table.trim();
        }
        if (rule.columnPattern.trim()) {
            nextRule.match.column_pattern = rule.columnPattern.trim();
        }
        if (rule.naStrategy.trim()) {
            nextRule.na_strategy = rule.naStrategy.trim();
        }
        acc.push(nextRule);
        return acc;
    }, []);

    const nextPlan = {
        ...(existingData ?? {}),
        encoder_overrides: buildEncoderOverrides(state.explicitOverrides),
        na_strategy: buildNaStrategy(state.explicitOverrides)
    } as Record<string, unknown>;

    if (Object.keys(encoderDefaults).length > 0) {
        nextPlan.encoder_defaults = encoderDefaults;
    } else {
        delete nextPlan.encoder_defaults;
    }

    if (encoderRules.length > 0) {
        nextPlan.encoder_rules = encoderRules;
    } else {
        delete nextPlan.encoder_rules;
    }

    return nextPlan as ModelPlan["column_processing"];
}

// ─── Directed edge helpers ───────────────────────────────────────────────────

export interface DirectedEdge {
    table1: string;
    column1: string;
    table2: string;
    column2: string;
    budget: string;
}

export function getDirectedEdges(
    hopObj: Record<string, unknown> | undefined
): DirectedEdge[] {
    if (!hopObj) return [];
    const edges: DirectedEdge[] = [];
    const hopDefault =
        typeof hopObj.default === "number" ? hopObj.default : undefined;
    for (const [key, value] of Object.entries(hopObj)) {
        if (key === "default") continue;
        const m = key.match(/^(.+)\.(.+)->(.+)\.(.+)$/);
        if (m) {
            const numericValue = typeof value === "number" ? value : undefined;
            edges.push({
                table1: m[1],
                column1: m[2],
                table2: m[3],
                column2: m[4],
                budget:
                    numericValue !== undefined && numericValue !== hopDefault
                        ? String(numericValue)
                        : ""
            });
        }
    }
    return edges;
}

function buildDirectedEdgeEntries(
    edges: DirectedEdge[],
    hopDefault?: number
): Record<string, number | undefined> {
    const result: Record<string, number | undefined> = {};
    for (const e of edges) {
        if (
            e.table1.trim() &&
            e.column1.trim() &&
            e.table2.trim() &&
            e.column2.trim()
        ) {
            const key = `${e.table1}.${e.column1}->${e.table2}.${e.column2}`;
            result[key] = e.budget.trim() ? parseFloat(e.budget) : hopDefault;
        }
    }
    return result;
}
