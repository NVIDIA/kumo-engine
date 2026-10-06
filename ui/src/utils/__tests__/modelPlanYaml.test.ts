// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelPlan } from "gen/v1/types";

import {
    buildColumnProcessingPlan,
    getActiveHops,
    getColumnProcessingFormState,
    normalizeEncoderValue,
    parseSplitString,
    setHopValue
} from "utils/modelPlanYaml";

describe("modelPlanYaml column processing helpers", () => {
    it("round-trips global defaults, scoped rules, and explicit overrides", () => {
        const columnProcessing = buildColumnProcessingPlan(
            {
                globalDefaults: [
                    {
                        stype: "numerical",
                        encoder: "Numerical()",
                        naStrategy: "separate"
                    }
                ],
                scopedRules: [
                    {
                        table: "facts",
                        columnPattern: "^revenue_.*",
                        encoder: "MaxLogNumerical()",
                        naStrategy: "mean"
                    }
                ],
                explicitOverrides: [
                    {
                        table: "companies",
                        column: "age",
                        encoder: "Numerical()",
                        naStrategy: "zero"
                    }
                ]
            },
            undefined
        );

        const parsed = getColumnProcessingFormState(
            columnProcessing as ModelPlan["column_processing"]
        );

        expect(parsed.isSupported).toBe(true);
        expect(parsed.state.globalDefaults).toEqual([
            {
                stype: "numerical",
                encoder: "Numerical()",
                naStrategy: "separate"
            }
        ]);
        expect(parsed.state.scopedRules).toEqual([
            {
                table: "facts",
                columnPattern: "^revenue_.*",
                encoder: "MaxLogNumerical()",
                naStrategy: "mean"
            }
        ]);
        expect(parsed.state.explicitOverrides).toEqual([
            {
                table: "companies",
                column: "age",
                encoder: "Numerical()",
                naStrategy: "zero"
            }
        ]);
    });

    it("marks unsupported encoder_rules shapes as YAML-only", () => {
        const parsed = getColumnProcessingFormState({
            encoder_overrides: null,
            encoder_rules: { invalid: true }
        } as unknown as ModelPlan["column_processing"]);

        expect(parsed.isSupported).toBe(false);
    });

    it("normalizes legacy encoder names to valid constructor syntax", () => {
        expect(normalizeEncoderValue("numerical")).toBe("Numerical()");
        expect(normalizeEncoderValue("Hash")).toBe("Hash()");
        expect(normalizeEncoderValue("Index")).toBe("Index()");
        expect(normalizeEncoderValue("Kumo Index")).toBe("Index()");
        expect(normalizeEncoderValue("KumoIndex()")).toBe("Index()");
        expect(normalizeEncoderValue("kumoapi.encoder.index.KumoIndex()")).toBe(
            "Index()"
        );
        expect(
            normalizeEncoderValue(
                "kumoapi.encoder.index.KumoIndex(min_occ=10, na_strategy=separate)"
            )
        ).toBe("Index(min_occ=10, na_strategy=separate)");
        expect(normalizeEncoderValue("Index(min_occ=10)")).toBe(
            "Index(min_occ=10)"
        );
    });

    it("writes text embedding models as the backend does", () => {
        for (const encoder of [
            "TextEmbedding(model='glove.6B-100d')",
            'TextEmbedding(model="glove.6B-100d")',
            "TextEmbedding('glove.6B-100d')",
            'kumoapi.encoder.TextEmbedding("glove.6B-100d")'
        ]) {
            expect(normalizeEncoderValue(encoder)).toBe(
                "TextEmbedding(model='glove.6B-100d')"
            );
        }
        expect(normalizeEncoderValue("TextEmbedding()")).toBe(
            "TextEmbedding()"
        );
    });

    it("hydrates Temporal Split defaults from generated TimeSplit fractions YAML", () => {
        expect(
            parseSplitString(
                "TimeSplit(timestamp_boundaries=None, offset_boundaries=None, fractions=[0.8, 0.1, 0.1], indices=None, strict_split_boundaries=True)"
            )
        ).toMatchObject({
            type: "TemporalSplit",
            train: 0.8,
            val: 0.1,
            test: 0.1
        });
    });

    it("hydrates Time Range Split defaults from YAML ranges", () => {
        expect(
            parseSplitString(
                "TimeRangeSplit([('2024-01-01 20:00:00', '2024-10-12 20:00:00'), ('2024-10-12 20:00:00', '2024-10-19 20:00:00')], unit='days', timeframe=7, anchor_start_offset=1)"
            )
        ).toMatchObject({
            type: "TimeRangeSplit",
            trainStart: "2024-01-01 20:00:00",
            trainEnd: "2024-10-12 20:00:00",
            valStart: "2024-10-12 20:00:00",
            valEnd: "2024-10-19 20:00:00",
            unit: "days",
            timeframe: "7",
            anchorStartOffset: "1"
        });
    });

    it("hydrates Date Offset Split defaults from generated TimeSplit offset boundaries YAML", () => {
        expect(
            parseSplitString(
                "TimeSplit(timestamp_boundaries=None, offset_boundaries=[['-INF', -14], [-14, -7], [-7, 0]], fractions=None, indices=None, strict_split_boundaries=True)"
            )
        ).toMatchObject({
            type: "DateOffsetSplit",
            offsets: "-14, -7"
        });
    });
});

describe("getActiveHops", () => {
    it("returns empty for undefined", () => {
        expect(getActiveHops(undefined)).toEqual([]);
    });

    it("modern object format", () => {
        const numNeighbors = [{ hop1: { default: 16 }, hop2: { default: 8 } }];
        expect(getActiveHops(numNeighbors as never)).toEqual([
            { hopKey: "hop1", budget: 16 },
            { hopKey: "hop2", budget: 8 }
        ]);
    });

    it("legacy array-of-arrays format", () => {
        expect(getActiveHops([[16, 8, 4]] as never)).toEqual([
            { hopKey: "hop1", budget: 16 },
            { hopKey: "hop2", budget: 8 },
            { hopKey: "hop3", budget: 4 }
        ]);
    });

    it("legacy format with single hop", () => {
        expect(getActiveHops([[16]] as never)).toEqual([
            { hopKey: "hop1", budget: 16 }
        ]);
    });
});

describe("setHopValue with legacy format", () => {
    it("converts legacy array to modern format, preserving all hop budgets", () => {
        const result = setHopValue([[16, 8, 4]] as never, "hop2", 32);
        expect(result).toEqual([
            {
                hop1: { default: 16 },
                hop2: { default: 32 },
                hop3: { default: 4 }
            }
        ]);
    });

    it("deletes a hop from legacy format, preserving remaining budgets", () => {
        const result = setHopValue([[16, 8, 4]] as never, "hop2", null);
        expect(result).toEqual([
            { hop1: { default: 16 }, hop3: { default: 4 } }
        ]);
    });

    it("modern format edit is unchanged", () => {
        const numNeighbors = [{ hop1: { default: 16 }, hop2: { default: 8 } }];
        const result = setHopValue(numNeighbors as never, "hop1", 32);
        expect(result).toEqual([
            { hop1: { default: 32 }, hop2: { default: 8 } }
        ]);
    });
});
