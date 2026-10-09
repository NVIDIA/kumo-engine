// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Renders the 5 TLP node types (circle / diamond / triangle, ± emerald GT ring).
// Diamond/triangle use SVG polygons rather than CSS rotate to avoid sqrt(2)x
// box expansion that can clip at the parent boundary.

import {
    GT_RING_HALO,
    type NodeShapeVariant
} from "@/utils/xai-subgraph-shapes";

interface CommonProps {
    /** Inscribed-box dimension of the inner shape (px). */
    size: number;
    fillColor: string;
    /** Emerald GT ring overlay (+2px halo each side). */
    hasRing?: boolean;
    isHovered?: boolean;
    isSelected?: boolean;
    opacity?: number;
}

// Hover: outline (single color). Selected: box-shadow (stacks white spacer + pink).
function ringStyle(
    isHovered?: boolean,
    isSelected?: boolean
): { outline?: string; boxShadow?: string } {
    if (isSelected) {
        return {
            boxShadow: "0 0 0 2px white, 0 0 0 3px var(--color-pink-500)"
        };
    }
    if (isHovered) {
        return { outline: "1px solid var(--color-pink-500)" };
    }
    return {};
}

function CircleShape({
    size,
    fillColor,
    hasRing,
    isHovered,
    isSelected,
    opacity = 1
}: CommonProps) {
    const halo = hasRing ? GT_RING_HALO : 0;
    const outer = size + halo * 2;
    const ring = ringStyle(isHovered, isSelected);
    if (!hasRing) {
        return (
            <div
                style={{
                    width: outer,
                    height: outer,
                    backgroundColor: fillColor,
                    borderRadius: "50%",
                    opacity,
                    pointerEvents: "none",
                    ...ring
                }}
            />
        );
    }
    const innerSize = outer - halo * 2;
    // Wrapper carries the hover/selected ring OUTSIDE the emerald GT ring.
    return (
        <div
            style={{
                width: outer,
                height: outer,
                position: "relative",
                borderRadius: "50%",
                opacity,
                pointerEvents: "none",
                ...ring
            }}>
            <div
                style={{
                    position: "absolute",
                    top: 0,
                    left: 0,
                    width: outer,
                    height: outer,
                    borderRadius: "50%",
                    border: "1px solid var(--color-emerald-600)",
                    boxSizing: "border-box"
                }}
            />
            <div
                style={{
                    position: "absolute",
                    top: halo,
                    left: halo,
                    width: innerSize,
                    height: innerSize,
                    borderRadius: "50%",
                    backgroundColor: fillColor
                }}
            />
        </div>
    );
}

// Diamond/triangle SVG renderer. Outer rings are PARALLEL OFFSETS (uniform
// perpendicular distance) so gaps look consistent on every side. Offset
// formulas: diamond uses sqrt(2)*offset extension; triangle (apex-up, slanted
// sides slope ±2) uses sqrt(5) on the apex and (1+sqrt(5))/2 on the base sides.
function SvgPolygonShape({
    variant,
    size,
    fillColor,
    hasRing,
    isHovered,
    isSelected,
    opacity = 1
}: CommonProps & { variant: "diamond" | "triangle" }) {
    const halo = hasRing ? GT_RING_HALO : 0;
    const visible = size + halo * 2;
    const ringPad = 6;
    const total = visible + ringPad * 2;
    const cx = total / 2;
    const cy = total / 2;
    const innerR = size / 2;
    // Selected ring sits at the same distance from the inner shape as the GT
    // ring (so non-GT selected matches hit's outer dimensions per Figma).
    const gtOffset = halo - 0.5;
    const outerEdgeOffset = halo;
    const hoverOffset = outerEdgeOffset + 0.5;
    const selectedWhiteOffset = outerEdgeOffset + 0.5;
    const selectedPinkOffset = outerEdgeOffset + 1.5;
    const pts = (offset: number) => {
        if (variant === "diamond") {
            const r = innerR + offset * Math.SQRT2;
            return `${cx},${cy - r} ${cx + r},${cy} ${cx},${cy + r} ${cx - r},${cy}`;
        }
        const sqrt5 = Math.sqrt(5);
        const ay = cy - innerR - offset * sqrt5;
        const bx = cx - innerR - (offset / 2) * (1 + sqrt5);
        const cxRight = cx + innerR + (offset / 2) * (1 + sqrt5);
        const by = cy + innerR + offset;
        return `${cx},${ay} ${bx},${by} ${cxRight},${by}`;
    };
    return (
        <div
            style={{
                width: visible,
                height: visible,
                position: "relative",
                opacity,
                pointerEvents: "none"
            }}>
            <svg
                width={total}
                height={total}
                viewBox={`0 0 ${total} ${total}`}
                style={{
                    position: "absolute",
                    top: -ringPad,
                    left: -ringPad,
                    display: "block"
                }}
                aria-hidden="true">
                <polygon points={pts(0)} fill={fillColor} />
                {hasRing && (
                    <polygon
                        points={pts(gtOffset)}
                        fill="none"
                        stroke="var(--color-emerald-600)"
                    />
                )}
                {isHovered && !isSelected && (
                    <polygon
                        points={pts(hoverOffset)}
                        fill="none"
                        stroke="var(--color-pink-500)"
                    />
                )}
                {isSelected && (
                    <>
                        {/* 1px stroke (not 2) so it doesn't bleed into the inner shape. */}
                        <polygon
                            points={pts(selectedWhiteOffset)}
                            fill="none"
                            stroke="white"
                        />
                        <polygon
                            points={pts(selectedPinkOffset)}
                            fill="none"
                            stroke="var(--color-pink-500)"
                        />
                    </>
                )}
            </svg>
        </div>
    );
}

export function NodeShape({
    variant,
    ...rest
}: CommonProps & { variant: NodeShapeVariant }) {
    if (variant === "circle") {
        return <CircleShape {...rest} />;
    }
    return <SvgPolygonShape variant={variant} {...rest} />;
}
