// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { cn } from "@/lib/utils";
import { Portal as RadixPortal } from "@radix-ui/react-portal";
import React from "react";

const Portal = RadixPortal;

interface BouncingballProps {
    overlay?: boolean;
    style?: React.CSSProperties;
    className?: string;
}

/**
 * Kumo full-page loader — the NVIDIA brand "mark" spinner: interlocking arrow
 * triangles (rows of 3/5/7/7) that pulse in a green wave.
 *
 * This is OUR OWN implementation — our component, class names (kumo-loader*),
 * keyframe, and color token (--brand-green). It reproduces the NVIDIA loader
 * SHAPE (the visual brand pattern) but does NOT use any KUI source code, KUI
 * class names, KUI CSS variables, or the KUI package.
 */
const UP = "17 0, 34 31, 0 31";
const DOWN = "0 0, 34 0, 17 31";

type Arrow = { dir: "up" | "down"; x: number; y: number; delay: string };
const ROWS: { tx: number; ty: number; arrows: Arrow[] }[] = [
    {
        tx: 36,
        ty: 0,
        arrows: [
            { dir: "up", x: 0, y: 1, delay: "-1s" },
            { dir: "down", x: 18, y: 0, delay: "-.9444s" },
            { dir: "up", x: 36, y: 1, delay: "-.8888s" }
        ]
    },
    {
        tx: 18,
        ty: 32,
        arrows: [
            { dir: "up", x: 0, y: 1, delay: "-1s" },
            { dir: "down", x: 18, y: 0, delay: "-.9444s" },
            { dir: "up", x: 36, y: 1, delay: "-.8888s" },
            { dir: "down", x: 54, y: 0, delay: "-.8333s" },
            { dir: "up", x: 72, y: 1, delay: "-.7777s" }
        ]
    },
    {
        tx: 0,
        ty: 64,
        arrows: [
            { dir: "up", x: 0, y: 1, delay: "0s" },
            { dir: "down", x: 18, y: 0, delay: "-.1111s" },
            { dir: "up", x: 36, y: 1, delay: "-1s" },
            { dir: "down", x: 54, y: 0, delay: "-.8333s" },
            { dir: "up", x: 72, y: 1, delay: "-.6666s" },
            { dir: "down", x: 90, y: 0, delay: "-.7222s" },
            { dir: "up", x: 108, y: 1, delay: "-.6666s" }
        ]
    },
    {
        tx: 0,
        ty: 96,
        arrows: [
            { dir: "down", x: 0, y: 0, delay: "-.1666s" },
            { dir: "up", x: 18, y: 1, delay: "-.2222s" },
            { dir: "down", x: 36, y: 0, delay: "-.1666s" },
            { dir: "up", x: 54, y: 1, delay: "-.3333s" },
            { dir: "down", x: 72, y: 0, delay: "-.5s" },
            { dir: "up", x: 90, y: 1, delay: "-.6111s" },
            { dir: "down", x: 108, y: 0, delay: "-.5555s" }
        ]
    }
];

const TriangleLoader: React.FC<{ className?: string }> = ({ className }) => (
    <div
        className={cn("kumo-loader", className)}
        role="status"
        aria-label="Loading">
        <svg
            className="kumo-loader-canvas"
            viewBox="0 0 142 128"
            width="100%"
            height="100%">
            {ROWS.map((row, ri) => (
                <g key={ri} transform={`translate(${row.tx}, ${row.ty})`}>
                    {row.arrows.map((a, ai) => (
                        <polygon
                            key={ai}
                            className="kumo-loader-seg"
                            points={a.dir === "up" ? UP : DOWN}
                            transform={`translate(${a.x}, ${a.y})`}
                            style={{ animationDelay: a.delay }}
                        />
                    ))}
                </g>
            ))}
        </svg>
        <style>{`
            @keyframes kumoLoaderPulse { 0% { opacity: 0.1 } 30% { opacity: 1 } 100% { opacity: 0.1 } }
            .kumo-loader {
                display: flex;
                align-items: center;
                justify-content: center;
                filter: drop-shadow(0 0 8px color-mix(in srgb, var(--brand-green) 45%, transparent));
            }
            .kumo-loader-canvas { width: auto; height: 64px; }
            .kumo-loader-seg { fill: var(--brand-green); animation: kumoLoaderPulse 1s infinite; }
        `}</style>
    </div>
);

export const Bouncingball: React.FC<BouncingballProps> = ({
    overlay = false,
    style = {},
    className
}) => {
    if (overlay) {
        return (
            <Portal>
                <div
                    data-testid="animated-loading"
                    className={cn(
                        "fixed inset-0 z-[9999] flex items-center justify-center bg-black/50",
                        className
                    )}>
                    <TriangleLoader />
                </div>
            </Portal>
        );
    }

    return (
        <div
            data-testid="bouncingball-loading"
            style={style}
            className={cn("flex items-center justify-center", className)}>
            <TriangleLoader />
        </div>
    );
};
