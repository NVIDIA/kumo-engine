// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";

type SizeType = "xs" | "sm" | "md" | "lg" | "xl" | number;

/**
 * CheckMark or Done component with refined animation using Tailwind CSS v4
 *
 * @param size - Size of the SVG ('xs', 'sm', 'md', 'lg', 'xl' or number in pixels)
 * @param className - Additional CSS classes to apply
 * @returns Animated checkmark SVG with improved proportions and timing
 */
interface AnimatedDoneProps {
    size?: SizeType;
    className?: string;
}

const AnimatedDone: React.FC<AnimatedDoneProps> = ({
    size = "md",
    className = ""
}) => {
    // Convert to pixel value for SVG rendering
    const dimensions = getSizeDimensions(size);

    return (
        <div
            className={`relative inline-block ${className}`}
            style={{ width: dimensions, height: dimensions }}>
            <svg
                version="1.1"
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 130.2 130.2"
                width={dimensions}
                height={dimensions}
                className="text-green-600">
                <style>
                    {`
            @keyframes drawCheckFirst {
              0% {
                stroke-dasharray: 0, 60;
                stroke-dashoffset: 0;
                opacity: 0;
              }
              30% {
                opacity: 1;
              }
              100% {
                stroke-dasharray: 60, 60;
                stroke-dashoffset: 0;
                opacity: 1;
              }
            }
            
            @keyframes drawCheckSecond {
              0% {
                stroke-dasharray: 0, 80;
                stroke-dashoffset: 0;
                opacity: 0;
              }
              30% {
                opacity: 0;
              }
              40% {
                opacity: 1;
              }
              100% {
                stroke-dasharray: 80, 80;
                stroke-dashoffset: 0;
                opacity: 1;
              }
            }
            
            .path-check-first {
              stroke-dasharray: 0, 60;
              stroke-dashoffset: 0;
              animation: drawCheckFirst 0.5s ease-in-out forwards;
            }
            
            .path-check-second {
              stroke-dasharray: 0, 80;
              stroke-dashoffset: 0;
              animation: drawCheckSecond 1s ease-in-out forwards;
            }
          `}
                </style>
                {/* The shorter upward line */}
                <line
                    className="path-check-first"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="10"
                    strokeLinecap="round"
                    strokeMiterlimit="10"
                    x1="30"
                    y1="70"
                    x2="50"
                    y2="90"
                />
                {/* The longer downward line */}
                <line
                    className="path-check-second"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="10"
                    strokeLinecap="round"
                    strokeMiterlimit="10"
                    x1="50"
                    y1="90"
                    x2="100"
                    y2="40"
                />
            </svg>
        </div>
    );
};

/**
 * Helper function to convert SizeType to pixel dimensions
 */
function getSizeDimensions(size: SizeType): number {
    if (typeof size === "number") {
        return size;
    }

    // Map string sizes to pixel values
    switch (size) {
        case "xs":
            return 16;
        case "sm":
            return 24;
        case "md":
            return 32;
        case "lg":
            return 40;
        case "xl":
            return 48;
        default:
            return 32;
    }
}

export default AnimatedDone;
