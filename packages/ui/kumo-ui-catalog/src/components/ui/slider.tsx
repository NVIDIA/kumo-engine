// SPDX-FileCopyrightText: Copyright (c) 2023 shadcn
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0 AND MIT

import { DragIndicator } from "@/icons";
import { cn } from "@/lib/utils";
import * as SliderPrimitive from "@radix-ui/react-slider";
import * as React from "react";

function Slider({
    className,
    defaultValue = [0.5],
    value,
    min = 0,
    max = 1,
    step = 0.01,
    ...props
}: React.ComponentProps<typeof SliderPrimitive.Root>) {
    const [displayValue, setDisplayValue] = React.useState(
        Array.isArray(value)
            ? value[0]
            : Array.isArray(defaultValue)
              ? defaultValue[0]
              : 0.5
    );

    React.useEffect(() => {
        if (Array.isArray(value)) {
            setDisplayValue(value[0]);
        }
    }, [value]);

    const handleValueChange = React.useCallback(
        (newValue: number[]) => {
            setDisplayValue(newValue[0]);
            if (props.onValueChange) {
                props.onValueChange(newValue);
            }
        },
        [props]
    );

    return (
        <div className="relative w-full">
            {/* Value display box */}
            <div
                className="!rounded-xs absolute flex h-6 w-14 items-center justify-center overflow-hidden border border-gray-400 bg-gray-50"
                style={{
                    left: `calc(${((displayValue - min) / (max - min)) * 100}% - 32px)`,
                    top: "-16px"
                }}>
                <span className="justify-start text-center text-sm font-semibold leading-tight text-gray-800">
                    {displayValue}
                </span>
            </div>

            <div className="relative h-10">
                {/* The actual slider track */}
                {
                    <div className="absolute left-0 right-0 top-6 h-0.5 bg-gray-200"></div>
                }

                {/* Start vertical indicator */}
                <div className="absolute left-0 top-6 h-3 w-[2.63px] bg-gray-200"></div>

                {/* End vertical indicator */}
                <div className="absolute right-0 top-6 h-3 w-[2.63px] bg-gray-200"></div>

                {/* The slider component */}
                <SliderPrimitive.Root
                    data-slot="slider"
                    defaultValue={defaultValue}
                    value={value}
                    min={min}
                    max={max}
                    step={step}
                    onValueChange={handleValueChange}
                    className={cn(
                        "top-6.5 absolute left-0 right-0 flex touch-none select-none items-center",
                        className
                    )}
                    {...props}>
                    <SliderPrimitive.Track
                        data-slot="slider-track"
                        className="relative h-px grow bg-transparent">
                        <SliderPrimitive.Range
                            data-slot="slider-range"
                            className="absolute h-full bg-transparent"
                        />
                    </SliderPrimitive.Track>

                    <SliderPrimitive.Thumb
                        data-slot="slider-thumb"
                        className="!rounded-xs flex h-6 w-4 -translate-y-px items-center justify-center bg-gray-400"
                        aria-label="Slider thumb">
                        <DragIndicator className="size-4.5" />
                    </SliderPrimitive.Thumb>
                </SliderPrimitive.Root>

                {/* Min value */}
                <div className="absolute left-0 top-10 text-sm font-semibold leading-tight text-gray-800">
                    <span>{min}</span>
                </div>

                {/* Max value */}
                <div className="absolute right-0 top-10 text-sm font-semibold leading-tight text-gray-800">
                    <span>{max}</span>
                </div>
            </div>
        </div>
    );
}

export { Slider };
