// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as React from "react";

//import { ChevronDown } from "@/icons";
import { cn } from "@/lib/utils";

import { ChevronDown } from "@/icons/MaterialIcons/ChevronDown";
import { format } from "date-fns";
import { useEffect, useRef, useState } from "react";
import { Calendar } from "./calendar";
import { Popover, PopoverContent, PopoverTrigger } from "./popover";

function DateTimeInput({
    className,
    placeholder,
    dateFormat = "yyyy-MM-dd HH:mm:ss",
    onDateTimeChange,
    defaultDate
}: React.ComponentProps<"input"> & {
    placeholder: string;
    dateFormat?: string;
    onDateTimeChange?: (date: Date) => void;
    defaultDate?: Date;
}) {
    const [date, setDate] = useState<Date | undefined>(defaultDate);
    const [open, setOpen] = useState(false);
    const prevDefaultTimestampRef = useRef<number | undefined>(
        defaultDate?.getTime()
    );

    useEffect(() => {
        const nextTimestamp = defaultDate?.getTime();
        if (prevDefaultTimestampRef.current === nextTimestamp) {
            return;
        }
        prevDefaultTimestampRef.current = nextTimestamp;
        setDate(defaultDate);
    }, [defaultDate]);

    const handleDateChange = (date: Date) => {
        setDate(date);
        if (onDateTimeChange) {
            onDateTimeChange(date);
        }
    };

    return (
        <Popover open={open} onOpenChange={setOpen}>
            <PopoverTrigger asChild>
                <button
                    type="button"
                    role="button"
                    data-slot="datetime-input"
                    className={cn(
                        "w-full items-center justify-between",
                        "rounded-sm! focus-visible:border-ring focus-visible:ring-ring/50 aria-invalid:border-destructive aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 inline-flex cursor-pointer gap-2 whitespace-nowrap font-medium outline-none focus-visible:ring-[3px] disabled:pointer-events-none disabled:opacity-50",
                        "text-sm [&_svg]:size-5 [&_svg]:shrink-0",
                        "shadow-xs [&_svg]:fill-title-secondary border-border-input border bg-white hover:bg-slate-100",
                        "h-8 px-2 px-3 py-1",
                        date ? "text-slate-800" : "text-slate-500"
                    )}
                    data-state="closed">
                    <div className="flex items-center overflow-hidden text-ellipsis">
                        {date
                            ? format(date, dateFormat ?? "yyyy-MM-dd HH:mm:ss")
                            : placeholder}
                    </div>
                    <ChevronDown />
                </button>
            </PopoverTrigger>
            <PopoverContent
                align="start"
                className={cn("w-auto p-0", className)}>
                <Calendar
                    mode="single"
                    selected={date}
                    onSelect={(selectedDate: Date | undefined) => {
                        if (!selectedDate) return;
                        handleDateChange(selectedDate as Date);
                    }}
                    setHMS={handleDateChange}
                    initialFocus
                />
            </PopoverContent>
        </Popover>
    );
}

export { DateTimeInput };
