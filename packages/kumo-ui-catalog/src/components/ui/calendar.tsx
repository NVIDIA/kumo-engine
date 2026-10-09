// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2023 shadcn
// SPDX-License-Identifier: Apache-2.0 AND MIT

import * as React from "react";
import { DayPicker } from "react-day-picker";

import { Button, buttonVariants } from "@/components/ui/button";
import { ChevronLeft } from "@/icons/MaterialIcons/ChevronLeft";
import { ChevronRight } from "@/icons/MaterialIcons/ChevronRight";
import { cn } from "@/lib/utils";
import { addMonths, Day, set, startOfMonth } from "date-fns";
import { enUS } from "date-fns/locale";
import { useEffect, useRef, useState } from "react";
import { Input } from "./input";
import { Separator } from "./separator";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue
} from "./singleselect";

const months = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December"
];

const years = Array.from({ length: 101 }, (_, i) => 2000 + i); // 2000 to 2100

function CustomCaption({
    displayMonth,
    onMonthChange
}: {
    displayMonth: Date;
    onMonthChange: (newDate: Date) => void;
}) {
    const month = displayMonth.getMonth();
    const year = displayMonth.getFullYear();

    const handleMonthChange = (value: string) => {
        onMonthChange(startOfMonth(new Date(year, parseInt(value))));
    };

    const handleYearChange = (value: string) => {
        onMonthChange(startOfMonth(new Date(parseInt(value), month)));
    };

    const goToPrevMonth = () => {
        onMonthChange(startOfMonth(addMonths(displayMonth, -1)));
    };

    const goToNextMonth = () => {
        onMonthChange(startOfMonth(addMonths(displayMonth, 1)));
    };

    return (
        <div className="mt-2 flex w-full gap-1">
            <button
                type="button"
                aria-label="Go to previous month"
                className="ml-0 cursor-pointer"
                onClick={goToPrevMonth}>
                <ChevronLeft className="size-7" />
            </button>
            <Select value={month.toString()} onValueChange={handleMonthChange}>
                <SelectTrigger
                    size="sm"
                    className="w-32"
                    aria-label="Select month">
                    <SelectValue placeholder="Month" />
                </SelectTrigger>
                <SelectContent>
                    {months.map((label, idx) => (
                        <SelectItem key={idx} value={idx.toString()}>
                            {label}
                        </SelectItem>
                    ))}
                </SelectContent>
            </Select>

            <Select value={year.toString()} onValueChange={handleYearChange}>
                <SelectTrigger
                    size="sm"
                    className="w-22"
                    aria-label="Select year">
                    <SelectValue placeholder="Year" />
                </SelectTrigger>
                <SelectContent>
                    {years.map((y) => (
                        <SelectItem key={y} value={y.toString()}>
                            {y}
                        </SelectItem>
                    ))}
                </SelectContent>
            </Select>
            <button
                type="button"
                aria-label="Go to next month"
                className="cursor-pointer"
                onClick={goToNextMonth}>
                <ChevronRight className="size-7" />
            </button>
        </div>
    );
}

function Calendar({
    setHMS, //Use this if hh:mm:ss inputs are required.
    className,
    classNames,
    showOutsideDays = true,
    ...props
}: React.ComponentProps<typeof DayPicker> & {
    setHMS?: (date: Date) => void;
}) {
    const selectedDate = props?.selected as Date | undefined;
    const selectedTimestamp = selectedDate?.getTime();
    const skipNextEmitRef = useRef(false);
    const [ampm, setAmPm] = useState<"AM" | "PM">(
        selectedDate && selectedDate.getHours() >= 12 ? "PM" : "AM"
    );
    const hrs = selectedDate
        ? (() => {
              const hour = selectedDate.getHours() % 12;
              return hour === 0 ? 12 : hour;
          })()
        : null;
    const [hours, setHoursState] = useState<number>(hrs ?? 12);
    const [minutes, setMinutesState] = useState<number>(
        selectedDate ? selectedDate.getMinutes() : 0
    );
    const [seconds, setSecondsState] = useState<number>(
        selectedDate ? selectedDate.getSeconds() : 0
    );
    const [month, setMonth] = useState<Date>(
        selectedDate || startOfMonth(new Date())
    );

    useEffect(() => {
        if (!selectedDate) return;
        skipNextEmitRef.current = true;

        const nextAmpm = selectedDate.getHours() >= 12 ? "PM" : "AM";
        const nextHour = selectedDate.getHours() % 12 || 12;
        const nextMinutes = selectedDate.getMinutes();
        const nextSeconds = selectedDate.getSeconds();
        const nextMonth = startOfMonth(selectedDate);

        setAmPm((prev) => (prev === nextAmpm ? prev : nextAmpm));
        setHoursState((prev) => (prev === nextHour ? prev : nextHour));
        setMinutesState((prev) => (prev === nextMinutes ? prev : nextMinutes));
        setSecondsState((prev) => (prev === nextSeconds ? prev : nextSeconds));
        setMonth((prev) =>
            prev.getTime() === nextMonth.getTime() ? prev : nextMonth
        );
    }, [selectedDate, selectedTimestamp]);

    // Update time in date when any field changes
    useEffect(() => {
        if (!setHMS || !selectedDate) return;
        if (skipNextEmitRef.current) {
            skipNextEmitRef.current = false;
            return;
        }

        const hour24: number = ampm === "AM" ? hours % 12 : (hours % 12) + 12;
        const updated = set(selectedDate, {
            year: month.getFullYear(),
            month: month.getMonth(),
            hours: hour24,
            minutes,
            seconds
        });
        // Only update if something actually changed
        if (selectedDate.getTime() !== updated.getTime()) {
            setHMS(updated);
        }
    }, [
        hours,
        minutes,
        seconds,
        ampm,
        month,
        setHMS,
        selectedDate,
        selectedTimestamp
    ]);

    const handleNumberInput = (
        val: string,
        max: number,
        setFunc: (v: number) => void
    ) => {
        let n = 0;
        const intVal = parseInt(val);
        if (intVal) {
            n = intVal;
            if (intVal > max) {
                n = 0;
            } else if (intVal < 0) {
                n = max;
            }
        }
        if (!isNaN(n)) setFunc(n);
    };

    const renderNumberInput = (
        value: number,
        setFunc: (v: number) => void,
        max: number,
        label: string,
        pad: boolean = true
    ) => (
        <Input
            type="number"
            aria-label={label}
            value={pad ? String(value).padStart(2, "0") : value}
            onChange={(e) => handleNumberInput(e.target.value, max, setFunc)}
            className="mx-0 h-8 w-14 rounded-sm border pl-0 pr-0 text-center"
        />
    );

    const customLocale: typeof enUS = {
        ...enUS,
        localize: {
            ...enUS.localize!,
            day: (n: Day) => enUS.localize!.day(n, { width: "abbreviated" }) // 'Sun', 'Mon', etc.
        }
    };
    return (
        <>
            <div className="w-full p-3">
                <DayPicker
                    locale={customLocale}
                    showOutsideDays={showOutsideDays}
                    className={cn("p-0", className)}
                    month={month}
                    onMonthChange={setMonth}
                    classNames={{
                        months: "", //"flex flex-col sm:flex-row gap-2",
                        month: "flex flex-col gap-4",
                        caption: "",
                        //"flex justify-center pt-1 relative items-center w-full",
                        caption_label: "", //"text-sm font-medium",
                        /* nav: "lex items-center gap-1",
                    nav_button: cn(
                        buttonVariants({ variant: "outline" }),
                        "size-7 bg-transparent p-0 opacity-50 hover:opacity-100"
                    ),
                    nav_button_previous:
                        "absolute left-1 inset-y-[2px] ",
                    nav_button_next:
                        "absolute right-1 inset-y-[2px] ", */
                        table: "w-full border-collapse",
                        head_row: "flex w-full mt-1 gap-2",
                        head_cell:
                            "text-slate-800 rounded-md w-8.5 font-semibold text-[0.8rem]",
                        row: "flex w-full mt-2 gap-2",
                        cell: cn(),
                        //"relative p-0 text-center text-sm focus-within:relative focus-within:z-20 [&:has([aria-selected])]:bg-accent [&:has([aria-selected].day-range-end)]:rounded-r-md"
                        /* props.mode === "range"
                        ? "[&:has(=.day-range-end)]:rounded-r-md [&:has(<.day-range-start)]:rounded-l-md first:[&:has([aria-selected])]:rounded-l-md last:[&:has([aria-selected])]:rounded-r-md"
                        : "[&:has([aria-selected])]:rounded-md" */

                        day: cn(
                            buttonVariants({ variant: "light-outline" }),
                            "size-8 p-0 font-normal shadow-none bg-transparent border-transparent focus:bg-slate-100 text-slate-800"
                        ),
                        day_range_start: "", //"day-range-start aria-selected:bg-primary aria-selected:text-primary-foreground",
                        day_range_end: "", //"day-range-end aria-selected:bg-primary aria-selected:text-primary-foreground",
                        day_selected: "", //"bg-primary text-primary-foreground hover:bg-primary hover:text-primary-foreground focus:bg-primary focus:text-primary-foreground",
                        day_today: "!border-slate-500",
                        day_outside:
                            "!text-slate-500 !aria-selected:text-slate-500",
                        day_disabled: "", //"text-muted-foreground opacity-50",
                        day_range_middle: "aria-selected:text-slate-800",
                        day_hidden: "invisible",
                        ...classNames
                    }}
                    components={{
                        IconLeft: () => <ChevronLeft className="size-4" />,
                        IconRight: () => <ChevronRight className="size-4" />,
                        Caption: (props) => (
                            <CustomCaption
                                displayMonth={props.displayMonth}
                                onMonthChange={setMonth}
                            />
                        )
                    }}
                    {...props}
                />
            </div>
            {setHMS && (
                <div className="pb-2">
                    <div className="">
                        <Separator className="ml-0 flex w-full" />
                    </div>

                    <div className="flex items-center justify-center gap-2 pb-2 pt-2">
                        {renderNumberInput(hours, setHoursState, 12, "Hours")}
                        <span aria-hidden="true">:</span>
                        {renderNumberInput(
                            minutes,
                            setMinutesState,
                            59,
                            "Minutes"
                        )}
                        <span aria-hidden="true">:</span>
                        {renderNumberInput(
                            seconds,
                            setSecondsState,
                            59,
                            "Seconds"
                        )}

                        <Button
                            onClick={() => {
                                setAmPm(ampm === "AM" ? "PM" : "AM");
                            }}
                            className="!text-slate-800"
                            variant="light-outline"
                            size="sm">
                            {ampm}
                        </Button>
                    </div>
                </div>
            )}
        </>
    );
}

export { Calendar };
