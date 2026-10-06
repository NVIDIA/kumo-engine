// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DateTimeInput } from "./datetimeinput";

interface DatePickerProps {
    onDateChange: (date: Date) => void;
    format?: string;
    placeholder?: string;
    placeholderDate?: Date | null;
}

function DatePicker({
    onDateChange,
    format = "yyyy-MM-dd HH:mm:ss",
    placeholder = "Select Date",
    placeholderDate = null
}: DatePickerProps) {
    return (
        <DateTimeInput
            placeholder={placeholder}
            defaultDate={placeholderDate ?? undefined}
            onDateTimeChange={onDateChange}
            dateFormat={format}
        />
    );
}

export { DatePicker };
