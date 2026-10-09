# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import os.path as osp
from pathlib import Path
from test.fixtures.backend_cache import (  # noqa: F401
    auto_setup_kumo_backend,
    backend_cache,
)
from test.fixtures.local import tempdir, valid_fake_private_key  # noqa: F401
from typing import Callable, Dict, List, Optional

import pytest
from kumoapi.typing import Dtype, Stype


@pytest.fixture
def csv_edge_case_data() -> Callable:
    def _csv_edge_case_data(timestamps: bool = False) -> Dict[str, List[str]]:
        if not timestamps:
            return {
                "01_missing_and_empty_fields": [
                    [
                        "transaction_id", "account_number", "amount",
                        "description", "date"
                    ],
                    [1, "ACC12345", 1000.50, "Salary Payment", "2025-05-01"],
                    [2, "ACC67890", "", "Monthly Subscription", ""],
                    [3, "ACC99999", -250.75, "Refund", ""],
                    ["", "", "", "", ""],
                    [11, "ACC66666", 75.00],  # Inconsistent columns (missing)
                    [
                        12, "ACC77777", 88.00, "Unexpected", "2025-05-22",
                        "Extra column"
                    ],
                ],
                "02_formatting_and_special_characters": [
                    [
                        "transaction_id", "account_number    ", "amount",
                        "description", "date"
                    ],
                    [
                        4, "ACC33333", 500.00, "Reimbursement, travel",
                        "2025-03-12"
                    ],  # Comma in field
                    [
                        5, "ACC77777", 200.00, "Multi-line\nexpense note",
                        "2025-05-18"
                    ],  # Newline
                    [
                        6, "ACC88888", 150.00,
                        'Client said, "Thanks for the fast processing"',
                        "2025-05-19"
                    ],  # Quotes
                    [
                        7, "ACC99999", 0.00, "Cleared ✅ and ready &",
                        "2025-05-10"
                    ],  # Emojis/special chars
                    [
                        8, " ACC11111 ", " 300.00 ", "   Bonus Payment   ",
                        " 2025-01-01 "
                    ],  # Whitespace
                    # additional quoting cases, and mixed nan representation:
                    [
                        9, "ACC88888", 150.00,
                        "Client said, 'Thanks for the fast processing'",
                        "2025-05-19"
                    ],
                    [
                        10, "ACC88888", 150.00, "Client said, \"Thanks...",
                        "NULL"
                    ],
                    [11, "ACC88888", 150.00, "Client said, 'Thanks...", 'NaN'],
                ],
                "03_complex_and_invalid_structure": [
                    [
                        "transaction_id", "account_number", "amount",
                        "description", "date"
                    ],
                    [
                        9, "ACC44444", 9999999.99, "Lorem ipsum " * 50,
                        "2025-05-20"
                    ],  # Very long field
                    [10, "ACC55555", 42.00, "Valid transaction", "2025-05-21"],
                    [13, "ACC12121", 300.00, "Valid", "2025-05-18"],
                    [14, "ACC12122", "NaN", "Unknown", "NULL"],
                    [15, "ACC12123", -9999, "Anomaly", "not-a-date"],
                    [16, "ACC00000", -150.00, "<error>",
                     "??"],  # Malformed data
                ],
                "04_column_duplication": [
                    [
                        "transaction_id", "account_number", "account_number",
                        "amount", "description", "date"
                    ],  # Duplicate column
                    [
                        17, "ACC10101", "ACC10101", 123.45, "Double header",
                        "2025-05-01"
                    ],
                    [
                        17, "ACC10101", "ACC10101", 123.45, "Double header",
                        "2025-05-01"
                    ],
                ],
                "05_delimiter_mixes": [
                    [
                        "transaction_id", "account_number", "amount",
                        "description", "date"
                    ],
                    ["transaction_id;account_number;amount;description;date"
                     ],  # results in duplicate header, mix of delimiters
                    ["18;ACC20202;500.00;Payment with semicolon;2025-05-03"],
                    [
                        "18", "ACC20202", "500.00", "Payment with semicolon",
                        "2025-05-03"
                    ],
                ]
            }

        return {
            "01_standard_iso8601": [
                ["transaction_id", "account_number", "amount", "timestamp"],
                [1, "ACC001", 150.00, "2025-05-20T14:30:00"],
                [2, "ACC002", 200.00, "2025-05-21T09:00:00"],
            ],
            "02_date_only": [
                ["transaction_id", "account_number", "amount", "timestamp"],
                [3, "ACC003", 300.00, "2025-05-20"],
                [4, "ACC004", 120.00, "2025-05-21"],
            ],
            "03_datetime_with_timezone": [
                ["transaction_id", "account_number", "amount", "timestamp"],
                [5, "ACC005", 400.00, "2025-05-20T14:30:00Z"],
                [6, "ACC006", 500.00, "2025-05-21T09:00:00+05:00"],
            ],
            "03_datetime_with_timezone_utc": [
                ["transaction_id", "account_number", "amount", "timestamp"],
                [5, "ACC005", 400.00, "2025-05-20T14:30:00Z"],
                [6, "ACC006", 500.00, "2025-05-21T09:00:00+00:00"],
            ],
            "04_twelve_hour_am_pm_format": [
                ["transaction_id", "account_number", "amount", "timestamp"],
                [7, "ACC007", 600.00, "05/20/2025 02:30 PM"],
                [8, "ACC008", 700.00, "05/21/2025 09:00 AM"],
            ],
            "05_mixed_formats_tz": [
                ["transaction_id", "account_number", "amount", "timestamp"],
                [9, "ACC009", 800.00, "2025-05-20T14:30:00Z"],
                [10, "ACC010", 900.00, "05/21/2025 09:00 AM"],
                [11, "ACC011", 1000.00, "2025-05-22"],
                [12, "ACC012", 1100.00, "21.05.2025"],
            ],
            "05_mixed_formats_ntz": [
                ["transaction_id", "account_number", "amount", "timestamp"],
                [9, "ACC009", 800.00, "05/20/2025 09:00 AM"],
                [10, "ACC010", 900.00, "05/21/2025 01:00 PM"],
                [11, "ACC011", 1000.00, "2025-05-22"],
                [12, "ACC012", 1100.00, "25.05.2025"],
            ],
            "06_malformed_timestamps": [
                ["transaction_id", "account_number", "amount", "timestamp"],
                [13, "ACC013", 1200.00, "not-a-date"],
                [14, "ACC014", 1300.00, "2025-02-30T25:61:00"],
                [15, "ACC015", 1400.00, "13/2025/05"],
                [16, "ACC016", 1500.00, "9999-99-99"],
            ],
            "07_unix_timestamps_s": [
                ["transaction_id", "account_number", "amount", "timestamp"],
                [17, "ACC017", 1600.00, "1747745400"],
                [18, "ACC018", 1700.00, "1747749000"],
            ],
            "07_unix_timestamps_ms": [
                ["transaction_id", "account_number", "amount", "timestamp"],
                [17, "ACC017", 1600.00, "1747745400000"],
                [18, "ACC018", 1700.00, "1747749000000"],
            ],
            "07_unix_timestamps_us": [
                ["transaction_id", "account_number", "amount", "timestamp"],
                [17, "ACC017", 1600.00, "1747745400000000"],
                [18, "ACC018", 1700.00, "1747749000000000"],
            ],
            "07_unix_timestamps_ns": [
                ["transaction_id", "account_number", "amount", "timestamp"],
                [17, "ACC017", 1600.00, "1747745400000000000"],
                [18, "ACC018", 1700.00, "1747749000000000000"],
            ],
            "08_missing_and_null_placeholders": [
                ["transaction_id", "account_number", "amount", "timestamp"],
                [19, "ACC019", 1800.00, ""],
                [20, "ACC020", 1900.00, "NULL"],
                [21, "ACC021", 2000.00, None],
                [27, "ACC027", 2600.00, "N/A"],
                [28, "ACC028", 2700.00, "01-01-2025"],
                [29, "ACC029", 2800.00, "01-02-2025"],
                [30, "ACC030", 2900.00, "01-03-2025"],
                [31, "ACC031", 3000.00, "invalid"],
                [32, "ACC032", 3100.00, "99-99-9999"],
            ],
            "09_milliseconds_TZ": [
                ["transaction_id", "account_number", "amount", "timestamp"],
                [22, "ACC022", 2100.00, "2025-05-21T15:30:45.123"],
                [23, "ACC023", 2200.00, "2025-05-21T15:30:45.987Z"],
            ],
            "09_milliseconds_NTZ": [
                ["transaction_id", "account_number", "amount", "timestamp"],
                [22, "ACC022", 2100.00, "2025-05-21T15:30:45.123"],
                [23, "ACC023", 2200.00, "2025-05-21T15:30:45.987"],
            ],
            "10_locale_formats_mixed": [
                ["transaction_id", "account_number", "amount", "timestamp"],
                [26, "ACC026", 2500.00, "21/05/2025"],
                [27, "ACC026", 2500.00, "21.06.2025"],
                [28, "ACC026", 2500.00, "08-21-2025"],
            ],
            "11_integer_encoded_dates_1": [
                ["transaction_id", "account_number", "amount", "timestamp"],
                [29, "ACC029", 2600.00, "2025"],
                [30, "ACC030", 2700.00, "2026"],
            ],
            "11_integer_encoded_dates_2": [
                ["transaction_id", "account_number", "amount", "timestamp"],
                [29, "ACC029", 2600.00, "202505"],
                [30, "ACC030", 2700.00, "202506"],
            ],
            "11_integer_encoded_dates_3": [
                ["transaction_id", "account_number", "amount", "timestamp"],
                [29, "ACC029", 2600.00, "20250520"],
                [30, "ACC030", 2700.00, "20250521"],
            ],
            "12_integer_encoded_dates_with_time": [
                ["transaction_id", "account_number", "amount", "timestamp"],
                [31, "ACC031", 2800.00, "20250520123000"],
                [32, "ACC032", 2900.00, "20250521123000"],
            ],
        }

    return _csv_edge_case_data


@pytest.fixture
def write_csv_edge_case_files() -> Callable:
    def _write_csv_edge_case_files(file_data: Dict[str,
                                                   List[str]], tmp_path: Path,
                                   separator: Optional[str] = None) -> None:
        for file_name, data in file_data.items():
            file_path = osp.join(tmp_path, f"{file_name}.csv")
            with open(file_path, "w") as f:
                writer = csv.writer(f,
                                    delimiter=separator if separator else ',')
                writer.writerows(data)

    return _write_csv_edge_case_files


@pytest.fixture
def write_stype_dtype_test_csv() -> Callable:
    def _write_stype_dtype_test_csv(tmp_path: Path) -> str:
        # Sample values for each Dtype
        dtype_sample_values = {
            Dtype.bool: [True, False],
            Dtype.int: [1749070681,
                        1749070682],  # valid epoch times also to test casting
            Dtype.int32: [100, 200],
            Dtype.int64: [10000000000, 20000000000],
            Dtype.float: [1.23, 4.56],
            Dtype.float32: [3.14, 6.28],
            Dtype.float64: [2.71, 8.909090909090909],
            Dtype.string: ["apple", "banana"],
            Dtype.binary: ["b64==", "b64=="],
            Dtype.date: ["2025-01-01", "2025-12-31"],
            Dtype.time: ["12:34:56", "23:59:59"],
            Dtype.floatlist: ["[1.1, 2.2]", "[3.3, 4.4]"],
            Dtype.intlist: ["[1, 2]", "[3, 4]"],
            Dtype.stringlist: ["[\"a\", \"b\"]", "[\"c\", \"d\"]"],
            Dtype.unsupported: ["?", "?"],
            Dtype.timedelta: ["1", "2"]
        }

        # https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
        # but "None" is listed there but not converted to null/nan by pandas
        pandas_nan_values = [
            "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
            "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "n/a",
            "nan", "null"
        ]

        # Generate valid (Stype, Dtype) combinations
        valid_columns = []
        for stype in Stype:
            if (stype
                    in [Stype.unsupported, Stype.image,
                        Stype.multicategorical]):
                continue
            for dtype in dtype_sample_values.keys(
            ):  # use sample set to skip depracated types
                if stype.supports_dtype(dtype):
                    colname = f"{stype.value}__{dtype.value}"

                    # Special case for 'image' stype -> generate http:// URLs
                    if stype == Stype.image:
                        values = [
                            f"http://imagehost.com/image_{i}.jpg"
                            for i in range(1000)
                        ]
                    else:
                        values = dtype_sample_values[dtype] * (
                            1000 // len(dtype_sample_values[dtype]) + 1)
                        values = values[:1000]
                    if dtype != Dtype.bool:
                        for i, value in enumerate(pandas_nan_values):
                            values[100 + i] = value

                    if dtype in [Dtype.time, Dtype.date]:
                        values[1] = 'NaT'

                    valid_columns.append((colname, values))

        # Add custom column: numerical__float__mixed
        mixed_numerical_values = [(i if i % 2 == 0 else i + 0.5)
                                  for i in range(1000)]
        valid_columns.append(
            ("numerical__float__mixed", mixed_numerical_values))

        # need to have 1.5 times as many unique rows as split values
        multicategorical_values = ['a;b;c', 'a;b', 'c;b', 'a;c', 'b', 'a', 'c'
                                   ] * (1000 // 7 + 1)
        valid_columns.append(
            ("multicategorical__string", multicategorical_values))

        # Create data rows
        num_rows = 1000
        rows = []
        for i in range(num_rows):
            row = {}
            for colname, values in valid_columns:
                row[colname] = values[i]
            rows.append(row)

        # Write CSV file
        filename = "stype_dtype_combinations.csv"
        with open(osp.join(tmp_path, filename), "w", newline='',
                  encoding='utf-8') as f:
            writer = csv.DictWriter(
                f, fieldnames=[col for col, _ in valid_columns])
            writer.writeheader()
            writer.writerows(rows)
        return osp.join(tmp_path, filename)

    return _write_stype_dtype_test_csv


@pytest.fixture
def databricks_hm_connector():
    from kumo.connector.databricks_connector import DatabricksConnector

    return DatabricksConnector('databricks', host='https://example.invalid',
                               warehouse_id='fake-warehouse',
                               cluster_id='fake-cluster', catalog='fake-catalog',
                               schema_name='hm')
