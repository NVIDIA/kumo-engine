# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import common_utils.string
from common_utils import string
from common_utils.string import contains_identifier, contains_number


def test_validate_id():
    # Valid ID should pass
    common_utils.string.validate_id('abA2_')

    # Invalid IDs should fail
    with pytest.raises(ValueError, match=r"contain alphanumeric"):
        common_utils.string.validate_id('a??')

    with pytest.raises(ValueError, match=r"begin with a letter"):
        common_utils.string.validate_id('2eee')

    with pytest.raises(ValueError, match=r"fill out this value"):
        common_utils.string.validate_id("")


def test_contains_identifier():
    assert contains_identifier("unpaid") is False
    assert contains_identifier("unicorn_ID") is True
    assert contains_identifier("unicornId") is True
    assert contains_identifier("first_bid") is False
    assert contains_identifier("SK_ID_CURR") is True
    assert contains_identifier("postal_code") is True
    assert contains_identifier("productkey") is True

    assert contains_identifier("user_hash")
    assert contains_identifier("user_id")
    assert contains_identifier("item_cate")
    assert contains_identifier("item_class")
    assert contains_identifier("target_label")
    assert not contains_identifier("target_num")


def test_contains_number():
    assert contains_number("price") is True
    assert contains_number("testNumber") is True
    assert contains_number("product_total_price") is True
    assert contains_number("AVG_SALARY") is True
    assert contains_number("NUM_EMPLOYEES") is True
    assert contains_number("sum_of_transactions") is True
    assert contains_number("TOTAL_NUMBER") is True


def test_contains_time():
    assert string.contains_time("TS") is True
    assert string.contains_time("starts") is False


def test_contains_date():
    assert string.contains_date("Dt") is True
    assert string.contains_date("dts") is False
    assert string.contains_date("updated_at") is True
    assert string.contains_date("last_updated") is True
    assert string.contains_date("creation_date") is True
    assert string.contains_date("timestamp") is False
    assert string.contains_date("datestring") is False
    assert string.contains_date("DATETIME") is True
    assert string.contains_date("date_time_stamp") is True


@pytest.mark.parametrize(
    'name',
    [
        'dataset.csv/',
        'dataset.parquet/',
        'nested/many/layers/dataset.parquet/',
    ],
)
def test_connector_relative_source_preserves_directory_intent(name):
    string.validate_connector_relative_source_name(name)
    assert string.normalize_connector_source_name(name) == name


@pytest.mark.parametrize(
    'name',
    [
        's3://company-data/nested/events.parquet',
        'gs://company-data/nested/events.parquet',
        ('abfss://curated@companylake.dfs.core.windows.net/'
         'nested/events.parquet'),
    ],
)
def test_connector_source_accepts_supported_rootless_cloud_uri(name):
    string.validate_connector_source_name(name)
    assert string.normalize_connector_source_name(name) == name


@pytest.mark.parametrize('name', [
    'https://company-data/events.parquet',
    'file:///tmp/events.parquet',
])
def test_connector_source_rejects_unsupported_full_uri(name):
    with pytest.raises(ValueError, match='s3://, gs://, or abfss://'):
        string.validate_connector_source_name(name)


@pytest.mark.parametrize(
    'name',
    [
        'dataset.csv//',
        'nested//dataset.csv/',
        'nested/./dataset.csv/',
        'nested/../dataset.csv/',
    ],
)
def test_connector_relative_source_rejects_unsafe_directory_paths(name):
    with pytest.raises(ValueError):
        string.validate_connector_relative_source_name(name)
