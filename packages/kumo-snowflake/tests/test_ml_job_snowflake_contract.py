# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in contracts against real Snowflake and the installed ML Jobs SDK.

Set KUMO_ML_JOB_LIVE_CONFIG to a JSON Snowpark connection configuration and
KUMO_ML_JOB_LIVE_POOL to an existing GPU pool. These tests create a temporary
stage and upload payloads, but never dispatch a service or resume the pool.
Run with ``python -m unittest discover -s packages/kumo-snowflake/tests
-p test_ml_job_snowflake_contract.py`` in the SDK environment.
"""
from __future__ import annotations

import json
import os
import unittest
import uuid

from kumo_snowflake._ml_job_stage import StageStore, _stage_object_missing
from kumo_snowflake.ml_jobs import _set_tags


def contract_function(value: int = 1) -> int:
    return value


@unittest.skipUnless(os.environ.get('KUMO_ML_JOB_LIVE_CONFIG'),
                     'explicit live Snowflake configuration required')
class SnowflakeContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from snowflake.snowpark import Session

        cls.session = Session.builder.configs(
            json.loads(os.environ['KUMO_ML_JOB_LIVE_CONFIG'])).create()

    @classmethod
    def tearDownClass(cls):
        cls.session.close()

    def setUp(self):
        self.suffix = uuid.uuid4().hex.upper()
        self.stage = ('KUMO3_PIP_TESTING.PUBLIC.KUMO_CONTRACT_' + self.suffix)
        self.session.sql(f'CREATE TEMPORARY STAGE {self.stage}').collect()
        self.addCleanup(
            lambda: self.session.sql(f'DROP STAGE {self.stage}').collect())
        self.store = StageStore(self.session, '@' + self.stage)

    def test_real_remote_registration_preserves_seeded_wheel(self):
        from snowflake.ml import jobs

        name = 'KUMO_CONTRACT_' + self.suffix
        wheel = 'contract_wheel-0.0.0-py3-none-any.whl'
        content = b'contract marker; never installed or executed'
        self.store.put_bytes('wheels/' + wheel, content)
        self.store.copy_file('wheels/' + wheel, name + '/app')
        definition = jobs.remote(
            os.environ['KUMO_ML_JOB_LIVE_POOL'],
            stage_name='@' + self.stage,
            name=name,
            session=self.session,
            runtime_environment='2.8.1-py312',
            pip_requirements=['./' + wheel],
        )(contract_function)
        # Real SDK registration and payload upload; do not execute this SQL.
        sql = definition.to_sql()
        self.assertIn('SYSTEM$EXECUTE_ML_JOB', sql.upper())
        self.assertEqual(definition.spec_options.stage_path,
                         '@' + self.stage + '/' + name)
        self.assertEqual(self.store.get_bytes(name + '/app/' + wheel), content)
        self.assertEqual(self.store.get_bytes(name + '/app/requirements.txt'),
                         ('./' + wheel).encode())
        self.assertTrue(
            any('/app/' in str(row.get('name', ''))
                and str(row.get('name', '')).endswith('.py')
                for row in self.store.list(name + '/app/')))

    def test_missing_file_is_distinct_from_invalid_stage(self):
        self.assertIsNone(self.store.get_json('absent.json', missing_ok=True))
        invalid = StageStore(self.session, '@' + self.stage + '_INVALID')
        with self.assertRaises(Exception) as caught:
            invalid.get_json('absent.json', missing_ok=True)
        self.assertFalse(_stage_object_missing(caught.exception))
        self.assertEqual(caught.exception.sql_error_code, 2003)

    def test_tag_literal_roundtrip_and_alter_service_parser(self):
        tag = 'KUMO3_PIP_TESTING.PUBLIC.KUMO_CONTRACT_TAG'
        value = "review's \\path; -- ?"
        # Check string semantics with the same SQL literal used by _set_tags.
        literal = "'" + value.replace('\\', '\\\\').replace("'", "''") + "'"
        row = self.session.sql('SELECT ' + literal).collect()[0]
        self.assertEqual(row[0], value)
        # An absent service must reach name resolution (2003), rather than
        # fail parsing a DDL bind or incorrectly terminated string literal.
        with self.assertRaises(Exception) as caught:
            _set_tags(self.session,
                      'KUMO3_PIP_TESTING.PUBLIC.KUMO_ABSENT_' + self.suffix,
                      {tag: value})
        self.assertEqual(caught.exception.sql_error_code, 2003)


if __name__ == '__main__':
    unittest.main()
