#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the install preflight."""

import contextlib
import importlib.util
import io
import pathlib
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "kumo_helm_preflight", ROOT / "conformance" / "preflight.py")
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)


class PreflightRedactionTest(unittest.TestCase):
    def setUp(self):
        PREFLIGHT.failures.clear()

    def capture_secret_check(self, response):
        output = io.StringIO()
        with mock.patch.object(PREFLIGHT, "get_secret", return_value=response):
            with contextlib.redirect_stdout(output):
                PREFLIGHT.check_secret(
                    object(),
                    "TEMPORAL_MTLS",
                    "customer-namespace",
                    "production-database-secret",
                    ["database-password"],
                )
        return output.getvalue()

    def assert_sensitive_values_redacted(self, output):
        self.assertNotIn("customer-namespace", output)
        self.assertNotIn("production-database-secret", output)
        self.assertNotIn("database-password", output)

    def test_secret_not_found_redacts_identifiers(self):
        output = self.capture_secret_check((None, "HTTP 404"))
        self.assertIn("required Secret not found", output)
        self.assert_sensitive_values_redacted(output)

    def test_secret_api_error_redacts_identifiers_and_error(self):
        output = self.capture_secret_check(
            (None, "apiserver error containing sensitive detail"))
        self.assertIn("cannot read required Secret", output)
        self.assertNotIn("sensitive detail", output)
        self.assert_sensitive_values_redacted(output)

    def test_missing_key_redacts_identifiers(self):
        output = self.capture_secret_check(({"data": {}}, None))
        self.assertIn("missing one or more required keys", output)
        self.assert_sensitive_values_redacted(output)

    def test_success_redacts_identifiers(self):
        output = self.capture_secret_check(({
            "data": {
                "database-password": "encoded-value"
            }
        }, None))
        self.assertIn("present with all required keys", output)
        self.assert_sensitive_values_redacted(output)
        self.assertNotIn("encoded-value", output)

    def test_transport_address_and_external_failures_block(self):
        self.assertEqual(
            PREFLIGHT.split_host_port("temporal.example:8233", "7233"),
            ("temporal.example", "8233"))
        self.assertEqual(
            PREFLIGHT.split_host_port("[2001:db8::1]:8233", "7233"),
            ("2001:db8::1", "8233"))
        self.assertIsNone(
            PREFLIGHT.split_host_port("temporal.example:not-a-port", "7233"))
        output = io.StringIO()
        with mock.patch.object(PREFLIGHT.socket, "create_connection",
                               side_effect=OSError("connection refused")):
            with mock.patch.object(PREFLIGHT.time, "sleep"):
                with contextlib.redirect_stdout(output):
                    PREFLIGHT.check_dial("TEMPORAL_ACCESS",
                                         "temporal.external.example", "8233")
                    PREFLIGHT.check_dial("PG_ACCESS",
                                         "postgres.external.example", "5432")
        self.assertIn("[FAIL] TEMPORAL_ACCESS", output.getvalue())
        self.assertIn("[FAIL] PG_ACCESS", output.getvalue())

    def test_postgres_ca_bundle_must_be_readable(self):
        env = {
            "TEMPORAL_HOST": "temporal.example:7233",
            "PG_CA_PATH": "/opt/kumo/certs/postgres.pem",
            "PG_HOST": "postgres.example",
            "PG_PORT": "5432",
        }
        with mock.patch.dict(PREFLIGHT.os.environ, env, clear=True):
            with mock.patch.object(PREFLIGHT, "check_dial"):
                with mock.patch("builtins.open",
                                mock.mock_open(read_data=b"certificate")):
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        PREFLIGHT.main()
                    self.assertIn("[PASS] POSTGRES_CA", output.getvalue())

                PREFLIGHT.failures.clear()
                with mock.patch("builtins.open",
                                side_effect=OSError("not found")):
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        with self.assertRaises(SystemExit):
                            PREFLIGHT.main()
                    self.assertIn("[FAIL] POSTGRES_CA", output.getvalue())


if __name__ == "__main__":
    unittest.main()
