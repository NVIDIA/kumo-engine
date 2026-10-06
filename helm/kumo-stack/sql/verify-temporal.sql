-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
-- All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

\set ON_ERROR_STOP on
\getenv expected_role KUMO_TEMPORAL_ROLE

SELECT
  pg_catalog.pg_get_userbyid(datdba) = :'expected_role' AND
  pg_catalog.pg_get_userbyid(nspowner) = :'expected_role' AND
  has_schema_privilege(:'expected_role', 'public', 'CREATE')
FROM pg_database, pg_namespace
WHERE datname = current_database() AND nspname = 'public';
