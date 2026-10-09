-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
-- All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

\set ON_ERROR_STOP on
\getenv temporal_role KUMO_TEMPORAL_ROLE
\getenv temporal_password KUMO_TEMPORAL_PASSWORD

-- PostgreSQL before 15 prevents this administrator from reassigning the public
-- schema, so fail before creating anything.
SELECT format(
  'DO $$ BEGIN RAISE EXCEPTION %L; END $$',
  'Temporal bootstrap requires PostgreSQL 15 or newer; this server reports ' ||
  current_setting('server_version')
)
WHERE current_setting('server_version_num')::int < 150000
\gexec

SELECT format('CREATE ROLE %I LOGIN', :'temporal_role')
WHERE NOT EXISTS (
  SELECT 1 FROM pg_roles WHERE rolname = :'temporal_role'
)
\gexec

SELECT format(
  'ALTER ROLE %I WITH LOGIN PASSWORD %L',
  :'temporal_role',
  :'temporal_password'
)
\gexec

-- PostgreSQL 16's implicit CREATEROLE grant has ADMIN but not SET; test SET
-- there before adding the membership needed to create owned databases.
SELECT pg_has_role(
  current_user,
  :'temporal_role',
  CASE WHEN current_setting('server_version_num')::int >= 160000
    THEN 'SET' ELSE 'MEMBER' END
) AS temporal_role_grant_preexisting
\gset

SELECT format('GRANT %I TO %I', :'temporal_role', current_user)
WHERE :'temporal_role_grant_preexisting' = 'f'
\gexec

SELECT format('CREATE DATABASE temporal OWNER %I', :'temporal_role')
WHERE NOT EXISTS (
  SELECT 1 FROM pg_database WHERE datname = 'temporal'
)
\gexec

SELECT format(
  'CREATE DATABASE temporal_visibility OWNER %I',
  :'temporal_role'
)
WHERE NOT EXISTS (
  SELECT 1 FROM pg_database WHERE datname = 'temporal_visibility'
)
\gexec

SELECT format('ALTER DATABASE temporal OWNER TO %I', :'temporal_role')
\gexec
SELECT format(
  'ALTER DATABASE temporal_visibility OWNER TO %I',
  :'temporal_role'
)
\gexec

\connect temporal
SELECT format('ALTER SCHEMA public OWNER TO %I', :'temporal_role')
\gexec

\connect temporal_visibility
SELECT format('ALTER SCHEMA public OWNER TO %I', :'temporal_role')
\gexec

-- Revoke only membership added by this script; implicit creator grants remain.
SELECT format('REVOKE %I FROM %I', :'temporal_role', current_user)
WHERE :'temporal_role_grant_preexisting' = 'f'
\gexec
