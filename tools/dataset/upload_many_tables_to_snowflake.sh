#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Please edit the env variable in this script before running
#
#   ./tools/dataset/upload_many_tables_to_snowflake.sh
#

set -e

TABLES=(
  award_winners
  awards
  bookings
  confederations
  goals
  group_standings
  groups
  host_countries
  manager_appearances
  manager_appointments
  managers
  matches
  penalty_kicks
  player_appearances
  players
  qualified_teams
  referee_appearances
  referee_appointments
  referees
  squads
  stadiums
  substitutions
  team_appearances
  teams
  tournament_stages
  tournament_standings
  tournaments
)
S3_ROOT="s3://kumo-public-datasets/worldcup/parquet/"
SNOWFLAKE_DB="KUMO"
SNOWFLAKE_SCHEMA="WORLDCUP"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "We will run the following commands:"
echo
for table in ${TABLES[@]}; do
  echo "${SCRIPT_DIR}/upload_table_to_snowflake.py" \""${S3_ROOT}${table}.parquet"\" \""${SNOWFLAKE_DB}.${SNOWFLAKE_SCHEMA}.${table}"\"
done
echo
read -p "Do you wish to continue? [Y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
  false
fi

for table in ${TABLES[@]}; do
  echo
  echo "Trying to copy ${S3_ROOT}${table}.parquet to ${SNOWFLAKE_DB}.${SNOWFLAKE_SCHEMA}.${table}"

  # Try original path format first (table.parquet)
  if "${SCRIPT_DIR}/upload_table_to_snowflake.py" "${S3_ROOT}${table}.parquet" "${SNOWFLAKE_DB}.${SNOWFLAKE_SCHEMA}.${table}"; then
    echo "Successfully uploaded ${table} using .parquet format"
  else
    # If that failed, try the alternative path format (table/)
    echo "S3 path ${S3_ROOT}${table}.parquet does not exist, trying ${S3_ROOT}${table}/"
    if "${SCRIPT_DIR}/upload_table_to_snowflake.py" "${S3_ROOT}${table}/" "${SNOWFLAKE_DB}.${SNOWFLAKE_SCHEMA}.${table}"; then
      echo "Successfully uploaded ${table} using directory format"
    else
      echo "ERROR: Failed to upload ${table} - neither ${S3_ROOT}${table}.parquet nor ${S3_ROOT}${table}/ exists (or an unrelated error occurred)"
      exit 1
    fi
  fi
done
