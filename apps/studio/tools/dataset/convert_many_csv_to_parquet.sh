#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Please edit the env variable in this script before running
#
#   ./tools/dataset/convert_many_csv_to_parquet.sh
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
CSV_S3_ROOT="s3://kumo-public-datasets/worldcup/"
OUTPUT_S3_DIR="s3://kumo-public-datasets/worldcup/parquet/"

echo "We will run the following commands:"
echo
for table in ${TABLES[@]}; do
  echo ./tools/dataset/convert_csv_to_parquet.py \""${CSV_S3_ROOT}${table}.csv"\" \""${OUTPUT_S3_DIR}${table}.parquet"\"
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
  echo Copying "${CSV_S3_ROOT}${table}.csv" to "${OUTPUT_S3_DIR}${table}.parquet"
  ./tools/dataset/convert_csv_to_parquet.py "${CSV_S3_ROOT}${table}.csv" "${OUTPUT_S3_DIR}${table}.parquet"
done
