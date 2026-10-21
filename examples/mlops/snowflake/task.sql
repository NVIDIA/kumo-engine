-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
-- All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

-- register_jobs.py fills the <...> slots once per job; the block swaps this
-- run's task name and graph run id into the job arguments.
CREATE OR REPLACE TASK <task_name>
  WAREHOUSE = <warehouse>
  SCHEDULE = '<schedule>'
  USER_TASK_TIMEOUT_MS = <timeout_ms>
  SUSPEND_TASK_AFTER_NUM_FAILURES = 3
AS
DECLARE
  job_sql STRING DEFAULT $$<execute_ml_job_sql>$$;
  task_name STRING;
  run_id STRING;
BEGIN
  SELECT SYSTEM$TASK_RUNTIME_INFO('CURRENT_TASK_NAME') INTO :task_name;
  SELECT SYSTEM$TASK_RUNTIME_INFO('CURRENT_TASK_GRAPH_RUN_GROUP_ID')
    INTO :run_id;
  job_sql := REPLACE(REPLACE(job_sql, '__KUMO_JOB_ID__', task_name),
                     '__KUMO_JOB_RUN_ID__', run_id);
  EXECUTE IMMEDIATE :job_sql;
  RETURN run_id;
END;
