This is the case where two independent PQs are used to produce one prediction outcome.

Here we use the following example:

- Original Task: Prediction E[SUM(purchases in next 28 days)]

- This task is combined by the following two prediction tasks:

  - Task 1: Binary Classification - P[SUM(purchases in next 28 days) > 0]
  - Task 2: Regression where SUM(purchases in next 28 days) > 0
  - Final outcome = (Probability from Task 1) * (Regression from Task 2)

- `train.py` and `train_poll.py`: Development scripts

  - Trains two PQs in parallel (i.e. non-blocking)
  - Combines the outcome once both PQ trainings are done
  - When some exception happened, there should be the capability of running only failed jobs.
  - Two PQs need to be aligned in the data and split, even after some time in the case of retraining only one job.
  - For potential iterations for one of the two jobs, we need to materialize/lock the graph / split. d
  - Eval on the combined outcome

- `train_eval.py`:

  - Run two BPs in parallel (i.e non-blocking)
  - Combines the outcome
