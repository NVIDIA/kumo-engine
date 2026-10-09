# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-specific Predictive Queries for RelBench datasets.

Each entry maps a (dataset_name, task_name) pair from RelBench to a
PQuery string suitable for Kumo Enterprise.
"""

from __future__ import annotations

from typing import Dict, Tuple

RELBENCH_PQUERIES: Dict[Tuple[str, str], str] = {
    # rel-event node-level tasks
    (
        "rel-event",
        "user-attendance",
    ):
    """
PREDICT COUNT(event_attendees.* WHERE
  event_attendees.status IN ('yes', 'maybe'),
  0, 7, days
)
FOR EACH users.user_id
""",
    (
        "rel-event",
        "user-repeat",
    ):
    """
PREDICT COUNT(event_attendees.* WHERE
  (event_attendees.status = 'yes' OR event_attendees.status = 'maybe'),
  0, 7, days
) > 0
FOR EACH users.user_id
WHERE COUNT(event_attendees.* WHERE
  (event_attendees.status = 'yes' OR event_attendees.status = 'maybe'),
  -14, 0, days
) > 0
ASSUMING COUNT(event_attendees.*, 0, 7, days) > 0
""",
    (
        "rel-event",
        "user-ignore",
    ):
    """
PREDICT COUNT(event_attendees.* WHERE
  event_attendees.status = 'invited',
  0, 7, days
) > 2
FOR EACH users.user_id
ASSUMING COUNT(event_attendees.*, 0, 7, days) > 0
""",
    # f1 node-level tasks
    (
        "rel-f1",
        "driver-position",
    ):
    """
PREDICT AVG(results.positionOrder, 0, 60, days)
FOR EACH drivers.driverId
WHERE COUNT(results.*, -365, 0, days) > 0
""",
    (
        "rel-f1",
        "driver-dnf",
    ):
    """
PREDICT COUNT(results.* WHERE results.statusId != 1, 0, 30, days) > 0
FOR EACH drivers.driverId
ASSUMING COUNT(results.*, 0, 30, days) > 0
""",
    (
        "rel-f1",
        "driver-top3",
    ):
    """
PREDICT MIN(qualifying.position, 0, 30, days) <= 3
FOR EACH drivers.driverId
ASSUMING COUNT(qualifying.*, 0, 30, days) > 0
""",
    # rel-avito node-level tasks
    (
        "rel-avito",
        "ad-ctr",
    ):
    """
PREDICT SUM(SearchStream.IsClick, 0, 4, days) /
  COUNT(SearchStream.SearchID, 0, 4, days)
FOR EACH AdsInfo.AdID
ASSUMING SUM(SearchStream.IsClick, 0, 4, days) > 0
""",
    (
        "rel-avito",
        "user-clicks",
    ):
    """
PREDICT COUNT(SearchStream.* WHERE SearchStream.IsClick = 1.0, 0, 4, days) > 1
FOR EACH UserInfo.UserID
ASSUMING COUNT(SearchInfo.*, 0, 4, days) > 0
""",
    (
        "rel-avito",
        "user-visits",
    ):
    """
PREDICT COUNT_DISTINCT(VisitStream.AdID, 0, 4, days) > 1
FOR EACH UserInfo.UserID
ASSUMING COUNT(VisitStream.*, 0, 4, days) > 0
""",
    # rel-hm churn tasks
    (
        "rel-hm",
        "item-sales",
    ):
    """
PREDICT SUM(transactions.price, 0, 7, days)
FOR EACH article.article_id
""",
    (
        "rel-hm",
        "user-churn",
    ):
    """
PREDICT COUNT(transactions.*, 0, 7, days)=0
FOR EACH customer.customer_id
WHERE COUNT(transactions.*, -7, 0, days)>0
""",
    # rel-stack node-level tasks
    (
        "rel-stack",
        "post-votes",
    ):
    """
PREDICT COUNT(votes.* WHERE votes.VoteTypeId = 2, 0, 91, days)
FOR EACH posts.Id
WHERE (
  posts.OwnerUserId != -1 AND
  posts.OwnerUserId IS NOT NULL AND
  posts.PostTypeId = 1
)
""",
    (
        "rel-stack",
        "user-badge",
    ):
    """
PREDICT COUNT(badges.*, 0, 91, days) > 0
FOR EACH users.Id
""",
    (
        "rel-stack",
        "user-engagement",
    ):
    """
PREDICT (
  COUNT(posts.*, 0, 91, days) > 0 OR
  COUNT(votes.*, 0, 91, days) > 0 OR
  COUNT(comments.*, 0, 91, days) > 0
)
FOR EACH users.Id
WHERE (
  COUNT(posts.*, -INF, 0, days) > 0 OR
  COUNT(votes.*, -INF, 0, days) > 0 OR
  COUNT(comments.*, -INF, 0, days) > 0
)
""",
    # rel-trial node-level tasks
    (
        "rel-trial",
        "site-success",
    ):
    """
PREDICT AVG(outcome_analyses.p_value < 0.05, 0, 365, days WHERE
  outcomes.outcome_type = 'Primary' AND
  outcome_analyses.p_value >= 0 AND
  outcome_analyses.p_value <= 1 AND
  (outcome_analyses.p_value_modifier IS NULL OR
   outcome_analyses.p_value_modifier != '>')
)
FOR EACH facilities.facility_id
ASSUMING COUNT(facilities_studies.*, 0, 365, days) > 0
""",
    (
        "rel-trial",
        "study-adverse",
    ):
    """
PREDICT SUM(outcome_analyses.reported_event_totals.subjects_affected,
  0, 365, days WHERE
  (
    outcome_analyses.reported_event_totals.event_type = 'serious' OR
    outcome_analyses.reported_event_totals.event_type = 'deaths'
  ) AND
  outcome_analyses.reported_event_totals.subjects_affected IS NOT NULL
)
FOR EACH studies.nct_id
ASSUMING COUNT(outcome_analyses.reported_event_totals.*,
  0, 365, days WHERE
  (
    outcome_analyses.reported_event_totals.event_type = 'serious' OR
    outcome_analyses.reported_event_totals.event_type = 'deaths'
  ) AND
  outcome_analyses.reported_event_totals.subjects_affected IS NOT NULL
) > 0
""",
    (
        "rel-trial",
        "study-outcome",
    ):
    """
PREDICT COUNT(outcome_analyses.*, 0, 365, days WHERE
  outcomes.outcome_type = 'Primary' AND
  outcome_analyses.p_value >= 0 AND
  outcome_analyses.p_value <= 1 AND
  (outcome_analyses.p_value_modifier IS NULL OR
   outcome_analyses.p_value_modifier != '>') AND
  outcome_analyses.p_value <= 0.05
) > 0
FOR EACH studies.nct_id
ASSUMING COUNT(outcome_analyses.*, 0, 365, days WHERE
  outcomes.outcome_type = 'Primary' AND
  outcome_analyses.p_value >= 0 AND
  outcome_analyses.p_value <= 1 AND
  (outcome_analyses.p_value_modifier IS NULL OR
   outcome_analyses.p_value_modifier != '>')
) > 0
""",
    # rel-amazon churn tasks
    (
        "rel-amazon",
        "item-ltv",
    ):
    """
PREDICT SUM(review.product.price, 0, 91, days)
FOR EACH product.product_id
ASSUMING COUNT(review.*, 0, 91, days) > 0
""",
    (
        "rel-amazon",
        "user-churn",
    ):
    """
PREDICT COUNT(review.*, 0, 91, days)=0
FOR EACH customer.customer_id
WHERE COUNT(review.*, -91, 0, days)>0
""",
    (
        "rel-amazon",
        "user-ltv",
    ):
    """
PREDICT SUM(review.product.price, 0, 91, days)
FOR EACH customer.customer_id
WHERE COUNT(review.*, -91, 0, days) > 0
""",
    (
        "rel-amazon",
        "item-churn",
    ):
    """
PREDICT COUNT(review.*, 0, 91, days)=0
FOR EACH product.product_id
WHERE COUNT(review.*, -91, 0, days)>0
""",
}
