"""GH Archive (githubarchive.{day,month} on BigQuery) queries.

Only `type`, `created_at`, `actor.id`, `actor.login`, `repo.id`, `repo.name` are read (~0.2-0.3 GB
per day); `payload` costs ~15 GB per day before GitHub trimmed Events-API payloads in Oct 2025, so
it is read only for the Oct-2025 attribution check (after the trim, PR payloads keep
`action`/`number` only). `actor.login` is used inside SQL only (bot
and owner filters) and is never selected.

Maintainers: an actor who pushed to a repo (PushEvent needs write access) is that repo's
maintainer; all of a maintainer's events on that repo are dropped, so the PR / issue / comment
actors that remain are external contributors (maintainers close and merge PRs, which also emits
PullRequestEvents).
"""

from typing import Any

INTERACTION_TYPES = {
    "WatchEvent": "star",
    "ForkEvent": "fork",
    "PullRequestEvent": "pr",
    "IssuesEvent": "issue",
    "IssueCommentEvent": "comment",
    "PullRequestReviewEvent": "review",
    "PullRequestReviewCommentEvent": "review",
}
MAINTAINER_TYPE = "PushEvent"
CONTRIB_KINDS = ("pr", "issue", "comment", "review")

_FILTERS = """
    AND NOT REGEXP_CONTAINS(LOWER(actor.login), @bot_regex)
    AND LOWER(actor.login) != LOWER(SPLIT(repo.name, '/')[SAFE_OFFSET(0)])"""


def _external_events(source: str, where: str) -> str:
    """CTEs `ev` (filtered events) and `ext` (events of non-maintainers, PushEvent removed)."""
    return f"""
WITH ev AS (
  SELECT type, actor.id AS actor_id, repo.id AS repo_id, repo.name AS repo_name, created_at
  FROM {source}
  WHERE {where}
    AND type IN UNNEST(@types){_FILTERS}
),
maint AS (SELECT DISTINCT actor_id, repo_id FROM ev WHERE type = '{MAINTAINER_TYPE}'),
ext AS (
  SELECT ev.* FROM ev LEFT JOIN maint USING (actor_id, repo_id)
  WHERE maint.actor_id IS NULL AND ev.type != '{MAINTAINER_TYPE}'
)"""


def month_source(start: str, end: str) -> tuple[str, str]:
    """Month wildcard table + suffix filter for months YYYYMM in [start, end] (prefix `20*`:
    the datasets also hold views, which wildcard queries cannot read)."""
    return (
        "`githubarchive.month.20*`",
        f"_TABLE_SUFFIX BETWEEN '{start[2:]}' AND '{end[2:]}'",
    )


def repo_activity_sql(month_start: str, month_end: str) -> str:
    """Per repo: distinct external actors by kind over a month range (scoping)."""
    source, where = month_source(month_start, month_end)
    contrib = ", ".join(f"'{t}'" for t, k in INTERACTION_TYPES.items() if k in CONTRIB_KINDS)
    return (
        _external_events(source, where)
        + f"""
SELECT repo_id,
  ARRAY_AGG(repo_name ORDER BY created_at DESC LIMIT 1)[OFFSET(0)] AS repo_name,
  COUNT(DISTINCT IF(type = 'WatchEvent', actor_id, NULL)) AS n_star,
  COUNT(DISTINCT IF(type = 'ForkEvent', actor_id, NULL)) AS n_fork,
  COUNT(DISTINCT IF(type = 'PullRequestEvent', actor_id, NULL)) AS n_pr,
  COUNT(DISTINCT IF(type = 'IssuesEvent', actor_id, NULL)) AS n_issue,
  COUNT(DISTINCT IF(type IN ({contrib}), actor_id, NULL)) AS n_contrib,
  COUNT(DISTINCT actor_id) AS n_actors
FROM ext
GROUP BY repo_id
HAVING n_actors >= @min_actors"""
    )


def co_touch_sql(month_start: str, month_end: str) -> str:
    """Per repo outside @seed_ids: distinct external actors it shares with the seed repos."""
    source, where = month_source(month_start, month_end)
    return (
        _external_events(source, where)
        + """,
seed_users AS (SELECT DISTINCT actor_id FROM ext WHERE repo_id IN UNNEST(@seed_ids))
SELECT repo_id,
  ARRAY_AGG(repo_name ORDER BY created_at DESC LIMIT 1)[OFFSET(0)] AS repo_name,
  COUNT(DISTINCT actor_id) AS n_shared
FROM ext JOIN seed_users USING (actor_id)
WHERE repo_id NOT IN UNNEST(@seed_ids)
GROUP BY repo_id
HAVING n_shared >= @min_shared"""
    )


def day_events_sql(day: str) -> str:
    """One day (YYYYMMDD) of events on @repo_ids, aggregated per (type, actor, repo):
    first timestamp and count. PushEvent rows are kept here (maintainer marker)."""
    return f"""
SELECT type, actor.id AS actor_id, repo.id AS repo_id,
  ARRAY_AGG(repo.name ORDER BY created_at DESC LIMIT 1)[OFFSET(0)] AS repo_name,
  MIN(created_at) AS first_at, COUNT(*) AS n
FROM `githubarchive.day.{day}`
WHERE repo.id IN UNNEST(@repo_ids)
  AND type IN UNNEST(@types){_FILTERS}
GROUP BY type, actor_id, repo_id"""


def payload_check_sql(day_start: str, day_end: str) -> str:
    """PR / issue events with payload facts for @repo_ids over days [start, end] (YYYYMMDD):
    action (after GitHub's Oct-2025 trim PR payloads keep `action` and `number` but no author or
    merge flag; `opened` identifies the author), issue authorship (login compared inside SQL,
    only the boolean leaves BigQuery) and labels."""
    return f"""
SELECT type, actor.id AS actor_id, repo.id AS repo_id, created_at,
  JSON_VALUE(payload, '$.action') AS action,
  COALESCE(JSON_VALUE(payload, '$.pull_request.number'),
           JSON_VALUE(payload, '$.issue.number'),
           JSON_VALUE(payload, '$.number')) AS number,
  LOWER(JSON_VALUE(payload, '$.issue.user.login')) = LOWER(actor.login) AS actor_is_issue_author,
  ARRAY(SELECT LOWER(JSON_VALUE(l, '$.name'))
        FROM UNNEST(IFNULL(JSON_QUERY_ARRAY(payload, '$.issue.labels'),
                           JSON_QUERY_ARRAY(payload, '$.labels'))) AS l) AS labels
FROM `githubarchive.day.20*`
WHERE _TABLE_SUFFIX BETWEEN '{day_start[2:]}' AND '{day_end[2:]}'
  AND repo.id IN UNNEST(@repo_ids)
  AND type IN ('PullRequestEvent', 'IssuesEvent'){_FILTERS}"""


def base_params(cfg: dict[str, Any], types: list[str] | None = None) -> dict[str, Any]:
    return {
        "types": types or [*INTERACTION_TYPES, MAINTAINER_TYPE],
        "bot_regex": cfg["bot_regex"],
    }
