"""End-to-end smoke test against the live deployment (API + frontend) and a CORS check.

    uv run python scripts/smoke_live.py --api https://... --web https://...
Exits non-zero on any failure. Uses a public account taken at runtime from the catalog; nothing
about it is printed or stored.
"""

import argparse
import json
import sys
from pathlib import Path

import requests


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--api", required=True)
    p.add_argument("--web", required=True)
    p.add_argument("--username-from", default="data/serving/static/repos.json")
    a = p.parse_args()
    api, web = a.api.rstrip("/"), a.web.rstrip("/")
    failures = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
        if not ok:
            failures.append(name)

    h = requests.get(f"{api}/health", timeout=90)
    check(
        "api /health",
        h.ok and h.json().get("status") == "ok",
        f"{h.json().get('open_issues')} open issues",
    )
    o = requests.get(f"{api}/options", timeout=30).json()
    check("api /options", bool(o.get("languages")) and bool(o.get("interests")))

    r = requests.post(
        f"{api}/recommend/onboarding",
        json={"languages": ["Python"], "interests": ["ml", "tools"], "hours": 3},
        timeout=30,
    )
    body = r.json()
    check(
        "onboarding path",
        r.ok and len(body["repos"]) >= 5 and all(x["issues"] for x in body["repos"]),
        f"{len(body.get('repos', []))} repos, {len(body.get('new_projects', []))} new projects",
    )

    owners = [x["name"].split("/")[0] for x in json.loads(Path(a.username_from).read_text())]
    user = None
    for login in dict.fromkeys(owners):
        t = requests.get(f"https://api.github.com/users/{login}", timeout=10)
        if t.ok and t.json().get("type") == "User":
            user = login
            break
    r = requests.post(f"{api}/recommend/github", json={"username": user, "hours": 5}, timeout=60)
    body = r.json()
    ok = r.ok and len(body["repos"]) >= 5 and user.lower() not in r.text.lower()
    check(
        "github path (username not echoed)",
        ok,
        f"{len(body.get('repos', []))} repos, retrieval={body.get('profile', {}).get('retrieval')}",
    )
    check(
        "github path rejects bad username",
        requests.post(
            f"{api}/recommend/github", json={"username": "no spaces!"}, timeout=30
        ).status_code
        == 422,
    )

    repo = body["repos"][0]
    e = requests.post(
        f"{api}/explain",
        json={
            "repo_id": repo["id"],
            "issue_number": repo["issues"][0]["number"],
            "co_starred": repo["signals"].get("co_starred_with", []),
            "skills": repo["signals"].get("matched_skills", []),
        },
        timeout=30,
    )
    check(
        "explain", e.ok and len(e.json().get("text", "")) > 20, f"source={e.json().get('source')}"
    )

    pre = {
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    }
    good = requests.options(
        f"{api}/recommend/onboarding", headers={"Origin": web, **pre}, timeout=30
    )
    check("cors: frontend origin allowed", good.headers.get("access-control-allow-origin") == web)
    bad = requests.options(
        f"{api}/recommend/onboarding", headers={"Origin": "https://example.org", **pre}, timeout=30
    )
    check(
        "cors: other origin refused",
        "access-control-allow-origin" not in {k.lower() for k in bad.headers},
    )

    w = requests.get(web, timeout=60)
    check("frontend loads", w.ok and "FirstPR" in w.text)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
