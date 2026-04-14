"""GitHub Action entrypoint.

Reads trace JSON files from TRACES_DIR, runs analysis on each,
then posts results as a PR comment via the GitHub API.

Environment variables (set by action.yml):
  TRACES_DIR          — directory containing *.trace.json files (default: agent_traces)
  GITHUB_TOKEN        — GitHub token for posting PR comments
  GITHUB_REPOSITORY   — e.g. "owner/repo" (set automatically by GitHub Actions)
  GITHUB_EVENT_NAME   — e.g. "pull_request"
  GITHUB_EVENT_PATH   — path to event JSON
  AGENT_DEBUG_PROVIDER — which LLM provider to use (default: auto-detect)
  ANTHROPIC_API_KEY / OPENAI_API_KEY / etc.
"""

import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path


def main() -> int:
    traces_dir = Path(os.environ.get("TRACES_DIR", "agent_traces"))
    github_token = os.environ.get("GITHUB_TOKEN", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    event_path = os.environ.get("GITHUB_EVENT_PATH", "")

    # Find trace files
    trace_files = list(traces_dir.glob("**/*.trace.json")) + list(
        traces_dir.glob("**/*.json")
    ) if traces_dir.exists() else []

    if not trace_files:
        print(f"[agent-debug] No trace files found in {traces_dir}/")
        print("[agent-debug] Tip: save your trace as agent_traces/my_agent.trace.json")
        return 0

    print(f"[agent-debug] Found {len(trace_files)} trace file(s)")

    # Run analysis on each trace
    from agent_debug.pipeline import DiagnosisPipeline
    from agent_debug.formatters.markdown import report_to_markdown

    pipeline = DiagnosisPipeline()
    comment_sections = []
    any_failure = False

    for trace_file in trace_files:
        print(f"[agent-debug] Analyzing {trace_file.name}...")
        try:
            raw = json.loads(trace_file.read_text())
            report = pipeline.run(raw)
            md = report_to_markdown(report, trace_file=str(trace_file))
            comment_sections.append(md)

            sev = report["severity"]["severity"]
            if sev >= 3:
                any_failure = True

            print(
                f"[agent-debug] ✓ {trace_file.name}: "
                f"{report['classification']['subcategory']} "
                f"severity={sev}/5"
            )
        except Exception as e:
            print(f"[agent-debug] ✗ Error analyzing {trace_file.name}: {e}")
            comment_sections.append(
                f"## 🤖 agent-debug\n\n"
                f"❌ Failed to analyze `{trace_file.name}`:\n```\n{e}\n```"
            )

    # Post PR comment if we're in a PR context
    if github_token and repo and event_name in ("pull_request", "pull_request_target"):
        pr_number = _get_pr_number(event_path)
        if pr_number:
            full_comment = "\n\n---\n\n".join(comment_sections)
            _post_or_update_comment(github_token, repo, pr_number, full_comment)
        else:
            print("[agent-debug] Could not determine PR number, skipping comment.")
    else:
        # Not in a PR — just print to stdout
        for section in comment_sections:
            print("\n" + section)

    return 1 if any_failure else 0


def _get_pr_number(event_path: str) -> int | None:
    if not event_path or not Path(event_path).exists():
        return None
    try:
        event = json.loads(Path(event_path).read_text())
        return event.get("pull_request", {}).get("number")
    except Exception:
        return None


def _post_or_update_comment(
    token: str, repo: str, pr_number: int, body: str
) -> None:
    """Post a new comment or update the existing agent-debug comment on the PR."""
    base = f"https://api.github.com/repos/{repo}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    marker = "<!-- agent-debug-comment -->"
    full_body = f"{marker}\n{body}"

    # Look for existing comment to update
    existing_id = _find_existing_comment(base, pr_number, headers, marker)

    if existing_id:
        url = f"{base}/issues/comments/{existing_id}"
        method = "PATCH"
        print(f"[agent-debug] Updating existing PR comment #{existing_id}")
    else:
        url = f"{base}/issues/{pr_number}/comments"
        method = "POST"
        print(f"[agent-debug] Posting new PR comment on PR #{pr_number}")

    data = json.dumps({"body": full_body}).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
            print(f"[agent-debug] Comment posted: {result.get('html_url', '')}")
    except urllib.error.HTTPError as e:
        print(f"[agent-debug] Failed to post comment: {e.code} {e.reason}")


def _find_existing_comment(
    base: str, pr_number: int, headers: dict, marker: str
) -> int | None:
    url = f"{base}/issues/{pr_number}/comments?per_page=100"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            comments = json.loads(resp.read())
            for c in comments:
                if marker in c.get("body", ""):
                    return c["id"]
    except Exception:
        pass
    return None


if __name__ == "__main__":
    sys.exit(main())
