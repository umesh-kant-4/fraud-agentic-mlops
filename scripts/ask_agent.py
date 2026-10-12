"""Talk to the deployed agent directly (ops role is read-only).

  python scripts/ask_agent.py --runtime-arn arn:... "Why did the monitor alert last night?"
  python scripts/ask_agent.py --runtime-arn arn:... --role triage "Run triage now"
"""
from __future__ import annotations

import argparse
import json
import uuid

import boto3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--runtime-arn", required=True)
    ap.add_argument("--role", default="ops", choices=["ops", "triage", "release"])
    args = ap.parse_args()

    resp = boto3.client("bedrock-agentcore").invoke_agent_runtime(
        agentRuntimeArn=args.runtime_arn,
        runtimeSessionId=f"cli-{uuid.uuid4()}",
        payload=json.dumps({"role": args.role, "prompt": args.prompt}).encode(),
        qualifier="DEFAULT",
    )
    body = resp["response"]
    raw = body.read().decode() if hasattr(body, "read") else "".join(c.decode() for c in body)
    try:
        print(json.loads(raw).get("report", raw))
    except (json.JSONDecodeError, AttributeError):
        print(raw)


if __name__ == "__main__":
    main()
