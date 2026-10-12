"""Turns MLOps events into agent invocations on AgentCore Runtime.

Event                                         -> agent role
--------------------------------------------------------------------------
Hourly schedule, latest monitor run has
  violations not yet triaged                  -> triage
CloudWatch alarm (endpoint 5XX) -> ALARM      -> triage
Training pipeline execution Succeeded         -> release
Training pipeline execution Failed            -> triage

Each event is de-duplicated in DynamoDB so a flapping alarm or a repeated
schedule tick never invokes the same agent run twice.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid

import boto3
from botocore.exceptions import ClientError

log = logging.getLogger()
log.setLevel(logging.INFO)
_clients: dict = {}


def client(name):
    if name not in _clients:
        _clients[name] = boto3.client(name)
    return _clients[name]


def table():
    return boto3.resource("dynamodb").Table(os.environ["OPS_TABLE"])


def claim(key: str) -> bool:
    """Return True the first time a key is seen (idempotency)."""
    try:
        table().put_item(Item={"pk": "DISPATCH", "sk": key, "at": int(time.time())},
                         ConditionExpression="attribute_not_exists(sk)")
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
            return False
        raise


def route(event: dict) -> tuple[str, str, str] | None:
    """Map an EventBridge event to (dedupe_key, role, prompt), or None to ignore."""
    source, dtype = event.get("source"), event.get("detail-type", "")
    detail = event.get("detail", {}) or {}

    if source == "aws.cloudwatch" and dtype == "CloudWatch Alarm State Change":
        if detail.get("state", {}).get("value") != "ALARM":
            return None
        name = detail.get("alarmName")
        ts = detail.get("state", {}).get("timestamp", event.get("time"))
        return (f"alarm/{name}/{ts}", "triage",
                f"CloudWatch alarm '{name}' entered ALARM at {ts}. Reason: "
                f"{detail.get('state', {}).get('reason')}. Investigate the fraud endpoint and decide "
                "whether this is an infrastructure problem, a data problem or model degradation.")

    if source == "aws.sagemaker" and dtype == "SageMaker Model Building Pipeline Execution Status Change":
        if detail.get("pipelineArn", "").split("/")[-1].lower() != os.environ["PIPELINE_NAME"].lower():
            return None
        status, arn = detail.get("currentPipelineExecutionStatus"), detail.get("pipelineExecutionArn")
        if status == "Succeeded":
            return (f"pipeline/{arn}/Succeeded", "release",
                    f"Training pipeline execution {arn} succeeded. Review the challenger it registered "
                    "and, only if the evidence supports it, request human approval for promotion.")
        if status == "Failed":
            return (f"pipeline/{arn}/Failed", "triage",
                    f"Training pipeline execution {arn} FAILED. Find the failing step and why, "
                    "and record an incident with a recommended fix.")
        return None

    if source == "aws.events" and dtype == "Scheduled Event":
        return check_monitor()

    return None


def check_monitor():
    schedule = os.environ["MONITOR_SCHEDULE"]
    try:
        execs = client("sagemaker").list_monitoring_executions(
            MonitoringScheduleName=schedule, SortBy="ScheduledTime", SortOrder="Descending",
            MaxResults=1)["MonitoringExecutionSummaries"]
    except ClientError:
        log.info("monitoring schedule not configured yet")
        return None
    if not execs or execs[0]["MonitoringExecutionStatus"] != "CompletedWithViolations":
        return None
    ts = execs[0]["ScheduledTime"]
    ts = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
    return (f"monitor/{schedule}/{ts}", "triage",
            f"Model Monitor run scheduled at {ts} completed WITH VIOLATIONS on the fraud endpoint. "
            "Determine whether this is a broken upstream feed or a genuine population/fraud-pattern "
            "shift, check label freshness, and act per your runbook.")


def invoke_agent(role: str, prompt: str) -> str:
    resp = client("bedrock-agentcore").invoke_agent_runtime(
        agentRuntimeArn=os.environ["AGENT_RUNTIME_ARN"],
        runtimeSessionId=f"{role}-{uuid.uuid4()}",  # must be >= 33 chars
        payload=json.dumps({"role": role, "prompt": prompt}).encode(),
        qualifier=os.environ.get("AGENT_QUALIFIER", "DEFAULT"),
    )
    body = resp.get("response")
    if hasattr(body, "read"):
        return body.read().decode()
    return "".join(c.decode() if isinstance(c, bytes) else str(c) for c in (body or []))


def lambda_handler(event, _context):
    routed = route(event)
    if not routed:
        return {"dispatched": False}
    key, role, prompt = routed
    if not claim(key):
        log.info(f"duplicate event {key}, skipping")
        return {"dispatched": False, "duplicate": key}
    log.info(json.dumps({"dispatch": key, "role": role}))
    result = invoke_agent(role, prompt)
    table().update_item(Key={"pk": "DISPATCH", "sk": key},
                        UpdateExpression="SET agent_role = :r, agent_result = :o",
                        ExpressionAttributeValues={":r": role, ":o": result[:20000]})
    return {"dispatched": True, "key": key, "role": role}
