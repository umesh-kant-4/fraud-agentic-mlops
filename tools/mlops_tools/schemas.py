"""MCP tool definitions exposed through AgentCore Gateway (Lambda target).

Kept in Python so the CDK stack, the Lambda router and the unit tests all
read the same list. Only the subset of JSON Schema that Gateway's
ToolDefinition supports is used (type, description, properties, required, items).

There is deliberately NO tool that approves a model or changes the prod
endpoint. Promotion is a human action; deployment is a deterministic Lambda.
"""

READ_ONLY = {
    "get_endpoint_status",
    "get_endpoint_metrics",
    "get_monitoring_report",
    "check_data_freshness",
    "list_model_versions",
    "get_model_evaluation",
    "list_pipeline_executions",
    "get_pipeline_execution",
}
WRITE = {"start_retraining", "record_incident", "request_promotion"}


def _obj(props: dict | None = None, required: list | None = None) -> dict:
    schema: dict = {"type": "object", "properties": props or {}}
    if required:
        schema["required"] = required
    return schema


TOOLS = [
    {
        "name": "get_endpoint_status",
        "description": "Status of the production fraud endpoint: state, production variants, "
                       "model names, instance counts, last update time and any failure reason.",
        "inputSchema": _obj(),
    },
    {
        "name": "get_endpoint_metrics",
        "description": "Hourly invocations, 5XX/4XX errors and p50/p99 model latency for the "
                       "production endpoint over the last N hours (default 24, max 168).",
        "inputSchema": _obj({"hours": {"type": "integer", "description": "Look-back window in hours."}}),
    },
    {
        "name": "get_monitoring_report",
        "description": "Latest Model Monitor data-quality executions. For the most recent one with "
                       "violations, returns each violated constraint plus per-feature mean/std/missing "
                       "rate compared with the training baseline. Use this to tell a real population "
                       "shift apart from a broken upstream feed (nulls, constant values, type changes).",
        "inputSchema": _obj({"limit": {"type": "integer", "description": "Executions to list (default 3)."}}),
    },
    {
        "name": "check_data_freshness",
        "description": "Newest objects and their timestamps under the raw training-data prefix. "
                       "Fraud labels arrive late (chargebacks take weeks); use this to judge whether "
                       "enough newly labelled data exists for retraining to help.",
        "inputSchema": _obj(),
    },
    {
        "name": "list_model_versions",
        "description": "Recent versions in the fraud model package group with approval status, "
                       "trigger reason and headline evaluation metrics. Identifies the current champion "
                       "(latest Approved).",
        "inputSchema": _obj({"max_results": {"type": "integer", "description": "Default 5, max 20."}}),
    },
    {
        "name": "get_model_evaluation",
        "description": "Full evaluation report for one model package: challenger metrics, champion "
                       "metrics on the same test window, deltas and release-policy verdict.",
        "inputSchema": _obj({"model_package_arn": {"type": "string",
                                                   "description": "ARN of the model package."}},
                            ["model_package_arn"]),
    },
    {
        "name": "list_pipeline_executions",
        "description": "Recent training pipeline executions with status, start/end time, trigger "
                       "reason and failure reason.",
        "inputSchema": _obj({"max_results": {"type": "integer", "description": "Default 5, max 20."}}),
    },
    {
        "name": "get_pipeline_execution",
        "description": "Step-by-step status of one pipeline execution, including the quality-gate "
                       "outcome and the model package it registered, if any.",
        "inputSchema": _obj({"execution_arn": {"type": "string",
                                               "description": "Pipeline execution ARN."}},
                            ["execution_arn"]),
    },
    {
        "name": "start_retraining",
        "description": "Start the training pipeline. Guardrails enforced server-side: refuses if an "
                       "execution is already running or one was started within the cooldown window, "
                       "and only accepts input data under the project bucket's data/ prefix. The new "
                       "model is registered as PendingManualApproval; this never touches production.",
        "inputSchema": _obj({
            "reason": {"type": "string", "description": "Why retraining is warranted, citing evidence "
                                                        "(e.g. incident id, violated features)."},
            "input_data_uri": {"type": "string", "description": "Optional s3:// prefix under the "
                                                                "project bucket data/ folder."},
        }, ["reason"]),
    },
    {
        "name": "record_incident",
        "description": "Record an MLOps incident and notify the on-call humans by email.",
        "inputSchema": _obj({
            "title": {"type": "string", "description": "Short title."},
            "severity": {"type": "string", "description": "One of: low, medium, high."},
            "summary": {"type": "string", "description": "What happened and the evidence."},
            "recommended_action": {"type": "string", "description": "What the agent recommends next."},
        }, ["title", "severity", "summary", "recommended_action"]),
    },
    {
        "name": "request_promotion",
        "description": "Ask a human to approve a model package for production. Verifies the package "
                       "is in this project's group, is PendingManualApproval and passed the release "
                       "policy, then emails the approvers. Does NOT approve or deploy anything.",
        "inputSchema": _obj({
            "model_package_arn": {"type": "string", "description": "Challenger model package ARN."},
            "summary": {"type": "string", "description": "The promotion case: metrics vs champion, "
                                                         "risks, and what to watch after release."},
        }, ["model_package_arn", "summary"]),
    },
]


def to_cfn_schema(s: dict) -> dict:
    """MCP-style (lowercase) JSON schema -> CloudFormation SchemaDefinition (PascalCase)."""
    out = {"Type": s["type"]}
    if "description" in s:
        out["Description"] = s["description"]
    if "properties" in s:
        out["Properties"] = {k: to_cfn_schema(v) for k, v in s["properties"].items()}
    if "required" in s:
        out["Required"] = s["required"]
    if "items" in s:
        out["Items"] = to_cfn_schema(s["items"])
    return out


def to_cfn_tools(tools: list[dict]) -> list[dict]:
    return [{"Name": t["name"], "Description": t["description"],
             "InputSchema": to_cfn_schema(t["inputSchema"])} for t in tools]



TOOL_NAMES = {t["name"] for t in TOOLS}
assert TOOL_NAMES == READ_ONLY | WRITE, "schemas and permission sets out of sync"
