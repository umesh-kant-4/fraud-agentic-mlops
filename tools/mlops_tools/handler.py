"""Lambda target behind AgentCore Gateway: MLOps tools for the fraud agents.

Gateway calls this function with the tool arguments as `event` and the tool
name in `context.client_context.custom["bedrockAgentCoreToolName"]`, prefixed
with the target name and "___". Every tool returns a JSON-serialisable dict.

Guardrails live HERE, not in the agent prompt, so a confused or manipulated
agent still cannot exceed them.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError

log = logging.getLogger()
log.setLevel(logging.INFO)

DELIMITER = "___"
_clients: dict = {}


def client(name: str):
    if name not in _clients:
        _clients[name] = boto3.client(name)
    return _clients[name]


def cfg(key: str, default: str | None = None) -> str:
    val = os.environ.get(key, default)
    if val is None:
        raise RuntimeError(f"missing env var {key}")
    return val


class ToolError(Exception):
    """Expected, user-facing refusal (guardrail or bad input)."""


# --------------------------------------------------------------------- helpers

def tool_name_from_context(context) -> str:
    custom = getattr(getattr(context, "client_context", None), "custom", None) or {}
    raw = custom.get("bedrockAgentCoreToolName", "")
    return raw.split(DELIMITER, 1)[1] if DELIMITER in raw else raw


def _iso(v):
    return v.isoformat() if isinstance(v, datetime) else v


def metadata_safe(text: str, limit: int = 250) -> str:
    """Model Registry CustomerMetadataProperties only allow [\\p{L}\\p{Z}\\p{N}_.:/=+\\-@]."""
    cleaned = re.sub(r"[^\w .:/=+\-@]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", cleaned).strip()[:limit] or "agent-retrain"


# Captured endpoint CSV has no header, so Model Monitor names columns _c0.._cN.
# The baseline is built header-less too; map positions back to feature names here.
FEATURES = [
    "amount", "hour_of_day", "merchant_category", "distance_from_home_km",
    "txn_count_1h", "txn_count_24h", "avg_amount_30d", "amount_to_avg_ratio",
    "is_foreign", "card_present", "account_age_days", "new_device",
]


def feature_label(col: str | None) -> str | None:
    m = re.fullmatch(r"_c(\d+)", col or "")
    if m and int(m.group(1)) < len(FEATURES):
        return f"{FEATURES[int(m.group(1))]} ({col})"
    return col


def _clamp(v, default: int, lo: int, hi: int) -> int:
    try:
        v = int(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def read_s3_json(uri: str) -> dict:
    u = urlparse(uri)
    body = client("s3").get_object(Bucket=u.netloc, Key=u.path.lstrip("/"))["Body"].read()
    return json.loads(body)


def feature_shift(baseline: dict, current: dict) -> list[dict]:
    """Compare Model Monitor statistics.json files feature by feature."""
    def index(stats):
        out = {}
        for f in stats.get("features", []):
            num = f.get("numerical_statistics", {})
            common = num.get("common", {})
            total = common.get("num_present", 0) + common.get("num_missing", 0)
            out[f["name"]] = {
                "mean": num.get("mean"),
                "std": num.get("std_dev"),
                "min": num.get("min"),
                "max": num.get("max"),
                "missing_rate": (common.get("num_missing", 0) / total) if total else None,
                "type": f.get("inferred_type"),
            }
        return out

    b, c = index(baseline), index(current)
    rows = []
    for name in sorted(set(b) | set(c)):
        bs, cs = b.get(name, {}), c.get(name, {})
        shift = None
        if bs.get("mean") is not None and cs.get("mean") is not None and bs.get("std"):
            shift = round((cs["mean"] - bs["mean"]) / bs["std"], 3)
        rows.append({
            "feature": feature_label(name),
            "baseline_mean": bs.get("mean"), "current_mean": cs.get("mean"),
            "mean_shift_in_baseline_std": shift,
            "baseline_missing_rate": bs.get("missing_rate"), "current_missing_rate": cs.get("missing_rate"),
            "current_std": cs.get("std"),
            "type_changed": bool(bs and cs and bs.get("type") != cs.get("type")),
        })
    rows.sort(key=lambda r: -abs(r["mean_shift_in_baseline_std"] or 0))
    return rows


def champion_package():
    pkgs = client("sagemaker").list_model_packages(
        ModelPackageGroupName=cfg("MODEL_PACKAGE_GROUP"), ModelApprovalStatus="Approved",
        SortBy="CreationTime", SortOrder="Descending", MaxResults=1)["ModelPackageSummaryList"]
    return pkgs[0] if pkgs else None


def evaluation_for(package_arn: str) -> dict | None:
    pkg = client("sagemaker").describe_model_package(ModelPackageName=package_arn)
    uri = (pkg.get("ModelMetrics", {}).get("ModelQuality", {}) or {}).get("Statistics", {}).get("S3Uri") \
        or (pkg.get("ModelMetrics", {}).get("ModelDataQuality", {}) or {}).get("Statistics", {}).get("S3Uri")
    if not uri:
        # RegisterModel(model_statistics=...) lands under ModelQuality.Statistics
        return None
    return read_s3_json(uri)


def notify(subject: str, message: str) -> None:
    topic = os.environ.get("NOTIFY_TOPIC_ARN")
    if topic:
        client("sns").publish(TopicArn=topic, Subject=subject[:99], Message=message)


def table():
    return boto3.resource("dynamodb").Table(cfg("OPS_TABLE"))


# ----------------------------------------------------------------- read tools

def get_endpoint_status(_args):
    try:
        ep = client("sagemaker").describe_endpoint(EndpointName=cfg("ENDPOINT_NAME"))
    except ClientError as e:
        if "Could not find endpoint" in str(e):
            return {"endpoint": cfg("ENDPOINT_NAME"), "status": "NotDeployed"}
        raise
    return {
        "endpoint": ep["EndpointName"],
        "status": ep["EndpointStatus"],
        "failure_reason": ep.get("FailureReason"),
        "last_modified": _iso(ep.get("LastModifiedTime")),
        "config": ep.get("EndpointConfigName"),
        "variants": [{"name": v["VariantName"],
                      "instances": v.get("CurrentInstanceCount"),
                      "weight": v.get("CurrentWeight")} for v in ep.get("ProductionVariants", [])],
        "data_capture": ep.get("DataCaptureConfig", {}).get("EnableCapture"),
    }


def get_endpoint_metrics(args):
    hours = _clamp(args.get("hours"), 24, 1, 168)
    end = datetime.now(timezone.utc)
    dims = [{"Name": "EndpointName", "Value": cfg("ENDPOINT_NAME")},
            {"Name": "VariantName", "Value": "AllTraffic"}]
    queries = [
        ("invocations", "Invocations", "Sum"),
        ("errors_5xx", "Invocation5XXErrors", "Sum"),
        ("errors_4xx", "Invocation4XXErrors", "Sum"),
        ("latency_p50_us", "ModelLatency", "p50"),
        ("latency_p99_us", "ModelLatency", "p99"),
    ]
    resp = client("cloudwatch").get_metric_data(
        StartTime=end - timedelta(hours=hours), EndTime=end,
        MetricDataQueries=[{
            "Id": qid, "ReturnData": True,
            "MetricStat": {"Metric": {"Namespace": "AWS/SageMaker", "MetricName": m, "Dimensions": dims},
                           "Period": 3600, "Stat": stat},
        } for qid, m, stat in queries])
    out = {"endpoint": cfg("ENDPOINT_NAME"), "hours": hours, "series": {}}
    for r in resp["MetricDataResults"]:
        pts = sorted(zip(r["Timestamps"], r["Values"]))
        out["series"][r["Id"]] = [{"t": _iso(t), "v": round(v, 2)} for t, v in pts]
    totals = {k: sum(p["v"] for p in v) for k, v in out["series"].items() if not k.startswith("latency")}
    out["totals"] = totals
    inv = totals.get("invocations") or 0
    out["error_rate_5xx"] = round(totals.get("errors_5xx", 0) / inv, 5) if inv else None
    return out


def get_monitoring_report(args):
    sm = client("sagemaker")
    schedule = cfg("MONITOR_SCHEDULE")
    limit = _clamp(args.get("limit"), 3, 1, 10)
    try:
        sched = sm.describe_monitoring_schedule(MonitoringScheduleName=schedule)
    except ClientError:
        return {"schedule": schedule, "status": "NotConfigured",
                "hint": "run ml/monitor/setup_monitor.py after the first deployment"}
    execs = sm.list_monitoring_executions(MonitoringScheduleName=schedule, SortBy="ScheduledTime",
                                          SortOrder="Descending", MaxResults=limit)
    summaries = [{"scheduled": _iso(e["ScheduledTime"]), "status": e["MonitoringExecutionStatus"],
                  "processing_job_arn": e.get("ProcessingJobArn"),
                  "failure_reason": e.get("FailureReason")}
                 for e in execs["MonitoringExecutionSummaries"]]
    report = {"schedule": schedule, "executions": summaries, "latest_violations": None}

    latest_bad = next((s for s in summaries if s["status"] == "CompletedWithViolations"
                       and s["processing_job_arn"]), None)
    if latest_bad:
        job = sm.describe_processing_job(ProcessingJobName=latest_bad["processing_job_arn"].split("/")[-1])
        out_uri = job["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
        violations = read_s3_json(f"{out_uri}/constraint_violations.json").get("violations", [])
        report["latest_violations"] = {
            "scheduled": latest_bad["scheduled"],
            "violations": [{"feature": feature_label(v.get("feature_name")),
                            "check": v.get("constraint_check_type"),
                            "description": v.get("description")} for v in violations][:50],
        }
        baseline_uri = (sched["MonitoringScheduleConfig"].get("MonitoringJobDefinition", {})
                        .get("BaselineConfig", {}).get("StatisticsResource", {}).get("S3Uri"))
        if not baseline_uri:  # schedules created from a job definition keep the baseline there
            jd_name = sched["MonitoringScheduleConfig"].get("MonitoringJobDefinitionName")
            if jd_name:
                jd = sm.describe_data_quality_job_definition(JobDefinitionName=jd_name)
                baseline_uri = jd.get("DataQualityBaselineConfig", {}).get("StatisticsResource", {}).get("S3Uri")
        if baseline_uri:
            report["latest_violations"]["feature_comparison"] = feature_shift(
                read_s3_json(baseline_uri), read_s3_json(f"{out_uri}/statistics.json"))[:15]
    return report


def check_data_freshness(_args):
    bucket, prefix = cfg("BUCKET"), cfg("RAW_DATA_PREFIX", "data/")
    objs, token = [], None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kw["ContinuationToken"] = token
        page = client("s3").list_objects_v2(**kw)
        objs += page.get("Contents", [])
        if not page.get("IsTruncated") or len(objs) >= 5000:
            break
        token = page["NextContinuationToken"]
    objs.sort(key=lambda o: o["LastModified"], reverse=True)
    now = datetime.now(timezone.utc)
    newest = objs[0]["LastModified"] if objs else None
    return {
        "prefix": f"s3://{bucket}/{prefix}",
        "object_count": len(objs),
        "newest_object_age_hours": round((now - newest).total_seconds() / 3600, 1) if newest else None,
        "newest": [{"key": o["Key"], "modified": _iso(o["LastModified"]), "bytes": o["Size"]}
                   for o in objs[:10]],
    }


def list_model_versions(args):
    n = _clamp(args.get("max_results"), 5, 1, 20)
    sm = client("sagemaker")
    pkgs = sm.list_model_packages(ModelPackageGroupName=cfg("MODEL_PACKAGE_GROUP"),
                                  SortBy="CreationTime", SortOrder="Descending",
                                  MaxResults=n)["ModelPackageSummaryList"]
    champ = champion_package()
    out = []
    for p in pkgs:
        detail = sm.describe_model_package(ModelPackageName=p["ModelPackageArn"])
        try:
            ev = evaluation_for(p["ModelPackageArn"]) or {}
        except ClientError:
            ev = {}
        m = ev.get("metrics", {})
        out.append({
            "arn": p["ModelPackageArn"],
            "version": p.get("ModelPackageVersion"),
            "status": p["ModelApprovalStatus"],
            "created": _iso(p["CreationTime"]),
            "is_champion": bool(champ and champ["ModelPackageArn"] == p["ModelPackageArn"]),
            "trigger_reason": detail.get("CustomerMetadataProperties", {}).get("trigger_reason"),
            "aucpr": m.get("aucpr"),
            "recall_at_alert_rate": m.get("recall_at_alert_rate"),
            "dollar_recall_at_alert_rate": m.get("dollar_recall_at_alert_rate"),
            "release_policy": ev.get("comparison", {}).get("reasons"),
        })
    return {"group": cfg("MODEL_PACKAGE_GROUP"), "versions": out}


def _own_package(arn: str) -> dict:
    pkg = client("sagemaker").describe_model_package(ModelPackageName=arn)
    if pkg.get("ModelPackageGroupName") != cfg("MODEL_PACKAGE_GROUP"):
        raise ToolError("model package is not in this project's model package group")
    return pkg


def get_model_evaluation(args):
    arn = args.get("model_package_arn") or ""
    pkg = _own_package(arn)
    ev = evaluation_for(arn)
    return {"arn": arn, "status": pkg["ModelApprovalStatus"], "evaluation": ev}


def list_pipeline_executions(args):
    n = _clamp(args.get("max_results"), 5, 1, 20)
    sm = client("sagemaker")
    execs = sm.list_pipeline_executions(PipelineName=cfg("PIPELINE_NAME"), SortBy="CreationTime",
                                        SortOrder="Descending", MaxResults=n)["PipelineExecutionSummaries"]
    out = []
    for e in execs:
        d = sm.describe_pipeline_execution(PipelineExecutionArn=e["PipelineExecutionArn"])
        out.append({"arn": e["PipelineExecutionArn"], "status": e["PipelineExecutionStatus"],
                    "started": _iso(e.get("StartTime")), "last_modified": _iso(d.get("LastModifiedTime")),
                    "description": d.get("PipelineExecutionDescription"),
                    "failure_reason": d.get("FailureReason")})
    return {"pipeline": cfg("PIPELINE_NAME"), "executions": out}


def get_pipeline_execution(args):
    arn = args.get("execution_arn") or ""
    if f":pipeline/{cfg('PIPELINE_NAME').lower()}/" not in arn.lower():
        raise ToolError("execution does not belong to this project's pipeline")
    sm = client("sagemaker")
    d = sm.describe_pipeline_execution(PipelineExecutionArn=arn)
    steps = sm.list_pipeline_execution_steps(PipelineExecutionArn=arn)["PipelineExecutionSteps"]
    out_steps, registered = [], None
    for s in steps:
        meta = s.get("Metadata", {})
        if "RegisterModel" in meta:
            registered = meta["RegisterModel"].get("Arn")
        if "Condition" in meta:
            meta_summary = {"condition_outcome": meta["Condition"].get("Outcome")}
        else:
            meta_summary = {k: v.get("Arn") for k, v in meta.items() if isinstance(v, dict) and "Arn" in v}
        out_steps.append({"step": s["StepName"], "status": s["StepStatus"],
                          "failure_reason": s.get("FailureReason"), **meta_summary})
    return {"arn": arn, "status": d["PipelineExecutionStatus"], "failure_reason": d.get("FailureReason"),
            "registered_model_package": registered, "steps": out_steps}


# ---------------------------------------------------------------- write tools

def start_retraining(args):
    reason = (args.get("reason") or "").strip()
    if len(reason) < 20:
        raise ToolError("reason must explain the evidence for retraining (min 20 chars)")
    bucket = cfg("BUCKET")
    input_uri = (args.get("input_data_uri") or f"s3://{bucket}/{cfg('RAW_DATA_PREFIX', 'data/')}raw/").strip()
    if not input_uri.startswith(f"s3://{bucket}/data/"):
        raise ToolError(f"input_data_uri must be under s3://{bucket}/data/")

    sm = client("sagemaker")
    running = sm.list_pipeline_executions(PipelineName=cfg("PIPELINE_NAME"), SortBy="CreationTime",
                                          SortOrder="Descending", MaxResults=5)["PipelineExecutionSummaries"]
    if any(e["PipelineExecutionStatus"] in ("Executing", "Stopping") for e in running):
        raise ToolError("a pipeline execution is already running; not starting another")

    # cooldown lock: conditional write fails if a retrain started within the window
    cooldown_h = float(cfg("RETRAIN_COOLDOWN_HOURS", "6"))
    now = int(time.time())
    try:
        table().put_item(
            Item={"pk": "LOCK", "sk": "retrain", "started_at": now, "reason": reason[:1000]},
            ConditionExpression="attribute_not_exists(pk) OR started_at < :cutoff",
            ExpressionAttributeValues={":cutoff": now - int(cooldown_h * 3600)},
        )
    except ClientError as e:
        if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
            raise ToolError(f"retraining cooldown active ({cooldown_h}h); a human can start the "
                            "pipeline manually if this is urgent")
        raise

    champ = champion_package()
    champion_uri = "none"
    if champ:
        d = sm.describe_model_package(ModelPackageName=champ["ModelPackageArn"])
        champion_uri = d["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]

    try:
        resp = sm.start_pipeline_execution(
            PipelineName=cfg("PIPELINE_NAME"),
            PipelineExecutionDisplayName=f"agent-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}",
            PipelineExecutionDescription=f"Started by agent: {reason}"[:3072],
            PipelineParameters=[
                {"Name": "InputDataUri", "Value": input_uri},
                {"Name": "ChampionModelUri", "Value": champion_uri},
                {"Name": "TriggerReason", "Value": metadata_safe(reason)},
            ],
            ClientRequestToken=str(uuid.uuid4()),
        )
    except Exception:
        # don't leave a cooldown lock behind when nothing actually started
        table().delete_item(Key={"pk": "LOCK", "sk": "retrain"},
                            ConditionExpression="started_at = :now",
                            ExpressionAttributeValues={":now": now})
        raise
    notify("[fraud-mlops] Retraining started by agent",
           f"Reason: {reason}\nInput: {input_uri}\nExecution: {resp['PipelineExecutionArn']}")
    return {"started": True, "execution_arn": resp["PipelineExecutionArn"],
            "input_data_uri": input_uri, "champion_model_uri": champion_uri,
            "note": "result will be registered as PendingManualApproval if it passes the quality gate"}


def record_incident(args):
    severity = (args.get("severity") or "").lower()
    if severity not in {"low", "medium", "high"}:
        raise ToolError("severity must be low, medium or high")
    for f in ("title", "summary", "recommended_action"):
        if not (args.get(f) or "").strip():
            raise ToolError(f"{f} is required")
    incident_id = f"inc-{datetime.now(timezone.utc):%Y%m%d%H%M%S}-{uuid.uuid4().hex[:6]}"
    item = {"pk": "INCIDENT", "sk": incident_id, "title": args["title"][:200], "severity": severity,
            "summary": args["summary"][:5000], "recommended_action": args["recommended_action"][:2000],
            "created_at": int(time.time()), "status": "open"}
    table().put_item(Item=item)
    notify(f"[fraud-mlops][{severity}] {args['title']}",
           f"Incident {incident_id}\n\n{args['summary']}\n\nRecommended action:\n{args['recommended_action']}")
    return {"incident_id": incident_id, "notified": bool(os.environ.get("NOTIFY_TOPIC_ARN"))}


def request_promotion(args):
    arn = args.get("model_package_arn") or ""
    summary = (args.get("summary") or "").strip()
    if len(summary) < 40:
        raise ToolError("summary must make the promotion case (min 40 chars)")
    pkg = _own_package(arn)
    if pkg["ModelApprovalStatus"] != "PendingManualApproval":
        raise ToolError(f"package is {pkg['ModelApprovalStatus']}, not PendingManualApproval")
    ev = evaluation_for(arn) or {}
    if not ev.get("comparison", {}).get("passes"):
        raise ToolError("package did not pass the release policy; cannot request promotion")

    try:
        table().put_item(
            Item={"pk": "PROMOTION", "sk": arn, "summary": summary[:5000], "status": "requested",
                  "requested_at": int(time.time())},
            ConditionExpression="attribute_not_exists(sk)")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
            return {"requested": False, "note": "promotion already requested for this package"}
        raise
    notify("[fraud-mlops] Model promotion requested",
           f"{summary}\n\nModel package: {arn}\n\nTo approve (deploys via blue/green canary with "
           f"auto-rollback):\n  python scripts/approve.py --model-package-arn {arn} --note \"...\"\n"
           f"To reject:\n  python scripts/approve.py --model-package-arn {arn} --reject --note \"...\"")
    return {"requested": True, "model_package_arn": arn, "metrics": ev.get("metrics"),
            "deltas": ev.get("comparison", {}).get("deltas")}


HANDLERS = {
    "get_endpoint_status": get_endpoint_status,
    "get_endpoint_metrics": get_endpoint_metrics,
    "get_monitoring_report": get_monitoring_report,
    "check_data_freshness": check_data_freshness,
    "list_model_versions": list_model_versions,
    "get_model_evaluation": get_model_evaluation,
    "list_pipeline_executions": list_pipeline_executions,
    "get_pipeline_execution": get_pipeline_execution,
    "start_retraining": start_retraining,
    "record_incident": record_incident,
    "request_promotion": request_promotion,
}


def lambda_handler(event, context):
    name = tool_name_from_context(context)
    args = event if isinstance(event, dict) else {}
    log.info(json.dumps({"tool": name, "args": {k: str(v)[:200] for k, v in args.items()}}))
    fn = HANDLERS.get(name)
    if not fn:
        return {"error": f"unknown tool '{name}'"}
    try:
        return json.loads(json.dumps(fn(args), default=str))
    except ToolError as e:
        log.warning(json.dumps({"tool": name, "refused": str(e)}))
        return {"error": str(e), "refused_by_guardrail": True}
    except ClientError as e:
        log.exception("AWS error")
        return {"error": f"AWS error: {e.response['Error'].get('Code')}: {e.response['Error'].get('Message')}"}
