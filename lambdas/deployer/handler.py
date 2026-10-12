"""Deterministic deployer: runs when a HUMAN approves a model package.

EventBridge rule: source aws.sagemaker, detail-type "SageMaker Model Package
State Change", ModelApprovalStatus Approved, our package group.

First approval  -> create the endpoint (with data capture for Model Monitor).
Later approvals -> blue/green canary update with automatic rollback on the
                   5XX alarm. No agent is involved in this path.
"""
from __future__ import annotations

import json
import logging
import os
import time

import boto3
from botocore.exceptions import ClientError

log = logging.getLogger()
log.setLevel(logging.INFO)
sm = boto3.client("sagemaker")
sns = boto3.client("sns")


class EndpointBusy(Exception):
    """Raised so EventBridge's async invoke retries later (up to ~6h by default)."""


def endpoint_status(name: str) -> str | None:
    try:
        return sm.describe_endpoint(EndpointName=name)["EndpointStatus"]
    except ClientError as e:
        if "Could not find endpoint" in str(e):
            return None
        raise


def build_endpoint_config(name: str, model_name: str, bucket: str, instance_type: str) -> dict:
    return {
        "EndpointConfigName": name,
        "ProductionVariants": [{
            "VariantName": "AllTraffic",
            "ModelName": model_name,
            "InitialInstanceCount": int(os.environ.get("INSTANCE_COUNT", "1")),
            "InstanceType": instance_type,
            "InitialVariantWeight": 1.0,
        }],
        "DataCaptureConfig": {
            "EnableCapture": True,
            "InitialSamplingPercentage": int(os.environ.get("CAPTURE_PERCENT", "100")),
            "DestinationS3Uri": f"s3://{bucket}/datacapture",
            "CaptureOptions": [{"CaptureMode": "Input"}, {"CaptureMode": "Output"}],
            "CaptureContentTypeHeader": {"CsvContentTypes": ["text/csv"]},
        },
    }


def canary_deployment_config(alarm_name: str) -> dict:
    return {
        "BlueGreenUpdatePolicy": {
            "TrafficRoutingConfiguration": {
                "Type": "CANARY",
                "CanarySize": {"Type": "CAPACITY_PERCENT", "Value": 10},
                "WaitIntervalInSeconds": 600,
            },
            "TerminationWaitInSeconds": 300,
            "MaximumExecutionTimeoutInSeconds": 3600,
        },
        "AutoRollbackConfiguration": {"Alarms": [{"AlarmName": alarm_name}]},
    }


def lambda_handler(event, _context):
    detail = event.get("detail", {})
    log.info(json.dumps(detail, default=str))
    if detail.get("ModelApprovalStatus") != "Approved":
        return {"skipped": "not an approval"}
    if detail.get("ModelPackageGroupName") != os.environ["MODEL_PACKAGE_GROUP"]:
        return {"skipped": "other group"}

    arn = detail["ModelPackageArn"]
    # Never trust the event payload alone: confirm with SageMaker that a human approved it.
    pkg = sm.describe_model_package(ModelPackageName=arn)
    if pkg.get("ModelApprovalStatus") != "Approved" or \
            pkg.get("ModelPackageGroupName") != os.environ["MODEL_PACKAGE_GROUP"]:
        log.warning(json.dumps({"refused": arn, "status": pkg.get("ModelApprovalStatus")}))
        return {"skipped": "package is not approved in this group"}

    endpoint = os.environ["ENDPOINT_NAME"]
    status = endpoint_status(endpoint)
    if status not in (None, "InService", "Failed"):
        # another rollout is in progress; fail so the async invocation is retried
        raise EndpointBusy(f"{endpoint} is {status}; will retry {arn}")
    suffix = f"v{arn.rsplit('/', 1)[-1]}-{int(time.time())}"
    model_name = f"fraud-model-{suffix}"
    config_name = f"fraud-epc-{suffix}"

    sm.create_model(ModelName=model_name, ExecutionRoleArn=os.environ["SAGEMAKER_ROLE_ARN"],
                    PrimaryContainer={"ModelPackageName": arn})
    sm.create_endpoint_config(**build_endpoint_config(
        config_name, model_name, os.environ["BUCKET"], os.environ.get("INSTANCE_TYPE", "ml.m5.large")))

    if status is not None:
        sm.update_endpoint(EndpointName=endpoint, EndpointConfigName=config_name,
                           DeploymentConfig=canary_deployment_config(os.environ["ROLLBACK_ALARM_NAME"]))
        action = "blue/green canary update (10% for 10 min, auto-rollback on 5XX alarm)"
    else:
        sm.create_endpoint(EndpointName=endpoint, EndpointConfigName=config_name)
        action = "initial endpoint creation"

    table = boto3.resource("dynamodb").Table(os.environ["OPS_TABLE"])
    table.put_item(Item={"pk": "DEPLOYMENT", "sk": f"{int(time.time())}", "model_package_arn": arn,
                         "endpoint_config": config_name, "action": action})
    if os.environ.get("NOTIFY_TOPIC_ARN"):
        sns.publish(TopicArn=os.environ["NOTIFY_TOPIC_ARN"], Subject="[fraud-mlops] Deployment started",
                    Message=f"{action}\nEndpoint: {endpoint}\nModel package: {arn}\nConfig: {config_name}")
    return {"endpoint": endpoint, "config": config_name, "action": action}
