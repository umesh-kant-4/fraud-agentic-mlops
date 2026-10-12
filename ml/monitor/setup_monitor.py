"""Baseline + hourly data-quality monitoring schedule for the prod endpoint.

Run once after the first model is approved and the deployer Lambda has created
the endpoint (data capture is enabled by the deployer).

  python ml/monitor/setup_monitor.py
"""
from __future__ import annotations

import argparse
import os

import boto3
import sagemaker
from sagemaker.model_monitor import CronExpressionGenerator, DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat

PREFIX = os.environ.get("FRAUD_SSM_PREFIX", "/fraud-mlops")


def ssm(name: str) -> str:
    return boto3.client("ssm").get_parameter(Name=f"{PREFIX}/{name}")["Parameter"]["Value"]


def latest_baseline_uri(sm, pipeline_name: str) -> str:
    """Baseline CSV written by the Preprocess step of the latest successful execution."""
    execs = sm.list_pipeline_executions(PipelineName=pipeline_name, SortBy="CreationTime",
                                        SortOrder="Descending")["PipelineExecutionSummaries"]
    for ex in execs:
        if ex["PipelineExecutionStatus"] != "Succeeded":
            continue
        steps = sm.list_pipeline_execution_steps(PipelineExecutionArn=ex["PipelineExecutionArn"])
        for step in steps["PipelineExecutionSteps"]:
            if step["StepName"] == "Preprocess":
                job_arn = step["Metadata"]["ProcessingJob"]["Arn"]
                job = sm.describe_processing_job(ProcessingJobName=job_arn.split("/")[-1])
                for out in job["ProcessingOutputConfig"]["Outputs"]:
                    if out["OutputName"] == "baseline":
                        return out["S3Output"]["S3Uri"] + "/baseline.csv"
    raise SystemExit("no successful pipeline execution found; run the pipeline first")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-uri", default="")
    args = ap.parse_args()

    session = sagemaker.Session()
    sm = session.sagemaker_client
    bucket, role = ssm("bucket"), ssm("sagemaker-role-arn")
    endpoint, schedule = ssm("endpoint-name"), ssm("monitor-schedule-name")
    baseline_uri = args.baseline_uri or latest_baseline_uri(sm, ssm("pipeline-name"))

    monitor = DefaultModelMonitor(role=role, instance_count=1, instance_type="ml.m5.xlarge",
                                  volume_size_in_gb=20, max_runtime_in_seconds=3300,
                                  sagemaker_session=session)
    print(f"baselining from {baseline_uri} ...")
    monitor.suggest_baseline(
        baseline_dataset=baseline_uri,
        # header-less to match captured traffic; tools map _cN back to feature names
        dataset_format=DatasetFormat.csv(header=False),
        output_s3_uri=f"s3://{bucket}/monitoring/baseline",
        wait=True,
    )
    monitor.create_monitoring_schedule(
        monitor_schedule_name=schedule,
        endpoint_input=endpoint,
        output_s3_uri=f"s3://{bucket}/monitoring/reports",
        statistics=monitor.baseline_statistics(),
        constraints=monitor.suggested_constraints(),
        schedule_cron_expression=CronExpressionGenerator.hourly(),
        enable_cloudwatch_metrics=True,
    )
    print(f"monitoring schedule {schedule} created on {endpoint}")


if __name__ == "__main__":
    main()
