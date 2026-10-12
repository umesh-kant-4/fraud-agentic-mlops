"""Define and upsert the SageMaker Pipeline: preprocess -> train -> evaluate -> gate -> register.

The pipeline itself stays deterministic. Agents only *start* it (with a
reason and a data window) and *read* its results; they never edit it.

Usage (after `cdk deploy FraudMlopsCore`):
  python ml/pipeline/pipeline.py --upsert
  python ml/pipeline/pipeline.py --upsert --start   # also run once

Requires: sagemaker>=2.230,<3 (see requirements.txt).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import boto3
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionEquals, ConditionGreaterThanOrEqualTo
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

HERE = Path(__file__).parent
PREFIX = os.environ.get("FRAUD_SSM_PREFIX", "/fraud-mlops")
XGB_VERSION = "1.7-1"


def ssm(name: str) -> str:
    return boto3.client("ssm").get_parameter(Name=f"{PREFIX}/{name}")["Parameter"]["Value"]


def build(session: sagemaker.Session) -> Pipeline:
    region = session.boto_region_name
    role = ssm("sagemaker-role-arn")
    bucket = ssm("bucket")
    group = ssm("model-package-group")
    name = ssm("pipeline-name")

    input_uri = ParameterString("InputDataUri", default_value=f"s3://{bucket}/data/raw/")
    champion_uri = ParameterString("ChampionModelUri", default_value="none")
    trigger_reason = ParameterString("TriggerReason", default_value="manual")
    aucpr_floor = ParameterFloat("AucPrFloor", default_value=0.30)
    train_instance = ParameterString("TrainingInstanceType", default_value="ml.m5.xlarge")

    # 1. preprocess
    sk = SKLearnProcessor(framework_version="1.2-1", role=role, instance_type="ml.m5.xlarge",
                          instance_count=1, base_job_name="fraud-preprocess", sagemaker_session=session)
    step_pre = ProcessingStep(
        name="Preprocess",
        processor=sk,
        inputs=[ProcessingInput(source=input_uri, destination="/opt/ml/processing/input")],
        outputs=[ProcessingOutput(output_name=n, source=f"/opt/ml/processing/{n}")
                 for n in ("train", "validation", "test", "baseline")],
        code=str(HERE / "preprocess.py"),
    )

    # 2. train (SageMaker built-in XGBoost)
    image = sagemaker.image_uris.retrieve("xgboost", region, XGB_VERSION)
    est = Estimator(image_uri=image, role=role, instance_count=1, instance_type=train_instance,
                    output_path=f"s3://{bucket}/models/", base_job_name="fraud-train",
                    sagemaker_session=session)
    est.set_hyperparameters(objective="binary:logistic", eval_metric="aucpr", num_round=300,
                            max_depth=6, eta=0.1, subsample=0.8, colsample_bytree=0.8,
                            min_child_weight=5, scale_pos_weight=50, early_stopping_rounds=30)
    outs = step_pre.properties.ProcessingOutputConfig.Outputs
    step_train = TrainingStep(
        name="Train",
        estimator=est,
        inputs={
            "train": TrainingInput(outs["train"].S3Output.S3Uri, content_type="text/csv"),
            "validation": TrainingInput(outs["validation"].S3Output.S3Uri, content_type="text/csv"),
        },
    )

    # 3. evaluate challenger vs champion on the same test window
    evaluator = ScriptProcessor(image_uri=image, command=["python3"], role=role,
                                instance_type="ml.m5.xlarge", instance_count=1,
                                base_job_name="fraud-evaluate", sagemaker_session=session)
    eval_report = PropertyFile(name="EvaluationReport", output_name="evaluation", path="evaluation.json")
    step_eval = ProcessingStep(
        name="Evaluate",
        processor=evaluator,
        inputs=[
            ProcessingInput(step_train.properties.ModelArtifacts.S3ModelArtifacts, "/opt/ml/processing/model"),
            ProcessingInput(outs["test"].S3Output.S3Uri, "/opt/ml/processing/test"),
            ProcessingInput(str(HERE), "/opt/ml/processing/lib"),  # fraud_metrics.py, preprocess.py
        ],
        # write into the project bucket so the tools Lambda (scoped to it) can read the report
        outputs=[ProcessingOutput(
            output_name="evaluation", source="/opt/ml/processing/evaluation",
            destination=Join(on="/", values=[f"s3://{bucket}/pipeline",
                                             ExecutionVariables.PIPELINE_EXECUTION_ID, "evaluation"]))],
        code=str(HERE / "evaluate.py"),
        job_arguments=["--champion-model-uri", champion_uri],
        property_files=[eval_report],
    )

    # 4. register as PendingManualApproval - a human approves, never an agent
    eval_s3 = Join(on="/", values=[
        step_eval.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,
        "evaluation.json"])
    step_register = RegisterModel(
        name="Register",
        estimator=est,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large", "ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name=group,
        approval_status="PendingManualApproval",
        model_metrics=ModelMetrics(model_statistics=MetricsSource(
            s3_uri=eval_s3, content_type="application/json")),
        # TriggerReason is sanitised by the tools Lambda to the characters this field allows
        customer_metadata_properties={"trigger_reason": trigger_reason},
        description="Fraud detector challenger",
    )

    step_fail = FailStep(name="QualityGateFailed", error_message=Join(on=" ", values=[
        "Challenger failed quality gate. AUC-PR floor:", aucpr_floor]))

    gate = ConditionStep(
        name="QualityGate",
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(step_name=step_eval.name, property_file=eval_report,
                             json_path="metrics.aucpr"),
                right=aucpr_floor),
            ConditionEquals(
                left=JsonGet(step_name=step_eval.name, property_file=eval_report,
                             json_path="comparison.passes_int"),
                right=1),
        ],
        if_steps=[step_register],
        else_steps=[step_fail],
    )

    return Pipeline(
        name=name,
        parameters=[input_uri, champion_uri, trigger_reason, aucpr_floor, train_instance],
        steps=[step_pre, step_train, step_eval, gate],
        sagemaker_session=session,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upsert", action="store_true")
    ap.add_argument("--start", action="store_true")
    args = ap.parse_args()
    session = sagemaker.Session()
    pipeline = build(session)
    if args.upsert:
        pipeline.upsert(role_arn=ssm("sagemaker-role-arn"))
        print(f"upserted pipeline {pipeline.name}")
    else:
        print(pipeline.definition())
    if args.start:
        execution = pipeline.start(parameters={"TriggerReason": "initial-training"})
        print(f"started {execution.arn}")


if __name__ == "__main__":
    main()
