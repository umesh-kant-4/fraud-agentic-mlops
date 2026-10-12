"""The human-only release path: model registry group + production deployer.

Deployed by YOU with your own (admin) credentials: `make release`.
- Uses CliCredentialsStackSynthesizer, so it deploys with the caller's credentials
  instead of the CDK bootstrap roles. Claude's roles cannot deploy or change it
  (FraudMlopsGuardrails denies CloudFormation writes on any other stack name, and
  the scoped CloudFormation execution role cannot create FraudMlopsRelease-* roles).
- Its deployer role is the ONLY project role without the FraudMlopsBoundary, because
  the boundary denies endpoint changes and approved-model creation to every other role.
"""
from __future__ import annotations

from pathlib import Path

from aws_cdk import CliCredentialsStackSynthesizer, Duration, RemovalPolicy, Stack
from aws_cdk import aws_events as events
from aws_cdk import aws_events_targets as targets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_sagemaker as sagemaker
from constructs import Construct

from .core_stack import CoreStack

ROOT = Path(__file__).resolve().parents[2]


class ReleaseStack(Stack):
    def __init__(self, scope: Construct, cid: str, *, core: CoreStack, **kw):
        super().__init__(scope, cid, synthesizer=CliCredentialsStackSynthesizer(qualifier="fraudmlops"), **kw)
        acct, region = self.account, self.region

        # Registry = audit trail of what was approved and by whom: retained on teardown, and
        # created here (not in Core) so redeploying Core never collides with a retained group.
        group = sagemaker.CfnModelPackageGroup(
            self, "ModelGroup", model_package_group_name=core.group_name,
            model_package_group_description="Fraud detector versions; promotion requires human approval")
        group.apply_removal_policy(RemovalPolicy.RETAIN)

        deployer = lambda_.Function(
            self, "Deployer",
            runtime=lambda_.Runtime.PYTHON_3_13, handler="handler.lambda_handler",
            code=lambda_.Code.from_asset(str(ROOT / "lambdas" / "deployer")),
            timeout=Duration.minutes(2),
            environment={
                "MODEL_PACKAGE_GROUP": core.group_name,
                "ENDPOINT_NAME": core.endpoint_name,
                "SAGEMAKER_ROLE_ARN": core.sm_role.role_arn,
                "BUCKET": core.bucket.bucket_name,
                "OPS_TABLE": core.table.table_name,
                "NOTIFY_TOPIC_ARN": core.topic.topic_arn,
                "ROLLBACK_ALARM_NAME": core.alarm_5xx.alarm_name,
                "INSTANCE_TYPE": "ml.m5.large",
            },
        )
        deployer.add_to_role_policy(iam.PolicyStatement(
            actions=["sagemaker:CreateModel", "sagemaker:CreateEndpointConfig", "sagemaker:CreateEndpoint",
                     "sagemaker:UpdateEndpoint", "sagemaker:DescribeEndpoint",
                     "sagemaker:DescribeModelPackage", "sagemaker:AddTags"],
            resources=[f"arn:aws:sagemaker:{region}:{acct}:model/fraud-model-*",
                       f"arn:aws:sagemaker:{region}:{acct}:endpoint-config/fraud-epc-*",
                       f"arn:aws:sagemaker:{region}:{acct}:endpoint/{core.endpoint_name}",
                       f"arn:aws:sagemaker:{region}:{acct}:model-package/{core.group_name}/*"]))
        deployer.add_to_role_policy(iam.PolicyStatement(
            actions=["iam:PassRole"], resources=[core.sm_role.role_arn],
            conditions={"StringEquals": {"iam:PassedToService": "sagemaker.amazonaws.com"}}))
        core.table.grant_write_data(deployer)
        core.topic.grant_publish(deployer)

        events.Rule(
            self, "OnModelApproved",
            event_pattern=events.EventPattern(
                source=["aws.sagemaker"],
                detail_type=["SageMaker Model Package State Change"],
                detail={"ModelPackageGroupName": [core.group_name], "ModelApprovalStatus": ["Approved"]},
            ),
            targets=[targets.LambdaFunction(deployer)],
        )

        # This stack's roles are deliberately outside the project boundary (see module docstring).
        iam.PermissionsBoundary.of(self).clear()
