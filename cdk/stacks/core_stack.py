"""Core fraud MLOps resources: data bucket, SageMaker role, model registry group,
ops table, notifications, rollback alarm and the human-approval deployer."""
from __future__ import annotations

from pathlib import Path

from aws_cdk import (
    Duration, RemovalPolicy, Stack,
    aws_cloudwatch as cw,
    aws_dynamodb as ddb,
    aws_events as events,
    aws_events_targets as targets,
    aws_iam as iam,
    aws_lambda as lambda_,
    aws_s3 as s3,
    aws_sagemaker as sagemaker,
    aws_sns as sns,
    aws_sns_subscriptions as subs,
    aws_ssm as ssm,
)
from constructs import Construct

ROOT = Path(__file__).resolve().parents[2]


class CoreStack(Stack):
    def __init__(self, scope: Construct, cid: str, *, prefix: str, notify_email: str | None, **kw):
        super().__init__(scope, cid, **kw)
        self.prefix = prefix
        self.ssm_prefix = f"/{prefix}"

        self.bucket = s3.Bucket(
            self, "Data",
            versioned=True, enforce_ssl=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            removal_policy=RemovalPolicy.RETAIN,
        )

        # SageMaker execution role for processing, training, monitoring and hosting.
        # Starter scope: tighten AmazonSageMakerFullAccess for production.
        self.sm_role = iam.Role(
            self, "SageMakerRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess")],
        )
        self.bucket.grant_read_write(self.sm_role)
        self.sm_role.add_to_policy(iam.PolicyStatement(
            actions=["iam:PassRole"], resources=[self.sm_role.role_arn]))

        self.group_name = f"{prefix}-fraud-detector"
        self.pipeline_name = f"{prefix}-fraud-training"
        self.endpoint_name = f"{prefix}-fraud-detector-prod"
        self.monitor_schedule = f"{prefix}-fraud-data-quality"

        sagemaker.CfnModelPackageGroup(
            self, "ModelGroup", model_package_group_name=self.group_name,
            model_package_group_description="Fraud detector versions; promotion requires human approval")

        self.table = ddb.Table(
            self, "Ops",
            partition_key=ddb.Attribute(name="pk", type=ddb.AttributeType.STRING),
            sort_key=ddb.Attribute(name="sk", type=ddb.AttributeType.STRING),
            billing_mode=ddb.BillingMode.PAY_PER_REQUEST,
            point_in_time_recovery_specification=ddb.PointInTimeRecoverySpecification(
                point_in_time_recovery_enabled=True),
            removal_policy=RemovalPolicy.RETAIN,
        )

        self.topic = sns.Topic(self, "Notify", display_name="Fraud MLOps notifications")
        if notify_email:
            self.topic.add_subscription(subs.EmailSubscription(notify_email))

        # Auto-rollback trigger for blue/green deployments, and a triage trigger.
        self.alarm_5xx = cw.Alarm(
            self, "Endpoint5xx",
            alarm_name=f"{prefix}-fraud-endpoint-5xx",
            metric=cw.Metric(namespace="AWS/SageMaker", metric_name="Invocation5XXErrors",
                             dimensions_map={"EndpointName": self.endpoint_name, "VariantName": "AllTraffic"},
                             statistic="Sum", period=Duration.minutes(5)),
            threshold=5, evaluation_periods=1,
            comparison_operator=cw.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
            treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
        )

        # Deterministic deployer - fires only on a human approval in the Model Registry.
        deployer = lambda_.Function(
            self, "Deployer",
            runtime=lambda_.Runtime.PYTHON_3_13, handler="handler.lambda_handler",
            code=lambda_.Code.from_asset(str(ROOT / "lambdas" / "deployer")),
            timeout=Duration.minutes(2),
            environment={
                "MODEL_PACKAGE_GROUP": self.group_name,
                "ENDPOINT_NAME": self.endpoint_name,
                "SAGEMAKER_ROLE_ARN": self.sm_role.role_arn,
                "BUCKET": self.bucket.bucket_name,
                "OPS_TABLE": self.table.table_name,
                "NOTIFY_TOPIC_ARN": self.topic.topic_arn,
                "ROLLBACK_ALARM_NAME": self.alarm_5xx.alarm_name,
                "INSTANCE_TYPE": "ml.m5.large",
            },
        )
        acct, region = self.account, self.region
        deployer.add_to_role_policy(iam.PolicyStatement(
            actions=["sagemaker:CreateModel", "sagemaker:CreateEndpointConfig", "sagemaker:CreateEndpoint",
                     "sagemaker:UpdateEndpoint", "sagemaker:DescribeEndpoint",
                     "sagemaker:DescribeModelPackage", "sagemaker:AddTags"],
            resources=[f"arn:aws:sagemaker:{region}:{acct}:model/fraud-model-*",
                       f"arn:aws:sagemaker:{region}:{acct}:endpoint-config/fraud-epc-*",
                       f"arn:aws:sagemaker:{region}:{acct}:endpoint/{self.endpoint_name}",
                       f"arn:aws:sagemaker:{region}:{acct}:model-package/{self.group_name}/*"]))
        deployer.add_to_role_policy(iam.PolicyStatement(
            actions=["iam:PassRole"], resources=[self.sm_role.role_arn]))
        self.table.grant_write_data(deployer)
        self.topic.grant_publish(deployer)

        events.Rule(
            self, "OnModelApproved",
            event_pattern=events.EventPattern(
                source=["aws.sagemaker"],
                detail_type=["SageMaker Model Package State Change"],
                detail={"ModelPackageGroupName": [self.group_name], "ModelApprovalStatus": ["Approved"]},
            ),
            targets=[targets.LambdaFunction(deployer)],
        )

        # Shared config read by the pipeline script, the monitor script and the agent.
        params = {
            "bucket": self.bucket.bucket_name,
            "sagemaker-role-arn": self.sm_role.role_arn,
            "model-package-group": self.group_name,
            "pipeline-name": self.pipeline_name,
            "endpoint-name": self.endpoint_name,
            "monitor-schedule-name": self.monitor_schedule,
            "ops-table": self.table.table_name,
            "notify-topic-arn": self.topic.topic_arn,
        }
        for key, value in params.items():
            ssm.StringParameter(self, f"P-{key}", parameter_name=f"{self.ssm_prefix}/{key}",
                                string_value=value)
