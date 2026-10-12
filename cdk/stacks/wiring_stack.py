"""Connects the deployed AgentCore Runtime agent to the platform.

Deployed AFTER `agentcore deploy`, because it needs the runtime ARN and the
runtime's execution role name (both shown by `agentcore status --json`):

  cdk deploy FraudAgentWiring -c agentRuntimeArn=arn:... -c agentRoleName=...
"""
from __future__ import annotations

from pathlib import Path

from aws_cdk import Duration, Stack
from aws_cdk import aws_events as events
from aws_cdk import aws_events_targets as targets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from constructs import Construct

from .core_stack import CoreStack

ROOT = Path(__file__).resolve().parents[2]


class WiringStack(Stack):
    def __init__(self, scope: Construct, cid: str, *, core: CoreStack, gateway_arn: str,
                 agent_runtime_arn: str, agent_role_name: str, **kw):
        super().__init__(scope, cid, **kw)
        acct, region = self.account, self.region

        # 1. Let the agent read its config and call the gateway (SigV4).
        agent_role = iam.Role.from_role_name(self, "AgentRole", agent_role_name, mutable=True)
        iam.Policy(self, "AgentAccess", roles=[agent_role], statements=[
            iam.PolicyStatement(actions=["ssm:GetParameter", "ssm:GetParameters"],
                                resources=[f"arn:aws:ssm:{region}:{acct}:parameter{core.ssm_prefix}/*"]),
            iam.PolicyStatement(actions=["bedrock-agentcore:InvokeGateway"], resources=[gateway_arn]),
            iam.PolicyStatement(actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
                                resources=["arn:aws:bedrock:*::foundation-model/*",
                                           f"arn:aws:bedrock:*:{acct}:inference-profile/*"]),
        ])

        # 2. Dispatcher: EventBridge -> agent.
        dispatcher = lambda_.Function(
            self, "Dispatcher",
            runtime=lambda_.Runtime.PYTHON_3_13, handler="handler.lambda_handler",
            code=lambda_.Code.from_asset(str(ROOT / "lambdas" / "dispatcher")),
            # Storm protection is the DynamoDB de-dup in the handler, not reserved
            # concurrency (which fails on new accounts with a 10-execution quota).
            timeout=Duration.minutes(15), memory_size=256,
            environment={
                "AGENT_RUNTIME_ARN": agent_runtime_arn,
                "OPS_TABLE": core.table.table_name,
                "PIPELINE_NAME": core.pipeline_name,
                "MONITOR_SCHEDULE": core.monitor_schedule,
            },
        )
        dispatcher.add_to_role_policy(iam.PolicyStatement(
            actions=["bedrock-agentcore:InvokeAgentRuntime"],
            resources=[agent_runtime_arn, f"{agent_runtime_arn}/*"]))
        dispatcher.add_to_role_policy(iam.PolicyStatement(
            actions=["sagemaker:ListMonitoringExecutions"], resources=["*"]))
        core.table.grant(dispatcher, "dynamodb:PutItem", "dynamodb:UpdateItem")

        target = targets.LambdaFunction(dispatcher, retry_attempts=0)
        events.Rule(self, "MonitorCheck", schedule=events.Schedule.cron(minute="20"), targets=[target])
        events.Rule(self, "EndpointAlarm", targets=[target], event_pattern=events.EventPattern(
            source=["aws.cloudwatch"], detail_type=["CloudWatch Alarm State Change"],
            detail={"alarmName": [core.alarm_5xx.alarm_name], "state": {"value": ["ALARM"]}}))
        events.Rule(self, "PipelineDone", targets=[target], event_pattern=events.EventPattern(
            source=["aws.sagemaker"],
            detail_type=["SageMaker Model Building Pipeline Execution Status Change"],
            detail={"currentPipelineExecutionStatus": ["Succeeded", "Failed"]}))
