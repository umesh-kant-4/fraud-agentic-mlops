"""MLOps tools Lambda + AgentCore Gateway (MCP, IAM/SigV4 inbound auth)."""
from __future__ import annotations

import sys
from pathlib import Path

from aws_cdk import CfnOutput, CfnResource, Duration, Stack
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_ssm as ssm
from constructs import Construct

from .core_stack import CoreStack

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "mlops_tools"))
from schemas import TOOLS, to_cfn_tools  # noqa: E402

TARGET_NAME = "mlops-tools"


class ToolsStack(Stack):
    def __init__(self, scope: Construct, cid: str, *, core: CoreStack, **kw):
        super().__init__(scope, cid, **kw)
        acct, region = self.account, self.region

        fn = lambda_.Function(
            self, "MlopsTools",
            runtime=lambda_.Runtime.PYTHON_3_13, handler="handler.lambda_handler",
            code=lambda_.Code.from_asset(str(ROOT / "tools" / "mlops_tools")),
            timeout=Duration.seconds(60), memory_size=512,
            environment={
                "BUCKET": core.bucket.bucket_name,
                "RAW_DATA_PREFIX": "data/",
                "MODEL_PACKAGE_GROUP": core.group_name,
                "PIPELINE_NAME": core.pipeline_name,
                "ENDPOINT_NAME": core.endpoint_name,
                "MONITOR_SCHEDULE": core.monitor_schedule,
                "OPS_TABLE": core.table.table_name,
                "NOTIFY_TOPIC_ARN": core.topic.topic_arn,
                "RETRAIN_COOLDOWN_HOURS": "6",
            },
        )
        # Least privilege: read-only on SageMaker/CloudWatch/S3, plus exactly three writes.
        fn.add_to_role_policy(iam.PolicyStatement(
            sid="ReadOnlySageMaker",
            actions=["sagemaker:Describe*", "sagemaker:List*"], resources=["*"]))
        fn.add_to_role_policy(iam.PolicyStatement(
            sid="StartOnlyOurPipeline",
            actions=["sagemaker:StartPipelineExecution"],
            resources=[f"arn:aws:sagemaker:{region}:{acct}:pipeline/{core.pipeline_name}"]))
        fn.add_to_role_policy(iam.PolicyStatement(
            sid="Metrics", actions=["cloudwatch:GetMetricData"], resources=["*"]))
        core.bucket.grant_read(fn)
        core.table.grant(fn, "dynamodb:PutItem", "dynamodb:DeleteItem")
        core.topic.grant_publish(fn)

        gw_role = iam.Role(self, "GatewayRole",
                           assumed_by=iam.ServicePrincipal("bedrock-agentcore.amazonaws.com"))
        fn.grant_invoke(gw_role)

        gateway = CfnResource(self, "Gateway", type="AWS::BedrockAgentCore::Gateway", properties={
            "Name": f"{core.prefix}-mlops-gateway",
            "Description": "MLOps tools for fraud detection agents",
            "ProtocolType": "MCP",
            "AuthorizerType": "AWS_IAM",   # callers sign with SigV4; no shared secrets
            "RoleArn": gw_role.role_arn,
        })
        gateway.node.add_dependency(gw_role)

        target = CfnResource(self, "GatewayTarget", type="AWS::BedrockAgentCore::GatewayTarget", properties={
            "GatewayIdentifier": gateway.get_att("GatewayIdentifier").to_string(),
            "Name": TARGET_NAME,
            "Description": "SageMaker pipeline, registry, endpoint and monitor tools",
            "CredentialProviderConfigurations": [{"CredentialProviderType": "GATEWAY_IAM_ROLE"}],
            "TargetConfiguration": {"Mcp": {"Lambda": {
                "LambdaArn": fn.function_arn,
                "ToolSchema": {"InlinePayload": to_cfn_tools(TOOLS)},
            }}},
        })
        target.node.add_dependency(fn)

        self.gateway_arn = gateway.get_att("GatewayArn").to_string()
        gateway_url = gateway.get_att("GatewayUrl").to_string()
        ssm.StringParameter(self, "P-gateway-url", parameter_name=f"{core.ssm_prefix}/gateway-url",
                            string_value=gateway_url)
        ssm.StringParameter(self, "P-gateway-arn", parameter_name=f"{core.ssm_prefix}/gateway-arn",
                            string_value=self.gateway_arn)
        CfnOutput(self, "GatewayUrl", value=gateway_url)
        CfnOutput(self, "ToolsFunction", value=fn.function_name)
