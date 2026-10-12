#!/usr/bin/env python3
"""CDK app for agentic fraud MLOps.

  # Claude (FraudMlopsDeployer role) or you:
  cdk deploy FraudMlopsCore FraudMlopsTools -c notifyEmail=you@example.com
  # YOU only, with your own admin credentials (the human-only release path):
  cdk deploy FraudMlopsRelease
  # after the agent is deployed with the AgentCore CLI:
  cdk deploy FraudAgentWiring -c agentRuntimeArn=... -c agentRoleName=...
"""
import os

import aws_cdk as cdk

from stacks.core_stack import CoreStack
from stacks.release_stack import ReleaseStack
from stacks.tools_stack import ToolsStack
from stacks.wiring_stack import WiringStack

app = cdk.App()
env = cdk.Environment(account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
                      region=os.environ.get("CDK_DEFAULT_REGION", "us-east-1"))
prefix = app.node.try_get_context("prefix") or "fraud-mlops"

core = CoreStack(app, "FraudMlopsCore", env=env, prefix=prefix,
                 notify_email=app.node.try_get_context("notifyEmail"))
ReleaseStack(app, "FraudMlopsRelease", env=env, core=core)
tools = ToolsStack(app, "FraudMlopsTools", env=env, core=core)

runtime_arn = app.node.try_get_context("agentRuntimeArn")
role_name = app.node.try_get_context("agentRoleName")
if runtime_arn and role_name:
    WiringStack(app, "FraudAgentWiring", env=env, core=core, gateway_arn=tools.gateway_arn,
                agent_runtime_arn=runtime_arn, agent_role_name=role_name)

cdk.Tags.of(app).add("project", prefix)
app.synth()
