"""Fraud MLOps agent on AgentCore Runtime (Strands + MCP tools via AgentCore Gateway).

Payload: {"role": "triage" | "release" | "ops", "prompt": "..."}

The agent discovers tools from the Gateway (SigV4-signed with the runtime's own
IAM role, no secrets), keeps only the ones its role allows, and runs the loop.
"""
from __future__ import annotations

import logging
import os

import boto3
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from mcp_proxy_for_aws.client import aws_iam_streamablehttp_client
from strands import Agent
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient

from roles import ROLES, allowed, resolve_role

log = logging.getLogger(__name__)
app = BedrockAgentCoreApp()

REGION = os.environ.get("AWS_REGION", "us-east-1")
SSM_PREFIX = os.environ.get("FRAUD_SSM_PREFIX", "/fraud-mlops")
# Use a cross-region inference profile you have access to; override with AGENT_MODEL_ID
# or the SSM parameter <prefix>/agent-model-id.
DEFAULT_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
_cache: dict = {}


def param(name: str, default: str | None = None) -> str:
    if name not in _cache:
        try:
            _cache[name] = boto3.client("ssm", region_name=REGION).get_parameter(
                Name=f"{SSM_PREFIX}/{name}")["Parameter"]["Value"]
        except Exception:
            if default is None:
                raise
            _cache[name] = default
    return _cache[name]


def list_all_tools(mcp: MCPClient) -> list:
    tools, token = [], None
    while True:
        page = mcp.list_tools_sync(pagination_token=token)
        tools.extend(page)
        token = page.pagination_token
        if token is None:
            return tools


@app.entrypoint
def invoke(payload: dict, context=None) -> dict:
    role = resolve_role(payload)
    prompt = (payload or {}).get("prompt") or "Give me a health summary of the fraud model."
    model_id = os.environ.get("AGENT_MODEL_ID") or param("agent-model-id", DEFAULT_MODEL_ID)
    gateway_url = param("gateway-url")
    if not gateway_url.rstrip("/").endswith("/mcp"):
        gateway_url = gateway_url.rstrip("/") + "/mcp"

    mcp = MCPClient(lambda: aws_iam_streamablehttp_client(
        endpoint=gateway_url, aws_region=REGION, aws_service="bedrock-agentcore"))
    with mcp:
        tools = [t for t in list_all_tools(mcp) if allowed(role, t.tool_name)]
        log.info("role=%s tools=%s", role, [t.tool_name for t in tools])
        agent = Agent(
            model=BedrockModel(model_id=model_id, region_name=REGION, temperature=0.1),
            system_prompt=ROLES[role]["prompt"],
            tools=tools,
        )
        result = agent(prompt)

    return {"role": role, "tools_available": len(tools), "report": str(result)}


if __name__ == "__main__":
    app.run()
