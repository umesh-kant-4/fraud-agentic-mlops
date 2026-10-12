#!/usr/bin/env bash
# Scaffold the AgentCore project with the AgentCore CLI, drop in our agent code,
# and deploy it to AgentCore Runtime.
#
# Prereqs: Node 20+, `npm install -g @aws/agentcore`, AWS credentials, Python 3.10+, uv.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_DIR="$HERE/agentcore-project"
AGENT=FraudOpsAgent

if [ ! -d "$PROJECT_DIR" ]; then
  agentcore create --project-name FraudOpsAgents --name "$AGENT" \
    --language Python --framework Strands --model-provider Bedrock \
    --memory none --build CodeZip --skip-git --output-dir "$HERE"
  mv "$HERE/FraudOpsAgents" "$PROJECT_DIR"
fi

APP="$PROJECT_DIR/app/$AGENT"
cp "$HERE/agents/fraud_ops_agent/main.py" "$HERE/agents/fraud_ops_agent/roles.py" "$APP/"

# Extra deps on top of the template's (bedrock-agentcore, strands-agents, mcp)
( cd "$APP" && uv add boto3 mcp-proxy-for-aws )

cd "$PROJECT_DIR"
agentcore validate
agentcore deploy --yes

echo
echo "Deployed. Now grab the runtime ARN and its execution role name:"
echo "  (cd $PROJECT_DIR && agentcore status --json)"
echo "then deploy the wiring stack (see README step 6)."
