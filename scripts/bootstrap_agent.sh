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

# Every IAM role the AgentCore CDK app creates must carry the project permissions
# boundary (the scoped CloudFormation execution policy refuses roles without it).
python3 - "$PROJECT_DIR/agentcore/cdk/cdk.json" <<'EOF'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
if not p.exists():
    sys.exit(f"{p} not found: the AgentCore CLI layout changed. Add the context key "
             '"@aws-cdk/core:permissionsBoundary": {"name": "FraudMlopsBoundary"} '
             "to the CDK app's cdk.json by hand, then re-run.")
cfg = json.loads(p.read_text())
ctx = cfg.setdefault("context", {})
ctx["@aws-cdk/core:permissionsBoundary"] = {"name": "FraudMlopsBoundary"}
ctx["@aws-cdk/core:bootstrapQualifier"] = "fraudmlops"   # deploy via the scoped bootstrap
cfg["toolkitStackName"] = "CDKToolkit-FraudMlops"
p.write_text(json.dumps(cfg, indent=2) + "\n")
print(f"permissions boundary + bootstrap qualifier set in {p}")
EOF

cd "$PROJECT_DIR"
agentcore validate
# No --yes: if the CLI offers to bootstrap CDK, decline. This project deploys through the
# dedicated 'fraudmlops' bootstrap created by iam/setup.sh.
agentcore deploy

echo
echo "Deployed. Now grab the runtime ARN and its execution role name:"
echo "  (cd $PROJECT_DIR && agentcore status --json)"
echo "then deploy the wiring stack (see README step 6)."
