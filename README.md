# Agentic MLOps for fraud detection on AWS

A deterministic SageMaker MLOps backbone for an XGBoost fraud model, with
agents on **Amazon Bedrock AgentCore** that watch it, investigate problems,
retrain when the evidence supports it, and make the promotion case to a human.

```
                         ┌───────────────── AgentCore Runtime ─────────────────┐
 EventBridge ──► Dispatcher Lambda ──►  FraudOpsAgent (Strands)                │
  • hourly monitor check │             roles: triage │ release │ ops (read-only) │
  • endpoint 5XX alarm   └──────────────────────┬──────────────────────────────┘
  • pipeline done                               │ MCP over SigV4 (no secrets)
                                       AgentCore Gateway (AWS_IAM inbound)
                                                │
                                      Tools Lambda (guardrails enforced here)
                                                │
 ┌──────────────────────── SageMaker ───────────┴─────────────────────────────┐
 │ Pipeline: preprocess → train XGBoost → evaluate vs champion → gate → register│
 │ Model Registry (PendingManualApproval)   Endpoint + data capture  Monitor   │
 └──────────────────────────────────────────────────────────────────────────────┘
       Human approves (scripts/approve.py) ──► Deployer Lambda ──► blue/green canary
                                                                   + auto-rollback
```

## What each piece does

| Piece | Where | Agent can… |
|---|---|---|
| Training pipeline (time-based split, XGBoost, champion-vs-challenger eval, quality gate) | `ml/pipeline/` | start it (guarded) and read results |
| Model Registry, new versions land as **PendingManualApproval** | CDK core stack | read only |
| Prod endpoint with data capture, blue/green canary, auto-rollback | `lambdas/deployer/` | read only |
| Model Monitor data-quality schedule (hourly) | `ml/monitor/` | read reports and feature shifts |
| Tools Lambda behind Gateway (11 tools) | `tools/mlops_tools/` | the agent's only way to act |
| Agent (3 roles, per-role tool allowlist) | `agents/fraud_ops_agent/` | — |
| Event → agent dispatcher, de-duplicated | `lambdas/dispatcher/` | — |

**Guardrails live in code, not prompts.** `start_retraining` refuses if a run is
already going, enforces a 6h cooldown (DynamoDB conditional write) and only accepts
data under the project bucket. `request_promotion` refuses packages from other groups,
not pending, or failing the release policy. There is no tool that approves a model or
touches the endpoint. The tools Lambda's IAM role can start only this project's pipeline.

**Fraud-specific choices:** time-based splits (no future leakage); metrics at the alert
budget (precision/recall/dollar recall at top 1%, FPs per 1k txns) instead of accuracy;
the challenger and champion are scored on the *same* test window; the triage runbook
checks label freshness and separates broken feeds (missingness, type changes) from real
fraud-pattern shift before retraining.

## Prerequisites

- AWS account + credentials (`aws sts get-caller-identity` works), region **us-east-1** recommended
- Python 3.11+, Node 20+, [uv](https://docs.astral.sh/uv/)
- `npm install -g aws-cdk @aws/agentcore`
- Bedrock access to the model in `agents/fraud_ops_agent/main.py` (`DEFAULT_MODEL_ID`),
  or set your own: `aws ssm put-parameter --name /fraud-mlops/agent-model-id --type String --value <inference-profile-id>`

## Deploy, step by step

```bash
make install && make test                        # 43 unit tests, no AWS calls

# 1. Core + tools + Gateway
cd cdk && cdk bootstrap && cd ..
make deploy-core EMAIL=you@example.com            # confirm the SNS email you receive

# 2. Data → S3, then define and run the pipeline once
make data
make pipeline                                     # ~15 min; registers v1 as PendingManualApproval

# 3. A human approves v1 → deployer creates the endpoint (~8 min to InService)
aws sagemaker list-model-packages --model-package-group-name fraud-mlops-fraud-detector
python scripts/approve.py --model-package-arn <arn> --note "initial model"

# 4. Baseline traffic + monitoring schedule
make monitor

# 5. Deploy the agent to AgentCore Runtime
make agent
(cd agentcore-project && agentcore status --json)  # note runtime ARN + execution role name

# 6. Wire events to the agent and grant it Gateway access
make wire ARN=<runtime-arn> ROLE=<runtime-role-name>
```

Talk to it any time (ops role is read-only):

```bash
python scripts/ask_agent.py --runtime-arn <arn> "Is the fraud model healthy? Anything I should worry about?"
```

## Demo the loop

```bash
make drift            # fraud shifts to small foreign card-not-present charges
```

Within ~1–2 hours (next monitor run + the dispatcher's :20 check):

1. **Triage agent** reads the violations, sees `card_present`, `is_foreign`, `txn_count_1h`
   and `amount` shifted (not missingness), checks freshness, records an incident and
   starts retraining. You get an email for each.
2. Pipeline trains a challenger and scores it against the champion on the same window.
3. **Release agent** reviews the evaluation and, if it passes, emails you the promotion
   case with the exact approve/reject command.
4. You approve → canary at 10% for 10 min → full rollout, or auto-rollback on 5XX.

Don't want to wait? `python scripts/ask_agent.py --runtime-arn <arn> --role triage "Run triage now"`.

> Note: the drifted replay set has no new *labels*, so a well-behaved triage agent may
> choose "genuine shift without fresh labels" and recommend a label review instead of
> retraining. To see the full retrain path, upload fresh labelled data first:
> `python3 ml/data/generate_synthetic.py --rows 50000 --drift --seed 11 --out data/new.csv && aws s3 cp data/new.csv s3://<bucket>/data/raw/new.csv`

## Things to verify on first deploy

I wrote this against the AWS docs as of October 2026 but could not run `cdk deploy` from
where I built it. Check these if something fails:

- `cdk/stacks/tools_stack.py` uses raw CloudFormation for `AWS::BedrockAgentCore::Gateway`
  and `GatewayTarget` (tool schemas converted to the PascalCase CloudFormation shape). If
  CloudFormation rejects a property, compare with the resource's page in the CloudFormation
  Template Reference.
- `agents/fraud_ops_agent/main.py` signs Gateway calls with `mcp-proxy-for-aws`
  (`aws_iam_streamablehttp_client`). If its signature differs in your version, see that
  package's README.
- `scripts/bootstrap_agent.sh` assumes the AgentCore CLI scaffold layout
  (`app/<AgentName>/main.py`, `uv` project). Run `agentcore create --help` if flags changed.
- After the first hourly monitor run, open its `statistics.json` and confirm columns are
  named `_c0`..`_c11` (the tools map them back to feature names).
- The dispatcher uses the Lambda runtime's bundled boto3 for `bedrock-agentcore`. If the
  client is missing, vendor a newer boto3 into `lambdas/dispatcher/`.

## Costs and cleanup

Main running costs: one `ml.m5.large` endpoint (~$0.12/h), hourly monitoring jobs, and
pipeline runs. Agent and Gateway are pay-per-use.

```bash
aws sagemaker delete-monitoring-schedule --monitoring-schedule-name fraud-mlops-fraud-data-quality
aws sagemaker delete-endpoint --endpoint-name fraud-mlops-fraud-detector-prod
(cd agentcore-project && agentcore remove all && agentcore deploy)
make destroy          # S3 bucket and DynamoDB table are retained on purpose
```

## Hardening for production

- Split roles into separate runtimes with separate Gateways (or AgentCore Policy) so tool
  permissions are enforced per agent at the IAM layer, not only by allowlist.
- Replace `AmazonSageMakerFullAccess` on the SageMaker role with scoped permissions.
- Add a shadow/challenger endpoint step before approval, and model-quality monitoring
  once ground-truth labels flow back.
- Put the dispatcher's agent reports into your ticketing system instead of DynamoDB.
