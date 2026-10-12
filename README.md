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

## Who can do what

`iam/` sets up least-privilege access so Claude (or any automation) can help manage the
project without admin rights and without long-lived keys. Run `iam/setup.sh` once as an admin.

| Identity | Used by | Can | Cannot |
|---|---|---|---|
| **You** (admin) | you | everything; deploy `FraudMlopsRelease`; approve models | — |
| `FraudMlopsOperator` | Claude, day to day | run the pipeline, set up monitoring, send test traffic, talk to the agent, read logs/metrics/costs, delete the endpoint and monitor during cleanup | deploy anything, approve or register models, change IAM, delete retained data, assume other roles, act outside the region |
| `FraudMlopsDeployer` | Claude, only when you ask | operator rights + deploy/destroy `FraudMlopsCore`, `FraudMlopsTools`, `FraudAgentWiring` and the agent, through the dedicated `fraudmlops` CDK bootstrap | touch any other stack (incl. `FraudMlopsRelease`), plus everything the operator cannot |

Both roles are assumable only by your own identity (MFA for IAM users), 1-hour sessions.
They contain no secrets: your `~/.aws/config` profile assumes them from your login.

Layers that make this hold even if an agent misbehaves:
- **Scoped CloudFormation execution policy** replaces `AdministratorAccess` in the CDK bootstrap,
  and the bootstrap's deploy role is pinned to this project's stacks.
- **Permissions boundary** on every role the stacks create: no IAM changes, no model approval,
  no "already approved" models, no changes to the production endpoint, project buckets in
  this account only.
- **Human-only release path**: the model registry and the production deployer live in
  `FraudMlopsRelease`, which deploys with your credentials only; the deployer re-checks
  with SageMaker that a model was really approved before rolling it out.

## Deploy, step by step

```bash
make install && make test                        # unit tests, no AWS calls

# 0. YOU, once, with admin credentials: policies, scoped CDK bootstrap, the two roles
iam/setup.sh --principal-arn <your IAM user or Identity Center role ARN>
# add the printed profiles to ~/.aws/config

# 1. Core + tools + Gateway                        (profile: fraud-mlops-deployer)
make deploy-core EMAIL=you@example.com            # confirm the SNS email you receive

# 2. Human-only release path                       (YOU, admin profile)
make release

# 3. Data → S3, then define and run the pipeline   (profile: fraud-mlops-operator)
make data
make pipeline                                     # ~15 min; registers v1 as PendingManualApproval

# 4. A human approves v1 → deployer creates the endpoint (YOU; ~8 min to InService)
aws sagemaker list-model-packages --model-package-group-name fraud-mlops-fraud-detector
python scripts/approve.py --model-package-arn <arn> --note "initial model"

# 5. Baseline traffic + monitoring schedule        (profile: fraud-mlops-operator)
make monitor

# 6. Deploy the agent and wire events to it        (profile: fraud-mlops-deployer)
make agent
(cd agentcore-project && agentcore status --json)  # note runtime ARN + execution role name
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
- The AgentCore CLI's own CDK app: `scripts/bootstrap_agent.sh` sets the permissions boundary
  and the `fraudmlops` bootstrap in its `cdk.json`. Check its stack and role names contain
  `FraudOps` (the scoped policies rely on that); if `agentcore deploy` offers to bootstrap
  CDK, decline. If CloudFormation reports AccessDenied, the stack event names the exact
  action and resource; add it to `iam/cfn-execution-policy.json` and re-run `iam/setup.sh`.
- The dispatcher uses the Lambda runtime's bundled boto3 for `bedrock-agentcore`. If the
  client is missing, vendor a newer boto3 into `lambdas/dispatcher/`.

## Costs and cleanup

Main running costs: one `ml.m5.large` endpoint (~$0.12/h), hourly monitoring jobs, and
pipeline runs. Agent and Gateway are pay-per-use.

```bash
# operator profile
aws sagemaker delete-monitoring-schedule --monitoring-schedule-name fraud-mlops-fraud-data-quality
aws sagemaker delete-endpoint --endpoint-name fraud-mlops-fraud-detector-prod
# deployer profile
(cd agentcore-project && agentcore remove all && agentcore deploy)
make destroy           # wiring + tools
# YOU (admin)
make destroy-release   # release + core; bucket, table and model registry are retained on purpose
```

## Hardening for production

- Run the project in its **own AWS account** (AWS Organizations member account) with an
  SCP that denies IAM user/key creation and cross-account trust. Inside a shared account,
  an identity that can deploy stacks which create roles always keeps some indirect power
  over those roles' resources; a dedicated account contains that blast radius.
- Review `cdk diff` for `FraudMlopsCore` yourself before Claude deploys it: CDK only
  prompts on IAM/security-group broadening, not on bucket or Lambda code changes.

- Split roles into separate runtimes with separate Gateways (or AgentCore Policy) so tool
  permissions are enforced per agent at the IAM layer, not only by allowlist.
- Replace `AmazonSageMakerFullAccess` on the SageMaker role with scoped permissions.
- Add a shadow/challenger endpoint step before approval, and model-quality monitoring
  once ground-truth labels flow back.
- Put the dispatcher's agent reports into your ticketing system instead of DynamoDB.
