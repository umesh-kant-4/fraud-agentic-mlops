SHELL := /bin/bash
PREFIX ?= /fraud-mlops
EMAIL ?=
BUCKET = $(shell aws ssm get-parameter --name $(PREFIX)/bucket --query Parameter.Value --output text)
ACCOUNT = $(shell aws sts get-caller-identity --query Account --output text)
REGION ?= $(shell aws configure get region || echo us-east-1)
CFN_EXEC = arn:aws:iam::$(ACCOUNT):role/cdk-fraudmlops-cfn-exec-role-$(ACCOUNT)-$(REGION)

.PHONY: install test synth deploy-core release data pipeline monitor agent wire drift destroy destroy-release

install:
	python3 -m pip install -r requirements.txt

test:
	cd tests && python3 -m pytest -q

synth:
	cd cdk && cdk synth FraudMlopsCore FraudMlopsTools

# Default approval level: CDK stops and asks a HUMAN to confirm any IAM/security broadening.
deploy-core:            ## make deploy-core EMAIL=you@example.com
	cd cdk && cdk deploy FraudMlopsCore FraudMlopsTools -c notifyEmail=$(EMAIL)

release:                ## YOU only, with your own admin credentials: registry + production deployer
	cd cdk && cdk deploy FraudMlopsRelease

data:                   ## generate + upload training data, plus live/drifted replay sets
	python3 ml/data/generate_synthetic.py --rows 200000 --seed 42 --out data/train.csv
	python3 ml/data/generate_synthetic.py --rows 5000 --seed 7 --start 2026-04-01 --out data/live.csv
	python3 ml/data/generate_synthetic.py --rows 5000 --seed 9 --drift --out data/drifted.csv
	aws s3 cp data/train.csv s3://$(BUCKET)/data/raw/train.csv

pipeline:
	python3 ml/pipeline/pipeline.py --upsert --start

monitor:
	python3 ml/monitor/send_traffic.py --csv data/live.csv --rows 2000
	python3 ml/monitor/setup_monitor.py

agent:
	bash scripts/bootstrap_agent.sh

wire:                   ## make wire ARN=arn:aws:bedrock-agentcore:... ROLE=AgentCoreRuntimeRole...
	# --exclusively: don't redeploy Core/Tools (would drop the email subscription without EMAIL)
	cd cdk && cdk deploy FraudAgentWiring --exclusively -c agentRuntimeArn=$(ARN) -c agentRoleName=$(ROLE)

drift:                  ## replay drifted traffic; the next hourly monitor run should trigger triage
	python3 ml/monitor/send_traffic.py --csv data/drifted.csv --rows 3000

destroy:                ## project stacks Claude may remove (deployer role): wiring, tools
	# --role-arn: delete with the scoped CloudFormation execution role, not the caller's rights
	aws cloudformation delete-stack --stack-name FraudAgentWiring --role-arn $(CFN_EXEC)
	aws cloudformation wait stack-delete-complete --stack-name FraudAgentWiring
	cd cdk && cdk destroy FraudMlopsTools --force

destroy-release:        ## YOU only: release stack (registry is retained), then core
	cd cdk && cdk destroy FraudMlopsRelease --force
	cd cdk && cdk destroy FraudMlopsCore --force
