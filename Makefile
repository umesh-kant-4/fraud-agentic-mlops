SHELL := /bin/bash
PREFIX ?= /fraud-mlops
EMAIL ?=
BUCKET = $(shell aws ssm get-parameter --name $(PREFIX)/bucket --query Parameter.Value --output text)

.PHONY: install test synth deploy-core data pipeline monitor agent wire drift destroy

install:
	python3 -m pip install -r requirements.txt

test:
	cd tests && python3 -m pytest -q

synth:
	cd cdk && cdk synth FraudMlopsCore FraudMlopsTools

deploy-core:            ## make deploy-core EMAIL=you@example.com
	cd cdk && cdk deploy FraudMlopsCore FraudMlopsTools -c notifyEmail=$(EMAIL) --require-approval never

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
	cd cdk && cdk deploy FraudAgentWiring --exclusively -c agentRuntimeArn=$(ARN) -c agentRoleName=$(ROLE) --require-approval never

drift:                  ## replay drifted traffic; the next hourly monitor run should trigger triage
	python3 ml/monitor/send_traffic.py --csv data/drifted.csv --rows 3000

destroy:                ## wiring stack first: it imports exports from Core/Tools
	aws cloudformation delete-stack --stack-name FraudAgentWiring
	aws cloudformation wait stack-delete-complete --stack-name FraudAgentWiring
	cd cdk && cdk destroy FraudMlopsTools FraudMlopsCore --force
