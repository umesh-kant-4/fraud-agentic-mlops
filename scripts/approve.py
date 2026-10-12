"""Human approval (or rejection) of a model package.

Approving fires the deployer Lambda, which does a blue/green canary rollout with
automatic rollback. This is the only path to production, by design.

  python scripts/approve.py --model-package-arn arn:... --note "reviewed eval, LGTM"
  python scripts/approve.py --model-package-arn arn:... --reject --note "FP rate too high"
"""
from __future__ import annotations

import argparse
import getpass
import os
import time

import boto3

PREFIX = os.environ.get("FRAUD_SSM_PREFIX", "/fraud-mlops")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-package-arn", required=True)
    ap.add_argument("--note", required=True)
    ap.add_argument("--reject", action="store_true")
    args = ap.parse_args()

    ssm = boto3.client("ssm")
    group = ssm.get_parameter(Name=f"{PREFIX}/model-package-group")["Parameter"]["Value"]
    table_name = ssm.get_parameter(Name=f"{PREFIX}/ops-table")["Parameter"]["Value"]
    sm = boto3.client("sagemaker")
    pkg = sm.describe_model_package(ModelPackageName=args.model_package_arn)
    if pkg["ModelPackageGroupName"] != group:
        raise SystemExit(f"package is not in {group}")

    who = boto3.client("sts").get_caller_identity()["Arn"]
    status = "Rejected" if args.reject else "Approved"
    sm.update_model_package(ModelPackageArn=args.model_package_arn, ModelApprovalStatus=status,
                            ApprovalDescription=f"{status} by {who} ({getpass.getuser()}): {args.note}"[:1024])
    boto3.resource("dynamodb").Table(table_name).put_item(Item={
        "pk": "DECISION", "sk": f"{int(time.time())}", "model_package_arn": args.model_package_arn,
        "decision": status, "by": who, "note": args.note})
    print(f"{status}: {args.model_package_arn}")
    if status == "Approved":
        print("Deployer Lambda will roll it out (canary 10% -> 100%, auto-rollback on 5XX alarm).")


if __name__ == "__main__":
    main()
