#!/usr/bin/env bash
# One-time setup, run by YOU with admin credentials. Claude never needs admin.
#
# Creates (all idempotent; re-run after editing the JSON templates):
#   FraudMlopsBoundary      permissions boundary carried by every role the stacks create
#   FraudMlopsCfnExecution(+Iam)  what CloudFormation may create (replaces AdministratorAccess)
#   CDK bootstrap 'fraudmlops' (stack CDKToolkit-FraudMlops) using that execution policy,
#                           separate from any other CDK app's bootstrap in the account
#   FraudMlopsOperator      role Claude uses day to day: run, observe, clean up. No deploys.
#   FraudMlopsDeployer      role Claude uses only when you hand it out: operator + CDK deploy/destroy
#   FraudMlopsGuardrails    explicit denies attached to both roles
#
# Usage:
#   iam/setup.sh --principal-arn arn:aws:iam::<account>:user/<you> [--region us-east-1]
#
# --principal-arn is YOUR identity, the only one allowed to assume the two roles
# (an IAM user ARN, or your IAM Identity Center role ARN). No access keys are created.
set -euo pipefail

REGION="us-east-1"
PRINCIPAL=""
while [ $# -gt 0 ]; do
  case "$1" in
    --principal-arn) PRINCIPAL="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    *) echo "unknown arg $1"; exit 1 ;;
  esac
done
[ -n "$PRINCIPAL" ] || { echo "--principal-arn is required"; exit 1; }

HERE="$(cd "$(dirname "$0")" && pwd)"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

render() {
  sed -e "s/\${ACCOUNT_ID}/${ACCOUNT_ID}/g" -e "s/\${REGION}/${REGION}/g" "$HERE/$1" \
    | python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin), separators=(",",":")))' \
    > "$OUT/$1"
}

upsert_policy() {  # name file
  local arn="arn:aws:iam::${ACCOUNT_ID}:policy/$1"
  if aws iam get-policy --policy-arn "$arn" >/dev/null 2>&1; then
    local versions
    versions=$(aws iam list-policy-versions --policy-arn "$arn" \
      --query 'Versions[?IsDefaultVersion==`false`].VersionId' --output text)
    if [ "$(echo "$versions" | wc -w)" -ge 4 ]; then
      aws iam delete-policy-version --policy-arn "$arn" --version-id "$(echo "$versions" | awk '{print $NF}')"
    fi
    aws iam create-policy-version --policy-arn "$arn" --policy-document "file://$OUT/$2" --set-as-default >/dev/null
    echo "updated  policy $1"
  else
    aws iam create-policy --policy-name "$1" --policy-document "file://$OUT/$2" \
      --tags Key=project,Value=fraud-mlops >/dev/null
    echo "created  policy $1"
  fi
}

upsert_role() {  # name description policy...
  local name="$1" desc="$2"; shift 2
  if aws iam get-role --role-name "$name" >/dev/null 2>&1; then
    aws iam update-assume-role-policy --role-name "$name" --policy-document "file://$OUT/trust.json"
    echo "updated  role $name"
  else
    aws iam create-role --role-name "$name" --max-session-duration 3600 \
      --assume-role-policy-document "file://$OUT/trust.json" --description "$desc" \
      --tags Key=project,Value=fraud-mlops >/dev/null
    echo "created  role $name"
  fi
  for p in "$@"; do
    aws iam attach-role-policy --role-name "$name" --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/$p"
  done
}

echo "Account ${ACCOUNT_ID}, region ${REGION}, trusted principal ${PRINCIPAL}"
for f in permissions-boundary.json cfn-execution-policy.json cfn-execution-iam-policy.json operator-policy.json \
         deployer-policy.json guardrails-policy.json deploy-role-restriction-policy.json; do
  render "$f"
done
upsert_policy FraudMlopsBoundary     permissions-boundary.json
upsert_policy FraudMlopsCfnExecution cfn-execution-policy.json
upsert_policy FraudMlopsCfnExecutionIam cfn-execution-iam-policy.json
upsert_policy FraudMlopsOperator     operator-policy.json
upsert_policy FraudMlopsDeployer     deployer-policy.json
upsert_policy FraudMlopsGuardrails   guardrails-policy.json

# Dedicated CDK bootstrap: does not touch the default bootstrap other CDK apps may use.
npx --yes aws-cdk@2 bootstrap "aws://${ACCOUNT_ID}/${REGION}" \
  --qualifier fraudmlops --toolkit-stack-name CDKToolkit-FraudMlops \
  --cloudformation-execution-policies "arn:aws:iam::${ACCOUNT_ID}:policy/FraudMlopsCfnExecution,arn:aws:iam::${ACCOUNT_ID}:policy/FraudMlopsCfnExecutionIam"

# The bootstrap template lets its deploy role change ANY stack. Pin it to this project's stacks
# and to the scoped execution role. Re-run this script after any future `cdk bootstrap`.
aws iam put-role-policy --role-name "cdk-fraudmlops-deploy-role-${ACCOUNT_ID}-${REGION}" \
  --policy-name FraudMlopsProjectStacksOnly \
  --policy-document "file://$OUT/deploy-role-restriction-policy.json"
echo "pinned   cdk-fraudmlops-deploy-role to project stacks"

# Only YOUR principal can assume the roles; MFA required for IAM users; 1h sessions.
case "$PRINCIPAL" in
  *:assumed-role/*) echo "pass the role ARN (arn:aws:iam::...:role/...), not an assumed-role session ARN"; exit 1 ;;
  *:user/*)
    TRUST="{\"Effect\":\"Allow\",\"Principal\":{\"AWS\":\"${PRINCIPAL}\"},\"Action\":\"sts:AssumeRole\",\"Condition\":{\"Bool\":{\"aws:MultiFactorAuthPresent\":\"true\"}}}" ;;
  *AWSReservedSSO_*)
    # Identity Center recreates this role when the permission set is reprovisioned, which would
    # break a trust policy that names it directly; match it by ARN pattern instead.
    PERMSET="${PRINCIPAL##*AWSReservedSSO_}"; PERMSET="${PERMSET%_*}"
    PATTERN="arn:aws:iam::${ACCOUNT_ID}:role/aws-reserved/sso.amazonaws.com/*AWSReservedSSO_${PERMSET}_*"
    TRUST="{\"Effect\":\"Allow\",\"Principal\":{\"AWS\":\"arn:aws:iam::${ACCOUNT_ID}:root\"},\"Action\":\"sts:AssumeRole\",\"Condition\":{\"ArnLike\":{\"aws:PrincipalArn\":\"${PATTERN}\"}}}" ;;
  *)
    TRUST="{\"Effect\":\"Allow\",\"Principal\":{\"AWS\":\"${PRINCIPAL}\"},\"Action\":\"sts:AssumeRole\"}" ;;
esac
echo "{\"Version\":\"2012-10-17\",\"Statement\":[${TRUST}]}" > "$OUT/trust.json"
python3 -m json.tool "$OUT/trust.json" >/dev/null

upsert_role FraudMlopsOperator "fraud-agentic-mlops: run, observe, clean up (no deploys)" \
  FraudMlopsOperator FraudMlopsGuardrails
upsert_role FraudMlopsDeployer "fraud-agentic-mlops: operator + CDK deploy/destroy" \
  FraudMlopsOperator FraudMlopsDeployer FraudMlopsGuardrails

MFA_LINE=""
[[ "$PRINCIPAL" == *":user/"* ]] && MFA_LINE="mfa_serial = arn:aws:iam::${ACCOUNT_ID}:mfa/<your-mfa-device-name>"
cat <<EOF

Done. Add these profiles to ~/.aws/config. They contain no secrets: each one
assumes a role using your own login.

[profile fraud-mlops-operator]
role_arn = arn:aws:iam::${ACCOUNT_ID}:role/FraudMlopsOperator
source_profile = <your existing profile>
region = ${REGION}
${MFA_LINE}

[profile fraud-mlops-deployer]
role_arn = arn:aws:iam::${ACCOUNT_ID}:role/FraudMlopsDeployer
source_profile = <your existing profile>
region = ${REGION}
${MFA_LINE}

Recommended once per account (not done automatically, it is account-wide):
  aws s3control put-public-access-block --account-id ${ACCOUNT_ID} \\
    --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
EOF
