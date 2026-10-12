"""Static checks on the IAM policy templates (no AWS calls)."""
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IAM = ROOT / "iam"
FILES = ["operator-policy.json", "deployer-policy.json", "guardrails-policy.json",
         "cfn-execution-policy.json", "cfn-execution-iam-policy.json", "permissions-boundary.json",
         "deploy-role-restriction-policy.json"]
BOUNDARY_KEY_ACTIONS = {"iam:CreateRole", "iam:PutRolePermissionsBoundary", "iam:PutRolePolicy",
                        "iam:DeleteRolePolicy", "iam:AttachRolePolicy", "iam:DetachRolePolicy"}


def render(name, account="123456789012", region="us-east-1"):
    text = (IAM / name).read_text().replace("${ACCOUNT_ID}", account).replace("${REGION}", region)
    return json.loads(text)


def stmts(doc):
    return doc["Statement"]


def actions(s):
    a = s.get("Action", s.get("NotAction", []))
    return [a] if isinstance(a, str) else a


def sid(doc, name):
    return next(s for s in stmts(doc) if s["Sid"] == name)


class TestIamTemplates(unittest.TestCase):
    def test_valid_rendered_unique_sids(self):
        for f in FILES:
            doc = render(f)
            self.assertEqual(doc["Version"], "2012-10-17")
            self.assertNotIn("${", json.dumps(doc), f)
            sids = [s["Sid"] for s in stmts(doc)]
            self.assertEqual(len(sids), len(set(sids)), f)

    def test_under_managed_policy_size_limit_in_longest_region(self):
        for f in FILES:
            size = len(json.dumps(render(f, region="ap-southeast-1"), separators=(",", ":")))
            self.assertLess(size, 6144, f"{f} is {size} chars")

    def test_setup_script_renders_and_uploads_every_template(self):
        script = (IAM / "setup.sh").read_text()
        for f in FILES:
            self.assertIn(f, script)

    def test_no_account_ids_in_templates(self):
        for f in FILES:
            self.assertIsNone(re.search(r"\b\d{12}\b", (IAM / f).read_text()), f)

    # --- operator (everyday) role
    def test_operator_has_no_wildcard_admin(self):
        for s in stmts(render("operator-policy.json")):
            for a in actions(s):
                self.assertFalse(a in ("*", "iam:*", "sts:*", "s3:*", "sagemaker:*",
                                       "bedrock-agentcore:*", "bedrock-agentcore:Get*"), a)

    def test_operator_cannot_deploy(self):
        acts = {a for s in stmts(render("operator-policy.json")) for a in actions(s)}
        self.assertNotIn("sts:AssumeRole", acts)
        writes = [a for a in acts if a.startswith("cloudformation:")
                  and not a.split(":")[1].startswith(("Describe", "List", "Get"))]
        self.assertEqual(writes, [])

    def test_unscoped_logs_actions_use_star(self):
        scoped = sid(render("operator-policy.json"), "ReadProjectLogs")
        self.assertFalse({"logs:GetQueryResults", "logs:StopQuery"} & set(actions(scoped)))

    # --- guardrails (attached to both roles)
    def test_guardrails_are_deny_only_and_block_approval_iam_data_loss(self):
        doc = render("guardrails-policy.json")
        self.assertTrue(all(s["Effect"] == "Deny" for s in stmts(doc)))
        denies = {a for s in stmts(doc) for a in actions(s)}
        for a in ("sagemaker:UpdateModelPackage", "iam:Create*", "iam:Attach*", "s3:DeleteBucket",
                  "dynamodb:DeleteTable", "bedrock-agentcore:GetResourceApiKey"):
            self.assertIn(a, denies)

    def test_guardrails_block_assuming_stack_created_roles(self):
        g = sid(render("guardrails-policy.json"), "DenyAssumingAnyOtherRole")
        self.assertIn("sts:AssumeRole", actions(g))
        self.assertTrue(all("cdk-fraudmlops-" in r for r in g["NotResource"]))

    def test_release_stack_is_outside_claudes_stack_scope(self):
        for f, key in (("guardrails-policy.json", "DenyTouchingOtherStacksAndBootstrap"),
                       ("deploy-role-restriction-policy.json", "OnlyProjectStacks")):
            allowed = sid(render(f), key)["NotResource"]
            self.assertFalse(any("Release" in r for r in allowed), f)

    def test_deploy_role_pinned_to_scoped_execution_role(self):
        doc = render("deploy-role-restriction-policy.json")
        s = sid(doc, "OnlyWithTheScopedExecutionRole")
        self.assertNotIn("cloudformation:ExecuteChangeSet", actions(s))  # key unsupported there
        self.assertIn("cdk-fraudmlops-cfn-exec-role", s["Condition"]["StringNotEquals"]["cloudformation:RoleArn"])
        self.assertIn("put-role-policy", (IAM / "setup.sh").read_text())

    # --- deployer role
    def test_deployer_uses_only_dedicated_bootstrap(self):
        doc = render("deployer-policy.json")
        res = sid(doc, "DeployThroughTheDedicatedCdkBootstrap")["Resource"]
        self.assertTrue(all("cdk-fraudmlops-" in r for r in res))
        self.assertFalse(any("lookup" in r or "image-publishing" in r for r in res))
        delete = sid(doc, "TeardownProjectStacksWithTheScopedExecutionRole")
        self.assertIn("cloudformation:RoleArn", delete["Condition"]["StringEquals"])

    # --- CloudFormation execution policies
    def test_boundary_condition_only_on_supporting_actions(self):
        for f in ("cfn-execution-policy.json", "cfn-execution-iam-policy.json"):
            for s in stmts(render(f)):
                cond = s.get("Condition", {}).get("StringEquals", {}).get("iam:PermissionsBoundary")
                if s["Effect"] == "Allow" and BOUNDARY_KEY_ACTIONS & set(actions(s)):
                    self.assertTrue(cond and cond.endswith(":policy/FraudMlopsBoundary"), s["Sid"])
                if cond:
                    self.assertTrue(set(actions(s)) <= BOUNDARY_KEY_ACTIONS, s["Sid"])

    def test_cfn_cannot_escalate(self):
        doc = render("cfn-execution-iam-policy.json")
        denies = {a for s in stmts(doc) if s["Effect"] == "Deny" for a in actions(s)}
        for a in ("iam:DeleteRolePermissionsBoundary", "iam:CreatePolicyVersion", "iam:CreateUser"):
            self.assertIn(a, denies)
        passrole = sid(doc, "PassOnlyProjectRolesToProjectServices")
        self.assertFalse(any(r.endswith("role/Fraud*") for r in passrole["Resource"]))
        for r in passrole["Resource"]:  # operator/deployer roles must not be passable
            self.assertFalse(re.fullmatch(r.split(":role/")[1].replace("*", ".*"), "FraudMlopsOperator"))

    def test_cfn_log_groups_scoped(self):
        s = sid(render("cfn-execution-policy.json"), "ProjectLogGroups")
        self.assertTrue(all(":log-group:" in r for r in s["Resource"]))

    # --- permissions boundary
    def test_boundary_is_a_tight_ceiling(self):
        doc = render("permissions-boundary.json")
        for s in stmts(doc):
            if s["Effect"] == "Allow":
                iam = [a for a in actions(s) if a.startswith("iam:")]
                self.assertTrue(set(iam) <= {"iam:PassRole", "iam:GetRole"}, iam)
                self.assertNotIn("logs:*", actions(s))
        denies = {a for s in stmts(doc) if s["Effect"] == "Deny" for a in actions(s)}
        self.assertIn("sagemaker:UpdateModelPackage", denies)
        self.assertIn("sagemaker:CreatePresignedNotebookInstanceUrl", denies)
        s3 = sid(doc, "ProjectBucketsInThisAccountOnly")
        self.assertNotIn("arn:aws:s3:::sagemaker-*", s3["Resource"])
        self.assertEqual(s3["Condition"]["StringEquals"]["aws:ResourceAccount"], "123456789012")
        self.assertIn("sagemaker:UpdateEndpoint", actions(sid(doc, "NoProductionEndpointChanges")))
        pending = sid(doc, "NewModelsOnlyAsPendingApproval")
        self.assertEqual(pending["Condition"]["StringNotEquals"]["sagemaker:ModelApprovalStatus"],
                         "PendingManualApproval")
        for st in stmts(doc):  # logs resource policies have no resource-level support
            if st["Effect"] == "Allow" and st.get("Resource") != "*":
                self.assertNotIn("logs:PutResourcePolicy", actions(st))
        pr = sid(doc, "PassOnlyTheSageMakerRoleToSageMaker")
        self.assertEqual(pr["Condition"]["StringEquals"]["iam:PassedToService"], "sagemaker.amazonaws.com")

    # --- wiring into the CDK apps
    def test_cdk_apps_use_boundary_and_dedicated_bootstrap(self):
        ctx = json.loads((ROOT / "cdk" / "cdk.json").read_text())["context"]
        self.assertEqual(ctx["@aws-cdk/core:permissionsBoundary"]["name"], "FraudMlopsBoundary")
        self.assertEqual(ctx["@aws-cdk/core:bootstrapQualifier"], "fraudmlops")
        script = (ROOT / "scripts" / "bootstrap_agent.sh").read_text()
        self.assertIn("@aws-cdk/core:permissionsBoundary", script)
        self.assertIn("@aws-cdk/core:bootstrapQualifier", script)
        self.assertNotIn("deploy --yes", script)
        setup = (IAM / "setup.sh").read_text()
        self.assertIn("--qualifier fraudmlops", setup)

    def test_release_path_is_human_only(self):
        rel = (ROOT / "cdk" / "stacks" / "release_stack.py").read_text()
        self.assertIn("CliCredentialsStackSynthesizer", rel)
        self.assertIn("PermissionsBoundary.of(self).clear()", rel)
        self.assertIn("RemovalPolicy.RETAIN", rel)
        core = (ROOT / "cdk" / "stacks" / "core_stack.py").read_text()
        self.assertNotIn("lambda_.Function", core)
        self.assertNotIn("CfnModelPackageGroup", core)
        cfn = render("cfn-execution-iam-policy.json")
        for r in sid(cfn, "RoleWritesThatSupportTheBoundaryCondition")["Resource"]:
            self.assertNotIn("Release", r)

    def test_makefile_keeps_human_in_the_loop(self):
        mk = (ROOT / "Makefile").read_text()
        self.assertNotIn("--require-approval never", mk)
        self.assertIn("--role-arn $(CFN_EXEC)", mk)


if __name__ == "__main__":
    unittest.main()
