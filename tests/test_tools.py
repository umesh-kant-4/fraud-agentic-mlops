import os
import types
import unittest
from unittest import mock

import _stubs
from _stubs import client_error

h = _stubs.load("tools_handler", "tools/mlops_tools/handler.py")
import schemas  # noqa: E402

ENV = {
    "BUCKET": "bkt", "RAW_DATA_PREFIX": "data/", "MODEL_PACKAGE_GROUP": "grp",
    "PIPELINE_NAME": "fraud-training", "ENDPOINT_NAME": "ep", "MONITOR_SCHEDULE": "sched",
    "OPS_TABLE": "ops", "RETRAIN_COOLDOWN_HOURS": "6",
}
ALLOWED_SCHEMA_KEYS = {"type", "description", "properties", "required", "items"}


def ctx(tool):
    return types.SimpleNamespace(client_context=types.SimpleNamespace(
        custom={"bedrockAgentCoreToolName": f"mlops-tools___{tool}"}))


class FakeTable:
    def __init__(self, fail_code=None):
        self.items, self.fail_code = [], fail_code

    def put_item(self, **kw):
        if self.fail_code and "ConditionExpression" in kw:
            raise client_error(self.fail_code)
        self.items.append(kw["Item"])


@mock.patch.dict(os.environ, ENV, clear=False)
class TestTools(unittest.TestCase):
    def setUp(self):
        self.sm = mock.MagicMock()
        self.sns = mock.MagicMock()
        h._clients.clear()
        h._clients.update({"sagemaker": self.sm, "sns": self.sns})
        self.table = FakeTable()
        self.p_table = mock.patch.object(h, "table", lambda: self.table)
        self.p_table.start()

    def tearDown(self):
        self.p_table.stop()

    # --- wiring
    def test_every_schema_has_a_handler(self):
        self.assertEqual(set(h.HANDLERS), schemas.TOOL_NAMES)

    def test_schemas_use_supported_subset(self):
        def walk(s):
            self.assertTrue(set(s) <= ALLOWED_SCHEMA_KEYS, s)
            for v in s.get("properties", {}).values():
                walk(v)
        for t in schemas.TOOLS:
            self.assertEqual(t["inputSchema"]["type"], "object")
            walk(t["inputSchema"])

    def test_no_tool_can_approve_or_deploy(self):
        for name in schemas.TOOL_NAMES:
            self.assertNotIn("approve", name)
            self.assertNotIn("deploy", name)

    def test_tool_name_prefix_stripped(self):
        self.assertEqual(h.tool_name_from_context(ctx("get_endpoint_status")), "get_endpoint_status")

    def test_unknown_tool(self):
        self.assertIn("unknown tool", h.lambda_handler({}, ctx("drop_tables"))["error"])

    # --- start_retraining guardrails
    def test_retrain_needs_real_reason(self):
        out = h.lambda_handler({"reason": "drift"}, ctx("start_retraining"))
        self.assertTrue(out["refused_by_guardrail"])
        self.sm.start_pipeline_execution.assert_not_called()

    def test_retrain_rejects_foreign_bucket(self):
        out = h.lambda_handler({"reason": "x" * 30, "input_data_uri": "s3://evil/data/"},
                               ctx("start_retraining"))
        self.assertIn("must be under", out["error"])

    def test_retrain_refuses_when_running(self):
        self.sm.list_pipeline_executions.return_value = {
            "PipelineExecutionSummaries": [{"PipelineExecutionStatus": "Executing"}]}
        out = h.lambda_handler({"reason": "x" * 30}, ctx("start_retraining"))
        self.assertIn("already running", out["error"])
        self.sm.start_pipeline_execution.assert_not_called()

    def test_retrain_cooldown(self):
        self.table.fail_code = "ConditionalCheckFailedException"
        self.sm.list_pipeline_executions.return_value = {"PipelineExecutionSummaries": []}
        out = h.lambda_handler({"reason": "x" * 30}, ctx("start_retraining"))
        self.assertIn("cooldown", out["error"])
        self.sm.start_pipeline_execution.assert_not_called()

    def test_retrain_happy_path_passes_champion(self):
        self.sm.list_pipeline_executions.return_value = {"PipelineExecutionSummaries": []}
        self.sm.list_model_packages.return_value = {"ModelPackageSummaryList": [{"ModelPackageArn": "champ"}]}
        self.sm.describe_model_package.return_value = {
            "InferenceSpecification": {"Containers": [{"ModelDataUrl": "s3://bkt/models/m.tar.gz"}]}}
        self.sm.start_pipeline_execution.return_value = {"PipelineExecutionArn": "exec-1"}
        out = h.lambda_handler({"reason": "card_present shift, incident inc-1"}, ctx("start_retraining"))
        self.assertTrue(out["started"])
        params = {p["Name"]: p["Value"] for p in
                  self.sm.start_pipeline_execution.call_args.kwargs["PipelineParameters"]}
        self.assertEqual(params["ChampionModelUri"], "s3://bkt/models/m.tar.gz")
        self.assertEqual(params["InputDataUri"], "s3://bkt/data/raw/")

    # --- request_promotion guardrails
    def test_promotion_rejects_other_group(self):
        self.sm.describe_model_package.return_value = {"ModelPackageGroupName": "other",
                                                       "ModelApprovalStatus": "PendingManualApproval"}
        out = h.lambda_handler({"model_package_arn": "a", "summary": "y" * 50}, ctx("request_promotion"))
        self.assertIn("not in this project", out["error"])

    def test_promotion_rejects_failed_policy(self):
        self.sm.describe_model_package.return_value = {
            "ModelPackageGroupName": "grp", "ModelApprovalStatus": "PendingManualApproval"}
        with mock.patch.object(h, "evaluation_for", return_value={"comparison": {"passes": False}}):
            out = h.lambda_handler({"model_package_arn": "a", "summary": "y" * 50}, ctx("request_promotion"))
        self.assertIn("release policy", out["error"])
        self.sns.publish.assert_not_called()

    def test_promotion_happy_path_notifies(self):
        os.environ["NOTIFY_TOPIC_ARN"] = "topic"
        self.sm.describe_model_package.return_value = {
            "ModelPackageGroupName": "grp", "ModelApprovalStatus": "PendingManualApproval"}
        with mock.patch.object(h, "evaluation_for",
                               return_value={"comparison": {"passes": True, "deltas": {}}, "metrics": {}}):
            out = h.lambda_handler({"model_package_arn": "a", "summary": "y" * 50}, ctx("request_promotion"))
        self.assertTrue(out["requested"])
        self.sns.publish.assert_called_once()
        self.sm.update_model_package.assert_not_called()
        del os.environ["NOTIFY_TOPIC_ARN"]

    def test_retrain_releases_lock_when_start_fails(self):
        self.sm.list_pipeline_executions.return_value = {"PipelineExecutionSummaries": []}
        self.sm.list_model_packages.return_value = {"ModelPackageSummaryList": []}
        self.sm.start_pipeline_execution.side_effect = client_error("ThrottlingException")
        deleted = []
        self.table.delete_item = lambda **kw: deleted.append(kw)
        out = h.lambda_handler({"reason": "x" * 30}, ctx("start_retraining"))
        self.assertIn("ThrottlingException", out["error"])
        self.assertEqual(deleted[0]["Key"], {"pk": "LOCK", "sk": "retrain"})

    def test_trigger_reason_is_registry_safe(self):
        import re
        reason = "Shift in card_present (-0.6σ), is_foreign; incident #inc-1 — 100% CNP!\nretrain"
        safe = h.metadata_safe(reason)
        self.assertRegex(safe, r"^[\w .:/=+\-@]+$")
        self.assertNotIn("\n", safe)
        self.assertLessEqual(len(h.metadata_safe("a" * 1000)), 250)
        self.assertEqual(h.metadata_safe("!!!"), "agent-retrain")
        self.assertTrue(re.search("card_present", safe))

    def test_monitor_columns_mapped_to_feature_names(self):
        self.assertEqual(h.feature_label("_c9"), "card_present (_c9)")
        self.assertEqual(h.feature_label("_c99"), "_c99")
        self.assertEqual(h.feature_label("amount"), "amount")

    def test_cfn_tool_schema_is_pascal_case(self):
        cfn = schemas.to_cfn_tools(schemas.TOOLS)
        self.assertEqual(len(cfn), len(schemas.TOOLS))
        for t in cfn:
            self.assertEqual(set(t), {"Name", "Description", "InputSchema"})
            self.assertEqual(t["InputSchema"]["Type"], "object")
            for prop in t["InputSchema"].get("Properties", {}).values():
                self.assertTrue(set(prop) <= {"Type", "Description", "Properties", "Required", "Items"})
        retrain = next(t for t in cfn if t["Name"] == "start_retraining")
        self.assertEqual(retrain["InputSchema"]["Required"], ["reason"])

    # --- misc
    def test_incident_severity_validated(self):
        out = h.lambda_handler({"title": "t", "severity": "critical", "summary": "s",
                                "recommended_action": "r"}, ctx("record_incident"))
        self.assertTrue(out["refused_by_guardrail"])

    def test_pipeline_execution_must_be_ours(self):
        out = h.lambda_handler({"execution_arn": "arn:aws:sagemaker:us-east-1:1:pipeline/other/execution/x"},
                               ctx("get_pipeline_execution"))
        self.assertTrue(out["refused_by_guardrail"])

    def test_feature_shift_ranks_biggest_mover(self):
        def stats(mean_amount, missing_cp):
            return {"features": [
                {"name": "amount", "inferred_type": "Fractional", "numerical_statistics": {
                    "mean": mean_amount, "std_dev": 10, "common": {"num_present": 100, "num_missing": 0}}},
                {"name": "card_present", "inferred_type": "Integral", "numerical_statistics": {
                    "mean": 0.7, "std_dev": 0.4,
                    "common": {"num_present": 100 - missing_cp, "num_missing": missing_cp}}},
            ]}
        rows = h.feature_shift(stats(50, 0), stats(80, 40))
        self.assertEqual(rows[0]["feature"], "amount")
        self.assertEqual(rows[0]["mean_shift_in_baseline_std"], 3.0)
        cp = next(r for r in rows if r["feature"] == "card_present")
        self.assertAlmostEqual(cp["current_missing_rate"], 0.4)


if __name__ == "__main__":
    unittest.main()
