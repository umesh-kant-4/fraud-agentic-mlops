import os
import unittest
from unittest import mock

import _stubs
import roles
import schemas

dispatcher = _stubs.load("dispatcher_handler", "lambdas/dispatcher/handler.py")
with mock.patch("boto3.client", create=True):
    deployer = _stubs.load("deployer_handler", "lambdas/deployer/handler.py")


class TestRoles(unittest.TestCase):
    def test_allowlists_reference_real_tools(self):
        for name, r in roles.ROLES.items():
            self.assertTrue(r["tools"] <= schemas.TOOL_NAMES, name)

    def test_ops_is_read_only(self):
        self.assertFalse(roles.ROLES["ops"]["tools"] & schemas.WRITE)

    def test_release_cannot_retrain_and_triage_cannot_promote(self):
        self.assertNotIn("start_retraining", roles.ROLES["release"]["tools"])
        self.assertNotIn("request_promotion", roles.ROLES["triage"]["tools"])

    def test_gateway_name_filtering(self):
        self.assertTrue(roles.allowed("triage", "mlops-tools___start_retraining"))
        self.assertFalse(roles.allowed("ops", "mlops-tools___start_retraining"))
        self.assertEqual(roles.resolve_role({"role": "root"}), "ops")


@mock.patch.dict(os.environ, {"PIPELINE_NAME": "fraud-mlops-fraud-training", "MONITOR_SCHEDULE": "s"})
class TestDispatcher(unittest.TestCase):
    def test_alarm_routes_to_triage(self):
        ev = {"source": "aws.cloudwatch", "detail-type": "CloudWatch Alarm State Change",
              "detail": {"alarmName": "a", "state": {"value": "ALARM", "timestamp": "t", "reason": "r"}}}
        key, role, _ = dispatcher.route(ev)
        self.assertEqual((key, role), ("alarm/a/t", "triage"))

    def test_alarm_ok_ignored(self):
        ev = {"source": "aws.cloudwatch", "detail-type": "CloudWatch Alarm State Change",
              "detail": {"alarmName": "a", "state": {"value": "OK"}}}
        self.assertIsNone(dispatcher.route(ev))

    def _pipeline_event(self, status, name="fraud-mlops-fraud-training"):
        return {"source": "aws.sagemaker",
                "detail-type": "SageMaker Model Building Pipeline Execution Status Change",
                "detail": {"pipelineArn": f"arn:aws:sagemaker:us-east-1:1:pipeline/{name}",
                           "pipelineExecutionArn": "exec", "currentPipelineExecutionStatus": status}}

    def test_pipeline_success_routes_to_release(self):
        self.assertEqual(dispatcher.route(self._pipeline_event("Succeeded"))[1], "release")

    def test_pipeline_failure_routes_to_triage(self):
        self.assertEqual(dispatcher.route(self._pipeline_event("Failed"))[1], "triage")

    def test_other_pipelines_ignored(self):
        self.assertIsNone(dispatcher.route(self._pipeline_event("Succeeded", name="someone-else")))

    def test_duplicate_event_not_dispatched(self):
        ev = {"source": "aws.cloudwatch", "detail-type": "CloudWatch Alarm State Change",
              "detail": {"alarmName": "a", "state": {"value": "ALARM", "timestamp": "t"}}}
        with mock.patch.object(dispatcher, "claim", return_value=False), \
                mock.patch.object(dispatcher, "invoke_agent") as inv:
            out = dispatcher.lambda_handler(ev, None)
        self.assertFalse(out["dispatched"])
        inv.assert_not_called()


class TestDeployer(unittest.TestCase):
    def test_endpoint_config_captures_data(self):
        cfg = deployer.build_endpoint_config("c", "m", "bkt", "ml.m5.large")
        dc = cfg["DataCaptureConfig"]
        self.assertTrue(dc["EnableCapture"])
        self.assertEqual(dc["DestinationS3Uri"], "s3://bkt/datacapture")
        self.assertEqual(cfg["ProductionVariants"][0]["VariantName"], "AllTraffic")

    def test_canary_has_auto_rollback(self):
        c = deployer.canary_deployment_config("alarm")
        self.assertEqual(c["BlueGreenUpdatePolicy"]["TrafficRoutingConfiguration"]["Type"], "CANARY")
        self.assertEqual(c["AutoRollbackConfiguration"]["Alarms"], [{"AlarmName": "alarm"}])

    @mock.patch.dict(os.environ, {"MODEL_PACKAGE_GROUP": "g", "ENDPOINT_NAME": "ep"})
    def test_busy_endpoint_raises_for_retry(self):
        sm = mock.MagicMock()
        sm.describe_model_package.return_value = {"ModelApprovalStatus": "Approved",
                                                  "ModelPackageGroupName": "g"}
        sm.describe_endpoint.return_value = {"EndpointStatus": "Updating"}
        with mock.patch.object(deployer, "sm", sm):
            with self.assertRaises(deployer.EndpointBusy):
                deployer.lambda_handler({"detail": {"ModelApprovalStatus": "Approved",
                                                    "ModelPackageGroupName": "g",
                                                    "ModelPackageArn": "arn/1"}}, None)
        sm.create_model.assert_not_called()

    @mock.patch.dict(os.environ, {"MODEL_PACKAGE_GROUP": "g", "ENDPOINT_NAME": "ep"})
    def test_forged_approval_event_is_refused(self):
        sm = mock.MagicMock()
        sm.describe_model_package.return_value = {"ModelApprovalStatus": "PendingManualApproval",
                                                  "ModelPackageGroupName": "g"}
        with mock.patch.object(deployer, "sm", sm):
            out = deployer.lambda_handler({"detail": {"ModelApprovalStatus": "Approved",
                                                      "ModelPackageGroupName": "g",
                                                      "ModelPackageArn": "arn/1"}}, None)
        self.assertIn("not approved", out["skipped"])
        sm.create_model.assert_not_called()

    def test_ignores_non_approvals(self):
        out = deployer.lambda_handler({"detail": {"ModelApprovalStatus": "Rejected"}}, None)
        self.assertIn("skipped", out)


if __name__ == "__main__":
    unittest.main()
