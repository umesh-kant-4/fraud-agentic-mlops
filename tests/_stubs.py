"""Make modules importable without AWS deps, and put source dirs on sys.path.

If boto3/botocore are installed they are used as-is (no network calls are made:
tests replace every client). If not, minimal stand-ins are registered.
"""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in ("tools/mlops_tools", "ml/pipeline", "ml/data", "agents/fraud_ops_agent"):
    sys.path.insert(0, str(ROOT / p))

try:  # pragma: no cover
    import boto3  # noqa: F401
    from botocore.exceptions import ClientError  # noqa: F401
except ImportError:
    class ClientError(Exception):
        def __init__(self, error_response, operation_name):
            super().__init__(f"{operation_name}: {error_response}")
            self.response = error_response
            self.operation_name = operation_name

    botocore = types.ModuleType("botocore")
    exc = types.ModuleType("botocore.exceptions")
    exc.ClientError = ClientError
    botocore.exceptions = exc
    sys.modules["botocore"] = botocore
    sys.modules["botocore.exceptions"] = exc

    boto3 = types.ModuleType("boto3")
    boto3.client = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no AWS in tests"))
    boto3.resource = boto3.client
    sys.modules["boto3"] = boto3

from botocore.exceptions import ClientError  # noqa: E402,F811


def client_error(code: str, msg: str = "") -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": msg}}, "op")


def load(name: str, path: str):
    """Import a file under a unique module name (two handlers are both called handler.py)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
