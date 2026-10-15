# -*- coding: utf-8 -*-
"""
pytest configuration for the inference test suite.

Sets ARTIFACTS_DIR to an empty temp directory BEFORE app.py (and backends.py)
are imported, so the service under test sees no real models. The suite then
drops a synthetic ONNX model into that directory (see test_inference.py).
"""

import os
import tempfile

os.environ["ARTIFACTS_DIR"] = tempfile.mkdtemp(prefix="cifar10_test_artifacts_")
