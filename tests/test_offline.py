import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from mcp_shield.core import inspect, train
from mcp_shield.data import dataset, split
from mcp_shield.offline import inspect_batch, read_document


class OfflineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = train(split(dataset())[0])
        cls.sample = json.loads((Path(__file__).resolve().parents[1] / "examples" / "catalog.json").read_text())

    def test_batch_atomic_validation_and_counts(self):
        denied = copy.deepcopy(self.sample)
        denied["signals"]["credential_request"] = True
        result = inspect_batch([self.sample, denied], self.model)
        self.assertEqual(result["decisions"], {"allow": 1, "review": 0, "deny": 1})
        self.assertEqual(result["results"][0], inspect(self.sample, self.model))
        with patch("mcp_shield.offline.inspect") as inference:
            with self.assertRaisesRegex(ValueError, "item 1"):
                inspect_batch([self.sample, {}], self.model)
            inference.assert_not_called()
        for value in ([], {}, [self.sample] * 101):
            with self.assertRaises(ValueError):
                inspect_batch(value, self.model)

    def test_reader_bounds_and_duplicates(self):
        import io
        for raw in (b"", b"x" * 11, b'{"x":1,"x":2}', b"\xff"):
            with patch.object(Path, "open", return_value=io.BytesIO(raw)):
                with self.assertRaises((ValueError, UnicodeError)):
                    read_document("unused", maximum=10)
