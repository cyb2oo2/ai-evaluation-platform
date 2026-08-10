from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_eval_protocol.artifact_io import OutputTransaction, OutputTransactionError


class OutputTransactionTests(unittest.TestCase):
    def test_force_replaces_complete_directory_without_stale_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "run"
            output_dir.mkdir()
            (output_dir / "stale.json").write_text("{}", encoding="utf-8")

            with OutputTransaction(output_dir, force=True) as stage_dir:
                (stage_dir / "current.json").write_text("{}", encoding="utf-8")

            self.assertFalse((output_dir / "stale.json").exists())
            self.assertTrue((output_dir / "current.json").is_file())

    def test_failed_force_run_preserves_previous_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "run"
            output_dir.mkdir()
            previous = output_dir / "previous.json"
            previous.write_text('{"status":"complete"}', encoding="utf-8")

            with (
                self.assertRaisesRegex(RuntimeError, "simulated failure"),
                OutputTransaction(output_dir, force=True) as stage_dir,
            ):
                (stage_dir / "partial.json").write_text("{}", encoding="utf-8")
                raise RuntimeError("simulated failure")

            self.assertEqual(previous.read_text(encoding="utf-8"), '{"status":"complete"}')
            self.assertFalse((output_dir / "partial.json").exists())

    def test_nonempty_directory_requires_force(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "run"
            output_dir.mkdir()
            (output_dir / "existing.json").write_text("{}", encoding="utf-8")

            with (
                self.assertRaisesRegex(OutputTransactionError, "use --force"),
                OutputTransaction(output_dir, force=False),
            ):
                pass

    def test_backup_cleanup_failure_restores_previous_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "run"
            output_dir.mkdir()
            previous = output_dir / "previous.json"
            previous.write_text('{"status":"complete"}', encoding="utf-8")
            real_rmtree = __import__("shutil").rmtree
            calls = 0

            def fail_first_cleanup(path: Path, *args: object, **kwargs: object) -> None:
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise OSError("simulated cleanup failure")
                real_rmtree(path, *args, **kwargs)

            with (
                patch("ai_eval_protocol.artifact_io.shutil.rmtree", fail_first_cleanup),
                self.assertRaisesRegex(OutputTransactionError, "previous output was restored"),
                OutputTransaction(output_dir, force=True) as stage_dir,
            ):
                (stage_dir / "current.json").write_text("{}", encoding="utf-8")

            self.assertEqual(previous.read_text(encoding="utf-8"), '{"status":"complete"}')
            self.assertFalse((output_dir / "current.json").exists())
            self.assertEqual(list(output_dir.parent.glob(".run.*-*")), [])


if __name__ == "__main__":
    unittest.main()
