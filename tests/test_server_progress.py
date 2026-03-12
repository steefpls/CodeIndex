"""Tests for stderr progress rendering in server reindex wrappers."""

import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src import server


class _BrokenStderr:
    def write(self, _text):
        raise OSError(22, "Invalid argument")

    def flush(self):
        raise OSError(22, "Invalid argument")


class TestServerProgressRendering(unittest.TestCase):
    def test_render_progress_writes_when_stderr_works(self):
        state = {"enabled": True}
        sink = io.StringIO()
        with patch("src.server.sys.stderr", sink):
            server._render_progress_to_stderr("hello", state)
            server._finish_progress_line(state)
        value = sink.getvalue()
        self.assertIn("hello", value)
        self.assertTrue(value.endswith("\n"))
        self.assertTrue(state["enabled"])

    def test_render_progress_disables_on_stderr_failure(self):
        state = {"enabled": True}
        with patch("src.server.sys.stderr", _BrokenStderr()):
            server._render_progress_to_stderr("hello", state)
        self.assertFalse(state["enabled"])

    def test_finish_progress_line_disables_on_stderr_failure(self):
        state = {"enabled": True}
        with patch("src.server.sys.stderr", _BrokenStderr()):
            server._finish_progress_line(state)
        self.assertFalse(state["enabled"])


if __name__ == "__main__":
    unittest.main()
