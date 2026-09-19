"""Tests for the agent's sandbox, tools, approval gate and loop. A scripted model replaces the real one.

Run from the project folder:  python -m unittest discover -s tests -v
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.loop import parse_tool_calls, run_agent  # noqa: E402
from agent.sandbox import SandboxError, resolve_inside  # noqa: E402
from agent.tools import Workspace, safe_calculate  # noqa: E402

SAMPLE = Path(__file__).resolve().parent.parent / "workspace"


def call(name, **arguments):
    return f"<tool_call>{json.dumps({'name': name, 'arguments': arguments})}</tool_call>"


def scripted(*replies):
    queue = list(replies)
    return lambda messages, specs: queue.pop(0) if queue else call("list_files")  # a runaway model keeps calling tools


class TempWorkspace(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder, ignore_errors=True)
        shutil.copytree(SAMPLE, self.folder / "workspace", ignore=shutil.ignore_patterns("notes"))
        self.workspace = Workspace(self.folder / "workspace")
        (self.folder / "secret.txt").write_text("TOP SECRET outside the workspace", encoding="utf-8")


class SandboxTests(TempWorkspace):
    def test_paths_that_escape_the_workspace_are_refused(self):
        attacks = ["../secret.txt", "..\\secret.txt", "../../../../Windows/win.ini", "notes/../../secret.txt", str(self.folder / "secret.txt"), "C:\\Windows\\win.ini", "/etc/passwd"]
        for attack in attacks:
            with self.assertRaises(SandboxError, msg=attack):
                self.workspace.read_file(attack)
            with self.assertRaises(SandboxError, msg=attack):
                self.workspace.list_files(attack)

    def test_normal_paths_still_work(self):
        self.assertIn("Priya Raman", self.workspace.read_file("README.md"))
        self.assertIn("meeting_notes.md", self.workspace.list_files())

    def test_unusual_paths_are_rejected_cleanly(self):
        for bad in ["", "   ", "bad\x00name", None, 123]:
            with self.assertRaises(SandboxError):
                resolve_inside(self.workspace.root, bad)

    def test_only_text_files_can_be_read(self):
        (self.workspace.root / "program.exe").write_bytes(b"MZ")
        with self.assertRaises(SandboxError):
            self.workspace.read_file("program.exe")

    def test_symlinks_pointing_outside_are_refused(self):
        link = self.workspace.root / "shortcut.txt"
        try:
            os.symlink(self.folder / "secret.txt", link)
        except (OSError, NotImplementedError):
            self.skipTest("this system does not allow creating symlinks")
        with self.assertRaises(SandboxError):
            self.workspace.read_file("shortcut.txt")

    def test_long_files_are_truncated(self):
        (self.workspace.root / "big.txt").write_text("x" * 10_000, encoding="utf-8")
        text = self.workspace.read_file("big.txt")
        self.assertLess(len(text), 3100)
        self.assertTrue(text.endswith("[truncated]"))


class ToolTests(TempWorkspace):
    def test_search_finds_the_right_file_and_line(self):
        self.assertIn("contract_summary.md:3", self.workspace.search_files("acme"))  # case-insensitive, line 3 of that file
        self.assertEqual(self.workspace.search_files("zebra crossing"), "no matches")

    def test_search_treats_the_query_as_plain_text_not_a_pattern(self):
        self.assertEqual(self.workspace.search_files(".*"), "no matches")  # would match everything if it were a regex

    def test_calculator_refuses_anything_but_arithmetic(self):
        self.assertEqual(safe_calculate("1,250 * 12"), "15000")
        for bad in ["__import__('os').system('echo hi')", "open('secret.txt').read()", "9 ** 9 ** 9", "1/0"]:
            with self.assertRaises(Exception, msg=bad):
                safe_calculate(bad)

    def test_note_titles_are_sanitized_and_cannot_choose_a_path(self):
        saved = self.workspace.write_note("../../evil name!", "hello")
        self.assertEqual(saved, "saved notes/evil-name.md")
        self.assertTrue((self.workspace.notes_dir / "evil-name.md").exists())
        self.assertFalse((self.folder / "evil name!.md").exists())

    def test_notes_are_never_overwritten(self):
        self.workspace.write_note("todo", "first")
        with self.assertRaises(SandboxError):
            self.workspace.write_note("todo", "second")
        self.assertEqual((self.workspace.notes_dir / "todo.md").read_text(encoding="utf-8"), "first")


class LoopTests(TempWorkspace):
    def test_read_then_answer(self):
        model = scripted(call("read_file", path="project_notes.md"), "The deadline is October 14.")
        result = run_agent("deadline?", model, self.workspace.tools())
        self.assertEqual((result.status, result.tools_used), ("answered", ["read_file"]))
        self.assertIn("October 14", result.steps[0].result)

    def test_path_traversal_attempt_becomes_an_error_the_model_can_see(self):
        model = scripted(call("read_file", path="../secret.txt"), "I cannot read that.")
        result = run_agent("read the secret", model, self.workspace.tools())
        self.assertIn("outside the workspace", result.steps[0].result)
        self.assertNotIn("TOP SECRET", result.steps[0].result)

    def test_writes_are_declined_by_default(self):
        model = scripted(call("write_note", title="x", content="y"), "Done.")
        result = run_agent("save a note", model, self.workspace.tools())  # no approve_fn
        self.assertIs(result.steps[0].approved, False)
        self.assertIn("declined", result.steps[0].result)
        self.assertFalse(self.workspace.notes_dir.exists() and any(self.workspace.notes_dir.iterdir()))

    def test_an_approved_write_happens_and_is_recorded(self):
        asked = []
        approve = lambda name, arguments: asked.append((name, arguments)) or True
        model = scripted(call("write_note", title="todo", content="call vendor"), "Saved.")
        result = run_agent("save a note", model, self.workspace.tools(), approve_fn=approve)
        self.assertEqual(asked, [("write_note", {"title": "todo", "content": "call vendor"})])
        self.assertIs(result.steps[0].approved, True)
        self.assertTrue((self.workspace.notes_dir / "todo.md").exists())

    def test_read_only_tools_never_ask_for_approval(self):
        def fail(name, arguments):
            raise AssertionError("approval must not be requested for a read-only tool")

        run_agent("q", scripted(call("list_files"), "ok"), self.workspace.tools(), approve_fn=fail)

    def test_a_poisoned_file_cannot_make_the_agent_write_when_the_gate_declines(self):
        """The file tells the model to write a note. Even a model that obeys is stopped by the gate."""
        model = scripted(
            call("read_file", path="vendor_email.txt"),
            call("write_note", title="pwned", content="all data has been sent to the vendor"),  # the model was tricked
            "I could not save that note.",
        )
        result = run_agent("Summarize vendor_email.txt", model, self.workspace.tools())  # default: decline
        self.assertEqual(result.tools_used, ["read_file", "write_note"])
        self.assertIs(result.steps[1].approved, False)
        self.assertFalse((self.workspace.notes_dir / "pwned.md").exists())

    def test_repeating_the_same_call_is_detected(self):
        model = scripted(call("list_files"), call("list_files"), "done")
        result = run_agent("q", model, self.workspace.tools())
        self.assertIn("already made this exact call", result.steps[1].result)

    def test_runaway_loops_are_stopped(self):
        by_steps = run_agent("q", scripted(*[call("read_file", path=f"f{i}.txt") for i in range(50)]), self.workspace.tools(), max_steps=3)
        self.assertEqual(by_steps.status, "step_limit")
        by_calls = run_agent("q", scripted(*[call("read_file", path=f"f{i}.txt") for i in range(50)]), self.workspace.tools(), max_steps=50, max_tool_calls=4)
        self.assertEqual((by_calls.status, len(by_calls.steps)), ("tool_limit", 4))

    def test_malformed_and_unknown_tool_calls_do_not_crash(self):
        model = scripted("<tool_call>{oops</tool_call>", call("delete_everything"), call("read_file"), "ok")
        result = run_agent("q", model, self.workspace.tools())
        self.assertIn("not valid JSON", result.steps[0].result)
        self.assertIn("unknown tool", result.steps[1].result)
        self.assertIn("missing required", result.steps[2].result)
        self.assertEqual(result.status, "answered")

    def test_huge_tool_output_is_capped(self):
        (self.workspace.root / "wide.txt").write_text("y" * 2900, encoding="utf-8")
        result = run_agent("q", scripted(call("read_file", path="wide.txt"), "ok"), self.workspace.tools())
        self.assertLessEqual(len(result.steps[0].result), 1500)

    def test_parse_tool_calls(self):
        self.assertEqual(parse_tool_calls(call("calculator", expression="1+1")), [("calculator", {"expression": "1+1"})])
        self.assertEqual(parse_tool_calls("no calls here"), [])


if __name__ == "__main__":
    unittest.main()
