"""Headless Textual tests (App.run_test()/Pilot) for the TUI, driven
against ScriptedModel -- confirms modal/worker/rendering mechanics
programmatically before ever pointing the app at slow, unreliable real
Ducky inference. Per the build order's Phase 6 (tasks/todo.md's
agent-harness phase): this is the direct TUI-layer analog of
verify_agent_harness.py's scripted checks for the loop layer.
"""

import pytest

from ducky_agent.model_adapter import ScriptedModel
from ducky_agent.permissions.types import PermissionMode
from ducky_agent.tui.app import DuckyAgentApp
from ducky_agent.tui.widgets import PermissionModal


@pytest.mark.anyio
async def test_app_mounts_and_shows_transcript():
    app = DuckyAgentApp(model=ScriptedModel(responses=["hi"]), permission_mode=PermissionMode.YOLO)
    async with app.run_test() as pilot:
        assert app.query_one("#transcript") is not None
        assert app.query_one("#task-input") is not None


@pytest.mark.anyio
async def test_submitting_task_runs_to_final_answer(tmp_path):
    model = ScriptedModel(responses=["A plain final answer, no Action."])
    app = DuckyAgentApp(model=model, permission_mode=PermissionMode.YOLO)
    async with app.run_test() as pilot:
        await pilot.click("#task-input")
        for ch in "do something":
            await pilot.press(ch if ch != " " else "space")
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

        transcript_text = "\n".join(str(line) for line in app.query_one("#transcript").lines)
        assert "A plain final answer" in transcript_text
        assert app._busy is False


@pytest.mark.anyio
async def test_write_action_shows_permission_modal(tmp_path):
    target = str(tmp_path / "out.txt")
    model = ScriptedModel(
        responses=[f'Action: write_file(path="{target}", content="x")', "done"]
    )
    app = DuckyAgentApp(model=model, permission_mode=PermissionMode.DEFAULT)
    async with app.run_test() as pilot:
        await pilot.click("#task-input")
        for ch in "write":
            await pilot.press(ch)
        await pilot.press("enter")
        await pilot.pause()

        # The worker thread is blocked inside on_ask waiting for the modal;
        # give it a moment to actually push the screen from the main thread.
        for _ in range(50):
            if isinstance(app.screen, PermissionModal):
                break
            await pilot.pause()
        assert isinstance(app.screen, PermissionModal)

        await pilot.click("#allow-once")
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert (tmp_path / "out.txt").read_text() == "x"


@pytest.mark.anyio
async def test_deny_via_modal_produces_zero_mutation(tmp_path):
    target = str(tmp_path / "out.txt")
    model = ScriptedModel(
        responses=[f'Action: write_file(path="{target}", content="x")', "done"]
    )
    app = DuckyAgentApp(model=model, permission_mode=PermissionMode.DEFAULT)
    async with app.run_test() as pilot:
        await pilot.click("#task-input")
        for ch in "write":
            await pilot.press(ch)
        await pilot.press("enter")
        await pilot.pause()

        for _ in range(50):
            if isinstance(app.screen, PermissionModal):
                break
            await pilot.pause()
        assert isinstance(app.screen, PermissionModal)

        await pilot.click("#deny")
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert not (tmp_path / "out.txt").exists()
