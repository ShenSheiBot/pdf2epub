import json
import shutil
from pathlib import Path

import pytest

from pdf2epub.processors.workspace_polisher import (
    WorkspacePolishError,
    _command,
    _input_token_coverage,
    _prompt,
    run_workspace_polish,
)
from pdf2epub.core.tracking import AttemptRecord, ProcessingTracker
from pdf2epub.utils.safety import ProcessLockError, exclusive_process_lock


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "ocr_markdown"
    source.mkdir()
    (source / "chapter_1.md").write_text(
        "The army with-\ndrew.\n\n![](../images/map.png)\n\nPrinted caption.\n",
        encoding="utf-8",
    )
    (source / "chapter_2.md").write_text("Second  chapter.\n", encoding="utf-8")
    return source


def _fake_codex(tmp_path: Path, *, fail: bool = False) -> Path:
    script = tmp_path / "fake-codex"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        "args=sys.argv[1:]\n"
        "root=pathlib.Path(args[args.index('-C')+1]) if '-C' in args else pathlib.Path.cwd()\n"
        "prompt=args[-1]\n"
        "assert 'every Markdown file' in prompt or 'every listed file' in prompt\n"
        "first=root/'chapter_1.md'\n"
        "second=root/'chapter_2.md'\n"
        "assert 'Printed caption' not in first.read_text()\n"
        "assert '![](../images/map.png)' not in first.read_text()\n"
        "assert first.read_text().count('PDF2EPUB_PROTECTED_FRAGMENT_') == 2\n"
        "first_text=first.read_text()\n"
        "second_text=second.read_text()\n"
        "first.write_text(first_text.replace('with-\\ndrew', 'withdrew'))\n"
        "second.write_text(second_text.replace('Second  chapter', 'Second chapter'))\n"
        "print(json.dumps({'type':'thread.started','thread_id':'test-session'}))\n"
        "print(json.dumps({'type':'item.completed','item':{'type':'command_execution','aggregated_output':first_text+'\\n'+second_text}}))\n"
        + (
            "print(json.dumps({'type':'turn.failed','error':{'message':'failed'}})); raise SystemExit(7)\n"
            if fail
            else "print(json.dumps({'type':'turn.completed','usage':{'input_tokens':1,'output_tokens':1}}))\n"
        ),
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _fake_codex_needing_coverage_followup(tmp_path: Path) -> Path:
    script = tmp_path / "fake-codex-coverage"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        "args=sys.argv[1:]\n"
        "root=pathlib.Path(args[args.index('-C')+1]) if '-C' in args else pathlib.Path.cwd()\n"
        "first=(root/'chapter_1.md').read_text()\n"
        "second=(root/'chapter_2.md').read_text()\n"
        "is_resume='resume' in args\n"
        "output=first+'\\n'+second if is_resume else first\n"
        "print(json.dumps({'type':'thread.started','thread_id':'coverage-session'}))\n"
        "print(json.dumps({'type':'item.completed','item':{'type':'command_execution','aggregated_output':output}}))\n"
        "print(json.dumps({'type':'turn.completed','usage':{'input_tokens':1,'output_tokens':1}}))\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _fake_codex_deleting_a_source_file(tmp_path: Path) -> Path:
    script = tmp_path / "fake-codex-file-set"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        "args=sys.argv[1:]\n"
        "root=pathlib.Path(args[args.index('-C')+1]) if '-C' in args else pathlib.Path.cwd()\n"
        "is_resume='resume' in args\n"
        "first=root/'chapter_1.md'\n"
        "second=root/'chapter_2.md'\n"
        "assert second.exists()\n"
        "first_text=first.read_text()\n"
        "second_text=second.read_text()\n"
        "if is_resume:\n"
        "    second.write_text(second_text.replace('Second  chapter', 'Second chapter'))\n"
        "else:\n"
        "    second.unlink()\n"
        "print(json.dumps({'type':'thread.started','thread_id':'file-set-session'}))\n"
        "print(json.dumps({'type':'item.completed','item':{'type':'command_execution','aggregated_output':first_text+'\\n'+second_text}}))\n"
        "print(json.dumps({'type':'turn.completed','usage':{'input_tokens':1,'output_tokens':1}}))\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _fake_codex_dropping_protected_fragment(
    tmp_path: Path, *, repairs: bool
) -> Path:
    script = tmp_path / f"fake-codex-protected-{int(repairs)}"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        "args=sys.argv[1:]\n"
        "root=pathlib.Path(args[args.index('-C')+1]) if '-C' in args else pathlib.Path.cwd()\n"
        "is_resume='resume' in args\n"
        "first=root/'chapter_1.md'\n"
        "second=root/'chapter_2.md'\n"
        "first_text=first.read_text()\n"
        "second_text=second.read_text()\n"
        "placeholder='<!--PDF2EPUB_PROTECTED_FRAGMENT_000001-->'\n"
        "if not is_resume:\n"
        "    first.write_text(first_text.replace(placeholder, 'PROTECTED_DROP_POINT', 1))\n"
        + (
            "else:\n    first.write_text(first_text.replace('PROTECTED_DROP_POINT', placeholder, 1))\n"
            if repairs
            else ""
        )
        + "print(json.dumps({'type':'thread.started','thread_id':'protected-session'}))\n"
        "print(json.dumps({'type':'item.completed','item':{'type':'command_execution','aggregated_output':first_text+'\\n'+second_text}}))\n"
        "print(json.dumps({'type':'turn.completed','usage':{'input_tokens':1,'output_tokens':1}}))\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _fake_codex_targeted_followup_fails_once(tmp_path: Path) -> Path:
    script = tmp_path / "fake-codex-targeted-resume"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        "args=sys.argv[1:]\n"
        "root=pathlib.Path(args[args.index('-C')+1]) if '-C' in args else pathlib.Path.cwd()\n"
        "prompt=args[-1]\n"
        "is_resume='resume' in args\n"
        "first=root/'chapter_1.md'\n"
        "second=root/'chapter_2.md'\n"
        "first_text=first.read_text()\n"
        "second_text=second.read_text()\n"
        "placeholder='<!--PDF2EPUB_PROTECTED_FRAGMENT_000001-->'\n"
        "marker=root/'.targeted-followup-failed'\n"
        "print(json.dumps({'type':'thread.started','thread_id':'targeted-session'}))\n"
        "if not is_resume:\n"
        "    first.write_text(first_text.replace(placeholder, 'PROTECTED_DROP_POINT', 1))\n"
        "    print(json.dumps({'type':'item.completed','item':{'type':'command_execution','aggregated_output':first_text+'\\n'+second_text}}))\n"
        "    print(json.dumps({'type':'turn.completed'}))\n"
        "elif not marker.exists():\n"
        "    assert 'protected image or' in prompt\n"
        "    marker.write_text('failed')\n"
        "    print(json.dumps({'type':'turn.failed','error':{'message':'simulated targeted failure'}}))\n"
        "    raise SystemExit(7)\n"
        "else:\n"
        "    assert 'protected image or' in prompt\n"
        "    first.write_text(first_text.replace('PROTECTED_DROP_POINT', placeholder, 1))\n"
        "    print(json.dumps({'type':'turn.completed'}))\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def test_direct_codex_polishes_folder_and_restores_protected_content(tmp_path):
    output = tmp_path / "polished"
    result = run_workspace_polish(
        input_dir=_source(tmp_path),
        output_dir=output,
        model="gpt-5.6-terra",
        codex_binary=str(_fake_codex(tmp_path)),
        protected_fragments=["Printed caption."],
        run_root=tmp_path / "workspace",
    )

    assert result.completed == 2
    first = (output / "validated" / "chapter_1.md").read_text()
    assert "withdrew" in first
    assert "![](../images/map.png)" in first
    assert "Printed caption." in first
    assert "PDF2EPUB_PROTECTED_FRAGMENT_" not in first
    assert "Second chapter" in (output / "validated" / "chapter_2.md").read_text()
    state = json.loads(result.state_path.read_text())
    assert state["status"] == "published"
    assert state["session_id"] == "test-session"
    assert state["attempts"] == 1
    coverage = json.loads(
        (output / "workspace_agent" / "input_token_coverage.json").read_text()
    )
    assert coverage["coverage"] == 1.0
    assert coverage["complete"] is True


def test_deleted_protected_fragment_is_repaired_in_same_session(tmp_path):
    output = tmp_path / "polished"
    result = run_workspace_polish(
        input_dir=_source(tmp_path),
        output_dir=output,
        codex_binary=str(
            _fake_codex_dropping_protected_fragment(tmp_path, repairs=True)
        ),
        protected_fragments=["Printed caption."],
        run_root=tmp_path / "workspace",
    )

    state = json.loads(result.state_path.read_text())
    assert state["attempts"] == 2
    assert state["protected_fragment_followups"] == 1
    assert "protected_fragment_mismatches" not in state
    assert "![](../images/map.png)" in (
        output / "validated" / "chapter_1.md"
    ).read_text()


def test_unrepaired_protected_fragment_is_not_published(tmp_path):
    output = tmp_path / "polished"
    source = _source(tmp_path)
    fake = _fake_codex_dropping_protected_fragment(tmp_path, repairs=False)
    with pytest.raises(WorkspacePolishError, match="did not preserve protected"):
        run_workspace_polish(
            input_dir=source,
            output_dir=output,
            codex_binary=str(fake),
            protected_fragments=["Printed caption."],
            run_root=tmp_path / "workspace",
        )

    validation = json.loads(
        (output / "workspace_agent" / "validation.json").read_text()
    )
    assert validation["valid"] is False
    assert not (output / "validated").exists()

    state_path = output / "workspace_agent" / "state.json"
    state = json.loads(state_path.read_text())
    attempts = state["attempts"]
    run_file = Path(state["run_dir"]) / "chapter_1.md"
    run_file.write_text(
        run_file.read_text().replace(
            "PROTECTED_DROP_POINT",
            "<!--PDF2EPUB_PROTECTED_FRAGMENT_000001-->",
            1,
        )
    )
    fake.unlink()

    result = run_workspace_polish(
        input_dir=source,
        output_dir=output,
        codex_binary=str(fake),
        protected_fragments=["Printed caption."],
        resume=True,
    )
    resumed = json.loads(result.state_path.read_text())
    assert resumed["attempts"] == attempts
    assert resumed["status"] == "published"


def test_failed_targeted_followup_resumes_the_same_targeted_prompt(tmp_path):
    output = tmp_path / "polished"
    source = _source(tmp_path)
    fake = _fake_codex_targeted_followup_fails_once(tmp_path)

    with pytest.raises(WorkspacePolishError, match="Codex failed"):
        run_workspace_polish(
            input_dir=source,
            output_dir=output,
            codex_binary=str(fake),
            protected_fragments=["Printed caption."],
            run_root=tmp_path / "workspace",
        )

    failed_state = json.loads(
        (output / "workspace_agent" / "state.json").read_text()
    )
    assert failed_state["pending_followup"]["kind"] == "protected"
    assert failed_state["attempts"] == 2

    result = run_workspace_polish(
        input_dir=source,
        output_dir=output,
        codex_binary=str(fake),
        protected_fragments=["Printed caption."],
        resume=True,
    )
    resumed_state = json.loads(result.state_path.read_text())
    assert resumed_state["attempts"] == 3
    assert resumed_state["session_id"] == "targeted-session"
    assert "pending_followup" not in resumed_state
    assert resumed_state["status"] == "published"


def test_workspace_publish_replaces_stale_api_tracker(tmp_path):
    output = tmp_path / "polished"
    output.mkdir()
    (output / "raw").mkdir()
    (output / "validated").mkdir()
    (output / "validated" / "chapter_1.md").write_text("old", encoding="utf-8")
    tracker = ProcessingTracker(output / "processing_tracker.json", "OldApiPolisher")
    tracker.record_attempt(
        "chapter_1",
        AttemptRecord(timestamp=1, status="completed", model="old-api"),
    )

    result = run_workspace_polish(
        input_dir=_source(tmp_path),
        output_dir=output,
        codex_binary=str(_fake_codex(tmp_path)),
        protected_fragments=["Printed caption."],
        run_root=tmp_path / "workspace",
    )

    current = json.loads((output / "processing_tracker.json").read_text())
    assert current["processor"] == "CodexWorkspacePolisher"
    assert set(current["units"]) == {"chapter_1", "chapter_2"}
    assert all(
        unit["attempts"][-1]["model"] == "codex-cli:gpt-5.6-terra"
        for unit in current["units"].values()
    )
    histories = list((output / "workspace_agent" / "published_history").glob("*"))
    assert len(histories) == 1
    assert (histories[0] / "processing_tracker.json").is_file()


def test_relative_codex_path_is_resolved_before_workspace_chdir(
    tmp_path, monkeypatch
):
    fake = _fake_codex(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = run_workspace_polish(
        input_dir=_source(tmp_path),
        output_dir=tmp_path / "polished",
        codex_binary=f"./{fake.name}",
        protected_fragments=["Printed caption."],
        run_root=tmp_path / "workspace",
    )

    assert result.completed == 2


def test_polish_stage_lock_rejects_a_second_writer(tmp_path):
    lock_path = tmp_path / ".polish.lock"
    with exclusive_process_lock(lock_path, "polish"):
        with pytest.raises(ProcessLockError, match="Another polish process"):
            with exclusive_process_lock(lock_path, "polish"):
                pass


def test_missing_markdown_file_is_restored_and_returned_to_same_session(tmp_path):
    output = tmp_path / "polished"
    result = run_workspace_polish(
        input_dir=_source(tmp_path),
        output_dir=output,
        model="gpt-5.6-terra",
        codex_binary=str(_fake_codex_deleting_a_source_file(tmp_path)),
        protected_fragments=["Printed caption."],
        run_root=tmp_path / "workspace",
    )

    assert result.completed == 2
    assert sorted(path.name for path in (output / "validated").glob("*.md")) == [
        "chapter_1.md",
        "chapter_2.md",
    ]
    assert "Second chapter" in (
        output / "validated" / "chapter_2.md"
    ).read_text()
    state = json.loads(result.state_path.read_text())
    assert state["status"] == "published"
    assert state["attempts"] == 2
    assert state["file_set_followups"] == 1
    assert "file_set_mismatch" not in state


def test_codex_failure_retains_direct_workspace_and_session(tmp_path):
    output = tmp_path / "polished"
    with pytest.raises(WorkspacePolishError, match="state retained"):
        run_workspace_polish(
            input_dir=_source(tmp_path),
            output_dir=output,
            codex_binary=str(_fake_codex(tmp_path, fail=True)),
            protected_fragments=["Printed caption."],
            run_root=tmp_path / "workspace",
        )

    state = json.loads((output / "workspace_agent" / "state.json").read_text())
    assert state["status"] == "codex_failed"
    assert state["session_id"] == "test-session"
    assert Path(state["run_dir"]).is_dir()
    assert not (output / "validated").exists()


def test_prompt_gives_codex_method_freedom_and_complete_scope():
    prompt = _prompt("academic", resume=False)
    assert "every Markdown file" in prompt
    assert "every repair you can make" in prompt
    assert "Choose your own" in prompt
    assert "reading order" in prompt
    assert "context compaction" in prompt
    assert "Preserve genuine Markdown footnotes" in prompt
    assert "semantic continuity rather than mere adjacency" in prompt
    assert "same physical-page material across adjacent files" in prompt
    assert "Never invent a footnote or key" in prompt


def test_command_uses_normal_codex_config_without_policy_overrides(tmp_path):
    command = _command(
        codex="codex",
        model="gpt-5.6-terra",
        run_dir=tmp_path,
        prompt="do the work",
        session_id=None,
    )
    joined = " ".join(command)
    assert "--ignore-user-config" not in command
    assert "--ignore-rules" not in command
    assert "model_provider" not in joined
    assert command[-1] == "do the work"
    assert command[command.index("-C") + 1] == str(tmp_path)


def test_command_resumes_the_same_normal_codex_session(tmp_path):
    command = _command(
        codex="codex",
        model="gpt-5.6-terra",
        run_dir=tmp_path,
        prompt="continue",
        session_id="session-123",
    )
    assert command[:3] == ["codex", "exec", "resume"]
    assert "session-123" in command
    assert command[-1] == "continue"


def test_input_token_coverage_uses_exact_tool_output_content(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    first = " ".join(f"alpha{i}" for i in range(80))
    second = " ".join(f"beta{i}" for i in range(80))
    (snapshot / "first.md").write_text(first)
    (snapshot / "second.md").write_text(second)
    events = tmp_path / "events.jsonl"
    events.write_text(
        "\n".join(
            json.dumps(event)
            for event in (
                {"type": "thread.started", "thread_id": "session"},
                {
                    "type": "item.completed",
                    "item": {
                        "type": "command_execution",
                        "aggregated_output": first + "\n" + " ".join(second.split()[:24]),
                    },
                },
                {"type": "turn.completed"},
            )
        )
    )

    report = _input_token_coverage(
        snapshot_dir=snapshot,
        events_path=events,
        session_id="session",
    )

    assert report["files"]["first.md"]["coverage"] == 1.0
    assert report["files"]["second.md"]["coverage"] < 0.8
    assert report["below_threshold"] == ["second.md"]


def test_input_token_coverage_recognizes_standard_unified_diff_content(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    source_line = "The exact source paragraph before a footnote-layout repair."
    (snapshot / "diffed.md").write_text(source_line + "\n")
    events = tmp_path / "events.jsonl"
    events.write_text(
        "\n".join(
            json.dumps(event)
            for event in (
                {"type": "thread.started", "thread_id": "session"},
                {
                    "type": "item.completed",
                    "item": {
                        "type": "command_execution",
                        "command": "sed -n '1,2000p' /tmp/footnote-layout.patch",
                        "aggregated_output": (
                            "*** Begin Patch\n"
                            "*** Update File: diffed.md\n"
                            "@@ -1 +1 @@\n"
                            f"-{source_line}\n"
                            "+The repaired paragraph after a footnote-layout repair.\n"
                            "*** End Patch\n"
                        ),
                    },
                },
                {"type": "turn.completed"},
            )
        )
    )

    report = _input_token_coverage(
        snapshot_dir=snapshot,
        events_path=events,
        session_id="session",
    )

    assert report["files"]["diffed.md"]["coverage"] == 1.0
    assert report["complete"] is True


def test_input_token_coverage_consumes_repeated_occurrences_once(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "repeated.md").write_text("alpha\n" * 100)
    events = tmp_path / "events.jsonl"
    events.write_text(
        "\n".join(
            json.dumps(event)
            for event in (
                {"type": "thread.started", "thread_id": "session"},
                {
                    "type": "item.completed",
                    "item": {
                        "type": "command_execution",
                        "aggregated_output": "alpha\n" * 9,
                    },
                },
                {"type": "turn.completed"},
            )
        )
    )

    report = _input_token_coverage(
        snapshot_dir=snapshot,
        events_path=events,
        session_id="session",
    )

    assert report["files"]["repeated.md"]["covered_tokens"] == 9
    assert report["files"]["repeated.md"]["source_tokens"] == 100
    assert report["complete"] is False


def test_input_token_coverage_recognizes_numbered_full_file_output(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    lines = [f"Unique source line {index}" for index in range(1, 101)]
    (snapshot / "numbered.md").write_text("\n".join(lines) + "\n")
    displayed = "\n".join(
        f"{index:6d}\t{line}" for index, line in enumerate(lines, start=1)
    )
    events = tmp_path / "events.jsonl"
    events.write_text(
        "\n".join(
            json.dumps(event)
            for event in (
                {"type": "thread.started", "thread_id": "session"},
                {
                    "type": "item.completed",
                    "item": {
                        "type": "command_execution",
                        "command": "nl -ba numbered.md",
                        "aggregated_output": displayed,
                    },
                },
                {"type": "turn.completed"},
            )
        )
    )

    report = _input_token_coverage(
        snapshot_dir=snapshot,
        events_path=events,
        session_id="session",
    )

    assert report["files"]["numbered.md"]["coverage"] == 1.0
    assert report["files"]["numbered.md"]["unmatched_content_lines"] == 0
    assert report["complete"] is True


def test_input_token_coverage_does_not_strip_unrecognized_source_text(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "literal.md").write_text("alpha\n")
    events = tmp_path / "events.jsonl"
    events.write_text(
        "\n".join(
            json.dumps(event)
            for event in (
                {"type": "thread.started", "thread_id": "session"},
                {
                    "type": "item.completed",
                    "item": {
                        "type": "command_execution",
                        "command": "sed -n '1,5p' literal.md",
                        "aggregated_output": "     1\talpha\n",
                    },
                },
                {"type": "turn.completed"},
            )
        )
    )

    report = _input_token_coverage(
        snapshot_dir=snapshot,
        events_path=events,
        session_id="session",
    )

    assert report["files"]["literal.md"]["coverage"] == 0.0
    assert report["complete"] is False


def test_input_token_coverage_recognizes_equivalent_display_commands(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "variants.md").write_text("alpha\nbeta\ngamma\n")
    commands_and_outputs = (
        ("nl<variants.md", "     1\talpha\n"),
        ("cat -bn variants.md", "     2\tbeta\n"),
        ("git -C . diff -- variants.md", "-gamma\n"),
    )
    events_data = [{"type": "thread.started", "thread_id": "session"}]
    events_data.extend(
        {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": command,
                "aggregated_output": output,
            },
        }
        for command, output in commands_and_outputs
    )
    events_data.append({"type": "turn.completed"})
    events = tmp_path / "events.jsonl"
    events.write_text("\n".join(json.dumps(event) for event in events_data))

    report = _input_token_coverage(
        snapshot_dir=snapshot,
        events_path=events,
        session_id="session",
    )

    assert report["files"]["variants.md"]["coverage"] == 1.0
    assert report["complete"] is True


def test_low_input_token_coverage_automatically_resumes_once(tmp_path):
    output = tmp_path / "polished"
    result = run_workspace_polish(
        input_dir=_source(tmp_path),
        output_dir=output,
        codex_binary=str(_fake_codex_needing_coverage_followup(tmp_path)),
        protected_fragments=["Printed caption."],
        run_root=tmp_path / "workspace",
    )

    state = json.loads(result.state_path.read_text())
    report = json.loads(
        (output / "workspace_agent" / "input_token_coverage.json").read_text()
    )
    assert state["attempts"] == 2
    assert state["coverage_followups"] == 1
    assert report["complete"] is True


def test_publish_failure_is_remasked_before_resume(tmp_path, monkeypatch):
    import pdf2epub.processors.workspace_polisher as module

    output = tmp_path / "polished"
    source = _source(tmp_path)
    fake = _fake_codex(tmp_path)
    real_publish = module._publish

    def fail_publish(**kwargs):
        raise OSError("simulated publish failure")

    monkeypatch.setattr(module, "_publish", fail_publish)
    with pytest.raises(OSError, match="simulated publish failure"):
        run_workspace_polish(
            input_dir=source,
            output_dir=output,
            codex_binary=str(fake),
            protected_fragments=["Printed caption."],
            run_root=tmp_path / "workspace",
        )

    state = json.loads((output / "workspace_agent" / "state.json").read_text())
    assert "Printed caption." in (
        Path(state["run_dir"]) / "chapter_1.md"
    ).read_text()

    monkeypatch.setattr(module, "_publish", real_publish)
    result = run_workspace_polish(
        input_dir=source,
        output_dir=output,
        codex_binary=str(fake),
        protected_fragments=["Printed caption."],
        resume=True,
    )
    assert result.completed == 2
    resumed_state = json.loads(result.state_path.read_text())
    assert resumed_state["attempts"] == 1
    assert "Printed caption." in (
        output / "validated" / "chapter_1.md"
    ).read_text()


def test_resume_after_completed_followup_does_not_run_it_twice(tmp_path, monkeypatch):
    import pdf2epub.processors.workspace_polisher as module

    output = tmp_path / "polished"
    source = _source(tmp_path)
    real_coverage = module._input_token_coverage
    coverage_calls = 0

    def crash_after_followup(**kwargs):
        nonlocal coverage_calls
        coverage_calls += 1
        if coverage_calls == 2:
            raise OSError("simulated crash after completed followup")
        return real_coverage(**kwargs)

    monkeypatch.setattr(module, "_input_token_coverage", crash_after_followup)
    with pytest.raises(OSError, match="completed followup"):
        run_workspace_polish(
            input_dir=source,
            output_dir=output,
            codex_binary=str(_fake_codex_needing_coverage_followup(tmp_path)),
            protected_fragments=["Printed caption."],
            run_root=tmp_path / "workspace",
        )

    state_path = output / "workspace_agent" / "state.json"
    crashed_state = json.loads(state_path.read_text())
    assert crashed_state["status"] == "codex_complete"
    assert crashed_state["attempts"] == 2
    assert crashed_state["coverage_followups"] == 1

    monkeypatch.setattr(module, "_input_token_coverage", real_coverage)
    result = run_workspace_polish(
        input_dir=source,
        output_dir=output,
        codex_binary=str(_fake_codex_needing_coverage_followup(tmp_path)),
        protected_fragments=["Printed caption."],
        resume=True,
    )

    resumed_state = json.loads(result.state_path.read_text())
    assert resumed_state["attempts"] == 2
    assert resumed_state["status"] == "published"


def test_unpublished_v2_state_builds_snapshot_on_resume(tmp_path):
    source = _source(tmp_path)
    output = tmp_path / "polished"
    with pytest.raises(WorkspacePolishError, match="state retained"):
        run_workspace_polish(
            input_dir=source,
            output_dir=output,
            codex_binary=str(_fake_codex(tmp_path, fail=True)),
            protected_fragments=["Printed caption."],
            run_root=tmp_path / "workspace",
        )

    state_path = output / "workspace_agent" / "state.json"
    state = json.loads(state_path.read_text())
    snapshot = Path(state.pop("input_snapshot_dir"))
    state["version"] = 2
    state_path.write_text(json.dumps(state))
    shutil.rmtree(snapshot)

    result = run_workspace_polish(
        input_dir=source,
        output_dir=output,
        codex_binary=str(_fake_codex(tmp_path, fail=False)),
        protected_fragments=["Printed caption."],
        resume=True,
    )

    resumed_state = json.loads(result.state_path.read_text())
    assert resumed_state["version"] == 3
    assert Path(resumed_state["input_snapshot_dir"]).is_dir()
    assert resumed_state["status"] == "published"
