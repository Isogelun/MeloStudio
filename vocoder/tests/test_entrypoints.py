from vocoder.entrypoints.__main__ import COMMANDS, main
from vocoder.entrypoints import preprocess as preprocess_entry


def test_unified_cli_exposes_four_workflows(capsys):
    assert set(COMMANDS) == {"preprocess", "train", "export", "infer"}
    main([])
    output = capsys.readouterr().out
    for command in COMMANDS:
        assert command in output


def test_preprocess_entry_selects_bwe_pipeline(monkeypatch):
    called = []
    monkeypatch.setattr(preprocess_entry.preprocess_bwe, "main", lambda: called.append("bwe"))
    preprocess_entry.main(["--pipeline", "bwe", "input", "output"])
    assert called == ["bwe"]
