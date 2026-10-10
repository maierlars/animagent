import json
from pathlib import Path

from animagent.analysis.report import build_report
from animagent.models.profile import ReadOnlyCheck, RigProfile, StateFingerprint
from animagent.pipeline import analyze, save_artifacts
from tests.fakes.synthetic import rigify_like_extraction


def test_profile_roundtrip(tmp_path: Path) -> None:
    p = analyze(rigify_like_extraction(tail=True))
    path = tmp_path / "profile.json"
    p.save(path)
    q = RigProfile.load(path)
    assert q.schema_version == "rig_profile/1"
    assert q.extraction.schema_version == "rig_extraction/1"
    assert q.anatomy.schema_version == "anatomy/1"
    assert q.controls.schema_version == "controls/1"
    assert q.model_dump() == p.model_dump()


def test_artifacts_written_per_stage(tmp_path: Path) -> None:
    files = save_artifacts(analyze(rigify_like_extraction()), tmp_path)
    assert {"01_extraction.json", "02_skeleton.json", "03_anatomy.json", "04_controls.json", "05_report.json"} <= set(
        files
    )
    for f in files.values():
        json.loads(f.read_text())


def test_report_is_compact() -> None:
    rep = build_report(analyze(rigify_like_extraction(tail=True)))
    text = json.dumps(rep)
    assert len(text) < 30_000
    assert rep["schema_version"] == "rig_report/1"
    assert {lb["id"] for lb in rep["limbs"]} >= {"arm.L", "leg.R", "tail"}


def test_readonly_check_detects_changes() -> None:
    a = StateFingerprint(hashes={"pose": "1", "meshes": "2"})
    b = StateFingerprint(hashes={"pose": "1", "meshes": "3"})
    assert ReadOnlyCheck.compare(a, a).identical
    rc = ReadOnlyCheck.compare(a, b)
    assert not rc.identical and rc.changed_categories == ["meshes"]
    p = analyze(rigify_like_extraction(), readonly_check=rc)
    assert any(d.stage == "readonly_check" and d.level == "error" for d in p.diagnostics)
