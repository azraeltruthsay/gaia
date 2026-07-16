"""Tests: CodeMind restart-manifest builder (nfi3) — the loop-closure step."""

from gaia_common.utils.codemind_manifest import (
    build_manifest_for_files,
    service_for_candidate_file,
)


class TestServiceMapping:
    def test_candidate_paths_map(self):
        assert service_for_candidate_file("candidates/gaia-study/gaia_study/main.py") == "study"
        assert service_for_candidate_file("./candidates/gaia-core/gaia_core/config.py") == "core"
        assert service_for_candidate_file("candidates/gaia-common/gaia_common/config.py") == "common"

    def test_non_candidate_and_unknown_rejected(self):
        assert service_for_candidate_file("gaia-study/gaia_study/main.py") is None
        assert service_for_candidate_file("candidates/gaia-doctor/doctor.py") is None
        assert service_for_candidate_file("candidates/not-a-service/x.py") is None


class TestBuild:
    def test_single_service_manifest(self):
        m, err = build_manifest_for_files(
            ["candidates/gaia-study/gaia_study/indexer.py"], bead="GAIA_Project-test")
        assert err is None
        assert m["services"] == ["study"]
        assert m["promote_files"] == ["candidates/gaia-study/gaia_study/indexer.py"]
        assert m["status"] == "pending"
        assert m["tier"] == "source"
        assert m["manifest_version"] == 1
        assert m["requested_by"] == "codemind"
        assert "codemind_study" in m["id"]

    def test_multi_service(self):
        m, err = build_manifest_for_files([
            "candidates/gaia-study/gaia_study/indexer.py",
            "candidates/gaia-web/gaia_web/main.py",
        ])
        assert err is None
        assert m["services"] == ["study", "web"]
        assert len(m["promote_files"]) == 2

    def test_common_only_rides_with_core(self):
        m, err = build_manifest_for_files(["candidates/gaia-common/gaia_common/config.py"])
        assert err is None
        assert m["services"] == ["common", "core"]

    def test_doctor_file_aborts_build(self):
        m, err = build_manifest_for_files([
            "candidates/gaia-study/gaia_study/indexer.py",
            "candidates/gaia-doctor/doctor.py",
        ])
        assert m is None and "not a deployable" in err

    def test_prod_path_aborts_build(self):
        m, err = build_manifest_for_files(["gaia-study/gaia_study/indexer.py"])
        assert m is None and err

    def test_empty_aborts(self):
        m, err = build_manifest_for_files([])
        assert m is None and err

    def test_quotes_in_evidence_survive(self):
        m, err = build_manifest_for_files(
            ["candidates/gaia-study/gaia_study/indexer.py"],
            evidence=['pytest said "41 passed" today'])
        assert err is None
        assert '"41 passed"' in m["tests_run"][0]
