"""A legacy PASS must never stand in for new-family native acceptance."""

import pytest

from scripts import campaign_native_ci as ci


@pytest.mark.parametrize("child", ["", "<skipped/>", "<failure/>", "<error/>"])
def test_distinct_native_target_requires_exact_non_skipped_pass(tmp_path, monkeypatch, child):
    monkeypatch.chdir(tmp_path)
    ci.CAUSAL_REPORT.write_text(
        f'<testsuites><testsuite><testcase name="{ci.CAUSAL_NATIVE_TARGET}">'
        f"{child}</testcase></testsuite></testsuites>",
        encoding="utf-8",
    )
    if child:
        with pytest.raises(ci.CampaignCIError):
            ci.check_result(causal=True)
    else:
        ci.check_result(causal=True)


@pytest.mark.parametrize("name", [ci.NATIVE_TARGET, "other", ""])
def test_legacy_or_wrong_target_cannot_be_accepted_as_causal(tmp_path, monkeypatch, name):
    monkeypatch.chdir(tmp_path)
    ci.CAUSAL_REPORT.write_text(f'<testsuite><testcase name="{name}"/></testsuite>', encoding="utf-8")
    with pytest.raises(ci.CampaignCIError):
        ci.check_result(causal=True)


def test_missing_new_receipt_not_substituted_by_legacy_receipt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ci.REPORT.write_text(f'<testsuite><testcase name="{ci.NATIVE_TARGET}"/></testsuite>', encoding="utf-8")
    ci.check_result()
    with pytest.raises(ci.CampaignCIError):
        ci.check_result(causal=True)
