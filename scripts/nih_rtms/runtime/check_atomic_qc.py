"""Exercise real concurrent QC writes inside an isolated /data test mount."""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from nipype.utils.filemanip import copyfile


TARGETS = (
    Path("/data/xcpd_surf/restfmri/desc-linc_qc.json"),
    Path("/data/xcpd/restfmri/desc-linc_qc.json"),
)


def writer(worker):
    source = Path(f"/data/source_{worker}.json")
    source.write_text(json.dumps({"worker": worker, "payload": "x" * 65536}))
    for _ in range(64):
        for target in TARGETS:
            result = copyfile(str(source), str(target), copy=True, use_hardlink=True)
            assert result == str(target)
            current = json.loads(target.read_bytes())
            assert current["worker"] in (0, 1)
            assert current["payload"] == "x" * 65536
    return worker


if __name__ == "__main__":
    assert getattr(copyfile, "_nih_rtms_atomic_qc_copy", False)
    for target in TARGETS:
        target.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        assert set(pool.map(writer, (0, 1))) == {0, 1}

    invalid = Path("/data/invalid.json")
    invalid.write_text("{invalid")
    before = TARGETS[0].read_bytes()
    try:
        copyfile(str(invalid), str(TARGETS[0]), copy=True)
    except json.JSONDecodeError:
        pass
    else:
        raise AssertionError("Malformed QC metadata was accepted")
    assert TARGETS[0].read_bytes() == before

    source = Path("/data/unrelated_source.txt")
    destination = Path("/data/unrelated_destination.txt")
    source.write_text("ordinary Nipype copy")
    copyfile(str(source), str(destination), copy=True)
    assert destination.read_bytes() == source.read_bytes()
    for target in TARGETS:
        assert not list(target.parent.glob(".nih_rtms_atomic_qc_*"))
    print("PASS: 256 concurrent atomic QC writes, malformed-input rejection, ordinary copy")
