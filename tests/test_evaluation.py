import json

import pytest
from PIL import Image

from grounding.config import load_config
from grounding.evaluation import evaluate, point_in_polygon, read_manifest, score_point


def test_polygon_uses_region_instead_of_enclosing_box():
    triangle = [[10, 10], [90, 10], [10, 90]]
    assert score_point([0.2, 0.2], None, 100, 100, polygon=triangle)
    assert not score_point([0.8, 0.8], None, 100, 100, polygon=triangle)
    assert not score_point(None, None, 100, 100, polygon=triangle)
    concave = [[10, 10], [90, 10], [90, 30], [30, 30], [30, 90], [10, 90]]
    assert point_in_polygon((20, 80), concave)
    assert not point_in_polygon((80, 80), concave)
    assert point_in_polygon((20, 80), list(reversed(concave)))


def test_polygon_edges_follow_upstream_ray_crossing():
    square = [[10, 10], [90, 10], [90, 90], [10, 90]]
    # Unlike rectangles, the upstream polygon test is semi-open at boundaries.
    for point in [(10, 10), (10, 50), (50, 10)]:
        assert point_in_polygon(point, square)
    for point in [(90, 90), (90, 50), (50, 90)]:
        assert not point_in_polygon(point, square)
    assert point_in_polygon((20, 20), square + [square[0]])


def test_rectangle_and_absence_contract_is_preserved():
    assert score_point([0.1, 0.2], [10, 20, 30, 40], 100, 100)
    assert score_point([0.3, 0.4], [10, 20, 30, 40], 100, 100)
    assert score_point(None, None, 100, 100)
    assert not score_point(None, [10, 20, 30, 40], 100, 100)
    assert not score_point([0.2, 0.3], None, 100, 100)
    assert not score_point([True, 0.3], None, 100, 100)


@pytest.mark.parametrize(
    "geometry",
    [
        {},
        {"bbox_xyxy": None, "polygon_xy": [[0, 0], [10, 0], [0, 10]]},
        {"polygon_xy": None},
        {"polygon_xy": [[0, 0], [10, 0]]},
        {"polygon_xy": [0, 0, 10, 0, 0, 10]},
        {"polygon_xy": [[0, 0], [10, 10], [20, 20]]},
        {"polygon_xy": [[0, 0], [101, 0], [0, 10]]},
        {"polygon_xy": [[0, 0], [10, 0], [0, -1]]},
        {"polygon_xy": [[0, 0], [10, 0], [0, float("nan")]]},
        {"polygon_xy": [[0, 0], [True, 0], [0, 10]]},
    ],
)
def test_invalid_geometry_is_rejected_before_execution(tmp_path, geometry):
    Image.new("RGB", (100, 100)).save(tmp_path / "screen.png")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(dict(image="screen.png", instruction="Locate target", **geometry)) + "\n")
    with pytest.raises(ValueError):
        list(read_manifest(manifest))


def test_mixed_geometry_accuracy_counts_all_cases_and_failures(tmp_path):
    Image.new("RGB", (100, 100)).save(tmp_path / "screen.png")
    candidate = tmp_path / "candidate.py"
    candidate.write_text(
        "def ground(screenshot, instruction, pool):\n"
        " if instruction == 'crash': raise RuntimeError('failed')\n"
        " return {'point': None if instruction == 'absent' else [0.8, 0.8]}\n"
    )
    triangle = [[10, 10], [90, 10], [10, 90]]
    rows = [
        {"instruction": "rectangle", "bbox_xyxy": [70, 70, 90, 90]},
        {"instruction": "polygon-hit", "polygon_xy": [[70, 70], [90, 70], [90, 90], [70, 90]]},
        {"instruction": "polygon-miss", "polygon_xy": triangle},
        {"instruction": "absent", "bbox_xyxy": None},
        {"instruction": "crash", "polygon_xy": triangle},
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(dict(row, id=str(i), image="screen.png")) + "\n"
                                for i, row in enumerate(rows)))
    summary = evaluate(manifest, candidate, load_config(), tmp_path / "results")
    assert summary["examples"] == 5
    assert summary["correct"] == 3
    assert summary["accuracy"] == 0.6
    assert summary["operational_failures"] == 1
    assert summary["by_target_type"]["polygon"] == {
        "examples": 3, "correct": 1, "accuracy": 1 / 3, "operational_failures": 1,
    }
    assert summary["by_target_type"]["rectangle"]["accuracy"] == 1
    assert summary["by_target_type"]["absent"]["accuracy"] == 1
    metadata = json.loads((tmp_path / "results/metadata.json").read_text())
    assert len(metadata["evaluator_sha256"]) == 64
