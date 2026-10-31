"""Explicit coordinate conventions. Never infer units from numerical magnitude."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass

SPACES = {"pixels", "norm1000", "norm999", "norm01"}


@dataclass(frozen=True)
class ImageFrame:
    original_size: tuple[int, int]
    image_size: tuple[int, int]
    # Input-image pixels -> original-image pixels, including resize/translation/letterboxing.
    affine: tuple[float, float, float, float, float, float]

    @classmethod
    def full(cls, size):
        return cls(tuple(size), tuple(size), (1, 0, 0, 0, 1, 0))

    @classmethod
    def crop(cls, original_size, crop_xyxy, image_size):
        x0, y0, x1, y1 = crop_xyxy
        width, height = image_size
        if width <= 0 or height <= 0 or x1 <= x0 or y1 <= y0:
            raise ValueError("Invalid crop or input-image dimensions")
        return cls(
            tuple(original_size), tuple(image_size), ((x1 - x0) / width, 0, x0, 0, (y1 - y0) / height, y0)
        )

    def to_dict(self):
        return asdict(self)


def finite_values(values, count):
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise ValueError(f"Expected {count} coordinates")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError("Coordinates must be finite numbers")
    return [float(v) for v in values]


def point_to_input_norm01(point, space, image_size):
    if space not in SPACES:
        raise ValueError(f"Unknown coordinate space: {space}")
    x, y = finite_values(point, 2)
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image dimensions")
    scale_x, scale_y = (
        (width, height)
        if space == "pixels"
        else ((1000, 1000) if space == "norm1000" else ((999, 999) if space == "norm999" else (1, 1)))
    )
    point = [x / scale_x, y / scale_y]
    if any(v < 0 or v > 1 for v in point):
        raise ValueError(f"Coordinates out of range for declared space {space}")
    return point


def to_original_point(point, coordinate_space, frame: ImageFrame | dict):
    frame = ImageFrame(**frame) if isinstance(frame, dict) else frame
    if len(frame.original_size) != 2 or any(v <= 0 for v in frame.original_size):
        raise ValueError("Invalid original screenshot dimensions")
    u, v = point_to_input_norm01(point, coordinate_space, frame.image_size)
    x, y = u * frame.image_size[0], v * frame.image_size[1]
    a, b, c, d, e, f = finite_values(frame.affine, 6)
    original = [(a * x + b * y + c) / frame.original_size[0], (d * x + e * y + f) / frame.original_size[1]]
    if any(v < 0 or v > 1 or not math.isfinite(v) for v in original):
        raise ValueError("Point maps outside the original screenshot; no silent clipping")
    return original


def _extract(value):
    if value is None:
        return None, "refusal"
    if isinstance(value, list) and not value:
        return None, "refusal"
    if isinstance(value, dict):
        for key in ("point_2d", "point", "coordinate", "coordinates", "position", "click_point"):
            if key in value:
                return _extract(value[key])
        for key in ("bbox_2d", "bbox_xyxy", "bbox", "box"):
            if key in value:
                box = finite_values(value[key], 4)
                if box[2] < box[0] or box[3] < box[1]:
                    raise ValueError("Reversed xyxy bounding box")
                return box, "bbox_xyxy"
        if "x" in value and "y" in value:
            return finite_values([value["x"], value["y"]], 2), "point"
        if "arguments" in value:
            arguments = value["arguments"]
            return _extract(json.loads(arguments) if isinstance(arguments, str) else arguments)
        raise ValueError("No supported point or bounding-box field")
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], (list, dict)):
        return _extract(value[0])
    if isinstance(value, list) and len(value) in (2, 4):
        coordinates = finite_values(value, len(value))
        if len(value) == 4 and (coordinates[2] < coordinates[0] or coordinates[3] < coordinates[1]):
            raise ValueError("Reversed xyxy bounding box")
        return coordinates, "point" if len(value) == 2 else "bbox_xyxy"
    raise ValueError("Ambiguous or unsupported grounding response")


def parse_grounding(text):
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if cleaned.lower() in {"null", "none"}:
        return None, "refusal"
    decoder = json.JSONDecoder()
    candidates = []
    index = 0
    while index < len(cleaned):
        if cleaned[index] in "[{":
            try:
                value, end = decoder.raw_decode(cleaned, index)
                candidates.append(value)
                index = end
                continue
            except ValueError:
                pass
        index += 1
    parsed = []
    for candidate in candidates:
        try:
            parsed.append(_extract(candidate))
        except ValueError:
            continue
    if len(parsed) == 1:
        return parsed[0]
    if len(parsed) > 1:
        raise ValueError("Multiple grounding results; explicit selection is required")
    # Official/tool outputs may wrap coordinates in XML-style point/box tags.
    tagged = re.search(r"<(point|box|bbox)>\s*(.*?)\s*</\1>", cleaned, re.DOTALL)
    if tagged:
        numbers = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", tagged[2])]
        return _extract(numbers)
    raise ValueError("No unambiguous grounding coordinates found")


def normalize_grounding(text, coordinate_space, image_size, frame=None):
    coordinates, kind = parse_grounding(text)
    result = {
        "raw_response": text,
        "coordinate_space": coordinate_space,
        "kind": kind,
        "input_image_size": list(image_size),
        "point_input_norm01": None,
        "point_original_norm01": None,
        "frame": frame.to_dict() if isinstance(frame, ImageFrame) else frame,
    }
    if coordinates is None:
        return result
    if kind == "bbox_xyxy":
        # Validate both corners before taking the box center.
        point_to_input_norm01(coordinates[:2], coordinate_space, image_size)
        point_to_input_norm01(coordinates[2:], coordinate_space, image_size)
        point = [(coordinates[0] + coordinates[2]) / 2, (coordinates[1] + coordinates[3]) / 2]
        result["bbox_raw_xyxy"] = coordinates
    else:
        point = coordinates
    result["point_input_norm01"] = point_to_input_norm01(point, coordinate_space, image_size)
    if frame is not None:
        checked = ImageFrame(**frame) if isinstance(frame, dict) else frame
        if tuple(checked.image_size) != tuple(image_size):
            raise ValueError("Frame dimensions do not match the image actually sent to the model")
        result["point_original_norm01"] = to_original_point(point, coordinate_space, checked)
    return result


def grounding_prompt(instruction, space, image_size):
    width, height = image_size
    description = {
        "norm999": "relative coordinates on a 0–999 grid; x=999 is the right edge and y=999 the bottom",
        "norm1000": "relative coordinates on a 0–1000 grid; x=1000 is the right edge and y=1000 the bottom",
        "norm01": "relative coordinates in [0,1]; x=1 is the right edge and y=1 the bottom",
        "pixels": f"absolute pixels in the supplied {width}×{height} image, with x increasing right and y down",
    }[space]
    return (
        f"Locate the UI target: {instruction}\nReturn its center using {description}. "
        'Return only JSON {"point": [x,y]}; use {"point": null} if the target is absent. '
        "Refer to the supplied image, not an imagined screen or a differently resized image."
    )
