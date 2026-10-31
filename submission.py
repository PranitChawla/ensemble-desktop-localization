"""Starting policy. The coding agent may replace this entire inference algorithm."""

from PIL import Image

from grounding.coordinates import ImageFrame


def ground(screenshot, instruction, pool):
    with Image.open(screenshot) as image:
        frame = ImageFrame.full(image.size)
    result = pool.ground(
        "venus", instruction, image=screenshot, frame=frame, processor_kwargs={"max_pixels": 4194304}
    )
    return {"point": result["point_original_norm01"]}
