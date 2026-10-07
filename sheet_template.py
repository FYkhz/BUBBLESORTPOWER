from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from omr import (
    BUBBLE_RADIUS,
    MARKER_MARGIN,
    MARKER_SIZE,
    PAGE_HEIGHT,
    PAGE_WIDTH,
    QUESTION_COUNT,
    QUESTION_GAP_Y,
    QUESTION_START_Y,
    bubble_x_for_choices,
    choices_for_count,
)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def create_sheet(
    answer_marks: Sequence[str | None] | None = None,
    *,
    title: str = "OMR ANSWER SHEET",
    choice_count: int = 4,
) -> np.ndarray:
    choices = choices_for_count(choice_count)
    bubble_x = bubble_x_for_choices(choice_count)
    if answer_marks is not None and len(answer_marks) != QUESTION_COUNT:
        raise ValueError(f"Expected {QUESTION_COUNT} answers.")

    image = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "white")
    draw = ImageDraw.Draw(image)

    marker_positions = [
        (MARKER_MARGIN, MARKER_MARGIN),
        (PAGE_WIDTH - MARKER_MARGIN - MARKER_SIZE, MARKER_MARGIN),
        (PAGE_WIDTH - MARKER_MARGIN - MARKER_SIZE, PAGE_HEIGHT - MARKER_MARGIN - MARKER_SIZE),
        (MARKER_MARGIN, PAGE_HEIGHT - MARKER_MARGIN - MARKER_SIZE),
    ]
    for x, y in marker_positions:
        draw.rectangle((x, y, x + MARKER_SIZE, y + MARKER_SIZE), fill="black")

    draw.text((PAGE_WIDTH // 2, 115), title, font=_font(54, bold=True), fill="black", anchor="ma")
    draw.text(
        (PAGE_WIDTH // 2, 180),
        "Fill one bubble completely. Keep all four corner squares visible when photographing.",
        font=_font(24),
        fill="black",
        anchor="ma",
    )
    draw.line((250, 250, 1450, 250), fill="black", width=3)
    draw.text((300, 295), "Name: __________________________________________", font=_font(28), fill="black")

    header_y = 365
    draw.text((505, header_y), "Question", font=_font(29, bold=True), fill="black", anchor="mm")
    for x, choice in zip(bubble_x, choices):
        draw.text((int(x), header_y), choice, font=_font(31, bold=True), fill="black", anchor="mm")

    for q_index in range(QUESTION_COUNT):
        q = q_index + 1
        y = QUESTION_START_Y + q_index * QUESTION_GAP_Y
        # Keep the sheet pure white: no alternating gray row fills.
        draw.text((505, y), str(q), font=_font(30, bold=True), fill="black", anchor="mm")
        for choice_index, x in enumerate(bubble_x):
            box = (int(x - BUBBLE_RADIUS), y - BUBBLE_RADIUS, int(x + BUBBLE_RADIUS), y + BUBBLE_RADIUS)
            draw.ellipse(box, outline="black", width=4)
            if answer_marks is not None and answer_marks[q_index] == choices[choice_index]:
                inner = BUBBLE_RADIUS - 7
                fill_box = (int(x - inner), y - inner, int(x + inner), y + inner)
                draw.ellipse(fill_box, fill="black")

    footer_y = QUESTION_START_Y + QUESTION_COUNT * QUESTION_GAP_Y + 25
    draw.line((250, footer_y, 1450, footer_y), fill="black", width=3)
    draw.text(
        (PAGE_WIDTH // 2, footer_y + 45),
        f"20 questions · choices {choices[0]}-{choices[-1]} · smart-grid compatible",
        font=_font(23),
        fill="black",
        anchor="ma",
    )

    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def save_sheet(
    path: str | Path,
    answer_marks: Sequence[str | None] | None = None,
    *,
    title: str = "OMR ANSWER SHEET",
    choice_count: int = 4,
) -> None:
    image = create_sheet(answer_marks, title=title, choice_count=choice_count)
    cv2.imwrite(str(path), image)


if __name__ == "__main__":
    output = Path(__file__).resolve().parent / "assets" / "omr_answer_sheet.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    save_sheet(output)
    print(output)
