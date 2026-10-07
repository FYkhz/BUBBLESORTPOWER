from __future__ import annotations

import cv2
import numpy as np

from omr import QUESTION_COUNT, answers_from_detections, grade_answers, scan_sheet
from sheet_template import create_sheet


def perspective_photo(image: np.ndarray, rotate_deg: float = 0.0) -> np.ndarray:
    h, w = image.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32([[280, 180], [1940, 300], [1780, 2400], [150, 2260]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    photo = cv2.warpPerspective(image, matrix, (2250, 2650), borderValue=(175, 175, 175))
    if rotate_deg:
        center = (photo.shape[1] / 2, photo.shape[0] / 2)
        r = cv2.getRotationMatrix2D(center, rotate_deg, 1.0)
        photo = cv2.warpAffine(photo, r, (photo.shape[1], photo.shape[0]), borderValue=(175, 175, 175))
    return photo


def run_case(choice_count: int) -> None:
    labels = [chr(ord('A') + i) for i in range(choice_count)]
    key = [labels[i % choice_count] for i in range(QUESTION_COUNT)]
    key_sheet = perspective_photo(create_sheet(key, title=f"SMART {choice_count}", choice_count=choice_count), 1.2)
    scan = scan_sheet(key_sheet, answer_key_mode=True, choice_count=choice_count)
    detected = answers_from_detections(scan.detections)
    assert detected == key, (choice_count, detected, key, scan.grid_confidence, scan.warning)

    student = key.copy()
    student[2] = labels[(labels.index(student[2]) + 1) % choice_count]
    student[8] = None
    student_sheet = perspective_photo(create_sheet(student, title="STUDENT", choice_count=choice_count), -1.0)
    student_scan = scan_sheet(student_sheet, answer_key_mode=False, choice_count=choice_count)
    answers = answers_from_detections(student_scan.detections)
    grade = grade_answers(answers, key)
    assert grade["score"] == QUESTION_COUNT - 2, grade
    assert grade["incorrect"] == 1, grade
    assert grade["blank"] == 1, grade


if __name__ == "__main__":
    run_case(4)
    run_case(6)
    print("Smart-grid OMR tests passed for 4 and 6 choices.")
