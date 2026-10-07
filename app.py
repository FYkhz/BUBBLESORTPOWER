from __future__ import annotations

import hashlib
from datetime import datetime

import cv2
import pandas as pd
import streamlit as st

from omr import (
    CHOICES,
    choices_for_count,
    QUESTION_COUNT,
    OMRProcessingError,
    answers_from_detections,
    crop_question,
    decode_image_bytes,
    detection_rows,
    draw_overlay,
    encode_png,
    grade_answers,
    scan_sheet,
)
from sheet_template import create_sheet


st.set_page_config(page_title="OMR Grader", page_icon="✅", layout="wide")

st.markdown(
    """
<style>
.block-container {max-width: 1180px; padding-top: 1.7rem; padding-bottom: 3rem;}
.omr-hero {padding: 1.2rem 1.4rem; border: 1px solid #dbe5f4; border-radius: 18px; background: linear-gradient(135deg,#f7fbff,#eef5ff); margin-bottom: 1rem;}
.omr-hero h1 {margin: 0; font-size: 2.25rem; color: #0b3975;}
.omr-hero p {margin: .35rem 0 0; color: #40536d;}
.status-card {padding: .8rem 1rem; border-radius: 12px; border: 1px solid #dfe6ef; background: white;}
.small-muted {font-size: .88rem; color: #66758a;}
[data-testid="stMetric"] {border: 1px solid #dfe6ef; padding: .8rem; border-radius: 12px; background: white;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="omr-hero">
  <h1>OMR Grader</h1>
  <p>Scan an answer key first, validate it, then upload student sheets for automatic grading and uncertainty review.</p>
</div>
""",
    unsafe_allow_html=True,
)


def initialize_state() -> None:
    defaults = {
        "key_scan": None,
        "key_hash": None,
        "key_overrides": {},
        "key_answers": None,
        "key_confirmed_at": None,
        "student_scan": None,
        "student_hash": None,
        "student_overrides": {},
        "history": [],
        "choice_count": 4,
        "previous_choice_count": 4,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_state()


def source_picker(prefix: str, label: str):
    upload_tab, camera_tab = st.tabs(["Upload image", "Scan with camera"])
    with upload_tab:
        upload = st.file_uploader(label, type=["jpg", "jpeg", "png"], key=f"{prefix}_upload")
        if upload is not None:
            return upload, False
    with camera_tab:
        st.caption("Camera photos are automatically cropped, perspective-corrected, and cleaned like a document scanner before OMR grading.")
        camera = st.camera_input("Capture the full sheet with all four corners visible", key=f"{prefix}_camera")
        if camera is not None:
            return camera, True
    return None, False


def file_bytes(uploaded) -> bytes:
    uploaded.seek(0)
    return uploaded.read()


def image_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def problem_detections(scan):
    return [d for d in scan.detections if d.status != "clear"]


def clear_review_widget_state(state_key: str) -> None:
    for question in range(1, QUESTION_COUNT + 1):
        st.session_state.pop(f"{state_key}_{question}", None)
        st.session_state.pop(f"{state_key}_clear_{question}", None)


def edit_clear_answers(scan, *, state_key: str) -> dict[int, str | None]:
    overrides = dict(st.session_state[state_key])
    clear_items = [d for d in scan.detections if d.status == "clear"]
    if not clear_items:
        return overrides

    with st.expander("Optional: correct an answer that was detected clearly"):
        st.caption("Use this only when the highlighted image shows that the automatic choice is wrong.")
        columns = st.columns(4)
        for index, detection in enumerate(clear_items):
            q = detection.question
            current = overrides.get(q, detection.selected)
            options = list(choices_for_count(st.session_state.choice_count))
            selected = columns[index % 4].selectbox(
                f"Question {q}",
                options,
                index=options.index(current) if current in options else 0,
                key=f"{state_key}_clear_{q}",
            )
            if selected == detection.selected:
                overrides.pop(q, None)
            else:
                overrides[q] = selected

    st.session_state[state_key] = overrides
    return overrides


def review_controls(scan, *, state_key: str, include_blank: bool) -> dict[int, str | None]:
    overrides = dict(st.session_state[state_key])
    problems = problem_detections(scan)
    if not problems:
        st.success("All 20 questions were detected clearly.")
        return overrides

    st.warning(f"{len(problems)} question(s) need review before continuing.")
    active_choices = choices_for_count(st.session_state.choice_count)
    valid_choices = list(active_choices) + (["Blank"] if include_blank else [])
    choices = ["Select..."] + valid_choices

    for detection in problems:
        q = detection.question
        with st.expander(f"Question {q}: {detection.status.title()} — confidence separation {detection.confidence:.0f}%", expanded=True):
            crop = crop_question(scan.aligned_bgr, q)
            st.image(bgr_to_rgb(crop), use_container_width=True)
            density_text = " · ".join(
                f"{choice}: {density * 100:.1f}%" for choice, density in zip(active_choices, detection.densities)
            )
            st.caption(f"Inner-bubble ink density — {density_text}")
            default_value = overrides.get(q)
            default_index = choices.index(default_value) if default_value in valid_choices else 0
            selected = st.radio(
                f"Choose the intended answer for Question {q}",
                choices,
                index=default_index,
                horizontal=True,
                key=f"{state_key}_{q}",
            )
            if selected == "Select...":
                overrides.pop(q, None)
            else:
                overrides[q] = selected

    st.session_state[state_key] = overrides
    return overrides


with st.sidebar:
    st.header("Sheet setup")
    selected_choice_count = st.selectbox(
        "Choices per question",
        options=list(range(2, 9)),
        index=list(range(2, 9)).index(int(st.session_state.choice_count)),
        help="Supports 2 through 8 bubbles per question (A-H). Use the same setting when generating and grading the sheet.",
    )
    if int(selected_choice_count) != int(st.session_state.previous_choice_count):
        st.session_state.choice_count = int(selected_choice_count)
        st.session_state.previous_choice_count = int(selected_choice_count)
        st.session_state.key_scan = None
        st.session_state.key_hash = None
        st.session_state.key_overrides = {}
        st.session_state.key_answers = None
        st.session_state.key_confirmed_at = None
        st.session_state.student_scan = None
        st.session_state.student_hash = None
        st.session_state.student_overrides = {}
        st.warning("Sheet layout changed, so the previous answer key was cleared.")
    else:
        st.session_state.choice_count = int(selected_choice_count)

    active_choices = choices_for_count(st.session_state.choice_count)
    st.caption(f"Current layout: 20 questions · choices {active_choices[0]}-{active_choices[-1]}")
    st.divider()
    st.header("Workflow")
    page = st.radio(
        "Go to",
        ["1. Home & sheet", "2. Create answer key", "3. Grade student paper", "4. Session results"],
        label_visibility="collapsed",
    )
    st.divider()
    if st.session_state.key_answers:
        st.success("Answer key ready")
        st.caption(f"20 questions · {len(active_choices)} choices · saved {st.session_state.key_confirmed_at}")
    else:
        st.info("No answer key saved yet")


if page == "1. Home & sheet":
    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        st.subheader("Start with the prototype sheet")
        st.write(
            "Print the same 20-question sheet for the teacher's answer key and for students. "
            "The four black corner squares allow OpenCV to correct rotation and camera angle."
        )
        template = create_sheet(choice_count=st.session_state.choice_count)
        st.image(bgr_to_rgb(template), caption=f"Printable 20-question OMR sheet ({len(active_choices)} choices)", use_container_width=True)
        st.download_button(
            "Download printable answer sheet (PNG)",
            data=encode_png(template),
            file_name=f"omr_answer_sheet_20_questions_{len(active_choices)}_choices.png",
            mime="image/png",
            use_container_width=True,
        )
    with right:
        st.subheader("How this MVP works")
        st.markdown(
            """
1. Print the answer sheet.
2. Fill one copy with the correct answers.
3. Scan that copy under **Create answer key**.
4. Review any uncertain key answers and confirm the key.
5. Upload each student sheet under **Grade student paper**.
6. Review only answers the app could not confidently interpret.

**Color meaning**
- Green: clear/correct
- Red: clear/incorrect
- Orange: unclear or multiple marks
- Gray: blank
            """
        )
        st.info(
            "Camera mode now behaves like a document scanner: it auto-crops, corrects perspective, reduces shadows, and then uses the four black squares for fine alignment."
        )

elif page == "2. Create answer key":
    st.subheader("Create and validate the answer key")
    st.write("Upload or photograph a correctly completed copy of the prototype sheet.")
    source, from_camera = source_picker("key", "Upload the completed answer-key sheet")

    if source is not None:
        data = file_bytes(source)
        digest = image_hash(data)
        if digest != st.session_state.key_hash:
            try:
                with st.spinner("Detecting the sheet and reading ink density..."):
                    image = decode_image_bytes(data)
                    st.session_state.key_scan = scan_sheet(image, answer_key_mode=True, scanner_mode=from_camera, choice_count=st.session_state.choice_count)
                    st.session_state.key_hash = digest
                    st.session_state.key_overrides = {}
                    st.session_state.key_answers = None
                    st.session_state.key_confirmed_at = None
                    st.session_state.student_scan = None
                    st.session_state.student_hash = None
                    st.session_state.student_overrides = {}
                    clear_review_widget_state("key_overrides")
                    clear_review_widget_state("student_overrides")
            except OMRProcessingError as exc:
                st.error(str(exc))

    scan = st.session_state.key_scan
    if scan is not None:
        if scan.warning:
            st.warning(scan.warning)
        if scan.scanner_bgr is not None:
            with st.expander("Camera scanner preview", expanded=True):
                st.caption("This is the cleaned document image produced before OMR alignment and bubble reading.")
                st.image(bgr_to_rgb(scan.scanner_bgr), caption="Auto-cropped and enhanced scan", use_container_width=True)
                if scan.scanner_page_found:
                    st.success("Paper boundary detected and perspective corrected.")
                else:
                    st.info("Paper edge was uncertain; registration squares were used as the next alignment fallback.")
        if scan.grid_refinement_used:
            st.success(
                f"Bubble-grid auto-alignment active · confidence {scan.grid_confidence:.0%} · residual tilt {scan.residual_angle_deg:+.2f}°"
            )
        else:
            st.caption(f"Bubble-grid auto-alignment fallback · confidence {scan.grid_confidence:.0%}")
        st.caption(f"Sharpness score: {scan.blur_score:.0f}")
        c1, c2 = st.columns([1.25, 0.75], gap="large")
        with c1:
            overlay = draw_overlay(scan.aligned_bgr, scan.detections, overrides=st.session_state.key_overrides, bubble_rows=scan.bubble_rows)
            st.image(bgr_to_rgb(overlay), caption="Detected answer key", use_container_width=True)
        with c2:
            rows = pd.DataFrame(detection_rows(scan.detections, choices=scan.choices))
            clear_count = int((rows["Status"] == "Clear").sum())
            issue_count = QUESTION_COUNT - clear_count
            m1, m2 = st.columns(2)
            m1.metric("Clear", clear_count)
            m2.metric("Needs review", issue_count)
            st.dataframe(rows[["Question", "Detected", "Status", "Confidence"]], hide_index=True, use_container_width=True)

        st.divider()
        overrides = review_controls(scan, state_key="key_overrides", include_blank=False)
        overrides = edit_clear_answers(scan, state_key="key_overrides")
        answers = answers_from_detections(scan.detections, overrides)
        unresolved = [i + 1 for i, answer in enumerate(answers) if answer is None]

        if unresolved:
            st.error("The answer key cannot be saved until every question has exactly one confirmed answer.")
        else:
            summary = " · ".join(f"{i + 1}:{answer}" for i, answer in enumerate(answers))
            st.code(summary, language=None)
            if st.button("Confirm and save answer key", type="primary", use_container_width=True):
                st.session_state.key_answers = [str(answer) for answer in answers]
                st.session_state.key_confirmed_at = datetime.now().strftime("%Y-%m-%d %I:%M %p")
                st.success("Answer key saved. You can now grade student papers.")
                st.rerun()

elif page == "3. Grade student paper":
    st.subheader("Grade a student answer sheet")
    if not st.session_state.key_answers:
        st.warning("Create and confirm an answer key first.")
        st.stop()

    st.caption("Answer key ready: " + " · ".join(f"{i+1}:{a}" for i, a in enumerate(st.session_state.key_answers)))
    source, from_camera = source_picker("student", "Upload the student's answer sheet")

    if source is not None:
        data = file_bytes(source)
        digest = image_hash(data)
        if digest != st.session_state.student_hash:
            try:
                with st.spinner("Reading student answers..."):
                    image = decode_image_bytes(data)
                    st.session_state.student_scan = scan_sheet(image, answer_key_mode=False, scanner_mode=from_camera, choice_count=st.session_state.choice_count)
                    st.session_state.student_hash = digest
                    st.session_state.student_overrides = {}
                    clear_review_widget_state("student_overrides")
            except OMRProcessingError as exc:
                st.error(str(exc))

    scan = st.session_state.student_scan
    if scan is not None:
        if scan.warning:
            st.warning(scan.warning)
        if scan.scanner_bgr is not None:
            with st.expander("Camera scanner preview", expanded=True):
                st.caption("This is the cleaned document image produced before OMR alignment and bubble reading.")
                st.image(bgr_to_rgb(scan.scanner_bgr), caption="Auto-cropped and enhanced scan", use_container_width=True)
                if scan.scanner_page_found:
                    st.success("Paper boundary detected and perspective corrected.")
                else:
                    st.info("Paper edge was uncertain; registration squares were used as the next alignment fallback.")
        if scan.grid_refinement_used:
            st.success(
                f"Bubble-grid auto-alignment active · confidence {scan.grid_confidence:.0%} · residual tilt {scan.residual_angle_deg:+.2f}°"
            )
        else:
            st.caption(f"Bubble-grid auto-alignment fallback · confidence {scan.grid_confidence:.0%}")
        st.caption(f"Sharpness score: {scan.blur_score:.0f}")

        initial_answers = answers_from_detections(scan.detections, st.session_state.student_overrides)
        initial_grade = grade_answers(initial_answers, st.session_state.key_answers)
        overlay = draw_overlay(
            scan.aligned_bgr,
            scan.detections,
            key_answers=st.session_state.key_answers,
            overrides=st.session_state.student_overrides,
            bubble_rows=scan.bubble_rows,
        )

        left, right = st.columns([1.2, 0.8], gap="large")
        with left:
            st.image(bgr_to_rgb(overlay), caption="Green = correct, red = incorrect, orange = review, gray = blank", use_container_width=True)
        with right:
            r1, r2 = st.columns(2)
            r1.metric("Current score", f"{initial_grade['score']}/{initial_grade['total']}")
            r2.metric("Percentage", f"{initial_grade['percentage']:.1f}%")
            r3, r4 = st.columns(2)
            r3.metric("Incorrect", initial_grade["incorrect"])
            r4.metric("Blank", initial_grade["blank"])
            st.dataframe(pd.DataFrame(initial_grade["rows"]), hide_index=True, use_container_width=True)

        st.divider()
        overrides = review_controls(scan, state_key="student_overrides", include_blank=True)
        final_answers = answers_from_detections(scan.detections, overrides)
        unresolved = [d.question for d in scan.detections if d.status != "clear" and d.question not in overrides]

        if unresolved:
            st.warning("Review all orange questions to finalize the grade.")
        else:
            final_grade = grade_answers(final_answers, st.session_state.key_answers)
            final_overlay = draw_overlay(
                scan.aligned_bgr,
                scan.detections,
                key_answers=st.session_state.key_answers,
                overrides=overrides,
                bubble_rows=scan.bubble_rows,
            )
            st.success(f"Final score: {final_grade['score']}/{final_grade['total']} ({final_grade['percentage']:.1f}%)")

            result_df = pd.DataFrame(final_grade["rows"])
            c1, c2, c3 = st.columns(3)
            with c1:
                st.download_button(
                    "Download question results (CSV)",
                    data=result_df.to_csv(index=False).encode("utf-8"),
                    file_name="omr_question_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with c2:
                st.download_button(
                    "Download marked image (PNG)",
                    data=encode_png(final_overlay),
                    file_name="omr_graded_sheet.png",
                    mime="image/png",
                    use_container_width=True,
                )
            with c3:
                if st.button("Add result to session", use_container_width=True):
                    st.session_state.history.append(
                        {
                            "Scan": len(st.session_state.history) + 1,
                            "Time": datetime.now().strftime("%I:%M:%S %p"),
                            "Score": final_grade["score"],
                            "Total": final_grade["total"],
                            "Percentage": round(final_grade["percentage"], 1),
                            "Incorrect": final_grade["incorrect"],
                            "Blank": final_grade["blank"],
                        }
                    )
                    st.success("Result added to this session.")

elif page == "4. Session results":
    st.subheader("Session results")
    if not st.session_state.history:
        st.info("No student results have been added during this session yet.")
    else:
        history = pd.DataFrame(st.session_state.history)
        st.dataframe(history, hide_index=True, use_container_width=True)
        st.download_button(
            "Download session results (CSV)",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="omr_session_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
        if st.button("Clear session results"):
            st.session_state.history = []
            st.rerun()

st.divider()
st.caption("Prototype MVP · Fixed 20-question A-D sheet · No student roster or QR codes yet")
