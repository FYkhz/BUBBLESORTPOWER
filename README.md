# OMR Grader MVP

A simple offline-first prototype that:

- generates a printable 20-question A-D answer sheet;
- scans a completed answer key first;
- validates blank, unclear, or multiple-marked key answers;
- scans student answer-sheet images or phone-camera photos;
- measures inner-bubble ink density;
- highlights uncertain questions for teacher review;
- grades against the confirmed answer key;
- downloads question results, a marked image, and session results.

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Create and activate a virtual environment (recommended).
4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Start the app:

```bash
streamlit run app.py
```

Streamlit will open the app in your browser, usually at `http://localhost:8501`.

## Run on Replit

1. Create a new Python Repl.
2. Upload all files from this folder.
3. Install packages from `requirements.txt`.
4. Use this run command:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8080
```

## Recommended test sequence

1. Open **Home & sheet** and download the answer-sheet PNG.
2. Print it several times.
3. Fill one copy as the answer key.
4. Open **Create answer key**, upload or photograph the key, review flags, and save it.
5. Fill another copy as a student sheet.
6. Open **Grade student paper**, upload it, review uncertain answers, and download the result.

## Important prototype limitation

This MVP uses a fixed sheet layout. It expects:

- 20 questions;
- four choices: A, B, C, D;
- the supplied registration squares and bubble positions;
- the entire sheet to be visible in the image.

Future versions can add customizable templates, student rosters, batch grading, local database storage, Android packaging, and teacher accounts.

## Camera document scanner

Camera captures now pass through `document_scanner.py` before OMR grading. The scanner attempts to:

- detect the full paper boundary,
- crop and perspective-correct the page,
- normalize uneven lighting and shadows,
- improve local contrast while preserving pencil/pen marks,
- then use the existing four registration squares for fine alignment.

Uploaded JPG/PNG files keep the original upload path, which is useful for images already processed by CamScanner or another scanning app.
