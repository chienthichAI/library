# Face Evaluation Workflow (Realtime Camera)

## 1) Run system normally
- Start app as usual (`run_app.bat`).
- Every authentication attempt is auto-written to:
  - `backend/logs/face_eval/face_eval_YYYY-MM-DD.csv`

## 2) Ground truth options
- Recommended (test mode): send `ground_truth_student_id` directly while calling API/WS.
- Manual fallback: open CSV at end of day and fill `ground_truth_student_id`.

### A) API test mode (`POST /auth/verify-face`)
- Send multipart form field:
  - `ground_truth_student_id=SV001` (or `UNKNOWN`)
  - optional `source=api_test`

### B) WebSocket test mode (`/auth/ws/stream`)
- Connect with query param:
  - `/auth/ws/stream?ground_truth_student_id=SV001`
- Server will auto-mark `source=ws_test` in CSV.

### C) Manual fallback
- Open that CSV at the end of day.
- Fill column `ground_truth_student_id` for each test row:
  - Use real student id (example: `SV001`) if known person is in front of camera.
  - Use `UNKNOWN` if person should be rejected (not registered).

## 3) Run evaluation script
From `backend/`:

```bash
python scripts/eval/eval_face_from_csv.py
```

Optional:

```bash
python scripts/eval/eval_face_from_csv.py --csv logs/face_eval/face_eval_2026-03-25.csv
python scripts/eval/eval_face_from_csv.py --only-events AUTH_SUCCESS AUTH_MISMATCH AUTH_FAIL SPOOF_DETECTED
```

## Output
- Accuracy (%)
- Macro Precision / Recall / F1 (%)
- Weighted F1 (%)
- Confusion Matrix (rows=true label, cols=pred label)
