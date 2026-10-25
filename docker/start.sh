#!/usr/bin/env bash
# Starts the API, waits until it answers, then the UI; if either process exits, the container exits
# so the Space restarts it instead of serving a half-working app.
set -u

uvicorn app.api.main:app --host 127.0.0.1 --port 8000 &

python - <<'PY'
import time, urllib.request
for _ in range(180):  # model loading takes ~30 s on the Space's CPU
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5)
        break
    except OSError:
        time.sleep(2)
PY

streamlit run ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true \
  --server.enableXsrfProtection false --server.maxUploadSize 10 --browser.gatherUsageStats false &

wait -n
exit 1
