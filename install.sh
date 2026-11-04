#!/usr/bin/env bash
# Install dependencies in an order that works on Linux with recent Python
# (e.g. 3.12/3.14), where dlib has no prebuilt wheels and must not be built.
set -euo pipefail
cd "$(dirname "$0")"

if [ -z "${VIRTUAL_ENV:-}" ]; then
    [ -d .venv ] || python3 -m venv .venv
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

python -m pip install --upgrade pip

# requirements.txt already excludes face-recognition. dlib-bin provides the
# prebuilt `dlib` module and setuptools<81 provides pkg_resources (needed by
# face_recognition_models).
pip install -r requirements.txt

# face-recognition is pure Python; install it without pulling dlib.
pip install --no-deps face-recognition face-recognition-models

echo
echo "Done. Verify with:"
echo "  python -c \"import face_recognition, dlib; print('ok')\""
echo "Run locally with:"
echo "  python app.py"
