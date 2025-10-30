#!/usr/bin/env bash
set -e

PROJECT_NAME="dj-hand-controller"

# 1. Make folder structure
mkdir -p "$PROJECT_NAME"/{src,tests}
cd "$PROJECT_NAME"

# 2. Git & venv
git init
python -m venv venv

# 3. .gitignore
cat > .gitignore <<EOF
venv/
__pycache__/
*.py[cod]
EOF

# 4. requirements.txt (empty for now)
touch requirements.txt

# 5. README.md
cat > README.md <<EOF
# $PROJECT_NAME

Gesture-to-MIDI/OSC DJ controller using MediaPipe.

## Setup

\`\`\`bash
source venv/bin/activate
pip install -r requirements.txt
\`\`\`

## Usage

\`\`\`bash
python src/main.py
\`\`\`
EOF

# 6. src/main.py stub
cat > src/main.py <<EOF
import cv2, mediapipe as mp

def main():
    print("Hello DJ hand controller")

if __name__ == "__main__":
    main()
EOF

# 7. tests/test_main.py stub
cat > tests/test_main.py <<EOF
def test_dummy():
    assert True
EOF

# 8. Activate venv and install core deps
source venv/bin/activate
pip install --upgrade pip
pip install --default-timeout=300 mediapipe opencv-python numpy mido python-rtmidi python-osc
pip freeze > requirements.txt

echo "✔️  Bootstrapped $PROJECT_NAME"
