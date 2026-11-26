#!/usr/bin/env bash
# Move the deployment from /opt/oneocr_linux to /home/ubuntu/oneocr_linux.
# /opt copy is left untouched as a rollback.
set -euo pipefail
SRC=/opt/oneocr_linux
DEST=/home/ubuntu/oneocr_linux
SRC_TAR=/home/ubuntu/oneocr_src_new.tar.gz

mkdir -p "$DEST"
for d in models samples tests tools docs deploy; do
  if [ -d "$SRC/$d" ]; then cp -a "$SRC/$d" "$DEST/"; fi
done
cp -a "$SRC/pyproject.toml" "$DEST/" 2>/dev/null || true
cp -a "$SRC/README.md" "$DEST/" 2>/dev/null || true

cd "$DEST"
tar -xzf "$SRC_TAR"
rm -f "$SRC_TAR"

if [ ! -d .venv ]; then python3 -m venv .venv; fi
. .venv/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q -e .

bash deploy/build_onefile.sh "$DEST"

mkdir -p bin
LATEST=$(ls -dt "$DEST"/dist_pkg_* | head -1)
cp -f "$LATEST/oneocr-linux" bin/oneocr-linux
chmod +x bin/oneocr-linux
sudo ln -sf "$DEST/bin/oneocr-linux" /usr/local/bin/oneocr-linux

echo "--- which ---"
which oneocr-linux
ls -lh "$DEST/bin/oneocr-linux"
echo "--- smoke ---"
cd /tmp && oneocr-linux "$DEST/samples/sample.png" | head -3