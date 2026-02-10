#!/bin/bash
set -euo pipefail

echo "⚙️ Installing Rust toolchain via rustup..."
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

echo "⚙️ Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh

# Change to parent directory of script
cd "$(dirname "$(realpath "$0")")/.."

# ----------------------------------------
# Build Python wheels using uv for multiple Python versions
# macOS universal2 wheels (Intel + Apple Silicon)
# ----------------------------------------

PY_VERSIONS=(
  cp39
  cp310
  cp311
  cp312
  cp313
)

echo "🔧 Building wheels for Python versions: ${PY_VERSIONS[*]}"
for PY in "${PY_VERSIONS[@]}"; do
  echo "▶ Building for $PY..."
  uv build --python "$PY"
done

# ----------------------------------------
# Copy all wheels and source distribution to wheelhouse
# ----------------------------------------

echo "📦 Organizing wheels and source distributions..."
mkdir -p wheelhouse/
cp dist/*.whl wheelhouse/ 2>/dev/null || true
cp dist/*.tar.gz wheelhouse/ 2>/dev/null || true

echo "✅ macOS build complete. Files are in ./wheelhouse/"

# ----------------------------------------
# Test all built wheels with pytest using uv
# ----------------------------------------
echo "🧪 Testing built wheels with pytest using uv..."
for PY in "${PY_VERSIONS[@]}"; do
  echo "▶ Testing for $PY..."
  WHEEL_FILE=$(find wheelhouse/ -name "*${PY}*" -name "*.whl" | head -n1)
  if [ -n "$WHEEL_FILE" ]; then
    echo "Testing wheel: $WHEEL_FILE"
    uv run --python "$PY" --with pytest --with "$WHEEL_FILE" pytest tests/ -v
    echo "✅ Tests passed for $PY"
  else
    echo "❌ No wheel file found for $PY"
    exit 1
  fi
done
echo "✅ Build and test complete. Files are in ./wheelhouse/"
