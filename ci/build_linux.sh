#!/bin/bash
set -euo pipefail

echo "⚙️ Installing Rust toolchain via rustup..."
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Change to parent directory of script
cd "$(dirname "$(realpath "$0")")/.."

# ----------------------------------------
# Build Python wheels using uv for multiple Python versions
# Assumes running inside a manylinux Docker container
# e.g. quay.io/pypa/manylinux_2_28_x86_64.
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
# Repair wheels with auditwheel to ensure manylinux compatibility
# ----------------------------------------

echo "🛠️ Repairing built wheels with auditwheel..."
for whl in dist/*.whl; do
  echo "▶ Repairing $whl"
  auditwheel repair "$whl" -w wheelhouse/
done

# ----------------------------------------
# Copy source distribution (.tar.gz) to wheelhouse/
# ----------------------------------------

echo "📦 Copying source distributions..."
cp dist/*.tar.gz wheelhouse/

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