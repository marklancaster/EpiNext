#!/usr/bin/env bash
# setup_and_test.sh
# Initializes the uv environment, runs the linter, and executes the test suite.

set -e # Exit immediately if a command exits with a non-zero status

echo "🚀 Setting up the EpiNext environment..."

# 1. Ensure uv is installed (Jules' VM might need this if not pre-installed)
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# 2. Sync the project dependencies
echo "🔄 Syncing dependencies via uv..."
uv sync

# 3. Run Ruff for linting and formatting checks
echo "🧹 Running Ruff..."
uv run ruff check .
uv run ruff format --check .

# 4. Execute the pytest suite
echo "🧪 Running Pytest suite..."
uv run pytest -v

echo "✅ Environment setup and tests completed successfully!"