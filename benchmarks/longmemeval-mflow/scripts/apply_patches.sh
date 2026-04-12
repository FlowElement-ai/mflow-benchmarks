#!/usr/bin/env bash
# Apply precise_mode patches to M-flow source code.
# Run from the benchmark root directory.
set -euo pipefail

MFLOW_ROOT="${1:?Usage: $0 <path-to-mflow-root>}"
PATCH_DIR="$(cd "$(dirname "$0")/../patches" && pwd)"

echo "Applying precise_mode patches to: $MFLOW_ROOT"

# Verify M-flow version
if [ -f "$MFLOW_ROOT/pyproject.toml" ]; then
    VER=$(grep '^version' "$MFLOW_ROOT/pyproject.toml" | head -1)
    echo "  M-flow version: $VER"
fi

# New files
echo "  Creating: knowledge/summarization/precise_summarize.py"
cp "$PATCH_DIR/NEW__precise_summarize.py" "$MFLOW_ROOT/m_flow/knowledge/summarization/precise_summarize.py"

echo "  Creating: llm/prompts/precise_compress.txt"
cp "$PATCH_DIR/NEW__precise_compress.txt" "$MFLOW_ROOT/m_flow/llm/prompts/precise_compress.txt"

# Modified files
declare -A FILE_MAP=(
    ["MOD__pipeline_contexts.py"]="m_flow/memory/episodic/episode_builder/pipeline_contexts.py"
    ["MOD__write_episodic_memories.py"]="m_flow/memory/episodic/write_episodic_memories.py"
    ["MOD__phase0a.py"]="m_flow/memory/episodic/episode_builder/phase0a.py"
    ["MOD__summarize_by_event.py"]="m_flow/knowledge/summarization/summarize_by_event.py"
    ["MOD__memorize.py"]="m_flow/api/v1/memorize/memorize.py"
    ["MOD__get_memorize_router.py"]="m_flow/api/v1/memorize/routers/get_memorize_router.py"
    ["MOD__adapter.py"]="m_flow/adapters/graph/kuzu/adapter.py"
)

for patch_file in "${!FILE_MAP[@]}"; do
    target="${FILE_MAP[$patch_file]}"
    echo "  Replacing: $target"
    cp "$PATCH_DIR/$patch_file" "$MFLOW_ROOT/$target"
done

echo ""
echo "Done. Patches applied to $MFLOW_ROOT"
echo "Use precise_mode=True in memorize() to activate."
