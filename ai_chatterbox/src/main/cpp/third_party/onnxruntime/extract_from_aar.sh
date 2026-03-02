#!/bin/bash
# Extract libonnxruntime.so from the Gradle-cached AAR for CMake linking.
# This is needed because the ORT Android AAR does not include prefab metadata.
# The .so at runtime comes from the AAR jni/ directory via Gradle;
# this extracted copy is only used at CMake link time.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LIB_DIR="$SCRIPT_DIR/lib"
ORT_VERSION="1.24.2"

# Find the AAR in Gradle cache
AAR_PATH=$(find "$HOME/.gradle/caches" -path "*/onnxruntime-android/${ORT_VERSION}/*" -name "*.aar" 2>/dev/null | head -1)

if [ -z "$AAR_PATH" ]; then
    echo "ERROR: onnxruntime-android-${ORT_VERSION}.aar not found in Gradle cache."
    echo "Run './gradlew :ai_chatterbox:dependencies' first to download it."
    exit 1
fi

echo "Found AAR: $AAR_PATH"

for ABI in arm64-v8a; do
    mkdir -p "$LIB_DIR/$ABI"
    unzip -o "$AAR_PATH" "jni/$ABI/libonnxruntime.so" -d /tmp/ort_extract_$$ >/dev/null
    cp "/tmp/ort_extract_$$/jni/$ABI/libonnxruntime.so" "$LIB_DIR/$ABI/"
    rm -rf /tmp/ort_extract_$$
    echo "Extracted: $LIB_DIR/$ABI/libonnxruntime.so ($(du -h "$LIB_DIR/$ABI/libonnxruntime.so" | cut -f1))"
done

echo "Done. ORT .so files ready for CMake linking."
