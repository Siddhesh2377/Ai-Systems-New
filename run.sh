#!/bin/bash

# Base directory
BASE_DIR="/home/home/AndroidStudioProjects/AiSystems/ai_sd/src/main/cpp/src"

# Create base directory if it doesn't exist
mkdir -p "$BASE_DIR"

# Navigate to base directory
cd "$BASE_DIR" || exit 1

echo "Creating directory structure in: $BASE_DIR"

# Create directories
echo "Creating directories..."
mkdir -p jni
mkdir -p core
mkdir -p models
mkdir -p inference
mkdir -p processing
mkdir -p schedulers
mkdir -p utils
mkdir -p common

# Create JNI files
echo "Creating JNI files..."
touch jni/StableDiffusionJNI.cpp
touch jni/StableDiffusionJNI.h
touch jni/JNITypes.cpp
touch jni/JNITypes.h

# Create Core files
echo "Creating Core files..."
touch core/ModelManager.cpp
touch core/ModelManager.h
touch core/GenerationConfig.cpp
touch core/GenerationConfig.h

# Create Models files
echo "Creating Models files..."
touch models/QnnModelLoader.cpp
touch models/QnnModelLoader.h
touch models/MnnModelLoader.cpp
touch models/MnnModelLoader.h
touch models/ModelCache.cpp
touch models/ModelCache.h

# Create Inference files
echo "Creating Inference files..."
touch inference/ImageGenerator.cpp
touch inference/ImageGenerator.h
touch inference/InferenceContext.cpp
touch inference/InferenceContext.h
touch inference/Upscaler.cpp
touch inference/Upscaler.h

# Create Processing files
echo "Creating Processing files..."
touch processing/TextProcessor.cpp
touch processing/TextProcessor.h
touch processing/PromptProcessor.cpp
touch processing/PromptProcessor.h
touch processing/VAEProcessor.cpp
touch processing/VAEProcessor.h
touch processing/ImageProcessor.cpp
touch processing/ImageProcessor.h

# Create Schedulers files
echo "Creating Schedulers files..."
touch schedulers/SchedulerFactory.cpp
touch schedulers/SchedulerFactory.h
touch schedulers/DPMSchedulerWrapper.cpp
touch schedulers/DPMSchedulerWrapper.h
touch schedulers/EulerSchedulerWrapper.cpp
touch schedulers/EulerSchedulerWrapper.h

# Create Utils files
echo "Creating Utils files..."
touch utils/TilingUtils.cpp
touch utils/TilingUtils.h
touch utils/PatchUtils.cpp
touch utils/PatchUtils.h
touch utils/BlendingUtils.cpp
touch utils/BlendingUtils.h
touch utils/SafetyChecker.cpp
touch utils/SafetyChecker.h

# Create Common files
echo "Creating Common files..."
touch common/Types.h
touch common/Constants.h
touch common/Logger.h

echo ""
echo "✓ Directory structure created successfully!"
echo ""
echo "Summary:"
echo "--------"
echo "JNI files:        $(ls jni/ | wc -l) files"
echo "Core files:       $(ls core/ | wc -l) files"
echo "Models files:     $(ls models/ | wc -l) files"
echo "Inference files:  $(ls inference/ | wc -l) files"
echo "Processing files: $(ls processing/ | wc -l) files"
echo "Schedulers files: $(ls schedulers/ | wc -l) files"
echo "Utils files:      $(ls utils/ | wc -l) files"
echo "Common files:     $(ls common/ | wc -l) files"
echo ""
echo "Total files created: $(find . -type f | wc -l)"
echo ""
echo "Structure:"
tree -L 2 2>/dev/null || find . -print | sed -e 's;[^/]*/;|____;g;s;____|; |;g'