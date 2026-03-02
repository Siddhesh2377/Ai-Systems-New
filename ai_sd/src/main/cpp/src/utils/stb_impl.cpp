/**
 * STB implementation file — single compilation unit for all STB libraries.
 *
 * STB headers are header-only with IMPLEMENTATION macros that emit definitions.
 * These MUST only be defined in ONE .cpp file to avoid ODR violations.
 * Previously they were in sd_utils.h which caused duplicate symbols when
 * sd_utils.h was included from multiple TUs.
 *
 * Phase 1.12 fix (pulled forward for model_loader extraction).
 */

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_resize2.h"
#include "stb_image_write.h"
