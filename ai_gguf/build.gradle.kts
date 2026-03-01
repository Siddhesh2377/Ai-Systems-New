plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.serialization)
}

android {
    namespace = "com.mp.ai_gguf"
    compileSdk {
        version = release(36)
    }

    defaultConfig {
        minSdk = 27

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        consumerProguardFiles("consumer-rules.pro")

        ndk {
            abiFilters += listOf("arm64-v8a") // Android-only: ARM64 only, x86_64 emulator support removed
        }

        externalNativeBuild {
            cmake {
                // Mobile-optimized build: CPU + OpenCL only
                arguments += "-DLLAMA_MOBILE=ON"
                arguments += "-DLLAMA_BUILD_COMMON=ON"
                arguments += "-DGGML_PAGE_SIZE=16384"
                arguments += "-DGGML_OPENMP=OFF"

                // ARM CPU: single build targeting dotprod (ARMv8.2+), KleidiAI disabled (SIGILL crash)
                arguments += "-DGGML_NATIVE=OFF"
                arguments += "-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16"
                arguments += "-DGGML_CPU_KLEIDIAI=OFF"

                // GPU: OpenCL disabled for GGUF (CPU-only text gen)
                // OpenCL is only used by ai_sd (diffusion). Having it here
                // adds ~5-10s to model load due to Adreno kernel compilation.
                arguments += "-DGGML_OPENCL=OFF"

                arguments += "-DCMAKE_BUILD_TYPE=Release"
                cppFlags += listOf(
                    "-O3",
                    "-ffast-math",
                    "-fno-finite-math-only",
                    "-ffp-contract=fast"
                )
                cFlags += listOf(
                    "-O3",
                    "-ffast-math",
                    "-fno-finite-math-only",
                    "-ffp-contract=fast"
                )
            }
        }

    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    externalNativeBuild {
        cmake {
            path("src/main/cpp/CMakeLists.txt")
            version = "3.31.4"
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }
}

dependencies {
    implementation(libs.kotlinx.serialization.json)
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
}