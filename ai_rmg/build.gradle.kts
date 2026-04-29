plugins {
    alias(libs.plugins.android.library)
}

android {
    namespace = "com.dark.ai_rmg"
    compileSdk {
        version = release(36) {
            minorApiLevel = 1
        }
    }
    ndkVersion = "27.3.13750724"

    defaultConfig {
        minSdk = 29

        consumerProguardFiles("consumer-rules.pro")

        externalNativeBuild {
            cmake {
                cppFlags("-std=c++20")
                arguments += listOf(
                    "-DANDROID_STL=c++_static",
                    "-Wno-deprecated",
                    "-Wno-dev",
                )
                // arm64-v8a only: rm-graph kernels need armv8.2-a (FP16/DOTPROD).
                // x86_64 fails because NDK clang rejects __builtin_cpu_supports("f16c")
                // in upstream cpu_features.cc.
                abiFilters += listOf("arm64-v8a")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
                "consumer-rules.pro"
            )
        }
        debug {
            isMinifyEnabled = false
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

    buildFeatures {
        buildConfig = false
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.kotlinx.coroutines.android)
}
