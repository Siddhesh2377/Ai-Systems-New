plugins {
    alias(libs.plugins.android.library)
}

android {
    namespace = "com.dark.ai_sherpa"
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
                arguments += listOf(
                    "-DANDROID_STL=c++_shared",
                    "-DCMAKE_BUILD_TYPE=Release",
                )
                // 32-bit is intentionally excluded for the foreseeable future:
                // sherpa-onnx + ORT mobile ship arm64 binaries we trust, and
                // every supported handset on minSdk=29 has a 64-bit kernel.
                abiFilters += listOf("arm64-v8a", "armeabi-v7a")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
    }

    externalNativeBuild {
        cmake {
            path("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildFeatures {
        prefab = true
    }
}

dependencies {
    api(project(":tn_security"))
}
