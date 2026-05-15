plugins {
    alias(libs.plugins.android.library)
}

android {
    namespace = "com.dark.ai_sd"
    compileSdk {
        // 36.1 required to match tn_security AAR (Prefab dependency).
        version = release(36) {
            minorApiLevel = 1
        }
    }
    ndkVersion = "27.3.13750724"

    defaultConfig {
        // Bumped 27 -> 29 to match tn_security's minSdk. The QNN backend
        // realistically only ships on devices >= API 29 anyway.
        minSdk = 29
        consumerProguardFiles("consumer-rules.pro")

        ndk {
            //noinspection ChromeOsAbiSupport
            abiFilters += listOf("arm64-v8a")
        }

        externalNativeBuild {
            cmake {
                arguments += listOf(
                    "-DCMAKE_BUILD_TYPE=Release",
                    // Shared STL required for Prefab consumers of tn_security
                    // (which exports public C++ ABI). Static STL would silently
                    // duplicate libc++ in each .so and break std::* type IDs.
                    "-DANDROID_STL=c++_shared",
                    "-Wno-deprecated",
                    "-Wno-dev",
                )
                targets += "ai_sd"
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
            isMinifyEnabled = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro"
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
    buildFeatures {
        prefab = true
    }
}

dependencies {
    api(project(":tn_security"))
    // tar.xz extraction for QNN libs — api scope so consumers get these transitively
    api(libs.commons.compress)
    api(libs.xz)
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
}
