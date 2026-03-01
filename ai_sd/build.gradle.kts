plugins {
    alias(libs.plugins.android.library)
}

android {
    namespace = "com.dark.ai_sd"
    compileSdk {
        version = release(36)
    }

    defaultConfig {
        minSdk = 27

        ndk {
            //noinspection ChromeOsAbiSupport
            abiFilters += listOf("arm64-v8a")
        }

        externalNativeBuild {
            cmake {
                arguments += "-DCMAKE_BUILD_TYPE=Release"
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
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro"
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
    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }
}

dependencies {
    // Keep: tar.xz extraction for QNN libs from assets
    implementation(libs.commons.compress)
    implementation(libs.xz)
    // Removed: okhttp (no more HTTP client)
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
}
