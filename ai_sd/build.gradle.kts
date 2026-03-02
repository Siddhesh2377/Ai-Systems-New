plugins {
    alias(libs.plugins.android.library)
}

android {
    namespace = "com.dark.ai_sd"
    compileSdk {
        version = release(36)
    }
    ndkVersion = "27.3.13750724"

    defaultConfig {
        minSdk = 27
        consumerProguardFiles("consumer-rules.pro")

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
}

dependencies {
    // tar.xz extraction for QNN libs
    // NOTE: AAR does not bundle transitive deps — consuming apps must also declare:
    //   implementation("org.apache.commons:commons-compress:1.28.0")
    //   implementation("org.tukaani:xz:1.11")
    implementation(libs.commons.compress)
    implementation(libs.xz)
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
}
