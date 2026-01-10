import org.gradle.kotlin.dsl.withType
import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jetbrains.kotlin.gradle.tasks.KotlinJvmCompile

plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
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
                cppFlags("")
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
            version = "3.22.1"
        }
    }
    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
        jniLibs {
            useLegacyPackaging = true
            // Exclude CMake build outputs to prevent duplicates
            excludes += "**/intermediates/cxx/**/*.so"
        }
        // Add this to handle duplicate .so files
        jniLibs.pickFirsts.add("lib/arm64-v8a/libstable_diffusion_core.so")
        jniLibs.pickFirsts.add("lib/armeabi-v7a/libstable_diffusion_core.so")
        jniLibs.pickFirsts.add("lib/x86/libstable_diffusion_core.so")
        jniLibs.pickFirsts.add("lib/x86_64/libstable_diffusion_core.so")
    }

    // Add sourceSets configuration to control jniLibs directory
    sourceSets {
        getByName("main") {
            // Only include jniLibs from the standard location
            jniLibs.setSrcDirs(listOf("src/main/jniLibs"))
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    tasks.withType<KotlinJvmCompile>().configureEach {
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_17)
        }
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
}