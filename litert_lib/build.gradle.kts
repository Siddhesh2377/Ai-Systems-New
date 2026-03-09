plugins {
    alias(libs.plugins.android.library)
}

android {
    namespace = "com.dark.litert_lib"
    compileSdk {
        version = release(36)
    }

    defaultConfig {
        minSdk = 29
        consumerProguardFiles("consumer-rules.pro")
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

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    implementation(project(":unified_inference"))
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.androidx.core.ktx)
    // LiteRT-LM SDK — API may need adjustment once resolved
    // Removed for now: actual artifact coordinates need verification
    // implementation("com.google.ai.edge.litertlm:litertlm-android:latest.release")
}
