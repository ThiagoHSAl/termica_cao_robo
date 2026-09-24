plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.verlab.thermaldetector"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.verlab.thermaldetector"
        minSdk = 29          // MediaProjection foreground service type
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }

    // Não comprimir o modelo .onnx no APK (ORT precisa mapear direto).
    androidResources {
        noCompress += "onnx"
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    // ONNX Runtime para Android (pacote completo, suporta todos os operadores).
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")
}
