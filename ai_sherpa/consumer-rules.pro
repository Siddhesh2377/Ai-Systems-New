# JNI classes — apps consuming ai_sherpa must not rename these
-keepclasseswithmembernames,includedescriptorclasses class com.dark.ai_sherpa.OnlineRecognizer {
    native <methods>;
}
-keepclasseswithmembernames,includedescriptorclasses class com.dark.ai_sherpa.OnlineStream {
    native <methods>;
}
-keepclasseswithmembernames,includedescriptorclasses class com.dark.ai_sherpa.OfflineRecognizer {
    native <methods>;
}
-keepclasseswithmembernames,includedescriptorclasses class com.dark.ai_sherpa.OfflineStream {
    native <methods>;
}
-keepclasseswithmembernames,includedescriptorclasses class com.dark.ai_sherpa.OfflineTts {
    native <methods>;
}
-keepclasseswithmembernames,includedescriptorclasses class com.dark.ai_sherpa.VoiceActivityDetector {
    native <methods>;
}
-keepclasseswithmembernames,includedescriptorclasses class com.dark.ai_sherpa.WaveReader {
    native <methods>;
}
-keepclasseswithmembernames,includedescriptorclasses class com.dark.ai_sherpa.WaveWriter {
    native <methods>;
}

# Public API surface — keep for consumers
-keep class com.dark.ai_sherpa.OnlineRecognizer { public *; }
-keep class com.dark.ai_sherpa.OnlineStream { public *; }
-keep class com.dark.ai_sherpa.OfflineRecognizer { public *; }
-keep class com.dark.ai_sherpa.OfflineStream { public *; }
-keep class com.dark.ai_sherpa.OfflineTts { public *; }
-keep class com.dark.ai_sherpa.VoiceActivityDetector { public *; }
-keep class com.dark.ai_sherpa.WaveReader { public *; }
-keep class com.dark.ai_sherpa.WaveWriter { public *; }

# Result types returned to callers
-keep class com.dark.ai_sherpa.OnlineRecognizerResult { *; }
-keep class com.dark.ai_sherpa.OfflineRecognizerResult { *; }
-keep class com.dark.ai_sherpa.GeneratedAudio { *; }
-keep class com.dark.ai_sherpa.SpeechSegment { *; }
-keep class com.dark.ai_sherpa.WaveData { *; }

# Config types passed by callers — fields read by JNI via GetFieldID
-keep class com.dark.ai_sherpa.FeatureConfig { *; }
-keep class com.dark.ai_sherpa.EndpointRule { *; }
-keep class com.dark.ai_sherpa.EndpointConfig { *; }
-keep class com.dark.ai_sherpa.OnlineTransducerModelConfig { *; }
-keep class com.dark.ai_sherpa.OnlineParaformerModelConfig { *; }
-keep class com.dark.ai_sherpa.OnlineZipformer2CtcModelConfig { *; }
-keep class com.dark.ai_sherpa.OnlineNeMoCtcModelConfig { *; }
-keep class com.dark.ai_sherpa.OnlineModelConfig { *; }
-keep class com.dark.ai_sherpa.OnlineLMConfig { *; }
-keep class com.dark.ai_sherpa.OnlineCtcFstDecoderConfig { *; }
-keep class com.dark.ai_sherpa.HomophoneReplacerConfig { *; }
-keep class com.dark.ai_sherpa.OnlineRecognizerConfig { *; }
-keep class com.dark.ai_sherpa.OfflineTransducerModelConfig { *; }
-keep class com.dark.ai_sherpa.OfflineParaformerModelConfig { *; }
-keep class com.dark.ai_sherpa.OfflineNemoEncDecCtcModelConfig { *; }
-keep class com.dark.ai_sherpa.OfflineWhisperModelConfig { *; }
-keep class com.dark.ai_sherpa.OfflineTdnnModelConfig { *; }
-keep class com.dark.ai_sherpa.OfflineModelConfig { *; }
-keep class com.dark.ai_sherpa.OfflineLMConfig { *; }
-keep class com.dark.ai_sherpa.OfflineRecognizerConfig { *; }
-keep class com.dark.ai_sherpa.OfflineTtsVitsModelConfig { *; }
-keep class com.dark.ai_sherpa.OfflineTtsKokoroModelConfig { *; }
-keep class com.dark.ai_sherpa.OfflineTtsModelConfig { *; }
-keep class com.dark.ai_sherpa.OfflineTtsConfig { *; }
-keep class com.dark.ai_sherpa.SileroVadModelConfig { *; }
-keep class com.dark.ai_sherpa.TenVadModelConfig { *; }
-keep class com.dark.ai_sherpa.VadModelConfig { *; }

# SherpaLib — JNI bridge + error tracker (Kotlin object: keep INSTANCE)
-keepclasseswithmembernames,includedescriptorclasses class com.dark.ai_sherpa.SherpaLib {
    native <methods>;
}
-keep class com.dark.ai_sherpa.SherpaLib { public *; }
-keepclassmembers class com.dark.ai_sherpa.SherpaLib {
    public static ** INSTANCE;
}
