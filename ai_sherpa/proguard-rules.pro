# Source info for crash reports
-keepattributes SourceFile,LineNumberTable
-renamesourcefileattribute SourceFile

# Kotlin metadata, generics, checked exceptions
-keepattributes Exceptions,Signature,InnerClasses,EnclosingMethod

# Native library loader
-keep class com.dark.ai_sherpa.SherpaLib { *; }

# JNI classes — method names must match C++ extern "C" declarations exactly.
# includedescriptorclasses also keeps parameter/return types from being renamed.
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

# Result data classes constructed by JNI via NewObject/GetMethodID
-keep class com.dark.ai_sherpa.OnlineRecognizerResult { *; }
-keep class com.dark.ai_sherpa.OfflineRecognizerResult { *; }
-keep class com.dark.ai_sherpa.GeneratedAudio { *; }
-keep class com.dark.ai_sherpa.SpeechSegment { *; }
-keep class com.dark.ai_sherpa.WaveData { *; }

# Config data classes — fields read by JNI via GetFieldID; names must not be obfuscated
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
