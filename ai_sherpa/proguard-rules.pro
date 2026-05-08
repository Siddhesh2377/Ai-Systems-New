# Source info for crash reports
-keepattributes SourceFile,LineNumberTable
-renamesourcefileattribute SourceFile

# Kotlin metadata, generics, checked exceptions
-keepattributes Exceptions,Signature,InnerClasses,EnclosingMethod

# JNI / public API: same set as consumer-rules. Library-internal R8 needs these
# too because some test paths run minification on the module itself.
-keep class com.dark.ai_sherpa.SherpaLib { *; }

-keepclasseswithmembernames,includedescriptorclasses class com.dark.ai_sherpa.OfflineRecognizer {
    native <methods>;
}
-keepclasseswithmembernames,includedescriptorclasses class com.dark.ai_sherpa.OfflineStream {
    native <methods>;
}
-keepclasseswithmembernames,includedescriptorclasses class com.dark.ai_sherpa.OfflineTts {
    native <methods>;
}

-keep class com.dark.ai_sherpa.OfflineRecognizerResult { *; }
-keep class com.dark.ai_sherpa.GeneratedAudio          { *; }

-keep class com.dark.ai_sherpa.FeatureConfig                       { *; }
-keep class com.dark.ai_sherpa.HomophoneReplacerConfig             { *; }
-keep class com.dark.ai_sherpa.OfflineTransducerModelConfig        { *; }
-keep class com.dark.ai_sherpa.OfflineParaformerModelConfig        { *; }
-keep class com.dark.ai_sherpa.OfflineNemoEncDecCtcModelConfig     { *; }
-keep class com.dark.ai_sherpa.OfflineWhisperModelConfig           { *; }
-keep class com.dark.ai_sherpa.OfflineTdnnModelConfig              { *; }
-keep class com.dark.ai_sherpa.OfflineNemoTransducerModelConfig    { *; }
-keep class com.dark.ai_sherpa.OfflineModelConfig                  { *; }
-keep class com.dark.ai_sherpa.OfflineLMConfig                     { *; }
-keep class com.dark.ai_sherpa.OfflineRecognizerConfig             { *; }
-keep class com.dark.ai_sherpa.OfflineTtsVitsModelConfig           { *; }
-keep class com.dark.ai_sherpa.OfflineTtsKokoroModelConfig         { *; }
-keep class com.dark.ai_sherpa.OfflineTtsModelConfig               { *; }
-keep class com.dark.ai_sherpa.OfflineTtsConfig                    { *; }
