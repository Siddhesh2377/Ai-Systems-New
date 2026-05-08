# Consumer rules for :ai_sherpa
#
# JNI-bridged classes: native method names must match C++ extern "C" symbols
# exactly. `includedescriptorclasses` also keeps parameter/return types from
# being renamed.
-keepclasseswithmembernames,includedescriptorclasses class com.dark.ai_sherpa.OfflineRecognizer {
    native <methods>;
}
-keepclasseswithmembernames,includedescriptorclasses class com.dark.ai_sherpa.OfflineStream {
    native <methods>;
}
-keepclasseswithmembernames,includedescriptorclasses class com.dark.ai_sherpa.OfflineTts {
    native <methods>;
}
-keepclasseswithmembernames,includedescriptorclasses class com.dark.ai_sherpa.SherpaLib {
    native <methods>;
}

# Public API: callers reference these by name.
-keep class com.dark.ai_sherpa.OfflineRecognizer { public *; }
-keep class com.dark.ai_sherpa.OfflineStream     { public *; }
-keep class com.dark.ai_sherpa.OfflineTts        { public *; }
-keep class com.dark.ai_sherpa.SherpaLib         { public *; }
-keepclassmembers class com.dark.ai_sherpa.SherpaLib {
    public static ** INSTANCE;
}

# Result classes: constructed by JNI via NewObject + cached MethodID. The
# constructor signature is hard-coded in C++ — never rename or reorder.
-keep class com.dark.ai_sherpa.OfflineRecognizerResult { *; }
-keep class com.dark.ai_sherpa.GeneratedAudio          { *; }

# Config data classes: fields read by JNI via GetFieldID; field names must
# match the strings in offline_recognizer.cpp / offline_tts.cpp exactly.
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
