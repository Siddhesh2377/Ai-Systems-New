-keep class com.dark.ai_chatterbox.ChatterboxNativeLib {
    native <methods>;
    <init>();
}
-keep interface com.dark.ai_chatterbox.ChatterboxCallback { *; }
-keep class * implements com.dark.ai_chatterbox.ChatterboxCallback { *; }
-keep class com.dark.ai_chatterbox.ChatterboxManager { public *; }
-keep class com.dark.ai_chatterbox.ChatterboxConfig { *; }
-keep class com.dark.ai_chatterbox.ChatterboxState { *; }
-keep class com.dark.ai_chatterbox.ChatterboxState$* { *; }
