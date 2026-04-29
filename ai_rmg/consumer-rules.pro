# JNI bridge: native methods are bound by JNI signature at runtime;
# renaming or stripping them crashes at the binding site.
-keep class com.dark.ai_rmg.RmgNativeLib {
    public static ** INSTANCE;
    native <methods>;
}
-keepclassmembers class com.dark.ai_rmg.RmgNativeLib {
    public static ** INSTANCE;
}

# JNI looks up onToken(I[B)Z by name + signature on the runtime callback class.
-keep interface com.dark.ai_rmg.RmgTokenCallback { *; }
-keepclassmembers class * implements com.dark.ai_rmg.RmgTokenCallback { *; }

-keep class com.dark.ai_rmg.RmgEngine { public *; }
-keep class com.dark.ai_rmg.RmgEngine$* { public *; }

-keep class com.dark.ai_rmg.models.RmgDims { *; }
-keep class com.dark.ai_rmg.models.DecodingMetrics { *; }
-keep class com.dark.ai_rmg.models.GenerationResult { *; }
-keep class com.dark.ai_rmg.models.GenerationEvent { *; }
-keep class com.dark.ai_rmg.models.GenerationEvent$* { *; }

-keep enum com.dark.ai_rmg.models.RmgLogLevel { *; }

-keep class kotlin.coroutines.Continuation
-keepclassmembers class * implements kotlin.coroutines.Continuation { *; }

-keep class kotlinx.coroutines.flow.** { *; }
-keepclassmembers class * extends kotlinx.coroutines.channels.ProducerScope { *; }
-dontwarn kotlinx.coroutines.**

-dontwarn com.dark.ai_rmg.RmgNativeLib
