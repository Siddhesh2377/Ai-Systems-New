# Library ProGuard rules for com.dark.gguf_lib.
# Applied when building the library itself with minification enabled.
# Consumer rules in consumer-rules.pro are automatically merged in.

# Crash-report friendly: keep source filenames + line numbers.
-keepattributes SourceFile,LineNumberTable
-renamesourcefileattribute SourceFile

-keepattributes *Annotation*,Signature,InnerClasses,EnclosingMethod,RuntimeVisibleAnnotations

-dontwarn kotlin.Unit
-dontwarn kotlin.**

# Coroutines: keep flow infrastructure; don't blanket-keep all of kotlinx.coroutines.
-keep class kotlinx.coroutines.flow.** { *; }
-keepclassmembers class * {
    @kotlinx.coroutines.** *;
}
-dontwarn kotlinx.coroutines.**

# callbackFlow lambdas extend ProducerScope subclasses; preserve their members.
-keepclassmembers class * extends kotlinx.coroutines.channels.ProducerScope { *; }

-ignorewarnings
