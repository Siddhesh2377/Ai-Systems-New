# tn_security — names are load-bearing via JNI auto-discovery + sealed-class
# reflection by hxs / serialization. Keep everything in the public API surface.

-keep class com.dark.tn_security.** { *; }
-keepclassmembers class com.dark.tn_security.** { *; }

# Enum value() lookups via reflection from JNI.
-keepclassmembers enum com.dark.tn_security.** {
    public static **[] values();
    public static ** valueOf(java.lang.String);
}
