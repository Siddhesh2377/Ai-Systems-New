package com.dark.tn_security

/**
 * Stable error code. Numeric values mirror `tn_code` in tn_security.h.
 * The UI dialog branches on these — never renumber, only append.
 */
enum class TnCode(val value: Int) {
    OK                       (0),
    UNKNOWN                  (1),
    CANCELLED                (2),
    INVALID_PARAM            (3),
    NOT_READY                (4),

    // resource
    OOM                      (100),
    DISK_FULL                (101),
    RESOURCE_EXHAUSTED       (102),
    THREAD_POOL_FULL         (103),

    // io
    IO_FAIL                  (200),
    FILE_NOT_FOUND           (201),
    FILE_CORRUPT             (202),
    PERMISSION_DENIED        (203),
    NETWORK_FAIL             (204),

    // model
    MODEL_LOAD_FAIL          (300),
    MODEL_ARCH_UNSUPPORTED   (301),
    MODEL_TEMPLATE_INVALID   (302),
    CONTEXT_OVERFLOW         (303),
    MMAP_FAIL                (304),
    QUANT_UNSUPPORTED        (305),

    // inference
    DECODE_FAIL              (400),
    TOKENIZE_FAIL            (401),
    SAMPLE_FAIL              (402),
    PROJECTOR_MISMATCH       (403),
    KV_CACHE_FAIL            (404),
    GRAPH_BUILD_FAIL         (405),

    // backend / hardware
    BACKEND_INIT_FAIL        (500),
    QNN_HTP_UNAVAILABLE      (501),
    SOC_INCOMPATIBLE         (502),
    GPU_UNAVAILABLE          (503),
    MNN_INIT_FAIL            (504),
    ZSTD_PATCH_FAIL          (505),

    // ipc / service
    AIDL_DEAD_OBJECT         (600),
    AIDL_TIMEOUT             (601),
    AIDL_TRANSACTION_LARGE   (602),
    SERVICE_BIND_FAIL        (603),

    // plugin
    PLUGIN_API_MISMATCH      (700),
    PLUGIN_CLASS_NOT_FOUND   (701),
    PLUGIN_INIT_FAIL         (702),
    PLUGIN_EXEC_FAIL         (703),

    // native crash
    NATIVE_CRASH             (900),
    NATIVE_ABORT             (901);

    companion object {
        fun fromInt(v: Int): TnCode = entries.firstOrNull { it.value == v } ?: UNKNOWN
    }
}
