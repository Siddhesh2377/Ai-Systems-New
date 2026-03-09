package com.dark.unified_inference.model

sealed class ModelSource {
    data class FilePath(val path: String) : ModelSource()
    data class FileDescriptor(val fd: Int) : ModelSource()
    data class ContentUri(val uri: String) : ModelSource()
    data class Directory(val path: String) : ModelSource()
}
