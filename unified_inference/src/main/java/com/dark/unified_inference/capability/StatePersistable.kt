package com.dark.unified_inference.capability

interface StatePersistable {
    fun getStateSize(): Long
    fun stateSaveToFile(path: String): Boolean
    fun stateLoadFromFile(path: String): Boolean
}
