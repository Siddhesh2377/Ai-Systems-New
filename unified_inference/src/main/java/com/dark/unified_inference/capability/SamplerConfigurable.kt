package com.dark.unified_inference.capability

interface SamplerConfigurable {
    fun updateSamplerParams(paramsJson: String): Boolean
    fun setLogitBias(biasJson: String): Boolean
}
