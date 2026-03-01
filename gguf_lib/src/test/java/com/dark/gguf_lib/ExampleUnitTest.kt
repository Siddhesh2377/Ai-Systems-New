package com.dark.gguf_lib

import org.junit.Test
import org.junit.Assert.*

/**
 * Unit tests for Tool-Neuron SDK data classes and utilities.
 * Note: JNI-dependent tests require instrumented tests on an Android device.
 */
class ToolNeuronUnitTest {

    @Test
    fun `SamplingParams defaults are reasonable`() {
        val params = SamplingParams()
        assertEquals(0.7f, params.temperature, 0.01f)
        assertEquals(40, params.topK)
        assertEquals(0.95f, params.topP, 0.01f)
        assertEquals(0.05f, params.minP, 0.01f)
        assertEquals(1.1f, params.repeatPenalty, 0.01f)
        assertEquals(256, params.maxTokens)
    }

    @Test
    fun `EngineStatus fromCode maps correctly`() {
        assertEquals(EngineStatus.OK, EngineStatus.fromCode(0))
        assertEquals(EngineStatus.LOAD_FAILED, EngineStatus.fromCode(1))
        assertEquals(EngineStatus.CONTEXT_FAIL, EngineStatus.fromCode(2))
        assertEquals(EngineStatus.NO_MODEL, EngineStatus.fromCode(3))
        assertEquals(EngineStatus.CANCELLED, EngineStatus.fromCode(6))
    }

    @Test
    fun `ToolCallResult parses tool name`() {
        val result = ToolCallResult("""{"tool": "get_weather", "arguments": {"city": "Tokyo"}}""")
        assertEquals("get_weather", result.toolName)
    }

    @Test
    fun `ToolCallResult parses arguments`() {
        val result = ToolCallResult("""{"tool": "search", "arguments": {"query": "hello"}}""")
        assertEquals("search", result.toolName)
        assertTrue(result.argumentsJson.contains("hello"))
    }

    @Test
    fun `Personality defaults are set`() {
        val p = Personality(name = "Test", persona = "A test character")
        assertEquals("Test", p.name)
        assertEquals(0.7f, p.temperature, 0.01f)
        assertEquals(0.5f, p.creativity, 0.01f)
    }

    @Test
    fun `Mood enum has all values`() {
        assertEquals(10, Mood.entries.size)
        assertEquals(Mood.NEUTRAL, Mood.entries[0])
        assertEquals(Mood.CUSTOM, Mood.entries[9])
    }

    @Test
    fun `ToolDefinition can be created`() {
        val tool = ToolDefinition(
            name = "calculator",
            description = "Perform math",
            params = listOf(
                ToolParam("expression", "Math expression to evaluate"),
                ToolParam("precision", "Decimal places", ToolParamType.NUMBER, false),
            )
        )
        assertEquals("calculator", tool.name)
        assertEquals(2, tool.params.size)
        assertTrue(tool.params[0].required)
        assertFalse(tool.params[1].required)
    }

    @Test
    fun `GenerationResult data class works`() {
        val perf = PerfMetrics(
            promptEvalMs = 100.0,
            generationMs = 500.0,
            promptTokens = 10,
            generatedTokens = 50,
            promptTokensPerSec = 100.0,
            generationTokensPerSec = 100.0,
        )
        val result = GenerationResult(
            status = EngineStatus.OK,
            text = "Hello world",
            perf = perf,
        )
        assertEquals(EngineStatus.OK, result.status)
        assertEquals("Hello world", result.text)
        assertEquals(50, result.perf.generatedTokens)
    }
}
