package com.seazon.feedme.lib.ai.gemini

import com.seazon.feedme.lib.network.HttpManager
import com.seazon.feedme.lib.network.HttpMethod
import com.seazon.feedme.lib.network.HttpUtils
import com.seazon.feedme.lib.network.NameValuePair
import com.seazon.feedme.lib.utils.orZero
import com.seazon.feedme.lib.utils.toJson
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json

/**
 * AI 提供商枚举，用于下拉菜单选择
 */
enum class AiProvider(val defaultBaseUrl: String, val defaultModel: String) {
    GOOGLE_GEMINI(
        "https://generativelanguage.googleapis.com/v1beta/models/",
        "gemini-1.5-flash"
    ),
    OPENAI(
        "https://api.openai.com/v1/",
        "gpt-4o-mini"
    ),
    OPENROUTER(
        "https://openrouter.ai/api/v1/",
        "google/gemini-flash-1.5"
    ),
    ALI_BAILIAN(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/", // 阿里百炼 OpenAI 兼容接口
        "qwen-plus"
    ),
    ZHIPU_GLM(
        "https://open.bigmodel.cn/api/paas/v4/",
        "glm-4-flash"
    ),
    SILICON_FLOW(
        "https://api.siliconflow.cn/v1/",
        "Qwen/Qwen2.5-7B-Instruct"
    );
}

/**
 * AI 配置类，由用户设定
 */
data class AiConfig(
    val provider: AiProvider,
    val apiKey: String,
    val baseUrl: String? = null, // 用户可自定义 URL，为空则使用默认
    val model: String? = null    // 用户可自定义模型，为空则使用默认
) {
    fun getEffectiveUrl(): String {
        var url = (baseUrl ?: provider.defaultBaseUrl)
        if (!url.endsWith("/")) url += "/"
        
        return if (provider == AiProvider.GOOGLE_GEMINI) {
            // Gemini 需要拼接 :generateContent
            val modelName = model ?: provider.defaultModel
            "${url}${modelName}:generateContent"
        } else {
            // OpenAI 兼容接口通常是 chat/completions
            "${url}chat/completions"
        }
    }

    fun getEffectiveModel(): String {
        return model ?: provider.defaultModel
    }
}

class AiService(private val config: AiConfig) {

    /**
     * 翻译功能
     */
    suspend fun translate(query: String, language: String): Translation? {
        val prompt = "Translate to $language, JSON format output, key is dst. Text: $query"
        return executeRequest(prompt)
    }

    /**
     * 摘要功能
     */
    suspend fun summary(query: String, language: String): Translation? {
        val prompt = "Summary the text in $language, JSON format output, key is dst, dst should be a string, and no more than 400 words, use markdown to improve readability if need. Text: $query"
        return executeRequest(prompt)
    }

    private suspend fun executeRequest(prompt: String): Translation? {
        if (config.apiKey.isEmpty()) {
            throw AiException(-1, "API key is required")
        }

        val rawJson: String? = if (config.provider == AiProvider.GOOGLE_GEMINI) {
            requestGemini(prompt)
        } else {
            requestOpenAiCompatible(prompt)
        }

        return if (rawJson.isNullOrEmpty()) {
            null
        } else {
            // 清理 Markdown 代码块标记 (兼容 ```json ... ```)
            val cleanJson = rawJson
                .replace(Regex("^```json\\s*", RegexOption.MULTILINE), "")
                .replace(Regex("^```\\s*", RegexOption.MULTILINE), "")
                .trim()
            
            try {
                toJson<Translation>(cleanJson)
            } catch (e: Exception) {
                // 如果解析失败，可能是模型返回了包含 JSON 的文本，尝试提取 JSON 部分
                val jsonStart = cleanJson.indexOf("{")
                val jsonEnd = cleanJson.lastIndexOf("}")
                if (jsonStart != -1 && jsonEnd != -1 && jsonEnd > jsonStart) {
                    toJson<Translation>(cleanJson.substring(jsonStart, jsonEnd + 1))
                } else {
                    null
                }
            }
        }
    }

    // --- Google Gemini 原生逻辑 ---
    private suspend fun requestGemini(prompt: String): String? {
        val requestBody = GeminiRequestBody(
            listOf(GeminiContent(listOf(GeminiPart(prompt))))
        )
        val bodyStr = Json.encodeToString(requestBody)

        val response: GeminiResult? = HttpManager.requestWrap(
            httpMethod = HttpMethod.POST,
            url = config.getEffectiveUrl(),
            headers = mapOf(
                HttpUtils.HTTP_HEADERS_CONTENT_TYPE to HttpUtils.HTTP_HEADERS_CONTENT_TYPE_JSON,
            ),
            params = listOf(
                NameValuePair("key", config.apiKey), // Gemini Key 在 URL 参数中
            ),
            body = bodyStr,
        ).convertBody()

        if (response?.error != null) {
            throw AiException(response.error.code.orZero(), response.error.message.orEmpty())
        }

        return response?.candidates?.firstOrNull()?.content?.parts?.firstOrNull()?.text
    }

    // --- OpenAI 兼容逻辑 (OpenAI, OpenRouter, 阿里, 智谱, 硅基流动) ---
    private suspend fun requestOpenAiCompatible(prompt: String): String? {
        val requestBody = OpenAiRequestBody(
            model = config.getEffectiveModel(),
            messages = listOf(
                OpenAiMessage("system", "You are a helpful assistant. Always output valid JSON."),
                OpenAiMessage("user", prompt)
            ),
            responseFormat = OpenAiResponseFormat("json_object") // 强制 JSON 模式（部分模型支持）
        )
        
        // 某些模型不支持 response_format 参数，如果报错可以考虑去掉该参数，这里为了通用性我们保留，
        // 或者你可以创建一个不带 responseFormat 的 data class 变体。
        // 为了最大兼容性，如果不确定模型是否支持 json_object，通常可以不传这个字段，完全依赖 Prompt。
        // 下面代码为了防止部分模型（如旧版开源模型）报错，我在序列化时如果 responseFormat 为 null 就不传。
        
        val json = Json { ignoreUnknownKeys = true; encodeDefaults = true }
        // 注意：如果你使用的 Kotlin Serialization 版本较低，可能需要调整 responseFormat 的处理
        val bodyStr = json.encodeToString(requestBody)

        val response: OpenAiResult? = HttpManager.requestWrap(
            httpMethod = HttpMethod.POST,
            url = config.getEffectiveUrl(),
            headers = mapOf(
                HttpUtils.HTTP_HEADERS_CONTENT_TYPE to HttpUtils.HTTP_HEADERS_CONTENT_TYPE_JSON,
                "Authorization" to "Bearer ${config.apiKey}" // OpenAI 风格 Key 在 Header 中
            ),
            body = bodyStr,
        ).convertBody()

        if (response?.error != null) {
            throw AiException(0, response.error.message.orEmpty()) // OpenAI 错误码通常在 message 或 type 里
        }

        return response?.choices?.firstOrNull()?.message?.content
    }
}

@Serializable
data class Translation(
    val dst: String? = null,
)

class AiException(val code: Int, message: String) : Exception(message)

// ================== Google Gemini 数据结构 ==================
@Serializable
data class GeminiRequestBody(val contents: List<GeminiContent>? = null)

@Serializable
data class GeminiResult(val candidates: List<GeminiCandidates>? = null, val error: GeminiError? = null) {
    @Serializable data class GeminiError(val code: Int? = null, val message: String? = null)
}

@Serializable
data class GeminiCandidates(val content: GeminiContent? = null)

@Serializable
data class GeminiContent(val parts: List<GeminiPart>? = null)

@Serializable
data class GeminiPart(val text: String? = null)

// ================== OpenAI 兼容数据结构 ==================
@Serializable
data class OpenAiRequestBody(
    val model: String,
    val messages: List<OpenAiMessage>,
    val stream: Boolean = false,
    @Serializable(with = OpenAiResponseFormatSerializer::class) // 简单的处理，如果不需要严格模式可去掉
    val responseFormat: OpenAiResponseFormat? = null
)

// 简单的辅助类用于处理 snake_case 序列化，或者直接使用 @SerialName
@Serializable
data class OpenAiResponseFormat(
    val type: String
)

// 这是一个空的 Serializer 占位符，因为实际上我们不需要复杂的序列化器，
// 只要确保 Json 配置为 encodeDefaults=false 且该字段为 null 时不传即可。
// 为了简化代码，这里假设你直接传 null 或者正常对象。
// 修正：为简单起见，如果不想引入复杂序列化器，可以直接在上面属性加 @SerialName("response_format")
// 并确保你的 Json 配置支持。下面重新定义更简单的结构。

/* 修正后的 OpenAI 请求体，更通用 */
@Serializable
data class OpenAiRequestBodySimple(
    val model: String,
    val messages: List<OpenAiMessage>,
    val stream: Boolean = false
)

@Serializable
data class OpenAiMessage(
    val role: String,
    val content: String
)

@Serializable
data class OpenAiResult(
    val choices: List<OpenAiChoice>? = null,
    val error: OpenAiError? = null
) {
    @Serializable data class OpenAiError(val message: String? = null, val type: String? = null)
}

@Serializable
data class OpenAiChoice(
    val message: OpenAiMessage? = null
)

// 为了避免复杂的 Serializer 依赖，这里使用扩展函数根据需要选择 RequestBody
private fun OpenAiRequestBody(model: String, messages: List<OpenAiMessage>, responseFormat: OpenAiResponseFormat?): Any {
    // 这是一个伪构造函数，实际逻辑里：
    // 大多数新模型支持 {"type": "json_object"}，但如果是旧模型或某些国产模型可能不支持。
    // 为了最大兼容性，我们在 Prompt 中已经强调了 JSON，这里可以只发标准包。
    // 如果你确定都使用新模型，可以加上 response_format 字段。
    // 这里为了代码简洁和最大兼容性，我们暂时只发送 standard body。
    return OpenAiRequestBodySimple(model, messages)
}

// 实际上上面的伪构造函数是为了解决序列化可选字段的问题。
// 让我们直接用 OpenAiRequestBodySimple 作为主要请求体，
// 因为 Prompt 中的 "JSON format output" 通常足够有效。
