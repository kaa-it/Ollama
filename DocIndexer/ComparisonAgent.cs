using System.Text.Json;

public class ComparisonAgent(
    ILlmService llmService,
    EnhancedRagPipeline enhancedRag,
    CitationValidator validator)
{
    private const int StructuredMaxTokens = 4096;

    public async Task<(CitationAnswer? answer, RagResult ragResult)> AskWithRagAsync(
        TestQuestion question,
        RagPipelineMode mode,
        CancellationToken ct = default)
    {
        var ragResult = await enhancedRag.ExecuteAsync(question.Question, mode, ct);

        // Жёсткое правило: если релевантность ниже порога — сразу "не знаю"
        if (ragResult.IsUnknown)
        {
            var unknownAnswer = new CitationAnswer(
                Answer: "",
                Confidence: ConfidenceLevel.Unknown,
                ClarificationRequest: $"I don't have enough relevant information to answer this question confidently. " +
                    $"The best matching content has a relevance score of {ragResult.MaxChunkSimilarity:F2}, " +
                    $"which is below my threshold. Please rephrase your question or ask about a different topic.",
                Sources: [],
                Citations: []
            );
            return (unknownAnswer, ragResult);
        }

        var systemPrompt = PromptBuilder.SystemPrompt;
        var userPrompt = PromptBuilder.BuildUserPrompt(question.Question, ragResult.Chunks, ragResult.Confidence);

        CitationAnswer? parsedAnswer = null;
        var maxRetries = GetEnvInt("RAG_MAX_RETRIES", 3);
        var lastException = "";

        for (int attempt = 1; attempt <= maxRetries; attempt++)
        {
            try
            {
                var rawResponse = await llmService.AskAsync(userPrompt, systemPrompt, StructuredMaxTokens, ct);

                parsedAnswer = CitationAnswerParser.Parse(rawResponse, ragResult.Chunks);

                // Если confidence = unknown от LLM, но RAG сказал что можно отвечать — 
                // это inconsistency. Перезапрашиваем с более чёткими инструкциями.
                if (parsedAnswer.Confidence == ConfidenceLevel.Unknown && !ragResult.IsUnknown)
                {
                    if (attempt < maxRetries)
                    {
                        userPrompt += "\n\n[SYSTEM: The context IS sufficient. Do NOT output confidence='unknown'. Provide a concrete answer with citations.]";
                        continue;
                    }
                }

                if (!GetEnvBool("RAG_ENABLE_VALIDATION", true))
                    break;

                var validation = validator.Validate(parsedAnswer, ragResult.Chunks);

                if (validation.IsValid)
                    break;

                if (attempt < maxRetries)
                {
                    userPrompt += $"\n\n[SYSTEM FEEDBACK: Previous response had validation errors: {string.Join("; ", validation.Errors)}. " +
                        $"Please fix and respond with valid JSON only. No markdown, no explanations outside JSON.]";
                }
                else
                {
                    parsedAnswer = CreateFallbackAnswer(ragResult);
                }
            }
            catch (JsonException ex)
            {
                lastException = ex.Message;
                if (attempt < maxRetries)
                {
                    userPrompt += "\n\n[SYSTEM FEEDBACK: Your previous response was not valid JSON. " +
                        "Respond with raw JSON only. No markdown code blocks, no extra text.]";
                }
                else
                {
                    parsedAnswer = CreateFallbackAnswer(ragResult, $"JSON parse error: {ex.Message}");
                }
            }
        }

        // Если после всех попыток parsedAnswer всё ещё null — создаём emergency fallback
        parsedAnswer ??= CreateFallbackAnswer(ragResult, $"All {maxRetries} retries failed. Last error: {lastException}");

        return (parsedAnswer, ragResult);
    }

    private static CitationAnswer CreateFallbackAnswer(RagResult ragResult, string? errorReason = null)
    {
        if (ragResult.Chunks.Count == 0)
        {
            return new CitationAnswer(
                Answer: "Unable to generate an answer based on the retrieved context.",
                Confidence: ConfidenceLevel.Unknown,
                ClarificationRequest: errorReason ?? "No relevant chunks were retrieved.",
                Sources: [],
                Citations: []
            );
        }

        var topChunk = ragResult.Chunks.OrderByDescending(c => c.FinalScore).First();

        // Берём quote без markdown-заголовков — ищем первое предложение с существительным
        var quote = CitationAnswerParser.ExtractSafeQuote(topChunk.Chunk.Content, 150);

        return new CitationAnswer(
            Answer: $"Based on the retrieved context [CITATION:0]: {quote}",
            Confidence: ConfidenceLevel.Low,
            ClarificationRequest: errorReason,
            Sources:
            [
                new SourceReference(
                    Source: topChunk.Chunk.Source,
                    Section: topChunk.Chunk.Section,
                    ChunkId: topChunk.Chunk.ChunkId,
                    RelevanceScore: (float)topChunk.FinalScore,
                    ChunkIndex: topChunk.Chunk.ChunkIndex,
                    TotalChunks: topChunk.Chunk.TotalChunks
                )
            ],
            Citations:
            [
                new Citation(Quote: quote, SourceIndex: 0, Explanation: errorReason ?? "Top retrieved chunk")
            ]
        );
    }

    private static bool GetEnvBool(string name, bool defaultValue) =>
        bool.TryParse(Environment.GetEnvironmentVariable(name), out var v) ? v : defaultValue;

    private static int GetEnvInt(string name, int defaultValue) =>
        int.TryParse(Environment.GetEnvironmentVariable(name), out var v) ? v : defaultValue;
}
