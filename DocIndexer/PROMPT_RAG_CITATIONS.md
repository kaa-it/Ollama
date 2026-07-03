# Промпт для доработки RAG: обязательные источники, цитаты и режим "не знаю"

> **Цель:** Модифицировать существующий RAG-pipeline (DocIndexer) так, чтобы каждый ответ гарантированно содержал источники, цитаты из найденных чанков, и при низкой релевантности модель отвечала "не знаю" с просьбой уточнить запрос.

---

## 1. Текущее состояние системы

Проект: `DocIndexer` — .NET 10 консольное приложение для семантического поиска по документации Rust Design Patterns.

### Ключевые компоненты:
- **`EnhancedRagPipeline`** — выполняет retrieval: embed → vector search → threshold filter → rerank → формирует `RagResult`
- **`ComparisonAgent`** — формирует простой промпт и вызывает `AnthropicLlmService.AskAsync()`
- **`AnthropicLlmService`** — обёртка над Claude API (system prompt + user prompt)
- **`EvaluationEngine`** — прогоняет 10 вопросов из `test-questions.json`, сохраняет результаты в SQLite
- **`Models`** — `DocumentChunk`, `IndexedChunk`, `ScoredChunk`, `RagResult`

### Текущие проблемы:
1. **Нет структурированных цитат** — LLM получает plaintext контекст без инструкций выделять цитаты
2. **Нет валидации источников** — не проверяется, что модель реально использовала найденные чанки
3. **Нет режима "не знаю"** — при similarity < threshold система всё равно задаёт вопрос LLM
4. **Промпт не требует JSON-ответа** — ответ непредсказуемый, парсинг невозможен

---

## 2. Архитектура изменений

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│  User Question  │────▶│  EnhancedRagPipeline │────▶│  Structured Prompt  │
└─────────────────┘     └──────────────────────┘     └─────────────────────┘
                               │                              │
                               ▼                              ▼
                        ┌──────────────┐              ┌──────────────┐
                        │  Retrieval   │              │  Claude API  │
                        │  + Rerank    │              │  (JSON mode) │
                        └──────────────┘              └──────────────┘
                               │                              │
                               ▼                              ▼
                        ┌──────────────────────────────────────────┐
                        │         CitationAnswer (NEW)             │
                        │  {                                       │
                        │    "answer": string,                     │
                        │    "confidence": "high" | "low",         │
                        │    "sources": [                          │
                        │      { "source": "...",                  │
                        │        "section": "...",                 │
                        │        "chunk_id": "...",                │
                        │        "relevance_score": 0.85 }         │
                        │    ],                                    │
                        │    "citations": [                        │
                        │      { "quote": "...",                   │
                        │        "source_index": 0 }               │
                        │    ]                                     │
                        │  }                                       │
                        └──────────────────────────────────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │  Validator   │  ◄── проверка: цитаты ∈ контекст?
                        │  (NEW)       │      источники совпадают с chunks?
                        └──────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │  Fallback    │  ◄── если validation failed → retry
                        │  / Retry     │      или fallback на "не знаю"
                        └──────────────┘
```

---

## 3. Пошаговые изменения

### 3.1 Добавить модели данных (новый файл `RagAnswerModels.cs`)

```csharp
public record SourceReference(
    string Source,           // Путь к файлу (нормализованный)
    string? Section,         // Заголовок секции
    string ChunkId,          // UUID чанка
    float RelevanceScore,    // similarity / finalScore
    int ChunkIndex,          // Порядковый номер чанка
    int TotalChunks          // Всего чанков в документе
);

public record Citation(
    string Quote,            // Дословная цитата из чанка (30-200 символов)
    int SourceIndex,         // Индекс в массиве sources
    string? Explanation      // Почему эта цитата релевантна (опционально)
);

public enum ConfidenceLevel { High, Medium, Low, Unknown }

public record CitationAnswer(
    string Answer,                    // Основной текст ответа
    ConfidenceLevel Confidence,       // Уверенность модели
    string? ClarificationRequest,     // Если Confidence == Unknown — просьба уточнить
    List<SourceReference> Sources,    // Список использованных источников
    List<Citation> Citations          // Цитаты с привязкой к sources
);

public record UnknownAnswer(
    string Reason,                    // Почему не удалось ответить
    string Suggestion,                // Что уточнить у пользователя
    float? MaxSimilarity              // Максимальная similarity найденных чанков
);
```

### 3.2 Добавить пороговый обработчик (новый файл `UnknownThresholdHandler.cs`)

```csharp
public class UnknownThresholdHandler
{
    private readonly float _minSimilarity;
    private readonly float _minHighConfidenceSimilarity;

    public UnknownThresholdHandler(
        float minSimilarity = 0.45f,      // Абсолютный минимум — ниже = "не знаю"
        float minHighConfidence = 0.65f   // Выше = можно отвечать уверенно
    )
    {
        _minSimilarity = minSimilarity;
        _minHighConfidenceSimilarity = minHighConfidence;
    }

    /// <summary>
    /// Определяет, достаточно ли релевантен контекст для ответа
    /// </summary>
    public RelevanceAssessment AssessRelevance(List<ScoredChunk> chunks)
    {
        if (chunks.Count == 0)
            return new RelevanceAssessment(false, 0, "No chunks found");

        var maxSim = chunks.Max(c => c.FinalScore);
        var avgSim = chunks.Average(c => c.FinalScore);
        var topChunk = chunks.OrderByDescending(c => c.FinalScore).First();

        // Правило: если даже лучший чанк ниже порога — контекста недостаточно
        if (maxSim < _minSimilarity)
        {
            return new RelevanceAssessment(
                canAnswer: false,
                maxSim,
                reason: $"Best chunk relevance ({maxSim:F3}) below threshold ({_minSimilarity})"
            );
        }

        // Правило: если максимальная релевантность низкая, но выше абсолютного минимума — отвечаем с осторожностью
        var confidence = maxSim >= _minHighConfidenceSimilarity
            ? ConfidenceLevel.High
            : (maxSim >= _minSimilarity + 0.1f ? ConfidenceLevel.Medium : ConfidenceLevel.Low);

        return new RelevanceAssessment(true, maxSim, null, confidence);
    }
}

public record RelevanceAssessment(
    bool CanAnswer,
    float MaxSimilarity,
    string? Reason,
    ConfidenceLevel Confidence = ConfidenceLevel.Unknown
);
```

### 3.3 Переработать `EnhancedRagPipeline.ExecuteAsync`

**Требования:**
1. После rerank вызвать `UnknownThresholdHandler.AssessRelevance(finalChunks)`
2. Если `CanAnswer == false` — вернуть `RagResult` с флагом `IsUnknown = true` и причиной
3. Иначе — продолжить формирование контекста как раньше
4. Добавить в `RagResult`: `float MaxChunkSimilarity`, `ConfidenceLevel Confidence`, `bool IsUnknown`

```csharp
public record RagResult(
    string Context,
    string[] Sources,
    string OriginalQuestion,
    string? RewrittenQuestion,
    List<ScoredChunk> Chunks,
    RagPipelineMode Mode,
    float SimilarityThreshold,
    int TopKPre,
    int TopKPost,
    int FilteredCount,
    // === NEW FIELDS ===
    float MaxChunkSimilarity,
    ConfidenceLevel Confidence,
    bool IsUnknown,              // true → ассистент должен сказать "не знаю"
    string? UnknownReason        // почему не знает
);
```

### 3.4 Создать структурированный промпт (новый файл `PromptBuilder.cs`)

```csharp
public class PromptBuilder
{
    private const string SystemPrompt = @"You are a precise technical assistant for Rust design patterns documentation. 
Your answers MUST be grounded exclusively in the provided context chunks.

STRICT RULES:
1. Answer ONLY using information from the provided context chunks.
2. If the context does not contain sufficient information, output JSON with confidence: ""unknown"".
3. Every factual claim MUST be backed by a citation from the context.
4. Citations must be EXACT substrings (30-200 chars) from the context — no paraphrasing.
5. NEVER fabricate sources, citations, or facts not present in the context.
6. Respond in the same language as the user's question.

OUTPUT FORMAT — strict JSON:
{
  ""answer"": ""<comprehensive answer with inline [CITATION:N] markers>"",
  ""confidence"": ""<high|medium|low|unknown>"",
  ""sources"": [
    { ""index"": 0, ""source"": ""<file path>"", ""section"": ""<section name>"", ""chunk_id"": ""<uuid>"", ""score"": 0.85 }
  ],
  ""citations"": [
    { ""index"": 0, ""quote"": ""<exact text from chunk>"", ""source_index"": 0 }
  ],
  ""clarification_request"": ""<if confidence=unknown, ask user to clarify>""
}";

    public static string BuildUserPrompt(string question, List<ScoredChunk> chunks, ConfidenceLevel confidence)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"User question: {question}");
        sb.AppendLine($"Estimated confidence based on retrieval: {confidence}");
        sb.AppendLine();
        sb.AppendLine("Context chunks (use ONLY these to answer):");
        sb.AppendLine("============================================");

        for (int i = 0; i < chunks.Count; i++)
        {
            var c = chunks[i];
            sb.AppendLine($"--- CHUNK [{i}] ---");
            sb.AppendLine($"source: {c.Chunk.Source}");
            sb.AppendLine($"section: {c.Chunk.Section ?? "N/A"}");
            sb.AppendLine($"chunk_id: {c.Chunk.ChunkId}");
            sb.AppendLine($"chunk_index: {c.Chunk.ChunkIndex}/{c.Chunk.TotalChunks}");
            sb.AppendLine($"relevance_score: {c.FinalScore:F3}");
            sb.AppendLine("--- CONTENT ---");
            sb.AppendLine(c.Chunk.Content);
            sb.AppendLine();
        }

        sb.AppendLine("============================================");
        sb.AppendLine();
        sb.AppendLine("INSTRUCTIONS:");
        sb.AppendLine("1. Provide a JSON response following the system format.");
        sb.AppendLine("2. Include at least one citation for every key claim.");
        sb.AppendLine("3. Use [CITATION:0], [CITATION:1], etc. inline in the answer text.");
        sb.AppendLine("4. If confidence is 'unknown', set answer to empty string and provide clarification_request.");
        sb.AppendLine("5. The 'quote' in each citation MUST be an exact substring of the corresponding chunk content.");

        return sb.ToString();
    }
}
```

### 3.5 Создать парсер ответа (новый файл `CitationAnswerParser.cs`)

```csharp
public class CitationAnswerParser
{
    public static CitationAnswer Parse(string jsonResponse, List<ScoredChunk> contextChunks)
    {
        // Парсинг JSON в CitationAnswer
        // Валидация: каждый source_index < sources.Count
        // Валидация: каждый quote присутствует в соответствующем chunk.Content
        // Валидация: confidence consistency (unknown → answer должен быть пустым)
    }

    private static bool ValidateQuoteExists(string quote, string chunkContent)
    {
        // Нормализовать whitespace (collapse multiple spaces/newlines)
        // Проверить substring containment
        // Допуск: quote может быть обрезан по границам предложения
    }
}
```

### 3.6 Создать валидатор цитат (новый файл `CitationValidator.cs`)

```csharp
public class CitationValidator
{
    public ValidationResult Validate(CitationAnswer answer, List<ScoredChunk> contextChunks)
    {
        var errors = new List<string>();
        var warnings = new List<string>();

        // 1. Проверить, что sources не пуст (если confidence != unknown)
        if (answer.Confidence != ConfidenceLevel.Unknown && answer.Sources.Count == 0)
            errors.Add("No sources provided despite non-unknown confidence");

        // 2. Проверить, что citations не пуст (если confidence != unknown)
        if (answer.Confidence != ConfidenceLevel.Unknown && answer.Citations.Count == 0)
            errors.Add("No citations provided despite non-unknown confidence");

        // 3. Проверить, что каждая цитата существует в контексте
        foreach (var citation in answer.Citations)
        {
            if (citation.SourceIndex >= contextChunks.Count)
            {
                errors.Add($"Citation references invalid source index {citation.SourceIndex}");
                continue;
            }

            var chunk = contextChunks[citation.SourceIndex];
            if (!chunk.Chunk.Content.Contains(citation.Quote, StringComparison.OrdinalIgnoreCase))
            {
                // Попробовать fuzzy match (нормализованный whitespace)
                var normalizedQuote = NormalizeWhitespace(citation.Quote);
                var normalizedContent = NormalizeWhitespace(chunk.Chunk.Content);
                if (!normalizedContent.Contains(normalizedQuote, StringComparison.OrdinalIgnoreCase))
                {
                    errors.Add($"Quote not found in chunk {citation.SourceIndex}: '{citation.Quote[..Math.Min(50, citation.Quote.Length)]}...'");
                }
            }
        }

        // 4. Проверить, что answer содержит [CITATION:N] ссылки
        if (answer.Confidence != ConfidenceLevel.Unknown)
        {
            var citationRefs = Regex.Matches(answer.Answer, @"\[CITATION:(\d+)\]");
            if (citationRefs.Count == 0)
                warnings.Add("Answer has no inline [CITATION:N] references");

            var referencedIndices = citationRefs.Select(m => int.Parse(m.Groups[1].Value)).ToHashSet();
            var availableIndices = answer.Citations.Select(c => c.Index).ToHashSet();
            var missing = referencedIndices.Except(availableIndices).ToList();
            if (missing.Count > 0)
                errors.Add($"Answer references missing citations: {string.Join(", ", missing)}");
        }

        // 5. Проверить consistency: если IsUnknown в RAG — confidence должен быть Unknown
        // (это проверяется на уровне ComparisonAgent)

        return new ValidationResult(errors.Count == 0, errors, warnings);
    }

    private static string NormalizeWhitespace(string text) =>
        Regex.Replace(text.Trim(), @"\s+", " ");
}

public record ValidationResult(bool IsValid, List<string> Errors, List<string> Warnings);
```

### 3.7 Переработать `ComparisonAgent`

```csharp
public class ComparisonAgent(
    ILlmService llmService,
    EnhancedRagPipeline enhancedRag,
    UnknownThresholdHandler thresholdHandler,
    CitationValidator validator)
{
    public async Task<(CitationAnswer? answer, RagResult ragResult)> AskWithRagAsync(
        TestQuestion question,
        RagPipelineMode mode,
        CancellationToken ct = default)
    {
        // 1. Retrieval
        var ragResult = await enhancedRag.ExecuteAsync(question.Question, mode, ct);

        // 2. Проверка на "не знаю" (жёсткое правило на уровне retrieval)
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

        // 3. Формирование промпта
        var systemPrompt = PromptBuilder.SystemPrompt;
        var userPrompt = PromptBuilder.BuildUserPrompt(
            question.Question,
            ragResult.Chunks,
            ragResult.Confidence
        );

        // 4. Вызов LLM с retry
        CitationAnswer? parsedAnswer = null;
        var maxRetries = 3;
        for (int attempt = 1; attempt <= maxRetries; attempt++)
        {
            try
            {
                var rawResponse = await llmService.AskAsync(userPrompt, systemPrompt, ct);

                // 5. Парсинг
                parsedAnswer = CitationAnswerParser.Parse(rawResponse, ragResult.Chunks);

                // 6. Валидация
                var validation = validator.Validate(parsedAnswer, ragResult.Chunks);

                if (validation.IsValid)
                    break;

                // Если валидация не прошла и это не последняя попытка — добавить feedback в промпт
                if (attempt < maxRetries)
                {
                    userPrompt += $"\n\n[SYSTEM FEEDBACK: Previous response had validation errors: {string.Join("; ", validation.Errors)}. Please fix and respond with valid JSON.]";
                }
                else
                {
                    // Fallback: если после всех попыток валидация не прошла — использовать fallback ответ
                    parsedAnswer = CreateFallbackAnswer(ragResult, validation);
                }
            }
            catch (JsonException ex)
            {
                if (attempt == maxRetries)
                    parsedAnswer = CreateFallbackAnswer(ragResult, new ValidationResult(false, [$"JSON parse error: {ex.Message}"], []));
            }
        }

        return (parsedAnswer, ragResult);
    }

    private CitationAnswer CreateFallbackAnswer(RagResult ragResult, ValidationResult validation)
    {
        // Создать минимальный валидный ответ на основе топ-чанка
        var topChunk = ragResult.Chunks.OrderByDescending(c => c.FinalScore).First();
        var quote = topChunk.Chunk.Content.Length > 150
            ? topChunk.Chunk.Content[..150] + "..."
            : topChunk.Chunk.Content;

        return new CitationAnswer(
            Answer: $"Based on the retrieved context: {quote}",
            Confidence: ConfidenceLevel.Low,
            ClarificationRequest: null,
            Sources: [
                new SourceReference(
                    Source: topChunk.Chunk.Source,
                    Section: topChunk.Chunk.Section,
                    ChunkId: topChunk.Chunk.ChunkId,
                    RelevanceScore: topChunk.FinalScore,
                    ChunkIndex: topChunk.Chunk.ChunkIndex,
                    TotalChunks: topChunk.Chunk.TotalChunks
                )
            ],
            Citations: [
                new Citation(Quote: quote, SourceIndex: 0, Explanation: "Top retrieved chunk")
            ]
        );
    }
}
```

### 3.8 Обновить `EvaluationEngine` для проверки на 10 вопросах

Добавить три проверки для каждого ответа:

```csharp
public class CitationEvaluationResult
{
    public int QuestionId { get; set; }
    public bool HasSources { get; set; }           // ✓ источники присутствуют
    public bool HasCitations { get; set; }         // ✓ цитаты присутствуют
    public bool CitationsMatchContext { get; set; } // ✓ цитаты найдены в чанках
    public bool AnswerConsistentWithCitations { get; set; } // ✓ смысл ответа соответствует цитатам
    public bool CorrectlySaidUnknown { get; set; }  // ✓ если релевантность низкая → сказал "не знаю"
    public List<string> Errors { get; set; } = [];
}

// В RunAsync, после получения ответа:
var eval = new CitationEvaluationResult
{
    QuestionId = q.Id,
    HasSources = answer?.Sources.Count > 0 ?? false,
    HasCitations = answer?.Citations.Count > 0 ?? false,
    CitationsMatchContext = validator.Validate(answer, ragResult.Chunks).IsValid,
    CorrectlySaidUnknown = ragResult.IsUnknown == (answer?.Confidence == ConfidenceLevel.Unknown)
};

// Сохранить eval в отдельную таблицу evaluation_citations
```

### 3.9 Добавить режим `RagPipelineMode.CitationEnforced`

```csharp
public enum RagPipelineMode
{
    Baseline,
    WithThreshold,
    WithReranker,
    FullPipeline,
    CitationEnforced  // NEW: полный pipeline + обязательные цитаты + режим "не знаю"
}
```

---

## 4. Тестовые проверки (на 10 вопросах из `test-questions.json`)

### Что проверять:

| # | Проверка | Критерий успеха |
|---|----------|----------------|
| 1 | В каждом ответе есть `sources` | `Sources.Count > 0` (кроме Unknown) |
| 2 | В каждом ответе есть `citations` | `Citations.Count > 0` (кроме Unknown) |
| 3 | Каждая цитата найдена в чанке | `Citations.All(c => chunk.Content.Contains(c.Quote))` |
| 4 | Смысл ответа соответствует цитатам | Answer.Contains(key concepts from citations) |
| 5 | При low relevance → "не знаю" | `MaxSimilarity < 0.45` → `Confidence == Unknown` |
| 6 | Inline citations [CITATION:N] присутствуют | Regex.Matches(Answer, @"\[CITATION:\d+\]").Count > 0 |
| 7 | Нет hallucinated sources | Все `Sources` ⊆ `ragResult.Chunks` |
| 8 | JSON parseable | Нет JsonException |
| 9 | Время ответа < 30 сек | Stopwatch.Elapsed < 30s |
| 10 | Consistency mode | Одинаковый вопрос даёт похожие sources |

### Ожидаемый результат для 10 вопросов:

```
Question 1/10: medium — "What is the RAII guard pattern..."
  Mode: CitationEnforced
  IsUnknown: false
  Confidence: High
  Sources: 3 (patterns/behavioural/RAII.md)
  Citations: 4 ✓
  Validation: PASSED ✓
  Citations in context: 4/4 ✓
  Inline refs: [CITATION:0], [CITATION:1] ✓

Question 5/10: medium — "What is the Newtype pattern..."
  Mode: CitationEnforced
  IsUnknown: false
  Confidence: High
  Sources: 2 (patterns/behavioural/newtype.md)
  Citations: 3 ✓
  Validation: PASSED ✓

... (для всех 10 вопросов)

╔══════════════════════════════════════════════════════════════════╗
║           CITATION EVALUATION SUMMARY                            ║
╠══════════════════════════════════════════════════════════════════╣
║ Total questions: 10                                              ║
║ With sources:    10/10 (100%) ✓                                  ║
║ With citations:  10/10 (100%) ✓                                  ║
║ Citations valid: 10/10 (100%) ✓                                  ║
║ Correctly unknown: N/M (depends on threshold)                    ║
║ Avg response time: X ms                                          ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 5. Environment Variables

Добавить новые переменные:

```bash
export RAG_UNKNOWN_THRESHOLD=0.45        # Минимальная similarity для ответа
export RAG_HIGH_CONFIDENCE_THRESHOLD=0.65 # Порог высокой уверенности
export RAG_CITATION_MIN_LENGTH=30        # Минимальная длина цитаты
export RAG_CITATION_MAX_LENGTH=200       # Максимальная длина цитаты
export RAG_MAX_RETRIES=3                 # Количество retry при валидации
export RAG_ENABLE_VALIDATION=true        # Включить валидацию цитат
```

---

## 6. Интеграция с существующим кодом

### Файлы, которые нужно изменить:

1. **`Models.cs`** — добавить `ConfidenceLevel`, `SourceReference`, `Citation`, `CitationAnswer`, `UnknownAnswer`
2. **`EnhancedRagPipeline.cs`** — интегрировать `UnknownThresholdHandler`, добавить поля в `RagResult`
3. **`ComparisonAgent.cs`** — полностью переработать: структурированный промпт, парсинг, валидация, retry
4. **`EvaluationEngine.cs`** — добавить `CitationEvaluationResult`, сохранение в БД, сводную таблицу
5. **`Program.cs`** — добавить `CitationEnforced` в switch, зарегистрировать новые сервисы

### Новые файлы:

- `RagAnswerModels.cs` — модели ответа
- `UnknownThresholdHandler.cs` — логика "не знаю"
- `PromptBuilder.cs` — генерация промптов
- `CitationAnswerParser.cs` — парсинг JSON-ответа LLM
- `CitationValidator.cs` — валидация цитат против контекста

### Критерий готовности:

- [ ] `dotnet build` проходит без ошибок
- [ ] `dotnet run -- compare` с режимом `CitationEnforced` прогоняет 10 вопросов
- [ ] Все 10 ответов содержат sources
- [ ] Все 10 ответов содержат citations
- [ ] Все цитаты подтверждены в чанках (валидация)
- [ ] При тестовом запросе с релевантностью < 0.45 — ответ "не знаю"
- [ ] Результаты сохраняются в БД в таблице `evaluations` + `evaluation_citations`
