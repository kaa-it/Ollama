Вот готовый промпт для OpenCoder. Он содержит полный контекст текущей архитектуры, детальные требования к новым классам, интеграции и тестированию.

---

## Промпт для OpenCoder

```
Работаем в проекте DocIndexer — .NET 10 консольное приложение (C# 10, top-level statements, без namespace'ов). Нужно реализовать интерактивный CLI мини-чат с RAG, источниками и памятью задачи.

### Текущая архитектура проекта

**Program.cs** (899 строк, top-level): содержит все базовые типы:
- `DotEnvLoader.Load()` — загружает `.env` в `Environment`
- `ChunkingStrategy` enum (Structural)
- `DocumentChunk`, `IndexedChunk`, `ChunkingStats`, `IndexingResult`
- `IChunkingStrategy`, `FixedSizeChunkingStrategy`, `StructuralChunkingStrategy`
- `IVectorStore`, `SqliteVectorStore` (SQLite, cosine similarity в памяти)
- `IEmbeddingService`, `OllamaEmbeddingService` (OllamaSharp, модель `nomic-embed-text`, 768 dim)
- `IQueryRewriteService`, `HeuristicQueryRewriteService`, `LlmQueryRewriteService`
- `IndexingPipeline` — индексация файлов с прогрессом
- `VectorMath.CosineSimilarity`
- Сейчас Program.cs просто: инициализирует store/ollama, индексирует `args[0]` (или `patterns`), читает `test-questions.json` и запускает `EvaluationEngine.RunAsync(questions)`.

**Отдельные файлы:**
- `AnthropicLlmService.cs` — `ILlmService` с методом `AskAsync(string prompt, string? systemPrompt = null, int? maxTokens = null, CancellationToken ct = default)`. Реализует `IDisposable`.
- `EnhancedRagPipeline.cs` — `ExecuteAsync(string question, RagPipelineMode mode, CancellationToken ct)` → `RagResult`. Режимы: `Baseline`, `WithThreshold`, `WithReranker`, `FullPipeline`, `CitationEnforced`. Использует `SimilarityThresholdFilter`, `HeuristicReranker`, `UnknownThresholdHandler`. Конструктор: `(OllamaEmbeddingService, SqliteVectorStore, IQueryRewriteService?)`.
- `ComparisonAgent.cs` — `AskWithRagAsync(TestQuestion, RagPipelineMode, CancellationToken)` → `(CitationAnswer?, RagResult)`. Использует `ILlmService`, `EnhancedRagPipeline`, `CitationValidator`. Внутри: `PromptBuilder.SystemPrompt`, `PromptBuilder.BuildUserPrompt`, `CitationAnswerParser.Parse`, retry-loop с валидацией.
- `PromptBuilder.cs` — `SystemPrompt` (константа, RAW JSON, CITATION markers) и `BuildUserPrompt(question, chunks, confidence)`.
- `CitationAnswerParser.cs` — `Parse(rawResponse, contextChunks)` → `CitationAnswer`. `ExtractJson` — убирает markdown blocks.
- `CitationValidator.cs` — `Validate(CitationAnswer, List<ScoredChunk>)` → `ValidationResult`. Проверяет inline `[CITATION:N]`, exact substring quotes, source indices.
- `RagAnswerModels.cs` — `SourceReference`, `Citation`, `CitationAnswer`, `ConfidenceLevel` enum.
- `EvaluationEngine.cs` — тестирование 10 вопросов.
- `Models.cs` — `TestQuestion`.
- `HeuristicReranker.cs`, `SimilarityThresholdFilter.cs`, `UnknownThresholdHandler.cs`.

**Переменные окружения (в `.env`):**
`ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` (default: claude-opus-4-5-20251101), `OLLAMA_HOST`, `EMBEDDING_MODEL`, `RAG_TOP_K_PRE=10`, `RAG_TOP_K_POST=3`, `RAG_SIMILARITY_THRESHOLD=0.5`, `RAG_UNKNOWN_THRESHOLD=0.45`, `RAG_HIGH_CONFIDENCE_THRESHOLD=0.65`, `RAG_ENABLE_VALIDATION=true`, `RAG_MAX_RETRIES=3`, `RAG_CITATION_MIN_LENGTH=30`, `RAG_CITATION_MAX_LENGTH=200`.

---

### Задача: реализовать мини-чат CLI

Нужно создать следующие **новые файлы** и **минимально изменить** `Program.cs`.

#### 1. `TaskState.cs`
```csharp
public record TaskState
{
    public string? Goal { get; set; }
    public List<string> Constraints { get; set; } = [];
    public List<string> Clarifications { get; set; } = [];
    public Dictionary<string, string> Terms { get; set; } = [];
    public string? ActiveTopic { get; set; }
    public ConfidenceLevel ConfidenceInGoal { get; set; } = ConfidenceLevel.Unknown;
    public DateTime LastUpdated { get; set; } = DateTime.UtcNow;
}
```

#### 2. `ChatMessage.cs`
```csharp
public enum ChatRole { User, Assistant }

public record ChatMessage(
    ChatRole Role,
    string Content,
    DateTime Timestamp,
    List<SourceReference>? Sources = null,
    List<Citation>? Citations = null
);
```

#### 3. `TaskMemoryService.cs`
Сервис анализа диалога и обновления TaskState.

**Конструктор:** `(ILlmService llm)`

**Методы:**
- `async Task UpdateStateAsync(string userMessage, CitationAnswer assistantAnswer, CancellationToken ct)`
    - Отправляет LLM краткий запрос анализа диалога. Промпт примерно:
      ```
      Analyze this dialog turn. Extract:
      GOAL: user's overall goal (1 sentence) or "unknown"
      CONFIDENCE: high/medium/low/unknown
      CONSTRAINTS: comma-separated list
      TERMS: term=definition pairs, comma-separated
      ACTIVE_TOPIC: current topic (1-3 words)
  
      User: {userMessage}
      Assistant: {assistantAnswer.Answer}
  
      Return pipe-delimited: GOAL|CONFIDENCE|CONSTRAINTS|TERMS|ACTIVE_TOPIC
      ```
    - Парсит ответ, обновляет `TaskState`.

- `string BuildContextPrompt()` — возвращает строку для вставки в LLM промпт:
  ```
  [TASK CONTEXT]
  Goal: {Goal}
  Constraints: {string.Join(", ", Constraints)}
  Terms: {string.Join(", ", Terms.Select(...))}
  Active Topic: {ActiveTopic}
  ```
  Если Goal == null, возвращает пустую строку.

- `bool NeedsGoalClarification()` — true если `Goal == null` или `ConfidenceInGoal <= ConfidenceLevel.Low`.

- `string GetStateSnapshot()` — форматированный вывод для консоли.

#### 4. `ChatSession.cs`
Сессия диалога.

**Свойства:**
- `List<ChatMessage> History`
- `TaskMemoryService TaskMemory`
- `string? SessionId`

**Методы:**
- `void AddUserMessage(string content)`
- `void AddAssistantMessage(CitationAnswer answer)` — копирует Sources и Citations.
- `string GetHistoryContext(int maxMessages = 6)` — последние `maxMessages` сообщений в формате:
  ```
  User: {content}
  Assistant: {content}
  ```
  (только текст, без sources).

#### 5. `ChatService.cs`
Основной сервис.

**Конструктор:** `(ILlmService llm, EnhancedRagPipeline rag, CitationValidator validator, TaskMemoryService taskMemory)`

**Методы:**

- `async Task<CitationAnswer> ProcessMessageAsync(string userMessage, ChatSession session, CancellationToken ct)`
    1. Если `session.History.Count == 0` и `taskMemory.NeedsGoalClarification()`:
        - Вернуть `CitationAnswer` с `ClarificationRequest = "Чтобы я мог лучше помочь, уточните: какая ваша цель? (например, 'изучить Rust паттерны' или 'найти анти-паттерны')"` и `Confidence = Unknown`.
    2. Иначе: `var ragResult = await _rag.ExecuteAsync(userMessage, RagPipelineMode.CitationEnforced, ct)`.
    3. Если `ragResult.IsUnknown`:
        - Вернуть fallback с `ClarificationRequest` = "Не найдено релевантного контекста. Переформулируйте вопрос.".
    4. Сформировать полный промпт:
        - `PromptBuilder.SystemPrompt`
        - `\n\n[TASK CONTEXT]\n{taskMemory.BuildContextPrompt()}`
        - `\n\n[DIALOG HISTORY]\n{session.GetHistoryContext(6)}`
        - `\n\n{PromptBuilder.BuildUserPrompt(userMessage, ragResult.Chunks, ragResult.Confidence)}`
    5. `var raw = await _llm.AskAsync(fullPrompt, null, 4096, ct);`
    6. `var answer = CitationAnswerParser.Parse(raw, ragResult.Chunks);`
    7. `var validation = _validator.Validate(answer, ragResult.Chunks);`
    8. Если не валидно и `answer.Confidence != Unknown`:
        - Попробовать ещё 1 раз с feedback-промптом (как в `ComparisonAgent`).
        - Если снова не валидно — использовать fallback аналогично `CreateFallbackAnswer` из `ComparisonAgent` (exact quote + [CITATION:0]).
    9. `await _taskMemory.UpdateStateAsync(userMessage, answer, ct);`
    10. Вернуть `answer`.

- `async Task RunInteractiveAsync(CancellationToken ct)`
    - REPL: выводить `Chat > `, читать `Console.ReadLine()`.
    - Команды:
        - `/exit` — выход.
        - `/reset` — `session = new ChatSession()` (новый `Guid.NewGuid().ToString()`).
        - `/state` — выводить `taskMemory.GetStateSnapshot()`.
        - `/goal` — выводить текущую цель.
        - `/help` — список команд.
    - Для каждого сообщения:
        - `session.AddUserMessage(input)`
        - `var answer = await ProcessMessageAsync(input, session, ct)`
        - `session.AddAssistantMessage(answer)`
        - Выводить `answer.Answer` (с inline [CITATION:N]).
        - Если `answer.Confidence == Unknown`, выводить `ClarificationRequest` жёлтым.
        - Выводить секцию **Sources** (заголовок, список `SourceReference` — Title, Source, Section).
        - Выводить секцию **Citations** (заголовок, список `Citation` — Quote и SourceIndex).
        - Если `taskMemory.State.Goal` изменился (или был установлен впервые), выводить зелёным `[Task Goal updated: {Goal}]`.

#### 6. `ChatScenarioTest.cs`
Тест на 2 длинных сценариях.

**Конструктор:** `(ChatService chatService, TaskMemoryService taskMemory)`

**Методы:**
- `async Task RunScenariosAsync(string scenariosPath, CancellationToken ct)`
    - Читает `test-chat-scenarios.json`.
    - Для каждого сценария:
        - Создаёт `ChatSession` (пустой).
        - Устанавливает `taskMemory.State.Goal = scenario.InitialGoal` (если задан).
        - Прогоняет `scenario.Messages` (List<string>).
        - Для каждого сообщения: `await _chat.ProcessMessageAsync(msg, session, ct)`.
        - Собирает метрики:
            - `goalPreserved` — Goal оставался стабильным после установки.
            - `sourcesAlwaysShown` — все ответы имели `Sources.Count > 0` или `IsUnknown`.
            - `citationsAlwaysShown` — все ответы имели `Citations.Count > 0` или `IsUnknown`.
            - `unknownCount` — сколько раз ответ был Unknown.
            - `avgResponseLength` — средняя длина `answer.Answer`.
        - Выводит отчёт по сценарию в консоль.

**Вывод финального отчёта:**
```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    CHAT SCENARIO TEST REPORT                                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Scenario: {name}                                                             ║
║ Messages: 15                                                                 ║
║ Goal Preserved:        YES / NO                                              ║
║ Sources Always Shown:  YES / NO                                              ║
║ Citations Always Shown: YES / NO                                             ║
║ Unknown Answers:       {N}                                                   ║
║ Avg Response Length:   {N} chars                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

#### 7. `test-chat-scenarios.json`
Создай в корне проекта:

```json
[
  {
    "name": "Rust Design Patterns Deep Dive",
    "initialGoal": "Learn about Rust design patterns",
    "messages": [
      "What is the Builder pattern in Rust?",
      "Why would I use it instead of a large constructor?",
      "Can you show a real code example?",
      "How does it compare to the Factory pattern?",
      "What about RAII guards?",
      "When should I prefer RAII over manual resource management?",
      "Explain the Strategy pattern without traits",
      "Is there a performance cost to using closures as strategies?",
      "How about the Newtype pattern?",
      "What are its primary use cases?",
      "When should I avoid Newtype?",
      "What is the Visitor pattern and can it be implemented in Rust?",
      "What are alternatives to Visitor in Rust?",
      "How does struct decomposition help with borrowing?",
      "Summarize all the patterns we discussed and their trade-offs"
    ]
  },
  {
    "name": "Rust Anti-patterns Analysis",
    "initialGoal": "Understand common anti-patterns in Rust",
    "messages": [
      "What is 'Clone to satisfy the borrow checker'?",
      "Why is it considered an anti-pattern?",
      "When is cloning actually acceptable?",
      "What is the #[deny(warnings)] anti-pattern?",
      "Why is it bad for library authors?",
      "What are the alternatives to denying warnings?",
      "Explain struct decomposition for independent borrowing",
      "When is it useful and when is it overkill?",
      "What anti-patterns exist with the Drop trait?",
      "How can I avoid them?",
      "Is using unwrap() everywhere an anti-pattern?",
      "What is the unwrap_or idiom and how does it help?",
      "What about shadowing variables in match arms?",
      "How does eager cloning affect performance?",
      "Summarize the anti-patterns to avoid in production Rust code"
    ]
  }
]
```

---

### Модификации Program.cs

**Замени** текущий поток (строки ~10-78) на:

```csharp
var mode = (args.Length > 0 ? args[0] : "help").ToLowerInvariant();
var rootDir = args.Length > 1 ? args[1] : "patterns";

// Инициализация общих сервисов
var store = new SqliteVectorStore(dbPath);
await store.InitializeAsync();
var ollama = new OllamaEmbeddingService(ollamaHost, embeddingModel);

// ... проверка Ollama (существующий код) ...

switch (mode)
{
    case "index":
        // текущая индексация
        break;
    case "citations":
        // текущий EvaluationEngine
        break;
    case "chat":
        await RunChatAsync(store, ollama);
        break;
    case "chat-test":
        await RunChatTestAsync(store, ollama);
        break;
    case "help" or "--help" or "-h":
        Console.WriteLine("Usage: dotnet run -- [mode] [rootDir]");
        Console.WriteLine("Modes: index, citations, chat, chat-test");
        break;
    default:
        Console.WriteLine("Unknown mode. Use: index, citations, chat, chat-test");
        break;
}

static async Task RunChatAsync(SqliteVectorStore store, OllamaEmbeddingService ollama)
{
    // Индексация если база пуста
    var count = await store.GetChunkCountAsync();
    if (count == 0)
    {
        // run indexing
    }

    using var llm = new AnthropicLlmService(4096);
    var rewrite = new HeuristicQueryRewriteService();
    var rag = new EnhancedRagPipeline(ollama, store, rewrite);
    var validator = new CitationValidator();
    var taskMemory = new TaskMemoryService(llm);
    var chat = new ChatService(llm, rag, validator, taskMemory);

    await chat.RunInteractiveAsync();
}

static async Task RunChatTestAsync(SqliteVectorStore store, OllamaEmbeddingService ollama)
{
    // Аналогично, но создаёт ChatScenarioTest и вызывает RunScenariosAsync
}
```

**Важно:** при `chat` и `chat-test` перед запуском проверить, что индекс не пустой. Если пустой — проиндексировать `rootDir` (Structural strategy) с сообщением "Индекс пуст, выполняется индексация...".

---

### Требования к качеству

1. **C# 10, top-level statements, target-typed new** (`new()`, `[]`). Без namespace.
2. `AnthropicLlmService` — `using var`, не забывай `IDisposable`.
3. `EnhancedRagPipeline` — конструктор `(OllamaEmbeddingService, SqliteVectorStore, IQueryRewriteService?)`.
4. `CitationValidator` — конструктор без параметров (использует `Environment.GetEnvironmentVariable`).
5. `PromptBuilder.SystemPrompt` — статическое поле, `PromptBuilder.BuildUserPrompt` — статический метод.
6. История в промпте: **максимум 6 последних сообщений** (3 пары).
7. TaskState обновляется **после** каждого ответа ассистента.
8. Если TaskState.Goal == null, ассистент в первом сообщении **не отвечает на вопрос**, а просит уточнить цель.
9. Все ответы ассистента (кроме Unknown) должны содержать `Sources` и `Citations`.
10. Fallback при validation error — exact quote из top chunk с `[CITATION:0]`.

---

### Проверка после реализации

1. `dotnet build` — должен пройти без ошибок.
2. `dotnet run -- chat-test` — должен прочитать `test-chat-scenarios.json`, прогнать 2 сценария по 15 сообщений, и вывести отчёт:
    - Goal Preserved: YES для обоих
    - Sources Always Shown: YES для обоих
    - Citations Always Shown: YES для обоих
    - Unknown Answers: допустимо 0-2 на сценарий
    - Без необработанных исключений.

Выведи полный исходный код **всех новых файлов** и **точный diff/вставку для Program.cs** (только нужные изменения).
```

---

**Want me to also generate a test script that validates the OpenCoder output, or prepare the `test-chat-scenarios.json` file right now so you can hand it to OpenCoder as a reference?**