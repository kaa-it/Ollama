Проведён детальный аудит кода. Найдено **5 критических багов**, **7 логических ошибок** и **1 мусорный класс**. Вот полный отчёт и исправленный промпт для OpenCode:

---

## 🔴 Критические баги

| # | Файл | Баг | Последствия |
|---|------|-----|-------------|
| 1 | `AnthropicLlmService.cs:28` | `Model = AnthropicModels.Claude45Opus` (hardcoded) вместо `_model` | Игнорируется env var, используется другая модель (дороже/медленнее) |
| 2 | `AnthropicLlmService.cs:29` | `Temperature = _temperature` — пользователь запретил temperature | Runtime error или непредсказуемое поведение SDK |
| 3 | `RagPipeline.cs:12` | Двойной префикс: `search_document: search_query: вопрос` | Эмбеддинг запроса совершенно неверен, поиск возвращает мусор |
| 4 | `EvaluationEngine.cs:22` | `new OllamaEmbeddingService()` — без параметров | Игнорируются `OLLAMA_HOST`/`EMBEDDING_MODEL`, ломается если Ollama не на localhost |
| 5 | `EvaluationEngine.cs:130-132` | `conn.Open()`, `cmd.ExecuteReader()`, `reader.Read()` — синхронные вызовы в async методе | Потенциальный deadlock, блокировка потока |

---

## 🟡 Логические ошибки и мусор

| # | Проблема | Где |
|---|----------|-----|
| 6 | `RagContext` record объявлен, но нигде не используется | `Models.cs` |
| 7 | Нет `key_concepts` в `TestQuestion` — невозможна автоматическая оценка | `Models.cs` + `test-questions.json` |
| 8 | `IndexOf("patterns/")` — не работает на Windows (`\`), с другим регистром | `RagPipeline.cs:17` |
| 9 | Default mode = `"compare"` — неожиданный запуск индексации + API-вызовов без аргументов | `Program.cs` |
| 10 | Нет проверки `File.Exists("test-questions.json")` | `Program.cs` |
| 11 | Нет команды `help` | `Program.cs` |
| 12 | Нет кэширования ответов — 20 API-вызовов каждый запуск | `EvaluationEngine.cs` |

---

## 📋 Исправленный промпт для OpenCode

```markdown
# ИСПРАВЛЕНИЕ БАГОВ В RAG-АГЕНТЕ

## Контекст

Код сгенерирован ранее, но при аудите найдены критические баги. Нужно исправить ВСЕ файлы.

## Критические баги (исправить обязательно)

### 1. AnthropicLlmService.cs — HARDCODED модель + Temperature

**Баг 1a:** `Model = Anthropic.SDK.Constants.AnthropicModels.Claude45Opus` — игнорируется поле `_model`.

```csharp
// БЫЛО (НЕПРАВИЛЬНО):
Model = Anthropic.SDK.Constants.AnthropicModels.Claude45Opus,

// ДОЛЖНО БЫТЬ:
Model = _model,
```

**Баг 1b:** `Temperature = _temperature` — пользователь явно указал что Temperature НЕ поддерживается в Anthropic SDK 12.24.1.

```csharp
// УДАЛИТЬ полностью:
// - private readonly decimal _temperature;
// - _temperature = 0.1m; (в конструкторе)
// - Temperature = _temperature (в MessageParameters)
```

### 2. RagPipeline.cs + OllamaEmbeddingService.cs — ДВОЙНОЙ префикс embeddings

**Баг:** `RagPipeline` вызывает `GenerateEmbeddingsAsync([$"search_query: {question}"])`, но `OllamaEmbeddingService.GenerateEmbeddingsAsync` ВНУТРИ добавляет префикс `"search_document:"` ко ВСЕМ текстам. Результат: `"search_document: search_query: What is RAII..."` — НЕПРАВИЛЬНО!

**Исправление:** Добавить в `OllamaEmbeddingService` отдельный метод для query embeddings:

```csharp
// В OllamaEmbeddingService добавить:
public async Task<float[]> GenerateQueryEmbeddingAsync(string query, CancellationToken ct = default)
{
    var textList = new List<string> { query };
    var batches = new List<List<string>> { textList };
    var results = new float[1][];
    var index = 0;

    foreach (var b in batches)
    {
        // Для query используем префикс "search_query:"
        var prefixedTexts = b.Select(t => $"search_query: {t}").ToList();
        var embeddings = await EmbedWithRetryAsync(prefixedTexts, ct);
        foreach (var emb in embeddings)
        {
            if (emb.Length != 768)
                throw new InvalidOperationException($"Expected 768-dimensional embedding, got {emb.Length}");
            results[index++] = emb;
        }
    }
    return results[0];
}
```

В `RagPipeline.cs` заменить:
```csharp
// БЫЛО:
var queryEmbedding = (await embeddingService.GenerateEmbeddingsAsync([$"search_query: {question}"], ct))[0];

// ДОЛЖНО БЫТЬ:
var queryEmbedding = await embeddingService.GenerateQueryEmbeddingAsync(question, ct);
```

### 3. EvaluationEngine.cs — RagPipeline создается без параметров

**Баг:** `new RagPipeline(new OllamaEmbeddingService(), vectorStore)` — создает OllamaEmbeddingService с дефолтами (localhost:11434), игнорируя env vars.

**Исправление:** Передавать `OllamaEmbeddingService` через конструктор:

```csharp
public class EvaluationEngine
{
    private readonly string _dbPath;
    private readonly OllamaEmbeddingService _embeddingService;
    
    public EvaluationEngine(string dbPath, OllamaEmbeddingService embeddingService)
    {
        _dbPath = dbPath;
        _embeddingService = embeddingService;
    }
    
    // В RunAsync:
    var ragPipeline = new RagPipeline(_embeddingService, vectorStore);
}
```

### 4. EvaluationEngine.cs — Синхронные методы в async контексте

**Баг:** В методе `PrintSummary` используются синхронные `conn.Open()`, `cmd.ExecuteReader()`, `reader.Read()`.

**Исправление:** Заменить на async версии:
```csharp
await conn.OpenAsync();
using var reader = await cmd.ExecuteReaderAsync();
while (await reader.ReadAsync())
```

### 5. RagPipeline.cs — Ненадежная нормализация путей

**Баг:** `IndexOf("patterns/")` не работает на Windows (`\`), с другим регистром.

**Исправление:**
```csharp
var sources = chunksList
    .Select(c => c.Source)
    .Select(p =>
    {
        var normalized = p.Replace('\\', '/');
        var idx = normalized.IndexOf("patterns/", StringComparison.OrdinalIgnoreCase);
        return idx >= 0 ? normalized[idx..] : Path.GetFileName(p);
    })
    .Distinct()
    .ToArray();
```

---

## Логические ошибки и улучшения

### 6. Удалить мусорный код

**Удалить** `RagContext` record из `Models.cs` — нигде не используется.

### 7. Добавить key_concepts в TestQuestion

**В `Models.cs`:**
```csharp
public record TestQuestion
{
    public required int Id { get; init; }
    public required string Question { get; init; }
    public string? ExpectedAnswer { get; init; }
    public string[]? ExpectedSources { get; init; }
    public string? Difficulty { get; init; }
    public string[]? KeyConcepts { get; init; }  // НОВОЕ
}
```

**В `test-questions.json` добавить поле `key_concepts`** для каждого вопроса (массив строк). Пример для Q1:
```json
"key_concepts": ["RAII", "guard object", "borrow checker", "MutexGuard", "Deref", "lifetime"]
```

### 8. Добавить автоматическую оценку качества

**В `EvaluationEngine.cs`:**
```csharp
private static double ScoreAnswer(string answer, string[]? keyConcepts)
{
    if (keyConcepts == null || keyConcepts.Length == 0) return 0;
    var lower = answer.ToLowerInvariant();
    var matched = keyConcepts.Count(kc => lower.Contains(kc.ToLowerInvariant()));
    return (double)matched / keyConcepts.Length;
}
```

Выводить score в отчете:
```
Key concepts matched: 6/7 (0.86)
```

### 9. Default mode и help

**В `Program.cs`:**
```csharp
// БЫЛО:
var mode = (args.Length > 0 ? args[0] : "compare").ToLowerInvariant();

// ДОЛЖНО БЫТЬ:
var mode = (args.Length > 0 ? args[0] : "help").ToLowerInvariant();
```

Добавить обработку help:
```csharp
case "help" or "--help" or "-h" or "":
    Console.WriteLine("Usage: dotnet run -- [mode] [rootDir]");
    Console.WriteLine("Modes:");
    Console.WriteLine("  index   - Index documents using FixedSize + Structural strategies");
    Console.WriteLine("  demo    - Interactive semantic search demo");
    Console.WriteLine("  compare - Run RAG vs No-RAG comparison (requires ANTHROPIC_API_KEY)");
    Console.WriteLine("\nEnvironment variables:");
    Console.WriteLine("  ANTHROPIC_API_KEY    - Required for compare mode");
    Console.WriteLine("  ANTHROPIC_MODEL      - Default: claude-3-5-sonnet-20240620");
    Console.WriteLine("  OLLAMA_HOST          - Default: http://localhost:11434");
    Console.WriteLine("  EMBEDDING_MODEL      - Default: nomic-embed-text");
    break;
```

### 10. Проверка существования test-questions.json

**В `Program.cs`, метод `RunCompareAsync`:**
```csharp
var questionsPath = "test-questions.json";
if (!File.Exists(questionsPath))
{
    Console.ForegroundColor = ConsoleColor.Red;
    Console.WriteLine($"❌ Файл {questionsPath} не найден. Убедитесь что он находится в рабочей директории.");
    Console.ResetColor();
    return;
}
```

### 11. Передача OllamaEmbeddingService в EvaluationEngine

**В `Program.cs`, режим `compare`:**
```csharp
case "compare":
    await RunIndexingAsync(store, ollama, rootDir, extensions, chunkSize, overlap);
    var comparator = new StrategyComparator(store);
    await comparator.CompareAndReportAsync();
    await RunCompareAsync(store, ollama, dbPath);  // Передаем ollama!
    break;
```

```csharp
static async Task RunCompareAsync(SqliteVectorStore store, OllamaEmbeddingService ollama, string dbPath)
{
    // ... проверка File.Exists ...
    var engine = new EvaluationEngine(dbPath, ollama);  // Передаем ollama!
    await engine.RunAsync(questions);
}
```

---

## Итоговый чек-лист изменений

### AnthropicLlmService.cs
- [ ] `Model = _model` (убрать hardcoded Claude45Opus)
- [ ] Удалить `_temperature` поле и `Temperature` из MessageParameters

### OllamaEmbeddingService.cs
- [ ] Добавить `GenerateQueryEmbeddingAsync(string query)`

### RagPipeline.cs
- [ ] Использовать `GenerateQueryEmbeddingAsync` вместо `GenerateEmbeddingsAsync` с ручным префиксом
- [ ] Исправить нормализацию путей (`Replace('\\', '/')`, `OrdinalIgnoreCase`)

### EvaluationEngine.cs
- [ ] Конструктор принимает `OllamaEmbeddingService`
- [ ] `PrintSummary`: `OpenAsync`, `ExecuteReaderAsync`, `ReadAsync`
- [ ] Добавить `ScoreAnswer` метод
- [ ] Выводить `key_concepts_score` в отчет

### Models.cs
- [ ] Добавить `KeyConcepts` в `TestQuestion`
- [ ] Удалить `RagContext`

### Program.cs
- [ ] Default mode = `"help"`
- [ ] Добавить case `"help"`
- [ ] Передавать `ollama` в `EvaluationEngine`
- [ ] Проверка `File.Exists("test-questions.json")`

### test-questions.json
- [ ] Добавить `"key_concepts": ["...", "..."]` для каждого из 10 вопросов
```

---

**Ключевое правило:** Все изменения должны быть **минимальными и точечными** — не переписывать файлы целиком, а только исправить указанные строки. Сохранить существующую функциональность индексации и демо-режима.