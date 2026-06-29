Вот подробный промпт для OpenCode, подготовленный последовательно на основе анализа требований и актуальных технологий.

---

## 🔬 Последовательное рассуждение

**Шаг 1 — Анализ стека.** Для C# 14 / .NET 10 актуальны: top-level statements, primary constructors, collection expressions `[]`, `required` init-only свойства. Для работы с Ollama лучший выбор — **OllamaSharp** (рекомендован Microsoft, поддерживает .NET 10, реализует `IEmbeddingGenerator`) . Для SQLite — **Microsoft.Data.Sqlite** + хранение эмбеддингов как JSON/BLOB. Расширение `sqlite-vec` доступно, но для простоты и полного контроля лучше brute-force cosine similarity в памяти .

**Шаг 2 — Ollama + nomic-embed-text.** Модель `nomic-embed-text` v1.5 выдаёт 768-мерные векторы. Важный нюанс: для документов нужен префикс `search_document:`, для запросов — `search_query:` — без них качество retrieval падает на 5-10 пунктов . Актуальный эндпоинт — `/api/embed` (batch input), старый `/api/embeddings` deprecated . Рекомендуется `num_ctx: 8192` для полного контекста.

**Шаг 3 — Chunking.** Две стратегии:
- **FixedSize** — разбиение по словам с перекрытием (overlap). Просто, предсказуемо, но может резать смысловые границы.
- **Structural** — разбиение по заголовкам Markdown (`# ## ###`), пустым строкам в `.txt`, `#region` в `.cs`. Сохраняет семантическую целостность, но требует парсинга формата.

**Шаг 4 — Метаданные.** Каждый чанк должен содержать: `source` (путь), `title` (имя файла), `section` (заголовок раздела), `chunk_id` (UUID), `chunk_index`, `total_chunks`, `strategy`, `indexed_at`.

**Шаг 5 — SQLite схема.** Таблица `chunks` с полями для метаданных и BLOB/JSON для эмбеддингов. Индексы по `strategy` и `source`.

---

## 📋 Промпт для OpenCode

```
================================================================================
ПРОМПТ: Пайплайн индексации документов с эмбеддингами (C# 14 / .NET 10)
================================================================================

ЦЕЛЬ
----
Реализовать консольное приложение на C# 14 / .NET 10, которое рекурсивно 
обходит папку с текстовыми файлами, разбивает их на чанки двумя стратегиями, 
генерирует эмбеддинги через Ollama (nomic-embed-text) и сохраняет индекс 
в SQLite с богатыми метаданными. Приложение сравнивает стратегии chunking'а 
и демонстрирует семантический поиск.

ТЕХНОЛОГИЧЕСКИЙ СТЕК
--------------------
- C# 14 / .NET 10 (top-level statements, primary constructors, collection expressions)
- OllamaSharp 5.4.6+ (NuGet) — официальный .NET SDK для Ollama
- Microsoft.Data.Sqlite 9.0+ (NuGet) — SQLite драйвер
- System.Text.Json — встроенная сериализация

ПРЕДВАРИТЕЛЬНЫЕ ТРЕБОВАНИЯ
----------------------------
Ollama должен быть запущен локально, модель nomic-embed-text должна быть 
скачана: `ollama pull nomic-embed-text`

АРХИТЕКТУРА (реализовать ВСЕ компоненты)
-----------------------------------------

1. МОДЕЛИ ДАННЫХ

```csharp
public enum ChunkingStrategy { FixedSize, Structural }

public record DocumentChunk
{
    public required string ChunkId { get; init; }      // Guid.NewGuid().ToString()
    public required string Source { get; init; }       // абсолютный путь к файлу
    public required string Title { get; init; }        // Path.GetFileName(Source)
    public required string? Section { get; init; }     // заголовок раздела (null для FixedSize)
    public required string Content { get; init; }      // текст чанка
    public required int ChunkIndex { get; init; }      // 0-based порядковый номер
    public required int TotalChunks { get; init; }     // общее количество чанков в файле
    public required ChunkingStrategy Strategy { get; init; }
    public required DateTime IndexedAt { get; init; }
}

public record IndexedChunk : DocumentChunk
{
    public required float[] Embedding { get; init; }   // 768 float для nomic-embed-text
}
```

2. ИНТЕРФЕЙСЫ СТРАТЕГИЙ ЧАНКИНГА

```csharp
public interface IChunkingStrategy
{
    ChunkingStrategy StrategyType { get; }
    IEnumerable<DocumentChunk> Chunk(string filePath, string content);
}

// === Стратегия 1: Фиксированный размер с перекрытием ===
public class FixedSizeChunkingStrategy(int chunkSize = 512, int overlap = 50) : IChunkingStrategy
{
    public ChunkingStrategy StrategyType => ChunkingStrategy.FixedSize;
    
    // Разбивает текст на чанки фиксированного размера ПО СЛОВАМ.
    // chunkSize — максимальное количество слов в чанке.
    // overlap — количество слов перекрытия между соседними чанками.
    // Алгоритм:
    //   1. Разбить content на слова: content.Split([' ', '\t', '\n'], StringSplitOptions.RemoveEmptyEntries)
    //   2. Идти окном: берём chunkSize слов, следующий чанк начинается с (chunkSize - overlap) слов
    //   3. Собрать слова обратно в строку через string.Join(' ', words)
    //   4. Для каждого чанка: ChunkId = Guid.NewGuid(), Section = null
    //   5. TotalChunks вычислить заранее
}

// === Стратегия 2: Структурный chunking ===
public class StructuralChunkingStrategy(int maxChunkSize = 512, int overlap = 50) : IChunkingStrategy
{
    public ChunkingStrategy StrategyType => ChunkingStrategy.Structural;
    
    // Разбивает текст по структурным элементам:
    // - .md файлы: по заголовкам (# ## ###). Заголовок + всё до следующего заголовка = 1 чанк.
    //   Section = текст заголовка (без #).
    // - .txt файлы: по двойным пустым строкам (\n\n\n+). Каждый блок = 1 чанк.
    //   Section = первая строка блока (если <= 100 символов) или null.
    // - .cs файлы: по #region / #endregion или по пустым строкам между методами/классами.
    //   Section = имя региона или "Class.Method" (первые 100 символов).
    // - Другие форматы: fallback на FixedSizeChunkingStrategy.
    // 
    // Если секция длиннее maxChunkSize слов — применить внутри FixedSizeChunkingStrategy
    // к содержимому секции, сохраняя Section в каждом под-чанке.
}
```

3. СЕРВИС ЭМБЕДДИНГОВ

```csharp
public interface IEmbeddingService
{
    Task<float[][]> GenerateEmbeddingsAsync(IEnumerable<string> texts, CancellationToken ct = default);
}

public class OllamaEmbeddingService : IEmbeddingService
{
    // Конфигурация через переменные окружения с fallback:
    //   OLLAMA_HOST  → default "http://localhost:11434"
    //   EMBEDDING_MODEL → default "nomic-embed-text"
    
    // Реализация:
    // 1. Создать OllamaApiClient(new Uri(baseUrl))
    // 2. Для КАЖДОГО текста добавить префикс: $"search_document: {text}"
    //    (это КРИТИЧЕСКИ важно для nomic-embed-text v1.5!)
    // 3. Отправлять батчами по 10-15 текстов за раз через client.EmbedAsync()
    //    или client.GenerateEmbeddingAsync() — используй актуальный метод OllamaSharp 5.4+
    // 4. Retry с exponential backoff: 3 попытки при HttpRequestException
    // 5. Проверить что Ollama доступен перед началом (ListLocalModels)
    // 6. Вернуть float[][] — массив эмбеддингов (каждый float[768])
}
```

4. ХРАНИЛИЩЕ SQLite

```csharp
public interface IVectorStore
{
    Task InitializeAsync();
    Task SaveChunksAsync(IEnumerable<IndexedChunk> chunks);
    Task<IEnumerable<IndexedChunk>> SearchSimilarAsync(float[] queryEmbedding, int topK = 5, ChunkingStrategy? strategy = null);
    Task<long> GetChunkCountAsync(ChunkingStrategy? strategy = null);
    Task ClearIndexAsync(ChunkingStrategy? strategy = null);
    Task<ChunkingStats> GetStatsAsync(ChunkingStrategy strategy);
}

public record ChunkingStats(
    int ChunkCount,
    double AvgChunkLengthChars,
    double AvgChunkLengthWords,
    int FilesWithMultipleChunks,
    int? SectionsDetected
);

public class SqliteVectorStore(string dbPath = "index.db") : IVectorStore
{
    // СХЕМА БД:
    // CREATE TABLE IF NOT EXISTS chunks (
    //     chunk_id TEXT PRIMARY KEY,
    //     source TEXT NOT NULL,
    //     title TEXT NOT NULL,
    //     section TEXT,
    //     content TEXT NOT NULL,
    //     chunk_index INTEGER NOT NULL,
    //     total_chunks INTEGER NOT NULL,
    //     strategy TEXT NOT NULL,
    //     indexed_at TEXT NOT NULL,
    //     embedding TEXT NOT NULL  -- JSON массив float[768]
    // );
    // CREATE INDEX IF NOT EXISTS idx_chunks_strategy ON chunks(strategy);
    // CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
    
    // SaveChunksAsync:
    //   - Использовать SQLite transaction для bulk insert
    //   - Сериализовать embedding через JsonSerializer.Serialize(chunk.Embedding)
    //   - INSERT OR REPLACE (на случай повторной индексации)
    
    // SearchSimilarAsync:
    //   - Загрузить ВСЕ чанки (или фильтр по strategy) из БД
    //   - Десериализовать embedding из JSON
    //   - Вычислить cosine similarity: dot(a,b) / (||a|| * ||b||)
    //   - Отсортировать по убыванию similarity, взять topK
    //   - Вернуть IEnumerable<IndexedChunk>
    
    // GetStatsAsync:
    //   - SELECT COUNT(*), AVG(LENGTH(content)), ... GROUP BY не нужен, 
    //     фильтруй в C# по strategy
    //   - Для Structural: посчитать уникальные non-null section
}
```

5. КОСИНУСНОЕ СХОДСТВО (утилита)

```csharp
public static class VectorMath
{
    public static float CosineSimilarity(float[] a, float[] b)
    {
        // dot(a,b) / (sqrt(dot(a,a)) * sqrt(dot(b,b)))
        // Проверить что длины массивов равны (768)
        // Обработать деление на ноль
    }
}
```

6. ПАЙПЛАЙН ИНДЕКСАЦИИ

```csharp
public class IndexingPipeline(
    IChunkingStrategy chunkingStrategy,
    IEmbeddingService embeddingService,
    IVectorStore vectorStore)
{
    public async Task<IndexingResult> RunAsync(
        string rootDirectory, 
        string[] fileExtensions, 
        IProgress<double>? progress = null,
        CancellationToken ct = default)
    {
        // 1. Найти файлы:
        //    var files = Directory.EnumerateFiles(rootDirectory, "*.*", SearchOption.AllDirectories)
        //        .Where(f => fileExtensions.Contains(Path.GetExtension(f), StringComparer.OrdinalIgnoreCase))
        //        .ToList();
        //
        // 2. Для каждого файла:
        //    - try/catch при чтении (пропускать недоступные/бинарные)
        //    - File.ReadAllTextAsync
        //    - chunkingStrategy.Chunk(filePath, content) → List<DocumentChunk>
        //
        // 3. Собрать ВСЕ чанки всех файлов в один список
        //
        // 4. Батчами по 10-15 отправить на embeddingService.GenerateEmbeddingsAsync
        //    - progress?.Report(current / total)
        //
        // 5. Собрать IndexedChunk (DocumentChunk + Embedding[i])
        //
        // 6. Сохранить в vectorStore.SaveChunksAsync
        //
        // 7. Вернуть статистику: количество файлов, чанков, время
        
        return new IndexingResult(files.Count, chunks.Count, sw.Elapsed);
    }
}

public record IndexingResult(int FilesProcessed, int ChunksCreated, TimeSpan Duration);
```

7. СРАВНЕНИЕ СТРАТЕГИЙ

```csharp
public class StrategyComparator(IVectorStore vectorStore)
{
    public async Task<ComparisonReport> CompareAndReportAsync()
    {
        // Получить stats для обеих стратегий через vectorStore.GetStatsAsync
        // Вывести красивую таблицу в консоль с ANSI-цветами:
        //
        // ╔══════════════════════════════════════════════════════════════╗
        // ║           СРАВНЕНИЕ СТРАТЕГИЙ CHUNKING'А                   ║
        // ╠════════════════════╦══════════════════╦════════════════════╣
        // ║ Метрика            ║ FixedSize        ║ Structural         ║
        // ╠════════════════════╬══════════════════╬════════════════════╣
        // ║ Чанков             ║ 1,234            ║ 892                ║
        // ║ Средний размер     ║ 487 симв         ║ 1,203 симв         ║
        // ║ Средний размер     ║ 82 слова         ║ 156 слов           ║
        // ║ Файлов >1 чанка    ║ 45               ║ 23                 ║
        // ║ Секций распознано  ║ N/A              ║ 67                 ║
        // ╚════════════════════╩══════════════════╩════════════════════╝
        //
        // Вывести вывод: какая стратегия лучше для каких типов документов.
    }
}
```

8. ТОЧКА ВХОДА (Program.cs — top-level statements)

```csharp
using System;
using System.Diagnostics;
using System.IO;
// ... остальные using

// === КОНФИГУРАЦИЯ ===
var rootDir = args.Length > 0 ? args[0] : "./test-files";
var extensions = new[] { ".txt", ".md", ".cs", ".json", ".xml", ".yaml", ".yml", ".html", ".js", ".py" };

// === ИНИЦИАЛИЗАЦИЯ ===
var store = new SqliteVectorStore("document_index.db");
await store.InitializeAsync();

var ollama = new OllamaEmbeddingService();

// Проверить доступность Ollama
Console.WriteLine("Проверка Ollama...");
try { /* ListLocalModels */ }
catch { Console.WriteLine("❌ Ollama недоступен. Запустите: ollama serve"); return; }

// === ИНДЕКСАЦИЯ: Fixed Size ===
Console.WriteLine("\n📦 Индексация: Fixed Size Strategy (chunk=512, overlap=50)");
var progress1 = new Progress<double>(p => Console.Write($"\r  Прогресс: {p:P0}"));
var pipeline1 = new IndexingPipeline(new FixedSizeChunkingStrategy(512, 50), ollama, store);
var result1 = await pipeline1.RunAsync(rootDir, extensions, progress1);
Console.WriteLine($"\n✅ Готово: {result1.ChunksCreated} чанков из {result1.FilesProcessed} файлов за {result1.Duration.TotalSeconds:F1}с");

// === ИНДЕКСАЦИЯ: Structural ===
Console.WriteLine("\n📐 Индексация: Structural Strategy");
var progress2 = new Progress<double>(p => Console.Write($"\r  Прогресс: {p:P0}"));
var pipeline2 = new IndexingPipeline(new StructuralChunkingStrategy(512, 50), ollama, store);
var result2 = await pipeline2.RunAsync(rootDir, extensions, progress2);
Console.WriteLine($"\n✅ Готово: {result2.ChunksCreated} чанков из {result2.FilesProcessed} файлов за {result2.Duration.TotalSeconds:F1}с");

// === СРАВНЕНИЕ ===
var comparator = new StrategyComparator(store);
await comparator.CompareAndReportAsync();

// === ДЕМО ПОИСКА ===
Console.WriteLine("\n🔍 ДЕМО СЕМАНТИЧЕСКОГО ПОИСКА");
while (true)
{
    Console.Write("\nВведите запрос (или 'exit'): ");
    var query = Console.ReadLine();
    if (string.IsNullOrWhiteSpace(query) || query == "exit") break;
    
    // Добавить префикс search_query: для nomic-embed-text!
    var queryEmbedding = (await ollama.GenerateEmbeddingsAsync([$"search_query: {query}"]))[0];
    
    Console.WriteLine("\n--- Fixed Size результаты ---");
    var fixedResults = await store.SearchSimilarAsync(queryEmbedding, 3, ChunkingStrategy.FixedSize);
    foreach (var r in fixedResults)
        PrintResult(r);
    
    Console.WriteLine("\n--- Structural результаты ---");
    var structResults = await store.SearchSimilarAsync(queryEmbedding, 3, ChunkingStrategy.Structural);
    foreach (var r in structResults)
        PrintResult(r);
}

static void PrintResult(IndexedChunk r)
{
    var preview = r.Content.Length > 150 ? r.Content[..150] + "..." : r.Content;
    Console.WriteLine($"  [{r.Strategy}] {r.Title} | {r.Section ?? "N/A"} | chunk {r.ChunkIndex + 1}/{r.TotalChunks}");
    Console.WriteLine($"    {preview.Replace('\n', ' ')}");
}
```

ТРЕБОВАНИЯ К РЕАЛИЗАЦИИ
------------------------
1. ВСЕ классы должны быть в одном файле Program.cs (или логически разбиты 
   на partial Program.cs)
2. Использовать C# 14 фичи:
   - Primary constructors для сервисов (уже показаны в сигнатурах)
   - Collection expressions: `var list = [];` вместо `new List<T>()`
   - `required` init-only properties в record'ах
   - Pattern matching где уместно
3. Обработка ошибок:
   - try/catch при File.ReadAllTextAsync (пропускать, не падать)
   - Retry при Ollama (3 попытки, exponential backoff)
   - Проверка размерности эмбеддингов (должно быть 768)
4. Производительность:
   - Асинхронные операции везде где возможно
   - SQLite transaction для bulk insert
   - Batch размер 10-15 для Ollama
5. Цветной вывод консоли:
   - 🟢 Зеленый: успешные операции
   - 🔴 Красный: ошибки
   - 🟡 Желтый: предупреждения
   - 🔵 Синий: информация

ДОПОЛНИТЕЛЬНО (если хватит контекста)
--------------------------------------
- Добавить аргументы командной строки через System.CommandLine:
  `--dir`, `--strategy`, `--chunk-size`, `--overlap`, `--clear`
- Добавить экспорт статистики в JSON файл
- Добавить поддержку sqlite-vec расширения для нативного vector search 
  (вместо brute-force в памяти)

ПРИМЕР ЗАПУСКА
--------------
```bash
# 1. Создать проект
dotnet new console -n DocIndexer
cd DocIndexer

# 2. Добавить пакеты
dotnet add package OllamaSharp
dotnet add package Microsoft.Data.Sqlite

# 3. Создать тестовые файлы
mkdir test-files
cat > test-files/readme.md << 'EOF'
# Введение
Это документация проекта. Он написан на C#.

# Архитектура
Проект использует чистую архитектуру с разделением на слои.

# API
REST API предоставляет endpoints для работы с данными.
EOF

cat > test-files/config.txt << 'EOF'
Database=localhost
Port=5432
Username=admin

CacheEnabled=true
CacheTTL=3600
EOF

# 4. Запуск
dotnet run -- ./test-files
```

ОЖИДАЕМЫЙ ВЫВОД
---------------
Приложение должно:
1. Обработать все файлы в ./test-files рекурсивно
2. Создать чанки обеими стратегиями
3. Сгенерировать эмбеддинги через Ollama
4. Сохранить в SQLite с полными метаданными
5. Вывести сравнительную таблицу
6. Позволить интерактивный семантический поиск
```

---

### Почему именно такая структура промпта

| Аспект | Решение | Обоснование |
|--------|---------|-------------|
| **Ollama SDK** | OllamaSharp | Рекомендован Microsoft, поддерживает .NET 10, `IEmbeddingGenerator`  |
| **Префиксы** | `search_document:` / `search_query:` | Критичны для качества nomic-embed-text v1.5  |
| **SQLite** | Microsoft.Data.Sqlite + JSON | Простота, переносимость, не требует нативных расширений  |
| **Batch size** | 10-15 чанков | Баланс между нагрузкой на Ollama и скоростью |
| **Chunking** | Слова, не символы | Сохраняет целостность слов, более предсказуемо |
| **C# 14** | Primary constructors, `[]`, `required` | Современный идиоматичный код |