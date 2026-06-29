================================================================================
ИСПРАВЛЕННЫЙ ПРОМПТ ДЛЯ OpenCode — Пайплайн индексации документов
================================================================================

⚠️  ЭТО ИСПРАВЛЕННАЯ ВЕРСИЯ. Все найденные баги и логические ошибки
    исправлены в спецификации ниже. Реализуй СТРОГО по этому промпту.

ЦЕЛЬ
-----
Реализовать на C# 14 / .NET 10 консольное приложение, которое рекурсивно
обходит папку с текстовыми файлами (.txt, .md, .cs, .json и др.), разбивает
их на чанки ДВУМЯ стратегиями, генерирует эмбеддинги через Ollama
(модель nomic-embed-text) и сохраняет индекс в SQLite с метаданными.

ТЕХНОЛОГИЧЕСКИЙ СТЕК
--------------------
- C# 14 / .NET 10 (top-level statements, primary constructors, collection expressions)
- OllamaSharp 5.4.6+ (NuGet)
- Microsoft.Data.Sqlite 9.0+ (NuGet)
- System.Text.Json

ВАЖНЫЕ ИСПРАВЛЕНИЯ (не повторять ошибки предыдущей версии)
----------------------------------------------------------

### ИСПРАВЛЕНИЕ 1: FixedSizeChunkingStrategy — TotalChunks
ПРЕДЫДУЩАЯ ОШИБКА: формула (words.Length + step - 1) / step давала неверный
результат. Например, 10 слов, chunkSize=5, overlap=2 -> step=3, формула давала 4,
но реально 3 чанка.

ПРАВИЛЬНАЯ РЕАЛИЗАЦИЯ:

```csharp
public class FixedSizeChunkingStrategy(int chunkSize = 512, int overlap = 50) : IChunkingStrategy
{
    public ChunkingStrategy StrategyType => ChunkingStrategy.FixedSize;

    public IEnumerable<DocumentChunk> Chunk(string filePath, string content)
    {
        var words = content.Split([' ', '\t', '\n'], StringSplitOptions.RemoveEmptyEntries);
        if (words.Length == 0) yield break;

        var step = chunkSize - overlap;
        if (step <= 0) step = 1;

        var now = DateTime.UtcNow;
        var title = Path.GetFileName(filePath);
        var index = 0;

        // СНАЧАЛА считаем сколько будет чанков, ПОТОМ генерируем
        int totalChunks;
        if (words.Length <= chunkSize)
            totalChunks = 1;
        else
            totalChunks = 1 + (words.Length - chunkSize + step - 1) / step;

        for (int offset = 0; offset < words.Length; offset += step, index++)
        {
            var count = Math.Min(chunkSize, words.Length - offset);
            var chunkWords = words[offset..(offset + count)];
            var chunkContent = string.Join(' ', chunkWords);

            yield return new DocumentChunk
            {
                ChunkId = Guid.NewGuid().ToString(),
                Source = filePath,
                Title = title,
                Section = null,
                Content = chunkContent,
                ChunkIndex = index,
                TotalChunks = totalChunks,
                Strategy = ChunkingStrategy.FixedSize,
                IndexedAt = now
            };
        }
    }
}
```

### ИСПРАВЛЕНИЕ 2: IndexingPipeline — НЕ использовать ConcurrentBag
ПРЕДЫДУЩАЯ ОШИБКА: ConcurrentBag.ToList() не гарантирует порядок, что ломает
соответствие между chunk и embedding при batch-обработке.

ПРАВИЛЬНАЯ РЕАЛИЗАЦИЯ RunAsync:

```csharp
public async Task<IndexingResult> RunAsync(
    string rootDirectory,
    string[] fileExtensions,
    IProgress<double>? progress = null,
    CancellationToken ct = default)
{
    var sw = Stopwatch.StartNew();

    var files = Directory.EnumerateFiles(rootDirectory, "*.*", SearchOption.AllDirectories)
        .Where(f => fileExtensions.Contains(Path.GetExtension(f), StringComparer.OrdinalIgnoreCase))
        .ToList();

    Console.ForegroundColor = ConsoleColor.Cyan;
    Console.WriteLine($"  Найдено файлов: {files.Count}");
    Console.ResetColor();

    // ИСПРАВЛЕНИЕ: НЕ ConcurrentBag, а обычный List
    // Чтение файлов быстрое, bottleneck — Ollama API вызовы
    var allChunks = new List<DocumentChunk>();
    foreach (var file in files)
    {
        try
        {
            var content = await File.ReadAllTextAsync(file, ct);
            var fileChunks = chunkingStrategy.Chunk(file, content).ToList();
            allChunks.AddRange(fileChunks);
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine($"\n  ⚠️  Пропущен {file}: {ex.Message}");
            Console.ResetColor();
        }
    }

    var total = allChunks.Count;
    if (total == 0)
        return new IndexingResult(files.Count, 0, sw.Elapsed);

    // Батчами отправляем на эмбеддинги
    var batchSize = 10;
    var processed = 0;

    for (int i = 0; i < allChunks.Count; i += batchSize)
    {
        var batch = allChunks.Skip(i).Take(batchSize).ToList();
        var texts = batch.Select(c => c.Content).ToList();
        var embeddings = await embeddingService.GenerateEmbeddingsAsync(texts, ct);

        var indexedBatch = new List<IndexedChunk>();
        for (int j = 0; j < batch.Count; j++)
        {
            indexedBatch.Add(new IndexedChunk
            {
                ChunkId = batch[j].ChunkId,
                Source = batch[j].Source,
                Title = batch[j].Title,
                Section = batch[j].Section,
                Content = batch[j].Content,
                ChunkIndex = batch[j].ChunkIndex,
                TotalChunks = batch[j].TotalChunks,
                Strategy = batch[j].Strategy,
                IndexedAt = batch[j].IndexedAt,
                Embedding = embeddings[j]
            });
        }

        await vectorStore.SaveChunksAsync(indexedBatch);
        processed += batch.Count;
        progress?.Report((double)processed / total);
    }

    sw.Stop();
    return new IndexingResult(files.Count, total, sw.Elapsed);
}
```

### ИСПРАВЛЕНИЕ 3: ChunkMarkdown — заголовки H2-H6
ПРЕДЫДУЩАЯ ОШИБКА: условие trimmed[1] == ' ' ловило ТОЛЬКО H1 (# Header),
для ## Header trimmed[1] == '#', условие false. Заголовки H2-H6 не распознавались.

ПРАВИЛЬНАЯ РЕАЛИЗАЦИЯ:

```csharp
private IEnumerable<DocumentChunk> ChunkMarkdown(string filePath, string content, DateTime now)
{
    var lines = content.Split('\n');
    var sections = new List<(string? section, string content)>();
    string? currentSection = null;
    var currentLines = new List<string>();

    foreach (var line in lines)
    {
        var trimmed = line.TrimStart();
        // ИСПРАВЛЕНИЕ: считаем количество # в начале строки
        var headingLevel = 0;
        while (headingLevel < trimmed.Length && headingLevel < 6 && trimmed[headingLevel] == '#')
            headingLevel++;

        // Заголовок: 1-6 # и затем пробел
        if (headingLevel > 0 && headingLevel < trimmed.Length && trimmed[headingLevel] == ' ')
        {
            if (currentLines.Count > 0)
                sections.Add((currentSection, string.Join('\n', currentLines)));

            currentSection = trimmed[(headingLevel + 1)..].Trim();
            currentLines = [line];
        }
        else
        {
            currentLines.Add(line);
        }
    }

    if (currentLines.Count > 0)
        sections.Add((currentSection, string.Join('\n', currentLines)));

    return ProcessSections(filePath, sections, now);
}
```

### ИСПРАВЛЕНИЕ 4: ChunkCSharp — не терять код между #endregion и #region
ПРЕДЫДУЩАЯ ОШИБКА: после #endregion currentLines = [], и код до следующего
#region терялся.

ПРАВИЛЬНАЯ РЕАЛИЗАЦИЯ:

```csharp
private IEnumerable<DocumentChunk> ChunkCSharp(string filePath, string content, DateTime now)
{
    var lines = content.Split('\n');
    var sections = new List<(string? section, string content)>();
    string? currentRegion = null;
    var currentLines = new List<string>();

    foreach (var line in lines)
    {
        var trimmed = line.Trim();
        if (trimmed.StartsWith("#region "))
        {
            // Сохраняем предыдущий накопленный код как "unregioned" секцию
            if (currentLines.Count > 0)
            {
                var unregioned = string.Join('\n', currentLines).Trim();
                if (!string.IsNullOrEmpty(unregioned))
                    sections.Add((currentRegion ?? "Код вне региона", unregioned));
            }

            currentRegion = trimmed["#region ".Length..].Trim();
            currentLines = [line];
        }
        else if (trimmed == "#endregion")
        {
            currentLines.Add(line);
            var regionContent = string.Join('\n', currentLines).Trim();
            if (!string.IsNullOrEmpty(regionContent))
                sections.Add((currentRegion, regionContent));
            // ИСПРАВЛЕНИЕ: НЕ сбрасываем currentLines = [], продолжаем собирать
            currentRegion = "Код вне региона";
            currentLines = [];
        }
        else
        {
            currentLines.Add(line);
        }
    }

    // Сохраняем оставшийся код
    if (currentLines.Count > 0)
    {
        var remaining = string.Join('\n', currentLines).Trim();
        if (!string.IsNullOrEmpty(remaining))
            sections.Add((currentRegion ?? "Код вне региона", remaining));
    }

    // Если регионов мало, fallback на разбиение по пустым строкам
    if (sections.Count <= 1)
    {
        sections = [];
        var blockLines = new List<string>();
        foreach (var line in lines)
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                if (blockLines.Count > 0)
                {
                    var block = string.Join('\n', blockLines).Trim();
                    if (!string.IsNullOrEmpty(block))
                    {
                        var first = blockLines[0].Trim();
                        var sec = IsCSharpSectionHeader(first) ? first : null;
                        sections.Add((sec, block));
                    }
                    blockLines = [];
                }
            }
            else
            {
                blockLines.Add(line);
            }
        }
        if (blockLines.Count > 0)
        {
            var block = string.Join('\n', blockLines).Trim();
            if (!string.IsNullOrEmpty(block))
            {
                var first = blockLines[0].Trim();
                var sec = IsCSharpSectionHeader(first) ? first : null;
                sections.Add((sec, block));
            }
        }
    }

    return ProcessSections(filePath, sections, now);
}

private static bool IsCSharpSectionHeader(string line)
{
    var trimmed = line.Trim();
    return trimmed.Length <= 100 && (
        trimmed.StartsWith("//") ||
        trimmed.StartsWith("/*") ||
        trimmed.StartsWith("public ") ||
        trimmed.StartsWith("private ") ||
        trimmed.StartsWith("internal ") ||
        trimmed.StartsWith("protected ") ||
        trimmed.StartsWith("class ") ||
        trimmed.StartsWith("interface ") ||
        trimmed.StartsWith("enum ") ||
        trimmed.StartsWith("record ") ||
        trimmed.StartsWith("struct ")
    );
}
```

### ИСПРАВЛЕНИЕ 5: ChunkTxt — нормализация переносов строк
ПРЕДЫДУЩАЯ ОШИБКА: разделители \n\n\n не ловили Windows-переносы \r\n.

ПРАВИЛЬНАЯ РЕАЛИЗАЦИЯ:

```csharp
private IEnumerable<DocumentChunk> ChunkTxt(string filePath, string content, DateTime now)
{
    // ИСПРАВЛЕНИЕ: нормализуем переносы строк
    var normalized = content.Replace("\r\n", "\n");
    var blocks = normalized.Split("\n\n", StringSplitOptions.RemoveEmptyEntries);

    // Если всего 1 блок, пробуем разделить по одиночным пустым строкам
    if (blocks.Length <= 1)
        blocks = normalized.Split('\n', StringSplitOptions.RemoveEmptyEntries);

    var sections = new List<(string? section, string content)>();
    foreach (var block in blocks)
    {
        var trimmed = block.Trim();
        if (string.IsNullOrEmpty(trimmed)) continue;

        var lines = trimmed.Split('\n');
        var firstLine = lines[0].Trim();
        // Считаем section только если первая строка выглядит как заголовок
        var isHeading = firstLine.Length <= 100
            && !firstLine.EndsWith('.')
            && !firstLine.EndsWith(',')
            && char.IsUpper(firstLine[0]);
        var section = isHeading ? firstLine : null;
        sections.Add((section, trimmed));
    }

    return ProcessSections(filePath, sections, now);
}
```

### ИСПРАВЛЕНИЕ 6: GetStatsAsync — правильный SQL
ПРЕДЫДУЩАЯ ОШИБКА: section в SELECT без GROUP BY; два запроса вместо одного.

ПРАВИЛЬНАЯ РЕАЛИЗАЦИЯ:

```csharp
public async Task<ChunkingStats> GetStatsAsync(ChunkingStrategy strategy)
{
    // Запрос 1: основная статистика
    using var conn = new SqliteConnection(ConnectionString);
    await conn.OpenAsync();

    var cmd = conn.CreateCommand();
    cmd.CommandText = """
        SELECT source, COUNT(*) as chunk_count, AVG(LENGTH(content)) as avg_chars
        FROM chunks
        WHERE strategy = $strategy
        GROUP BY source
    """;
    cmd.Parameters.AddWithValue("$strategy", strategy.ToString());

    var totalChunks = 0;
    var totalChars = 0.0;
    var filesWithMultiple = 0;
    var allSections = new HashSet<string>();

    using (var reader = await cmd.ExecuteReaderAsync())
    {
        while (await reader.ReadAsync())
        {
            var count = reader.GetInt32(reader.GetOrdinal("chunk_count"));
            totalChunks += count;
            totalChars += count * reader.GetDouble(reader.GetOrdinal("avg_chars"));
            if (count > 1) filesWithMultiple++;
        }
    }

    // Запрос 2: уникальные секции (только для Structural)
    if (strategy == ChunkingStrategy.Structural)
    {
        var cmd2 = conn.CreateCommand();
        cmd2.CommandText = """
            SELECT DISTINCT section FROM chunks
            WHERE strategy = $strategy AND section IS NOT NULL AND section != ''
        """;
        cmd2.Parameters.AddWithValue("$strategy", strategy.ToString());
        using var reader2 = await cmd2.ExecuteReaderAsync();
        while (await reader2.ReadAsync())
            allSections.Add(reader2.GetString(0));
    }

    var avgChars = totalChunks > 0 ? totalChars / totalChunks : 0;

    // Запрос 3: среднее количество слов (считаем в C# для точности)
    var cmd3 = conn.CreateCommand();
    cmd3.CommandText = "SELECT content FROM chunks WHERE strategy = $strategy";
    cmd3.Parameters.AddWithValue("$strategy", strategy.ToString());

    double totalWords = 0;
    using var reader3 = await cmd3.ExecuteReaderAsync();
    while (await reader3.ReadAsync())
    {
        var content = reader3.GetString(0);
        var wordCount = content.Split([' ', '\t', '\n'], StringSplitOptions.RemoveEmptyEntries).Length;
        totalWords += wordCount;
    }

    var avgWords = totalChunks > 0 ? totalWords / totalChunks : 0;

    return new ChunkingStats(
        totalChunks,
        avgChars,
        avgWords,
        filesWithMultiple,
        strategy == ChunkingStrategy.Structural ? allSections.Count : null
    );
}
```

### ИСПРАВЛЕНИЕ 7: SqliteVectorStore — using для соединений
ПРЕДЫДУДЩАЯ ОШИБКА: нет using для SqliteConnection.

ПРАВИЛЬНАЯ РЕАЛИЗАЦИЯ: во ВСЕХ методах SqliteVectorStore использовать:
using var conn = new SqliteConnection(ConnectionString);
await conn.OpenAsync();
// ... работа с conn
// conn.Dispose() вызовется автоматически

### ИСПРАВЛЕНИЕ 8: ChunkSections (fallback) — обновлять TotalChunks
ПРЕДЫДУЩАЯ ОШИБКА: TotalChunks оставался от fallback.Chunk().

ПРАВИЛЬНАЯ РЕАЛИЗАЦИЯ:

```csharp
private IEnumerable<DocumentChunk> ChunkSections(string filePath, string content, FixedSizeChunkingStrategy fallback, DateTime now)
{
    var fallbackChunks = fallback.Chunk(filePath, content).ToList();
    var total = fallbackChunks.Count;
    for (int i = 0; i < fallbackChunks.Count; i++)
    {
        yield return fallbackChunks[i] with
        {
            Strategy = ChunkingStrategy.Structural,
            ChunkIndex = i,
            TotalChunks = total  // ИСПРАВЛЕНИЕ: явно устанавливаем TotalChunks
        };
    }
}
```

### ИСПРАВЛЕНИЕ 9: OllamaEmbeddingService — убрать бесполезный catch

```csharp
public async Task<IEnumerable<Model>> CheckAvailabilityAsync()
{
    return await _client.ListLocalModelsAsync();
}
```

### ИСПРАВЛЕНИЕ 10: OllamaEmbeddingService — batch embeddings
ВАЖНО: согласно исследованиям, batch-вызовы Ollama могут давать слегка
отличающиеся результаты по сравнению с single-вызовами.
Для баланса скорости/точности используем batch size = 10.

Полный код OllamaEmbeddingService:

```csharp
public class OllamaEmbeddingService(string host = "http://localhost:11434", string model = "nomic-embed-text") : IEmbeddingService
{
    private readonly OllamaApiClient _client = new(new Uri(host)) { SelectedModel = model };

    public async Task<IEnumerable<Model>> CheckAvailabilityAsync()
    {
        return await _client.ListLocalModelsAsync();
    }

    public async Task<float[][]> GenerateEmbeddingsAsync(IEnumerable<string> texts, CancellationToken ct = default)
    {
        var textList = texts.ToList();
        if (textList.Count == 0) return [];

        var batches = new List<List<string>>();
        var batch = new List<string>();
        foreach (var t in textList)
        {
            batch.Add(t);
            if (batch.Count >= 10)
            {
                batches.Add([.. batch]);
                batch = [];
            }
        }
        if (batch.Count > 0) batches.Add([.. batch]);

        var results = new float[textList.Count][];
        var index = 0;

        foreach (var b in batches)
        {
            var prefixedTexts = b.Select(t => $"search_document: {t}").ToList();
            var embeddings = await EmbedWithRetryAsync(prefixedTexts, ct);
            foreach (var emb in embeddings)
            {
                if (emb.Length != 768)
                    throw new InvalidOperationException($"Expected 768-dimensional embedding, got {emb.Length}");
                results[index++] = emb;
            }
        }

        return results;
    }

    private async Task<List<float[]>> EmbedWithRetryAsync(List<string> texts, CancellationToken ct)
    {
        var maxRetries = 3;
        var delay = 1000;

        for (int attempt = 1; attempt <= maxRetries; attempt++)
        {
            try
            {
                var request = new EmbedRequest
                {
                    Input = texts,
                    Model = _client.SelectedModel
                };
                var response = await _client.EmbedAsync(request, ct);
                return response.Embeddings;
            }
            catch (HttpRequestException) when (attempt < maxRetries)
            {
                await Task.Delay(delay * attempt, ct);
            }
        }

        throw new HttpRequestException("Failed to generate embeddings after 3 retries");
    }
}
```

ПОЛНЫЙ КОД ВСЕХ ОСТАЛЬНЫХ КОМПОНЕНТОВ
----------------------------------------

Модели данных, VectorMath, интерфейсы IChunkingStrategy/IEmbeddingService/IVectorStore,
SqliteVectorStore (кроме GetStatsAsync, см. ИСПРАВЛЕНИЕ 6), IndexingPipeline
(кроме RunAsync, см. ИСПРАВЛЕНИЕ 2), StrategyComparator, Program.cs —
остаются как в предыдущей версии, с учётом:
1. ВСЕ using var conn = new SqliteConnection(...) в SqliteVectorStore
2. ChunkMarkdown по ИСПРАВЛЕНИЮ 3
3. ChunkCSharp по ИСПРАВЛЕНИЮ 4
4. ChunkTxt по ИСПРАВЛЕНИЮ 5
5. ChunkSections по ИСПРАВЛЕНИЮ 8
6. FixedSizeChunkingStrategy по ИСПРАВЛЕНИЮ 1
7. IndexingPipeline.RunAsync по ИСПРАВЛЕНИЮ 2

ТРЕБОВАНИЯ К КОДУ
-----------------
1. C# 14 фичи: primary constructors, collection expressions [], required init-only
2. Обработка ошибок: try/catch при чтении файлов, retry при Ollama
3. Производительность: batching эмбеддингов, SQLite transaction
4. Цветной вывод консоли
5. Проверка доступности Ollama перед стартом

ПРИМЕР ЗАПУСКА
--------------
dotnet new console -n DocIndexer
cd DocIndexer
dotnet add package OllamaSharp
dotnet add package Microsoft.Data.Sqlite
dotnet run -- ./test-files
