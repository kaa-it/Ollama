using System.Diagnostics;
using System.Text.Json;
using Microsoft.Data.Sqlite;
using OllamaSharp;
using OllamaSharp.Models;

Console.OutputEncoding = System.Text.Encoding.UTF8;

var mode = (args.Length > 0 ? args[0] : "help").ToLowerInvariant();
var rootDir = args.Length > 1 ? args[1] : "../../../patterns";
var extensions = new[] { ".txt", ".md", ".cs", ".json", ".xml", ".yaml", ".yml", ".html", ".js", ".py" };
var ollamaHost = Environment.GetEnvironmentVariable("OLLAMA_HOST") ?? "http://localhost:11434";
var embeddingModel = Environment.GetEnvironmentVariable("EMBEDDING_MODEL") ?? "nomic-embed-text";
var dbPath = Environment.GetEnvironmentVariable("DB_PATH") ?? "document_index.db";
var chunkSize = int.TryParse(Environment.GetEnvironmentVariable("CHUNK_SIZE"), out var cs) ? cs : 512;
var overlap = int.TryParse(Environment.GetEnvironmentVariable("OVERLAP"), out var ov) ? ov : 50;

var store = new SqliteVectorStore(dbPath);
await store.InitializeAsync();
var ollama = new OllamaEmbeddingService(ollamaHost, embeddingModel);

Console.ForegroundColor = ConsoleColor.Blue;
Console.WriteLine("Проверка Ollama...");
Console.ResetColor();

try
{
    var models = await ollama.CheckAvailabilityAsync();
    if (!models.Any(m => m.Name.Contains("nomic-embed-text", StringComparison.OrdinalIgnoreCase)))
    {
        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine("⚠️  Модель nomic-embed-text не найдена. Выполните: ollama pull nomic-embed-text");
        Console.ResetColor();
    }
}
catch (Exception ex)
{
    Console.ForegroundColor = ConsoleColor.Red;
    Console.WriteLine($"❌ Ollama недоступен: {ex.Message}. Запустите: ollama serve");
    Console.ResetColor();
    return;
}

Console.ForegroundColor = ConsoleColor.Green;
Console.WriteLine("\n✅ Ollama доступен");
Console.ResetColor();

switch (mode)
{
    case "index":
        await RunIndexingAsync(store, ollama, rootDir, extensions, chunkSize, overlap);
        break;
    case "demo":
        await RunDemoAsync(store, ollama);
        break;
    case "compare":
        await RunIndexingAsync(store, ollama, rootDir, extensions, chunkSize, overlap);
        var comparator = new StrategyComparator(store);
        await comparator.CompareAndReportAsync();
        await RunCompareAsync(ollama, dbPath);
        break;
    case "help" or "--help" or "-h":
        Console.WriteLine("Usage: dotnet run -- [mode] [rootDir]");
        Console.WriteLine("Modes:");
        Console.WriteLine("  index   - Index documents using FixedSize + Structural strategies");
        Console.WriteLine("  demo    - Interactive semantic search demo");
        Console.WriteLine("  compare - Run RAG vs No-RAG comparison (requires ANTHROPIC_API_KEY)");
        Console.WriteLine("\nEnvironment variables:");
        Console.WriteLine("  ANTHROPIC_API_KEY       - Required for compare mode");
        Console.WriteLine("  ANTHROPIC_MODEL         - Default: claude-3-5-sonnet-20240620");
        Console.WriteLine("  OLLAMA_HOST             - Default: http://localhost:11434");
        Console.WriteLine("  EMBEDDING_MODEL         - Default: nomic-embed-text");
        Console.WriteLine("  RAG_TOP_K_PRE=10        - Pre-filter K (vector search)");
        Console.WriteLine("  RAG_TOP_K_POST=3        - Post-filter K (sent to LLM)");
        Console.WriteLine("  RAG_SIMILARITY_THRESHOLD=0.5 - Min similarity for filter");
        Console.WriteLine("  RAG_ENABLE_REWRITE=true - Enable query rewrite");
        Console.WriteLine("  RAG_ENABLE_RERANK=true  - Enable heuristic reranker");
        Console.WriteLine("  RAG_USE_LLM_REWRITE=false - Use LLM for rewrite (vs heuristic)");
        break;
    default:
        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine("Неизвестный режим. Используйте: index, demo, или compare");
        Console.ResetColor();
        break;
}

static async Task RunIndexingAsync(SqliteVectorStore store, OllamaEmbeddingService ollama, string rootDir, string[] extensions, int chunkSize, int overlap)
{
    Console.ForegroundColor = ConsoleColor.Blue;
    Console.WriteLine("\n📦 Индексация: Fixed Size Strategy (chunk=512, overlap=50)");
    Console.ResetColor();

    var progress1 = new Progress<double>(p =>
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.Write($"\r  Прогресс: {p:P0}");
        Console.ResetColor();
    });
    var pipeline1 = new IndexingPipeline(new FixedSizeChunkingStrategy(chunkSize, overlap), ollama, store);
    var result1 = await pipeline1.RunAsync(rootDir, extensions, progress1);
    Console.ForegroundColor = ConsoleColor.Green;
    Console.WriteLine($"\n✅ Готово: {result1.ChunksCreated} чанков из {result1.FilesProcessed} файлов за {result1.Duration.TotalSeconds:F1}с");
    Console.ResetColor();

    Console.ForegroundColor = ConsoleColor.Blue;
    Console.WriteLine("\n📐 Индексация: Structural Strategy");
    Console.ResetColor();

    var progress2 = new Progress<double>(p =>
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.Write($"\r  Прогресс: {p:P0}");
        Console.ResetColor();
    });
    var pipeline2 = new IndexingPipeline(new StructuralChunkingStrategy(chunkSize, overlap), ollama, store);
    var result2 = await pipeline2.RunAsync(rootDir, extensions, progress2);
    Console.ForegroundColor = ConsoleColor.Green;
    Console.WriteLine($"\n✅ Готово: {result2.ChunksCreated} чанков из {result2.FilesProcessed} файлов за {result2.Duration.TotalSeconds:F1}с");
    Console.ResetColor();
}

static async Task RunDemoAsync(SqliteVectorStore store, OllamaEmbeddingService ollama)
{
    Console.ForegroundColor = ConsoleColor.Blue;
    Console.WriteLine("\n🔍 ДЕМО СЕМАНТИЧЕСКОГО ПОИСКА");
    Console.ResetColor();

    while (true)
    {
        Console.Write("\nВведите запрос (или 'exit'): ");
        var query = Console.ReadLine();
        if (string.IsNullOrWhiteSpace(query) || query == "exit") break;

        try
        {
            var queryEmbedding = await ollama.GenerateQueryEmbeddingAsync(query);

            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("\n--- Fixed Size результаты ---");
            Console.ResetColor();
            var fixedResults = await store.SearchSimilarAsync(queryEmbedding, 3, ChunkingStrategy.FixedSize);
            foreach (var r in fixedResults)
                PrintResult(r);

            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("\n--- Structural результаты ---");
            Console.ResetColor();
            var structResults = await store.SearchSimilarAsync(queryEmbedding, 3, ChunkingStrategy.Structural);
            foreach (var r in structResults)
                PrintResult(r);
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"❌ Ошибка поиска: {ex.Message}");
            Console.ResetColor();
        }
    }
}

static async Task RunCompareAsync(OllamaEmbeddingService ollama, string dbPath)
{
    var questionsPath = "test-questions.json";
    if (!File.Exists(questionsPath))
    {
        Console.ForegroundColor = ConsoleColor.Red;
        Console.WriteLine($"❌ Файл {questionsPath} не найден. Убедитесь что он находится в рабочей директории.");
        Console.ResetColor();
        return;
    }

    var questionsJson = await File.ReadAllTextAsync(questionsPath);
    var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
    var questions = JsonSerializer.Deserialize<List<TestQuestion>>(questionsJson, options) ?? [];

    var engine = new EvaluationEngine(dbPath, ollama);
    await engine.RunAsync(questions);
}

static void PrintResult(IndexedChunk r)
{
    var preview = r.Content.Length > 150 ? r.Content[..150] + "..." : r.Content;
    Console.ForegroundColor = ConsoleColor.Magenta;
    Console.Write($"  [{r.Strategy}] ");
    Console.ResetColor();
    Console.WriteLine($"{r.Title} | {r.Section ?? "N/A"} | chunk {r.ChunkIndex + 1}/{r.TotalChunks}");
    Console.WriteLine($"    {preview.Replace('\n', ' ')}");
}

// ============================================================
// MODELS
// ============================================================

public enum ChunkingStrategy { FixedSize, Structural }

public record DocumentChunk
{
    public required string ChunkId { get; init; }
    public required string Source { get; init; }
    public required string Title { get; init; }
    public required string? Section { get; init; }
    public required string Content { get; init; }
    public required int ChunkIndex { get; init; }
    public required int TotalChunks { get; init; }
    public required ChunkingStrategy Strategy { get; init; }
    public required DateTime IndexedAt { get; init; }
}

public record IndexedChunk : DocumentChunk
{
    public required float[] Embedding { get; init; }
}

public record IndexingResult(int FilesProcessed, int ChunksCreated, TimeSpan Duration);

public record ChunkingStats(
    int ChunkCount,
    double AvgChunkLengthChars,
    double AvgChunkLengthWords,
    int FilesWithMultipleChunks,
    int? SectionsDetected
);

// ============================================================
// VECTOR MATH
// ============================================================

public static class VectorMath
{
    public static float CosineSimilarity(float[] a, float[] b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException($"Embedding dimensions mismatch: {a.Length} vs {b.Length}");

        float dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        var magnitude = MathF.Sqrt(normA) * MathF.Sqrt(normB);
        return magnitude == 0 ? 0 : dot / magnitude;
    }
}

// ============================================================
// CHUNKING STRATEGIES
// ============================================================

public interface IChunkingStrategy
{
    ChunkingStrategy StrategyType { get; }
    IEnumerable<DocumentChunk> Chunk(string filePath, string content);
}

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

        int totalChunks;
        if (words.Length <= chunkSize)
            totalChunks = 1;
        else
            totalChunks = 1 + (words.Length - chunkSize + step - 1) / step;

        for (int offset = 0; offset < words.Length; offset += step, index++)
        {
            var count = Math.Min(chunkSize, words.Length - offset);
            var chunkWords = words[offset..(offset + count)];
            var content_ = string.Join(' ', chunkWords);

            yield return new DocumentChunk
            {
                ChunkId = Guid.NewGuid().ToString(),
                Source = filePath,
                Title = title,
                Section = null,
                Content = content_,
                ChunkIndex = index,
                TotalChunks = totalChunks,
                Strategy = ChunkingStrategy.FixedSize,
                IndexedAt = now
            };
        }
    }
}

public class StructuralChunkingStrategy(int maxChunkSize = 512, int overlap = 50) : IChunkingStrategy
{
    public ChunkingStrategy StrategyType => ChunkingStrategy.Structural;

    public IEnumerable<DocumentChunk> Chunk(string filePath, string content)
    {
        var ext = Path.GetExtension(filePath).ToLowerInvariant();
        var fallback = new FixedSizeChunkingStrategy(maxChunkSize, overlap);
        var now = DateTime.UtcNow;

        return ext switch
        {
            ".md" => ChunkMarkdown(filePath, content, now),
            ".txt" => ChunkTxt(filePath, content, now),
            ".cs" => ChunkCSharp(filePath, content, now),
            _ => ChunkSections(filePath, content, fallback, now)
        };
    }

    private IEnumerable<DocumentChunk> ChunkMarkdown(string filePath, string content, DateTime now)
    {
        var lines = content.Split('\n');
        var sections = new List<(string? section, string content)>();
        string? currentSection = null;
        var currentLines = new List<string>();

        foreach (var line in lines)
        {
            var trimmed = line.TrimStart();
            var headingLevel = 0;
            while (headingLevel < trimmed.Length && headingLevel < 6 && trimmed[headingLevel] == '#')
                headingLevel++;

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

    private IEnumerable<DocumentChunk> ChunkTxt(string filePath, string content, DateTime now)
    {
        var normalized = content.Replace("\r\n", "\n");
        var blocks = normalized.Split("\n\n", StringSplitOptions.RemoveEmptyEntries);

        if (blocks.Length <= 1)
            blocks = normalized.Split('\n', StringSplitOptions.RemoveEmptyEntries);

        var sections = new List<(string? section, string content)>();
        foreach (var block in blocks)
        {
            var trimmed = block.Trim();
            if (string.IsNullOrEmpty(trimmed)) continue;

            var lines = trimmed.Split('\n');
            var firstLine = lines[0].Trim();
            var isHeading = firstLine.Length <= 100
                && !firstLine.EndsWith('.')
                && !firstLine.EndsWith(',')
                && char.IsUpper(firstLine[0]);
            var section = isHeading ? firstLine : null;
            sections.Add((section, trimmed));
        }

        return ProcessSections(filePath, sections, now);
    }

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
                currentRegion = "Код вне региона";
                currentLines = [];
            }
            else
            {
                currentLines.Add(line);
            }
        }

        if (currentLines.Count > 0)
        {
            var remaining = string.Join('\n', currentLines).Trim();
            if (!string.IsNullOrEmpty(remaining))
                sections.Add((currentRegion ?? "Код вне региона", remaining));
        }

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
                TotalChunks = total
            };
        }
    }

    private IEnumerable<DocumentChunk> ProcessSections(string filePath, List<(string? section, string content)> sections, DateTime now)
    {
        var fallback = new FixedSizeChunkingStrategy(maxChunkSize, overlap);
        var fileChunks = new List<DocumentChunk>();

        foreach (var (section, sectionContent) in sections)
        {
            if (string.IsNullOrWhiteSpace(sectionContent)) continue;

            var wordCount = sectionContent.Split([' ', '\t', '\n'], StringSplitOptions.RemoveEmptyEntries).Length;

            if (wordCount <= maxChunkSize)
            {
                fileChunks.Add(new DocumentChunk
                {
                    ChunkId = Guid.NewGuid().ToString(),
                    Source = filePath,
                    Title = Path.GetFileName(filePath),
                    Section = section,
                    Content = sectionContent,
                    ChunkIndex = 0,
                    TotalChunks = 0,
                    Strategy = ChunkingStrategy.Structural,
                    IndexedAt = now
                });
            }
            else
            {
                var subChunks = fallback.Chunk(filePath, sectionContent).ToList();
                foreach (var sc in subChunks)
                {
                    fileChunks.Add(new DocumentChunk
                    {
                        ChunkId = sc.ChunkId,
                        Source = sc.Source,
                        Title = sc.Title,
                        Section = section,
                        Content = sc.Content,
                        ChunkIndex = 0,
                        TotalChunks = 0,
                        Strategy = ChunkingStrategy.Structural,
                        IndexedAt = now
                    });
                }
            }
        }

        var total = fileChunks.Count;
        for (int i = 0; i < fileChunks.Count; i++)
        {
            yield return fileChunks[i] with { ChunkIndex = i, TotalChunks = total };
        }
    }
}

// ============================================================
// EMBEDDING SERVICE
// ============================================================

public interface IEmbeddingService
{
    Task<float[][]> GenerateEmbeddingsAsync(IEnumerable<string> texts, CancellationToken ct = default);
}

public class OllamaEmbeddingService(string host = "http://localhost:11434", string model = "nomic-embed-text") : IEmbeddingService
{
    private readonly OllamaApiClient _client = new(new Uri(host)) { SelectedModel = model };

    public async Task<IEnumerable<Model>> CheckAvailabilityAsync()
    {
        return await _client.ListLocalModelsAsync();
    }

    public async Task<float[]> GenerateQueryEmbeddingAsync(string query, CancellationToken ct = default)
    {
        var textList = new List<string> { query };
        var batches = new List<List<string>> { textList };
        var results = new float[1][];
        var index = 0;

        foreach (var b in batches)
        {
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
                batches.Add(batch);
                batch = [];
            }
        }
        if (batch.Count > 0) batches.Add(batch);

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

// ============================================================
// SQLITE VECTOR STORE
// ============================================================

public interface IVectorStore
{
    Task InitializeAsync();
    Task SaveChunksAsync(IEnumerable<IndexedChunk> chunks);
    Task<IEnumerable<IndexedChunk>> SearchSimilarAsync(float[] queryEmbedding, int topK = 5, ChunkingStrategy? strategy = null);
    Task<IEnumerable<(IndexedChunk chunk, float similarity)>> SearchSimilarWithScoresAsync(float[] queryEmbedding, int topK = 5, ChunkingStrategy? strategy = null);
    Task<long> GetChunkCountAsync(ChunkingStrategy? strategy = null);
    Task ClearIndexAsync(ChunkingStrategy? strategy = null);
    Task<ChunkingStats> GetStatsAsync(ChunkingStrategy strategy);
}

public class SqliteVectorStore(string dbPath = "index.db") : IVectorStore
{
    private string ConnectionString => $"Data Source={dbPath}";

    public async Task InitializeAsync()
    {
        using var conn = new SqliteConnection(ConnectionString);
        await conn.OpenAsync();

        var cmd = conn.CreateCommand();
        cmd.CommandText = """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                title TEXT NOT NULL,
                section TEXT,
                content TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                total_chunks INTEGER NOT NULL,
                strategy TEXT NOT NULL,
                indexed_at TEXT NOT NULL,
                embedding TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_strategy ON chunks(strategy);
            CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
        """;
        await cmd.ExecuteNonQueryAsync();
    }

    public async Task SaveChunksAsync(IEnumerable<IndexedChunk> chunks)
    {
        using var conn = new SqliteConnection(ConnectionString);
        await conn.OpenAsync();

        using var transaction = conn.BeginTransaction();
        var cmd = conn.CreateCommand();
        cmd.Transaction = transaction;
        cmd.CommandText = """
            INSERT OR REPLACE INTO chunks
                (chunk_id, source, title, section, content, chunk_index, total_chunks, strategy, indexed_at, embedding)
            VALUES
                ($chunk_id, $source, $title, $section, $content, $chunk_index, $total_chunks, $strategy, $indexed_at, $embedding)
        """;

        var pChunkId = cmd.Parameters.Add("$chunk_id", SqliteType.Text);
        var pSource = cmd.Parameters.Add("$source", SqliteType.Text);
        var pTitle = cmd.Parameters.Add("$title", SqliteType.Text);
        var pSection = cmd.Parameters.Add("$section", SqliteType.Text);
        var pContent = cmd.Parameters.Add("$content", SqliteType.Text);
        var pChunkIndex = cmd.Parameters.Add("$chunk_index", SqliteType.Integer);
        var pTotalChunks = cmd.Parameters.Add("$total_chunks", SqliteType.Integer);
        var pStrategy = cmd.Parameters.Add("$strategy", SqliteType.Text);
        var pIndexedAt = cmd.Parameters.Add("$indexed_at", SqliteType.Text);
        var pEmbedding = cmd.Parameters.Add("$embedding", SqliteType.Text);

        foreach (var chunk in chunks)
        {
            pChunkId.Value = chunk.ChunkId;
            pSource.Value = chunk.Source;
            pTitle.Value = chunk.Title;
            pSection.Value = chunk.Section ?? (object)DBNull.Value;
            pContent.Value = chunk.Content;
            pChunkIndex.Value = chunk.ChunkIndex;
            pTotalChunks.Value = chunk.TotalChunks;
            pStrategy.Value = chunk.Strategy.ToString();
            pIndexedAt.Value = chunk.IndexedAt.ToString("O");
            pEmbedding.Value = JsonSerializer.Serialize(chunk.Embedding);
            await cmd.ExecuteNonQueryAsync();
        }

        transaction.Commit();
    }

    public async Task<IEnumerable<IndexedChunk>> SearchSimilarAsync(float[] queryEmbedding, int topK = 5, ChunkingStrategy? strategy = null)
    {
        using var conn = new SqliteConnection(ConnectionString);
        await conn.OpenAsync();

        var cmd = conn.CreateCommand();
        if (strategy.HasValue)
        {
            cmd.CommandText = "SELECT * FROM chunks WHERE strategy = $strategy";
            cmd.Parameters.AddWithValue("$strategy", strategy.Value.ToString());
        }
        else
        {
            cmd.CommandText = "SELECT * FROM chunks";
        }

        var chunks = new List<(IndexedChunk chunk, float similarity)>();
        using var reader = await cmd.ExecuteReaderAsync();

        while (await reader.ReadAsync())
        {
            var embeddingJson = reader.GetString(reader.GetOrdinal("embedding"));
            var embedding = JsonSerializer.Deserialize<float[]>(embeddingJson) ?? [];
            var similarity = VectorMath.CosineSimilarity(queryEmbedding, embedding);

            var chunk = new IndexedChunk
            {
                ChunkId = reader.GetString(reader.GetOrdinal("chunk_id")),
                Source = reader.GetString(reader.GetOrdinal("source")),
                Title = reader.GetString(reader.GetOrdinal("title")),
                Section = reader.IsDBNull(reader.GetOrdinal("section")) ? null : reader.GetString(reader.GetOrdinal("section")),
                Content = reader.GetString(reader.GetOrdinal("content")),
                ChunkIndex = reader.GetInt32(reader.GetOrdinal("chunk_index")),
                TotalChunks = reader.GetInt32(reader.GetOrdinal("total_chunks")),
                Strategy = Enum.Parse<ChunkingStrategy>(reader.GetString(reader.GetOrdinal("strategy"))),
                IndexedAt = DateTime.Parse(reader.GetString(reader.GetOrdinal("indexed_at"))),
                Embedding = embedding
            };
            chunks.Add((chunk, similarity));
        }

        return chunks.OrderByDescending(c => c.similarity).Take(topK).Select(c => c.chunk);
    }

    public async Task<IEnumerable<(IndexedChunk chunk, float similarity)>> SearchSimilarWithScoresAsync(float[] queryEmbedding, int topK = 5, ChunkingStrategy? strategy = null)
    {
        using var conn = new SqliteConnection(ConnectionString);
        await conn.OpenAsync();

        var cmd = conn.CreateCommand();
        if (strategy.HasValue)
        {
            cmd.CommandText = "SELECT * FROM chunks WHERE strategy = $strategy";
            cmd.Parameters.AddWithValue("$strategy", strategy.Value.ToString());
        }
        else
        {
            cmd.CommandText = "SELECT * FROM chunks";
        }

        var chunks = new List<(IndexedChunk chunk, float similarity)>();
        using var reader = await cmd.ExecuteReaderAsync();

        while (await reader.ReadAsync())
        {
            var embeddingJson = reader.GetString(reader.GetOrdinal("embedding"));
            var embedding = JsonSerializer.Deserialize<float[]>(embeddingJson) ?? [];
            var similarity = VectorMath.CosineSimilarity(queryEmbedding, embedding);

            var chunk = new IndexedChunk
            {
                ChunkId = reader.GetString(reader.GetOrdinal("chunk_id")),
                Source = reader.GetString(reader.GetOrdinal("source")),
                Title = reader.GetString(reader.GetOrdinal("title")),
                Section = reader.IsDBNull(reader.GetOrdinal("section")) ? null : reader.GetString(reader.GetOrdinal("section")),
                Content = reader.GetString(reader.GetOrdinal("content")),
                ChunkIndex = reader.GetInt32(reader.GetOrdinal("chunk_index")),
                TotalChunks = reader.GetInt32(reader.GetOrdinal("total_chunks")),
                Strategy = Enum.Parse<ChunkingStrategy>(reader.GetString(reader.GetOrdinal("strategy"))),
                IndexedAt = DateTime.Parse(reader.GetString(reader.GetOrdinal("indexed_at"))),
                Embedding = embedding
            };
            chunks.Add((chunk, similarity));
        }

        return chunks.OrderByDescending(c => c.similarity).Take(topK);
    }

    public async Task<long> GetChunkCountAsync(ChunkingStrategy? strategy = null)
    {
        using var conn = new SqliteConnection(ConnectionString);
        await conn.OpenAsync();

        var cmd = conn.CreateCommand();
        if (strategy.HasValue)
        {
            cmd.CommandText = "SELECT COUNT(*) FROM chunks WHERE strategy = $strategy";
            cmd.Parameters.AddWithValue("$strategy", strategy.Value.ToString());
        }
        else
        {
            cmd.CommandText = "SELECT COUNT(*) FROM chunks";
        }

        return (long)(await cmd.ExecuteScalarAsync())!;
    }

    public async Task ClearIndexAsync(ChunkingStrategy? strategy = null)
    {
        using var conn = new SqliteConnection(ConnectionString);
        await conn.OpenAsync();

        var cmd = conn.CreateCommand();
        if (strategy.HasValue)
        {
            cmd.CommandText = "DELETE FROM chunks WHERE strategy = $strategy";
            cmd.Parameters.AddWithValue("$strategy", strategy.Value.ToString());
        }
        else
        {
            cmd.CommandText = "DELETE FROM chunks";
        }

        await cmd.ExecuteNonQueryAsync();
    }

    public async Task<ChunkingStats> GetStatsAsync(ChunkingStrategy strategy)
    {
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
}

// ============================================================
// INDEXING PIPELINE
// ============================================================

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
        var sw = Stopwatch.StartNew();

        var files = Directory.EnumerateFiles(rootDirectory, "*.*", SearchOption.AllDirectories)
            .Where(f => fileExtensions.Contains(Path.GetExtension(f), StringComparer.OrdinalIgnoreCase))
            .ToList();

        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine($"  Найдено файлов: {files.Count}");
        Console.ResetColor();

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
}

// ============================================================
// STRATEGY COMPARATOR
// ============================================================

public class StrategyComparator(IVectorStore vectorStore)
{
    public async Task CompareAndReportAsync()
    {
        var fixedStats = await vectorStore.GetStatsAsync(ChunkingStrategy.FixedSize);
        var structStats = await vectorStore.GetStatsAsync(ChunkingStrategy.Structural);

        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("\n╔══════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║           СРАВНЕНИЕ СТРАТЕГИЙ CHUNKING'А                     ║");
        Console.WriteLine("╠════════════════════╦══════════════════╦══════════════════════╣");
        Console.WriteLine("║ Метрика            ║ FixedSize        ║ Structural           ║");
        Console.WriteLine("╠════════════════════╬══════════════════╬══════════════════════╣");
        Console.WriteLine($"║ Чанков             ║ {fixedStats.ChunkCount,16} ║ {structStats.ChunkCount,17}    ║");
        Console.WriteLine($"║ Средний размер     ║ {fixedStats.AvgChunkLengthChars,8:F0} симв    ║ {structStats.AvgChunkLengthChars,9:F0} симв       ║");
        Console.WriteLine($"║ Средний размер     ║ {fixedStats.AvgChunkLengthWords,8:F0} слова   ║ {structStats.AvgChunkLengthWords,9:F0} слова      ║");
        Console.WriteLine($"║ Файлов >1 чанка    ║ {fixedStats.FilesWithMultipleChunks,16} ║ {structStats.FilesWithMultipleChunks,17}    ║");
        Console.WriteLine($"║ Секций распознано  ║ {"N/A",16} ║ {structStats.SectionsDetected?.ToString() ?? "N/A",17}    ║");
        Console.WriteLine("╚════════════════════╩══════════════════╩══════════════════════╝");
        Console.ResetColor();

        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("\n  ВЫВОД:");
        Console.ResetColor();

        if (structStats.ChunkCount < fixedStats.ChunkCount)
        {
            Console.WriteLine("  ✅ Structural стратегия создала меньше чанков, что указывает на");
            Console.WriteLine("     более осмысленное разбиение документа.");
        }
        else
        {
            Console.WriteLine("  ℹ️  FixedSize стратегия создала меньше или столько же чанков.");
        }

        if (structStats.AvgChunkLengthChars > fixedStats.AvgChunkLengthChars)
        {
            Console.WriteLine("  ✅ Structural чанки в среднем длиннее — секции сохраняют");
            Console.WriteLine("     контекст и целостность смысловых блоков.");
        }

        if (structStats.SectionsDetected > 0)
        {
            Console.WriteLine($"  ✅ Structural распознала {structStats.SectionsDetected} секций,");
            Console.WriteLine("     что помогает в поиске по структурным элементам.");
        }

        Console.WriteLine("\n  📌 Для Markdown, C# и TXT файлов Structural стратегия предпочтительнее.");
        Console.WriteLine("  📌 Для неизвестных форматов используется fallback на FixedSize.\n");
    }
}
