# УЛУЧШЕННЫЙ RAG: ФИЛЬТРАЦИЯ, RERANKING, QUERY REWRITE + СРАВНЕНИЕ РЕЖИМОВ

## Контекст

Проект уже содержит базовый RAG-агент (RagPipeline, ComparisonAgent, EvaluationEngine). 
Нужно добавить **второй этап** после векторного поиска: фильтрация/реранкинг + query rewriting, 
и сравнить качество 4 режимов работы.

## Цель

Реализовать улучшенный RAG pipeline с:
1. **Query Rewrite** — переписывание вопроса для лучшего поиска
2. **Similarity Threshold Filter** — отсечение нерелевантных чанков по порогу
3. **Heuristic Reranker** — переранжирование чанков по комбинированному score
4. **Сравнение 4 режимов** — baseline, с фильтром, с reranker'ом, полный pipeline

## Архитектура улучшенного pipeline

```
Вопрос пользователя
        |
[QueryRewriteService] -> Переписанный запрос (опционально)
        |
[OllamaEmbeddingService] -> Эмбеддинг запроса
        |
[SqliteVectorStore.SearchSimilarWithScoresAsync] -> topK=PRE (default 10)
        |
[SimilarityThresholdFilter] -> Отсечь чанки с similarity < threshold
        |
[HeuristicReranker] -> Переранжировать по комбинированному score
        |
[TopKPostFilter] -> Взять topK=POST (default 3)
        |
[BuildContext] -> Формирование контекста для LLM
        |
[AnthropicLlmService] -> Ответ
```

## Что нужно реализовать

### 1. QueryRewriteService

```csharp
public interface IQueryRewriteService
{
    Task<string> RewriteAsync(string query, CancellationToken ct = default);
}

/// <summary>
/// Эвристический рерайтер: добавляет "Rust" и "pattern" если отсутствуют.
/// Быстрый, не требует API-вызовов.
/// </summary>
public class HeuristicQueryRewriteService : IQueryRewriteService
{
    public Task<string> RewriteAsync(string query, CancellationToken ct = default)
    {
        var lower = query.ToLowerInvariant();
        var rewritten = query;

        if (!lower.Contains("rust"))
            rewritten += " Rust";
        if (!lower.Contains("pattern") && !lower.Contains("idiom") && !lower.Contains("anti-pattern"))
            rewritten += " pattern";

        return Task.FromResult(rewritten.Trim());
    }
}

/// <summary>
/// LLM-based рерайтер: использует Claude для переписывания вопроса.
/// Более качественный, но тратит токены.
/// </summary>
public class LlmQueryRewriteService(ILlmService llm) : IQueryRewriteService
{
    public async Task<string> RewriteAsync(string query, CancellationToken ct = default)
    {
        var prompt = "Rewrite the following question into a concise search query optimized for retrieving Rust design patterns documentation. Keep only key technical terms and concepts. Output ONLY the rewritten query, nothing else.\n\nQuestion: " + query + "\nSearch query:";

        var rewritten = await llm.AskAsync(prompt, "You are a search query optimization assistant.", ct);
        return rewritten.Trim().Trim('"', '\'');
    }
}
```

### 2. SimilarityThresholdFilter

```csharp
public class SimilarityThresholdFilter
{
    private readonly float _threshold;

    public SimilarityThresholdFilter(float threshold = 0.5f)
    {
        _threshold = threshold;
    }

    public List<(IndexedChunk chunk, float similarity)> Filter(
        List<(IndexedChunk chunk, float similarity)> candidates)
    {
        return candidates.Where(c => c.similarity >= _threshold).ToList();
    }
}
```

### 3. HeuristicReranker

```csharp
public record ScoredChunk(
    IndexedChunk Chunk,
    float OriginalSimilarity,
    double FinalScore,
    double KeywordScore
);

public class HeuristicReranker
{
    /// <summary>
    /// Переранжирует чанки по комбинированному score:
    /// FinalScore = 0.6 * cosine_similarity + 0.3 * keyword_match_ratio + 0.1 * length_boost
    /// </summary>
    public List<ScoredChunk> Rerank(string query, List<(IndexedChunk chunk, float similarity)> candidates)
    {
        var queryWords = query.ToLowerInvariant()
            .Split(new[] { ' ', '\t', '\n', '?', '!', '.', ',', ':', ';' }, StringSplitOptions.RemoveEmptyEntries)
            .Where(w => w.Length > 2)
            .Distinct()
            .ToList();

        return candidates.Select(c => {
            var chunkText = c.chunk.Content.ToLowerInvariant();
            var chunkWords = chunkText.Split(new[] { ' ', '\t', '\n' }, StringSplitOptions.RemoveEmptyEntries);

            var matchedKeywords = queryWords.Count(qw => chunkText.Contains(qw));
            var keywordScore = queryWords.Count > 0 ? (double)matchedKeywords / queryWords.Count : 0;

            var wordCount = chunkWords.Length;
            var lengthBoost = wordCount switch
            {
                < 50 => 0.5,
                < 200 => 1.0,
                < 500 => 0.9,
                _ => 0.7
            };

            var finalScore = 0.6 * c.similarity + 0.3 * keywordScore + 0.1 * lengthBoost;

            return new ScoredChunk(c.chunk, c.similarity, finalScore, keywordScore);
        }).OrderByDescending(s => s.FinalScore).ToList();
    }
}
```

### 4. EnhancedRagPipeline

```csharp
public enum RagPipelineMode
{
    Baseline,
    WithThreshold,
    WithReranker,
    FullPipeline
}

public class EnhancedRagPipeline
{
    private readonly OllamaEmbeddingService _embeddingService;
    private readonly SqliteVectorStore _vectorStore;
    private readonly IQueryRewriteService? _rewriteService;
    private readonly SimilarityThresholdFilter _thresholdFilter;
    private readonly HeuristicReranker _reranker;
    private readonly int _topKPre;
    private readonly int _topKPost;
    private readonly float _similarityThreshold;
    private readonly bool _enableRewrite;
    private readonly bool _enableRerank;

    public EnhancedRagPipeline(
        OllamaEmbeddingService embeddingService,
        SqliteVectorStore vectorStore,
        IQueryRewriteService? rewriteService = null)
    {
        _embeddingService = embeddingService;
        _vectorStore = vectorStore;
        _rewriteService = rewriteService;
        _thresholdFilter = new SimilarityThresholdFilter(GetEnvFloat("RAG_SIMILARITY_THRESHOLD", 0.5f));
        _reranker = new HeuristicReranker();
        _topKPre = GetEnvInt("RAG_TOP_K_PRE", 10);
        _topKPost = GetEnvInt("RAG_TOP_K_POST", 3);
        _similarityThreshold = GetEnvFloat("RAG_SIMILARITY_THRESHOLD", 0.5f);
        _enableRewrite = GetEnvBool("RAG_ENABLE_REWRITE", true);
        _enableRerank = GetEnvBool("RAG_ENABLE_RERANK", true);
    }

    public async Task<RagResult> ExecuteAsync(
        string question, 
        RagPipelineMode mode,
        CancellationToken ct = default)
    {
        var originalQuestion = question;
        string? rewrittenQuestion = null;

        if (mode == RagPipelineMode.FullPipeline && _enableRewrite && _rewriteService != null)
        {
            rewrittenQuestion = await _rewriteService.RewriteAsync(question, ct);
            question = rewrittenQuestion;
        }

        var queryEmbedding = await _embeddingService.GenerateQueryEmbeddingAsync(question, ct);

        var searchK = mode == RagPipelineMode.Baseline ? _topKPost : _topKPre;
        var rawResults = await _vectorStore.SearchSimilarWithScoresAsync(queryEmbedding, searchK, ChunkingStrategy.Structural);
        var rawResultsList = rawResults.ToList();

        List<(IndexedChunk chunk, float similarity)> filtered = rawResultsList;
        if (mode != RagPipelineMode.Baseline)
        {
            filtered = _thresholdFilter.Filter(rawResultsList);
            if (filtered.Count == 0)
                filtered = rawResultsList.Take(_topKPost).ToList();
        }

        List<ScoredChunk> ranked;
        if ((mode == RagPipelineMode.WithReranker || mode == RagPipelineMode.FullPipeline) && _enableRerank)
            ranked = _reranker.Rerank(question, filtered);
        else
            ranked = filtered.Select(f => new ScoredChunk(f.chunk, f.similarity, f.similarity, 0)).ToList();

        var finalChunks = ranked.Take(_topKPost).ToList();

        var contextParts = finalChunks.Select((c, i) =>
        {
            var sectionInfo = c.Chunk.Section != null ? $"[Section: {c.Chunk.Section}]" : "";
            return $"--- Source {i + 1}: {c.Chunk.Title} {sectionInfo} (score: {c.FinalScore:F3}) ---\n{c.Chunk.Content}";
        });

        var context = string.Join("\n\n", contextParts);

        var sources = finalChunks
            .Select(c => c.Chunk.Source)
            .Select(p => {
                var normalized = p.Replace('\\', '/');
                var idx = normalized.IndexOf("patterns/", StringComparison.OrdinalIgnoreCase);
                return idx >= 0 ? normalized[idx..] : Path.GetFileName(p);
            })
            .Distinct()
            .ToArray();

        return new RagResult(context, sources, originalQuestion, rewrittenQuestion, finalChunks, mode, _similarityThreshold, _topKPre, _topKPost);
    }

    private static int GetEnvInt(string name, int defaultValue) =>
        int.TryParse(Environment.GetEnvironmentVariable(name), out var v) ? v : defaultValue;
    private static float GetEnvFloat(string name, float defaultValue) =>
        float.TryParse(Environment.GetEnvironmentVariable(name), out var v) ? v : defaultValue;
    private static bool GetEnvBool(string name, bool defaultValue) =>
        bool.TryParse(Environment.GetEnvironmentVariable(name), out var v) ? v : defaultValue;
}

public record RagResult(
    string Context,
    string[] Sources,
    string OriginalQuestion,
    string? RewrittenQuestion,
    List<ScoredChunk> Chunks,
    RagPipelineMode Mode,
    float SimilarityThreshold,
    int TopKPre,
    int TopKPost
);
```

### 5. Изменения в SqliteVectorStore

```csharp
// Добавить в интерфейс IVectorStore:
Task<IEnumerable<(IndexedChunk chunk, float similarity)>> SearchSimilarWithScoresAsync(
    float[] queryEmbedding, int topK = 5, ChunkingStrategy? strategy = null);

// Реализация (копия SearchSimilarAsync но возвращает tuple):
public async Task<IEnumerable<(IndexedChunk chunk, float similarity)>> SearchSimilarWithScoresAsync(
    float[] queryEmbedding, int topK = 5, ChunkingStrategy? strategy = null)
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
```

### 6. Изменения в ComparisonAgent

```csharp
public class ComparisonAgent(
    ILlmService llmService,
    EnhancedRagPipeline enhancedRag)
{
    public async Task<ComparisonResult> CompareAsync(
        TestQuestion question, 
        RagPipelineMode mode,
        CancellationToken ct = default)
    {
        var systemPrompt = "You are an expert in Rust design patterns.";
        var sw1 = Stopwatch.StartNew();
        var answerWithoutRag = await llmService.AskAsync(question.Question, systemPrompt, ct);
        sw1.Stop();

        var ragSystemPrompt = "You are an expert in Rust design patterns. Answer based ONLY on the provided context. If the context doesn't contain the answer, say so.";

        var ragResult = await enhancedRag.ExecuteAsync(question.Question, mode, ct);

        var ragPrompt = "Context information is below.\n---------------------\n" + ragResult.Context + 
            "\n---------------------\nGiven the context information and not prior knowledge, answer the following question:\n" + question.Question;

        var sw2 = Stopwatch.StartNew();
        var answerWithRag = await llmService.AskAsync(ragPrompt, ragSystemPrompt, ct);
        sw2.Stop();

        return new ComparisonResult
        {
            Question = question,
            AnswerWithoutRag = answerWithoutRag,
            AnswerWithRag = answerWithRag,
            TimeWithoutRagMs = sw1.ElapsedMilliseconds,
            TimeWithRagMs = sw2.ElapsedMilliseconds,
            SourcesUsed = ragResult.Sources,
            RagMode = mode,
            RewrittenQuestion = ragResult.RewrittenQuestion,
            SimilarityThreshold = ragResult.SimilarityThreshold,
            TopKPre = ragResult.TopKPre,
            TopKPost = ragResult.TopKPost,
            ChunksUsed = ragResult.Chunks.Select(c => new ChunkInfo(
                c.Chunk.Title, c.Chunk.Section, c.OriginalSimilarity, c.FinalScore, c.KeywordScore
            )).ToList()
        };
    }
}

public record ChunkInfo(
    string Title,
    string? Section,
    float OriginalSimilarity,
    double FinalScore,
    double KeywordScore
);
```

### 7. Изменения в Models.cs

```csharp
public record ComparisonResult
{
    public required TestQuestion Question { get; init; }
    public required string AnswerWithoutRag { get; init; }
    public required string AnswerWithRag { get; init; }
    public required long TimeWithoutRagMs { get; init; }
    public required long TimeWithRagMs { get; init; }
    public required string[] SourcesUsed { get; init; }
    public RagPipelineMode RagMode { get; init; }
    public string? RewrittenQuestion { get; init; }
    public float SimilarityThreshold { get; init; }
    public int TopKPre { get; init; }
    public int TopKPost { get; init; }
    public List<ChunkInfo>? ChunksUsed { get; init; }
}
```

### 8. Изменения в EvaluationEngine

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

    public async Task RunAsync(List<TestQuestion> questions, CancellationToken ct = default)
    {
        var apiKey = Environment.GetEnvironmentVariable("ANTHROPIC_API_KEY");
        if (string.IsNullOrEmpty(apiKey))
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("ANTHROPIC_API_KEY не задан. Пропускаем сравнение.");
            Console.ResetColor();
            return;
        }

        await InitializeEvaluationTableAsync();

        using var llmService = new AnthropicLlmService();
        var vectorStore = new SqliteVectorStore(_dbPath);

        IQueryRewriteService rewriteService = GetEnvBool("RAG_USE_LLM_REWRITE", false) 
            ? new LlmQueryRewriteService(llmService) 
            : new HeuristicQueryRewriteService();

        var enhancedRag = new EnhancedRagPipeline(_embeddingService, vectorStore, rewriteService);
        var agent = new ComparisonAgent(llmService, enhancedRag);

        var modes = new[] { RagPipelineMode.Baseline, RagPipelineMode.WithThreshold, 
                           RagPipelineMode.WithReranker, RagPipelineMode.FullPipeline };

        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("\n╔══════════════════════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║         УЛУЧШЕННЫЙ RAG: СРАВНЕНИЕ РЕЖИМОВ                                    ║");
        Console.WriteLine("╚══════════════════════════════════════════════════════════════════════════════╝");
        Console.ResetColor();

        foreach (var q in questions)
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine($"\n--- Question {q.Id}: {q.Question} ---");
            Console.ResetColor();

            var withoutRag = await agent.CompareAsync(q, RagPipelineMode.Baseline, ct);
            await SaveEvaluationAsync(q, "without_rag", withoutRag);

            foreach (var mode in modes)
            {
                try
                {
                    var withRag = await agent.CompareAsync(q, mode, ct);
                    await SaveEvaluationAsync(q, $"with_rag_{mode}", withRag);
                    PrintComparison(withRag, mode);
                }
                catch (Exception ex)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine($"Ошибка в режиме {mode}: {ex.Message}");
                    Console.ResetColor();
                }
            }
        }

        await PrintSummaryAsync(questions);
    }

    private void PrintComparison(ComparisonResult result, RagPipelineMode mode)
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine($"  Mode: {mode}");
        if (result.RewrittenQuestion != null)
            Console.WriteLine($"  Rewritten: {result.RewrittenQuestion}");
        Console.WriteLine($"  Chunks: {result.ChunksUsed?.Count ?? 0} (threshold: {result.SimilarityThreshold})");
        if (result.ChunksUsed != null)
        {
            foreach (var c in result.ChunksUsed)
                Console.WriteLine($"    - {c.Title} | orig_sim: {c.OriginalSimilarity:F3} | final: {c.FinalScore:F3} | keywords: {c.KeywordScore:F3}");
        }
        Console.ResetColor();

        var score = ScoreAnswer(result.AnswerWithRag, result.Question.KeyConcepts);
        Console.WriteLine($"  Key concepts: {score.matched}/{result.Question.KeyConcepts?.Length ?? 0} ({score.score:F2})");
    }

    private async Task PrintSummaryAsync(List<TestQuestion> questions)
    {
        using var conn = new SqliteConnection(ConnectionString);
        await conn.OpenAsync();

        var cmd = conn.CreateCommand();
        cmd.CommandText = @"
            SELECT mode, AVG(response_time_ms) as avg_time, COUNT(*) as cnt
            FROM evaluations 
            WHERE mode LIKE 'with_rag_%' OR mode = 'without_rag'
            GROUP BY mode";

        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("\n╔══════════════════════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║                    СРАВНЕНИЕ РЕЖИМОВ RAG PIPELINE                            ║");
        Console.WriteLine("╠════════════════════╦══════════════════╦══════════════════╦═══════════════════╣");
        Console.WriteLine("║ Режим              ║ Avg Time (ms)    ║ Count            ║                   ║");
        Console.WriteLine("╠════════════════════╬══════════════════╬══════════════════╬═══════════════════╣");

        using var reader = await cmd.ExecuteReaderAsync();
        while (await reader.ReadAsync())
        {
            var mode = reader.GetString(0);
            var avgTime = reader.GetDouble(1);
            var count = reader.GetInt32(2);
            Console.WriteLine($"║ {mode,-18} ║ {avgTime,14:F0} ║ {count,14} ║                   ║");
        }

        Console.WriteLine("╚════════════════════╩══════════════════╩══════════════════╩═══════════════════╝");
        Console.ResetColor();
    }

    private static (int matched, double score) ScoreAnswer(string answer, string[]? keyConcepts)
    {
        if (keyConcepts == null || keyConcepts.Length == 0) return (0, 0);
        var lower = answer.ToLowerInvariant();
        var matched = keyConcepts.Count(kc => lower.Contains(kc.ToLowerInvariant()));
        return (matched, (double)matched / keyConcepts.Length);
    }

    private static bool GetEnvBool(string name, bool defaultValue) =>
        bool.TryParse(Environment.GetEnvironmentVariable(name), out var v) ? v : defaultValue;
}
```

### 9. Environment Variables

```
RAG_TOP_K_PRE=10              # Сколько чанков искать в векторной БД (pre-filter)
RAG_TOP_K_POST=3              # Сколько чанков отправлять в LLM (post-filter)
RAG_SIMILARITY_THRESHOLD=0.5  # Минимальная cosine similarity
RAG_ENABLE_REWRITE=true       # Включить query rewriting
RAG_ENABLE_RERANK=true        # Включить heuristic reranker
RAG_USE_LLM_REWRITE=false     # Использовать LLM вместо эвристики для rewrite
```

## Чек-лист

### Новые файлы:
- [ ] QueryRewriteService.cs
- [ ] SimilarityThresholdFilter.cs
- [ ] HeuristicReranker.cs
- [ ] EnhancedRagPipeline.cs

### Изменения:
- [ ] SqliteVectorStore.cs — SearchSimilarWithScoresAsync
- [ ] ComparisonAgent.cs — EnhancedRagPipeline + RagPipelineMode
- [ ] Models.cs — RagResult, ChunkInfo, RagPipelineMode, обновить ComparisonResult
- [ ] EvaluationEngine.cs — 4 режима, детальная статистика
- [ ] Program.cs — передача EnhancedRagPipeline
