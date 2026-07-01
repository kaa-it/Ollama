using System.Diagnostics;
using System.Text.Json;
using Microsoft.Data.Sqlite;

public class EvaluationEngine
{
    private readonly string _dbPath;
    private readonly OllamaEmbeddingService _embeddingService;
    private string ConnectionString => $"Data Source={_dbPath}";

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
            Console.WriteLine("\n⚠️  ANTHROPIC_API_KEY не задан. Пропускаем сравнение RAG vs No-RAG.");
            Console.WriteLine("   Установите: export ANTHROPIC_API_KEY=sk-ant-...");
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
        Console.WriteLine("║         СРАВНЕНИЕ РЕЖИМОВ RAG PIPELINE                                        ║");
        Console.WriteLine("╚══════════════════════════════════════════════════════════════════════════════╝");
        Console.ResetColor();

        foreach (var q in questions)
        {
            if (ct.IsCancellationRequested) break;

            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine($"\n--- Question {q.Id}/10: {q.Difficulty} ---");
            Console.ResetColor();
            Console.WriteLine($"Q: {q.Question}");

            foreach (var mode in modes)
            {
                try
                {
                    var sw = Stopwatch.StartNew();
                    var (answerWithRag, ragResult) = await agent.AskWithRagAsync(q, mode, ct);
                    sw.Stop();

                    var score = ScoreAnswer(answerWithRag, q.KeyConcepts);

                    PrintComparison(answerWithRag, ragResult, mode);
                    Console.WriteLine($"  Key concepts: {score.matched}/{q.KeyConcepts?.Length ?? 0} ({score.score:F2})");

                    var chunksInfo = ragResult.Chunks?.Select(c => new
                    {
                        c.Chunk.Title,
                        c.Chunk.Section,
                        c.OriginalSimilarity,
                        c.FinalScore,
                        c.KeywordScore
                    });
                    var chunksInfoJson = chunksInfo != null ? JsonSerializer.Serialize(chunksInfo) : null;

                    await SaveEvaluationAsync(q, $"with_rag_{mode}", answerWithRag, ragResult.Sources, sw.ElapsedMilliseconds,
                        pipelineMode: mode.ToString(), rewrittenQuestion: ragResult.RewrittenQuestion,
                        similarityThreshold: ragResult.SimilarityThreshold,
                        topKPre: ragResult.TopKPre, topKPost: ragResult.TopKPost,
                        keyConceptsScore: score.score, chunksInfo: chunksInfoJson);
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

    private async Task InitializeEvaluationTableAsync()
    {
        using var conn = new SqliteConnection(ConnectionString);
        await conn.OpenAsync();

        var cmd = conn.CreateCommand();
        cmd.CommandText = """
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id INTEGER NOT NULL,
                question TEXT NOT NULL,
                mode TEXT NOT NULL,
                answer TEXT NOT NULL,
                sources_used TEXT,
                response_time_ms INTEGER,
                pipeline_mode TEXT,
                rewritten_question TEXT,
                similarity_threshold REAL,
                top_k_pre INTEGER,
                top_k_post INTEGER,
                key_concepts_score REAL,
                chunks_info TEXT,
                created_at TEXT NOT NULL
            )
        """;
        await cmd.ExecuteNonQueryAsync();
    }

    private async Task SaveEvaluationAsync(
        TestQuestion question, string mode, string answer, string[]? sources, long timeMs,
        string? pipelineMode, string? rewrittenQuestion, float? similarityThreshold,
        int? topKPre, int? topKPost, double? keyConceptsScore, string? chunksInfo)
    {
        using var conn = new SqliteConnection(ConnectionString);
        await conn.OpenAsync();

        var cmd = conn.CreateCommand();
        cmd.CommandText = """
            INSERT INTO evaluations
                (question_id, question, mode, answer, sources_used, response_time_ms,
                 pipeline_mode, rewritten_question, similarity_threshold,
                 top_k_pre, top_k_post, key_concepts_score, chunks_info, created_at)
            VALUES
                ($id, $question, $mode, $answer, $sources, $time,
                 $pipelineMode, $rewritten, $threshold,
                 $topKPre, $topKPost, $score, $chunks, $now)
        """;
        cmd.Parameters.AddWithValue("$id", question.Id);
        cmd.Parameters.AddWithValue("$question", question.Question);
        cmd.Parameters.AddWithValue("$mode", mode);
        cmd.Parameters.AddWithValue("$answer", answer);
        cmd.Parameters.AddWithValue("$sources", sources != null ? JsonSerializer.Serialize(sources) : DBNull.Value);
        cmd.Parameters.AddWithValue("$time", timeMs);
        cmd.Parameters.AddWithValue("$pipelineMode", (object?)pipelineMode ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$rewritten", (object?)rewrittenQuestion ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$threshold", similarityThreshold.HasValue ? (object)similarityThreshold.Value : DBNull.Value);
        cmd.Parameters.AddWithValue("$topKPre", topKPre.HasValue ? (object)topKPre.Value : DBNull.Value);
        cmd.Parameters.AddWithValue("$topKPost", topKPost.HasValue ? (object)topKPost.Value : DBNull.Value);
        cmd.Parameters.AddWithValue("$score", keyConceptsScore.HasValue ? (object)keyConceptsScore.Value : DBNull.Value);
        cmd.Parameters.AddWithValue("$chunks", (object?)chunksInfo ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$now", DateTime.UtcNow.ToString("O"));
        await cmd.ExecuteNonQueryAsync();
    }

    private void PrintComparison(string answerWithRag, RagResult ragResult, RagPipelineMode mode)
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine($"  Mode: {mode}");
        if (ragResult.RewrittenQuestion != null)
            Console.WriteLine($"  Rewritten: {ragResult.RewrittenQuestion}");
        Console.WriteLine($"  Chunks: {ragResult.Chunks.Count} (threshold: {ragResult.SimilarityThreshold})");
        foreach (var c in ragResult.Chunks)
            Console.WriteLine($"    - {c.Chunk.Title} | orig_sim: {c.OriginalSimilarity:F3} | final: {c.FinalScore:F3} | keywords: {c.KeywordScore:F3}");
        Console.ResetColor();

        Console.WriteLine($"  {Truncate(answerWithRag, 300)}");
        Console.WriteLine($"  Length: {answerWithRag.Length} chars");
    }

    private async Task PrintSummaryAsync(List<TestQuestion> questions)
    {
        using var conn = new SqliteConnection(ConnectionString);
        await conn.OpenAsync();

        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("\n╔══════════════════════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║                    СРАВНЕНИЕ РЕЖИМОВ RAG PIPELINE                            ║");
        Console.WriteLine("╠════════════════════╦══════════════════╦══════════════════╦═══════════════════╣");
        Console.WriteLine("║ Режим              ║ Avg Time (ms)    ║ Avg Key Concept  ║ Count             ║");
        Console.WriteLine("╠════════════════════╬══════════════════╬══════════════════╬═══════════════════╣");

        var cmd = conn.CreateCommand();
        cmd.CommandText = @"
            SELECT mode,
                   AVG(response_time_ms) as avg_time,
                   AVG(key_concepts_score) as avg_score,
                   COUNT(*) as cnt
            FROM evaluations
            WHERE mode LIKE 'with_rag_%'
            GROUP BY mode
            ORDER BY mode";
        using var reader = await cmd.ExecuteReaderAsync();
        while (await reader.ReadAsync())
        {
            var modeLabel = reader.GetString(0);
            var avgTime = reader.IsDBNull(1) ? 0 : reader.GetDouble(1);
            var avgScore = reader.IsDBNull(2) ? 0 : reader.GetDouble(2);
            var count = reader.GetInt32(3);
            Console.WriteLine($"║ {modeLabel,-18} ║ {avgTime,14:F0} ║ {avgScore,14:F3} ║ {count,14} ║");
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

    private static string Truncate(string text, int maxLen)
    {
        if (text.Length <= maxLen) return text;
        return text[..maxLen] + "...";
    }

    private static bool GetEnvBool(string name, bool defaultValue) =>
        bool.TryParse(Environment.GetEnvironmentVariable(name), out var v) ? v : defaultValue;
}
