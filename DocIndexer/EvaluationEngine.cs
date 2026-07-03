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

    public async Task RunAsync(List<TestQuestion> questions, RagPipelineMode[]? modes = null, CancellationToken ct = default)
    {
        var apiKey = Environment.GetEnvironmentVariable("ANTHROPIC_API_KEY");
        if (string.IsNullOrEmpty(apiKey))
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("\n⚠️  ANTHROPIC_API_KEY не задан. Пропускаем сравнение.");
            Console.WriteLine("   Установите: export ANTHROPIC_API_KEY=sk-ant-...");
            Console.ResetColor();
            return;
        }

        await InitializeEvaluationTableAsync();
        await InitializeCitationEvaluationTableAsync();

        using var llmService = new AnthropicLlmService();
        var vectorStore = new SqliteVectorStore(_dbPath);

        IQueryRewriteService rewriteService = GetEnvBool("RAG_USE_LLM_REWRITE", false)
            ? new LlmQueryRewriteService(llmService)
            : new HeuristicQueryRewriteService();

        var enhancedRag = new EnhancedRagPipeline(_embeddingService, vectorStore, rewriteService);
        var validator = new CitationValidator();
        var agent = new ComparisonAgent(llmService, enhancedRag, validator);

        modes ??= new[]
        {
            RagPipelineMode.Baseline,
            RagPipelineMode.WithThreshold,
            RagPipelineMode.WithReranker,
            RagPipelineMode.FullPipeline,
            RagPipelineMode.CitationEnforced
        };

        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("\n╔══════════════════════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║         СРАВНЕНИЕ РЕЖИМОВ RAG (CITATION-ENABLED)                             ║");
        Console.WriteLine("╚══════════════════════════════════════════════════════════════════════════════╝");
        Console.ResetColor();

        var citationEvals = new List<CitationEvaluationResult>();

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
                    var (answer, ragResult) = await agent.AskWithRagAsync(q, mode, ct);
                    sw.Stop();

                    var score = ScoreAnswer(answer?.Answer ?? "", q.KeyConcepts);

                    if (mode == RagPipelineMode.CitationEnforced && answer != null)
                    {
                        var eval = EvaluateCitation(q, answer, ragResult, validator);
                        citationEvals.Add(eval);
                        PrintCitationResult(answer, ragResult, eval, sw.ElapsedMilliseconds);

                        await SaveEvaluationAsync(q, $"with_rag_{mode}", answer.Answer, ragResult.Sources, sw.ElapsedMilliseconds,
                            pipelineMode: mode.ToString(), rewrittenQuestion: ragResult.RewrittenQuestion,
                            similarityThreshold: ragResult.SimilarityThreshold,
                            topKPre: ragResult.TopKPre, topKPost: ragResult.TopKPost,
                            keyConceptsScore: score.score, chunksInfo: SerializeChunks(ragResult.Chunks));

                        await SaveCitationEvaluationAsync(q, mode, eval);
                    }
                    else
                    {
                        PrintAnswer(answer?.Answer ?? "", ragResult, mode);
                        Console.WriteLine($"  Key concepts: {score.matched}/{q.KeyConcepts?.Length ?? 0} ({score.score:F2})");

                        await SaveEvaluationAsync(q, $"with_rag_{mode}", answer?.Answer ?? "", ragResult.Sources, sw.ElapsedMilliseconds,
                            pipelineMode: mode.ToString(), rewrittenQuestion: ragResult.RewrittenQuestion,
                            similarityThreshold: ragResult.SimilarityThreshold,
                            topKPre: ragResult.TopKPre, topKPost: ragResult.TopKPost,
                            keyConceptsScore: score.score, chunksInfo: SerializeChunks(ragResult.Chunks));
                    }
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

        if (citationEvals.Count > 0)
            PrintCitationSummary(citationEvals);
    }

    private static string? SerializeChunks(List<ScoredChunk>? chunks)
    {
        if (chunks == null) return null;
        var info = chunks.Select(c => new
        {
            c.Chunk.Title,
            c.Chunk.Section,
            c.OriginalSimilarity,
            c.FinalScore,
            c.KeywordScore
        });
        return JsonSerializer.Serialize(info);
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

    private async Task InitializeCitationEvaluationTableAsync()
    {
        using var conn = new SqliteConnection(ConnectionString);
        await conn.OpenAsync();

        var cmd = conn.CreateCommand();
        cmd.CommandText = """
            CREATE TABLE IF NOT EXISTS evaluation_citations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id INTEGER NOT NULL,
                question TEXT NOT NULL,
                mode TEXT NOT NULL,
                has_sources INTEGER NOT NULL DEFAULT 0,
                has_citations INTEGER NOT NULL DEFAULT 0,
                citations_match_context INTEGER NOT NULL DEFAULT 0,
                answer_consistent INTEGER NOT NULL DEFAULT 0,
                correctly_said_unknown INTEGER NOT NULL DEFAULT 0,
                confidence TEXT,
                is_unknown INTEGER NOT NULL DEFAULT 0,
                max_similarity REAL,
                chunk_count INTEGER,
                validation_errors TEXT,
                validation_warnings TEXT,
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

    private async Task SaveCitationEvaluationAsync(TestQuestion question, RagPipelineMode mode, CitationEvaluationResult eval)
    {
        using var conn = new SqliteConnection(ConnectionString);
        await conn.OpenAsync();

        var cmd = conn.CreateCommand();
        cmd.CommandText = """
            INSERT INTO evaluation_citations
                (question_id, question, mode, has_sources, has_citations, citations_match_context,
                 answer_consistent, correctly_said_unknown, confidence, is_unknown, max_similarity,
                 chunk_count, validation_errors, validation_warnings, created_at)
            VALUES
                ($id, $question, $mode, $hasSources, $hasCitations, $match,
                 $consistent, $unknown, $confidence, $isUnknown, $maxSim,
                 $chunkCount, $errors, $warnings, $now)
        """;
        cmd.Parameters.AddWithValue("$id", question.Id);
        cmd.Parameters.AddWithValue("$question", question.Question);
        cmd.Parameters.AddWithValue("$mode", mode.ToString());
        cmd.Parameters.AddWithValue("$hasSources", eval.HasSources ? 1 : 0);
        cmd.Parameters.AddWithValue("$hasCitations", eval.HasCitations ? 1 : 0);
        cmd.Parameters.AddWithValue("$match", eval.CitationsMatchContext ? 1 : 0);
        cmd.Parameters.AddWithValue("$consistent", eval.AnswerConsistentWithCitations ? 1 : 0);
        cmd.Parameters.AddWithValue("$unknown", eval.CorrectlySaidUnknown ? 1 : 0);
        cmd.Parameters.AddWithValue("$confidence", eval.Confidence.ToString());
        cmd.Parameters.AddWithValue("$isUnknown", eval.IsUnknown ? 1 : 0);
        cmd.Parameters.AddWithValue("$maxSim", eval.MaxSimilarity);
        cmd.Parameters.AddWithValue("$chunkCount", eval.ChunkCount);
        cmd.Parameters.AddWithValue("$errors", eval.Errors.Count > 0 ? JsonSerializer.Serialize(eval.Errors) : DBNull.Value);
        cmd.Parameters.AddWithValue("$warnings", eval.Warnings.Count > 0 ? JsonSerializer.Serialize(eval.Warnings) : DBNull.Value);
        cmd.Parameters.AddWithValue("$now", DateTime.UtcNow.ToString("O"));
        await cmd.ExecuteNonQueryAsync();
    }

    private CitationEvaluationResult EvaluateCitation(TestQuestion question, CitationAnswer answer, RagResult ragResult, CitationValidator validator)
    {
        var validation = validator.Validate(answer, ragResult.Chunks);

        var answerConsistent = IsAnswerConsistentWithCitations(answer);

        return new CitationEvaluationResult
        {
            QuestionId = question.Id,
            HasSources = answer.Sources.Count > 0,
            HasCitations = answer.Citations.Count > 0,
            CitationsMatchContext = validation.IsValid,
            AnswerConsistentWithCitations = answerConsistent,
            CorrectlySaidUnknown = ragResult.IsUnknown == (answer.Confidence == ConfidenceLevel.Unknown),
            Errors = validation.Errors,
            Warnings = validation.Warnings,
            Confidence = answer.Confidence,
            IsUnknown = ragResult.IsUnknown,
            MaxSimilarity = ragResult.MaxChunkSimilarity,
            ChunkCount = ragResult.Chunks.Count
        };
    }

    private static bool IsAnswerConsistentWithCitations(CitationAnswer answer)
    {
        if (string.IsNullOrWhiteSpace(answer.Answer) || answer.Citations.Count == 0)
            return false;

        var answerLower = answer.Answer.ToLowerInvariant();

        var stopWords = new HashSet<string> { "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "and", "or", "but", "it", "its", "this", "that" };

        var citationWords = answer.Citations
            .SelectMany(c => c.Quote.ToLowerInvariant()
                .Split([' ', '\t', '\n', '.', ',', ';', ':', '!', '?', '(', ')', '[', ']', '"', '\'', '`'], StringSplitOptions.RemoveEmptyEntries))
            .Where(w => w.Length > 3 && !stopWords.Contains(w))
            .Distinct()
            .ToList();

        if (citationWords.Count == 0) return true;

        var matched = citationWords.Count(w => answerLower.Contains(w));
        var ratio = (double)matched / citationWords.Count;
        return ratio >= 0.3;
    }

    private void PrintCitationResult(CitationAnswer answer, RagResult ragResult, CitationEvaluationResult eval, long timeMs)
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine($"  Mode: {ragResult.Mode}");
        Console.WriteLine($"  IsUnknown: {ragResult.IsUnknown}");
        Console.WriteLine($"  Confidence: {answer.Confidence}");
        Console.WriteLine($"  Sources: {answer.Sources.Count}");
        Console.WriteLine($"  Citations: {answer.Citations.Count} {(eval.CitationsMatchContext ? "✓" : "✗")}");
        Console.WriteLine($"  Answer consistent: {(eval.AnswerConsistentWithCitations ? "✓" : "✗")}");
        if (eval.Errors.Count > 0)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            foreach (var err in eval.Errors)
                Console.WriteLine($"    Error: {err}");
        }
        if (eval.Warnings.Count > 0)
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            foreach (var warn in eval.Warnings)
                Console.WriteLine($"    Warning: {warn}");
        }
        Console.ResetColor();

        Console.WriteLine($"  {Truncate(answer.Answer, 300)}");
        Console.WriteLine($"  Length: {answer.Answer.Length} chars | Time: {timeMs}ms");
    }

    private void PrintAnswer(string answer, RagResult ragResult, RagPipelineMode mode)
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine($"  Mode: {mode}");
        if (ragResult.RewrittenQuestion != null)
            Console.WriteLine($"  Rewritten: {ragResult.RewrittenQuestion}");
        Console.WriteLine($"  Chunks: {ragResult.Chunks.Count} (threshold: {ragResult.SimilarityThreshold})");
        foreach (var c in ragResult.Chunks)
            Console.WriteLine($"    - {c.Chunk.Title} | orig_sim: {c.OriginalSimilarity:F3} | final: {c.FinalScore:F3} | keywords: {c.KeywordScore:F3}");
        Console.ResetColor();

        Console.WriteLine($"  {Truncate(answer, 300)}");
        Console.WriteLine($"  Length: {answer.Length} chars");
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

    private void PrintCitationSummary(List<CitationEvaluationResult> evals)
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("\n╔══════════════════════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║           CITATION EVALUATION SUMMARY                                        ║");
        Console.WriteLine("╠══════════════════════════════════════════════════════════════════════════════╣");
        Console.ResetColor();

        var total = evals.Count;
        var withSources = evals.Count(e => e.HasSources);
        var withCitations = evals.Count(e => e.HasCitations);
        var citationsValid = evals.Count(e => e.CitationsMatchContext);
        var consistent = evals.Count(e => e.AnswerConsistentWithCitations);
        var correctlyUnknown = evals.Count(e => e.CorrectlySaidUnknown);

        Console.WriteLine($"  Total questions (CitationEnforced): {total}");
        Console.WriteLine($"  With sources:    {withSources}/{total} ({100.0 * withSources / total:F0}%) {(withSources == total ? "✓" : "✗")}");
        Console.WriteLine($"  With citations:  {withCitations}/{total} ({100.0 * withCitations / total:F0}%) {(withCitations == total ? "✓" : "✗")}");
        Console.WriteLine($"  Citations valid: {citationsValid}/{total} ({100.0 * citationsValid / total:F0}%) {(citationsValid == total ? "✓" : "✗")}");
        Console.WriteLine($"  Answer consistent: {consistent}/{total} ({100.0 * consistent / total:F0}%) {(consistent == total ? "✓" : "✗")}");
        Console.WriteLine($"  Correctly unknown: {correctlyUnknown}/{total}");

        var hasErrors = evals.Any(e => e.Errors.Count > 0);
        if (hasErrors)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("\n  Errors found in some responses:");
            foreach (var e in evals.Where(e => e.Errors.Count > 0))
            {
                Console.WriteLine($"    Q{e.QuestionId}: {string.Join("; ", e.Errors)}");
            }
            Console.ResetColor();
        }

        Console.WriteLine();
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

public class CitationEvaluationResult
{
    public int QuestionId { get; set; }
    public bool HasSources { get; set; }
    public bool HasCitations { get; set; }
    public bool CitationsMatchContext { get; set; }
    public bool AnswerConsistentWithCitations { get; set; }
    public bool CorrectlySaidUnknown { get; set; }
    public ConfidenceLevel Confidence { get; set; }
    public bool IsUnknown { get; set; }
    public float MaxSimilarity { get; set; }
    public int ChunkCount { get; set; }
    public List<string> Errors { get; set; } = [];
    public List<string> Warnings { get; set; } = [];
}
