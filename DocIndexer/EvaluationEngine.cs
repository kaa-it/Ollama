using System.Diagnostics;
using System.Text.Json;
using Microsoft.Data.Sqlite;

public class EvaluationEngine
{
    private readonly string _dbPath;
    private readonly IEmbeddingService _embeddingService;
    private readonly ILlmService _llmService;
    private string ConnectionString => $"Data Source={_dbPath}";

    private readonly bool _dbAlreadyExisted;

    public EvaluationEngine(string dbPath, IEmbeddingService embeddingService, ILlmService llmService, bool dbAlreadyExisted = false)
    {
        _dbPath = dbPath;
        _embeddingService = embeddingService;
        _llmService = llmService;
        _dbAlreadyExisted = dbAlreadyExisted;
    }

    public async Task RunAsync(List<TestQuestion> questions, CancellationToken ct = default)
    {
        await InitializeCitationEvaluationTableAsync();

        var vectorStore = new SqliteVectorStore(_dbPath);

        IQueryRewriteService rewriteService = GetEnvBool("RAG_USE_LLM_REWRITE", false)
            ? new LlmQueryRewriteService(_llmService)
            : new HeuristicQueryRewriteService();

        var enhancedRag = new EnhancedRagPipeline(_embeddingService, vectorStore, rewriteService, _dbAlreadyExisted);
        var validator = new CitationValidator();
        var agent = new ComparisonAgent(_llmService, enhancedRag, validator);

        var mode = RagPipelineMode.CitationEnforced;

        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("\n╔══════════════════════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║                    CITATION-ENABLED RAG EVALUATION                           ║");
        Console.WriteLine("╚══════════════════════════════════════════════════════════════════════════════╝");
        Console.ResetColor();

        var citationEvals = new List<CitationEvaluationResult>();

        foreach (var q in questions)
        {
            if (ct.IsCancellationRequested) break;

            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine($"\n--- Question {q.Id}/{questions.Count}: {q.Difficulty} ---");
            Console.ResetColor();
            Console.WriteLine($"Q: {q.Question}");

            try
            {
                var sw = Stopwatch.StartNew();
                var (answer, ragResult) = await agent.AskWithRagAsync(q, mode, ct);
                sw.Stop();

                if (answer != null)
                {
                    var eval = EvaluateCitation(q, answer, ragResult, validator);
                    citationEvals.Add(eval);
                    PrintCitationResult(answer, ragResult, eval, sw.ElapsedMilliseconds);
                    await SaveCitationEvaluationAsync(q, mode, eval);
                }
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"Ошибка: {ex.Message}");
                Console.ResetColor();
            }
        }

        if (citationEvals.Count > 0)
            PrintCitationSummary(citationEvals);
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
        foreach (var src in answer.Sources)
            Console.WriteLine($"    - {src.Source}{(src.Section != null ? $" ({src.Section})" : "")}");
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

        if (!string.IsNullOrEmpty(answer.Answer))
        {
            Console.WriteLine($"  {answer.Answer}");
            Console.WriteLine($"  Length: {answer.Answer.Length} chars | Time: {timeMs}ms");
        }
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
                Console.WriteLine($"    Q{e.QuestionId}: {string.Join("; ", e.Errors)}");
            Console.ResetColor();
        }
        Console.WriteLine();
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
