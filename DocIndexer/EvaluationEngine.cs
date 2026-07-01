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
        var ragPipeline = new RagPipeline(
            _embeddingService,
            vectorStore);
        var agent = new ComparisonAgent(llmService, ragPipeline);

        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("\n╔══════════════════════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║                    RAG vs NO-RAG COMPARISON REPORT                           ║");
        Console.WriteLine("╚══════════════════════════════════════════════════════════════════════════════╝");
        Console.ResetColor();

        for (int i = 0; i < questions.Count; i++)
        {
            if (ct.IsCancellationRequested) break;

            var q = questions[i];
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine($"\n--- Question {q.Id}/10: {q.Difficulty} ---");
            Console.ResetColor();
            Console.WriteLine($"Q: {q.Question}");

            try
            {
                var result = await agent.CompareAsync(q, ct);

                await SaveEvaluationAsync(q, "without_rag", result.AnswerWithoutRag, null, result.TimeWithoutRagMs);
                await SaveEvaluationAsync(q, "with_rag", result.AnswerWithRag, result.SourcesUsed, result.TimeWithRagMs);

                PrintComparison(result);
                PrintKeyConceptsScore(result);
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"❌ Ошибка при обработке вопроса {q.Id}: {ex.Message}");
                Console.ResetColor();
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
                created_at TEXT NOT NULL
            )
        """;
        await cmd.ExecuteNonQueryAsync();
    }

    private async Task SaveEvaluationAsync(TestQuestion question, string mode, string answer, string[]? sources, long timeMs)
    {
        using var conn = new SqliteConnection(ConnectionString);
        await conn.OpenAsync();

        var cmd = conn.CreateCommand();
        cmd.CommandText = """
            INSERT INTO evaluations (question_id, question, mode, answer, sources_used, response_time_ms, created_at)
            VALUES ($id, $question, $mode, $answer, $sources, $time, $now)
        """;
        cmd.Parameters.AddWithValue("$id", question.Id);
        cmd.Parameters.AddWithValue("$question", question.Question);
        cmd.Parameters.AddWithValue("$mode", mode);
        cmd.Parameters.AddWithValue("$answer", answer);
        cmd.Parameters.AddWithValue("$sources", sources != null ? JsonSerializer.Serialize(sources) : DBNull.Value);
        cmd.Parameters.AddWithValue("$time", timeMs);
        cmd.Parameters.AddWithValue("$now", DateTime.UtcNow.ToString("O"));
        await cmd.ExecuteNonQueryAsync();
    }

    private void PrintComparison(ComparisonResult result)
    {
        var q = result.Question;

        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine("  WITHOUT RAG:");
        Console.ResetColor();
        Console.WriteLine($"  {Truncate(result.AnswerWithoutRag, 300)}");
        Console.WriteLine($"  Length: {result.AnswerWithoutRag.Length} chars | Time: {result.TimeWithoutRagMs / 1000.0:F1}s");

        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine("  WITH RAG:");
        Console.ResetColor();
        if (result.SourcesUsed.Length > 0)
        {
            Console.ForegroundColor = ConsoleColor.Magenta;
            Console.WriteLine($"  Sources: {string.Join(", ", result.SourcesUsed)}");
            Console.ResetColor();
        }
        Console.WriteLine($"  {Truncate(result.AnswerWithRag, 300)}");
        Console.WriteLine($"  Length: {result.AnswerWithRag.Length} chars | Time: {result.TimeWithRagMs / 1000.0:F1}s");

        // Simple analysis
        var ragLonger = result.AnswerWithRag.Length > result.AnswerWithoutRag.Length;
        var ragHasSources = result.SourcesUsed.Length > 0;
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.Write($"  Analysis: RAG ответ {(ragLonger ? "подробнее" : "короче")}");
        if (ragHasSources)
            Console.Write(" со ссылками на источники");
        Console.WriteLine();
        Console.ResetColor();
    }

    private async Task PrintSummaryAsync(List<TestQuestion> questions)
    {
        using var conn = new SqliteConnection(ConnectionString);
        await conn.OpenAsync();

        var cmd = conn.CreateCommand();
        cmd.CommandText = "SELECT mode, AVG(response_time_ms), AVG(LENGTH(answer)) FROM evaluations GROUP BY mode";
        using var reader = await cmd.ExecuteReaderAsync();

        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("\n╔══════════════════════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║                              ИТОГОВАЯ СТАТИСТИКА                             ║");
        Console.WriteLine("╠═══════════════╦════════════════╦══════════════════╦══════════════════════════╣");
        Console.WriteLine("║ Режим         ║ Среднее время  ║ Средняя длина    ║ Всего вопросов           ║");
        Console.WriteLine("╠═══════════════╬════════════════╬══════════════════╬══════════════════════════╣");

        var stats = new List<(string mode, double avgTime, double avgLen)>();
        while (await reader.ReadAsync())
        {
            stats.Add((
                reader.GetString(0),
                reader.GetDouble(1),
                reader.GetDouble(2)
            ));
        }

        foreach (var s in stats)
        {
            var label = s.mode == "without_rag" ? "Без RAG" : "С RAG";
            Console.WriteLine($"║ {label,-13} ║ {s.avgTime,12:F0} мс ║ {s.avgLen,14:F0} симв ║ {questions.Count,23} ║");
        }

        Console.WriteLine("╚═══════════════╩════════════════╩══════════════════╩══════════════════════════╝");
        Console.ResetColor();
    }

    private void PrintKeyConceptsScore(ComparisonResult result)
    {
        var keyConcepts = result.Question.KeyConcepts;
        if (keyConcepts == null || keyConcepts.Length == 0) return;

        var scoreWithoutRag = ScoreAnswer(result.AnswerWithoutRag, keyConcepts);
        var scoreWithRag = ScoreAnswer(result.AnswerWithRag, keyConcepts);

        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine($"  Key concepts matched without RAG: {scoreWithoutRag.Item1}/{keyConcepts.Length} ({scoreWithoutRag.Item2:F2})");
        Console.WriteLine($"  Key concepts matched with RAG:    {scoreWithRag.Item1}/{keyConcepts.Length} ({scoreWithRag.Item2:F2})");
        Console.ResetColor();
    }

    private static (int matched, double score) ScoreAnswer(string answer, string[] keyConcepts)
    {
        if (keyConcepts.Length == 0) return (0, 0);
        var lower = answer.ToLowerInvariant();
        var matched = keyConcepts.Count(kc => lower.Contains(kc.ToLowerInvariant()));
        return (matched, (double)matched / keyConcepts.Length);
    }

    private static string Truncate(string text, int maxLen)
    {
        if (text.Length <= maxLen) return text;
        return text[..maxLen] + "...";
    }
}
