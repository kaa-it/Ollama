using System.Diagnostics;

public class ComparisonAgent(
    ILlmService llmService,
    RagPipeline ragPipeline)
{
    public async Task<ComparisonResult> CompareAsync(TestQuestion question, CancellationToken ct = default)
    {
        var systemPrompt = "You are an expert in Rust design patterns.";

        var sw1 = Stopwatch.StartNew();
        var answerWithoutRag = await llmService.AskAsync(question.Question, systemPrompt, ct);
        sw1.Stop();

        var ragSystemPrompt = "You are an expert in Rust design patterns. Answer based ONLY on the provided context. If the context doesn't contain the answer, say so.";

        var (ragContext, sources) = await ragPipeline.BuildContextAsync(question.Question, ct);
        var ragPrompt = $"""
Context information is below.
---------------------
{ragContext}
---------------------
Given the context information and not prior knowledge, answer the following question:
{question.Question}
""";

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
            SourcesUsed = sources
        };
    }
}
