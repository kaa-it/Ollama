public class ComparisonAgent(
    ILlmService llmService,
    EnhancedRagPipeline enhancedRag)
{
    public async Task<(string answer, RagResult ragResult)> AskWithRagAsync(
        TestQuestion question,
        RagPipelineMode mode,
        CancellationToken ct = default)
    {
        var ragSystemPrompt = "You are an expert in Rust design patterns. Answer based ONLY on the provided context. If the context doesn't contain the answer, say so.";

        var ragResult = await enhancedRag.ExecuteAsync(question.Question, mode, ct);

        var ragPrompt = "Context information is below.\n---------------------\n" + ragResult.Context +
            "\n---------------------\nGiven the context information and not prior knowledge, answer the following question:\n" + question.Question;

        var answer = await llmService.AskAsync(ragPrompt, ragSystemPrompt, ct);
        return (answer, ragResult);
    }
}
