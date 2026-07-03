public interface IQueryRewriteService
{
    Task<string> RewriteAsync(string query, CancellationToken ct = default);
}

public class HeuristicQueryRewriteService : IQueryRewriteService
{
    public Task<string> RewriteAsync(string query, CancellationToken ct = default)
    {
        var words = query.Split([' ', '\t', '\n', '?', '!', '.', ','], StringSplitOptions.RemoveEmptyEntries);
        var lowerWords = words.Select(w => w.ToLowerInvariant()).ToHashSet();

        var rewritten = query;

        if (!lowerWords.Contains("rust"))
            rewritten += " Rust";
        if (!lowerWords.Contains("pattern") && !lowerWords.Contains("idiom") && !lowerWords.Contains("anti-pattern"))
            rewritten += " pattern";

        return Task.FromResult(rewritten.Trim());
    }
}

public class LlmQueryRewriteService(ILlmService llm) : IQueryRewriteService
{
    public async Task<string> RewriteAsync(string query, CancellationToken ct = default)
    {
        var prompt = "Rewrite the following question into a concise search query optimized for retrieving Rust design patterns documentation. Keep only key technical terms and concepts. Output ONLY the rewritten query, nothing else.\n\nQuestion: " + query + "\nSearch query:";

        var rewritten = await llm.AskAsync(prompt, "You are a search query optimization assistant.", ct: ct);
        var result = rewritten.Trim().Trim('"', '\'');
        return string.IsNullOrWhiteSpace(result) ? query : result;
    }
}
