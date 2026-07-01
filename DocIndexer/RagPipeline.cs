using System.Text.Json;

public class RagPipeline(
    OllamaEmbeddingService embeddingService,
    SqliteVectorStore vectorStore)
{
    private readonly int _topK = int.TryParse(Environment.GetEnvironmentVariable("RAG_TOP_K"), out var k) ? k : 3;

    public async Task<(string context, string[] sources)> BuildContextAsync(string question, CancellationToken ct = default)
    {
        var queryEmbedding = (await embeddingService.GenerateEmbeddingsAsync([$"search_query: {question}"], ct))[0];

        var chunks = await vectorStore.SearchSimilarAsync(queryEmbedding, _topK, ChunkingStrategy.Structural);
        var chunksList = chunks.ToList();

        var sources = chunksList
            .Select(c => c.Source)
            .Select(p =>
            {
                var idx = p.IndexOf("patterns/", StringComparison.OrdinalIgnoreCase);
                return idx >= 0 ? p[idx..] : Path.GetFileName(p);
            })
            .Distinct()
            .ToArray();

        var contextParts = chunksList.Select((c, i) =>
        {
            var sectionInfo = c.Section != null ? $"[Section: {c.Section}]" : "";
            return $"--- Source {i + 1}: {c.Title} {sectionInfo} ---\n{c.Content}";
        });

        var context = string.Join("\n\n", contextParts);
        return (context, sources);
    }
}
