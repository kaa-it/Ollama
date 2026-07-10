using OllamaSharp;
using OllamaSharp.Models;

public class OllamaEmbeddingService(string host = "http://localhost:11434", string model = "nomic-embed-text") : IEmbeddingService
{
    private readonly OllamaApiClient _client = new(new Uri(host)) { SelectedModel = model };

    public async Task<IEnumerable<Model>> CheckAvailabilityAsync()
    {
        return await _client.ListLocalModelsAsync();
    }

    public async Task<float[]> GenerateQueryEmbeddingAsync(string query, CancellationToken ct = default)
    {
        var prefixedTexts = new[] { $"search_query: {query}" };
        var embeddings = await EmbedWithRetryAsync(prefixedTexts.ToList(), ct);
        if (embeddings[0].Length != 768)
            throw new InvalidOperationException($"Expected 768-dimensional embedding, got {embeddings[0].Length}");
        return embeddings[0];
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
