public enum RagPipelineMode
{
    Baseline,
    WithThreshold,
    WithReranker,
    FullPipeline
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
    int TopKPost,
    int FilteredCount
);

public class EnhancedRagPipeline
{
    private readonly OllamaEmbeddingService _embeddingService;
    private readonly SqliteVectorStore _vectorStore;
    private readonly IQueryRewriteService? _rewriteService;
    private readonly SimilarityThresholdFilter _thresholdFilter;
    private readonly HeuristicReranker _reranker;
    private readonly int _topKPre;
    private readonly int _topKPost;
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
        var filteredCount = 0;
        if (mode != RagPipelineMode.Baseline)
        {
            filtered = _thresholdFilter.Filter(rawResultsList);
            filteredCount = filtered.Count;
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
            .Select(p =>
            {
                var normalized = p.Replace('\\', '/');
                var idx = normalized.IndexOf("patterns/", StringComparison.OrdinalIgnoreCase);
                return idx >= 0 ? normalized[idx..] : Path.GetFileName(p);
            })
            .Distinct()
            .ToArray();

        return new RagResult(context, sources, originalQuestion, rewrittenQuestion, finalChunks, mode, _thresholdFilter.Threshold, _topKPre, _topKPost, filteredCount);
    }

    private static int GetEnvInt(string name, int defaultValue) =>
        int.TryParse(Environment.GetEnvironmentVariable(name), out var v) ? v : defaultValue;
    private static float GetEnvFloat(string name, float defaultValue) =>
        float.TryParse(Environment.GetEnvironmentVariable(name), out var v) ? v : defaultValue;
    private static bool GetEnvBool(string name, bool defaultValue) =>
        bool.TryParse(Environment.GetEnvironmentVariable(name), out var v) ? v : defaultValue;
}
