// Additional models for RAG comparison agent

public record TestQuestion
{
    public required int Id { get; init; }
    public required string Question { get; init; }
    public string? ExpectedAnswer { get; init; }
    public string[]? ExpectedSources { get; init; }
    public string? Difficulty { get; init; }
}

public record ComparisonResult
{
    public required TestQuestion Question { get; init; }
    public required string AnswerWithoutRag { get; init; }
    public required string AnswerWithRag { get; init; }
    public required long TimeWithoutRagMs { get; init; }
    public required long TimeWithRagMs { get; init; }
    public required string[] SourcesUsed { get; init; }
}

public record RagContext
{
    public required string Title { get; init; }
    public string? Section { get; init; }
    public required string Content { get; init; }
}
