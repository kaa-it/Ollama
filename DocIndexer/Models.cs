using System.Text.Json.Serialization;

public record TestQuestion
{
    public required int Id { get; init; }
    public required string Question { get; init; }
    [JsonPropertyName("expected_answer")]
    public string? ExpectedAnswer { get; init; }
    [JsonPropertyName("expected_sources")]
    public string[]? ExpectedSources { get; init; }
    public string? Difficulty { get; init; }
    [JsonPropertyName("key_concepts")]
    public string[]? KeyConcepts { get; init; }
}
