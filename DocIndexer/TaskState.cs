public record TaskState
{
    public string? Goal { get; set; }
    public List<string> Constraints { get; set; } = [];
    public Dictionary<string, string> Terms { get; set; } = [];
    public string? ActiveTopic { get; set; }
    public ConfidenceLevel ConfidenceInGoal { get; set; } = ConfidenceLevel.Unknown;
    public DateTime LastUpdated { get; set; } = DateTime.UtcNow;
}
