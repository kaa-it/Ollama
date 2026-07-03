using System.Text;

public class TaskMemoryService
{
    private readonly ILlmService _llm;
    public TaskState State { get; } = new();

    public TaskMemoryService(ILlmService llm)
    {
        _llm = llm;
    }

    public void Reset()
    {
        State.Goal = null;
        State.Constraints = [];
        State.Terms = [];
        State.ActiveTopic = null;
        State.ConfidenceInGoal = ConfidenceLevel.Unknown;
        State.LastUpdated = DateTime.UtcNow;
    }

    public async Task InferGoalFromMessageAsync(string userMessage, CancellationToken ct = default)
    {
        var prompt = $"This is the user's first message in a chat about Rust design patterns. " +
            $"Extract their overall goal in 1 sentence. If unclear, respond 'unknown'.\n\n" +
            $"User message: {userMessage}\n\nGoal:";

        try
        {
            var raw = await _llm.AskAsync(prompt, "You extract user goals from chat messages.", 256, ct);
            var goal = raw.Trim().Trim('"', '\'');
            if (!goal.Equals("unknown", StringComparison.OrdinalIgnoreCase) && !string.IsNullOrWhiteSpace(goal))
            {
                State.Goal = goal;
                State.ConfidenceInGoal = ConfidenceLevel.Medium;
                State.LastUpdated = DateTime.UtcNow;
            }
        }
        catch { /* ignore */ }
    }

    public async Task UpdateStateAsync(string userMessage, CitationAnswer assistantAnswer, CancellationToken ct = default)
    {
        if (assistantAnswer.Confidence == ConfidenceLevel.Unknown)
            return;

        var prompt = $"Analyze this dialog turn. Extract:\n" +
            $"GOAL: user's overall goal (1 sentence) or \"unknown\"\n" +
            $"CONFIDENCE: high/medium/low/unknown\n" +
            $"CONSTRAINTS: comma-separated list or \"none\"\n" +
            $"TERMS: term=definition pairs, comma-separated or \"none\"\n" +
            $"ACTIVE_TOPIC: current topic (1-3 words) or \"none\"\n\n" +
            $"User: {userMessage}\n" +
            $"Assistant: {assistantAnswer.Answer}\n\n" +
            $"Return pipe-delimited: GOAL|CONFIDENCE|CONSTRAINTS|TERMS|ACTIVE_TOPIC";

        string raw;
        try
        {
            raw = await _llm.AskAsync(prompt, "You are a dialog analyzer.", 512, ct);
        }
        catch
        {
            return;
        }

        var parts = raw.Split('|', 5);
        if (parts.Length >= 1)
        {
            var goal = parts[0].Trim();
            if (!goal.Equals("unknown", StringComparison.OrdinalIgnoreCase) && State.Goal == null)
                State.Goal = goal;
        }
        if (parts.Length >= 2)
        {
            var newConfidence = parts[1].Trim().ToLowerInvariant() switch
            {
                "high" => ConfidenceLevel.High,
                "medium" => ConfidenceLevel.Medium,
                "low" => ConfidenceLevel.Low,
                _ => ConfidenceLevel.Unknown
            };
            // Only overwrite if we got a meaningful value, preserve existing otherwise
            if (newConfidence != ConfidenceLevel.Unknown || State.ConfidenceInGoal == ConfidenceLevel.Unknown)
                State.ConfidenceInGoal = newConfidence;
        }
        if (parts.Length >= 3)
        {
            var constraintsText = parts[2].Trim();
            if (!constraintsText.Equals("none", StringComparison.OrdinalIgnoreCase))
            {
                var newConstraints = constraintsText.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).ToList();
                foreach (var c in newConstraints)
                {
                    if (!State.Constraints.Contains(c, StringComparer.OrdinalIgnoreCase))
                        State.Constraints.Add(c);
                }
            }
        }
        if (parts.Length >= 4)
        {
            var termsText = parts[3].Trim();
            if (!termsText.Equals("none", StringComparison.OrdinalIgnoreCase))
            {
                foreach (var pair in termsText.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
                {
                    var eqIdx = pair.IndexOf('=');
                    if (eqIdx > 0)
                    {
                        var key = pair[..eqIdx].Trim();
                        var value = pair[(eqIdx + 1)..].Trim();
                        if (!string.IsNullOrEmpty(key))
                            State.Terms[key] = value;
                    }
                }
            }
        }
        if (parts.Length >= 5)
        {
            var topic = parts[4].Trim();
            State.ActiveTopic = topic.Equals("none", StringComparison.OrdinalIgnoreCase) ? null : topic;
        }

        State.LastUpdated = DateTime.UtcNow;
    }

    public string BuildContextPrompt()
    {
        if (State.Goal == null)
            return "";

        var sb = new StringBuilder();
        sb.AppendLine("[TASK CONTEXT]");
        sb.AppendLine($"Goal: {State.Goal}");
        if (State.Constraints.Count > 0)
            sb.AppendLine($"Constraints: {string.Join(", ", State.Constraints)}");
        if (State.Terms.Count > 0)
            sb.AppendLine($"Terms: {string.Join(", ", State.Terms.Select(kv => $"{kv.Key}={kv.Value}"))}");
        if (State.ActiveTopic != null)
            sb.AppendLine($"Active Topic: {State.ActiveTopic}");
        return sb.ToString();
    }

    public bool NeedsGoalClarification() =>
        State.Goal == null || State.ConfidenceInGoal <= ConfidenceLevel.Low;

    public string GetStateSnapshot()
    {
        var sb = new StringBuilder();
        sb.AppendLine("=== Task State ===");
        sb.AppendLine($"Goal: {State.Goal ?? "(not set)"}");
        sb.AppendLine($"Confidence: {State.ConfidenceInGoal}");
        sb.AppendLine($"Constraints: {(State.Constraints.Count > 0 ? string.Join(", ", State.Constraints) : "(none)")}");
        sb.AppendLine($"Terms: {(State.Terms.Count > 0 ? string.Join(", ", State.Terms.Select(kv => $"{kv.Key}={kv.Value}")) : "(none)")}");
        sb.AppendLine($"Active Topic: {State.ActiveTopic ?? "(none)"}");
        sb.AppendLine($"Last Updated: {State.LastUpdated:O}");
        return sb.ToString();
    }
}
