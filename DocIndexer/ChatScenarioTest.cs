using System.Text.Json;

public class ChatScenarioTest
{
    private readonly ChatService _chatService;
    private readonly TaskMemoryService _taskMemory;

    public ChatScenarioTest(ChatService chatService, TaskMemoryService taskMemory)
    {
        _chatService = chatService;
        _taskMemory = taskMemory;
    }

    public async Task RunScenariosAsync(string scenariosPath, CancellationToken ct = default)
    {
        if (!File.Exists(scenariosPath))
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"❌ File {scenariosPath} not found.");
            Console.ResetColor();
            return;
        }

        var json = await File.ReadAllTextAsync(scenariosPath, ct);
        var scenarios = JsonSerializer.Deserialize<List<ChatScenario>>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

        if (scenarios == null || scenarios.Count == 0)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("❌ No scenarios found.");
            Console.ResetColor();
            return;
        }

        foreach (var scenario in scenarios)
        {
            _taskMemory.Reset();
            await RunScenarioAsync(scenario, ct);
        }
    }

    private async Task RunScenarioAsync(ChatScenario scenario, CancellationToken ct)
    {
        var session = new ChatSession();

        if (!string.IsNullOrEmpty(scenario.InitialGoal))
        {
            _taskMemory.State.Goal = scenario.InitialGoal;
            _taskMemory.State.ConfidenceInGoal = ConfidenceLevel.High;
        }

        var sourcesAlwaysShown = true;
        var citationsAlwaysShown = true;
        var unknownCount = 0;
        var totalLength = 0;

        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine($"\n--- Scenario: {scenario.Name} ({scenario.Messages.Count} messages) ---");
        Console.ResetColor();

        for (int i = 0; i < scenario.Messages.Count; i++)
        {
            if (ct.IsCancellationRequested) break;

            var msg = scenario.Messages[i];

            session.AddUserMessage(msg);

            try
            {
                var answer = await _chatService.ProcessMessageAsync(msg, session, ct);
                session.AddAssistantMessage(answer);

                if (answer.Confidence == ConfidenceLevel.Unknown)
                    unknownCount++;

                if (answer.Confidence != ConfidenceLevel.Unknown)
                {
                    if (answer.Sources.Count == 0)
                        sourcesAlwaysShown = false;
                    if (answer.Citations.Count == 0)
                        citationsAlwaysShown = false;
                }

                totalLength += answer.Answer.Length;

                Console.Write($"  [{i + 1}/{scenario.Messages.Count}] {msg.Truncate(60)} → ");
                Console.ForegroundColor = answer.Confidence switch
                {
                    ConfidenceLevel.High => ConsoleColor.Green,
                    ConfidenceLevel.Medium => ConsoleColor.Cyan,
                    ConfidenceLevel.Low => ConsoleColor.Yellow,
                    _ => ConsoleColor.Red
                };
                Console.WriteLine($"{answer.Confidence} ({answer.Answer.Length} chars, {answer.Sources.Count} sources, {answer.Citations.Count} citations)");
                Console.ResetColor();
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"  [{i + 1}/{scenario.Messages.Count}] Error: {ex.Message}");
                Console.ResetColor();
                sourcesAlwaysShown = false;
                citationsAlwaysShown = false;
            }
        }

        var goalPreserved = !string.IsNullOrEmpty(_taskMemory.State.Goal)
            && _taskMemory.State.ConfidenceInGoal != ConfidenceLevel.Unknown;

        var avgLength = scenario.Messages.Count > 0 ? totalLength / scenario.Messages.Count : 0;

        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("\n╔══════════════════════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║                    CHAT SCENARIO TEST REPORT                                 ║");
        Console.WriteLine("╠══════════════════════════════════════════════════════════════════════════════╣");
        Console.WriteLine($"║ Scenario: {scenario.Name,-59}║");
        Console.WriteLine($"║ Messages: {scenario.Messages.Count,-71}║");
        Console.WriteLine($"║ Goal Preserved:        {(goalPreserved ? "YES" : "NO"),-52}║");
        Console.WriteLine($"║ Sources Always Shown:  {(sourcesAlwaysShown ? "YES" : "NO"),-52}║");
        Console.WriteLine($"║ Citations Always Shown: {(citationsAlwaysShown ? "YES" : "NO"),-52}║");
        Console.WriteLine($"║ Unknown Answers:       {unknownCount,-52}║");
        Console.WriteLine($"║ Avg Response Length:   {avgLength,-5} chars                                       ║");
        Console.WriteLine("╚══════════════════════════════════════════════════════════════════════════════╝");
        Console.ResetColor();
    }

    public class ChatScenario
    {
        public string Name { get; set; } = "";
        public string? InitialGoal { get; set; }
        public List<string> Messages { get; set; } = [];
    }
}

public static class StringExtensions
{
    public static string Truncate(this string s, int maxLen)
    {
        if (s.Length <= maxLen) return s;
        return s[..maxLen] + "...";
    }
}
