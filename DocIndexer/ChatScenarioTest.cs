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

    private async Task<List<ChatScenario>> LoadScenariosAsync(string scenariosPath, CancellationToken ct)
    {
        if (!File.Exists(scenariosPath))
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"❌ File {scenariosPath} not found.");
            Console.ResetColor();
            return [];
        }

        var json = await File.ReadAllTextAsync(scenariosPath, ct);
        var scenarios = JsonSerializer.Deserialize<List<ChatScenario>>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

        if (scenarios == null || scenarios.Count == 0)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("❌ No scenarios found.");
            Console.ResetColor();
            return [];
        }

        return scenarios;
    }

    private static string BoxLine(string content) => $"║{content.Truncate(75).PadRight(78)}║";

    public async Task RunScenariosAsync(string scenariosPath, CancellationToken ct = default)
    {
        var scenarios = await LoadScenariosAsync(scenariosPath, ct);
        if (scenarios.Count == 0) return;

        foreach (var scenario in scenarios)
        {
            _taskMemory.Reset();
            await RunScenarioAsync(scenario, ct);
        }
    }

    public async Task RunFirstScenarioVerboseAsync(string scenariosPath, int messageCount = 2, CancellationToken ct = default)
    {
        var scenarios = await LoadScenariosAsync(scenariosPath, ct);
        if (scenarios.Count == 0) return;

        var scenario = scenarios[0];
        _taskMemory.Reset();

        var session = new ChatSession();

        if (!string.IsNullOrEmpty(scenario.InitialGoal))
        {
            _taskMemory.State.Goal = scenario.InitialGoal;
            _taskMemory.State.ConfidenceInGoal = ConfidenceLevel.High;
        }

        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("\n" + "╔" + new string('═', 78) + "╗");
        Console.WriteLine(BoxLine($"  CHAT TEST 2 — Verbose: {scenario.Name}"));
        Console.WriteLine(BoxLine($"  First {messageCount} messages only"));
        Console.WriteLine("╚" + new string('═', 78) + "╝");
        Console.ResetColor();

        var limitedMessages = scenario.Messages.Take(messageCount).ToList();

        if (limitedMessages.Count == 0)
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("⚠️  No messages to process.");
            Console.ResetColor();
            return;
        }

        for (int i = 0; i < limitedMessages.Count; i++)
        {
            if (ct.IsCancellationRequested) break;

            var msg = limitedMessages[i];

            session.AddUserMessage(msg);

            try
            {
                var answer = await _chatService.ProcessMessageAsync(msg, session, ct);
                session.AddAssistantMessage(answer);

                Console.WriteLine();
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine($"─── Question [{i + 1}/{limitedMessages.Count}] ───");
                Console.ResetColor();
                Console.WriteLine(msg);
                Console.WriteLine();

                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine($"─── Answer (Confidence: {answer.Confidence}) ───");
                Console.ResetColor();
                Console.WriteLine(answer.Answer);
                Console.WriteLine();

                if (answer.ClarificationRequest != null)
                {
                    Console.ForegroundColor = ConsoleColor.DarkYellow;
                    Console.WriteLine($"─── Clarification Request ───");
                    Console.ResetColor();
                    Console.WriteLine(answer.ClarificationRequest);
                    Console.WriteLine();
                }

                if (answer.Sources.Count > 0)
                {
                    Console.ForegroundColor = ConsoleColor.DarkGray;
                    Console.WriteLine($"─── Sources ({answer.Sources.Count}) ───");
                    foreach (var src in answer.Sources)
                    {
                        Console.WriteLine($"  [{src.ChunkIndex}] {src.Source}{(src.Section != null ? $" ({src.Section})" : "")} (score: {src.RelevanceScore:F3})");
                    }
                    Console.ResetColor();
                    Console.WriteLine();
                }

                if (answer.Citations.Count > 0)
                {
                    Console.ForegroundColor = ConsoleColor.DarkGray;
                    Console.WriteLine($"─── Citations ({answer.Citations.Count}) ───");
                    foreach (var cit in answer.Citations)
                    {
                        var quotePreview = cit.Quote.Length > 200 ? cit.Quote[..200] + "..." : cit.Quote;
                        Console.WriteLine($"  [{cit.SourceIndex}] \"{quotePreview}\"");
                    }
                    Console.ResetColor();
                    Console.WriteLine();
                }

                Console.WriteLine(new string('═', 80));
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"  [{i + 1}/{limitedMessages.Count}] Error: {ex.Message}");
                Console.ResetColor();
            }
        }

        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("╔" + new string('═', 78) + "╗");
        Console.WriteLine(BoxLine("  CHAT TEST 2 REPORT"));
        Console.WriteLine("╠" + new string('═', 78) + "╣");
        Console.WriteLine(BoxLine($" Scenario: {scenario.Name}"));
        Console.WriteLine(BoxLine($" Messages processed: {limitedMessages.Count}"));
        Console.WriteLine(BoxLine($" Goal: {_taskMemory.State.Goal ?? "(not set)"}"));
        Console.WriteLine("╚" + new string('═', 78) + "╝");
        Console.ResetColor();
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
        Console.WriteLine("\n╔" + new string('═', 78) + "╗");
        Console.WriteLine(BoxLine("  CHAT SCENARIO TEST REPORT"));
        Console.WriteLine("╠" + new string('═', 78) + "╣");
        Console.WriteLine(BoxLine($" Scenario: {scenario.Name}"));
        Console.WriteLine(BoxLine($" Messages: {scenario.Messages.Count}"));
        Console.WriteLine(BoxLine($" Goal Preserved: {(goalPreserved ? "YES" : "NO")}"));
        Console.WriteLine(BoxLine($" Sources Always Shown: {(sourcesAlwaysShown ? "YES" : "NO")}"));
        Console.WriteLine(BoxLine($" Citations Always Shown: {(citationsAlwaysShown ? "YES" : "NO")}"));
        Console.WriteLine(BoxLine($" Unknown Answers: {unknownCount}"));
        Console.WriteLine(BoxLine($" Avg Response Length: {avgLength} chars"));
        Console.WriteLine("╚" + new string('═', 78) + "╝");
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
