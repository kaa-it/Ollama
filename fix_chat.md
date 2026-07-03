Вот детальный промпт с найденными багами и точными указаниями по исправлениям.

---

## Промпт для OpenCoder: исправление багов в мини-чате

```
Проект DocIndexer, .NET 10, C# 10, top-level statements. Ты уже реализовал мини-чат (ChatService, ChatSession, TaskMemoryService, ChatScenarioTest, TaskState, ChatMessage, Program.cs, test-chat-scenarios.json). Сборка проходит без ошибок, но обнаружены КРИТИЧЕСКИЕ логические ошибки, мусорный код и runtime-проблемы. Исправь их. Нужен полный исходный код ВСЕХ новых файлов после исправлений. Файлы, которые не менялись (AnthropicLlmService, EvaluationEngine и т.д.), переписывать НЕ надо.

### 🔴 Критические баги (приводят к неверной работе)

#### 1. Дублирование текущего вопроса в LLM промпте — ChatService.cs
**Проблема:** В `RunInteractiveAsync` вызывается `session.AddUserMessage(trimmed)` ПЕРЕД `ProcessMessageAsync`. Внутри `ProcessMessageAsync` `session.GetHistoryContext(6)` возвращает это сообщение. Затем `PromptBuilder.BuildUserPrompt` добавляет `User question: {userMessage}`. LLM видит вопрос ДВАЖДЫ.

**Исправление:** В `ChatSession.GetHistoryContext` исключать последнее сообщение пользователя из возвращаемого контекста, если оно совпадает с текущим вопросом. ИЛИ: в `ChatService.ProcessMessageAsync` передавать `maxMessages` так, чтобы история содержала только предыдущие пары, без текущего вопроса. Самый простой способ — в `GetHistoryContext` возвращать `History.TakeLast(maxMessages).SkipLast(1)` когда последнее сообщение — User. Или проще: в `ProcessMessageAsync` не добавлять `User question:` из `BuildUserPrompt` в конец промпта, а использовать историю как единственный источник user-вопросов. Но лучше: в `ChatService.ProcessMessageAsync` строить `userPrompt` БЕЗ `BuildUserPrompt` — вместо этого передавать всё через history + system prompt. Однако это сломает существующий `PromptBuilder.BuildUserPrompt`. Поэтому самый безопасный фикс:

В `ChatService.ProcessMessageAsync`, строка ~52:
```csharp
var userPrompt = PromptBuilder.BuildUserPrompt(userMessage, ragResult.Chunks, ragResult.Confidence);
```
Замени `BuildUserPrompt` на inline-формирование промпта, где `userMessage` передаётся через history, а не через отдельный блок. Или проще: перед вызовом `GetHistoryContext(6)` временно не добавлять текущее сообщение в историю. Но история уже добавлена в `RunInteractiveAsync`.

**Самое простое и правильное исправление:**
В `ChatService.ProcessMessageAsync` замени:
```csharp
var systemPrompt = PromptBuilder.SystemPrompt + "\n\n" +
    _taskMemory.BuildContextPrompt() + "\n\n" +
    "[DIALOG HISTORY]\n" + session.GetHistoryContext(6);

var userPrompt = PromptBuilder.BuildUserPrompt(userMessage, ragResult.Chunks, ragResult.Confidence);
var fullPrompt = systemPrompt + "\n\n" + userPrompt;
```
на:
```csharp
var systemPrompt = PromptBuilder.SystemPrompt + "\n\n" +
    _taskMemory.BuildContextPrompt();

var history = session.GetHistoryContext(6);
// Исключаем текущий вопрос из истории, он будет в userPrompt
var historyWithoutLast = string.IsNullOrEmpty(history) ? "" : 
    string.Join("\n", history.Split('\n').Reverse().Skip(1).Reverse());

if (!string.IsNullOrEmpty(historyWithoutLast))
    systemPrompt += "\n\n[DIALOG HISTORY]\n" + historyWithoutLast;

var userPrompt = PromptBuilder.BuildUserPrompt(userMessage, ragResult.Chunks, ragResult.Confidence);
var fullPrompt = systemPrompt + "\n\n" + userPrompt;
```

#### 2. TaskState никогда не устанавливается через естественный диалог — ChatService.cs + TaskMemoryService.cs
**Проблема:** Когда `NeedsGoalClarification()` = true, `ProcessMessageAsync` return'ит на строке ~28-34 (Unknown-ответ) и НЕ доходит до `UpdateStateAsync` (строка 118). Пользователь отвечает "Learn Rust patterns", но `UpdateStateAsync` для этого сообщения НЕ извлекает goal, потому что она анализирует assistant-ответ, а не user-сообщение. Goal остаётся null.

**Исправление:**
В `ChatService.ProcessMessageAsync`, после блока "if isFirstTurn && NeedsGoalClarification":
```csharp
if (isFirstTurn && _taskMemory.NeedsGoalClarification())
{
    // Извлекаем goal прямо из первого сообщения пользователя
    await _taskMemory.InferGoalFromMessageAsync(userMessage, ct);
    
    if (_taskMemory.NeedsGoalClarification())
    {
        return new CitationAnswer(... clarification request ...);
    }
}
```

Добавь в `TaskMemoryService` новый метод:
```csharp
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
```

#### 3. TaskMemory не сбрасывается между сценариями — ChatScenarioTest.cs
**Проблема:** `ChatScenarioTest` использует один `_taskMemory` для всех сценариев. После первого сценария Terms, Constraints, ActiveTopic остаются и портят второй сценарий.

**Исправление:**
1. Добавь в `TaskMemoryService`:
```csharp
public void Reset()
{
    State.Goal = null;
    State.Constraints = [];
    State.Clarifications = [];
    State.Terms = [];
    State.ActiveTopic = null;
    State.ConfidenceInGoal = ConfidenceLevel.Unknown;
    State.LastUpdated = DateTime.UtcNow;
}
```

2. В `ChatScenarioTest.RunScenarioAsync`, сразу после `var session = new ChatSession(_taskMemory);` добавь:
```csharp
_taskMemory.Reset();
```

#### 4. Constraints и Terms перезаписываются вместо аккумуляции — TaskMemoryService.cs
**Проблема:** В `UpdateStateAsync`:
- `State.Constraints = constraintsText.Split(...).ToList();` — полная замена, старые теряются.
- `State.Terms.Clear();` — очищает все термины перед добавлением новых.

**Исправление:**
```csharp
// Constraints: merge, не replace
var newConstraints = constraintsText.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).ToList();
foreach (var c in newConstraints)
    if (!State.Constraints.Contains(c, StringComparer.OrdinalIgnoreCase))
        State.Constraints.Add(c);

// Terms: merge, не replace
foreach (var pair in termsText.Split(...))
{
    var eqIdx = pair.IndexOf('=');
    if (eqIdx > 0)
    {
        var key = pair[..eqIdx].Trim();
        var value = pair[(eqIdx + 1)..].Trim();
        if (!string.IsNullOrEmpty(key))
            State.Terms[key] = value; // Dictionary сам перезапишет или добавит
    }
}
// Убрать State.Terms.Clear();
```

### 🟡 Средние баги

#### 5. Дублирование systemPrompt в retry-loop — ChatService.cs
**Проблема:** `systemPrompt` строится до цикла (строки 50-52) и внутри цикла как `systemOnly` (строки 66-68). Одинаковый код.

**Исправление:** Вынести построение `systemOnly` за цикл for:
```csharp
var systemOnly = PromptBuilder.SystemPrompt + "\n\n" +
    _taskMemory.BuildContextPrompt() + "\n\n" +
    "[DIALOG HISTORY]\n" + session.GetHistoryContext(6);

for (int attempt = 1; attempt <= maxRetries; attempt++)
{
    var rawResponse = await _llm.AskAsync(userPrompt, systemOnly, MaxTokens, ct);
    // ...
}
```

#### 6. Мёртвое свойство Clarifications — TaskState.cs
**Проблема:** `Clarifications` нигде не используется. Мёртвый код.

**Исправление:** Удалить свойство `Clarifications` из `TaskState`. Удалить его упоминание из `GetStateSnapshot()`.

#### 7. ChatSession хранит TaskMemory но не использует — ChatSession.cs
**Проблема:** `public TaskMemoryService TaskMemory { get; }` — ни один метод ChatSession не обращается к нему. Лишняя связь.

**Исправление:** Убрать `TaskMemory` из конструктора и свойств `ChatSession`. В `ChatService.RunInteractiveAsync` и `ChatScenarioTest` создавать `ChatSession` без параметра.

#### 8. Неверная проверка goalPreserved — ChatScenarioTest.cs
**Проблема:** `goalPreserved = _taskMemory.State.Goal == firstGoal` — слишком строго. Goal может эволюционировать (уточняться).

**Исправление:**
```csharp
var goalPreserved = !string.IsNullOrEmpty(_taskMemory.State.Goal) 
    && _taskMemory.State.ConfidenceInGoal != ConfidenceLevel.Unknown;
```

#### 9. ExtractSafeQuote дублируется — ChatService.cs + ComparisonAgent.cs
**Проблема:** Идентичный метод `ExtractSafeQuote` существует и в `ChatService`, и в `ComparisonAgent`.

**Исправление:** Перенести `ExtractSafeQuote` в `CitationAnswerParser` как `public static string ExtractSafeQuote(string content, int maxLength)`. Удалить из `ChatService` и `ComparisonAgent`. Обновить вызовы.

### 📋 Требования к выходному коду

1. Все исправленные файлы — полный исходный код (ChatService.cs, ChatSession.cs, TaskMemoryService.cs, TaskState.cs, ChatScenarioTest.cs, Program.cs, test-chat-scenarios.json).
2. `Program.cs` должен оставаться с existing кодом (VectorMath, chunking strategies, vector store, indexing pipeline и т.д.) + local functions RunChatAsync/RunChatTestAsync. Ничего не удалять из Program.cs.
3. `dotnet build` должен проходить.
4. `ChatScenarioTest` должен запускаться: `dotnet run -- chat-test ./patterns`. Метрики должны показывать:
    - Goal Preserved: YES (оба сценария)
    - Sources Always Shown: YES
    - Citations Always Shown: YES
    - Unknown Answers: 0-2 на сценарий (допустимо)
5. Не используй namespace'ы. C# 10, target-typed new.

Выведи только исправленные файлы с полным содержимым.
```

____


Вот результат аудита реализации OpenCoder и промпт с оставшимися багами.

---

## 🔍 Результат аудита

**Сборка:** ✅ проходит без ошибок.

**Что исправлено хорошо:**
- Дублирование вопроса в LLM промпте — исправлено (`GetHistoryContext` пропускает последнее User-сообщение).
- `TaskMemory.Reset()` добавлен и используется между сценариями.
- `Clarifications` удалён из `TaskState` — мусор убран.
- `TaskMemory` убран из `ChatSession` — лишняя связь удалена.
- `ExtractSafeQuote` дедуплицирован в `CitationAnswerParser`.
- `Constraints` и `Terms` теперь merge вместо replace.
- `GoalPreserved` в тесте теперь проверяет наличие goal и не-Unknown confidence.

**Оставшиеся баги (2 штуки):**

---

## Промпт для OpenCoder

```
Проект DocIndexer, .NET 10, C# 10, top-level statements. Ты уже реализовал мини-чат (ChatService, ChatSession, TaskMemoryService, ChatScenarioTest, TaskState, ChatMessage). Сборка проходит, но обнаружены 2 оставшихся бага. Исправь их. Выведи полный исходный код ТОЛЬКО тех файлов, которые меняются. Не трогай Program.cs, EvaluationEngine, AnthropicLlmService и другие существующие файлы.

### 🔴 Баг 1: Инвертированная логика NeedsGoalClarification — TaskMemoryService.cs

**Файл:** TaskMemoryService.cs, строка 146

**Текущий код (НЕВЕРНЫЙ):**
```csharp
public bool NeedsGoalClarification() =>
    State.Goal == null || State.ConfidenceInGoal >= ConfidenceLevel.Low;
```

**Проблема:** `>= ConfidenceLevel.Low` означает "Low ИЛИ Medium ИЛИ High". То есть если goal установлен с confidence Medium, метод ВСЁ РАВНО возвращает true, и пользователь постоянно получает clarification request вместо ответа.

**ConfidenceLevel enum:**
```csharp
public enum ConfidenceLevel { Unknown = 0, Low = 1, Medium = 2, High = 3 }
```

**Исправление:** Замени `>=` на `<=`:
```csharp
public bool NeedsGoalClarification() =>
    State.Goal == null || State.ConfidenceInGoal <= ConfidenceLevel.Low;
```

Логика: clarification нужна только если goal не установлен ИЛИ confidence слишком низкий (Unknown или Low).

---

### 🔴 Баг 2: /reset не сбрасывает TaskMemory — ChatService.cs

**Файл:** ChatService.cs, метод RunInteractiveAsync, блок обработки команды `/reset`

**Текущий код (НЕПОЛНЫЙ):**
```csharp
if (trimmed.Equals("/reset", StringComparison.OrdinalIgnoreCase))
{
    session = new ChatSession();
    previousGoal = null;
    Console.ForegroundColor = ConsoleColor.Yellow;
    Console.WriteLine("Session reset.");
    Console.ResetColor();
    continue;
}
```

**Проблема:** Создаётся новый ChatSession, но TaskMemoryService (_taskMemory) НЕ сбрасывается. Goal, Constraints, Terms, ActiveTopic остаются от предыдущей сессии и портят новый диалог.

**Исправление:** Добавить `_taskMemory.Reset();` после `session = new ChatSession();`:
```csharp
if (trimmed.Equals("/reset", StringComparison.OrdinalIgnoreCase))
{
    session = new ChatSession();
    _taskMemory.Reset();
    previousGoal = null;
    Console.ForegroundColor = ConsoleColor.Yellow;
    Console.WriteLine("Session reset.");
    Console.ResetColor();
    continue;
}
```

---

### Требования

1. Выведи полный исходный код ИСПРАВЛЕННЫХ файлов: ChatService.cs и TaskMemoryService.cs.
2. Остальные файлы (ChatSession.cs, ChatScenarioTest.cs, TaskState.cs, Program.cs и т.д.) НЕ переписывай.
3. `dotnet build` должен проходить.
4. После исправлений `NeedsGoalClarification()` должен возвращать false для goal с confidence Medium/High.
5. После `/reset` TaskMemory должен быть чистым (Goal=null, Constraints=[], Terms=[]).

Выведи только два файла с полным содержимым.
```