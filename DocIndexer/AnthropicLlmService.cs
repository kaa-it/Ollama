using Anthropic.SDK;
using Anthropic.SDK.Messaging;

public interface ILlmService
{
    Task<string> AskAsync(string prompt, string? systemPrompt = null, CancellationToken ct = default);
}

public class AnthropicLlmService : ILlmService, IDisposable
{
    private readonly AnthropicClient _client;
    private readonly string _model;
    private readonly int _maxTokens;

    public AnthropicLlmService()
    {
        var apiKey = Environment.GetEnvironmentVariable("ANTHROPIC_API_KEY");
        if (string.IsNullOrEmpty(apiKey))
            throw new InvalidOperationException("ANTHROPIC_API_KEY environment variable is not set");

        _client = new AnthropicClient(apiKey);
        _model = Environment.GetEnvironmentVariable("ANTHROPIC_MODEL") ?? "claude-opus-4-5-20251101";
        _maxTokens = 1024;
    }

    public async Task<string> AskAsync(string prompt, string? systemPrompt = null, CancellationToken ct = default)
    {
        var messages = new List<Message>
        {
            new Message(RoleType.User, prompt)
        };

        var parameters = new MessageParameters
        {
            Messages = messages,
            MaxTokens = _maxTokens,
            Model = _model,
            Stream = false,
        };

        if (!string.IsNullOrEmpty(systemPrompt))
        {
            parameters.System = new List<SystemMessage>
            {
                new SystemMessage(systemPrompt)
            };
        }

        var response = await _client.Messages.GetClaudeMessageAsync(parameters, ct);
        return response.Message.ToString();
    }

    public void Dispose()
    {
        _client?.Dispose();
    }
}
