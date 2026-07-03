using System.Text.RegularExpressions;

/// <summary>
/// Simple .env file loader. Reads KEY=VALUE pairs and sets them as environment variables.
/// Supports: comments (#), empty lines, quoted values, inline comments.
/// Must be called early in Program.cs before any other code accesses Environment variables.
/// </summary>
public static class DotEnvLoader
{
    public static void Load(string filePath = ".env")
    {
        if (!File.Exists(filePath))
            return;

        foreach (var rawLine in File.ReadAllLines(filePath))
        {
            var line = rawLine.Trim();

            // Skip empty lines and comments
            if (string.IsNullOrWhiteSpace(line) || line.StartsWith('#'))
                continue;

            // Remove inline comments (but not inside quotes)
            line = RemoveInlineComment(line);

            var separatorIndex = line.IndexOf('=');
            if (separatorIndex <= 0)
                continue;

            var key = line[..separatorIndex].Trim();
            var value = line[(separatorIndex + 1)..].Trim();

            // Remove surrounding quotes
            value = Unquote(value);

            if (!string.IsNullOrEmpty(key))
            {
                Environment.SetEnvironmentVariable(key, value);
            }
        }
    }

    private static string RemoveInlineComment(string line)
    {
        var inSingleQuote = false;
        var inDoubleQuote = false;

        for (int i = 0; i < line.Length; i++)
        {
            var c = line[i];

            if (c == '\'' && !inDoubleQuote)
                inSingleQuote = !inSingleQuote;
            else if (c == '"' && !inSingleQuote)
                inDoubleQuote = !inDoubleQuote;
            else if (c == '#' && !inSingleQuote && !inDoubleQuote)
                return line[..i].TrimEnd();
        }

        return line;
    }

    private static string Unquote(string value)
    {
        if (value.Length >= 2)
        {
            if ((value.StartsWith('"') && value.EndsWith('"')) ||
                (value.StartsWith('\'') && value.EndsWith('\'')))
            {
                return value[1..^1];
            }
        }
        return value;
    }
}
