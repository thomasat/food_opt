namespace FoodOptimizer.Core;

/// <summary>shell.log never holds the server's base path (M3): it is the
/// per-launch secret in the app's address. Every address the shell logs
/// passes through <see cref="Redact"/> with the current base.</summary>
public static class LogRedaction
{
    /// <summary>What stands in for the base.</summary>
    public const string Mark = "…";

    /// <summary>A shorter base would blank ordinary parts of an address
    /// (the launcher's bases are 32 characters).</summary>
    public const int MinBaseLength = 8;

    public static string Redact(string text, string? basePath) =>
        string.IsNullOrEmpty(text) || basePath is null || basePath.Length < MinBaseLength
            ? text
            : text.Replace(basePath, Mark, StringComparison.Ordinal);
}
