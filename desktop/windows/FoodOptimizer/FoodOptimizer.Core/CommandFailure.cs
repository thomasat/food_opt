namespace FoodOptimizer.Core;

/// <summary>A failed menu command's reason. The dialog never shows it
/// (Windows' or .NET's text is not ours); shell.log keeps it.</summary>
public static class CommandFailure
{
    /// <summary>The exception's type and message, as shell.log keeps them.</summary>
    public static string Logged(Exception ex) => ex.GetType().Name + ": " + ex.Message;
}
