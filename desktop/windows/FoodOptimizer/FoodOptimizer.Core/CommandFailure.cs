using System.ComponentModel;

namespace FoodOptimizer.Core;

/// <summary>What a failed menu command may tell the user: only a failure
/// that came from the OS (a file, a folder, ShellExecute) has a reason
/// worth showing. A code error is never shown as Windows' text.</summary>
public static class CommandFailure
{
    /// <summary>The OS's message, or null when the failure is not the OS's.</summary>
    public static string? Reason(Exception ex) =>
        ex is IOException or UnauthorizedAccessException or Win32Exception ? ex.Message : null;
}
