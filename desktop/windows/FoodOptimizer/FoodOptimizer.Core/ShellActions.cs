namespace FoodOptimizer.Core;

/// <summary>What the launch machine asks the shell to do; the shell turns
/// each one into a page call, a timer or a navigation.</summary>
public abstract record ShellAction;

public sealed record PageStatusLine(string Raw) : ShellAction;
public sealed record PageServerStarted : ShellAction;
public sealed record PageShowStalled : ShellAction;
public sealed record PageShowError(PageError Kind, string? DetailId) : ShellAction;
public sealed record PageShowSteps(PageView View) : ShellAction;
public sealed record CheckHealth(Uri Uri, int Port, string Base) : ShellAction;
public sealed record Navigate(Uri Uri) : ShellAction;
public sealed record StartHealthWatch(Uri Uri) : ShellAction;
public sealed record StopHealthWatch : ShellAction;
public sealed record StopPolling : ShellAction;
public sealed record RecordSuccessfulOpen : ShellAction;
public sealed record StartLauncher : ShellAction;

/// <summary>One poll's reads of status.txt, server.port and server.base
/// (null: missing or unreadable).</summary>
public sealed record PollSnapshot(FileRead? Status, FileRead? Port, FileRead? Base);

public enum LaunchPhase { Waiting, Loaded, Failed, ShuttingDown }
