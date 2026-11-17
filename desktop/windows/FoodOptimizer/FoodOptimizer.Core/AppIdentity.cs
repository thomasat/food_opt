using System.Reflection;

namespace FoodOptimizer.Core;

/// <summary>The names that identify Food Optimizer to Windows: the taskbar
/// group, the single-instance mutex, the running-app names the installer
/// checks, the activation event, and the version (desktop/VERSION, stamped
/// at build time).</summary>
public static class AppIdentity
{
    public const string AppUserModelId = "FoodSystemInnovations.FoodOptimizer";
    public const string SingleInstanceMutex = @"Local\FoodOptimizer.SingleInstance.3CC8CB77-0313-4A7D-9582-9896F0A3A8B3";

    /// <summary>A per-user copy's running app: seen only in its own session
    /// (no other user can make a name there), so nobody else's app can hold
    /// up this user's install. setup.iss's RunningName checks it.</summary>
    public const string RunningPerUser = @"Local\FoodOptimizer.Running.PerUser.3CC8CB77-0313-4A7D-9582-9896F0A3A8B3";

    /// <summary>The all-users copy's running app, in every user's session:
    /// an IT upgrade (SYSTEM) waits for every user's app.</summary>
    public const string RunningAllUsers = @"Global\FoodOptimizer.Running.AllUsers.3CC8CB77-0313-4A7D-9582-9896F0A3A8B3";
    public const string ActivateEvent = @"Local\FoodOptimizer.Activate.3CC8CB77-0313-4A7D-9582-9896F0A3A8B3";

    /// <summary>desktop/VERSION: every assembly of the shell carries it as
    /// its informational version (Directory.Build.props).</summary>
    public static string Version { get; } =
        typeof(AppIdentity).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()!.InformationalVersion;
}
