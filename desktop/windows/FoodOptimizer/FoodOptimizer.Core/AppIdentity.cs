using System.Reflection;

namespace FoodOptimizer.Core;

/// <summary>The names that identify Food Optimizer to Windows: the taskbar
/// group, the single-instance and installer mutexes, the activation event,
/// and the version (desktop/VERSION, stamped at build time).</summary>
public static class AppIdentity
{
    public const string AppUserModelId = "FoodSystemInnovations.FoodOptimizer";
    public const string SingleInstanceMutex = @"Local\FoodOptimizer.SingleInstance.3CC8CB77-0313-4A7D-9582-9896F0A3A8B3";
    public const string InstallerMutex = @"Global\FoodOptimizer.Running.3CC8CB77-0313-4A7D-9582-9896F0A3A8B3";
    public const string ActivateEvent = @"Local\FoodOptimizer.Activate.3CC8CB77-0313-4A7D-9582-9896F0A3A8B3";

    /// <summary>desktop/VERSION: every assembly of the shell carries it as
    /// its informational version (Directory.Build.props).</summary>
    public static string Version { get; } =
        typeof(AppIdentity).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()!.InformationalVersion;
}
