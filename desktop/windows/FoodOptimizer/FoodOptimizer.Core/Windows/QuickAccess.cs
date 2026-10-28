using System.Runtime.InteropServices;
using System.Runtime.Versioning;

namespace FoodOptimizer.Core.Windows;

/// <summary>D2: pins a folder to File Explorer's Quick access through the
/// Shell's own verb (no admin). Call it on an STA thread.</summary>
[SupportedOSPlatform("windows")]
public static class QuickAccess
{
    const string Home = "shell:::{679f85cb-0220-4080-b29b-5540cc05aab6}";

    /// <summary>How long a pin may take to show in Quick access.</summary>
    static readonly TimeSpan ListWait = TimeSpan.FromSeconds(2);
    static readonly TimeSpan ListPoll = TimeSpan.FromMilliseconds(100);

    /// <summary>Pins the folder unless Quick access lists it already, then
    /// looks again: Pinned only when this call pinned it and it is listed
    /// now. Logs quickaccess.pinned, listed, not_listed or failed. Each
    /// look uses a Shell object of its own, so none outlives an await.
    /// Never throws.</summary>
    public static async Task<PinOutcome> PinAsync(string dir, ShellLog log)
    {
        try
        {
            switch (WithShell(s => Listed(s, dir)))
            {
                case null:
                    log.Write("quickaccess.failed", "no_shell");
                    return PinOutcome.Failed;
                case true:
                    log.Write("quickaccess.listed", dir);
                    return PinOutcome.AlreadyListed;
            }
            var invoked = WithShell(s =>
            {
                dynamic? folder = s.Namespace(dir);
                if (folder == null) return false;
                folder.Self.InvokeVerb("pintohome");
                return true;
            });
            if (invoked != true)
            {
                log.Write("quickaccess.failed", invoked == null ? "no_shell" : "no_folder");
                return PinOutcome.Failed;
            }
            for (var waited = TimeSpan.Zero; ; waited += ListPoll)
            {
                if (WithShell(s => Listed(s, dir)) == true)
                {
                    log.Write("quickaccess.pinned", dir);
                    return PinOutcome.Pinned;
                }
                if (waited >= ListWait) break;
                await Task.Delay(ListPoll);
            }
            log.Write("quickaccess.not_listed", dir);
            return PinOutcome.NotListed;
        }
        catch (Exception e)
        {
            log.Write("quickaccess.failed", e.GetType().Name);
            return PinOutcome.Failed;
        }
    }

    /// <summary>f on a new Shell.Application, released after; null when
    /// there is no Shell.</summary>
    static bool? WithShell(Func<dynamic, bool> f)
    {
        object? shell = null;
        try
        {
            var type = Type.GetTypeFromProgID("Shell.Application");
            if (type == null || (shell = Activator.CreateInstance(type)) == null) return null;
            return f(shell);
        }
        finally
        {
            if (shell != null && Marshal.IsComObject(shell)) Marshal.FinalReleaseComObject(shell);
        }
    }

    static bool Listed(dynamic shell, string dir)
    {
        dynamic? home = shell.Namespace(Home);
        if (home == null) return false;
        var want = Path.TrimEndingDirectorySeparator(dir);
        // By index, as Explorer's own scripts do: enumerating a late-bound
        // FolderItems is not something every runtime offers.
        dynamic items = home.Items();
        int count = items.Count;
        for (var i = 0; i < count; i++)
        {
            string? path = items.Item(i).Path;
            if (path != null && string.Equals(Path.TrimEndingDirectorySeparator(path), want, StringComparison.OrdinalIgnoreCase))
                return true;
        }
        return false;
    }
}
