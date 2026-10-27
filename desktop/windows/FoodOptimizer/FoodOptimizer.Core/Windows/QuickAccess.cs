using System.Runtime.InteropServices;
using System.Runtime.Versioning;

namespace FoodOptimizer.Core.Windows;

/// <summary>D2: pins a folder to File Explorer's Quick access through the
/// Shell's own verb (no admin). Call it on an STA thread.</summary>
[SupportedOSPlatform("windows")]
public static class QuickAccess
{
    const string Home = "shell:::{679f85cb-0220-4080-b29b-5540cc05aab6}";

    /// <summary>True when the folder is pinned now or was already listed;
    /// logs quickaccess.pinned, listed or failed. Never throws.</summary>
    public static bool Pin(string dir, ShellLog log)
    {
        object? shell = null;
        try
        {
            var type = Type.GetTypeFromProgID("Shell.Application");
            if (type == null || (shell = Activator.CreateInstance(type)) == null)
            {
                log.Write("quickaccess.failed", "no_shell");
                return false;
            }
            dynamic s = shell;
            if (Listed(s, dir))
            {
                log.Write("quickaccess.listed", dir);
                return true;
            }
            dynamic? folder = s.Namespace(dir);
            if (folder == null)
            {
                log.Write("quickaccess.failed", "no_folder");
                return false;
            }
            folder.Self.InvokeVerb("pintohome");
            log.Write("quickaccess.pinned", dir);
            return true;
        }
        catch (Exception e)
        {
            log.Write("quickaccess.failed", e.GetType().Name);
            return false;
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
