namespace FoodOptimizer.Core;

/// <summary>D1 hand-off: the user's own shortcuts open their per-user copy,
/// which never cleans itself up. Opened on a PC where IT has installed the
/// all-users copy, it starts that copy and quits, before it holds any name;
/// the all-users copy then runs the per-user copy's own uninstaller
/// (PerUserCleanup). The all-users copy itself never hands off.</summary>
public static class ItCopy
{
    public static string Exe(string dir) => Path.Combine(dir, "FoodOptimizer.exe");

    /// <summary>The all-users copy's folder to hand off to, or null.
    /// Candidates in order: the all-users uninstall entry's InstallLocation
    /// (HKLM, which no user can write), then the default folder under
    /// Program Files. One qualifies only when it holds FoodOptimizer.exe and
    /// the all-users marker, is not this copy's own folder (however
    /// spelled), and no folder on its path is a link or has a name Win32
    /// would trim (PerUserCopy's rules).</summary>
    public static string? HandOffTarget(bool allUsers, string ownRoot, string? installLocation, string? defaultDir,
        Func<string, bool> fileExists, Func<string, bool> isReparsePoint)
    {
        if (allUsers) return null;
        string own;
        try { own = PerUserCopy.Normalize(ownRoot); }
        catch (Exception) { return null; }
        foreach (var candidate in new[] { installLocation, defaultDir })
        {
            if (string.IsNullOrWhiteSpace(candidate) || PerUserCopy.OddName(candidate)) continue;
            string dir;
            try { dir = PerUserCopy.Normalize(candidate); }
            catch (Exception) { continue; }
            if (string.Equals(dir, own, StringComparison.OrdinalIgnoreCase)) continue;
            if (PerUserCopy.OddName(dir) || PerUserCopy.OnALink(dir, isReparsePoint)) continue;
            if (fileExists(Path.Combine(dir, "FoodOptimizer.exe")) && fileExists(Path.Combine(dir, "all-users")))
                return dir;
        }
        return null;
    }
}
