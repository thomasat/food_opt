namespace FoodOptimizer.Core;

/// <summary>D1: the all-users app finds this user's own per-user copy and
/// hands it to that copy's uninstaller (PerUserCleanup). The all-users
/// installer never deletes anything in a per-user location.</summary>
public static class PerUserCopy
{
    /// <summary>The uninstaller's switches: no window, no box, no restart.</summary>
    public static readonly IReadOnlyList<string> UninstallArguments = ["/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"];

    /// <summary>The install folder's name (setup.iss DefaultDirName, under
    /// {userpf} for a per-user install).</summary>
    public const string FolderName = "Food Optimizer";

    public static string Uninstaller(string dir) => Path.Combine(dir, "unins000.exe");

    /// <summary>The folder whose unins000.exe to run, or null. Candidates in
    /// order: the user's uninstall entry's InstallLocation, then the default
    /// per-user folder. One qualifies only when it is not this copy's own
    /// folder (however spelled), no folder on its path is a link or has a
    /// name Win32 would trim (a trailing space or dot), and it holds
    /// FoodOptimizer.exe, unins000.exe and unins000.dat.</summary>
    // TODO(signing): also require unins000.exe's Authenticode signer = Food System Innovations.
    public static string? Choose(string ownRoot, string? installLocation, string defaultDir,
        Func<string, bool> fileExists, Func<string, bool> isReparsePoint)
    {
        var own = Normalize(ownRoot);
        foreach (var candidate in new[] { installLocation, defaultDir })
        {
            if (string.IsNullOrWhiteSpace(candidate) || OddName(candidate)) continue;
            string dir;
            try { dir = Normalize(candidate); }
            catch (Exception) { continue; }
            if (own != null && string.Equals(dir, own, StringComparison.OrdinalIgnoreCase)) continue;
            if (OddName(dir) || OnALink(dir, isReparsePoint)) continue;
            if (new[] { "FoodOptimizer.exe", "unins000.exe", "unins000.dat" }
                .All(f => fileExists(Path.Combine(dir, f))))
                return dir;
        }
        return null;
    }

    static string Normalize(string path) => Path.TrimEndingDirectorySeparator(Path.GetFullPath(path));

    /// <summary>A folder name ending in a space or dot ("." and ".." aside):
    /// Win32 drops those from a path's last part but keeps them in a middle
    /// one, so the folder checked and the folder run from could differ.</summary>
    static bool OddName(string path) =>
        path.Split('\\', '/').Any(part => part is not ("." or "..") && part.Length > 0
            && (part[^1] == ' ' || part[^1] == '.'));

    /// <summary>Whether the folder or any folder above it is a junction or
    /// other link: one could lead to a folder that is not this user's.</summary>
    static bool OnALink(string dir, Func<string, bool> isReparsePoint)
    {
        for (var p = dir; !string.IsNullOrEmpty(p); p = Path.GetDirectoryName(p))
            if (isReparsePoint(p)) return true;
        return false;
    }
}
