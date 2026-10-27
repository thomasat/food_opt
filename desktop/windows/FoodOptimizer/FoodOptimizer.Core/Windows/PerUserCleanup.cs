using System.Diagnostics;
using System.Globalization;
using System.Runtime.Versioning;
using System.Security.Principal;
using Microsoft.Win32;

namespace FoodOptimizer.Core.Windows;

/// <summary>D1: started from an all-users install, the app looks for this
/// user's own per-user copy and runs that copy's own uninstaller, silently
/// and as this user. The all-users installer never deletes anything in a
/// user's profile. The uninstaller keeps the projects folder, and keeps
/// the state folder and the Quick access pin while an all-users copy is
/// installed (setup.iss).</summary>
[SupportedOSPlatform("windows")]
public static class PerUserCleanup
{
    /// <summary>The per-user uninstall entry Inno Setup writes (setup.iss AppId).</summary>
    public const string UninstallKey =
        @"Software\Microsoft\Windows\CurrentVersion\Uninstall\{DE876ED9-927A-45AD-AD9B-D6E4B4C9A4A3}_is1";

    static readonly TimeSpan ExitWait = TimeSpan.FromSeconds(120);

    /// <summary>Starts the per-user copy's uninstaller and returns at once;
    /// its exit code is logged from a background wait. Never throws.</summary>
    public static void Run(InstallLayout layout, ShellLog log)
    {
        if (!layout.AllUsers) return;
        try
        {
            if (!PerUserCopy.ShouldRun(layout.AllUsers, Elevated()))
            {
                log.Write("percopy.skipped", "elevated");
                return;
            }
            var dir = PerUserCopy.Choose(layout.Root, InstallLocation(), DefaultDir(), File.Exists, IsReparsePoint);
            if (dir == null)
            {
                log.Write("percopy.none");
                return;
            }
            var info = new ProcessStartInfo(PerUserCopy.Uninstaller(dir))
            {
                UseShellExecute = false,
                CreateNoWindow = true,
                // Not inside the folder the uninstaller deletes.
                WorkingDirectory = Path.GetTempPath(),
            };
            foreach (var a in PerUserCopy.UninstallArguments) info.ArgumentList.Add(a);
            var process = Process.Start(info);
            if (process == null)
            {
                log.Write("percopy.failed", "no_process");
                return;
            }
            log.Write("percopy.uninstall", dir);
            _ = Task.Run(() =>
            {
                try
                {
                    using (process)
                    {
                        if (process.WaitForExit(ExitWait))
                            log.Write("percopy.exit", process.ExitCode.ToString(CultureInfo.InvariantCulture));
                        else
                            log.Write("percopy.timeout");
                    }
                }
                catch (Exception e)
                {
                    log.Write("percopy.failed", e.GetType().Name);
                }
            });
        }
        catch (Exception e)
        {
            log.Write("percopy.failed", e.GetType().Name);
        }
    }

    /// <summary>This process runs with admin rights: SYSTEM, or an elevated
    /// administrator (a filtered UAC token is not in the role). Unknown
    /// counts as elevated.</summary>
    static bool Elevated()
    {
        try
        {
            using var id = WindowsIdentity.GetCurrent();
            return id.IsSystem || new WindowsPrincipal(id).IsInRole(WindowsBuiltInRole.Administrator);
        }
        catch (Exception)
        {
            return true;
        }
    }

    /// <summary>This user's uninstall entry's InstallLocation, read-only.</summary>
    static string? InstallLocation()
    {
        using var key = Registry.CurrentUser.OpenSubKey(UninstallKey);
        return key?.GetValue("InstallLocation") as string;
    }

    /// <summary>Where a per-user install goes by default ({userpf}\Food Optimizer).</summary>
    static string DefaultDir() =>
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Programs", PerUserCopy.FolderName);

    /// <summary>A link, or a folder that can't be looked at (so it can't be
    /// ruled out); a missing one isn't (the file checks fail for it).</summary>
    static bool IsReparsePoint(string path)
    {
        try { return (File.GetAttributes(path) & FileAttributes.ReparsePoint) != 0; }
        catch (Exception e) when (e is FileNotFoundException or DirectoryNotFoundException) { return false; }
        catch (Exception) { return true; }
    }
}
