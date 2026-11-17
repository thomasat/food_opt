using System.Diagnostics;
using System.Runtime.Versioning;
using Microsoft.Win32;

namespace FoodOptimizer.Core.Windows;

/// <summary>D1 hand-off: a per-user copy opened on a PC with the all-users
/// (IT) copy starts that copy and tells the caller to quit, before any
/// name is held: the IT copy's cleanup then runs this copy's uninstaller,
/// which checks this copy's running name (setup.iss AppClosed).</summary>
[SupportedOSPlatform("windows")]
public static class ItCopyHandoff
{
    /// <summary>True when the IT copy was started (the caller exits 0).
    /// Never from the IT copy itself. Never throws.</summary>
    public static bool Run(InstallLayout layout, ShellLog log)
    {
        if (layout.AllUsers) return false;
        try
        {
            var dir = ItCopy.HandOffTarget(layout.AllUsers, layout.Root, InstallLocation(), DefaultDir(), File.Exists, PerUserCleanup.IsReparsePoint);
            if (dir == null) return false;
            log.Write("percopy.handoff", dir);
            // No shell execute and no arguments: this exe, in its own folder.
            var info = new ProcessStartInfo(ItCopy.Exe(dir))
            {
                UseShellExecute = false,
                WorkingDirectory = dir,
            };
            using var process = Process.Start(info);
            if (process != null) return true;
            log.Write("percopy.handoff_failed", "no_process");
        }
        catch (Exception e)
        {
            log.Write("percopy.handoff_failed", e.GetType().Name);
        }
        return false;
    }

    /// <summary>The all-users uninstall entry's InstallLocation, read-only.
    /// The shell is x64, so this is the 64-bit view (setup.iss HKLM64).</summary>
    static string? InstallLocation()
    {
        using var key = Registry.LocalMachine.OpenSubKey(PerUserCleanup.UninstallKey);
        return key?.GetValue("InstallLocation") as string;
    }

    /// <summary>Where an all-users install goes by default
    /// ({commonpf64}\Food Optimizer).</summary>
    static string? DefaultDir()
    {
        var pf = Environment.GetEnvironmentVariable("ProgramW6432");
        return string.IsNullOrEmpty(pf) ? null : Path.Combine(pf, PerUserCopy.FolderName);
    }
}
