using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;

namespace FoodOptimizer.Core.Windows;

/// <summary>Calls into the Windows shell: folders, Explorer, sounds, the
/// taskbar identity and the web mark on saved files.</summary>
[SupportedOSPlatform("windows")]
public static class WinShell
{
    public static string? DownloadsFolder()
    {
        if (NativeMethods.SHGetKnownFolderPath(NativeMethods.FOLDERID_Downloads, 0, IntPtr.Zero, out var p) != 0)
            return null;
        try { return Marshal.PtrToStringUni(p); }
        finally { NativeMethods.CoTaskMemFree(p); }
    }

    /// <summary>Opens the file's folder in Explorer with the file selected.</summary>
    public static bool RevealInExplorer(string path)
    {
        if (!File.Exists(path) && !Directory.Exists(path)) return false;
        if (NativeMethods.SHParseDisplayName(Path.GetFullPath(path), IntPtr.Zero, out var pidl, 0, out _) != 0)
            return false;
        try { return NativeMethods.SHOpenFolderAndSelectItems(pidl, 0, IntPtr.Zero, 0) == 0; }
        finally { NativeMethods.CoTaskMemFree(pidl); }
    }

    public static void PlayNotificationSound() =>
        NativeMethods.PlaySoundW("Notification.Default", IntPtr.Zero,
            NativeMethods.SND_ALIAS | NativeMethods.SND_ASYNC | NativeMethods.SND_NODEFAULT | NativeMethods.SND_SYSTEM);

    public static void SetAppUserModelId(string id) => NativeMethods.SetCurrentProcessExplicitAppUserModelID(id);

    /// <summary>Deletes the file's Zone.Identifier stream; true when it is
    /// gone or was never there.</summary>
    public static bool RemoveZoneIdentifier(string path)
    {
        if (NativeMethods.DeleteFileW(DownloadRules.ZoneIdentifierPath(path))) return true;
        return Marshal.GetLastPInvokeError() == NativeMethods.ERROR_FILE_NOT_FOUND;
    }

    /// <summary>Opens a file, folder or web address the way Explorer would.</summary>
    public static bool Open(string target)
    {
        try
        {
            using var p = Process.Start(new ProcessStartInfo(target) { UseShellExecute = true });
            return true;
        }
        catch (Exception)
        {
            return false;
        }
    }
}
