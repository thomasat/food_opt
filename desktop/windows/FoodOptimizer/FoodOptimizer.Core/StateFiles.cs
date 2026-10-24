using System.Text;
using System.Text.RegularExpressions;

namespace FoodOptimizer.Core;

/// <summary>A file's text and its last-write time (UTC), read together.</summary>
public sealed record FileRead(string Text, DateTime ModifiedUtc);

/// <summary>server.port: the port, and the server's pid when it is written.</summary>
public sealed record PortInfo(int Port, int? Pid);

/// <summary>The files the launcher and the shell keep in the state folder,
/// and how the shell reads the launcher's ones: shared, so the launcher can
/// replace them while the shell reads, and retried past brief locks
/// (Defender and the search indexer hold files for a moment).</summary>
public static partial class StateFiles
{
    public const string Port = "server.port", Base = "server.base", Status = "status.txt",
        StatusTmp = "status.txt.tmp", LauncherLog = "launcher.log", Settings = "settings.json",
        ShellLog = "shell.log", WebViewData = "WebView2";

    /// <summary>The file's text and mtime, or null when it is missing or
    /// stays locked. Never throws.</summary>
    public static FileRead? ReadShared(string path, int attempts = 3, int delayMs = 20)
    {
        for (var i = 0; i < Math.Max(1, attempts); i++)
        {
            try
            {
                using var fs = new FileStream(path, FileMode.Open, FileAccess.Read,
                    FileShare.ReadWrite | FileShare.Delete);
                var modified = File.GetLastWriteTimeUtc(fs.SafeFileHandle);
                using var reader = new StreamReader(fs, new UTF8Encoding(false));
                return new FileRead(reader.ReadToEnd(), DateTime.SpecifyKind(modified, DateTimeKind.Utc));
            }
            catch (FileNotFoundException) { return null; }
            catch (DirectoryNotFoundException) { return null; }
            catch (IOException)
            {
                if (i + 1 < attempts) Thread.Sleep(delayMs);
            }
            catch (Exception) { return null; }
        }
        return null;
    }

    /// <summary>"PORT PID": the first token all digits in 1-65535; the pid
    /// when the second token is all digits.</summary>
    public static PortInfo? ParsePort(string text)
    {
        var parts = text.TrimEnd().Split(' ');
        if (parts.Length == 0 || !AllDigits(parts[0]) || parts[0].Length > 5) return null;
        var port = int.Parse(parts[0]);
        if (port < 1 || port > 65535) return null;
        int? pid = parts.Length > 1 && AllDigits(parts[1]) && parts[1].Length <= 9
            ? int.Parse(parts[1]) : null;
        return new PortInfo(port, pid);
    }

    static bool AllDigits(string s) => s.Length > 0 && s.All(char.IsAsciiDigit);

    /// <summary>The server's base path: one URL-safe token, or null.</summary>
    public static string? ParseBase(string text)
    {
        var t = text.Trim();
        return BaseToken().IsMatch(t) ? t : null;
    }

    /// <summary>Written by this launch: no older than its start, less 2 s of
    /// clock and file-system granularity.</summary>
    public static bool IsFresh(DateTime modifiedUtc, DateTime launchStartedUtc) =>
        modifiedUtc >= launchStartedUtc - TimeSpan.FromSeconds(2);

    [GeneratedRegex("^[A-Za-z0-9_-]{1,64}$")]
    private static partial Regex BaseToken();
}
