using System.Text;
using System.Text.RegularExpressions;

namespace FoodOptimizer.Core;

/// <summary>shell.log: one line per event, "yyyy-MM-dd HH:mm:ss event
/// [detail]". Events are dotted identifiers, never prose. Writing never
/// throws: a log that cannot be written is not worth stopping for.</summary>
public sealed partial class ShellLog
{
    readonly string _path;
    readonly Func<DateTime> _now;
    readonly object _lock = new();

    public ShellLog(string path, Func<DateTime>? now = null)
    {
        _path = path;
        _now = now ?? (() => DateTime.Now);
    }

    public string Path => _path;

    public void Write(string evt, string? detail = null)
    {
        if (!EventName().IsMatch(evt)) throw new ArgumentException("log.event: " + evt, nameof(evt));
        var now = _now();
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        var line = now.ToString("yyyy-MM-dd", inv) + " " + now.ToString("HH:mm:ss", inv) + " " + evt;
        if (!string.IsNullOrEmpty(detail)) line += " " + detail.Replace('\r', ' ').Replace('\n', ' ');
        try
        {
            lock (_lock)
                File.AppendAllText(_path, line + "\n", new UTF8Encoding(false));
        }
        catch (Exception)
        {
        }
    }

    /// <summary>Moves a log over <paramref name="maxBytes"/> to "&lt;path&gt;.1".</summary>
    public static void Rotate(string path, long maxBytes = 1_000_000)
    {
        try
        {
            var info = new FileInfo(path);
            if (info.Exists && info.Length > maxBytes) File.Move(path, path + ".1", overwrite: true);
        }
        catch (Exception)
        {
        }
    }

    [GeneratedRegex("^[a-z0-9_.]+$")]
    private static partial Regex EventName();
}
