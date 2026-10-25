using System.Text.RegularExpressions;

namespace FoodOptimizer.Core;

/// <summary>Where a download is offered and how its file is named.</summary>
public static partial class DownloadRules
{
    const string Fallback = "download";

    /// <summary>The name the server gave in Content-Disposition; a name the
    /// server gave is kept as is, even when it ends in " (2)". Without one,
    /// the file name WebView2 proposed, without its folder. When WebView2
    /// numbered it because the name was taken in its folder ("probe (4).csv"
    /// beside "probe.csv"), the number is dropped: the Save dialog offers the
    /// app's own name and asks before replacing.</summary>
    public static string SuggestedFileName(string resultFilePath, string? contentDisposition, Func<string, bool>? exists = null)
    {
        if (DispositionFileName(contentDisposition) is { } given) return given;
        var cut = resultFilePath.LastIndexOfAny(['\\', '/']);
        var name = cut >= 0 ? resultFilePath[(cut + 1)..] : resultFilePath;
        if (name.Length == 0) return Fallback;
        if (exists != null && Numbered().Match(name) is { Success: true } m)
        {
            var original = m.Groups[1].Value + m.Groups[2].Value;
            if (exists(resultFilePath[..(cut + 1)] + original)) return original;
        }
        return name;
    }

    /// <summary>The file name in a Content-Disposition header: filename*
    /// (RFC 5987, percent-decoded) when it is valid, else filename (quoted or
    /// token). Null when there is none, it is empty, or it holds \ / : or ..</summary>
    public static string? DispositionFileName(string? contentDisposition)
    {
        if (string.IsNullOrWhiteSpace(contentDisposition)) return null;
        string? plain = null, extended = null;
        foreach (var part in SplitParameters(contentDisposition).Skip(1))
        {
            var eq = part.IndexOf('=');
            if (eq < 0) continue;
            var key = part[..eq].Trim();
            var value = part[(eq + 1)..].Trim();
            if (key.Equals("filename*", StringComparison.OrdinalIgnoreCase)) extended ??= DecodeExtended(value);
            else if (key.Equals("filename", StringComparison.OrdinalIgnoreCase)) plain ??= Unquote(value);
        }
        var name = extended ?? plain;
        return IsSafeName(name) ? name : null;
    }

    static bool IsSafeName(string? name) =>
        !string.IsNullOrEmpty(name) && name.IndexOfAny(['\\', '/', ':']) < 0 && !name.Contains("..");

    // Split on ';' outside double quotes; a backslash inside quotes escapes.
    static IEnumerable<string> SplitParameters(string header)
    {
        var current = new System.Text.StringBuilder();
        bool quoted = false, escaped = false;
        foreach (var c in header)
        {
            if (c == ';' && !quoted)
            {
                yield return current.ToString();
                current.Clear();
                continue;
            }
            current.Append(c);
            if (escaped) escaped = false;
            else if (quoted && c == '\\') escaped = true;
            else if (c == '"') quoted = !quoted;
        }
        yield return current.ToString();
    }

    static string Unquote(string value)
    {
        if (value.Length < 2 || value[0] != '"' || value[^1] != '"') return value;
        var sb = new System.Text.StringBuilder();
        for (var i = 1; i < value.Length - 1; i++)
        {
            if (value[i] == '\\' && i + 1 < value.Length - 1) i++;
            sb.Append(value[i]);
        }
        return sb.ToString();
    }

    // charset'language'percent-encoded; only UTF-8 is read. Null if invalid.
    static string? DecodeExtended(string value)
    {
        var parts = value.Split('\'', 3);
        if (parts.Length != 3 || !parts[0].Equals("UTF-8", StringComparison.OrdinalIgnoreCase)) return null;
        var bytes = new List<byte>();
        var text = parts[2];
        for (var i = 0; i < text.Length; i++)
        {
            if (text[i] != '%')
            {
                if (text[i] > 0x7f) return null;
                bytes.Add((byte)text[i]);
            }
            else if (i + 2 < text.Length && byte.TryParse(text.AsSpan(i + 1, 2), System.Globalization.NumberStyles.AllowHexSpecifier, null, out var b))
            {
                bytes.Add(b);
                i += 2;
            }
            else return null;
        }
        try { return new System.Text.UTF8Encoding(false, true).GetString(bytes.ToArray()); }
        catch (ArgumentException) { return null; }
    }

    [GeneratedRegex(@"^(.+) \(\d+\)(\.[^.]*)?$")]
    private static partial Regex Numbered();

    /// <summary>The Downloads folder when it exists, else the user's folder.</summary>
    public static string InitialDirectory(string? downloadsFolder, string userProfile, Func<string, bool> exists) =>
        !string.IsNullOrEmpty(downloadsFolder) && exists(downloadsFolder) ? downloadsFolder : userProfile;

    /// <summary>The NTFS stream that marks a file as from the internet.</summary>
    public static string ZoneIdentifierPath(string path) => path + ":Zone.Identifier";

    public static bool IsFromOurServer(string uri, int? port)
    {
        if (port == null) return false;
        var u = uri.StartsWith("blob:", StringComparison.OrdinalIgnoreCase) ? uri[5..] : uri;
        return Uri.TryCreate(u, UriKind.Absolute, out var parsed)
            && parsed.Scheme == "http" && parsed.Host == "127.0.0.1" && parsed.Port == port;
    }
}
