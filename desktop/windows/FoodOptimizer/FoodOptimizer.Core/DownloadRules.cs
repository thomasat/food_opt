namespace FoodOptimizer.Core;

/// <summary>Where a download is offered and how its file is named.</summary>
public static class DownloadRules
{
    const string Fallback = "download";

    /// <summary>The file name WebView2 proposed, without its folder.</summary>
    public static string SuggestedFileName(string resultFilePath)
    {
        var cut = resultFilePath.LastIndexOfAny(['\\', '/']);
        var name = cut >= 0 ? resultFilePath[(cut + 1)..] : resultFilePath;
        return name.Length > 0 ? name : Fallback;
    }

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
