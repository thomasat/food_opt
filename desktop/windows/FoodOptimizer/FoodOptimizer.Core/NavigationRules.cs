using System.Net;

namespace FoodOptimizer.Core;

public enum NavDecision { Allow, Cancel, OpenExternally }

/// <summary>What the navigation rules need to know: the server's port (null
/// before it is known), whether the app has loaded, and whether the shell
/// itself asked for its page just now.</summary>
public sealed record NavContext(int? Port, bool Loaded, bool PageNavigationExpected);

/// <summary>The window only ever shows our server and our own page. Other
/// web links open in the default browser; everything else is cancelled.</summary>
public static class NavigationRules
{
    static Uri? Parse(string uri) =>
        Uri.TryCreate(uri, UriKind.Absolute, out var u) ? u : null;

    static bool IsWeb(Uri u) => u.Scheme is "http" or "https";

    static bool IsLocal(Uri u) =>
        u.IsLoopback || string.Equals(u.Host, "localhost", StringComparison.OrdinalIgnoreCase)
        || (IPAddress.TryParse(u.Host.Trim('[', ']'), out var ip) && IPAddress.IsLoopback(ip));

    static bool IsOurPage(Uri u) =>
        u.Scheme == "https" && u.IsDefaultPort
        && string.Equals(u.Host, PageBridge.Host, StringComparison.OrdinalIgnoreCase);

    /// <summary>Our server: http on 127.0.0.1 at the launch's port. The
    /// server refuses any other Host, localhost included.</summary>
    static bool IsOurServer(Uri u, int? port, string scheme = "http") =>
        port is { } p && u.Scheme == scheme && u.Host == "127.0.0.1" && u.Port == p;

    public static NavDecision ForMainFrame(string uri, bool isHistory, NavContext ctx)
    {
        if (isHistory || Parse(uri) is not { } u) return NavDecision.Cancel;
        if (IsOurPage(u))
            return ctx.Loaded && !ctx.PageNavigationExpected ? NavDecision.Cancel : NavDecision.Allow;
        if (IsOurServer(u, ctx.Port)) return NavDecision.Allow;
        if (!IsWeb(u) || IsLocal(u)) return NavDecision.Cancel;
        return NavDecision.OpenExternally;
    }

    public static NavDecision ForFrame(string uri, NavContext ctx)
    {
        if (uri is "about:blank" or "about:srcdoc") return NavDecision.Allow;
        return Parse(uri) is { } u && IsOurServer(u, ctx.Port) ? NavDecision.Allow : NavDecision.Cancel;
    }

    public static NavDecision ForNewWindow(string uri) =>
        Parse(uri) is { } u && IsWeb(u) && !IsLocal(u) && !IsOurPage(u)
            ? NavDecision.OpenExternally
            : NavDecision.Cancel;

    /// <summary>The page may load only from our server, our page folder,
    /// and data: or blob: URLs; nothing from the internet.</summary>
    public static bool IsResourceAllowed(string uri, int? port)
    {
        if (uri.StartsWith("data:", StringComparison.OrdinalIgnoreCase)
            || uri.StartsWith("blob:", StringComparison.OrdinalIgnoreCase))
            return true;
        if (Parse(uri) is not { } u) return false;
        return IsOurPage(u) || IsOurServer(u, port) || IsOurServer(u, port, "ws");
    }
}
