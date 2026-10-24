using System.Text.Json;
using System.Text.Json.Nodes;

namespace FoodOptimizer.Core;

/// <summary>The setup and error pages desktop/shared/pages/status.js shows.</summary>
public enum PageError { CouldNotStart, MissingPart, Unsupported, Stopped }

/// <summary>How the steps page opens: <see cref="FirstOpening"/> is the
/// first opening after an install or upgrade.</summary>
public sealed record PageView(bool FirstOpening);

/// <summary>The JavaScript the shell runs on the shared page: the boot
/// object and calls to window.FoodOpt. Every argument is JSON, so no text
/// from a file can break out of the call.</summary>
public static class PageBridge
{
    public const string Host = "page.foodopt.example";
    public const string Origin = "https://" + Host;
    public const string StatusPageUrl = Origin + "/status.html";
    public const string RetryMessage = "retry";

    public static string KindName(PageError k) => k switch
    {
        PageError.CouldNotStart => "could_not_start",
        PageError.MissingPart => "missing_part",
        PageError.Unsupported => "unsupported",
        PageError.Stopped => "stopped",
        _ => throw new ArgumentOutOfRangeException(nameof(k)),
    };

    static string Json(string s) => JsonSerializer.Serialize(s);

    static JsonObject ViewJson(PageView v) => new()
    {
        ["firstRun"] = false,
        ["firstOpening"] = v.FirstOpening,
    };

    /// <summary>Sets window.FOODOPT_BOOT, but only on our own page.</summary>
    public static string BootScript(Messages m, string version, PageView? view)
    {
        var messages = new JsonObject();
        foreach (var (id, text) in m.All.OrderBy(kv => kv.Key, StringComparer.Ordinal))
            messages[id] = text;
        var boot = new JsonObject
        {
            ["os"] = "windows",
            ["version"] = version,
            ["messages"] = messages,
        };
        if (view != null) boot["view"] = ViewJson(view);
        return "if (location.origin === " + Json(Origin) + ") {\nwindow.FOODOPT_BOOT = "
            + boot.ToJsonString() + ";\n}";
    }

    public static string StatusLine(string raw) => "FoodOpt.statusLine(" + Json(raw) + ")";

    public static string ServerStarted() => "FoodOpt.serverStarted()";

    public static string ShowStalled() => "FoodOpt.showStalled()";

    public static string ShowError(PageError k, string? detailId) =>
        detailId == null
            ? "FoodOpt.showError(" + Json(KindName(k)) + ")"
            : "FoodOpt.showError(" + Json(KindName(k)) + "," + Json(detailId) + ")";

    public static string ShowSteps(PageView v) => "FoodOpt.showSteps(" + ViewJson(v).ToJsonString() + ")";
}
