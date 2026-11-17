using FoodOptimizer.Core;
using Microsoft.Web.WebView2.Core;

namespace FoodOptimizer.Shell;

/// <summary>The WebView2 environment: its data in the state folder, crash
/// reports kept on this PC (they could hold formulation data), and the
/// runtime's own networking off (I4). Measured on WebView2 156 by Level 2's
/// test_l2_40_traffic (Task 13): without switches, a 60 s session reached
/// config.edge.skype.com (stopped by --disable-background-networking),
/// edge.microsoft.com's component updater and its download host (stopped
/// by --disable-component-update), looked up wpad (stopped by
/// --no-proxy-server) and sent DNS queries for the status page's own name
/// (stopped by the host rule; the app is at the IP 127.0.0.1 and needs no
/// name). --disable-domain-reliability and --no-pings are the standard
/// background switches for uploads a session here never triggers.</summary>
static class WebEnvironment
{
    public const string BrowserArguments =
        "--disable-background-networking --disable-component-update --disable-domain-reliability " +
        "--no-pings --no-proxy-server --host-resolver-rules=\"MAP * ~NOTFOUND, EXCLUDE 127.0.0.1\"";

    public static Task<CoreWebView2Environment> CreateAsync(UserPaths p) =>
        CoreWebView2Environment.CreateAsync(null, p.WebViewData,
            new CoreWebView2EnvironmentOptions
            {
                IsCustomCrashReportingEnabled = true,
                AdditionalBrowserArguments = BrowserArguments,
            });
}
