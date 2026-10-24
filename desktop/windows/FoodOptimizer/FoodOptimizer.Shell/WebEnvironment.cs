using FoodOptimizer.Core;
using Microsoft.Web.WebView2.Core;

namespace FoodOptimizer.Shell;

/// <summary>The WebView2 environment: its data in the state folder, and
/// crash reports kept on this PC (they could hold formulation data).</summary>
static class WebEnvironment
{
    public static Task<CoreWebView2Environment> CreateAsync(UserPaths p) =>
        CoreWebView2Environment.CreateAsync(null, p.WebViewData,
            new CoreWebView2EnvironmentOptions { IsCustomCrashReportingEnabled = true });
}
