using System.Text.Json;
using System.Text.Json.Nodes;
using FoodOptimizer.Core;

namespace FoodOptimizer.Tests;

public class PageBridgeTests
{
    static readonly Messages M = Messages.LoadEmbedded();

    static string ArgOf(string call, string name)
    {
        var prefix = "FoodOpt." + name + "(";
        Assert.StartsWith(prefix, call);
        Assert.EndsWith(")", call);
        return call[prefix.Length..^1];
    }

    [Fact]
    public void Status_line_is_one_json_string_argument()
    {
        var raw = "Install \"x\" </script>|38";
        var arg = ArgOf(PageBridge.StatusLine(raw), "statusLine");
        Assert.Equal(raw, JsonSerializer.Deserialize<string>(arg));
        Assert.DoesNotContain("<", arg);
    }

    static JsonNode BootJson(string script)
    {
        const string marker = "window.FOODOPT_BOOT = ";
        var start = script.IndexOf(marker, StringComparison.Ordinal) + marker.Length;
        var end = script.LastIndexOf(';');
        return JsonNode.Parse(script[start..end])!;
    }

    [Fact]
    public void Boot_script_runs_only_on_our_page()
    {
        var script = PageBridge.BootScript(M, "0.9.0", new PageView(true));
        Assert.StartsWith("if (location.origin === \"https://page.foodopt.example\") {", script);
        var boot = BootJson(script);
        Assert.Equal("windows", (string?)boot["os"]);
        Assert.Equal("0.9.0", (string?)boot["version"]);
        Assert.Equal(M.All.Count, boot["messages"]!.AsObject().Count);
        Assert.Equal("Try again", (string?)boot["messages"]!["button.try_again"]);
        Assert.Equal("{\"firstRun\":false,\"firstOpening\":true}", boot["view"]!.ToJsonString());

        var none = BootJson(PageBridge.BootScript(M, "0.9.0", null));
        Assert.False(none.AsObject().ContainsKey("view"));
    }

    [Fact]
    public void Show_error_names_the_page_kind()
    {
        Assert.Equal("FoodOpt.showError(\"stopped\")", PageBridge.ShowError(PageError.Stopped, null));
        Assert.Equal("FoodOpt.showError(\"could_not_start\",\"error.could_not_start.another_copy\")",
            PageBridge.ShowError(PageError.CouldNotStart, "error.could_not_start.another_copy"));
        Assert.Equal("missing_part", PageBridge.KindName(PageError.MissingPart));
        Assert.Equal("unsupported", PageBridge.KindName(PageError.Unsupported));
    }

    [Fact]
    public void Show_steps_never_claims_a_download()
    {
        Assert.Equal("FoodOpt.showSteps({\"firstRun\":false,\"firstOpening\":false})", PageBridge.ShowSteps(new PageView(false)));
    }

    [Fact]
    public void Page_names_are_fixed()
    {
        Assert.Equal("page.foodopt.example", PageBridge.Host);
        Assert.Equal("https://page.foodopt.example/status.html", PageBridge.StatusPageUrl);
        Assert.Equal("retry", PageBridge.RetryMessage);
        Assert.Equal("FoodOpt.serverStarted()", PageBridge.ServerStarted());
        Assert.Equal("FoodOpt.showStalled()", PageBridge.ShowStalled());
    }
}
