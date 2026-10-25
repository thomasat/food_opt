using FoodOptimizer.Core;

namespace FoodOptimizer.Tests;

public class NavigationRulesTests
{
    static readonly NavContext Ctx = new(18501, Loaded: true, PageNavigationExpected: false);
    const string Page = "https://page.foodopt.example/status.html";

    [Theory]
    [InlineData("http://127.0.0.1:18501/abc/", NavDecision.Allow)]
    [InlineData("http://127.0.0.1:18501/abc/?x=1#y", NavDecision.Allow)]
    [InlineData("http://127.0.0.1:18502/", NavDecision.Cancel)]
    [InlineData("http://localhost:18501/", NavDecision.Cancel)]
    [InlineData("https://localhost/", NavDecision.Cancel)]
    [InlineData("https://127.0.0.1:18501/", NavDecision.Cancel)]
    [InlineData("https://example.com/x", NavDecision.OpenExternally)]
    [InlineData("HTTPS://Example.com", NavDecision.OpenExternally)]
    [InlineData("http://example.com/y", NavDecision.OpenExternally)]
    [InlineData("file:///C:/Windows/win.ini", NavDecision.Cancel)]
    [InlineData("javascript:alert(1)", NavDecision.Cancel)]
    [InlineData("data:text/html,x", NavDecision.Cancel)]
    [InlineData("foodopt://retry", NavDecision.Cancel)]
    [InlineData("mailto:a@b.c", NavDecision.Cancel)]
    [InlineData("not a uri", NavDecision.Cancel)]
    [InlineData("", NavDecision.Cancel)]
    [InlineData(Page, NavDecision.Cancel)]
    public void Main_frame_when_loaded(string uri, NavDecision expected)
    {
        Assert.Equal(expected, NavigationRules.ForMainFrame(uri, false, Ctx));
    }

    [Fact]
    public void Our_page_is_allowed_before_load_or_when_the_shell_asked_for_it()
    {
        Assert.Equal(NavDecision.Allow, NavigationRules.ForMainFrame(Page, false, Ctx with { Loaded = false }));
        Assert.Equal(NavDecision.Allow, NavigationRules.ForMainFrame(Page, false, Ctx with { PageNavigationExpected = true }));
        Assert.Equal(NavDecision.OpenExternally, NavigationRules.ForMainFrame("https://page.foodopt.example.evil.com/", false, Ctx with { Loaded = false }));
    }

    [Theory]
    [InlineData("http://127.0.0.1:18501/abc/")]
    [InlineData(Page)]
    [InlineData("https://example.com/x")]
    public void A_history_navigation_is_cancelled(string uri)
    {
        Assert.Equal(NavDecision.Cancel, NavigationRules.ForMainFrame(uri, true, Ctx with { PageNavigationExpected = true }));
    }

    [Fact]
    public void No_port_yet_means_no_server_to_allow()
    {
        Assert.Equal(NavDecision.Cancel, NavigationRules.ForMainFrame("http://127.0.0.1:18501/", false, new NavContext(null, false, false)));
    }

    [Theory]
    [InlineData("about:blank", NavDecision.Allow)]
    [InlineData("about:srcdoc", NavDecision.Allow)]
    [InlineData("http://127.0.0.1:18501/abc/component/x.html", NavDecision.Allow)]
    [InlineData("http://127.0.0.1:18502/", NavDecision.Cancel)]
    [InlineData("https://example.com/", NavDecision.Cancel)]
    [InlineData("file:///C:/x", NavDecision.Cancel)]
    public void Frames(string uri, NavDecision expected)
    {
        Assert.Equal(expected, NavigationRules.ForFrame(uri, Ctx));
    }

    [Theory]
    [InlineData("https://example.com/foodopt-open", NavDecision.OpenExternally)]
    [InlineData("http://example.com/", NavDecision.OpenExternally)]
    [InlineData("http://127.0.0.1:18501/abc/", NavDecision.Cancel)]
    [InlineData("http://localhost:9/", NavDecision.Cancel)]
    [InlineData("file:///C:/x", NavDecision.Cancel)]
    [InlineData("about:blank", NavDecision.Cancel)]
    public void New_windows(string uri, NavDecision expected)
    {
        Assert.Equal(expected, NavigationRules.ForNewWindow(uri));
    }

    [Theory]
    [InlineData("http://127.0.0.1:18501/abc/static/x.js", 18501, true)]
    [InlineData("ws://127.0.0.1:18501/abc/_stcore/stream", 18501, true)]
    [InlineData("https://page.foodopt.example/status.css", 18501, true)]
    [InlineData("https://page.foodopt.example/status.js", null, true)]
    [InlineData("data:image/png;base64,AAAA", 18501, true)]
    [InlineData("blob:http://127.0.0.1:18501/1234", 18501, true)]
    [InlineData("https://fonts.gstatic.com/x", 18501, false)]
    [InlineData("http://127.0.0.1:18501/", null, false)]
    [InlineData("http://127.0.0.1:18502/", 18501, false)]
    [InlineData("http://localhost:18501/", 18501, false)]
    [InlineData("wss://127.0.0.1:18501/", 18501, false)]
    [InlineData("file:///C:/x", 18501, false)]
    public void Resources(string uri, int? port, bool allowed)
    {
        Assert.Equal(allowed, NavigationRules.IsResourceAllowed(uri, port));
    }
}
