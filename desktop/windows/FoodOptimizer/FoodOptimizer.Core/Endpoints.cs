namespace FoodOptimizer.Core;

/// <summary>The local server's addresses, under its per-launch base path.</summary>
public static class Endpoints
{
    public static Uri App(int port, string @base) => new($"http://127.0.0.1:{port}/{@base}/");

    public static Uri Health(int port, string @base) => new(App(port, @base), "_stcore/health");
}
