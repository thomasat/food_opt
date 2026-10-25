using System.Net;
using System.Net.Sockets;
using FoodOptimizer.Core;

namespace FoodOptimizer.Tests;

public class HealthProbeTests
{
    static Uri HealthOf(int port) => Endpoints.Health(port, "abc");

    [Fact]
    public async Task Ok_is_healthy()
    {
        using var server = new TestHttpServer(200, TimeSpan.Zero);
        using var probe = new HealthProbe();
        Assert.True(await probe.IsHealthyAsync(HealthOf(server.Port)));
    }

    [Fact]
    public async Task Service_unavailable_is_not_healthy()
    {
        using var server = new TestHttpServer(503, TimeSpan.Zero);
        using var probe = new HealthProbe();
        Assert.False(await probe.IsHealthyAsync(HealthOf(server.Port)));
    }

    [Fact]
    public async Task A_closed_port_is_not_healthy()
    {
        var l = new TcpListener(IPAddress.Loopback, 0);
        l.Start();
        var port = ((IPEndPoint)l.LocalEndpoint).Port;
        l.Stop();
        using var probe = new HealthProbe();
        Assert.False(await probe.IsHealthyAsync(HealthOf(port)));
    }

    [Fact]
    public async Task A_slow_answer_times_out()
    {
        using var server = new TestHttpServer(200, TimeSpan.FromSeconds(2));
        using var probe = new HealthProbe(TimeSpan.FromSeconds(0.5));
        var start = DateTime.UtcNow;
        Assert.False(await probe.IsHealthyAsync(HealthOf(server.Port)));
        Assert.True(DateTime.UtcNow - start < TimeSpan.FromSeconds(1.5));
    }

    [Fact]
    public async Task Health_check_ignores_the_default_proxy()
    {
        var saved = HttpClient.DefaultProxy;
        try
        {
            HttpClient.DefaultProxy = new WebProxy("http://127.0.0.1:9");
            using var server = new TestHttpServer(200, TimeSpan.Zero);
            using var probe = new HealthProbe();
            Assert.True(await probe.IsHealthyAsync(HealthOf(server.Port)));
        }
        finally
        {
            HttpClient.DefaultProxy = saved;
        }
    }

    [Fact]
    public void The_handler_has_no_proxy_and_no_redirects()
    {
        using var h = HealthProbe.CreateHandler();
        Assert.False(h.UseProxy);
        Assert.False(h.AllowAutoRedirect);
        Assert.False(h.UseCookies);
        Assert.Equal(TimeSpan.FromSeconds(2), h.ConnectTimeout);
    }
}
