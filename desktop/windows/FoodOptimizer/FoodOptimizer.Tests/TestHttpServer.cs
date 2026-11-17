using System.Net;
using System.Net.Sockets;
using System.Text;

namespace FoodOptimizer.Tests;

/// <summary>A tiny HTTP server on 127.0.0.1: every request gets
/// <c>status</c> and the body "ok" after <c>delay</c>.</summary>
sealed class TestHttpServer : IDisposable
{
    readonly TcpListener _listener = new(IPAddress.Loopback, 0);
    readonly CancellationTokenSource _stop = new();

    public TestHttpServer(int status, TimeSpan delay)
    {
        _listener.Start();
        Port = ((IPEndPoint)_listener.LocalEndpoint).Port;
        _ = Task.Run(() => Serve(status, delay));
    }

    public int Port { get; }

    async Task Serve(int status, TimeSpan delay)
    {
        while (!_stop.IsCancellationRequested)
        {
            TcpClient client;
            try { client = await _listener.AcceptTcpClientAsync(_stop.Token); }
            catch { return; }
            _ = Task.Run(async () =>
            {
                using (client)
                {
                    try
                    {
                        var stream = client.GetStream();
                        var buffer = new byte[4096];
                        var seen = new StringBuilder();
                        while (!seen.ToString().Contains("\r\n\r\n"))
                        {
                            var n = await stream.ReadAsync(buffer, _stop.Token);
                            if (n == 0) return;
                            seen.Append(Encoding.ASCII.GetString(buffer, 0, n));
                        }
                        await Task.Delay(delay, _stop.Token);
                        var reply = $"HTTP/1.1 {status} X\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok";
                        await stream.WriteAsync(Encoding.ASCII.GetBytes(reply), _stop.Token);
                    }
                    catch { }
                }
            });
        }
    }

    public void Dispose()
    {
        _stop.Cancel();
        _listener.Stop();
    }
}
