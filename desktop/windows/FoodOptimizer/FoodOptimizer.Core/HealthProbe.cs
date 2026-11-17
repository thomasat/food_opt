namespace FoodOptimizer.Core;

/// <summary>Asks the local server's health endpoint, never through a proxy
/// (Windows' system proxy would otherwise catch 127.0.0.1).</summary>
public sealed class HealthProbe : IDisposable
{
    readonly HttpClient _client;

    public HealthProbe(TimeSpan? timeout = null, HttpMessageHandler? handler = null)
    {
        _client = new HttpClient(handler ?? CreateHandler(), disposeHandler: true)
        {
            Timeout = timeout ?? TimeSpan.FromSeconds(3),
        };
    }

    public static SocketsHttpHandler CreateHandler() => new()
    {
        UseProxy = false,
        AllowAutoRedirect = false,
        UseCookies = false,
        ConnectTimeout = TimeSpan.FromSeconds(2),
    };

    /// <summary>True only for a 200 answer; any error or timeout is false.</summary>
    public async Task<bool> IsHealthyAsync(Uri uri, CancellationToken ct = default)
    {
        try
        {
            using var response = await _client.GetAsync(uri, HttpCompletionOption.ResponseHeadersRead, ct)
                .ConfigureAwait(false);
            return (int)response.StatusCode == 200;
        }
        catch (Exception)
        {
            return false;
        }
    }

    public void Dispose() => _client.Dispose();
}
