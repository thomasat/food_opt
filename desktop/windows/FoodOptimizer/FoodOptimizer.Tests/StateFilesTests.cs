using FoodOptimizer.Core;

namespace FoodOptimizer.Tests;

public sealed class StateFilesTests : IDisposable
{
    readonly string _dir = Directory.CreateTempSubdirectory("foodopt-state-").FullName;

    public void Dispose() => Directory.Delete(_dir, true);

    [Theory]
    [InlineData("18501 4242\n", 18501, 4242)]
    [InlineData("18501\n", 18501, null)]
    [InlineData("18501 x", 18501, null)]
    public void ParsePort_reads_port_and_pid(string text, int port, int? pid)
    {
        Assert.Equal(new PortInfo(port, pid), StateFiles.ParsePort(text));
    }

    [Theory]
    [InlineData("")]
    [InlineData("abc 1")]
    [InlineData(" 18501 1")]
    [InlineData("0 1")]
    [InlineData("70000 1")]
    public void ParsePort_refuses_garbage(string text)
    {
        Assert.Null(StateFiles.ParsePort(text));
    }

    [Fact]
    public void ParseBase_takes_one_safe_token()
    {
        var b = "Ab0_-cdefghijklmnopqrstuvwxyz012";
        Assert.Equal(32, b.Length);
        Assert.Equal(b, StateFiles.ParseBase(b + "\r\n"));
        Assert.Null(StateFiles.ParseBase(""));
        Assert.Null(StateFiles.ParseBase("a/b"));
        Assert.Null(StateFiles.ParseBase("a b"));
        Assert.Null(StateFiles.ParseBase(new string('a', 65)));
        Assert.Equal(new string('a', 64), StateFiles.ParseBase(new string('a', 64)));
    }

    [Fact]
    public void IsFresh_allows_two_seconds_of_clock_skew()
    {
        var start = new DateTime(2026, 10, 24, 12, 0, 0, DateTimeKind.Utc);
        Assert.True(StateFiles.IsFresh(start.AddSeconds(-1.9), start));
        Assert.False(StateFiles.IsFresh(start.AddSeconds(-2.1), start));
        Assert.True(StateFiles.IsFresh(start.AddSeconds(60), start));
    }

    [Fact]
    public void ReadShared_missing_file_is_null()
    {
        Assert.Null(StateFiles.ReadShared(Path.Combine(_dir, "nope.txt")));
    }

    [Fact]
    public void ReadShared_reads_a_file_another_writer_holds_open()
    {
        var path = Path.Combine(_dir, StateFiles.Status);
        using var writer = new FileStream(path, FileMode.Create, FileAccess.Write,
            FileShare.ReadWrite | FileShare.Delete);
        writer.Write("Starting the app|\n"u8);
        writer.Flush();
        var read = StateFiles.ReadShared(path);
        Assert.NotNull(read);
        Assert.Equal("Starting the app|\n", read.Text);
        Assert.True(Math.Abs((read.ModifiedUtc - File.GetLastWriteTimeUtc(path)).TotalSeconds) < 1);
        Assert.Equal(DateTimeKind.Utc, read.ModifiedUtc.Kind);
    }

    [Fact]
    public async Task ReadShared_retries_a_file_held_without_sharing()
    {
        Assert.SkipUnless(OperatingSystem.IsWindows(), "Windows only");
        var path = Path.Combine(_dir, StateFiles.Port);
        File.WriteAllText(path, "18501 4242\n");

        var held = new FileStream(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None);
        var release = Task.Run(async () => { await Task.Delay(30); held.Dispose(); });
        var read = StateFiles.ReadShared(path, attempts: 10, delayMs: 20);
        await release;
        Assert.NotNull(read);
        Assert.Equal("18501 4242\n", read.Text);

        using (new FileStream(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None))
        {
            var start = DateTime.UtcNow;
            Assert.Null(StateFiles.ReadShared(path));
            Assert.True(DateTime.UtcNow - start < TimeSpan.FromMilliseconds(500));
        }
    }

    [Fact]
    public void Endpoints_name_the_base_path()
    {
        Assert.Equal(new Uri("http://127.0.0.1:18501/abc/_stcore/health"), Endpoints.Health(18501, "abc"));
        Assert.Equal("http://127.0.0.1:18501/abc/", Endpoints.App(18501, "abc").AbsoluteUri);
    }
}
