using FoodOptimizer.Core;

namespace FoodOptimizer.Tests;

public sealed class ShellLogTests : IDisposable
{
    readonly string _dir = Directory.CreateTempSubdirectory("foodopt-log-").FullName;
    string Path => System.IO.Path.Combine(_dir, StateFiles.ShellLog);

    public void Dispose() => Directory.Delete(_dir, true);

    [Fact]
    public void Line_format()
    {
        new ShellLog(Path).Write("launcher.start", "x");
        Assert.Matches(@"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d launcher.start x\n$", File.ReadAllText(Path));
    }

    [Fact]
    public void Lines_append_with_local_time_and_flattened_detail()
    {
        var log = new ShellLog(Path, () => new DateTime(2026, 10, 24, 1, 2, 3, DateTimeKind.Local));
        log.Write("shell.start", "0.9.0");
        log.Write("shell.quit");
        log.Write("page.error", "a\r\nb\nc");
        Assert.Equal("2026-10-24 01:02:03 shell.start 0.9.0\n2026-10-24 01:02:03 shell.quit\n2026-10-24 01:02:03 page.error a  b c\n",
            File.ReadAllText(Path));
    }

    [Theory]
    [InlineData("Launcher started")]
    [InlineData("Shell.Start")]
    [InlineData("")]
    public void Prose_event_names_are_refused(string evt)
    {
        Assert.Throws<ArgumentException>(() => new ShellLog(Path).Write(evt));
    }

    [Fact]
    public void A_log_in_a_missing_folder_never_throws()
    {
        new ShellLog(System.IO.Path.Combine(_dir, "no", "such", "shell.log")).Write("shell.start");
    }

    [Fact]
    public void Rotate_moves_a_large_log()
    {
        File.WriteAllText(Path, new string('x', 2000));
        File.WriteAllText(Path + ".1", "old");
        ShellLog.Rotate(Path, 1000);
        Assert.False(File.Exists(Path));
        Assert.Equal(2000, new FileInfo(Path + ".1").Length);

        File.WriteAllText(Path, "small");
        ShellLog.Rotate(Path, 1000);
        Assert.Equal("small", File.ReadAllText(Path));
        ShellLog.Rotate(System.IO.Path.Combine(_dir, "absent.log"));
    }
}
