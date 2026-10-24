using System.Runtime.Versioning;
using FoodOptimizer.Core;
using FoodOptimizer.Core.Windows;

namespace FoodOptimizer.Tests;

[SupportedOSPlatform("windows")]
public sealed class WinShellTests : IDisposable
{
    readonly string _dir = Directory.CreateTempSubdirectory("foodopt-shell-").FullName;

    public void Dispose() => Directory.Delete(_dir, true);

    [Fact]
    public void Downloads_folder_exists()
    {
        Assert.SkipUnless(OperatingSystem.IsWindows(), "Windows only");
        var d = WinShell.DownloadsFolder();
        Assert.NotNull(d);
        Assert.True(Directory.Exists(d), d);
    }

    [Fact]
    public void Remove_zone_identifier_strips_the_mark()
    {
        Assert.SkipUnless(OperatingSystem.IsWindows(), "Windows only");
        var file = Path.Combine(_dir, "a.csv");
        File.WriteAllText(file, "a,b\n1,2\n");
        var stream = DownloadRules.ZoneIdentifierPath(file);
        File.WriteAllText(stream, "[ZoneTransfer]\r\nZoneId=3\r\n");
        Assert.True(File.Exists(stream));
        Assert.True(WinShell.RemoveZoneIdentifier(file));
        Assert.False(File.Exists(stream));
        Assert.Equal("a,b\n1,2\n", File.ReadAllText(file));

        var plain = Path.Combine(_dir, "b.csv");
        File.WriteAllText(plain, "x");
        Assert.True(WinShell.RemoveZoneIdentifier(plain));
        Assert.Equal("x", File.ReadAllText(plain));
    }

    [Fact]
    public void Open_of_a_missing_target_is_false()
    {
        Assert.SkipUnless(OperatingSystem.IsWindows(), "Windows only");
        Assert.False(WinShell.Open(Path.Combine(_dir, "no such file.txt")));
        Assert.False(WinShell.RevealInExplorer(Path.Combine(_dir, "no such file.txt")));
    }
}
