using System.Text;
using System.Text.Json;
using FoodOptimizer.Core;

namespace FoodOptimizer.Tests;

public sealed class ShellSettingsTests : IDisposable
{
    readonly string _dir = Directory.CreateTempSubdirectory("foodopt-settings-").FullName;
    string Path => System.IO.Path.Combine(_dir, StateFiles.Settings);

    public void Dispose() => Directory.Delete(_dir, true);

    [Fact]
    public void Round_trip_keeps_every_field()
    {
        var s = new ShellSettings
        {
            Zoom = 1.3,
            LastOpenedVersion = "0.9.0",
            Window = new SavedWindow(@"\\.\DISPLAY1", 320, 90, 1280, 860, true),
        };
        Assert.True(s.Save(Path));
        var back = ShellSettings.Load(Path);
        Assert.Equal(1, back.Version);
        Assert.Equal(1.3, back.Zoom);
        Assert.Equal("0.9.0", back.LastOpenedVersion);
        Assert.Equal(s.Window, back.Window);
        using var doc = JsonDocument.Parse(File.ReadAllBytes(Path));
        Assert.Equal(1, doc.RootElement.GetProperty("version").GetInt32());
        Assert.Equal(@"\\.\DISPLAY1", doc.RootElement.GetProperty("window").GetProperty("monitor").GetString());
        Assert.True(doc.RootElement.GetProperty("window").GetProperty("maximized").GetBoolean());
    }

    static void AssertDefaults(ShellSettings s)
    {
        Assert.Equal(1, s.Version);
        Assert.Equal(1.0, s.Zoom);
        Assert.Null(s.LastOpenedVersion);
        Assert.Null(s.Window);
    }

    [Fact]
    public void Missing_file_gives_defaults() => AssertDefaults(ShellSettings.Load(Path));

    [Theory]
    [InlineData("{")]
    [InlineData("")]
    [InlineData("[1,2]")]
    [InlineData("null")]
    [InlineData("\"zoom\"")]
    public void Truncated_file_gives_defaults(string text)
    {
        File.WriteAllText(Path, text);
        AssertDefaults(ShellSettings.Load(Path));
    }

    [Fact]
    public void One_bad_field_keeps_the_others()
    {
        File.WriteAllText(Path, """{"zoom":"abc","lastOpenedVersion":"0.9.0"}""");
        var s = ShellSettings.Load(Path);
        Assert.Equal(1.0, s.Zoom);
        Assert.Equal("0.9.0", s.LastOpenedVersion);

        File.WriteAllText(Path, """{"zoom":1.4,"lastOpenedVersion":7,"window":{"monitor":"M","left":1,"top":2,"width":0,"height":5,"maximized":false}}""");
        s = ShellSettings.Load(Path);
        Assert.Equal(1.4, s.Zoom);
        Assert.Null(s.LastOpenedVersion);
        Assert.Null(s.Window);

        File.WriteAllText(Path, """{"zoom":1.4,"window":{"left":1,"top":2,"width":900,"height":600}}""");
        Assert.Null(ShellSettings.Load(Path).Window);

        File.WriteAllText(Path, """{"window":{"monitor":"M","left":1,"top":2,"width":900,"height":600,"maximized":"yes"}}""");
        Assert.Equal(new SavedWindow("M", 1, 2, 900, 600, false), ShellSettings.Load(Path).Window);
    }

    [Theory]
    [InlineData("100", 2.5)]
    [InlineData("0.1", 0.5)]
    [InlineData("1.234", 1.23)]
    public void Zoom_out_of_range_is_clamped(string zoom, double expected)
    {
        File.WriteAllText(Path, "{\"zoom\":" + zoom + "}");
        Assert.Equal(expected, ShellSettings.Load(Path).Zoom);
    }

    [Fact]
    public void A_bom_is_tolerated()
    {
        File.WriteAllText(Path, """{"zoom":1.2}""", new UTF8Encoding(true));
        Assert.Equal(0xEF, File.ReadAllBytes(Path)[0]);
        Assert.Equal(1.2, ShellSettings.Load(Path).Zoom);
    }

    [Fact]
    public void Save_leaves_no_temp_file()
    {
        Assert.True(new ShellSettings().Save(Path));
        Assert.True(new ShellSettings { Zoom = 2 }.Save(Path));
        Assert.Equal([StateFiles.Settings], Directory.GetFiles(_dir).Select(System.IO.Path.GetFileName));
        Assert.Equal(2.0, ShellSettings.Load(Path).Zoom);
    }

    [Fact]
    public void Save_to_a_locked_file_returns_false()
    {
        Assert.SkipUnless(OperatingSystem.IsWindows(), "Windows only");
        File.WriteAllText(Path, "{}");
        using (new FileStream(Path, FileMode.Open, FileAccess.ReadWrite, FileShare.None))
        {
            Assert.False(new ShellSettings { Zoom = 2 }.Save(Path));
        }
        Assert.Equal("{}", File.ReadAllText(Path));
    }
}
