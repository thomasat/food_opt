using System.Diagnostics;
using System.Drawing;
using FoodOptimizer.Core;

namespace FoodOptimizer.UiTests;

public class IdentityTests
{
    static string Exe => Path.Combine(Uia.Stage, "FoodOptimizer.exe");

    [Fact]
    public void The_exe_says_food_optimizer()
    {
        var v = FileVersionInfo.GetVersionInfo(Exe);
        Assert.Equal("Food Optimizer", v.FileDescription);
        Assert.Equal("Food Optimizer", v.ProductName);
        Assert.Equal("Food System Innovations", v.CompanyName);
        Assert.Equal(AppIdentity.Version, v.ProductVersion);
        Assert.Equal(AppIdentity.Version + ".0", v.FileVersion);
    }

    [Fact]
    public void The_exe_is_x64()
    {
        using var f = File.OpenRead(Exe);
        using var r = new BinaryReader(f);
        f.Position = 0x3C;
        var pe = r.ReadInt32();
        f.Position = pe;
        Assert.Equal(0x00004550u, r.ReadUInt32());
        Assert.Equal(0x8664, r.ReadUInt16());
    }

    /// The window sets the taskbar ID itself (not only the process-wide
    /// one), so it groups with the Start menu shortcut.
    [Fact]
    public void The_window_carries_the_app_id()
    {
        using var app = ShellApp.Launch(ShellRun.Fresh());
        Assert.Equal(AppIdentity.AppUserModelId, AppIdProbe.OfWindow(app.Hwnd));
    }

    [Fact]
    public void The_exe_has_an_icon()
    {
        using var icon = Icon.ExtractAssociatedIcon(Exe);
        Assert.NotNull(icon);
        Assert.True(icon.Width > 0);
    }
}
