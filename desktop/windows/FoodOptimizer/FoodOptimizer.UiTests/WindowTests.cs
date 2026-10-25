using System.Runtime.InteropServices;
using System.Text.Json;
using System.Windows.Automation;
using FoodOptimizer.Core;
using WinForms = System.Windows.Forms;

namespace FoodOptimizer.UiTests;

public class WindowTests
{
    [StructLayout(LayoutKind.Sequential)] struct POINT { public int X, Y; }
    [DllImport("user32.dll")] static extern IntPtr MonitorFromPoint(POINT p, uint flags);
    [DllImport("shcore.dll")] static extern int GetDpiForMonitor(IntPtr m, int type, out uint x, out uint y);

    static int DpiOf(WinForms.Screen s)
    {
        var p = new POINT { X = s.Bounds.X + s.Bounds.Width / 2, Y = s.Bounds.Y + s.Bounds.Height / 2 };
        return GetDpiForMonitor(MonitorFromPoint(p, 2), 0, out var x, out _) == 0 ? (int)x : 96;
    }

    internal static IReadOnlyList<MonitorInfo> Monitors() =>
        WinForms.Screen.AllScreens.Select(s => new MonitorInfo(s.DeviceName,
            new PxRect(s.WorkingArea.X, s.WorkingArea.Y, s.WorkingArea.Width, s.WorkingArea.Height), DpiOf(s), s.Primary)).ToList();

    static WindowPattern Pattern(ShellApp app) => (WindowPattern)app.Window.GetCurrentPattern(WindowPattern.Pattern);
    static TransformPattern Transform(ShellApp app) => (TransformPattern)app.Window.GetCurrentPattern(TransformPattern.Pattern);

    static void AssertBounds(PxRect want, System.Windows.Rect got)
    {
        Assert.InRange(got.X, want.X - 2, want.X + 2);
        Assert.InRange(got.Y, want.Y - 2, want.Y + 2);
        Assert.InRange(got.Width, want.Width - 2, want.Width + 2);
        Assert.InRange(got.Height, want.Height - 2, want.Height + 2);
    }

    static void AssertDefaultPlacement(ShellApp app)
    {
        var expected = WindowPlacement.Initial(Monitors(), null);
        Uia.WaitUntil(() => false, TimeSpan.FromSeconds(1));
        if (expected.Maximized)
            Assert.Equal(WindowVisualState.Maximized, Pattern(app).Current.WindowVisualState);
        else
            AssertBounds(expected.Bounds, app.Window.Current.BoundingRectangle);
    }

    [Fact]
    public void First_window_uses_the_default_placement()
    {
        using var app = ShellApp.Launch(ShellRun.Fresh());
        AssertDefaultPlacement(app);
    }

    static void Restore(ShellApp app)
    {
        Pattern(app).SetWindowVisualState(WindowVisualState.Normal);
        Thread.Sleep(500);
    }

    /// The plan's 1100x750 is below the minimum at 200 % (1400x1000), so the
    /// size is halfway between the minimum and the work area.
    [Fact]
    public async Task Window_size_is_saved_and_restored()
    {
        PxRect want;
        using (var app = ShellApp.Launch(ShellRun.Fresh()))
        {
            Restore(app);
            var screen = WinForms.Screen.FromHandle(app.Hwnd);
            var wa = screen.WorkingArea;
            var (minW, minH) = WindowPlacement.MinimumSize(DpiOf(screen));
            want = new PxRect(wa.X + 10, wa.Y + 10, minW + (wa.Width - minW) / 2, minH + (wa.Height - minH) / 2);
            Transform(app).Move(want.X, want.Y);
            Transform(app).Resize(want.Width, want.Height);
            Thread.Sleep(500);
            AssertBounds(want, app.Window.Current.BoundingRectangle);
            app.Close();
            Assert.True(await app.WaitForExitAsync(TimeSpan.FromSeconds(10)));
            var w = app.Settings()["window"]!;
            Assert.Equal(want.X, (int)w["left"]!);
            Assert.Equal(want.Y, (int)w["top"]!);
            Assert.Equal(want.Width, (int)w["width"]!);
            Assert.Equal(want.Height, (int)w["height"]!);
            Assert.False((bool)w["maximized"]!);
        }
        using var again = ShellApp.Launch(ShellRun.Same());
        Thread.Sleep(1000);
        AssertBounds(want, again.Window.Current.BoundingRectangle);
    }

    [Fact]
    public void A_position_on_a_missing_monitor_is_ignored()
    {
        var settings = """{"version":1,"zoom":1.0,"window":{"monitor":"\\\\.\\DISPLAY9","left":5000,"top":4000,"width":900,"height":600,"maximized":false}}""";
        using var app = ShellApp.Launch(ShellRun.Fresh() with { SettingsJson = settings });
        AssertDefaultPlacement(app);
    }

    [Fact]
    public async Task A_corrupt_settings_file_still_opens()
    {
        using var app = ShellApp.Launch(ShellRun.Fresh() with { SettingsJson = "{" });
        app.WaitFor(ControlType.Edit, "Type here", ShellApp.Load);
        app.Close();
        Assert.True(await app.WaitForExitAsync(TimeSpan.FromSeconds(10)));
        using var doc = JsonDocument.Parse(File.ReadAllText(app.SettingsPath));
        Assert.Equal(JsonValueKind.Object, doc.RootElement.ValueKind);
    }

    [Fact]
    public void The_window_cannot_be_smaller_than_the_minimum()
    {
        using var app = ShellApp.Launch(ShellRun.Fresh());
        Restore(app);
        Transform(app).Resize(300, 200);
        Thread.Sleep(500);
        var dpi = DpiOf(WinForms.Screen.FromHandle(app.Hwnd));
        var (minW, minH) = WindowPlacement.MinimumSize(dpi);
        var r = app.Window.Current.BoundingRectangle;
        Assert.True(r.Width >= minW - 2 && r.Height >= minH - 2, $"{r.Width}x{r.Height} < {minW}x{minH}");
    }
}
