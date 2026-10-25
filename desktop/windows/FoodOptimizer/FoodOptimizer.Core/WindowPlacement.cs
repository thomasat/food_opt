namespace FoodOptimizer.Core;

/// <summary>A rectangle in physical pixels.</summary>
public readonly record struct PxRect(int X, int Y, int Width, int Height)
{
    public int Right => X + Width;
    public int Bottom => Y + Height;
}

public sealed record MonitorInfo(string DeviceName, PxRect WorkArea, int Dpi, bool IsPrimary);

/// <summary>The window as settings.json keeps it: the monitor's device name
/// and the restore bounds in physical pixels.</summary>
public sealed record SavedWindow(string Monitor, int Left, int Top, int Width, int Height, bool Maximized);

public sealed record Placement(PxRect Bounds, bool Maximized);

/// <summary>Where the window opens: the saved place if its monitor is still
/// there (pulled fully on screen), else 1280x860 centred on the primary
/// monitor, or maximized when that does not fit. Never off-screen.</summary>
public static class WindowPlacement
{
    public const int DefaultWidth = 1280, DefaultHeight = 860, MinWidth = 700, MinHeight = 500;

    public static int Scale(int logical, int dpi) => (int)Math.Round(logical * dpi / 96.0, MidpointRounding.AwayFromZero);

    public static (int Width, int Height) MinimumSize(int dpi) => (Scale(MinWidth, dpi), Scale(MinHeight, dpi));

    public static Placement Initial(IReadOnlyList<MonitorInfo> monitors, SavedWindow? saved)
    {
        if (monitors.Count == 0)
            return new Placement(new PxRect(0, 0, DefaultWidth, DefaultHeight), false);

        var home = saved == null ? null
            : monitors.FirstOrDefault(m => string.Equals(m.DeviceName, saved.Monitor, StringComparison.OrdinalIgnoreCase));
        if (saved != null && home != null)
        {
            var wa = home.WorkArea;
            var (minW, minH) = MinimumSize(home.Dpi);
            var w = Math.Min(Math.Max(saved.Width, minW), wa.Width);
            var h = Math.Min(Math.Max(saved.Height, minH), wa.Height);
            var x = Math.Clamp(saved.Left, wa.X, wa.Right - w);
            var y = Math.Clamp(saved.Top, wa.Y, wa.Bottom - h);
            return new Placement(new PxRect(x, y, w, h), saved.Maximized);
        }

        var primary = monitors.FirstOrDefault(m => m.IsPrimary) ?? monitors[0];
        var area = primary.WorkArea;
        var dw = Scale(DefaultWidth, primary.Dpi);
        var dh = Scale(DefaultHeight, primary.Dpi);
        if (dw > area.Width || dh > area.Height) return new Placement(area, true);
        return new Placement(
            new PxRect(area.X + (area.Width - dw) / 2, area.Y + (area.Height - dh) / 2, dw, dh), false);
    }

    public static SavedWindow ToSaved(string monitor, PxRect restoreBounds, bool maximized) =>
        new(monitor, restoreBounds.X, restoreBounds.Y, restoreBounds.Width, restoreBounds.Height, maximized);
}
