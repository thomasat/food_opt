using System.Runtime.InteropServices;
using System.Text;

namespace FoodOptimizer.UiTests;

/// <summary>Watches for console windows (conhost or Windows Terminal) that
/// appear between Start and Stop.</summary>
sealed class ConsoleWatch
{
    static readonly string[] Classes = ["ConsoleWindowClass", "CASCADIA_HOSTING_WINDOW_CLASS"];

    delegate bool EnumProc(IntPtr hwnd, IntPtr lParam);
    [DllImport("user32.dll")] static extern bool EnumWindows(EnumProc proc, IntPtr lParam);
    [DllImport("user32.dll")] static extern bool IsWindowVisible(IntPtr hwnd);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] static extern int GetClassNameW(IntPtr hwnd, StringBuilder name, int max);

    readonly HashSet<IntPtr> _before = [];
    readonly HashSet<IntPtr> _seen = [];
    CancellationTokenSource? _stop;
    Task? _task;

    static List<IntPtr> Consoles()
    {
        var found = new List<IntPtr>();
        EnumWindows((h, _) =>
        {
            if (!IsWindowVisible(h)) return true;
            var sb = new StringBuilder(256);
            GetClassNameW(h, sb, sb.Capacity);
            if (Classes.Contains(sb.ToString())) found.Add(h);
            return true;
        }, IntPtr.Zero);
        return found;
    }

    public void Start()
    {
        _before.UnionWith(Consoles());
        _stop = new CancellationTokenSource();
        var token = _stop.Token;
        _task = Task.Run(async () =>
        {
            while (!token.IsCancellationRequested)
            {
                foreach (var h in Consoles())
                    if (!_before.Contains(h)) lock (_seen) _seen.Add(h);
                try { await Task.Delay(200, token); } catch (OperationCanceledException) { }
            }
        });
    }

    /// <summary>Stops watching; the console windows that appeared.</summary>
    public IReadOnlyCollection<IntPtr> Stop()
    {
        _stop?.Cancel();
        _task?.Wait();
        lock (_seen) return _seen.ToList();
    }
}
