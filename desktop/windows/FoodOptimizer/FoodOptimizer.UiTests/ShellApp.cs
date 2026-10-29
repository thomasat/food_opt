using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.Management;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text.Json.Nodes;
using System.Windows.Automation;
using WinForms = System.Windows.Forms;

namespace FoodOptimizer.UiTests;

/// <summary>One run's folders: the staged install, and a temp root (with a
/// non-ASCII user name in it) holding the state and projects folders.</summary>
public sealed record ShellRun(string Stage, string TempRoot, IDictionary<string, string>? Env = null, string? SettingsJson = null)
{
    public string StateDir => Path.Combine(TempRoot, "state");
    public string ProjectsDir => Path.Combine(TempRoot, "FoodOptimizer");

    static string RootFor(string test) => Path.Combine(Path.GetTempPath(), "foodopt-ui", test, "José Müller");

    /// <summary>A run with empty folders.</summary>
    public static ShellRun Fresh([CallerMemberName] string test = "")
    {
        var root = RootFor(test);
        if (Directory.Exists(root)) Directory.Delete(root, true);
        return new ShellRun(Uia.Stage, root);
    }

    /// <summary>A run that keeps what an earlier run of this test left.</summary>
    public static ShellRun Same([CallerMemberName] string test = "") => new(Uia.Stage, RootFor(test));
}

/// <summary>A running FoodOptimizer.exe and its main window, driven
/// through UI Automation.</summary>
public sealed class ShellApp : IDisposable
{
    public const int LoadSeconds = 120;
    public static TimeSpan Load => TimeSpan.FromSeconds(LoadSeconds);

    [DllImport("user32.dll")] static extern bool SetForegroundWindow(IntPtr hwnd);
    [DllImport("user32.dll")] static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll")] static extern IntPtr WindowFromPoint(POINT p);
    [DllImport("user32.dll")] static extern IntPtr GetAncestor(IntPtr hwnd, uint flags);
    [DllImport("user32.dll")] static extern uint GetWindowThreadProcessId(IntPtr hwnd, out uint pid);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] static extern int GetWindowText(IntPtr hwnd, System.Text.StringBuilder text, int max);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] static extern int GetClassName(IntPtr hwnd, System.Text.StringBuilder text, int max);
    [DllImport("user32.dll")] static extern bool SetCursorPos(int x, int y);
    [DllImport("user32.dll")] static extern void keybd_event(byte vk, byte scan, uint flags, UIntPtr extra);
    [DllImport("user32.dll")] static extern void mouse_event(uint flags, int dx, int dy, int data, UIntPtr extra);

    [StructLayout(LayoutKind.Sequential)] struct POINT { public int X, Y; }

    ShellApp(ShellRun run, Process process)
    {
        Run = run;
        Process = process;
    }

    public ShellRun Run { get; }
    public Process Process { get; }
    public AutomationElement Window { get; private set; } = null!;
    public double WindowSeconds { get; private set; }
    public string StateDir => Run.StateDir;
    public string ProjectsDir => Run.ProjectsDir;
    public string Stage => Run.Stage;

    public static Process StartExe(ShellRun run)
    {
        Directory.CreateDirectory(run.StateDir);
        if (run.SettingsJson != null) File.WriteAllText(Path.Combine(run.StateDir, "settings.json"), run.SettingsJson);
        var info = new ProcessStartInfo(Path.Combine(run.Stage, "FoodOptimizer.exe"))
        {
            UseShellExecute = false,
            WorkingDirectory = run.Stage,
        };
        info.Environment["FOODOPT_STATE_DIR"] = run.StateDir;
        info.Environment["FOODOPT_PROJECTS_DIR"] = run.ProjectsDir;
        foreach (var (k, v) in run.Env ?? new Dictionary<string, string>()) info.Environment[k] = v;
        return Process.Start(info)!;
    }

    public static ShellApp Launch(ShellRun run, bool waitForWindow = true)
    {
        var sw = Stopwatch.StartNew();
        var app = new ShellApp(run, StartExe(run));
        if (waitForWindow)
        {
            AutomationElement? w = null;
            if (!Uia.WaitUntil(() => (w = app.MainWindow()) != null, TimeSpan.FromSeconds(30), 50))
            {
                app.Dispose();
                throw new Xunit.Sdk.XunitException("the Food Optimizer window never appeared");
            }
            app.WindowSeconds = sw.Elapsed.TotalSeconds;
            app.Window = w!;
        }
        return app;
    }

    /// <summary>The process's WinForms main window, or null.</summary>
    public AutomationElement? MainWindow() =>
        Uia.TopWindows(Process.Id).FirstOrDefault(w => Uia.SafeClass(w).StartsWith("WindowsForms", StringComparison.Ordinal));

    /// <summary>A top-level window of this process named <paramref name="name"/> (a dialog).</summary>
    /// UI Automation lists an owned dialog under its owner window, so both
    /// the desktop's children and the main window's are searched.
    public AutomationElement? Dialog(string name)
    {
        var windows = new List<AutomationElement>(Uia.TopWindows(Process.Id));
        var main = MainWindow();
        if (main != null)
            windows.AddRange(main.FindAll(TreeScope.Children,
                new PropertyCondition(AutomationElement.ControlTypeProperty, ControlType.Window)).Cast<AutomationElement>());
        return windows.FirstOrDefault(w => Uia.NameOf(w) == name
            && (main == null || w.Current.NativeWindowHandle != main.Current.NativeWindowHandle));
    }

    public IntPtr Hwnd => new(Window.Current.NativeWindowHandle);

    public AutomationElement WaitFor(ControlType t, string name, TimeSpan timeout) => Uia.WaitFor(Window, t, name, timeout);

    public bool Exists(ControlType t, string name) => Uia.Find(Window, t, name) != null;

    public void Activate()
    {
        try { Window.SetFocus(); } catch (Exception) { }
        SetForegroundWindow(Hwnd);
        Thread.Sleep(300);
    }

    public bool IsForeground => GetForegroundWindow() == Hwnd;

    /// <summary>Brings the window back to the foreground when another one
    /// has it (a browser window an external link opened can come up late,
    /// during the next test): keys sent meanwhile would go to that window.
    /// True when the window has the foreground.</summary>
    public bool EnsureForeground() => EnsureForeground(Window);

    /// <summary><see cref="EnsureForeground()"/> for any top-level window
    /// (a dialog of the app, a file dialog).</summary>
    public static bool EnsureForeground(AutomationElement window)
    {
        var hwnd = new IntPtr(window.Current.NativeWindowHandle);
        if (GetForegroundWindow() == hwnd) return true;
        try { window.SetFocus(); } catch (Exception) { }
        SetForegroundWindow(hwnd);
        return Uia.WaitUntil(() => GetForegroundWindow() == hwnd, TimeSpan.FromSeconds(3), 50);
    }

    /// <summary>Fails, naming the window that has it, unless
    /// <paramref name="window"/> has (or gets back) the foreground.</summary>
    public static void AssertForeground(AutomationElement window) =>
        Assert.True(EnsureForeground(window), $"'{Uia.NameOf(window)}' does not have the foreground; {Foreground()} has it");

    public void AssertForeground() => AssertForeground(Window);

    /// <summary>The foreground window, described for a failure message.</summary>
    public static string Foreground() => Describe(GetForegroundWindow());

    static string Describe(IntPtr hwnd)
    {
        if (hwnd == IntPtr.Zero) return "no window";
        var title = new System.Text.StringBuilder(256);
        var cls = new System.Text.StringBuilder(256);
        GetWindowText(hwnd, title, title.Capacity);
        GetClassName(hwnd, cls, cls.Capacity);
        GetWindowThreadProcessId(hwnd, out var pid);
        string name;
        try { using var p = System.Diagnostics.Process.GetProcessById((int)pid); name = p.ProcessName; }
        catch (Exception) { name = "?"; }
        return $"{cls} '{title}' of {name} ({pid})";
    }

    /// <summary>Gives <paramref name="target"/> (in <paramref name="window"/>)
    /// the keyboard focus while its window has the foreground; fails,
    /// naming what has them, when it cannot. Keys sent next reach it.</summary>
    public static void Focus(AutomationElement window, AutomationElement target)
    {
        AssertForeground(window);
        if (HasFocus(target)) return;
        Assert.True(Uia.WaitUntil(() =>
        {
            if (HasFocus(target)) return true;
            // The window first: Win32 focus back on it (the web view takes
            // it on), then the element (DOM focus inside the page).
            try { window.SetFocus(); target.SetFocus(); }
            catch (InvalidOperationException) { }
            return HasFocus(target);
        }, TimeSpan.FromSeconds(3), 100), $"'{Uia.NameOf(target)}' never got the keyboard focus; {Foreground()} has the foreground");
    }

    /// <summary>Whether UI Automation's focused element (which starts from
    /// the Win32 focus, then asks the page) is <paramref name="e"/>: a web
    /// element's own HasKeyboardFocus reports only the page's focus.</summary>
    static bool HasFocus(AutomationElement e)
    {
        try { return Automation.Compare(AutomationElement.FocusedElement, e); }
        catch (Exception) { return false; }
    }

    public void Focus(AutomationElement target) => Focus(Window, target);

    /// <summary>Sends keys to <paramref name="target"/> only while the app
    /// has the foreground and the target has the keyboard focus.</summary>
    public void Send(AutomationElement target, string keys) => Send(Window, target, keys);

    public static void Send(AutomationElement window, AutomationElement target, string keys)
    {
        Focus(window, target);
        WinForms.SendKeys.SendWait(keys);
    }

    /// <summary>Puts the cursor on (x, y) and fails unless that point is
    /// this window's, with the window in the foreground: a click or wheel
    /// sent there then reaches the app, not a window over it.</summary>
    void PointAt(double x, double y)
    {
        AssertForeground();
        SetCursorPos((int)x, (int)y);
        Thread.Sleep(100);
        var under = GetAncestor(WindowFromPoint(new POINT { X = (int)x, Y = (int)y }), 2 /* GA_ROOT */);
        Assert.True(under == Hwnd, $"the point ({x:0}, {y:0}) is on {Describe(under)}, not the app");
    }

    /// <summary>Ctrl + mouse wheel over the middle of the window.</summary>
    public void CtrlWheel(int notches)
    {
        var r = Window.Current.BoundingRectangle;
        PointAt(r.X + r.Width / 2, r.Y + r.Height / 2);
        const byte VK_CONTROL = 0x11;
        keybd_event(VK_CONTROL, 0, 0, UIntPtr.Zero);
        for (var i = 0; i < Math.Abs(notches); i++)
        {
            mouse_event(0x0800, 0, 0, Math.Sign(notches) * 120, UIntPtr.Zero);
            Thread.Sleep(150);
        }
        keybd_event(VK_CONTROL, 0, 2, UIntPtr.Zero);
        Thread.Sleep(300);
    }

    /// <summary>The menu bar's top-level items.</summary>
    public IReadOnlyList<string> MenuBarItems()
    {
        var bar = Uia.All(Window, ControlType.MenuBar).First(b => Uia.NameOf(b) != "System Menu Bar" && Uia.NameOf(b) != "System");
        return bar.FindAll(TreeScope.Children, new PropertyCondition(AutomationElement.ControlTypeProperty, ControlType.MenuItem))
            .Cast<AutomationElement>().Select(Uia.NameOf).ToList();
    }

    AutomationElement TopMenu(string name)
    {
        var bar = Uia.All(Window, ControlType.MenuBar).First(b => Uia.NameOf(b) != "System Menu Bar" && Uia.NameOf(b) != "System");
        return bar.FindAll(TreeScope.Children, new PropertyCondition(AutomationElement.ControlTypeProperty, ControlType.MenuItem))
            .Cast<AutomationElement>().First(e => Uia.NameOf(e) == name);
    }

    /// <summary>Opens a menu; its items (separators left out).</summary>
    public IReadOnlyList<AutomationElement> OpenMenu(string name)
    {
        Activate();
        var top = TopMenu(name);
        ((ExpandCollapsePattern)top.GetCurrentPattern(ExpandCollapsePattern.Pattern)).Expand();
        IReadOnlyList<AutomationElement> items = [];
        Uia.WaitUntil(() =>
        {
            items = top.FindAll(TreeScope.Descendants, new PropertyCondition(AutomationElement.ControlTypeProperty, ControlType.MenuItem))
                .Cast<AutomationElement>().ToList();
            if (items.Count == 0)
                items = Uia.TopWindows(Process.Id)
                    .SelectMany(w => w.FindAll(TreeScope.Descendants, new PropertyCondition(AutomationElement.ControlTypeProperty, ControlType.MenuItem)).Cast<AutomationElement>())
                    .Where(e => !MenuBarItems().Contains(Uia.NameOf(e)))
                    .ToList();
            return items.Count > 0;
        }, TimeSpan.FromSeconds(5));
        return items;
    }

    public void CloseMenus()
    {
        AssertForeground();
        WinForms.SendKeys.SendWait("{ESC}{ESC}");
    }

    /// <summary>Opens a menu and clicks one of its items.</summary>
    public void Menu(string menu, string item)
    {
        var e = OpenMenu(menu).First(i => Uia.NameOf(i) == item);
        Uia.Invoke(e);
        Thread.Sleep(300);
    }

    /// <summary>A dialog of this process with the caption Food Optimizer, once it appears.</summary>
    public AutomationElement WaitForDialog(TimeSpan timeout)
    {
        AutomationElement? d = null;
        if (!Uia.WaitUntil(() => (d = Dialog("Food Optimizer")) != null, timeout))
            throw new Xunit.Sdk.XunitException("no Food Optimizer dialog");
        return d!;
    }

    public static string Clipboard()
    {
        var text = "";
        OnSta(() => { text = WinForms.Clipboard.GetText(); });
        return text;
    }

    /// <summary>Empties the clipboard, so a later check cannot pass on
    /// what an earlier run (or anything else) left there.</summary>
    public static void ClearClipboard()
    {
        Assert.True(Uia.WaitUntil(() =>
        {
            try
            {
                OnSta(() => { WinForms.Clipboard.Clear(); });
                return !OnSta(() => WinForms.Clipboard.ContainsText());
            }
            catch (ExternalException) { return false; }   // another program holds it for a moment
        }, TimeSpan.FromSeconds(5)), "the clipboard could not be emptied");
    }

    static void OnSta(Action a)
    {
        Exception? failed = null;
        var t = new Thread(() => { try { a(); } catch (Exception ex) { failed = ex; } });
        t.SetApartmentState(ApartmentState.STA);
        t.Start();
        t.Join();
        if (failed != null) System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(failed).Throw();
    }

    static T OnSta<T>(Func<T> f)
    {
        T result = default!;
        OnSta(() => { result = f(); });
        return result;
    }

    public double? SavedZoom()
    {
        try { return (double?)Settings()["zoom"]; }
        catch (Exception) { return null; }
    }

    void ClickAt(double x, double y, uint down, uint up, int data = 0)
    {
        PointAt(x, y);
        mouse_event(down, 0, 0, data, UIntPtr.Zero);
        mouse_event(up, 0, 0, data, UIntPtr.Zero);
        Thread.Sleep(300);
    }

    static (double X, double Y) Center(AutomationElement e)
    {
        var r = e.Current.BoundingRectangle;
        return (r.X + r.Width / 2, r.Y + r.Height / 2);
    }

    public void RightClick(AutomationElement e)
    {
        var (x, y) = Center(e);
        ClickAt(x, y, 0x0008, 0x0010);
    }

    /// <summary>Right-clicks a point given as fractions of the window.</summary>
    public void RightClickWindowAt(double fx, double fy)
    {
        var r = Window.Current.BoundingRectangle;
        ClickAt(r.X + r.Width * fx, r.Y + r.Height * fy, 0x0008, 0x0010);
    }

    /// <summary>The mouse's back button over the middle of the window.</summary>
    public void MouseBack()
    {
        var (x, y) = Center(Window);
        ClickAt(x, y, 0x0080, 0x0100, 1);
    }

    public string SettingsPath => Path.Combine(StateDir, "settings.json");

    public JsonNode Settings() => JsonNode.Parse(ReadShared(SettingsPath))!;

    public IReadOnlyList<string> Log()
    {
        var path = Path.Combine(StateDir, "shell.log");
        return File.Exists(path) ? ReadShared(path).Split('\n', StringSplitOptions.RemoveEmptyEntries) : [];
    }

    public string LauncherLog()
    {
        var path = Path.Combine(StateDir, "launcher.log");
        return File.Exists(path) ? ReadShared(path) : "";
    }

    public static string ReadShared(string path)
    {
        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
        using var r = new StreamReader(fs);
        return r.ReadToEnd();
    }

    /// <summary>The Python processes running from this stage's python folder.</summary>
    public IReadOnlyList<(int Pid, string CommandLine)> StagePythons() => StagePythonsOf(Stage);

    public static IReadOnlyList<(int Pid, string CommandLine)> StagePythonsOf(string stage)
    {
        var prefix = Path.Combine(stage, "python") + Path.DirectorySeparatorChar;
        var found = new List<(int, string)>();
        using var searcher = new ManagementObjectSearcher(
            "SELECT ProcessId, ExecutablePath, CommandLine FROM Win32_Process WHERE Name = 'python.exe'");
        foreach (ManagementObject o in searcher.Get())
        {
            using (o)
            {
                var exe = o["ExecutablePath"] as string;
                if (exe != null && exe.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                    found.Add((Convert.ToInt32(o["ProcessId"]), o["CommandLine"] as string ?? ""));
            }
        }
        return found;
    }

    /// <summary>The WebView2 browser processes whose command line names
    /// <paramref name="folder"/> (their --user-data-dir).</summary>
    public static IReadOnlyList<int> WebViewBrowsersUsing(string folder)
    {
        var found = new List<int>();
        using var searcher = new ManagementObjectSearcher(
            "SELECT ProcessId, CommandLine FROM Win32_Process WHERE Name = 'msedgewebview2.exe'");
        foreach (ManagementObject o in searcher.Get())
        {
            using (o)
            {
                if ((o["CommandLine"] as string ?? "").Contains(folder, StringComparison.OrdinalIgnoreCase))
                    found.Add(Convert.ToInt32(o["ProcessId"]));
            }
        }
        return found;
    }

    public void Close()
    {
        var w = MainWindow() ?? Window;
        ((WindowPattern)w.GetCurrentPattern(WindowPattern.Pattern)).Close();
    }

    public async Task<bool> WaitForExitAsync(TimeSpan timeout)
    {
        using var cts = new CancellationTokenSource(timeout);
        try
        {
            await Process.WaitForExitAsync(cts.Token);
            return true;
        }
        catch (OperationCanceledException)
        {
            return Process.HasExited;
        }
    }

    public void Screenshot(string name)
    {
        try
        {
            var r = Window.Current.BoundingRectangle;
            using var bmp = new Bitmap((int)r.Width, (int)r.Height);
            using (var g = Graphics.FromImage(bmp)) g.CopyFromScreen((int)r.X, (int)r.Y, 0, 0, bmp.Size);
            bmp.Save(Path.Combine(Uia.Results, name + ".png"), ImageFormat.Png);
        }
        catch (Exception) { }
    }

    /// <summary>The whole primary screen, into the results folder (never committed).</summary>
    public static void ScreenshotScreen(string name)
    {
        try
        {
            var b = WinForms.Screen.PrimaryScreen!.Bounds;
            using var bmp = new Bitmap(b.Width, b.Height);
            using (var g = Graphics.FromImage(bmp)) g.CopyFromScreen(b.X, b.Y, 0, 0, bmp.Size);
            bmp.Save(Path.Combine(Uia.Results, name + ".png"), ImageFormat.Png);
        }
        catch (Exception) { }
    }

    /// <summary>Kills anything this stage left running (between tests).</summary>
    public static void KillStage(string stage)
    {
        foreach (var p in Process.GetProcessesByName("FoodOptimizer"))
        {
            try
            {
                if (p.MainModule?.FileName?.StartsWith(stage, StringComparison.OrdinalIgnoreCase) == true)
                    p.Kill(entireProcessTree: true);
            }
            catch (Exception) { }
        }
        foreach (var (pid, _) in StagePythonsOf(stage))
        {
            try { Process.GetProcessById(pid).Kill(entireProcessTree: true); }
            catch (Exception) { }
        }
    }

    public void Dispose()
    {
        if (!Process.HasExited)
        {
            try { Close(); } catch (Exception) { }
            if (!Process.WaitForExit(8000))
            {
                try { Process.Kill(entireProcessTree: true); } catch (Exception) { }
            }
        }
        Uia.WaitUntil(() => StagePythons().Count == 0, TimeSpan.FromSeconds(5));
        KillStage(Stage);
        Process.Dispose();
    }
}
