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
    [DllImport("user32.dll")] static extern bool SetCursorPos(int x, int y);
    [DllImport("user32.dll")] static extern void keybd_event(byte vk, byte scan, uint flags, UIntPtr extra);
    [DllImport("user32.dll")] static extern void mouse_event(uint flags, int dx, int dy, int data, UIntPtr extra);

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
    public AutomationElement? Dialog(string name) =>
        Uia.TopWindows(Process.Id).FirstOrDefault(w => Uia.SafeClass(w) == "#32770" && Uia.NameOf(w) == name);

    public IntPtr Hwnd => new(Window.Current.NativeWindowHandle);

    public AutomationElement WaitFor(ControlType t, string name, TimeSpan timeout) => Uia.WaitFor(Window, t, name, timeout);

    public bool Exists(ControlType t, string name) => Uia.Find(Window, t, name) != null;

    public void Activate()
    {
        try { Window.SetFocus(); } catch (Exception) { }
        SetForegroundWindow(Hwnd);
        Thread.Sleep(300);
    }

    public void Keys(string sendKeys)
    {
        Activate();
        WinForms.SendKeys.SendWait(sendKeys);
        Thread.Sleep(300);
    }

    /// <summary>Ctrl + mouse wheel over the middle of the window.</summary>
    public void CtrlWheel(int notches)
    {
        Activate();
        var r = Window.Current.BoundingRectangle;
        SetCursorPos((int)(r.X + r.Width / 2), (int)(r.Y + r.Height / 2));
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
