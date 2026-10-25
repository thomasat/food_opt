using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;
using System.Windows.Automation;

namespace FoodOptimizer.UiTests;

/// <summary>The installer (FoodOptimizer-*-Windows*.exe) on the desktop.
/// Inno Setup's own controls (TNewButton, TNewCheckBox, TNewStaticText)
/// reach UI Automation as plain panes with names but no patterns, so the
/// wizard is read through UI Automation and driven with Win32 button
/// messages on each control's window handle.</summary>
public sealed class SetupWizard : IDisposable
{
    const uint BM_GETCHECK = 0x00F0, BM_CLICK = 0x00F5;
    [DllImport("user32.dll")] static extern IntPtr SendMessageW(IntPtr hwnd, uint msg, IntPtr wParam, IntPtr lParam);
    [DllImport("user32.dll")] static extern bool PostMessageW(IntPtr hwnd, uint msg, IntPtr wParam, IntPtr lParam);
    [DllImport("user32.dll")] static extern bool PrintWindow(IntPtr hwnd, IntPtr hdc, uint flags);

    SetupWizard(Process process) => Process = process;

    public Process Process { get; }
    public HashSet<string> Seen { get; } = new(StringComparer.Ordinal);

    /// <summary>The installer the UI tests run (run_ui_tests.ps1 -Installer).</summary>
    public static string Installer =>
        Environment.GetEnvironmentVariable("FOODOPT_UI_INSTALLER") is { Length: > 0 } s
            ? s : throw new InvalidOperationException("FOODOPT_UI_INSTALLER is not set (run_ui_tests.ps1 -Installer)");

    /// <summary>A wizard over a process already started (the uninstaller).</summary>
    public static SetupWizard Attach(Process process) => new(process);

    public static SetupWizard Start(string installer, string args, IDictionary<string, string> env)
    {
        var info = new ProcessStartInfo(installer, args) { UseShellExecute = false };
        foreach (var (k, v) in env) info.Environment[k] = v;
        return new SetupWizard(Process.Start(info)!);
    }

    /// <summary>The wizard's window ("Install Food Optimizer"), owned by the
    /// installer's unpacked .tmp process.</summary>
    public AutomationElement Window
    {
        get
        {
            AutomationElement? w = null;
            if (!Uia.WaitUntil(() => (w = FindWindow()) != null, TimeSpan.FromSeconds(60)))
                throw new Xunit.Sdk.XunitException("the installer's window never appeared");
            return w!;
        }
    }

    static AutomationElement? FindWindow() =>
        AutomationElement.RootElement.FindAll(TreeScope.Children, Condition.TrueCondition).Cast<AutomationElement>()
            .FirstOrDefault(w => Uia.SafeClass(w) == "TWizardForm" || (Uia.NameOf(w) == "Install Food Optimizer"
                && Uia.SafeClass(w).StartsWith("T", StringComparison.Ordinal)));

    /// <summary>The window's controls that are real windows (so not the
    /// title bar's parts), on screen.</summary>
    static IReadOnlyList<AutomationElement> Controls(AutomationElement root) =>
        root.FindAll(TreeScope.Descendants, Condition.TrueCondition).Cast<AutomationElement>()
            .Where(e =>
            {
                try { return e.Current.NativeWindowHandle != 0 && !e.Current.IsOffscreen; }
                catch (Exception) { return false; }
            }).ToList();

    /// <summary>The top-most text of the current page.</summary>
    public string Heading()
    {
        var texts = Controls(Window).Where(e => Uia.SafeClass(e) == "TNewStaticText" && Uia.NameOf(e).Trim().Length > 0)
            .OrderBy(e => e.Current.BoundingRectangle.Top).ThenBy(e => e.Current.BoundingRectangle.Left).ToList();
        Note(Window);
        return texts.Count == 0 ? "" : Uia.NameOf(texts[0]).Trim();
    }

    /// <summary>Every named control of a window, also kept in Seen; a
    /// list box's items are read with LB_GETTEXT where it keeps strings.</summary>
    public IReadOnlyList<string> Texts() => Note(Window);

    IReadOnlyList<string> Note(AutomationElement root)
    {
        var names = new List<string>();
        foreach (var e in Controls(root))
        {
            var n = Uia.NameOf(e).Trim();
            if (n.Length > 0) names.Add(n);
            if (Uia.SafeClass(e) == "TNewCheckListBox") names.AddRange(ListItems(new IntPtr(e.Current.NativeWindowHandle)));
        }
        foreach (var n in names) Seen.Add(n);
        return names;
    }

    const uint LB_GETCOUNT = 0x018B, LB_GETTEXT = 0x0189, LB_GETTEXTLEN = 0x018A;

    static IEnumerable<string> ListItems(IntPtr hwnd)
    {
        var count = (int)SendMessageW(hwnd, LB_GETCOUNT, IntPtr.Zero, IntPtr.Zero);
        for (var i = 0; i < count && i < 20; i++)
        {
            var len = (int)SendMessageW(hwnd, LB_GETTEXTLEN, i, IntPtr.Zero);
            if (len <= 0 || len > 1000) continue;
            var buf = Marshal.AllocHGlobal((len + 1) * 2);
            try
            {
                SendMessageW(hwnd, LB_GETTEXT, i, buf);
                var s = Marshal.PtrToStringUni(buf);
                if (!string.IsNullOrWhiteSpace(s)) yield return s.Trim();
            }
            finally { Marshal.FreeHGlobal(buf); }
        }
    }

    AutomationElement Control(AutomationElement root, string name) =>
        Controls(root).FirstOrDefault(e => Uia.NameOf(e).Replace("&", "").Trim() == name)
        ?? throw new Xunit.Sdk.XunitException($"no control '{name}'; saw: {string.Join(" | ", Note(root))}");

    /// <summary>A check box's state (BM_GETCHECK), or null when it is not one.</summary>
    public bool? IsChecked(string name)
    {
        var e = Control(Window, name);
        if (Uia.SafeClass(e) != "TNewCheckBox") return null;
        return SendMessageW(new IntPtr(e.Current.NativeWindowHandle), BM_GETCHECK, IntPtr.Zero, IntPtr.Zero) == 1;
    }

    public void SetChecked(string name, bool value)
    {
        if (IsChecked(name) != value) Click(name);
    }

    /// <summary>Clicks a button or check box of the wizard.</summary>
    public void Click(string button) => Click(Window, button);

    public void Click(AutomationElement root, string button)
    {
        Note(root);
        PostMessageW(new IntPtr(Control(root, button).Current.NativeWindowHandle), BM_CLICK, IntPtr.Zero, IntPtr.Zero);
        Thread.Sleep(500);
    }

    /// <summary>A message box (a top-level window, or one the wizard owns,
    /// whatever its title) holding a text that contains
    /// <paramref name="textContains"/>.</summary>
    public AutomationElement WaitForDialog(string textContains, TimeSpan t)
    {
        AutomationElement? found = null;
        bool Match()
        {
            var main = FindWindow();
            // UI Automation lists a dialog the wizard owns under the wizard.
            var candidates = AutomationElement.RootElement.FindAll(TreeScope.Children, Condition.TrueCondition)
                .Cast<AutomationElement>()
                .Where(w => main == null || w.Current.NativeWindowHandle != main.Current.NativeWindowHandle)
                .ToList();
            if (main != null)
                candidates.AddRange(main.FindAll(TreeScope.Descendants,
                    new PropertyCondition(AutomationElement.ControlTypeProperty, ControlType.Window)).Cast<AutomationElement>());
            foreach (var w in candidates)
            {
                var texts = w.FindAll(TreeScope.Descendants, Condition.TrueCondition).Cast<AutomationElement>().Select(Uia.NameOf);
                if (texts.Any(x => x.Contains(textContains, StringComparison.Ordinal))) { found = w; return true; }
            }
            return false;
        }
        if (!Uia.WaitUntil(Match, t, 250))
            throw new Xunit.Sdk.XunitException($"no installer dialog saying '{textContains}' within {t.TotalSeconds:0} s");
        Note(found!);
        Seen.Add(Uia.NameOf(found!));
        return found!;
    }

    public async Task<int> WaitForExitAsync(TimeSpan timeout)
    {
        using var cts = new CancellationTokenSource(timeout);
        await Process.WaitForExitAsync(cts.Token);
        return Process.ExitCode;
    }

    /// <summary>The wizard's window as a picture, into the results folder (never committed).</summary>
    public void Screenshot(string name)
    {
        try
        {
            var w = Window;
            var r = w.Current.BoundingRectangle;
            using var bmp = new Bitmap((int)r.Width, (int)r.Height);
            using (var g = Graphics.FromImage(bmp))
            {
                var hdc = g.GetHdc();
                PrintWindow(new IntPtr(w.Current.NativeWindowHandle), hdc, 2);
                g.ReleaseHdc(hdc);
            }
            bmp.Save(Path.Combine(Uia.Results, name + ".png"), ImageFormat.Png);
        }
        catch (Exception) { }
    }

    static string Repo([System.Runtime.CompilerServices.CallerFilePath] string here = "") =>
        Path.GetFullPath(Path.Combine(Path.GetDirectoryName(here)!, "..", "..", "..", "..", ".."));

    /// <summary>installer_messages.isl as section -> key -> value
    /// (tests/test_windows_installer.py's parse_isl).</summary>
    public static Dictionary<string, Dictionary<string, string>> ParseIsl(string path)
    {
        var o = new Dictionary<string, Dictionary<string, string>>();
        Dictionary<string, string>? section = null;
        foreach (var raw in File.ReadAllText(path, Encoding.ASCII).Split('\n'))
        {
            var line = raw.TrimEnd('\r');
            if (line.Trim().Length == 0 || line.TrimStart().StartsWith(';')) continue;
            var m = Regex.Match(line.Trim(), @"^\[(\w+)\]$");
            if (m.Success) { o[m.Groups[1].Value] = section = new(); continue; }
            var eq = line.IndexOf('=');
            if (eq < 0 || section == null) throw new FormatException("not a key=value line: " + line);
            section[line[..eq].Trim()] = line[(eq + 1)..];
        }
        return o;
    }

    public static readonly string[] Buttons = ["Next", "Install", "Cancel", "Back", "Finish", "OK", "Yes", "No"];

    /// <summary>Every text the installer may show: our messages with their
    /// placeholders filled, one entry per paragraph, plus Inno's buttons.</summary>
    public static IReadOnlySet<string> AllowedTexts(string version)
    {
        var isl = ParseIsl(Path.Combine(Repo(), "desktop", "windows", "installer_messages.isl"));
        var allowed = new HashSet<string>(Buttons, StringComparer.Ordinal);
        foreach (var section in isl.Values)
            foreach (var value in section.Values)
            {
                var text = value.Replace("&", "").Replace("[name]", "Food Optimizer").Replace("[ver]", version)
                    .Replace("%1", "Food Optimizer");
                foreach (var para in text.Split("%n%n")) allowed.Add(para.Replace("%n", " ").Trim());
            }
        return allowed;
    }

    /// <summary>The texts of Seen that are not allowed: each is split into
    /// paragraphs first (Inno joins a page's messages with blank lines), and
    /// a path on the install folder's drive (the file being copied, which
    /// Inno shortens with "\...\") is allowed.</summary>
    public IReadOnlyList<string> NotAllowed(string version, string appDir)
    {
        var allowed = AllowedTexts(version);
        var bad = new List<string>();
        foreach (var text in Seen)
            foreach (var para in Regex.Split(text.Replace("\r\n", "\n"), @"\n\s*\n"))
            {
                var p = Regex.Replace(para, @"\s+", " ").Trim();
                if (p.Length == 0 || allowed.Contains(p)) continue;
                if (p.StartsWith(appDir, StringComparison.OrdinalIgnoreCase)) continue;
                if (p.StartsWith(Path.GetPathRoot(appDir)!, StringComparison.OrdinalIgnoreCase)) continue;
                bad.Add(p);
            }
        return bad;
    }

    public void Dispose()
    {
        try { if (!Process.HasExited) Process.Kill(entireProcessTree: true); } catch (Exception) { }
        foreach (var p in Process.GetProcesses())
        {
            try
            {
                if (p.ProcessName.EndsWith(".tmp", StringComparison.OrdinalIgnoreCase) &&
                    Regex.IsMatch(p.ProcessName, "^FoodOptimizer-.*Windows", RegexOptions.IgnoreCase))
                    p.Kill(entireProcessTree: true);
            }
            catch (Exception) { }
        }
        Process.Dispose();
    }
}
