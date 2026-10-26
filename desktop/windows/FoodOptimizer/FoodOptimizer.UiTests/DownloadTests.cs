using System.Runtime.CompilerServices;
using System.Windows.Automation;
using WinForms = System.Windows.Forms;

namespace FoodOptimizer.UiTests;

public class DownloadTests
{
    static readonly TimeSpan Three = TimeSpan.FromSeconds(3);
    static readonly TimeSpan Ten = TimeSpan.FromSeconds(10);
    const string Csv = "a,b\n1,2\n";

    static (ShellApp App, string Saved) Loaded([CallerMemberName] string test = "")
    {
        var app = ShellApp.Launch(ShellRun.Fresh(test));
        app.WaitFor(ControlType.Edit, "Type here", ShellApp.Load);
        var saved = Path.Combine(app.Run.TempRoot, "saved");
        Directory.CreateDirectory(saved);
        return (app, saved);
    }

    static AutomationElement WaitForWindow(ShellApp app, string name)
    {
        AutomationElement? w = null;
        Assert.True(Uia.WaitUntil(() => (w = app.Dialog(name) ?? Uia.TopWindow(name)) != null, Ten), "no window " + name);
        Thread.Sleep(1500);   // a file dialog keeps building itself after it appears
        return app.Dialog(name) ?? Uia.TopWindow(name) ?? w!;
    }

    /// <summary>The file dialog's File name box (control id 1001 in Save
    /// As, 1148 in Open), once the dialog has finished building itself.</summary>
    static AutomationElement FileNameBox(AutomationElement dialog)
    {
        AutomationElement? box = null;
        Assert.True(Uia.WaitUntil(() => (box = Uia.All(dialog, ControlType.Edit).FirstOrDefault(e =>
            e.Current.AutomationId is "1001" or "1148" || Uia.NameOf(e).StartsWith("File name", StringComparison.Ordinal))) != null, Ten),
            "no File name box");
        return box!;
    }

    /// <summary>A window a file dialog opened over itself (UI Automation
    /// lists it under that dialog).</summary>
    static AutomationElement ChildWindow(AutomationElement parent, string name)
    {
        AutomationElement? w = null;
        Assert.True(Uia.WaitUntil(() => (w = parent.FindFirst(TreeScope.Children,
            new AndCondition(new PropertyCondition(AutomationElement.ControlTypeProperty, ControlType.Window),
                             new PropertyCondition(AutomationElement.NameProperty, name)))) != null, Ten), "no window " + name);
        return w!;
    }

    /// <summary>Clicks a dialog button without waiting for what it opens
    /// (a Win32 button's Invoke returns only when its handler does).</summary>
    static void Press(AutomationElement dialog, string button)
    {
        var b = Uia.WaitFor(dialog, ControlType.Button, button, Three);
        _ = Task.Run(() => { try { Uia.Invoke(b); } catch (Exception) { } });
        Thread.Sleep(800);
    }

    static void SetFileName(AutomationElement dialog, string value)
    {
        var box = FileNameBox(dialog);
        ((ValuePattern)box.GetCurrentPattern(ValuePattern.Pattern)).SetValue(value);
        // SetValue alone changes the text without the dialog noticing: a
        // typed space and backspace make it read the box again.
        box.SetFocus();
        Thread.Sleep(200);
        WinForms.SendKeys.SendWait("{END} {BACKSPACE}");
        Thread.Sleep(300);
        Assert.Equal(value, Uia.Value(box));
    }

    /// <summary>Clicks Download a CSV and answers the Save As dialog with <paramref name="path"/>.</summary>
    static AutomationElement SaveTo(ShellApp app, string path)
    {
        Uia.Invoke(app.WaitFor(ControlType.Button, "Download a CSV", Ten));
        var dialog = WaitForWindow(app, "Save As");
        // Explorer may hide known extensions in the box.
        Assert.Contains(Uia.Value(FileNameBox(dialog)), new[] { "probe.csv", "probe" });
        SetFileName(dialog, path);
        Press(dialog, "Save");
        return dialog;
    }

    static void CloseExplorer(string name)
    {
        if (Uia.TopWindow(name, "CabinetWClass") is { } w)
        {
            try { ((WindowPattern)w.GetCurrentPattern(WindowPattern.Pattern)).Close(); } catch (Exception) { }
        }
    }

    [Fact]
    public void Download_asks_where_and_saves_without_the_web_mark()
    {
        var (app, saved) = Loaded();
        using var _ = app;
        var path = Path.Combine(saved, "probe.csv");
        SaveTo(app, path);
        Assert.True(Uia.WaitUntil(() => File.Exists(path) && File.ReadAllText(path) == Csv, Ten), "not saved");

        var control = Path.Combine(saved, "control.txt");
        File.WriteAllText(control, "x");
        File.WriteAllText(control + ":Zone.Identifier", "[ZoneTransfer]\r\nZoneId=3\r\n");
        Assert.True(File.Exists(control + ":Zone.Identifier"));
        File.Delete(control);
        Assert.True(Uia.WaitUntil(() => app.Log().Any(l => l.Contains("download.done")), Ten), string.Join("\n", app.Log()));
        Assert.False(File.Exists(path + ":Zone.Identifier"));

        Assert.True(Uia.WaitUntil(() => Uia.TopWindow("saved", "CabinetWClass") != null, Ten), "no Explorer window");
        CloseExplorer("saved");
        Assert.Null(Uia.Find(app.Window, ControlType.Text, "Downloads"));
        Assert.Null(Uia.Find(app.Window, ControlType.Button, "Downloads"));
    }

    [Fact]
    public void Download_over_an_existing_file_replaces_it()
    {
        var (app, saved) = Loaded();
        using var _ = app;
        var path = Path.Combine(saved, "probe.csv");
        File.WriteAllText(path, "old contents that are longer than the new ones\n");
        Press(ChildWindow(SaveTo(app, path), "Confirm Save As"), "Yes");
        Assert.True(Uia.WaitUntil(() => File.Exists(path) && File.ReadAllText(path) == Csv, Ten), "not replaced");
        Assert.True(Uia.WaitUntil(() => app.Log().Any(l => l.Contains("download.done")), Ten));
        CloseExplorer("saved");
        Assert.Equal(["probe.csv"], Directory.GetFiles(saved).Select(Path.GetFileName));
    }

    [Fact]
    public void Cancelled_download_saves_nothing()
    {
        var (app, saved) = Loaded();
        using var _ = app;
        Uia.Invoke(app.WaitFor(ControlType.Button, "Download a CSV", Ten));
        Press(WaitForWindow(app, "Save As"), "Cancel");
        Thread.Sleep(2000);
        Assert.Empty(Directory.GetFiles(saved));
        Assert.Null(app.Dialog("Food Optimizer"));
        Assert.True(app.Log().Any(l => l.Contains("download.cancelled")), string.Join("\n", app.Log()));
    }

    [Fact]
    public void Download_over_a_file_open_elsewhere_says_it_could_not_be_saved()
    {
        var (app, saved) = Loaded();
        using var _ = app;
        var path = Path.Combine(saved, "probe.csv");
        File.WriteAllText(path, "held\n");
        using (new FileStream(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None))
        {
            Press(ChildWindow(SaveTo(app, path), "Confirm Save As"), "Yes");
            var d = app.WaitForDialog(Ten);
            Uia.WaitFor(d, ControlType.Text, "The file could not be saved", Three);
            Uia.WaitFor(d, ControlType.Text, "Try again and pick another folder, such as your FoodOptimizer folder.", Three);
            Uia.Invoke(Uia.WaitFor(d, ControlType.Button, "OK", Three));
        }
        Assert.Equal("held\n", File.ReadAllText(path));
        Assert.Empty(Directory.GetFiles(saved, "*.crdownload"));
        Assert.Equal(["probe.csv"], Directory.GetFiles(saved).Select(Path.GetFileName));
    }

    /// <summary>Clicks <paramref name="button"/>, whose file the server
    /// names "&lt;base&gt; (2).csv", with "&lt;base&gt;.csv" in the real Downloads
    /// folder (where WebView2 proposes names), and checks the Save As box
    /// offers the server's name. The old rule took the " (2)" for a number
    /// WebView2 added. Only the files this test creates are removed after.</summary>
    static void OffersTheServersNumberedName(string button, string baseName, [CallerMemberName] string test = "")
    {
        var downloads = FoodOptimizer.Core.Windows.WinShell.DownloadsFolder();
        Assert.False(string.IsNullOrEmpty(downloads), "no Downloads folder");
        var plain = Path.Combine(downloads!, baseName + ".csv");
        var numbered = Path.Combine(downloads!, baseName + " (2).csv");
        Assert.False(File.Exists(numbered), numbered + " is already there; move it away to run this test");
        var before = new HashSet<string>(Directory.GetFiles(downloads!), StringComparer.OrdinalIgnoreCase);
        var madePlain = !File.Exists(plain);
        if (madePlain) File.WriteAllText(plain, Csv);
        try
        {
            var (app, _) = Loaded(test);
            using var _app = app;
            Uia.Invoke(app.WaitFor(ControlType.Button, button, Ten));
            var dialog = WaitForWindow(app, "Save As");
            // Explorer may hide known extensions in the box.
            Assert.Contains(Uia.Value(FileNameBox(dialog)), new[] { baseName + " (2).csv", baseName + " (2)" });
            Press(dialog, "Cancel");
            Assert.True(Uia.WaitUntil(() => app.Log().Any(l => l.Contains("download.cancelled")), Ten), string.Join("\n", app.Log()));
        }
        finally
        {
            // WebView2 downloads into "<guid>.tmp" beside the proposal while
            // the dialog is open; a test that fails there leaves it behind,
            // still held for a moment by the closing web view.
            var made = Directory.GetFiles(downloads!).Where(f => !before.Contains(f) && (f.Equals(numbered, StringComparison.OrdinalIgnoreCase)
                || f.EndsWith(".tmp", StringComparison.OrdinalIgnoreCase) || f.EndsWith(".crdownload", StringComparison.OrdinalIgnoreCase)));
            if (madePlain) made = made.Append(plain);
            var left = made.ToList();
            Uia.WaitUntil(() =>
            {
                left.RemoveAll(f =>
                {
                    try { File.Delete(f); return true; }
                    catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { return false; }
                });
                return left.Count == 0;
            }, Ten);
        }
    }

    [Fact]
    public void A_file_really_named_with_a_number_keeps_it() =>
        OffersTheServersNumberedName("Download a numbered CSV", "probe");

    [Fact]
    public void An_accented_name_keeps_its_accents_and_number() =>
        // Streamlit sends a latin-1 name as raw bytes in filename="…".
        OffersTheServersNumberedName("Download an accented CSV", "Crème");

    [Fact]
    public void Upload_picks_several_files()
    {
        var (app, saved) = Loaded();
        using var _ = app;
        File.WriteAllText(Path.Combine(saved, "a.csv"), "a\n1\n");
        File.WriteAllText(Path.Combine(saved, "b.csv"), "b\n2\n");
        Uia.Invoke(app.WaitFor(ControlType.Button, "Browse files", Ten));
        var open = WaitForWindow(app, "Open");
        SetFileName(open, saved);
        Press(open, "Open");
        Thread.Sleep(1500);
        open = WaitForWindow(app, "Open");
        SetFileName(open, "\"a.csv\" \"b.csv\"");
        Press(open, "Open");
        app.WaitFor(ControlType.Text, "uploaded: a.csv, b.csv", Ten);
    }
}
