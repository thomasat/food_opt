using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Windows.Automation;

namespace FoodOptimizer.UiTests;

/// <summary>Where the staged install is and where results go (set by
/// uitests/run_ui_tests.ps1), plus small UI Automation helpers.</summary>
static class Uia
{
    public static string Stage =>
        Environment.GetEnvironmentVariable("FOODOPT_UI_STAGE") is { Length: > 0 } s
            ? s : throw new InvalidOperationException("FOODOPT_UI_STAGE is not set");

    public static string Results
    {
        get
        {
            var r = Environment.GetEnvironmentVariable("FOODOPT_UI_RESULTS") is { Length: > 0 } s
                ? s : Path.Combine(Path.GetTempPath(), "foodopt-ui-results");
            Directory.CreateDirectory(r);
            return r;
        }
    }

    /// <summary>Appends "name value" to timings.txt in the results folder.</summary>
    public static void Record(string name, object value) =>
        File.AppendAllText(Path.Combine(Results, "timings.txt"), $"{DateTime.Now:HH:mm:ss} {name} {value}\r\n");

    [DllImport("user32.dll")]
    static extern bool SetProcessDpiAwarenessContext(IntPtr value);

    [ModuleInitializer]
    internal static void DpiAware() => SetProcessDpiAwarenessContext(new IntPtr(-4));

    public static bool WaitUntil(Func<bool> condition, TimeSpan timeout, int pollMs = 200)
    {
        var sw = Stopwatch.StartNew();
        while (sw.Elapsed < timeout)
        {
            try { if (condition()) return true; }
            catch (ElementNotAvailableException) { }
            catch (ExternalException) { }   // COM errors, and the clipboard held for a moment
            Thread.Sleep(pollMs);
        }
        try { return condition(); }
        catch (Exception) { return false; }
    }

    static Condition TypeIs(ControlType t) => new PropertyCondition(AutomationElement.ControlTypeProperty, t);

    /// <summary>Elements under <paramref name="root"/> of this type (any
    /// type for Text: Chromium names headings and paragraphs variously).</summary>
    public static IEnumerable<AutomationElement> All(AutomationElement root, ControlType t)
    {
        var condition = t == ControlType.Text ? Condition.TrueCondition : TypeIs(t);
        return root.FindAll(TreeScope.Descendants, condition).Cast<AutomationElement>();
    }

    public static string NameOf(AutomationElement e)
    {
        try { return e.Current.Name ?? ""; }
        catch (Exception) { return ""; }
    }

    /// <summary>The element whose Name is <paramref name="name"/>, else one
    /// whose Name contains it; null when there is none.</summary>
    public static AutomationElement? Find(AutomationElement root, ControlType t, string name)
    {
        var all = All(root, t).ToList();
        return all.FirstOrDefault(e => NameOf(e) == name)
            ?? all.FirstOrDefault(e => NameOf(e).Contains(name, StringComparison.Ordinal));
    }

    public static AutomationElement WaitFor(AutomationElement root, ControlType t, string name, TimeSpan timeout)
    {
        AutomationElement? found = null;
        if (WaitUntil(() => (found = Find(root, t, name)) != null, timeout, 250)) return found!;
        var seen = string.Join(" | ", All(root, t).Select(NameOf).Where(n => n.Length > 0).Distinct().Take(40));
        throw new Xunit.Sdk.XunitException($"no {t.ProgrammaticName} named '{name}' within {timeout.TotalSeconds:0} s; saw: {seen}");
    }

    public static void Invoke(AutomationElement e) =>
        ((InvokePattern)e.GetCurrentPattern(InvokePattern.Pattern)).Invoke();

    public static string Value(AutomationElement e) =>
        ((ValuePattern)e.GetCurrentPattern(ValuePattern.Pattern)).Current.Value;

    /// <summary>Top-level windows of a process.</summary>
    public static IReadOnlyList<AutomationElement> TopWindows(int pid) =>
        AutomationElement.RootElement.FindAll(TreeScope.Children,
            new PropertyCondition(AutomationElement.ProcessIdProperty, pid)).Cast<AutomationElement>().ToList();

    /// <summary>A top-level window, of any process, whose Name is (or contains) <paramref name="name"/>.</summary>
    public static AutomationElement? TopWindow(string name, string? className = null)
    {
        var all = AutomationElement.RootElement.FindAll(TreeScope.Children, Condition.TrueCondition)
            .Cast<AutomationElement>()
            .Where(w => className == null || SafeClass(w) == className).ToList();
        return all.FirstOrDefault(w => NameOf(w) == name)
            ?? all.FirstOrDefault(w => NameOf(w).Contains(name, StringComparison.Ordinal));
    }

    public static string SafeClass(AutomationElement e)
    {
        try { return e.Current.ClassName ?? ""; }
        catch (Exception) { return ""; }
    }
}
