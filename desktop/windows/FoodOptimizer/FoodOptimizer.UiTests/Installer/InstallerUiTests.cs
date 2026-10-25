using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Windows.Automation;
using FoodOptimizer.Core;

namespace FoodOptimizer.UiTests;

/// <summary>Helpers the installer and installed-app tests share: a per-test
/// user folder (with a non-ASCII name), silent installs and uninstalls, and
/// the app's own title.</summary>
static class InstallerRun
{
    public static readonly string[] Silent = ["/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CLOSEAPPLICATIONS"];

    /// <summary>C:\FOUI\<tag>\José Müller: short, so the install folder
    /// beside it stays inside the installer's 80-character budget.</summary>
    public static string Root(string tag) =>
        Path.Combine(Path.GetPathRoot(Path.GetTempPath())!, "FOUI", tag, "José Müller");

    public static ShellRun Fresh(string tag)
    {
        var root = Root(tag);
        if (Directory.Exists(root)) Directory.Delete(root, true);
        Directory.CreateDirectory(root);
        return new ShellRun(Path.Combine(root, "Food Optimizer"), root);
    }

    /// <summary>The shell's and the installer's folders for a run.</summary>
    public static Dictionary<string, string> Env(ShellRun run) => new()
    {
        ["FOODOPT_STATE_DIR"] = run.StateDir,
        ["FOODOPT_PROJECTS_DIR"] = run.ProjectsDir,
    };

    public static int Run(string exe, IEnumerable<string> args, IDictionary<string, string> env, TimeSpan timeout)
    {
        var info = new ProcessStartInfo(exe) { UseShellExecute = false };
        foreach (var a in args) info.ArgumentList.Add(a);
        foreach (var (k, v) in env) info.Environment[k] = v;
        using var p = Process.Start(info)!;
        if (!p.WaitForExit(timeout)) { p.Kill(true); throw new Xunit.Sdk.XunitException(exe + " timed out"); }
        return p.ExitCode;
    }

    public static void SilentInstall(ShellRun run, string? log = null)
    {
        var args = new List<string>(Silent) { "/DIR=" + run.Stage };
        if (log != null) args.Add("/LOG=" + log);
        Assert.Equal(0, Run(SetupWizard.Installer, args, Env(run), TimeSpan.FromMinutes(30)));
    }

    /// <summary>Uninstalls the copy in <paramref name="dir"/> silently, if
    /// there is one, and waits for its folder to go.</summary>
    public static void SilentUninstall(string dir, IDictionary<string, string> env)
    {
        ShellApp.KillStage(dir);
        var unins = Path.Combine(dir, "unins000.exe");
        if (!File.Exists(unins)) return;
        Run(unins, ["/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"], env, TimeSpan.FromMinutes(10));
        Uia.WaitUntil(() => !Directory.Exists(dir), TimeSpan.FromSeconds(60), 500);
    }

    /// <summary>wording.APP_TITLE, read with the install's own Python.</summary>
    public static string AppTitle(string dir)
    {
        var info = new ProcessStartInfo(Path.Combine(dir, "python", "python.exe"))
        {
            UseShellExecute = false,
            RedirectStandardOutput = true,
            CreateNoWindow = true,
        };
        foreach (var a in new[] { "-I", "-c", "import sys, wording; sys.stdout.write(wording.APP_TITLE)" })
            info.ArgumentList.Add(a);
        using var p = Process.Start(info)!;
        var title = p.StandardOutput.ReadToEnd();
        p.WaitForExit();
        Assert.Equal(0, p.ExitCode);
        Assert.False(string.IsNullOrEmpty(title));
        return title;
    }

    public static bool ShowsDocument(AutomationElement window, string title) =>
        Uia.All(window, ControlType.Document).Any(d => Uia.NameOf(d) == title);

    [DllImport("advapi32.dll", SetLastError = true)]
    static extern bool OpenProcessToken(IntPtr process, uint access, out IntPtr token);
    [DllImport("advapi32.dll", SetLastError = true)]
    static extern bool GetTokenInformation(IntPtr token, int cls, out int value, int size, out int returned);
    [DllImport("kernel32.dll")] static extern bool CloseHandle(IntPtr h);

    /// <summary>GetTokenInformation(TokenElevation) of another process.</summary>
    public static bool IsElevated(Process p)
    {
        if (!OpenProcessToken(p.Handle, 0x0008, out var token)) throw new System.ComponentModel.Win32Exception();
        try
        {
            if (!GetTokenInformation(token, 20, out var elevated, 4, out _)) throw new System.ComponentModel.Win32Exception();
            return elevated != 0;
        }
        finally { CloseHandle(token); }
    }

    /// <summary>FoodOptimizer.exe processes started from <paramref name="dir"/>.</summary>
    public static IReadOnlyList<Process> Shells(string dir) =>
        Process.GetProcessesByName("FoodOptimizer").Where(p =>
        {
            try { return p.MainModule?.FileName?.StartsWith(dir, StringComparison.OrdinalIgnoreCase) == true; }
            catch (Exception) { return false; }
        }).ToList();
}

/// <summary>The installer's three pages, its app-running and uninstall
/// dialogs, all in our words (spec section 5; Review Focus 1).</summary>
[Trait("Category", "Installer")]
public class InstallerUiTests
{
    static readonly string Version = AppIdentity.Version;

    [Fact]
    public async Task Wizard_shows_three_pages_in_our_words_and_opens_the_app()
    {
        var run = InstallerRun.Fresh("wizard");
        var env = InstallerRun.Env(run);
        var dir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Programs", "Food Optimizer");
        var log = Path.Combine(Uia.Results, "wizard-setup.log");
        File.Delete(log);
        try
        {
            using var w = SetupWizard.Start(SetupWizard.Installer, "\"/LOG=" + log + "\"", env);
            Assert.True(Uia.WaitUntil(() => w.Heading() == "Install Food Optimizer", TimeSpan.FromSeconds(60)), w.Heading());
            Assert.True(w.IsChecked("Add a desktop shortcut"));
            Assert.Contains("Install", w.Texts());
            Assert.DoesNotContain("Next", w.Texts());
            w.Screenshot("installer-1-welcome");
            w.Click("Install");

            var headings = new HashSet<string>();
            Assert.True(Uia.WaitUntil(() =>
            {
                var h = w.Heading();
                if (h.Length > 0) headings.Add(h);
                return h == "Food Optimizer is installed";
            }, TimeSpan.FromSeconds(1800), 500), string.Join(" | ", headings));
            headings.ExceptWith(["Install Food Optimizer", "Food Optimizer is installed"]);
            Assert.Subset(new HashSet<string> { "Installing", "Getting ready" }, headings);

            var finish = w.Texts();
            w.Screenshot("installer-3-finish");
            w.Click("Finish");
            Assert.Equal(0, await w.WaitForExitAsync(TimeSpan.FromSeconds(60)));
            Assert.Empty(w.NotAllowed(Version, dir));

            // The Finish page ran "Open Food Optimizer" (ticked) and not
            // "Open Start Here" (unticked): Inno logs each [Run] entry it runs.
            var setupLog = File.ReadAllText(log);
            Assert.Contains("Filename: " + Path.Combine(dir, "FoodOptimizer.exe"), setupLog);
            Assert.DoesNotContain("Filename: " + Path.Combine(dir, "Start Here.txt"), setupLog);

            Process? shell = null;
            Assert.True(Uia.WaitUntil(() => (shell = InstallerRun.Shells(dir).FirstOrDefault()) != null
                && Uia.TopWindows(shell.Id).Any(x => Uia.NameOf(x) == "Food Optimizer"), TimeSpan.FromSeconds(15)),
                "no Food Optimizer window within 15 s of Finish");
            Assert.False(InstallerRun.IsElevated(shell!));
            var window = Uia.TopWindows(shell!.Id).First(x => Uia.NameOf(x) == "Food Optimizer");
            var title = InstallerRun.AppTitle(dir);
            Assert.True(Uia.WaitUntil(() => InstallerRun.ShowsDocument(window, title), TimeSpan.FromSeconds(300), 1000),
                "no page titled " + title);
            ((WindowPattern)window.GetCurrentPattern(WindowPattern.Pattern)).Close();
            Assert.True(shell.WaitForExit(10000));
            Assert.True(Uia.WaitUntil(() => ShellApp.StagePythonsOf(dir).Count == 0, TimeSpan.FromSeconds(5)));
            Uia.Record("installer_finish_texts", string.Join(" | ", finish));
        }
        finally
        {
            InstallerRun.SilentUninstall(dir, env);
        }
    }

    [Fact]
    public async Task Running_app_holds_the_installer_with_our_words()
    {
        var run = InstallerRun.Fresh("running");
        var env = InstallerRun.Env(run);
        try
        {
            InstallerRun.SilentInstall(run);
            var title = InstallerRun.AppTitle(run.Stage);
            using var app = ShellApp.Launch(run);
            Assert.True(Uia.WaitUntil(() => InstallerRun.ShowsDocument(app.Window, title), TimeSpan.FromSeconds(300), 1000));

            using var w = SetupWizard.Start(SetupWizard.Installer, "\"/DIR=" + run.Stage + "\"", env);
            var d = w.WaitForDialog("Close Food Optimizer, then click OK.", TimeSpan.FromSeconds(60));
            w.Click(d, "OK");
            Thread.Sleep(1500);
            d = w.WaitForDialog("Close Food Optimizer, then click OK.", TimeSpan.FromSeconds(30));
            app.Close();
            Assert.True(await app.WaitForExitAsync(TimeSpan.FromSeconds(10)));
            Assert.True(Uia.WaitUntil(() => app.StagePythons().Count == 0, TimeSpan.FromSeconds(10)));
            w.Click(d, "OK");
            Assert.True(Uia.WaitUntil(() => w.Heading() == "Install Food Optimizer", TimeSpan.FromSeconds(60)), w.Heading());
            w.Click("Cancel");
            var stop = w.WaitForDialog("Food Optimizer is not installed yet. Stop installing?", TimeSpan.FromSeconds(30));
            w.Click(stop, "Yes");
            Assert.Equal(2, await w.WaitForExitAsync(TimeSpan.FromSeconds(60)));
            Assert.Empty(w.NotAllowed(Version, run.Stage));
        }
        finally
        {
            InstallerRun.SilentUninstall(run.Stage, env);
        }
    }

    [Fact]
    public async Task Uninstall_by_hand_says_the_projects_are_kept()
    {
        var run = InstallerRun.Fresh("unins");
        var env = InstallerRun.Env(run);
        try
        {
            InstallerRun.SilentInstall(run);
            Directory.CreateDirectory(run.ProjectsDir);
            var project = Path.Combine(run.ProjectsDir, "My project.pkl");
            File.WriteAllBytes(project, [0x80, 0x04, 0x4b, 0x01]);

            var info = new ProcessStartInfo(Path.Combine(run.Stage, "unins000.exe")) { UseShellExecute = false };
            foreach (var (k, v) in env) info.Environment[k] = v;
            using var unins = Process.Start(info)!;
            using var w = new UninstallWatch(unins);
            var confirm = w.Wizard.WaitForDialog("Uninstall Food Optimizer? Your projects are kept.", TimeSpan.FromSeconds(60));
            w.Wizard.Click(confirm, "Yes");
            var done = w.Wizard.WaitForDialog("under FoodOptimizer", TimeSpan.FromSeconds(600));
            w.Wizard.Click(done, "OK");
            Assert.True(Uia.WaitUntil(() => !Directory.Exists(run.Stage), TimeSpan.FromSeconds(60), 500));
            Assert.Equal(new byte[] { 0x80, 0x04, 0x4b, 0x01 }, File.ReadAllBytes(project));
            Assert.Empty(w.Wizard.NotAllowed(Version, run.Stage));
            await Task.CompletedTask;
        }
        finally
        {
            InstallerRun.SilentUninstall(run.Stage, env);
        }
    }

    /// <summary>The uninstaller's dialogs, read like the wizard's.</summary>
    sealed class UninstallWatch(Process unins) : IDisposable
    {
        public SetupWizard Wizard { get; } = SetupWizardFor(unins);

        static SetupWizard SetupWizardFor(Process p) => SetupWizard.Attach(p);

        public void Dispose() => Wizard.Dispose();
    }
}
