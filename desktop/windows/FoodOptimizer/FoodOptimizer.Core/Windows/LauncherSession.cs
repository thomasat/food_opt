using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.Versioning;

namespace FoodOptimizer.Core.Windows;

/// <summary>The launcher could not be started (the message is Windows' reason).</summary>
public sealed class LauncherStartException(string message, Exception? inner = null) : Exception(message, inner);

/// <summary>One running launcher: hidden, in its own Job Object, with its
/// stdin held open as the lifeline (closing it asks the launcher to stop).
/// Its output is read and dropped so it never blocks on a full pipe.</summary>
[SupportedOSPlatform("windows")]
public sealed class LauncherSession : IDisposable
{
    readonly Process _process;
    readonly JobObject _job;
    readonly ShellLog _log;
    int _lifelineClosed, _jobClosed;

    LauncherSession(Process process, JobObject job, ShellLog log, DateTime startedUtc)
    {
        _process = process;
        _job = job;
        _log = log;
        StartedUtc = startedUtc;
        ProcessId = process.Id;
    }

    public int ProcessId { get; }
    public DateTime StartedUtc { get; }

    /// <summary>Raised with the exit code, on a thread-pool thread.</summary>
    public event EventHandler<int>? Exited;

    public static LauncherSession Start(string python, IReadOnlyList<string> args, string workingDir, ShellLog log)
    {
        var info = new ProcessStartInfo(python)
        {
            UseShellExecute = false,
            CreateNoWindow = true,
            WorkingDirectory = workingDir,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        foreach (var a in args) info.ArgumentList.Add(a);

        JobObject job;
        try { job = new JobObject(); }
        catch (Win32Exception e) { throw new LauncherStartException(e.Message, e); }

        var process = new Process { StartInfo = info, EnableRaisingEvents = true };
        var started = DateTime.UtcNow;
        try
        {
            process.Start();
        }
        catch (Exception e) when (e is Win32Exception or InvalidOperationException or IOException)
        {
            job.Dispose();
            process.Dispose();
            log.Write("launcher.start_failed", e.Message);
            throw new LauncherStartException(e.Message, e);
        }
        try
        {
            job.Assign(process);
        }
        catch (Win32Exception e)
        {
            try { process.Kill(entireProcessTree: true); } catch (Exception) { }
            job.Dispose();
            process.Dispose();
            log.Write("launcher.start_failed", e.Message);
            throw new LauncherStartException(e.Message, e);
        }

        var session = new LauncherSession(process, job, log, started);
        process.OutputDataReceived += (_, _) => { };
        process.ErrorDataReceived += (_, _) => { };
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();
        process.Exited += session.OnExited;
        log.Write("launcher.start", session.ProcessId.ToString(System.Globalization.CultureInfo.InvariantCulture));
        if (process.HasExited) session.OnExited(process, EventArgs.Empty);
        return session;
    }

    int _exitRaised;

    void OnExited(object? sender, EventArgs e)
    {
        if (Interlocked.Exchange(ref _exitRaised, 1) == 1) return;
        int code;
        try { code = _process.ExitCode; }
        catch (InvalidOperationException) { code = -1; }
        _log.Write("launcher.exit", code.ToString(System.Globalization.CultureInfo.InvariantCulture));
        Exited?.Invoke(this, code);
    }

    /// <summary>Closes the launcher's stdin; it stops the server and exits.</summary>
    public void CloseLifeline()
    {
        if (Interlocked.Exchange(ref _lifelineClosed, 1) == 1) return;
        try { _process.StandardInput.Close(); }
        catch (Exception) { }
    }

    public async Task<bool> WaitForExitAsync(TimeSpan timeout)
    {
        using var cts = new CancellationTokenSource(timeout);
        try
        {
            await _process.WaitForExitAsync(cts.Token).ConfigureAwait(false);
            return true;
        }
        catch (OperationCanceledException)
        {
            return _process.HasExited;
        }
    }

    /// <summary>Kills the launcher and everything it started.</summary>
    public void KillAll()
    {
        if (Interlocked.Exchange(ref _jobClosed, 1) == 1) return;
        _job.Dispose();
    }

    public void Dispose()
    {
        CloseLifeline();
        KillAll();
    }
}
