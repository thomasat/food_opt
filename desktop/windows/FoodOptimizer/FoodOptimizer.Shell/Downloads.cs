using FoodOptimizer.Core;
using FoodOptimizer.Core.Windows;
using Microsoft.Web.WebView2.Core;

namespace FoodOptimizer.Shell;

/// <summary>Downloads ask where to save (starting in Downloads), replace a
/// file cleanly, drop the web mark from files our server sent (so Excel
/// doesn't open round sheets in Protected View), and then show the file in
/// Explorer with the Windows notification sound. No Edge download flyout.</summary>
sealed class Downloads
{
    readonly Form _owner;
    readonly LaunchController _launch;
    readonly Messages _messages;
    readonly ShellLog _log;

    public Downloads(Form owner, LaunchController launch, Messages messages, ShellLog log)
    {
        _owner = owner;
        _launch = launch;
        _messages = messages;
        _log = log;
    }

    public void Attach(CoreWebView2 core) => core.DownloadStarting += OnDownloadStarting;

    void ShowFailed() =>
        ShellDialogs.Info(_owner, _messages.Text("download.failed.title"), _messages.Text("download.failed.body"));

    void OnDownloadStarting(object? sender, CoreWebView2DownloadStartingEventArgs e)
    {
        var deferral = e.GetDeferral();
        _owner.BeginInvoke(() =>
        {
            var failed = false;
            try
            {
                var name = DownloadRules.SuggestedFileName(e.ResultFilePath, e.DownloadOperation.ContentDisposition, File.Exists);
                using var dialog = new SaveFileDialog
                {
                    FileName = name,
                    InitialDirectory = DownloadRules.InitialDirectory(WinShell.DownloadsFolder(),
                        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), Directory.Exists),
                    OverwritePrompt = true,
                    AddExtension = true,
                    DefaultExt = Path.GetExtension(name).TrimStart('.'),
                };
                if (dialog.ShowDialog(_owner) != DialogResult.OK)
                {
                    e.Cancel = true;
                    e.Handled = true;
                    _log.Write("download.cancelled");
                    return;
                }
                var path = dialog.FileName;
                if (File.Exists(path))
                {
                    try
                    {
                        File.Delete(path);
                    }
                    catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
                    {
                        _log.Write("download.failed", ex.GetType().Name);
                        e.Cancel = true;
                        e.Handled = true;
                        failed = true;
                        return;
                    }
                }
                e.ResultFilePath = path;
                e.Handled = true;
                Track(e.DownloadOperation, path);
            }
            catch (Exception ex)
            {
                _log.Write("download.failed", ex.GetType().Name);
                e.Cancel = true;
                failed = true;
            }
            finally
            {
                deferral.Complete();
                if (failed) ShowFailed();
            }
        });
    }

    void Track(CoreWebView2DownloadOperation op, string path)
    {
        var uri = op.Uri;
        op.StateChanged += (_, _) =>
        {
            switch (op.State)
            {
                case CoreWebView2DownloadState.Completed:
                    if (DownloadRules.IsFromOurServer(uri, _launch.Port) && !WinShell.RemoveZoneIdentifier(path))
                        _log.Write("download.mark_kept");
                    WinShell.RevealInExplorer(path);
                    WinShell.PlayNotificationSound();
                    _log.Write("download.done");
                    break;
                case CoreWebView2DownloadState.Interrupted:
                    if (op.InterruptReason == CoreWebView2DownloadInterruptReason.UserCanceled)
                    {
                        _log.Write("download.cancelled");
                        break;
                    }
                    _log.Write("download.failed", op.InterruptReason.ToString());
                    // A modal dialog inside a WebView2 event handler would
                    // hold WebView2 up; show it once the handler has returned.
                    _owner.BeginInvoke(ShowFailed);
                    break;
            }
        };
    }
}
