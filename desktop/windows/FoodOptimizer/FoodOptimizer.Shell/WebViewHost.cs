using Microsoft.Web.WebView2.Core;

namespace FoodOptimizer.Shell;

/// <summary>Hosts a raw CoreWebView2Controller in a WinForms control: it
/// keeps the web view's bounds, visibility and focus in step with the form.</summary>
sealed class WebViewHost : Control
{
    static readonly Color Background = Color.FromArgb(0xF7, 0xF6, 0xF2);

    public WebViewHost()
    {
        BackColor = Background;
        SetStyle(ControlStyles.Selectable, true);
        TabStop = true;
    }

    public CoreWebView2Controller? Controller { get; private set; }
    public CoreWebView2Environment? Environment { get; private set; }
    public CoreWebView2? Core => Controller?.CoreWebView2;

    public async Task InitializeAsync(CoreWebView2Environment env, double zoom)
    {
        Environment = env;
        var controller = await env.CreateCoreWebView2ControllerAsync(Handle);
        controller.DefaultBackgroundColor = Background;
        controller.ZoomFactor = zoom;
        controller.Bounds = ClientRectangle;
        controller.IsVisible = FindForm()?.WindowState != FormWindowState.Minimized;
        Controller = controller;
    }

    /// <summary>Closes the web view (its browser process is gone or the window is closing).</summary>
    public void CloseController()
    {
        var c = Controller;
        Controller = null;
        try { c?.Close(); } catch (Exception) { }
    }

    /// <summary>After the browser process has gone: closes the dead view,
    /// waits (at most 5 s) for the old environment's processes to exit, and
    /// creates a new environment and view.</summary>
    public async Task RecreateAsync(Func<Task<CoreWebView2Environment>> create, double zoom)
    {
        var old = Environment;
        var exited = new TaskCompletionSource();
        if (old != null) old.BrowserProcessExited += (_, _) => exited.TrySetResult();
        CloseController();
        await Task.WhenAny(exited.Task, Task.Delay(TimeSpan.FromSeconds(5)));
        await InitializeAsync(await create(), zoom);
    }

    protected override void OnResize(EventArgs e)
    {
        base.OnResize(e);
        if (Controller != null) Controller.Bounds = ClientRectangle;
    }

    protected override void OnGotFocus(EventArgs e)
    {
        base.OnGotFocus(e);
        try { Controller?.MoveFocus(CoreWebView2MoveFocusReason.Programmatic); } catch (Exception) { }
    }

    protected override void OnParentChanged(EventArgs e)
    {
        base.OnParentChanged(e);
        if (FindForm() is { } form)
        {
            form.Move += (_, _) => { try { Controller?.NotifyParentWindowPositionChanged(); } catch (Exception) { } };
            form.Resize += (_, _) =>
            {
                if (Controller != null) Controller.IsVisible = form.WindowState != FormWindowState.Minimized;
            };
        }
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing) CloseController();
        base.Dispose(disposing);
    }
}
