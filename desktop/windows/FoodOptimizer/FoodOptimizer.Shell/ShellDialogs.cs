using FoodOptimizer.Core;

namespace FoodOptimizer.Shell;

/// <summary>Native dialogs captioned "Food Optimizer", with button labels
/// from messages.json (never MessageBox, whose buttons the shell can't name).</summary>
static class ShellDialogs
{
    static Messages? _messages;

    public static Messages Messages
    {
        get => _messages ?? throw new InvalidOperationException(nameof(Messages));
        set => _messages = value;
    }

    static TaskDialogPage Page(string heading, string body) => new()
    {
        Caption = Messages.Text("dialog.title"),
        Heading = string.IsNullOrEmpty(heading) ? null : heading,
        Text = string.IsNullOrEmpty(body) ? null : body,
        AllowCancel = true,
        SizeToContent = true,
    };

    static TaskDialogButton Show(IWin32Window? owner, TaskDialogPage page) =>
        owner != null
            ? TaskDialog.ShowDialog(owner, page, TaskDialogStartupLocation.CenterOwner)
            : TaskDialog.ShowDialog(page, TaskDialogStartupLocation.CenterScreen);

    public static void Info(IWin32Window? owner, string heading, string body)
    {
        var page = Page(heading, body);
        page.Buttons.Add(new TaskDialogButton(Messages.Text("dialog.ok")));
        Show(owner, page);
    }

    /// <summary>A heading alone, with OK.</summary>
    public static void Info(IWin32Window? owner, string heading) => Info(owner, heading, "");

    /// <summary>OK / Cancel; true for OK.</summary>
    public static bool Confirm(IWin32Window owner, string message)
    {
        var page = Page("", message);
        var ok = new TaskDialogButton(Messages.Text("dialog.ok"));
        page.Buttons.Add(ok);
        page.Buttons.Add(new TaskDialogButton(Messages.Text("dialog.cancel")));
        return Show(owner, page) == ok;
    }

    /// <summary>The text typed, or null for Cancel.</summary>
    public static string? Prompt(IWin32Window owner, string message, string? defaultText)
    {
        using var form = new PromptForm(Messages, message, defaultText);
        return form.ShowDialog(owner) == DialogResult.OK ? form.Value : null;
    }
}
