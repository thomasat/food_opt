using FoodOptimizer.Core;

namespace FoodOptimizer.Shell;

/// <summary>The page's prompt(): its message, a text box and OK / Cancel.</summary>
sealed class PromptForm : Form
{
    readonly TextBox _box = new() { Dock = DockStyle.Top };

    public PromptForm(Messages m, string message, string? defaultText)
    {
        Text = m.Text("dialog.title");
        FormBorderStyle = FormBorderStyle.FixedDialog;
        MinimizeBox = false;
        MaximizeBox = false;
        ShowInTaskbar = false;
        StartPosition = FormStartPosition.CenterParent;
        AutoScaleMode = AutoScaleMode.Dpi;
        AutoSize = true;
        AutoSizeMode = AutoSizeMode.GrowAndShrink;
        Padding = new Padding(12);

        var label = new Label { Text = message, AutoSize = true, MaximumSize = new Size(420, 0), Dock = DockStyle.Top, Padding = new Padding(0, 0, 0, 8) };
        _box.Text = defaultText ?? "";
        _box.Width = 420;
        var ok = new Button { Text = m.Text("dialog.ok"), DialogResult = DialogResult.OK, AutoSize = true };
        var cancel = new Button { Text = m.Text("dialog.cancel"), DialogResult = DialogResult.Cancel, AutoSize = true };
        var buttons = new FlowLayoutPanel
        {
            FlowDirection = FlowDirection.RightToLeft,
            Dock = DockStyle.Top,
            AutoSize = true,
            Padding = new Padding(0, 10, 0, 0),
        };
        buttons.Controls.Add(cancel);
        buttons.Controls.Add(ok);
        var layout = new TableLayoutPanel { AutoSize = true, ColumnCount = 1, Dock = DockStyle.Fill };
        layout.Controls.Add(label);
        layout.Controls.Add(_box);
        layout.Controls.Add(buttons);
        Controls.Add(layout);
        AcceptButton = ok;
        CancelButton = cancel;
    }

    public string Value => _box.Text;
}
