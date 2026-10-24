using System.Text;
using FoodOptimizer.Core;

namespace FoodOptimizer.Tests;

public class MessagesTests
{
    static readonly Messages M = Messages.LoadEmbedded();

    static string Lf(byte[] bytes) => Encoding.UTF8.GetString(bytes).Replace("\r\n", "\n");

    [Fact]
    public void Embedded_messages_equal_the_shared_file()
    {
        var shared = File.ReadAllBytes(TestRepo.Path("desktop", "shared", "messages.json"));
        Assert.Equal(Lf(shared), Lf(Messages.EmbeddedBytes()));
    }

    [Fact]
    public void Text_returns_the_windows_text()
    {
        Assert.Equal("This PC is not supported", M.Text("error.unsupported.title"));
        Assert.Equal("Exit", M.Text("menu.app.quit"));
        Assert.Equal("OK", M.Text("dialog.ok"));
    }

    [Fact]
    public void Text_of_a_mac_only_entry_throws_naming_it()
    {
        var e = Assert.Throws<KeyNotFoundException>(() => M.Text("steps.download"));
        Assert.Contains("steps.download", e.Message);
        var absent = Assert.Throws<KeyNotFoundException>(() => M.Text("no.such.id"));
        Assert.Contains("no.such.id", absent.Message);
    }

    [Fact]
    public void Format_fills_placeholders()
    {
        Assert.Equal("Food Optimizer 0.9.0", M.Format("about.body", ("version", "0.9.0")));
        Assert.Throws<FormatException>(() => M.Format("about.body"));
    }

    [Fact]
    public void All_has_only_windows_texts()
    {
        Assert.False(M.All.ContainsKey("steps.note.first_run"));
        Assert.Equal("Try again", M.All["button.try_again"]);
        Assert.Equal("Exit", M.All["menu.app.quit"]);
    }

    [Fact]
    public void Menu_layout_is_file_view_help()
    {
        var layout = M.MenuLayout;
        Assert.Equal(["menu.file", "menu.view", "menu.help"], layout.Select(m => m.Id));
        Assert.Equal(["menu.file.open_projects", MenuSpec.Separator, "menu.app.quit"], layout[0].Items);
        Assert.Equal(["menu.view.bigger", "menu.view.smaller", "menu.view.actual"], layout[1].Items);
        Assert.Equal(["menu.help.start_here", "menu.help.get_help", "menu.help.show_log", MenuSpec.Separator, "menu.app.about"], layout[2].Items);
    }

    [Fact]
    public void Keys_for_bigger_text()
    {
        Assert.Equal(["Ctrl+=", "Ctrl++", "Ctrl+Add"], M.Keys("menu.view.bigger"));
        Assert.Empty(M.Keys("menu.app.quit"));
    }

    [Fact]
    public void Load_reads_another_os()
    {
        using var s = new MemoryStream(Messages.EmbeddedBytes());
        var mac = Messages.Load(s, "mac");
        Assert.Equal("Quit Food Optimizer", mac.Text("menu.app.quit"));
    }
}
