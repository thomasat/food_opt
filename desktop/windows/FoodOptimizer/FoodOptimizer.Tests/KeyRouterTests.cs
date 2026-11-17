using System.Text;
using FoodOptimizer.Core;

namespace FoodOptimizer.Tests;

public class KeyRouterTests
{
    static readonly KeyRouter Router = KeyRouter.FromMessages(Messages.LoadEmbedded());

    [Theory]
    [InlineData(0xBB, '=', true, false, false, MenuIds.Bigger)]
    [InlineData(0xBB, '=', true, true, false, MenuIds.Bigger)]      // Ctrl++ on US
    [InlineData(0xBB, '+', true, false, false, MenuIds.Bigger)]     // German
    [InlineData(0x6B, null, true, false, false, MenuIds.Bigger)]    // numpad +
    [InlineData(0xBD, '-', true, false, false, MenuIds.Smaller)]
    [InlineData(0x6D, null, true, false, false, MenuIds.Smaller)]   // numpad -
    [InlineData(0x36, '-', true, false, false, MenuIds.Smaller)]    // AZERTY
    [InlineData(0x30, 'à', true, false, false, MenuIds.Actual)]     // AZERTY
    [InlineData(0x60, null, true, false, false, MenuIds.Actual)]    // numpad 0
    [InlineData(0x4F, 'o', true, false, false, MenuIds.OpenProjects)]
    [InlineData(0xBB, '=', false, false, false, null)]
    [InlineData(0x43, 'c', true, false, false, null)]
    [InlineData(0x74, null, false, false, false, null)]             // F5
    public void Routes(int vk, char? ch, bool ctrl, bool shift, bool alt, string? id)
    {
        Assert.Equal(id, Router.Route(vk, ch, ctrl, shift, alt));
    }

    [Theory]
    [InlineData(0x51, 'q')]   // AltGr+Q = @ on German
    [InlineData(0x30, '0')]   // AltGr+0 = }
    [InlineData(0xBB, '=')]
    [InlineData(0x4F, 'o')]
    public void AltGr_chords_are_never_routed(int vk, char ch)
    {
        Assert.Null(Router.Route(vk, ch, ctrl: true, shift: false, alt: true));
    }

    [Fact]
    public void Every_windows_key_in_messages_parses()
    {
        KeyRouter.FromMessages(Messages.LoadEmbedded());
    }

    [Fact]
    public void An_unknown_key_token_is_refused()
    {
        var json = """
            {"messages": [], "menus": [
              {"id": "menu.view.bigger", "kind": "menu", "text": {"mac": null, "windows": "Bigger Text"},
               "keys": {"mac": [], "windows": ["Ctrl+Banana"]}}
            ]}
            """;
        var m = Messages.Load(new MemoryStream(Encoding.UTF8.GetBytes(json)));
        var e = Assert.Throws<ArgumentException>(() => KeyRouter.FromMessages(m));
        Assert.Contains("Ctrl+Banana", e.Message);
    }

    [Fact]
    public void Menu_ids_are_the_messages_ids()
    {
        var m = Messages.LoadEmbedded();
        foreach (var id in MenuIds.Commands) Assert.NotNull(m.Text(id));
        Assert.Equal(9, MenuIds.Commands.Count);
        Assert.Equal("menu.file.open_projects", MenuIds.OpenProjects);
        Assert.Equal("menu.app.quit", MenuIds.Quit);
    }
}
