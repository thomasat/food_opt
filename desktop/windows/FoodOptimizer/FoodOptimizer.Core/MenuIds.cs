namespace FoodOptimizer.Core;

/// <summary>The menu and command ids: messages.json ids, so the label, the
/// shortcut keys and the command share one name.</summary>
public static class MenuIds
{
    public const string File = "menu.file", View = "menu.view", Help = "menu.help";

    public const string OpenProjects = "menu.file.open_projects", Quit = "menu.app.quit",
        Bigger = "menu.view.bigger", Smaller = "menu.view.smaller", Actual = "menu.view.actual",
        StartHere = "menu.help.start_here", GetHelp = "menu.help.get_help",
        ShowLog = "menu.help.show_log", About = "menu.app.about";

    /// <summary>Every command the menus and shortcuts can run.</summary>
    public static readonly IReadOnlyList<string> Commands =
        [OpenProjects, Quit, Bigger, Smaller, Actual, StartHere, GetHelp, ShowLog, About];
}
