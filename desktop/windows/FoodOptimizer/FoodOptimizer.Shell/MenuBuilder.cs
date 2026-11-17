using FoodOptimizer.Core;

namespace FoodOptimizer.Shell;

/// <summary>The menu bar, built from menu_layout in messages.json. Items
/// show their first shortcut but own none: KeyRouter runs every shortcut,
/// so nothing fires twice.</summary>
static class MenuBuilder
{
    public static MenuStrip Build(Messages m, Action<string> execute)
    {
        var strip = new MenuStrip();
        foreach (var spec in m.MenuLayout)
        {
            var top = new ToolStripMenuItem(m.Text(spec.Id)) { Name = spec.Id };
            foreach (var id in spec.Items)
            {
                if (id == MenuSpec.Separator)
                {
                    top.DropDownItems.Add(new ToolStripSeparator());
                    continue;
                }
                var item = new ToolStripMenuItem(m.Text(id)) { Name = id };
                var keys = m.Keys(id);
                if (keys.Count > 0)
                {
                    item.ShortcutKeyDisplayString = keys[0];
                    item.ShowShortcutKeys = true;
                }
                item.Click += (_, _) => execute(id);
                top.DropDownItems.Add(item);
            }
            strip.Items.Add(top);
        }
        return strip;
    }
}
