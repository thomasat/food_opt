namespace FoodOptimizer.Core;

/// <summary>The right-click menu keeps only the editing commands.</summary>
public static class ContextMenuRules
{
    static readonly HashSet<string> Kept = new(StringComparer.Ordinal) { "cut", "copy", "paste", "selectAll" };

    public static bool Keep(string itemName) => Kept.Contains(itemName);
}
