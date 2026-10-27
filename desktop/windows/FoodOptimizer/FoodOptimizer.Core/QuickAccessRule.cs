namespace FoodOptimizer.Core;

/// <summary>D2: the app pins the projects folder to Quick access once per
/// user (at the first launch); a user who unpins it later keeps it unpinned.</summary>
public static class QuickAccessRule
{
    public static bool ShouldPin(ShellSettings s) => !s.QuickAccessPinned;
}
