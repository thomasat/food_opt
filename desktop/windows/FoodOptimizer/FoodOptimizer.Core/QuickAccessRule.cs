namespace FoodOptimizer.Core;

/// <summary>What one pin attempt found (QuickAccess.PinAsync).</summary>
public enum PinOutcome
{
    /// <summary>This launch pinned it, and Quick access lists it now.</summary>
    Pinned,
    /// <summary>Listed before the app did anything: the user's own pin, or
    /// Frequent folders. Not ours.</summary>
    AlreadyListed,
    /// <summary>The verb ran, but Quick access does not list the folder.</summary>
    NotListed,
    Failed,
}

/// <summary>D2: the app pins the projects folder to Quick access once per
/// user (at the first launch); a user who unpins it later keeps it
/// unpinned. "Ours" (QuickAccessPinned, which the uninstaller reads) only
/// when this launch pinned it.</summary>
public static class QuickAccessRule
{
    public static bool ShouldPin(ShellSettings s) => !s.QuickAccessPinned && !s.QuickAccessChecked;

    /// <summary>Records what a launch found; true when the settings changed
    /// (save them). A pin that did not take is tried at the next launch.</summary>
    public static bool Record(ShellSettings s, PinOutcome outcome)
    {
        switch (outcome)
        {
            case PinOutcome.Pinned:
                s.QuickAccessPinned = true;
                s.QuickAccessChecked = true;
                return true;
            case PinOutcome.AlreadyListed:
                s.QuickAccessChecked = true;
                return true;
            default:
                return false;
        }
    }
}
