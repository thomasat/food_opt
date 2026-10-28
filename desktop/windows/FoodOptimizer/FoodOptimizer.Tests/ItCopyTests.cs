using FoodOptimizer.Core;

namespace FoodOptimizer.Tests;

/// <summary>D1 hand-off: a per-user copy opened on a PC with an IT copy
/// starts the IT copy and quits, so the IT copy can run this copy's own
/// uninstaller. Real folders under a temp root, so the paths are this OS's.</summary>
public sealed class ItCopyTests : IDisposable
{
    readonly string _dir = Directory.CreateTempSubdirectory("foodopt-itcopy-").FullName;
    static readonly char Sep = Path.DirectorySeparatorChar;

    public void Dispose() => Directory.Delete(_dir, true);

    string Own => Path.Combine(_dir, "Local", "Programs", "Food Optimizer");
    string Default => Path.Combine(_dir, "Program Files", "Food Optimizer");
    string Custom => Path.Combine(_dir, "Apps", "FO");

    /// <summary>A folder holding the app and, unless left out, the marker
    /// the all-users installer adds.</summary>
    string Copy(string dir, bool marker = true)
    {
        Directory.CreateDirectory(dir);
        File.WriteAllText(Path.Combine(dir, "FoodOptimizer.exe"), "exe");
        if (marker) File.WriteAllText(Path.Combine(dir, "all-users"), "all-users");
        return dir;
    }

    static bool NoLinks(string _) => false;

    string? Target(string? location, bool allUsers = false, Func<string, bool>? isReparsePoint = null) =>
        ItCopy.HandOffTarget(allUsers, Own, location, Default, File.Exists, isReparsePoint ?? NoLinks);

    [Fact]
    public void The_it_copy_never_hands_off()
    {
        Copy(Own);
        Copy(Default);
        Assert.Null(Target(Default, allUsers: true));
        Assert.Null(Target(null, allUsers: true));
    }

    [Fact]
    public void The_uninstall_key_location_wins()
    {
        Copy(Custom);
        Copy(Default);
        Assert.Equal(Custom, Target(Custom + Sep));
        Assert.Equal(Custom, Target(Custom));
    }

    [Fact]
    public void Program_files_is_the_fallback()
    {
        Copy(Default);
        Assert.Equal(Default, Target(null));
        Assert.Equal(Default, Target(""));
        // The key names a folder that no longer holds the IT copy.
        Directory.CreateDirectory(Custom);
        Assert.Equal(Default, Target(Custom));
        Assert.Null(ItCopy.HandOffTarget(false, Own, null, Path.Combine(_dir, "nothing"), File.Exists, NoLinks));
        Assert.Null(ItCopy.HandOffTarget(false, Own, null, null, File.Exists, NoLinks));
    }

    [Fact]
    public void A_folder_without_the_marker_or_the_app_is_not_the_it_copy()
    {
        Copy(Default, marker: false);
        Assert.Null(Target(null));
        Copy(Custom);
        File.Delete(Path.Combine(Custom, "FoodOptimizer.exe"));
        Assert.Null(Target(Custom));
        // A folder named all-users is not the marker.
        File.WriteAllText(Path.Combine(Custom, "FoodOptimizer.exe"), "exe");
        File.Delete(Path.Combine(Custom, "all-users"));
        Directory.CreateDirectory(Path.Combine(Custom, "all-users"));
        Assert.Null(Target(Custom));
    }

    [Fact]
    public void Own_root_is_never_the_target()
    {
        // However spelled: never a loop back into this copy.
        Copy(Own);
        var parent = Path.GetDirectoryName(Own)!;
        foreach (var spelling in new[]
        {
            Own, Own + Sep, Own.ToUpperInvariant(),
            Path.Combine(parent, "x", "..", "Food Optimizer"),
            Path.Combine(parent, ".", "Food Optimizer") + Sep,
        })
        {
            Assert.Null(ItCopy.HandOffTarget(false, Own, spelling, Path.Combine(_dir, "nothing"), File.Exists, NoLinks));
            Assert.Null(ItCopy.HandOffTarget(false, Own, null, spelling, File.Exists, NoLinks));
            Assert.Null(ItCopy.HandOffTarget(false, spelling, Own, Path.Combine(_dir, "nothing"), File.Exists, NoLinks));
        }
    }

    [Fact]
    public void A_link_or_odd_name_on_the_path_is_never_the_target()
    {
        Copy(Custom);
        Copy(Default);
        var apps = Path.Combine(_dir, "Apps");
        Assert.Equal(Default, Target(Custom, isReparsePoint: p => string.Equals(p, apps, StringComparison.OrdinalIgnoreCase)));
        Assert.Null(Target(null, isReparsePoint: p => string.Equals(p, Default, StringComparison.OrdinalIgnoreCase)));
        var odd = Copy(Path.Combine(_dir, "Link ", "FO"));
        Assert.Null(ItCopy.HandOffTarget(false, Own, odd, Path.Combine(_dir, "nothing"), File.Exists, NoLinks));
    }

    [Fact]
    public void The_target_exe_is_the_shell_in_that_folder()
    {
        Assert.Equal(Path.Combine(Default, "FoodOptimizer.exe"), ItCopy.Exe(Default));
    }
}
