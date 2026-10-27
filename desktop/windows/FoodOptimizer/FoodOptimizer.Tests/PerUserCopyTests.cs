using FoodOptimizer.Core;

namespace FoodOptimizer.Tests;

/// <summary>D1: which per-user copy the all-users app may hand to its own
/// uninstaller. Real folders under a temp root, so the paths are this OS's.</summary>
public sealed class PerUserCopyTests : IDisposable
{
    readonly string _dir = Directory.CreateTempSubdirectory("foodopt-percopy-").FullName;
    static readonly char Sep = Path.DirectorySeparatorChar;

    public void Dispose() => Directory.Delete(_dir, true);

    string Own => Path.Combine(_dir, "Program Files", "Food Optimizer");
    string Default => Path.Combine(_dir, "Local", "Programs", "Food Optimizer");
    string Custom => Path.Combine(_dir, "Tools", "FO");

    /// <summary>A folder holding the files that make it a copy of ours.</summary>
    string Copy(string dir, params string[] skip)
    {
        Directory.CreateDirectory(dir);
        foreach (var name in new[] { "FoodOptimizer.exe", "unins000.exe", "unins000.dat" }.Except(skip))
            File.WriteAllText(Path.Combine(dir, name), name);
        return dir;
    }

    static bool NoLinks(string _) => false;

    string? Choose(string? location, Func<string, bool>? isReparsePoint = null) =>
        PerUserCopy.Choose(Own, location, Default, File.Exists, isReparsePoint ?? NoLinks);

    [Fact]
    public void Own_root_is_never_chosen()
    {
        Copy(Own);
        Assert.Null(PerUserCopy.Choose(Own, Own, Own, File.Exists, NoLinks));
        Assert.Null(Choose(Own));
    }

    [Fact]
    public void The_uninstall_key_location_wins()
    {
        Copy(Custom);
        Copy(Default);
        Assert.Equal(Custom, Choose(Custom + Sep));
        Assert.Equal(Custom, Choose(Custom));
    }

    [Fact]
    public void The_default_folder_is_the_fallback()
    {
        Copy(Default);
        Assert.Equal(Default, Choose(null));
        Assert.Equal(Default, Choose(""));
        // The key names a folder that no longer holds a copy.
        Directory.CreateDirectory(Custom);
        Assert.Equal(Default, Choose(Custom));
        Assert.Null(PerUserCopy.Choose(Own, null, Path.Combine(_dir, "nothing"), File.Exists, NoLinks));
    }

    [Fact]
    public void A_folder_missing_unins000_dat_is_not_ours()
    {
        Copy(Custom, "unins000.dat");
        Assert.Null(Choose(Custom));
        Copy(Default, "unins000.exe");
        Assert.Null(Choose(Custom));
        File.WriteAllText(Path.Combine(Default, "unins000.exe"), "x");
        File.Delete(Path.Combine(Default, "FoodOptimizer.exe"));
        Assert.Null(Choose(null));
    }

    [Fact]
    public void A_junction_on_the_path_is_never_chosen()
    {
        Copy(Custom);
        Copy(Default);
        var link = Path.Combine(_dir, "Tools");
        // A link at the folder itself, or at any folder above it.
        Assert.Null(PerUserCopy.Choose(Own, Custom, Path.Combine(_dir, "nothing"), File.Exists,
            p => string.Equals(p, Custom, StringComparison.OrdinalIgnoreCase)));
        Assert.Null(PerUserCopy.Choose(Own, Custom, Path.Combine(_dir, "nothing"), File.Exists,
            p => string.Equals(p, link, StringComparison.OrdinalIgnoreCase)));
        // The default folder is checked the same way.
        Assert.Equal(Default, Choose(Custom, p => string.Equals(p, link, StringComparison.OrdinalIgnoreCase)));
        Assert.Null(Choose(null, p => p.StartsWith(Path.Combine(_dir, "Local"), StringComparison.OrdinalIgnoreCase)));
    }

    [Fact]
    public void Different_spelling_of_own_root_is_still_own_root()
    {
        Copy(Own);
        var parent = Path.GetDirectoryName(Own)!;
        foreach (var spelling in new[]
        {
            Own + Sep,
            Own.ToUpperInvariant(),
            Path.Combine(parent, "x", "..", "Food Optimizer"),
            Path.Combine(parent, ".", "Food Optimizer") + Sep,
        })
        {
            Assert.Null(PerUserCopy.Choose(Own, spelling, Path.Combine(_dir, "nothing"), File.Exists, NoLinks));
            Assert.Null(PerUserCopy.Choose(spelling, Own, Path.Combine(_dir, "nothing"), File.Exists, NoLinks));
        }
    }

    [Fact]
    public void A_name_ending_in_a_space_or_dot_is_never_chosen()
    {
        // Win32 drops a trailing space or dot from a path's last part but
        // keeps it in a middle one: the folder checked and the folder run
        // from could differ.
        var odd = Path.Combine(_dir, "Link ", "FO");
        var dot = Path.Combine(_dir, "Link.", "FO");
        Copy(odd);
        Copy(dot);
        Assert.Null(Choose(odd));
        Assert.Null(Choose(dot));
        Assert.Null(Choose(Path.Combine(_dir, "Tools", "FO ")));
    }

    [Fact]
    public void The_uninstaller_command_is_silent_and_ours()
    {
        Assert.Equal(Path.Combine(Custom, "unins000.exe"), PerUserCopy.Uninstaller(Custom));
        Assert.Equal(["/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"], PerUserCopy.UninstallArguments);
    }
}
