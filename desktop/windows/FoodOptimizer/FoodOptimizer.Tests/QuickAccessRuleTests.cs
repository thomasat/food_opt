using FoodOptimizer.Core;

namespace FoodOptimizer.Tests;

/// <summary>D2: the app pins the projects folder once per user; a user's
/// own unpin afterwards is kept.</summary>
public sealed class QuickAccessRuleTests : IDisposable
{
    readonly string _dir = Directory.CreateTempSubdirectory("foodopt-quickaccess-").FullName;
    string Path => System.IO.Path.Combine(_dir, StateFiles.Settings);

    public void Dispose() => Directory.Delete(_dir, true);

    [Fact]
    public void First_launch_pins()
    {
        Assert.True(QuickAccessRule.ShouldPin(new ShellSettings()));
        Assert.True(QuickAccessRule.ShouldPin(ShellSettings.Load(Path)));
    }

    [Fact]
    public void A_user_unpin_is_kept()
    {
        Assert.False(QuickAccessRule.ShouldPin(new ShellSettings { QuickAccessPinned = true }));
        Assert.True(new ShellSettings { QuickAccessPinned = true }.Save(Path));
        Assert.False(QuickAccessRule.ShouldPin(ShellSettings.Load(Path)));
    }
}
