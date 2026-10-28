using FoodOptimizer.Core;

namespace FoodOptimizer.Tests;

/// <summary>D2: the app pins the projects folder once per user; a user's
/// own unpin afterwards is kept. It records "ours" only when this launch
/// pinned it: a folder already listed (the user's own pin, or Frequent
/// folders) is not ours, and is not tried again.</summary>
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

    [Fact]
    public void A_pin_this_launch_made_is_ours()
    {
        var s = new ShellSettings();
        Assert.True(QuickAccessRule.Record(s, PinOutcome.Pinned));
        Assert.True(s.QuickAccessPinned);
        Assert.True(s.QuickAccessChecked);
        Assert.False(QuickAccessRule.ShouldPin(s));
    }

    [Fact]
    public void A_folder_already_listed_is_not_ours_and_not_tried_again()
    {
        var s = new ShellSettings();
        Assert.True(QuickAccessRule.Record(s, PinOutcome.AlreadyListed));
        Assert.False(s.QuickAccessPinned);
        Assert.True(s.QuickAccessChecked);
        Assert.False(QuickAccessRule.ShouldPin(s));
        Assert.True(s.Save(Path));
        var back = ShellSettings.Load(Path);
        Assert.False(QuickAccessRule.ShouldPin(back));
        Assert.False(back.QuickAccessPinned);
    }

    [Theory]
    [InlineData(PinOutcome.NotListed)]
    [InlineData(PinOutcome.Failed)]
    public void A_pin_that_did_not_take_is_tried_again(PinOutcome outcome)
    {
        var s = new ShellSettings();
        Assert.False(QuickAccessRule.Record(s, outcome));
        Assert.False(s.QuickAccessPinned);
        Assert.False(s.QuickAccessChecked);
        Assert.True(QuickAccessRule.ShouldPin(s));
    }
}
