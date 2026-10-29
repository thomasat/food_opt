using FoodOptimizer.Core;

namespace FoodOptimizer.Tests;

public class WindowPlacementTests
{
    static MonitorInfo Mon(string name, int x, int y, int w, int h, int dpi = 96, bool primary = true) =>
        new(name, new PxRect(x, y, w, h), dpi, primary);

    static readonly MonitorInfo FullHd = Mon(@"\\.\DISPLAY1", 0, 0, 1920, 1040);

    [Fact]
    public void Default_at_100_percent_is_1280_by_860_centred()
    {
        Assert.Equal(new Placement(new PxRect(320, 90, 1280, 860), false), WindowPlacement.Initial([FullHd], null));
    }

    [Fact]
    public void Default_at_150_percent_on_a_1080p_laptop_is_maximized()
    {
        var m = Mon(@"\\.\DISPLAY1", 0, 0, 1920, 1032, 144);
        Assert.Equal(new Placement(new PxRect(0, 0, 1920, 1032), true), WindowPlacement.Initial([m], null));
    }

    [Fact]
    public void Default_at_150_percent_on_a_1440p_screen_fits()
    {
        var m = Mon(@"\\.\DISPLAY1", 0, 0, 2560, 1392, 144);
        Assert.Equal(new Placement(new PxRect(320, 51, 1920, 1290), false), WindowPlacement.Initial([m], null));
    }

    [Fact]
    public void Default_uses_the_primary_monitor()
    {
        var left = Mon(@"\\.\DISPLAY2", 0, 0, 1920, 1040, primary: false);
        var primary = Mon(@"\\.\DISPLAY1", 1920, 0, 1920, 1040);
        Assert.Equal(new PxRect(2240, 90, 1280, 860), WindowPlacement.Initial([left, primary], null).Bounds);
        Assert.Equal(new PxRect(320, 90, 1280, 860),
            WindowPlacement.Initial([left with { IsPrimary = false }], null).Bounds);
    }

    [Fact]
    public void Saved_position_on_a_present_monitor_is_restored()
    {
        var second = Mon(@"\\.\DISPLAY2", 1920, 0, 1920, 1040, primary: false);
        var saved = new SavedWindow(@"\\.\display2", 2000, 100, 1100, 750, false);
        Assert.Equal(new Placement(new PxRect(2000, 100, 1100, 750), false),
            WindowPlacement.Initial([FullHd, second], saved));
    }

    [Fact]
    public void Saved_on_a_missing_monitor_uses_the_default()
    {
        var saved = new SavedWindow(@"\\.\DISPLAY9", 4000, 100, 1100, 750, false);
        Assert.Equal(WindowPlacement.Initial([FullHd], null), WindowPlacement.Initial([FullHd], saved));
    }

    [Fact]
    public void Saved_partly_off_screen_is_pulled_inside()
    {
        var saved = new SavedWindow(@"\\.\DISPLAY1", -500, 900, 1280, 860, false);
        Assert.Equal(new PxRect(0, 180, 1280, 860), WindowPlacement.Initial([FullHd], saved).Bounds);
        var right = new SavedWindow(@"\\.\DISPLAY1", 1800, -40, 1280, 860, false);
        Assert.Equal(new PxRect(640, 0, 1280, 860), WindowPlacement.Initial([FullHd], right).Bounds);
    }

    [Fact]
    public void Saved_larger_than_the_work_area_is_shrunk_to_it()
    {
        var saved = new SavedWindow(@"\\.\DISPLAY1", 10, 10, 3000, 2000, false);
        Assert.Equal(new PxRect(0, 0, 1920, 1040), WindowPlacement.Initial([FullHd], saved).Bounds);
    }

    [Fact]
    public void Saved_smaller_than_the_minimum_grows()
    {
        var saved = new SavedWindow(@"\\.\DISPLAY1", 100, 100, 300, 200, false);
        Assert.Equal(new PxRect(100, 100, 700, 500), WindowPlacement.Initial([FullHd], saved).Bounds);
        var hi = Mon(@"\\.\DISPLAY1", 0, 0, 2560, 1392, 144);
        Assert.Equal(new PxRect(100, 100, 1050, 750), WindowPlacement.Initial([hi], saved).Bounds);
    }

    [Fact]
    public void Saved_maximized_stays_maximized()
    {
        var saved = new SavedWindow(@"\\.\DISPLAY1", 100, 100, 1100, 750, true);
        Assert.Equal(new Placement(new PxRect(100, 100, 1100, 750), true), WindowPlacement.Initial([FullHd], saved));
    }

    [Fact]
    public void Scale_and_minimum_follow_the_dpi()
    {
        Assert.Equal(1280, WindowPlacement.Scale(1280, 96));
        Assert.Equal(1600, WindowPlacement.Scale(1280, 120));
        Assert.Equal((700, 500), WindowPlacement.MinimumSize(96));
        Assert.Equal((1050, 750), WindowPlacement.MinimumSize(144));
    }

    [Fact]
    public void To_saved_keeps_the_restore_bounds()
    {
        Assert.Equal(new SavedWindow("M", 1, 2, 3, 4, true), WindowPlacement.ToSaved("M", new PxRect(1, 2, 3, 4), true));
    }
}
