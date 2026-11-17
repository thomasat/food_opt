using FoodOptimizer.Core;

namespace FoodOptimizer.Tests;

public class ZoomTests
{
    [Theory]
    [InlineData(0.3, 0.5)]
    [InlineData(3.0, 2.5)]
    [InlineData(double.NaN, 1.0)]
    [InlineData(double.PositiveInfinity, 1.0)]
    [InlineData(double.NegativeInfinity, 1.0)]
    [InlineData(1.234, 1.23)]
    public void Clamp(double z, double expected) => Assert.Equal(expected, Zoom.Clamp(z), 9);

    [Theory]
    [InlineData(1.0, 1.1)]
    [InlineData(2.45, 2.5)]
    [InlineData(2.5, 2.5)]
    [InlineData(1.2000000001, 1.3)]
    public void Bigger(double z, double expected) => Assert.Equal(expected, Zoom.Bigger(z), 9);

    [Theory]
    [InlineData(0.55, 0.5)]
    [InlineData(1.25, 1.15)]
    [InlineData(0.5, 0.5)]
    public void Smaller(double z, double expected) => Assert.Equal(expected, Zoom.Smaller(z), 9);

    [Fact]
    public void Same_allows_float_noise()
    {
        Assert.True(Zoom.Same(1.1, 1.0 + 0.1));
        Assert.True(Zoom.Same(1.1, 1.1009));
        Assert.False(Zoom.Same(1.1, 1.102));
    }

    [Theory]
    [InlineData(3.0, 2.3, 2.5, 2.5)]
    [InlineData(1.1, 1.0, null, 1.1)]
    [InlineData(2.5, 2.5, null, null)]
    [InlineData(0.333333, 0.5, 0.5, null)]
    [InlineData(1.1000000001, 1.1, null, null)]
    public void OnReported_reapplies_and_saves(double reported, double saved, double? reapply, double? save) =>
        Assert.Equal((reapply, save), Zoom.OnReported(reported, saved));
}
