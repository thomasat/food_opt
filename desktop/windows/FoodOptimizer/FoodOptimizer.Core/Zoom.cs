namespace FoodOptimizer.Core;

/// <summary>The page zoom: steps of 0.1 between 0.5 and 2.5.</summary>
public static class Zoom
{
    public const double Min = 0.5, Max = 2.5, Step = 0.1, Default = 1.0;

    public static double Clamp(double z) =>
        double.IsFinite(z)
            ? Math.Clamp(Math.Round(z, 2, MidpointRounding.AwayFromZero), Min, Max)
            : Default;

    public static double Bigger(double z) => Clamp(z + Step);

    public static double Smaller(double z) => Clamp(z - Step);

    public static bool Same(double a, double b) => Math.Abs(a - b) < 0.001;
}
