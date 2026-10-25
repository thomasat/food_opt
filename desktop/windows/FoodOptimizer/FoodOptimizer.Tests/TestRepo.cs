namespace FoodOptimizer.Tests;

/// <summary>The repository the tests were built from: the first folder
/// above the test binaries that holds desktop/VERSION.</summary>
static class TestRepo
{
    public static string Root { get; } = Find();

    public static string Path(params string[] parts) =>
        System.IO.Path.Combine([Root, .. parts]);

    static string Find()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null)
        {
            if (File.Exists(System.IO.Path.Combine(dir.FullName, "desktop", "VERSION")))
                return dir.FullName;
            dir = dir.Parent;
        }
        throw new InvalidOperationException("no desktop/VERSION above " + AppContext.BaseDirectory);
    }
}
