using FoodOptimizer.Core;

namespace FoodOptimizer.Tests;

/// M3: shell.log never holds the server's base path.
public class LogRedactionTests
{
    const string Base = "Ab_cd-0123456789EFGHIJKLMNOPQRST";

    [Fact]
    public void Base_is_replaced_everywhere()
    {
        var text = $"http://127.0.0.1:8123/{Base}/_stcore/health?next=/{Base}/x#{Base}";
        Assert.Equal("http://127.0.0.1:8123/…/_stcore/health?next=/…/x#…", LogRedaction.Redact(text, Base));
        Assert.Equal("…", LogRedaction.Mark);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    public void Null_base_changes_nothing(string? @base)
    {
        var text = $"http://127.0.0.1:8123/{Base}/";
        Assert.Equal(text, LogRedaction.Redact(text, @base));
    }

    [Fact]
    public void Short_strings_are_not_redacted()
    {
        // A base of under 8 characters would blank ordinary parts of an
        // address (the launcher's bases are 32).
        Assert.Equal("http://127.0.0.1:8123/abcdefg/x", LogRedaction.Redact("http://127.0.0.1:8123/abcdefg/x", "abcdefg"));
        Assert.Equal("http://127.0.0.1:8123/…/x", LogRedaction.Redact("http://127.0.0.1:8123/abcdefgh/x", "abcdefgh"));
        Assert.Equal("", LogRedaction.Redact("", Base));
    }
}
