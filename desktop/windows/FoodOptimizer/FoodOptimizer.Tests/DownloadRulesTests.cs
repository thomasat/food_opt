using FoodOptimizer.Core;

namespace FoodOptimizer.Tests;

public class DownloadRulesTests
{
    [Fact]
    public void Initial_directory_is_downloads_when_it_exists()
    {
        Assert.Equal(@"D:\Dl", DownloadRules.InitialDirectory(@"D:\Dl", @"C:\U", _ => true));
        Assert.Equal(@"C:\U", DownloadRules.InitialDirectory(@"D:\Dl", @"C:\U", _ => false));
        Assert.Equal(@"C:\U", DownloadRules.InitialDirectory(null, @"C:\U", _ => true));
        Assert.Equal(@"C:\U", DownloadRules.InitialDirectory("", @"C:\U", _ => true));
    }

    [Fact]
    public void Suggested_name_is_the_file_name()
    {
        Assert.Equal("Round 6 sheet.xlsx", DownloadRules.SuggestedFileName(@"C:\x\Round 6 sheet.xlsx"));
        Assert.Equal("probe.csv", DownloadRules.SuggestedFileName("probe.csv"));
        Assert.Equal("download", DownloadRules.SuggestedFileName(""));
        Assert.Equal("download", DownloadRules.SuggestedFileName(@"C:\x\"));
    }

    [Fact]
    public void Zone_identifier_is_the_alternate_stream()
    {
        Assert.Equal(@"C:\x\a.csv:Zone.Identifier", DownloadRules.ZoneIdentifierPath(@"C:\x\a.csv"));
    }

    [Theory]
    [InlineData("http://127.0.0.1:18501/abc/media/x.csv", 18501, true)]
    [InlineData("blob:http://127.0.0.1:18501/1234", 18501, true)]
    [InlineData("http://127.0.0.1:18502/x.csv", 18501, false)]
    [InlineData("http://127.0.0.1:18501/x.csv", null, false)]
    [InlineData("https://example.com/x.csv", 18501, false)]
    [InlineData("data:text/csv,a", 18501, false)]
    public void From_our_server(string uri, int? port, bool expected)
    {
        Assert.Equal(expected, DownloadRules.IsFromOurServer(uri, port));
    }
}
