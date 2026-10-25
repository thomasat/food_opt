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
        Assert.Equal("Round 6 sheet.xlsx", DownloadRules.SuggestedFileName(@"C:\x\Round 6 sheet.xlsx", contentDisposition: null));
        Assert.Equal("probe.csv", DownloadRules.SuggestedFileName("probe.csv", contentDisposition: null));
        Assert.Equal("download", DownloadRules.SuggestedFileName("", contentDisposition: null));
        Assert.Equal("download", DownloadRules.SuggestedFileName(@"C:\x\", contentDisposition: null));
    }

    [Fact]
    public void Suggested_name_drops_the_number_webview_added()
    {
        // WebView2 numbers the proposal when the name is taken in Downloads;
        // the dialog offers the server's name, and the user may replace it.
        var taken = new HashSet<string> { Path.Combine("D", "probe.csv"), Path.Combine("D", "Round 6 sheet.xlsx") };
        Assert.Equal("probe.csv", DownloadRules.SuggestedFileName(Path.Combine("D", "probe (4).csv"), taken.Contains));
        Assert.Equal("Round 6 sheet.xlsx", DownloadRules.SuggestedFileName(Path.Combine("D", "Round 6 sheet (1).xlsx"), taken.Contains));
        Assert.Equal("notes (2).csv", DownloadRules.SuggestedFileName(Path.Combine("D", "notes (2).csv"), taken.Contains));
        Assert.Equal("probe (x).csv", DownloadRules.SuggestedFileName(Path.Combine("D", "probe (x).csv"), taken.Contains));
        Assert.Equal("probe.csv", DownloadRules.SuggestedFileName(Path.Combine("D", "probe (4).csv"), null, taken.Contains));
        Assert.Equal("Round 6 sheet.xlsx", DownloadRules.SuggestedFileName(Path.Combine("D", "Round 6 sheet (1).xlsx"), null, taken.Contains));
        Assert.Equal("notes (2).csv", DownloadRules.SuggestedFileName(Path.Combine("D", "notes (2).csv"), null, taken.Contains));
        Assert.Equal("probe (x).csv", DownloadRules.SuggestedFileName(Path.Combine("D", "probe (x).csv"), null, taken.Contains));
    }

    [Theory]
    [InlineData("attachment; filename=\"Round (2).csv\"", "Round (2).csv")]
    [InlineData("attachment; filename*=UTF-8''Round%20%C3%A9.csv", "Round é.csv")]
    [InlineData("attachment; filename=\"a.csv\"; filename*=UTF-8''b.csv", "b.csv")]
    [InlineData("attachment; filename=plain.csv", "plain.csv")]
    [InlineData("attachment; filename=\"..\\\\evil.csv\"", null)]
    [InlineData("attachment", null)]
    [InlineData(null, null)]
    [InlineData("attachment; FILENAME*=utf-8'en'a%20b.csv", "a b.csv")]
    [InlineData("attachment; filename=\"say \\\"hi\\\".csv\"", "say \"hi\".csv")]
    [InlineData("attachment; filename=\"a;b.csv\"", "a;b.csv")]
    [InlineData("attachment; filename=\"\"", null)]
    [InlineData("attachment; filename*=UTF-8''%ZZ.csv; filename=ok.csv", "ok.csv")]
    [InlineData("attachment; filename*=UTF-8''%ZZ.csv", null)]
    [InlineData("attachment; filename*=UTF-8''..%2Fx.csv", null)]
    [InlineData("attachment; filename*=UTF-8''a%3Ab.csv", null)]
    public void DispositionFileName_reads_every_form(string? cd, string? want) =>
        Assert.Equal(want, DownloadRules.DispositionFileName(cd));

    [Fact]
    public void A_file_really_named_with_a_number_keeps_it() =>
        Assert.Equal("Round (2).csv", DownloadRules.SuggestedFileName(@"C:\D\Round (2).csv", "attachment; filename=\"Round (2).csv\"", _ => true));

    [Fact]
    public void A_name_numbered_by_webview_goes_back_to_the_servers() =>
        Assert.Equal("Round.csv", DownloadRules.SuggestedFileName(@"C:\D\Round (3).csv", "attachment; filename=\"Round.csv\"", _ => true));

    [Fact]
    public void Without_a_disposition_name_the_current_rule_applies() =>
        Assert.Equal("probe.csv", DownloadRules.SuggestedFileName(Path.Combine("D", "probe (4).csv"), "attachment", p => p == Path.Combine("D", "probe.csv")));

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
