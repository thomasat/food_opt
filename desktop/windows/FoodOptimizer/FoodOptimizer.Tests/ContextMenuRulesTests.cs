using FoodOptimizer.Core;

namespace FoodOptimizer.Tests;

public class ContextMenuRulesTests
{
    [Theory]
    [InlineData("cut")]
    [InlineData("copy")]
    [InlineData("paste")]
    [InlineData("selectAll")]
    public void Editing_items_are_kept(string name) => Assert.True(ContextMenuRules.Keep(name));

    [Theory]
    [InlineData("back")]
    [InlineData("reload")]
    [InlineData("saveAs")]
    [InlineData("print")]
    [InlineData("share")]
    [InlineData("webCapture")]
    [InlineData("inspectElement")]
    [InlineData("emoji")]
    [InlineData("copyLinkLocation")]
    [InlineData("other")]
    [InlineData("")]
    public void Everything_else_goes(string name) => Assert.False(ContextMenuRules.Keep(name));
}
