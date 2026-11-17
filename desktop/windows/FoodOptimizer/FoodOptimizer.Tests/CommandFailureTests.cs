using System.ComponentModel;
using System.Runtime.InteropServices;
using FoodOptimizer.Core;

namespace FoodOptimizer.Tests;

public class CommandFailureTests
{
    public static TheoryData<Exception, string> Cases => new()
    {
        { new IOException("io text"), "IOException: io text" },
        { new DirectoryNotFoundException("dir text"), "DirectoryNotFoundException: dir text" },
        { new UnauthorizedAccessException("access text"), "UnauthorizedAccessException: access text" },
        { new Win32Exception(5, "win32 text"), "Win32Exception: win32 text" },
        { new COMException("com text (0x8007139F)", unchecked((int)0x8007139F)), "COMException: com text (0x8007139F)" },
        { new InvalidOperationException("invalid text"), "InvalidOperationException: invalid text" },
    };

    /// A failed menu command's reason goes to shell.log, never on screen:
    /// the exception's type and its message, on one line.
    [Theory]
    [MemberData(nameof(Cases))]
    public void The_logged_reason_is_the_type_and_message(Exception ex, string logged) =>
        Assert.Equal(logged, CommandFailure.Logged(ex));

    /// The dialog's body is the fixed support line: no placeholder for a reason.
    [Fact]
    public void The_dialog_body_has_no_reason()
    {
        Assert.Equal("Try again, or email sohum.patnaik@fsi.org.",
            Messages.LoadEmbedded().Text("error.command_failed.body"));
    }
}
