using System.ComponentModel;
using System.Runtime.InteropServices;
using FoodOptimizer.Core;

namespace FoodOptimizer.Tests;

public class CommandFailureTests
{
    public static TheoryData<Exception, bool> Cases => new()
    {
        { new IOException("io text"), true },
        { new DirectoryNotFoundException("dir text"), true },
        { new PathTooLongException("long text"), true },
        { new FileNotFoundException("file text"), true },
        { new UnauthorizedAccessException("access text"), true },
        { new Win32Exception(5, "win32 text"), true },
        { new COMException("com text (0x8007139F)", unchecked((int)0x8007139F)), false },
        { new InvalidOperationException("invalid text"), false },
        { new NullReferenceException("null text"), false },
        { new ArgumentException("argument text"), false },
        { new FormatException("format text"), false },
    };

    /// Only a failure that came from the OS (files, folders, ShellExecute)
    /// has a reason worth showing; a code error is never shown as Windows' text.
    [Theory]
    [MemberData(nameof(Cases))]
    public void Only_os_failures_have_a_reason(Exception ex, bool shown) =>
        Assert.Equal(shown ? ex.Message : null, CommandFailure.Reason(ex));
}
