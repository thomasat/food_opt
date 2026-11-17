using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using FoodOptimizer.Core.Windows;

namespace FoodOptimizer.Tests;

public class JobObjectTests
{
    [Fact]
    public void Struct_sizes_match_win64()
    {
        Assert.Equal(144, Marshal.SizeOf<NativeMethods.JOBOBJECT_EXTENDED_LIMIT_INFORMATION>());
        Assert.Equal(64, Marshal.SizeOf<NativeMethods.JOBOBJECT_BASIC_LIMIT_INFORMATION>());
        Assert.Equal(48, Marshal.SizeOf<NativeMethods.IO_COUNTERS>());
    }

    internal static HashSet<int> Pids(string name) =>
        Process.GetProcessesByName(name).Select(p => p.Id).ToHashSet();

    internal static bool WaitUntil(Func<bool> condition, TimeSpan timeout)
    {
        var end = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < end)
        {
            if (condition()) return true;
            Thread.Sleep(50);
        }
        return condition();
    }

    internal static bool Alive(int pid)
    {
        try { return !Process.GetProcessById(pid).HasExited; }
        catch (ArgumentException) { return false; }
    }

    [Fact]
    [SupportedOSPlatform("windows")]
    public void Job_kills_the_whole_tree_when_closed()
    {
        Assert.SkipUnless(OperatingSystem.IsWindows(), "Windows only");
        var before = Pids("PING");
        var job = new JobObject();
        var cmd = Process.Start(new ProcessStartInfo("cmd.exe", "/c ping -n 60 127.0.0.1")
        {
            UseShellExecute = false,
            CreateNoWindow = true,
        })!;
        job.Assign(cmd);
        int ping = 0;
        Assert.True(WaitUntil(() => (ping = Pids("PING").Except(before).FirstOrDefault()) != 0, TimeSpan.FromSeconds(3)));
        job.Dispose();
        Assert.True(WaitUntil(() => !Alive(cmd.Id) && !Alive(ping), TimeSpan.FromSeconds(3)));
    }
}
