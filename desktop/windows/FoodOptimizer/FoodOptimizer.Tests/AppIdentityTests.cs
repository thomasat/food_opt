using FoodOptimizer.Core;

namespace FoodOptimizer.Tests;

public class AppIdentityTests
{
    [Fact]
    public void Version_is_desktop_VERSION()
    {
        var expected = File.ReadAllText(TestRepo.Path("desktop", "VERSION")).Trim();
        Assert.Equal(expected, AppIdentity.Version);
        Assert.DoesNotContain("+", AppIdentity.Version);
    }

    [Fact]
    public void Identity_names_are_fixed()
    {
        Assert.Equal("FoodSystemInnovations.FoodOptimizer", AppIdentity.AppUserModelId);
        Assert.Equal(@"Local\FoodOptimizer.SingleInstance.3CC8CB77-0313-4A7D-9582-9896F0A3A8B3", AppIdentity.SingleInstanceMutex);
        Assert.Equal(@"Local\FoodOptimizer.Running.PerUser.3CC8CB77-0313-4A7D-9582-9896F0A3A8B3", AppIdentity.RunningPerUser);
        Assert.Equal(@"Global\FoodOptimizer.Running.AllUsers.3CC8CB77-0313-4A7D-9582-9896F0A3A8B3", AppIdentity.RunningAllUsers);
        Assert.Equal(@"Local\FoodOptimizer.Activate.3CC8CB77-0313-4A7D-9582-9896F0A3A8B3", AppIdentity.ActivateEvent);
    }
}
