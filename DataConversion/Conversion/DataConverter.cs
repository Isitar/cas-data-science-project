using DataConversion.DataModels.Input;
using DataConversion.DataModels.Output;
using PlacementType = DataConversion.DataModels.Output.PlacementType;

namespace DataConversion.Conversion;

public class DataConverter
{
    public static IEnumerable<ClimbingRouteAtAngle> ConvertClimbingRoute(ClimbingRoute climbingRoute)
    {
        var output = new ClimbingRouteAtAngle(climbingRoute.Uuid, 0, new PlacementType[35, 36], 0);
        FillPlacements(output, climbingRoute.ConvertedPlacements);
        foreach (var climbStat in climbingRoute.ClimbStats)
        {
            yield return output with { Angle = climbStat.Angle, DifficultyAverage = climbStat.DifficultyAverage };
        }
    }

    private static void FillPlacements(ClimbingRouteAtAngle routeAtAngle, IEnumerable<Placement> placements)
    {
        foreach (var placement in placements)
        {
            routeAtAngle.Placements[placement.X - 1, placement.Y] = placement.Type switch
            {
                null
                    or DataModels.Input.PlacementType.Middle
                    or DataModels.Input.PlacementType.None => PlacementType.M,
                DataModels.Input.PlacementType.Start => PlacementType.S,
                DataModels.Input.PlacementType.Finish => PlacementType.F,
                DataModels.Input.PlacementType.FeetOnly => PlacementType.FO
            };
        }
    }
}