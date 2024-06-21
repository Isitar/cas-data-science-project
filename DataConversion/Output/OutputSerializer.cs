using System.Text.Json;
using DataConversion.Conversion;
using DataConversion.DataModels.Output;

namespace DataConversion.Output;

public class OutputSerializer
{
    public static void Serialize(IEnumerable<ClimbingRouteAtAngle> climbingRoutesAtAngle)
    {
        var jsonSerializerOptions = new JsonSerializerOptions();
        jsonSerializerOptions.Converters.Add(new ClimbingRouteAtAngleConverter());

        var outputText = JsonSerializer.Serialize(climbingRoutesAtAngle, jsonSerializerOptions);
        File.WriteAllText("output.json", outputText);
    }
}