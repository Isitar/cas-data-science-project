using DataConversion.DataModels.Input;
using System.Text.Json;

namespace DataConversion.Extraction;

public class JsonLoader
{
    public static ClimbingRoute[] Load(string filename)
    {
        var jsonContent = File.ReadAllText(filename);

        var serializerOptions = new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true,
            PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        };


        ClimbingRoute[] data = JsonSerializer.Deserialize<ClimbingRoute[]>(jsonContent, serializerOptions)!;
        
        data = data.Select(dataPoint =>
                dataPoint with
                {
                    ConvertedPlacements = dataPoint.Placements
                        .Select(placement => new Placement(placement.X, placement.Y, placement.Type switch
                        {
                            null => PlacementType.None,
                            "FEET-ONLY" => PlacementType.FeetOnly,
                            "FINISH" => PlacementType.Finish,
                            "MIDDLE" => PlacementType.Middle,
                            "START" => PlacementType.Start
                        }, placement.LedPosition)).ToList()
                })
            .ToArray();
        
        return data;
    }
}