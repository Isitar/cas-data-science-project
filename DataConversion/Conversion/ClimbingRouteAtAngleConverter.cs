using System.Text.Json;
using System.Text.Json.Serialization;
using DataConversion.DataModels.Input;
using DataConversion.DataModels.Output;
using PlacementType = DataConversion.DataModels.Output.PlacementType;

namespace DataConversion.Conversion;

public class ClimbingRouteAtAngleConverter : JsonConverter<ClimbingRouteAtAngle>
{
    public override ClimbingRouteAtAngle Read(ref Utf8JsonReader reader, Type typeToConvert,
        JsonSerializerOptions options)
    {
        throw new NotImplementedException();
    }

    public override void Write(Utf8JsonWriter writer, ClimbingRouteAtAngle value, JsonSerializerOptions options)
    {
        writer.WriteStartObject();

        writer.WriteString("Uuid", value.Uuid);
        writer.WriteNumber("Angle", value.Angle);
        writer.WriteNumber("DifficultyAverage", value.DifficultyAverage);

        for (int i = 0; i < value.Placements.GetLength(0); i++)
        {
            for (int j = 0; j < value.Placements.GetLength(1); j++)
            {
                foreach (var placementType in Enum.GetValues<PlacementType>())
                {
                    string fieldName = $"{placementType}_{i}_{j}";
                    bool fieldValue = value.Placements[i, j] == placementType;
                    writer.WriteBoolean(fieldName, fieldValue);
                }
            }
        }

        writer.WriteEndObject();
    }
}