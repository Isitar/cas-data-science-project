using System.Text.Json.Serialization;

namespace DataConversion.DataModels.Input;

public record ClimbStat(
    int Angle,
    string FaAt,
    string ClimbUuid,
    string FaUsername,
    double QualityAverage,
    int AscensionistCount,
    double DifficultyAverage
);

public enum PlacementType
{
    None,
    FeetOnly,
    Finish,
    Middle,
    Start
};

public record JsonPlacement(    
    int X,
    int Y,
    string? Type,
    [property: JsonPropertyName("ledPosition")]
    int? LedPosition);

public record Placement(
    int X,
    int Y,
    PlacementType? Type,
    int? LedPosition);

public record ClimbingRoute(
    string Uuid,
    int? LayoutId,
    int? LayoutDeviantId,
    int? SetterId,
    string? SetterUsername,
    string? Name,
    string? Description,
    int Hsm,
    int EdgeLeft,
    int EdgeRight,
    int EdgeBottom,
    int EdgeTop,
    int FramesCount,
    int FramesPace,
    bool IsDraft,
    bool IsListed,
    DateTime CreatedAt,
    List<ClimbStat> ClimbStats,
    List<JsonPlacement> Placements,
    [property: JsonIgnore] List<Placement> ConvertedPlacements,
    int TotalAscents);