namespace DataConversion.DataModels.Output;

public record ClimbingRouteAtAngle(
    string Uuid,
    int Angle,
    PlacementType[,] Placements,
    double DifficultyAverage // our target variable
);

public enum PlacementType
{
    FO,
    F,
    M,
    S
};
