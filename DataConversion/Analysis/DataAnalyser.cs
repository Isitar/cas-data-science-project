using DataConversion.DataModels.Input;

namespace DataConversion.Analysis;

public class DataAnalyser
{
    public static void AnalysePlacements(IEnumerable<JsonPlacement> placements)
    {
        var intermediate = placements.ToList();
        ExtractAndOutput(intermediate, p => p.Type, nameof(Placement.Type));
        ExtractAndOutput(intermediate, p => p.X, nameof(Placement.X));
        ExtractAndOutput(intermediate, p => p.Y, nameof(Placement.Y));
        ExtractAndOutput(intermediate, p => p.X + "_" + p.Y, "XY");
        ExtractAndOutput(intermediate, p => (p.Type ?? "MIDDLE") + "_" + p.X + "_" + p.Y, "TypeXY");
        ExtractAndOutput(intermediate, p => p.LedPosition, nameof(Placement.LedPosition));
    }

    public static void AnalyseStats(IEnumerable<ClimbStat> stats)
    {
        var intermediate = stats.ToList();
        ExtractAndOutput(intermediate, p => p.Angle, nameof(ClimbStat.Angle));
    }

    private static IEnumerable<TK?> Extract<T, TK>(IEnumerable<T> datas, Func<T, TK?> selector, string name)
        => datas.Select(selector).Distinct().Order();

    private static void ExtractAndOutput<T, TK>(IEnumerable<T> datas, Func<T, TK?> selector, string name)
    {
        var extracted = Extract(datas, selector, name).ToList();
        Console.WriteLine($"{name} ({extracted.Count}): {string.Join(", ", extracted)}");
    }
}