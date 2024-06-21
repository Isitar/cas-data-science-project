using DataConversion.Analysis;
using DataConversion.Conversion;
using DataConversion.Extraction;
using DataConversion.Output;

public class JsonEntryPoint
{
    public static void Run(string[] args)
    {
        var fileName = args.Any() ? args[0] : "sample_data/samples.json";
        var data = JsonLoader.Load(fileName);

        Console.WriteLine($"got {data.Length} input data");
        var dataWithLayout = data
            .Where(d => d.LayoutId == 1)
            .Where(d => d.LayoutDeviantId == 9)
            .ToList();


        Console.WriteLine(dataWithLayout.Where(d => d.Placements.Any(p => p.Y > 35)).ToList().Count);
        Console.WriteLine(dataWithLayout.Where(d => d.Placements.Any(p => p.Y < 0)).ToList().Count);

        var dataWithValidY = dataWithLayout.Where(d => !d.Placements.Any(p => p.Y > 35));

        var finalData = dataWithValidY;

// Console.WriteLine(dataWithLayout.Where(d => d.Placements.Any(p => p.Y > 35)).ToList().First());


// dataWithLayout.Where(d => d.Name.ToLower().Contains("the pearl")).ToList().ForEach(Console.WriteLine);
        DataAnalyser.AnalysePlacements(finalData.SelectMany(route => route.Placements));
        DataAnalyser.AnalyseStats(finalData.SelectMany(route => route.ClimbStats));

        finalData.Where(fd => fd.Name.Contains("bob the dino")).SelectMany(fd => fd.Placements).ToList()
            .ForEach(Console.WriteLine);

        var convertedData = finalData.SelectMany(DataConverter.ConvertClimbingRoute).ToList();
        Console.WriteLine($"Converted to {convertedData.Count} routes");
        OutputSerializer.Serialize(convertedData);
        Console.WriteLine("Done");
    }
}