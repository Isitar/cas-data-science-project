using DataConversion.DataModels.Input;

namespace DataConversion.EntryPoints;

public class SqliteEntryPoint
{
    public static void Run()
    {
        var repo = new ClimbingRouteRepository(
            "Data Source=/home/palu/Projects/cas-data-science/00_Projektarbeit/data/boardlib-kilter-db.sqlite3;");
        repo.CreateOutputTable();
        // repo.CreatePlacementRoleColumns();
        // var routes = repo.Routes();
        // foreach (var route in routes)
        // {
        //     Console.WriteLine($"{route.Name}, {route.Angle}, {route.Difficulty}, {repo.Difficulty(route.Difficulty)}");
        //     var b = Console.ReadLine();
        //     if (Equals(b, "x"))
        //     {
        //         break;
        //     }
        // }
    }
}