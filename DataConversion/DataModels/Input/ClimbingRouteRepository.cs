using Microsoft.Data.Sqlite;

namespace DataConversion.DataModels.Input;

public class ClimbingRouteRepository
{
    private readonly string connectionString;

    public ClimbingRouteRepository(string connectionString)
    {
        this.connectionString = connectionString;
    }


    /// <summary>
    /// ordered by difficulty
    /// </summary>
    private List<(int difficulty, string display)> difficulties = new();

    public string Difficulty(double displayDifficulty)
    {
        if (difficulties.Count == 0)
        {
            LoadDifficulties();
        }

        return difficulties
            .First(d => d.difficulty >= displayDifficulty)
            .display;
    }

    private void LoadDifficulties()
    {
        using (var connection = new SqliteConnection(connectionString))
        {
            connection.Open();

            var command = connection.CreateCommand();
            command.CommandText =
                @"select difficulty, boulder_name
from difficulty_grades
order by difficulty";

            using (var reader = command.ExecuteReader())
            {
                while (reader.Read())
                {
                    difficulties.Add((reader.GetInt32(0), reader.GetString(1)));
                }
            }
        }
    }

    public void CreatePlacementRoleColumns()
    {
        var placementIds = new List<int>();
        var roleIds = new List<int>();

        using (var connection = new SqliteConnection(connectionString))
        {
            connection.Open();

            var command = connection.CreateCommand();
            command.CommandText =
                @"select id from placements where layout_id = 1";

            using (var reader = command.ExecuteReader())
            {
                while (reader.Read())
                {
                    var id = reader.GetInt32(0);
                    placementIds.Add(id);
                }
            }

            command = connection.CreateCommand();
            command.CommandText = @"select id from placement_roles where product_id=1";

            using (var reader = command.ExecuteReader())
            {
                while (reader.Read())
                {
                    var id = reader.GetInt32(0);
                    roleIds.Add(id);
                }
            }

            // foreach (var roleId in roleIds)
            // {
            foreach (var placementId in placementIds)
            {
                var columnName = $"p{placementId}";
                command = connection.CreateCommand();
                // command.CommandText = @$"alter table climbs drop column {columnName}";
                command.CommandText = @$"alter table climbs add column {columnName} BOOLEAN default 0";
                command.ExecuteNonQuery();
                command = connection.CreateCommand();
                command.CommandText = $"UPDATE climbs SET {columnName} = 1 WHERE frames LIKE '%{columnName}%';";
                command.ExecuteNonQuery();
            }
            // }
        }
    }

    public void CreateOutputTable()
    {
        var placementIds = new List<int>();
        var roleIds = new List<int>();

        using (var connection = new SqliteConnection(connectionString))
        {
            connection.Open();

            var command = connection.CreateCommand();
            command.CommandText =
                @"select id from placements where layout_id = 1";

            using (var reader = command.ExecuteReader())
            {
                while (reader.Read())
                {
                    var id = reader.GetInt32(0);
                    placementIds.Add(id);
                }
            }

            command = connection.CreateCommand();
            command.CommandText = @"select id from placement_roles where product_id=1";

            using (var reader = command.ExecuteReader())
            {
                while (reader.Read())
                {
                    var id = reader.GetInt32(0);
                    roleIds.Add(id);
                }
            }

            var climbColumns = new List<string>();
            climbColumns.Add("c.uuid");


            // foreach (var roleId in roleIds)
            // {
            foreach (var placementId in placementIds)
            {
                var columnName = $"c.p{placementId}";
                climbColumns.Add(columnName);
            }
            // }

            command = connection.CreateCommand();
            command.CommandText =
                $"""
                 CREATE TABLE climbs_cleaned as 
                 SELECT {string.Join(",", climbColumns)}, cs.angle, dg.boulder_name as difficulty
                 FROM climbs c 
                     JOIN climb_stats cs on cs.climb_uuid = c.uuid
                     JOIN difficulty_grades dg on floor(cs.display_difficulty) = dg.difficulty
                 WHERE c.layout_id = 1
                   and c.is_listed
                   and not c.is_draft
                   and c.frames_count = 1
                 """;

            command.ExecuteNonQuery();
        }
    }

    public IEnumerable<SqLiteClimbingRoute> Routes()
    {
        using (var connection = new SqliteConnection(connectionString))
        {
            connection.Open();

            var command = connection.CreateCommand();
            command.CommandText =
                @"
select 
    c.uuid, 
    c.name,
    c.frames, 
    cs.angle, 
    cs.display_difficulty 
from climbs c 
    join climb_stats cs on cs.climb_uuid == c.uuid";

            using (var reader = command.ExecuteReader())
            {
                while (reader.Read())
                {
                    var uuid = reader.GetString(0);
                    var name = reader.GetString(1);
                    var frames = reader.GetString(2);
                    var angle = reader.GetInt32(3);
                    var displayDifficulty = reader.GetDouble(4);
                    yield return new SqLiteClimbingRoute(uuid, name, frames, angle, displayDifficulty);
                }
            }
        }
    }
}