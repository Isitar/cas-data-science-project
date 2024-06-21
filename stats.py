import json

def load_and_count(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
     
        return data, len(data)


def main():
    data, cnt = load_and_count('extracted_all.json')

    for stat in  data[0]['climb_stats']:
        print(f'angle: {stat["angle"]}, dif: {stat["difficulty_average"]}')
    print(f"cnt: {cnt}")

if __name__ == "__main__":
    main()