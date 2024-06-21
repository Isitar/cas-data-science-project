import json
import os

# Function to load and combine JSON files
def combine_json_files(input_folder, output_file):
    combined_data = []
    for filename in os.listdir(input_folder):
        if filename.startswith("extracted_") and filename.endswith(".json"):
            with open(os.path.join(input_folder, filename), 'r') as file:
                data = json.load(file)
                combined_data.extend(data)

    # Writing combined data to a single JSON file
    with open(output_file, 'w') as outfile:
        json.dump(combined_data, outfile)

# Main function
def main():
    input_folder = '.'  # Assuming JSON files are in the current directory
    output_file = 'extracted_all.json'

    combine_json_files(input_folder, output_file)
    print("Combined JSON file saved as", output_file)

if __name__ == "__main__":
    main()