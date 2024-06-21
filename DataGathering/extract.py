import requests
import json

def download_json_from_api(url, headers):
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for 4xx and 5xx status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print("Error downloading JSON:", e)
        return None


def main():     
    api_url = "https://drmzubrwofxyzyhicvvx.supabase.co/rest/v1/climbs?select=*&order=total_ascents.desc.nullslast"  # Replace with your API endpoint
    auth_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJyb2xlIjoiYW5vbiIsImlhdCI6MTYzMTI3NDA1MiwiZXhwIjoxOTQ2ODUwMDUyfQ.vBZ8uBgVI3Wc9RaJ2STinaVnd0dY2HHyK42YkqBxUR0"  # Replace with your authentication token
    api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJyb2xlIjoiYW5vbiIsImlhdCI6MTYzMTI3NDA1MiwiZXhwIjoxOTQ2ODUwMDUyfQ.vBZ8uBgVI3Wc9RaJ2STinaVnd0dY2HHyK42YkqBxUR0"  # Replace with your API key
    
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "apikey": api_key,
        "Content-Type": "application/json"
    }

    limit = 500
    for i in range(100):
        api_url = f"https://drmzubrwofxyzyhicvvx.supabase.co/rest/v1/climbs?select=*&order=total_ascents.desc.nullslast&offset={i*limit}&limit={limit}"  
        json_data = download_json_from_api(api_url, headers)
        

        if json_data:
            print(f"JSON downloaded successfully: {i}")
            filename = f'extracted_{i}.json'
            with open(filename, 'w') as file:
                json.dump(json_data, file)
        else:
            print("Failed to download JSON data.")

if __name__ == "__main__":
    main()
