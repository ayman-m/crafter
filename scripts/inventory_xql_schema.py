import os
import requests
import json
from tqdm import tqdm
from bs4 import BeautifulSoup

# File paths
toc_filename = 'data/source/xql_files/xql_toc.json'
details_filename = 'data/source/xql_files/xql_details.json'

# Base URLs
toc_url_template = 'https://docs-cortex.paloaltonetworks.com/internal/api/webapp/maps/{documentId}/toc'
details_url = 'https://docs-cortex.paloaltonetworks.com/internal/api/webapp/reader/topics/request'

# Headers
headers = {
    'accept': 'application/json',
    'accept-language': 'en-US,en;q=0.9',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
    'ft-calling-app': 'ft/turnkey-portal',
    'ft-calling-app-version': '5.0.29',
    'x-http-method-override': 'GET',
}

# Helper function to create missing files
def create_file_if_missing(filename, default_content=None):
    if not os.path.exists(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as file:
            json.dump(default_content or {}, file)
        print(f"Created missing file: {filename}")

# Fetch TOC data
def fetch_toc(document_id):
    toc_url = toc_url_template.format(documentId=document_id)
    response = requests.get(toc_url, headers=headers)
    response.raise_for_status()
    toc_data = response.json()
    with open(toc_filename, 'w') as file:
        json.dump(toc_data, file)
    print(f"TOC data saved to {toc_filename}")
    return toc_data

# Fetch details for topics
def fetch_details(toc_data):
    topics = [
        {
            "sourceType": "OFFICIAL",
            "originMapId": topic['topic'].get('mapId'),
            "originTocId": topic['topic'].get('tocId'),
            "contentId": topic['topic']['link'].get('contentId')
        }
    
        for topic in toc_data.get('toc', [])
        if topic['topic'].get('link') and topic['topic']['link'].get('contentId')
    ]
    print (topics)
    if not topics:
        print("No topics found in TOC data.")
        return []

    payload = {"topics": topics}
    response = requests.post(details_url, headers=headers, json=payload)
    response.raise_for_status()
    details_data = response.json()
    with open(details_filename, 'w') as file:
        json.dump(details_data, file)
    print(f"Details data saved to {details_filename}")
    return details_data

# Main function
def main():
    # Ensure necessary files exist
    create_file_if_missing(toc_filename, {})
    create_file_if_missing(details_filename, [])

    # Example document ID
    document_id = "0aa6Nsr7VgBlIGGUI8TKHg"  # Replace with the actual document ID

    # Fetch TOC and details
    toc_data = fetch_toc(document_id)
    if toc_data:
        fetch_details(toc_data)
    else:
        print("Failed to fetch TOC data.")

if __name__ == "__main__":
    main()
