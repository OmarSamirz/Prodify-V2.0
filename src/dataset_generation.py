import requests
import json
import time
import sys
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

API_KEY = "<API-KEY>"
MODEL_NAME = "deepseek/deepseek-chat-v3.1:free"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 1  # seconds
MAX_RETRY_DELAY = 300    # 5 minutes
RATE_LIMIT_DELAY = 60    # 1 minute base delay for rate limits
REQUEST_TIMEOUT = 30     # 30 seconds timeout

def create_session_with_retries():
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    return session

def make_api_request_with_retry(prompt: str, session: requests.Session):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
    }
    
    retry_delay = INITIAL_RETRY_DELAY
    
    for attempt in range(MAX_RETRIES):
        try:
            response = session.post(
                API_URL, 
                headers=headers, 
                json=payload, 
                timeout=REQUEST_TIMEOUT
            )
            
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"].strip()
                else:
                    print("Warning: Empty or invalid response from API.")
                    return ""
            
            elif response.status_code == 429:
                print(f"Rate limit hit. Waiting {RATE_LIMIT_DELAY} seconds before retry...")
                time.sleep(RATE_LIMIT_DELAY)
                retry_delay = RATE_LIMIT_DELAY
            
            else:
                print(f"API request failed with status {response.status_code}: {response.text}")
                response.raise_for_status()
                
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            print(f"An error occurred on attempt {attempt + 1}: {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, MAX_RETRY_DELAY)
            else:
                print("All retry attempts failed. Pausing for max delay before continuing...")
                time.sleep(MAX_RETRY_DELAY)
                retry_delay = INITIAL_RETRY_DELAY
    
    #if all retries fail, wait and return empty to continue script
    print("Failed after max retries. Waiting max delay and continuing...")
    time.sleep(MAX_RETRY_DELAY)
    return ""

def parse_list_from_response(response: str, limit: int):
    # basic preprocessing
    lines = response.split('\n')
    items = []
    for line in lines:
        stripped = line.strip()
        if stripped and (stripped[0].isdigit() or stripped.startswith('-') or stripped.startswith('*')):
            item = stripped.lstrip('0123456789.-* ').strip()
            if item:
                items.append(item)
    return items[:limit]

def generate_dataset(test_mode=False):
    api_session = create_session_with_retries()
    
    # Limits numbers for test mode
    num_sectors = 3 if test_mode else 10
    num_brands = 3 if test_mode else 10
    num_products = 3 if test_mode else 20
    
    dataset = {}
    
    #Step 1: get top sectors
    print("Fetching top consumer product sectors...")
    sectors_prompt = f"Search online and list the top {num_sectors} consumer product sectors (however it mustn't be included in the these: []). Provide as a numbered list."
    sectors_response = make_api_request_with_retry(sectors_prompt, api_session)
    sectors = parse_list_from_response(sectors_response, num_sectors)
    
    if not sectors:
        print("Failed to retrieve sectors. Exiting.")
        return
    
    print(f"Retrieved {len(sectors)} sectors: {sectors}")
    
    total_steps = len(sectors) * (1 + num_brands)
    progress_bar = tqdm(total=total_steps, desc="Overall Progress", unit="step")
    
    for sector in sectors:
        dataset[sector] = {}
        
        #Step 2: Get top brands for this sector
        print(f"\nProcessing sector: {sector}")
        brands_prompt = f"Search online and list the top {num_brands} brands by valuation in the {sector} consumer product sector. Provide as a numbered list."
        brands_response = make_api_request_with_retry(brands_prompt, api_session)
        brands = parse_list_from_response(brands_response, num_brands)
        
        if not brands:
            print(f"Failed to retrieve brands for {sector}. Skipping.")
            progress_bar.update(1 + num_brands)
            continue
        
        print(f"Retrieved {len(brands)} brands for {sector}: {brands}")
        progress_bar.update(1)
        
        for brand in brands:
            #Step 3: Get top products for this brand
            print(f"  Processing brand: {brand}")
            products_prompt = f"Search online and list the top {num_products} products of the brand {brand} in the {sector} sector. Provide as a numbered list."
            products_response = make_api_request_with_retry(products_prompt, api_session)
            products = parse_list_from_response(products_response, num_products)
            
            if not products:
                print(f"    Failed to retrieve products for {brand}. Skipping.")
                progress_bar.update(1)
                continue
            
            dataset[sector][brand] = products
            print(f"    Retrieved {len(products)} products for {brand}: {products}")
            progress_bar.update(1)
    
    progress_bar.close()
    
    output_file = "consumer_products_dataset_test.json" if test_mode else "consumer_products_dataset.json"
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=4)
    
    print(f"\nDataset generation complete. Saved to {output_file}")

if __name__ == "__main__":
    test_mode = len(sys.argv) > 1 and sys.argv[1].lower() == 'test'
    print(f"Running in {'test' if test_mode else 'full'} mode...")
    
    try:
        generate_dataset(test_mode)
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C). Exiting gracefully.")
        sys.exit(0)
