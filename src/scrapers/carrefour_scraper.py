import requests
import pandas as pd
from tqdm.auto import tqdm
from dotenv import load_dotenv

import os
from typing import Optional, Dict, Any, List
from constants import COUTNRY_LOCATION, CARREFOUR_PARAMS, CARREFOUR_HEADER, CARREFOUR_CSV_DATA_PATH

from modules.logger import logger

load_dotenv()


class CarrefourScraper:

    def __init__(
        self,
        csv_path_en: str,
        csv_path_ar: Optional[str] = None,
        page_size: int = 20,
    ) -> None:
        self.csv_path_en = csv_path_en
        self.csv_path_ar = csv_path_ar
        self.page_size = page_size
        self.params = CARREFOUR_PARAMS
        self.headers = CARREFOUR_HEADER
        self.products_url = os.getenv("CARREFOUR_PRODUCTS_API")
        self.locations_url = os.getenv("CARREFOUR_LOCATIONS_API")
        self.session = requests.Session()

    def get_response(
        self,
        url: str,
        headers: Dict[str, Any], 
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        response = self.session.get(url, headers=headers, params=params)

        if response.status_code != 200:
            logger.error(f"Could not get any response, code: {response.status_code}.")
            return {"success": False, "data": None}

        return {
            "success": True,
            "data": response
        }
    
    def get_locations(self) -> Dict[str, Any]:
        # response = self.session.get(self.locations_url, headers=self.headers)
        response = self.get_response(self.locations_url, headers=self.headers)

        if not response["success"]:
            return []
        
        locations = {}
        data = response["data"].json()
        for loc in data["data"]:
            if loc["displayName"] is not None:
                locations[loc["displayName"]] = {
                    "displayName": loc["displayName"],
                    "name": loc["name"],
                    "languages": [lang["code"] for lang in loc["languages"]],
                    # "currency": loc["currencies"][0]["name"],
                    "domain": loc["countryDomain"]
                }

        return locations

    def _get_categories(self, data: Dict[str, Any]) -> List[str]:
        if not data.get("children"):
            return [data["id"]]

        categories = []
        for child in data["children"]:
            categories.extend(self._get_categories(child))

        return categories

    def get_categories(
        self, 
        country_domain: str, 
        location_name: str, 
        # currency: str, 
        lang: str, 
        location_code: str
    ) -> Dict[str, Any]:
        base_url = self.products_url.split(".")[0] + "." + country_domain
        url = f"{base_url}/api/v1/menu"

        params = COUTNRY_LOCATION[location_name]
        # self.headers["currency"] = currency
        self.headers["langcode"] = lang
        self.headers["storeid"] = location_code
        # self.headers["Referer"] =
        response = self.get_response(url, headers=self.headers, params=params)

        if not response["success"]:
            return []

        data = response["data"].json()
        categories = self._get_categories(data[0])

        return categories
    
    def get_products(self, location_name: str, lang: str, category: str):
        url = self.products_url + category

        location = COUTNRY_LOCATION[location_name]
        params = {
            "sortBy": "relevance",
            "categoryCode": "",
            "needFilter": "false",
            "pageSize": 40,
            "requireSponsProducts": "true",
            "verticalCategory": "true",
            "needVariantsData": "true",
            "currentPage": 0,
            "responseWithCatTree": "true",
            "depth": 3,
            "lang": "en",
            "categoryId": "F21630500",
            "latitude": 25.2171003,
            "longitude": 55.3613635
        }
        params["latitude"] = location["latitude"]
        params["longitude"] = location["longitude"]
        params["lang"] = lang
        params["pageSize"] = self.page_size
        params["categoryId"] = category

        response = self.get_response(url, self.headers, params)

        if not response["success"]:
            return []

        try:
            data = response["data"].json()
        except:
            return []
        num_pages = data["numOfPages"] if "numOfPages" in data else 0
        products = data["products"]
        for page in range(1, num_pages + 1):
            self.params["currentPage"] = page
            response = self.get_response(url, self.headers, params)
            
            if not response["success"]:
                continue
            try:
                data = response["data"].json()
                products.extend(data["products"])
            except:
                continue

        return products
    
    def save_products_csv(self, products: List[Dict[str, Any]]) -> None:
        products_id = []
        products_name = []
        products_url = []
        products_brand = []
        products_cat_level_1 = []
        products_cat_level_2 = []
        products_cat_level_3 = []
        for prod in products:
            products_id.append(prod["id"])
            products_name.append(prod["name"])
            products_url.append(prod["url"])
            products_brand.append(prod["brand"]["name"])
            cat1 = cat2 = cat3 = None

            for cat in prod["categories"]:
                if cat["level"] == 1:
                    cat1 = cat["name"]
                if cat["level"] == 2:
                    cat2 = cat["name"]
                if cat["level"] == 3:
                    cat3 = cat["name"]
            
            products_cat_level_1.append(cat1)
            products_cat_level_2.append(cat2)
            products_cat_level_3.append(cat3)

        products_csv = pd.DataFrame({
            "id": products_id,
            "name": products_name,
            "lavel_1": products_cat_level_1,
            "level_2": products_cat_level_2,
            "level_3": products_cat_level_3,
            "url": products_url
        })

        products_csv.to_csv(self.csv_path_en, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    scraper = CarrefourScraper(CARREFOUR_CSV_DATA_PATH)
    locations = scraper.get_locations()
    all_products = []
    for loc_name, loc_data in tqdm(locations.items(), total=len(locations)):
        code = loc_data["name"]
        languages = loc_data["languages"]
        # currency = loc_data["currency"]
        country_domain = loc_data["domain"]
        for lang in languages:
            categories = scraper.get_categories(country_domain, loc_name, lang, code)
            for category in tqdm(categories, total=len(categories)):
                products = scraper.get_products(loc_name, lang, category)
                if not products:
                    logger.warning(f"No products found category {category}, lang={lang}, store={scraper.headers['storeid']}, location={loc_name}")
                all_products.extend(products)
                logger.info(f"Total number of products are {len(all_products)}")

    print(len(all_products))
    scraper.save_products_csv(all_products)