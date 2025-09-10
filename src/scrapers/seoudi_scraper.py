import requests
import pandas as pd
from tqdm.auto import tqdm
from dotenv import load_dotenv

import os
import string
from typing import Dict, Any, Optional, List

from modules.logger import logger
from constants import SEOUDI_QUERY, SEOUDI_EN_DATA_PATH, SEOUDI_AR_DATA_PATH

load_dotenv()


class SeoudiScraper:

    def __init__(
        self,
        csv_path_en: str,
        csv_path_ar: Optional[str] = None,
        page_size: int = 1000,
    ) -> None:
        self.csv_path_en = csv_path_en
        self.csv_path_ar = csv_path_ar
        self.page_size = page_size
        self.url = os.getenv("SEOUDI_API")
        self. query = SEOUDI_QUERY
        self.prefixes = list(string.ascii_uppercase) + list(string.digits)
        self.session = requests.Session()
    
    def get_response(self, variables: Dict[str, Any], headers: Dict[str, Any]) -> Dict[str, Any]:
        response = self.session.post(self.url, json={"query": self.query, "variables": variables}, headers=headers)

        if response.status_code != 200:
            logger.error(f"Could not get any response, code: {response.status_code}.")
            return {"success": False, "data": None}

        return {
            "success": True,
            "data": response
        }

    def get_products(self, lang: str = "en") -> List[Dict[str, Any]]:
        store = "default" if lang == "en" else "ar_EG"
        language = "en-US,en;q=0.9" if lang == "en" else "ar-EG"
        all_products = []
        for prefix in tqdm(self.prefixes, total=len(self.prefixes)):
            variables = {
                "page": 1,
                "pageSize": self.page_size,
                "search": f"{prefix}"
            }
            headers = {
                "Store": store,
                "Accept-Language": language,
                "Accept-Encoding": "gzip, deflate, br, zstd",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36",
            }
            response = self.get_response(variables, headers)
            
            if not response["success"]:
                continue

            data = response["data"].json()
            data = data["data"]["connection"]
            total_pages = data["page_info"]["total_pages"]
            all_products.extend(data["nodes"])

            for page in range(2, total_pages + 1):
                variables["page"] = page
                response = self.get_response(variables, headers)
                data = response["data"].json()["data"]["connection"]
                all_products.extend(data["nodes"])

            logger.info(f"Total number of scrapped products `{len(all_products)}` of prefix `{prefix}`.")

        return all_products

    def save_products_csv(self, products: List[Dict[str, Any]], lang: str = "en") -> None:
        products_id = []
        products_name = []
        products_brand = []
        products_link = []
        products_cat_level_1 = []
        products_cat_level_2 = []
        products_cat_level_3 = []
        for prod in products:
            if not prod["categories"]:
                continue

            products_id.append(prod["id"])
            products_name.append(prod["name"])
            products_brand.append(prod["brand"]["name"] if prod["brand"] is not None else None)
            products_link.append(prod["url_key"])
            cat1 = cat2 = cat3 = None

            if not isinstance(prod["categories"], List):
                prod["categories"] = [prod["categories"]]

            for category in prod["categories"]:
                if "top-offeres" not in category["url_path"] and category["level"] == 2:
                    cat1 = category["name"]
                if "top-offeres" not in category["url_path"] and category["level"] == 3:
                    cat2 = category["name"]
                if "top-offeres" not in category["url_path"] and category["level"] == 4:
                    cat3 = category["name"]

            products_cat_level_1.append(cat1)
            products_cat_level_2.append(cat2)
            products_cat_level_3.append(cat3)

        df = pd.DataFrame({
            "id": products_id,
            "name": products_name,
            "brand": products_brand,
            "url": products_link,
            "level_1": products_cat_level_1,
            "level_2": products_cat_level_2,
            "level_3": products_cat_level_3,
        })

        path = self.csv_path_en if lang == "en" else self.csv_path_ar
        df.drop_duplicates(subset=["id"], inplace=True)
        df.to_csv(path, index=False, encoding="utf-8-sig")


def main():
    scraper = SeoudiScraper(csv_path_en=SEOUDI_EN_DATA_PATH, csv_path_ar=SEOUDI_AR_DATA_PATH)
    
    en_products = scraper.get_products()
    scraper.save_products_csv(en_products)

    ar_products = scraper.get_products("ar")
    scraper.save_products_csv(ar_products, "ar")

if __name__ == "__main__":
    main()