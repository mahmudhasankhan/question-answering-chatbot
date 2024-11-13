import time
import argparse
import pandas as pd
import os
import logging

from bs4 import BeautifulSoup
from selenium import webdriver
from tqdm import tqdm

# consts
PATH = "./data/product_list.csv"
# class attrs for sub category link scraping
SUB_CATEGORY_CLASS_ATTR = "list-card-image"
CATEGORY_CLASS_ATTR = "text-dark"
# class attrs for product link scraping for each sub category
PRODUCT_CLASS_ATTR = 'row product-row-scroll'
PRODUCT_LINK_CLASS_ATTR = "details-link"
PRODUCT_DIV_CLASS_ATTR = "osahan-main-body py-3 py-md-4"


def initialize_browser():
    driver = webdriver.Chrome()
    logging.info("Web driver initialized")
    return driver


def sub_category_link_scraper(args, driver):

    logging.info("Started scraping sub-category links")
    URL = args.url

    if URL is None:
        logging.critical("Exception has occurred, URL is None!")
        raise Exception("Webpage URL must not be None!")
    else:
        driver = webdriver.Chrome()
        driver.get(URL)

    time.sleep(15)
    soup = BeautifulSoup(driver.page_source, "html.parser")

    special_divs = soup.find_all(
        'div', {'class': SUB_CATEGORY_CLASS_ATTR})
    links_of_sub_categories = []

    for div in tqdm(special_divs, desc="scraping sub-category links"):
        for item in div.find_all('a', attrs={"class": CATEGORY_CLASS_ATTR}):
            links_of_sub_categories.append(item['href'])
    print(links_of_sub_categories, end='\n')
    driver.quit()
    logging.info("web driver stopped")
    return links_of_sub_categories


def product_link_scraper(driver, sub_category_links):
    logging.info("Started scraping products from sub-category links")

    links_of_products = []

    for i, url in enumerate(tqdm(sub_category_links,
                                 desc=f"scraping products for each sub category urls")):
        # if i == 3:
        #     break
        url = "https://sindabad.com" + url
        driver.get(url)
        time.sleep(15)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        special_divs = soup.find_all(
            'div', {'class': PRODUCT_CLASS_ATTR})

        for div in tqdm(special_divs, desc="scraping product links"):
            for item in div.find_all('a', attrs={"class": PRODUCT_LINK_CLASS_ATTR}):
                links_of_products.append(item['href'])

    driver.quit()
    logging.info("web driver stopped")
    logging.info("Scraping products from sub-category links is complete")
    print(links_of_products, end='\n')
    return links_of_products


def product_scraper(driver, product_links):

    logging.info("Product scraper has started")

    data = {
        "product link": [],
        "product details": [],
    }

    for i, link in enumerate(tqdm(product_links, desc="scraping product details")):
        # if i == 3:
        #     break
        product_link = "https://sindabad.com" + link
        driver.get(product_link)
        time.sleep(15)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        product_desc = soup.find(
            "div", {"class": PRODUCT_DIV_CLASS_ATTR})
        print(f"Product Number: {i+1}")

        data["product link"].append(product_link)
        texts = ""
        for li in product_desc.find_all("li"):
            if li.text != "":
                texts += li.text + ", "

        data["product details"].append(texts)

    driver.quit()
    logging.info("Web driver stopped")
    logging.info("Finished scraping products")
    return data


def main():

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename="./logs/scrape_data.log")

    parser = argparse.ArgumentParser()
    parser.add_argument("--url",
                        type=str,
                        default=None,
                        help="URL of the webpage that you want to scrape data from")

    args = parser.parse_args()
    logging.info("Web scraping started")
    start_time = time.time()

    driver = initialize_browser()
    sub_category_links = sub_category_link_scraper(args, driver)

    driver = initialize_browser()
    links = product_link_scraper(driver, sub_category_links)

    driver = initialize_browser()
    data = product_scraper(driver, links)

    dataframe = pd.DataFrame.from_dict(data)

    logging.info("Saving the data into a csv file")
    if os.path.exists(PATH):
        print(f"csv file exists at {PATH}")
        dataframe.to_csv(PATH, mode="a", header=False, index=False)
    else:
        print(f"file doesn't exist, creating at {PATH}")
        dataframe.to_csv(PATH, header=True, index=False)

    logging.info(f"Finished saving the data in a csv file at {PATH}")

    elapsed = (time.time() - start_time) / 60

    print(f'Total elapsed time: {elapsed:.2f} min')

    logging.info(f'Total elapsed time: {elapsed:.2f} min')


if __name__ == "__main__":
    main()
