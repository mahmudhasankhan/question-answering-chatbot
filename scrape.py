from bs4 import BeautifulSoup
from selenium import webdriver
import time
# import json
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--url",
                    type=str,
                    default=None,
                    help="URL of the webpage that you want to scrape data from")
args = parser.parse_args()

URL = args.url

if URL is None:
    raise Exception("Webpage URL must not be None!")
else:
    driver = webdriver.Chrome()
    driver.get(URL)

time.sleep(4)

soup = BeautifulSoup(driver.page_source, "html.parser")
special_divs = soup.find_all(
    'div', {'class': 'row product-row-scroll'})

item_links = []
for div in special_divs:
    for item in div.find_all('a', attrs={"class": "details-link"}):
        item_links.append(item['href'])

driver.quit()

product_driver = webdriver.Chrome()

data = {
    "product link": [],
    "product details": [],
}

for i, link in enumerate(item_links):
    # if i == 3:
    #     break

    product_link = "https://sindabad.com" + link
    product_driver.get(product_link)
    time.sleep(8)
    new_soup = BeautifulSoup(product_driver.page_source, "html.parser")
    product_desc = new_soup.find(
        "div", {"class": "osahan-main-body py-3 py-md-4"})
    print(f"Product Number: {i+1}")

    data["product link"].append(product_link)
    texts = ""
    for li in product_desc.find_all("li"):
        if li.text != "":
            texts += li.text + ", "
    # text_list["product details"].append([
    #     li.text for li in product_desc.find_all("li") if li.text != ""])

    data["product details"].append(texts)

# print(data["product details"])
# with open("./data/product_list.json", "w") as f:
#     json.dump(data, f)

dataframe = pd.DataFrame.from_dict(data)
dataframe.to_csv("./data/product_list.csv", header=True, index=False)
product_driver.quit()
