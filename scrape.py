from bs4 import BeautifulSoup
from selenium import webdriver
import time
import json
import argparse

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

time.sleep(8)

soup = BeautifulSoup(driver.page_source, "html.parser")
special_divs = soup.find_all(
    'div', {'class': 'row product-row-scroll'})

item_links = []
for div in special_divs:
    for item in div.find_all('a', attrs={"class": "details-link"}):
       item_links.append(item['href'])

driver.quit()

product_driver = webdriver.Chrome()

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
    
    text_list = []
    text_list.append(f"Product link: {product_link}")
    
    for li in product_desc.find_all("li"):
        text_list.append(li.text)
    with open(f"./data/product_id_{i+1}.json", "w") as f:
        json.dump(text_list, f)
product_driver.quit()
