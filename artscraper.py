from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import requests
import time
import os
import json

ARTIST = 'vincent-van-gogh'
BASE_URL = f'https://www.wikiart.org/en/vincent-van-gogh/all-works/text-list'
IMG_DIR = f'wikiart/{ARTIST}'

def setup_driver():
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    return webdriver.Chrome(options=options)

def save_image(img_url, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = img_url.split('/')[-1].split('!')[0]
    img_path = os.path.join(folder, filename)
    r = requests.get(img_url, stream=True)
    if r.status_code == 200:
        with open(img_path, 'wb') as f:
            f.write(r.content)
        return filename
    return None

def scroll_to_load_all(driver):
    scroll_pause = 2
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pause)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

def scrape_paintings(driver):
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    paintings = []
    grid_items = soup.select('li[data-id]')  # each painting list item
    print(f"Found {len(grid_items)} paintings")

    for item in grid_items:
        a = item.find('a')
        if not a or not a.has_attr('href'):
            continue
        img = item.find('img')
        if not img or not img.has_attr('src'):
            continue

        painting_url = 'https://www.wikiart.org' + a['href']
        img_url = img['src'].split('!')[0]
        title = img.get('title', 'Untitled')
        filename = save_image(img_url, IMG_DIR)

        paintings.append({
            'title': title,
            'url': painting_url,
            'img_filename': filename,
        })

    return paintings

def main():
    driver = setup_driver()
    driver.get(BASE_URL)
    time.sleep(5)

    print("Scrolling to load all paintings...")
    scroll_to_load_all(driver)
    print("Finished scrolling.")

    paintings = scrape_paintings(driver)
    driver.quit()

    if not os.path.exists(IMG_DIR):
        os.makedirs(IMG_DIR)

    json_path = os.path.join(IMG_DIR, f"{ARTIST}_metadata.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(paintings, f, indent=2, ensure_ascii=False)

    print(f"Saved metadata for {len(paintings)} paintings to {json_path}")

if __name__ == '__main__':
    main()
