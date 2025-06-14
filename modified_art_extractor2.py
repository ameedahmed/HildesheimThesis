from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import os
from pathlib import Path
import regex as re
from itertools import repeat
from multiprocessing import Pool
from tqdm import tqdm
import time
from selenium.webdriver.common.keys import Keys
import pandas as pd
import csv
from selenium.webdriver.common.action_chains import ActionChains
import time


def get_art_urls_from_page(art_html):
    soup = BeautifulSoup(art_html, 'html.parser')
    art_links = soup.select('div.title-block a')
    print(f'\nNumber of art titles: {len(art_links)}')
    art_urls = [a['href'] for a in art_links if 'href' in a.attrs]
    return art_urls


def go_to_art_page(website_url, title_block):
    art_address = title_block.find('a', 'artwork-name ng-binding')
    art_url = urljoin(website_url, art_address['href'])
    art_page_html = requests.get(art_url).text
    return art_page_html


def clean_string(string):
    string = re.sub("[?():/]", "", string)
    return string.strip().replace('"', '').lower()


def get_art_details_from_page(driver, art_url, style_name):

    driver.get(art_url)
    actions = ActionChains(driver)
    # Try closing the initial popup if it appears
    try:
        driver.execute_script("document.body.style.zoom='80%'")
        time.sleep(0.1)
        WebDriverWait(driver, 15).until(
            EC.element_to_be_clickable((By.ID, "close-popup"))
        ).click()
        driver.execute_script("document.body.style.zoom='90%'")
        print(f"1st stage {i} out of 7")
    except:
        print("Popup did not appear – nothing to close")

        # Wait until the <article> tag is present
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "article"))
        )
    except:
        print("[ERROR] Article not found.")
        return {}

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    art_article = soup.find('article')
    if not art_article:
        return {}

    # Extract metadata
    art_title_name = art_article.h3.text if art_article.h3 else 'unknown'

    art_author_name = art_article.h5.text if art_article.h5 else 'unknown'

    style_genre_container = art_article.find_all('li', class_='dictionary-values')
    try:
        art_style_name = style_genre_container[0].find('a').text.strip()
    except:
        art_style_name = 'unknown'
    try:
        genre_tag = style_genre_container[1].find("span", {"itemprop": "genre"})
        genre_name = genre_tag.text.strip() if genre_tag else 'unknown'
    except:
        genre_name = 'unknown'
    try:
        media_links = style_genre_container[2].find_all("a")
        media_name_list = [a.text.strip() for a in media_links if a.text]
    except:
        media_name_list = []
    
    # Handle view all sizes and possible overlay popups
    try:
        # Hide EU overlay if it exists
        for i in range(4):
            try:
                driver.execute_script("document.body.style.zoom='80%'")
                time.sleep(0.2)
                WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.ID, "close-popup"))
                ).click()
                print(f"2nd stage {i} out of 7")
                driver.execute_script("document.body.style.zoom='70%'")
            except:
                print(f"Popup did not appear for 2nd time in {art_title_name}– nothing to close")

        # Click "View all sizes"
        view_all_sizes_btn = WebDriverWait(driver, 4).until(
            EC.element_to_be_clickable((By.CLASS_NAME, "all-sizes"))
        )
        ActionChains(driver).move_to_element(view_all_sizes_btn).click().perform()

        # Wait for image URLs to appear
        WebDriverWait(driver, 4).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "a[href^='https://uploads']"))
        )

    except Exception as e:
        print("[ERROR] Failed to click 'View all sizes':{art_title_name}", e)
        return {}

    # Parse new page source for upload links
    soup2 = BeautifulSoup(driver.page_source, "html.parser")
    upload_links = [
        a['href'] for a in soup2.find_all('a', href=True)
        if a['href'].startswith("https://uploads")
    ]

    if not upload_links:
        print("[ERROR] No image URLs found.")
        return {}

    return {
        'title': clean_string(art_title_name),
        'artist': clean_string(art_author_name),
        'style': clean_string(art_style_name),
        'genre': clean_string(genre_name),
        'media': media_name_list,
        'image_url': upload_links[-1]
    }
    
def get_art_page_url(website_url, title_block):
    art_url = urljoin(website_url, title_block)
    return art_url


def parallel_get_art_details(art_url, style_name, write_folder):
    # Setup headless driver inside worker process
    try:
        # Path to Brave browser (adjust as per your OS and installation)
        brave_path = "C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe"  # Windows

        options = Options()
        options.binary_location = brave_path
        driver = webdriver.Chrome(options=options)
        driver.minimize_window()
        
        art_metadata = get_art_details_from_page(driver, art_url,style_name)


        csv_filepath = write_folder/f'{style_name}.csv'
        url = art_metadata['image_url']
        response = requests.get(url, stream=True)
        ext = response.headers['content-type'].split('/')[-1]

        write_sub_folder = write_folder/f"{style_name}"
        write_sub_folder.mkdir(exist_ok=True)

        if len(art_metadata['title']) >= 150:
            fname = art_metadata['title'][:150] + ('.' + ext)
        else:
            fname = art_metadata['title'] + ('.' + ext)

        image_path = write_sub_folder/fname
        image_path = image_path.relative_to(os.getcwd())

        with open(image_path, 'wb') as f:
            f.write(response.content)

        if not csv_filepath.exists():
            with open(csv_filepath, 'w', encoding="utf-8") as f:
                with open(csv_filepath, "w", newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["artist", "title", "style", "image_path"])  # Header
                    writer.writerow([
                        art_metadata['artist'],
                        art_metadata['title'],
                        art_metadata['style'],
                        str(image_path)
                    ])
        else:
            with open(csv_filepath, 'a', encoding="utf-8") as f:
                f.writelines(
                    f"{art_metadata['artist']},{art_metadata['title']},{art_metadata['style']},{str(image_path)}\n"
                )
    except Exception as e:
        print(f"[ERROR] Failed to get art details for {art_url}: {e}")

    finally:
        driver.quit()


if __name__ == '__main__':
    write_folder = Path(os.path.join(os.getcwd(), 'realism_data'))
    write_folder.mkdir(exist_ok=True)

    website_url = 'https://www.wikiart.org/'
    style_name = 'Academicism'
    artist_works_url = urljoin(website_url, f'en/paintings-by-style/{style_name}?select=al#!#filterName:featured,viewType:masonry')

    
    # Path to Brave browser (adjust as per your OS and installation)
    brave_path = "C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe"  # Windows

    options = Options()
    options.binary_location = brave_path
    driver = webdriver.Chrome(options=options)
        
    driver.minimize_window()
    driver.set_page_load_timeout(3600)
    driver.get(artist_works_url)
            # Wait for the button to be present and clickable
    wait = WebDriverWait(driver, 3)

    # Keep clicking 'Load More' until it disappears
    while True:
        try:
            start_time = time.time()

            # Wait for and click the 'Load More' button
            load_more_button = wait.until(
                EC.element_to_be_clickable((By.CLASS_NAME, "masonry-load-more-button"))
            )
            load_more_button.click()
            print("Clicked 'Load More' button, waiting for new content to load...")

            time.sleep(1)  # Let new content load

            # Attempt to close popup if it appears
            try:
                driver.execute_script("document.body.style.zoom='80%'")
                time.sleep(0.1)
                WebDriverWait(driver, 6).until(
                    EC.element_to_be_clickable((By.ID, "close-popup"))
                ).click()
                driver.execute_script("document.body.style.zoom='90%'")
                print("Popup closed successfully")
            except TimeoutException:
                print("Popup did not appear – nothing to close")

            print("--- %s seconds ---" % (time.time() - start_time))

        except TimeoutException:
            print("No more 'Load More' button found or it is not clickable.")
            break


    artist_works_page = driver.page_source
    driver.quit()

    art_title_blocks = get_art_urls_from_page(artist_works_page)
    art_urls = []
    for title_block in art_title_blocks:
        art_urls.append(get_art_page_url(website_url, title_block))

    num_iters = len(art_urls)
    arguments = zip(art_urls, repeat(style_name, num_iters), repeat(write_folder, num_iters))

    with Pool() as p:
        p.starmap(parallel_get_art_details, arguments)
