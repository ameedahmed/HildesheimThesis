from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
from selenium import webdriver
import os
from pathlib import Path
import time
import regex as re
from itertools import repeat
from multiprocessing import Pool
from tqdm import tqdm
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, NoSuchElementException

def get_art_urls_from_page(art_html):
    """
    input: html of page that contains all art titles
    output: list of beautiful soup objects containing all art titles on the page
    """
    soup = BeautifulSoup(art_html, 'html.parser')
    
    # Find all <a> tags within <li class="painting-list-text-row">
    art_links = soup.select('li.painting-list-text-row a')
    
    print(f'\nNumber of art titles: {len(art_links)}')
    return art_links


def go_to_art_page(website_url, title_block):
    """
    input: block soup object containing details of art details
    output: return art bs4 object
    """
    art_address = title_block.find('a', 'artwork-name ng-binding')
    art_url = urljoin(website_url, art_address['href'])
    art_page_html = requests.get(art_url).text

    return art_page_html


def clean_string(string):
    # remove filename illegal chars
    string = re.sub("[?():/]", "", string)
    return string.strip().replace('"', '').lower()


def get_art_details_from_page(art_page_html):
    """
    input: soup object of art page
    output: dictionary of details of art from page
    """
    art_page_soup = BeautifulSoup(art_page_html, 'html.parser')
    # access block that contains art metadata
    art_article = art_page_soup.find_all('article')
    if len(art_article) == 0:
        return {}
    else:
        art_article = art_page_soup.find_all('article')[0]

    art_title_name = art_article.h3.text
    art_author_name = art_article.h5.text

    style_genre_container = art_article.find_all('li', 'dictionary-values')
    art_style = style_genre_container[0]
    art_style_name = art_style.find('a').text    
    genre_container = style_genre_container[1]
    genre_tag = genre_container.find("span", {"itemprop": "genre"})
    genre_name = genre_tag.text if genre_tag else None
    media_container = style_genre_container[2]
    art_medium_name = media_container.find_all("a")
    media_name_list = [a.text.strip() for a in art_medium_name if a.text]

    image_soups = art_page_soup.find_all('img')
    art_image_url = image_soups[0]['src']
    #Make sure art_image_url iterates. It is not iterating at the moment
    return {'title': clean_string(art_title_name),
            'artist': clean_string(art_author_name),
            'style': clean_string(art_style_name),
            'image_url': art_image_url}


def get_art_page_url(website_url, title_block):
    """
    input: block soup object containing details of art details
    output: return art bs4 object
    """
    #art_address = title_block.find('a', 'artwork-name ng-binding')
    art_url = urljoin(website_url, title_block['href'])
    return art_url



def parallel_get_art_details(art_url, artist_name, write_folder):
    art_page = requests.get(art_url).text
    art_metadata = get_art_details_from_page(art_page)
    if not art_metadata:
        return

    csv_filepath = write_folder/f'{artist_name}.csv'
    url = art_metadata['image_url']

    response = requests.get(url, stream=True)
    ext = response.headers['content-type'].split('/')[-1]

    write_sub_folder = write_folder/f"{artist_name}"

    if not write_sub_folder.exists():
        write_sub_folder.mkdir()

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
            f.writelines('artist,title,style,image_path\n')
            f.writelines(
                f"{art_metadata['artist']},{art_metadata['title']},{art_metadata['style']},{str(image_path)}\n"
            )
    else:
        with open(csv_filepath, 'a', encoding="utf-8") as f:
            f.writelines(
                f"{art_metadata['artist']},{art_metadata['title']},{art_metadata['style']},{str(image_path)}\n"
            )


if __name__ == '__main__':
    # destination
    write_folder = Path(os.path.join(os.getcwd(), 'van_gogh_data'))

    if not write_folder.exists():
        write_folder.mkdir()

    # crawl source
    website_url = 'https://www.wikiart.org/'
    artist_name = 'vincent-van-gogh'
    
    # go directly to Van Gogh's works page
    artist_works_url = urljoin(website_url, f'en/{artist_name}/all-works#!#filterName:all-paintings-chronologically,resultType:text')
    
    options = Options()
    options.add_argument("--headless")  # Run in background
    driver = webdriver.Chrome(options=options)
    
    driver = webdriver.Chrome()
    driver.minimize_window()
    driver.get(artist_works_url)
    
            # Wait for the load more button to be visible and clickable
    for i in range(1000):
        try:
            load_more_btn = WebDriverWait(driver, 2).until(
            EC.element_to_be_clickable((By.CLASS_NAME, "masonry-load-more-button"))
        )
            print("Clicking LOAD MORE...")
            load_more_btn.click()
        except Exception as e:    
            try:
                close_btn = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.ID, "close-popup"))
                )
                close_btn.click()
                print("Geo popup closed.")
                break
            except:
                print("Geo popup did not appear or was already closed.") 
#        WebDriverWait(driver, 10).until(
#        lambda d: d.find_element(By.CLASS_NAME, "count").text != 1932
 #   )
#        prev_count_text = driver.find_element(By.CLASS_NAME, "count").text
                

    artist_works_page = driver.page_source
    
    art_title_blocks = get_art_urls_from_page(artist_works_page)

    art_urls = []
    num_repeats = len(art_title_blocks)

    for website_url, title_block in zip(repeat(website_url, num_repeats), art_title_blocks):
        art_urls.append(get_art_page_url(website_url, title_block))
        
    num_iters = len(art_urls)
    arguments = zip(art_urls,
                    repeat(artist_name, num_iters),
                    repeat(write_folder, num_iters))

    with Pool() as p:
        p.starmap(parallel_get_art_details, arguments)
        
    driver.quit()