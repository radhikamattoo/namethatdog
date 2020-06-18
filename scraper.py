# Code modified from:
# https://gist.github.com/genekogan/ebd77196e4bf0705db51f86431099e57#gistcomment-2389793

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import json
import os
import time
import urllib
import argparse
import requests

DRIVER_PATH = "/Users/mattoor/Downloads/chromedriver"
MAX_IMAGES = 100
# 80-20 training-val split per dog breed
MAX_TRAINING_IMAGES = int(MAX_IMAGES*.8)
MAX_VALIDATION_IMAGES = MAX_IMAGES - MAX_TRAINING_IMAGES
MAX_TEST_IMAGES = 1

def fetch_image_urls(query, max_links_to_fetch, wd, sleep_between_interactions=1):
    def scroll_to_end(wd):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(sleep_between_interactions)

    # build the google query
    search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

    # load the page
    wd.get(search_url.format(q=query))

    image_urls = set()
    image_count = 0
    results_start = 0
    while image_count < max_links_to_fetch:
        scroll_to_end(wd)

        # get all image thumbnail results
        thumbnail_results = wd.find_elements_by_css_selector("img.Q4LuWd")
        number_results = len(thumbnail_results)

        print(f"Found: {number_results} search results. Extracting links from {results_start}:{number_results}")

        for img in thumbnail_results[results_start:number_results]:
            # try to click every thumbnail such that we can get the real image behind it
            try:
                img.click()
                time.sleep(sleep_between_interactions)
            except Exception:
                continue

            # extract image urls
            actual_images = wd.find_elements_by_css_selector('img.n3VNCb')
            for actual_image in actual_images:
                if actual_image.get_attribute('src') and 'http' in actual_image.get_attribute('src'):
                    image_urls.add(actual_image.get_attribute('src'))

            image_count = len(image_urls)

            if len(image_urls) >= max_links_to_fetch:
                print(f"Found: {len(image_urls)} image links, done!")
                break
        else:
            print("Found:", len(image_urls), "image links, looking for more ...")
            time.sleep(30)
            return
            load_more_button = wd.find_element_by_css_selector(".mye4qd")
            if load_more_button:
                wd.execute_script("document.querySelector('.mye4qd').click();")

        # move the result startpoint further down
        results_start = len(thumbnail_results)

    return image_urls

def persist_image(folder_path,url):
    try:
        image_content = requests.get(url).content

    except Exception as e:
        print(f"ERROR - Could not download {url} - {e}")

    try:
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file).convert('RGB')
        file_path = os.path.join(folder_path,hashlib.sha1(image_content).hexdigest()[:10] + '.jpg')
        with open(file_path, 'wb') as f:
            image.save(f, "JPEG", quality=85)
        print(f"SUCCESS - saved {url} - as {file_path}")
    except Exception as e:
        print(f"ERROR - Could not save {url} - {e}")

def search_and_download(search_term):
    folder_name = '_'.join(search_term.lower().split(' '))
    print("Scraping images for {}".format(folder_name))

    training_dir = os.path.join('data', 'train', folder_name)
    validation_dir = os.path.join('data','val', folder_name)
    test_dir = os.path.join('data','test', folder_name)
    if not os.path.isdir(training_dir):
        os.mkdir(training_dir)
    if not os.path.isdir(validation_dir):
        os.mkdir(validation_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    wd = webdriver.Chrome(executable_path=DRIVER_PATH)
    res = fetch_image_urls(search_term, MAX_IMAGES + MAX_TEST_IMAGES, wd=wd, sleep_between_interactions=0.5)
    wd.close()

    image_count = 0
    download_dir = training_dir
    for elem in res:
        persist_image(download_dir,elem)
        image_count += 1
        if image_count == MAX_TRAINING_IMAGES:
            download_dir = validation_dir
        elif image_count == MAX_IMAGES:
            download_dir = test_dir

def get_data():
    with open('data/dogs.json') as f:
        breednames = json.load(f)["dogs"]
        num_breeds = len(breednames)
        print("Downloading images for {} breeds".format(num_breeds))
        for breedname in breednames:
            breedname = breedname.replace(" Dog", "").lower()
            search_and_download(breedname)




if __name__ == '__main__':
    get_data()
