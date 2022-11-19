import io
import os
import time
import argparse
import requests

from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support import expected_conditions as EC


def scrape_artwork(args):
    chromedriver, url, save_path = \
    args.chromedriver, args.url, args.save_path

    img_urls = set()

    wd = webdriver.Chrome(chromedriver)

    wd.get(url)

    num = len(wd.find_elements(By.CLASS_NAME, "ent-name"))

    for i in range(num):
        time.sleep(s)

        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(s)

        wd.execute_script("window.scrollTo(0, 0);")
        time.sleep(s)

        links = wd.find_elements(By.CLASS_NAME, "ent-name")

        wd.get(links[i].get_attribute('href'))
        time.sleep(s)

        try:
            wd.find_element_by_partial_link_text("Additional artwork").click()
            time.sleep(s)

            imgs = wd.find_elements(By.XPATH,"//img[contains(@class,'')]")

            for img in imgs:
                if img.get_attribute('src') and 'http' in img.get_attribute('src'):
                    img_urls.add(img.get_attribute('src'))

            wd.back()
            wd.back()

        except NoSuchElementException:
            imgs = wd.find_elements(By.XPATH,"//img[contains(@class,'')]")

            for img in imgs:
                if img.get_attribute('src') and 'http' in img.get_attribute('src'):
                    img_urls.add(img.get_attribute('src'))

            wd.back()

    wd.quit()

    if not os.path.isdir(save_path):
        os.mkdir(os.path.join(save_path))

    for _, url in enumerate(img_urls):
        image_content = requests.get(url).content
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file)
        file_path = os.path.join(save_path, os.path.basename(os.path.normpath(url))[:-4] + ".png")

        if os.path.isfile(file_path):
            with open(file_path[:-4] + '_2.jpg', "wb") as f:
                image.save(f, "PNG")
        else:
            with open(file_path, "wb") as f:
                image.save(f, "PNG")
    
    os.remove("{save_path}\\header-sm.jpg")

    print(f"Total of {len(img_urls)} saved to {save_path}/")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chromedriver', type=str, default='chromedriver.exe', help='path to chromedriver.exe')
    parser.add_argument('--url', type=str, default='https://pokemondb.net/pokedex/national', help='Pokemon Database URL')
    parser.add_argument('--save_path', type=str, default='artwork', help='folder to save images')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    scrape_artwork(args)