import io
import os
import time
import argparse
import requests

from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By


def scrape_sprites(args):
    chromedriver, url, save_path = \
        args.chromedriver, args.url, args.save_path

    img_urls = set()
    gens = ["gen1", "gen2", "gen3", "gen4", "gen5", "gen6", "gen7", "gen8"]

    wd = webdriver.Chrome(chromedriver)

    wd.get(url)
    print("\nScraping Pokemon pixel art; this may take a few minutes...")
    print("\n   DO NOT CLOSE, MINIMIZE, OR SCROLL ON THE CHROME TAB")

    ele = wd.find_element(By.ID, gens[0])
    wd.execute_script("arguments[0].scrollIntoView();", ele)

    for gen in gens[1:]:
        thumbnails = wd.find_elements(By.CLASS_NAME, "icon-pkmn")

        for img in thumbnails:
            if img.get_attribute('src') and 'http' in img.get_attribute('src'):
                img_urls.add(img.get_attribute('src'))

        ele = wd.find_element(By.ID, gen)
        wd.execute_script("arguments[0].scrollIntoView();", ele)
        time.sleep(1)

    wd.quit()

    if not os.path.isdir(save_path):
        os.mkdir(os.path.join(save_path))

    for _, url in enumerate(img_urls):
        image_content = requests.get(url).content
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file)
        file_path = os.path.join(save_path, os.path.basename(os.path.normpath(url))[:-4] + ".png")

        with open(file_path, "wb") as f:
            image.save(f, "PNG")

    os.remove("images\\s.png")

    if len(os.listdir('images')) == 898:
        print("\nCongratulations, all images saved!")
    elif len(os.listdir('images')) < 898:
        print("Sorry, some images did not save for some reason...")
    else:
        print("Sorry, extra images were saved somehow...")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chromedriver', type=str, default='chromedriver.exe', help='path to chromedriver.exe')
    parser.add_argument('--url', type=str, default='https://pokemondb.net/sprites', help='Pokemon Database URL')
    parser.add_argument('--save_path', type=str, default='sprites', help='folder to save images')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    scrape_sprites(args)