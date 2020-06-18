# Code modified from:
# https://gist.github.com/genekogan/ebd77196e4bf0705db51f86431099e57#gistcomment-2389793

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import json
import os
import urllib2
import argparse
import boto3

MAX_IMAGES = 250
bucket_name = 'namethatdog'
s3 = boto3.resource('s3')

def upload_data():
    bucket = s3.Bucket(bucket_name)
    with open('data/dogs.json') as f:
        breednames = json.load(f)["dogs"]
        for breedname in breednames:
            successounter = 0
            dirname = breedname.replace(' ', '-')
            searchterm = breedname.replace(' ', '+')

            url = "https://www.google.co.in/search?q="+searchterm+"&source=lnms&tbm=isch"
            browser = webdriver.Chrome("/Users/mattoor/Downloads/chromedriver")
            browser.get(url)
            header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}

            for _ in range(500):
                browser.execute_script("window.scrollBy(0,10000)")

            for x in browser.find_elements_by_xpath('//div[contains(@class,"rg_meta")]'):
                print "Count:", successounter
                print "URL:",json.loads(x.get_attribute('innerHTML'))["ou"]

                img = json.loads(x.get_attribute('innerHTML'))["ou"]
                imgtype = json.loads(x.get_attribute('innerHTML'))["ity"]
                try:
                    req = urllib2.Request(img, headers={'User-Agent': header})
                    raw_img = urllib2.urlopen(req).read()
                    # TODO: SPLIT INTO TRAIN/VALIDATION DIRECTORIES FOR EASIER SETUP
                    filename = breedname.replace(' ', '') + '/' + str(successounter)
                    bucket.put_object(Key=filename, Body=raw_img)
                    successounter = successounter + 1
                except:
                        print "Can't get img"
                finally:
                    if successounter == MAX_IMAGES:
                        print "Downloaded 200 images, moving on to next breed..."
                        break

            print successounter, "pictures succesfully downloaded"
            browser.close()

if __name__ == '__main__':
    with open('data/dogs.json') as f:
        breednames = json.load(f)["dogs"]
        for breedname in breednames:
            breedname = breedname.title()
            if " Dog" in breedname:
                breedname = breedname.replace(" Dog", "")
