"""
A web scraper that queries Amazon
"""
from bs4 import BeautifulSoup
import requests
import json
import re
# from lxml import html
# import pickle
import time

"""
AMAZON = "https://www.amazon.com"

AMAZON_URL = "https://www.amazon.com/s/?keywords=laptop"
AMAZON_DP = "https://www.amazon.com/dp/"
"""

flipkart = "https://www.flipkart.com"
flipkart_url = "https://www.flipkart.com/search?q=laptops&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=off&as=off&sort=popularity"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.90 Safari/537.36'}


def is_result_div(tag):
    """ Returns true for divs with ids starting with 'result_'. """
    if not tag.name == "div":
        return False
    # id="result_17"
    return tag.has_key("id") and str(tag["id"]).startswith("result_")


def get_hrefs(html):
    # keep the values in list
    # data-asin list
    # results = []

    #
    soup = BeautifulSoup(html, 'html5lib')
    """
    result_divs = soup.find_all(is_result_div)
    for result_div in is_result_divs:
        asin = result_div.find('li')
    """
    #_refs_div = soup.find_all(
    #        'div', {'class': "_31qSD5"})

    hrefs = [] 
    for link in soup.findAll('a', attrs={'class': "_31qSD5"}):
        try:
            href = link['href']
            print("The href: %s" % href)
            hrefs.append(href)
        except KeyError:
            pass

    return hrefs


def num_there(s):
    return re.search(r'[2]\d{3}', s)


def get_reviews(html, cnt):
    pages = round(cnt / 10) + 1 # bug
    _reviews = []
    page = 1

    #import pdb
    # pdb.set_trace()
    #_reviews = []
    
    while page <= pages:
        url = html + "&pageNumber=" +  str(page)
       
        print("The review page %d url: %s" % (page, url))
        page = page + 1

        r = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(r.content, 'html5lib')

        # TODO: get reviews - qwjRop
        _reviews_div = soup.find_all(
            'div', {'class': "qwjRop"})
        #_reviews = []
        for item in _reviews_div:
            _txt = item.get_text().lstrip().rstrip().rstrip("\n").lstrip("\n")
            _txt_list = _txt.encode('utf-8').split()
            if len(_txt_list) > 1:  # bad option
                _reviews.append(_txt_list)

    return _reviews


def get_results(href):

    # Added Retrying
    _dict = {}
    #url = 'http://www.amazon.com/dp/' + asin
    url = flipkart + href
    # headers = {
    #    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.90 Safari/537.36'}
    r = requests.get(url, headers=HEADERS)
    # page_response = r.content
    soup = BeautifulSoup(r.content, 'html5lib')
    # reviews, technical specifications, price, brand
    # title
    # price - id "style_name_0_price"
    # product_title = soup.title.string
    # #product_price = soup.find(id="style_name_0_price").text
    #_dict["asin"] = asin

    
    #HP 15q Pentium Quad Core - (4 GB/1 TB HDD/DOS) 15q-ds0004TU Laptop  (15.6 inch, Sparkling Black, 1.77 kg)
    #lst = parameters.split()
    #_dict['tech'] = parameters
    #title = soup.find_all('span', {'class' : '_1VpSqZ'}).get_text()
    if soup.find('span', {'class' : '_38sUEc'}):
        _reviews_cnt = soup.find('span', {'class' : '_38sUEc'}).text
    else:
        return _dict
    if _reviews_cnt and '&' in _reviews_cnt:
        _reviews_cnt_ = _reviews_cnt.split('&')[1]
        num = re.search('[\d]+', _reviews_cnt_).group()
        if num:  # reviews 
            _reviews_div = soup.find('div', attrs={'class': 'col _39LH-M'})
            if len(_reviews_div.find_all('a')) > 1:
                _reviews_href = _reviews_div.find_all('a')[-1]['href']
            else:
                return _dict
            if _reviews_href:
                # parameters firstly
                if soup.find('span', {'class' : '_35KyD6'}):
                    parameters = soup.find('span', {'class' : '_35KyD6'}).text
                    if parameters:
                        _dict["tech"] = parameters

                _reviews_html = flipkart + _reviews_href
                _dict["reviews"] = get_reviews(_reviews_html, int(num))
     
    else: # skip this one
        pass

    return _dict


def main():
    # get all asins of some product
    hrefs = []
    
    # scrape all the pages
    # total = 0
    page = 1  # by default
    pages = 28
    # while True:
    while page < pages:
        url = flipkart_url + "&page=" + str(page)
        print("Scraping page %d url: %s " % (page, url))
        page = page + 1

        r = requests.get(url, headers=HEADERS)
        #ref = get_ref(r.content)
        results = get_hrefs(r.content)

        # store
        hrefs = hrefs + results
        #print(hrefs)

    """
    # write the file
    with open("flipkart_hrefs_1005.md", 'a') as f:
        f.write(str(tuple(hrefs)))
    """

    #hrefs = "/acer-aspire-3-pentium-quad-core-4-gb-1-tb-hdd-linux-a315-32-laptop/p/itmf5q4jsr8hqcqx?pid=COMF5Q4J7UKUAXZU&srno=s_2_25&otracker=search&lid=LSTCOMF5Q4J7UKUAXZUWWZTVP&fm=organic&iid=fb582eec-28a8-44a9-9cc1-f491a51ae6a3.COMF5Q4J7UKUAXZU.SEARCH&ppt=SearchPage&ppn=Search&qH=c06ea84a1e3dc3c6"
    #hrefs = "/asus-core-i5-8th-gen-8-gb-1-tb-hdd-windows-10-home-2-gb-graphics-r542uq-dm252t-laptop/p/itmf4hcwgbcde33t?pid=COMF4GGQ43VVZNGP&srno=s_2_26&otracker=search&lid=LSTCOMF4GGQ43VVZNGPX56ZXY&fm=SEARCH&iid=131b9f30-9558-4d31-a258-8df28bdc1108.COMF4GGQ43VVZNGP.SEARCH&ppt=SearchPage&ppn=Search&ssid=dxzzqwntdc0000001538717528269&qH=c06ea84a1e3dc3c6"
    #ASIN = hrefs.split('/')[1]
    f = open('flipkart_reviews_1008.json', 'a', encoding="utf-8")

    for link in hrefs:
        _ret = {}
        #link = "/acer-aspire-3-pentium-quad-core-4-gb-1-tb-hdd-linux-a315-32-laptop/p/itmf5q4jsr8hqcqx?pid=COMF5Q4J7UKUAXZU&srno=s_2_25&otracker=search&lid=LSTCOMF5Q4J7UKUAXZUWWZTVP&fm=organic&iid=fb582eec-28a8-44a9-9cc1-f491a51ae6a3.COMF5Q4J7UKUAXZU.SEARCH&ppt=SearchPage&ppn=Search&qH=c06ea84a1e3dc3c6"
        ASIN = link.split('/')[1]
        _ret[ASIN] = get_results(link)
        print(_ret)
        f.write(str(_ret) + "\n")

    f.close()
    
    # get the infor based on ASINS
    # reviews, technical specifications, price, brand
    # title
    # price - id "style_name_0_price"

    #ASINS = ["B017XR0XWC"]
    # f = open()
    
    #f1 = open("amazon_asin_0628.md", 'r')
    #ASINS = eval(f1.readline())
    """
    with open("amazon_asin_0628.md", 'a') as f:
        f.write(str(set(ASINS)))
    """

    #ASINS = flipkart + str1
    #ASINS = ["B07C8BJ1NT","B01JJQVNLK","B078KNND2S", "B005OSFT90", "B01AP5AJFA","B01AP5AJFA", "ACSVBGNA01", "B06WWKYM1X"]

    # print(_ret)
    # print(get_results)
    return 0


if __name__ == "__main__":
    main()
