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

AMAZON = "https://www.amazon.com"

AMAZON_URL = "https://www.amazon.com/s/?keywords=laptop"
AMAZON_DP = "https://www.amazon.com/dp/"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.90 Safari/537.36'}


def is_result_div(tag):
    """ Returns true for divs with ids starting with 'result_'. """
    if not tag.name == "div":
        return False
    # id="result_17"
    return tag.has_key("id") and str(tag["id"]).startswith("result_")


def get_asins(html):
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
    asins = soup.findAll("div", attrs={"data-asin": re.compile(r".*")})
    if len(asins) == 0:
        asins = soup.findAll("li", attrs={"data-asin": re.compile(r".*")})

    lst_asins = []
    i = 0
    while i < len(list(asins)):
        print(asins[i])
        # asin

        lst_asins.append(asins[i]["data-asin"])
        i = i + 1

    return lst_asins


def num_there(s):
    return re.search(r'[2]\d{3}', s)


def get_reviews(html, cnt):
    pages = round(cnt / 10)
    _reviews = []
    page = 1

    #import pdb
    # pdb.set_trace()
    #_reviews = []
    while page < pages:
        url = html + str(page)
        page = page + 1
        print("The review page %d url: %s" % (page, url))
        r = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(r.content, 'html5lib')

        # TODO: get reviews
        _reviews_div = soup.find_all(
            'div', {'class': "a-row a-spacing-medium review-data"})
        #_reviews = []
        for item in _reviews_div:
            _txt = item.get_text().lstrip().rstrip().rstrip("\n").lstrip("\n")
            _txt_list = _txt.encode('utf-8').split()
            if len(_txt_list) > 5:  # bad option
                _reviews.append(_txt_list)

    return _reviews


def get_results(asin):
    # Added Retrying
    _dict = {}
    url = 'http://www.amazon.com/dp/' + asin
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
    _dict["asin"] = asin
    title = soup.title.string
    """
    if "Robot Check" in title:
        time.sleep(60)
        title = soup.title.string
        if "Robot Check" in title:
            _dict["title"] = ""
        else:
            _dict["title"] = soup.title.string
    else:
        _dict["title"] = soup.title.string
    """
    if title:
        _dict["title"] = title

    product_price = soup.find(id="style_name_0_price")
    if product_price:
        _dict["price"] = product_price.text.rstrip('\n').lstrip("\n").rstrip().rstrip()
    else:
        _dict["price"] = ""

        # product_date = soup.find(id="averageCustomerReviews").text
    #
    #_add_info =  soup.find_all('tr.th', {'class': "a-color-secondary a-size-base prodDetSectionEntry"})
    _add_info = soup.find(id="productDetails_detailBullets_sections1")
    if _add_info:
        rows = _add_info.findChildren(['td'])
        for row in rows:
            _td = str(row.get_text().rstrip("\n").lstrip("\n"))
            if num_there(_td):
                #_td = _tr.td.get_text().lstrip().rstrip().rstrip("\n").lstrip("\n")
                _dict["date"] = _td
    else:
        _dict["date"] = ""

    _tech_details = {}
    _tech_details_1 = soup.find(id="productDetails_techSpec_section_1")
    if _tech_details_1:
        tech_details_1 = _tech_details_1.find_all("tr")
        for _tr in tech_details_1:
            _th = _tr.th.get_text()
            _td = _tr.td.get_text()
            _tech_details[_th] = _td.lstrip().rstrip().rstrip("\n").lstrip("\n")

    _tech_details_2 = soup.find(id="productDetails_techSpec_section_2")
    if _tech_details_2:
        tech_details_2 = _tech_details_2.find_all("tr")
        for _tr in tech_details_2:
            _th = _tr.th.get_text()
            _td = _tr.td.get_text()
            _tech_details[_th] = _td.lstrip().rstrip().rstrip("\n").lstrip("\n")

    if _tech_details:
        _dict["tech"] = _tech_details
    else:
        _dict["tech"] = ""

    #_reviews = soup.find(id="customerReviews")
    reviews = []
    _reviews_div = soup.find('div', attrs={'class': 'a-row a-spacing-large'})
    _reviews_cnt = 0
    _reviews_href = ""
    #_reviews_div = '<div class="a-row a-spacing-large"><a class="a-link-emphasis a-text-bold" data-hook="see-all-reviews-link-foot" href="/HP-17-3-Laptop-Intel-Memory/product-reviews/B06X1869F3/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&amp;reviewerType=all_reviews">See all 129 reviews</a></div>'
    if _reviews_div:
        try:
            for a in _reviews_div.find('a', href=True, text=True):
                _reviews_href = a['href']
            #_reviews_txt = a.txt
                _reviews_txt = re.search(r'\d+', a.txt)
                _reviews_cnt = int(_reviews_txt.group())
        #reviews_href = "/".join(str(x) for x in _reviews_href)
        except TypeError:
            pass

    if _reviews_href:
        _reviews_html = AMAZON + _reviews_href + "&pageNumber="

    #import pdb
    # pdb.set_trace()
    if _reviews_cnt > 1 : #assume the review is more than 1 word
        _dict["reviews"] = get_reviews(_reviews_html, _reviews_cnt)
    else:
        _reviews_txt = soup.find_all(
            'div', {'class': "a-expander-content a-expander-partial-collapse-content"})
    #reviews = []
        for item in _reviews_txt:

            #import pdb
            # pdb.set_trace()
            _txt = item.get_text().lstrip().rstrip().rstrip("\n").lstrip("\n")
            _txt_list = _txt.encode('utf-8').split()
            if len(_txt_list) > 1:  # assume the review is more than 1 word
                reviews.append(_txt_list)

        _dict["reviews"] = reviews
    # if not reviews:
    #	reviews = parser.xpath(XPATH_REVIEW_SECTION_2)

    return _dict


def main():
    """
    # get all asins of some product
    ASINS = []

    # scrape all the pages
    # total = 0
    page = 1  # by default
    pages = 400
    # while True:
    while page < pages:
        url = AMAZON_URL + "&page=" + str(page)
        print("Scraping page %d url: %s " % (page, url))
        page = page + 1

        r = requests.get(url, headers=HEADERS)
        results = get_asins(r.content)

        # store
        ASINS = ASINS + results
        print(ASINS)

    # write the file
    with open("amazon_asin_0627.md", 'a') as f:
        f.write(str(tuple(ASINS)))
    
    """
    # get the infor based on ASINS
    # reviews, technical specifications, price, brand
    # title
    # price - id "style_name_0_price"

    #ASINS = ["B017XR0XWC"]
    # f = open()
    """
    f1 = open("amazon_asin_0627.md", 'r')
    ASINS = eval(f1.readline())

    with open("amazon_asin_0628.md", 'a') as f:
        f.write(str(set(ASINS)))
    """

    ASINS = ["B06X1869F3"]
    #ASINS = ["B07C8BJ1NT","B01JJQVNLK","B078KNND2S", "B005OSFT90", "B01AP5AJFA","B01AP5AJFA", "ACSVBGNA01", "B06WWKYM1X"]


    f = open('amazon_update_0628_latest_.json', 'a', encoding="utf-8")
    _ret = {}
    for asin in ASINS:
        _ret = get_results(asin)
        print(_ret)
        f.write(str(_ret) + "\n")

    f.close()
   
    # print(_ret)
    # print(get_results)
    return 0


if __name__ == "__main__":
    main()
