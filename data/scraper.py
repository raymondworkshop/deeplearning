"""
A web scraper that queries Amazon
"""
from bs4 import BeautifulSoup
import requests
import json
import re
# from lxml import html


AMAZON_URL = "https://www.amazon.com/s/?keywords=laptop"
AMAZON_DP = "https://www.amazon.com/dp/"
headers = {
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


def get_results(asin):
        # Added Retrying
                        # This script has only been tested with Amazon.com
    _dict = {}
    url = 'http://www.amazon.com/dp/' + asin
    # Add some recent user agent to prevent amazon from blocking the request
    # Find some chrome user agent strings  here https://udger.com/resources/ua-list/browser-detail?browser=Chrome
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.90 Safari/537.36'}
    r = requests.get(url, headers=headers)
    # page_response = r.content
    soup = BeautifulSoup(r.content, 'html5lib')
    # reviews, technical specifications, price, brand
    # title
    # price - id "style_name_0_price"
    # product_title = soup.title.string
    # product_price = soup.find(id="style_name_0_price").text
    _dict["title"] = soup.title.string
    _product_price = soup.find(id="style_name_0_price")

    import pdb
    pdb.set_trace()
    if _product_price:
        _dict["price"] = _product_price.text

    # product_date = soup.find(id="averageCustomerReviews").text

    _tech_details = {}
    _tech_details_1 = soup.find(
        id="productDetails_techSpec_section_1")
    if _tech_details_1:
        tech_details_1 = _tech_details_1.find_all("tr")
        for _tr in tech_details_1:
            _th = _tr.th.get_text()
            _td = _tr.td.get_text()
            _tech_details[_th] = _td

    _tech_details_2 = soup.find(
        id="productDetails_techSpec_section_2")
    if _tech_details_2:
        tech_details_2 = _tech_details_2.find_all("tr")
        for _tr in tech_details_2:
            _th = _tr.th.get_text()
            _td = _tr.td.get_text()
            _tech_details[_th] = _td

    _dict["tech"] = _tech_details

    _reviews = soup.find_all(
        'div', {'class': "a-expander-content a-expander-partial-collapse-content"})
    """
	reviews = []
    for item in _reviews:
	    reviews.append(item.get_text().encode('utf-8'))
	"""

    _dict["reviews"] = str(_reviews)
    # if not reviews:
    #	reviews = parser.xpath(XPATH_REVIEW_SECTION_2)

    return _dict


def main():
    """
    # get all asins of some product
    ASINS =  []

    # scrape all the pages
    # total = 0
    page = 1 # by default
    pages = 20
    # while True:
    while page < pages:
        url = AMAZON_URL + "&page=" + str(page)
        print("Scraping page %d url: %s " % (page, url))
        page = page + 1

        r = requests.get(url,headers = headers)
        results = get_asins(r.content)

        # store
        ASINS = ASINS + results
        print(ASINS)
    """

    # get the infor based on ASINS
    # reviews, technical specifications, price, brand
    # title
    # price - id "style_name_0_price"
    ASINS = ["B07C7XY7GS", "B07C8J1NT"]
    f = open('amazon.json', 'a')
    # f = open('amazon.json', 'a')
    _ret = {}
    for asin in ASINS:
        _ret = get_results(asin)
        print(_ret)
        f.write(json.dumps(_ret))

    f.close()

    # print(_ret)
    # print(get_results)
    return 0


if __name__ == "__main__":
    main()
