"""
A web scraper that queries Amazon

Instruction: 
    * get the asins OF the products - manually, in a list
       - do some parameter selections on the left in amazons web 
       - for example ["B01ATX3PBS", "B07G77QHV8"] 

    * 

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
    pages = round(cnt / 10) + 1 # bug
    _reviews = []
    page = 1

    #import pdb
    # pdb.set_trace()
    #_reviews = []
    while page <= pages:
        url = html + str(page)
        
        print("The review page %d url: %s" % (page, url))
        page = page + 1

        r = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(r.content, 'html5lib')

        # TODO: get reviews
        _reviews_div = soup.find_all(
            'div', {'class': "a-row a-spacing-small review-data"})
        
        """   
        _reviews_div = soup.find_all(
            'div', {'id': "cm_cr-review_list"})
        """

        #_reviews = []
        for item in _reviews_div:
            _txt = item.get_text().lstrip().rstrip().rstrip("\n").lstrip("\n")
            _txt_list = _txt.encode('utf-8').split()
            if len(_txt_list) > 1:  # bad option
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
    if "robot" in title.lower():
        print("Warning: Robot Check - Please wait")
        return 0

    if title:
        _dict["title"] = title

    product_price = soup.find(id="style_name_0_price") #priceblock_ourprice
    if product_price:
        _dict["price"] = product_price.text.rstrip(
            '\n').lstrip("\n").rstrip().rstrip()
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
                _dict["date"] = _td.lstrip().rstrip().rstrip("\n").lstrip("\n")
                
                year = re.search('\d{4}', _dict["date"]).group()
                if int(year) < 2012: # only get the ones from 2015 
                    return _dict
        
    else:
        _dict["date"] = ""

    _tech_details = {}
    _tech_details_1 = soup.find(id="productDetails_techSpec_section_1")
    if _tech_details_1:
        tech_details_1 = _tech_details_1.find_all("tr")
        for _tr in tech_details_1:
            _th = _tr.th.get_text()
            _td = _tr.td.get_text()
            _tech_details[_th] = _td.rstrip("\n").lstrip("\n").lstrip().rstrip()

    _tech_details_2 = soup.find(id="productDetails_techSpec_section_2")
    if _tech_details_2:
        tech_details_2 = _tech_details_2.find_all("tr")
        for _tr in tech_details_2:
            _th = _tr.th.get_text()
            _td = _tr.td.get_text()
            _tech_details[_th] = _td.rstrip("\n").lstrip("\n").lstrip().rstrip()

    if _tech_details:
        _dict["tech"] = _tech_details
    else:
        _dict["tech"] = ""

    #_reviews = soup.find(id="customerReviews")
    reviews = []
    _reviews_div = soup.find('div', attrs={'class': 'a-row a-spacing-large'})
    _reviews_cnt = 0
    _reviews_href = ""

    #import pdb
    #pdb.set_trace()
    if _reviews_div:
        _reviews_div_a = _reviews_div.find('a')
        if _reviews_div_a:
            try:
                a_txt = _reviews_div.a.text
                if a_txt:
                    _reviews_txt = re.search(r'\d+', a_txt)
                    if _reviews_txt:
                        _reviews_cnt = int(_reviews_txt.group())
                        #href
                        #if _reviews_div.a.find('href'):
                        _reviews_href = _reviews_div.a['href']

            except TypeError:
                pass

    if _reviews_href:
        _reviews_html = AMAZON + _reviews_href + "&pageNumber="

    #import pdb
    #pdb.set_trace()
    if _reviews_href and _reviews_cnt > 4:  # assume the review is more than 1 word
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

    #import pdb
    #pdb.set_trace()
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
    
    #f1 = open("amazon_asin_0628.md", 'r')
    #ASINS = eval(f1.readline())
    """
    with open("amazon_asin_0628.md", 'a') as f:
        f.write(str(set(ASINS)))
    """

    # laptop
    #ASINS = ["B07193JRJR"]
    #ASINS = ["B07C8BJ1NT","B01JJQVNLK","B078KNND2S", "B005OSFT90", "B01AP5AJFA","B01AP5AJFA", "ACSVBGNA01", "B06WWKYM1X"]

    LAPTOP_ASINS = ["B077X1ZB7H", "B079TGL2BZ", "B07KQXQWK1","B07L49MY9H","B07L9MM5RN",
 "B07N41YXP5", "B07Q478DHY", "B07KNLVRJ2", "B07KDQW7Q1",
"B0762S8PYM", "B075FLBJV7", "B07MGT236W", "B07NXTKWMX", "B0795W86N3",
"B07Q147J19", "B07D97S1CR", "B07K23MWKV", "B07MBR3D8C", "B07PB5M8DS",
"B07KWG73RQ", "B07FSFRWS4", "B07KWG73RQ", "B07FSFRWS4", "B0748YG81P",
"B07MY6Z3QC", "B07L28M2CB", "B07HZL1NWW", "B07BJ9T8XC", "B07KWK3N7V",
"B07FW7D635", "B077GN5H7D", "B07DMJGV9W", "B07P2JH6MS", "B07JF6HRJ1",
"B07ML5ZVZY", "B07HB744Y1", "B07QRWRPJ6", "B07PX4B36S", "B07K8T4SG1",
"B07MQBN2X1", "B078YCMD67", "B07C9J1PY6", "B07MZ8XBNR", "B01NBE6Y5D",
"B07GKZJ8CX", "B07N79NPSQ", "B07MPVF5LF", "B0716J3CYJ", "B07MKNL71F",
"B07KTLGQ4R", "B07MM2TSFB", "B07PPDFFG9", "B07P6T1VBJ", "B07D5H84NL",
"B07PWVH4FW", "B07MW5VL59", "B07KWRD7DD", "B07P54RSPY", "B07M88S533",
"B07MKLLVFZ", "B07NCCDCNK", "B07H3KT9RV", "B07CD3MRZD", "B01NBE6Y5D",
"B076BFW7VZ", "B07MH2YHTX", "B07L6PLLWN", "B07LF6BTWN", "B07BWF4H3W",
"B07DRGBS61", "B07DTR113H", "B07KVFTVHX", "B07CD3MRZF", "B07K1STZHB",
"B07DQR6DQT", "B07N48XQ1V", "B077XFNXX1", "B07N48XQ1V", "B077ZLH5LC",
"B07BRHXCZL", "B01APA6K6M", "B07L9LCYMJ", "B07LF6BTWN", "B07L8BX6NB",
"B077P7D2B2", "B07KWS13NY", "B07BXG7725", "B07KJTRTLX", "B07MNTRQ8Q",
"B07Q8TQVRX", "B07MKZM4Y2", "B07MD1FMN4", "B01GQVA114", "B07HPSR3Z9",
"B01DBGVB7K", "B07KNFW5NJ", "B0744GWBXC", "B00AZL0M34", "B07GDYNK3B",
"B07QQB6DC1", "B07DT78VJJ", "B07P5RT1P5", "B07L9JX6KC", "B07BLPHRX9",
"B07FZZRG2M", "B07L519KGY", "B07JBJM275", "B07FKHYHCQ", "B07QYXQ277"]

    #desktop
    DESKTOP_ASINS = ["B01ATX3PBS", "B07G77QHV8"]

    #mobile phone 
    #PHONE_ASINS = ["B072271F5K", "B06Y6J869C"]
    
    #camera
    #CAMERA_ASINS = ["B00I8BIBCW", "B00I8BIC9E"]

    ASINS = DESKTOP_ASINS
    ASINS = ["B077X1ZB7H"]

    #f = open('amazon_update_0628_latest_4.json', 'a', encoding="utf-8")
    #f = open('amazon_update_0629_draft.json', 'a', encoding="utf-8")
    #f = open('new_amazon_laptop_0506_draft.json', 'a', encoding="utf-8")
    #f = open('new_amazon_phone_0506_draft.json', 'a', encoding="utf-8")
    f = open('new_amazon_camera_0506_draft.json', 'a', encoding="utf-8")

    _ret = {}
    for asin in ASINS:
        _ret = get_results(asin)
        print(_ret)
        f.write(str(_ret) + "\n")

    f.close()

    # print(_ret)
    # print(get_results)

    print("Done")
    return 0


if __name__ == "__main__":
    main()
