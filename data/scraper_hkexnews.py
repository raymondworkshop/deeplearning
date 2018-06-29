"""
"""

from bs4 import BeautifulSoup
import requests
#import urllib.request
import json

# search based on Advanced search to list docs
HKEXNEWS = "http://www.hkexnews.hk/"
HKEXNEWS_search = "http://www.hkexnews.hk/listedco/listconews/advancedsearch/search_active_main.aspx"

HEADERS = {
   'User-Agent':  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'
}

search_request = {
            "__VIEWSTATEGENERATOR": "D8AF0445",
            "__VIEWSTATEENCRYPTED": None, 
            "ctl00$txt_today": "20180629",
            "ctl00$hfStatus": "AEM",
            "ctl00$hfAlert": None, 
            "ctl00$txt_stock_code": None, 
            "ctl00$txt_stock_name": None, 
            "ctl00$rdo_SelectDocType": "rbAfter2006",
            "ctl00$sel_tier_1": "3",
            "ctl00$sel_DocTypePrior2006": "-1",
            "ctl00$sel_tier_2_group": "-2",
            "ctl00$sel_tier_2": "-2",
            "ctl00$ddlTierTwo": "176,5,22",
            "ctl00$ddlTierTwoGroup": "22,5",
            "ctl00$txtKeyWord": None, 
            "ctl00$rdo_SelectDateOfRelease": "rbManualRange",
            "ctl00$sel_DateOfReleaseFrom_d": "28",
            "ctl00$sel_DateOfReleaseFrom_m": "03",
            "ctl00$sel_DateOfReleaseFrom_y": "2017",
            "ctl00$sel_DateOfReleaseTo_d": "27",
            "ctl00$sel_DateOfReleaseTo_m": "03",
            "ctl00$sel_DateOfReleaseTo_y": "2018",
            "ctl00$sel_defaultDateRange": "SevenDays",

            "ctl00$rdo_SelectSortBy": "rbDateTime"
            
        }

def download(url, file):
    #urllib.request.urlretrieve(url, file)
    print("downloading the %s" % file)

    r = requests.get(url)
    with open(file, 'wb') as f:
        f.write(r.content)

def main():
    #scraper = HKexnewsScraper()
    playload = {
        'searchRequestJson': json.dumps(search_request)
    }

    r = requests.get(HKEXNEWS_search, data=playload, headers=HEADERS)
    # page_response = r.content
    soup = BeautifulSoup(r.content, 'html5lib')

    table = soup.find("table", id='ctl00_gvMain')
    rows = table.findAll("tr")
    for row in rows:
        cells = row.find_all('td')
        for cell in cells:
            txt = cell.get_text()
            if "SHARE OFFER" in txt:
                stockcode = cell.id["ctl00_gvMain_ctl10_lbStockCode"]
                url = HKEXNEWS + cell.href
                download(url, stockcode+".pdf")


if __name__ == "__main__":
    main()