import urllib.request
from bs4 import UnicodeDammit
from bs4 import BeautifulSoup
from scrapy.selector import Selector
import threading
import os
import time

url = 'https://www.ncei.noaa.gov/data/geostationary-ir-channel-brightness-temperature-gridsat-b1/access/1980/'


def getHref():
    try:
        links = []
        headers = {
            'Cookie': '_ga=GA1.3.660922014.1587209913; _ga=GA1.2.1861630245.1587212008; _gid=GA1.2.1017360702.1588485543; _gid=GA1.3.1017360702.1588485543',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.122 Safari/537.36'
        }
        req = urllib.request.Request(url, headers=headers)
        # print(1)
        data = urllib.request.urlopen(req)
        data = data.read()
        dammit = UnicodeDammit(data, ['utf-8', 'gbk'])
        data = dammit.unicode_markup
        # print(data)
        selector = Selector(text=data)
        s = selector.xpath("//td/a/text()").extract()
        for i in s:
            links.append(i)
        return links
    except Exception as e:
        print(e)


def cbk(a, b, c):
    per = 100.0 * a * b / c
    if per > 100:
        per = 100
        print('download finish')


def downLoad(links):
    print(len(links))
    print(links)
    count = 0
    for i in links[2577:]:
        count += 1
        print('thread:', i)
        link = url + i
        path = 'D:\GridSat'
        filename = os.path.join(path, i)
        try:
            # req = urllib.request.Request(link)
            urllib.request.urlretrieve(link, filename)
            print("%d/%d" % (count, len(links)))
        except Exception as e:
            print(e)
            time.sleep(60)
            continue


if __name__ == '__main__':
    links = getHref()
    downLoad(links)
    # t = threading.Thread(target=downLoad, args=(links,))
    # t.setDaemon(False)
    # t.start()
    # t.join()
    print('The end')
