import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule


class TranscriptsSpider(CrawlSpider):
    name = 'transcripts'
    allowed_domains = ['subslikescript.com']
    # start_urls = ['https://subslikescript.com/movies_letter-X']

    # Thiết lập biến user-agent
    user_agent = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
    ]

    # Sử dụng user-agent khi gửi request
    def start_requests(self):
        yield scrapy.Request(
            url='https://subslikescript.com/movies_letter-X',
            headers={'user-agent': self.user_agent}
        )

    # Thiết lập các rule cho crawler
    rules = (
        Rule(
            LinkExtractor(restrict_xpaths=("//ul[@class='scripts-list']/a")),
            callback='parse_item', follow=True, process_request='set_user_agent'
        ),
        Rule(
            LinkExtractor(restrict_xpaths=("(//a[@rel='next'])[1]")),
            process_request='set_user_agent'
        ),
    )

    # Thiết lập user-agent cho request
    def set_user_agent(self, request, spider):
        request.headers['User-Agent'] = self.user_agent
        return request

    def parse_item(self, response):
        # Lấy phần tử article chứa dữ liệu cần thiết (title, plot, v.v...)
        article = response.xpath("//article[@class='main-article']")

        # Trích xuất dữ liệu và trả về dưới dạng dictionary
        yield {
            'title': article.xpath("./h1/text()").get(),
            'plot': article.xpath("./p/text()").get(),
            'transcript': article.xpath("./div[@class='full-script']/text()").getall(),
            'url': response.url,
            'user-agent': response.request.headers['User-Agent'],
        }
