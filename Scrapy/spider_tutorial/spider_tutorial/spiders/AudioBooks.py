import scrapy


class AudibleSpider(scrapy.Spider):
    name = 'audible'
    allowed_domains = ['www.audible.com']
    start_urls = ['https://www.audible.com/search/']

    def start_requests(self):
        # Chỉnh sửa headers mặc định (user-agent)
        yield scrapy.Request(
            url='https://www.audible.com/search/',
            callback=self.parse,
            headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'}
        )

    def parse(self, response):
        # Lấy các box chứa thông tin sản phẩm (tiêu đề, tác giả, độ dài)
        product_container = response.xpath('//div[@class="adbl-impression-container "]//li[contains(@class, "productListItem")]')

        # Lặp qua từng sản phẩm trong product_container
        for product in product_container:
            book_title = product.xpath('.//h3[contains(@class , "bc-heading")]/a/text()').get()
            book_author = product.xpath('.//li[contains(@class , "authorLabel")]/span/a/text()').getall()
            book_length = product.xpath('.//li[contains(@class , "runtimeLabel")]/span/text()').get()

            # Trả về dữ liệu đã thu thập được và user-agent đã định nghĩa
            yield {
                'title': book_title,
                'author': book_author,
                'length': book_length,
                'User-Agent': response.request.headers['User-Agent'],
            }

        # Lấy phần phân trang và tìm liên kết của nút trang tiếp theo (next_page_url)
        pagination = response.xpath('//ul[contains(@class , "pagingElements")]')
        next_page_url = pagination.xpath('.//span[contains(@class , "nextButton")]/a/@href').get()
        button_disabled = pagination.xpath('.//span[contains(@class , "nextButton")]/a/@aria-disabled').get()

        # Chuyển tới liên kết "next_page_url" nếu có, sử dụng user-agent đã định nghĩa
        if next_page_url and button_disabled is None:
            yield response.follow(
                url=next_page_url,
                callback=self.parse,
                headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'}
            )
