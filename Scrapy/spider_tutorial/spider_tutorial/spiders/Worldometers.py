import scrapy


class WorldometersSpider(scrapy.Spider):
    name = 'worldometers'
    allowed_domains = ['www.worldometers.info']
    start_urls = ['https://www.worldometers.info/world-population/population-by-country/']

    def parse(self, response):
        # Lấy các phần tử "a" cho mỗi quốc gia
        countries = response.xpath('//td/a')

        # Lặp qua danh sách các quốc gia
        for country in countries:
            country_name = country.xpath(".//text()").get()
            link = country.xpath(".//@href").get()

            # Trả về URL tương đối (gửi request với URL tương đối)
            yield response.follow(
                url=link,
                callback=self.parse_country,
                meta={'country': country_name}
            )

    def parse_country(self, response):
        # Lấy tên quốc gia và các phần tử hàng trong bảng dân số
        country = response.request.meta['country']
        rows = response.xpath(
            "(//table[contains(@class,'table')])[1]/tbody/tr"
        )  # Bạn cũng có thể sử dụng giá trị đầy đủ của class

        # Lặp qua danh sách các hàng
        for row in rows:
            year = row.xpath(".//td[1]/text()").get()
            population = row.xpath(".//td[2]/strong/text()").get()

            # Trả về dữ liệu đã được trích xuất
            yield {
                'country': country,
                'year': year,
                'population': population,
            }
