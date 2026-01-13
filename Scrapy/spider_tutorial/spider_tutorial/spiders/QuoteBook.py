import scrapy
from scrapy import FormRequest


class QuotesLoginSpider(scrapy.Spider):
    name = 'quotes_login'
    allowed_domains = ['quotes.toscrape.com']
    start_urls = ['https://quotes.toscrape.com/login']

    def parse(self, response):
        # Parsing the csrf_token, username and password
        csrf_token = response.xpath("//input[@name='csrf_token']/@value").get()

        # Sending FormRequest (FormRequest extends the base Request with functionality for dealing with HTML forms)
        # FormRequest.from_response() simulates a user login
        yield FormRequest.from_response(
            response,
            formxpath='//form',
            formdata={
                'csrf_token': csrf_token,
                'username': 'admin',
                'password': 'admin'
            },
            callback=self.after_login
        )

    def after_login(self, response):
        # Checking if there is a "logout" link to verify successful login
        if response.xpath("//a[@href='/logout']/text()").get():
            print('Successfully logged in!')
