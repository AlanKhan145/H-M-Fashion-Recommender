from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import time
from scraper import extract_real_estate_info
from database import get_database
from config import PAGE_START, PAGE_END, BASE_URL, WAIT_TIME, COLLECTION_NAME

# Kết nối MongoDB
database = get_database()
if database is not None:
    properties_col = database[COLLECTION_NAME]

    # Khởi tạo trình duyệt Selenium
    service = Service()
    driver = webdriver.Chrome(service=service)

    try:
        for i in range(PAGE_START, PAGE_END + 1):
            time.sleep(WAIT_TIME)
            website_url = BASE_URL.format(page=i)
            driver.get(website_url)
            data_list = extract_real_estate_info(driver)

            for data in data_list:
                if properties_col.find_one({"title": data["title"]}) is None:
                    properties_col.insert_one(data)
                    print(f"Đã lưu: {data['title']}")
                else:
                    print(f"Dữ liệu đã tồn tại: {data['title']}")
    finally:
        driver.quit()
