from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.remote.webdriver import WebDriver
from config import WAIT_TIME

def extract_real_estate_info(driver: WebDriver) -> list:
    """
    Trích xuất thông tin bất động sản từ trang web.

    Tham số:
        driver (WebDriver): Đối tượng Selenium WebDriver.

    Trả về:
        list: Danh sách thông tin bất động sản, mỗi mục là một từ điển chứa:
            - title (str): Tiêu đề.
            - description (str): Mô tả.
            - price (str | None): Giá.
            - area (str | None): Diện tích.
            - price_per_m2 (str | None): Giá mỗi m².
            - bedrooms (str | None): Số phòng ngủ.
            - toilets (str | None): Số WC.
            - location (str | None): Địa điểm.
            - agent_name (str | None): Tên môi giới.
            - published_date (str | None): Ngày đăng.
    """
    real_estate_list = []

    try:
        # Tìm tất cả các phần tử chứa thông tin bất động sản
        info_contents = driver.find_elements(By.CLASS_NAME, "re__card-info")

        for info_content in info_contents:
            try:
                # Lấy tiêu đề
                title = info_content.find_element(By.XPATH, ".//span[@class='pr-title js__card-title']").text.strip()

                # Lấy giá
                price = info_content.find_element(By.XPATH, ".//span[contains(@class, 're__card-config-price')]").text.strip()

                # Lấy diện tích
                area = info_content.find_element(By.XPATH, ".//span[contains(@class, 're__card-config-area')]").text.strip()

                # Lấy giá mỗi mét vuông
                price_per_m2 = info_content.find_element(By.XPATH, ".//span[contains(@class, 're__card-config-price_per_m2')]").text.strip()

                # Lấy số phòng ngủ
                bedrooms_element = info_content.find_elements(By.XPATH, ".//span[contains(@class, 're__card-config-bedroom')]/span")
                bedrooms = bedrooms_element[0].text.strip() if bedrooms_element else None

                # Lấy số WC
                toilets_element = info_content.find_elements(By.XPATH, ".//span[contains(@class, 're__card-config-toilet')]/span")
                toilets = toilets_element[0].text.strip() if toilets_element else None

                # Lấy vị trí
                location_element = info_content.find_elements(By.XPATH, ".//div[@class='re__card-location']/span")
                location = location_element[0].text.strip() if location_element else None

                # Lấy mô tả
                description = info_content.find_element(By.XPATH, ".//div[contains(@class, 're__card-description')]").text.strip()

                # Lấy tên môi giới
                agent_name_element = info_content.find_elements(By.XPATH, ".//div[@class='agent-name agent-item']")
                agent_name = agent_name_element[0].text.strip() if agent_name_element else None

                # Lấy ngày đăng
                published_date_element = info_content.find_elements(By.XPATH, ".//span[contains(@class, 're__card-published-info-published-at')]")
                published_date = published_date_element[0].text.strip() if published_date_element else None

                # Thêm vào danh sách
                real_estate_list.append({
                    "title": title,
                    "description": description,
                    "price": price if price else None,
                    "area": area if area else None,
                    "price_per_m2": price_per_m2 if price_per_m2 else None,
                    "bedrooms": bedrooms if bedrooms else None,
                    "toilets": toilets if toilets else None,
                    "location": location if location else None,
                    "agent_name": agent_name if agent_name else None,
                    "published_date": published_date if published_date else None
                })

            except Exception as e:
                print(f"")
                continue

    except Exception as e:
        print(f"")

    return real_estate_list
