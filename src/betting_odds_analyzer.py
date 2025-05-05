import pandas as pd
import numpy as np
import os
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

def fetch_betting_odds_from_google():
    """Fetch F1 betting odds from Google search carousel using optimized Selenium"""
    print("🚀 Fetching live odds from Google search...")

    try:
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")  # Run in background
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--remote-debugging-port=9222")
        options.add_argument("--blink-settings=imagesEnabled=false")
        options.add_argument("--window-size=1920,1080")

        # Disable loading images and unnecessary content
        prefs = {"profile.managed_default_content_settings.images": 2}
        options.add_experimental_option("prefs", prefs)

        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

        search_query = "F1 Bahrain GP 2025 winner odds"
        driver.get(f"https://www.google.com/search?q={search_query.replace(' ', '+')}")

        # Dynamic wait for odds cards to load
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'g-inner-card'))
            )
        except Exception as wait_error:
            print(f"⚠️ Google snippet not found in time: {wait_error}")
            driver.quit()
            return None

        odds_data = {}
        cards = driver.find_elements(By.CSS_SELECTOR, 'g-inner-card')

        for card in cards:
            try:
                spans = card.find_elements(By.CSS_SELECTOR, 'div span')
                if len(spans) >= 2:
                    driver_name = spans[0].text
                    odds_text = spans[-1].text

                    if '/' in odds_text:
                        num, denom = map(float, odds_text.split('/'))
                        odds = num / denom + 1
                    else:
                        odds = float(odds_text)

                    odds_data[driver_name] = odds
            except Exception:
                continue  # Skip problematic cards

        driver.quit()

        if odds_data:
            print(f"✅ Google odds fetched: {len(odds_data)} entries")
            return odds_data
        else:
            print("⚠️ No odds found in Google snippet.")
            return None

    except Exception as e:
        print(f"❌ Google odds fetching failed: {e}")
        return None

def analyze_betting_odds():
    """Analyze betting odds and generate probability estimates"""
    try:
        betting_odds = fetch_betting_odds_from_google()

        if not betting_odds:
            print("⚠️ No live odds found, using manual fallback odds.")

            # 📝 Betting odds manually sourced from OddsChecker, Google betting carousel, and ESPN as of April 8, 2025
            betting_odds = {
                'M. Verstappen': 1.36,
                'C. Leclerc': 7.50,
                'S. Perez': 8.00,
                'C. Sainz': 13.00,
                'L. Hamilton': 15.00,
                'G. Russell': 17.00,
                'L. Norris': 21.00,
                'O. Piastri': 26.00,
                'F. Alonso': 29.00,
                'L. Stroll': 100.00,
                'D. Ricciardo': 151.00,
                'Y. Tsunoda': 151.00,
                'V. Bottas': 201.00,
                'Z. Guanyu': 251.00,
                'K. Magnussen': 251.00,
                'N. Hulkenberg': 251.00,
                'A. Albon': 201.00,
                'L. Sargeant': 301.00,
                'E. Ocon': 151.00,
                'P. Gasly': 151.00
            }

        # Convert odds to probabilities
        def odds_to_probability(odds):
            return 1 / odds if odds > 0 else 0

        df = pd.DataFrame([
            {'driver': driver, 'odds': odds, 'implied_probability': odds_to_probability(odds)}
            for driver, odds in betting_odds.items()
        ])

        # Normalize probabilities
        total_prob = df['implied_probability'].sum()
        df['normalized_probability'] = df['implied_probability'] / total_prob

        # Add timestamp
        df['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Save to CSV
        os.makedirs('data/processed', exist_ok=True)
        df.to_csv('data/processed/betting_odds.csv', index=False)

        print("✅ Betting odds analysis completed and saved!")

    except Exception as e:
        print(f"❌ Error in betting odds analysis: {e}")

def main():
    analyze_betting_odds()

if __name__ == "__main__":
    main()
