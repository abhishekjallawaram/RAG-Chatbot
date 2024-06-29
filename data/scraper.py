import requests
from bs4 import BeautifulSoup
import os
from datetime import datetime, timedelta

ZODIAC_SIGNS = [
    "aries", "taurus", "gemini", "cancer", "leo",
    "virgo", "libra", "scorpio", "sagittarius",
    "capricorn", "aquarius", "pisces"
]

TIME_PERIODS = {
    "today": datetime.now(),
    "yesterday": datetime.now() - timedelta(days=1),
    "tomorrow": datetime.now() + timedelta(days=1)
}

URL_PATTERNS = {
    "daily": {
        "today": "https://www.astrology.com/horoscope/daily/{sign}.html",
        "yesterday": "https://www.astrology.com/horoscope/daily/yesterday/{sign}.html",
        "tomorrow": "https://www.astrology.com/horoscope/daily/tomorrow/{sign}.html"
    },
    "weekly": {
        "this_week": "https://www.astrology.com/horoscope/weekly-overview/{sign}.html",
        "last_week": "https://www.astrology.com/horoscope/weekly-overview/last-week/{sign}.html",
        "next_week": "https://www.astrology.com/horoscope/weekly-overview/next-week/{sign}.html"
    },
    "monthly": {
        "this_month": "https://www.astrology.com/horoscope/monthly-overview/{sign}.html",
        "last_month": "https://www.astrology.com/horoscope/monthly-overview/last-month/{sign}.html"
    },
    "yearly": {
        "2023": "https://www.astrology.com/us/horoscope/yearly-overview-2023.aspx?sign={sign}",
        "2024": "https://www.astrology.com/us/horoscope/yearly-overview-2024.aspx?sign={sign}"
    }
}

def get_horoscope(url):
    response = requests.get(url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        prediction_element = soup.select_one('div.horoscope-content-wrapper p')
        if prediction_element:
            return prediction_element.get_text(strip=True)
        else:
            return 'Unable to fetch horoscope content'
    else:
        return 'Unable to fetch horoscope content'

def convert_to_markdown(data):
    md_content = f"# Horoscope for {data['sign'].capitalize()} on {data['period'].capitalize()}\n\n"
    md_content += f"**Date:** {data['date']}\n\n"
    md_content += f"**Sign:** {data['sign'].capitalize()}\n\n"
    md_content += f"**Prediction:**\n\n{data['prediction']}\n"
    return md_content

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_horoscope(data, directory, filename):
    ensure_directory(directory)
    file_path = f"{directory}/{filename}"
    
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            existing_content = file.read()
        new_content = convert_to_markdown(data)
        if existing_content == new_content:
            print(f"File {file_path} already exists and data is accurate. Ignoring.")
        else:
            with open(file_path, "w") as file:
                file.write(new_content)
            print(f"File {file_path} already exists but data was updated.")
    else:
        markdown_content = convert_to_markdown(data)
        with open(file_path, "w") as file:
            file.write(markdown_content)
        print(f"Markdown file created: {file_path}")

if __name__ == "__main__":
    current_date = datetime.now().strftime('%m-%d-%Y')
    base_directory = "data"
    
    for period, date in TIME_PERIODS.items():
        date_str = date.strftime('%m-%d-%Y')
        for sign in ZODIAC_SIGNS:
            url = URL_PATTERNS['daily'][period].format(sign=sign)
            prediction = get_horoscope(url)
            data = {
                'period': period,
                'sign': sign,
                'prediction': prediction,
                'date': date.strftime('%B %d, %Y')
            }
            save_horoscope(data, f"{base_directory}/{date_str}", f"{sign}.md")

    # Weekly horoscopes
    weekly_periods = ['this_week', 'last_week', 'next_week']
    current_week = datetime.now().isocalendar()[1]
    last_week = (datetime.now() - timedelta(weeks=1)).isocalendar()[1]
    next_week = (datetime.now() + timedelta(weeks=1)).isocalendar()[1]
    week_mapping = {'this_week': current_week, 'last_week': last_week, 'next_week': next_week}

    for period in weekly_periods:
        for sign in ZODIAC_SIGNS:
            url = URL_PATTERNS['weekly'][period].format(sign=sign)
            prediction = get_horoscope(url)
            data = {
                'period': period.replace('_', ' '),
                'sign': sign,
                'prediction': prediction,
                'date': datetime.now().strftime('%B %d, %Y')
            }
            week_dir = week_mapping[period]
            save_horoscope(data, f"{base_directory}/week_{week_dir}", f"{sign}_{period}.md")

    # Monthly horoscopes
    monthly_periods = ['this_month', 'last_month']
    current_month = datetime.now().strftime('%B')
    last_month = (datetime.now() - timedelta(days=30)).strftime('%B')

    for period in monthly_periods:
        for sign in ZODIAC_SIGNS:
            url = URL_PATTERNS['monthly'][period].format(sign=sign)
            prediction = get_horoscope(url)
            data = {
                'period': period.replace('_', ' '),
                'sign': sign,
                'prediction': prediction,
                'date': datetime.now().strftime('%B %d, %Y')
            }
            month_dir = current_month if period == 'this_month' else last_month
            save_horoscope(data, f"{base_directory}/{month_dir}", f"{sign}_{period}.md")

    # Yearly horoscopes
    yearly_periods = ['2023', '2024']
    for period in yearly_periods:
        for sign in ZODIAC_SIGNS:
            url = URL_PATTERNS['yearly'][period].format(sign=sign)
            prediction = get_horoscope(url)
            data = {
                'period': period,
                'sign': sign,
                'prediction': prediction,
                'date': datetime.now().strftime('%B %d, %Y')
            }
            save_horoscope(data, f"{base_directory}/{period}", f"{sign}_{period}.md")
