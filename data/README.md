# Horoscope Scraper

This project is a Python script designed to scrape daily, weekly, monthly, and yearly horoscopes for all zodiac signs from a specified website and save them in a structured format. The horoscopes are saved in Markdown files organized by date for daily horoscopes, by week number for weekly horoscopes, and by month and year for monthly and yearly horoscopes, respectively.

## Table of Contents
- [Horoscope Scraper](#horoscope-scraper)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Directory Structure](#directory-structure)
    - [Daily Horoscopes](#daily-horoscopes)
    - [Weekly Horoscopes](#weekly-horoscopes)
    - [Monthly Horoscopes](#monthly-horoscopes)
    - [Yearly Horoscopes](#yearly-horoscopes)
  - [Key Components](#key-components)
    - [URL Patterns](#url-patterns)
    - [Functions](#functions)
  - [Main Script Logic](#main-script-logic)
  - [How It Works](#how-it-works)
  - [Usage](#usage)
  - [Conclusion](#conclusion)

## Overview

The script performs the following tasks:
1. Scrapes daily horoscopes for "today," "yesterday," and "tomorrow."
2. Scrapes weekly horoscopes for "this week," "last week," and "next week."
3. Scrapes monthly horoscopes for the current and previous months.
4. Scrapes yearly horoscopes for the years 2023 and 2024.
5. Saves the scraped horoscope data into appropriately named directories and files.

## Directory Structure

The script organizes the horoscopes into the following directory structure:

### Daily Horoscopes

- **Directory:** `data/YYYY-MM-DD/`
  - **Files:** `sign.md`
  - **Description:** Contains daily horoscopes for each zodiac sign. The directory name is based on the date in `YYYY-MM-DD` format.

### Weekly Horoscopes

- **Directory:** `data/week_W/`
  - **Files:**
    - `sign_this_week.md`
    - `sign_last_week.md`
    - `sign_next_week.md`
  - **Description:** Contains weekly horoscopes for each zodiac sign. The directory name includes the week number (`W`).

### Monthly Horoscopes

- **Directory:** `data/Month/`
  - **Files:**
    - `sign_this_month.md`
    - `sign_last_month.md`
  - **Description:** Contains monthly horoscopes for each zodiac sign. The directory name is the month name.

### Yearly Horoscopes

- **Directory:** `data/Year/`
  - **Files:** `sign_YYYY.md`
  - **Description:** Contains yearly horoscopes for each zodiac sign. The directory name is the year (`YYYY`).

## Key Components

### URL Patterns

The script uses predefined URL patterns to fetch horoscopes for different periods (daily, weekly, monthly, and yearly). The URLs are formatted using the zodiac sign and the specific period.

### Functions

1. **get_horoscope(url)**: Fetches the horoscope data from the provided URL.
2. **convert_to_markdown(data)**: Converts the horoscope data into Markdown format.
3. **ensure_directory(directory)**: Ensures that the specified directory exists, creating it if necessary.
4. **save_horoscope(data, directory, filename)**: Saves the horoscope data into a Markdown file in the specified directory.

## Main Script Logic

1. **Daily Horoscopes**: 
   - Iterates over the periods "today," "yesterday," and "tomorrow."
   - Fetches the horoscope for each zodiac sign.
   - Saves the horoscopes in directories named by date (e.g., `2023-06-29`).

2. **Weekly Horoscopes**:
   - Determines the current, last, and next week numbers.
   - Fetches the horoscope for each zodiac sign for the corresponding weeks.
   - Saves the horoscopes in directories named by the week number (e.g., `week_26`).

3. **Monthly Horoscopes**:
   - Determines the current and previous month names.
   - Fetches the horoscope for each zodiac sign for the corresponding months.
   - Saves the horoscopes in directories named by the month (e.g., `June`).

4. **Yearly Horoscopes**:
   - Fetches the horoscope for each zodiac sign for the years 2023 and 2024.
   - Saves the horoscopes in directories named by the year (e.g., `2024`).

## How It Works

1. **Initialize**: The script starts by defining the zodiac signs, time periods, and URL patterns.
2. **Fetch Data**: For each period and zodiac sign, the script constructs the URL, fetches the data, and processes it.
3. **Save Data**: The fetched data is converted to Markdown format and saved in the appropriate directory and file.

## Usage

- The script can be run to fetch and save the latest horoscopes.
- It ensures that duplicate data is not saved by checking existing files and updating only if there is new data.

Usage from data directory
```bash
  python3 scraper.py
```
Usage from home directory

```bash
  python3 data/scraper.py
```

## Conclusion

This script provides an automated way to scrape and organize horoscope data for all zodiac signs. By structuring the data in directories based on dates, weeks, months, and years, it ensures easy access and management of the horoscope information.
