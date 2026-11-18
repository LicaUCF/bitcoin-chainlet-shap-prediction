#Assignment 2 - LLM Document Analysis: SEC Form 8-K Product Extraction
#Aline Silva- CAP5619



#In this single-student assignment, you will analyze SEC Form 8-K filings
#using large language models to extract new product releases and structure the data in a tabular format.
#
#Data Description:
#You will process SEC 8-K filings and extract the following information from each document:
#- Company Name: The name of the company filing the 8-K.
#- Stock Name: The ticker symbol of the company.
#- Filing Time: The timestamp of the filing.
#- New Product: The name of the newly announced product.
#- Product Description: A concise description of the product, limited to less than 180 characters.
#
#Task:
#Extract and structure data into the following format:
#company_name, stock_name, filing_time, new_product, product_description
#
#Rubric:
#Extraction Accuracy (60 pts): Full marks if code runs properly and uses local LLM extraction.
#Output Formatting (20 pts): Full marks for required columns, proper CSV, and 100+ entries.
#Report Quality (20 pts): 1-2 page report detailing challenges, LLM specs, and GitHub URL.
#





# libraries needed
import os
import re
import csv
import time
import json
import requests
from bs4 import BeautifulSoup

# local path and output file configuration
working_dir ='/Users/alinesilva/Library/CloudStorage/OneDrive-UniversityofCentralFlorida/UCF Fintech/CAP 5619 - Artificial Intelligence for FinTech/Assignments/Assignment2'
if not os.path.exists(working_dir):
    os.makedirs(working_dir, exist_ok=True)

csv_output_path =os.path.join(working_dir, 'extracted_8k_products.csv')

# sec fair access headers
sec_headers = {
    'User-Agent': 'AlineSilva al570598@ucf.edu',
    'Accept-Encoding': 'gzip, deflate'
}

# local ollama configuration
ollama_url ='http://localhost:11434/api/generate'
model_name ='llama3.2'
target_count =100


# 1 download company tickers mapping
tickers_url ='https://www.sec.gov/files/company_tickers.json'
resp =requests.get(tickers_url, headers=sec_headers)
if resp.status_code == 200:
    companies_data =resp.json()
    print(f'loaded sec tickers successfully, total companies: {len(companies_data)}')
else:
    print('failed to load tickers from sec')
    companies_data ={}


# helper function to pad cik to 10 digits
def pad_cik(cik):
    return str(cik).zfill(10)


# 2 fetch 8-k filing entries using sec atom feed
def get_company_8k_filings(cik_padded, count=15):
    feed_url =f'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik_padded}&type=8-K&count={count}&output=atom'
    time.sleep(0.12) # sec rate limit compliance
    try:
        r =requests.get(feed_url, headers=sec_headers, timeout=2.0)
        if r.status_code != 200:
            return []
    except Exception as e:
        return []

    soup =BeautifulSoup(r.content, 'xml')
    entries =[]
    for entry in soup.find_all('entry'):
        title_tag =entry.find('title')
        updated_tag =entry.find('updated')
        link_tag =entry.find('link')

        title =title_tag.text if title_tag else ''
        if '8-K' not in title:
            continue

        filing_time =updated_tag.text[:10] if updated_tag else ''
        doc_url =link_tag.get('href') if link_tag else None

        if doc_url:
# resolve html index link to plain text document
            if doc_url.endswith('-index.htm') or doc_url.endswith('-index.html'):
                doc_url =doc_url.replace('-index.htm', '.txt').replace('-index.html', '.txt')
            entries.append({
                'filing_time': filing_time,
                'url': doc_url
            })
    return entries


# 3 clean filing text by removing tags and non-text elements
def clean_document_text(raw_html):
    soup =BeautifulSoup(raw_html, 'html.parser')
    for tag in soup(['script', 'style', 'header', 'footer']):
        tag.decompose()
    text =soup.get_text(separator=' ')
    clean_text =re.sub(r'\s+', ' ', text).strip()
    return clean_text


# 4 query local ollama llm for named entity recognition (ner)
def extract_product_with_ner(company_name, ticker, filing_time, filing_text):
# filter non product filings before running llm
    keywords =['launch', 'unveil', 'introduce', 'announced', 'release', 'new product', 'commercial availability']
    if not any(k in filing_text.lower() for k in keywords):
        return None

# prompt structure
    prompt =f"""You are an NLP model performing Named Entity Recognition (NER) on an SEC 8-K filing.

Document Context:
Company: {company_name}
Stock Ticker: {ticker}
Filing Date: {filing_time}

Filing Content:
\"\"\"{filing_text[:3500]}\"\"\"

Instructions:
Determine if this filing announces a new product or service.
If NO new product is announced, return: {{"has_product": false}}
If a new product IS announced, extract:
1. "new_product": exact entity name of the new product
2. "product_description": clear description under 170 characters

Return ONLY valid JSON format."""

    payload = {
        'model': model_name,
        'prompt': prompt,
        'stream': False,
        'format': 'json',
        'options': {
            'temperature': 0.0,
            'top_p': 0.1
        }
    }

    try:
        response =requests.post(ollama_url, json=payload, timeout=30)
        if response.status_code == 200:
            parsed =json.loads(response.json().get('response', '{}'))
            if parsed.get('has_product'):
                prod_name =parsed.get('new_product', '').strip()
                desc =parsed.get('product_description', '').strip()

                if not prod_name or not desc:
                    return None

# enforce limit of under 180 characters
                if len(desc) >= 180:
                    desc =desc[:174].rstrip() + '...'

                return {
                    'company_name': company_name,
                    'stock_name': ticker,
                    'filing_time': filing_time,
                    'new_product': prod_name,
                    'product_description': desc
                }
    except Exception as e:
        return None
    return None


# 5 main extraction loop and csv generation
# write headers if file does not exist or has 0 bytes
if not os.path.exists(csv_output_path) or os.path.getsize(csv_output_path) == 0:
    with open(csv_output_path, mode='w', newline='', encoding='utf-8') as f:
        writer =csv.writer(f)
        writer.writerow(['company_name', 'stock_name', 'filing_time', 'new_product', 'product_description'])

# count existing saved records to avoid duplicate counts
existing_lines =0
if os.path.exists(csv_output_path):
    with open(csv_output_path, mode='r', encoding='utf-8') as f:
        existing_lines =max(0, len(f.readlines()) - 1)

total_saved =existing_lines
print(f'current records in csv: {total_saved}/{target_count}')

# priority tickers with active commercial releases to finish rapidly
priority_tickers =[
    'AAPL', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AMD', 'INTC', 'QCOM',
    'AVGO', 'CSCO', 'ORCL', 'IBM', 'TXN', 'CRM', 'NOW', 'PANW', 'SNOW', 'PLTR',
    'DELL', 'HPQ', 'HPE', 'SMCI', 'ARM', 'MU', 'ADI', 'LRCX', 'KLAC', 'AMAT',
    'GILD', 'MRNA', 'BIIB', 'VRTX', 'REGN', 'LLY', 'JNJ', 'PFE', 'ABT', 'BMY',
    'F', 'GM', 'RIVN', 'LCID', 'SONY', 'SPOT', 'NFLX', 'UBER', 'DASH', 'SHOP',
    'SQ', 'PYPL', 'COIN', 'CRWD', 'FTNT', 'ZS', 'DDOG', 'NET', 'MDB', 'SNPS',
    'CDNS', 'ANSS', 'WDAY', 'ADBE', 'INTU', 'TEAM', 'DOCU', 'ZM', 'OKTA', 'ON'
]

ticker_to_info ={comp['ticker']: comp for comp in companies_data.values() if 'ticker' in comp}
ordered_companies =[ticker_to_info[t] for t in priority_tickers if t in ticker_to_info]

for comp in companies_data.values():
    if comp.get('ticker') not in priority_tickers:
        ordered_companies.append(comp)

for comp in ordered_companies:
    if total_saved >= target_count:
        break

    cik_raw =comp.get('cik_str')
    ticker =comp.get('ticker')
    company_name =comp.get('title')

    if not cik_raw or not ticker:
        continue


    cik_padded =pad_cik(cik_raw)
    filings =get_company_8k_filings(cik_padded, count=15)

    for item in filings:
        if total_saved >= target_count:
            break

        time.sleep(0.12)
        try:
            doc_resp =requests.get(item['url'], headers=sec_headers, timeout=4.0)
            if doc_resp.status_code != 200:
                continue
            cleaned_text =clean_document_text(doc_resp.text[:150000])
        except Exception:
            continue

        result =extract_product_with_ner(company_name, ticker, item['filing_time'], cleaned_text)
        if result:
            total_saved += 1
# append to csv immediately
            with open(csv_output_path, mode='a', newline='', encoding='utf-8') as f:
                writer =csv.writer(f)
                writer.writerow([
                    result['company_name'],
                    result['stock_name'],
                    result['filing_time'],
                    result['new_product'],
                    result['product_description']
                ])
            print(f'[{total_saved}/{target_count}] added: {result["stock_name"]} -> {result["new_product"]}')

# print summary
print(f'completed extraction. total entries saved: {total_saved}')