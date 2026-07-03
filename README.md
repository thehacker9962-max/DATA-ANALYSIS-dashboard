# Data Analysis Dashboard

A Django-based analytics web app that lets users upload CSV or Excel files, clean the data, inspect quality issues, view visual insights, and receive business-focused recommendations to improve performance.

## Project Overview

This project is designed for fast, no-code data analysis. Users can:

- upload a dataset from CSV or Excel
- automatically clean and standardize column names
- detect missing values, duplicates, and trends
- view KPI cards and charts
- receive recommendations for sales, growth, and business improvement

## Main Features

- Upload and analyze CSV/Excel data
- Automatic data cleaning and normalization
- Data quality metrics and weakness detection
- Trend and average visualizations
- Business recommendations for better decisions
- Responsive dashboard UI for desktop and mobile

## Technology Stack

- Python
- Django
- Pandas
- NumPy
- scikit-learn
- Chart.js
- Render deployment support

## Local Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

## Render Deployment

1. Push this repository to GitHub.
2. Create a new Web Service on Render.
3. Connect the repository and use the included Render configuration.
4. Set the environment variables in Render:
   - `DJANGO_SECRET_KEY`
   - `DJANGO_DEBUG=False`
   - `DJANGO_ALLOWED_HOSTS=your-service.onrender.com`
5. Deploy the service.

## Example Use Case

Upload sales, customer, or product data to identify:

- weak data quality areas
- missing information
- performance trends
- opportunities to improve sales or operations

## Folder Structure

- `analytics/` — main dashboard logic, templates, and analysis functions
- `analysisapp/` — Django project configuration
- `templates/` — UI views and dashboard layout

