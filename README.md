# DATA-ANALYSIS-dashboard

Django analytics app with CSV/Excel upload, cleaning, KPI cards, charts, export, and streaming support for large datasets.

## Deploy to Render

1. Push this repo to GitHub.
2. Create a new Web Service on Render and connect this repo.
3. Set the following environment variables:
   - `DJANGO_SECRET_KEY` — secure secret
   - `DJANGO_DEBUG` = `False`
   - `DJANGO_ALLOWED_HOSTS` = `your-service.onrender.com`
   - `DATABASE_URL` — provided by Render Postgres or your database service
4. Render build command:
   ```bash
   pip install -r requirements.txt && python manage.py collectstatic --noinput
   ```
5. Render start command:
   ```bash
   gunicorn analysisapp.wsgi:application --log-file -
   ```

Do not paste `envVars:` into the Render start command. Environment variables must be added in Render's Environment tab, or supplied by `render.yaml` when using Blueprint deploys.

## Local setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```
