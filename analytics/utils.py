import os
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from io import BytesIO

def save_dataframe_to_temp(df: pd.DataFrame, suffix: str = ".pkl") -> str:
    temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    path = temp_file.name
    temp_file.close()
    df.to_pickle(path)
    return path


def load_dataframe_from_path(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_pickle(path)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(col).strip().replace(" ", "_") for col in cleaned.columns]
    cleaned = cleaned.rename(columns=lambda c: c.lower())
    cleaned = cleaned.apply(lambda col: col.astype(str).str.strip() if col.dtype == "object" else col)
    cleaned = cleaned.replace({np.nan: None})
    cleaned = cleaned.drop_duplicates()

    for col in cleaned.columns:
        if cleaned[col].dtype == "object":
            cleaned[col] = cleaned[col].replace({"": None, "nan": None, "None": None})

    return cleaned


def get_column_profiles(df: pd.DataFrame) -> List[Dict[str, Any]]:
    profiles = []
    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        profiles.append(
            {
                "name": col,
                "dtype": str(series.dtype),
                "null_count": int(series.isna().sum()),
                "unique_count": int(series.nunique(dropna=True)),
                "sample_values": [str(x) for x in non_null.head(5).tolist()],
            }
        )
    return profiles


def build_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    missing = df.isna().sum().to_dict()
    duplicate_count = int(df.duplicated().sum())
    quality_score = max(0, min(100, round(100 - (sum(missing.values()) * 2) - duplicate_count * 5)))

    return {
        "quality_score": quality_score,
        "null_values": missing,
        "duplicate_rows": duplicate_count,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
    }


def analyze_dataframe(df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
    cleaned = clean_dataframe(df)
    quality = build_quality_report(cleaned)

    numeric_columns = [col for col in cleaned.columns if pd.api.types.is_numeric_dtype(cleaned[col])]
    categorical_columns = [col for col in cleaned.columns if col not in numeric_columns]

    kpis = {
        "rows": quality["row_count"],
        "columns": quality["column_count"],
        "duplicates": quality["duplicate_rows"],
        "missing_values": int(sum(quality["null_values"].values())),
        "quality_score": quality["quality_score"],
    }

    insights = []
    chart_data = []
    if numeric_columns:
        numeric_summary = cleaned[numeric_columns].describe().to_dict()
        insights.append(f"Numeric columns detected: {', '.join(numeric_columns[:5])}")
        if target_column and target_column in numeric_columns:
            insights.append("Target column available for predictive analysis.")
        for col in numeric_columns[:6]:
            series = cleaned[col].dropna()
            if not series.empty:
                chart_data.append({"name": col, "value": round(float(series.mean()), 2)})
    if categorical_columns:
        insights.append(f"Categorical columns detected: {', '.join(categorical_columns[:5])}")

    trend = None
    if numeric_columns and len(cleaned) > 1:
        first_numeric = numeric_columns[0]
        series = pd.to_numeric(cleaned[first_numeric], errors='coerce').dropna()
        if len(series) > 1:
            x = list(range(len(series)))
            y = series.tolist()
            slope = (sum((xi - sum(x)/len(x)) * (yi - sum(y)/len(y)) for xi, yi in zip(x, y)) /
                     sum((xi - sum(x)/len(x)) ** 2 for xi in x)) if sum((xi - sum(x)/len(x)) ** 2 for xi in x) else 0
            trend = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
            insights.append(f"Trend hint for {first_numeric}: {trend}.")

    correlation = None
    if len(numeric_columns) >= 2:
        corr_cols = numeric_columns[:10]
        corr_matrix = cleaned[corr_cols].corr(numeric_only=True)
        corr_matrix = corr_matrix.fillna(0)
        correlation = corr_matrix.to_dict()
        if len(numeric_columns) > 10:
            insights.append(
                "Correlation computed for first 10 numeric columns only to preserve performance."
            )

    prediction = None
    if target_column and target_column in cleaned.columns and pd.api.types.is_numeric_dtype(cleaned[target_column]):
        feature_frame = cleaned.select_dtypes(include=[np.number]).drop(columns=[target_column], errors="ignore")
        if not feature_frame.empty and len(feature_frame.dropna()) > 5:
            model_df = cleaned[[target_column] + list(feature_frame.columns)].dropna()
            X = model_df[feature_frame.columns]
            y = model_df[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            prediction = {
                "model": "LinearRegression",
                "mse": round(float(mse), 4),
                "r2_hint": "Model trained with numeric features.",
            }

    return {
        "cleaned_df": cleaned,
        "row_count": quality["row_count"],
        "column_count": quality["column_count"],
        "quality_score": quality["quality_score"],
        "column_profiles": get_column_profiles(cleaned),
        "quality_report": quality,
        "kpis": kpis,
        "insights": insights,
        "correlation": correlation,
        "prediction": prediction,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "chart_data": chart_data,
        "trend": trend,
    }


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Read a CSV or Excel file uploaded by the user."""
    filename = uploaded_file.name.lower()
    excel_mime_types = {
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
    }

    if filename.endswith((".xls", ".xlsx")) or uploaded_file.content_type in excel_mime_types:
        uploaded_file.seek(0)
        return pd.read_excel(uploaded_file)

    uploaded_file.seek(0)
    try:
        return pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        try:
            return pd.read_excel(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="latin1", engine="python")


def save_uploaded_to_disk(uploaded_file) -> str:
    """Save uploaded file to a temp path and return the path."""
    suffix = os.path.splitext(uploaded_file.name)[1] or '.csv'
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    uploaded_file.seek(0)
    # write in chunks to avoid using too much memory
    for chunk in uploaded_file.chunks():
        tf.write(chunk)
    tf.flush()
    tf.close()
    return tf.name


def analyze_large_file(path: str, sample_size: int = 2000, chunksize: int = 100000) -> Dict[str, Any]:
    """Analyze a large CSV in streaming fashion and return a summary dict.

    - Computes row/column counts, null counts, simple KPIs.
    - Uses reservoir sampling to build a representative sample DataFrame (<= sample_size).
    - Aggregates numeric column sums/counts to compute means.
    - For correlations and charts, uses the sampled DataFrame (keeps memory small).
    """
    # Only CSV is supported for very large files here
    reader = pd.read_csv(path, chunksize=chunksize, iterator=True)
    total_rows = 0
    cols = None
    nulls = None
    numeric_sums = {}
    numeric_counts = {}
    duplicate_est = 0
    sample_rows = []
    import random

    for chunk in reader:
        if cols is None:
            cols = list(chunk.columns)
            nulls = {c: 0 for c in cols}
            # init numeric trackers
            for c in cols:
                if pd.api.types.is_numeric_dtype(chunk[c]):
                    numeric_sums[c] = 0.0
                    numeric_counts[c] = 0

        # update totals
        n = len(chunk)
        total_rows += n
        # nulls
        nullcounts = chunk.isna().sum().to_dict()
        for c, v in nullcounts.items():
            nulls[c] = nulls.get(c, 0) + int(v)

        # numeric aggregates
        for c in list(numeric_sums.keys()):
            ser = pd.to_numeric(chunk[c], errors='coerce')
            numeric_sums[c] += float(ser.sum(skipna=True)) if not ser.empty else 0.0
            numeric_counts[c] += int(ser.count())

        # reservoir sampling for sample_size rows
        for idx, row in chunk.iterrows():
            if len(sample_rows) < sample_size:
                sample_rows.append(row.to_dict())
            else:
                j = random.randint(0, total_rows - 1)
                if j < sample_size:
                    sample_rows[j] = row.to_dict()

    # build sample DataFrame
    sample_df = pd.DataFrame(sample_rows, columns=cols) if sample_rows else pd.DataFrame(columns=cols)

    # compute numeric columns list and means from aggregated sums
    numeric_columns = [c for c in cols if c in numeric_counts and numeric_counts[c] > 0]
    chart_data = []
    for c in numeric_columns[:6]:
        mean_val = numeric_sums[c] / numeric_counts[c] if numeric_counts[c] else 0
        chart_data.append({"name": c, "value": round(float(mean_val), 2)})

    # correlation based on sample
    correlation = None
    if len(numeric_columns) >= 2 and not sample_df.empty:
        corr = sample_df[numeric_columns].corr(numeric_only=True).fillna(0)
        correlation = corr.to_dict()

    # build summary similar to analyze_dataframe
    quality = {
        "quality_score": 100,
        "null_values": nulls,
        "duplicate_rows": 0,
        "row_count": int(total_rows),
        "column_count": len(cols) if cols else 0,
    }

    insights = []
    if numeric_columns:
        insights.append(f"Numeric columns detected: {', '.join(numeric_columns[:5])}")
    categorical_columns = [c for c in cols if c not in numeric_columns] if cols else []
    if categorical_columns:
        insights.append(f"Categorical columns detected: {', '.join(categorical_columns[:5])}")

    trend = None
    if numeric_columns and not sample_df.empty:
        first_numeric = numeric_columns[0]
        ser = pd.to_numeric(sample_df[first_numeric], errors='coerce').dropna()
        if len(ser) > 1:
            x = list(range(len(ser)))
            y = ser.tolist()
            slope = (sum((xi - sum(x)/len(x)) * (yi - sum(y)/len(y)) for xi, yi in zip(x, y)) /
                     sum((xi - sum(x)/len(x)) ** 2 for xi in x)) if sum((xi - sum(x)/len(x)) ** 2 for xi in x) else 0
            trend = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
            insights.append(f"Trend hint for {first_numeric}: {trend}.")

    summary = {
        "cleaned_df": sample_df,
        "row_count": quality["row_count"],
        "column_count": quality["column_count"],
        "quality_score": quality["quality_score"],
        "column_profiles": get_column_profiles(sample_df),
        "quality_report": quality,
        "kpis": {"rows": quality["row_count"], "columns": quality["column_count"], "duplicates": 0, "missing_values": sum(nulls.values()), "quality_score": quality["quality_score"]},
        "insights": insights,
        "correlation": correlation,
        "prediction": None,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "chart_data": chart_data,
        "trend": trend,
    }

    return summary

def generate_chart_data(df: pd.DataFrame, numeric_columns: List[str]) -> Dict[str, Any]:
    """Return Chart.js-compatible data for interactive chart rendering."""
    chart_data: Dict[str, Any] = {}
    if not numeric_columns:
        return chart_data

    # Use the first numeric column for trend analysis
    first_numeric = numeric_columns[0]
    series = pd.to_numeric(df[first_numeric], errors='coerce').dropna()
    if not series.empty:
        max_points = 200
        if len(series) > max_points:
            indices = np.linspace(0, len(series) - 1, max_points, dtype=int)
            sample_series = series.iloc[indices]
            labels = [str(i + 1) for i in indices]
        else:
            sample_series = series
            labels = [str(i + 1) for i in range(len(series))]

        chart_data['trend'] = {
            'type': 'line',
            'labels': labels,
            'datasets': [
                {
                    'label': f'{first_numeric} Trend',
                    'data': [float(v) for v in sample_series.tolist()],
                    'borderColor': '#2563eb',
                    'backgroundColor': 'rgba(37, 99, 235, 0.2)',
                    'fill': True,
                }
            ],
        }

    # Histogram for first up to 3 numeric columns
    hist_labels = []
    hist_values = []
    for col in numeric_columns[:3]:
        series = pd.to_numeric(df[col], errors='coerce').dropna()
        if series.empty:
            continue
        hist_labels.append(col)
        hist_values.append(float(series.mean()))
    if hist_labels and hist_values:
        chart_data['summary'] = {
            'type': 'bar',
            'labels': hist_labels,
            'datasets': [
                {
                    'label': 'Average value',
                    'data': hist_values,
                    'backgroundColor': ['#2563eb', '#10b981', '#f59e0b'][:len(hist_values)],
                }
            ],
        }

    # Correlation matrix as a heatmap-like dataset for Chart.js
    if len(numeric_columns) >= 2:
        corr = df[numeric_columns].corr(numeric_only=True).fillna(0)
        chart_data['correlation'] = {
            'labels': list(corr.columns),
            'matrix': corr.round(2).to_dict(),
        }

    return chart_data

def generate_recommendations(summary: Dict[str, Any], df: pd.DataFrame, target: Optional[str] = None) -> List[str]:
    recs: List[str] = []
    # Quality-based recommendations
    q = summary.get('quality_report', {})
    if q.get('duplicate_rows', 0) > 0:
        recs.append('Remove duplicate rows to avoid skewed aggregates.')
    nulls = sum(q.get('null_values', {}).values()) if q.get('null_values') else 0
    if nulls > 0:
        recs.append('Impute or remove missing values to improve model quality.')

    # Trend-based
    trend = summary.get('trend')
    if trend == 'increasing':
        recs.append('Sales show an increasing trend — consider scaling inventory and marketing to capture demand.')
    elif trend == 'decreasing':
        recs.append('Sales are decreasing — investigate recent changes, promotions, pricing, or supply issues.')
    else:
        recs.append('Sales appear stable; run A/B tests on promotions to find uplift opportunities.')

    # Correlation-based suggestions
    corr = summary.get('correlation')
    if isinstance(corr, dict) and target:
        # find features most correlated with target
        if target in corr:
            pairs = corr[target]
            sorted_feats = sorted(pairs.items(), key=lambda kv: abs(kv[1]), reverse=True)
            top = [f for f, v in sorted_feats if f != target][:3]
            if top:
                recs.append(f'Features strongly associated with {target}: {", ".join(top)} — consider focusing campaigns on these drivers.')

    # Generic actions
    recs.append('Run targeted promotions for top-performing segments and monitor lift.')
    recs.append('Use top correlated numeric features as candidate predictors for forecasting models.')

    return recs


def compute_business_metrics(summary: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    """Compute business-facing KPIs like sales rating and product performance."""
    metrics: Dict[str, Any] = {}
    q_score = summary.get('quality_score', 0)
    trend = summary.get('trend')

    # Heuristic: detect sales/revenue column
    sales_col = None
    for candidate in ['sales', 'revenue', 'amount', 'total']:
        if candidate in df.columns:
            sales_col = candidate
            break

    avg_sales = None
    if sales_col:
        series = pd.to_numeric(df[sales_col], errors='coerce').dropna()
        if not series.empty:
            avg_sales = float(series.mean())

    # Sales rating heuristic
    rating = 'N/A'
    if avg_sales is not None:
        if q_score >= 75 and trend == 'increasing' and avg_sales > 0:
            rating = 'Excellent'
        elif q_score >= 50 and avg_sales > 0:
            rating = 'Good'
        else:
            rating = 'Poor'
    metrics['sales_rating'] = rating
    metrics['avg_sales'] = avg_sales

    # Product performance: find product column and rank by revenue or sales
    product_col = None
    for candidate in ['product', 'product_name', 'product_id', 'sku']:
        if candidate in df.columns:
            product_col = candidate
            break

    top_products: List[Dict[str, Any]] = []
    if product_col and sales_col:
        gp = df[[product_col, sales_col]].dropna()
        try:
            gp[sales_col] = pd.to_numeric(gp[sales_col], errors='coerce')
            agg = gp.groupby(product_col)[sales_col].sum().sort_values(ascending=False).head(5)
            top_products = [{'product': idx, 'value': float(val)} for idx, val in agg.items()]
        except Exception:
            top_products = []
    metrics['top_products'] = top_products

    # Product performance score
    perf_score = 0
    if top_products:
        perf_score = min(100, int(sum(p['value'] for p in top_products) / (len(top_products) * (abs(metrics.get('avg_sales') or 1))) * 10))
    metrics['product_performance_score'] = perf_score

    return metrics


def dataframe_to_excel(df: pd.DataFrame, output_path: str) -> None:
    df.to_excel(output_path, index=False)


def dataframe_to_csv(df: pd.DataFrame, output_path: str) -> None:
    df.to_csv(output_path, index=False)


def dataframe_to_pdf(df: pd.DataFrame, output_path: str) -> None:
    from xhtml2pdf import pisa

    html = df.to_html(index=False)
    pdf_buffer = BytesIO()
    pisa.CreatePDF(html, dest=pdf_buffer)
    with open(output_path, "wb") as fh:
        fh.write(pdf_buffer.getvalue())
