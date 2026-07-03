import os
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from io import BytesIO


def make_json_safe(value: Any) -> Any:
    """Convert pandas/numpy values into JSON/session-safe Python values."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [make_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if pd.isna(value):
        return None
    return value

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

    for col in cleaned.columns:
        try:
            if cleaned[col].dtype == "object":
                cleaned[col] = cleaned[col].astype(str).str.strip()
            elif pd.api.types.is_datetime64_any_dtype(cleaned[col]):
                cleaned[col] = cleaned[col].astype(str).str.strip()
        except Exception:
            cleaned[col] = cleaned[col].astype(str)

    try:
        cleaned = cleaned.replace({np.nan: None})
        cleaned = cleaned.drop_duplicates()
    except Exception:
        cleaned = cleaned.fillna(value=None)

    for col in cleaned.columns:
        try:
            if cleaned[col].dtype == "object":
                cleaned[col] = cleaned[col].replace({"": None, "nan": None, "None": None})
                numeric_candidate = cleaned[col].astype(str).str.replace(",", "", regex=False)
                converted_numeric = pd.to_numeric(numeric_candidate, errors="coerce")
                if converted_numeric.notna().mean() >= 0.6:
                    cleaned[col] = converted_numeric
                    continue

                converted_date = pd.to_datetime(cleaned[col], errors="coerce")
                if converted_date.notna().mean() >= 0.8:
                    cleaned[col] = converted_date
        except Exception:
            continue

    return cleaned


def choose_prediction_target(df: pd.DataFrame, target_column: Optional[str] = None) -> Optional[str]:
    """Return a numeric target column for regression, preferring user and business fields."""
    numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if target_column:
        normalized = target_column.strip().lower().replace(" ", "_")
        if normalized in numeric_columns:
            return normalized

    for candidate in ["sales", "revenue", "profit", "amount", "total", "price"]:
        if candidate in numeric_columns:
            return candidate

    return numeric_columns[-1] if numeric_columns else None


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
    try:
        cleaned = clean_dataframe(df)
    except Exception:
        cleaned = df.copy()
        cleaned = cleaned.astype(str)

    try:
        quality = build_quality_report(cleaned)
    except Exception:
        quality = {
            "quality_score": 0,
            "null_values": {},
            "duplicate_rows": 0,
            "row_count": int(len(cleaned)),
            "column_count": int(len(cleaned.columns)),
        }

    try:
        numeric_columns = [col for col in cleaned.columns if pd.api.types.is_numeric_dtype(cleaned[col])]
        categorical_columns = [col for col in cleaned.columns if col not in numeric_columns]
        target_column = choose_prediction_target(cleaned, target_column)
    except Exception:
        numeric_columns = []
        categorical_columns = list(cleaned.columns)
        target_column = None

    kpis = {
        "rows": quality["row_count"],
        "columns": quality["column_count"],
        "duplicates": quality["duplicate_rows"],
        "missing_values": int(sum(quality["null_values"].values())) if quality.get("null_values") else 0,
        "quality_score": quality["quality_score"],
    }

    insights = []
    chart_data = []
    try:
        if numeric_columns:
            insights.append(f"Numeric columns detected: {', '.join(numeric_columns[:5])}")
            if target_column and target_column in numeric_columns:
                insights.append("Target column available for predictive analysis.")
            for col in numeric_columns[:6]:
                series = cleaned[col].dropna()
                if not series.empty:
                    chart_data.append({"name": col, "value": round(float(series.mean()), 2)})
        if categorical_columns:
            insights.append(f"Categorical columns detected: {', '.join(categorical_columns[:5])}")
    except Exception:
        insights.append("Basic inspection completed with fallback handling.")

    trend = None
    try:
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
    except Exception:
        trend = None

    correlation = None
    try:
        if len(numeric_columns) >= 2:
            corr_cols = numeric_columns[:10]
            corr_matrix = cleaned[corr_cols].corr(numeric_only=True)
            corr_matrix = corr_matrix.fillna(0)
            correlation = corr_matrix.to_dict()
            if len(numeric_columns) > 10:
                insights.append(
                    "Correlation computed for first 10 numeric columns only to preserve performance."
                )
    except Exception:
        correlation = None

    prediction = None
    try:
        if target_column and target_column in cleaned.columns and pd.api.types.is_numeric_dtype(cleaned[target_column]):
            feature_frame = cleaned.select_dtypes(include=[np.number]).drop(columns=[target_column], errors="ignore")
            if not feature_frame.empty and len(feature_frame.dropna()) > 5:
                model_df = cleaned[[target_column] + list(feature_frame.columns)].dropna()
                X = model_df[feature_frame.columns]
                y = model_df[target_column]
                if len(model_df) >= 8:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    mse = mean_squared_error(y_test, preds)
                    prediction = {
                        "model": "LinearRegression",
                        "target": target_column,
                        "features": list(feature_frame.columns)[:8],
                        "mse": round(float(mse), 4),
                        "sample_prediction": round(float(preds[0]), 4) if len(preds) else None,
                        "r2_hint": "Model trained with available numeric features.",
                    }
    except Exception:
        prediction = None

    try:
        weaknesses = detect_dataset_weaknesses(
            {
                "quality_report": quality,
                "numeric_columns": numeric_columns,
                "categorical_columns": categorical_columns,
                "trend": trend,
                "correlation": correlation,
                "quality_score": quality["quality_score"],
                "target_column": target_column,
            },
            cleaned,
            target_column,
        )
        recommendations = generate_recommendations(
            {
                "quality_report": quality,
                "numeric_columns": numeric_columns,
                "categorical_columns": categorical_columns,
                "trend": trend,
                "correlation": correlation,
                "quality_score": quality["quality_score"],
                "target_column": target_column,
                "weaknesses": weaknesses,
            },
            cleaned,
            target_column,
        )
    except Exception as exc:
        weaknesses = [f"Basic review completed with fallback handling: {exc}"]
        recommendations = ["Review the uploaded data carefully and clean obvious missing or duplicate values."]

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
        "target_column": target_column,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "chart_data": chart_data,
        "trend": trend,
        "weaknesses": weaknesses,
        "recommendations": recommendations,
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
    missing_total = sum(nulls.values()) if nulls else 0
    total_cells = max(total_rows * max(len(cols or []), 1), 1)
    missing_ratio = missing_total / total_cells
    quality_score = max(0, min(100, round(100 - (missing_ratio * 100))))

    quality = {
        "quality_score": quality_score,
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
        "kpis": {"rows": quality["row_count"], "columns": quality["column_count"], "duplicates": 0, "missing_values": missing_total, "quality_score": quality["quality_score"]},
        "insights": insights,
        "correlation": correlation,
        "prediction": None,
        "target_column": None,
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

    try:
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

        if len(numeric_columns) >= 2:
            corr = df[numeric_columns].corr(numeric_only=True).fillna(0)
            chart_data['correlation'] = {
                'labels': list(corr.columns),
                'matrix': corr.round(2).to_dict(),
            }
    except Exception:
        return {}

    return chart_data

def detect_dataset_weaknesses(summary: Dict[str, Any], df: pd.DataFrame, target: Optional[str] = None) -> List[str]:
    weaknesses: List[str] = []
    q = summary.get('quality_report', {})
    if q.get('duplicate_rows', 0) > 0:
        weaknesses.append('Duplicate rows are present and can distort the analysis.')

    null_values = sum(q.get('null_values', {}).values()) if q.get('null_values') else 0
    if null_values > 0:
        null_columns = [name for name, count in q.get('null_values', {}).items() if count]
        if null_columns:
            weaknesses.append(f'Missing values were found in {", ".join(null_columns[:5])}.')

    if not summary.get('numeric_columns'):
        weaknesses.append('No numeric columns were detected, which limits trend and forecasting analysis.')
    elif not target or target not in summary.get('numeric_columns', []):
        weaknesses.append('A clear numeric target column was not identified, so predictive modeling may be weak.')

    trend = summary.get('trend')
    if trend == 'decreasing':
        weaknesses.append('The main metric is trending downward, suggesting a performance decline that needs investigation.')
    elif trend == 'stable':
        weaknesses.append('The main metric is flat, so the dataset may need segmentation or stronger driver variables to reveal opportunities.')

    if summary.get('quality_score', 100) < 70:
        weaknesses.append('Overall data quality is below the threshold needed for confident business decisions.')

    if not summary.get('correlation'):
        weaknesses.append('The correlation structure is weak or unavailable, so the key business drivers are not yet clear.')

    return weaknesses


def generate_recommendations(summary: Dict[str, Any], df: pd.DataFrame, target: Optional[str] = None) -> List[str]:
    recs: List[str] = []
    weaknesses = summary.get('weaknesses', []) or detect_dataset_weaknesses(summary, df, target)

    business_columns = [col for col in ["sales", "revenue", "profit", "amount", "total", "price"] if col in df.columns]
    if business_columns:
        recs.append(f'Use {", ".join(business_columns[:3])} as the primary business KPI to guide decisions and monitor performance.')

    if any('Missing values' in weakness for weakness in weaknesses):
        recs.append('Fill missing values with a business-safe default or remove incomplete rows before drawing conclusions.')
    if any('Duplicate rows' in weakness for weakness in weaknesses):
        recs.append('Remove duplicate records so summaries and forecasts reflect the true business pattern.')
    if any('No numeric columns' in weakness for weakness in weaknesses):
        recs.append('Convert the key business fields to numeric values to unlock trend charts and actionable insights.')
    if any('clear numeric target' in weakness for weakness in weaknesses):
        recs.append('Choose a numeric outcome column such as sales, revenue, profit, or amount to improve forecasting quality.')
    if any('trending downward' in weakness for weakness in weaknesses):
        recs.append('Review pricing, promotions, and segment mix to recover sales momentum and stop the decline.')
    if any('flat' in weakness for weakness in weaknesses):
        recs.append('Segment the data by product, region, or customer group and test promotions to uncover hidden growth opportunities.')
    if any('quality is below' in weakness for weakness in weaknesses):
        recs.append('Clean and standardize the dataset before making operational decisions from the report.')
    if any('correlation structure' in weakness for weakness in weaknesses):
        recs.append('Add richer driver variables such as price, discount, seasonality, or marketing spend to improve explanation.')

    if any(col in df.columns for col in ['product', 'product_name', 'sku']) and any(col in df.columns for col in ['sales', 'revenue', 'amount', 'total']):
        recs.append('Analyze sales by product category and focus promotions on the segments that generate the highest value.')
    if any(col in df.columns for col in ['region', 'city', 'country']) and any(col in df.columns for col in ['sales', 'revenue', 'amount', 'total']):
        recs.append('Compare regional performance to identify where sales growth or losses are concentrated.')

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
        recs.append('The main numeric trend is increasing. Consider scaling inventory, budget, or capacity around the strongest segments.')
    elif trend == 'decreasing':
        recs.append('The main numeric trend is decreasing. Investigate pricing, promotions, operational changes, and recent demand shifts.')
    else:
        recs.append('Sales appear stable; run A/B tests on promotions to find uplift opportunities.')

    # Correlation-based suggestions
    corr = summary.get('correlation')
    if isinstance(corr, dict) and target:
        if target in corr:
            pairs = corr[target]
            sorted_feats = sorted(pairs.items(), key=lambda kv: abs(kv[1]), reverse=True)
            top = [f for f, v in sorted_feats if f != target][:3]
            if top:
                recs.append(f'Features strongly associated with {target}: {", ".join(top)}. Use these as candidate business drivers.')

    # Generic actions
    if not recs:
        recs.append('Run targeted promotions for top-performing segments and monitor lift.')
    recs.append('Use top correlated numeric features as candidate predictors for forecasting models.')

    return recs


def compute_business_metrics(summary: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    """Compute business-facing KPIs like sales rating and product performance."""
    metrics: Dict[str, Any] = {}
    try:
        q_score = summary.get('quality_score', 0)
        trend = summary.get('trend')

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

        perf_score = 0
        if top_products:
            perf_score = min(100, int(sum(p['value'] for p in top_products) / (len(top_products) * (abs(metrics.get('avg_sales') or 1))) * 10))
        metrics['product_performance_score'] = perf_score
    except Exception:
        metrics = {
            'sales_rating': 'N/A',
            'avg_sales': None,
            'top_products': [],
            'product_performance_score': 0,
        }

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
