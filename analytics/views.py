import json
import os
import tempfile
from datetime import datetime, timezone

from django.http import HttpResponse
from django.shortcuts import redirect, render

from .forms import UploadCSVForm
from .utils import (
    analyze_dataframe,
    analyze_large_file,
    compute_business_metrics,
    dataframe_to_csv,
    dataframe_to_excel,
    dataframe_to_pdf,
    generate_chart_data,
    generate_recommendations,
    load_dataframe_from_path,
    make_json_safe,
    read_uploaded_file,
    save_dataframe_to_temp,
    save_uploaded_to_disk,
)


def _save_analysis_result_to_mongodb(summary):
    uri = os.environ.get('MONGODB_URI')
    if not uri:
        return

    try:
        from pymongo import MongoClient

        client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        db_name = os.environ.get('MONGODB_DB', 'data_analysis_dashboard')
        client[db_name]['analysis_results'].insert_one(
            {
                'created_at': datetime.now(timezone.utc),
                'row_count': summary.get('row_count'),
                'column_count': summary.get('column_count'),
                'quality_score': summary.get('quality_score'),
                'numeric_columns': summary.get('numeric_columns', []),
                'categorical_columns': summary.get('categorical_columns', []),
                'insights': summary.get('insights', []),
                'prediction': summary.get('prediction'),
                'trend': summary.get('trend'),
            }
        )
    except Exception:
        return


def _empty_context(form, error_message=""):
    return {
        'form': form,
        'summary': None,
        'preview_columns': [],
        'preview_rows': [],
        'correlation_preview': '',
        'chart_data_json': '{}',
        'recommendations': [],
        'business_metrics': {},
        'error_message': error_message,
    }


def _analysis_context(form, summary):
    cleaned_df = summary['cleaned_df']
    preview_records = make_json_safe(cleaned_df.head(10).to_dict(orient='records'))

    correlation_preview = ''
    if summary.get('correlation'):
        correlation_preview = json.dumps(make_json_safe(summary['correlation']), indent=2)

    chart_data = make_json_safe(generate_chart_data(cleaned_df, summary.get('numeric_columns', [])))
    recommendations = make_json_safe(
        generate_recommendations(summary, cleaned_df, summary.get('target_column'))
    )
    business_metrics = make_json_safe(compute_business_metrics(summary, cleaned_df))
    safe_summary = {
        key: make_json_safe(value)
        for key, value in summary.items()
        if key != 'cleaned_df'
    }

    return {
        'form': form,
        'summary': safe_summary,
        'preview_columns': cleaned_df.columns.tolist(),
        'preview_rows': [list(row.values()) for row in preview_records],
        'correlation_preview': correlation_preview,
        'chart_data_json': json.dumps(chart_data),
        'recommendations': recommendations,
        'business_metrics': business_metrics,
        'error_message': '',
    }


def _remember_dataframe(request, df):
    old_path = request.session.get('analysis_file_path')
    if old_path and os.path.exists(old_path):
        try:
            os.remove(old_path)
        except OSError:
            pass

    try:
        request.session['analysis_file_path'] = save_dataframe_to_temp(df)
    except OSError:
        request.session.pop('analysis_file_path', None)


def dashboard(request):
    form = UploadCSVForm(request.POST or None, request.FILES or None)

    if request.method == 'POST' and not form.is_valid():
        return render(
            request,
            'analytics/dashboard.html',
            _empty_context(form, 'Please choose a valid CSV or Excel file before analyzing.'),
        )

    if request.method == 'POST':
        uploaded_file = request.FILES['csv_file']
        target_column = form.cleaned_data.get('target_column') or None
        _, ext = os.path.splitext(uploaded_file.name.lower())
        large_threshold = 50 * 1024 * 1024

        if uploaded_file.size and uploaded_file.size > large_threshold:
            if ext in ('.xls', '.xlsx'):
                return render(
                    request,
                    'analytics/dashboard.html',
                    _empty_context(
                        form,
                        'Large Excel files are not supported. Please convert to CSV and re-upload.',
                    ),
                )

            try:
                disk_path = save_uploaded_to_disk(uploaded_file)
                summary = analyze_large_file(disk_path)
            except Exception as exc:
                return render(
                    request,
                    'analytics/dashboard.html',
                    _empty_context(form, f'Failed to analyze large file: {exc}'),
                )
        else:
            try:
                df = read_uploaded_file(uploaded_file)
                summary = analyze_dataframe(df, target_column)
            except Exception as exc:
                return render(
                    request,
                    'analytics/dashboard.html',
                    _empty_context(form, f'Failed to analyze file: {exc}'),
                )

        _remember_dataframe(request, summary['cleaned_df'])
        _save_analysis_result_to_mongodb(summary)
        return render(request, 'analytics/dashboard.html', _analysis_context(form, summary))

    file_path = request.session.get('analysis_file_path')
    if file_path:
        try:
            df = load_dataframe_from_path(file_path)
            summary = analyze_dataframe(df)
            return render(request, 'analytics/dashboard.html', _analysis_context(form, summary))
        except Exception:
            request.session.pop('analysis_file_path', None)

    return render(request, 'analytics/dashboard.html', _empty_context(form))


def _get_session_dataframe(request):
    file_path = request.session.get('analysis_file_path')
    if not file_path or not os.path.exists(file_path):
        return None
    return load_dataframe_from_path(file_path)


def download_csv(request):
    df = _get_session_dataframe(request)
    if df is None:
        return redirect('dashboard')
    temp_path = tempfile.NamedTemporaryFile(suffix='.csv', delete=False).name
    dataframe_to_csv(df, temp_path)
    with open(temp_path, 'rb') as fh:
        response = HttpResponse(fh.read(), content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="analysis.csv"'
    os.remove(temp_path)
    return response


def download_excel(request):
    df = _get_session_dataframe(request)
    if df is None:
        return redirect('dashboard')
    temp_path = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False).name
    dataframe_to_excel(df, temp_path)
    with open(temp_path, 'rb') as fh:
        response = HttpResponse(
            fh.read(),
            content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        )
        response['Content-Disposition'] = 'attachment; filename="analysis.xlsx"'
    os.remove(temp_path)
    return response


def download_pdf(request):
    df = _get_session_dataframe(request)
    if df is None:
        return redirect('dashboard')
    temp_path = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False).name
    dataframe_to_pdf(df, temp_path)
    with open(temp_path, 'rb') as fh:
        response = HttpResponse(fh.read(), content_type='application/pdf')
        response['Content-Disposition'] = 'attachment; filename="analysis.pdf"'
    os.remove(temp_path)
    return response
