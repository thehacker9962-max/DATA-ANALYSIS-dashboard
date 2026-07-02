import json
import os
import tempfile
from io import StringIO

import pandas as pd
from django.http import HttpResponse
from django.shortcuts import redirect, render

from .forms import UploadCSVForm
from .utils import (
    analyze_dataframe,
    compute_business_metrics,
    dataframe_to_csv,
    dataframe_to_excel,
    dataframe_to_pdf,
    generate_chart_data,
    generate_recommendations,
    read_uploaded_file,
    save_uploaded_to_disk,
    analyze_large_file,
    save_dataframe_to_temp,
    load_dataframe_from_path,
)


def dashboard(request):
    form = UploadCSVForm(request.POST or None, request.FILES or None)
    summary = None
    preview_columns = []
    preview_rows = []
    correlation_preview = ""

    if request.method == 'POST' and form.is_valid():
        uploaded_file = request.FILES['csv_file']
        # If file is large, prefer streaming CSV processing; Excel files cannot be streamed reliably
        large_threshold = 50 * 1024 * 1024  # 50 MB
        _, ext = os.path.splitext(uploaded_file.name.lower())
        if hasattr(uploaded_file, 'size') and uploaded_file.size and uploaded_file.size > large_threshold:
            if ext in ('.xls', '.xlsx'):
                # Large Excel uploads are not supported for streaming; ask user to convert to CSV
                context = {
                    'form': form,
                    'summary': None,
                    'preview_columns': [],
                    'preview_rows': [],
                    'correlation_preview': '',
                    'chart_data_json': '{}',
                    'recommendations': [],
                    'business_metrics': {},
                    'error_message': 'Large Excel files are not supported. Please convert to CSV and re-upload (or upload smaller Excel files).',
                }
                return render(request, 'analytics/dashboard.html', context)

            try:
                disk_path = save_uploaded_to_disk(uploaded_file)
                summary = analyze_large_file(disk_path)
            except Exception as e:
                context = {
                    'form': form,
                    'summary': None,
                    'preview_columns': [],
                    'preview_rows': [],
                    'correlation_preview': '',
                    'chart_data_json': '{}',
                    'recommendations': [],
                    'business_metrics': {},
                    'error_message': f'Failed to analyze large file: {e}',
                }
                return render(request, 'analytics/dashboard.html', context)

            # ensure cleaned_df persisted to disk for session
            if request.session.get('analysis_file_path'):
                try:
                    old_path = request.session.get('analysis_file_path')
                    if old_path and os.path.exists(old_path):
                        os.remove(old_path)
                except Exception:
                    pass
            file_path = save_dataframe_to_temp(summary['cleaned_df'])
            request.session['analysis_file_path'] = file_path
        else:
            df = read_uploaded_file(uploaded_file)
            summary = analyze_dataframe(df)

        if request.session.get('analysis_file_path'):
            try:
                old_path = request.session.get('analysis_file_path')
                if old_path and os.path.exists(old_path):
                    os.remove(old_path)
            except Exception:
                pass

        file_path = save_dataframe_to_temp(summary['cleaned_df'])
        request.session['analysis_file_path'] = file_path
        request.session['analysis_columns'] = summary['cleaned_df'].columns.tolist()
        request.session['analysis_preview'] = summary['cleaned_df'].head(10).to_dict(orient='records')
        request.session['analysis_summary'] = {
            key: value
            for key, value in summary.items()
            if key != 'cleaned_df'
        }

        preview_columns = request.session['analysis_columns']
        preview_rows = [list(row.values()) for row in request.session['analysis_preview']]
        if summary['correlation']:
            correlation_preview = json.dumps(summary['correlation'], indent=2)
        chart_data = generate_chart_data(summary['cleaned_df'], summary.get('numeric_columns', []))
        chart_data_json = json.dumps(chart_data)
        request.session['analysis_chart_data'] = chart_data
        request.session['analysis_recommendations'] = generate_recommendations(summary, summary['cleaned_df'], None)
        request.session['analysis_business_metrics'] = compute_business_metrics(summary, summary['cleaned_df'])
        recommendations = request.session['analysis_recommendations']
        business_metrics = request.session['analysis_business_metrics']

    elif request.session.get('analysis_file_path') and request.session.get('analysis_summary'):
        preview_columns = request.session.get('analysis_columns', [])
        preview_rows = [list(row.values()) for row in request.session.get('analysis_preview', [])]
        summary = request.session.get('analysis_summary')
        if summary.get('correlation'):
            correlation_preview = json.dumps(summary['correlation'], indent=2)
        chart_data_json = json.dumps(request.session.get('analysis_chart_data', {}))
        recommendations = request.session.get('analysis_recommendations', [])
        business_metrics = request.session.get('analysis_business_metrics', {})

    context = {
        'form': form,
        'summary': summary,
        'preview_columns': preview_columns,
        'preview_rows': preview_rows,
        'correlation_preview': correlation_preview,
        'chart_data_json': chart_data_json if 'chart_data_json' in locals() else '{}',
        'recommendations': recommendations if 'recommendations' in locals() else [],
        'business_metrics': business_metrics if 'business_metrics' in locals() else {},
    }
    return render(request, 'analytics/dashboard.html', context)


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
        response = HttpResponse(fh.read(), content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
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
