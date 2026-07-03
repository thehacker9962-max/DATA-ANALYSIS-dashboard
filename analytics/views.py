import json
import os

from django.shortcuts import redirect, render

from .forms import UploadCSVForm
from .utils import (
    analyze_dataframe,
    compute_business_metrics,
    generate_chart_data,
    generate_recommendations,
    make_json_safe,
    read_uploaded_file,
)


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
        'downloads_available': False,
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
        'downloads_available': False,
    }


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

            return render(
                request,
                'analytics/dashboard.html',
                _empty_context(
                    form,
                    'This deployment is in no-storage mode. Please upload a CSV under 50 MB, or split a larger file before analyzing.',
                ),
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

        return render(request, 'analytics/dashboard.html', _analysis_context(form, summary))

    return render(request, 'analytics/dashboard.html', _empty_context(form))


def download_csv(request):
    return redirect('dashboard')


def download_excel(request):
    return redirect('dashboard')


def download_pdf(request):
    return redirect('dashboard')
