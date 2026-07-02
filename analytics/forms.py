from django import forms


class UploadCSVForm(forms.Form):
    csv_file = forms.FileField(label="Upload CSV or Excel")
    target_column = forms.CharField(
        label="Prediction target column",
        required=False,
        help_text="Optional. Leave blank to let the app choose a numeric target automatically.",
    )
