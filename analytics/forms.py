from django import forms


class UploadCSVForm(forms.Form):
    csv_file = forms.FileField(label="Upload CSV or Excel")
    # Removed target column input; predictions are generated automatically by the server
