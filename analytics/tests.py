from io import BytesIO

from django.test import Client, SimpleTestCase, TestCase
from django.core.files.uploadedfile import SimpleUploadedFile
import pandas as pd

from analytics.utils import analyze_dataframe


class AnalyzeDataframeTests(SimpleTestCase):
    def test_analyze_dataframe_returns_summary(self):
        df = pd.DataFrame(
            {
                "Name": ["Alice", "Alice", "Bob"],
                "Age": [25, None, 30],
                "Salary": [1000, 1200, 1500],
                "City": ["New York", "Los Angeles", "New York"],
            }
        )

        summary = analyze_dataframe(df, "Salary")

        self.assertEqual(summary["row_count"], 3)
        self.assertEqual(summary["column_count"], 4)
        self.assertGreaterEqual(summary["quality_score"], 0)
        self.assertIn("insights", summary)
        self.assertIn("kpis", summary)

    def test_analyze_dataframe_exposes_weakness_analysis(self):
        df = pd.DataFrame(
            {
                "Product": ["A", "A", "B"],
                "Sales": [100, None, 120],
                "Region": ["North", "South", "North"],
            }
        )

        summary = analyze_dataframe(df, "Sales")

        self.assertIn("weaknesses", summary)
        self.assertGreaterEqual(len(summary["weaknesses"]), 1)
        self.assertIn("recommendations", summary)
        self.assertGreaterEqual(len(summary["recommendations"]), 1)


class DashboardViewTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_dashboard_post_renders_successfully(self):
        csv_data = "sales,profit,product\n100,10,A\n200,20,B\n150,15,A\n"
        upload = SimpleUploadedFile("test.csv", csv_data.encode("utf-8"), content_type="text/csv")
        response = self.client.post("/", {"csv_file": upload})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "trendChart")
        self.assertContains(response, "chartData =")

    def test_dashboard_post_reads_excel_file(self):
        df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
        buffer = BytesIO()
        df.to_excel(buffer, index=False)
        buffer.seek(0)
        upload = SimpleUploadedFile(
            "test.xlsx",
            buffer.read(),
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        response = self.client.post("/", {"csv_file": upload})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "trendChart")
        self.assertContains(response, "chartData =")
