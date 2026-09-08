import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class AppSmokeTests(unittest.TestCase):
    def test_demo_opens_and_analyzes(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=15).run()
        self.assertFalse(app.exception)
        self.assertTrue(app.info)

        app.toggle[1].set_value(False).run(timeout=15)
        app.toggle[0].set_value(True).run(timeout=15)
        app.button[0].click().run(timeout=15)

        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value, "10")
        self.assertEqual(app.metric[1].value, "3")
        self.assertTrue(app.error)


if __name__ == "__main__":
    unittest.main()
