import unittest
from unittest.mock import patch

from PIL import Image

from led_inspector.ollama import analyze_with_vision, list_vision_models


class OllamaClientTests(unittest.TestCase):
    @patch("led_inspector.ollama._request_json")
    def test_lists_only_vision_models(self, request_json):
        request_json.return_value = {
            "models": [
                {"name": "text-only", "capabilities": ["completion"]},
                {"name": "vision-model", "capabilities": ["vision", "completion"]},
            ]
        }

        self.assertEqual(list_vision_models(), ["vision-model"])

    @patch("led_inspector.ollama._request_json")
    def test_parses_structured_inspection(self, request_json):
        request_json.return_value = {
            "message": {
                "content": (
                    '{"verdict":"OK","confidence":96,"summary":"Peça aprovada",'
                    '"suspect_leds":[],"observations":["LEDs equivalentes"]}'
                )
            }
        }
        image = Image.new("RGB", (32, 32), "white")

        result = analyze_with_vision(image, image, "vision-model", [])

        self.assertEqual(result.verdict, "OK")
        self.assertEqual(result.confidence, 96.0)
        self.assertEqual(result.suspect_leds, [])

    @patch("led_inspector.ollama._request_json")
    def test_normalizes_fractional_confidence(self, request_json):
        request_json.return_value = {
            "message": {
                "content": (
                    '{"verdict":"REVISAR","confidence":0.9,"summary":"Falha visual",'
                    '"suspect_leds":["LD1"],"observations":[]}'
                )
            }
        }
        image = Image.new("RGB", (32, 32), "white")

        result = analyze_with_vision(image, image, "vision-model", [])

        self.assertEqual(result.confidence, 90.0)

    @patch("led_inspector.ollama._request_json")
    def test_uses_physical_integrity_prompt(self, request_json):
        request_json.return_value = {
            "message": {
                "content": (
                    '{"verdict":"REJEITADO","confidence":0.95,"summary":"Rachadura",'
                    '"suspect_leds":["LD1"],"observations":["Rachadura no fósforo"]}'
                )
            }
        }
        image = Image.new("RGB", (32, 32), "white")

        result = analyze_with_vision(image, image, "vision-model", [])

        prompt = request_json.call_args.args[1]["messages"][0]["content"]
        self.assertEqual(result.verdict, "REJEITADO")
        self.assertIn("INSPEÇÃO DE INTEGRIDADE DO LED", prompt)
        self.assertIn("rachadura no fósforo", prompt)
        self.assertIn("OK / REJEITADO", prompt)


if __name__ == "__main__":
    unittest.main()
