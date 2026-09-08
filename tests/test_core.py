import unittest

from PIL import Image, ImageDraw

from led_inspector.core import (
    LedRegion,
    align_image,
    generate_grid_regions,
    inspect_leds,
    inspect_leds_buffer,
)


class GridTests(unittest.TestCase):
    def test_grid_ids_and_bounds(self):
        regions = generate_grid_regions((600, 300), 2, 3, bounds=(30, 20, 570, 280))
        self.assertEqual([region.led_id for region in regions], ["LD1", "LD2", "LD3", "LD4", "LD5", "LD6"])
        for region in regions:
            region.validated((600, 300))

    def test_invalid_region_is_rejected(self):
        with self.assertRaises(ValueError):
            LedRegion("LD1", 90, 90, 20, 20).validated((100, 100))


class InspectionTests(unittest.TestCase):
    def setUp(self):
        self.reference = Image.new("RGB", (180, 80), (20, 80, 55))
        draw = ImageDraw.Draw(self.reference)
        draw.ellipse((20, 20, 60, 60), fill=(230, 205, 70))
        draw.ellipse((120, 20, 160, 60), fill=(230, 205, 70))
        self.regions = [LedRegion("LD1", 15, 15, 50, 50), LedRegion("LD2", 115, 15, 50, 50)]

    def test_identical_image_passes(self):
        results, _ = inspect_leds(self.reference, self.reference.copy(), self.regions, threshold=5)
        self.assertTrue(all(item.status == "OK" for item in results))

    def test_damaged_led_is_flagged(self):
        candidate = self.reference.copy()
        ImageDraw.Draw(candidate).ellipse((20, 20, 60, 60), fill=(15, 15, 15))
        results, _ = inspect_leds(self.reference, candidate, self.regions, threshold=5)
        self.assertEqual(results[0].status, "SUSPEITO")
        self.assertEqual(results[1].status, "OK")

    def test_concentrated_damage_is_not_hidden_by_neighbor(self):
        reference = Image.new("RGB", (240, 90), (20, 80, 55))
        draw = ImageDraw.Draw(reference)
        draw.rectangle((20, 20, 80, 70), fill=(230, 205, 70))
        draw.rectangle((140, 20, 200, 70), fill=(230, 205, 70))
        candidate = reference.copy()
        ImageDraw.Draw(candidate).rectangle((20, 20, 80, 70), fill=(15, 15, 15))

        results, _ = inspect_leds(
            reference,
            candidate,
            [LedRegion("LD1", 10, 10, 100, 70)],
            threshold=18,
        )

        self.assertEqual(results[0].status, "SUSPEITO")

    def test_small_translation_is_corrected(self):
        shifted = Image.new("RGB", self.reference.size, (20, 80, 55))
        shifted.paste(self.reference, (3, 2))
        aligned = align_image(self.reference, shifted, max_shift=6)
        results, _ = inspect_leds(self.reference, aligned, self.regions, threshold=8, max_shift=0)
        self.assertTrue(all(item.status == "OK" for item in results))

    def test_buffer_keeps_defect_seen_in_one_frame(self):
        damaged = self.reference.copy()
        ImageDraw.Draw(damaged).ellipse((20, 20, 60, 60), fill=(15, 15, 15))

        results, _ = inspect_leds_buffer(
            self.reference,
            [self.reference.copy(), damaged],
            self.regions,
            threshold=5,
        )

        self.assertEqual(results[0].status, "SUSPEITO")
        self.assertEqual(results[1].status, "OK")


if __name__ == "__main__":
    unittest.main()
