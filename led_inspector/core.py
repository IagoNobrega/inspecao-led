from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class LedRegion:
    led_id: str
    x: int
    y: int
    width: int
    height: int

    def validated(self, image_size: tuple[int, int]) -> "LedRegion":
        image_width, image_height = image_size
        if not self.led_id.strip():
            raise ValueError("Todo LED precisa ter um identificador.")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"{self.led_id}: largura e altura devem ser positivas.")
        if self.x < 0 or self.y < 0:
            raise ValueError(f"{self.led_id}: x e y não podem ser negativos.")
        if self.x + self.width > image_width or self.y + self.height > image_height:
            raise ValueError(f"{self.led_id}: região ultrapassa os limites da imagem.")
        return self


@dataclass(frozen=True)
class InspectionResult:
    led_id: str
    status: str
    score: float
    difference: float
    edge_difference: float
    dark_difference: float
    diagnosis: str
    region: LedRegion


def generate_grid_regions(
    image_size: tuple[int, int],
    rows: int,
    columns: int,
    bounds: tuple[int, int, int, int] | None = None,
    fill_ratio: float = 0.58,
) -> list[LedRegion]:
    """Cria regiões LD1...LDn, em ordem da esquerda para a direita."""
    if rows < 1 or columns < 1:
        raise ValueError("A grade precisa ter ao menos uma linha e uma coluna.")
    if not 0.1 <= fill_ratio <= 1.0:
        raise ValueError("fill_ratio deve estar entre 0.1 e 1.0.")

    image_width, image_height = image_size
    left, top, right, bottom = bounds or (0, 0, image_width, image_height)
    if not (0 <= left < right <= image_width and 0 <= top < bottom <= image_height):
        raise ValueError("Os limites da grade são inválidos.")

    cell_width = (right - left) / columns
    cell_height = (bottom - top) / rows
    roi_width = max(2, int(round(cell_width * fill_ratio)))
    roi_height = max(2, int(round(cell_height * fill_ratio)))
    regions: list[LedRegion] = []
    index = 1
    for row in range(rows):
        for column in range(columns):
            center_x = left + (column + 0.5) * cell_width
            center_y = top + (row + 0.5) * cell_height
            x = max(0, int(round(center_x - roi_width / 2)))
            y = max(0, int(round(center_y - roi_height / 2)))
            width = min(roi_width, image_width - x)
            height = min(roi_height, image_height - y)
            regions.append(LedRegion(f"LD{index}", x, y, width, height))
            index += 1
    return regions


def _rgb_array(image: Image.Image, size: tuple[int, int] | None = None) -> np.ndarray:
    converted = image.convert("RGB")
    if size is not None and converted.size != size:
        converted = converted.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(converted, dtype=np.float32)


def _gray(array: np.ndarray) -> np.ndarray:
    return array[..., 0] * 0.299 + array[..., 1] * 0.587 + array[..., 2] * 0.114


def _best_translation(reference: np.ndarray, candidate: np.ndarray, max_shift: int) -> tuple[int, int]:
    reference_gray = _gray(reference)
    candidate_gray = _gray(candidate)
    height, width = reference_gray.shape
    sample_step = max(1, max(height, width) // 300)
    ref = reference_gray[::sample_step, ::sample_step]
    test = candidate_gray[::sample_step, ::sample_step]
    scaled_shift = max(1, max_shift // sample_step)
    best_error = float("inf")
    best = (0, 0)

    for dy in range(-scaled_shift, scaled_shift + 1):
        for dx in range(-scaled_shift, scaled_shift + 1):
            y0_ref, y1_ref = max(0, dy), min(ref.shape[0], ref.shape[0] + dy)
            x0_ref, x1_ref = max(0, dx), min(ref.shape[1], ref.shape[1] + dx)
            y0_test, y1_test = max(0, -dy), min(test.shape[0], test.shape[0] - dy)
            x0_test, x1_test = max(0, -dx), min(test.shape[1], test.shape[1] - dx)
            if y1_ref <= y0_ref or x1_ref <= x0_ref:
                continue
            error = float(
                np.mean(
                    np.abs(
                        ref[y0_ref:y1_ref, x0_ref:x1_ref]
                        - test[y0_test:y1_test, x0_test:x1_test]
                    )
                )
            )
            if error < best_error:
                best_error = error
                best = (dx * sample_step, dy * sample_step)
    return best


def _translate(array: np.ndarray, dx: int, dy: int) -> np.ndarray:
    height, width = array.shape[:2]
    fill = np.median(array.reshape(-1, 3), axis=0)
    translated = np.empty_like(array)
    translated[...] = fill
    x0_dst, x1_dst = max(0, dx), min(width, width + dx)
    y0_dst, y1_dst = max(0, dy), min(height, height + dy)
    x0_src, x1_src = max(0, -dx), min(width, width - dx)
    y0_src, y1_src = max(0, -dy), min(height, height - dy)
    translated[y0_dst:y1_dst, x0_dst:x1_dst] = array[y0_src:y1_src, x0_src:x1_src]
    return translated


def align_image(reference: Image.Image, candidate: Image.Image, max_shift: int = 12) -> Image.Image:
    """Redimensiona e corrige pequenos deslocamentos x/y da foto inspecionada."""
    ref_array = _rgb_array(reference)
    candidate_array = _rgb_array(candidate, reference.size)
    dx, dy = _best_translation(ref_array, candidate_array, max_shift=max_shift)
    return Image.fromarray(np.uint8(np.clip(_translate(candidate_array, dx, dy), 0, 255)))


def _normalize_lighting(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    # A mediana é deliberadamente usada no lugar da média/desvio padrão.
    # Assim, um componente queimado ou ausente não muda a correção aplicada
    # a todos os outros LEDs da placa.
    ref_median = np.median(reference, axis=(0, 1), keepdims=True)
    test_median = np.median(candidate, axis=(0, 1), keepdims=True)
    gain = ref_median / np.maximum(test_median, 8.0)
    adjusted = candidate * np.clip(gain, 0.4, 2.2)
    return np.clip(adjusted, 0, 255)


def _edge_strength(gray: np.ndarray) -> np.ndarray:
    horizontal = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    vertical = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    return np.hypot(horizontal, vertical)


def inspect_leds(
    reference: Image.Image,
    candidate: Image.Image,
    regions: Iterable[LedRegion],
    threshold: float = 18.0,
    max_shift: int = 12,
) -> tuple[list[InspectionResult], Image.Image]:
    """Compara LEDs com uma placa boa e devolve resultados e imagem alinhada."""
    aligned = align_image(reference, candidate, max_shift=max_shift)
    ref = _rgb_array(reference)
    test = _rgb_array(aligned)
    normalized_test = _normalize_lighting(ref, test)
    results: list[InspectionResult] = []

    for raw_region in regions:
        region = raw_region.validated(reference.size)
        x0, y0 = region.x, region.y
        x1, y1 = x0 + region.width, y0 + region.height
        ref_crop = ref[y0:y1, x0:x1]
        test_crop = normalized_test[y0:y1, x0:x1]
        ref_gray = _gray(ref_crop)
        test_gray = _gray(test_crop)

        pixel_difference = np.mean(np.abs(ref_crop - test_crop), axis=2) / 255.0
        average_difference = float(np.mean(pixel_difference))
        concentrated_difference = float(np.percentile(pixel_difference, 75))
        difference = max(average_difference, concentrated_difference)
        edge_difference = float(
            np.mean(np.abs(_edge_strength(ref_gray) - _edge_strength(test_gray))) / 255.0
        )
        reference_dark = float(np.mean(ref_gray < 65))
        candidate_dark = float(np.mean(test_gray < 65))
        dark_difference = abs(candidate_dark - reference_dark)
        score = 100.0 * (0.60 * difference + 0.25 * edge_difference + 0.15 * dark_difference)

        if score >= threshold:
            status = "SUSPEITO"
            dominant = max(
                (difference, "aparência diferente da referência"),
                (edge_difference, "estrutura ou contorno alterado"),
                (dark_difference, "possível mancha ou queimadura"),
            )[1]
            diagnosis = dominant
        else:
            status = "OK"
            diagnosis = "sem alteração visual relevante"

        results.append(
            InspectionResult(
                led_id=region.led_id,
                status=status,
                score=round(score, 2),
                difference=round(difference * 100, 2),
                edge_difference=round(edge_difference * 100, 2),
                dark_difference=round(dark_difference * 100, 2),
                diagnosis=diagnosis,
                region=region,
            )
        )
    return results, aligned


def inspect_leds_buffer(
    reference: Image.Image,
    candidates: Iterable[Image.Image],
    regions: Iterable[LedRegion],
    threshold: float = 18.0,
    max_shift: int = 12,
) -> tuple[list[InspectionResult], Image.Image]:
    """Compara vários quadros e preserva o pior resultado de cada LED."""
    candidate_images = list(candidates)
    if not candidate_images:
        raise ValueError("O buffer de inspeção está vazio.")

    frame_results: list[list[InspectionResult]] = []
    aligned_images: list[Image.Image] = []
    for candidate in candidate_images:
        results, aligned = inspect_leds(
            reference,
            candidate,
            regions,
            threshold=threshold,
            max_shift=max_shift,
        )
        frame_results.append(results)
        aligned_images.append(aligned)

    worst_by_led: dict[str, InspectionResult] = {}
    for results in frame_results:
        for result in results:
            previous = worst_by_led.get(result.led_id)
            if previous is None or result.score > previous.score:
                worst_by_led[result.led_id] = result

    selected_frame = max(
        range(len(frame_results)),
        key=lambda index: sum(item.score for item in frame_results[index]),
    )
    return list(worst_by_led.values()), aligned_images[selected_frame]


def annotate_image(image: Image.Image, results: Iterable[InspectionResult]) -> Image.Image:
    marked = image.convert("RGB").copy()
    draw = ImageDraw.Draw(marked)
    font = ImageFont.load_default()
    for result in results:
        region = result.region
        color = (225, 55, 65) if result.status == "SUSPEITO" else (32, 184, 116)
        x0, y0 = region.x, region.y
        x1, y1 = x0 + region.width, y0 + region.height
        line_width = max(2, min(marked.size) // 250)
        draw.rectangle((x0, y0, x1, y1), outline=color, width=line_width)
        label = f"{result.led_id} {result.score:.1f}"
        box = draw.textbbox((x0, y0), label, font=font)
        label_height = box[3] - box[1] + 6
        label_width = box[2] - box[0] + 8
        label_top = max(0, y0 - label_height)
        draw.rectangle((x0, label_top, x0 + label_width, y0), fill=color)
        draw.text((x0 + 4, label_top + 2), label, fill="white", font=font)
    return marked


def generate_demo_images() -> tuple[Image.Image, Image.Image]:
    """Gera uma placa sintética para conhecer o fluxo sem fotografias reais."""
    size = (900, 420)
    reference = Image.new("RGB", size, (22, 83, 64))
    draw = ImageDraw.Draw(reference)
    draw.rounded_rectangle((28, 28, 872, 392), radius=24, outline=(185, 198, 154), width=5)
    centers = [(130 + column * 160, 135 + row * 150) for row in range(2) for column in range(5)]
    for index, (x, y) in enumerate(centers, start=1):
        draw.ellipse((x - 40, y - 40, x + 40, y + 40), fill=(205, 208, 190), outline=(70, 73, 66), width=5)
        draw.ellipse((x - 24, y - 24, x + 24, y + 24), fill=(238, 198, 62), outline=(250, 238, 160), width=4)
        draw.line((x - 50, y, x - 40, y), fill=(205, 181, 92), width=8)
        draw.line((x + 40, y, x + 50, y), fill=(205, 181, 92), width=8)
        draw.text((x - 13, y + 48), f"LD{index}", fill=(236, 242, 220))

    candidate = reference.copy()
    damaged = ImageDraw.Draw(candidate)
    for index in (1, 2, 6):
        x, y = centers[index - 1]
        damaged.ellipse((x - 25, y - 25, x + 25, y + 25), fill=(35, 31, 27))
        damaged.line((x - 15, y - 15, x + 18, y + 12), fill=(125, 70, 40), width=9)
    return reference, candidate
