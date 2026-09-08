from __future__ import annotations

import os
import threading
import tkinter as tk
from dataclasses import dataclass

import customtkinter as ctk
import cv2
from PIL import Image, ImageTk

from led_inspector import (
    DEFAULT_OLLAMA_URL,
    DEFAULT_VISION_MODEL,
    LedRegion,
    OllamaError,
    analyze_with_vision,
    annotate_image,
    generate_grid_regions,
    inspect_leds_buffer,
    list_vision_models,
)


CAMERA_INTERVAL_MS = 30


@dataclass
class CanvasTransform:
    scale: float
    offset_x: int
    offset_y: int


class LedInspectorDesktop(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("LED Inspector — câmera contínua")
        self.geometry("1480x900")
        self.minsize(1180, 760)
        self.protocol("WM_DELETE_WINDOW", self.close_app)

        self.capture: cv2.VideoCapture | None = None
        self.camera_running = False
        self.current_frame: Image.Image | None = None
        self.inspection_buffer: list[Image.Image] = []
        self.golden_image: Image.Image | None = None
        self.roi: tuple[int, int, int, int] | None = None
        self.roi_start: tuple[int, int] | None = None
        self.golden_transform: CanvasTransform | None = None
        self.live_transform: CanvasTransform | None = None
        self.live_photo: ImageTk.PhotoImage | None = None
        self.golden_photo: ImageTk.PhotoImage | None = None
        self.result_photo: ImageTk.PhotoImage | None = None
        self.busy = False

        self.camera_index = tk.StringVar(value="auto")
        self.rows = tk.StringVar(value="2")
        self.columns = tk.StringVar(value="5")
        self.single_led = tk.BooleanVar(value=True)
        self.threshold = tk.StringVar(value="22")
        self.max_shift = tk.StringVar(value="15")
        self.fill_ratio = tk.StringVar(value="58")
        self.ollama_url = tk.StringVar(value=DEFAULT_OLLAMA_URL)
        self.ollama_model = tk.StringVar(value=DEFAULT_VISION_MODEL)
        self.use_llm = tk.BooleanVar(value=True)
        self.status_text = tk.StringVar(value="Inicie a câmera para começar.")
        self.verdict_text = tk.StringVar(value="AGUARDANDO")

        self._build_interface()
        self.after(CAMERA_INTERVAL_MS, self._camera_loop)
        self.after(350, self.start_camera)

    def _build_interface(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        sidebar = ctk.CTkScrollableFrame(self, width=285, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")

        ctk.CTkLabel(
            sidebar,
            text="LED Inspector",
            font=ctk.CTkFont(size=25, weight="bold"),
        ).pack(anchor="w", padx=16, pady=(18, 2))
        ctk.CTkLabel(
            sidebar,
            text="Inspeção contínua com golden e LLM",
            text_color=("gray35", "gray70"),
        ).pack(anchor="w", padx=16, pady=(0, 18))

        self._section_title(sidebar, "Câmera")
        self._entry(sidebar, "Índice da câmera", self.camera_index)
        self.camera_button = ctk.CTkButton(
            sidebar,
            text="Iniciar câmera",
            command=self.toggle_camera,
        )
        self.camera_button.pack(fill="x", padx=16, pady=6)
        ctk.CTkButton(
            sidebar,
            text="Capturar golden",
            command=self.capture_golden,
            fg_color="#B7791F",
            hover_color="#975A16",
        ).pack(fill="x", padx=16, pady=6)
        ctk.CTkButton(
            sidebar,
            text="Adicionar foto ao buffer",
            command=self.add_frame_to_buffer,
            fg_color="#2563EB",
            hover_color="#1D4ED8",
        ).pack(fill="x", padx=16, pady=(10, 4))
        ctk.CTkButton(
            sidebar,
            text="Limpar buffer",
            command=self.clear_buffer,
            fg_color="transparent",
            border_width=1,
        ).pack(fill="x", padx=16, pady=4)
        self.buffer_text = tk.StringVar(value="Buffer: 0 fotos")
        ctk.CTkLabel(
            sidebar,
            textvariable=self.buffer_text,
            text_color=("gray35", "gray70"),
        ).pack(anchor="w", padx=16, pady=(0, 8))

        self._section_title(sidebar, "Região de interesse")
        ctk.CTkLabel(
            sidebar,
            text="Depois de capturar a golden, arraste o mouse sobre ela para marcar a placa.",
            wraplength=245,
            justify="left",
            text_color=("gray35", "gray70"),
        ).pack(anchor="w", padx=16, pady=(0, 8))
        ctk.CTkSwitch(
            sidebar,
            text="Região marcada = uma única LED",
            variable=self.single_led,
        ).pack(anchor="w", padx=16, pady=6)
        self._entry(sidebar, "Linhas de LEDs", self.rows)
        self._entry(sidebar, "Colunas de LEDs", self.columns)
        self._entry(sidebar, "Limite de rejeição", self.threshold)
        self._entry(sidebar, "Correção de posição (px)", self.max_shift)
        self._entry(sidebar, "Tamanho da região do LED (%)", self.fill_ratio)

        self._section_title(sidebar, "Ollama")
        ctk.CTkSwitch(
            sidebar,
            text="Usar análise da LLM",
            variable=self.use_llm,
        ).pack(anchor="w", padx=16, pady=6)
        self._entry(sidebar, "Servidor", self.ollama_url)
        ctk.CTkLabel(sidebar, text="Modelo de visão").pack(anchor="w", padx=16, pady=(8, 2))
        self.model_menu = ctk.CTkOptionMenu(
            sidebar,
            values=[DEFAULT_VISION_MODEL],
            variable=self.ollama_model,
        )
        self.model_menu.pack(fill="x", padx=16, pady=(0, 6))
        ctk.CTkButton(
            sidebar,
            text="Conectar e buscar modelos",
            command=self.connect_ollama,
            fg_color="transparent",
            border_width=1,
        ).pack(fill="x", padx=16, pady=6)

        self.inspect_button = ctk.CTkButton(
            sidebar,
            text="INSPECIONAR QUADRO ATUAL",
            command=self.inspect_current_frame,
            height=48,
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.inspect_button.pack(fill="x", padx=16, pady=(22, 8))

        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew", padx=18, pady=14)
        main.grid_columnconfigure((0, 1), weight=1, uniform="camera")
        main.grid_rowconfigure(1, weight=3)
        main.grid_rowconfigure(3, weight=2)

        ctk.CTkLabel(
            main,
            text="Câmera ao vivo",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=4, pady=(0, 6))
        ctk.CTkLabel(
            main,
            text="Golden — arraste para marcar a região de interesse",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).grid(row=0, column=1, sticky="w", padx=4, pady=(0, 6))

        self.live_canvas = tk.Canvas(main, bg="#111827", highlightthickness=0, cursor="cross")
        self.live_canvas.grid(row=1, column=0, sticky="nsew", padx=(0, 7))
        self.golden_canvas = tk.Canvas(main, bg="#111827", highlightthickness=0, cursor="cross")
        self.golden_canvas.grid(row=1, column=1, sticky="nsew", padx=(7, 0))
        self.golden_canvas.bind("<ButtonPress-1>", self._roi_press)
        self.golden_canvas.bind("<B1-Motion>", self._roi_drag)
        self.golden_canvas.bind("<ButtonRelease-1>", self._roi_release)

        result_header = ctk.CTkFrame(main, fg_color="transparent")
        result_header.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(14, 6))
        ctk.CTkLabel(
            result_header,
            text="Último resultado",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(side="left")
        self.verdict_label = ctk.CTkLabel(
            result_header,
            textvariable=self.verdict_text,
            width=150,
            height=34,
            corner_radius=8,
            fg_color="#4B5563",
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.verdict_label.pack(side="right")

        result_frame = ctk.CTkFrame(main)
        result_frame.grid(row=3, column=0, columnspan=2, sticky="nsew")
        result_frame.grid_columnconfigure(0, weight=3)
        result_frame.grid_columnconfigure(1, weight=2)
        result_frame.grid_rowconfigure(0, weight=1)
        self.result_label = ctk.CTkLabel(result_frame, text="A imagem marcada aparecerá aqui")
        self.result_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.report_box = ctk.CTkTextbox(result_frame, wrap="word", font=ctk.CTkFont(size=13))
        self.report_box.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        self.report_box.insert("1.0", "Capture a golden, marque a região e inspecione o quadro atual.")
        self.report_box.configure(state="disabled")

        ctk.CTkLabel(
            main,
            textvariable=self.status_text,
            anchor="w",
            text_color=("gray30", "gray70"),
        ).grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 0))

    @staticmethod
    def _section_title(parent: ctk.CTkBaseClass, text: str) -> None:
        ctk.CTkLabel(
            parent,
            text=text,
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(anchor="w", padx=16, pady=(16, 6))

    @staticmethod
    def _entry(parent: ctk.CTkBaseClass, label: str, variable: tk.StringVar) -> None:
        ctk.CTkLabel(parent, text=label).pack(anchor="w", padx=16, pady=(7, 2))
        ctk.CTkEntry(parent, textvariable=variable).pack(fill="x", padx=16)

    def toggle_camera(self) -> None:
        if self.camera_running:
            self.stop_camera()
        else:
            self.start_camera()

    def _camera_candidates(self) -> list[int]:
        configured_index = self.camera_index.get().strip().lower()
        if configured_index in {"", "auto", "automático", "automatico"}:
            return list(range(5))
        try:
            return [int(configured_index)]
        except ValueError:
            self._set_status("Informe um índice inteiro (0, 1, 2...) ou use 'auto'.")
            return []

    def start_camera(self) -> None:
        camera_candidates = self._camera_candidates()
        if not camera_candidates:
            return

        self.stop_camera()
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        capture = None
        camera_index = None
        for candidate in camera_candidates:
            candidate_capture = cv2.VideoCapture(candidate, backend)
            if not candidate_capture.isOpened() and backend != cv2.CAP_ANY:
                candidate_capture.release()
                candidate_capture = cv2.VideoCapture(candidate)
            if candidate_capture.isOpened():
                capture = candidate_capture
                camera_index = candidate
                break
            candidate_capture.release()

        if capture is None or camera_index is None:
            self._set_status(
                "Nenhuma câmera foi encontrada. Conecte uma câmera, feche outros apps que a usam "
                "e confira as permissões do Windows."
            )
            return

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture = capture
        self.camera_running = True
        self.camera_index.set(str(camera_index))
        self.camera_button.configure(text="Parar câmera", fg_color="#B91C1C", hover_color="#991B1B")
        self._set_status(f"Câmera {camera_index} ativa. A visualização continuará durante toda a inspeção.")

    def stop_camera(self) -> None:
        self.camera_running = False
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if hasattr(self, "camera_button"):
            self.camera_button.configure(text="Iniciar câmera", fg_color=["#3B8ED0", "#1F6AA5"])

    def _camera_loop(self) -> None:
        if self.camera_running and self.capture is not None:
            ok, frame = self.capture.read()
            if ok:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.current_frame = Image.fromarray(rgb)
                self.live_photo, self.live_transform = self._draw_image(
                    self.live_canvas,
                    self.current_frame,
                    self.live_photo,
                    self.roi,
                )
            else:
                self._set_status("A câmera parou de entregar imagens.")
        self.after(CAMERA_INTERVAL_MS, self._camera_loop)

    @staticmethod
    def _draw_image(
        canvas: tk.Canvas,
        image: Image.Image,
        previous_photo: ImageTk.PhotoImage | None,
        roi: tuple[int, int, int, int] | None,
    ) -> tuple[ImageTk.PhotoImage, CanvasTransform]:
        del previous_photo
        canvas_width = max(320, canvas.winfo_width())
        canvas_height = max(240, canvas.winfo_height())
        scale = min(canvas_width / image.width, canvas_height / image.height)
        display_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        offset_x = (canvas_width - display_size[0]) // 2
        offset_y = (canvas_height - display_size[1]) // 2
        resized = image.resize(display_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(resized)
        canvas.delete("all")
        canvas.create_image(offset_x, offset_y, image=photo, anchor="nw")
        if roi is not None:
            left, top, right, bottom = roi
            canvas.create_rectangle(
                offset_x + left * scale,
                offset_y + top * scale,
                offset_x + right * scale,
                offset_y + bottom * scale,
                outline="#22C55E",
                width=3,
                tags="roi",
            )
        return photo, CanvasTransform(scale, offset_x, offset_y)

    def capture_golden(self) -> None:
        if self.current_frame is None:
            self._set_status("Inicie a câmera antes de capturar a golden.")
            return
        self.golden_image = self.current_frame.copy()
        self.roi = None
        self.golden_photo, self.golden_transform = self._draw_image(
            self.golden_canvas,
            self.golden_image,
            self.golden_photo,
            None,
        )
        self._set_status("Golden capturada. Arraste o mouse sobre a golden para marcar a região da placa.")

    def add_frame_to_buffer(self) -> None:
        if self.current_frame is None:
            self._set_status("Inicie a câmera antes de adicionar uma foto ao buffer.")
            return
        if len(self.inspection_buffer) >= 8:
            self._set_status("O buffer já tem 8 fotos. Limpe-o para iniciar uma nova sequência.")
            return
        self.inspection_buffer.append(self.current_frame.copy())
        self.buffer_text.set(f"Buffer: {len(self.inspection_buffer)} foto(s)")
        self._set_status(
            f"Foto {len(self.inspection_buffer)} adicionada. Adicione outras fotos ou inicie a análise."
        )

    def clear_buffer(self) -> None:
        self.inspection_buffer.clear()
        self.buffer_text.set("Buffer: 0 fotos")
        self._set_status("Buffer limpo. Posicione a peça e adicione novas fotos.")

    def _roi_press(self, event: tk.Event) -> None:
        if self.golden_image is None or self.golden_transform is None:
            return
        self.roi_start = (event.x, event.y)
        self.golden_canvas.delete("selection")

    def _roi_drag(self, event: tk.Event) -> None:
        if self.roi_start is None:
            return
        self.golden_canvas.delete("selection")
        self.golden_canvas.create_rectangle(
            self.roi_start[0],
            self.roi_start[1],
            event.x,
            event.y,
            outline="#F59E0B",
            width=3,
            dash=(7, 4),
            tags="selection",
        )

    def _roi_release(self, event: tk.Event) -> None:
        if self.roi_start is None or self.golden_image is None or self.golden_transform is None:
            return
        start_x, start_y = self.roi_start
        self.roi_start = None
        transform = self.golden_transform

        def to_image(x: int, y: int) -> tuple[int, int]:
            image_x = round((x - transform.offset_x) / transform.scale)
            image_y = round((y - transform.offset_y) / transform.scale)
            return (
                max(0, min(self.golden_image.width, image_x)),
                max(0, min(self.golden_image.height, image_y)),
            )

        x0, y0 = to_image(start_x, start_y)
        x1, y1 = to_image(event.x, event.y)
        left, right = sorted((x0, x1))
        top, bottom = sorted((y0, y1))
        if right - left < 20 or bottom - top < 20:
            self._set_status("A região selecionada é muito pequena. Arraste sobre toda a placa.")
            self.golden_canvas.delete("selection")
            return

        self.roi = (left, top, right, bottom)
        self.golden_photo, self.golden_transform = self._draw_image(
            self.golden_canvas,
            self.golden_image,
            self.golden_photo,
            self.roi,
        )
        self._set_status(f"Região marcada: x={left}:{right}, y={top}:{bottom}.")

    def connect_ollama(self) -> None:
        if self.busy:
            return
        self.busy = True
        self._set_status("Buscando modelos de visão no Ollama...")
        server = self.ollama_url.get()

        def worker() -> None:
            try:
                models = list_vision_models(server)
                if not models:
                    raise OllamaError("Nenhum modelo com visão foi encontrado.")
                self.after(0, lambda: self._finish_model_connection(models))
            except Exception as error:
                message = str(error)
                self.after(0, lambda: self._finish_error(message))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_model_connection(self, models: list[str]) -> None:
        self.model_menu.configure(values=models)
        chosen = DEFAULT_VISION_MODEL if DEFAULT_VISION_MODEL in models else models[0]
        self.ollama_model.set(chosen)
        self.busy = False
        self._set_status(f"Ollama conectado. Modelo selecionado: {chosen}.")

    def inspect_current_frame(self) -> None:
        if self.busy:
            return
        if self.golden_image is None:
            self._set_status("Capture primeiro uma peça boa como golden.")
            return
        if self.roi is None:
            self._set_status("Marque a região de interesse na imagem golden.")
            return
        if self.current_frame is None:
            self._set_status("A câmera ainda não possui um quadro para analisar.")
            return

        try:
            threshold = float(self.threshold.get().replace(",", "."))
            max_shift = int(self.max_shift.get())
            if self.single_led.get():
                rows = columns = 1
            else:
                rows = int(self.rows.get())
                columns = int(self.columns.get())
        except ValueError as error:
            self._set_status(f"Configuração inválida: {error}")
            return

        golden = self.golden_image.copy()
        candidates = [image.copy() for image in self.inspection_buffer]
        if not candidates:
            candidates = [self.current_frame.copy()]
        buffer_size = len(candidates)
        roi = self.roi
        use_llm = self.use_llm.get()
        server = self.ollama_url.get()
        model = self.ollama_model.get()
        self.busy = True
        self.inspect_button.configure(state="disabled", text="ANALISANDO...")
        self._set_status(
            f"Analisando {buffer_size} foto(s) em segundo plano. A câmera permanece ativa..."
        )

        def worker() -> None:
            try:
                if self.single_led.get():
                    left, top, right, bottom = roi
                    regions = [LedRegion("LD1", left, top, right - left, bottom - top)]
                else:
                    regions = generate_grid_regions(
                        golden.size,
                        rows,
                        columns,
                        bounds=roi,
                        fill_ratio=int(self.fill_ratio.get()) / 100.0,
                    )
                results, aligned = inspect_leds_buffer(
                    golden,
                    candidates,
                    regions,
                    threshold=threshold,
                    max_shift=max_shift,
                )
                marked = annotate_image(aligned, results)
                if not use_llm:
                    self.after(0, lambda: self._complete_local_only(marked, results))
                    return

                local_results = [
                    {"led": item.led_id, "status": item.status, "score": item.score}
                    for item in results
                ]
                try:
                    inspection = analyze_with_vision(
                        golden.crop(roi),
                        aligned.crop(roi),
                        model,
                        local_results,
                        base_url=server,
                    )
                    self.after(0, lambda: self._complete_with_llm(marked, results, inspection))
                except Exception as error:
                    message = str(error)
                    self.after(0, lambda: self._complete_with_llm_error(marked, results, message))
            except Exception as error:
                message = str(error)
                self.after(0, lambda: self._finish_processing_error(message))

        threading.Thread(target=worker, daemon=True).start()

    def _complete_local_only(self, marked: Image.Image, results: list) -> None:
        self._show_local_result(marked, results)
        self._finish_without_llm(results)

    def _complete_with_llm(self, marked: Image.Image, results: list, inspection) -> None:
        self._show_local_result(marked, results)
        self._finish_inspection(inspection, results)

    def _complete_with_llm_error(self, marked: Image.Image, results: list, message: str) -> None:
        self._show_local_result(marked, results)
        self._finish_inspection_error(message, results)

    def _finish_processing_error(self, message: str) -> None:
        self.busy = False
        self.inspect_button.configure(state="normal", text="INSPECIONAR QUADRO ATUAL")
        self._set_status(f"Não foi possível analisar o quadro: {message}")

    def _show_local_result(self, image: Image.Image, results: list) -> None:
        display = image.copy()
        display.thumbnail((760, 300), Image.Resampling.LANCZOS)
        self.result_photo = ImageTk.PhotoImage(display)
        self.result_label.configure(image=self.result_photo, text="")
        suspects = [item.led_id for item in results if item.status == "SUSPEITO"]
        lines = ["ANÁLISE VISUAL LOCAL", ""]
        lines.append("Suspeitos: " + (", ".join(suspects) if suspects else "nenhum"))
        lines.extend(f"{item.led_id}: {item.status} — score {item.score:.2f}" for item in results)
        self._write_report("\n".join(lines))

    def _finish_without_llm(self, results: list) -> None:
        suspects = [item.led_id for item in results if item.status == "SUSPEITO"]
        self._set_verdict("REVISAR" if suspects else "OK")
        self.busy = False
        self.inspect_button.configure(state="normal", text="INSPECIONAR QUADRO ATUAL")
        self._set_status("Inspeção local concluída. A câmera permanece ativa.")

    def _finish_inspection(self, inspection, results: list) -> None:
        self._set_verdict(inspection.verdict)
        existing = self.report_box.get("1.0", "end").strip()
        llm_lines = [
            existing,
            "",
            "PARECER DA LLM",
            f"Modelo: {inspection.model}",
            f"Veredito: {inspection.verdict}",
            f"Confiança: {inspection.confidence:.1f}%",
            f"Resumo: {inspection.summary}",
        ]
        if inspection.suspect_leds:
            llm_lines.append("LEDs indicados: " + ", ".join(inspection.suspect_leds))
        if inspection.observations:
            llm_lines.append("Observações:")
            llm_lines.extend(f"• {item}" for item in inspection.observations)
        self._write_report("\n".join(llm_lines))
        self.busy = False
        self.inspect_button.configure(state="normal", text="INSPECIONAR QUADRO ATUAL")
        self._set_status("Inspeção concluída. A câmera permanece ativa para a próxima peça.")

    def _finish_inspection_error(self, message: str, results: list) -> None:
        suspects = [item.led_id for item in results if item.status == "SUSPEITO"]
        self._set_verdict("REVISAR" if suspects else "OK LOCAL")
        existing = self.report_box.get("1.0", "end").strip()
        self._write_report(f"{existing}\n\nLLM indisponível: {message}")
        self.busy = False
        self.inspect_button.configure(state="normal", text="INSPECIONAR QUADRO ATUAL")
        self._set_status("A análise local terminou, mas a LLM não respondeu. A câmera continua ativa.")

    def _finish_error(self, message: str) -> None:
        self.busy = False
        self._set_status(message)

    def _set_verdict(self, verdict: str) -> None:
        self.verdict_text.set(verdict)
        if verdict == "OK":
            self.verdict_label.configure(fg_color="#15803D")
        elif verdict == "OK LOCAL":
            self.verdict_label.configure(fg_color="#A16207")
        else:
            self.verdict_label.configure(fg_color="#B91C1C")

    def _write_report(self, text: str) -> None:
        self.report_box.configure(state="normal")
        self.report_box.delete("1.0", "end")
        self.report_box.insert("1.0", text)
        self.report_box.configure(state="disabled")

    def _set_status(self, text: str) -> None:
        self.status_text.set(text)

    def close_app(self) -> None:
        self.stop_camera()
        self.destroy()


def main() -> None:
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    app = LedInspectorDesktop()
    app.mainloop()


if __name__ == "__main__":
    main()
