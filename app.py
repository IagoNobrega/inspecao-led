from __future__ import annotations

from hashlib import sha256
from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError
from streamlit.typing import UploadedFile

from led_inspector import (
    DEFAULT_OLLAMA_URL,
    DEFAULT_VISION_MODEL,
    LedRegion,
    OllamaError,
    analyze_with_vision,
    annotate_image,
    generate_demo_images,
    generate_grid_regions,
    inspect_leds,
    list_vision_models,
)


st.set_page_config(page_title="LED Inspector", page_icon=":material/lightbulb:", layout="wide")
st.title("LED Inspector")
st.caption("Inspeção por câmera ao vivo usando uma peça golden e inteligência artificial local")

st.session_state.setdefault("ollama_models", [])
st.session_state.setdefault("ollama_connection_error", None)


def open_upload(uploaded_file: UploadedFile) -> Image.Image:
    try:
        image = Image.open(BytesIO(uploaded_file.getvalue()))
        image.load()
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("O arquivo enviado não é uma imagem válida.") from exc


def image_to_png_bytes(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def regions_to_frame(regions: list[LedRegion]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"LED": item.led_id, "x": item.x, "y": item.y, "largura": item.width, "altura": item.height}
            for item in regions
        ]
    )


def frame_to_regions(frame: pd.DataFrame) -> list[LedRegion]:
    return [
        LedRegion(
            led_id=str(row["LED"]),
            x=int(row["x"]),
            y=int(row["y"]),
            width=int(row["largura"]),
            height=int(row["altura"]),
        )
        for _, row in frame.iterrows()
    ]


with st.sidebar:
    st.header("Configuração")
    use_demo = st.toggle(
        "Usar demonstração",
        value=False,
        help="Ative apenas para testar o sistema sem uma câmera.",
    )
    rows = st.number_input("Linhas de LEDs", 1, 20, 2)
    columns = st.number_input("Colunas de LEDs", 1, 30, 5)
    fill_ratio = st.slider(
        "Tamanho da região do LED (%)",
        min_value=30,
        max_value=90,
        value=58,
        step=5,
        help="Porcentagem da célula ocupada pelo LED. Menor = mais focado no LED, maior = inclui mais entorno.",
    )
    threshold = st.slider(
        "Limite de rejeição",
        min_value=3.0,
        max_value=45.0,
        value=10.0 if use_demo else 22.0,
        step=0.5,
        help="Valores menores tornam a inspeção mais sensível. Aumente se peças boas forem rejeitadas.",
    )
    max_shift = st.slider(
        "Correção de posição (px)",
        0,
        30,
        15,
        help="Compensa pequenos deslocamentos da peça em relação à golden. Aumente se a peça se move.",
    )
    st.divider()
    st.subheader("Inteligência artificial")
    use_llm = st.toggle(
        "Usar análise da LLM",
        value=True,
        help="Envia a golden e a captura para um modelo de visão no Ollama.",
    )
    if use_llm:
        ollama_url = st.text_input(
            "Servidor Ollama",
            value=DEFAULT_OLLAMA_URL,
            key="ollama_url",
        )
        if st.button("Conectar e buscar modelos", icon=":material/sync:", width="stretch"):
            try:
                with st.spinner("Consultando modelos de visão..."):
                    st.session_state.ollama_models = list_vision_models(ollama_url)
                st.session_state.ollama_connection_error = None
            except OllamaError as error:
                st.session_state.ollama_models = []
                st.session_state.ollama_connection_error = str(error)

        if st.session_state.ollama_models:
            default_index = (
                st.session_state.ollama_models.index(DEFAULT_VISION_MODEL)
                if DEFAULT_VISION_MODEL in st.session_state.ollama_models
                else 0
            )
            llm_model = st.selectbox(
                "Modelo de visão",
                st.session_state.ollama_models,
                index=default_index,
                key="ollama_model_select",
            )
            st.success("Ollama conectado.", icon=":material/check_circle:")
        else:
            llm_model = st.text_input(
                "Modelo de visão",
                value=DEFAULT_VISION_MODEL,
                key="ollama_model_manual",
                help="Use Conectar para carregar automaticamente os modelos com capacidade de visão.",
            )
            if st.session_state.ollama_connection_error:
                st.error(st.session_state.ollama_connection_error)
    else:
        ollama_url = DEFAULT_OLLAMA_URL
        llm_model = DEFAULT_VISION_MODEL

if use_demo:
    reference, candidate = generate_demo_images()
    reference_signature = "demo"
else:
    st.subheader("1 · Defina a referência golden")
    st.caption(
        "Envie uma foto validada de uma peça boa. Ela será a referência para comparar todos os LEDs."
    )
    reference_file = st.file_uploader(
        "Imagem da peça boa (golden)",
        type=["png", "jpg", "jpeg", "webp"],
        key="golden_reference",
        help="Use uma foto com a mesma câmera, suporte, distância e iluminação da inspeção.",
    )

    if reference_file is None:
        st.info("Carregue a imagem golden para liberar a câmera de inspeção.")
        st.stop()

    try:
        reference = open_upload(reference_file)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    reference_signature = sha256(reference_file.getvalue()).hexdigest()[:12]

    st.subheader("2 · Capture a peça pela câmera")
    st.caption("Posicione a peça como na imagem golden e tire a foto quando o enquadramento estiver correto.")
    camera_file = st.camera_input(
        "Câmera ao vivo — peça a inspecionar",
        key="inspection_camera",
        resolution="1080p",
        help="O navegador pode solicitar permissão para acessar a câmera.",
    )

    if camera_file is None:
        with st.container(border=True):
            st.caption("Referência golden carregada")
            st.image(reference, width="stretch")
        st.info("Tire uma foto com a câmera para iniciar a comparação.")
        st.stop()

    try:
        candidate = open_upload(camera_file)
    except ValueError as error:
        st.error(str(error))
        st.stop()

image_width, image_height = reference.size
default_bounds = (50, 60, 850, 360) if use_demo else (0, 0, image_width, image_height)

preview_step = "1" if use_demo else "3"
st.subheader(f"{preview_step} · Confira o enquadramento")
preview_left, preview_right = st.columns(2)
with preview_left:
    st.caption("Referência golden — peça boa")
    st.image(reference, width="stretch")
with preview_right:
    st.caption("Captura da câmera — peça em inspeção")
    st.image(candidate, width="stretch")

regions_step = "2" if use_demo else "4"
st.subheader(f"{regions_step} · Delimite a área e confira os LEDs")
with st.expander("Ajustar limites da grade", expanded=not use_demo, icon=":material/crop:"):
    bound_columns = st.columns(4)
    left = bound_columns[0].number_input("Esquerda (x)", 0, image_width - 1, default_bounds[0])
    top = bound_columns[1].number_input("Topo (y)", 0, image_height - 1, default_bounds[1])
    right = bound_columns[2].number_input("Direita (x)", 1, image_width, default_bounds[2])
    bottom = bound_columns[3].number_input("Base (y)", 1, image_height, default_bounds[3])

try:
    generated_regions = generate_grid_regions(
        reference.size,
        int(rows),
        int(columns),
        bounds=(int(left), int(top), int(right), int(bottom)),
        fill_ratio=fill_ratio / 100.0,
    )
except ValueError as error:
    st.error(str(error))
    st.stop()

editor_signature = f"{reference_signature}_{rows}_{columns}_{left}_{top}_{right}_{bottom}_{fill_ratio}"
edited_regions = st.data_editor(
    regions_to_frame(generated_regions),
    width="stretch",
    hide_index=True,
    num_rows="dynamic",
    column_config={
        "LED": st.column_config.TextColumn("LED", required=True),
        "x": st.column_config.NumberColumn("x", min_value=0, step=1, required=True),
        "y": st.column_config.NumberColumn("y", min_value=0, step=1, required=True),
        "largura": st.column_config.NumberColumn("largura", min_value=1, step=1, required=True),
        "altura": st.column_config.NumberColumn("altura", min_value=1, step=1, required=True),
    },
    key=f"regions_editor_{editor_signature}",
)

if st.button("Analisar captura", type="primary", icon=":material/search:", width="stretch"):
    try:
        regions = frame_to_regions(edited_regions)
        results, aligned = inspect_leds(reference, candidate, regions, threshold=threshold, max_shift=max_shift)
    except (ValueError, TypeError) as error:
        st.error(f"Não foi possível analisar: {error}")
        st.stop()

    result_rows = [
        {
            "LED": result.led_id,
            "Status": result.status,
            "Pontuação": result.score,
            "Diagnóstico": result.diagnosis,
            "Diferença visual (%)": result.difference,
            "Diferença de bordas (%)": result.edge_difference,
            "Diferença de áreas escuras (%)": result.dark_difference,
        }
        for result in results
    ]
    result_frame = pd.DataFrame(result_rows)
    marked = annotate_image(aligned, results)
    suspects = [result.led_id for result in results if result.status == "SUSPEITO"]

    llm_inspection = None
    llm_error = None
    if use_llm:
        local_results = [
            {"led": result.led_id, "status": result.status, "score": result.score}
            for result in results
        ]
        try:
            with st.spinner(f"{llm_model} está comparando a golden com a captura..."):
                llm_inspection = analyze_with_vision(
                    reference=reference,
                    candidate=aligned,
                    model=llm_model,
                    local_results=local_results,
                    base_url=ollama_url,
                )
        except OllamaError as error:
            llm_error = str(error)

    result_step = "3" if use_demo else "5"
    st.subheader(f"{result_step} · Resultado")
    with st.container(horizontal=True):
        st.metric("LEDs analisados", len(results), icon=":material/lightbulb:", border=True)
        st.metric("Suspeitos", len(suspects), icon=":material/warning:", border=True)
        st.metric("Aprovados", len(results) - len(suspects), icon=":material/check_circle:", border=True)

    if suspects:
        st.error("Revisar: " + ", ".join(suspects))
    else:
        st.success("Nenhum LED suspeito foi detectado com o limite atual.")

    if use_llm:
        st.subheader("Parecer da inteligência artificial")
        if llm_inspection is not None:
            if llm_inspection.verdict == "OK":
                st.success(f"LLM: OK — {llm_inspection.summary}", icon=":material/check_circle:")
            else:
                st.error(f"LLM: REVISAR — {llm_inspection.summary}", icon=":material/warning:")
            with st.container(horizontal=True):
                st.metric("Confiança da LLM", f"{llm_inspection.confidence:.1f}%", border=True)
                st.metric("Modelo", llm_inspection.model, border=True)
            if llm_inspection.suspect_leds:
                st.write("**LEDs indicados pela LLM:** " + ", ".join(llm_inspection.suspect_leds))
            if llm_inspection.observations:
                with st.expander("Observações da LLM", icon=":material/visibility:"):
                    for observation in llm_inspection.observations:
                        st.write(f"- {observation}")
        else:
            st.warning(
                f"A análise local foi concluída, mas a LLM não respondeu: {llm_error}",
                icon=":material/cloud_off:",
            )

    st.image(marked, caption="Vermelho = suspeito · Verde = aprovado", width="stretch")
    st.dataframe(result_frame, width="stretch", hide_index=True)

    download_left, download_right = st.columns(2)
    download_left.download_button(
        "Baixar laudo CSV",
        data=result_frame.to_csv(index=False).encode("utf-8-sig"),
        file_name="laudo_inspecao_leds.csv",
        mime="text/csv",
        on_click="ignore",
        icon=":material/download:",
        width="stretch",
    )
    download_right.download_button(
        "Baixar imagem marcada",
        data=image_to_png_bytes(marked),
        file_name="placa_inspecionada.png",
        mime="image/png",
        on_click="ignore",
        icon=":material/download:",
        width="stretch",
    )

st.caption(
    "A inspeção é visual e comparativa. Para resultados consistentes, mantenha a câmera fixa e use sempre o mesmo "
    "suporte, foco, distância e iluminação da imagem golden. O resultado deve ser confirmado por um responsável."
)
