"""Streamlit UI for PetVision AI."""

import base64
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

API_URL = "http://api:8000"

MIME_BY_EXTENSION = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def _image_content_type(uploaded_file) -> str:
    """Return a MIME type the API accepts for multipart uploads."""
    if uploaded_file.type and uploaded_file.type.startswith("image/"):
        return uploaded_file.type

    extension = uploaded_file.name.rsplit(".", 1)[-1].lower() if uploaded_file.name else ""
    return MIME_BY_EXTENSION.get(f".{extension}", "image/jpeg")


def _api_error_message(response: requests.Response) -> str:
    try:
        detail = response.json().get("detail")
        if detail:
            return f"{response.status_code}: {detail}"
    except ValueError:
        pass
    return f"{response.status_code}: {response.text or response.reason}"


def main() -> None:
    st.set_page_config(page_title="PetVision AI", page_icon="🐾", layout="centered")
    st.title("🐾 PetVision AI")
    st.caption("Production-grade image classification platform")

    with st.sidebar:
        st.header("Configurações")
        api_url = st.text_input("API URL", value=API_URL)
        show_grad_cam = st.checkbox("Mostrar Grad-CAM", value=True)

        if st.button("Verificar API"):
            try:
                response = requests.get(f"{api_url}/health", timeout=5)
                st.json(response.json())
            except requests.RequestException as exc:
                st.error(f"API indisponível: {exc}")

    uploaded_file = st.file_uploader(
        "Envie uma imagem",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Imagem enviada", use_container_width=True)

        if st.button("Classificar"):
            endpoint = "explain" if show_grad_cam else "predict"
            with st.spinner("Classificando..."):
                try:
                    content_type = _image_content_type(uploaded_file)
                    response = requests.post(
                        f"{api_url}/{endpoint}",
                        params={"top_k": 1},
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                content_type,
                            )
                        },
                        timeout=60,
                    )
                    if not response.ok:
                        st.error(f"Erro na predição: {_api_error_message(response)}")
                        return
                    result = response.json()

                    st.success(f"Modelo: **{result['model_name']}**")
                    if result["predictions"]:
                        top_pred = result["predictions"][0]
                        st.markdown(f"**Categoria prevista:** {top_pred['label']}")
                        st.progress(
                            top_pred["confidence"],
                            text=f"Confiança: {top_pred['confidence']:.1%}",
                        )

                    if show_grad_cam and "heatmap_base64" in result:
                        heatmap = Image.open(BytesIO(base64.b64decode(result["heatmap_base64"])))
                        st.image(
                            heatmap,
                            caption=f"Grad-CAM — classe explicada: {result['explained_class']}",
                            use_container_width=True,
                        )
                except requests.RequestException as exc:
                    st.error(f"Erro na predição: {exc}")


if __name__ == "__main__":
    main()
