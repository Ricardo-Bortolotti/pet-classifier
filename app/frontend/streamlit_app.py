"""Streamlit UI for PetVision AI."""

import requests
import streamlit as st

API_URL = "http://api:8000"


def main() -> None:
    st.set_page_config(page_title="PetVision AI", page_icon="🐾", layout="centered")
    st.title("🐾 PetVision AI")
    st.caption("Production-grade image classification platform")

    with st.sidebar:
        st.header("Configurações")
        api_url = st.text_input("API URL", value=API_URL)
        top_k = st.slider("Top-K predictions", min_value=1, max_value=10, value=3)

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
            with st.spinner("Classificando..."):
                try:
                    response = requests.post(
                        f"{api_url}/predict",
                        params={"top_k": top_k},
                        files={"file": (uploaded_file.name, uploaded_file.getvalue())},
                        timeout=30,
                    )
                    response.raise_for_status()
                    result = response.json()

                    st.success(f"Modelo: **{result['model_name']}**")
                    for pred in result["predictions"]:
                        st.progress(
                            pred["confidence"],
                            text=f"{pred['label']}: {pred['confidence']:.1%}",
                        )
                except requests.RequestException as exc:
                    st.error(f"Erro na predição: {exc}")


if __name__ == "__main__":
    main()
