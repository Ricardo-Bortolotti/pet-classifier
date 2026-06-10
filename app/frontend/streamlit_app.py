"""Streamlit UI for PetVision AI."""

import base64
import os
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

API_URL = os.environ.get("PETVISION_API_URL", "http://localhost:8000")

MIME_BY_EXTENSION = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

CLASS_LABELS = {
    "cat": "Cat",
    "cats": "Cat",
    "dog": "Dog",
    "dogs": "Dog",
}


def _format_class_label(label: str) -> str:
    return CLASS_LABELS.get(label.lower(), label.title())


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

    st.markdown(
        "This application uses a **deep learning model** trained to predict whether "
        "an uploaded image shows a **cat** or a **dog**."
    )
    st.markdown(
        "Upload a photo of a cat or a dog. Supported formats: **JPG**, **JPEG**, **PNG**, "
        "and **WebP**. After you upload an image, click **Classify** below and the model "
        "will return its prediction."
    )

    with st.sidebar:
        st.header("Settings")
        show_grad_cam = st.checkbox("Show Grad-CAM", value=True)

        if st.button("Check API health"):
            try:
                response = requests.get(f"{API_URL}/health", timeout=5)
                st.json(response.json())
            except requests.RequestException as exc:
                st.error(f"API unavailable: {exc}")

        st.markdown("---")
        st.markdown(
            "Developed by **Ricardo Bortolotti**  \n"
            "[LinkedIn](https://www.linkedin.com/in/ricardo-bortolotti/) · "
            "[GitHub](https://github.com/Ricardo-Bortolotti)"
        )

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded image", use_container_width=True)

        if st.button("Classify"):
            endpoint = "explain" if show_grad_cam else "predict"
            with st.spinner("Classifying..."):
                try:
                    content_type = _image_content_type(uploaded_file)
                    response = requests.post(
                        f"{API_URL}/{endpoint}",
                        params={"top_k": 1},
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                content_type,
                            )
                        },
                        timeout=300,
                    )
                    if not response.ok:
                        st.error(f"Prediction error: {_api_error_message(response)}")
                        return
                    result = response.json()

                    if result["predictions"]:
                        top_pred = result["predictions"][0]
                        display_label = _format_class_label(top_pred["label"])

                        st.markdown("**Predicted class**")
                        st.markdown(
                            f'<p style="font-size: 2.5rem; font-weight: 700; margin: 0;">'
                            f"{display_label}</p>",
                            unsafe_allow_html=True,
                        )
                        st.progress(
                            top_pred["confidence"],
                            text=f"Confidence: {top_pred['confidence']:.1%}",
                        )

                        if show_grad_cam:
                            st.markdown(
                                "**What is Grad-CAM?**  \n"
                                "Grad-CAM (Gradient-weighted Class Activation Mapping) highlights "
                                "the regions of the image that most influenced the model's "
                                "prediction. In the overlay below, **warm colors (red and "
                                "yellow)** indicate areas the model paid the most attention to "
                                "when choosing "
                                "this class. **Cool colors (blue and green)** mark regions with "
                                "less influence on the decision. This helps you see *where* the "
                                "model looked—not just *what* it predicted."
                            )

                    if show_grad_cam and "heatmap_base64" in result:
                        explained = _format_class_label(result["explained_class"])
                        heatmap = Image.open(BytesIO(base64.b64decode(result["heatmap_base64"])))
                        st.image(
                            heatmap,
                            caption=f"Grad-CAM overlay — explained class: {explained}",
                            use_container_width=True,
                        )
                except requests.RequestException as exc:
                    st.error(f"Prediction error: {exc}")


if __name__ == "__main__":
    main()
