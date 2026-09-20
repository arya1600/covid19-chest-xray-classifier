"""Streamlit demo: upload a chest X-ray, see the model's prediction and a
Grad-CAM overlay of what it's attending to.

Run with:
    streamlit run demo/app.py

THIS IS A RESEARCH/PORTFOLIO DEMO, NOT A DIAGNOSTIC TOOL. See the
disclaimer rendered on every page, and read reports/ for why this
project's own Experiment D / Grad-CAM findings mean the underlying
model's apparent accuracy should not be taken at face value — it shows
signs of relying on dataset-source artifacts rather than confirmed lung
pathology. This demo exists to make the pipeline tangible, not to claim
the model works.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from covidcxr.data.transforms import ResizeAndPad, build_transforms  # noqa: E402
from covidcxr.explain.gradcam import GradCAM, overlay_heatmap  # noqa: E402
from covidcxr.models.baseline_cnn import BaselineCNN  # noqa: E402
from covidcxr.training.config import load_experiment_config  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURES = {"baseline_cnn": BaselineCNN}

# Friendly names for the technical run folders, so the dropdown doesn't
# require knowing what "exp_a_baseline" means. Add an entry here whenever
# a new experiment is trained.
MODEL_DISPLAY_INFO = {
    "exp_a_baseline": {
        "label": "Main model — COVID vs Normal",
        "description": "The project's primary classifier. See the disclaimer above before trusting its output.",
    },
    "exp_d_control": {
        "label": "Comparison model — Normal vs Pneumonia (not COVID-related)",
        "description": "A sanity-check model trained on a different task (single hospital, no COVID images at all), used only to test the main model for bias. Not for classifying COVID.",
    },
}

DISCLAIMER = (
    "**This is a research / MSc-portfolio demo, not a medical device.** "
    "It has not been clinically validated. Independent evaluation "
    "(`reports/`, Experiment D and Grad-CAM analysis in this project) "
    "found evidence the underlying model may partly rely on dataset-source "
    "artifacts rather than confirmed lung pathology. **Do not use this for "
    "diagnosis or any real clinical decision.**"
)


def find_available_runs() -> list[Path]:
    runs_dir = REPO_ROOT / "runs"
    if not runs_dir.is_dir():
        return []
    return sorted(p for p in runs_dir.iterdir() if (p / "best.pt").exists() and (p / "config.yaml").exists())


@st.cache_resource
def load_model(config_path: str, checkpoint_path: str):
    config = load_experiment_config(config_path)
    model_cls = ARCHITECTURES[config.model.architecture]
    model = model_cls(
        in_channels=config.model.in_channels,
        channels=tuple(config.model.channels),
        dropout=config.model.dropout,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model, config


def main() -> None:
    st.set_page_config(page_title="CXR classifier demo (research only)", page_icon="⚠️")
    st.title("Chest X-ray classifier — research demo")
    st.warning(DISCLAIMER)

    runs = find_available_runs()
    if not runs:
        st.error("No trained runs found under runs/. Train a model first (scripts/train.py).")
        return

    run_names = [r.name for r in runs]
    display_labels = [MODEL_DISPLAY_INFO.get(name, {}).get("label", name) for name in run_names]
    selected_label = st.sidebar.selectbox("Choose a model", display_labels)
    selected_name = run_names[display_labels.index(selected_label)]
    selected_run = next(r for r in runs if r.name == selected_name)

    st.sidebar.caption(MODEL_DISPLAY_INFO.get(selected_name, {}).get("description", ""))

    model, config = load_model(str(selected_run / "config.yaml"), str(selected_run / "best.pt"))
    label_map = config.data.label_to_index or {"NORMAL": 0, "COVID": 1}
    index_to_label = {v: k for k, v in label_map.items()}
    positive_label = index_to_label.get(1, "class 1")
    negative_label = index_to_label.get(0, "class 0")

    uploaded = st.file_uploader("Upload a chest X-ray image", type=["png", "jpg", "jpeg"])
    if uploaded is None:
        st.info("Upload an image to run inference. Try images from data/raw/ — never upload real patient data you don't have rights to.")
        return

    from PIL import Image
    raw_image = Image.open(uploaded).convert("L")

    transform = build_transforms(
        config.data.image_size, train=False, mean=config.data.mean, std=config.data.std,
        num_channels=config.data.num_channels,
    )
    display_image = ResizeAndPad(config.data.image_size)(raw_image)
    input_tensor = transform(raw_image).unsqueeze(0)
    input_tensor.requires_grad_(True)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)
    pred_index = int(logits.argmax(dim=1).item())
    pred_label = index_to_label.get(pred_index, str(pred_index))
    pred_prob = probs[0, pred_index].item()

    cam_tool = GradCAM(model, target_layer=model.features)
    cam, _ = cam_tool.generate(input_tensor, target_class=pred_index)
    overlay = overlay_heatmap(display_image, cam)

    col1, col2 = st.columns(2)
    with col1:
        st.image(display_image, caption="Input (resized, as fed to the model)", width="stretch")
    with col2:
        st.image(overlay, caption="Grad-CAM: what the model attended to", width="stretch")

    st.divider()

    normal_pct = probs[0, 0].item() * 100
    positive_pct = probs[0, 1].item() * 100

    if pred_index == 0:
        st.success(f"### ✅ Result: looks like **{negative_label}** ({pred_prob:.0%} confidence)")
    else:
        st.error(f"### ⚠️ Result: shows signs consistent with **{positive_label}** ({pred_prob:.0%} confidence)")

    st.caption(
        "This is the model's single verdict above. The two numbers below show "
        "how confident it was in EACH possible answer — they always add up to 100%."
    )
    m1, m2 = st.columns(2)
    m1.metric(f"Chance it's {negative_label}", f"{normal_pct:.0f}%")
    m2.metric(f"Chance it's {positive_label}", f"{positive_pct:.0f}%")

    st.warning(DISCLAIMER)


if __name__ == "__main__":
    main()
