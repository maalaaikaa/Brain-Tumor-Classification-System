from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import threading
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from config import DISPLAY_NAMES, UPLOAD_DIR, HISTORY_PATH, TRAINING_SUMMARY_PATH
from prediction.predictor import (
    confidence_message,
    load_model,
    predict_image,
    list_trained_models,
    load_trained_models,
)
from preprocessing.eda import scan_dataset
from ui.components import hero, inject_css, metrics_dashboard
from preprocessing.validator import validate_mri_scan
from utils.pdf_generator import generate_pdf_report
from database import db_manager

st.set_page_config(page_title="Brain Tumor Classification System", page_icon=":material/radiology:", layout="wide")
inject_css()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
db_manager.init_db()

@st.cache_resource
def cached_model():
    try:
        return load_model()
    except FileNotFoundError:
        return None

# Initialize session state for login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.markdown(
        """
        <div class="login-card">
            <div class="login-title">Clinical Access Portal</div>
            <p style="text-align: center; color: #64748b; font-size: 0.95rem; margin-bottom: 2rem;">
                Brain Tumor Classification Decision Support System
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_submit = st.form_submit_button("Access Portal")
        
        if login_submit:
            if username == "admin" and password == "admin123":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid Username or Password.")
                
    st.info(
        "🔐 **Demo Credentials**:\n\n"
        "- **Username**: `admin`  \n"
        "- **Password**: `admin123`"
    )
    
    st.markdown(
        '<div class="disclaimer" style="margin-top: 2rem;">This prototype requires clinical login to protect simulated patient records. Use the demo credentials above to access the dashboard.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# Sidebar user profile
st.sidebar.markdown(
    """
    <div style="padding: 1rem; background: #f8fafc; border-radius: 8px; border: 1px solid #cbd5e1; margin-bottom: 1.5rem;">
        <span style="font-size: 0.85rem; color: #64748b;">Logged in as:</span><br>
        <span style="font-weight: bold; color: #0f766e;">Dr. Administrator</span>
    </div>
    """,
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏥 Clinical Workspace",
        "📋 Patient Records & Logs",
        "📊 Model & Data Insights",
        "📖 Info & Resources",
    ],
)

st.sidebar.write("---")
if st.sidebar.button("Log Out", type="secondary"):
    st.session_state.logged_in = False
    st.rerun()

if page == "🏥 Clinical Workspace":
    hero()
    st.write("This application classifies MRI scans into Glioma, Meningioma, Pituitary, or No Tumor classes.")
    
    st.write("---")
    st.write("### 🧠 MRI Classification Workspace")
    st.caption("Upload a brain MRI image, fill in patient information, and review the model's class probabilities.")
    
    # Search available models
    trained_model_paths = list_trained_models()
    num_trained_models = len(trained_model_paths)
    
    # Prediction settings columns
    col_opts = st.columns(2)
    with col_opts[0]:
        prediction_mode = st.radio("Prediction Mode", ["Single Model", "Multi-Model Ensemble"], horizontal=True)
    with col_opts[1]:
        gradcam_cmap = st.selectbox("Grad-CAM Colormap", ["jet", "viridis", "inferno", "plasma", "magma"], index=0)
        
    with st.sidebar.expander("Advanced Settings"):
        gradcam_alpha = st.slider("Grad-CAM Alpha Transparency", 0.1, 1.0, 0.42)
        custom_layer = st.text_input("Custom Target Conv Layer (Optional)", value="")
        
    with st.form("patient_info_form"):
        st.write("#### Patient & Scan Metadata")
        c_meta = st.columns(3)
        p_id = c_meta[0].text_input("Patient ID", value=f"PT-{datetime.now().strftime('%y%m%d%H%M')}")
        p_age = c_meta[1].number_input("Age (Years)", min_value=1, max_value=120, value=45)
        p_gender = c_meta[2].selectbox("Biological Sex", ["Unspecified", "Male", "Female", "Other"])
        notes = st.text_area("Clinician Comments / Notes", placeholder="Enter notes or observations here...")
        
        upload = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png", "bmp"])
        submit_button = st.form_submit_button("Analyze Scan")
        
    if upload and submit_button:
        # Heuristics check
        validation = validate_mri_scan(upload)
        if not validation["is_valid"]:
            st.warning("⚠️ **MRI Structure Check Warning**")
            for r in validation["reasons"]:
                st.write(f"- {r}")
            st.info("The system will attempt classification, but results may be highly inaccurate due to out-of-distribution inputs.")
            
        try:
            with st.spinner("Analyzing MRI scan..."):
                if prediction_mode == "Multi-Model Ensemble":
                    if num_trained_models <= 1:
                        st.info("Ensemble mode requested, but only one trained model is available. Falling back to the default model.")
                        model = cached_model()
                    else:
                        model = load_trained_models(trained_model_paths)
                else:
                    model = cached_model()
                    
                if model is None:
                    raise FileNotFoundError("No trained model found. Please train a model first.")
                    
                result = predict_image(
                    upload,
                    model=model,
                    layer_name=custom_layer if custom_layer else None,
                    colormap=gradcam_cmap,
                    alpha=gradcam_alpha
                )
            
            # Save files permanently in uploads directory
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            orig_filename = f"{p_id}_{timestamp_str}_orig.png"
            grad_filename = f"{p_id}_{timestamp_str}_grad.png"
            
            orig_path = UPLOAD_DIR / orig_filename
            grad_path = UPLOAD_DIR / grad_filename
            
            Image.fromarray(result["original"]).save(orig_path)
            Image.fromarray(result["highlighted"]).save(grad_path)
            
            # Save record to SQLite database
            record_id = db_manager.save_record(
                patient_id=p_id,
                patient_age=p_age,
                patient_gender=p_gender,
                prediction_class=result["prediction"],
                confidence=result["confidence"],
                probabilities_dict=result["probabilities"],
                image_path=str(orig_path),
                physician_notes=notes
            )
            
            # Get full record for PDF
            db_record = db_manager.get_record_by_id(record_id)
            
            # Generate PDF Report
            pdf_report_path = UPLOAD_DIR / f"report_{record_id}.pdf"
            generate_pdf_report(
                record=db_record,
                original_path=orig_path,
                highlighted_path=grad_path,
                output_path=pdf_report_path
            )
            
            # Render predictions
            st.success(f"Analysis Complete & Logged! Record ID: #{record_id}")
            risk, msg = confidence_message(result["confidence"])
            
            summary_cols = st.columns(3)
            summary_cols[0].metric("Prediction", result["prediction"])
            summary_cols[1].metric("Confidence", f"{result['confidence']:.2%}")
            summary_cols[2].metric("Reliability", risk)
            st.info(msg)
            
            # PDF download button
            with open(pdf_report_path, "rb") as pdf_file:
                pdf_data = pdf_file.read()
            st.download_button(
                label="Download Clinical PDF Report",
                data=pdf_data,
                file_name=f"Brain_Tumor_Report_{p_id}.pdf",
                mime="application/pdf",
                type="primary"
            )
            
            # Visual overlay
            col1, col2 = st.columns(2)
            col1.image(result["original"], caption="Original MRI Scan", use_container_width=True)
            col2.image(result["highlighted"], caption=f"Grad-CAM Highlight ({gradcam_cmap.upper()})", use_container_width=True)
            
            if result.get("explanation_error"):
                st.warning(
                    "Prediction completed, but the Grad-CAM explanation could not be generated for this model. "
                    f"Details: {result['explanation_error']}"
                )
                
            probs = pd.DataFrame(
                {"Class": list(result["probabilities"].keys()), "Probability": list(result["probabilities"].values())}
            )
            fig = px.bar(
                probs.sort_values("Probability", ascending=False),
                x="Class",
                y="Probability",
                range_y=[0, 1],
                text_auto=".1%",
                color="Class",
            )
            fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
            st.warning("Consult a qualified radiologist or physician before making clinical decisions.")
            
        except Exception as exc:
            st.error(f"Analysis Failed: {exc}")

elif page == "📋 Patient Records & Logs":
    st.title("Patient Records & Logs")
    st.caption("Browse past diagnostic entries, search records, update clinician notes, and download print-ready PDF reports.")
    
    records = db_manager.get_all_records()
    if not records:
        st.info("No scan records found in the database. Scan an MRI to log your first case.")
    else:
        # Search & Filter widgets
        st.write("### 🔍 Search and Filter Records")
        col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
        
        search_query = col_f1.text_input("Search Patient ID or Notes", value="")
        filter_class = col_f2.multiselect(
            "Filter by Tumor Class",
            ["Glioma Tumor", "Meningioma Tumor", "Pituitary Tumor", "No Tumor"],
            default=["Glioma Tumor", "Meningioma Tumor", "Pituitary Tumor", "No Tumor"]
        )
        min_conf = col_f3.slider("Min Confidence", 0.0, 1.0, 0.0, step=0.05)
        
        # Apply filters
        filtered_records = []
        for r in records:
            search_match = (
                search_query.lower() in r["patient_id"].lower() or 
                (r["physician_notes"] and search_query.lower() in r["physician_notes"].lower())
            )
            class_match = r["prediction_class"] in filter_class
            conf_match = r["confidence"] >= min_conf
            
            if search_match and class_match and conf_match:
                filtered_records.append(r)
                
        if not filtered_records:
            st.info("No records matched your search filters.")
        else:
            df_records = pd.DataFrame([
                {
                    "ID": r["id"],
                    "Patient ID": r["patient_id"],
                    "Age": r["patient_age"],
                    "Gender": r["patient_gender"],
                    "Prediction": r["prediction_class"],
                    "Confidence": f"{r['confidence']:.2%}",
                    "Timestamp": r["timestamp"],
                    "Has Notes": "Yes" if r["physician_notes"] else "No"
                } for r in filtered_records
            ])
            st.dataframe(df_records, use_container_width=True, hide_index=True)
            
            st.write("---")
            st.write("### Record Detail & Actions")
            record_ids = [r["id"] for r in filtered_records]
            selected_id = st.selectbox("Select Record ID to view", record_ids)
            
            if selected_id:
                record = db_manager.get_record_by_id(selected_id)
                if record:
                    col_det1, col_det2 = st.columns([1, 1])
                    with col_det1:
                        st.write(f"**Patient ID:** {record['patient_id']}")
                        st.write(f"**Age / Gender:** {record['patient_age']} yrs / {record['patient_gender']}")
                        st.write(f"**Scan Time:** {record['timestamp']}")
                        st.write(f"**Prediction:** {record['prediction_class']} ({(record['confidence']):.2%})")
                        
                        # Notes update form
                        with st.form(f"update_notes_form_{selected_id}"):
                            notes_val = st.text_area("Update Physician Notes", value=record.get("physician_notes") or "")
                            btn_update = st.form_submit_button("Update Notes")
                            if btn_update:
                                db_manager.update_notes(selected_id, notes_val)
                                st.success("Notes updated successfully!")
                                st.rerun()
                                
                        # Download PDF button
                        orig_path = Path(record["image_path"])
                        grad_path = Path(str(orig_path).replace("_orig.png", "_grad.png"))
                        pdf_report_path = UPLOAD_DIR / f"report_{selected_id}.pdf"
                        
                        try:
                            generate_pdf_report(
                                record=record,
                                original_path=orig_path,
                                highlighted_path=grad_path,
                                output_path=pdf_report_path
                            )
                            with open(pdf_report_path, "rb") as pdf_file:
                                pdf_data = pdf_file.read()
                            st.download_button(
                                label="Download PDF Report",
                                data=pdf_data,
                                file_name=f"Brain_Tumor_Report_{record['patient_id']}.pdf",
                                mime="application/pdf",
                                key=f"dl_pdf_{selected_id}"
                            )
                        except Exception as exc:
                            st.error(f"Could not generate PDF: {exc}")
                            
                        # Delete record button
                        if st.button("Delete Record", type="secondary", key=f"del_rec_{selected_id}"):
                            db_manager.delete_record(selected_id)
                            orig_path.unlink(missing_ok=True)
                            grad_path.unlink(missing_ok=True)
                            pdf_report_path.unlink(missing_ok=True)
                            st.success("Record deleted successfully!")
                            st.rerun()
                            
                    with col_det2:
                        orig_path = Path(record["image_path"])
                        grad_path = Path(str(orig_path).replace("_orig.png", "_grad.png"))
                        if orig_path.exists() and grad_path.exists():
                            st.image(str(orig_path), caption="Original MRI Scan", use_container_width=True)
                            st.image(str(grad_path), caption="Grad-CAM Activation Highlight", use_container_width=True)
                        else:
                            st.warning("MRI Scan image files were not found on disk.")

elif page == "📊 Model & Data Insights":
    st.title("Model & Dataset Insights")
    st.caption("Review evaluation performance metrics, dataset analytics, and launch background backbone retraining.")
    
    tab_perf, tab_data, tab_train = st.tabs(["📈 Model Performance", "🗂️ Dataset Analytics", "⚙️ Background Trainer"])
    
    with tab_perf:
        st.write("### Model Performance Dashboard")
        metrics_dashboard()
        
    with tab_data:
        st.write("### Dataset Specifications & Distributions")
        try:
            df = scan_dataset()
            st.dataframe(df.groupby(["split", "class"]).size().reset_index(name="images"), use_container_width=True)
            st.plotly_chart(px.histogram(df, x="class", color="split", barmode="group"), use_container_width=True)
            st.plotly_chart(px.box(df, x="class", y="mean_intensity", color="split"), use_container_width=True)
        except Exception:
            st.info("Dataset not found yet. Run `python -m dataset.download_dataset`.")
            
    with tab_train:
        st.write("### Retrain Classifier Backbone")
        # Check if dataset exists
        dataset_ready = False
        try:
            df_ds = scan_dataset()
            if not df_ds.empty:
                dataset_ready = True
        except Exception:
            pass
            
        if not dataset_ready:
            st.error("Dataset not found. Please download the dataset first by running: `python -m dataset.download_dataset`")
        else:
            is_training = False
            if "train_thread" in st.session_state:
                is_training = st.session_state.train_thread.is_alive()
                
            if is_training:
                st.info("⏳ **Training in Progress**")
                st.spinner("Training model backbone... please wait.")
                
                # Show live plot from HISTORY_PATH
                if HISTORY_PATH.exists():
                    try:
                        df_hist = pd.read_csv(HISTORY_PATH)
                        st.write(f"Completed epochs: {len(df_hist)}")
                        col_plots = st.columns(2)
                        loss_cols = [c for c in df_hist.columns if "loss" in c]
                        accuracy_cols = [c for c in df_hist.columns if "accuracy" in c]
                        if loss_cols:
                            col_plots[0].line_chart(df_hist[loss_cols])
                        if accuracy_cols:
                            col_plots[1].line_chart(df_hist[accuracy_cols])
                    except Exception:
                        pass
                if st.button("Refresh Status"):
                    st.rerun()
            else:
                with st.form("train_hyperparameters"):
                    st.write("#### Hyperparameters")
                    col_h1, col_h2 = st.columns(2)
                    t_model = col_h1.selectbox("Backbone Model", ["mobilenetv2", "efficientnetb0", "resnet50", "custom_cnn"])
                    t_epochs = col_h1.number_input("Epochs", min_value=1, max_value=100, value=10)
                    t_batch = col_h2.selectbox("Batch Size", [16, 32, 64], index=1)
                    t_lr = col_h2.number_input("Learning Rate", min_value=1e-6, max_value=1e-1, value=1e-4, format="%e")
                    t_dropout = col_h2.slider("Dropout Rate", 0.1, 0.7, 0.35)
                    
                    btn_start = st.form_submit_button("Start Training")
                    
                if btn_start:
                    HISTORY_PATH.unlink(missing_ok=True)
                    TRAINING_SUMMARY_PATH.unlink(missing_ok=True)
                    
                    from training.trainer import train_model
                    
                    def run_training():
                        try:
                            train_model(
                                model_name=t_model,
                                epochs=int(t_epochs),
                                batch_size=int(t_batch),
                                learning_rate=t_lr,
                                dropout=t_dropout
                            )
                        except Exception as e:
                            print(f"Training thread error: {e}")
                            
                    thread = threading.Thread(target=run_training)
                    st.session_state.train_thread = thread
                    thread.start()
                    st.success("Training started in the background! Refresh to see live progress.")
                    st.rerun()

elif page == "📖 Info & Resources":
    st.title("Info & Resources")
    st.caption("Access medical context, REST API guidelines, and developer details.")
    
    with st.expander("🧠 About Brain Tumors", expanded=True):
        st.write("""
        This decision support system classifies brain MRI scans into four distinct categories:
        1. **Glioma**: Tumors originating in the glial cells of the brain or spine.
        2. **Meningioma**: Tumors arising from the meninges (membranes protecting the brain).
        3. **Pituitary Tumor**: Growths that develop in the pituitary gland.
        4. **No Tumor**: Healthy brain MRI scans with no detectable mass.
        
        *Note: While MRI scans give crucial structural clues, definitive diagnosis requires comprehensive biopsy and pathology reports.*
        """)
        
    with st.expander("❓ Frequently Asked Questions (FAQ)"):
        st.write("""
        **Q: Is this a final clinical diagnosis?**  
        *No. This prototype is an educational AI decision support assistant. It should be used exclusively for screening support and educational verification.*
        
        **Q: What is the purpose of the Grad-CAM visualization?**  
        *Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the specific convolutional regions the network focused on to arrive at its classification. This makes the neural network's decision process explainable to clinicians.*
        
        **Q: How do I train alternative model backbones?**  
        *Go to the 'Model & Data Insights' tab, open the 'Background Trainer' tab, set your hyperparameters, and click 'Start Training'.*
        """)
        
    with st.expander("🔌 REST API Integrations"):
        st.write("A FastAPI backend microservice runs concurrently on port 8000. Start it via CLI:")
        st.code("uvicorn api.main:app --port 8000 --reload")
        st.markdown("""
        **Available API Endpoints:**
        * `POST /predict`: Upload scan image, runs inferences, and logs records.
        * `GET /history`: Returns database entries in JSON.
        * `GET /report/{record_id}`: Compiles and downloads PDF reports.
        * `GET /metrics`: Returns evaluation metrics.
        """)
        st.markdown("""
        **Python Connection Example:**
        ```python
        import requests
        
        res = requests.post(
            "http://127.0.0.1:8000/predict",
            data={"patient_id": "PT-A1", "use_ensemble": "true"},
            files={"file": open("mri.png", "rb")}
        )
        print(res.json())
        ```
        """)
        
    with st.expander("💻 Developer & Technical Info"):
        st.write("""
        * **Frameworks**: Python, TensorFlow, Keras, Streamlit, FastAPI, ReportLab.
        * **Backbones**: MobileNetV2, EfficientNetB0, ResNet50, Custom CNN.
        * **Acceleration**: Mixed-precision computation (enabled when CUDA GPU is active).
        * **Credits**: Built by DeepMind Advanced Agentic Coding for portfolio development.
        """)
