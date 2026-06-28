import os

import requests
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("API_URL", "http://localhost:8000/predict/")

map_sino = {"Sí": "Yes", "No": "No"}
map_genero = {"Femenino": "Female", "Masculino": "Male"}
map_internet = {"Fibra Óptica": "Fiber optic", "DSL": "DSL", "No": "No"}
map_contrato = {
    "Mes a mes": "Month-to-month",
    "Un año": "One year",
    "Dos años": "Two year",
}
map_pago = {
    "Cheque Electrónico": "Electronic check",
    "Cheque por Correo": "Mailed check",
    "Transferencia Bancaria": "Bank transfer (automatic)",
    "Tarjeta de Crédito": "Credit card (automatic)",
}


def main():
    st.title("Plataforma de Predicción de Fuga de Clientes")
    st.markdown(
        "Este dashboard consume un modelo de **Machine Learning (CatBoost)** a través de una **API REST (FastAPI)** para evaluar el riesgo de que un cliente abandone los servicios de la empresa."
    )

    tab_predict, tab_analytics = st.tabs(["Nueva Predicción", "Análisis del Modelo"])

    with tab_predict:
        render_prediction_tab()

    with tab_analytics:
        render_analytics_tab()


def render_prediction_tab():
    st.header("Perfil del Cliente")
    st.write("Ingrese los datos del cliente para evaluar su riesgo de fuga.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demografía")
        gender_ui = st.selectbox("Género", list(map_genero.keys()))
        senior_ui = st.selectbox("¿Es de la tercera edad?", list(map_sino.keys()))
        partner_ui = st.selectbox("¿Tiene pareja?", list(map_sino.keys()))
        dependents_ui = st.selectbox("¿Tiene dependientes?", list(map_sino.keys()))

    with col2:
        st.subheader("Servicios y Contrato")
        tenure = st.number_input(
            "Meses de antigüedad", min_value=0, max_value=100, value=1
        )
        contract_ui = st.selectbox("Tipo de Contrato", list(map_contrato.keys()))
        paperless_ui = st.selectbox("Facturación Electrónica", list(map_sino.keys()))
        payment_method_ui = st.selectbox("Método de Pago", list(map_pago.keys()))

    with col3:
        st.subheader("Configuración Técnica")
        internet_service_ui = st.selectbox(
            "Servicio de Internet", list(map_internet.keys())
        )
        multiple_lines_ui = st.selectbox(
            "Líneas Múltiples", ["Sí", "No", "Sin servicio telefónico"]
        )

        is_no_internet = internet_service_ui == "No"
        internet_options = (
            ["Sin servicio de internet"] if is_no_internet else ["Sí", "No"]
        )

        online_security_ui = st.selectbox("Seguridad Online", internet_options)
        online_backup_ui = st.selectbox("Respaldo Online", internet_options)
        tech_support_ui = st.selectbox("Soporte Técnico", internet_options)

    st.subheader("Finanzas")
    fin_col1, fin_col2 = st.columns(2)
    with fin_col1:
        monthly_charges = st.number_input(
            "Cargos Mensuales ($)", min_value=0.0, value=29.85
        )
    with fin_col2:
        total_charges = st.number_input(
            "Cargos Totales ($)", min_value=0.0, value=29.85
        )

    st.markdown("---")
    if st.button("Evaluar Riesgo de Fuga", type="primary", use_container_width=True):

        map_multiple_lines = {
            "Sí": "Yes",
            "No": "No",
            "Sin servicio telefónico": "No phone service",
        }
        map_tech_options = {
            "Sí": "Yes",
            "No": "No",
            "Sin servicio de internet": "No internet service",
        }

        payload = {
            "customerID": "STREAMLIT-USER",
            "gender": map_genero[gender_ui],
            "SeniorCitizen": 1 if senior_ui == "Sí" else 0,
            "Partner": map_sino[partner_ui],
            "Dependents": map_sino[dependents_ui],
            "tenure": tenure,
            "PhoneService": (
                "Yes" if multiple_lines_ui != "Sin servicio telefónico" else "No"
            ),
            "MultipleLines": map_multiple_lines[multiple_lines_ui],
            "InternetService": map_internet[internet_service_ui],
            "OnlineSecurity": map_tech_options[online_security_ui],
            "OnlineBackup": map_tech_options[online_backup_ui],
            "DeviceProtection": "No" if is_no_internet else "Yes",
            "TechSupport": map_tech_options[tech_support_ui],
            "StreamingTV": "No" if is_no_internet else "Yes",
            "StreamingMovies": "No" if is_no_internet else "Yes",
            "Contract": map_contrato[contract_ui],
            "PaperlessBilling": map_sino[paperless_ui],
            "PaymentMethod": map_pago[payment_method_ui],
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        with st.spinner("Consultando al motor de inferencia..."):
            try:
                response = requests.post(API_URL, json=payload, timeout=30)

                if response.status_code == 200:
                    result = response.json()
                    display_prediction_result(result, payload)
                elif response.status_code == 422:
                    st.warning(
                        f"Error de validación: {response.json()['detail'][0]['msg']}"
                    )
                else:
                    st.error(f"Error de la API: Código {response.status_code}")
                    st.json(response.json())
            except requests.exceptions.ConnectionError:
                st.error(
                    "No se pudo conectar a la API. Asegúrese de que FastAPI esté corriendo en el puerto 8000."
                )
            except requests.exceptions.ReadTimeout:
                st.error(
                    "El servidor está despertando (Cold Start). Por favor, intenta de nuevo en unos segundos."
                )


def display_prediction_result(result: dict, payload: dict):
    risk_score = result["churn_risk_score"]
    will_churn = result["will_churn"]

    st.subheader("Resultados de la Evaluación")
    st.divider()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric(
            label=f"ID: #{result.get('prediction_id', 'N/A')}",
            value=f"{risk_score:.1%}",
        )
        if will_churn:
            st.error("ALERTA: Alta Probabilidad")
            st.warning(
                "Se recomienda enviar una oferta de retención o contactar al cliente inmediatamente."
            )
        else:
            st.success("CLIENTE SEGURO")

    with col2:
        st.markdown("### 🔍 ¿Por qué?")
        st.markdown("Principales factores que influyen en esta predicción:")

        if risk_score > 0.5:
            for factor in result.get("top_risk_factors", []):
                st.markdown(f"- 🔴 **{factor}**")
        else:
            for factor in result.get("top_retention_factors", []):
                st.markdown(f"- 🟢 **{factor}**")


def render_analytics_tab():
    st.header("Auditoría del Modelo (Explainable AI)")
    st.write(
        "Este panel muestra el rendimiento del modelo Campeón activo (CatBoost) y explica cómo toma sus decisiones."
    )

    img_dir = "docs/assets/images"

    if not os.path.exists(img_dir):
        st.error(
            "No se encontraron las imágenes. Por favor ejecute la evaluación del modelo primero."
        )
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Matriz de Confusión")
        st.image(Image.open(f"{img_dir}/confusion_matrix.png"), use_column_width=True)

        st.subheader("Importancia de Variables (SHAP)")
        st.image(Image.open(f"{img_dir}/shap_bar.png"), use_column_width=True)

    with col2:
        st.subheader("Curva ROC (Poder de Separación)")
        st.image(Image.open(f"{img_dir}/roc_curve.png"), use_column_width=True)

        st.subheader("Impacto Direccional (SHAP Summary)")
        st.image(Image.open(f"{img_dir}/shap_summary.png"), use_column_width=True)


if __name__ == "__main__":
    main()
