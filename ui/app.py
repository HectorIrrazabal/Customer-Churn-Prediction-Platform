"""Dashboard interactivo para la predicción de Churn utilizando Streamlit."""

import os

import requests
import streamlit as st
from PIL import Image

# Configuración de la página
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# URL de nuestra API REST local
API_URL = os.getenv("API_URL", "http://localhost:8000/predict/")


def main():
    st.title("Plataforma de Predicción de Fuga de Clientes ")
    st.markdown("""
    Este dashboard consume un modelo de **Machine Learning (CatBoost)** a través de una **API REST (FastAPI)** para evaluar el riesgo de que un cliente abandone los servicios de la empresa.
    """)

    # Crear pestañas para organizar la interfaz
    tab_predict, tab_analytics = st.tabs(
        ["Nueva Predicción", "Análisis del Modelo"]
    )

    with tab_predict:
        render_prediction_tab()

    with tab_analytics:
        render_analytics_tab()


def render_prediction_tab():
    """Renderiza el formulario para realizar predicciones."""
    st.header("Perfil del Cliente")
    st.write("Ingrese los datos del cliente para evaluar su riesgo de fuga.")

    # Agrupar campos lógicamente en columnas
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demografía")
        gender = st.selectbox("Género", ["Male", "Female"])
        senior_citizen = st.selectbox("¿Es de la tercera edad?", ["No", "Yes"])
        partner = st.selectbox("¿Tiene pareja?", ["Yes", "No"])
        dependents = st.selectbox("¿Tiene dependientes?", ["Yes", "No"])

    with col2:
        st.subheader("Servicios y Contrato")
        tenure = st.number_input(
            "Meses de antigüedad ", min_value=0, max_value=100, value=1
        )
        contract = st.selectbox(
            "Tipo de Contrato", ["Month-to-month", "One year", "Two year"]
        )
        paperless = st.selectbox("Facturación Electrónica", ["Yes", "No"])
        payment_method = st.selectbox(
            "Método de Pago",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

    with col3:
        st.subheader("Configuración Técnica")
        internet_service = st.selectbox(
            "Servicio de Internet", ["DSL", "Fiber optic", "No"]
        )
        multiple_lines = st.selectbox(
            "Líneas Múltiples", ["Yes", "No", "No phone service"]
        )

        # Deshabilitar servicios si no hay internet
        is_no_internet = internet_service == "No"
        internet_options = ["No internet service"] if is_no_internet else ["Yes", "No"]

        online_security = st.selectbox("Seguridad Online", internet_options)
        online_backup = st.selectbox("Respaldo Online", internet_options)
        tech_support = st.selectbox("Soporte Técnico", internet_options)

    # Campos financieros en una fila separada
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

    # Botón de envío
    st.markdown("---")
    if st.button("Evaluar Riesgo de Fuga", type="primary", use_container_width=True):
        # Mapeo de datos para el payload JSON
        payload = {
            "customerID": "STREAMLIT-USER",
            "gender": gender,
            "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": "Yes" if multiple_lines != "No phone service" else "No",
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": (
                "No" if is_no_internet else "Yes"
            ),  # Simplificado por UX
            "TechSupport": tech_support,
            "StreamingTV": "No" if is_no_internet else "Yes",  # Simplificado por UX
            "StreamingMovies": "No" if is_no_internet else "Yes",  # Simplificado por UX
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        with st.spinner("Consultando al motor de inferencia..."):
            try:
                response = requests.post(API_URL, json=payload, timeout=30)

                if response.status_code == 200:
                    result = response.json()
                    display_prediction_result(result)
                else:
                    st.error(f"Error de la API: Código {response.status_code}")
                    st.json(response.json())
            except requests.exceptions.ConnectionError:
                st.error(
                    "No se pudo conectar a la API. Asegúrese de que FastAPI esté corriendo en el puerto 8000."
                )


def display_prediction_result(result: dict):
   
    risk_score = result["churn_risk_score"]
    will_churn = result["will_churn"]

    st.subheader("Resultados de la Evaluación")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="ID de Predicción", value=f"#{result['prediction_id']}")

    with col2:
        # Colorear según el riesgo
        color = "red" if will_churn else "green"
        st.markdown(
            f"<h3 style='color:{color}; text-align:center;'>Riesgo de Fuga: {risk_score:.1%}</h3>",
            unsafe_allow_html=True,
        )

    with col3:
        status = "ALERTA: Alta Probabilidad" if will_churn else "CLIENTE SEGURO"
        st.info(status)

    if will_churn:
        st.warning(
            "Se recomienda enviar una oferta de retención o contactar al cliente inmediatamente."
        )


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
