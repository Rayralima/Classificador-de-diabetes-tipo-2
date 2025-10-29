# Colocar aqui o código do streamlit app
import streamlit as st
import pickle
import numpy as np

# Carregar o modelo e o scaler
try:
    with open("modelo_diabetes.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("Erro: Arquivos 'modelo_diabetes.pkl' ou 'scaler.pkl' não encontrados.")
    st.stop()

# --- Interface do Streamlit ---
st.title("Predição de Diabetes - Modelo de IA")
st.write("Insira os dados do paciente para prever o risco de diabetes.")

# Campos de entrada das 8 feactures
pregnancies = st.number_input("Número de Gestações", min_value=0, step=1)
glucose = st.number_input("Glicose (nível de glicose no plasma)", min_value=0.0)
blood_pressure = st.number_input("Pressão Arterial (diastólica, mm Hg)", min_value=0.0)
skin_thickness = st.number_input("Espessura da Dobra Cutânea (tríceps, mm)", min_value=0.0)
insulin = st.number_input("Insulina (soro de 2 horas, mu U/ml)", min_value=0.0)
bmi = st.number_input("IMC (Índice de Massa Corporal)", min_value=0.0)
dpf = st.number_input("Função de Pedigree de Diabetes", min_value=0.0, format="%.3f")
age = st.number_input("Idade (anos)", min_value=0, step=1)

# Botão para iniciar a predição
if st.button("Prever Risco de Diabetes"):
    if 'model' in locals() and 'scaler' in locals():
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                insulin, bmi, dpf, age]])

        # Aplicando o Scaler
        input_scaled = scaler.transform(input_data)

        # Fazendo a predição
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled) # ver a probabilidade

        # Exibindo o resultado
        if prediction[0] == 1:
            st.error(f"Resultado: Diabético (Probabilidade: {probability[0][1]:.2f})")
        else:
            st.success(f"Resultado: Não Diabético (Probabilidade: {probability[0][0]:.2f})")
    else:
        st.warning("Modelo ou Scaler não carregados. Verifique os arquivos .pkl.")