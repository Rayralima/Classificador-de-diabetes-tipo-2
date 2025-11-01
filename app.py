import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuração da Página ---
# Usar st.set_page_config é a primeira coisa a se fazer
st.set_page_config(
    page_title="Análise de Diabetes",
    page_icon="🩺",
    layout="wide"
)

# --- Carregamento dos Modelos ---
# Usar @st.cache_data previne recarregar os modelos a cada clique
@st.cache_data
def carregar_modelos_e_dados():
    try:
        with open("modelo_diabetes.pkl", "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        st.error("Erro: Arquivo 'modelo_diabetes.pkl' não encontrado.")
        model = None
        
    try:
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        st.error("Erro: Arquivo 'scaler.pkl' não encontrado.")
        scaler = None
    
    # Carregar dados para os gráficos de EDA
    try:
        df = pd.read_csv("diabetes.csv")
    except FileNotFoundError:
        st.error("Erro: Arquivo 'diabetes.csv' não encontrado para a EDA.")
        df = None

    return model, scaler, df

model, scaler, df = carregar_modelos_e_dados()

# --- Título Principal ---
st.title("🩺 Projeto Final: Machine Learning Aplicado à Saúde")
st.write("Análise preditiva e exploratória de risco de diabetes usando o dataset Pima.")

# --- Abas para Organização ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Classificação (Supervisionado)", 
    "Clusterização (Não Supervisionado)", 
    "Análise Exploratória (EDA)",
    "Resultados do Modelo"
])

# --- Aba 1: Classificação (Predição) ---
with tab1:
    st.header("Ferramenta de Predição de Risco")
    st.write("Insira os dados da paciente para prever o risco de diabetes.")
    
    if (model is None) or (scaler is None):
        st.warning("Modelos de predição não carregados. A ferramenta está desativada.")
    else:
        # Dividir em colunas para um layout mais limpo
        col1, col2 = st.columns(2)
        
        with col1:
            pregnancies = st.number_input("Número de Gestações", min_value=0, step=1, value=1)
            glucose = st.number_input("Glicose (nível de glicose no plasma)", min_value=0.0, value=120.0)
            blood_pressure = st.number_input("Pressão Arterial (diastólica, mm Hg)", min_value=0.0, value=70.0)
            skin_thickness = st.number_input("Espessura da Dobra Cutânea (tríceps, mm)", min_value=0.0, value=20.0)
            
        with col2:
            insulin = st.number_input("Insulina (soro de 2 horas, mu U/ml)", min_value=0.0, value=80.0)
            bmi = st.number_input("IMC (Índice de Massa Corporal)", min_value=0.0, value=30.0)
            dpf = st.number_input("Função de Pedigree de Diabetes", min_value=0.0, format="%.3f", value=0.470)
            age = st.number_input("Idade (anos)", min_value=0, step=1, value=30)

        # Botão para fazer a predição
        if st.button("Analisar Risco", type="primary"):
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                    insulin, bmi, dpf, age]])

            # Aplicar o scaler
            input_scaled = scaler.transform(input_data)

            # Fazer a predição
            prediction = model.predict(input_scaled)
            probability = model.predict_proba(input_scaled)
            
            # Pegar a probabilidade de Risco (Classe 1)
            prob_risco = probability[0][1]

            # Mostrar o resultado
            if prediction[0] == 1:
                st.error(f"**Resultado: Risco Elevado de Diabetes** (Probabilidade: {prob_risco*100:.2f}%)")
            else:
                st.success(f"**Resultado: Risco Baixo de Diabetes** (Probabilidade: {prob_risco*100:.2f}%)")

# --- Aba 2: Clusterização (Não Supervisionado) ---
with tab2:
    st.header("Análise de Perfis de Pacientes (K-Means)")
    st.write("Usamos a aprendizagem não supervisionada para encontrar grupos naturais (clusters) de pacientes com características semelhantes.")
    
    # Carregar os arquivos de análise de cluster
    pca_plot_path = 'kmeans_pca_plot.png'
    analysis_csv_path = 'cluster_analysis.csv'
    
    if not os.path.exists(pca_plot_path) or not os.path.exists(analysis_csv_path):
        st.warning("Arquivos de análise de cluster (`kmeans_pca_plot.png`, `cluster_analysis.csv`) não encontrados.")
        st.write("Por favor, execute o script de análise K-Means primeiro.")
    else:
        st.subheader("Visualização dos Clusters (PCA)")
        st.image(pca_plot_path, caption="Dispersão dos pacientes reduzida a 2 dimensões (PCA)")
        
        st.subheader("Interpretação dos Perfis dos Clusters")
        st.write("A tabela abaixo mostra o perfil médio de cada grupo encontrado:")
        
        # Carregar e exibir a tabela de análise
        df_cluster_analysis = pd.read_csv(analysis_csv_path).set_index('Cluster')
        st.dataframe(df_cluster_analysis.style.background_gradient(cmap='viridis_r', subset=['Proporcao_Diabetes', 'Glucose', 'BMI', 'Age', 'Insulin']))
        
        st.subheader("Conclusões da Análise de Cluster")
        st.markdown(r"""
        Com base na tabela, identificamos 3 perfis principais:

        * **Cluster 1 (Perfil de Baixo Risco):**
            * **Descrição:** É o maior grupo (335 pacientes) e o mais saudável.
            * **Características:** Pacientes mais jovens (26 anos), com menor IMC (28.5) e menor Glicose (106).
            * **Resultado:** Apenas 13% deste grupo têm diabetes.

        * **Cluster 0 (Perfil de Risco - Idade/Gestações):**
            * **Descrição:** O segundo maior grupo (248 pacientes).
            * **Características:** Pacientes com idade média mais alta (46 anos) e um número muito elevado de gestações (média de 7.4).
            * **Resultado:** 52% deste grupo têm diabetes.

        * **Cluster 2 (Perfil de Risco - Metabólico):**
            * **Descrição:** O menor grupo (185 pacientes), mas com os piores indicadores.
            * **Características:** Embora jovens (29 anos), têm os maiores níveis médios de **Glicose (137)**, **IMC (39.1)** e **Insulina (191.6)**.
            * **Resultado:** 52% deste grupo têm diabetes, indicando um risco metabólico severo.
        """)

# --- Aba 3: Análise Exploratória (EDA) ---
with tab3:
    st.header("Análise Exploratória dos Dados (EDA)")
    
    if df is None:
        st.warning("Arquivo 'diabetes.csv' não carregado. Não é possível mostrar a EDA.")
    else:
        st.subheader("Justificativa do Pré-processamento")
        st.write("""
        A análise inicial mostrou que 5 colunas (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) 
        tinham valores `0`, o que é biologicamente impossível.
        Estes gráficos de distribuição (histogramas) provam visualmente a presença desses dados 'ausentes'
        disfarçados de zero, justificando sua substituição pela mediana durante o pré-processamento.
        """)
        
        # Gráfico de Histograma (como o da sua imagem)
        fig, ax = plt.subplots(figsize=(10, 6))
        df.hist(ax=ax, bins=20, figsize=(15, 10))
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Correlação entre as Variáveis")
        st.write("""
        Este mapa de calor (heatmap) mostra a correlação entre todas as variáveis. 
        Podemos ver que `Glucose`, `BMI` e `Age` são as features com a correlação
        positiva mais forte com o `Outcome` (o diagnóstico de diabetes), 
        o que as torna preditoras importantes para o modelo.
        """)
        
        # Gráfico de Correlação (como o da sua imagem)
        fig_corr, ax_corr = plt.subplots(figsize=(10, 7))
        corr_matrix = df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
        st.pyplot(fig_corr)

# --- Aba 4: Resultados do Modelo ---
with tab4:
    st.header("Avaliação do Modelo Supervisionado")
    st.write("Aqui estão as métricas de performance do modelo Random Forest, obtidas na fase de treinamento e teste (do notebook).")

    st.subheader("Relatório de Classificação")
    st.write("Acurácia Geral: **74,03%**")
    
    # Texto do Relatório
    report_texto = """
                  precision    recall  f1-score   support

               0       0.80      0.79      0.80        99
               1       0.63      0.65      0.64        55

        accuracy                           0.74       154
       macro avg       0.72      0.72      0.72       154
    weighted avg       0.74      0.74      0.74       154
    """
    st.code(report_texto)
    
    st.subheader("Matriz de Confusão")
    st.write("A matriz mostra exatamente onde o modelo acertou e onde errou.")
    
    # Dados da Matriz
    cm = np.array([[78, 21], 
                   [19, 36]])

    # Criar a figura do heatmap
    fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, 
                xticklabels=['Predito 0 (Não)', 'Predito 1 (Sim)'], 
                yticklabels=['Real 0 (Não)', 'Real 1 (Sim)'])
    ax_cm.set_xlabel('Predição')
    ax_cm.set_ylabel('Valor Real')
    ax_cm.set_title('Matriz de Confusão')
    
    # Mostrar o gráfico no Streamlit
    st.pyplot(fig_cm)
    
    st.subheader("Validação Cruzada (Cross-Validation)")
    st.write("""
    Para garantir que o modelo é estável e a acurácia de 74% não foi sorte, 
    rodamos uma validação cruzada (K-Fold=5).
    O resultado foi uma acurácia média de **76.82%**, com baixo desvio padrão. 
    Isso prova que o modelo é robusto e confiável.
    """)
    st.image('image_b6a1cc.png', caption='Resultado da Validação Cruzada (K-Fold=5) executada no notebook.')