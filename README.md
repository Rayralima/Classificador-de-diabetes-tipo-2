# 🩺 Classificador de Diabetes tipo 2
1. Introdução 

O objetivo deste projeto é aplicar técnicas de Machine Learning para analisar o dataset "Pima Indians Diabetes" da UCI. Este dataset busca prever o diagnóstico de diabetes tipo 2 em mulheres da etnia Pima, com base em 8 variáveis diagnósticas, como nível de glicose, IMC, idade e pressão arterial.

O projeto demonstra um fluxo completo de ciência de dados, incluindo a exploração e pré-processamento dos dados, a aplicação de um modelo supervisionado (Random Forest) para classificação (prever se a paciente tem ou não diabetes) e um modelo não supervisionado (K-Means) para segmentação (encontrar perfis de risco entre as pacientes). A aplicação final foi desenvolvida em Streamlit para permitir uma interação prática com os modelos.

2. Análise Exploratória (EDA) e Pré-Processamento
(Use no README.md ou roteiro do vídeo)

A análise exploratória (EDA) revelou que features como Glucose, BMI e Age possuem a correlação mais forte com o Outcome (diagnóstico).

Um desafio crítico deste dataset é a presença de valores "0" em colunas onde isso é biologicamente impossível (ex: Glucose=0 ou BMI=0). Esses zeros são, na verdade, dados ausentes. Para corrigir isso, aplicamos uma estratégia de imputation, substituindo os valores 0 de colunas como Glucose, BMI e Insulin pela mediana da respectiva coluna. Esta abordagem é mais robusta que a média, pois não é afetada por outliers.

Por fim, todos os dados foram padronizados com StandardScaler para garantir que as diferentes escalas das features não distorcessem a performance dos modelos.

3. 🚀 Análise: Aprendizagem Supervisionada (Classificação)
Justificativa do Modelo: Foi escolhido o Random Forest Classifier como modelo principal. Esta escolha se justifica por sua alta performance em problemas de classificação, sua capacidade de lidar com relações não-lineares e sua relativa robustez contra overfitting (especialmente quando validado corretamente).

Resultados (Baseados na sua execução): O modelo foi treinado em 80% dos dados e avaliado nos 20% restantes.

Acurácia (Teste): O modelo atingiu 74,03% de acurácia no conjunto de teste.

Validação Cruzada (100% dos dados): Para garantir a robustez, uma validação cruzada (K-Fold=5) foi aplicada em todo o dataset, resultando em uma acurácia média estável de 76,82%.

Relatório de Classificação (Análise Detalhada):

              precision    recall  f1-score
           0 (Não)   0.80      0.79      0.80
           1 (Sim)   0.63      0.65      0.64
Interpretação:

O modelo é muito bom em identificar pacientes Não Diabéticas (Precisão de 80%).

Ele é razoável em identificar pacientes Diabéticas (Recall de 65%). Isso significa que, de cada 100 pacientes que realmente têm diabetes, o modelo consegue identificar corretamente 65 delas (as outras 35 são "falsos negativos").

A Matriz de Confusão ([[78, 21], [19, 36]]) mostra que 19 pacientes diabéticas foram erroneamente classificadas como saudáveis, sendo este o erro mais crítico.

4. 🚀 Análise: Aprendizagem Não Supervisionada (Clusterização)

Justificativa da Técnica: Foi aplicado o algoritmo K-Means para segmentar as pacientes em grupos (clusters) com características semelhantes, sem usar a variável Outcome. O objetivo é descobrir "perfis de risco" naturais nos dados.

Justificativa de k (Número de Clusters): O Método do Cotovelo (Elbow Method) foi utilizado para encontrar o número ideal de clusters. O gráfico (salvo como kmeans_elbow_plot.png) mostra que a inércia (WCSS) diminui drasticamente até k=3, formando um "cotovelo". A partir de k=3, a redução se torna menos acentuada. Portanto, k=3 foi escolhido como o número ideal de clusters.

Interpretação dos Clusters (A Análise Principal): A análise da tabela de médias de cada cluster revelou 3 perfis de pacientes muito distintos:

                Tabela de Análise dos Clusters:

     Size  Proporcao_Diabetes  Glucose    BMI    Age  Insulin  ...
    0   248                0.52   130.79  32.81  46.04   139.63  ...
    1   335                0.13   106.62  28.50  25.94   113.33  ...
    2   185                0.52   136.64  39.15  29.30   191.57  ...
    Cluster 1: "Perfil de Baixo Risco" (335 pacientes)

Este é o maior grupo e o mais saudável. Possui a menor proporção de diabetes (apenas 13%).

Perfil: São as pacientes mais jovens (média de 26 anos), com o menor IMC (28.5) e menor Glicose (106).

Cluster 0: "Perfil de Risco por Idade/Gestações" (248 pacientes)

Este grupo tem uma alta proporção de diabetes (52%).

Perfil: O risco aqui é fortemente associado à idade (média de 46 anos) e ao número de gestações (média de 7.4), que são os mais altos de todos os grupos.

Cluster 2: "Perfil de Risco Metabólico" (185 pacientes)

Este grupo também tem alta proporção de diabetes (52%), mas por motivos diferentes do Cluster 0.

Perfil: Embora sejam jovens (média de 29 anos), elas possuem os piores indicadores metabólicos: a Glicose mais alta (137), o IMC mais alto (39) e a Insulina mais alta (191).

5. Conclusão do Projeto

O projeto foi bem-sucedido em criar duas abordagens de Machine Learning. O modelo supervisionado (Random Forest) alcançou uma performance robusta de 76,8% (validação cruzada), provando ser uma ferramenta viável para prever o risco de diabetes.

Mais importante, o modelo não supervisionado (K-Means) revelou que o risco de diabetes não é único; ele se manifesta em pelo menos dois perfis distintos: um associado à idade e histórico de gestações, e outro associado a indicadores metabólicos severos em pacientes mais jovens.
