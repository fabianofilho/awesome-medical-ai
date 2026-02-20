<p align="center">
  <h1 align="center">Awesome Medical AI</h1>
</p>

<p align="center">
  <a href="https://awesome.re"><img src="https://awesome.re/badge.svg" alt="Awesome"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://github.com/fabianofilho/awesome-medical-ai/stargazers"><img src="https://img.shields.io/github/stars/fabianofilho/awesome-medical-ai.svg?style=social" alt="Stars"></a>
  <a href="https://github.com/fabianofilho/awesome-medical-ai/network/members"><img src="https://img.shields.io/github/forks/fabianofilho/awesome-medical-ai.svg?style=social" alt="Forks"></a>
</p>

<p align="center">
  Uma lista curada de recursos incríveis para <strong>Inteligência Artificial na Medicina</strong>: artigos científicos, datasets, ferramentas, modelos pré-treinados, código prático e competições.
</p>

---

## Conteúdo

- [Artigos (Papers)](#artigos-papers)
  - [Revisões e Surveys](#revisões-e-surveys)
  - [Modelos Preditivos e EHR](#modelos-preditivos-e-ehr)
  - [Imagem Médica](#imagem-médica)
  - [NLP em Saúde](#nlp-em-saúde)
  - [ECG e Sinais Biológicos](#ecg-e-sinais-biológicos)
  - [Drug Discovery](#drug-discovery)
  - [IA Explicável (XAI) em Saúde](#ia-explicável-xai-em-saúde)
  - [Ética e Fairness em IA na Saúde](#ética-e-fairness-em-ia-na-saúde)
- [Mão na Massa & Código](#mão-na-massa--código)
  - [Repositórios e Tutoriais Gerais](#repositórios-e-tutoriais-gerais)
  - [ECG e Sinais](#ecg-e-sinais)
  - [Imagem Médica](#imagem-médica-1)
  - [NLP Clínico](#nlp-clínico)
  - [EHR e Dados Tabulares](#ehr-e-dados-tabulares)
- [Conjuntos de Dados (Datasets)](#conjuntos-de-dados-datasets)
  - [Prontuários Eletrônicos (EHR)](#prontuários-eletrônicos-ehr)
  - [Imagens Médicas](#imagens-médicas)
  - [Sinais Biológicos](#sinais-biológicos)
  - [Texto e NLP](#texto-e-nlp)
  - [Genômica e Bioinformática](#genômica-e-bioinformática)
  - [Fármacos e Moléculas](#fármacos-e-moléculas)
- [Ferramentas & Bibliotecas](#ferramentas--bibliotecas)
  - [Imagem Médica](#imagem-médica-2)
  - [Análise de Sobrevida e EHR](#análise-de-sobrevida-e-ehr)
  - [NLP em Saúde](#nlp-em-saúde-1)
  - [Explicabilidade e Fairness](#explicabilidade-e-fairness)
  - [Interoperabilidade e Padrões](#interoperabilidade-e-padrões)
- [Modelos Pré-treinados](#modelos-pré-treinados)
  - [Modelos de Linguagem (LLMs)](#modelos-de-linguagem-llms)
  - [Modelos de Visão](#modelos-de-visão)
  - [Modelos Multimodais](#modelos-multimodais)
  - [Modelos para Sinais (ECG/EEG)](#modelos-para-sinais-ecgeeg)
- [Competições](#competições)
  - [Kaggle](#kaggle)
  - [Grand Challenge](#grand-challenge)
  - [PhysioNet](#physionet)
  - [MICCAI](#miccai)
  - [Outros](#outros)
- [Contribuindo](#contribuindo)
- [Licença](#licença)

---

## Artigos (Papers)

### Revisões e Surveys

- [Artificial Intelligence in Healthcare: 2024 Year in Review (2025)](https://www.medrxiv.org/content/10.1101/2025.02.26.25322978v1.full-text) - Uma revisão abrangente das publicações de IA na saúde em 2024.
- [AI, Health, and Health Care Today and Tomorrow (2025)](https://jamanetwork.com/journals/jama/fullarticle/2840175) - Discussão sobre como a IA na saúde deve ser desenvolvida, avaliada e regulada. (JAMA)
- [Artificial intelligence in healthcare and medicine (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12455834/) - Revisão do papel da IA na detecção de doenças, cuidado personalizado, drug discovery e análise preditiva.
- [Foundation Model in Biomedicine: A Survey (2025)](https://arxiv.org/abs/2503.02104) - Survey sobre o uso de modelos de fundação em áreas biomédicas.
- [Future of AI-ML in Pathology and Medicine (2025)](https://www.sciencedirect.com/science/article/pii/S0893395225000018) - Revisão sobre adoção e direções futuras de plataformas de IA-ML em patologia.
- [A Comprehensive Survey of Foundation Models in Medicine (2024)](https://arxiv.org/abs/2406.10729) - Survey completo sobre modelos de fundação na medicina.
- [Deep Learning for Healthcare Review (2017)](https://www.ncbi.nlm.nih.gov/pubmed/28481991) - Revisão seminal sobre oportunidades e desafios de DL na saúde.

### Modelos Preditivos e EHR

- [A deep learning model for clinical outcome prediction using longitudinal inpatient EHR (2025)](https://academic.oup.com/jamiaopen/article/8/2/ooaf026/8110091) - Modelo TECO baseado em Transformer para prever desfechos clínicos a partir de dados de prontuário.
- [Deep Learning prediction models based on EHR trajectories: A systematic review (2023)](https://www.sciencedirect.com/science/article/pii/S153204642300151X) - Revisão sistemática de modelos de DL baseados em trajetórias de EHR.
- [Predicting healthcare trajectories from medical records (2017)](https://www.sciencedirect.com/science/article/pii/S1532046417300710) - Rede neural profunda para ler registros médicos e prever desfechos futuros.
- [Clinical prediction models using machine learning in oncology (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12506039/) - Modelos de predição clínica usando ML em oncologia.
- [Updating methods for AI-based clinical prediction models (2024)](https://www.sciencedirect.com/science/article/pii/S0895435624003925) - Métodos para atualizar modelos de predição clínica baseados em IA.
- [MiME: Multilevel Medical Embedding of EHR for Predictive Healthcare (2018)](https://nips.cc/Conferences/2018/Schedule?showEvent=11448) - Embedding multinível de EHR para saúde preditiva. (NeurIPS)
- [Patient2Vec: A Personalized Interpretable Deep Representation of the Longitudinal EHR (2018)](https://arxiv.org/abs/1810.04793) - Representação profunda e interpretável de EHR longitudinal.
- [Opportunistic Learning: Budgeted Cost-Sensitive Learning from Data Streams (2019)](https://openreview.net/pdf?id=S1eOHo09KX) - Aprendizado sensível a custo com aplicação em diabetes.

### Imagem Médica

- [Deep learning-based image classification for integrating pathology and radiology (2025)](https://www.nature.com/articles/s41598-025-07883-w) - Framework para integrar dados multimodais de patologia e radiologia.
- [Generative AI and Foundation Models in Radiology (2025)](https://pubs.rsna.org/doi/10.1148/radiol.242961) - Revisão sobre modelos de fundação em radiologia.
- [A novel explainable AI framework for medical image classification (2025)](https://www.sciencedirect.com/science/article/pii/S1361841525002129) - Método de XAI para análise de imagens médicas com explicações estatísticas, visuais e baseadas em regras.
- [Deep learning-based image classification for AI-assisted medical imaging (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12171332/) - Framework de DL para classificação de imagens médicas.
- [Adversarial Attacks Against Medical Deep Learning Systems (2018)](https://arxiv.org/pdf/1804.05296.pdf) - Estudo sobre ataques adversariais contra sistemas de DL médico.

### NLP em Saúde

- [The Growing Impact of NLP in Healthcare (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11475376/) - Revisão narrativa sobre os usos atuais de PNL na saúde.
- [BioBERT: a pre-trained biomedical language representation model (2020)](https://academic.oup.com/bioinformatics/article/36/4/1234/5566506) - Modelo de linguagem pré-treinado em literatura biomédica.
- [Large language models encode clinical knowledge (2023)](https://www.nature.com/articles/s41586-023-06291-2) - Avaliação do Med-PaLM em exames de competência médica. (Nature)
- [Ascle: A Python NLP Toolkit for Medical Text Generation (2024)](https://www.jmir.org/2024/1/e60601/) - Toolkit de PNL para geração de texto médico.
- [Towards Automating Healthcare Question Answering in a Noisy Multilingual Low-Resource Setting (2019)](https://www.aclweb.org/anthology/P19-1090) - QA automatizado em saúde em cenários multilíngues.
- [Label-aware Double Transfer Learning for Cross-Specialty Medical NER (2018)](http://aclweb.org/anthology/N18-1001) - Transferência de aprendizado para reconhecimento de entidades médicas.
- [EMER-QA: A large corpus for question answering on electronic medical records (2018)](https://arxiv.org/pdf/1809.00732.pdf) - Grande corpus para QA em prontuários eletrônicos.

### ECG e Sinais Biológicos

- [Cardiologist-level arrhythmia detection and classification in ambulatory ECGs using a deep neural network (2019)](https://www.nature.com/articles/s41591-018-0268-3) - Detecção de arritmias em nível de cardiologista usando DNN. (Nature Medicine)
- [Deep learning prediction of all-cause mortality from 12-lead ECGs (2020)](https://www.nature.com/articles/s41591-020-0870-z) - Predição de mortalidade por todas as causas a partir de ECG de 12 derivações. (Nature Medicine)
- [Automatic diagnosis of the 12-lead ECG using a deep neural network (2020)](https://www.nature.com/articles/s41467-020-15432-4) - Diagnóstico automático de ECG de 12 derivações usando DNN. (Nature Communications)
- [An artificial intelligence-enabled ECG algorithm for the identification of patients with atrial fibrillation (2019)](https://www.nature.com/articles/s41591-018-0339-5) - Algoritmo de IA para identificar fibrilação atrial a partir de ECG em ritmo sinusal. (Nature Medicine)

### Drug Discovery

- [GAMENet: Graph Augmented MEmory Networks for Recommending Medication Combination (2018)](https://arxiv.org/abs/1809.01852) - Redes de memória aumentadas por grafos para recomendação de combinações de medicamentos.
- [Novel Exploration Techniques (NETs) for Malaria Policy Interventions (2017)](https://arxiv.org/pdf/1712.00428.pdf) - Técnicas de exploração para intervenções políticas contra malária.

### IA Explicável (XAI) em Saúde

- [Uncertainty-Aware Attention for Reliable Interpretation and Prediction (2018)](https://nips.cc/Conferences/2018/Schedule?showEvent=11112) - Atenção com consciência de incerteza para interpretação confiável. (NeurIPS)
- [Direct Uncertainty Prediction for Medical Second Opinions (2018)](https://arxiv.org/abs/1807.01771) - Predição direta de incerteza para segundas opiniões médicas.
- [Leveraging uncertainty information from deep neural networks for disease detection (2017)](https://www.nature.com/articles/s41598-017-17876-z) - Uso de informação de incerteza de DNNs para detecção de doenças.

### Ética e Fairness em IA na Saúde

- [Artificial Intelligence In Health And Health Care: Priorities (2025)](https://www.healthaffairs.org/doi/10.1377/hlthaff.2024.01003) - Prioridades para uso seguro e eficaz de IA na saúde. (Health Affairs)
- [Implementation and Updating of Clinical Prediction Models (2025)](https://www.mcpdigitalhealth.org/article/S2949-7612(25)00035-5/fulltext) - Abordagens de implementação e atualização de modelos clínicos.

---

## Mão na Massa & Código

### Repositórios e Tutoriais Gerais

- [aws-samples/aws-healthcare-lifescience-ai-ml-sample-notebooks](https://github.com/aws-samples/aws-healthcare-lifescience-ai-ml-sample-notebooks) - Notebooks de exemplo para IA/ML em saúde e ciências da vida na AWS.
- [NIGMS/AI-ML-For-Biomedical-Researchers](https://github.com/NIGMS/AI-ML-For-Biomedical-Researchers) - Módulo prático de IA/ML centrado em dados para pesquisadores biomédicos, cobrindo R, preparação de dados e construção de modelos.
- [ageron/handson-ml3](https://github.com/ageron/handson-ml3) - Notebooks Jupyter que percorrem os fundamentos de ML e DL em Python usando Scikit-Learn, Keras e TensorFlow.
- [BridgingAISocietySummerSchools/Hands-On-Notebooks](https://github.com/BridgingAISocietySummerSchools/Hands-On-Notebooks) - Notebooks para introduzir conceitos essenciais de ML usando código real.
- [Project-MONAI/tutorials](https://github.com/Project-MONAI/tutorials) - Tutoriais oficiais do MONAI para segmentação, classificação e detecção em imagens médicas.
- [medtorch/awesome-healthcare-ai](https://github.com/medtorch/awesome-healthcare-ai) - Lista curada de ferramentas, algoritmos e datasets de IA na saúde.
- [kakoni/awesome-healthcare](https://github.com/kakoni/awesome-healthcare) - Lista curada de software open source, bibliotecas e recursos para saúde.

### ECG e Sinais

- [DeepPSP/torch_ecg](https://github.com/DeepPSP/torch_ecg) - Modelos de DL para ECG em PyTorch, com ferramentas para processamento de sinais e treinamento.
- [eddymina/ECG_Classification_Pytorch](https://github.com/eddymina/ECG_Classification_Pytorch) - Aplicação de redes convolucionais 1D para classificação de ECG.
- [jarek-pawlowski/machine-learning-applications (ECG notebook)](https://github.com/jarek-pawlowski/machine-learning-applications/blob/main/ecg_classification.ipynb) - Notebook sobre classificação automática de sinais de ECG.
- [cepdnaclk/Comprehensive-ECG-analysis-with-Deep-Learning](https://github.com/cepdnaclk/e17-4yp-Comprehensive-ECG-analysis-with-Deep-Learning-on-GPU-accelerators) - Análise abrangente de ECG com DL em aceleradores GPU.

### Imagem Médica

- [Project-MONAI/tutorials](https://github.com/Project-MONAI/tutorials) - Tutoriais oficiais do MONAI para segmentação, classificação e detecção.
- [intel/fl-tutorial](https://github.com/intel/fl-tutorial) - Tutorial oficial da MICCAI 2022 sobre Aprendizado Federado para Saúde.
- [openmedlab/Awesome-Medical-Dataset](https://github.com/openmedlab/Awesome-Medical-Dataset) - Lista de datasets médicos proeminentes, especialmente para imagem.

### NLP Clínico

- [EmilyAlsentzer/clinicalBERT](https://github.com/EmilyAlsentzer/clinicalBERT) - Modelos BERT para notas clínicas, ajustados em dados do MIMIC-III.
- [dmis-lab/biobert](https://github.com/dmis-lab/biobert) - Modelo de linguagem pré-treinado em literatura biomédica para mineração de texto.
- [Ascle NLP Toolkit](https://www.jmir.org/2024/1/e60601/) - Toolkit de PNL para geração de texto médico.

### EHR e Dados Tabulares

- [sunlabuiuc/PyHealth](https://github.com/sunlabuiuc/PyHealth) - Biblioteca Python para construir modelos de ML em dados clínicos.
- [baeseongsu/awesome-machine-learning-for-healthcare](https://github.com/baeseongsu/awesome-machine-learning-for-healthcare) - Coleção curada de pesquisas na interseção de ML e saúde.

---

## Conjuntos de Dados (Datasets)

### Prontuários Eletrônicos (EHR)

| Dataset | Descrição | Link |
|---------|-----------|------|
| **MIMIC-IV** | Grande banco de dados de acesso público com dados de UTI do Beth Israel Deaconess Medical Center, cobrindo admissões de 2008 a 2019. | [PhysioNet](https://physionet.org/content/mimiciv/) |
| **MIMIC-III** | Versão anterior do MIMIC com 58.976 admissões hospitalares de 38.597 pacientes. | [PhysioNet](https://physionet.org/content/mimiciii/) |
| **eICU** | Base de dados multicêntrica com dados de alta granularidade de UTIs nos EUA. | [PhysioNet](https://eicu-crd.mit.edu/) |
| **CPRD** | Banco de dados de atenção primária do Reino Unido com mais de 60 milhões de pacientes. | [CPRD](https://cprd.com/data) |

### Imagens Médicas

| Dataset | Descrição | Link |
|---------|-----------|------|
| **CheXpert** | 224.316 radiografias de tórax com labels estruturados de 14 observações. | [Stanford ML Group](https://stanfordmlgroup.github.io/competitions/chexpert/) |
| **MIMIC-CXR** | 227.835 estudos de imagem para 64.588 pacientes com relatórios de radiologia. | [PhysioNet](https://physionet.org/content/mimic-cxr/) |
| **PadChest** | 160.000 imagens de raio-X de tórax com anotações de mais de 170 achados. | [BIMCV](http://bimcv.cipf.es/bimcv-projects/padchest/) |
| **TCIA** | Grande arquivo de imagens médicas de câncer em formato DICOM. | [TCIA](https://www.cancerimagingarchive.net/) |
| **Medical Segmentation Decathlon** | 10 tarefas de segmentação cobrindo vários órgãos e patologias. | [MSD](http://medicaldecathlon.com/) |
| **ISIC Archive** | Grande coleção de imagens de lesões de pele para classificação de melanoma. | [ISIC](https://www.isic-archive.com/) |
| **ROCO** | 81.000 imagens de radiologia com legendas correspondentes. | [GitHub](https://github.com/razorx89/roco-dataset) |
| **PMC-OA** | 1.6M pares imagem-legenda de artigos do PubMed Central. | [HuggingFace](https://huggingface.co/datasets/axiong/pmc_oa_beta) |
| **OpenI** | 3.7 milhões de imagens de cerca de 1.2 milhões de artigos. | [OpenI](https://openi.nlm.nih.gov/) |

### Sinais Biológicos

| Dataset | Descrição | Link |
|---------|-----------|------|
| **PTB-XL** | Grande dataset de ECG com 21.837 registros de 12 derivações e anotações clínicas. | [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/) |
| **PhysioNet Databases** | Repositório de sinais fisiológicos incluindo ECG, EEG, EMG e mais. | [PhysioNet](https://physionet.org/about/database/) |
| **MIT-BIH Arrhythmia Database** | Dataset clássico de arritmias com 48 registros de ECG de meia hora. | [PhysioNet](https://physionet.org/content/mitdb/1.0.0/) |
| **AFDB** | Banco de dados de fibrilação atrial do MIT-BIH. | [PhysioNet](https://physionet.org/content/afdb/1.0.0/) |

### Texto e NLP

| Dataset | Descrição | Link |
|---------|-----------|------|
| **PubMed** | 35M citações e resumos de literatura biomédica. | [PubMed](https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/) |
| **PMC** | 8 milhões de registros de artigos completos. | [PMC](https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk) |
| **CORD-19** | 1M artigos sobre COVID-19 e coronavírus. | [GitHub](https://github.com/allenai/cord19) |
| **MedDialog** | 3.66 milhões de conversas médicas. | [GitHub](https://github.com/UCSD-AI4H/COVID-Dialogue) |
| **MeQSum** | 1.000 instâncias de sumarização de perguntas médicas. | [GitHub](https://github.com/abachaa/MeQSum) |
| **UMLS** | Base de conhecimento com 2M entidades para 900K conceitos. | [NLM](https://www.nlm.nih.gov/research/umls/index.html) |
| **Medical Transcriptions** | Transcrições médicas para NLP. | [Kaggle](https://www.kaggle.com/tboyle10/medicaltranscriptions) |

### Genômica e Bioinformática

| Dataset | Descrição | Link |
|---------|-----------|------|
| **UK Biobank** | Grande estudo de coorte com dados genômicos e de saúde de 500.000 participantes. | [UK Biobank](https://www.ukbiobank.ac.uk/) |
| **TCGA** | The Cancer Genome Atlas: dados genômicos de mais de 20.000 amostras de câncer. | [NCI](https://www.cancer.gov/tcga) |
| **1000 Genomes** | Catálogo de variação genética humana. | [1000 Genomes](https://www.internationalgenome.org/) |

### Fármacos e Moléculas

| Dataset | Descrição | Link |
|---------|-----------|------|
| **ChEMBL** | Banco de dados de moléculas bioativas com propriedades drug-like. | [ChEMBL](https://www.ebi.ac.uk/chembl/) |
| **DrugBank** | Base de dados abrangente de informações sobre fármacos. | [DrugBank](https://go.drugbank.com/) |
| **PubChem** | Banco de dados de química com informações sobre estruturas, propriedades e atividades biológicas. | [PubChem](https://pubchem.ncbi.nlm.nih.gov/) |

---

## Ferramentas & Bibliotecas

### Imagem Médica

- [MONAI](https://monai.io/) - Toolkit de IA de código aberto para imagem médica, baseado em PyTorch. [![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/MONAI.svg?style=social)](https://github.com/Project-MONAI/MONAI)
- [MONAI Label](https://github.com/Project-MONAI/MONAILabel) - Ferramenta inteligente de rotulagem de imagens e aprendizado para criar datasets anotados.
- [TorchXRayVision](https://github.com/mlmed/torchxrayvision) - Biblioteca para trabalhar com radiografias de tórax com modelos pré-treinados.
- [MedPy](https://loli.github.io/medpy/) - Biblioteca de processamento de imagens médicas em Python.
- [SimpleITK](https://simpleitk.org/) - Interface simplificada para o Insight Toolkit (ITK) para análise de imagens.
- [3D Slicer](https://www.slicer.org/) - Plataforma de código aberto para visualização e análise de imagens médicas.

### Análise de Sobrevida e EHR

- [PyHealth](https://pyhealth.readthedocs.io/) - Biblioteca Python abrangente para IA na saúde, facilitando a construção de modelos de ML em dados clínicos. [![GitHub stars](https://img.shields.io/github/stars/sunlabuiuc/PyHealth.svg?style=social)](https://github.com/sunlabuiuc/PyHealth)
- [scikit-survival](https://scikit-survival.readthedocs.io/) - Análise de sobrevida construída sobre o Scikit-learn.
- [lifelines](https://lifelines.readthedocs.io/) - Implementação de modelos de análise de sobrevida em Python (Kaplan-Meier, Cox PH, etc.).
- [CatBoost](https://catboost.ai/) - Gradient boosting com suporte nativo a variáveis categóricas, amplamente usado em dados tabulares de saúde.
- [XGBoost](https://xgboost.readthedocs.io/) - Implementação otimizada de gradient boosting.
- [LightGBM](https://lightgbm.readthedocs.io/) - Framework de gradient boosting rápido e eficiente.
- [TabPFN](https://github.com/automl/TabPFN) - Prior-data Fitted Networks para classificação tabular.

### NLP em Saúde

- [John Snow Labs Spark NLP for Healthcare](https://www.johnsnowlabs.com/healthcare-nlp/) - NLP de estado da arte para saúde, com modelos para NER, de-identificação e extração de relações.
- [spaCy](https://spacy.io/) - Biblioteca de PNL industrial com suporte a modelos biomédicos (scispaCy).
- [scispaCy](https://allenai.github.io/scispacy/) - Modelos spaCy treinados em texto biomédico.
- [Hugging Face Transformers](https://huggingface.co/transformers/) - Biblioteca de modelos de linguagem com muitos modelos biomédicos disponíveis.
- [cTAKES](http://ctakes.apache.org/) - Clinical Text Analysis Knowledge Extraction System da Apache.
- [CliNER](https://github.com/text-machine-lab/CliNER) - Sistema de reconhecimento de entidades nomeadas clínicas.

### Explicabilidade e Fairness

- [SHAP](https://github.com/shap/shap) - SHapley Additive exPlanations para explicar a saída de qualquer modelo de ML.
- [LIME](https://github.com/marcotcr/lime) - Local Interpretable Model-agnostic Explanations.
- [InterpretML](https://interpret.ml/) - Toolkit de interpretabilidade de ML da Microsoft.
- [Fairlearn](https://fairlearn.org/) - Toolkit para avaliar e melhorar a fairness de modelos de ML.
- [AIF360](https://aif360.mybluemix.net/) - AI Fairness 360 da IBM.
- [MCBoost](https://github.com/mlr-org/mcboost) - Multicalibração para melhorar a calibração de modelos por subgrupos.

### Interoperabilidade e Padrões

- [FHIR (Fast Healthcare Interoperability Resources)](https://www.hl7.org/fhir/) - Padrão de interoperabilidade para troca de dados de saúde.
- [fhir.resources](https://pypi.org/project/fhir.resources/) - Modelos Pydantic para recursos FHIR em Python.
- [HAPI FHIR](https://hapifhir.io/) - Implementação de referência do FHIR em Java.
- [PySUS](https://github.com/AlertaDengue/PySUS) - Biblioteca Python para acessar dados do DATASUS (Brasil).

---

## Modelos Pré-treinados

### Modelos de Linguagem (LLMs)

| Modelo | Descrição | Link |
|--------|-----------|------|
| **Med-PaLM 2** | LLM do Google projetado para responder perguntas médicas com qualidade de especialista. | [Google Research](https://sites.research.google/gr/med-palm/) |
| **MedGemma** | Modelos abertos do Google para IA na saúde (Health AI Developer Foundations). | [Kaggle](https://www.kaggle.com/competitions/med-gemma-impact-challenge) |
| **BioGPT** | LLM da Microsoft pré-treinado em literatura biomédica. | [GitHub](https://github.com/microsoft/BioGPT) |
| **GatorTron** | Modelo de linguagem clínica treinado em 90B palavras de texto clínico. | [GitHub](https://github.com/uf-hobi-informatics-lab/GatorTron) |
| **PubMedBERT** | BERT pré-treinado do zero em resumos e textos completos do PubMed. | [HuggingFace](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext) |
| **ClinicalBERT** | BERT ajustado em notas clínicas do MIMIC-III. | [GitHub](https://github.com/EmilyAlsentzer/clinicalBERT) |
| **BioBERT** | BERT pré-treinado em literatura biomédica (PubMed + PMC). | [GitHub](https://github.com/dmis-lab/biobert) |
| **SciBERT** | BERT pré-treinado em artigos científicos. | [GitHub](https://github.com/allenai/scibert) |

### Modelos de Visão

| Modelo | Descrição | Link |
|--------|-----------|------|
| **RadImageNet** | Modelo de classificação treinado em grande dataset de radiologia. | [GitHub](https://github.com/BMEII-AI/RadImageNet) |
| **MedSAM** | Segment Anything Model adaptado para segmentação de imagens médicas. | [GitHub](https://github.com/bowang-lab/MedSAM) |
| **TorchXRayVision Models** | Modelos pré-treinados para classificação de radiografias de tórax. | [GitHub](https://github.com/mlmed/torchxrayvision) |
| **RetFound** | Foundation model para imagens de retina. | [GitHub](https://github.com/rmaphoh/RETFound_MAE) |

### Modelos Multimodais

| Modelo | Descrição | Link |
|--------|-----------|------|
| **BiomedCLIP** | Modelo de visão-linguagem contrastivo para o domínio biomédico. | [HuggingFace](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) |
| **MedCLIP** | Modelo CLIP adaptado para imagens médicas e texto clínico. | [GitHub](https://github.com/RyanWangZf/MedCLIP) |
| **LLaVA-Med** | Modelo multimodal de linguagem e visão para biomedicina. | [GitHub](https://github.com/microsoft/LLaVA-Med) |

### Modelos para Sinais (ECG/EEG)

| Modelo | Descrição | Link |
|--------|-----------|------|
| **torch_ecg models** | Modelos de DL para ECG em PyTorch. | [GitHub](https://github.com/DeepPSP/torch_ecg) |
| **ECG-DL** | Modelos de deep learning para classificação de ECG. | [GitHub](https://github.com/eddymina/ECG_Classification_Pytorch) |

---

## Competições

### Kaggle

- [The MedGemma Impact Challenge (2026)](https://www.kaggle.com/competitions/med-gemma-impact-challenge) - Construir aplicações de IA centradas no ser humano com MedGemma e outros modelos abertos do Google.
- [Medical AI: Medical Diagnosis Challenge (2025)](https://www.kaggle.com/competitions/cm-medical-prognosis) - Explorar fatores demográficos, clínicos e de estilo de vida que influenciam o prognóstico de longo prazo.
- [Kaggle Healthcare Competitions](https://www.kaggle.com/competitions?tagIds=4202-Healthcare) - Todas as competições de saúde no Kaggle.
- [RSNA Screening Mammography Breast Cancer Detection](https://www.kaggle.com/competitions/rsna-breast-cancer-detection) - Detecção de câncer de mama em mamografias.
- [RSNA Intracranial Hemorrhage Detection](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection) - Detecção de hemorragia intracraniana em tomografias.
- [SIIM-ISIC Melanoma Classification](https://www.kaggle.com/competitions/siim-isic-melanoma-classification) - Classificação de melanoma em imagens de lesões de pele.

### Grand Challenge

- [Grand Challenge Platform](https://grand-challenge.org/challenges/) - Plataforma que hospeda desafios em imagem médica e IA, com dezenas de desafios ativos.

### PhysioNet

- [PhysioNet/CinC Challenges](https://physionet.org/challenge/) - Desafios anuais focados em problemas clínicos usando dados fisiológicos.
- [George B. Moody PhysioNet Challenge 2022](https://physionet.org/content/challenge-2022/) - Detecção de sopros cardíacos a partir de fonocardiogramas.
- [George B. Moody PhysioNet Challenge 2021](https://physionet.org/content/challenge-2021/) - Classificação de ECG de múltiplas derivações.

### MICCAI

- [MICCAI Registered Challenges](https://miccai.org/index.php/special-interest-groups/challenges/miccai-registered-challenges/) - Desafios realizados em conjunto com a conferência MICCAI.

### Outros

- [DrivenData](https://www.drivendata.org/) - Competições de ciência de dados para impacto social, incluindo saúde.
- [ISBI Challenges](https://biomedicalimaging.org/) - Desafios de imagem biomédica do IEEE International Symposium on Biomedical Imaging.

---

## Contribuindo

Suas contribuições são sempre bem-vindas! Por favor, dê uma olhada nas [diretrizes de contribuição](CONTRIBUTING.md) primeiro.

## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para mais detalhes.
