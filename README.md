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
- [Eventos & Conferências](#eventos--conferências)
  - [CHIL 2026](#chil-2026-conference-on-health-inference-and-learning)
- [Contribuindo](#contribuindo)
- [Licença](#licença)

---

## Artigos (Papers)

### Revisões e Surveys

- [Decoding the language of sleep with artificial intelligence (2026)](https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(26)00375-2/abstract) - James Zou e Eric Topol discutem como a IA pode transformar o sono em um dado médico acionável, integrando sinais cerebrais, cardíacos e respiratórios. (The Lancet)
- [A comparison of self-rated health, frailty index, and multimorbidity as predictors of four-year all-cause mortality: a multi-cohort study in 30+ adult populations (2026)](https://www.researchsquare.com/article/rs-8099259/latest.pdf) - Estudo multicoorte em 30+ populações adultas comparando preditores de mortalidade. (Research Square, relacionado a Chiavegatto Filho)
- [The contribution of air pollution to the socioeconomic inequalities in dementia risk in later life: The Whitehall II cohort study (2026)](https://wellcomeopenresearch.org/articles/11-158) - Contribuição da poluição do ar para desigualdades socioeconômicas no risco de demência. (Wellcome Open Research)
- [COVID-19 Lockdowns and Gestational Diabetes Risk: Identifying Critical Windows for Prevention (2026)](https://www.researchsquare.com/article/rs-8927540/latest) - Lockdowns e risco de diabetes gestacional: janelas críticas para prevenção. (Research Square)
- [Socioeconomic status and suicidal behaviour in the UK Biobank: prospective cohort study on effect modifiers (2026)](https://www.sciencedirect.com/science/article/pii/S1353829226000328) - SES e comportamento suicida no UK Biobank com modificadores de efeito. (Health & Place)
- [Physical Activity Trajectories Throughout the COVID-19 Pandemic in Southern Brazil: Evidence From a Cohort Study (2026)](https://journals.humankinetics.com/view/journals/jpah/aop/article-10.1123-jpah.2025-0158/article-10.1123-jpah.2025-0158.xml) - Estudo de coorte sobre trajetórias de atividade física durante a pandemia no Sul do Brasil. (Journal of Physical Activity and Health)
- [Impact of Long-Term Cumulative Exposure to Wildfire Smoke PM2.5 on Cardiovascular Hospital Admissions Among Older Adults (2026)](https://www.sciencedirect.com/science/article/pii/S0735109725106608) - Estudo de coorte sobre o impacto da exposição cumulativa à fumaça de incêndios florestais em internações cardiovasculares. (JACC)
- [Extensive antibiotic use during early life and risk of frailty: a population-based cohort study in the UK Biobank (2026)](https://link.springer.com/article/10.1186/s12889-026-26612-0) - Uso extensivo de antibióticos na infância e risco de fragilidade na vida adulta. (BMC Public Health)
- [Cohort Prevalence Estimates Are Sensitive to Prebaseline Mortality: HAALSI Cohort in Rural South Africa (2026)](https://read.dukeupress.edu/demography/article-pdf/doi/10.1215/00703370-12479730/2379471/12479730.pdf) - Nota metodológica sobre viés de mortalidade pré-baseline em estudos de coorte. (Demography)
- [Severe infections, domain-specific cognitive vulnerability, and future infection risk in older adults (2026)](https://www.medrxiv.org/content/10.64898/2026.02.17.26346454.full.pdf) - Associação entre infecções graves e vulnerabilidade cognitiva em idosos. (medRxiv)
- [Impact of the COVID-19 pandemic on the health situation of the Brazilian population (2026)](https://www.sciencedirect.com/science/article/pii/S1413867026011803) - Análise do impacto da pandemia de COVID-19 na situação de saúde da população brasileira. (Brazilian Journal of Infectious Diseases)
- [Experiências adversas na infância e desempenho físico na vida adulta: ELSA-Brasil (2026)](https://www.scielo.br/j/csp/a/SPTpMqZV94sR5DhKfKYptkx/abstract/?lang=pt) - Associações entre experiências adversas na infância e desempenho físico em adultos brasileiros. (Cadernos de Saúde Pública)
- [Sustained physical activity offers benefits beyond activity volume in chronic disease prevention (2026)](https://www.nature.com/articles/s41467-026-69552-4_reference.pdf) - Estudo longitudinal sobre os benefícios da atividade física sustentada na prevenção de doenças crônicas. (Nature Communications)
- [Neonatal mortality trends in the 21st century: findings from the global burden of disease study 2021 (2026)](https://pubmed.ncbi.nlm.nih.gov/41688040/) - Análise de tendências e causas de mortalidade neonatal de 2000 a 2021. (Jornal de Pediatria)
- [Impact of cooking with liquefied petroleum gas compared with traditional cooking practices on perinatal and early neonatal mortality (2026)](https://gh.bmj.com/content/11/2/e020391) - Ensaio clínico randomizado sobre o impacto do uso de gás de cozinha na mortalidade perinatal. (BMJ Global Health)
- [Artificial Intelligence in Healthcare: 2024 Year in Review (2025)](https://www.medrxiv.org/content/10.1101/2025.02.26.25322978v1.full-text) - Uma revisão abrangente das publicações de IA na saúde em 2024.
- [AI, Health, and Health Care Today and Tomorrow (2025)](https://jamanetwork.com/journals/jama/fullarticle/2840175) - Discussão sobre como a IA na saúde deve ser desenvolvida, avaliada e regulada. (JAMA)
- [Artificial intelligence in healthcare and medicine (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12455834/) - Revisão do papel da IA na detecção de doenças, cuidado personalizado, drug discovery e análise preditiva.
- [Foundation Model in Biomedicine: A Survey (2025)](https://arxiv.org/abs/2503.02104) - Survey sobre o uso de modelos de fundação em áreas biomédicas.
- [Future of AI-ML in Pathology and Medicine (2025)](https://www.sciencedirect.com/science/article/pii/S0893395225000018) - Revisão sobre adoção e direções futuras de plataformas de IA-ML em patologia.
- [A Comprehensive Survey of Foundation Models in Medicine (2024)](https://arxiv.org/abs/2406.10729) - Survey completo sobre modelos de fundação na medicina.
- [Deep Learning for Healthcare Review (2017)](https://www.ncbi.nlm.nih.gov/pubmed/28481991) - Revisão seminal sobre oportunidades e desafios de DL na saúde.

### Modelos Preditivos e EHR

- [Evidence of Unreliable Data and Poor Data Provenance in Clinical Prediction Model Research and Clinical Practice (2026)](https://www.medrxiv.org/content/medrxiv/early/2026/02/26/2026.02.24.26347028.full.pdf) - Gibson, White, Collins e Barnett demonstram que modelos de predição clínica frequentemente são criados com dados não confiáveis e sem proveniência adequada. (medRxiv, Gary S. Collins)
- [NIVPredict: AI Tool for Early Prediction of NIV Outcome in Acute Respiratory Failure (2026)](https://healthmanagement.org/s/nivpredict-ai-tool-for-early-prediction-of-niv-outcome-in-acute-respiratory-failure) - Ferramenta web baseada em TabPFN para predição de falha de ventilação não-invasiva. AUC 0.858, Brier Score 0.093, validado em 38 hospitais de 4 países. (Critical Care)
- [Human-Guided Agentic AI for Multimodal Clinical Prediction (2026)](https://arxiv.org/html/2602.19502v1) - Sistemas de IA agnósticos com orientação humana para predição clínica multimodal. F1=0.8986 em readmissão hospitalar. (arXiv)
- [A Feature-Stable and Explainable ML Framework for Trustworthy Decision-Making (2026)](https://arxiv.org/html/2602.17364v1) - Framework de ML explicável com dados incompletos usando CatBoost, XGBoost, LightGBM e SHAP. (arXiv)
- [Machine Learning Based Prediction of Surgical Outcomes in Chronic Rhinosinusitis (2026)](https://arxiv.org/html/2602.17888v1) - ML para recomendação cirúrgica em rinossinusite crônica: ~85% de acurácia, superando média de especialistas (75.6%). (Computers in Biology and Medicine)
- [Benchmarking LLMs for Predictive Modeling in Biomedical Research (2026)](https://www.cell.com/cell-reports-medicine/fulltext/S2666-3791(26)00011-X) - LLMs gerando código R/Python para tarefas de predição em dados ômicos, igualando performance mediana de participantes do DREAM Challenge. (Cell Reports Medicine)
- [Hybrid Federated and Split Learning for Privacy Preserving Clinical Prediction (2026)](https://arxiv.org/abs/2602.15304) - Arquitetura híbrida de aprendizado federado e split learning para predição clínica sem centralizar dados. (arXiv)
- [Making Conformal Predictors Robust in Healthcare Settings (2026)](https://arxiv.org/html/2602.19483v1) - Predição conformal com garantias estatísticas para segurança do paciente em IA clínica. (arXiv)
- [Query Optimization with LLMs (2026)](https://arxiv.org/pdf/2602.10387) - Uso de modelos de linguagem grandes para otimização de queries em bases de dados. (arXiv)
- [Protocol for development of a reporting guideline (TRIPOD-Code) for code repositories associated with diagnostic and prognostic prediction model studies (2026)](https://scholar.google.com/scholar?q=TRIPOD-Code) - Protocolo para guidelines de reporte de repositórios de código em estudos de modelos preditivos diagnósticos e prognósticos.
- [A deep learning model for clinical outcome prediction using longitudinal inpatient EHR (2025)](https://academic.oup.com/jamiaopen/article/8/2/ooaf026/8110091) - Modelo TECO baseado em Transformer para prever desfechos clínicos a partir de dados de prontuário.
- [Deep Learning prediction models based on EHR trajectories: A systematic review (2023)](https://www.sciencedirect.com/science/article/pii/S153204642300151X) - Revisão sistemática de modelos de DL baseados em trajetórias de EHR.
- [Predicting healthcare trajectories from medical records (2017)](https://www.sciencedirect.com/science/article/pii/S1532046417300710) - Rede neural profunda para ler registros médicos e prever desfechos futuros.
- [Clinical prediction models using machine learning in oncology (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12506039/) - Modelos de predição clínica usando ML em oncologia.
- [Updating methods for AI-based clinical prediction models (2024)](https://www.sciencedirect.com/science/article/pii/S0895435624003925) - Métodos para atualizar modelos de predição clínica baseados em IA.
- [MiME: Multilevel Medical Embedding of EHR for Predictive Healthcare (2018)](https://nips.cc/Conferences/2018/Schedule?showEvent=11448) - Embedding multinível de EHR para saúde preditiva. (NeurIPS)
- [Patient2Vec: A Personalized Interpretable Deep Representation of the Longitudinal EHR (2018)](https://arxiv.org/abs/1810.04793) - Representação profunda e interpretável de EHR longitudinal.
- [Opportunistic Learning: Budgeted Cost-Sensitive Learning from Data Streams (2019)](https://openreview.net/pdf?id=S1eOHo09KX) - Aprendizado sensível a custo com aplicação em diabetes.

### Imagem Médica

- [Pillar-0: Novo SOTA em Radiologia 3D (2026)](https://arxiv.org/abs/2602.pillar0) - Modelo que supera MedImageInsight e MedGemma em tarefas de radiologia 3D. Apresentado internamente na Hapvida como referência de estado da arte.
- [Pneumonia Detection from Enhanced Chest X-Ray Images Based on Double SGAN (2026)](https://www.nature.com/articles/s41598-026-39785-w) - Double SGAN com ResNet18-SA para diagnóstico de pneumonia: 95.83% acurácia, F1 95.52% no Pneumonia-MNIST. (Scientific Reports, Nature)
- [Are H&E-based computational models transforming molecular pathology diagnostics in cancer? (2025)](https://www.academia.edu/resource/work/145550860) - Revisão sistemática (PRISMA) sobre modelos computacionais baseados em H&E para predição de biomarcadores moleculares em câncer. (Academia Oncology)
- [Deep learning-based image classification for integrating pathology and radiology (2025)](https://www.nature.com/articles/s41598-025-07883-w) - Framework para integrar dados multimodais de patologia e radiologia.
- [Generative AI and Foundation Models in Radiology (2025)](https://pubs.rsna.org/doi/10.1148/radiol.242961) - Revisão sobre modelos de fundação em radiologia.
- [A novel explainable AI framework for medical image classification (2025)](https://www.sciencedirect.com/science/article/pii/S1361841525002129) - Método de XAI para análise de imagens médicas com explicações estatísticas, visuais e baseadas em regras.
- [Deep learning-based image classification for AI-assisted medical imaging (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12171332/) - Framework de DL para classificação de imagens médicas.
- [Adversarial Attacks Against Medical Deep Learning Systems (2018)](https://arxiv.org/pdf/1804.05296.pdf) - Estudo sobre ataques adversariais contra sistemas de DL médico.

### NLP em Saúde

- [DSGym: A Standardized and Holistic Framework for Advancing Data Science Agents (2026)](https://openreview.net/forum?id=5t0nVFY1Ld) - Framework padronizado para avaliar agentes de ciência de dados em benchmarks fragmentados. (James Zou group, OpenReview)
- [AI Agents for Data Science: a Discussion of LAMBDA: A Large Model Based Data Agent (2026)](https://www.polyu.edu.hk/ama/cmfai/discussions/diss3.pdf) - Discussão sobre agentes de IA para ciência de dados com LAMBDA. (Yuksekgonul, Zou)
- [Eubiota: Modular Agentic AI for Autonomous Discovery in the Gut Microbiome (2026)](https://www.biorxiv.org/content/biorxiv/early/2026/02/28/2026.02.27.708412.full.pdf) - IA agentiva modular para descoberta autônoma no microbioma intestinal. (bioRxiv, James Zou group)
- [Rethinking Memory Mechanisms of Foundation Agents in the Second Half: A Survey (2026)](https://arxiv.org/abs/2602.06052) - Survey sobre mecanismos de memória em agentes de fundação com ênfase em avaliação no mundo real. (arXiv, James Zou group)
- ["Sorry, I Didn't Catch That": How Speech Models Miss What Matters Most (2026)](https://arxiv.org/pdf/2602.12249) - Análise de como modelos de reconhecimento de fala falham em capturar informações críticas. (arXiv)
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

- [Resolução CFM 2.454/2026: Regulamentação de IA na Medicina (2026)](https://www.cfm.org.br/resolucoes/2454-2026) - Resolução do Conselho Federal de Medicina que regulamenta o uso de IA na prática médica brasileira. Prazo de 180 dias para adequação e designação de responsável técnico de IA. (CFM Brasil)
- [Position: AI Development Should Prioritize Cognitive Security (2026)](https://openreview.net/forum?id=5pShDP8LSm) - Artigo de posição sobre a necessidade de priorizar segurança cognitiva no desenvolvimento de sistemas de IA generativa. (James Zou group, OpenReview)
- [Racial Income Inequality and COVID-19 Burden in Louisiana: A Spatial Analysis of Public Health Disparities (2026)](https://muse.jhu.edu/pub/1/article/982963/summary) - Análise espacial de disparidades raciais e socioeconômicas na carga de COVID-19 na Louisiana. (Journal of Health Care for the Poor and Underserved)
- [Artificial Intelligence In Health And Health Care: Priorities (2025)](https://www.healthaffairs.org/doi/10.1377/hlthaff.2024.01003) - Prioridades para uso seguro e eficaz de IA na saúde. (Health Affairs)
- [Implementation and Updating of Clinical Prediction Models (2025)](https://www.mcpdigitalhealth.org/article/S2949-7612(25)00035-5/fulltext) - Abordagens de implementação e atualização de modelos clínicos.

---

## Mão na Massa & Código

### Repositórios e Tutoriais Gerais

- [SPR 2026 - BERTimbau v5 Gamma Search (Kaggle Notebook)](https://www.kaggle.com/code/fabianofilho/spr-2026-bertimbau-v5-gamma-search) - Notebook com BERTimbau v5 usando busca gamma para otimização de threshold na competição SPR 2026.
- [SPR 2026 - BERTimbau v5 Threshold Grid Search (Kaggle Notebook)](https://www.kaggle.com/code/fabianofilho/spr-2026-bertimbau-v5-threshold-grid) - Notebook com grid search de threshold para maximizar F1-Score Macro.
- [Validação: BioBERTpt + Focal Loss (Kaggle Notebook)](https://www.kaggle.com/code/fabianofilho/validacao-biobertpt-focal-loss) - Notebook de validação com BioBERTpt e Focal Loss para NLP clínico em português.
- [Validação: BERTimbau Large (Kaggle Notebook)](https://www.kaggle.com/code/fabianofilho/validacao-bertimbau-large) - Notebook de validação com BERTimbau Large para classificação de relatórios de mamografia.
- [Validação: DistilBERT Multilingual (Kaggle Notebook)](https://www.kaggle.com/code/fabianofilho/validacao-distilbert-multilingual) - Notebook de validação com DistilBERT multilingual para NLP médico.
- [open-wearables](https://github.com/the-momentum/open-wearables) - Plataforma self-hosted para unificar dados de wearables (Garmin, Apple Watch, Suunto) em uma API pronta para ML.
- [stenoai](https://github.com/ruzin/stenoai) - IA para notas clínicas estruturadas com LLMs locais (Qwen, DeepSeek, Llama3), com foco em privacidade total de dados do paciente.
- [bio - Open-Source Bio AI Assistant](https://github.com/yorkeccak/bio) - Assistente de IA open-source para pesquisa biomédica com acesso a literatura, ensaios clínicos e bases de dados de efeitos adversos.
- [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) - Toolbox para medição remota de sinais fisiológicos (FC, SpO2) via câmera. NeurIPS 2023. (UbiComp Lab, Univ. de Washington)
- [Triagegeist Kaggle Hackathon](https://www.kaggle.com/competitions/triagegeist) - Competição para construir ferramentas de IA para suporte à triagem em departamentos de emergência. Prêmio: $10.000. (Laitinen-Fredriksson Foundation)
- [SPR 2026 - BERTimbau \+ Focal Loss v3(Kaggle Notebook)](https://www.kaggle.com/code/fabianofilho/spr-2026-bertimbau-focal-loss-v3) - Notebook com BERTimbau fine-tuned com Focal Loss para a competição SPR 2026 de NLP em português.
- [SPR 2026 - BioBERTpt + Focal Loss v2 (Kaggle Notebook)](https://www.kaggle.com/code/fabianofilho/spr-2026-biobertpt-focal-loss-v2) - Notebook com BioBERTpt para classificação de textos médicos em português.
- [SPR 2026 - BERTimbau + LoRA Offline (Kaggle Notebook)](https://www.kaggle.com/code/fabianofilho/spr-2026-bertimbau-lora-offline) - Notebook com BERTimbau + LoRA (Low-Rank Adaptation) para fine-tuning eficiente em modo offline.
- [Webinar: Unlocking Causal Insights with TabPFN (Prior Labs, Mar 2026)](https://app.livestorm.co/prior-labs/unlocking-causal-insights-with-tabpfn) - Webinar com Bernhard Schölkopf e Frank Hutter sobre inferência causal a partir de dados observacionais com TabPFN.
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

- [Prior Labs - TabPFN](https://priorlabs.ai/) - Plataforma de inferência causal e classificação tabular com Prior-data Fitted Networks, com aplicações em dados clínicos.
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

- [Timber](https://github.com/kossisoroyce/timber) - Compilador AOT que converte modelos de ML (XGBoost, LightGBM, scikit-learn) em código C99 nativo, com inferência até 336x mais rápida. Ideal para predição de risco em tempo real.
- [snputils](https://www.biorxiv.org/content/10.64898/2026.02.28.708618v1) - Biblioteca de alta performance para análise de variação genética e estrutura populacional. (bioRxiv 2026)
- [discovery-engine-api](https://pypi.org/project/discovery-engine-api/) - API para encontrar padrões estatísticos validados em dados tabulares, incluindo interações de features e efeitos de subgrupos.
- [skforecast](https://pypi.org/project/skforecast/) - Forecasting de séries temporais com modelos scikit-learn. Versão 0.20.1 com intervalos de predição 10x mais rápidos e AutoARIMA.
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

- [xai-cola](https://arxiv.org/html/2602.21845v1) - Biblioteca para esparsificar explicações contrafatuais, reduzindo mudanças superflúas nas features em até 50%. Útil para explicar negações de autorizações médicas.
- [SymTorch](https://www.marktechpost.com/2026/03/03/meet-symtorch-a-pytorch-library-that-translates-deep-learning-models-into-human-readable-equations/) - Biblioteca PyTorch que usa regressão simbólica para traduzir redes neurais em equações matemáticas legíveis. Melhora a interpretabilidade de modelos complexos.
- [SHAP](https://github.com/shap/shap) - SHapley Additive exPlanations para explicar a saída de qualquer modelo de ML.
- [LIME](https://github.com/marcotcr/lime) - Local Interpretable Model-agnostic Explanations.
- [InterpretML](https://interpret.ml/) - Toolkit de interpretabilidade de ML da Microsoft.
- [Fairlearn](https://fairlearn.org/) - Toolkit para avaliar e melhorar a fairness de modelos de ML.
- [AIF360](https://aif360.mybluemix.net/) - AI Fairness 360 da IBM.
- [MCBoost](https://github.com/mlr-org/mcboost) - Multicalibração para melhorar a calibração de modelos por subgrupos.

### Interoperabilidade e Padrões

- [ccda-to-fhir](https://pypi.org/project/ccda-to-fhir/) - Biblioteca Python para conversão de documentos C-CDA para recursos FHIR R4B (HL7). Essencial para interoperabilidade de dados de saúde.
- [pyomop](https://github.com/dermatologist/pyomop) - Canivete suíço para trabalhar com o OHDSI OMOP Common Data Model v5.4/v6 via SQLAlchemy.
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

- [Triagegeist Hackathon (2026)](https://www.kaggle.com/competitions/triagegeist) - Desafio para construir ferramentas de IA para suporte à triagem em departamentos de emergência. Prêmio: $10.000. Dados recomendados: MIMIC-IV-ED, NHAMCS. (Laitinen-Fredriksson Foundation)
- [SPR 2026 - Mammography Report Classification (2026)](https://www.kaggle.com/competitions/spr-2026-mammography) - Classificação de relatórios de mamografia em português. Métrica: F1-Score Macro. Prêmio: $3.000. (Felipe Kitamura, MD, PhD)
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

## Eventos & Conferências

### CHIL 2026 (Conference on Health, Inference, and Learning)

- [CHIL Doctoral Symposium 2026](https://chil.ahli.cc/submit/doctoral-symposium/) - Programa para doutorandos apresentarem pesquisa e conectarem com mentores sêniors em IA na saúde. Prazo: 13 Mar 2026. (AHLI - Association for Health Learning and Inference)
- [Health AI Builders Unconference 2.0 - CHIL 2026](https://chil.ahli.cc/submit/unconference/) - Evento full-day sobre implantação real de IA na saúde: segurança, compliance, integração e perspectiva clínica. 28 Jun 2026, Seattle.

---

## Contribuindo

Suas contribuições são sempre bem-vindas! Por favor, dê uma olhada nas [diretrizes de contribuição](CONTRIBUTING.md) primeiro.

## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para mais detalhes.


## Atualização Semanal (14 de Março de 2026)

### Papers

- [A clinical environment simulator for dynamic AI evaluation (2026)](https://www.nature.com/articles/s41591-026-04252-6) - Simulação de ambiente hospitalar para avaliação dinâmica de IA em cenários realistas. (Nature)
- [Can AI help predict which heart-failure patients will worsen within a year? (2026)](https://news.mit.edu/2026/can-ai-help-predict-which-heart-failure-patients-will-worsen-0312) - Modelo de deep learning do MIT para prever a trajetória de insuficiência cardíaca com até um ano de antecedência.
- [External Validation of a Commercially Available AI Tool for Nasogastric Tube Position Decision Support in the NHS (2026)](https://t.n.nejm.org/r/?id=hcb1c7be1,8d45159,3ee0cb9&cid=DM2441800_Non_Subscriber&bid=-887325727&p1=U2FsdGVkX1%2FQ43k6f9dTu7CKOmJPKUC4vS3qIgW0n8D7d5phuagMy7OT839g0Dgdij1fRavivnwDBa17BDBMqRVD8vcTqND995QuMffAYsznDrrkYpRjCIrEwIrWjSos08Q%2BNw7EaXDYSEary%2FMAu7AcXB%2B%2Fvz5nhYy6VfKPaiD9%2BehTSDo%2Fu2dOka6WJI1%2BoJVNf9uD1AcPemgvzgy3aA%3D%3D) - Estudo de validação prospectivo e multicêntrico sobre o desempenho de uma ferramenta de IA para detecção de posicionamento de sonda nasogástrica. (NEJM AI)
- [Real World Human-LLM Interactions - Prospective blinded versus unblinded expert physician assessments of LLM responses to complex medical dilemmas (2026)](https://pubmed.ncbi.nlm.nih.gov/41818246/) - Avaliação prospectiva de respostas de LLMs a dilemas médicos complexos por médicos especialistas. (PubMed)
- [Inteligência artificial identifica dor em bebês e pode auxiliar decisões médicas em UTI neonatal (2026)](https://www.google.com/url?rct=j&sa=t&url=https://revistagalileu.globo.com/saude/noticia/2026/03/inteligencia-artificial-identifica-dor-em-bebes-e-pode-auxiliar-decisoes-medicas-em-uti-neonatal.ghtml&ct=ga&cd=CAEYACoUMTY5NzAxNTA2ODU1MzQ1Nzk2NzgyHTQ4YzZmZDE0MWJhYWMwN2Q6Y29tLmJyOnB0OlVT&usg=AOvVaw3-KIP7ZaWN8npj6MNWkI8q) - Ferramenta desenvolvida por pesquisadores da FEI e da Unifesp que utiliza IA para interpretar expressões faciais de recém-nascidos. (Galileu - Globo)

### Hands-On

- [ISU-OSF Data Science Competition (2026)](https://news.illinoisstate.edu/2026/03/isu-osf-data-science-competition-returns-to-tackle-missed-medical-appointments/) - Competição de ciência de dados para aplicar técnicas de machine learning e modelagem preditiva para identificar potenciais faltas em consultas médicas.
- [Longitudinal Ageing Study (2026)](https://www.kaggle.com/datasets/boidoingthings/longitudinal-ageing-study) - Dataset de um estudo longitudinal representativo sobre envelhecimento e saúde. (Kaggle)

### Bibliotecas

- [FreedomIntelligence/OpenClaw-Medical-Skills (2026)](https://github.com/FreedomIntelligence/OpenClaw-Medical-Skills) - A maior biblioteca de skills médicos de IA de código aberto para OpenClaw, com 869 skills curados.
- [LabClaw – Operating Layer for LabOS (2026)](https://github.com/wu-yc/LabClaw) - Pacote com 211 arquivos SKILL.md prontos para produção para workflows de IA biomédica. (Stanford-Princeton)
- [Tuva Health (2026)](https://www.ycombinator.com/companies/industry/digital-health) - Software de código aberto que limpa e transforma dados de saúde desorganizados. (YC-funded)

### Competições

- [ISU-OSF Data Science Competition (2026)](https://news.illinoisstate.edu/2026/03/isu-osf-data-science-competition-returns-to-tackle-missed-medical-appointments/) - Desafio para identificar potenciais faltas em consultas médicas usando ML.

### Ética

- [TEMPO: Experimenting with AI Sandboxes in the United States (2026)](https://t.n.nejm.org/r/?id=hcb1c7be1,8d45159,3ee0cba&cid=DM2441800_Non_Subscriber&bid=-887325727&p1=U2FsdGVkX1%2FQ43k6f9dTu7CKOmJPKUC4vS3qIgW0n8D7d5phuagMy7OT839g0Dgdij1fRavivnwDBa17BDBMqRVD8vcTqND995QuMffAYsznDrrkYpRjCIrEwIrWjSos08Q%2BNw7EaXDYSEary%2FMAu7AcXB%2B%2Fvz5nhYy6VfKPaiD9%2BehTSDo%2Fu2dOka6WJI1%2BoJVNf9uD1AcPemgvzgy3aA%3D%3D) - Análise sobre o novo piloto TEMPO da FDA, que introduz sandboxes regulatórios para acelerar o uso de IA em saúde. (NEJM AI)
- [Nova Resolução do CFM define regras para uso de inteligência artificial na saúde (2026)](https://www.google.com/url?rct=j&sa=t&url=https://diariodocomercio.com.br/opiniao/artigo/nova-resolucao-do-cfm-define-regras-para-uso-de-inteligencia-artificial-na-saude/&ct=ga&cd=CAEYACoTMzQ3ODUwMDI3MTE5NDk0NjI1MzIdNDhjNmZkMTQxYmFhYzA3ZDpjb20uYnI6cHQ6VVM&usg=AOvVaw3B6YDXX8ABKYXJQil96mwX) - Nova resolução do Conselho Federal de Medicina que orienta o uso da IA na medicina brasileira. (Diário do Comércio)
- [Huntsman Mental Health Institute contributes to new framework for ethical AI (2026)](https://healthcare.utah.edu/press-releases/2026/03/huntsman-mental-health-institute-contributes-new-framework-ensuring-ethical) - Contribuição para um novo framework para garantir o desenvolvimento ético da IA para a saúde. (University of Utah)


## Atualização Semanal (21 de Março de 2026)

### Papers

- [Inteligência artificial na saúde pública brasileira: potencialidades, desafios e implicações éticas para o Sistema Único de Saúde (2026)](https://www.scielosp.org/article/rbepid/2026.v29/e260006/pt/) - Análise crítica sobre a incorporação da IA no SUS, à luz de seus princípios. (Revista Brasileira de Epidemiologia)
- [CRIMSON: A Clinically-Grounded LLM-Based Metric for Generative Radiology Report Evaluation (2026)](https://arxiv.org/abs/2603.06183) - Framework de avaliação para laudos de radiologia gerados por IA, focado em correção diagnóstica, relevância contextual e segurança do paciente. (arXiv)
- [Evaluation of SOFA-2 Score Performance Across Demographic Subgroups: An External Validation Study Using MIMIC-IV (2026)](https://www.medrxiv.org/content/10.64898/2026.03.10.26348061) - Validação externa do score SOFA-2 em subgrupos demográficos usando dados do MIMIC-IV. (medRxiv)
- [Health care access by atherosclerotic cardiovascular disease status: a JACC data report on trends in the United States, 2019-2024 (2026)](https://www.jacc.org/doi/full/10.1016/j.jacc.2025.12.045) - Análise de tendências no acesso à saúde para pacientes com doença cardiovascular aterosclerótica nos EUA. (JACC)
- [The Prime Diet Quality Score (PDQS), chronic disease and cause-specific mortality in UK Biobank: a prospective study (2026)](https://link.springer.com/article/10.1007/s00394-025-03877-6) - Avaliação do PDQS como métrica de qualidade da dieta em relação à mortalidade e doenças crônicas no UK Biobank. (European Journal of Nutrition)
- [Innovating global regulatory frameworks for generative AI in healthcare (2026)](https://www.nature.com/articles/s41746-026-02552-2) - Discussão sobre as oportunidades e desafios da integração de GenAI e LLMs na saúde. (Nature)
- [Machine learning analysis of CT scans can diagnose and predict disease risk (2026)](https://www.nih.gov/news-events/nih-research-matters/machine-learning-analysis-ct-scans) - Modelo de ML que interpreta tomografias abdominais para diagnóstico e predição de risco de doenças crônicas. (NIH Research Matters)
- [Multimodal Deep Learning for Early Prediction of Patient Outcomes (2026)](https://arxiv.org/abs/2603.14719) - Abordagem de deep learning multimodal que combina dados estruturados e não estruturados para prever desfechos de pacientes. (arXiv)

### Ferramentas & Bibliotecas

- [Perplexity Health (2026)](https://www.perplexity.ai/hub/blog/introducing-perplexity-health) - Suite de conectores para dados de saúde pessoais que permite o uso do Perplexity Computer para responder a perguntas de saúde com base em seus próprios registros médicos.
- [NVIDIA Healthcare AI Open Models (2026)](http://nvidianews.nvidia.com/news/nvidia-expands-open-model-families-to-power-the-next-wave-of-agentic-physical-and-healthcare-ai) - Expansão do ecossistema de modelos de IA abertos da NVIDIA para apoiar casos de uso em saúde e ciências da vida, incluindo descoberta de medicamentos e pesquisa biomédica.
- [RightNow-AI/openfang (2026)](https://github.com/RightNow-AI/openfang) - Sistema operacional de agente (Agent Operating System) com lançamento da versão v0.3.30 focada em segurança.

### Competições & Eventos

- [Machine Learning for Healthcare (MLHC) 2026](https://www.mlforhc.org/) - Chamada de artigos para a conferência MLHC 2026, que ocorrerá de 12 a 14 de agosto de 2026 em Baltimore, EUA. Prazo de submissão: 17 de abril.
- [Employee Mental Health & Burnout Dataset (Kaggle)](https://www.kaggle.com/datasets/suhanigupta04/employee-mental-health-and-burnout-dataset) - Novo dataset no Kaggle que cobre o pipeline completo de saúde mental no trabalho, incluindo estresse, estilo de vida, sono e acesso à terapia.
