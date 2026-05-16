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
  Uma lista curada de recursos incriveis para <strong>Inteligencia Artificial na Medicina</strong>: artigos cientificos, datasets, ferramentas, modelos pre-treinados, codigo pratico e competicoes.
</p>

---

## Novidades -- Semana de 09-16 mai 2026

> Adicionados automaticamente pelo cron #08 (awesome-medical-ai).

### Papers & Benchmarks (8 novos)

**Benchmarks de LLMs em Saúde:**
- **HealthBench** (OpenAI, Mai 2026) -- Benchmark com 5.000 conversas médicas avaliadas por 262 médicos de 60 países; o3 atinge 60% (arXiv 2505.08775)
- **HealthBench Professional** (OpenAI, 2026) -- Extensão do HealthBench para fluxos de trabalho clínicos (consulta, documentação, pesquisa médica)
- **MedAgentBench** (Stanford/NEJM AI, 2026) -- Ambiente virtual FHIR-compatível para benchmark de agentes LLM em EHR; 300 tarefas, 100 pacientes, 700k+ elementos; Claude 3.5 Sonnet lidera com 69,67%

**Drug Discovery:**
- **IsoDDE $2.1B Series B** (Isomorphic Labs, Mai 2026) -- Motor de drug design com +2× a precisão do AlphaFold 3 em benchmarks de generalização; Isomorphic Labs capta US$ 2,1B (12 mai 2026) para levar primeiro candidato a fármaco à Fase I até 2026

**NLP Clínico:**
- **VeriFact** (NEJM AI, 2026) -- Sistema RAG + LLM-as-Judge para verificação automática de fatos em documentos de cuidado ao paciente; 93,2% de concordância com clínicos

**Ética e Fairness:**
- **AI-Driven Healthcare: Ensuring Fairness and Mitigating Bias** (PLOS Digital Health, 2026) -- Revisão abrangente sobre estratégias para garantir equidade e mitigar viés em IA na saúde

**XAI e Imagem Médica:**
- **Clinician-Centric Explainable AI for Medical Imaging** (PMC 2026) -- Revisão sistemática de frameworks XAI em diagnósticos por imagem com foco em adoção clínica; destaca detecção de pneumonia como caso de uso

### Modelos & Ferramentas (3 novos)

**LLMs para Ciências da Vida:**
- **GPT-Rosalind** (OpenAI, Abr 2026) -- Modelo frontier especializado em drug discovery, genômica, raciocínio sobre proteínas e pesquisa científica; nomeado em homenagem a Rosalind Franklin; parcerias com Amgen, Moderna, Allen Institute

**Agentes Clínicos:**
- **DR. INFO** (2026) -- Assistente clínico RAG-based agentivo que supera LLMs frontier no HealthBench
- **TissueLab** (2026) -- Sistema de IA agentivo para análise de imagens médicas; abstrai problemas de imagem em operações fundamentais (segmentação, classificação) sem necessidade de código

### Datasets (1 novo)

- **HealthBench Dataset** (OpenAI, Mai 2026) -- 5.000 conversas multi-turno com rubricas específicas por médico (48.562 critérios únicos); aberto para a comunidade de pesquisa

### Eventos (em destaque esta semana)

- **Bio-IT World Expo 2026** -- AI for Drug Discovery & Development (18-20 **mai** 2026, Boston & Virtual) — acontece **esta semana**
- **IJCAI-ECAI 2026: AI and Health Track** -- Call for papers encerrado; conferência em Yokohama, jul 2026

---

## Novidades -- Semana de 02-09 mai 2026

> Adicionados automaticamente pelo cron #07 (awesome-medical-ai).

### Papers (15 novos)

**ECG & Sinais:**
- MEETI: Multimodal ECG Dataset (MIMIC-IV-ECG) com sinais, imagens e interpretações textuais (Scientific Data, Mai 2026)
- Multimodal Transformer for Cardiovascular Comorbidity Detection via ECG (JMIR Formative Research, 2026)

**NLP & Processamento de Linguagem:**
- Scalable Micro-Credentials for AI Literacy in Healthcare com infraestrutura baseada em LLMs (medRxiv, Mar 2026)
- Benchmarking LLM-based Information Extraction Tools for Medical Documents (medRxiv, Jan 2026)
- Authority Signals in AI Cited Health Sources: A Framework (medRxiv, Jan 2026)

**Imagem Médica:**
- Measuring the Impact of AI on Report-Drafting Efficiency in Chest Computed Tomography Interpretation (JMIR, 2026)
- The Diagnostic Value of Image-Based Machine Learning for Osteoporosis: Systematic Review & Meta-Analysis (JMIR, 2026)
- From Intelligent Models to Clinical Tools: AI in Medical Imaging transição pesquisa-clínica (Scientific Reports, 2026)
- Explainable AI for Medical Imaging Systems Review com foco em transparência (Cluster Computing, 2025)

**Drug Discovery:**
- Artificial Intelligence in Drug Discovery and Development: Pharmaceutical Innovation (Drug Development Research, 2026)
- AI in Drug Discovery Predictions 2026: crescimento de 5-7B para 8-10B USD (Drug Target Review, 2026)
- MSU Study: Faster Discovery of Therapeutic Drugs Through AI (MSUToday, Mar 2026)

### Modelos & Ferramentas (10 novos)

**Clinical & NLP:**
- **ClinicBot** -- Chatbot clínico com RAG priorizado por evidências (KIT, Helmholtz Ulm, TU Munich, 2026)
- **BioMedArena** -- Toolkit open-source para agentes de pesquisa biomédica (arXiv, 2026)
- **LAOS System** -- Clinical documentation com voice processing e LLMs (npj Digital Medicine, 2026)

**Medical Imaging & Vision:**
- **MedHorizon** -- Long-context medical video understanding para procedimentos cirúrgicos (arXiv, 2026)
- **Bridging Visual Saliency and LLMs** -- Framework para XAI em imagem médica (arXiv, 2026)

**LLM Benchmarks & Modelos:**
- **MedGemma 1.5** -- Versão aprimorada do Google com suporte para imagem médica e MedASR (2026)
- **Me-LLaMA** -- LLaMA médico com 129B tokens e 214k instruções (2026)
- **Gemini-2 & GPT-4o Clinical Benchmarks** -- Top performers em avaliações clínicas 2026 (Nature, 2026)
- **BioELMo, BlueBERT, BioMegatron** -- Modelos BERT especializados para biomedicina

**Visão & Multimodal:**
- **Merlin** -- Modelo de fundação visão-linguagem 3D para TC abdominal, 6M+ imagens (Stanford MIMI, 2026)

**ECG & Sinais:**
- **CaMPNet** -- Cross-attention fusion de sinais 12-derivações, métricas e demografia (MIMIC-IV-ECG, 2026)
- **Multimodal Transformer for ECG** -- Integração de sinais brutos, imagens e metadados (Scientific Data, 2026)

### Datasets (6 novos)

- **MEETI (MIMIC-IV-ECG)** -- Dataset multimodal sincronizando ECG, imagens de alta resolução e interpretações (Scientific Data, 2026)
- **EHRSHOT, INSPECT, MedAlign** -- 3 benchmarks EHR de-identificados com 25.991 pacientes, 441.680 visitas (Stanford HAI, 2026)
- **CT-EBM-SP Corpus** -- Corpus anotado de ensaios clínicos em espanhol para extração de relações (Scientific Data, 2026)
- **Swedish Medical LLM Benchmark (SMLB)** -- 4 datasets para avaliação de LLMs médicos (2026)
- **EchoNext** -- Dataset curado de ECGs com diagnósticos estruturais confirmados (PhysioNet, 2026)
- **Top 50+ Open-Source Healthcare Datasets** -- Curadoria atualizada de datasets públicos (Medium, 2026)

### Competições

- **AI-Med Future Competition** (2026)
- **MedGemma Impact Challenge winners** -- Aplicações em cuidado crítico, saúde mental, patologia, pneumonia (Google, Fev 2026)

### Eventos

- **MIDL 2026** -- Conferência internacional deep learning em imagem biomédica (Taipei, Taiwan)
- **BioNLP 2026** -- Workshop sobre NLP biomédico, transparência e DEIA
- **Keystone Symposia** -- Computational Advances in Drug Discovery
- **Bio-IT World Expo** -- AI for Drug Discovery & Development (18-20 Mai 2026, Boston & Virtual)

---

---

## Indice

- [Papers](papers/) -- artigos cientificos por area
  - [Revisoes e Surveys](papers/reviews-surveys.md)
  - [Modelos Preditivos e EHR](papers/predictive-models-ehr.md)
  - [Imagem Medica](papers/medical-imaging.md)
  - [NLP em Saude](papers/nlp-health.md)
  - [ECG e Sinais](papers/ecg-signals.md)
  - [Drug Discovery](papers/drug-discovery.md)
  - [IA Explicavel (XAI)](papers/xai-health.md)
  - [Etica e Fairness](papers/ethics-fairness.md)
- [Ferramentas & Bibliotecas](tools/) -- softwares, libs, frameworks
  - [Imagem Medica](tools/imaging.md)
  - [Sobrevida e EHR](tools/survival-ehr.md)
  - [NLP](tools/nlp.md)
  - [Explicabilidade e Fairness](tools/explainability-fairness.md)
  - [Interoperabilidade](tools/interoperability.md)
- [Datasets](datasets/) -- conjuntos de dados por modalidade
  - [EHR](datasets/ehr.md)
  - [Imagem Medica](datasets/medical-imaging.md)
  - [Sinais](datasets/signals.md)
  - [Texto e NLP](datasets/nlp-text.md)
  - [Genomica](datasets/genomics.md)
  - [Farmacos](datasets/pharma.md)
- [Modelos Pre-treinados](models/) -- LLMs, visao, multimodal, sinais
  - [LLMs](models/llms.md)
  - [Visao](models/vision.md)
  - [Multimodal](models/multimodal.md)
  - [Sinais (ECG/EEG)](models/signals.md)
- [Mao na Massa & Codigo](hands-on/) -- tutoriais e notebooks
  - [Geral](hands-on/general.md)
  - [ECG e Sinais](hands-on/ecg-signals.md)
  - [Imagem Medica](hands-on/medical-imaging.md)
  - [NLP Clinico](hands-on/nlp-clinical.md)
  - [EHR e Tabulares](hands-on/ehr-tabular.md)
- [Competicoes](competitions/) -- ativas e historicas
- [Eventos & Conferencias](events/)

---

## Contribuindo

Suas contribuicoes sao sempre bem-vindas! Por favor, veja as [diretrizes de contribuicao](CONTRIBUTING.md) primeiro.

## Licenca

Este projeto esta licenciado sob a Licenca MIT -- veja o arquivo [LICENSE](LICENSE) para mais detalhes.
