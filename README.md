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

## Novidades -- Semana de 23-30 mai 2026

> Adicionados automaticamente pelo cron #09 (awesome-medical-ai).

### Papers (8 novos)

**Oncologia & Patologia Computacional:**
- [Artera Multimodal AI (MMAI) — ASCO 2026 Abstracts](https://www.businesswire.com/news/home/20260521757142/en/Artera-Presents-Multiple-ASCO-2026-Abstracts-on-Multimodal-AI-MMAI-Risk-Stratification-and-Treatment-Decisions-in-Cancer) -- IA multimodal para estratificacao de risco em cancer de mama (HR+ com 1-3 linfonodos positivos) e prostata; apresentacao oral no ASCO 2026 comparando MMAI vs. teste de 21 genes (Recurrence Score). (Chicago, Mai 2026)
- [Translating Multimodal Foundation Models into Oncology](https://pmc.ncbi.nlm.nih.gov/articles/PMC12989828/) -- Revisao de FMs multimodais (MUSK, M3FM, MICE) integrando patologia, radiologia e dados genomicos; MICE: melhoria de C-index de 3,8-11,2% em 30 tipos de cancer. (PMC, 2026)
- [What's New in Digital and Computational Pathology 2026](https://jpatholtm.org/journal/view.php?number=17219) -- Avancos em whole-slide imaging, modelos de IA (TITAN: 335k slides, 20 orgaos, self-supervised), integracao multimodal e infraestrutura clinica para patologia digital. (J Pathol TM, 2026)
- [HISTAI: Open-Source Large-Scale Whole Slide Image Dataset](https://arxiv.org/abs/2505.12120) -- 112.000 WSIs de 46.000 casos com metadados clinicos completos; um dos maiores datasets publicos de histopatologia. (arXiv 2505.12120, Mai 2026)

**NLP & IA Generativa:**
- [Towards a Medical AI Scientist](https://arxiv.org/abs/2603.28589) -- Primeiro framework autonomo de pesquisa clinica: geracao de hipoteses, experimentos e manuscritos com qualidade proxima a MICCAI em revisao cega por especialistas. (arXiv 2603.28589, 2026)
- [Barriers and Opportunities of Scaling Ambient AI Scribes](https://www.nature.com/articles/s41746-026-02554-0) -- Analise de desafios de escala de assistentes de documentacao clinica com IA em cenarios ambulatorial, hospitalar e de baixa renda. (npj Digital Medicine, 2026)
- [Generative AI in Medicine -- Evaluating Progress and Challenges](https://www.nejm.org/doi/abs/10.1056/NEJMsb2503956) -- Estado atual da IA generativa em medicina: geracao de evidencias em tempo real, estudos dinamicos e suporte personalizado. (NEJM, 2026)
- [Bedside to Bench: AI and the New Science of Medicine](https://www.nejm.org/doi/full/10.1056/NEJMp2510203) -- Como IA reorganiza o ciclo translacional da pesquisa medica, da beira do leito ao laboratorio, com implicacoes para sistemas autonomos de conhecimento clinico. (NEJM, 2026)

### Modelos & Ferramentas (2 novos)

- **Chai-2** (Chai Discovery, 2026) -- Primeiro sistema de IA para design de anticorpos funcionais de novo sem triagem laboratorial; taxa de sucesso 16-20% (100x melhor que metodos anteriores); parceria com Eli Lilly, US$130M Series B
- **MedASR** (Google Research, 2026) -- Primeiro modelo open-weight de reconhecimento de fala medica otimizado para ditado clinico, lancado junto ao MedGemma 1.5

### Dataset (1 novo)

- **HISTAI** (2026) -- 112.000 WSIs de 46.000 casos com metadados clinicos; open-source via [GitHub](https://github.com/HistAI/HISTAI)

### Eventos (4 novos)

- **ASCO Annual Meeting 2026** -- 29/05-02/06/2026, Chicago, IL -- maior conferencia de oncologia clinica com forte presenca de IA em abstracts
- **WHA79 -- OMS, Genebra** -- 22-28/05/2026 -- resolucao aprovada integrando teleradiologia e IA em estrategias nacionais de saude digital
- **FDA RFI: AI Early-Phase Clinical Trials** -- Prazo 28/05/2026 -- comentarios sobre piloto de IA para otimizacao de ensaios clinicos de fase inicial
- **3rd International Webinar on AI for Health 2026** -- 30/05/2026, virtual -- tema: IA agentica e generativa na assistencia ao paciente em tempo real

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
