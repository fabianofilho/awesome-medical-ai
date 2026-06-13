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

## Novidades -- Semana de 07-13 jun 2026

> Adicionados automaticamente pelo cron #09 (awesome-medical-ai).

### Papers (4 novos)

**Benchmarks & IA Agentiva Clínica:**
- [MedCTA: A Benchmark for Clinical Tool Agents](https://arxiv.org/abs/2606.11702) -- Ashraf, Jeong, Thoker, Ghanem (KAUST + MIT) — 107 tarefas clínicas reais, 5 ferramentas implantadas, 18 modelos; melhor acurácia: 31,54%; nenhum modelo atinge sucesso de trajetória estrita (arXiv 2606.11702, Jun 2026)
- [AgentClinic: a multimodal benchmark for tool-using clinical AI agents](https://www.nature.com/articles/s41746-026-02674-7) -- benchmark multimodal em 9 especialidades e 7 idiomas; Claude 3.5 supera demais backbones; acurácia diagnóstica pode cair a <10% vs. QA estático (npj Digital Medicine, 2026)

**NLP & Triagem Especializada:**
- [Advancing medical AI through benchmarking and competition for specialty triage](https://www.nature.com/articles/s41746-026-02433-8) -- MedGPT-Guide com estratégia "10 Relevant + 10 Random + Ensemble" para triagem de especialidade; novo benchmark MedTriage (npj Digital Medicine, Jun 2026)
- [Benchmarking LLM-based agent systems for clinical decision tasks](https://pmc.ncbi.nlm.nih.gov/articles/PMC13031301/) -- avaliação de sistemas agentes baseados em LLMs em tarefas clínicas complexas de decisão (npj Digital Medicine / PMC, 2026)

### Datasets & Ferramentas (1 novo)

- **NVIDIA Open-H + Cosmos-H** -- primeiro dataset aberto de robótica cirúrgica: 776h de vídeo cirúrgico, 35 organizações, 9 sistemas robóticos; acompanha Isaac GR00T-H (primeiro VLA model aberto para robótica hospitalar) e simulador Cosmos-H para dados sintéticos [Hugging Face](https://huggingface.co/blog/nvidia/physical-ai-for-healthcare-robotics) (Jun 2026)

### Eventos (2 novos)

- **DHAI 2026 (Digital Health & AI Innovation Summit)** — 8-9 Jun 2026, Boston — 2000+ participantes, 150+ palestrantes, foco em IA clínica e decisão hospitalar
- **Medica AI – The Conference 2026** — 16 Jul 2026, Lisboa (Champalimaud Foundation) — IA como transformação sistêmica na saúde, não apenas tecnológica

---

## Novidades -- Semana de 19-25 mai 2026

> Adicionados automaticamente pelo cron #08 (awesome-medical-ai).

### Papers (10 novos)

**NLP & LLMs Clínicos:**
- [Large Language Models for AI-Assisted Radiotherapy Scheduling: A Feasibility Study Under Realistic Operational Constraints](https://arxiv.org/abs/2605.12896) -- avalia LLMs para geração de agendas de radioterapia com restrições clínicas reais (arXiv 2605.12896, Mai 2026)
- [Dissecting clinical reasoning failures in frontier artificial intelligence using 10,000 synthetic cases](https://www.medrxiv.org/content/10.64898/2026.04.22.26351488v1) -- análise sistemática de falhas de raciocínio clínico em modelos de fronteira com 10k casos sintéticos (medRxiv, Abr 2026)
- [Forecasting Scientific Progress with Artificial Intelligence](https://arxiv.org/abs/2605.22681) -- Wu, Lu, Chen, Clark et al. — framework de avaliação temporal para previsão de progresso científico com IA (arXiv 2605.22681, Mai 2026)

**Imagem Médica:**
- [Efficient Deep Learning for Medical Imaging: Bridging the Gap Between High-Performance AI and Clinical Deployment](https://arxiv.org/abs/2602.00910) -- revisão de arquiteturas leves (CNNs, Transformers, Linear Complexity) e estratégias de compressão para deploy clínico (arXiv 2602.00910, 2026)

**Modelos Preditivos & EHR:**
- [Machine Intelligence-Driven Forecasting for ED Triage and Dynamic Hospital Patient Routing](https://www.medrxiv.org/content/10.64898/2026.02.18.26346566v2.full) -- gradient boosting supera scoring clínico tradicional em triagem de pronto-socorro (medRxiv, 2026)
- [Generalizing intensive care AI across time scales in resource-limited settings](https://www.medrxiv.org/content/10.64898/2026.04.23.26351588v1.full) -- limitações de generalização de modelos de UTI em ambientes com recursos limitados (medRxiv, Abr 2026)

**Ética, Fairness & Causal AI:**
- [Causal Machine Learning Is Not a Panacea: A Roadmap for Observational Causal Inference in Health](https://arxiv.org/abs/2605.20782) -- Tjandra, Chang, Parbhoo, Ranganath, Celi et al. — roadmap crítico para inferência causal observacional em saúde (arXiv 2605.20782, Mai 2026)
- [Qualitative Framework for Evaluating Clinical Data Science Systems: Beyond Technical Validation](https://pmc.ncbi.nlm.nih.gov/articles/PMC13193742/) -- Heslin — framework qualitativo para avaliação de sistemas além de métricas técnicas (Healthcare Informatics Research, 2026)

**Saúde Digital & Policy:**
- [A global call to action on the digital determinants of health](https://www.nature.com/articles/s41746-026-02765-5) -- Shaw, Atuire, Celi, Davis et al. — perspectiva sobre determinantes digitais da saúde e impacto das tecnologias emergentes (npj Digital Medicine, 2026)
- [Artificial Intelligence in Healthcare: 2025 Year in Review](https://www.medrxiv.org/content/10.64898/2026.02.23.26346888v1) -- ML clássico em declínio pela primeira vez; foundation models dominando o campo (medRxiv, 2026)

### Ética & Fairness

- **Causal ML roadmap** -- Celi et al. alerta: ML causal não é panaceia para inferência em dados observacionais de saúde
- **Framework de avaliação qualitativa** -- Heslin propõe avaliação de sistemas de ciência de dados clínicos além das métricas técnicas convencionais

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
