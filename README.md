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

> Adicionados automaticamente pelo cron #09 (awesome-medical-ai).

### Papers (7 novos)

**NLP & Colaboração Humano-IA:**
- [Human–large language model collaboration in clinical medicine: a systematic review and meta-analysis](https://www.nature.com/articles/s41746-026-02382-2) -- 10 estudos; melhoria de 4,88 pp em scores compostos de diagnóstico/manejo com colaboração humano-LLM (npj Digital Medicine, Jun 2026)
- [Independent and collaborative performance of LLMs and healthcare professionals in diagnosis and triage](https://www.nature.com/articles/s41746-026-02409-8) -- 50 estudos, 25 LLMs; acurácia relativa LLM vs. profissionais evolui de 0,89 (top-1) a 1,17 (top-10) (npj Digital Medicine, Jun 2026)
- [From tool to teammate: RCT of clinician-AI collaborative workflows for diagnosis](https://www.nature.com/articles/s41746-026-02545-1) -- RCT com 70 clínicos testando IA como primeira vs. segunda opinião em raciocínio diagnóstico (npj Digital Medicine, Jun 2026)
- [AutoMedBench: Towards Medical AutoResearch with Agentic AI Models](https://arxiv.org/abs/2606.01961) -- benchmark de pesquisa médica autônoma em 5 tracks (segmentação, VQA, geração de laudo, detecção de lesão); média 33 turnos de agente (arXiv 2606.01961, Jun 2026)

**Imagem Médica:**
- [A Vision-Language Framework for Comparative Reasoning in Radiology (MedReCo)](https://arxiv.org/abs/2606.06407) -- MedReCo-DB: 690k imagens, 160k pacientes, 8 instituições; melhoria de 14,5–46,5 pp em raio-X e 13–27,9 pp em TC para seguimento longitudinal (arXiv 2606.06407, Jun 2026)

**Ética & Educação Médica em IA:**
- [What the AI era doctor should know: a scoping review of AI competencies for medical education](https://www.nature.com/articles/s41746-026-02761-9) -- revisão PRISMA-ScR de competências de IA para graduação médica; Hunt, Sprehe, de Lomba et al. (npj Digital Medicine, Jun 2026)
- [Protecting clinical value judgment in the age of AI](https://www.nature.com/articles/s41746-026-02561-1) -- sistemas de IA adaptativa operacionalizam tradeoffs de valores ocultos em escala e criam pressões de responsabilidade; Patil, Myers, Dai (npj Digital Medicine, Jun 2026)

### Eventos (2 realizados)

- **Health AI Summit 2026** (2ª ed.) — 1-2 Jun 2026, San Francisco — automação por IA, GenAI para workflow e aplicações hospitalares reais
- **Mayo Clinic AI Research Summit** — 4-5 Jun 2026, Rochester, MN — geração de evidências do mundo real em IA na saúde

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
