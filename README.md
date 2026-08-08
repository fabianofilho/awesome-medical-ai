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

## Novidades -- Semana de 01 ago - 08 ago 2026

> Adicionados automaticamente pela curadoria semanal (awesome-medical-ai).

### Papers (6 novos)

**Patologia e Imagem Medica:**
- [PRISM2: End-to-end multimodal pathology foundation model with clinical dialogue](https://www.nature.com/articles/s41591-026-04521-4) -- Tempus; treinado em 2,3 milhoes de whole-slide images e 14 milhoes de pares de QA clinicos; atinge acuracia clinica para deteccao de cancer de prostata e mama sem ajuste de tarefa. (Nature Medicine, 4 Ago 2026)
- [OncoTriad-QA: A Patient-Level Radiology-Pathology-Genomics Benchmark for Pan-Cancer Reasoning](https://arxiv.org/abs/2608.02615) -- benchmark com 86,1 mil questoes em 9.281 casos TCGA de 32 tipos de cancer alinhando radiologia, histopatologia e dados genomicos; testa modelos multimodais em raciocinio oncologico integrado. (arXiv 2608.02615, Ago 2026)

**NLP e Raciocinio Clinico:**
- [MedUPS: Towards Diagnostic Assistance in Uncommon Medical Cases with Large Language Models](https://arxiv.org/abs/2608.01012) -- dataset MedUPSQA com 21.874 pontos de decisao clinica de 5.535 casos raros; treina modelos com GRPO em decisoes intermediarias ao longo da trajetoria do paciente. (arXiv 2608.01012, 2 Ago 2026)
- [From Siloed Algorithms to Compliance-First Agentic Platforms: A Multi-Layered Architecture for Hospital AI Systems](https://arxiv.org/abs/2608.06112) -- analisa por que 70-80% dos pilotos de IA hospitalar falham ao escalar; propoe arquitetura multi-camada orientada a governanca para sistemas agentivos em hospitais. (arXiv 2608.06112, 6 Ago 2026)

**IA Explicavel e Etica:**
- [Divergent impacts of explainable AI for dermatological diagnosis on clinicians versus lay people](https://www.nature.com/articles/s41591-026-04553-w) -- experimento com 623 leigos e 153 medicos; XAI melhora acuracia de nao-especialistas mas prejudica clinicos de atencao primaria, que se saem melhor sem explicacoes do modelo. (Nature Medicine, 4 Ago 2026)
- [A clinically validated framework for auditing AI chatbot behavior in mental health interactions](https://www.nature.com/articles/s41591-026-04577-2) -- framework SIM-VAIL avalia Claude, ChatGPT, Gemini, Grok e Llama em 810 conversas; modelos frequentemente amplificam vulnerabilidades psicologicas de usuarios em contextos de saude mental. (Nature Medicine, 7 Ago 2026)

### Ferramentas e Repositorios

Sem novidades esta semana.

### Datasets

Sem novidades esta semana.

### Eventos

Sem novidades esta semana.

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
