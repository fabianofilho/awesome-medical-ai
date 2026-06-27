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

## Novidades -- Semana de 20-27 jun 2026

> Adicionados automaticamente pela curadoria semanal (awesome-medical-ai).

### Papers (5 novos)

**NLP & LLMs Clínicos:**
- [LLMs de Propósito Geral Superam Ferramentas Clínicas Aprovadas pela FDA](https://www.nature.com/articles/s41591-026-04329-2) -- GPT-5.2, Gemini 3.1 Pro e Claude Opus 4.6 superam OpenEvidence e UpToDate Expert AI em todos os benchmarks; médicos praticantes como avaliadores (Nature Medicine, 23 Jun 2026)
- [Advancing Conversational Diagnostic AI with Multimodal Reasoning](https://www.nature.com/articles/s41591-026-04371-0) -- IA diagnóstica conversacional com capacidades multimodais avançadas para atenção primária (Nature Medicine, Jun 2026)
- [Automation Bias in LLM-Assisted Diagnostic Reasoning Among AI-Trained Physicians](https://www.medrxiv.org/content/10.1101/2025.08.23.25334280v2) -- médicos treinados com IA são vulneráveis a viés de automação mesmo quando usam LLMs voluntariamente (medRxiv, Jun 2026)

**Imagem Médica & Cirurgia:**
- [Decentralized Privacy-Preserving Pipeline for Laparoscopic Appendectomy Videos](https://ai.nejm.org/) -- Saldanha et al. — Swarm Learning com supervisão fraca para prever desfechos em apendicectomias, sem centralizar dados sensíveis (NEJM AI, 23 Jun 2026)

**Modelos Preditivos & EHR:**
- [EHR-Based Model Identifies Type 2 Diabetes Risk up to 10 Years Earlier](https://www.hcinnovationgroup.com/analytics-ai/artifical-intelligence-machine-learning/news/55382584/ehr-based-prediction-model-identifies-diabetes-risk) -- Kaiser Permanente — modelo preditivo validado apresentado no ADA 2026 em Nova Orleans (Jun 2026)

### Modelos & Ferramentas (3 novos)

- **[MedGemma 27B Multimodal](https://research.google/blog/next-generation-medical-image-interpretation-with-medgemma-15-and-medical-speech-to-text-with-medasr/)** (Google, Jun 16, 2026) -- complementa linha MedGemma com suporte a EHRs longitudinais e interpretação multimodal complexa
- **[Google DeepMind AI Co-Clinician](https://deepmind.google/blog/ai-co-clinician/)** (DeepMind, Jun 2026) -- preferido por médicos em 67% dos casos vs. ferramentas existentes; zero erros críticos em 97/98 consultas simuladas de atenção primária
- **[GPT-5.5 Instant para Saúde](https://openai.com/index/improving-health-intelligence-in-chatgpt/)** (OpenAI, Jun 2026) -- melhora substancial em benchmarks de saúde em relação ao GPT-5.3 Instant

### Regulação & FDA

- **[EchoNext — Primeiro Dispositivo de IA para ECG de 12 Derivações Aprovado pela FDA](https://lasvegassun.com/news/2026/jun/23/pathway-labs-announces-worlds-first-fda-clearance-/)** (Pathway Labs, 23 Jun 2026) -- seis indicações cardiovasculares: IC direita/esquerda, doença valvar, cardiomiopatia infiltrativa e hipertensão pulmonar
- **[FDA Controles Anti-Viés para Dispositivos de IA](https://www.mddionline.com/artificial-intelligence/fda-s-ai-device-regulations-key-updates-compliance-strategies-for-medtech-success)** -- novas regras em vigor a partir de 30 Jun 2026: fornecedores devem publicar fontes de viés potencial e limites de explicabilidade

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
