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

## Novidades -- Semana de 15-22 ago 2026

> Adicionados automaticamente pela curadoria semanal (awesome-medical-ai).

### Papers (2 novos)

**Segurança e Revisões:**
- [Safety and security of large language models in healthcare](https://www.nature.com/articles/s41586-026-10687-1) -- Clusmann, Freyer, Ostermann et al. (TU Dresden / Kather group) -- revisão em Nature integrando evidências de IA médica, cibersegurança, ética e psicologia comportamental; apresenta framework unificado de ameaças relevantes, camadas de proteção e responsabilidades de mitigação para uso seguro de LLMs na prática clínica. (Nature, 19 ago 2026)
- [90% dos papers biomédicos mostram sinais de uso de IA em escrita](https://www.nature.com/articles/d41586-026-02551-z) -- Nature News sobre preprint (arXiv, 12 ago): estima 77% de uso de LLMs nos artigos do PubMed Central de 2025 e 90% nos artigos de dezembro de 2025; uso mais frequente em resumo, introdução e discussão do que em métodos e resultados. (Nature, 20 ago 2026)

### Ferramentas e Repositórios

Sem novidades nesta semana.

### Datasets

Sem novidades nesta semana.

### Eventos

Sem novidades nesta semana.

---

## Novidades -- Semana de 27 jun - 04 jul 2026

> Adicionados automaticamente pela curadoria semanal (awesome-medical-ai).

### Papers (2 novos)

**NLP & LLMs Clinicos:**
- [MedGellan: LLM-Generated Medical Guidance to Support Physicians](https://arxiv.org/abs/2507.04431) -- Framework leve com prompting bayesiano que gera orientacao clinica a partir de prontuarios brutos para apoio diagnostico; melhora recall e F1 sem substituir o medico. (arXiv 2507.04431, Jul 2026)
- [VISTA Architect: A graph database-oriented health AI system demonstrated in multidisciplinary tumor boards](https://arxiv.org/abs/2606.22692) -- Kiiskinen et al. (Stanford) — arquitetura em dois niveis (MEDS Graph + Timeline Object Architecture) para fornecer contexto clinico longitudinal completo a LLMs em tumor boards. (arXiv 2606.22692, Jun 2026)

### Ferramentas e Repositorios (3 novos)

- **[UniMedVL](https://github.com/uni-medical/UniMedVL)** (ICML 2026) -- modelo fundacional unificado de visao e linguagem medica; suporta VQA, geracao de laudos, sintese de imagens e traducao cross-modal em 9 modalidades de imagem. Aceito no ICML 2026.
- **[PhysicianBench](https://github.com/HealthRex/PhysicianBench)** (Stanford HealthRex, 2026) -- harness de avaliacao de agentes LLM em ambientes EHR reais via FHIR; 100 tarefas de longo horizonte em 21 especialidades; melhor modelo atinge apenas 46% de sucesso (pass@1).
- **[Awesome-AI-Agents-for-Healthcare](https://github.com/AgenticHealthAI/Awesome-AI-Agents-for-Healthcare)** (AgenticHealthAI, 2026) -- lista curada e atualizada de avancos em IA agentiva para saude, com papers de 2026 sobre agentes medicos multimodais e agenticos.

### Datasets (1 novo)

- **[UniMedVL-5M](https://huggingface.co/datasets/General-Medical-AI/UniMedVL-5M)** (HuggingFace, Jun 2026) -- primeiro dataset medico em grande escala com mais de 5,6 milhoes de instancias em 8 modalidades de imagem para tarefas multimodais; disponivel para pesquisa academica nao-comercial.

### Eventos (3 novos)

- **[HealthAI2026](https://healthai.pagesconferences.org/)** -- 27-28 Jul 2026, Praga, Republica Tcheca. Conferencia sobre medicina de precisao e IA em saude com clinicos, pesquisadores e reguladores.
- **[MLHC 2026](https://mlhc.org/)** -- 12-14 Ago 2026, Machine Learning for Healthcare Conference. Principal venue para ML aplicado a saude; decisoes de aceitacao em Jul 2026.
- **[AIiH 2026](https://aiih.cc/)** -- 26-28 Ago 2026, Imperial College Londres. Conferencia internacional sobre IA em saude clinica.

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
