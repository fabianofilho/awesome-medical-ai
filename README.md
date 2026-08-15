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

## Novidades -- Semana de 09 ago - 15 ago 2026

> Adicionados automaticamente pela curadoria semanal (awesome-medical-ai).

### Papers (3 novos)

**Infraestrutura e Dados em Saude:**
- [An AI-Health infrastructure for the Nordic region: technical foundations, data assets, and a roadmap for deployment](https://www.nature.com/articles/s41591-026-04575-4) -- Andreassen, Heimer, Brunak, Kallioniemi et al. -- Proposta de infraestrutura federada de IA em saude conectando 6 paises nordicos com grandes coortes longitudinais multimodais; modelos computacionais viajam entre ambientes seguros nacionais sem mover dados de pacientes; compativel com European Health Data Space (EHDS). (Nature Medicine, 14 Ago 2026)

**Imagem Medica:**
- [Auditable Agentic AI for Evidence-Grounded Thyroid Ultrasound Diagnosis and Reporting (ThyroidXAgent)](https://arxiv.org/abs/2608.12590) -- Sistema agentivo multicentrico para ultrassom de tireoide com coordenacao de ferramentas e trilha auditavel; treinado em ~300k imagens (OpenThyroidDB, 35 centros); Dice 87,21% em segmentacao, AUROC 0,9466 em classificacao benigno-maligno; reducao de 27,4% no tempo de laudagem e aumento de consistencia diagnostica de 70,3% para 86,2%. (arXiv 2608.12590, 12 Ago 2026)
- [An Information Theory Analysis of Whole Slide Image Pathology AI Under Limited Resources](https://arxiv.org/abs/2608.10846) -- Tsuruyama -- Compara WSI-AI (compressao automatica da lamina inteira) versus Diagnostic Field Selection AI (expert seleciona regioes primeiro); orienta alocacao de recursos computacionais em patologia digital com base em principios de teoria da informacao. (arXiv 2608.10846, 11 Ago 2026)

### Ferramentas e Repositorios

Sem novidades nesta semana.

### Datasets

Sem novidades nesta semana.

### Eventos

Sem novidades nesta semana.

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
