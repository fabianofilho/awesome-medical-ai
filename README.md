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

## Novidades -- Semana de 04 jul - 11 jul 2026

> Adicionados automaticamente pela curadoria semanal (awesome-medical-ai).

### Papers (4 novos)

**NLP, LLMs e Agentes Clinicos:**
- [DynamiCare: A Dynamic Multi-Agent Framework for Interactive and Open-Ended Medical Decision-Making](https://arxiv.org/abs/2507.02616) -- Framework multi-agente dinamico que modela o diagnostico como um loop interativo multi-rodada; inclui MIMIC-Patient, dataset estruturado de pacientes virtuais derivado do MIMIC-III. (arXiv 2507.02616, Jul 2026)
- [Large Language Models for Automating Clinical Data Standardization: HL7 FHIR Use Case](https://arxiv.org/abs/2507.03067) -- Demonstra a viabilidade de LLMs para transformar dados clinicos heterogeneos no padrao HL7 FHIR de forma automatizada. (arXiv 2507.03067, Jul 2026)
- [Empowering Healthcare Practitioners with Language Models: Structuring Speech Transcripts in Two Real-World Clinical Applications](https://arxiv.org/abs/2507.05517) -- Avaliacao de LLMs para estruturacao de transcricoes de fala clinica em dois cenarios reais de uso hospitalar. (arXiv 2507.05517, Jul 2026)
- [PhysAssistBench: Are LLMs Ready to Assist Physicians?](https://arxiv.org/abs/2606.18613) -- Benchmark para assistencia interativa medico-paciente-EHR com 1.296 turnos validados por clinicos extraidos de casos MIMIC-IV; modelos atuais mostram desempenho insatisfatorio. (arXiv 2606.18613, Jun 2026)

### Ferramentas e Repositorios (2 novos)

- **[OpenMed](https://github.com/maziyarpanahi/openmed)** (2026) -- IA clinica local-first com NER biomedico e de-identificacao HIPAA 100% on-device; suporta 1.000+ modelos em 12 linguas; Apple MLX, Python, sem dados enviados para nuvem; Apache-2.0.
- **[MedAgentGym](https://github.com/wshi83/MedAgentGym)** (ICLR 2026) -- Ambiente de treinamento escalavel para agentes LLM em raciocinio biomedico orientado a codigo; 72.413 instancias em 129 categorias de 12 cenarios reais; Med-Copilot atinge +43% via RL. (arXiv 2506.04405)

### Eventos (3 novos ou atualizados)

- **[AIME 2026 — 24th International Conference on AI in Medicine](https://aime26.aimedicine.info/)** -- 7-10 Jul 2026, Universidade de Ottawa, Canada. Tema: conhecimento, raciocinio e integracao semantica. Proceedings publicados pela Springer Nature.
- **[AIMed26](https://www.aimed26.com/)** -- 10-13 Nov 2026, Orlando, FL (Renaissance Orlando at Seaworld). Principal encontro de IA e medicina com keynotes, workshops e networking entre clinicos e cientistas de dados.
- **[DoctorsAI Global Summit 2026](https://www.doctorsaiglobalsummit.com/)** -- Tema: Precision, Partnership, Possibility. Aplicacoes de ago 20 a set 10, 2026. Evento voltado para medicos e pesquisadores que aplicam IA na pratica clinica.

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
