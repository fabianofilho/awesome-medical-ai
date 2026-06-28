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

## Novidades -- Semana de 14-20 jun 2026

> Adicionados automaticamente pelo cron #10 (awesome-medical-ai).

### Papers & Benchmarks (2 novos)

**Benchmarks de IA para Ciências da Vida:**
- [LifeSciBench](https://openai.com/index/introducing-life-sci-bench/) (OpenAI, Jun 17, 2026) — 750 tarefas de pesquisa em ciências da vida elaboradas por 173 cientistas PhD de biotech/pharma; rubrics com 25 critérios cada (19.020 critérios no total); melhor modelo passa apenas 36,1%; principal gargalo: processamento de artefatos (sequências genômicas, estruturas químicas, figuras experimentais)
- [GPT-5.4 como Agente Autônomo em Química Medicinal](https://openai.com/index/introducing-new-capabilities-to-gpt-rosalind/) (OpenAI + Molecule.one, Jun 17, 2026) — primeira instância documentada publicamente de modelo frontier como agente quase-autônomo em workflow de laboratório úmido; melhoria de Chan-Lam coupling em 10.080 reações; TEMPO como oxidante inesperado para sulfonamidas primárias (farmacóforo em >91 medicamentos aprovados pela FDA)

### Ferramentas (2 novas)

- **[ChatGPT for Clinicians](https://openai.com/index/making-chatgpt-better-for-clinicians/)** (OpenAI, Abr 22, 2026) — acesso gratuito para médicos, NPs, PAs e farmacêuticos americanos verificados; assistência em documentação, pesquisa médica e raciocínio clínico com GPT-5
- **[MUSE](https://news.cuanschutz.edu/dbmi/health-ai-tools-support-clinicians)** — Multi-LLM Uncertainty via Subset Ensemble (CU/DBMI); identifica subconjunto confiável de LLMs para cada questão clínica, auxiliando clínicos a interpretar a confiança das predições

### Datasets (2 novos)

- **[NCI IDC AI Annotations — Cancer Collections](https://pmc.ncbi.nlm.nih.gov/articles/PMC12307902/)** (Scientific Data, 2025–2026) — anotações AI-geradas para 11 coleções do NCI Imaging Data Commons; TC e RM de mama, cérebro, fígado, pulmões e próstata; modelos nnU-Net validados por radiologistas certificados; conformidade DICOM
- **[Project Imaging-X](https://arxiv.org/abs/2603.27460)** (arXiv 2603.27460, 2026) — survey de 1.000+ datasets públicos de imagem médica para desenvolvimento de foundation models; curadoria por modalidade, patologia e licença

### Arquivos Atualizados

`README.md` · `papers/nlp-health.md` · `papers/drug-discovery.md` · `tools/nlp.md` · `datasets/medical-imaging.md`

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
