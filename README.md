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

## Novidades -- Semana de 11-17 out 2026

> Adicionados automaticamente pela curadoria semanal (awesome-medical-ai).

### Papers (5 novos)

**Epilepsia e Monitoramento Cerebral com IA:**
- [Patient-in-the-loop epilepsy monitoring via AI chat agent](https://www.nih.gov/news-events/news-releases/patients-engage-epilepsy-monitor-real-time-improving-seizure-detection) (Nature Medicine, 14 Out 2026) -- 13 adultos em unidade de monitoramento de epilepsia da Penn usaram app para consultar agente de IA com acesso ao EEG em tempo real; rotulos dos proprios pacientes sobre alarmes reduziram falsos alarmes em cerca de 80% em simulacao pos-hoc. (Goldblum, Shi, Litt et al.)

**LLMs no Cuidado Clinico:**
- [Large Language Models Recommend Costlier Care Than Treating Physicians](https://ai.nejm.org/) (NEJM AI, 15 Out 2026) -- estudo original compara o custo do cuidado recomendado por LLMs com o recomendado por medicos tratantes; LLMs tendem a indicar cuidado mais caro em analise de casos reais.
- [AI-Assisted Molecular Workflow in Adult Medulloblastoma](https://ai.nejm.org/) (NEJM AI, 16 Out 2026) -- caso clinico descrevendo como workflow molecular assistido por IA orientou decisao terapeutica em caso de meduloblastoma em adulto; primeira descricao detalhada desse uso clinico.
- [Auditing Escalation Reliability in Healthcare LLM Agents](https://arxiv.org/abs/2610.21270) (arXiv 2610.21270, Out 2026) -- benchmark de 400 episodios sinteticos de workflow hospitalar avaliando se agentes LLM respondem, pedem esclarecimento, absteem ou escalam quando evidencias estao ausentes ou desatualizadas.
- [Tool-Augmented LLMs for Reliable Medical Calculation from Patient Notes](https://arxiv.org/abs/2610.21477) (arXiv 2610.21477, Out 2026) -- framework que separa extracao de variaveis clinicas (LLM) do calculo clinico (ferramenta deterministica), reduzindo erros aritmeticos em notas medicas.

### Ferramentas e Repositorios

Sem novidades confirmadas nesta semana.

### Datasets

Sem novidades nesta semana.

### Eventos (3 novos)

- **[IEEE MedAI 2026](https://medai2026.zzuli.edu.cn/)** -- 4a Conferencia Internacional IEEE sobre IA Medica, 20-22 Nov 2026, Zhengzhou, China. Foco em aplicacoes de IA em biomed, diagnostico e robotica cirurgica; camera-ready deadline: 25 Out 2026.
- **[AI Care Conference 2026 (AIDH)](https://members.digitalhealth.org.au/m/event_details.asp?id=2059763)** -- 23-24 Nov 2026, Crown Melbourne, Southbank, Australia. Conferencia pratica da Associacao Australiana de Saude Digital reunindo clinicos, pesquisadores e formuladores de politicas em torno de adocao responsavel de IA na saude.
- **[iAIM 2026 — International Conference on AI in Medicine](https://easychair.org/cfp/iAIM2026)** -- 3-5 Dez 2026, Nanyang Technological University, Singapura. Papers aceitos para apresentacao presencial; deadline de submissao encerrado em Set 2026.

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
