# Awesome Medical AI

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

Uma lista curada de recursos, frameworks e ferramentas incríveis para Inteligência Artificial na Medicina.

## Conteúdo

- [Artigos (Papers)](#artigos-papers)
- [Mão na Massa & Código (Hands-On & Code)](#mão-na-massa--código-hands-on--code)
- [Conjuntos de Dados (Datasets)](#conjuntos-de-dados-datasets)
- [Ferramentas & Bibliotecas (Tools & Libraries)](#ferramentas--bibliotecas-tools--libraries)
- [Modelos Pré-treinados (Pre-trained Models)](#modelos-pré-treinados-pre-trained-models)
- [Competições (Competitions)](#competições-competitions)
- [Contribuindo (Contributing)](#contribuindo-contributing)
- [Licença (License)](#licença-license)

---

## Artigos (Papers)

Uma seleção de artigos seminais e recentes que moldam o campo da IA na saúde.

### Modelos Preditivos e EHR
*   [A deep learning model for clinical outcome prediction using longitudinal inpatient electronic health records (2025)](https://academic.oup.com/jamiaopen/article/8/2/ooaf026/8110091) - Apresenta o TECO, um modelo baseado em Transformer para prever desfechos clínicos a partir de dados de prontuário eletrônico.
*   [Large language models encode clinical knowledge (2023)](https://www.nature.com/articles/s41586-023-06291-2) - Avaliação do Med-PaLM, um LLM para medicina, demonstrando sua capacidade em exames de competência médica.
*   [Predicting healthcare trajectories from medical records: A deep learning approach (2017)](https://www.sciencedirect.com/science/article/pii/S1532046417300710) - Uma abordagem de rede neural profunda para ler registros médicos e prever futuros desfechos.

### Imagem Médica (Radiologia, Patologia)
*   [Deep learning-based image classification for integrating pathology and radiology in AI-assisted medical imaging (2025)](https://www.nature.com/articles/s41598-025-07883-w) - Um framework para integrar dados multimodais em imagem médica.
*   [Generative AI and Foundation Models in Radiology (2025)](https://pubs.rsna.org/doi/10.1148/radiol.242961) - Uma revisão sobre o uso e o potencial de modelos de fundação em radiologia.
*   [A novel explainable AI framework for medical image classification (2025)](https://www.sciencedirect.com/science/article/pii/S1361841525002129) - Propõe um novo método de IA explicável (XAI) projetado especificamente para análise de imagens médicas.

### NLP em Saúde
*   [The Growing Impact of Natural Language Processing in Healthcare (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11475376/) - Uma revisão narrativa que resume os usos atuais de PNL na saúde.
*   [BioBERT: a pre-trained biomedical language representation model for biomedical text mining (2020)](https://academic.oup.com/bioinformatics/article/36/4/1234/5566506) - Introduz o BioBERT, um modelo de linguagem pré-treinado em literatura biomédica.

## Mão na Massa & Código (Hands-On & Code)

Repositórios, notebooks e tutoriais para colocar em prática seus conhecimentos em IA na saúde.

*   [aws-samples/aws-healthcare-lifescience-ai-ml-sample-notebooks](https://github.com/aws-samples/aws-healthcare-lifescience-ai-ml-sample-notebooks) - Notebooks de exemplo para IA/ML em saúde e ciências da vida na AWS.
*   [ageron/handson-ml3](https://github.com/ageron/handson-ml3) - Uma série de notebooks Jupyter que percorrem os fundamentos de Machine Learning e Deep Learning em Python usando Scikit-Learn, Keras e TensorFlow.
*   [NIGMS/AI-ML-For-Biomedical-Researchers](https://github.com/NIGMS/AI-ML-For-Biomedical-Researchers) - Um módulo prático de IA/ML centrado em dados para pesquisadores biomédicos.
*   [DeepPSP/torch_ecg](https://github.com/DeepPSP/torch_ecg) - Modelos de Deep Learning para ECG implementados em PyTorch, com ferramentas para processamento de sinais e treinamento.
*   [intel/fl-tutorial](https://github.com/intel/fl-tutorial) - Tutorial oficial da MICCAI 2022 sobre Aprendizado Federado para Saúde.

## Conjuntos de Dados (Datasets)

Datasets abertos e de alta qualidade para treinar e validar seus modelos.

### Prontuários Eletrônicos (EHR)
*   [MIMIC-IV](https://physionet.org/content/mimiciv/) - Um grande banco de dados de acesso público compreendendo dados de saúde não identificados associados a pacientes que permaneceram em unidades de terapia intensiva.
*   [eICU Collaborative Research Database](https://eicu-crd.mit.edu/) - Uma base de dados multicêntrica com dados de alta granularidade de UTIs nos EUA.

### Imagens Médicas
*   [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) - Um grande dataset de radiografias de tórax com 14 observações comuns como labels.
*   [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/) - Um grande arquivo de imagens médicas de câncer em formato DICOM.
*   [Medical Segmentation Decathlon](http://medicaldecathlon.com/) - Um conjunto de dados para segmentação de imagens médicas, abrangendo vários órgãos e patologias.
*   [ISIC Archive](https://www.isic-archive.com/#!/topWithHeader/onlyHeaderTop/gallery) - Grande coleção de imagens de lesões de pele para classificação de melanoma.
*   [PadChest](http://bimcv.cipf.es/bimcv-projects/padchest/) - Um grande conjunto de dados de radiografias de tórax com anotações de mais de 170 achados radiológicos.

### Sinais Biológicos
*   [PhysioNet](https://physionet.org/about/database/) - Um grande repositório de sinais fisiológicos gravados, incluindo ECG, EEG, e mais.
*   [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) - Um grande conjunto de dados de eletrocardiografia com anotações clínicas detalhadas.

## Ferramentas & Bibliotecas (Tools & Libraries)

Frameworks e bibliotecas essenciais para acelerar o desenvolvimento de soluções de IA na saúde.

### Imagem Médica
*   [MONAI](https://monai.io/) - Um toolkit de IA de código aberto para imagem médica, baseado em PyTorch.
*   [TorchXRayVision](https://github.com/mlmed/torchxrayvision) - Uma biblioteca para trabalhar com radiografias de tórax com modelos pré-treinados.
*   [MedPy](https://loli.github.io/medpy/) - Uma biblioteca de processamento de imagens médicas em Python.

### Análise de Sobrevida e EHR
*   [PyHealth](https://pyhealth.readthedocs.io/) - Uma biblioteca Python abrangente para IA na saúde, facilitando a construção de modelos de ML em dados clínicos.
*   [scikit-survival](https://scikit-survival.readthedocs.io/) - Análise de sobrevida construída sobre o Scikit-learn.
*   [lifelines](https://lifelines.readthedocs.io/) - Implementação de modelos de análise de sobrevida em Python.

### NLP em Saúde
*   [John Snow Labs Spark NLP for Healthcare](https://www.johnsnowlabs.com/healthcare-nlp/) - NLP de estado da arte para saúde, com modelos para NER, de-identificação e extração de relações.
*   [ClinicalBERT](https://github.com/EmilyAlsentzer/clinicalBERT) - Modelos BERT para notas clínicas, ajustados em dados do MIMIC-III.
*   [BioBERT](https://github.com/dmis-lab/biobert) - Um modelo de linguagem pré-treinado em literatura biomédica para mineração de texto.

## Modelos Pré-treinados (Pre-trained Models)

Modelos de fundação e outros modelos pré-treinados para acelerar a pesquisa e o desenvolvimento.

### Modelos de Linguagem (LLMs)
*   [Med-PaLM](https://sites.research.google/gr/med-palm/) - Um grande modelo de linguagem projetado para fornecer respostas de alta qualidade a perguntas médicas.
*   [BioGPT](https://github.com/microsoft/BioGPT) - Um LLM da Microsoft pré-treinado em literatura biomédica.
*   [GatorTron](https://github.com/uf-hobi-informatics-lab/GatorTron) - Um modelo de linguagem clínica da Universidade da Flórida, treinado em um grande corpus de dados de EHR.
*   [PubMedBERT](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext) - Um modelo BERT pré-treinado do zero em resumos e textos completos do PubMed.

### Modelos de Visão
*   [RadImageNet](https://github.com/BMEII-AI/RadImageNet) - Um modelo de classificação de imagens treinado em um grande conjunto de dados de radiologia.
*   [MedSAM](https://github.com/bowang-lab/MedSAM) - O Segment Anything Model (SAM) adaptado para segmentação de imagens médicas.
*   [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) - Um modelo de visão-linguagem contrastivo para o domínio biomédico.

## Competições (Competitions)

Desafios para testar suas habilidades e contribuir para o avanço da área.

*   [Kaggle Competitions - Healthcare](https://www.kaggle.com/competitions?tagIds=4202-Healthcare) - Competições de ciência de dados focadas em saúde.
*   [Grand Challenge](https://grand-challenge.org/challenges/) - Uma plataforma que hospeda desafios em imagem médica e IA.
*   [PhysioNet/Computers in Cardiology Challenges](https://physionet.org/challenge/) - Desafios anuais focados em problemas clínicos interessantes usando dados do PhysioNet.
*   [MICCAI Challenges](https://miccai.org/index.php/special-interest-groups/challenges/miccai-registered-challenges/) - Desafios realizados em conjunto com a conferência MICCAI.

---

## Contribuindo (Contributing)

Suas contribuições são sempre bem-vindas! Por favor, dê uma olhada nas [diretrizes de contribuição](CONTRIBUTING.md) primeiro.

## Licença (License)

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para mais detalhes.
