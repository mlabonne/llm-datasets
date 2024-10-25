<div align="center">
  <h1>💾 LLM Datasets</h1>
  <p>
    🐦 <a href="https://twitter.com/maximelabonne">Follow me on X</a> • 
    🤗 <a href="https://huggingface.co/mlabonne">Hugging Face</a> • 
    💻 <a href="https://mlabonne.github.io/blog">Blog</a> • 
    📙 <a href="https://github.com/PacktPublishing/Hands-On-Graph-Neural-Networks-Using-Python">Hands-on GNN</a>
  </p>
   <p><em>High-quality datasets, tools, and concepts for LLM fine-tuning.</em></p>
</div>
<br/>

## 👍 What is a good dataset?

Data is the most valuable asset in LLM development. While datasets can't be directly evaluated like models, high-quality datasets have the following characteristics:

* **Accuracy**: Samples should be factually correct, helpful to users, and well-written. Answers should also be relevant to their corresponding instructions.
* **Diversity**: You want to cover as many use cases as possible to ensure proper instruction-following and relevant answers. This requires a wide range of topics, contexts, lengths, writing styles, etc. sampled in a representative way.
* **Complexity**: Answers should be nontrivial and a/ representative of tasks you expect the model to handle or b/ include complex tasks involving multi-step reasoning, planning, etc. 

Measuring accuracy can be easy in the case of mathematical problems using a Python interpreter, or near-impossible with open-ended, subjective questions. On the other hand, clustering datasets by topic is a good way of measuring diversity. Finally, complexity can be assessed using other LLMs acting like judges.

## 📅 Open SFT datasets

Once a model has been pre-trained on a next-token prediction task, supervised fine-tuning is used to turn it into an assistant capable of answering questions and achieving tasks. These datasets contain pairs of instructions and outputs to train LLMs to go beyond their pre-training objective. All the datasets listed here should be under permissive licensing (Apache 2.0, MIT, cc-by-4.0, etc.).

### General-purpose

The goal of general-purpose datasets is to transform base models into versatile and capable assistants by exposing them to a wide range of high-quality data. These datasets often include a diverse mix of real-world and synthetic data, commonly generated using models like GPT-4.

| Dataset                                                                                                       | #     | Authors                      | Date     | Notes                                                                             |
| ------------------------------------------------------------------------------------------------------------- | ----- | ---------------------------- | -------- | --------------------------------------------------------------------------------- |
| [Infinity-Instruct](https://huggingface.co/datasets/BAAI/Infinity-Instruct)                          | 7.45M  | BAAI | Aug 2024 | High-quality evolved samples based on a collection of open-source datasets. |
| [WebInstructSub](https://huggingface.co/datasets/chargoddard/WebInstructSub-prometheus)  | 2.39M  | Yue et al. | May 2024 | Instructions created by retrieving document from Common Crawl, extracting QA pairs, and refining them. See the [MAmmoTH2 paper](https://arxiv.org/abs/2405.03548) (this is a subset). |
| [The-Tome](https://huggingface.co/datasets/arcee-ai/The-Tome)               | 1.75M | Arcee AI | Jul 2024 | Reranked and filtered collection of datasets with a focus on instruction following. See my [100k subset](https://huggingface.co/datasets/mlabonne/FineTome-100k). |
| [Hercules v4.5](https://huggingface.co/datasets/Locutusque/hercules-v4.5)   | 1.72M | Sebastian Gabarain | Apr 2024 | Large-scale general-purpose dataset with math, code, RP, etc. See [v4](https://huggingface.co/datasets/Locutusque/hercules-v4.0) for the list of datasets. |
| [Dolphin-2.9](https://huggingface.co/datasets/cognitivecomputations/Dolphin-2.9) | 1.39M | Cognitive Computations | Apr 2023 | Large-scale general-purpose dataset used by the Dolphin models. |
| [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M)          | 1.04M  | Zhao et al. | May 2023 | Real conversations between human users and GPT-3.5/4, including metadata. See the [WildChat paper](https://arxiv.org/abs/2405.01470). |
| [OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5)    | 1M    | Teknium | Nov 2023 | Another large-scale dataset used by the OpenHermes models. |
| [SlimOrca](https://huggingface.co/datasets/Open-Orca/SlimOrca)              | 518k  | Lian et al. | Sep 2023 | Curated subset of [OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca) using GPT-4-as-a-judge to remove wrong answers. |
| [Tulu V2 Mix](https://huggingface.co/datasets/allenai/tulu-v2-sft-mixture)  | 326k  | Ivison et al. | Nov 2023 | Mix of high-quality datasets. See [Tulu 2 paper](https://arxiv.org/abs/2311.10702). |
| [UltraInteract SFT](https://huggingface.co/datasets/openbmb/UltraInteract_sft) | 289k  | Yuan et al. | Apr 2024 | Focus on math, coding, and logic tasks with step-by-step answers. See [Eurus paper](https://arxiv.org/abs/2404.02078). |
| [NeurIPS-LLM-data](https://huggingface.co/datasets/upaya07/NeurIPS-LLM-data)   | 204k  | Jindal et al. | Nov 2023 | Winner of [NeurIPS LLM Efficiency Challenge](https://llm-efficiency-challenge.github.io/), with an interesting data preparation strategy. |
| [UltraChat 200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | 200k  | Tunstall et al., Ding et al. | Oct 2023 | Heavily filtered version of the [UItraChat](https://github.com/thunlp/UltraChat) dataset, consisting of 1.4M dialogues generated by ChatGPT. |
| [WizardLM_evol_instruct_V2](https://huggingface.co/datasets/mlabonne/WizardLM_evol_instruct_v2_196K-ShareGPT) | 143k  | Xu et al. | Jun 2023 | Latest version of Evol-Instruct applied to Alpaca and ShareGPT data. See [WizardLM paper](https://arxiv.org/abs/2304.12244). |
| [Synthia-v1.3](https://huggingface.co/datasets/migtissera/Synthia-v1.3)  | 119k  | Migel Tissera | Nov 2023 | High-quality synthetic data generated using GPT-4. |
| [oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1)                                                | 84.4k | Köpf et al.                  | Mar 2023 | Human-generated assistant-style conversation corpus in 35 different languages. See [OASST1 paper](https://arxiv.org/abs/2304.07327) and [oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2). |
| [WizardLM_evol_instruct_70k](https://huggingface.co/datasets/mlabonne/WizardLM_evol_instruct_70k-ShareGPT) | 70k   | Xu et al.                    | Apr 2023 | Evol-Instruct applied to Alpaca and ShareGPT data. See [WizardLM paper](https://arxiv.org/abs/2304.12244).                                                                                              |
| [airoboros-3.2](https://huggingface.co/datasets/jondurbin/airoboros-3.2)                                      | 58.7k | Jon Durbin                   | Dec 2023 | High-quality uncensored dataset.                                                                                                                                                                        |
| [ShareGPT_Vicuna_unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)       | 53k   | anon823 1489123              | Mar 2023 | Filtered version of the ShareGPT dataset, consisting of real conversations between users and ChatGPT.                                                                                                   |
| [lmsys-chat-1m-smortmodelsonly](https://huggingface.co/datasets/Nebulous/lmsys-chat-1m-smortmodelsonly)       | 45.8k | Nebulous, Zheng et al.       | Sep 2023 | Filtered version of [lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) with responses from GPT-4, GPT-3.5-turbo, Claude-2, Claude-1, and Claude-instant-1.                            |
| [Open-Platypus](https://huggingface.co/datasets/garage-bAInd/Open-Platypus)                                   | 24.9k | Lee et al.                   | Sep 2023 | Collection of datasets that were deduplicated using Sentence Transformers (it contains an NC dataset). See [Platypus paper](https://arxiv.org/abs/2308.07317).                                          |
| [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)                       | 15k   | Conover et al.               | May 2023 | Generated by Databricks employees, prompt/response pairs in eight different instruction categories, including the seven outlined in the InstructGPT paper.                                              |

### Math & Logic

LLMs often struggle with mathematical reasoning and formal logic, which has led to the creation of specialized datasets. These datasets extend beyond pure mathematics, encompassing a wide range of problems that require systematic thinking and step-by-step reasoning, ultimately enabling LLMs to tackle complex real-world challenges that involve logical deduction and quantitative analysis.

| Dataset                                                                             | #    | Authors      | Date     | Notes                                                                                                                                  |
| ----------------------------------------------------------------------------------- | ---- | ------------ | -------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| [OpenMathInstruct-1](https://huggingface.co/datasets/nvidia/OpenMathInstruct-1)     | 5.75M | Toshniwal et al. | Feb 2024 | Problems from GSM8K and MATH, solutions generated by Mixtral-8x7B                                                                      |
| [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA)                  | 395k | Yu et al.    | Dec 2023 | Bootstrap mathematical questions by rewriting them from multiple perspectives. See [MetaMath paper](https://arxiv.org/abs/2309.12284). |
| [MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct)              | 262k | Yue et al.   | Sep 2023 | Compiled from 13 math rationale datasets, six of which are newly curated, and focuses on chain-of-thought and program-of-thought.      |
| [Orca-Math](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k) | 200k | Mitra et al. | Feb 2024 | Grade school math world problems generated using GPT4-Turbo. See [Orca-Math paper](https://arxiv.org/pdf/2402.14830.pdf).              |
|[NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) | 859k | Jia LI et al. | 2024 |The sources of the dataset range from Chinese high school math exercises to US and international mathematics olympiad competition problems. |

### Code

Code is another challenging domain for LLMs that lack specialized pre-training. Code datasets, containing diverse programming language examples, are used to fine-tune LLMs and enhance their ability to understand, generate, and analyze code, enabling them to serve as effective coding assistants.

| Dataset                                                                                                      | #     | Authors       | Date     | Notes                                                                                                                                                                                                                                                                                                                                                 |
| ------------------------------------------------------------------------------------------------------------ | ----- | ------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [CodeFeedback-Filtered-Instruction](https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction) | 157k  | Zheng et al.  | Feb 2024 | Filtered version of Magicoder-OSS-Instruct, ShareGPT (Python), Magicoder-Evol-Instruct, and Evol-Instruct-Code.                                                                                                                                                                                                                                       |
| [Tested-143k-Python-Alpaca](https://huggingface.co/datasets/Vezora/Tested-143k-Python-Alpaca)                | 143k  | Vezora        | Mar 2024 | Collection of generated Python code that passed automatic tests to ensure high quality.                                                                                                                                                                                                                                                               |
| [glaive-code-assistant](https://huggingface.co/datasets/glaiveai/glaive-code-assistant)                      | 136k  | Glaive.ai     | Sep 2023 | Synthetic data of problems and solutions with ~60% Python samples. Also see the [v2](https://huggingface.co/datasets/glaiveai/glaive-code-assistant-v2) version.                                                                                                                                                                                      |
| [Magicoder-Evol-Instruct-110K](https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K)        | 110k  | Wei et al.    | Nov 2023 | A decontaminated version of [evol-codealpaca-v1](https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1). Decontamination is done in the same way as StarCoder ([bigcode decontamination process](https://github.com/bigcode-project/bigcode-dataset/tree/main/decontamination)). See [Magicoder paper](https://arxiv.org/abs/2312.02120). |
| [dolphin-coder](https://huggingface.co/datasets/cognitivecomputations/dolphin-coder)                         | 109k  | Eric Hartford | Nov 2023 | Dataset transformed from [leetcode-rosetta](https://www.kaggle.com/datasets/erichartford/leetcode-rosetta).                                                                                                                                                                                                                                           |
| [synthetic_tex_to_sql](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql)                       | 100k  | Gretel.ai     | Apr 2024 | Synthetic text-to-SQL samples (~23M tokens), covering diverse domains.                                                                                                                                                                                                                                                                                |
| [sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context)                               | 78.6k | b-mc2         | Apr 2023 | Cleansed and augmented version of the [WikiSQL](https://huggingface.co/datasets/wikisql) and [Spider](https://huggingface.co/datasets/spider) datasets.                                                                                                                                                                                               |
| [Magicoder-OSS-Instruct-75K](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K)            | 75k   | Wei et al.    | Nov 2023 | OSS-Instruct dataset generated by `gpt-3.5-turbo-1106`. See [Magicoder paper](https://arxiv.org/abs/2312.02120).                                                                                                                                                                                                                                      |
| [Code-Feedback](https://huggingface.co/datasets/m-a-p/Code-Feedback)                                         | 66.4k | Zheng et al.  | Feb 2024 | Diverse Code Interpreter-like dataset with multi-turn dialogues and interleaved text and code responses. See [OpenCodeInterpreter paper](https://arxiv.org/abs/2402.14658).                                                                                                                                                                           |
| [Open-Critic-GPT](https://huggingface.co/datasets/Vezora/Open-Critic-GPT) | 55.1k | Vezora  | Jul 2024 | Use a local model to create, introduce, and identify bugs in code across multiple programming languages. |
| [self-oss-instruct-sc2-exec-filter-50k](https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k) | 50.7k | Lozhkov et al.  | Apr 2024 | Created in three steps with seed functions from TheStack v1, self-instruction with StarCoder2, and self-validation. See the [blog post](https://huggingface.co/blog/sc2-instruct). |

### Conversation & Role-Play

Many datasets focus on pairs of instructions and outputs, but chat models are often used in conversational settings. Conversational and role-play datasets expose LLMs to the patterns, nuances, and context-dependent nature of real conversations, allowing them to generate more natural, and engaging dialogues.

| Dataset                                                                             | #     | Authors                 | Date     | Notes                                                                                                         |
| ----------------------------------------------------------------------------------- | ----- | ----------------------- | -------- | ------------------------------------------------------------------------------------------------------------- |
| [Bluemoon](https://huggingface.co/datasets/Squish42/bluemoon-fandom-1-1-rp-cleaned) | 290k  | Squish42                | Jun 2023 | Posts from the Blue Moon roleplaying forum cleaned and scraped by a third party.                              |
| [PIPPA](https://huggingface.co/datasets/kingbri/PIPPA-shareGPT)                     | 16.8k | Gosling et al., kingbri | Aug 2023 | Deduped version of Pygmalion's [PIPPA](https://huggingface.co/datasets/PygmalionAI/PIPPA) in ShareGPT format. |
| [Capybara](https://huggingface.co/datasets/LDJnr/Capybara)                          | 16k   | LDJnr                   | Dec 2023 | Strong focus on information diversity across a wide range of domains with multi-turn conversations.           |
| [RPGPT_PublicDomain-alpaca](https://huggingface.co/datasets/practical-dreamer/RPGPT_PublicDomain-alpaca) | 4.26k | practical dreamer                   | May 2023 | Synthetic dataset of public domain character dialogue in roleplay format made with [build-a-dataset](https://github.com/practical-dreamer/build-a-dataset)                                        |
| [Pure-Dove](https://huggingface.co/datasets/LDJnr/Pure-Dove)                        | 3.86k | LDJnr                   | Sep 2023 | Highly filtered multi-turn conversations between GPT-4 and real humans                                        |
| [Opus Samantha](https://huggingface.co/datasets/macadeliccc/opus_samantha)          | 1.85k | macadelicc              | Apr 2024 | Multi-turn conversations with Claude 3 Opus.                                                                  |
| [LimaRP-augmented](https://huggingface.co/datasets/grimulkan/LimaRP-augmented)      | 804   | lemonilia, grimulkan    | Jan 2024 | Augmented and cleansed version of LimaRP, consisting of human roleplaying conversations.                      |

### Multilingual

Learning new languages "from scratch" is a pre-training task, but providing multilingual instruction samples is useful to boost performance in the languages of interest.

| Dataset                                                                                                       | #     | Authors                      | Date     | Notes                                                                             |
| ------------------------------------------------------------------------------------------------------------- | ----- | ---------------------------- | -------- | --------------------------------------------------------------------------------- |
| [M2Lingual](https://huggingface.co/datasets/ServiceNow-AI/M2Lingual)                          | 175K  | ServiceNow AI | June 2024 | Dataset spanning 70+ langauges and 20 NLP tasks generated from GPT-4 using task-based taxonomy guided evolutions. More details in [M2Lingual](https://arxiv.org/abs/2406.16783) paper.|

### Agent & Function calling

Function calling allows large language models (LLMs) to execute predefined functions with parameters inferred from user prompts, rather than generating standard text responses. This enables LLMs to seamlessly integrate with external systems, perform complex operations, and provide more accurate and contextually relevant responses.

| Dataset                                                                                           | #     | Authors         | Date     | Notes                                                                               |
| ------------------------------------------------------------------------------------------------- | ----- | --------------- | -------- | ----------------------------------------------------------------------------------- |
| [glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) | 113k  | Sahil Chaudhary | Sep 2023 | High-quality dataset with pairs of instructions and answers in different languages. <br>See [Locutusque/function-calling-chatml](https://huggingface.co/datasets/Locutusque/function-calling-chatml) for a variant without conversation tags. |
| [xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)                          | 60k   | Salesforce      | Jun 2024 | Samples created using a data generation pipeline designed to produce verifiable data for function-calling applications |
| [Agent-FLAN](https://huggingface.co/datasets/internlm/Agent-FLAN)                                 | 34.4k | internlm        | Mar 2024 | Mix of AgentInstruct, ToolBench, and ShareGPT datasets.                             |

## ⚖️ Preference alignment

 Preference alignment datasets are collections of data used to address the alignment of an LLM's goals with those of its human operators or users. These datasets help LLM's stay consistent with human values and preferences.

| Dataset                                                                             | #     | Authors                 | Date     | Notes                                                                                                         |
| ----------------------------------------------------------------------------------- | ----- | ----------------------- | -------- | ------------------------------------------------------------------------------------------------------------- |
|[ultrafeedback-binarized-preferences-cleaned](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned) | 158k | Argilla | 2023 | This dataset has been created to explore whether DPO fine-tuning with more than one rejection per chosen response helps the model perform better in the AlpacaEval, MT-Bench, and LM Eval Harness benchmarks.
|[ultrafeedback-multi-binarized-preferences-cleaned](https://huggingface.co/datasets/argilla/ultrafeedback-multi-binarized-preferences-cleaned) | 158k | Argilla | 2023 | This dataset represents a new iteration on top of argilla/ultrafeedback-binarized-preferences-cleaned, and has been created to explore whether DPO fine-tuning with more than one rejection per chosen response helps the model perform better in the AlpacaEval, MT-Bench, and LM Eval Harness benchmarks.



## 🔧 Tools

To create a high-quality dataset, focus on carefully curating a diverse set of relevant, accurate, and informative examples rather than simply maximizing dataset size.

Start by aggregating available data from various sources (open-source or not) and applying filters like data deduplication and data quality. If the initial dataset is small or insufficient, consider synthetically generating additional data that mirrors its quality and style. Iteratively explore and refine the dataset by assessing model performance, identifying gaps, and collecting or generating data to address those shortcomings.

Tools listed in this section may belong to several categories but appear in only one for clarity.

### Data deduplication and decontamination
* **Exact deduplication**: Remove identical samples with data normalization (e.g., convert text to lowercase), hash generation (e.g., create an MD5 or SHA-256 hash for each sample), and duplicate removal.
* **Fuzzy deduplication**
  * **MinHash**: Fuzzy deduplication with hashing, sorting, and Jaccard similarity (preferred technique).
  * **BLOOM filters**: Fuzzy deduplication with hashing and fixed-size vector.
* **Decontamination**: Remove samples too close to test sets, using either exact or fuzzy filtering.

### Data quality evaluation
* **Rule-based filtering**: Remove samples based on a list of unwanted words, like refusals and "As an AI assistant" ([example](https://huggingface.co/datasets/cognitivecomputations/WizardLM_alpaca_evol_instruct_70k_unfiltered/blob/main/wizardlm_clean.py)).
* [**Argilla**](https://argilla.io/): Open-source data curation platform that allows you to filter and annotate datasets in a collaborative way.
* [**LLM-as-a-judge**](https://colab.research.google.com/github/huggingface/cookbook/blob/main/notebooks/en/llm_judge.ipynb): Colab notebook that provides code to rate outputs with Mixtral-7x8B.
* [**Data Prep Kit**](https://github.com/IBM/data-prep-kit): Framework for data preparation for both code and language, with modules in Python, Ray, and Spark, and a wide range of scale from laptops to data centers.
* [**DataTrove**](https://github.com/huggingface/datatrove/): HuggingFace library for large-scale data processing, used in the creation of [Fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb).

### Data generation

#### SFT datasets
* [**Distilabel**](https://github.com/argilla-io/distilabel): General-purpose framework that can generate and augment data (SFT, DPO) with techniques like UltraFeedback and DEITA.
* [**Auto Data**](https://github.com/Itachi-Uchiha581/Auto-Data): Lightweight library to automatically generate fine-tuning datasets with API models.
* [**Bonito**](https://github.com/BatsResearch/bonito): Library for generating synthetic instruction tuning datasets for your data without GPT (see also [AutoBonito](https://colab.research.google.com/drive/1l9zh_VX0X4ylbzpGckCjH5yEflFsLW04?usp=sharing)).
* [**Augmentoolkit**](https://github.com/e-p-armstrong/augmentoolkit): Framework to convert raw text into datasets using open-source and closed-source models.
* [**Magpie**](https://github.com/magpie-align/magpie): Your efficient and high-quality synthetic data generation pipeline by prompting aligned LLMs with nothing.
* [**Genstruct**](https://huggingface.co/NousResearch/Genstruct-7B): An instruction generation model, which is designed to generate valid instructions from raw data.
* [**DataDreamer**](https://datadreamer.dev/docs/latest/): A python library for prompting and synthetic data generation.

#### Pre-training datasets
* [**llm-swarm**](https://github.com/huggingface/llm-swarm): Generate synthetic datasets for pretraining or fine-tuning using either local LLMs or Inference Endpoints on the Hugging Face Hub.
* [**Cosmopedia**](https://github.com/huggingface/cosmopedia): Hugging Face's code for generating the [Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) dataset.
* [**textbook_quality**](https://github.com/VikParuchuri/textbook_quality): A repository for generating textbook-quality data, mimicking the approach of the Microsoft's Phi models.

### Data exploration
* [**sentence-transformers**](https://sbert.net/): A python module for working with popular language embedding models.
* [**Lilac**](https://github.com/lilacai/lilac): Tool to curate better data for LLMs, used by NousResearch, databricks, cohere, Alignment Lab AI. It can also apply filters.
* [**Nomic Atlas**](https://github.com/nomic-ai/nomic): Interact with instructed data to find insights and store embeddings.
* [**text-clustering**](https://github.com/huggingface/text-clustering): A framework from Huggingface for clustering textual data.
* [**BunkaTopics**](https://github.com/charlesdedampierre/BunkaTopics): Data cleaning and topic modeling visualization.
* [**Autolabel**](https://github.com/refuel-ai/autolabel): Automatically label data using popular language models.

### Data scraping
* [**Trafilatura**](https://github.com/adbar/trafilatura**): Python and command-line tool to gather text and metadata on the web. Used for the creation of [RefinedWeb](https://arxiv.org/abs/2306.01116).
* [**Marker**](https://github.com/VikParuchuri/marker): Quickly convert PDFs to markdown text.

## Acknowledgments

Special thanks to [geronimi73](https://github.com/geronimi73), [Bytes-Explorer](https://github.com/Bytes-Explorer), [euclaise](https://github.com/euclaise), [RishabhMaheshwary](https://github.com/RishabhMaheshwary), and [ParagEkbote](https://github.com/ParagEkbote) for their PRs.

## References

Please let me know if a dataset is not properly credited.

- Wei-Lin Chiang et al, "Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%* ChatGPT Quality," 2023.
- Yihan Cao et al, "Instruction Mining: When Data Mining Meets Large Language Model Finetuning," 2023.
- Subhabrata Mukherjee et al, "Orca: Progressive Learning from Complex Explanation Traces of GPT-4," 2023.
- Chunting Zhou et al, "LIMA: Less Is More for Alignment," 2023.
- Suriya Gunasekar et al, "Textbooks Are All You Need," 2023.
- Lichang Chen et al, "AlpaGasus: Training A Better Alpaca with Fewer Data," 2024.
- Zheng Cai et al, "InternLM2 Technical Report," 2024.
- Lifan Yuan et al, "Advancing LLM Reasoning Generalists with Preference Trees," 2024.
- Wei Liu et al, "What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning," 2024.
- Xingyao Wang et al, "MINT: Evaluating LLMs in Multi-turn Interaction with Tools and Language Feedback," 2024.
