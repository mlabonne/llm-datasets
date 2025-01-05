<div align="center">
<img src="https://i.imgur.com/SekZcgb.png" alt="Image">
  <p>
    🐦 <a href="https://twitter.com/maximelabonne">Follow me on X</a> • 
    🤗 <a href="https://huggingface.co/mlabonne">Hugging Face</a> • 
    💻 <a href="https://mlabonne.github.io/blog">Blog</a> • 
    📙 <a href="https://github.com/PacktPublishing/Hands-On-Graph-Neural-Networks-Using-Python">Hands-on GNN</a>
  </p>
   <p><em>Curated list of datasets and tools for post-training.</em></p>
</div>
<br/>

## 👍 What is a good dataset?

Data is the most valuable asset in LLM development. When building a dataset, we target the three following characteristics:

* **Accuracy**: Samples should be factually correct and relevant to their corresponding instructions. This can involve using solvers for math and unit tests for code.
* **Diversity**: You want to cover as many use cases as possible to make sure you're never out of distribution. High diversity is essential as it leads to better generalization.
* **Complexity**: Answers should be both detailed (to maximize helpfulness) and include system 2 techniques like chain of thought (to force step-by-step reasoning).

Measuring accuracy is easy in most cases but near-impossible with open-ended, subjective questions. On the other hand, clustering datasets by topic is a good way of evaluating data mixture diversity. Finally, complexity can be assessed using other LLMs acting like judges.

## 📅 Open SFT datasets

Once a model has been pre-trained on a next-token prediction task, Supervised Fine-Tuning (SFT) is used to turn it into an assistant capable of answering questions and following instructions. These datasets contain pairs of instructions and outputs to train LLMs to understand conversational structure. Unless otherwise noted, all datasets listed here are under permissive licenses (Apache 2.0, MIT, CC-BY-4.0, etc.).

### General-purpose mixtures

General-purpose datasets offer balanced mixtures of different types of data, including chat, code, and math. These datasets can be used to create general-purpose models that can handle various types of queries.

| Dataset                                                                                               | #     | Authors            | Date     | Notes                                                                                                                                                                                                                                                        |
| ----------------------------------------------------------------------------------------------------- | ----- | ------------------ | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [Infinity-Instruct](https://huggingface.co/datasets/BAAI/Infinity-Instruct)                           | 7.45M | BAAI               | Aug 2024 | High-quality evolved samples based on a collection of open-source datasets.                                                                                                                                                                                  |
| [WebInstructSub](https://huggingface.co/datasets/chargoddard/WebInstructSub-prometheus)               | 2.39M | Yue et al.         | May 2024 | Instructions created by retrieving document from Common Crawl, extracting QA pairs, and refining them. See the [MAmmoTH2 paper](https://arxiv.org/abs/2405.03548) and [full set](https://huggingface.co/datasets/TIGER-Lab/WebInstructFull) (13.5M samples). |
| [The-Tome](https://huggingface.co/datasets/arcee-ai/The-Tome)                                         | 1.75M | Arcee AI           | Jul 2024 | Reranked and filtered collection of datasets with a focus on instruction following. See my [100k subset](https://huggingface.co/datasets/mlabonne/FineTome-100k).                                                                                            |
| [open-perfectblend](https://huggingface.co/datasets/mlabonne/open-perfectblend)                       | 1.42M | Xu et al., Labonne | Oct 2024 | Open reproduction of the dataset described [in this paper](https://arxiv.org/abs/2409.20370). It's a solid general-purpose instruction dataset with chat, math, code, and instruction-following data.                                                        |
| [smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)                                    | 1.1M  | Hugging Face       | Nov 2024 | Mix of existing and new datasets used to train [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) with proper evaluations.                                                                                                                |
| [orca-agentinstruct-1M-v1](https://huggingface.co/datasets/mlabonne/orca-agentinstruct-1M-v1-cleaned) | 1.05M | Microsoft          | Nov 2024 | Subset of the AgentInstruct dataset (~25 samples) designed for Orca-3-Mistral, using raw text publicly available on the web as seed data.                                                                                                                    |
| [tulu3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture)                       | 939k  | AI2                | Nov 2024 | (CC-BY-NC-4.0) SFT mixture used to train the [Tulu 3](https://huggingface.co/collections/allenai/tulu-3-models-673b8e0dc3512e30e7dc54f5). It uses public datasets and new synthetic versions, including persona-based answers for diversity.                 |
| [Open-Platypus](https://huggingface.co/datasets/garage-bAInd/Open-Platypus)                           | 24.9k | Lee et al.         | Sep 2023 | Collection of datasets that were deduplicated using Sentence Transformers (it contains an NC dataset). See [Platypus paper](https://arxiv.org/abs/2308.07317).                                                                                               |
### Math

LLMs often struggle with mathematical reasoning and formal logic, which has led to the creation of specialized datasets. These datasets can include systematic thinking and step-by-step reasoning.

| Dataset                                                                             | #    | Authors       | Date      | Notes                                                                                                                                                                      |
| ----------------------------------------------------------------------------------- | ---- | ------------- | --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2)     | 14M  | Nvidia        | Sep 2024  | Augmented samples from GSM8K and MATH (training set) using Llama-3.1-405B-Instruct.                                                                                        |
| [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT)              | 859k | Jia Li et al. | July 2024 | Data used to win the first progress prize of the AI Math Olympiad. See the tool-integrated reasoning version [here](https://huggingface.co/datasets/AI-MO/NuminaMath-TIR). |
| [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA)                  | 395k | Yu et al.     | Dec 2023  | Bootstrap mathematical questions by rewriting them from multiple perspectives. See [MetaMath paper](https://arxiv.org/abs/2309.12284).                                     |
| [MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct)              | 262k | Yue et al.    | Sep 2023  | Compiled from 13 math rationale datasets, six of which are newly curated, and focuses on chain-of-thought and program-of-thought.                                          |
| [Orca-Math](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k) | 200k | Mitra et al.  | Feb 2024  | Grade school math world problems generated using GPT4-Turbo. See [Orca-Math paper](https://arxiv.org/pdf/2402.14830.pdf).                                                  |
### Code

Code is another challenging domain for LLMs. Code datasets, containing diverse programming language examples, are used to fine-tune LLMs and enhance their ability to understand, generate, and analyze code.

| Dataset                                                                                                                | #     | Authors        | Date     | Notes                                                                                                                                                                                                                                                                                                                                                 |
| ---------------------------------------------------------------------------------------------------------------------- | ----- | -------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [opc-sft-stage2](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2)                                         | 436k  | Huang et al.   | Nov 2024 | Dataset used in OpenCoder's Stage 2, based on four seed datasets. See [OpenCoder paper](https://arxiv.org/abs/2411.04905).                                                                                                                                                                                                                            |
| [CodeFeedback-Filtered-Instruction](https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction)           | 157k  | Zheng et al.   | Feb 2024 | Filtered version of Magicoder-OSS-Instruct, ShareGPT (Python), Magicoder-Evol-Instruct, and Evol-Instruct-Code.                                                                                                                                                                                                                                       |
| [Tested-143k-Python-Alpaca](https://huggingface.co/datasets/Vezora/Tested-143k-Python-Alpaca)                          | 143k  | Vezora         | Mar 2024 | Collection of generated Python code that passed automatic tests to ensure high quality.                                                                                                                                                                                                                                                               |
| [glaive-code-assistant](https://huggingface.co/datasets/glaiveai/glaive-code-assistant)                                | 136k  | Glaive.ai      | Sep 2023 | Synthetic data of problems and solutions with ~60% Python samples. Also see the [v2](https://huggingface.co/datasets/glaiveai/glaive-code-assistant-v2) version.                                                                                                                                                                                      |
| [Magicoder-Evol-Instruct-110K](https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K)                  | 110k  | Wei et al.     | Nov 2023 | A decontaminated version of [evol-codealpaca-v1](https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1). Decontamination is done in the same way as StarCoder ([bigcode decontamination process](https://github.com/bigcode-project/bigcode-dataset/tree/main/decontamination)). See [Magicoder paper](https://arxiv.org/abs/2312.02120). |
| [synthetic_tex_to_sql](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql)                                 | 100k  | Gretel.ai      | Apr 2024 | Synthetic text-to-SQL samples (~23M tokens), covering diverse domains.                                                                                                                                                                                                                                                                                |
| [sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context)                                         | 78.6k | b-mc2          | Apr 2023 | Cleansed and augmented version of the [WikiSQL](https://huggingface.co/datasets/wikisql) and [Spider](https://huggingface.co/datasets/spider) datasets.                                                                                                                                                                                               |
| [Code-Feedback](https://huggingface.co/datasets/m-a-p/Code-Feedback)                                                   | 66.4k | Zheng et al.   | Feb 2024 | Diverse Code Interpreter-like dataset with multi-turn dialogues and interleaved text and code responses. See [OpenCodeInterpreter paper](https://arxiv.org/abs/2402.14658).                                                                                                                                                                           |
| [Open-Critic-GPT](https://huggingface.co/datasets/Vezora/Open-Critic-GPT)                                              | 55.1k | Vezora         | Jul 2024 | Use a local model to create, introduce, and identify bugs in code across multiple programming languages.                                                                                                                                                                                                                                              |
| [self-oss-instruct-sc2-exec-filter-50k](https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k) | 50.7k | Lozhkov et al. | Apr 2024 | Created in three steps with seed functions from TheStack v1, self-instruction with StarCoder2, and self-validation. See the [blog post](https://huggingface.co/blog/sc2-instruct).                                                                                                                                                                    |
### Instruction following

Instruction following corresponds to the ability to properly follow constraints in the user prompt, such as "write only two paragraphs", "write your answer in French", etc. Strong instruction-following capabilities is a must-have for modern LLMs.

| Dataset                                                                                                                        | #     | Authors     | Date     | Notes                                                                                                                                                                                                      |
| ------------------------------------------------------------------------------------------------------------------------------ | ----- | ----------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [AutoIF-instruct-61k-with-funcs](https://huggingface.co/datasets/Post-training-Data-Flywheel/AutoIF-instruct-61k-with-funcs)   | 61.5k | Diao et al. | Oct 2024 | Samples generated with [this code](https://github.com/shizhediao/Post-Training-Data-Flywheel/tree/main/IF-generation) and gpt-4o-mini, based on Qwen's [AutoIF](https://github.com/QwenLM/AutoIF) library. |
| [ifeval-like-data](https://huggingface.co/datasets/argilla/ifeval-like-data)                                                   | 56.3k | Argilla     | Oct 2024 | Only use the "filtered" subset. Samples generated by Qwen2.5-72B and verified with lm-evaluation-harness.                                                                                                  |
| [tulu-3-sft-personas-instruction-following](https://huggingface.co/datasets/allenai/tulu-3-sft-personas-instruction-following) | 30k   | AI2         | Nov 2024 | Synthetic samples created with personas, following the methodology introduced by [Ge et al., 2024](https://arxiv.org/pdf/2406.20094).                                                                      |
### Multilingual

Learning new languages "from scratch" is a pre-training task, but providing multilingual instruction samples is useful to boost performance in the languages of interest.

| Dataset                                                                                                       | #     | Authors                      | Date     | Notes                                                                             |
| ------------------------------------------------------------------------------------------------------------- | ----- | ---------------------------- | -------- | --------------------------------------------------------------------------------- |
| [aya dataset](https://huggingface.co/datasets/CohereForAI/aya_dataset) | 204k | Singh et al. | Feb 2024 | Multilingual instruction fine-tuning dataset curated by an open-science community via Aya Annotation Platform. |
| [M2Lingual](https://huggingface.co/datasets/ServiceNow-AI/M2Lingual)                          | 175K  | ServiceNow AI | June 2024 | Dataset spanning 70+ langauges and 20 NLP tasks generated from GPT-4 using task-based taxonomy guided evolutions. More details in [M2Lingual](https://arxiv.org/abs/2406.16783) paper.|
### Agent & Function calling

Function calling allows large language models (LLMs) to execute predefined functions with parameters inferred from user prompts, rather than generating standard text responses. This enables LLMs to seamlessly integrate with external systems, perform complex operations, and provide more accurate and contextually relevant responses.

| Dataset                                                                                               | #     | Authors         | Date     | Notes                                                                                                                                                                                                                                         |
| ----------------------------------------------------------------------------------------------------- | ----- | --------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)     | 113k  | Sahil Chaudhary | Sep 2023 | High-quality dataset with pairs of instructions and answers in different languages. <br>See [Locutusque/function-calling-chatml](https://huggingface.co/datasets/Locutusque/function-calling-chatml) for a variant without conversation tags. |
| [xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)     | 60k   | Salesforce      | Jun 2024 | Samples created using a data generation pipeline designed to produce verifiable data for function-calling applications                                                                                                                        |
| [Agent-FLAN](https://huggingface.co/datasets/internlm/Agent-FLAN)                                     | 34.4k | internlm        | Mar 2024 | Mix of AgentInstruct, ToolBench, and ShareGPT datasets.                                                                                                                                                                                       |
| [hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) | 11.6k | Nous            | Aug 2024 | Compilation of structured output and function calling data used in the Hermes 2 Pro series of models.                                                                                                                                         |
| [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE)                                           | 11.3k | Liu et al.      | Aug 2024 | Agentic pipeline self-evolution synthesis process to curate a comprehensive API pool                                                                                                                                                                                                                             |
### Real conversations

Real-world conversations provide valuable insights into how people naturally interact with LLMs, helping us identify the most important use cases and understand typical usage patterns.

| Dataset                                                              | #     | Authors     | Date     | Notes                                                                                                                                                  |
| -------------------------------------------------------------------- | ----- | ----------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M)   | 1.04M | Zhao et al. | May 2023 | Real conversations between human users and GPT-3.5/4, including metadata. See the [WildChat paper](https://arxiv.org/abs/2405.01470).                  |
| [lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) | 1M    | LMSYS       | Sep 2023 | Real-world conversations with 25 LLMs, collected from 210K unique IP addresses on the Vicuna demo and Chatbot Arena website from April to August 2023. |
| [oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2)       | 135k  | Köpf et al. | Dec 2023 | Human-generated conversations trees with multiple replies. See [OASST1 paper](https://arxiv.org/abs/2304.07327).                                       |
| [ShareGPT52K](https://huggingface.co/datasets/RyokoAI/ShareGPT52K)   | 90k   | ShareGPT    | Apr 2023 | Conversations scraped via the ShareGPT API before it was shut down. They include both user prompts and responses from GPT-3.5.                         |
| [oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1)       | 84.4k | Köpf et al. | Mar 2023 | Human-generated assistant-style conversation corpus in 35 different languages. See [OASST1 paper](https://arxiv.org/abs/2304.07327).                   |
## ⚖️ Preference alignment

Unlike instruction data, preference datasets consist of chosen and rejected answers. Preference alignment is used to align LLM's answers with human preferences to adopt the desired style and values.

| Dataset                                                                                                                            | #     | Authors          | Date     | Notes                                                                                                                                                                     |
| ---------------------------------------------------------------------------------------------------------------------------------- | ----- | ---------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Skywork-Reward-Preference-80K-v0.2](https://huggingface.co/datasets/Skywork/Skywork-Reward-Preference-80K-v0.2)                   | 77k   | Skywork          | 2024     | Preference pairs compiled from public sources like HelpSteer2, OffsetBias, WildGuard, and Magpie.                                                                         |
| [ultrafeedback-binarized-preferences-cleaned](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned) | 61.1k | Argilla          | 2023     | Decontaminated version of the UltraChat dataset, scored by GPT-4 and binarized into "chosen" and "rejected" answers based on these scores.                                |
| [Infinity-Preference](https://huggingface.co/datasets/BAAI/Infinity-Preference)                                                    | 59k   | BAAI             | Sep 2024 | Adjusts preference attribute weights per task using Infinity-Instruct's labeling system. Each instruction is accompanied by a preference pair sampled from Gemma-2-9B-IT. |
| [Code-Preference-Pairs](https://huggingface.co/datasets/Vezora/Code-Preference-Pairs)                                              | 53k   | Vezora           | Jul 2024 | Pairs of code examples, where the chosen sample is correct and the rejected one contains a bug.                                                                           |
| [orpo-dpo-mix-40k](https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k)                                                      | 44k   | Argilla, Labonne | May 2024 | Combination of the following high-quality DPO datasets, mostly from Argilla.                                                                                              |
| [chatbot_arena_conversations](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations)                                   | 33k   | LMSYS            | Jul 2023 | Cleaned real conversations with pairwise human preferences collected on the [Chatbot Arena](https://lmsys.org/blog/2023-05-03-arena/) from April to June 2023             |
| [tulu-3-pref-personas-instruction-following](https://huggingface.co/datasets/allenai/tulu-3-pref-personas-instruction-following)   | 19.9k | AI2              | Nov 2024 | Instruction following data in the form of chosen and rejected answers to teach the model to follow precise constraints.                                                   |
| [Human-Like-DPO-Dataset](https://huggingface.co/datasets/HumanLLMs/Human-Like-DPO-Dataset)                                         | 10.9k | Weyaxi           | May 2024 | Teach to output more human-like answers instead of the formal slop LLMS usually output.                                                                                   |

## 🔧 Tools

Tools listed in this section can help you evaluate, generate, and explore datasets. Start by aggregating available data from various sources (open-source or not) and applying filters like data deduplication and data quality. If the initial dataset is small or insufficient, consider synthetically generating additional data to fill the gap. Iteratively explore and refine the dataset by assessing model performance, identifying gaps, and collecting or generating data to address those shortcomings.
### Data scraping
* [**Trafilatura**](https://github.com/adbar/trafilatura): Python and command-line tool to gather text and metadata on the web. Used for the creation of [RefinedWeb](https://arxiv.org/abs/2306.01116).
* [**Marker**](https://github.com/VikParuchuri/marker): Quickly convert PDFs to markdown text.

### Data quality evaluation
* **Rule-based filtering**: Remove samples based on a list of unwanted words, like refusals and "As an AI assistant" ([example](https://huggingface.co/datasets/cognitivecomputations/WizardLM_alpaca_evol_instruct_70k_unfiltered/blob/main/wizardlm_clean.py)).
* [**Argilla**](https://argilla.io/): Platform that allows you to manually filter and annotate datasets in a collaborative way.
* [**judges**](https://github.com/quotient-ai/judges): Small library of LLM judges with various classifiers and graders (early development).

### Data generation
* [**Distilabel**](https://github.com/argilla-io/distilabel): General-purpose framework that can generate and augment data (SFT, DPO) with techniques like UltraFeedback and DEITA.
* [**Augmentoolkit**](https://github.com/e-p-armstrong/augmentoolkit): Framework to convert raw text into datasets using open-source and closed-source models.
* [**Data Prep Kit**](https://github.com/IBM/data-prep-kit): Framework for data preparation for both code and language, with modules in Python, Ray, and Spark, and a wide range of scale from laptops to data centers.

### Data exploration
* [**Lilac**](https://github.com/lilacai/lilac): Tool for exploration, curation and quality control of datasets.
* [**Nomic Atlas**](https://github.com/nomic-ai/nomic): Interact with instructed data to find insights and store embeddings.
* [**text-clustering**](https://github.com/huggingface/text-clustering): A framework from Huggingface for clustering textual data.
* [**Autolabel**](https://github.com/refuel-ai/autolabel): Automatically label data using popular language models.

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
