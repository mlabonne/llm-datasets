<div align="center">
<img src="https://i.imgur.com/SekZcgb.png" alt="Image">
  <p>
    𝕏 <a href="https://twitter.com/maximelabonne">Follow me on X</a> • 
    🤗 <a href="https://huggingface.co/mlabonne">Hugging Face</a> • 
    💻 <a href="https://mlabonne.github.io/blog">Blog</a> • 
    📙 <a href="https://packt.link/a/9781836200079">LLM Engineer's Handbook</a>
  </p>
   <p><em>Curated list of datasets and tools for post-training.</em></p>
</div>
<br/>

## 👍 What is a good dataset?

Data is the most valuable asset in LLM development. When building a high-quality dataset, we target the three following characteristics:

* **Accuracy**: Samples should be factually correct and relevant to their corresponding instructions. This can involve using solvers for math and unit tests for code.
* **Diversity**: You want to cover as many use cases as possible to make sure you're never out of distribution. High diversity is essential as it leads to better generalization.
* **Complexity**: Samples should be multi-turn, multilingual, well-written, and include step-by-step reasoning when relevant.

To ensure the quality of a dataset, it is essential to combine various techniques, such as manuals reviews, heuristics like rule-based filtering, and scoring via judge LLMs or reward models.

## 📅 Instruction Datasets

Once a model has been pre-trained on a next-token prediction task, Supervised Fine-Tuning (SFT) is used to turn it into an assistant capable of answering questions and following instructions. During SFT, models learn a chat template and are specialized in one or more domains.

> [!NOTE]
> Unless specified otherwise, all datasets listed here are under permissive licenses (Apache 2.0, MIT, CC-BY-4.0, etc.).

### General-purpose mixtures

General-purpose datasets offer balanced mixtures of different types of data, including chat, code, and math. These datasets can be used to create general-purpose models that can handle various types of queries.

| Dataset                                                                                               | #     | Authors            | Date     | Notes                                                                                                                                                                                                                                                        |
| ----------------------------------------------------------------------------------------------------- | ----- | ------------------ | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2)   | 6.34M  | Nvidia            | Aug 2025 | Large-scale dataset with five target languages (Spanish, French, German, Italian, Japanese) for math, code, general reasoning, and instruction following. Used to train [Nemotron-Nano-9B-v2](https://arxiv.org/abs/2508.14444).                                                                       |
| [smoltalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2)                                  | 3.38M | Hugging Face       | July 2025 | Dataset used to train SmolLM3 models with and without reasoning traces. Includes OpenThoughts3, Tulu 3, and multilingual data. See the [SmolLM3 blog post](https://huggingface.co/blog/smollm3).                              |
| [open-perfectblend](https://huggingface.co/datasets/mlabonne/open-perfectblend)                       | 1.42M | Xu et al., Labonne | Oct 2024 | Open reproduction of the dataset described [in this paper](https://arxiv.org/abs/2409.20370). It's a solid general-purpose instruction dataset with chat, math, code, and instruction-following data.                                                        |
| [orca-agentinstruct-1M-v1](https://huggingface.co/datasets/mlabonne/orca-agentinstruct-1M-v1-cleaned) | 1.05M | Microsoft          | Nov 2024 | Subset of the AgentInstruct dataset (~25 samples) designed for Orca-3-Mistral, using raw text publicly available on the web as seed data.                                                                                                                    |
| [tulu3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture)                       | 939k  | AllenAI                | Nov 2024 | (CC-BY-NC-4.0) SFT mixture used to train the [Tulu 3](https://huggingface.co/collections/allenai/tulu-3-models-673b8e0dc3512e30e7dc54f5). It uses public datasets and new synthetic versions, including persona-based answers for diversity.                 |
| [FuseChat-Mixture](https://huggingface.co/datasets/FuseAI/FuseChat-Mixture)                           | 95k   | Wan et al.         | Feb 2024 | Comprehensive training dataset covering different styles and capabilities, featuring both human-written and model-generated samples. See [FuseChat paper](https://arxiv.org/abs/2402.16107).                                                                 |


### Math

LLMs often struggle with mathematical reasoning and formal logic, which has led to the creation of specialized datasets. These datasets can include systematic thinking and step-by-step reasoning.

| Dataset                                                                             | #    | Authors       | Date      | Notes                                                                                                                                                                      |
| ----------------------------------------------------------------------------------- | ---- | ------------- | --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2)     | 14M  | Nvidia        | Sep 2024  | Augmented samples from GSM8K and MATH (training set) using Llama-3.1-405B-Instruct.                                                                                        |
| [MegaScience](https://huggingface.co/datasets/MegaScience/MegaScience)              | 1.25M | GAIR-NLP | July 2025 | (CC-BY-NC-SA-4.0) High-quality scientific dataset with diverse domains and abaltion studies. See the paper [here](https://arxiv.org/abs/2507.16812). |
| [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT)              | 859k | Jia Li et al. | July 2024 | Data used to win the first progress prize of the AI Math Olympiad. See the tool-integrated reasoning version [here](https://huggingface.co/datasets/AI-MO/NuminaMath-TIR). |
| [Orca-Math](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k) | 200k | Mitra et al.  | Feb 2024  | Grade school math world problems generated using GPT4-Turbo. See [Orca-Math paper](https://arxiv.org/pdf/2402.14830.pdf).                                                  |

### Code

Code is another challenging domain for LLMs. Code datasets, containing diverse programming language examples, are used to fine-tune LLMs and enhance their ability to understand, generate, and analyze code.

| Dataset                                                                                                                | #     | Authors        | Date     | Notes                                                                                                                                                                                                                                                                                                                                                 |
| ---------------------------------------------------------------------------------------------------------------------- | ----- | -------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Ling-Coder-SFT](https://huggingface.co/datasets/inclusionAI/Ling-Coder-SFT)                                         | 4.48M  | InclusionAI   | Mar 2025 | Large-scale coding dataset in English and Chinese with 20 programming languages and various topics. See their [tech report](http://arxiv.org/abs/2503.17793).                                                                                                                                                                                                                            |
| [opc-sft-stage2](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2)                                         | 436k  | Huang et al.   | Nov 2024 | Dataset used in OpenCoder's Stage 2, based on four seed datasets. See [OpenCoder paper](https://arxiv.org/abs/2411.04905).                                                                                                                                                                                                                            |
| [CodeFeedback-Filtered-Instruction](https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction)           | 157k  | Zheng et al.   | Feb 2024 | Filtered version of Magicoder-OSS-Instruct, ShareGPT (Python), Magicoder-Evol-Instruct, and Evol-Instruct-Code.                                                                                                                                                                                                                                       |
| [synthetic_tex_to_sql](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql)                                 | 100k  | Gretel.ai      | Apr 2024 | Synthetic text-to-SQL samples (~23M tokens), covering diverse domains.                                                                                                                                                          
### Instruction following

Instruction following corresponds to the ability to properly follow constraints in the user prompt, such as "write only two paragraphs", "write your answer in French", etc. Strong instruction-following capabilities is a must-have for modern LLMs.

| Dataset                                                                                                                        | #     | Authors     | Date     | Notes                                                                                                                                                                      |
| ------------------------------------------------------------------------------------------------------------------------------ | ----- | ----------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [AutoIF-instruct-61k-with-funcs](https://huggingface.co/datasets/Post-training-Data-Flywheel/AutoIF-instruct-61k-with-funcs)   | 61.5k | Diao et al. | Oct 2024 | Samples generated with [this code](https://github.com/shizhediao/Post-Training-Data-Flywheel/tree/main/IF-generation) and gpt-4o-mini, based on Qwen's [AutoIF](https://github.com/QwenLM/AutoIF) library. |
| [ifeval-like-data](https://huggingface.co/datasets/argilla/ifeval-like-data)                                                   | 56.3k | Argilla     | Oct 2024 | Only use the "filtered" subset. Samples generated by Qwen2.5-72B and verified with lm-evaluation-harness.                                                                  |
| [tulu-3-sft-personas-instruction-following](https://huggingface.co/datasets/allenai/tulu-3-sft-personas-instruction-following) | 30k   | AllenAI         | Nov 2024 | Synthetic samples created with personas, following the methodology introduced by [Ge et al., 2024](https://arxiv.org/pdf/2406.20094).                                      |

### Multilingual

Learning new languages "from scratch" is a pre-training task, but providing multilingual instruction samples is useful to boost performance in the languages of interest.

| Dataset                                                                                                       | #     | Authors                      | Date     | Notes                                                                             |
| ------------------------------------------------------------------------------------------------------------- | ----- | ---------------------------- | -------- | --------------------------------------------------------------------------------- |
| [luth-sft ](https://huggingface.co/datasets/kurakurai/luth-sft)                          | 570K  | kurakurai | August 2025 | French/English dataset with original data and good curation. More details in the [tech report](https://arxiv.org/abs/2510.05846v1).|
| [aya dataset](https://huggingface.co/datasets/CohereForAI/aya_dataset) | 204k | Singh et al. | Feb 2024 | Multilingual instruction fine-tuning dataset curated by an open-science community via Aya Annotation Platform. |
| [M2Lingual](https://huggingface.co/datasets/ServiceNow-AI/M2Lingual)                          | 175K  | ServiceNow AI | June 2024 | Dataset spanning 70+ languages and 20 NLP tasks generated from GPT-4 using task-based taxonomy guided evolutions. More details in [M2Lingual](https://arxiv.org/abs/2406.16783) paper.|

### Agent & Function calling

Function calling allows large language models (LLMs) to execute predefined functions with parameters inferred from user prompts, rather than generating standard text responses. This enables LLMs to seamlessly integrate with external systems, perform complex operations, and provide more accurate and contextually relevant responses.

| Dataset                                                                                               | #     | Authors         | Date     | Notes                                                                                                                                                                                                                                         |
| ----------------------------------------------------------------------------------------------------- | ----- | --------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)     | 60k   | Salesforce      | Jun 2024 | Samples created using a data generation pipeline designed to produce verifiable data for function-calling applications                                                                                                                        |
| [FunReason-MT](https://huggingface.co/datasets/Bingguang/FunReason-MT)                                | 17k   | Hao et al.      | Oct 2025 | Multi-turn function calling dataset with complex trajectories requiring environment-API graph interactions and chain-of-thought reasoning. See [FunReason-MT paper](https://arxiv.org/abs/2510.24645).                                        |
| [hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) | 11.6k | Nous            | Aug 2024 | Compilation of structured output and function calling data used in the Hermes 2 Pro series of models.                                                                                                                                         |
| [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE)                                           | 11.3k | Liu et al.      | Aug 2024 | Agentic pipeline self-evolution synthesis process to curate a comprehensive API pool                                                                                                                                                          |
| [APIGen-MT-5k](https://huggingface.co/datasets/Salesforce/APIGen-MT-5k)                               | 5k    | Salesforce      | Apr 2025 | (CC-BY-NC-4.0) Multi-turn agentic trajectories generated via simulated agent-human interplay with verified task blueprints. See [APIGen-MT paper](https://arxiv.org/abs/2504.03601).                            |


### Real conversations

Real-world conversations provide valuable insights into how people naturally interact with LLMs, helping us identify the most important use cases and understand typical usage patterns.

| Dataset                                                              | #     | Authors     | Date     | Notes                                                                                                                                                  |
| -------------------------------------------------------------------- | ----- | ----------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [WildChat-4.8M](https://huggingface.co/datasets/allenai/WildChat-4.8M) | 3.2M | Allen AI | Aug 2025 | Non-toxic conversations between human users and ChatGPT, filtered using OpenAI Moderation API. See [WildChat paper](https://arxiv.org/abs/2405.01470). |
| [lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) | 1M    | LMSYS       | Sep 2023 | Real-world conversations with 25 LLMs, collected from 210K unique IP addresses on the Vicuna demo and Chatbot Arena website from April to August 2023. |
| [arena-human-preference-100k](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-100k)           | 110k  | LMSYS             | Feb 2025 | Human preference evaluations collected from Chatbot Arena between June-August 2024. Used in [Arena Explorer](https://arxiv.org/abs/2403.04132) for conversation analysis and categorization. Includes precomputed embeddings.                                                                      |

## ⚖️ Preference dataset

Unlike instruction data, preference datasets consist of chosen and rejected answers. Preference alignment is used to align LLM's answers with human preferences to adopt the desired style and values.

| Dataset                                                                                                                            | #     | Authors          | Date     | Notes                                                                                                                                                                     |
| ---------------------------------------------------------------------------------------------------------------------------------- | ----- | ---------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Skywork-Reward-Preference-80K-v0.2](https://huggingface.co/datasets/Skywork/Skywork-Reward-Preference-80K-v0.2)                   | 77k   | Skywork          | 2024     | Preference pairs compiled from public sources like HelpSteer2, OffsetBias, WildGuard, and Magpie.                                                                         |
| [ultrafeedback-binarized-preferences-cleaned](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned) | 61.1k | Argilla          | 2023     | Decontaminated version of the UltraChat dataset, scored by GPT-4 and binarized into "chosen" and "rejected" answers based on these scores.                                |
| [Infinity-Preference](https://huggingface.co/datasets/BAAI/Infinity-Preference)                                                    | 59k   | BAAI             | Sep 2024 | Adjusts preference attribute weights per task using Infinity-Instruct's labeling system. Each instruction is accompanied by a preference pair sampled from Gemma-2-9B-IT. |
| [Code-Preference-Pairs](https://huggingface.co/datasets/Vezora/Code-Preference-Pairs)                                              | 53k   | Vezora           | Jul 2024 | Pairs of code examples, where the chosen sample is correct and the rejected one contains a bug.                                                                           |
| [orpo-dpo-mix-40k](https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k)                                                      | 44k   | Argilla, Labonne | May 2024 | Combination of the following high-quality DPO datasets, mostly from Argilla.                                                                                              |
| [HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3)                                                  | 40.5k | Wang et al.       | Oct 2024 | Multi-attribute helpfulness dataset with 40,476 preference samples and 40,821 feedback samples across General, STEM, Code, and Multilingual domains (14 languages). See [HelpSteer3 paper](https://arxiv.org/abs/2505.11475).                                                                     |
| [chatbot_arena_conversations](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations)                                   | 33k   | LMSYS            | Jul 2023 | Cleaned real conversations with pairwise human preferences collected on the [Chatbot Arena](https://lmsys.org/blog/2023-05-03-arena/) from April to June 2023             |
| [FalseReject](https://huggingface.co/datasets/AmazonScience/FalseReject)                                        | 28.8k | Amazon Science    | May 2025 | (CC-BY-NC-4.0) Dataset for mitigating over-refusal behavior in LLMs across 44 safety-related categories. Contains adversarially generated but benign prompts with context-aware responses. See [FalseReject paper](https://arxiv.org/abs/2505.08054).                                             |
| [tulu-3-pref-personas-instruction-following](https://huggingface.co/datasets/allenai/tulu-3-pref-personas-instruction-following)   | 19.9k | AllenAI              | Nov 2024 | Instruction following data in the form of chosen and rejected answers to teach the model to follow precise constraints.                                                   |
| [Human-Like-DPO-Dataset](https://huggingface.co/datasets/HumanLLMs/Human-Like-DPO-Dataset)                                         | 10.9k | Weyaxi           | May 2024 | Teach to output more human-like answers instead of the formal slop LLMS usually output.                                                                                   |

## 🔧 Tools

Tools listed in this section can help you evaluate, generate, and explore datasets. Start by aggregating available data from various sources (open-source or not) and applying filters like data deduplication and data quality. If the initial dataset is small or insufficient, consider synthetically generating additional data to fill the gap. Iteratively explore and refine the dataset by assessing model performance, identifying gaps, and collecting or generating data to address those shortcomings.

### Data scraping

* [**Trafilatura**](https://github.com/adbar/trafilatura): Python and command-line tool to gather text and metadata on the web. Used for the creation of [RefinedWeb](https://arxiv.org/abs/2306.01116).
* [**Marker**](https://github.com/VikParuchuri/marker): Quickly convert PDFs to markdown text.

### Data filtering

* **Rule-based filtering**: Remove samples based on a list of unwanted words, like refusals and "As an AI assistant" ([example](https://huggingface.co/datasets/cognitivecomputations/WizardLM_alpaca_evol_instruct_70k_unfiltered/blob/main/wizardlm_clean.py)).
* [**SemHash**](https://github.com/MinishLab/semhash): Fuzzy deduplication based on fast embedding generation with a distilled model.
* [**Argilla**](https://argilla.io/): Platform that allows you to manually filter and annotate datasets in a collaborative way.
* [**judges**](https://github.com/quotient-ai/judges): Small library of LLM judges with various classifiers and graders (early development).

### Data generation

* [**Curator**](https://github.com/bespokelabsai/curator/): Synthetic data generation tool that makes it easy to build pipelines around LLMs, use batching, and view data in progress.
* [**Distilabel**](https://github.com/argilla-io/distilabel): General-purpose framework that can generate and augment data (SFT, DPO) with techniques like UltraFeedback and DEITA.
* [**Augmentoolkit**](https://github.com/e-p-armstrong/augmentoolkit): Framework to convert raw text into datasets using open-source and closed-source models.
* [**Data Prep Kit**](https://github.com/IBM/data-prep-kit): Framework for data preparation for both code and language, with modules in Python, Ray, and Spark, and a wide range of scale from laptops to data centers.

### Data exploration

* [**Lilac**](https://www.lilacml.com/): Tool for exploration, curation, and quality control of datasets.
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
