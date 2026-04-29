<div align="center">
<img src="https://i.imgur.com/SekZcgb.png" alt="Image">
  <p>
    𝕏 <a href="https://twitter.com/maximelabonne">Follow me on X</a> • 
    🤗 <a href="https://huggingface.co/mlabonne">Hugging Face</a> • 
    💻 <a href="https://maximelabonne.substack.com/">Blog</a> • 
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

### General

General-purpose datasets offer balanced mixtures of different types of data, including chat, code, and math. These datasets can be used to create general-purpose models that can handle various types of queries.

| Dataset | # | Thinking | Notes |
| ------- | - | -------- | ----- |
| [Nemotron-Cascade-2-SFT-Data](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data) (Mar 2026) | 15.87M | Mixed | (NVIDIA Open Model License) Large-scale SFT mixture used to train [Nemotron-Cascade-2-30B-A3B](https://huggingface.co/nvidia/Nemotron-Cascade-2-30B-A3B), covering math, science, chat, instruction following, coding agents, and SWE. Responses generated from DeepSeek-V3.2, GPT-OSS-120B, and Qwen3. See also [Nemotron-Cascade-2-RL-data](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-RL-data). |
| [SYNTHETIC-2-SFT-verified](https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-2-SFT-verified) (Jun 2025) | 4M | Yes | Large-scale reasoning dataset with verified traces from DeepSeek-R1-0528, spanning math, coding, puzzles, and instruction following. See [SYNTHETIC-2 release post](https://www.primeintellect.ai/blog/synthetic-2-release). |
| [Dolci-Instruct-SFT](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT) (Dec 2025) | 2.15M | No | (CC-BY-NC-4.0) Large-scale instruction following mixture combining ~800K samples from public datasets with ~1.3M new synthetic samples. Covers math, science, coding, safety, and 70+ languages. See also [Dolci-Think-SFT-7B](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-7B) for the reasoning variant. |
| [open-perfectblend](https://huggingface.co/datasets/mlabonne/open-perfectblend) (Oct 2024) | 1.42M | No | Open reproduction of the dataset described [in this paper](https://arxiv.org/abs/2409.20370). It's a baseline instruction dataset with chat, math, code, and instruction following data. |
| [orca-agentinstruct-1M-v1](https://huggingface.co/datasets/mlabonne/orca-agentinstruct-1M-v1-cleaned) (Nov 2024) | 1.05M | No | Subset of the AgentInstruct dataset (~25 samples) designed for Orca-3-Mistral, using raw text publicly available on the web as seed data. |
| [KIMI-K2.5-1000000x](https://huggingface.co/datasets/ianncity/KIMI-K2.5-1000000x) (Apr 2026) | 693k | Yes | ~5B tokens of reasoning traces distilled from Kimi K2.5 in high-reasoning mode. 50% code, 20% science, 15% math, plus logic, creative writing, and multilingual STEM. See [Kimi K2.5 report](https://github.com/MoonshotAI/Kimi-k1.5). |


### Math

LLMs often struggle with mathematical reasoning and formal logic, which has led to the creation of specialized datasets. These datasets can include systematic thinking and step-by-step reasoning.

| Dataset | # | Thinking | Notes |
| ------- | - | -------- | ----- |
| [MathX-5M](https://huggingface.co/datasets/Modotte/MathX-5M) (Feb 2026) | 5.05M | Yes | High-quality, synthetically curated and meticulously filtered dataset for advanced mathematical reasoning. See the extended [MathX-20M](https://huggingface.co/datasets/Modotte/MathX-20M) version. |
| [Nemotron-SFT-Math-v3](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Math-v3) (Mar 2026) | 1.24M | Yes | (NVIDIA Open Model License) Mathematical reasoning and problem-solving dataset. Component of the [Nemotron Post-Training v3](https://huggingface.co/collections/nvidia/nemotron-post-training-v3) collection. |
| [Nemotron-Math-Proofs-v1](https://huggingface.co/datasets/nvidia/Nemotron-Math-Proofs-v1) (Dec 2025) | 925k | Yes | (CC-BY-SA-4.0) ~580k natural language proof problems with ~550k Lean 4 formalizations and ~900k verified proof trajectories. Sourced from AoPS, Math StackExchange, and MathOverflow. |
| [OpenThoughts3-1.2M](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) (Jun 2025) | 1.2M | Yes | Mixture with 850k math, 250k code, 100k science samples, annotated with QwQ-32B. |
| [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) (Jul 2024) | 859k | Yes | Data used to win the first progress prize of the AI Math Olympiad. See the tool-integrated reasoning version [here](https://huggingface.co/datasets/AI-MO/NuminaMath-TIR). |
| [AM-Thinking-v1-Distilled (Math)](https://huggingface.co/datasets/a-m-team/AM-Thinking-v1-Distilled/blob/main/math.jsonl) (May 2025) | 558k | Yes | Math dataset with verified responses distilled from AM-Thinking-v1 and Qwen3-235B-A22B. See the paper [here](https://arxiv.org/abs/2505.14464). |

### Science

Science datasets cover domains like physics, chemistry, and biology with reasoning-heavy questions, often in GPQA-style multiple-choice or long-form formats.

| Dataset | # | Thinking | Notes |
| ------- | - | -------- | ----- |
| [MegaScience](https://huggingface.co/datasets/MegaScience/MegaScience) (Jul 2025) | 1.25M | Yes | (CC-BY-NC-SA-4.0) High-quality scientific dataset with diverse domains and ablation studies. See the paper [here](https://arxiv.org/abs/2507.16812). |
| [Nemotron-Science-v1](https://huggingface.co/datasets/nvidia/Nemotron-Science-v1) (Dec 2025) | 226k | Yes | (CC-BY-4.0) 174k GPQA-style multiple-choice questions and 52k chemistry questions, synthetically generated to enhance scientific reasoning. |

### Code

Code is another challenging domain for LLMs. Code datasets, containing diverse programming language examples, are used to fine-tune LLMs and enhance their ability to understand, generate, and analyze code.

| Dataset | # | Thinking | Notes |
| ------- | - | -------- | ----- |
| [CodeX-7M-Non-Thinking](https://huggingface.co/datasets/Modotte/CodeX-7M-Non-Thinking) (Feb 2026) | 7.36M | No | Large-scale curated coding dataset with direct solutions. Covers Python, Java, C++, JavaScript, and more across algorithms, data structures, ML, and competitive programming. 40% advanced complexity. See also [CodeX-2M-Thinking](https://huggingface.co/datasets/Modotte/CodeX-2M-Thinking) for the reasoning-trace variant. |
| [Ling-Coder-SFT](https://huggingface.co/datasets/inclusionAI/Ling-Coder-SFT) (Mar 2025) | 4.48M | No | Large-scale coding dataset in English and Chinese with 20 programming languages and various topics. Direct-answer responses without reasoning traces. See their [tech report](http://arxiv.org/abs/2503.17793). |
| [Nemotron-SFT-Competitive-Programming-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Competitive-Programming-v2) (Mar 2026) | 845k | Yes | (NVIDIA Open Model License) Competitive programming instruction data with `reasoning_content` traces, from the [Nemotron Post-Training v3](https://huggingface.co/collections/nvidia/nemotron-post-training-v3) collection. |
| [Nemotron-SFT-OpenCode-v1](https://huggingface.co/datasets/nvidia/Nemotron-SFT-OpenCode-v1) (Mar 2026) | 459k | No | (NVIDIA Open Model License) Open-source agentic code instruction dataset (OpenCode CLI tool calling) from the Nemotron Post-Training v3 collection. |
| [Nemotron-SFT-SWE-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-SWE-v2) (Mar 2026) | 256k | Yes | (NVIDIA Open Model License) Software engineering trajectories generated with DeepSeek-R1-0528, from the Nemotron Post-Training v3 collection. |
| [rStar-Coder](https://huggingface.co/datasets/microsoft/rStar-Coder) (May 2025) | 1M | Yes | Large-scale competitive code problem dataset, targeting LiveCodeBench, HumanEval, and MBPP. See the paper [here](https://arxiv.org/abs/2505.21297). |
| [opc-sft-stage2](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2) (Nov 2024) | 436k | No | Dataset used in OpenCoder's Stage 2, based on four seed datasets. See [OpenCoder paper](https://arxiv.org/abs/2411.04905). |
| [AM-Thinking-v1-Distilled (Code)](https://huggingface.co/datasets/a-m-team/AM-Thinking-v1-Distilled/blob/main/code.jsonl) (May 2025) | 324k | Yes | Code dataset with verified responses distilled from AM-Thinking-v1 and Qwen3-235B-A22B. See the paper [here](https://arxiv.org/abs/2505.14464). |
| [CodeFeedback-Filtered-Instruction](https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction) (Feb 2024) | 157k | No | Filtered version of Magicoder-OSS-Instruct, ShareGPT (Python), Magicoder-Evol-Instruct, and Evol-Instruct-Code. |
| [open-r1/codeforces](https://huggingface.co/datasets/open-r1/codeforces) (Feb 2025) | 29k | Yes | 10K+ Codeforces competitive programming problems covering contests through 2025, paired with ~100K DeepSeek-R1 reasoning solutions. CC-BY-4.0. Released as part of [Open-R1](https://github.com/huggingface/open-r1). |
| [synthetic_tex_to_sql](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql) (Apr 2024) | 100k | No | Synthetic text-to-SQL samples (~23M tokens), covering diverse domains. |

### Instruction following

Instruction following corresponds to the ability to properly follow constraints in the user prompt, such as "write only two paragraphs", "write your answer in French", etc. Strong instruction following capabilities is a must-have for modern LLMs.

| Dataset | # | Thinking | Notes |
| ------- | - | -------- | ----- |
| [Nemotron-SFT-Instruction-Following-Chat-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Instruction-Following-Chat-v2) (Mar 2026) | 2M | Mixed | (NVIDIA Open Model License) Multi-turn chat instruction following dataset with `reasoning_on` and `reasoning_off` splits, from the [Nemotron Post-Training v3](https://huggingface.co/collections/nvidia/nemotron-post-training-v3) collection. |
| [AutoIF-instruct-61k-with-funcs](https://huggingface.co/datasets/Post-training-Data-Flywheel/AutoIF-instruct-61k-with-funcs) (Oct 2024) | 61.5k | No | Samples generated with [this code](https://github.com/shizhediao/Post-Training-Data-Flywheel/tree/main/IF-generation) and gpt-4o-mini, based on Qwen's [AutoIF](https://github.com/QwenLM/AutoIF) library. |
| [ifeval-like-data](https://huggingface.co/datasets/argilla/ifeval-like-data) (Oct 2024) | 56.3k | No | Only use the "filtered" subset. Samples generated by Qwen2.5-72B and verified with lm-evaluation-harness. |
| [tulu-3-sft-personas-instruction-following](https://huggingface.co/datasets/allenai/tulu-3-sft-personas-instruction-following) (Nov 2024) | 30k | No | Synthetic samples created with personas, following the methodology introduced by [Ge et al., 2024](https://arxiv.org/pdf/2406.20094). |

### Multilingual

Learning new languages "from scratch" is a pre-training task, but providing multilingual instruction samples is useful to boost performance in the languages of interest.

| Dataset | # | Thinking | Notes |
| ------- | - | -------- | ----- |
| [Nemotron-SFT-Multilingual-v1](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Multilingual-v1) (Mar 2026) | 3.07M | Yes | (NVIDIA Open Model License) Multilingual reasoning data translated from Nemotron-Math/Code/Science (prompts and answers in target language, reasoning trace in English). From the [Nemotron Post-Training v3](https://huggingface.co/collections/nvidia/nemotron-post-training-v3) collection. |
| [luth-sft](https://huggingface.co/datasets/kurakurai/luth-sft) (Aug 2025) | 570K | No | French/English dataset with original data and good curation. More details in the [tech report](https://arxiv.org/abs/2510.05846v1). |
| [aya dataset](https://huggingface.co/datasets/CohereForAI/aya_dataset) (Feb 2024) | 204k | No | Multilingual instruction fine-tuning dataset curated by an open-science community via Aya Annotation Platform. |
| [M2Lingual](https://huggingface.co/datasets/ServiceNow-AI/M2Lingual) (Jun 2024) | 175K | No | Dataset spanning 70+ languages and 20 NLP tasks generated from GPT-4 using task-based taxonomy guided evolutions. More details in [M2Lingual](https://arxiv.org/abs/2406.16783) paper. |

### Agent & Function calling

Function calling allows large language models (LLMs) to execute predefined functions with parameters inferred from user prompts, rather than generating standard text responses. This enables LLMs to seamlessly integrate with external systems, perform complex operations, and provide more accurate and contextually relevant responses.

| Dataset | # | Thinking | Notes |
| ------- | - | -------- | ----- |
| [AgentTrove](https://huggingface.co/datasets/open-thoughts/AgentTrove) (Apr 2026) | 1.7M | No | Samples drawn from 219 source datasets spanning code repair, shell scripting, mathematical problem-solving, competitive programming, and general computer-use tasks. |
| [Nemotron-SFT-Agentic-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2) (Mar 2026) | 992k | Mixed | (NVIDIA Open Model License) Agentic task instruction data with both thinking and non-thinking trajectories (toggled via `chat_template_kwargs.thinking`), from the [Nemotron Post-Training v3](https://huggingface.co/collections/nvidia/nemotron-post-training-v3) collection. |
| [ToolMind](https://huggingface.co/datasets/Nanbeige/ToolMind) (Nov 2025) | 369k | Yes | Large-scale reasoning-enhanced tool-use dataset with 20K+ tools, using a multi-agent framework simulating user-assistant-tool interactions with fine-grained quality filtering. See [ToolMind paper](https://arxiv.org/abs/2511.15718). |
| [xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) (Jun 2024) | 60k | No | Samples created using a data generation pipeline designed to produce verifiable data for function-calling applications. |
| [FunReason-MT](https://huggingface.co/datasets/Bingguang/FunReason-MT) (Oct 2025) | 17k | Yes | Multi-turn function calling dataset with complex trajectories requiring environment-API graph interactions and chain-of-thought reasoning. See [FunReason-MT paper](https://arxiv.org/abs/2510.24645). |
| [hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) (Aug 2024) | 11.6k | No | Compilation of structured output and function calling data used in the Hermes 2 Pro series of models. |
| [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) (Aug 2024) | 11.3k | No | Agentic pipeline self-evolution synthesis process to curate a comprehensive API pool. |
| [APIGen-MT-5k](https://huggingface.co/datasets/Salesforce/APIGen-MT-5k) (Apr 2025) | 5k | No | (CC-BY-NC-4.0) Multi-turn agentic trajectories generated via simulated agent-human interplay with verified task blueprints. See [APIGen-MT paper](https://arxiv.org/abs/2504.03601). |


### Real conversations

Real-world conversations provide valuable insights into how people naturally interact with LLMs, helping us identify the most important use cases and understand typical usage patterns.

| Dataset | # | Thinking | Notes |
| ------- | - | -------- | ----- |
| [WildChat-4.8M](https://huggingface.co/datasets/allenai/WildChat-4.8M) (Aug 2025) | 3.2M | No | Non-toxic conversations between human users and ChatGPT, filtered using OpenAI Moderation API. See [WildChat paper](https://arxiv.org/abs/2405.01470). |
| [lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) (Sep 2023) | 1M | No | Real-world conversations with 25 LLMs, collected from 210K unique IP addresses on the Vicuna demo and Chatbot Arena website from April to August 2023. |
| [arena-human-preference-140k](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k) (Jul 2025) | 136k | No | Human preference evaluations from Chatbot Arena (April–July 2025). Updated and extended version of the 100k release. Includes precomputed embeddings and CC-BY-4.0 license. |

## ⚖️ Preference dataset

Unlike instruction data, preference datasets consist of chosen and rejected answers. Preference alignment is used to align LLM's answers with human preferences to adopt the desired style and values.

| Dataset | # | Thinking | Notes |
| ------- | - | -------- | ----- |
| [Skywork-Reward-Preference-80K-v0.2](https://huggingface.co/datasets/Skywork/Skywork-Reward-Preference-80K-v0.2) (2024) | 77k | No | Preference pairs compiled from public sources like HelpSteer2, OffsetBias, WildGuard, and Magpie. |
| [ultrafeedback-binarized-preferences-cleaned](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned) (2023) | 61.1k | No | Decontaminated version of the UltraChat dataset, scored by GPT-4 and binarized into "chosen" and "rejected" answers based on these scores. |
| [Infinity-Preference](https://huggingface.co/datasets/BAAI/Infinity-Preference) (Sep 2024) | 59k | No | Adjusts preference attribute weights per task using Infinity-Instruct's labeling system. Each instruction is accompanied by a preference pair sampled from Gemma-2-9B-IT. |
| [Code-Preference-Pairs](https://huggingface.co/datasets/Vezora/Code-Preference-Pairs) (Jul 2024) | 53k | No | Pairs of code examples, where the chosen sample is correct and the rejected one contains a bug. |
| [orpo-dpo-mix-40k](https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k) (May 2024) | 44k | No | Combination of the following high-quality DPO datasets, mostly from Argilla. |
| [HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3) (Oct 2024) | 40.5k | No | Multi-attribute helpfulness dataset with 40,476 preference samples and 40,821 feedback samples across General, STEM, Code, and Multilingual domains (14 languages). See [HelpSteer3 paper](https://arxiv.org/abs/2505.11475). |
| [chatbot_arena_conversations](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations) (Jul 2023) | 33k | No | Cleaned real conversations with pairwise human preferences collected on the [Chatbot Arena](https://lmsys.org/blog/2023-05-03-arena/) from April to June 2023. |
| [FalseReject](https://huggingface.co/datasets/AmazonScience/FalseReject) (May 2025) | 28.8k | No | (CC-BY-NC-4.0) Dataset for mitigating over-refusal behavior in LLMs across 44 safety-related categories. Contains adversarially generated but benign prompts with context-aware responses. See [FalseReject paper](https://arxiv.org/abs/2505.08054). |
| [tulu-3-pref-personas-instruction-following](https://huggingface.co/datasets/allenai/tulu-3-pref-personas-instruction-following) (Nov 2024) | 19.9k | No | Instruction following data in the form of chosen and rejected answers to teach the model to follow precise constraints. |
| [Human-Like-DPO-Dataset](https://huggingface.co/datasets/HumanLLMs/Human-Like-DPO-Dataset) (May 2024) | 10.9k | No | Teach to output more human-like answers instead of the formal slop LLMS usually output. |

## 🔧 Tools

Tools listed in this section can help you evaluate, generate, and explore datasets. Start by aggregating available data from various sources (open-source or not) and applying filters like data deduplication and data quality. If the initial dataset is small or insufficient, consider synthetically generating additional data to fill the gap. Iteratively explore and refine the dataset by assessing model performance, identifying gaps, and collecting or generating data to address those shortcomings.

### Data scraping

* [**Trafilatura**](https://github.com/adbar/trafilatura): Python and command-line tool to gather text and metadata on the web. Used for the creation of [RefinedWeb](https://arxiv.org/abs/2306.01116).
* [**Marker**](https://github.com/VikParuchuri/marker): Quickly convert PDFs to markdown text.

### Data filtering

* **Rule-based filtering**: Remove samples based on a list of unwanted words, like refusals and "As an AI assistant" ([example](https://huggingface.co/datasets/cognitivecomputations/WizardLM_alpaca_evol_instruct_70k_unfiltered/blob/main/wizardlm_clean.py)).
* [**NeMo Curator**](https://github.com/NVIDIA/NeMo-Curator): NVIDIA's GPU-accelerated toolkit for large-scale data curation, with exact/fuzzy/semantic deduplication, 30+ heuristic filters, and quality/safety classifiers. Powers the Nemotron-CC pipeline.
* [**DataTrove**](https://github.com/huggingface/datatrove): Hugging Face's library for large-scale text processing with platform-agnostic pipeline blocks (filters, dedup, readers/writers) that run locally or on Slurm without code changes.
* [**Dolma**](https://github.com/allenai/dolma): AllenAI's high-performance toolkit with ready-to-use taggers, fast deduplication, and cloud support. Built to curate the 3T-token Dolma corpus for OLMo.
* [**SemHash**](https://github.com/MinishLab/semhash): Fuzzy deduplication based on fast embedding generation with a distilled model.
* [**Argilla**](https://argilla.io/): Platform that allows you to manually filter and annotate datasets in a collaborative way.

### Data generation

* [**Curator**](https://github.com/bespokelabsai/curator/): Synthetic data generation tool that makes it easy to build pipelines around LLMs, use batching, and view data in progress.
* [**Distilabel**](https://github.com/argilla-io/distilabel): General-purpose framework that can generate and augment data (SFT, DPO) with techniques like UltraFeedback and DEITA.
* [**Augmentoolkit**](https://github.com/e-p-armstrong/augmentoolkit): Framework to convert raw text into datasets using open-source and closed-source models.
* [**Data Prep Kit**](https://github.com/IBM/data-prep-kit): Framework for data preparation for both code and language, with modules in Python, Ray, and Spark, and a wide range of scale from laptops to data centers.

### Data exploration

* [**Nomic Atlas**](https://github.com/nomic-ai/nomic): Interact with instructed data to find insights and store embeddings.

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
