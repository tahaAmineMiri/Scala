Welcome to LLM evaluation!

This notes will quickly get you started on understanding the essence of LLM evaluation in no time :bulb:



# What are LLM Evaluation Metrics?

![image](https://hackmd.io/_uploads/Syiz5aQ2p.png)


# Different Ways to Compute Metric Scores

![image](https://hackmd.io/_uploads/S10KqaQ2a.png)


==**Statistical Scorers**==

statistical scoring methods in my opinion are non essential, This is because statistical methods performs poorly whenever reasoning is required, making it too inaccurate as a scorer for most LLM evaluation criteria.

1. **The BLEU** **(BiLingual Evaluation Understudy)** scorer evaluates the output of your LLM application against annotated ground truths (or, expected outputs). It calculates the precision for each matching n-gram (n consecutive words) between an LLM output and expected output to calculate their geometric mean and applies a brevity penalty if needed.
1. **The ROUGE** **(Recall-Oriented Understudy for Gisting Evaluation)** scorer is s primarily used for evaluating text summaries from NLP models, and calculates recall by comparing the overlap of n-grams between LLM outputs and expected outputs. It determines the proportion (0–1) of n-grams in the reference that are present in the LLM output.
1. **The METEOR** **(Metric for Evaluation of Translation with Explicit Ordering)** scorer is more comprehensive since it calculates scores by assessing both precision (n-gram matches) and recall (n-gram overlaps), adjusted for word order differences between LLM outputs and expected outputs. It also leverages external linguistic databases like WordNet to account for synonyms. The final score is the harmonic mean of precision and recall, with a penalty for ordering discrepancies.
1. **Levenshtein distance** **(or edit distance, you probably recognize this as a LeetCode hard DP problem)** scorer calculates the minimum number of single-character edits (insertions, deletions, or substitutions) required to change one word or text string into another, which can be useful for evaluating spelling corrections, or other tasks where the precise alignment of characters is critical.

:::danger
:radioactive_sign: Since purely statistical scorers hardly not take any semantics into account and have extremely limited reasoning capabilities, they are not accurate enough for evaluating LLM outputs that are often long and complex.
:::

==**Model-Based Scorers**==

Scorers that are purely statistical are reliable but inaccurate, as they struggle to take semantics into account. In this section, it is more of the opposite — scorers that purely rely on NLP models are comparably more accurate, but are also more unreliable due to their probabilistic nature.

* The **NLI** scorer, which uses Natural Language Inference models (which is a type of NLP classification model) to classify whether an LLM output is logically consistent (entailment), contradictory, or unrelated (neutral) with respect to a given reference text. The score typically ranges between entailment (with a value of 1) and contradiction (with a value of 0), providing a measure of logical coherence.
* The **BLEURT (Bilingual Evaluation Understudy with Representations from Transformers)** scorer, which uses pre-trained models like BERT to score LLM outputs on some expected outputs.

:::danger
:radioactive_sign: Apart from inconsistent scores, the reality is there are several shortcomings of these approaches. For example, NLI scorers can also struggle with accuracy when processing long texts, while BLEURT are limited by the quality and representativeness of its training data.
:::



* **G-Eval** is a framework from a paper titled  [“NLG Evaluation using GPT-4 with Better Human Alignment”](https://arxiv.org/pdf/2303.16634.pdf) that uses LLMs to evaluate LLM outputs.

![image](https://hackmd.io/_uploads/B1Dzp6m3p.png)

Let’s run through the G-Eval **algorithm** using this example. First, to generate evaluation steps:

1. Introduce an evaluation task to the LLM of your choice (eg. rate this output from 1–5 based on coherence).
1. Give a definition for your criteria (eg. “Coherence — the collective quality of all sentences in the actual output”).
1. Create a prompt by concatenating the evaluation steps with all the arguments listed in your evaluation steps (eg., if you’re looking to evaluate coherence for an LLM output, the LLM output would be a required argument).
1. At the end of the prompt, ask it to generate a score between 1–5, where 5 is better than 1.

it first generates a series of evaluation steps using chain of thoughts (CoTs) before using the generated steps to determine the final score via a form-filling paradigm (this is just a fancy way of saying G-Eval requires several pieces of information to work). 


:::danger
:radioactive_sign: **Note**:  that in the original G-Eval paper, the authors only used GPT-3.5 and GPT-4 for experiments, I would highly recommend you stick with these models.
:::


:::info
:information_source: 
G-Eval is great because as an LLM-Eval it is able to take the full semantics of LLM outputs into account, making it much more accurate.
:::


:::warning
:warning: Although G-Eval correlates much more with human judgment when compared to its counterparts, it can still be unreliable, as asking an LLM to come up with a score is indisputably arbitrary.
:::


* **Prometheus** Prometheus is a fully open-source LLM that is comparable to GPT-4’s evaluation capabilities when the appropriate reference materials (reference answer, score rubric) are provided. It is also use case agnostic, similar to G-Eval. Prometheus is a language model using [Llama-2-Chat](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) as a base model and fine-tuned on 100K feedback (generated by GPT-4) within the [Feedback Collection](https://huggingface.co/datasets/kaist-ai/Feedback-Collection)

[Here are the results from the parer](https://arxiv.org/pdf/2310.08491.pdf)


![image](https://hackmd.io/_uploads/r1_5fA7n6.png)

Prometheus follows the same principles as G-Eval. However, there are several differences:

1. While G-Eval is a framework that uses GPT-3.5/4, Prometheus is an LLM fine-tuned for evaluation.
1. While G-Eval generates the score rubric/evaluation steps via CoTs, the score rubric for Prometheus is provided in the prompt instead.
1. Prometheus requires reference/example evaluation results.

:::danger
:radioactive_sign: Prometheus was designed to make evaluation open source instead of depending on proprietary models such as OpenAI’s GPTs
:::

==**Combining Statistical and Model-Based Scorers**==

* The **BERTScore** scorer, which relies on pre-trained language models like BERT and computes the cosine similarity between the contextual embeddings of words in the reference and the generated texts. These similarities are then aggregated to produce a final score. A higher BERTScore indicates a greater degree of semantic overlap between the LLM output and the reference text.
* The **MoverScore** scorer, which first uses embedding models, specifically pre-trained language models like BERT to obtain deeply contextualized word embeddings for both the reference text and the generated text before using something called the Earth Mover’s Distance (EMD) to compute the minimal cost that must be paid to transform the distribution of words in an LLM output to the distribution of words in the reference text.


:::warning
:warning: Both the BERTScore and MoverScore scorer is vulnerable to contextual awareness and bias due to their reliance on contextual embeddings from pre-trained models like BERT. 
:::

***what about LLM-Evals?***.

* **GPTScore** Unlike G-Eval which directly performs the evaluation task with a form-filling paradigm, [GPTScore](https://arxiv.org/pdf/2302.04166.pdf) uses the conditional probability of generating the target text as an evaluation metric.

![image](https://hackmd.io/_uploads/r16nvbNnT.png)


* **SelfCheckGPT** SelfCheckGPT [is an odd one. It is a simple sampling-based approach that is used to fact-check LLM outputs](https://arxiv.org/pdf/2303.08896.pdf). It assumes that  hallucinated outputs are not reproducible, whereas if an LLM has knowledge of a given concept, sampled responses are likely to be similar and contain consistent facts.

SelfCheckGPT is an interesting approach because it makes detecting hallucination a reference-less process, which is extremely useful in a production setting.

![image](https://hackmd.io/_uploads/HJkauZVha.png)

:::info
:information_source: However, although you’ll notice that **G-Eval** and **Prometheus** is use case agnostic, SelfCheckGPT is not. It is only suitable for hallucination detection, and not for evaluating other use cases such as summarization, coherence, etc.
:::

* **QAG Score** QAG (Question Answer Generation) Score is a scorer that leverages LLMs’ high reasoning capabilities to reliably evaluate LLM outputs. It uses answers (usually either a ‘yes’ or ‘no’) to close-ended questions (which can be generated or preset) to compute a final metric score. It is reliable because it does NOT use LLMs to directly generate scores. For example, if you want to compute a score for faithfulness (which measures whether an LLM output was hallucinated or not), you would:

1. Use an LLM to extract all claims made in an LLM output.
1. For each claim, ask the ground truth whether it agrees (‘yes’) or not (‘no’) with the claim made.

In the case of faithfulness, if we define it as as the proportion of claims in an LLM output that are accurate and consistent with the ground truth, it  can easily be calculated by dividing the number of accurate (truthful) claims by the total number of claims made by the LLM. Since we are not using LLMs to directly generate evaluation scores but still leveraging its superior reasoning ability, we get scores that are both accurate and reliable.


# RAG Metrics
[RAG](https://arxiv.org/pdf/2309.15217.pdf) serves as a method to supplement LLMs with extra context to generate tailored outputs, and is great for building chatbots. It is made up of two components — the retriever, and the generator.

![image](https://hackmd.io/_uploads/Sk-qsZE3p.png)

Here’s how a RAG workflow typically works:

1. Your RAG system receives an input.
1. The **retriever** uses this input to perform a vector search in your knowledge base (which nowadays in most cases is a vector database).
1. The **generator** receives the retrieval context and the user input as additional context to generate a tailor output.


 
 :::info
:information_source:  **High quality LLM** outputs is the product of a great **retriever** and **generator**, For this reason, great RAG metrics focuses on evaluating either your RAG retriever or generator in a reliable and accurate way.
:::

* **Faithfulness** is a RAG metric that evaluates whether the LLM/generator in your RAG pipeline is generating LLM outputs that factually aligns with the information presented in the retrieval context. **But which scorer should we use for the faithfulness metric?**


:::success
:thumbsup: The QAG Scorer is the best scorer for RAG metrics
:::


we can calculate faithfulness using QAG by following this algorithm:

1. Use LLMs to extract all claims made in the output.
1. For each claim, check whether the it agrees or contradicts with each individual node in the retrieval context. In this case, the close-ended question in QAG will be something like: “Does the given claim agree with the reference text”, where the “reference text” will be each individual retrieved node. (Note that you need to confine the answer to either a ‘yes’, ‘no’, or ‘idk’. The ‘idk’ state represents the edge case where the retrieval context does not contain relevant information to give a yes/no answer.)
1. Add up the total number of truthful claims (‘yes’ and ‘idk’), and divide it by the total number of claims made.


* **Answer Relevancy**  is a RAG metric that assesses whether your RAG generator outputs concise answers, and can be calculated by determining the proportion of sentences in an LLM output that a relevant to the input (ie. divide the number relevant sentences by the total number of sentences).


:::info
:information_source: **(Using QAG for all RAG metrics)**
:::

* **Contextual Precision** is a RAG metric that assesses the quality of your RAG pipeline’s retriever. When we’re talking about contextual metrics, we’re mainly concerned about the relevancy of the retrieval context. A high contextual precision score means nodes that are relevant in the retrieval contextual are ranked higher than irrelevant ones. This is important because LLMs gives more weighting to information in nodes that appear earlier in the retrieval context, which affects the quality of the final output.

* **Contextual Relevancy**  the simplest metric to understand, contextual relevancy is simply the proportion of sentences in the retrieval context that are relevant to a given input.

:::warning
:warning: Checking for crazy talk (**Hallucination**) , **Toxicity**, and favoritism (**Bias**) with G-Eval sounds cool, or maybe trying out QAG, but we're still on the hunt for the best fit!
:::

# Evaluating an LLM Text Summarization:

**Existing Problems with Text Summarization Metrics**

Historically, model-based scorers (e.g., BertScore and ROUGE) have been used to evaluate the quality of text summaries. These metrics, while useful, often focus on surface-level features like word overlap and semantic similarity.

* **Word Overlap Metrics**: Metrics like ==ROUGE== (Recall-Oriented Understudy for Gisting Evaluation) often compare the overlap of words or phrases between the generated summary and a reference summary. If both summaries are of similar length, the likelihood of a higher overlap increases, potentially leading to higher scores.
* **Semantic Similarity Metrics**: Tools like ==BertScore== evaluate the semantic similarity between the generated summary and the reference. Longer summaries might cover more content from the reference text, which could result in a higher similarity score, even if the summary isn’t necessarily better in terms of quality or conciseness.


:::warning
:warning: These metrics struggle especially when the original text is composed of concatenated text chunks, which is often the case for a retrieval augmented generation (RAG) summarization use case. This is because they often fail to effectively assess summaries for disjointed information within the combined text chunks.
:::

**LLM-Evals**

==**G-Eval**==, an LLM-Evals framework that can be used for a summarization task. It usually involves providing the original text to an LLM like GPT-4 and asking it to generate a score and provide a reason for its evaluation. However, although better than traditional approaches, evaluating text summarization with LLMs presents its own set of challenges:

1. **Arbitrariness**: LLM Chains of Thought (CoTs) are often arbitrary, which is particularly noticeable when the models omit details that humans would typically consider essential to include in the summary.
1. **Bias**: LLMs often overlook factual inconsistencies between the summary and original text as they tend to prioritize summaries that reflect the style and content present in their training data.

In a nutshell, arbitrariness causes LLM-Evals to overlook the exclusion of essential details(or at least hinders their ability to identify what should be considered essential), while bias causes LLM-Evals to overlook factual inconsistencies between the original text and the summary.

:::success
:thumbsup: LLM-Evals can be Engineered to Overcome Arbitrariness and Bias Using QAG framework here is the paper : https://arxiv.org/pdf/2004.04228.pdf
:::

![image](https://hackmd.io/_uploads/HygXz7436.png)


**A Text Summarization Metric is the Combination of Inclusion and Alignment Scores :**

At the end of the day, you only care about two things in a summarization task:

* Inclusion of details.
* Factual alignment between the original text and summary.

**==Calculating Inclusion Score:==**

here’s the algorithm:

1. Generate n questions from the original text in a summarization task.
1. For each question, generate either a ‘yes’, ‘no’, or ‘idk’ answer using information from the original text and summary individually. The ‘idk’ answer from the summary represents the case where the summary does not contain enough information to answer the question.


:::info
:information_source: The higher the number of **Yes** answers, the greater the inclusion score. This is because matching answers indicate the summary is both factually correct and contains sufficient detail to answer the question. A ‘no’ from the summary indicates a contradiction, whereas an ‘idk’ indicates omission. (Since the questions are generated from the original text, answers from the original text should all be ‘yes’.)
:::

==**Calculating Alignment Score**==

The general algorithm to calculate the alignment score is identical to the one used for inclusion. However, ***note that in the case of alignment, we utilize the summary as the reference text to generate close-ended questions instead***. This is because for alignment, we only want to detect cases of hallucination and contradiction, so we are more concerned with the original text’s answers to the summary’s questions.

The ‘idk’ or ‘no’ answer from the original text indicates either a hallucination or contradiction respectively. (Again, answers from the summary should all be ‘yes’.)

**Combining the Scores**

There are several ways you can combine the scores to generate a final summarization score. You can take an **average**, use geometric/harmonic **means**, or take the **minimum** of the two. 


