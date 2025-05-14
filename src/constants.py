SYSTEM_PROMPT = """Respond in the following format:
<reasoning>
Your detailed reasoning goes here...
</reasoning>
<answer>
Relevant passages: title1, title2, ... 
(If no passages are relevant, respond with: "Relevant passages: No relevant passages")
</answer>
"""

USER_INSTRUCTIONS = """Identify all the passages with relevant information for answering the given question. Explain your reasoning and then only include the list of relevant passages at the end."""

USER_INSTRUCTIONS_CUSTOM = {
    # BRIGHT
    "biology": "Identify all the passages with relevant information for answering the given question. Explain your reasoning and then only include the list of relevant passages at the end.",
    "earth_science": "Identify all the passages with relevant information for answering the given question. Explain your reasoning and then only include the list of relevant passages at the end.",
    "economics": "Identify all the passages with relevant information for answering the given question. Explain your reasoning and then only include the list of relevant passages at the end.",
    "psychology": "Identify all the passages with relevant information for answering the given question. Explain your reasoning and then only include the list of relevant passages at the end.",
    "robotics": "Identify all the passages with relevant information for answering the given question. Explain your reasoning and then only include the list of relevant passages at the end.",
    "stackoverflow": "Identify all the passages with relevant information for answering the given question. Explain your reasoning and then only include the list of relevant passages at the end.",
    "sustainable_living": "Identify all the passages with relevant information for answering the given question. Explain your reasoning and then only include the list of relevant passages at the end.",
    "leetcode": "Given the following coding problem, identify all the passages with relevant examples that can help answer the given question. Explain your reasoning and then only include the list of relevant passages at the end.",
    "pony": "Identify all the passages with relevant information for answering the given question. Explain your reasoning and then only include the list of relevant passages at the end.",
    "aops": "Given the following math problem, identify all the passages with relevant examples that can help answer the given question. Explain your reasoning and then only include the list of relevant passages at the end.",
    "theoremqa_questions": "Given the following math problem, identify all the passages with relevant examples that can help answer the given question. Explain your reasoning and then only include the list of relevant passages at the end.",
    "theoremqa_theorems": "Given the following math problem, identify all the passages with relevant theorems that can help answer the given question. Explain your reasoning and then only include the list of relevant passages at the end.",
    # BEIR
    "arguana": "I am looking to write an essay and need to find counterarguments against this statement.\nIdentify all the passages with any counterarguments or evidence that could be used to help me.",
    "climate-fever": "I am looking to write an essay and need to find evidence that either supports or contradicts the given statement",
}
