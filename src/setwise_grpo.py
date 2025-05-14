import string
import logging
from typing import Dict, Any, Callable, Optional, List
import re

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Respond in the following format:
<reasoning>
Your detailed reasoning goes here...
</reasoning>
<answer>
Relevant passages: title1, title2, ... 
(If no passages are relevant, respond with: "Relevant passages: No relevant passages")
</answer>
"""

USER_INSTRUCTIONS = """Identify all the relevant passages for answering the given query. Explain your reasoning step by step."""


def normalize_text(text: str) -> str:
    return text.lower().translate(str.maketrans("", "", string.punctuation)).strip()


def axo_rank_rl_transform(cfg: Dict[str, Any], *args, **kwargs) -> Callable:
    def transform_fn(
        example: Dict[str, str], tokenizer: Optional[Callable] = None
    ) -> Dict[str, Any]:
        def remove_non_utf8(s: Optional[str]) -> str:
            if not isinstance(s, str):
                return ""
            return s.encode("utf-8", "ignore").decode("utf-8", "ignore")

        question = remove_non_utf8(example.get("question", example.get("query")))
        context = remove_non_utf8(example.get("context", [""])[0])
        citations = example.get("citations")
        if len(citations) == 0:
            citations = ["No relevant passages"]

        prompt = f"{SYSTEM_PROMPT}\n\n{USER_INSTRUCTIONS}\n\nQuestion: {question}\nPassages:\n{context}"
        prompt = remove_non_utf8(prompt)

        return {
            "prompt": [{"role": "user", "content": prompt}],
            "citations_": citations,
        }

    return transform_fn, {
        "remove_columns": ["question", "answer", "context", "citations"]
    }


def extract_xml_section(text: str, tag: str) -> str:
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"
    if open_tag not in text or close_tag not in text:
        return ""
    return text.split(open_tag)[-1].split(close_tag)[0].strip()


def extract_supporting_passages(answer_text: str) -> List[str]:
    match = re.search(
        r"(?i)relevant\s*(?:passages)?\s*[:\-]?\s*(.*)", answer_text, re.DOTALL
    )
    if match:
        passages_str = match.group(1).strip()
        passages = re.split(r"[,;\n]+", passages_str)
        return [p.strip() for p in passages if p.strip()]
    return []


def fbeta_citation_reward_func(
    completions: List[Any],
    citations_: List[List[str]],
    beta: float = 2.0,
    reward_value: float = 5.0,
    **kwargs,
) -> List[float]:
    rewards = []
    for response, allowed_citations_list in zip(completions, citations_):
        content = response[0]["content"]
        if "<answer>" not in content or "</answer>" not in content:
            rewards.append(0.0)
            continue

        allowed_set = set(normalize_text(p) for p in allowed_citations_list if p)

        answer_section = extract_xml_section(content, "answer")
        predicted_passages = extract_supporting_passages(answer_section)
        predicted_set = [normalize_text(p) for p in predicted_passages if p]

        true_positives = sum(1 for p in predicted_set if p in allowed_set)
        predicted_count = len(predicted_set)
        allowed_count = len(allowed_set)

        precision = true_positives / predicted_count if predicted_count > 0 else 0.0
        recall = true_positives / allowed_count if allowed_count > 0 else 0.0

        if precision == 0 and recall == 0:
            fbeta = 0.0
        else:
            fbeta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)

        reward = fbeta * reward_value
        rewards.append(reward)
    return rewards


def xmlcount_reward_func(completions: List[Any], **kwargs) -> List[float]:
    rewards = []
    required_tags = ["<reasoning>", "</reasoning>", "<answer>", "</answer>"]
    for completion in completions:
        content = completion[0]["content"]
        if all(tag in content for tag in required_tags):
            rewards.append(0.5)
        else:
            rewards.append(0)
    return rewards


def formatting_reward_func(completions: List[Any], **kwargs) -> List[float]:
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    for r in responses:
        reward = 0.0
        if "<answer>" not in r or "</answer>" not in r:
            rewards.append(reward)
            continue

        answer_section = extract_xml_section(r, "answer")
        normalized_answer = normalize_text(answer_section)
        if "supporting passages" in normalized_answer:
            reward += 0.50
        rewards.append(reward)
    return rewards


def length_reward_func(completions: List[Any], **kwargs) -> List[float]:
    responses = [completion[0]["content"] for completion in completions]
    rewards = []

    for r in responses:
        if "<answer>" not in r or "</answer>" not in r:
            rewards.append(-5)
        else:
            answer_section = extract_xml_section(r, "answer")
            answer_length = len(answer_section)

            if answer_length <= 200:
                reward = 0
            else:
                reward = -5

            rewards.append(reward)

    return rewards


def unicode_reward_func(completions: List[Any], **kwargs) -> List[float]:
    responses = [completion[0]["content"] for completion in completions]
    rewards = []

    for r in responses:
        unicode_chars = re.findall(r"[\u0080-\U0010FFFF]", r)
        if len(unicode_chars) > 36:
            rewards.append(-5.0)
        else:
            rewards.append(0)
    return rewards
