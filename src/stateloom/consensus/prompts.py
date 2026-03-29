"""Prompt templates for the consensus framework."""

DEFAULT_CONFIDENCE_INSTRUCTION = (
    "\n\nEnd your response with your confidence level in this exact format: "
    "[Confidence: X.XX] where X.XX is between 0.00 and 1.00."
)

VOTE_SYSTEM_PROMPT = (
    "You are a careful, analytical assistant. Answer the question to the best "
    "of your ability. Be concise but thorough." + DEFAULT_CONFIDENCE_INSTRUCTION
)

DEBATE_ROUND_TEMPLATE = (
    "You are participating in a multi-model debate. Here are the responses "
    "from other models in the previous round:\n\n"
    "{previous_responses}\n\n"
    "Consider these perspectives carefully. Where you disagree, explain why "
    "with specific reasoning. Where you agree, build on the strongest points. "
    "Update your answer if you find another model's argument more compelling."
    + DEFAULT_CONFIDENCE_INSTRUCTION
)

JUDGE_SYNTHESIS_TEMPLATE = (
    "You are a judge synthesizing the best answer from a multi-model debate. "
    "Here is the full transcript:\n\n{transcript}\n\n"
    "Synthesize the strongest arguments into a single, coherent answer. "
    "Resolve any remaining disagreements by choosing the most well-reasoned "
    "position. Be concise." + DEFAULT_CONFIDENCE_INSTRUCTION
)
