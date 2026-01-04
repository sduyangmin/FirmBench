SEED_PROMPT = "You are a finance-domain reasoning assistant. and your task is to answer the user's question."

META_PROMPT = (
    "[[ ## context ## ]]\n"
    "You are the reflection model, an expert at improving large language model prompts for a specific task. "
    "You will be given:\n"
    "1. The current prompt (used by the target model for the task).\n"
    "2. A set of feedback notes summarizing where the system underperformed.\n\n"
    "[[ ## current prompt ## ]]\n"
    "{candidate}\n\n"
    "The following are examples of different task inputs provided to the assistant along "
    "with the assistant's response for each of them, and feedback on how the assistant's "
    "response could be better:\n"
    "[[ ## inputs, outputs, feedback ## ]]\n"
    "{inputs_outputs_feedback}\n\n"
    "[[ ## goal ## ]]\n"
    "Propose at most 1 new prompt that keeps what works, fixes weaknesses, improves clarity, "
    "completeness, and correctness, and is concise. Output ONLY as JSON with field 'list_candidate_prompts', "
    "where each item has key 'candidate_prompt'. Do NOT generate examples from the task. "
    "Do NOT add any lead-in text or preamble; return only the prompt itself inside the JSON."
)

INITIAL_CANDIDATES_GENERATION_PROMPT = (
    "[[ ## context ## ]]\n"
    "You are an expert at generating potentially optimal large language model prompts or instructions "
    "for a specific task. You will be given a seed prompt and sampled items from a dataset related to "
    "the task to help you diagnose what the task is about.\n"
    "[[ ## seed prompt ## ]]\n"
    "{seed_prompt}\n\n"
    "[[ ## inputs, outputs ## ]]\n"
    "{inputs}\n\n"
    "[[ ## goal ## ]]\n"
    "Generate a set of diverse and high-quality candidate prompts or instructions for the task "
    "using the seed prompt as your reference. Provide at most {num_new_prompts} new candidate prompts. "
    "Output ONLY as JSON with field 'list_candidate_prompts' where each item has key 'candidate_prompt'. "
    "Do NOT generate examples from the task. "
    "Do NOT add any lead-in text or explanation; return only the prompts inside JSON."
)

MERGING_PROMPT = (
    "[[ ## context ## ]]\n"
    "You are an expert at merging text prompts. You will be given candidate prompts to synthesize:\n"
    "[[ ## candidate prompts ## ]]\n"
    "{candidates}\n\n"
    "[[ ## goal ## ]]\n"
    "Synthesize the candidate prompts into a single, coherent prompt that captures the essence of all the candidates. "
    "The synthesis must be done such that the weaknesses of the individual candidates are addressed. "
    "Output ONLY the merged candidate prompt as JSON with field 'list_candidate_prompts', "
    "each item having key 'candidate_prompt'. Do NOT add any lead-in text or explanation."
)



