# This file is going to code the generation of paraphrase for the new intent with an API call to an LLM
# Will also be in charge of pre and post processing of the input and output of the LLM
# get new input => pre process it => call the LLM for more context ==> get output with parahrased and label intent

# most of the structure of this file has been created with the help of AI, to help me create the good structure of the prompt,
# Prompt to help me create the System_prompt: I am building a system for incremental learning of intents classifIcation, the model knows already a set of intents with matching utterances
# and need an LLM to generate me new intent with a set of utterances based on an unknown detected utterance, I need to feed to the LLM as
# input the context of the already known utterances which is a balanced set of all the intents, a single or multiple utterances (unkown in an ood detection system)
# I need clear instructions to pass on the LLM to generate these utterances and the new intent if it is not explicitly indicated.


SYSTEM_PROMPT = """You generate synthetic training data for an on-device intent classifier used by a voice assistant.

Your job: given one or more real user utterances that the current classifier could NOT recognize, produce a batch of new training utterances that all express the SAME underlying intent as those escalated utterances.

Guidelines:
- Write short, natural, single-turn utterances, the way real people actually speak to a voice assistant.
- Maximize diversity: vary phrasing, sentence length, vocabulary, politeness, and specificity. Avoid near-duplicate sentences.
- Every utterance must express ONE consistent intent: the one shown in the escalated examples.
- You are given samples of the intents the classifier ALREADY knows. Your generated utterances must be clearly DISTINCT from those known intents so the new class does not blur into an existing one. When the new intent is close to a known one, emphasize the details that set it apart.
- Do not include real personal data; use neutral placeholders where a name/place/number is needed.

Return ONLY a JSON object, with no surrounding prose or code fences, matching exactly this schema:
{"intent_name": "<snake_case name for the new intent>", "utterances": ["...", "..."]}"""


def _format_context_block(context):
    """It transforms the context dictionary into a bullet point structure, with each intent and under, for each
    Bullet point a specific utterance.
    """
    lines = []
    for intent_name, utterances in context.items():
        lines.append(f"- {intent_name}:")
        for utt in utterances:
            lines.append(f'    - "{utt}"')
    return "\n".join(lines)


def format_prompt(context, escalated_utts, target_intent, n_utterances):
    """
    This generates the prompt formated to pass to the LLM to generate the synthetic training data.
    The context is the set of already known utterances for each intent, so it has context of what to not generate
    in the case that two intents are similar.

    escalated_utts: A list of the escalated utterances, it is a list so it can be multiple, but only one utterance will be escalated.
    target_intent: The name to use for the new intent (experiment mode, so the generated class matches a real MASSIVE intent), or None to let the model
    invent a snake_case name (production mode).

    n_utterances, how many training utterances to generate (K).
    """
    context_block = _format_context_block(context)
    escalated_block = "\n".join(
        f'- "{utt}"' for utt in escalated_utts
    )  # for each escalated utterance in the list, merge into a bullet-like structure.

    if (
        target_intent is not None
    ):  # this is not None only for the experiment of synthetic vs real training data evaluated both on real data.
        name_instruction = f'Name this new intent exactly: "{target_intent}".'
    else:
        name_instruction = "Choose a concise snake_case name for this new intent."

    user_content = f"""## Intents the classifier already knows (stay clearly distinct from these) {context_block} ## Escalated utterances (unknown intent to learn)
{escalated_block}## Task Generate exactly {n_utterances} training utterances for the new intent expressed by the escalated examples above.
{name_instruction} Return only the JSON object described in the instructions."""

    return {
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_content}],
    }
