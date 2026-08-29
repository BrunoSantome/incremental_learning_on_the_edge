# This file is going to code the generation of paraphrase for the new intent with an API call to an LLM
# Will also be in charge of pre and post processing of the input and output of the LLM
# get new input => pre process it => call the LLM for more context ==> get output with parahrased and label intent

# most of the structure of this file has been created with the help of AI, to help me create the good structure of the prompt,
# Prompt to help me create the System_prompt: I am building a system for incremental learning of intents classifIcation, the model knows already a set of intents with matching utterances
# and need an LLM to generate me new intent with a set of utterances based on an unknown detected utterance, I need to feed to the LLM as
# input the context of the already known utterances which is a balanced set of all the intents, a single or multiple utterances (unkown in an ood detection system)
# I need clear instructions to pass on the LLM to generate these utterances and the new intent if it is not explicitly indicated.

import json
import os
import sys

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

    user_content = f"""## Intents the classifier already knows (stay clearly distinct from these)
{context_block}

## Escalated utterances (unknown intent to learn)
{escalated_block}

## Task
Generate exactly {n_utterances} training utterances for the new intent expressed by the escalated examples above.
{name_instruction} Return only the JSON object described in the instructions."""

    return {
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_content}],
    }


class LLMClient:
    def generate(self, prompt):
        raise NotImplementedError


class AnthropicClient(LLMClient):
    """LLMClient backed by the Anthropic Messages API."""

    def __init__(self, model, max_tokens):
        from anthropic import (
            Anthropic,
        )  # lazy: only needed if you actually use this provider

        self.model = model
        self.max_tokens = max_tokens
        # Zero-arg Anthropic() resolves credentials from the environment
        # (ANTHROPIC_API_KEY, or an `ant auth login` profile).
        self.client = Anthropic()

    def generate(self, prompt):
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=prompt["system"],
            messages=prompt["messages"],
        )
        # Concatenate the text blocks into the raw JSON string the model was told to emit.
        return "".join(block.text for block in response.content if block.type == "text")


PROVIDERS = {
    "anthropic": AnthropicClient,
}


def build_llm_client(config):
    """Build the LLM client from the llm block of the config."""
    llm_cfg = dict(config["llm"])  # copy so the pop doesn't mutate the loaded config
    provider = llm_cfg.pop("provider")
    return PROVIDERS[provider](**llm_cfg)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.configuration import load_config
    from core.dataloader import DataClass

    #  ['takeaway_order', 'general_joke', 'recommendation_locations', 'play_podcasts', 'transport_traffic']
    TARGET_INTENT = "takeaway_order"
    N_UTTERANCES = 10
    config = load_config()
    data = DataClass()
    context = data.build_intent_context()

    train_split = data.sets_names[0]
    seed_rows = data.dataset_totrain[train_split].filter(
        lambda ex: ex[data.label_col] == TARGET_INTENT
    )
    escalated_utts = [seed_rows[0]["utt"]]

    prompt = format_prompt(
        context=context,
        escalated_utts=escalated_utts,
        target_intent=TARGET_INTENT,
        n_utterances=N_UTTERANCES,
    )

    print("SYSTEM \n", prompt["system"])
    print("\nUSER\n", prompt["messages"][0]["content"])

    llm = build_llm_client(config)
    raw = llm.generate(prompt)

    print("\nRAW LLM OUTPUT (repr\n")
    print(repr(raw))
    print("\n readable\n")
    print(raw)

"""
 You generate synthetic training data for an on-device intent classifier used by a voice assistant.

Your job: given one or more real user utterances that the current classifier could NOT recognize, produce a batch of new training utterances that all express the SAME underlying intent as those escalated utterances.

Guidelines:
- Write short, natural, single-turn utterances, the way real people actually speak to a voice assistant.
- Maximize diversity: vary phrasing, sentence length, vocabulary, politeness, and specificity. Avoid near-duplicate sentences.
- Every utterance must express ONE consistent intent: the one shown in the escalated examples.
- You are given samples of the intents the classifier ALREADY knows. Your generated utterances must be clearly DISTINCT from those known intents so the new class does not blur into an existing one. When the new intent is close to a known one, emphasize the details that set it apart.
- Do not include real personal data; use neutral placeholders where a name/place/number is needed.

Return ONLY a JSON object, with no surrounding prose or code fences, matching exactly this schema:
{"intent_name": "<snake_case name for the new intent>", "utterances": ["...", "..."]}

USER
 ## Intents the classifier already knows (stay clearly distinct from these)
- alarm_set:
    - "set an alarm for the dance classes two hours before the classes start"
    - "set my alarm for six am tomorrow"
    - "change alarm to start at midnight"
    - "can you set an alarm for seven am"
    - "turn on an alarm for nine am"
    - "alarm time for twelve p. m."
    - "put alarm for this meeting"
    - "call me at six am"
- audio_volume_down:
    - "turn down media volume"
    - "can you make it a little quieter"
    - "your volume is too high please repeat that lower"
    - "turn speaker volume down"
    - "please lower the volume"
    - "can you speak quieter please"
    - "ok google lower all volume on speakers please"
    - "can you turn that down"
- audio_volume_mute:
    - "stop it"
    - "mute"
    - "cut off noise"
    - "will you please mute my speakers"
    - "set volume to zero"
    - "please mute the volume control"
    - "please be silent until i tell you not to be"
    - "be quiet"
- audio_volume_up:
    - "on the music player set the volume level at sixty"
    - "increase the volume to max please"
    - "i need to hear the volume of the current music"
    - "i need to hear sound on my speakers"
    - "turn up the speakers"
    - "please raise the volume of speaker"
    - "increase volume by one"
    - "loudly"
- datetime_query:
    - "what time are we looking at right now"
    - "please give me today's date"
    - "what time is it in england"
    - "what day of the week is valentines on"
    - "where does the twenty second fall this month"
    - "i need to know what time it is right now in new york city"
    - "what time is it now in g. m. t."
    - "tell me the today's date"
- email_addcontact:
    - "add a new email in my contacts from john"
    - "add new email to anna"
    - "add dale at gmail dot com to my contacts"
    - "add this email to my contact"
    - "add this new email"
    - "add work email address"
    - "create new contact with email"
    - "add this email to my address book"
- lists_createoradd:
    - "start a new list"
    - "make a list"
    - "is there room on my grocery list for an extra item"
    - "please order me this months groceries"
    - "add mow lawn to things to do list"
    - "make a new list for me please"
    - "add milk to the grocery list"
    - "i need to make a grocery list"
- news_query:
    - "what are the news lately"
    - "headlines from dhaka tribune"
    - "what's the news on b. b. c. news"
    - "what is the news in my area"
    - "what's the latest news from c. n. n."
    - "what's the news"
    - "ok google add a notification from bob's news on the weather"
    - "set notification for six o'clock news update"
- play_audiobook:
    - "i want you to fast forward the audio then resume"
    - "continue the smurfs book"
    - "can you play my favorite audio book of king kong"
    - "resume the girl on the train on audible"
    - "open lyrics name"
    - "music one"
    - "play for me madonna song from audiobook"
    - "play audiobook of jacob"
- play_game:
    - "twenty questions you start"
    - "games"
    - "play chess app"
    - "play golf"
    - "start candy crush"
    - "please play my cricket game"
    - "would you like to play a game with me"
    - "let's play a game how about tic tac toe"
- play_music:
    - "hey siri play me an automated playlist based on songs heard in gaana app this week"
    - "play for me hip hop music"
    - "i want to listen arijit singh song once again"
    - "let me listen to my top rock songs playlist"
    - "please start the playlist huey lewis and the news"
    - "play something random from google play"
    - "olly play me an upbeat song through your speakers"
    - "start playing jazz music"
- play_radio:
    - "on the radio it is time for good music"
    - "activate the radio please"
    - "play new radio channel"
    - "load up doctor demento and play it by my bed"
    - "play alex jones in radio"
    - "play the last radio channel"
    - "turn on some radio station"
    - "tune in to nine hundred and thirty five on radio"
- transport_query:
    - "what's the next train at westcombe park"
    - "how far is orlando from my house"
    - "how long is the walk from nyack to valley cottage"
    - "can you give me directions from mcdonalds manhattan to my home"
    - "hey google hey what time can i catch the nearest train to me"
    - "what is the next metro to d. c."
    - "how to get to cracow before noon by train"
    - "could you find a train back home at around seven p. m."
- transport_taxi:
    - "find a ride to the bar"
    - "i need a taxi to pick me up at the house and take me downtown"
    - "please send a taxi to my house"
    - "book a taxi for tomorrow morning"
    - "olly can you call a taxi"
    - "i need to book a taxi"
    - "book a taxi to the airport"
    - "call a taxi for me to arrive tonight at five"
- weather_query:
    - "will it be hot"
    - "what is the weather in utah"
    - "is it going to be cloudy in london during the weekend"
    - "is it going to be hot in two hours"
    - "is it gonna snow"
    - "is it going to be windy tomorrow"
    - "ok google what is the weather where i am at today"
    - "should i bring snow shoes"

## Escalated utterances (unknown intent to learn)
- "please order some sushi for dinner"

## Task
Generate exactly 10 training utterances for the new intent expressed by the escalated examples above.
Name this new intent exactly: "takeaway_order". Return only the JSON object described in the instructions.

RAW LLM OUTPUT (repr

'{"intent_name": "takeaway_order", "utterances": ["please order some sushi for dinner", "i\'d like to get a large pepperoni pizza delivered", "order me some thai food from the place down the street", "can you get takeout from the chinese restaurant tonight", "place a delivery order for two burgers and fries", "i want to order food to be delivered to my house", "order my usual from the sandwich shop", "get me some tacos delivered please", "order dinner for two from the indian place", "can you order a curry for pickup in thirty minutes"]}'

----- readable -----

{"intent_name": "takeaway_order", "utterances": ["please order some sushi for dinner", "i'd like to get a large pepperoni pizza delivered", "order me some thai food from the place down the street", "can you get takeout from the chinese restaurant tonight", "place a delivery order for two burgers and fries", "i want to order food to be delivered to my house", "order my usual from the sandwich shop", "get me some tacos delivered please", "order dinner for two from the indian place", "can you order a curry for pickup in thirty minutes"]}
"""
