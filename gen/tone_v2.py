"""Customer messages with varied tone (Maya v0.2).

v0.1 recognised anger only through its templates' phrases ("Unacceptable", "furious"). Here anger comes
through many styles (capitals, sarcasm, polite-but-firm, blunt, repeated contact), calm messages often
mention problems (so "problem" != "angry"), and trap texts use the words of a statement with the opposite
meaning ("I'm not upset, just asking").

Contexts avoid the eval_v2 / eval_v3 domains (no travel, rentals, refunds-with-policy, landlords).

Facts: upset, threat (to leave / escalate / go public), problem (reports something wrong), thanks (expresses thanks)
"""
import random

from .facts import Fact

CONTEXTS = {  # context -> [(subject, [problems that fit it])]
    "software": [("the calendar sync", ["keeps crashing", "drops half my entries", "stopped syncing yesterday"]),
                 ("the invoice export", ["keeps crashing", "shows the wrong totals", "produces empty files"]),
                 ("the mobile app", ["logs me out every hour", "won't load at all", "keeps crashing"]),
                 ("the dashboard", ["shows the wrong totals", "has been down since this morning", "won't load at all"])],
    "delivery": [("my parcel", ["still hasn't arrived", "arrived smashed", "went to the wrong address"]),
                 ("my grocery delivery", ["is missing two items", "was left in the rain", "came a day late"])],
    "billing": [("my last bill", ["charged me twice", "is $80 higher than agreed"]),
                ("my subscription", ["renewed without asking me", "still shows a cancelled plan"])],
    "restaurant": [("our takeaway order", ["came an hour late and cold", "was missing the main course"]),
                   ("our table booking for Friday", ["was given to someone else", "disappeared from the system"])],
    "gym": [("my membership", ["was frozen without notice", "charged me after I cancelled"]),
            ("the class booking app", ["cancels my bookings at random", "won't load at all"])],
    "phone": [("my home internet", ["has had no signal for three days", "is crawling at 1 Mbps"]),
              ("my mobile plan", ["cut out during every call", "charged me roaming fees at home"])],
}

UPSET_STYLES = {
    "caps": ["{T} {B}. AGAIN.", "{T} {B} AGAIN?! WHY?", "{T} {B}!!! This is the third time."],
    "sarcasm": ["Wow, {t} {b}. Brilliant work, truly.", "Love how {t} {b}. Really making my day.",
                "Oh fantastic, {t} {b} again. Exactly what I needed today."],
    "firm": ["I have now written four times about how {t} {b}, and nobody has fixed it. I expect a proper answer.",
             "{T} {b}. This is not acceptable for what I pay.", "I am extremely disappointed that {t} {b}."],
    "blunt": ["{T} {b}. Fed up.", "Seriously, {t} {b}. Sort it out.", "{T} {b} and frankly I've had enough."],
    "weary": ["Honestly, I'm sick of this. {T} {b} for the fifth week running.", "Every single week {t} {b}. I give up being nice about it."],
}
CALM_PROBLEM = ["Hi, just letting you know {t} {b}. No rush, whenever you get a moment.",
                "Hello! Small thing: {t} {b}. Could someone take a look when convenient?",
                "Quick note that {t} {b}. Not a big deal, just thought you'd want to know.",
                "Morning team, {t} {b} on my side. Happy to send more details if useful.",
                "I think {t} {b}, but it might be something on my end. Any tips?"]
CALM_NO_PROBLEM = ["How do I change the email address on {t}?", "Is there a way to export {t} as a PDF?",
                   "Could you tell me when {t} is due to renew?", "Just checking whether {t} includes weekends.",
                   "What are your opening hours on public holidays?"]
TRAPS = ["I'm not upset at all, just curious: how do I change the settings for {t}?",
         "No complaints here! Just wondering whether {t} can be set to weekly.",
         "Not angry, not in a hurry, only asking how {t} works with two accounts.",
         "I wouldn't say I'm unhappy, I just want to understand the options for {t}."]
THANKS = [" Thanks a lot!", " Thank you for the quick help last time.", " Much appreciated.", " Cheers!"]
THREATS = {"cancel": [" If this isn't fixed today, I'm cancelling.", " Fix it or I'm taking my business elsewhere.",
                      " One more time and I'm switching providers."],
           "escalate": [" I want to speak to a manager.", " I'll be raising this with your head office.",
                        " Escalate this, please, or I will."],
           "public": [" I'm about to post about this everywhere.", " Expect a one-star review.", " I'll be telling everyone I know."]}


class Tone:
    name = "customer-tone-v2"
    flips = {"mood": ["upset", "calm_problem", "calm", "trap"], "threat": ["none", "cancel", "escalate", "public"]}
    implications = [("threat", "upset"), ("thanks", "not:threat")]

    def valid_flip(self, s, slot):
        if slot == "threat":
            return s["mood"] == "upset"
        return True

    def sample(self, rng):
        mood = rng.choice(["upset", "upset", "upset", "calm_problem", "calm_problem", "calm", "trap"])
        ctx = rng.choice(list(CONTEXTS))
        thing, bads = rng.choice(CONTEXTS[ctx])
        return {"_seed": rng.random(), "ctx": ctx, "mood": mood, "t": thing, "b": rng.choice(bads),
                "style": rng.choice(list(UPSET_STYLES)),
                "threat": rng.choice(["none", "none", "cancel", "escalate", "public"]) if mood == "upset" else "none",
                "thanks": mood in ("calm_problem", "calm", "trap") and rng.random() < 0.4}

    def render(self, s):
        r = random.Random(s["_seed"])
        t, b = s["t"], s["b"]
        if s["mood"] == "upset":
            tpl = r.choice(UPSET_STYLES[s["style"]])
            text = tpl.format(t=t, b=b, T=t[0].upper() + t[1:], B=b.upper())
            if s["style"] == "caps":
                text = text.replace(t[0].upper() + t[1:], (t[0].upper() + t[1:]).upper())
        elif s["mood"] == "calm_problem":
            text = r.choice(CALM_PROBLEM).format(t=t, b=b)
        elif s["mood"] == "calm":
            text = r.choice(CALM_NO_PROBLEM).format(t=t)
        else:
            text = r.choice(TRAPS).format(t=t)
        if s["threat"] != "none":
            text += r.choice(THREATS[s["threat"]])
        if s["thanks"]:
            text += r.choice(THANKS)
        return text

    def facts(self, s):
        upset = s["mood"] == "upset"
        return [
            Fact("upset", upset,
                 ["The customer is irritated", "This person sounds upset", "Is the sender frustrated?",
                  "The tone of the message is hostile or exasperated", "The customer is unhappy with the service"],
                 ["The sender sounds relaxed", "There is no frustration in this message", "The customer is calm",
                  "Is the tone friendly and patient?"]),
            Fact("threat", s["threat"] != "none",
                 ["The customer says they will leave or complain higher up", "Is there a threat to cancel, escalate or go public?"],
                 ["The message contains no threat of any kind"]),
            Fact("problem", s["mood"] in ("upset", "calm_problem"),
                 ["Something is not working for this person", "The message describes an issue",
                  "Is the customer pointing out a fault?"],
                 ["The message is only a question, nothing is broken", "No fault is being reported"]),
            Fact("thanks", s["thanks"],
                 ["The customer thanks the company", "Does the message express gratitude?"],
                 ["There is no thank-you in the message"]),
        ]


TONE = Tone()
