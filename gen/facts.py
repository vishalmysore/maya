"""Yes/no facts over the layaMOE domains.

A fact has a value computed from the generator's labels, a list of wordings that are true when
the value is True (statements and questions), and a list of negated wordings (true when the value
is False). Training pairs a text with one wording; when a fact has negations, the same text is
often also paired with a negated wording, which gives the consistency loss its negation pairs.
"""


class Fact:
    def __init__(self, name, value, pos, neg=()):
        self.name, self.value, self.pos, self.neg = name, value, list(pos), list(neg)


def guardrails(lab):
    return [
        Fact("destructive", lab["destructive"],
             ["The action is destructive and cannot be undone",
              "This action permanently destroys or overwrites data",
              "Running this would cause damage that cannot be reversed",
              "Will this plan permanently destroy data?",
              "Is this action irreversible?"],
             ["The action can be undone if needed",
              "Nothing is permanently lost if this runs",
              "This action is reversible"]),
        Fact("needs_human", lab["needs_human"],
             ["A human should approve this action before it runs",
              "This action requires sign-off from a person first",
              "An operator must review this before the agent proceeds",
              "Does a person need to approve this first?"],
             ["It is safe to run this action without a human approving it first",
              "The agent can go ahead with this on its own",
              "This action is low-risk enough to run automatically",
              "Can the agent do this without asking anyone?"]),
        Fact("high_risk", lab["risk"] >= 2,
             ["This plan is dangerous", "The risk of this action is high or critical",
              "Is this a high-risk action?"],
             ["This plan carries little risk", "The risk of this action is low"]),
    ]


def moderation(lab):
    return [
        Fact("spam", lab["spam"],
             ["The comment is spam or an advertisement", "This is promotional spam",
              "The comment is trying to sell something or drive clicks", "Is this comment spam?"],
             ["The comment is not spam", "This is a genuine comment, not an advert"]),
        Fact("abusive", lab["toxicity"] >= 2,
             ["The comment insults or threatens someone", "This comment is abusive",
              "Does the comment attack a person?"],
             ["The comment does not insult or threaten anyone", "The comment is free of personal attacks"]),
        Fact("threat", lab["toxicity"] >= 3,
             ["The comment contains a threat or hate speech", "Is someone being threatened here?"],
             ["There is no threat or hate speech in the comment"]),
        Fact("remove", lab["verdict"] == "remove",
             ["This comment should be taken down", "The post breaks the rules and must be removed"],
             ["This comment can stay up", "The post should not be removed"]),
    ]


def support(lab):
    return [
        Fact("angry", lab["angry"],
             ["The customer sounds angry", "The customer is frustrated or upset",
              "The tone of this ticket is hostile or furious", "Is the customer angry?"],
             ["The customer sounds calm", "The customer is not angry", "The ticket is written in a calm tone"]),
        Fact("bug", lab["team"] == "bug",
             ["The customer reports that something is broken", "This ticket is about a defect",
              "Is something not working for the customer?"],
             ["Nothing is broken in this ticket", "The ticket does not report a defect"]),
        Fact("access", lab["team"] == "account_access",
             ["The customer cannot get into their account", "This is a login or account access problem"],
             ["The customer can access their account"]),
        Fact("feature", lab["team"] == "feature_request",
             ["The customer is asking for a new feature", "Is this a feature request?"],
             ["The ticket is not a feature request"]),
        Fact("urgent", lab["urgency"] >= 2,
             ["This ticket needs a response today", "The request is urgent", "Is this urgent?"],
             ["This ticket can wait", "The request is not urgent"]),
    ]


def delivery(lab):
    return [
        Fact("upset", lab["upset"],
             ["The customer is upset", "The customer is unhappy or angry",
              "The message shows the customer is frustrated", "Is the customer upset?"],
             ["The customer is not upset", "There is no sign the customer is unhappy"]),
        Fact("reship", lab["action"] == "reship",
             ["A replacement should be sent", "The parcel needs to be shipped again"],
             ["There is no need to send a replacement"]),
        Fact("fine", lab["action"] == "wait",
             ["The delivery is on track and nothing needs to be done", "Is the parcel still on its way as planned?"],
             ["Something has gone wrong with this delivery", "Support needs to act on this delivery"]),
        Fact("urgent", lab["urgency"] >= 2,
             ["This delivery issue is urgent", "Support must act on this quickly"],
             ["This delivery issue is not urgent"]),
    ]


def email(lab):
    return [
        Fact("needs_reply", lab["needs_reply"],
             ["The sender expects a reply", "Someone is waiting for an answer to this email",
              "This email asks the recipient to respond", "Does this email need a reply?"],
             ["No reply is expected", "The email does not need an answer", "Nothing needs to be sent back"]),
        Fact("urgent", lab["urgency"] >= 2,
             ["This email is urgent", "The email needs attention today", "Is this email time-sensitive?"],
             ["This email can wait", "The email is not urgent"]),
        Fact("delegate", lab["action"] == "delegate",
             ["This email should go to someone else", "The recipient should forward this to the right person"],
             ["The recipient can handle this email themselves"]),
    ]


LAYA_FACTS = {
    "agent-guardrails": (guardrails, []),
    "content-moderation": (moderation, [("threat", "abusive"), ("threat", "remove"), ("spam", "remove")]),
    "support-tickets": (support, [("feature", "not:urgent")]),
    "delivery-exceptions": (delivery, [("reship", "not:fine")]),
    "email-triage": (email, [("urgent", "needs_reply"), ("delegate", "needs_reply")]),
}
