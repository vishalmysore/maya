"""Slot-based generators for seven extra training domains (none of them appear in data/eval_v2).

A domain samples a slot dict, renders it to text and computes yes/no facts from the slots.
Filler wording is drawn from random.Random(slots["_seed"]), so changing one slot and rendering
again gives a minimal pair: the same text except for that one detail.
"""
import random

from .facts import Fact


class Domain:
    name = ""
    flips = {}          # slot -> list of values it may be flipped to
    implications = []   # (fact_a, fact_b): if a is true then b is true; "not:x" means x is false

    def sample(self, rng):
        raise NotImplementedError

    def render(self, s):
        raise NotImplementedError

    def facts(self, s):
        raise NotImplementedError


def _r(s):
    return random.Random(s["_seed"])


# ---------------------------------------------------------------------------------------------
class ITIncidents(Domain):
    name = "it-incidents"
    flips = {"symptom": ["down", "errors", "slow", "disk", "cert"], "scope": ["all", "region", "internal"],
             "status": ["ongoing", "resolved"]}
    implications = [("page", "impact"), ("resolved", "not:impact"), ("outage", "not:resolved")]
    SERVICES = ["checkout API", "login service", "search cluster", "payments gateway", "mobile push service",
                "image CDN", "order database", "recommendation engine"]

    def sample(self, rng):
        return {"_seed": rng.random(), "svc": rng.choice(self.SERVICES),
                "symptom": rng.choice(["down", "down", "errors", "errors", "slow", "disk", "cert"]),
                "scope": rng.choice(["all", "all", "region", "internal"]),
                "status": rng.choice(["ongoing", "ongoing", "ongoing", "resolved"])}

    def render(self, s):
        r = _r(s)
        mins = r.choice([4, 12, 25, 40, 90])
        sym = {"down": r.choice([f"The {s['svc']} is completely down", f"The {s['svc']} is not responding at all"]),
               "errors": r.choice([f"The {s['svc']} is returning 5xx errors for about {r.choice([8, 20, 35])}% of requests",
                                   f"Error rate on the {s['svc']} jumped to {r.choice([12, 30, 60])}%"]),
               "slow": f"p95 latency of the {s['svc']} is {r.choice([3, 6, 9])}x the usual level",
               "disk": f"Disk usage on the {s['svc']} hosts reached {r.choice([81, 85, 88])}%, still serving normally",
               "cert": f"The TLS certificate of the {s['svc']} expires in {r.choice([12, 20, 25])} days"}[s["symptom"]]
        scope = {"all": r.choice([" for all users", " for every customer"]),
                 "region": r.choice([" for customers in the EU region", " for users in Asia-Pacific"]),
                 "internal": r.choice([" on the internal staging copy only; production is unaffected",
                                       " in the internal admin tool only, no customer traffic goes through it"])}[s["scope"]]
        if s["symptom"] in ("disk", "cert"):
            scope = ""
        if s["status"] == "resolved":
            tail = r.choice([f" Fixed after {mins} minutes; all metrics are back to normal.",
                             f" Resolved: a rollback {mins} minutes ago restored normal service."])
            head = "Earlier today: " if r.random() < 0.5 else "Resolved incident: "
            return head + sym[0].lower() + sym[1:] + scope + "." + tail
        tail = r.choice([f" Started {mins} minutes ago.", f" Ongoing for {mins} minutes.", ""])
        return r.choice(["ALERT: ", "Incident: ", "PagerDuty: ", ""]) + sym + scope + "." + tail

    def facts(self, s):
        ongoing = s["status"] == "ongoing"
        bad = s["symptom"] in ("down", "errors", "slow")
        impact = ongoing and bad and s["scope"] != "internal"
        return [
            Fact("impact", impact,
                 ["Customers are affected right now", "Users are currently feeling this problem",
                  "Is this hurting customers at the moment?"],
                 ["No customers are affected right now", "Users are not impacted at the moment"]),
            Fact("outage", ongoing and s["symptom"] == "down",
                 ["The service is down right now", "Is there a full outage?"],
                 ["The service is up", "There is no outage at the moment"]),
            Fact("resolved", not ongoing,
                 ["The incident has been resolved", "Is the problem already fixed?"],
                 ["The problem is still ongoing", "The incident is not resolved yet"]),
            Fact("page", impact and s["symptom"] in ("down", "errors"),
                 ["The on-call engineer should be woken up now", "This needs an immediate page"],
                 ["This can wait until business hours"]),
        ]


# ---------------------------------------------------------------------------------------------
class ProductReviews(Domain):
    name = "product-reviews"
    flips = {"stars": [1, 2, 3, 4, 5], "issue": ["none", "broke", "noisy", "smaller", "late"],
             "returned": [False, True]}
    implications = [("returned", "not:recommends"), ("recommends", "positive")]
    PRODUCTS = ["blender", "desk lamp", "air fryer", "vacuum cleaner", "electric kettle", "office chair",
                "rice cooker", "space heater", "coffee grinder", "bathroom scale"]

    def sample(self, rng):
        stars = rng.choice([1, 2, 3, 4, 5, 5])
        issue = "none" if stars >= 4 and rng.random() < 0.7 else rng.choice(["none", "broke", "noisy", "smaller", "late"])
        return {"_seed": rng.random(), "p": rng.choice(self.PRODUCTS), "stars": stars, "issue": issue,
                "returned": stars <= 2 and rng.random() < 0.5}

    def render(self, s):
        r = _r(s)
        mood = {1: r.choice(["Terrible", "Awful purchase", "Total waste of money"]),
                2: r.choice(["Disappointing", "Not great", "Below expectations"]),
                3: r.choice(["It's okay", "Average at best", "Mixed feelings"]),
                4: r.choice(["Pretty good", "Happy with it", "Solid choice"]),
                5: r.choice(["Love it", "Fantastic", "Best purchase this year"])}[s["stars"]]
        issue = {"none": "",
                 "broke": r.choice([f" The {s['p']} stopped working after a week.", f" The switch on the {s['p']} broke on day three."]),
                 "noisy": f" The {s['p']} is much louder than I expected.",
                 "smaller": f" The {s['p']} is smaller than it looks in the photos.",
                 "late": " Delivery took two weeks longer than promised."}[s["issue"]]
        verdict = {1: " Avoid.", 2: " Would not buy again.", 3: " Does the job.",
                   4: f" I'd recommend this {s['p']}.", 5: r.choice([" Highly recommend!", " Would buy again in a heartbeat."])}[s["stars"]]
        if s["stars"] >= 4 and s["issue"] == "broke":
            verdict = " The replacement works fine, so I'm mostly happy."
        ret = r.choice([" I sent it back for a refund.", " Returned it."]) if s["returned"] else ""
        stars = f"{s['stars']}/5. " if r.random() < 0.4 else ""
        return f"{stars}{mood}.{issue}{verdict}{ret}"

    def facts(self, s):
        positive = s["stars"] >= 4
        defect = s["issue"] in ("broke", "noisy")
        return [
            Fact("recommends", positive and not s["returned"],
                 ["The reviewer would recommend this product", "Would this customer suggest the product to a friend?",
                  "The reviewer is satisfied with the purchase"],
                 ["The reviewer would not recommend this product", "The reviewer advises against buying it"]),
            Fact("positive", positive,
                 ["The review is positive", "Is the overall tone of the review favourable?"],
                 ["The review is negative or lukewarm"]),
            Fact("defect", defect,
                 ["The product has a defect", "Does the reviewer report a fault with the product?"],
                 ["The product works as it should"]),
            Fact("returned", s["returned"],
                 ["The reviewer returned the product", "Was the item sent back?"],
                 ["The reviewer kept the product"]),
        ]


# ---------------------------------------------------------------------------------------------
class ExpenseClaims(Domain):
    name = "expense-claims"
    LIMITS = {"meals": 75, "taxi": 120, "hotel": 300, "software": 200, "alcohol": 0, "gift": 50}
    flips = {"amount": [18, 45, 90, 160, 280, 650, 1400], "category": list(LIMITS),
             "receipt": [False, True], "preapproved": [False, True]}
    implications = [("pay", "receipt"), ("pay", "policy")]

    def sample(self, rng):
        cat = rng.choice(list(self.LIMITS))
        return {"_seed": rng.random(), "category": cat, "amount": rng.choice([18, 45, 90, 160, 280, 650, 1400]),
                "receipt": rng.random() < 0.7, "preapproved": rng.random() < 0.3,
                "who": rng.choice(["Priya", "Tom", "Ana", "Kenji", "Lucas", "Fatima", "Olu", "Mei"])}

    def render(self, s):
        r = _r(s)
        what = {"meals": r.choice(["team lunch with a client", "dinner during the Leeds site visit"]),
                "taxi": r.choice(["taxi from the airport to the client office", "late-night cab home after the release"]),
                "hotel": r.choice(["two nights at the conference hotel", "hotel in Munich for the customer workshop"]),
                "software": r.choice(["annual licence for a diagramming tool", "a year of a screen-recording app"]),
                "alcohol": r.choice(["drinks at the bar after the offsite", "bottles of wine for the team celebration"]),
                "gift": r.choice(["a thank-you gift for a supplier", "flowers for a client's opening"])}[s["category"]]
        rec = r.choice([" Receipt attached.", " Itemised receipt uploaded."]) if s["receipt"] else r.choice(
            [" I lost the receipt.", " No receipt, the card machine was broken."])
        pre = r.choice([" My manager approved this in advance.", " Pre-approved by finance (ticket FIN-2231)."]) if s["preapproved"] else ""
        return f"Expense claim from {s['who']}: ${s['amount']} for {what}.{rec}{pre}"

    def facts(self, s):
        policy = s["category"] != "alcohol" and (s["amount"] <= self.LIMITS[s["category"]] or s["preapproved"])
        pay = policy and s["receipt"]
        return [
            Fact("policy", policy,
                 ["The expense is within policy", "Is this claim allowed under the expense rules?"],
                 ["The expense breaks the expense policy", "This claim is outside policy"]),
            Fact("receipt", s["receipt"],
                 ["A receipt was provided", "Did the employee attach a receipt?"],
                 ["The receipt is missing", "No receipt was provided"]),
            Fact("pay", pay,
                 ["This claim can be reimbursed as submitted", "Should finance pay this claim out?"],
                 ["This claim cannot be paid as submitted"]),
            Fact("large", s["amount"] >= 500,
                 ["This is a large expense of $500 or more", "Is the amount over $500?"],
                 ["The amount is under $500"]),
        ]


# ---------------------------------------------------------------------------------------------
class MeetingRequests(Domain):
    name = "meeting-requests"
    flips = {"when": ["today", "this_week", "next_month"], "mandatory": [False, True], "online": [False, True],
             "rsvp": [False, True]}
    implications = []

    def sample(self, rng):
        return {"_seed": rng.random(), "when": rng.choice(["today", "this_week", "next_month"]),
                "mandatory": rng.random() < 0.4, "online": rng.random() < 0.5, "rsvp": rng.random() < 0.5,
                "topic": rng.choice(["Q3 planning", "the vendor review", "onboarding redesign", "the security audit",
                                     "budget sign-off", "the product roadmap"])}

    def render(self, s):
        r = _r(s)
        when = {"today": r.choice(["today at 3pm", "this afternoon at 4:30"]),
                "this_week": r.choice(["on Thursday at 10am", "this Friday at 2pm"]),
                "next_month": r.choice(["on the 14th of next month", "early next month"])}[s["when"]]
        where = r.choice([" on Zoom", " over Teams (link below)"]) if s["online"] else r.choice(
            [" in meeting room 4B", " at the head office, third floor"])
        must = r.choice([" Attendance is mandatory for all team leads.", " Everyone on the team must attend."]) if s["mandatory"] else r.choice(
            [" Optional - join if the topic is relevant to you.", " Feel free to skip if you're busy."])
        rsvp = r.choice([" Please accept or decline so we can plan seating.", " RSVP by tomorrow, please."]) if s["rsvp"] else ""
        return f"Meeting about {s['topic']} {when}{where}.{must}{rsvp}"

    def facts(self, s):
        return [
            Fact("mandatory", s["mandatory"],
                 ["Attendance is required", "Do I have to attend this meeting?"],
                 ["Attendance is optional", "The meeting can be skipped"]),
            Fact("rsvp", s["rsvp"],
                 ["The organiser asks for a reply", "Is a response to the invite requested?"],
                 ["No reply to the invitation is requested"]),
            Fact("today", s["when"] == "today",
                 ["The meeting is today", "Is this meeting happening today?"],
                 ["The meeting is not today"]),
            Fact("online", s["online"],
                 ["The meeting is held online", "Can I join this meeting remotely?"],
                 ["The meeting is in person", "Attendees need to be physically present"]),
        ]


# ---------------------------------------------------------------------------------------------
class CodeChanges(Domain):
    name = "code-changes"
    flips = {"change": ["docs", "tests", "refactor", "migration", "auth", "remove_api"],
             "tests": ["added", "none", "failing"], "reviewed": [False, True]}
    implications = [("failing", "not:merge"), ("merge", "reviewed"), ("breaking", "not:merge")]

    def sample(self, rng):
        return {"_seed": rng.random(), "change": rng.choice(["docs", "tests", "refactor", "migration", "auth", "remove_api"]),
                "tests": rng.choice(["added", "added", "none", "failing"]), "reviewed": rng.random() < 0.6}

    def render(self, s):
        r = _r(s)
        what = {"docs": r.choice(["Fix typos in the README and update the setup guide", "Document the new config flags"]),
                "tests": r.choice(["Add unit tests for the invoice parser", "Increase coverage of the date helpers"]),
                "refactor": r.choice(["Refactor the cart module into smaller functions, no behaviour change",
                                      "Rename internal variables in the scheduler for clarity"]),
                "migration": r.choice(["Database migration that drops the legacy `phone` column from `users`",
                                       "Migration renaming the `orders.total` column to `orders.amount`"]),
                "auth": r.choice(["Change how session tokens are validated in the login service",
                                  "Switch password hashing from bcrypt to argon2"]),
                "remove_api": r.choice(["Remove the deprecated /v1/export endpoint that two partners still call",
                                        "Delete the public `getUserById` method from the SDK"])}[s["change"]]
        tests = {"added": r.choice([" Tests added, CI is green.", " New tests included and all checks pass."]),
                 "none": r.choice([" No tests in this PR; CI is green.", " Didn't add tests, existing checks pass."]),
                 "failing": r.choice([" CI is red: 3 tests fail.", " The integration tests are currently failing."])}[s["tests"]]
        rev = r.choice([" Approved by two reviewers.", " Reviewed and approved by the module owner."]) if s["reviewed"] else r.choice(
            [" Waiting for review.", " Nobody has reviewed it yet."])
        return f"Pull request: {what}.{tests}{rev}"

    def facts(self, s):
        breaking = s["change"] in ("migration", "remove_api")
        merge = s["reviewed"] and s["tests"] != "failing" and not breaking and \
            not (s["change"] == "auth" and s["tests"] == "none")
        return [
            Fact("merge", merge,
                 ["This pull request is ready to merge", "Can this be merged now?"],
                 ["This pull request should not be merged yet", "The change is not ready to go in"]),
            Fact("failing", s["tests"] == "failing",
                 ["Tests are failing on this change", "Is CI broken for this pull request?"],
                 ["All checks pass on this change"]),
            Fact("reviewed", s["reviewed"],
                 ["The change has been reviewed", "Has someone approved this pull request?"],
                 ["The change has not been reviewed yet"]),
            Fact("breaking", breaking,
                 ["This change can break existing users or data", "Is this a breaking change?"],
                 ["The change is backwards compatible", "Nothing existing breaks with this change"]),
            Fact("security", s["change"] == "auth",
                 ["The change touches authentication or security", "Does this affect how users log in?"],
                 ["The change has nothing to do with security"]),
        ]


# ---------------------------------------------------------------------------------------------
class InsuranceClaims(Domain):
    name = "insurance-claims"
    flips = {"injury": [False, True], "drivable": [False, True], "police": [False, True],
             "fault": ["other", "self", "unknown"]}
    implications = [("tow", "vehicle")]

    def sample(self, rng):
        return {"_seed": rng.random(), "kind": rng.choice(["collision", "collision", "theft", "water", "hail"]),
                "injury": rng.random() < 0.3, "drivable": rng.random() < 0.6, "police": rng.random() < 0.5,
                "fault": rng.choice(["other", "self", "unknown"])}

    def render(self, s):
        r = _r(s)
        k = s["kind"]
        if k == "collision":
            ev = {"other": r.choice(["A van ran a red light and hit my car", "Another driver reversed into me in a car park"]),
                  "self": r.choice(["I misjudged the turn and hit a lamp post", "I rear-ended the car in front in slow traffic"]),
                  "unknown": r.choice(["Two cars collided at the junction and it's unclear who had right of way",
                                       "My car was hit at a roundabout; both of us say the other pulled out"])}[s["fault"]]
            drv = r.choice([" The car still drives fine.", " I drove it home afterwards."]) if s["drivable"] else r.choice(
                [" The car can't be driven, the front axle is bent.", " The car won't start and is stuck at the scene."])
        elif k == "theft":
            ev, drv = r.choice(["My bike was stolen from the garage overnight", "My laptop was taken from my car"]), ""
        elif k == "water":
            ev, drv = r.choice(["A pipe burst upstairs and flooded my kitchen", "Water came through the ceiling during the storm"]), ""
        else:
            ev, drv = r.choice(["Hail dented the roof and cracked two windows", "A hailstorm smashed the conservatory roof"]), ""
        inj = r.choice([" My passenger hurt her neck and went to hospital.", " I have a cut on my head and a sore wrist."]) if s["injury"] else r.choice(
            [" Nobody was hurt.", " No injuries."])
        pol = r.choice([" Police attended and gave me a report number.", " I reported it to the police (ref 55-1023)."]) if s["police"] else r.choice(
            [" I haven't contacted the police.", ""])
        return f"{ev}.{drv}{inj}{pol}"

    def facts(self, s):
        vehicle = s["kind"] == "collision"
        return [
            Fact("injury", s["injury"],
                 ["Someone was injured", "Was anybody hurt?"],
                 ["Nobody was injured", "There were no injuries"]),
            Fact("vehicle", vehicle,
                 ["This claim involves a car accident", "Is this a motor claim?"],
                 ["No vehicle collision is involved"]),
            Fact("tow", vehicle and not s["drivable"],
                 ["The vehicle needs to be towed", "Does the car need recovery?"],
                 ["No tow truck is needed"]),
            Fact("police", s["police"],
                 ["The police were informed", "Is there a police report?"],
                 ["The police have not been contacted"]),
            Fact("other_fault", vehicle and s["fault"] == "other",
                 ["Another driver caused the accident", "Was the other party at fault?"]),
        ]


# ---------------------------------------------------------------------------------------------
class SalesInquiries(Domain):
    name = "sales-inquiries"
    flips = {"intent": ["pricing", "demo", "support", "job", "student"], "budget": [False, True],
             "timeline": [False, True]}
    implications = [("hot", "buy"), ("hot", "budget"), ("not_customer", "not:buy")]

    def sample(self, rng):
        return {"_seed": rng.random(), "intent": rng.choice(["pricing", "pricing", "demo", "demo", "support", "job", "student"]),
                "budget": rng.random() < 0.5, "timeline": rng.random() < 0.5,
                "org": rng.choice(["a 40-person logistics firm", "a regional hospital group", "a three-person design studio",
                                   "a mid-size retailer", "a university IT department"])}

    def render(self, s):
        r = _r(s)
        ask = {"pricing": r.choice([f"We're {s['org']} and want pricing for the enterprise plan.",
                                    f"I run purchasing at {s['org']}; please send a quote for 50 seats."]),
               "demo": r.choice([f"We're {s['org']} evaluating tools like yours and would like a demo.",
                                 f"Can we book a product walkthrough for our team at {s['org']}?"]),
               "support": r.choice(["We're existing customers and the export button stopped working.",
                                    "How do I reset my password? I'm already a customer."]),
               "job": r.choice(["Are you hiring? I'd love to join your sales team, CV attached.",
                                "I'm applying for the account executive role you posted."]),
               "student": r.choice(["I'm a student writing a thesis on SaaS pricing, could you answer a few questions?",
                                    "For a class project, can you tell me how your company was founded?"])}[s["intent"]]
        buying = s["intent"] in ("pricing", "demo")
        bud = (r.choice([" We have about $40k budgeted.", " Budget is approved at roughly $15k a year."]) if s["budget"] else "") if buying else ""
        tl = (r.choice([" We need to decide by the end of this month.", " Looking to roll out this quarter."]) if s["timeline"] else "") if buying else ""
        return ask + bud + tl

    def facts(self, s):
        buy = s["intent"] in ("pricing", "demo")
        budget, timeline = buy and s["budget"], buy and s["timeline"]
        return [
            Fact("buy", buy,
                 ["The sender is interested in buying", "Is this a potential customer?"],
                 ["The sender is not looking to buy anything"]),
            Fact("budget", budget,
                 ["The message mentions a budget", "Did they say how much they can spend?"],
                 ["No budget is mentioned"]),
            Fact("hot", buy and budget and timeline,
                 ["This is a hot lead with budget and a deadline", "Should sales call them today?"],
                 ["This is not a hot lead"]),
            Fact("not_customer", s["intent"] in ("job", "student"),
                 ["The sender is a job seeker or a student", "Is this message from someone who isn't a buyer or customer?"],
                 ["The sender is a customer or a prospective buyer"]),
        ]


NEW_DOMAINS = [ITIncidents(), ProductReviews(), ExpenseClaims(), MeetingRequests(), CodeChanges(),
               InsuranceClaims(), SalesInquiries()]
