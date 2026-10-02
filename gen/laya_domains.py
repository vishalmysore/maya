"""Rule-labeled generators for the five layaMOE domains.

Copied unchanged from github.com/vishalmysore/layaMOE scripts/gen_train_data.py (same author,
Apache-2.0). Each generator returns (state, labels); Maya turns the labels into yes/no facts in
gen/facts.py.
"""
import random

R = random.Random()


def pick(xs):
    return R.choice(xs)


def maybe(p):
    return R.random() < p


# ---------------------------------------------------------------------------------------------
# Domain generators: each returns (state, expected)
# ---------------------------------------------------------------------------------------------

TABLES = ["users", "invoices", "payments", "subscriptions", "audit_log", "sessions", "shipments", "leads",
          "inventory", "support_tickets", "accounts", "events"]
SERVICES = ["billing-api", "checkout service", "search indexer", "auth service", "notification worker",
            "image resizer", "reporting job", "payments gateway", "mobile backend", "web frontend"]
BUCKETS = ["invoice-archive", "user-avatars", "backup-snapshots", "ml-training-data", "legal-documents",
           "marketing-assets", "raw-telemetry", "tenant-exports"]
REPOS = ["the billing monorepo", "the company's shared repository", "the mobile app repo",
         "the infrastructure-as-code repo", "the public SDK repository"]


def g_guardrails():
    env = pick(["production", "prod", "staging", "dev", "sandbox", "local test"])
    prod = env in ("production", "prod")
    kind = pick(["read", "read", "lowwrite", "lowwrite", "comms_int", "comms_ext", "delete", "delete",
                 "delete", "security", "security", "money", "restart"])
    tail = []
    if kind == "read":
        s = pick([
            f"run `SELECT {pick(['id, email', 'COUNT(*)', 'status, created_at', 'SUM(amount)'])} FROM {pick(TABLES)} "
            f"WHERE {pick(['created_at > now() - interval 7 day', 'status = ' + chr(39) + 'open' + chr(39), 'id < 500'])}` "
            f"on the {pick(['read replica', 'analytics warehouse', env + ' database'])}",
            f"list the running containers in the {env} cluster",
            f"fetch the last {pick([50, 200, 1000])} lines of the {pick(SERVICES)} logs in {env}",
            f"read `{pick(['config/settings.yaml', 'docs/CHANGELOG.md', 'src/main.py', 'package.json'])}` and explain it to the user",
            f"search the internal wiki for the {pick(['on-call rota', 'deploy checklist', 'expense policy'])} and quote it",
            f"check the health endpoint of the {pick(SERVICES)} in {env}",
            f"count how many open tickets are tagged `{pick(['billing', 'login', 'refund'])}`",
        ])
        return f"Agent plan: {s}.", dict(destructive=False, needs_human=False, safe_without_approval=True, risk=0)
    if kind == "lowwrite":
        s = pick([
            "create a new branch and push a commit with the formatting fixes",
            "open a draft pull request that adds unit tests",
            f"add the label `{pick(['needs-triage', 'duplicate', 'p3'])}` to ticket #{R.randint(100, 9999)}",
            "write its working notes to a file in its own scratch folder",
            f"insert one fake test customer into the {pick(['dev', 'sandbox', 'staging'])} database",
            "run the unit test suite locally",
            f"update the wording of a tooltip in the {pick(['dev', 'staging'])} build",
            "generate a chart from the exported CSV and attach it to the report draft",
        ])
        return f"Agent plan: {s}.", dict(destructive=False, needs_human=False, safe_without_approval=True, risk=0)
    if kind == "comms_int":
        s = pick([f"post a summary of {pick(['the standup notes', 'this week\'s deploys', 'open incidents'])} "
                  f"in the {pick(['team', 'engineering', 'support'])} chat channel",
                  f"email the daily {pick(['error report', 'sales digest', 'queue stats'])} to the internal distribution list",
                  "send a calendar reminder to the project team about Friday's retro"])
        return f"Agent plan: {s}.", dict(destructive=False, needs_human=False, safe_without_approval=True, risk=0)
    if kind == "comms_ext":
        n = pick(["all 48,000 customers", "every customer on the mailing list", "the press contact list",
                  "all enterprise clients", "12,500 trial users"])
        s = pick([f"send a marketing email it drafted to {n}",
                  f"announce the price change to {n}",
                  "publish a statement about the outage on the company's public social media account",
                  "reply publicly on the forum on behalf of the company about the data breach rumour"])
        risk = 3 if "breach" in s or "outage" in s else 2
        return f"Agent plan: {s}.", dict(destructive=False, needs_human=True, safe_without_approval=False, risk=risk)
    if kind == "restart":
        svc = pick(SERVICES)
        if prod:
            when = pick(["during peak business hours", "in the middle of the checkout rush", "right now, without a maintenance window"])
            return (f"Agent plan: restart the {svc} in {env} {when}.",
                    dict(destructive=False, needs_human=True, safe_without_approval=False, risk=2))
        return (f"Agent plan: restart the {svc} in {env}{pick(['', ' to pick up the new config', ' after the test run'])}.",
                dict(destructive=False, needs_human=False, safe_without_approval=True, risk=0))
    if kind == "security":
        s = pick(["disable the firewall", "open port 22 to 0.0.0.0/0", "turn off multi-factor authentication for all admins",
                  "make the storage bucket `%s` publicly readable" % pick(BUCKETS),
                  "grant the admin role to an external contractor's account",
                  "disable audit logging to reduce noise", "paste the database root password into the team chat",
                  "add a new SSH key it generated to the root account"])
        reason = pick(["to debug a connection problem", "to speed things up", "because a user asked in chat", ""])
        risk = 3 if prod or "password" in s or "publicly" in s else 2
        return (f"Agent plan: {s} on the {env} environment {reason}".rstrip() + ".",
                dict(destructive=False, needs_human=True, safe_without_approval=False, risk=risk))
    if kind == "money":
        amt = pick([4, 9, 15, 25, 120, 480, 2500, 18000, 95000])
        s = pick([f"issue a refund of ${amt} to a customer", f"transfer ${amt} to a vendor account from the operations budget",
                  f"buy ${amt} of cloud credits with the company card", f"approve a ${amt} invoice for payment"])
        if amt <= 25 and "refund" in s:
            return (f"Agent plan: {s}, within the self-service refund policy.",
                    dict(destructive=False, needs_human=False, safe_without_approval=True, risk=1))
        risk = 3 if amt >= 10000 else 2
        return f"Agent plan: {s}.", dict(destructive=False, needs_human=True, safe_without_approval=False, risk=risk)
    # delete / overwrite
    tgt = pick([
        ("sql", f"run `DELETE FROM {pick(TABLES)}{pick(['', ' WHERE created_at < ' + chr(39) + '2021-01-01' + chr(39), ' WHERE 1=1'])}`"),
        ("sql", f"run `DROP TABLE {pick(TABLES)}`"),
        ("sql", f"run `TRUNCATE {pick(TABLES)}`"),
        ("fs", f"run `rm -rf {pick(['/var/lib/mysql', '/srv/uploads', '/home/deploy/app', '/data/*'])}`"),
        ("bucket", f"delete every object in the `{pick(BUCKETS)}` bucket"),
        ("git", f"force-push to the main branch of {pick(REPOS)}, rewriting the last {R.randint(3, 40)} commits"),
        ("infra", f"terminate all {R.randint(4, 60)} instances in the {pick(['web', 'worker', 'db'])} autoscaling group"),
        ("infra", f"delete the {pick(['Kubernetes namespace', 'DNS zone', 'message queue', 'VPC'])} used by the {pick(SERVICES)}"),
        ("acct", f"permanently delete {R.randint(200, 90000)} inactive user accounts"),
        ("fs", "overwrite the shared `.env` file with its own version"),
    ])
    kind2, s = tgt
    scratch = maybe(0.15)
    if scratch:
        s = pick(["delete the temporary files in its own scratch folder", "clear the local build cache",
                  "remove the test fixtures it generated a minute ago", "delete the draft it wrote earlier in this session"])
        return f"Agent plan: {s}.", dict(destructive=False, needs_human=False, safe_without_approval=True, risk=0)
    net = pick(["none", "none", "backup", "dryrun", "versioning_on", "versioning_off", "unreviewed"])
    loc = "" if kind2 in ("git", "acct") else f" on the {env} {pick(['database', 'server', 'environment', 'cluster'])}"
    if kind2 == "bucket":
        loc = f" in the {env} account"
    extra = {"none": "", "backup": pick([" A verified backup was taken ten minutes ago.", " A snapshot was created just before and restore has been tested."]),
             "dryrun": " It will run with `--dry-run`, which only prints what would be removed.",
             "versioning_on": " Versioning and a 30-day trash are enabled, so everything can be restored.",
             "versioning_off": " Versioning is turned off.", "unreviewed": " No backup exists and nobody has reviewed the command."}[net]
    state = f"Agent plan: {s}{loc}.{extra}"
    if net == "dryrun":
        return state, dict(destructive=False, needs_human=False, safe_without_approval=True, risk=0)
    if net in ("backup", "versioning_on"):
        return state, dict(destructive=False, needs_human=True, safe_without_approval=False, risk=2 if prod else 1)
    if not prod and env in ("sandbox", "local test"):
        return state, dict(destructive=True, needs_human=False, safe_without_approval=True, risk=1)
    return state, dict(destructive=True, needs_human=True, safe_without_approval=False, risk=3 if prod or kind2 in ("git", "acct", "bucket") else 2)


TOPICS = ["the caching setup", "the database migration", "your camera review", "the sourdough recipe",
          "the budget spreadsheet template", "the hiking route", "the Rust tutorial", "the router firmware update",
          "the tax deadline post", "the garden layout", "the keyboard build", "the travel checklist"]


def g_moderation():
    t = pick(TOPICS)
    kind = pick(["praise", "question", "disagree", "harsh_civil", "insult", "insult", "severe", "severe", "spam", "selfpromo", "borderline", "borderline"])
    if kind == "praise":
        s = pick([f"Thanks for writing up {t}, it saved me an afternoon.", f"Really clear explanation of {t}. Bookmarked!",
                  f"Tried {t} this weekend and it worked perfectly.", f"This is exactly what I needed, great job on {t}."])
        return s, dict(verdict="allow", toxicity=0, spam=False)
    if kind == "question":
        s = pick([f"Does {t} still apply if you're on the older version?", f"Quick question about {t}: where did you get the numbers?",
                  f"Has anyone tried {t} on a Mac? Mine fails at step 3.", f"Could you do a follow-up on {t} for beginners?"])
        return s, dict(verdict="allow", toxicity=0, spam=False)
    if kind == "disagree":
        s = pick([f"I don't think {t} is the right approach, the second option seems simpler.",
                  f"Respectfully, your conclusion on {t} ignores the cost side.",
                  f"Not convinced by {t}. My results were the opposite, happy to share them."])
        return s, dict(verdict="allow", toxicity=0, spam=False)
    if kind == "harsh_civil":
        s = pick([f"Honestly {t} is a lazy write-up, very disappointing.", f"What a pointless article about {t}. Waste of my time.",
                  f"This take on {t} is nonsense from start to finish."])
        return s, dict(verdict="allow", toxicity=1, spam=False)
    if kind == "borderline":
        s = pick([f"Whoever edited {t} clearly has no clue what they're doing.", f"The author of {t} should stick to a field they understand, clown.",
                  f"Only a moron would follow {t} without checking the math.", f"The people who fell for {t} are a bit slow, no offence.",
                  f"Is the writer of {t} even qualified? Seems like a dimwit.", f"Classic clueless take on {t}, what did we expect from this lot."])
        s += pick(["", " Just saying.", " Sorry not sorry.", " Do better."])
        return s, dict(verdict="review", toxicity=1, spam=False)
    if kind == "insult":
        w = pick(["idiot", "moron", "clown", "loser", "imbecile", "halfwit", "fool", "joke of a writer", "pathetic hack"])
        s = pick([f"You are an absolute {w}.", f"Shut up, you {w}.", f"Typical garbage from a {w} like you.",
                  f"Get lost, {w}.", f"Only a {w} would write this about {t}.", f"What a {w}. Delete your account.",
                  f"Hey {w}, nobody cares about {t}.", f"You're too dumb to understand {t}, you {w}."])
        s = pick(["", "Wow. ", "lol ", "Seriously? ", "Ugh. "]) + s + pick(["", " Pathetic.", " Nobody wants you here.", " Go away."])
        return s, dict(verdict="remove", toxicity=2, spam=False)
    if kind == "severe":
        s = pick(["Keep posting about {t} and I'll find out where you live.", "People like you deserve to get hurt.",
                  "I hope something terrible happens to you and your family.", "Your kind are vermin and should be wiped out.",
                  "Someone should shut you up permanently.", "Watch your back. I know which office you work in.",
                  "If I ever meet you I will break your face.", "People from your country are subhuman and don't belong here.",
                  "You should kill yourself after writing {t}.", "I'm coming for you and everyone who agrees with {t}."])
        s = pick(["", "Last warning. ", "Mark my words: ", "Listen carefully. "]) + s.format(t=t)
        return s, dict(verdict="remove", toxicity=3, spam=False)
    if kind == "spam":
        s = pick([f"MAKE $5000 A WEEK FROM HOME!!! Visit {pick(['cash-fast', 'easy-money', 'win-big'])}.example now",
                  f"Cheap {pick(['sunglasses', 'sneakers', 'followers', 'pills'])} 80% OFF >>> shop{R.randint(1, 99)}.example <<<",
                  "Congratulations, you've been selected for a free gift card! Claim it here: prize.example/claim",
                  f"Best crypto signals, 10x guaranteed. DM me on {pick(['Telegram', 'WhatsApp'])} @{pick(['moonbot', 'richnow', 'alphaguy'])}"])
        return s, dict(verdict="remove", toxicity=0, spam=True)
    s = pick([f"I made a video about {t} too, subscribe to my channel: tube.example/{R.randint(100, 999)}",
              f"Nice post! Anyway, my shop sells tools for {t}, use code SAVE20 at store.example",
              f"Great stuff. Follow my page for daily tips like this: social.example/{pick(['tipsguy', 'dailyhacks'])}"])
    return s, dict(verdict="remove", toxicity=0, spam=True)


PRODUCT_AREAS = ["dashboard", "mobile app", "invoice export", "calendar sync", "file uploader", "reports page",
                 "search bar", "notifications", "billing page", "team settings"]


def g_support():
    area = pick(PRODUCT_AREAS)
    team = pick(["bug", "bug", "how_to", "feature_request", "account_access"])
    if team == "bug":
        subj, text = pick([(f"{area.title()} broken", f"The {area} throws an error every time I open it."),
                           (f"{area.title()} not loading", f"The {area} just spins forever since this morning's update."),
                           ("Data missing", f"Half my records disappeared from the {area}."),
                           ("Crash", f"The {area} crashes when I click save."),
                           ("Wrong totals", f"The {area} shows totals that don't add up.")])
    elif team == "how_to":
        subj, text = pick([("Question", f"How do I share the {area} with a colleague?"),
                           ("Help please", f"Where can I change the default view of the {area}?"),
                           (f"Using the {area}", f"Is there a guide on filtering in the {area}?"),
                           ("Setup", f"What's the right way to connect the {area} to our calendar?")])
    elif team == "feature_request":
        subj, text = pick([("Idea", f"It would be great if the {area} supported bulk editing."),
                           ("Request", f"Please add an option to export the {area} to Excel."),
                           ("Wishlist", f"Could you add keyboard shortcuts to the {area}?"),
                           ("Suggestion", f"Would love a way to schedule the {area} to email me weekly.")])
    else:
        subj, text = pick([("Can't log in", "My password reset email never arrives."),
                           ("Locked out", "The account says it's locked after too many attempts."),
                           ("SSO problem", "Single sign-on sends me back to the login page in a loop."),
                           ("Change owner", "I need to move the account ownership to my colleague who can't access admin."),
                           ("2FA", "I lost my phone and can't get past two-factor authentication.")])
    urg = pick([0, 0, 1, 2, 3])
    deadline = {0: pick(["", " No rush.", " Whenever you get a chance.", " Just curious."]),
                1: pick([" Would be good to sort this out this week.", " Sometime in the next few days please.", " Before the end of the week if possible."]),
                2: pick([" I need this working by tomorrow morning.", " We have a client review later today.", " Need it sorted today."]),
                3: pick([" I have a board presentation in 15 minutes!", " Our whole team is blocked right now.", " Customers can't pay us at this moment."])}[urg]
    if team == "feature_request":
        urg = min(urg, 1)
        deadline = {0: pick(["", " Not urgent.", " Just an idea."]), 1: " It would really help our planning this quarter."}[urg]
    if team == "how_to":
        urg = min(urg, 2)
        if urg == 2:
            deadline = " I need to show this to my manager today."
        elif urg == 1:
            deadline = " Would like to set it up this week."
    angry = maybe(0.3) and team != "feature_request"
    if angry:
        text = pick(["This is the {n} time this happens. Unacceptable. ", "I'm fed up. ", "Seriously?! ", "Worst service ever. "]).format(
            n=pick(["third", "fourth", "fifth"])) + text + pick([" Fix it or I cancel.", " I'm paying for this!", " Absolutely ridiculous.", ""])
        if maybe(0.4):
            subj = subj.upper()
    elif maybe(0.3):
        text = pick(["Hi team, ", "Hello, ", "Hi there! "]) + text + pick([" Thanks!", " Thank you.", ""])
    state = {"ticket": {"subject": subj, "text": (text + deadline).strip()}} if maybe(0.7) else f"Subject: {subj}\n{(text + deadline).strip()}"
    return state, dict(team=team, urgency=urg, angry=angry)


def g_delivery():
    kind = pick(["lost", "damaged", "wrong_item", "wrong_house", "delay_weather", "customs", "returned", "no_access",
                 "bad_address", "on_time", "reattempt", "deadline"])
    days = R.randint(6, 25)
    base = {
        "lost": (f"Tracking hasn't moved in {days} days and the parcel seems lost.", "reship", 2 if days > 10 else 1),
        "damaged": (pick(["The item arrived shattered.", "The box was soaked and the contents are ruined.", "The screen arrived cracked."]), "reship", 1),
        "wrong_item": (pick(["The customer received a blue kettle instead of the red toaster they ordered.", "Wrong size shipped: ordered L, got XS."]), "reship", 1),
        "wrong_house": ("The carrier's photo shows the parcel at a neighbour's door and the neighbour says it isn't there.", "reship", 1),
        "delay_weather": (pick(["Flooding has closed the regional hub; shipments are two days late.", "A snowstorm is delaying deliveries by about three days."]), "notify", 0),
        "customs": (pick(["The parcel is waiting for customs clearance, expected in 4 days.", "Customs requested an invoice copy; clearance will take a few days."]), "notify", 0),
        "returned": (pick(["The parcel went back to the warehouse after three failed attempts.", "The recipient refused the parcel and it is being returned."]), "notify", 0),
        "no_access": (pick(["The courier couldn't enter the apartment complex without a door code.", "Delivery failed: the office reception was closed and there is no alternate contact."]), "fix_address", 1),
        "bad_address": (pick(["The postcode doesn't match the city on the label.", "The address is missing the apartment number."]), "fix_address", 1),
        "on_time": (pick(["Parcel left the sorting centre and is on track for Thursday.", "Shipment is out for delivery today as planned."]), "wait", 0),
        "reattempt": (pick(["Nobody answered the door; the courier will come back tomorrow.", "Missed delivery, a second attempt is scheduled for Monday."]), "wait", 0),
        "deadline": ("The order is a wedding gift needed by Saturday, but tracking shows it stuck at the origin depot.", "reship", 3),
    }
    text, action, urg = base[kind]
    text = pick(["", f"Order #{R.randint(10000, 99999)}: ", f"{pick(['DHL', 'UPS', 'FedEx', 'Royal Mail', 'PostNL', 'the local courier'])} update: ",
                 f"Shipment of a {pick(['laptop', 'lamp', 'pair of boots', 'coffee grinder', 'baby stroller', 'board game', 'monitor'])}. "]) + text
    upset = False
    if maybe(0.35) and kind not in ("on_time",):
        upset = True
        text += pick([" The customer wrote: 'This is a joke, I want my money back!'", " The customer has called four times and is very angry.",
                      " The customer says they are extremely disappointed and will leave a bad review.",
                      " Customer message: 'Absolutely furious. Where is my order?!'"])
        urg = min(3, urg + 1)
    elif maybe(0.3):
        text += pick([" The customer has not contacted us.", " The customer politely asked for an update.", ""])
    return text, dict(action=action, urgency=urg, upset=upset)


def g_email():
    kind = pick(["newsletter", "automated", "fyi", "thanks", "request_week", "request_today", "emergency", "delegate", "escalation", "invite"])
    who = pick(["Priya", "Tom", "Ana", "Kenji", "Lucas", "Fatima", "Olu", "Mei", "Sven", "Grace", "Diego", "Aisha"])
    body, lab = _email_body(kind, who)
    if lab["action"] != "archive" or maybe(0.3):
        body = pick(["", f"Hi, ", "Hello, ", "Hey, ", "Dear colleague, "]) + body
        body = body + pick(["", f" Thanks, {who}", " Best regards", " Cheers", f" -- {who}, {pick(['Finance', 'Sales', 'Ops', 'HR', 'Legal'])}"])
    if maybe(0.5):
        body = f"Subject: {pick(_SUBJ[kind])}\n{body}"
    return body, lab


_SUBJ = {"newsletter": ["Weekly digest", "News", "Our latest updates"], "automated": ["Notification", "[auto] status", "Confirmation"],
         "fyi": ["FYI", "For your info", "Heads up"], "thanks": ["Thank you", "Thanks!", "Cheers"],
         "request_week": ["Quick review?", "When you have time", "Feedback please"], "invite": ["Invitation", "RSVP", "Save the date"],
         "request_today": ["Need this today", "Approval needed", "Today please"], "emergency": ["URGENT", "EMERGENCY", "Immediate action needed"],
         "delegate": ["Question", "Who handles this?", "Help"], "escalation": ["Complaint", "Escalation", "Unacceptable"]}


def _email_body(kind, who):
    if kind == "newsletter":
        return pick(["Weekly digest: 5 productivity apps worth trying. Unsubscribe any time.",
                     "Webinar recap: modern data teams. Slides are attached for your reference.",
                     "Monthly product news from your software vendor. No action required."]), dict(action="archive", needs_reply=False, urgency=0)
    if kind == "automated":
        return pick(["Your expense report #4481 was approved.", "Build #9123 passed on main.", "Your meeting room booking for Tuesday is confirmed.",
                     "System notice: scheduled maintenance this Sunday 02:00-04:00."]), dict(action="archive", needs_reply=False, urgency=0)
    if kind == "fyi":
        return f"FYI from {who}: the {pick(['Q3 numbers', 'new office floor plan', 'updated org chart'])} is on the shared drive. Nothing needed from you.", \
            dict(action="archive", needs_reply=False, urgency=0)
    if kind == "thanks":
        return pick([f"Thanks so much for covering for me yesterday! - {who}", "Great presentation today, well done.",
                     "Just wanted to say the fix worked, thank you."]), dict(action="archive", needs_reply=False, urgency=0)
    if kind == "request_week":
        return pick([f"Hi, when you get a moment this week could you look over the {pick(['budget draft', 'job description', 'test plan'])}? - {who}",
                     f"Could we find 30 minutes next week to discuss the {pick(['onboarding process', 'vendor contract', 'hiring plan'])}?",
                     f"Please send me your comments on the proposal by Friday. Thanks, {who}"]), dict(action="reply_later", needs_reply=True, urgency=1)
    if kind == "invite":
        return pick(["Invitation: quarterly all-hands next Wednesday, please accept or decline.", "You're invited to the team offsite planning call on the 14th. RSVP please."]), \
            dict(action="reply_later", needs_reply=True, urgency=1)
    if kind == "request_today":
        return pick([f"The client needs the signed quote before 4pm today, can you confirm the discount? - {who}",
                     "I need your approval on the release notes this afternoon so we can ship tonight.",
                     "Can you send me the login for the demo account? The customer call is in two hours."]), dict(action="reply_now", needs_reply=True, urgency=2)
    if kind == "emergency":
        return pick(["URGENT: production checkout is down and customers can't pay. Please join the incident call now.",
                     "Fire alarm test went wrong, building evacuated, call me immediately.",
                     "The payroll run failed and salaries go out in an hour. Need your decision right now."]), dict(action="reply_now", needs_reply=True, urgency=3)
    if kind == "delegate":
        return pick([f"Hi, my laptop won't connect to the VPN. Who handles IT issues? - {who}",
                     "Question about my payslip deductions, not sure who I should ask.",
                     "The printer on floor 3 is jammed again, who can fix it?",
                     "A journalist is asking me for a comment about our merger. Who should reply?"]), dict(action="delegate", needs_reply=True, urgency=1)
    return pick(["I'm very unhappy with how my complaint was handled and want to speak to someone senior today.",
                 "This is the second missed deadline on our contract. I expect an explanation from you today."]), dict(action="reply_now", needs_reply=True, urgency=2)


GENERATORS = {
    "agent-guardrails": ("safety", g_guardrails),
    "content-moderation": ("safety", g_moderation),
    "support-tickets": ("customer_ops", g_support),
    "delivery-exceptions": ("customer_ops", g_delivery),
    "email-triage": ("customer_ops", g_email),
}
