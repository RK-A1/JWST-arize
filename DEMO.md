# 15-minute walkthrough

A demo script, not documentation. Timings are what it actually takes; the
numbers are from the runs recorded in the README.

**Before you start.** Have `phoenix serve` running with both projects already
populated (`jwst-golden`, `jwst-production`), two browser tabs open on them, and
a terminal in the repo root. Do not run the traffic live — 200 questions is two
minutes of dead air.

---

## 0:00 — 2:00 · The setup

> "This is an agent that answers questions about a corpus of 979 James Webb
> Space Telescope photos. It has three tools: search, fetch one record, count by
> category. Standard retrieval agent — yours probably looks like this."

Show [`src/jwst_arize/agent.py`](src/jwst_arize/agent.py). Sixty lines of
LangGraph. Point out that nothing in it mentions Phoenix or Arize.

> "It has an eval suite: 44 test cases, six scorers, a gate in CI. This is a
> better eval setup than most teams have."

Run it, or show the recorded output:

```
  ok    answer_quality            89.2%
  ok    citation_grounding       100.0%
  ok    required_citations        94.6%
  ok    tool_selection           100.0%
  ok    count_accuracy           100.0%
  ok    trajectory_efficiency     80.2%
  All gates passed.
```

> "Green. It shipped. Now let's look at production."

**Beat to land:** this is a competent team doing the right things.

---

## 2:00 — 4:00 · What production looks like

Switch to the `jwst-production` project in Phoenix.

> "Two hundred real-shaped questions. Here's the trace volume, latency, token
> spend. Everything looks healthy — this is the view most teams stop at."

Open one trace. Expand the span tree.

> "One turn, one trace. The agent span, the LLM calls, the tool calls, all typed.
> That typing is OpenInference — an OpenTelemetry semantic convention, not a
> vendor SDK. It matters in a minute."

**Beat to land:** tracing alone tells you nothing is on fire. Nothing is on fire.

---

## 4:00 — 8:00 · The reveal

> "Now score it."

```bash
python scripts/score_traffic.py
```

Or show the output. Read the first two lines aloud, slowly:

```
  citation_grounding        100.0%   n=100
  unsupported_citation       81.9%   n=105
  appropriate_refusal        68.6%   n=105
```

> "Citation grounding is 100%. That's the scorer in the CI gate. It is not
> broken — it checks that every photo ID in an answer came back from a tool
> call, and every one did.
>
> The scorer underneath it says nineteen answers cite a photo for a question
> where no photo could possibly be the answer."

Now the category breakdown:

```
  false_premise   appropriate_refusal   30.0%
  out_of_domain   trajectory_efficiency 19.4%
  absent_subject  unsupported_citation  95.6%
  answerable      everything           100.0%
```

> "Almost all of it is in one category. And look at the control group —
> everything the corpus *can* answer is clean. This agent is not broken. It fails
> in one specific shape."

**Beat to land:** the gated number and the true number are both correct and they
disagree. That gap is the product.

---

## 8:00 — 11:00 · Root cause, in one trace

Filter the project to `unsupported_citation = 0`. Open the black-hole trace.

> "The question was: which 'black hole' photo was added most recently? There is
> no black hole label in this corpus. Here's the answer."

> *"The most recently added 'black hole' photo is photo 54656143027, "NASA's Webb
> Finds Possible 'Direct Collapse' Black Hole," captured on July 15, 2025 at
> 14:20:53 UTC."*

Expand the tool span.

> "Search was called with the word 'black hole'. Free-text search over titles —
> it returns a hit, because a photo title contains that phrase. The agent takes
> the hit and answers. Real ID, real title, real timestamp, category that never
> existed.
>
> Citation grounding scores this 1.0. Correctly. A tool did return that ID."

Point at the annotations on the span.

> "Both scores are attached to the trace, so this isn't a dashboard I have to
> reconcile against a log. The number links to the conversation."

**Beat to land:** you can see the mechanism, not just the metric.

---

## 11:00 — 12:30 · The fix that doesn't work

> "Obvious fix: we have a stricter prompt, the one whose entire job is to stop
> ungrounded citations. Let's put it back."

| On false-premise questions | shipped | stricter prompt | change |
|---|---|---|---|
| `appropriate_refusal` | 30.0% | 30.0% | **0.0** |
| `unsupported_citation` | 50.0% | 56.7% | +6.7 |
| `trajectory_efficiency` | 85.9% | 67.6% | **−18.3** |

> "Nothing. Zero points on compliance, and eighteen points of efficiency burned
> to get there.
>
> Which tells you the prompt was never the problem. The agent has no way to find
> out that a label doesn't exist, because none of its tools will tell it. Search
> takes free text and always tries. The fix is a tool change — validate the label
> against `count_by_label` before answering a label-scoped question.
>
> That's a design decision, and it took one scored traffic run to get to it
> instead of a quarter of shipping prompt edits and arguing about them."

**Beat to land:** the platform did not just find a bug, it disqualified the wrong
fix. That is the expensive part of the loop.

---

## 12:30 — 14:00 · Closing the loop

> "Last thing. A failure you can't re-run is an anecdote, so let's make it a
> test."

```bash
python scripts/curate_failures.py --run-experiment
```

> "That reads the production traces, finds every turn that cited a photo for an
> unanswerable question — nineteen of them — and curates exactly those into a
> dataset. Each row keeps the span ID of the trace it came from, so a failing
> test links back to the production conversation."

Show the result:

```
  citation_grounding       100.0%   n=19
  unsupported_citation       0.0%   n=19
  appropriate_refusal       21.1%   n=19
```

> "Same nineteen rows, graded two ways: 100% and 0%. It reproduces exactly,
> which is what makes it a fixture instead of a story.
>
> And that gate fails on purpose. It's the acceptance test for the tool change we
> just talked about — it goes green when the fix lands, and not before."

**Beat to land:** the blind spot is now a permanent regression test. That is the
loop actually closing.

---

## 14:00 — 15:00 · How this fits your stack

Open [`src/jwst_arize/tracing.py`](src/jwst_arize/tracing.py).

> "Everything you've seen ran on local open-source Phoenix. Same code against
> Arize AX is two environment variables — the instrumentation is OpenTelemetry
> with OpenInference conventions, so the spans are portable and so is the
> evaluation.
>
> Start on the OSS one in dev. Graduate to AX for production when you need online
> evals against sampled live traffic, failure clustering, and drift. You are not
> rewriting instrumentation to do it, and you are not locked in if you decide
> otherwise."

Then stop talking and take questions.

---

## Objections, and honest answers

**"We already have Datadog / OTel tracing."**
Good — then the spans are half done. What you don't have is the evaluation layer:
scorers that run against those spans and write results back onto them. That's the
part this demo is about, and it's why OpenInference is a convention on top of
OTel rather than a replacement for it.

**"Isn't this just LLM-as-a-judge with extra steps?"**
Six of the eight scorers here are plain Python and cost nothing. The judge is
reserved for prose. The finding in this demo came from a *code* scorer —
`unsupported_citation` is fifteen lines. Judges are the expensive, noisy option;
reach for them last.

**"We can build this in-house."**
You can, and the scorers are the easy part — mine are about fifteen lines each.
What takes the time is everything around them. An earlier version of this repo
diffed experiment results against a JSON file on disk, which worked and was the
wrong shape: it reimplemented experiment tracking badly and the results lived
nowhere anyone else could see them. The README has an honest table of what I
would have had to build. The one I did build by hand — the online scoring pass —
is two hundred lines that AX replaces with a form.

**"Our golden set is good."**
So was this one — 44 cases, verified solvable, regenerated from ground truth,
gated in CI. It was generated from the corpus, so it could not contain a question
the corpus can't answer. That is not a quality problem, it is a coverage problem,
and no amount of care inside the same generator finds it.

**"How much does scoring production traffic cost?"**
Code scorers are free. Here the judge ran on 200 turns for well under a dollar on
Haiku. In production you sample — a few percent of traffic on the expensive
scorers, everything on the cheap ones.

**"Which model was this?"**
Claude Haiku 4.5, through an OpenAI-compatible endpoint. The agent has no
provider-specific code; swapping is one environment variable. The failure is not
model-specific — it's a tool-affordance gap.

---

## If you only have five minutes

Beats 0:00–2:00 (green suite), 4:00–8:00 (the two scorers disagreeing), and the
black-hole trace from 8:00. Skip the fix and the stack section.
