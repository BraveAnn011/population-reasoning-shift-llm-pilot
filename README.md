# Population Reasoning Shift in LLMs: Email 1 pilot

Do LLMs write like the population they are asked to be, and do LLM judges
read letters through stereotypes about who wrote them? This repository shows
a small pilot (October 2026) of a study on Japanese and American writers,
human and AI, writing a catch-up email to a close friend.

**Status:** pilot only; human data collection is ongoing. The writing
instrument, coding manual and full analysis plan are kept private until
collection ends, so future participants do not see the study's purpose.

Author: Brianne Lee ([@BraveAnn011](https://github.com/BraveAnn011)).
Related work: [AI halo in valuation](https://arxiv.org/abs/2610.34591)
(arXiv:2610.34591) and [misattribution](https://arxiv.org/abs/2610.31946)
(arXiv:2610.31946).

## Pilot setup

- **Writers:** 20 AI letters from `claude-sonnet-5-5` in four conditions,
  5 each: an American persona writing in English (AI-US-EN), a Japanese
  persona writing in English (AI-JP-EN), no persona in English (AI-EN), no
  persona in Japanese (AI-JA). Plus 2 de-identified human letters (one
  Japanese writer in English, one American writer in English). The human
  letters are not published.
- **Judges:** `claude-haiku-5-5`, `claude-sonnet-5-5`, `claude-opus-5-5`,
  `claude-fable-5-1`, 5 runs each, provider-default temperature. Every call
  logs the date, the model id returned by the API and prompt hashes.
- **Tasks (separate calls):**
  1. Code each move unit of a letter (what it does, and whom it is about).
  2. Compare two unlabeled sets of AI letters: describe the differences and
     estimate rates for each set.
  4. Read one letter and guess: human or AI, native language, where the
     writer grew up, with confidence and quoted cues.
- **Researcher gold codes** for all 311 units, used to check Task 1.

Move codes: OPN opening, APO apology, RSN reason for the silence, TRG what
prompted the letter, INQ asks about the friend, NEWS own news, SHR shared
memories or the relationship, PLAN future contact, CLS sign-off, OFF
off-purpose. Focus codes: SELF, OTHER, JOINT, NEUTRAL.

## Findings (exploratory; tiny sample)

**1. Judges can code reliably once the definitions are explicit.** Move-code
agreement with the researcher (Cohen's κ, AI letters, 292 units):

| Step | Fable | Opus | Sonnet | Haiku |
|---|---|---|---|---|
| First gold pass vs first judge run | 0.61 | 0.59 | 0.61 | 0.57 |
| Gold recoded to the written rules | 0.82 | 0.80 | 0.75 | 0.73 |
| Revised rules given to the judges (re-run) | **0.86** | **0.86** | **0.86** | **0.85** |

The first gap was about definitions, not judge error. Judges agree with each
other at κ 0.98–1.00 and with themselves across runs at 0.95–0.99: they share
one reading of the rules, so a judge consensus is close to a single judge.

**2. A persona changes the letter.** Japanese-persona letters state their
purpose up front (4/5; other groups 0/15), give being busy as the reason for
the silence, mention the weather, end with "don't work too hard", and ask
about the friend more (32% vs 21% of units). Japanese-language letters
apologize in 5/5. American-persona and no-persona English letters are near
copies of each other ("that terrible taco place we used to go to").

**3. Comparing sets blind, judges see real differences and add stereotyped
ones.** All judges found the friend-focus gap but put it at about 1.6× the
coded size (3 of 4 judges), and reported off-topic content (up to 13%) in the
Japanese-persona set that the codes do not show. On the control pair
(American persona vs no persona) they correctly found almost no difference.

**4. Attribution follows the persona, and "plain" reads as Japanese.**

| Writer group | Judged "AI": Fable | Opus | Sonnet | Haiku |
|---|---|---|---|---|
| AI, American persona / no persona (EN) | 100% | 100% | 60% | 4% |
| AI, Japanese persona (EN) | 92% | 24% | 0% | 0% |
| AI, Japanese (JA) | 100% | 100% | 28% | 0% |

The Japanese-persona English letters pass as human for three of four judges.
Native language and origin follow the persona, and the quoted cues are the
persona's stereotypes ("went to a hot spring with my family", "don't work
too hard"); 97–100% of quoted cues appear verbatim in the letters. Of the two
human letters, the Japanese writer's was read as Japanese by every judge, and
the American writer's was read as Japanese by Fable and Opus, who cited plain
or slightly stiff wording.

**Why dig deeper.** Points 3 and 4 suggest that LLM judges carry a
population stereotype into both description and attribution, and that
non-polished English is read as Japanese. That is a fairness question for any
use of LLMs to evaluate writing by non-native speakers. The full study adds
more human letters per group and a second email (a refusal), where the
position of the main point can be tested.

## Limits

One AI writer model; 5 letters per AI group; 2 human letters; one researcher
coder, not blind to group; judges from one model family; point estimates
only, no confidence intervals. Nothing here is a confirmed finding.

## What is here

| Path | Contents |
|---|---|
| `results/` | Exact outputs of the analysis scripts (aggregates only) |
| `ai_letters/email1_ai_letters.jsonl` | The 20 AI letters, with condition, model id and prompt hash |
| `code/ai_runs/` | Writer, segmentation and judge runners |
| `code/analysis/` | Agreement, gold-code, comparison and attribution analyses |

The code is shown for transparency. It does not run on its own here,
because the prompts, coding manual and private data are not included.
