"""Generate the model's own reasoning traces to use as probe text.

Kennings are meant to compress the model's reasoning, so the most relevant
text to probe is reasoning the model itself wrote. This module asks the model
a set of problems with thinking switched on and saves what it writes.
"""

from __future__ import annotations

import json
import os
import re

import torch

from .models import LM

# Problems chosen to draw out long, multi-step reasoning in several styles.
PROMPTS = [
    "A water tank holds 2,400 litres. Pump A fills it in 6 hours, pump B in 8 hours, and a leak empties a full tank in 24 hours. Starting empty with both pumps on and the leak open, how long until it is full? After 2 hours pump A fails. How long does the whole fill take then?",
    "Find all integers n between 1 and 200 for which n squared plus n plus 41 is not prime. Explain how you search efficiently instead of testing every n by hand.",
    "Five friends (Ana, Bo, Cy, Dee, Eli) sit in a row of five seats. Ana is not at either end. Bo sits immediately left of Cy. Dee is not next to Eli. Eli is in an odd-numbered seat. List every valid seating and show the elimination.",
    "Write a Python function that returns the length of the longest substring without repeating characters. Reason about the sliding window invariants first, then give the code, then trace it by hand on the input 'abcabcbbxyzx'.",
    "A train leaves city P at 08:00 at 90 km/h. A second train leaves city Q, 510 km away, at 09:30 at 120 km/h toward P. Where and when do they meet? If the first train stops for 20 minutes at the 180 km mark, how does the answer change?",
    "Prove that the sum of the first n odd numbers is n squared, once by induction and once with a geometric argument. Then explain which proof generalises better to the sum of the first n cubes.",
    "You have 12 coins, one of which is counterfeit and either heavier or lighter. Using a balance scale three times, find the coin and whether it is heavy or light. Give the full decision tree.",
    "Design a rate limiter that allows 100 requests per user per minute across 20 servers. Compare token bucket, fixed window and sliding window log. Pick one and explain the failure cases you accept.",
    "Compute the determinant of the 4x4 matrix with rows (2,0,1,3), (1,1,0,2), (0,3,1,1), (4,1,2,0) by cofactor expansion, then check the result by row reduction.",
    "A shop sells pens at 3 for 4 kronor and notebooks at 7 kronor each. With exactly 100 kronor and at least one of each, how many ways can you spend all the money? Enumerate systematically.",
    "Explain step by step why quicksort has average time n log n but worst case n squared. Work through the recurrence for the average case and give a concrete worst-case input for a first-element pivot.",
    "A ladder 10 m long leans against a wall. The bottom slides away at 0.5 m/s. How fast is the top sliding down when the bottom is 6 m from the wall? Then find when the top is moving at exactly 1 m/s.",
    "Three boxes are labelled 'apples', 'oranges' and 'mixed', and every label is wrong. You may take one fruit from one box. Which box do you pick from and how do you relabel all three? Then generalise to four boxes with four wrong labels and say whether one draw is still enough.",
    "Given the list [14, 3, 27, 8, 19, 3, 42, 11], trace merge sort completely, showing every split and merge. Count the comparisons made.",
    "A project has tasks A(3 days), B(5), C(2), D(4), E(6), F(1). B and C need A. D needs B. E needs C. F needs D and E. Find the critical path and the slack of every task. If one extra worker can shorten one task by 2 days, which task should it be?",
    "Solve the system: x + 2y - z = 4, 2x - y + 3z = 9, 3x + y + 2z = 13. Say whether it has one solution, none, or infinitely many, and describe the solution set precisely.",
    "A bag has 5 red, 4 blue and 3 green marbles. Three are drawn without replacement. Find the probability of exactly two colours appearing. Check the answer by computing the complement.",
    "Write a function that checks whether a string of brackets ()[]{} is balanced, then extend it to report the index of the first error. Walk through the inputs '([{}])', '([)]' and '((('.",
    "How many trailing zeros does 1000 factorial have? How many does it have in base 12? Explain the general method for any base.",
    "A farmer has 240 m of fence for a rectangular field along a straight river, with no fence needed on the river side, and wants to split the field into three equal pens with fences perpendicular to the river. What dimensions maximise the area?",
    "Explain how a hash table with open addressing handles deletions. Why are tombstones needed? Simulate inserting keys 10, 22, 31, 4, 15, 28 into a table of size 11 with linear probing, then delete 22 and look up 15.",
    "Find the number of paths from the bottom-left to the top-right of a 6 by 6 grid of points moving only right or up, that never go above the main diagonal. Derive the formula instead of quoting it.",
    "A loan of 200,000 kronor at 4.8 percent annual interest, compounded monthly, is repaid in equal monthly payments over 10 years. Derive the payment formula from the geometric series and compute the payment and the total interest.",
    "You are given a sorted array that has been rotated an unknown number of times, with no duplicates. Describe a binary search that finds a target in logarithmic time, and prove the loop invariant. Trace it on [15, 18, 2, 3, 6, 12] looking for 3.",
    "Four people must cross a bridge at night with one torch. They take 1, 2, 7 and 10 minutes, at most two cross at a time, and a pair walks at the slower pace. Find the minimum total time and prove no schedule is faster.",
    "Estimate how many piano tuners work in Stockholm. State each assumption, give a range for it, and carry the uncertainty through to the final estimate.",
    "Factor the polynomial x^4 - 5x^3 + 5x^2 + 5x - 6 completely over the integers, explaining how you chose candidate roots.",
    "Explain the difference between a mutex, a semaphore and a condition variable by solving the bounded-buffer producer-consumer problem with each, and point out one bug that is easy to write in each version.",
    "A cube is painted red on all faces and cut into 125 equal small cubes. How many small cubes have exactly 0, 1, 2 and 3 red faces? Generalise to an n by n by n cut.",
    "Two players alternately remove 1, 3 or 4 stones from a pile of 37. The player who takes the last stone wins. Who wins with perfect play? Work out the pattern of losing positions.",
    "Convert the decimal number 2026.6875 to binary, octal and hexadecimal. Then explain why 0.1 has no finite binary expansion and what that means for floating-point sums.",
    "A recipe for 6 people needs 450 g flour, 3 eggs and 300 ml milk. You have 1 kg flour, 5 eggs and 1 litre milk. What is the largest whole number of people you can serve, which ingredient limits you, and how much of the others is left over?",
    "Given the recurrence T(n) = 3T(n/2) + n with T(1) = 1, find a closed form for n a power of 2 by unrolling, and confirm it with the master theorem.",
    "Write SQL to find, for each customer, the month with their highest total spending, given a table orders(customer_id, order_date, amount). Explain how ties are handled and how the query would use an index.",
    "A clock's hour and minute hands overlap at 12:00. Find the exact times of all overlaps in the next 12 hours, and the times when the hands are exactly opposite.",
    "Explain why the harmonic series diverges but the sum of reciprocals of squares converges. Give the comparison arguments in full and estimate how many terms are needed for the harmonic sum to pass 10.",
    "Plan a 3-day schedule to prepare for an exam covering 8 chapters of unequal difficulty (2 hard, 3 medium, 3 easy) with 5 study hours a day, including review. Justify the ordering and how you would adjust after a bad first day.",
    "Implement breadth-first search to find the shortest path in an unweighted grid with obstacles, returning the path itself. Reason about how to reconstruct the path, then trace the algorithm on a 5 by 5 grid of your choosing with at least four obstacles.",
    "A 5-digit number is divisible by 9, its digits are all different, it reads the same when the first and last digits are swapped, and the middle digit is the sum of the second and fourth. How many such numbers are there?",
    "Derive the formula for the volume of a sphere using integration, once with discs and once with shells, and explain why both give the same result.",
]


def _prompt_ids(lm: LM, prompt: str, thinking: bool) -> torch.Tensor:
    tok = lm.tok
    if getattr(tok, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        try:
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
            )
        except TypeError:
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids
    else:
        ids = torch.tensor([[lm.start_id] + lm.encode(prompt + "\n")])
    return ids.to(lm.device)


def split_reasoning(raw: str) -> str | None:
    """Return the text inside <think> ... </think>, or None if there is none.

    Handles a chat template that already opened the tag in the prompt, in
    which case the completion only contains the closing tag.
    """
    if "</think>" in raw:
        head = raw.split("</think>", 1)[0]
        return head.split("<think>", 1)[-1].strip() or None
    match = re.search(r"<think>(.*)", raw, flags=re.DOTALL)
    if match:
        return match.group(1).strip() or None
    return None


@torch.no_grad()
def generate_traces(
    lm: LM,
    out_path: str,
    prompts: list[str] | None = None,
    samples_per_prompt: int = 2,
    max_new_tokens: int = 2048,
    temperature: float = 1.0,
    top_p: float = 0.95,
    thinking: bool = True,
    seed: int = 0,
) -> int:
    """Write one JSON line per generated trace. Returns the number written.

    Rows already in the file are kept and their prompts skipped, so an
    interrupted run can simply be started again.
    """
    prompts = prompts or PROMPTS
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            done = {json.loads(line)["prompt_id"] for line in f if line.strip()}

    torch.manual_seed(seed)
    eos = lm.tok.eos_token_id
    pad = lm.tok.pad_token_id if lm.tok.pad_token_id is not None else lm.start_id
    written = 0
    with open(out_path, "a") as f:
        for pid, prompt in enumerate(prompts):
            if pid in done:
                continue
            ids = _prompt_ids(lm, prompt, thinking)
            out = lm.model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                num_return_sequences=samples_per_prompt,
                pad_token_id=pad,
            )
            for s, seq in enumerate(out[:, ids.shape[1] :].tolist()):
                if eos is not None and eos in seq:
                    seq = seq[: seq.index(eos)]
                raw = lm.tok.decode(seq, skip_special_tokens=False)
                row = {
                    "prompt_id": pid,
                    "sample": s,
                    "prompt": prompt,
                    "text": lm.tok.decode(seq, skip_special_tokens=True),
                    "reasoning": split_reasoning(raw),
                    "n_tokens": len(seq),
                }
                f.write(json.dumps(row) + "\n")
                written += 1
            f.flush()
            print(f"[traces] prompt {pid + 1}/{len(prompts)} done", flush=True)
    return written
