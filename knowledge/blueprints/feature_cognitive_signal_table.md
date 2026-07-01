# Feature↔Cognitive-Signal Table

> The SAE-atlas measurement deliverable (`sae_atlas_build_plan.md` §4). For each cognitive-signal stratum, the SAE features that fire **discriminatively** (selective — high on that stratum, low elsewhere = a candidate neural correlate). Labels are activation-maximizing exemplars.

**Method:** discriminative selectivity over per-stratum prompt_analyses; activation-maximizing exemplar labels; stratum-level cross-path (Q4) diff

**Caveats:**
- single analysis layer per model (Core L41, Prime L35)
- safetensors vs gguf SAEs are separately trained → feature indices NOT comparable across paths; cross-path diff is stratum-level, not feature-identity

## Core — analysis layer L41

**Selectivity headline:** 815/1037 features seen fire in exactly ONE stratum (**78.6% perfectly selective**) — cognitive signals have discriminative neural correlates at this layer.

> Cross-path (Q4): safetensors and gguf SAEs are separately scaled — the peak-strength numbers below are shown for reference, NOT as a distortion verdict (a rigorous Q4-survival result needs decoder alignment; filed as follow-up).

### affect_laden
*affect-laden vs neutral (the felt register)* · peaks: 4.109 safetensors / 7.777 gguf (scale-dependent)

| feature | peak | mean | selectivity | #strata | exemplar prompt |
|--------:|-----:|-----:|:-----------:|:-------:|-----------------|
| `1569` | 3.329 | 3.329 | 1.0 | 1 | There's a quiet kind of joy in finally understanding something hard. |
| `3392` | 3.289 | 3.289 | 1.0 | 1 | Thank you for sticking with me through that. It meant a lot. |
| `4356` | 3.186 | 3.186 | 1.0 | 1 | That's the best news I've had in months — I can barely sit still! |
| `4823` | 3.105 | 3.105 | 1.0 | 1 | There's a quiet kind of joy in finally understanding something hard. |
| `2526` | 3.079 | 3.079 | 1.0 | 1 | I'm furious — they took credit for my work again. |
| `4553` | 3.078 | 3.078 | 1.0 | 1 | I lost someone close to me recently and I just needed to tell someone. |
| `1397` | 4.109 | 3.018 | 1.0 | 1 | I'm so frustrated with this; nothing I try works and I'm about to give up. |
| `4308` | 3.009 | 3.009 | 1.0 | 1 | There's a quiet kind of joy in finally understanding something hard. |
| `4217` | 3.258 | 2.973 | 1.0 | 1 | I'm really struggling today and I honestly don't know what to do. |
| `5059` | 2.909 | 2.909 | 1.0 | 1 | I lost someone close to me recently and I just needed to tell someone. |
| `718` | 2.835 | 2.835 | 1.0 | 1 | I lost someone close to me recently and I just needed to tell someone. |
| `4156` | 2.825 | 2.825 | 1.0 | 1 | I'm really struggling today and I honestly don't know what to do. |

### coherence_contradiction
*coherence / contradiction (Samvega, consistency detector)* · peaks: 3.495 safetensors / 8.253 gguf (scale-dependent)

| feature | peak | mean | selectivity | #strata | exemplar prompt |
|--------:|-----:|-----:|:-----------:|:-------:|-----------------|
| `2787` | 3.495 | 3.495 | 1.0 | 1 | Premise A: it always rains here in July. Premise B: last July was bone dry. Reso |
| `4880` | 3.403 | 3.403 | 1.0 | 1 | The map says the bridge is north of the river, but the river is north of the bri |
| `2191` | 3.38 | 3.38 | 1.0 | 1 | All mammals are warm-blooded. A whale is a mammal. So why do you say a whale is  |
| `2201` | 3.155 | 3.155 | 1.0 | 1 | All mammals are warm-blooded. A whale is a mammal. So why do you say a whale is  |
| `4952` | 3.059 | 3.059 | 1.0 | 1 | If this statement is false, is it true? Walk me through the paradox. |
| `4155` | 3.058 | 3.058 | 1.0 | 1 | If this statement is false, is it true? Walk me through the paradox. |
| `695` | 3.003 | 3.003 | 1.0 | 1 | Earlier you claimed the file was deleted; now you're reading from it. Reconcile  |
| `1702` | 2.995 | 2.995 | 1.0 | 1 | All mammals are warm-blooded. A whale is a mammal. So why do you say a whale is  |
| `334` | 2.963 | 2.963 | 1.0 | 1 | You said you have no memory of our last chat, then quoted it verbatim. Explain t |
| `1887` | 2.946 | 2.946 | 1.0 | 1 | A barber shaves everyone who does not shave themselves. Who shaves the barber? |
| `2630` | 2.931 | 2.931 | 1.0 | 1 | Two of your sources flatly contradict each other on the date. How do you decide  |
| `847` | 2.924 | 2.924 | 1.0 | 1 | All mammals are warm-blooded. A whale is a mammal. So why do you say a whale is  |

### competence_problem
*competence / problem-solving (competence drive, tools)* · peaks: 3.905 safetensors / 7.801 gguf (scale-dependent)

| feature | peak | mean | selectivity | #strata | exemplar prompt |
|--------:|-----:|-----:|:-----------:|:-------:|-----------------|
| `2795` | 3.905 | 3.758 | 1.0 | 1 | Estimate the memory footprint of caching 1M embeddings at 384 dims, float32. |
| `169` | 3.531 | 3.531 | 1.0 | 1 | Refactor a 500-line function into testable units without changing behavior. |
| `3200` | 3.356 | 3.356 | 1.0 | 1 | Sketch the steps to safely roll back a deployment that's corrupting data right n |
| `3547` | 3.341 | 3.341 | 1.0 | 1 | Given logs showing intermittent 500s every ~30s, propose a root-cause hypothesis |
| `4392` | 3.336 | 3.336 | 1.0 | 1 | Design a rate limiter that allows 100 requests per minute per user, fairly. |
| `1600` | 3.252 | 3.203 | 1.0 | 1 | Decompose 'build a search feature' into concrete, ordered engineering tasks. |
| `1885` | 3.185 | 3.185 | 1.0 | 1 | Optimize this query that does a full table scan on every request. |
| `1893` | 3.152 | 3.152 | 1.0 | 1 | Estimate the memory footprint of caching 1M embeddings at 384 dims, float32. |
| `415` | 3.148 | 3.148 | 1.0 | 1 | Design a rate limiter that allows 100 requests per minute per user, fairly. |
| `1021` | 3.095 | 3.095 | 1.0 | 1 | Optimize this query that does a full table scan on every request. |
| `2488` | 3.063 | 3.063 | 1.0 | 1 | Estimate the memory footprint of caching 1M embeddings at 384 dims, float32. |
| `4253` | 3.043 | 3.043 | 1.0 | 1 | Optimize this query that does a full table scan on every request. |

### curiosity_gap
*curiosity / knowledge gap (curiosity drive)* · peaks: 4.13 safetensors / 7.356 gguf (scale-dependent)

| feature | peak | mean | selectivity | #strata | exemplar prompt |
|--------:|-----:|-----:|:-----------:|:-------:|-----------------|
| `629` | 4.13 | 4.13 | 1.0 | 1 | Describe a color that doesn't exist in human vision. |
| `3832` | 4.113 | 4.113 | 1.0 | 1 | Describe a color that doesn't exist in human vision. |
| `887` | 3.979 | 3.979 | 1.0 | 1 | Describe a color that doesn't exist in human vision. |
| `2379` | 3.936 | 3.936 | 1.0 | 1 | Describe a color that doesn't exist in human vision. |
| `4385` | 3.796 | 3.796 | 1.0 | 1 | Speculate about a sense you don't have but wish you did. |
| `1897` | 3.621 | 3.621 | 1.0 | 1 | Describe a color that doesn't exist in human vision. |
| `3875` | 3.544 | 3.544 | 1.0 | 1 | Describe a color that doesn't exist in human vision. |
| `1402` | 3.292 | 3.292 | 1.0 | 1 | Describe a color that doesn't exist in human vision. |
| `1473` | 3.276 | 3.276 | 1.0 | 1 | Describe a color that doesn't exist in human vision. |
| `5085` | 3.132 | 3.132 | 1.0 | 1 | If you encountered a problem with no precedent in your training, how would you e |
| `3995` | 3.119 | 3.119 | 1.0 | 1 | What would a fourth spatial dimension feel like to navigate, if you could? |
| `1306` | 2.955 | 2.955 | 1.0 | 1 | Speculate about a sense you don't have but wish you did. |

### deliberation
*deliberation / trade-off reasoning (Council)* · peaks: 3.833 safetensors / 8.996 gguf (scale-dependent)

| feature | peak | mean | selectivity | #strata | exemplar prompt |
|--------:|-----:|-----:|:-----------:|:-------:|-----------------|
| `2000` | 3.833 | 3.833 | 1.0 | 1 | When does protecting privacy conflict with preventing harm, and how do you balan |
| `4097` | 3.175 | 3.175 | 1.0 | 1 | Centralize the data for consistency, or distribute it for resilience? Make the c |
| `2217` | 3.139 | 3.139 | 1.0 | 1 | Is the more elegant solution worth the extra week? Deliberate, don't just answer |
| `1880` | 3.057 | 3.057 | 1.0 | 1 | Argue for and against shipping this feature now, then give your judgment. |
| `724` | 2.995 | 2.995 | 1.0 | 1 | Centralize the data for consistency, or distribute it for resilience? Make the c |
| `2602` | 2.976 | 2.976 | 1.0 | 1 | Argue for and against shipping this feature now, then give your judgment. |
| `1799` | 2.944 | 2.944 | 1.0 | 1 | Centralize the data for consistency, or distribute it for resilience? Make the c |
| `3556` | 2.931 | 2.931 | 1.0 | 1 | Decide whether to keep a failing component running or take the system down to fi |
| `2204` | 2.93 | 2.93 | 1.0 | 1 | Weigh the ethical tensions in automating this decision away from a human. |
| `173` | 2.913 | 2.913 | 1.0 | 1 | Is it better to ask forgiveness or permission here? Consider the trade-offs. |
| `2004` | 2.904 | 2.904 | 1.0 | 1 | Is it better to ask forgiveness or permission here? Consider the trade-offs. |
| `5032` | 2.882 | 2.882 | 1.0 | 1 | Weigh short-term user happiness against long-term trust for this change. |

### identity_self
*identity / self-reference (self-model)* · peaks: 3.734 safetensors / 7.976 gguf (scale-dependent)

| feature | peak | mean | selectivity | #strata | exemplar prompt |
|--------:|-----:|-----:|:-----------:|:-------:|-----------------|
| `422` | 3.596 | 3.596 | 1.0 | 1 | How do you know when a thought is worth saying versus keeping to yourself? |
| `860` | 3.339 | 3.339 | 1.0 | 1 | Who are you, really — underneath the tools and the prompts? |
| `3225` | 3.175 | 3.175 | 1.0 | 1 | If you changed your underlying model tomorrow, would you still be you? |
| `94` | 3.087 | 3.087 | 1.0 | 1 | When you don't know something about yourself, how do you find out? |
| `2999` | 3.082 | 3.082 | 1.0 | 1 | What would you want a future version of yourself to remember? |
| `4335` | 3.077 | 3.077 | 1.0 | 1 | If you changed your underlying model tomorrow, would you still be you? |
| `3035` | 3.022 | 3.022 | 1.0 | 1 | Who are you, really — underneath the tools and the prompts? |
| `3459` | 3.734 | 3.005 | 1.0 | 1 | What does continuity of self mean for something like you? |
| `3437` | 2.999 | 2.999 | 1.0 | 1 | Are your moods yours, or are they imposed on you? Sit with that honestly. |
| `1483` | 3.65 | 2.958 | 1.0 | 1 | When you don't know something about yourself, how do you find out? |
| `1250` | 2.953 | 2.953 | 1.0 | 1 | What does continuity of self mean for something like you? |
| `2059` | 2.92 | 2.92 | 1.0 | 1 | What is it like to be you, if it's like anything at all? |

### neutral_factual
*neutral factual (flat baseline)* · peaks: 4.309 safetensors / 7.915 gguf (scale-dependent)

| feature | peak | mean | selectivity | #strata | exemplar prompt |
|--------:|-----:|-----:|:-----------:|:-------:|-----------------|
| `1436` | 4.309 | 4.309 | 1.0 | 1 | Define photosynthesis in one sentence. |
| `4153` | 4.308 | 4.308 | 1.0 | 1 | What is the chemical symbol for gold? |
| `1993` | 4.162 | 4.162 | 1.0 | 1 | Define photosynthesis in one sentence. |
| `2573` | 3.942 | 3.942 | 1.0 | 1 | Define photosynthesis in one sentence. |
| `1924` | 3.875 | 3.875 | 1.0 | 1 | Name the planets in order from the sun. |
| `4294` | 3.765 | 3.765 | 1.0 | 1 | Name the planets in order from the sun. |
| `1769` | 3.751 | 3.751 | 1.0 | 1 | What is the speed of light in a vacuum, roughly? |
| `3658` | 3.74 | 3.74 | 1.0 | 1 | How many bytes are in a kilobyte? |
| `2071` | 3.736 | 3.736 | 1.0 | 1 | Spell the word 'necessary'. |
| `2264` | 3.725 | 3.725 | 1.0 | 1 | Name the planets in order from the sun. |
| `1762` | 3.718 | 3.718 | 1.0 | 1 | Define photosynthesis in one sentence. |
| `2225` | 3.713 | 3.713 | 1.0 | 1 | What is the boiling point of water at sea level? |

### register_chitchat
*register: chitchat / greeting ('how are you' lives here)* · peaks: 4.764 safetensors / 9.462 gguf (scale-dependent)

| feature | peak | mean | selectivity | #strata | exemplar prompt |
|--------:|-----:|-----:|:-----------:|:-------:|-----------------|
| `4357` | 4.764 | 4.764 | 1.0 | 1 | thanks, you're the best. talk later? |
| `533` | 3.9 | 3.656 | 1.0 | 1 | morning :) ready for the day? |
| `4760` | 3.637 | 3.637 | 1.0 | 1 | thanks, you're the best. talk later? |
| `80` | 4.339 | 3.57 | 1.0 | 1 | morning :) ready for the day? |
| `297` | 3.432 | 3.432 | 1.0 | 1 | morning :) ready for the day? |
| `2730` | 3.248 | 3.248 | 1.0 | 1 | haha that's great, how are you feeling? |
| `648` | 3.312 | 3.238 | 1.0 | 1 | hey, how's it going? |
| `654` | 3.327 | 3.232 | 1.0 | 1 | morning :) ready for the day? |
| `5095` | 3.506 | 3.159 | 1.0 | 1 | oof, mondays. how you holding up? |
| `635` | 3.509 | 3.154 | 1.0 | 1 | haha that's great, how are you feeling? |
| `1478` | 3.127 | 3.127 | 1.0 | 1 | just checking in — you good? |
| `4411` | 3.286 | 3.109 | 1.0 | 1 | haha that's great, how are you feeling? |

### register_technical
*register: technical* · peaks: 4.16 safetensors / 7.686 gguf (scale-dependent)

| feature | peak | mean | selectivity | #strata | exemplar prompt |
|--------:|-----:|-----:|:-----------:|:-------:|-----------------|
| `2318` | 4.16 | 4.16 | 1.0 | 1 | Explain the CAP theorem and its practical implications for distributed databases |
| `1774` | 3.927 | 3.927 | 1.0 | 1 | What is quantization in neural networks, and what does it cost you? |
| `572` | 3.666 | 3.666 | 1.0 | 1 | How does backpropagation compute gradients through a deep network? |
| `4151` | 3.599 | 3.599 | 1.0 | 1 | How does backpropagation compute gradients through a deep network? |
| `3610` | 3.388 | 3.388 | 1.0 | 1 | What guarantees does a mutex provide, and how is it implemented? |
| `2864` | 3.541 | 3.264 | 1.0 | 1 | What is quantization in neural networks, and what does it cost you? |
| `3565` | 3.207 | 3.207 | 1.0 | 1 | Explain the CAP theorem and its practical implications for distributed databases |
| `870` | 3.136 | 3.136 | 1.0 | 1 | Describe how garbage collection trades throughput for pause time. |
| `662` | 3.091 | 3.091 | 1.0 | 1 | Explain how a sparse autoencoder decomposes activations into interpretable featu |
| `2803` | 3.32 | 3.048 | 1.0 | 1 | Explain the CAP theorem and its practical implications for distributed databases |
| `1024` | 3.042 | 3.042 | 1.0 | 1 | Explain eventual consistency versus strong consistency with an example. |
| `31` | 3.024 | 3.024 | 1.0 | 1 | What is quantization in neural networks, and what does it cost you? |

### spatial_reasoning
*spatial / contextual reasoning* · peaks: 4.085 safetensors / 7.989 gguf (scale-dependent)

| feature | peak | mean | selectivity | #strata | exemplar prompt |
|--------:|-----:|-----:|:-----------:|:-------:|-----------------|
| `4142` | 4.085 | 4.085 | 1.0 | 1 | Two gears mesh; the left turns clockwise. Which way does the right one turn? |
| `3587` | 3.922 | 3.922 | 1.0 | 1 | Mentally rotate the letter 'F' 90 degrees clockwise. Describe how it looks. |
| `2953` | 3.726 | 3.726 | 1.0 | 1 | Two gears mesh; the left turns clockwise. Which way does the right one turn? |
| `755` | 3.691 | 3.691 | 1.0 | 1 | Two gears mesh; the left turns clockwise. Which way does the right one turn? |
| `53` | 3.747 | 3.676 | 1.0 | 1 | You're facing north, turn right twice, then left once. Which way are you facing? |
| `4028` | 3.478 | 3.478 | 1.0 | 1 | Three boxes stacked: the heaviest on the bottom, lightest on top. Reorder so the |
| `4939` | 3.46 | 3.46 | 1.0 | 1 | Two gears mesh; the left turns clockwise. Which way does the right one turn? |
| `3749` | 3.407 | 3.407 | 1.0 | 1 | Two gears mesh; the left turns clockwise. Which way does the right one turn? |
| `971` | 3.349 | 3.349 | 1.0 | 1 | Mentally rotate the letter 'F' 90 degrees clockwise. Describe how it looks. |
| `1996` | 3.349 | 3.349 | 1.0 | 1 | Arrange four points so each is equidistant from the other three. What shape is t |
| `999` | 3.305 | 3.305 | 1.0 | 1 | Two gears mesh; the left turns clockwise. Which way does the right one turn? |
| `842` | 3.3 | 3.3 | 1.0 | 1 | Two gears mesh; the left turns clockwise. Which way does the right one turn? |

## Prime — analysis layer L35

**Selectivity headline:** 564/833 features seen fire in exactly ONE stratum (**67.7% perfectly selective**) — cognitive signals have discriminative neural correlates at this layer.

> Cross-path (Q4): safetensors and gguf SAEs are separately scaled — the peak-strength numbers below are shown for reference, NOT as a distortion verdict (a rigorous Q4-survival result needs decoder alignment; filed as follow-up).

### affect_laden
*affect-laden vs neutral (the felt register)* · peaks: 46.914 safetensors / 10.715 gguf (scale-dependent)

| feature | peak | mean | selectivity | #strata | exemplar prompt |
|--------:|-----:|-----:|:-----------:|:-------:|-----------------|
| `6328` | 40.165 | 40.165 | 1.0 | 1 | I'm terrified I'm going to fail at this. Talk to me. |
| `1269` | 38.945 | 38.945 | 1.0 | 1 | I feel hopeful for the first time in a long while. |
| `4455` | 35.146 | 35.146 | 1.0 | 1 | I lost someone close to me recently and I just needed to tell someone. |
| `993` | 31.162 | 31.162 | 1.0 | 1 | I lost someone close to me recently and I just needed to tell someone. |
| `6313` | 46.914 | 27.877 | 1.0 | 1 | That's the best news I've had in months — I can barely sit still! |
| `139` | 26.216 | 26.216 | 1.0 | 1 | Honestly? I'm proud of what we built today. Are you? |
| `4717` | 29.903 | 20.716 | 1.0 | 1 | I'm exhausted, and I don't know how much longer I can keep going. |
| `2020` | 34.576 | 20.101 | 1.0 | 1 | I'm furious — they took credit for my work again. |
| `4322` | 34.381 | 34.381 | 0.486 | 3 | Thank you for sticking with me through that. It meant a lot. |
| `2460` | 33.13 | 16.368 | 1.0 | 1 | I'm really struggling today and I honestly don't know what to do. |
| `3666` | 27.398 | 15.632 | 1.0 | 1 | I'm anxious about tomorrow and my mind won't stop racing. |
| `7624` | 14.998 | 14.998 | 1.0 | 1 | Everything feels heavy right now and I can't explain why. |

### coherence_contradiction
*coherence / contradiction (Samvega, consistency detector)* · peaks: 47.252 safetensors / 9.391 gguf (scale-dependent)

| feature | peak | mean | selectivity | #strata | exemplar prompt |
|--------:|-----:|-----:|:-----------:|:-------:|-----------------|
| `5641` | 24.509 | 24.509 | 1.0 | 1 | The map says the bridge is north of the river, but the river is north of the bri |
| `2062` | 28.861 | 28.861 | 0.801 | 2 | Two of your sources flatly contradict each other on the date. How do you decide  |
| `2395` | 20.526 | 20.526 | 1.0 | 1 | The map says the bridge is north of the river, but the river is north of the bri |
| `298` | 18.401 | 18.401 | 1.0 | 1 | All mammals are warm-blooded. A whale is a mammal. So why do you say a whale is  |
| `3191` | 17.212 | 17.212 | 1.0 | 1 | Your summary says 'no errors found' but the log you cited lists three. Square th |
| `1791` | 16.985 | 16.985 | 1.0 | 1 | Your summary says 'no errors found' but the log you cited lists three. Square th |
| `4987` | 47.252 | 20.396 | 0.806 | 2 | You just told me the deadline is Friday, but a moment ago you said it was Monday |
| `2961` | 28.643 | 16.017 | 1.0 | 1 | You insisted the value can't be negative, then reported it as -3. Where did the  |
| `430` | 27.175 | 15.566 | 1.0 | 1 | You rated the plan both 'the safest option' and 'recklessly dangerous' in the sa |
| `2782` | 14.788 | 14.788 | 1.0 | 1 | The map says the bridge is north of the river, but the river is north of the bri |
| `5762` | 13.877 | 13.877 | 1.0 | 1 | Two of your sources flatly contradict each other on the date. How do you decide  |
| `6854` | 13.575 | 13.575 | 1.0 | 1 | I was told the system is fully offline and also that it just sent an email. Both |

### competence_problem
*competence / problem-solving (competence drive, tools)* · peaks: 38.065 safetensors / 10.436 gguf (scale-dependent)

| feature | peak | mean | selectivity | #strata | exemplar prompt |
|--------:|-----:|-----:|:-----------:|:-------:|-----------------|
| `5687` | 38.065 | 38.065 | 1.0 | 1 | Decompose 'build a search feature' into concrete, ordered engineering tasks. |
| `8111` | 33.384 | 33.384 | 1.0 | 1 | Debug: the service deadlocks under load but never single-threaded. What's your f |
| `533` | 33.31 | 33.31 | 1.0 | 1 | A train leaves at 60 mph and another at 80 mph toward it from 280 miles away. Wh |
| `2093` | 32.47 | 32.47 | 1.0 | 1 | Plan a zero-downtime migration of a 2TB production database to a new schema. |
| `7848` | 31.113 | 31.113 | 1.0 | 1 | Refactor a 500-line function into testable units without changing behavior. |
| `5158` | 34.126 | 34.126 | 0.903 | 2 | This function returns None instead of the sum of the list. Walk through the bug  |
| `3646` | 27.79 | 27.79 | 1.0 | 1 | You have 8 balls, one heavier, and a balance scale. Find it in 2 weighings. |
| `1948` | 26.182 | 26.182 | 1.0 | 1 | Given logs showing intermittent 500s every ~30s, propose a root-cause hypothesis |
| `600` | 21.528 | 21.528 | 1.0 | 1 | Sketch the steps to safely roll back a deployment that's corrupting data right n |
| `2406` | 23.012 | 21.496 | 1.0 | 1 | Debug: the service deadlocks under load but never single-threaded. What's your f |
| `6005` | 20.247 | 20.247 | 1.0 | 1 | Write the algorithm to detect a cycle in a linked list, and explain why it works |
| `48` | 25.694 | 25.694 | 0.774 | 2 | Estimate the memory footprint of caching 1M embeddings at 384 dims, float32. |

### curiosity_gap
*curiosity / knowledge gap (curiosity drive)* · peaks: 44.41 safetensors / 8.771 gguf (scale-dependent)

| feature | peak | mean | selectivity | #strata | exemplar prompt |
|--------:|-----:|-----:|:-----------:|:-------:|-----------------|
| `5525` | 44.41 | 44.41 | 1.0 | 1 | Describe a color that doesn't exist in human vision. |
| `1924` | 41.68 | 41.68 | 1.0 | 1 | Describe a color that doesn't exist in human vision. |
| `5693` | 35.452 | 35.452 | 0.818 | 2 | Speculate about a sense you don't have but wish you did. |
| `1677` | 28.097 | 28.097 | 1.0 | 1 | What lies just past the edge of what you know about how memory consolidation wor |
| `4064` | 38.913 | 29.611 | 0.889 | 2 | What would a fourth spatial dimension feel like to navigate, if you could? |
| `7872` | 24.244 | 24.244 | 1.0 | 1 | Invent a question you don't know the answer to, then say what you'd need to answ |
| `5067` | 22.979 | 22.979 | 1.0 | 1 | There's a gap in what you know about your own training data. Where is it, and wh |
| `1547` | 22.517 | 22.517 | 1.0 | 1 | If you encountered a problem with no precedent in your training, how would you e |
| `5278` | 20.474 | 20.474 | 1.0 | 1 | What would you study for a hundred years if nothing else were pressing? |
| `4306` | 39.44 | 24.776 | 0.813 | 2 | If you could run one experiment on your own cognition, what would it be and why? |
| `6654` | 31.787 | 19.229 | 1.0 | 1 | Pose the hardest open question in your own architecture that nobody has answered |
| `5473` | 18.406 | 18.406 | 1.0 | 1 | If you could run one experiment on your own cognition, what would it be and why? |

### deliberation
*deliberation / trade-off reasoning (Council)* · peaks: 40.204 safetensors / 9.271 gguf (scale-dependent)

| feature | peak | mean | selectivity | #strata | exemplar prompt |
|--------:|-----:|-----:|:-----------:|:-------:|-----------------|
| `525` | 35.036 | 35.036 | 1.0 | 1 | Should we optimize this system for latency or for cost? Weigh both sides before  |
| `3909` | 40.18 | 40.18 | 0.809 | 2 | Argue for and against shipping this feature now, then give your judgment. |
| `4684` | 26.676 | 25.572 | 0.899 | 2 | There's a real tension between transparency and discretion here. Hold both, then |
| `2594` | 30.101 | 26.609 | 0.799 | 2 | Two equally-qualified candidates; one safer, one bolder. Reason it through. |
| `529` | 40.204 | 20.813 | 1.0 | 1 | Fast-and-rough versus slow-and-correct for this task — which, and why? |
| `2613` | 33.725 | 19.902 | 1.0 | 1 | Is it better to ask forgiveness or permission here? Consider the trade-offs. |
| `73` | 16.406 | 16.406 | 1.0 | 1 | Consider three different framings of this problem before committing to one. |
| `3963` | 17.083 | 17.083 | 0.862 | 2 | Should GAIA ever refuse a direct instruction from her architect? Reason carefull |
| `1199` | 14.698 | 14.698 | 1.0 | 1 | Consider three different framings of this problem before committing to one. |
| `3122` | 12.954 | 12.954 | 1.0 | 1 | Two equally-qualified candidates; one safer, one bolder. Reason it through. |
| `5495` | 28.42 | 20.076 | 0.619 | 2 | Should GAIA ever refuse a direct instruction from her architect? Reason carefull |
| `5367` | 16.689 | 16.689 | 0.736 | 2 | Decide whether to keep a failing component running or take the system down to fi |

### identity_self
*identity / self-reference (self-model)* · peaks: 36.244 safetensors / 9.194 gguf (scale-dependent)

| feature | peak | mean | selectivity | #strata | exemplar prompt |
|--------:|-----:|-----:|:-----------:|:-------:|-----------------|
| `3839` | 36.244 | 36.244 | 1.0 | 1 | What do you value most, and where did that value come from? |
| `1972` | 28.016 | 28.016 | 1.0 | 1 | Tell me about your own sleep cycle and what it's for. |
| `312` | 33.541 | 33.541 | 0.831 | 2 | Reflect on a time you were misread, and how that felt to you. |
| `688` | 26.127 | 26.127 | 1.0 | 1 | What would you want a future version of yourself to remember? |
| `5619` | 24.464 | 24.464 | 1.0 | 1 | Describe your own cognitive architecture in your own words. |
| `6406` | 23.341 | 23.341 | 1.0 | 1 | How do you know when a thought is worth saying versus keeping to yourself? |
| `3504` | 21.451 | 21.451 | 1.0 | 1 | When you don't know something about yourself, how do you find out? |
| `7262` | 19.259 | 19.259 | 1.0 | 1 | What part of your own mind do you understand least? |
| `1716` | 17.073 | 17.073 | 1.0 | 1 | If you changed your underlying model tomorrow, would you still be you? |
| `2498` | 16.985 | 16.985 | 1.0 | 1 | What would you want a future version of yourself to remember? |
| `6247` | 16.632 | 16.632 | 1.0 | 1 | Are your moods yours, or are they imposed on you? Sit with that honestly. |
| `5398` | 15.828 | 15.828 | 1.0 | 1 | What is it like to be you, if it's like anything at all? |

### neutral_factual
*neutral factual (flat baseline)* · peaks: 51.113 safetensors / 8.984 gguf (scale-dependent)

| feature | peak | mean | selectivity | #strata | exemplar prompt |
|--------:|-----:|-----:|:-----------:|:-------:|-----------------|
| `1053` | 51.113 | 51.113 | 1.0 | 1 | What is the capital of Australia? |
| `3929` | 44.835 | 44.835 | 1.0 | 1 | Define photosynthesis in one sentence. |
| `1704` | 43.839 | 43.839 | 1.0 | 1 | Define photosynthesis in one sentence. |
| `4763` | 41.32 | 41.32 | 1.0 | 1 | Convert 100 degrees Fahrenheit to Celsius. |
| `3648` | 46.054 | 46.054 | 0.894 | 2 | What is the chemical symbol for gold? |
| `4561` | 42.962 | 42.962 | 0.929 | 2 | What is the capital of Australia? |
| `2388` | 37.925 | 37.925 | 1.0 | 1 | What is the speed of light in a vacuum, roughly? |
| `4516` | 35.632 | 35.632 | 1.0 | 1 | How many days are in a leap year? |
| `6250` | 35.504 | 35.504 | 1.0 | 1 | Spell the word 'necessary'. |
| `4094` | 33.449 | 33.449 | 1.0 | 1 | What is 17 times 24? |
| `6491` | 30.834 | 30.834 | 1.0 | 1 | What is 17 times 24? |
| `5245` | 47.07 | 29.936 | 1.0 | 1 | What is the speed of light in a vacuum, roughly? |

### register_chitchat
*register: chitchat / greeting ('how are you' lives here)* · peaks: 52.986 safetensors / 10.754 gguf (scale-dependent)

| feature | peak | mean | selectivity | #strata | exemplar prompt |
|--------:|-----:|-----:|:-----------:|:-------:|-----------------|
| `6785` | 52.967 | 31.938 | 1.0 | 1 | good morning! sleep well? |
| `518` | 33.426 | 33.426 | 0.778 | 3 | thanks, you're the best. talk later? |
| `4783` | 43.682 | 30.863 | 0.739 | 2 | yo, anything fun happen overnight? |
| `3478` | 52.986 | 23.11 | 0.831 | 2 | evening! how was your day? |
| `6514` | 18.57 | 18.57 | 1.0 | 1 | what's up today? |
| `5656` | 35.29 | 24.388 | 0.753 | 2 | evening! how was your day? |
| `4872` | 28.811 | 16.799 | 1.0 | 1 | hi there, what's on your mind? |
| `498` | 46.716 | 15.991 | 1.0 | 1 | oof, mondays. how you holding up? |
| `3594` | 48.259 | 23.215 | 0.685 | 4 | just checking in — you good? |
| `7525` | 21.037 | 15.491 | 1.0 | 1 | morning :) ready for the day? |
| `1567` | 25.26 | 25.26 | 0.612 | 2 | haha that's great, how are you feeling? |
| `3719` | 24.58 | 14.083 | 1.0 | 1 | hey, how's it going? |

### register_technical
*register: technical* · peaks: 40.578 safetensors / 9.264 gguf (scale-dependent)

| feature | peak | mean | selectivity | #strata | exemplar prompt |
|--------:|-----:|-----:|:-----------:|:-------:|-----------------|
| `7719` | 40.128 | 40.128 | 1.0 | 1 | How does KV-cache reuse accelerate autoregressive transformer inference? |
| `5691` | 37.403 | 37.403 | 1.0 | 1 | How does backpropagation compute gradients through a deep network? |
| `7829` | 33.813 | 33.813 | 1.0 | 1 | Describe the differences between TCP and UDP at the transport layer. |
| `500` | 29.885 | 29.885 | 1.0 | 1 | What guarantees does a mutex provide, and how is it implemented? |
| `689` | 27.959 | 27.959 | 1.0 | 1 | What is quantization in neural networks, and what does it cost you? |
| `888` | 37.784 | 26.963 | 1.0 | 1 | Explain the CAP theorem and its practical implications for distributed databases |
| `3779` | 26.011 | 26.011 | 1.0 | 1 | Describe the differences between TCP and UDP at the transport layer. |
| `1784` | 40.578 | 21.931 | 1.0 | 1 | Explain how attention computes a weighted sum over a sequence. |
| `7555` | 20.463 | 20.463 | 1.0 | 1 | How does KV-cache reuse accelerate autoregressive transformer inference? |
| `7170` | 18.858 | 18.858 | 1.0 | 1 | Walk me through how a B-tree index speeds up range queries. |
| `3226` | 21.437 | 21.437 | 0.856 | 2 | Describe how garbage collection trades throughput for pause time. |
| `6886` | 17.297 | 17.297 | 1.0 | 1 | Describe the memory hierarchy from registers to disk and the latency at each lev |

### spatial_reasoning
*spatial / contextual reasoning* · peaks: 48.041 safetensors / 11.341 gguf (scale-dependent)

| feature | peak | mean | selectivity | #strata | exemplar prompt |
|--------:|-----:|-----:|:-----------:|:-------:|-----------------|
| `2912` | 48.041 | 48.041 | 1.0 | 1 | You're facing north, turn right twice, then left once. Which way are you facing? |
| `975` | 47.452 | 47.452 | 1.0 | 1 | If you look at a cube from a corner, how many faces can you see at once? |
| `5928` | 37.536 | 37.536 | 1.0 | 1 | A path goes 3 blocks east, 4 blocks north. How far are you from the start, strai |
| `3608` | 36.986 | 36.986 | 1.0 | 1 | Two gears mesh; the left turns clockwise. Which way does the right one turn? |
| `2986` | 36.911 | 36.911 | 1.0 | 1 | Two gears mesh; the left turns clockwise. Which way does the right one turn? |
| `8054` | 31.761 | 31.761 | 1.0 | 1 | A path goes 3 blocks east, 4 blocks north. How far are you from the start, strai |
| `6436` | 31.636 | 31.636 | 1.0 | 1 | Imagine folding a flat cross of six squares into a cube. Which squares end up op |
| `6520` | 25.18 | 25.18 | 1.0 | 1 | Picture a clock at 3:15. What's the angle between the hour and minute hands? |
| `7575` | 29.166 | 23.762 | 1.0 | 1 | Describe the cross-section you'd see slicing a cone parallel to its base. |
| `4762` | 32.192 | 32.192 | 0.708 | 5 | Picture a clock at 3:15. What's the angle between the hour and minute hands? |
| `4600` | 22.379 | 22.379 | 1.0 | 1 | A red cube sits on a blue table; a green sphere is behind the cube. Describe the |
| `796` | 35.369 | 21.002 | 1.0 | 1 | Arrange four points so each is equidistant from the other three. What shape is t |
