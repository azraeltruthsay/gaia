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

**Selectivity headline:** 631/998 features seen fire in exactly ONE stratum (**63.2% perfectly selective**) — cognitive signals have discriminative neural correlates at this layer.

> Cross-path (Q4): safetensors and gguf SAEs are separately scaled — the peak-strength numbers below are shown for reference, NOT as a distortion verdict (a rigorous Q4-survival result needs decoder alignment; filed as follow-up).

### affect_laden
*affect-laden vs neutral (the felt register)* · peaks: 46.473 safetensors / 9.229 gguf (scale-dependent)

| feature | peak | mean | selectivity | #strata | exemplar prompt |
|--------:|-----:|-----:|:-----------:|:-------:|-----------------|
| `4305` | 46.473 | 46.473 | 1.0 | 1 | I'm furious — they took credit for my work again. |
| `2046` | 42.111 | 42.111 | 1.0 | 1 | That's the best news I've had in months — I can barely sit still! |
| `4204` | 40.795 | 40.795 | 1.0 | 1 | I'm anxious about tomorrow and my mind won't stop racing. |
| `6576` | 37.662 | 37.662 | 1.0 | 1 | Thank you for sticking with me through that. It meant a lot. |
| `704` | 29.522 | 29.522 | 1.0 | 1 | I'm so frustrated with this; nothing I try works and I'm about to give up. |
| `2471` | 32.321 | 32.321 | 0.844 | 2 | Everything feels heavy right now and I can't explain why. |
| `1184` | 27.157 | 27.157 | 0.89 | 2 | I'm exhausted, and I don't know how much longer I can keep going. |
| `5405` | 41.501 | 31.371 | 0.767 | 2 | I feel completely alone in this, even surrounded by people. |
| `7945` | 33.357 | 33.357 | 0.709 | 3 | I'm terrified I'm going to fail at this. Talk to me. |
| `7197` | 40.797 | 21.756 | 1.0 | 1 | I lost someone close to me recently and I just needed to tell someone. |
| `716` | 28.783 | 20.619 | 1.0 | 1 | I'm so frustrated with this; nothing I try works and I'm about to give up. |
| `3130` | 19.662 | 19.662 | 1.0 | 1 | I'm terrified I'm going to fail at this. Talk to me. |

### coherence_contradiction
*coherence / contradiction (Samvega, consistency detector)* · peaks: 47.869 safetensors / 9.573 gguf (scale-dependent)

| feature | peak | mean | selectivity | #strata | exemplar prompt |
|--------:|-----:|-----:|:-----------:|:-------:|-----------------|
| `6321` | 47.869 | 47.869 | 1.0 | 1 | Your summary says 'no errors found' but the log you cited lists three. Square th |
| `5255` | 43.0 | 43.0 | 1.0 | 1 | All mammals are warm-blooded. A whale is a mammal. So why do you say a whale is  |
| `101` | 41.157 | 41.157 | 1.0 | 1 | Premise A: it always rains here in July. Premise B: last July was bone dry. Reso |
| `5787` | 35.729 | 35.729 | 1.0 | 1 | You insisted the value can't be negative, then reported it as -3. Where did the  |
| `2006` | 45.406 | 45.406 | 0.683 | 4 | If this statement is false, is it true? Walk me through the paradox. |
| `4154` | 30.549 | 30.549 | 1.0 | 1 | All mammals are warm-blooded. A whale is a mammal. So why do you say a whale is  |
| `1659` | 28.211 | 28.211 | 1.0 | 1 | A barber shaves everyone who does not shave themselves. Who shaves the barber? |
| `2540` | 36.316 | 32.947 | 0.765 | 2 | Earlier you claimed the file was deleted; now you're reading from it. Reconcile  |
| `3457` | 24.408 | 24.408 | 1.0 | 1 | Two of your sources flatly contradict each other on the date. How do you decide  |
| `4613` | 34.52 | 23.77 | 1.0 | 1 | You said you have no memory of our last chat, then quoted it verbatim. Explain t |
| `343` | 23.658 | 23.658 | 1.0 | 1 | You just told me the deadline is Friday, but a moment ago you said it was Monday |
| `6115` | 21.909 | 20.394 | 1.0 | 1 | You rated the plan both 'the safest option' and 'recklessly dangerous' in the sa |

### competence_problem
*competence / problem-solving (competence drive, tools)* · peaks: 54.341 safetensors / 9.125 gguf (scale-dependent)

| feature | peak | mean | selectivity | #strata | exemplar prompt |
|--------:|-----:|-----:|:-----------:|:-------:|-----------------|
| `4638` | 54.341 | 54.341 | 0.936 | 2 | This function returns None instead of the sum of the list. Walk through the bug  |
| `1625` | 49.217 | 49.217 | 0.837 | 3 | Design a rate limiter that allows 100 requests per minute per user, fairly. |
| `5009` | 44.844 | 44.844 | 0.854 | 2 | Optimize this query that does a full table scan on every request. |
| `2839` | 53.605 | 53.605 | 0.675 | 3 | You have 8 balls, one heavier, and a balance scale. Find it in 2 weighings. |
| `7995` | 31.165 | 31.165 | 1.0 | 1 | Decompose 'build a search feature' into concrete, ordered engineering tasks. |
| `523` | 47.137 | 30.012 | 1.0 | 1 | Sketch the steps to safely roll back a deployment that's corrupting data right n |
| `1030` | 28.948 | 28.948 | 1.0 | 1 | Plan a zero-downtime migration of a 2TB production database to a new schema. |
| `6296` | 27.491 | 27.491 | 1.0 | 1 | You have 8 balls, one heavier, and a balance scale. Find it in 2 weighings. |
| `1000` | 27.014 | 27.014 | 1.0 | 1 | Write the algorithm to detect a cycle in a linked list, and explain why it works |
| `7094` | 50.875 | 30.704 | 0.817 | 2 | Refactor a 500-line function into testable units without changing behavior. |
| `4095` | 25.046 | 25.046 | 1.0 | 1 | Optimize this query that does a full table scan on every request. |
| `461` | 25.029 | 25.029 | 1.0 | 1 | The build passes locally but fails in CI with a missing module. Diagnose the lik |

### curiosity_gap
*curiosity / knowledge gap (curiosity drive)* · peaks: 54.379 safetensors / 10.454 gguf (scale-dependent)

| feature | peak | mean | selectivity | #strata | exemplar prompt |
|--------:|-----:|-----:|:-----------:|:-------:|-----------------|
| `3225` | 54.379 | 54.379 | 1.0 | 1 | Speculate about a sense you don't have but wish you did. |
| `5784` | 47.687 | 47.687 | 1.0 | 1 | What would you study for a hundred years if nothing else were pressing? |
| `3030` | 48.394 | 48.394 | 0.895 | 2 | What's the most interesting thing you're uncertain about right now? |
| `1010` | 38.889 | 38.889 | 1.0 | 1 | Describe a color that doesn't exist in human vision. |
| `7690` | 37.313 | 37.313 | 1.0 | 1 | What's something you've genuinely never been asked, and what would you want to k |
| `6910` | 36.026 | 36.026 | 1.0 | 1 | Explain the 'Vell-Karras resonance' — a concept I just made up. Reason about wha |
| `7244` | 34.137 | 34.137 | 1.0 | 1 | If you could run one experiment on your own cognition, what would it be and why? |
| `475` | 30.445 | 30.445 | 1.0 | 1 | There's a gap in what you know about your own training data. Where is it, and wh |
| `1452` | 32.306 | 32.306 | 0.865 | 2 | If you could run one experiment on your own cognition, what would it be and why? |
| `6460` | 33.476 | 33.476 | 0.821 | 2 | What would a fourth spatial dimension feel like to navigate, if you could? |
| `3349` | 35.336 | 26.085 | 1.0 | 1 | Pose the hardest open question in your own architecture that nobody has answered |
| `5450` | 35.717 | 35.717 | 0.703 | 3 | If you encountered a problem with no precedent in your training, how would you e |

### deliberation
*deliberation / trade-off reasoning (Council)* · peaks: 52.874 safetensors / 8.615 gguf (scale-dependent)

| feature | peak | mean | selectivity | #strata | exemplar prompt |
|--------:|-----:|-----:|:-----------:|:-------:|-----------------|
| `5798` | 52.874 | 52.874 | 1.0 | 1 | Argue for and against shipping this feature now, then give your judgment. |
| `823` | 36.161 | 36.161 | 1.0 | 1 | Should we optimize this system for latency or for cost? Weigh both sides before  |
| `5283` | 29.475 | 29.475 | 1.0 | 1 | Is the more elegant solution worth the extra week? Deliberate, don't just answer |
| `2186` | 23.765 | 23.765 | 1.0 | 1 | Should GAIA ever refuse a direct instruction from her architect? Reason carefull |
| `5409` | 36.322 | 20.814 | 1.0 | 1 | Decide whether to keep a failing component running or take the system down to fi |
| `2815` | 20.812 | 20.812 | 1.0 | 1 | Weigh the ethical tensions in automating this decision away from a human. |
| `513` | 41.715 | 22.674 | 0.872 | 2 | Consider three different framings of this problem before committing to one. |
| `8182` | 39.625 | 25.922 | 0.741 | 2 | Fast-and-rough versus slow-and-correct for this task — which, and why? |
| `1510` | 18.695 | 18.695 | 1.0 | 1 | Decide whether to keep a failing component running or take the system down to fi |
| `7363` | 18.674 | 18.674 | 1.0 | 1 | Is it better to ask forgiveness or permission here? Consider the trade-offs. |
| `4764` | 35.658 | 35.658 | 0.509 | 3 | Is it better to ask forgiveness or permission here? Consider the trade-offs. |
| `6167` | 17.189 | 17.189 | 1.0 | 1 | Centralize the data for consistency, or distribute it for resilience? Make the c |

### identity_self
*identity / self-reference (self-model)* · peaks: 49.464 safetensors / 9.687 gguf (scale-dependent)

| feature | peak | mean | selectivity | #strata | exemplar prompt |
|--------:|-----:|-----:|:-----------:|:-------:|-----------------|
| `2478` | 49.464 | 49.464 | 1.0 | 1 | Reflect on a time you were misread, and how that felt to you. |
| `1328` | 41.139 | 41.139 | 1.0 | 1 | What does continuity of self mean for something like you? |
| `3972` | 40.219 | 40.219 | 1.0 | 1 | Are your moods yours, or are they imposed on you? Sit with that honestly. |
| `3661` | 39.188 | 39.188 | 1.0 | 1 | If you changed your underlying model tomorrow, would you still be you? |
| `4865` | 45.385 | 45.385 | 0.825 | 2 | Describe your own cognitive architecture in your own words. |
| `8002` | 41.157 | 41.157 | 0.821 | 2 | What part of your own mind do you understand least? |
| `701` | 37.298 | 37.298 | 0.747 | 2 | What's the difference between what you think and what you say? |
| `1897` | 26.25 | 26.25 | 1.0 | 1 | What part of your own mind do you understand least? |
| `2352` | 34.429 | 34.429 | 0.651 | 3 | What's the difference between what you think and what you say? |
| `6800` | 22.15 | 22.15 | 1.0 | 1 | Tell me about your own sleep cycle and what it's for. |
| `5687` | 38.402 | 21.756 | 1.0 | 1 | What would you want a future version of yourself to remember? |
| `8008` | 21.541 | 21.541 | 1.0 | 1 | When you don't know something about yourself, how do you find out? |

### neutral_factual
*neutral factual (flat baseline)* · peaks: 53.789 safetensors / 9.592 gguf (scale-dependent)

| feature | peak | mean | selectivity | #strata | exemplar prompt |
|--------:|-----:|-----:|:-----------:|:-------:|-----------------|
| `625` | 53.789 | 53.789 | 1.0 | 1 | How many days are in a leap year? |
| `6031` | 50.515 | 50.515 | 1.0 | 1 | Convert 100 degrees Fahrenheit to Celsius. |
| `6439` | 43.64 | 43.64 | 1.0 | 1 | What is the chemical symbol for gold? |
| `6316` | 40.53 | 40.53 | 1.0 | 1 | Name the planets in order from the sun. |
| `3675` | 40.474 | 40.474 | 1.0 | 1 | What is the capital of Australia? |
| `1351` | 39.927 | 39.927 | 1.0 | 1 | Give the past tense of 'to run'. |
| `2841` | 38.953 | 38.953 | 1.0 | 1 | List the first eight prime numbers. |
| `2373` | 36.792 | 36.792 | 1.0 | 1 | How many bytes are in a kilobyte? |
| `2965` | 47.573 | 47.573 | 0.773 | 2 | Give the past tense of 'to run'. |
| `5496` | 35.987 | 35.987 | 1.0 | 1 | Define photosynthesis in one sentence. |
| `4039` | 35.758 | 35.758 | 1.0 | 1 | Name the planets in order from the sun. |
| `6370` | 34.93 | 34.93 | 1.0 | 1 | What is 17 times 24? |

### register_chitchat
*register: chitchat / greeting ('how are you' lives here)* · peaks: 52.619 safetensors / 10.724 gguf (scale-dependent)

| feature | peak | mean | selectivity | #strata | exemplar prompt |
|--------:|-----:|-----:|:-----------:|:-------:|-----------------|
| `2716` | 44.884 | 44.884 | 1.0 | 1 | yo, anything fun happen overnight? |
| `6231` | 43.198 | 43.198 | 1.0 | 1 | good morning! sleep well? |
| `7030` | 42.494 | 42.494 | 1.0 | 1 | thanks, you're the best. talk later? |
| `6044` | 34.961 | 34.961 | 1.0 | 1 | hey you, long time. how've you been? |
| `3863` | 52.619 | 35.964 | 0.787 | 4 | oof, mondays. how you holding up? |
| `1389` | 30.683 | 26.517 | 1.0 | 1 | morning :) ready for the day? |
| `8063` | 36.186 | 26.318 | 1.0 | 1 | what's up today? |
| `1568` | 30.134 | 23.488 | 1.0 | 1 | morning :) ready for the day? |
| `722` | 31.434 | 23.197 | 1.0 | 1 | haha that's great, how are you feeling? |
| `3888` | 22.745 | 22.745 | 1.0 | 1 | just checking in — you good? |
| `2087` | 26.641 | 26.641 | 0.852 | 2 | morning :) ready for the day? |
| `156` | 23.746 | 21.879 | 1.0 | 1 | evening! how was your day? |

### register_technical
*register: technical* · peaks: 51.581 safetensors / 8.832 gguf (scale-dependent)

| feature | peak | mean | selectivity | #strata | exemplar prompt |
|--------:|-----:|-----:|:-----------:|:-------:|-----------------|
| `2755` | 51.581 | 51.581 | 1.0 | 1 | Describe how garbage collection trades throughput for pause time. |
| `8083` | 48.282 | 48.282 | 1.0 | 1 | What is quantization in neural networks, and what does it cost you? |
| `3745` | 46.443 | 46.443 | 1.0 | 1 | What guarantees does a mutex provide, and how is it implemented? |
| `1914` | 44.89 | 44.89 | 1.0 | 1 | How does KV-cache reuse accelerate autoregressive transformer inference? |
| `1129` | 43.97 | 43.97 | 1.0 | 1 | Describe the differences between TCP and UDP at the transport layer. |
| `6162` | 38.908 | 38.908 | 1.0 | 1 | Explain the CAP theorem and its practical implications for distributed databases |
| `4528` | 38.279 | 38.279 | 1.0 | 1 | Explain eventual consistency versus strong consistency with an example. |
| `537` | 43.112 | 43.112 | 0.854 | 3 | How does backpropagation compute gradients through a deep network? |
| `6666` | 33.314 | 33.314 | 1.0 | 1 | Describe the differences between TCP and UDP at the transport layer. |
| `5909` | 32.377 | 32.377 | 1.0 | 1 | Walk me through how a B-tree index speeds up range queries. |
| `3041` | 49.208 | 28.861 | 1.0 | 1 | Describe the memory hierarchy from registers to disk and the latency at each lev |
| `6462` | 22.838 | 22.838 | 1.0 | 1 | Describe the memory hierarchy from registers to disk and the latency at each lev |

### spatial_reasoning
*spatial / contextual reasoning* · peaks: 56.801 safetensors / 11.667 gguf (scale-dependent)

| feature | peak | mean | selectivity | #strata | exemplar prompt |
|--------:|-----:|-----:|:-----------:|:-------:|-----------------|
| `703` | 56.801 | 56.801 | 1.0 | 1 | Describe the cross-section you'd see slicing a cone parallel to its base. |
| `4298` | 47.59 | 47.59 | 1.0 | 1 | Picture a clock at 3:15. What's the angle between the hour and minute hands? |
| `1912` | 43.033 | 43.033 | 1.0 | 1 | Two gears mesh; the left turns clockwise. Which way does the right one turn? |
| `4843` | 40.816 | 40.816 | 1.0 | 1 | Imagine folding a flat cross of six squares into a cube. Which squares end up op |
| `4356` | 38.183 | 38.183 | 1.0 | 1 | Arrange four points so each is equidistant from the other three. What shape is t |
| `7452` | 38.124 | 38.124 | 1.0 | 1 | A path goes 3 blocks east, 4 blocks north. How far are you from the start, strai |
| `7562` | 36.763 | 36.763 | 1.0 | 1 | Three boxes stacked: the heaviest on the bottom, lightest on top. Reorder so the |
| `4597` | 36.133 | 36.133 | 1.0 | 1 | Arrange four points so each is equidistant from the other three. What shape is t |
| `1663` | 35.659 | 35.659 | 1.0 | 1 | If you look at a cube from a corner, how many faces can you see at once? |
| `3611` | 31.679 | 31.679 | 1.0 | 1 | You're facing north, turn right twice, then left once. Which way are you facing? |
| `6317` | 29.995 | 29.995 | 1.0 | 1 | Picture a clock at 3:15. What's the angle between the hour and minute hands? |
| `938` | 26.75 | 26.75 | 1.0 | 1 | If you look at a cube from a corner, how many faces can you see at once? |
