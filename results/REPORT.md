# LLM-embeds rerun — report

Generated 2026-09-04 19:37 by `python -m src.run_all` at git 59a3fab; python 3.12.3, numpy 2.5.2, host promaxgb10-e746.
Seed 2026 throughout. Original notebooks untouched; code in `src/`.

## Phase 1 — define(a+b), controls, define2, analogies

### 1.1 Reproduction of the p. 16 table (GPT-3 Curie, raw)

| a + b → target | cos(a, a+b) | paper | cos(target, a+b) | paper | match | next three |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| young + dog → puppy | 0.9423 | 0.9423 | 0.9045 | 0.9045 | ✓ | puppy 0.905, youth 0.901, child 0.898 |
| young + cat → kitten | 0.9331 | 0.9331 | 0.8997 | 0.8997 | ✓ | kitten 0.900, kid 0.890, baby 0.889 |
| young + duck → duckling | 0.9337 | 0.9337 | 0.9064 | 0.9064 | ✓ | duckling 0.906, youth 0.893, baby 0.880 |
| female + spouse → wife | 0.9573 | 0.9573 | 0.9390 | 0.9389 | ✓ | wife 0.939, woman 0.930, girlfriend 0.909 |
| male + spouse → husband | 0.9542 | 0.9542 | 0.9303 | 0.9303 | ✓ | husband 0.930, wife 0.909, partner 0.903 |
| male + sibling → brother | 0.9515 | 0.9515 | 0.9233 | 0.9233 | ✓ | brother 0.923, siblings 0.909, adult 0.890 |
| female + sibling → sister | 0.9507 | 0.9507 | 0.9318 | 0.9318 | ✓ | sister 0.932, siblings 0.911, daughter 0.903 |

**Reading.** All seven cosines reproduce to four decimals, so the cached Curie file and this loader are the same data the paper used. cos(a, a+b) equals cos(b, a+b) exactly because Curie vectors are unit length, so a and b heading the list is arithmetic, not a finding.

### 1.2 Controls, Curie, raw

| a + b → target | rank under v(a) | cos | rank under v(b) | cos | rank under v(a)+v(b) | cos |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| young + dog → puppy | 283 | 0.8000 | 1 | 0.9046 | 1 | 0.9045 |
| young + cat → kitten | 885 | 0.7828 | 1 | 0.8962 | 1 | 0.8997 |
| young + duck → duckling | 537 | 0.7909 | 1 | 0.9017 | 1 | 0.9064 |
| female + spouse → wife | 10 | 0.8719 | 2 | 0.9258 | 1 | 0.9390 |
| male + spouse → husband | 11 | 0.8568 | 3 | 0.9186 | 1 | 0.9303 |
| male + sibling → brother | 21 | 0.8477 | 3 | 0.9093 | 1 | 0.9233 |
| female + sibling → sister | 21 | 0.8595 | 2 | 0.9122 | 1 | 0.9318 |
| man + unmarried → bachelor | 373 | 0.8029 | 23 | 0.8245 | 18 | 0.8651 |

Random-pair baseline (1000 seeded pairs, best non-a/b neighbour of v(a)+v(b)): mean 0.9007, p10 0.8824, p50 0.9010, p90 0.9187.

### 1.2 Controls, Curie, center

| a + b → target | rank under v(a) | cos | rank under v(b) | cos | rank under v(a)+v(b) | cos |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| young + dog → puppy | 73 | 0.2024 | 1 | 0.6361 | 1 | 0.5780 |
| young + cat → kitten | 246 | 0.1283 | 1 | 0.6369 | 1 | 0.5548 |
| young + duck → duckling | 37 | 0.2502 | 1 | 0.6769 | 1 | 0.6691 |
| female + spouse → wife | 13 | 0.4203 | 2 | 0.6870 | 1 | 0.7025 |
| male + spouse → husband | 8 | 0.3901 | 3 | 0.6720 | 1 | 0.6857 |
| male + sibling → brother | 20 | 0.3337 | 3 | 0.6263 | 1 | 0.6321 |
| female + sibling → sister | 14 | 0.4003 | 2 | 0.6488 | 1 | 0.6966 |
| man + unmarried → bachelor | 74 | 0.2062 | 11 | 0.3557 | 6 | 0.3881 |

Random-pair baseline (1000 seeded pairs, best non-a/b neighbour of v(a)+v(b)): mean 0.5009, p10 0.3838, p50 0.5021, p90 0.6153.

**Reading.** Ranks exclude a and b themselves. For puppy, kitten, duckling the target is already the nearest neighbour of v(b) alone; adding v(a) changes the cosine by a few thousandths and the rank not at all. The kinship cases move the target from rank 2–3 under v(b) to rank 1 under the sum, a real but small effect. The bachelor example stays at rank 18. The baseline shows what a random sum scores against its best neighbour; a p. 16 cosine near the baseline mean is typical, not evidence of composition. Mean-centering removes the common component that inflates every raw cosine; compare the two tables to see which effects survive.

### 1.3 define2 with one cognate filter (plural + 3-letter prefix + edit distance ≤ 2), top-3 per target

**raw**

| target | paper (Curie) | curie | davinci | opt-13b | t5-3b | flan-t5-xxl |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| wife | spouse + woman 0.949 | spouse + woman 0.949<br>mom + spouse 0.945<br>Mrs + spouse 0.944 | husband + woman 0.951<br>spouse + woman 0.950<br>husband + spouse 0.948 | husband + worker 0.994<br>worker + writer 0.993<br>mate + worker 0.992 | husband + woman 0.905<br>father + woman 0.888<br>father + husband 0.878 | husband + spouse 0.870<br>girlfriend + husband 0.852<br>girlfriend + spouse 0.844 |
| duckling | duck + youngster 0.926 | chicken + youngster 0.881<br>chicken + cute 0.878<br>feather + youngster 0.877 | bird + puppy 0.880<br>chicken + puppy 0.880<br>bird + kitten 0.879 | cheerful + puppy 0.736<br>produce + puppy 0.733<br>neighbouring + puppy 0.725 | dancer + poisonous 0.739<br>dancer + disgusting 0.715<br>dancer + thirsty 0.711 | boar + chicken 0.519<br>chicken + kitten 0.514<br>chicken + puppy 0.514 |
| puppy | dog + kitten 0.932 | dog + kitten 0.932<br>cute + dog 0.926<br>baby + dog 0.921 | dog + kitten 0.938<br>cute + dog 0.925<br>kitten + pet 0.918 | furniture + kitten 0.764<br>competence + kitten 0.761<br>kitten + sophisticated 0.761 | buddy + kitten 0.768<br>dog + kitten 0.764<br>dog + niece 0.755 | dog + kitten 0.866<br>baby + dog 0.833<br>dog + infant 0.829 |
| foal | horse + puppy 0.909 | — | — | — | — | — |
| freedom | independence + liberty 0.946 | independence + liberty 0.946<br>autonomy + liberty 0.936<br>independence + liberation 0.936 | independence + liberty 0.947<br>independence + liberation 0.945<br>liberation + liberty 0.941 | daughter + knowledge 0.989<br>awareness + dog 0.988<br>determination + dog 0.988 | ease + liberty 0.765<br>frustration + liberty 0.761<br>liberty + love 0.761 | independence + liberty 0.871<br>democracy + independence 0.851<br>independence + joy 0.851 |
| autonomy | control + independence 0.918 | control + independence 0.918<br>ability + independence 0.917<br>independence + voluntary 0.916 | independent + sovereignty 0.934<br>independence + solo 0.929<br>independence + sovereignty 0.929 | ideology + literacy 0.801<br>literacy + ministry 0.799<br>literacy + sovereignty 0.796 | methodology + sovereignty 0.752<br>reliability + sovereignty 0.751<br>complexity + sovereignty 0.750 | independence + sovereignty 0.817<br>flexibility + independence 0.817<br>independence + mobility 0.815 |
| justice | fairness + judicial 0.917 | fairness + judicial 0.917<br>judicial + revenge 0.913<br>equality + judicial 0.912 | judicial + moral 0.934<br>judicial + peace 0.933<br>fairness + judicial 0.933 | bass + existence 0.981<br>cheat + existence 0.980<br>existence + life 0.980 | judge + mercy 0.788<br>mercy + prison 0.777<br>crime + mercy 0.773 | court + equality 0.777<br>equality + judge 0.772<br>equality + judicial 0.768 |
| knowledge | information + wisdom 0.934 | information + wisdom 0.934<br>educated + information 0.930<br>expertise + information 0.928 | information + wisdom 0.940<br>information + learn 0.935<br>educated + information 0.934 | awareness + product 0.990<br>direction + information 0.989<br>information + product 0.989 | awareness + skill 0.779<br>skill + understanding 0.773<br>expertise + understanding 0.766 | awareness + expertise 0.772<br>aware + expertise 0.768<br>awareness + information 0.765 |
| rationality | logical + reasoning 0.924 | — | — | — | — | — |
| causation | consequence + correlation 0.910 | — | — | — | — | — |

Null distribution of the best define2 score (100 random unit-vector targets; 100 random real-word targets with the same filter):

| model | unit mean | unit p90 | unit max | word mean | word p90 | word max |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| curie | 0.0292 | 0.0478 | 0.0757 | 0.9172 | 0.9378 | 0.9552 |
| davinci | 0.0148 | 0.0237 | 0.0387 | 0.9238 | 0.9403 | 0.9518 |
| opt-13b | 0.0434 | 0.0575 | 0.0688 | 0.8406 | 0.9891 | 0.9935 |
| t5-3b | 0.1072 | 0.1337 | 0.1987 | 0.7744 | 0.8434 | 0.9273 |
| flan-t5-xxl | 0.0514 | 0.0674 | 0.0899 | 0.7862 | 0.8512 | 0.9003 |

**center**

| target | paper (Curie) | curie | davinci | opt-13b | t5-3b | flan-t5-xxl |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| wife | spouse + woman 0.949 | spouse + woman 0.756<br>mom + spouse 0.738<br>girlfriend + spouse 0.738 | husband + spouse 0.730<br>husband + woman 0.727<br>spouse + woman 0.721 | husband + worker 0.985<br>worker + writer 0.984<br>mate + worker 0.983 | husband + woman 0.818<br>father + woman 0.783<br>father + husband 0.765 | husband + spouse 0.767<br>girlfriend + husband 0.732<br>girlfriend + spouse 0.717 |
| duckling | duck + youngster 0.926 | chicken + youngster 0.568<br>chicken + cute 0.546<br>feather + youngster 0.546 | chicken + puppy 0.471<br>chicken + kitten 0.458<br>bird + puppy 0.457 | produce + puppy 0.775<br>producer + puppy 0.769<br>cheerful + puppy 0.768 | dancer + poisonous 0.733<br>dancer + disgusting 0.707<br>dancer + thirsty 0.699 | boar + chicken 0.438<br>chicken + pig 0.431<br>boar + kitten 0.431 |
| puppy | dog + kitten 0.932 | dog + kitten 0.721<br>cute + dog 0.687<br>baby + dog 0.663 | dog + kitten 0.715<br>cute + dog 0.640<br>kitten + pet 0.611 | furniture + kitten 0.787<br>kitten + prestigious 0.786<br>competence + kitten 0.786 | buddy + kitten 0.575<br>dog + kitten 0.570<br>dog + niece 0.548 | dog + kitten 0.748<br>dog + pet 0.669<br>baby + dog 0.663 |
| foal | horse + puppy 0.909 | — | — | — | — | — |
| freedom | independence + liberty 0.946 | independence + liberty 0.729<br>independence + liberation 0.675<br>autonomy + liberty 0.668 | independence + liberty 0.688<br>independence + liberation 0.681<br>liberation + liberty 0.662 | daughter + knowledge 0.972<br>awareness + dog 0.971<br>determination + mother 0.971 | liberty + love 0.571<br>ease + liberty 0.571<br>liberty + peace 0.569 | independence + liberty 0.670<br>democracy + independence 0.628<br>autonomy + liberty 0.595 |
| autonomy | control + independence 0.918 | independence + mobility 0.595<br>independence + sovereignty 0.594<br>independence + independent 0.591 | independent + sovereignty 0.672<br>independence + independent 0.666<br>independence + independently 0.663 | morality + predecessor 0.657<br>literacy + territory 0.657<br>literacy + sovereignty 0.655 | complexity + sovereignty 0.585<br>methodology + sovereignty 0.585<br>reliability + sovereignty 0.582 | independence + sovereignty 0.598<br>flexibility + independence 0.590<br>independence + mobility 0.580 |
| justice | fairness + judicial 0.917 | fairness + judicial 0.574<br>judicial + revenge 0.543<br>equality + judicial 0.542 | judicial + moral 0.598<br>judgement + judicial 0.595<br>fairness + judicial 0.592 | bass + existence 0.952<br>corruption + everything 0.950<br>cheat + existence 0.950 | judge + mercy 0.583<br>mercy + prison 0.572<br>crime + mercy 0.559 | equality + fairness 0.539<br>democracy + fairness 0.530<br>court + equality 0.526 |
| knowledge | information + wisdom 0.934 | information + wisdom 0.660<br>educated + information 0.639<br>expertise + information 0.623 | information + wisdom 0.649<br>educated + information 0.625<br>information + learn 0.614 | awareness + product 0.975<br>awareness + system 0.973<br>direction + information 0.973 | awareness + skill 0.554<br>skill + understanding 0.542<br>expertise + understanding 0.536 | awareness + expertise 0.437<br>awareness + information 0.434<br>awareness + wisdom 0.432 |
| rationality | logical + reasoning 0.924 | — | — | — | — | — |
| causation | consequence + correlation 0.910 | — | — | — | — | — |

Null distribution of the best define2 score (100 random unit-vector targets; 100 random real-word targets with the same filter):

| model | unit mean | unit p90 | unit max | word mean | word p90 | word max |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| curie | 0.0765 | 0.0843 | 0.0971 | 0.6027 | 0.7299 | 0.7712 |
| davinci | 0.0442 | 0.0481 | 0.0526 | 0.5712 | 0.6951 | 0.7857 |
| opt-13b | 0.0628 | 0.0707 | 0.0859 | 0.7792 | 0.9723 | 0.9851 |
| t5-3b | 0.1481 | 0.1652 | 0.1881 | 0.6416 | 0.7568 | 0.9156 |
| flan-t5-xxl | 0.0775 | 0.0843 | 0.0912 | 0.5468 | 0.6988 | 0.8493 |

Targets marked — (foal, rationality, causation) are not in the cached vocabulary: the notebooks fetched their vectors live from the retired OpenAI endpoint and never saved them, so the paper's numbers for them cannot be reproduced from the caches. Phase 3 embeds them with the open models.

**Reading.** The paper's pp. 18–19 numbers came from two filter settings (no filter for puppy/duckling/foal; plural + prefix for the rest). With one filter, compare the Curie column to the paper column: pairs that change are ones the original filter let through. In particular the paper's duckling = duck + youngster uses a cognate of the definiendum, which the paper's own definition of a definition forbids; with duck excluded the best Curie pair is chicken + youngster. The random-word null is the honest comparison: a target's best pair only means something if its score is well above what an arbitrary vocabulary word gets. Random unit vectors score far lower in the raw spaces because real words occupy a narrow cone; after centering the two nulls converge.

### 1.4 Analogies with the mikolov() bug fixed

Score = cos(v(a) − v(b) + v(c), v(d)); rank excludes a, b, c. The notebooks' values were cos(·, v("gpt")) and are not comparable.

**raw** — cells are rank (cosine; top-1 word)

| analogy | curie | davinci | opt-13b | t5-3b | flan-t5-xxl |
| :--- | :--- | :--- | :--- | :--- | :--- |
| king - man + woman -> queen | 1 (0.880; top: queen) | 2 (0.858; top: female) | 2423 (0.475; top: feeding) | 152 (0.379; top: cliff) | 1 (0.642; top: queen) |
| man - king + queen -> woman | 1 (0.869; top: woman) | 1 (0.844; top: woman) | 1716 (0.360; top: minister) | 1 (0.710; top: woman) | 1 (0.712; top: woman) |
| husband - man + woman -> wife | 1 (0.884; top: wife) | 1 (0.882; top: wife) | 2 (0.968; top: writer) | 1 (0.804; top: wife) | 1 (0.806; top: wife) |
| brother - male + female -> sister | 1 (0.912; top: sister) | 1 (0.924; top: sister) | 2720 (0.440; top: father) | 1 (0.809; top: sister) | 1 (0.863; top: sister) |
| father - man + woman -> mother | 1 (0.886; top: mother) | 1 (0.884; top: mother) | 6 (0.966; top: wife) | 7 (0.650; top: wife) | 1 (0.793; top: mother) |
| kitten - cat + dog -> puppy | 1 (0.911; top: puppy) | 1 (0.905; top: puppy) | 2926 (0.506; top: cocktail) | 1 (0.639; top: puppy) | 1 (0.826; top: puppy) |
| bigger - big + small -> smaller | skip | skip | skip | skip | skip |
| walked - walk + run -> ran | skip | skip | skip | skip | skip |

**center** — cells are rank (cosine; top-1 word)

| analogy | curie | davinci | opt-13b | t5-3b | flan-t5-xxl |
| :--- | :--- | :--- | :--- | :--- | :--- |
| king - man + woman -> queen | 1 (0.615; top: queen) | 2 (0.409; top: female) | 2687 (-0.401; top: feeding) | 301 (0.238; top: cliff) | 1 (0.463; top: queen) |
| man - king + queen -> woman | 1 (0.520; top: woman) | 2 (0.331; top: manner) | 3553 (-0.234; top: prince) | 1 (0.525; top: woman) | 1 (0.448; top: woman) |
| husband - man + woman -> wife | 1 (0.672; top: wife) | 1 (0.639; top: wife) | 2 (0.937; top: writer) | 1 (0.693; top: wife) | 1 (0.688; top: wife) |
| brother - male + female -> sister | 1 (0.709; top: sister) | 1 (0.644; top: sister) | 2732 (-0.393; top: father) | 1 (0.685; top: sister) | 1 (0.784; top: sister) |
| father - man + woman -> mother | 1 (0.648; top: mother) | 1 (0.538; top: mother) | 6 (0.932; top: daughter) | 7 (0.454; top: wife) | 1 (0.663; top: mother) |
| kitten - cat + dog -> puppy | 1 (0.682; top: puppy) | 1 (0.689; top: puppy) | 4 (0.285; top: boyfriend) | 1 (0.460; top: puppy) | 1 (0.701; top: puppy) |
| bigger - big + small -> smaller | skip | skip | skip | skip | skip |
| walked - walk + run -> ran | skip | skip | skip | skip | skip |

_Phase 1 ran in 2301 s._

## Phase 2 — magnitude study, redone on the cached embeddings

### 2.1 Coverage: pairs with both words in the cached vocabulary

| model | us_uk | plural | verb |
| :--- | ---: | ---: | ---: |
| opt-1.3b | 2/11 | 3/27 | 5/36 |
| opt-13b | 2/11 | 3/27 | 5/36 |
| t5-large | 2/11 | 3/27 | 5/36 |
| t5-3b | 2/11 | 3/27 | 5/36 |
| flan-t5-xxl | 2/11 | 3/27 | 5/36 |

The notebooks embedded out-of-vocabulary words on the fly and never saved them, so from the caches only these pairs can be scored. Missing words are listed in results/phase2.json; the GPU re-extraction in Phase 3 (`phase2_extra`) fills them where the original model could be loaded.

### 2.2 |%diff| in L1, raw — similar pairs vs 5000 random pairs (mean / median; p = Mann–Whitney, similar < random)

| model | random | us_uk | plural | verb | all pooled |
| :--- | :--- | :--- | :--- | :--- | :--- |
| opt-1.3b | 15.7 / 13.3 | 3.9 / 3.9 (n=2, p=0.0487*) | 10.8 / 7.9 (n=3, p=0.269) | 2.5 / 3.0 (n=5, p=0.00111**) | 5.3 / 3.3 (n=10, p=0.000601***) |
| opt-13b | 12.5 / 10.4 | 6.5 / 6.5 (n=2, p=0.192) | 7.8 / 9.5 (n=3, p=0.225) | 11.3 / 11.6 (n=5, p=0.519) | 9.3 / 8.0 (n=10, p=0.221) |
| t5-large | 13.3 / 9.4 | 21.1 / 21.1 (n=2, p=0.627) | 13.3 / 3.3 (n=3, p=0.321) | 15.6 / 6.0 (n=5, p=0.567) | 16.0 / 5.9 (n=10, p=0.504) |
| t5-3b | 12.8 / 9.7 | 1.1 / 1.1 (n=2, p=0.015*) | 11.2 / 13.9 (n=3, p=0.578) | 17.5 / 12.0 (n=5, p=0.842) | 12.4 / 11.5 (n=10, p=0.439) |
| flan-t5-xxl | 7.7 / 6.1 | 8.5 / 8.5 (n=2, p=0.743) | 3.1 / 2.5 (n=3, p=0.0792) | 2.4 / 2.1 (n=5, p=0.0121*) | 3.8 / 2.8 (n=10, p=0.0191*) |

### 2.2 |%diff| in L2, raw — similar pairs vs 5000 random pairs (mean / median; p = Mann–Whitney, similar < random)

| model | random | us_uk | plural | verb | all pooled |
| :--- | :--- | :--- | :--- | :--- | :--- |
| opt-1.3b | 16.5 / 13.8 | 1.8 / 1.8 (n=2, p=0.0187*) | 16.5 / 16.8 (n=3, p=0.615) | 3.9 / 4.0 (n=5, p=0.00435**) | 7.3 / 4.6 (n=10, p=0.00434**) |
| opt-13b | 30.5 / 24.3 | 36.5 / 36.5 (n=2, p=0.588) | 17.9 / 22.1 (n=3, p=0.231) | 5.6 / 4.9 (n=5, p=0.00409**) | 15.5 / 5.0 (n=10, p=0.0149*) |
| t5-large | 14.5 / 9.9 | 24.9 / 24.9 (n=2, p=0.718) | 18.0 / 5.6 (n=3, p=0.486) | 18.0 / 8.9 (n=5, p=0.707) | 19.4 / 8.1 (n=10, p=0.734) |
| t5-3b | 12.8 / 9.7 | 1.7 / 1.7 (n=2, p=0.0246*) | 10.9 / 13.8 (n=3, p=0.544) | 17.3 / 12.5 (n=5, p=0.842) | 12.3 / 12.1 (n=10, p=0.457) |
| flan-t5-xxl | 6.4 / 5.1 | 6.0 / 6.0 (n=2, p=0.624) | 3.0 / 2.7 (n=3, p=0.113) | 2.3 / 1.6 (n=5, p=0.0195*) | 3.2 / 2.6 (n=10, p=0.0239*) |

### 2.2 |%diff| in L1, center — similar pairs vs 5000 random pairs (mean / median; p = Mann–Whitney, similar < random)

| model | random | us_uk | plural | verb | all pooled |
| :--- | :--- | :--- | :--- | :--- | :--- |
| opt-1.3b | 18.9 / 16.0 | 9.6 / 9.6 (n=2, p=0.172) | 11.7 / 8.1 (n=3, p=0.199) | 3.2 / 2.9 (n=5, p=0.00136**) | 7.1 / 4.0 (n=10, p=0.00134**) |
| opt-13b | 15.3 / 12.9 | 10.1 / 10.1 (n=2, p=0.306) | 16.6 / 18.4 (n=3, p=0.725) | 17.0 / 15.5 (n=5, p=0.737) | 15.5 / 13.8 (n=10, p=0.708) |
| t5-large | 12.0 / 10.0 | 13.4 / 13.4 (n=2, p=0.556) | 2.9 / 3.3 (n=3, p=0.0188*) | 7.2 / 8.4 (n=5, p=0.147) | 7.1 / 5.1 (n=10, p=0.0347*) |
| t5-3b | 13.8 / 11.3 | 13.9 / 13.9 (n=2, p=0.538) | 9.0 / 9.7 (n=3, p=0.29) | 12.4 / 14.9 (n=5, p=0.48) | 11.7 / 10.0 (n=10, p=0.383) |
| flan-t5-xxl | 14.9 / 12.5 | 14.3 / 14.3 (n=2, p=0.512) | 2.0 / 2.1 (n=3, p=0.00657**) | 3.9 / 3.2 (n=5, p=0.00508**) | 5.4 / 3.0 (n=10, p=0.000787***) |

### 2.2 |%diff| in L2, center — similar pairs vs 5000 random pairs (mean / median; p = Mann–Whitney, similar < random)

| model | random | us_uk | plural | verb | all pooled |
| :--- | :--- | :--- | :--- | :--- | :--- |
| opt-1.3b | 19.7 / 16.7 | 9.9 / 9.9 (n=2, p=0.171) | 15.8 / 10.7 (n=3, p=0.382) | 5.2 / 3.8 (n=5, p=0.00558**) | 9.3 / 7.0 (n=10, p=0.00862**) |
| opt-13b | 23.5 / 20.5 | 27.1 / 27.1 (n=2, p=0.639) | 27.8 / 33.1 (n=3, p=0.716) | 7.4 / 7.2 (n=5, p=0.00831**) | 17.4 / 10.1 (n=10, p=0.111) |
| t5-large | 12.1 / 10.0 | 9.5 / 9.5 (n=2, p=0.346) | 3.1 / 3.0 (n=3, p=0.0203*) | 5.8 / 5.3 (n=5, p=0.0568) | 5.7 / 4.4 (n=10, p=0.00785**) |
| t5-3b | 13.4 / 10.9 | 13.5 / 13.5 (n=2, p=0.537) | 8.9 / 8.4 (n=3, p=0.298) | 12.7 / 13.6 (n=5, p=0.535) | 11.7 / 10.0 (n=10, p=0.426) |
| flan-t5-xxl | 14.3 / 11.9 | 14.4 / 14.4 (n=2, p=0.584) | 2.0 / 2.0 (n=3, p=0.00771**) | 3.5 / 3.9 (n=5, p=0.0037**) | 5.2 / 3.5 (n=10, p=0.00089***) |

**Reading (L2, raw, pooled).** opt-1.3b: similar pairs closer (median 4.6 vs 13.8, p=0.0043); opt-13b: similar pairs closer (median 5.0 vs 24.3, p=0.015); t5-large: no significant difference (median 8.1 vs 9.9); t5-3b: no significant difference (median 12.1 vs 9.7); flan-t5-xxl: similar pairs closer (median 2.6 vs 5.1, p=0.024). 
The paper's sentence "similar pairs were not closer in magnitude than random pairs" is a claim about the pooled comparison; where p < 0.05 in the "closer" direction it is contradicted on the authors' own pairs. Where results differ across models, the defensible statement is "no consistent effect across models". None of this tests a complexity–magnitude *correlation*: the design only checks that magnitude is invariant under near-synonymy, a necessary condition.

### 2.2b The notebooks' own printed values for the 74 similar pairs — record only, NOT a valid test

Parsed from the saved cell outputs of magnitudes*.ipynb: signed L1 percent difference |n1/n2 − 1|·100 per similar pair, computed by the notebooks from vectors they embedded on the fly (only 10 of the 74 pairs are in the cache). Typo pairs excluded.

**Invalid as a test, and the reviewer's letter must not cite it.** No random arm exists in the notebook outputs (they print only a signed 100-word mean per word, which cancels). Any random baseline for these values has to come from the cached matrices, so a test would compare on-the-fly vectors against cached vectors. `opt/1_3B.txt.orig` shows the cache was re-extracted at least once, so the two sets are not guaranteed to be the same extraction, and a p-value from mixed arms is not evidence. The same mixing was inside the notebooks' own `Rand100` column. The only clean 74-pair comparison is the GPU re-extraction in Phase 3, which embeds the similar pairs *and* the random pairs with one recipe in one run and reports the cosine between re-extracted and cached vectors for the words present in both.

| model | pairs parsed | us_uk mean / median abs %diff | plural mean / median abs %diff | verb mean / median abs %diff | all | largest abs diff |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| opt-1.3b | 71/74 | 5.3 / 1.8 (n=11) | 9.3 / 7.2 (n=27) | 12.2 / 7.1 (n=33) | 10.0 / 6.1 (n=71) | driving/drove +30%, drive/drives +29%, persons/people -29% |
| t5-large | 71/74 | 28.1 / 23.6 (n=11) | 20.1 / 11.0 (n=27) | 16.9 / 13.5 (n=33) | 19.8 / 13.6 (n=71) | index/indices +101%, apologize/apologise +94%, radius/radii +62% |
| opt-13b | 70/74 | 6.7 / 5.6 (n=11) | 9.8 / 9.3 (n=26) | 11.0 / 8.3 (n=33) | 9.9 / 8.0 (n=70) | thinks/thought +33%, think/thinks -29%, talk/talked -27% |
| t5-3b | 70/74 | 12.1 / 4.8 (n=11) | 19.1 / 14.5 (n=26) | 9.8 / 7.3 (n=33) | 13.6 / 8.2 (n=70) | persons/people +84%, appendix/appendixes +48%, apologize/apologise +46% |
| flan-t5-xxl | 70/74 | 4.8 / 1.9 (n=11) | 6.0 / 4.2 (n=26) | 4.6 / 3.1 (n=33) | 5.2 / 3.2 (n=70) | datum/data +22%, appendix/appendices +21%, chew/chewing -20% |

### 2.3 Tokens per word (tokenizer of each model family, word alone, no special tokens)

| model | words with >1 token (count) |
| :--- | :--- |
| opt-1.3b | alumnae=4, alumna=3, alumnus=3, appendixes=3, bacterium=3, maneuver=3, manoeuvre=3, nuclei=3, nucleus=3, paediatric=3, phenomena=3, phenomenon=3, travelled=3, alumni=2 … |
| opt-13b | alumnae=4, alumna=3, alumnus=3, appendixes=3, bacterium=3, maneuver=3, manoeuvre=3, nuclei=3, nucleus=3, paediatric=3, phenomena=3, phenomenon=3, travelled=3, alumni=2 … |
| t5-large | alumnae=6, apologise=6, alumna=5, alumnus=5, appendixes=5, appendices=4, analogue=3, appendix=3, bacterium=3, chewed=3, cries=3, indices=3, nuclei=3, nucleus=3 … |
| t5-3b | alumnae=6, apologise=6, alumna=5, alumnus=5, appendixes=5, appendices=4, analogue=3, appendix=3, bacterium=3, chewed=3, cries=3, indices=3, nuclei=3, nucleus=3 … |
| flan-t5-xxl | alumnae=6, apologise=6, alumna=5, alumnus=5, appendixes=5, appendices=4, analogue=3, appendix=3, bacterium=3, chewed=3, cries=3, indices=3, nuclei=3, nucleus=3 … |

Because the cached OPT/T5 vectors are means over subword tokens (T5 also averages in the EOS token), a pair whose members differ in token count is not a clean magnitude comparison. Outliers in the pair tables above line up with these words.

### 2.4 L1 vs L2 and rogue dimensions

| model | Pearson(L1, L2) | Spearman | top-3 variance dims: share of a word's squared L2 norm (mean, median over words) |
| :--- | :--- | :--- | :--- |
| opt-1.3b | 0.957 | 0.973 | d1346: mean 5.1%, median 3.6%; d1359: mean 2.1%, median 1.1%; d1279: mean 1.2%, median 0.6% |
| opt-13b | -0.185 | -0.007 | d902: mean 28.8%, median 11.1%; d4960: mean 1.9%, median 1.0%; d3964: mean 1.9%, median 1.2% |
| t5-large | 0.971 | 0.937 | d79: mean 2.5%, median 2.2%; d58: mean 3.1%, median 3.0%; d300: mean 1.5%, median 1.2% |
| t5-3b | 0.995 | 0.993 | d792: mean 0.2%, median 0.1%; d71: mean 0.2%, median 0.1%; d17: mean 0.2%, median 0.1% |
| flan-t5-xxl | 0.973 | 0.965 | d1478: mean 11.5%, median 12.3%; d1814: mean 0.1%, median 0.1%; d2152: mean 0.1%, median 0.0% |

The notebooks' "magnitude" was L1 (sum of absolute components). Where L1 and L2 are weakly correlated, the choice of norm changes the result; a dimension holding a large share of squared norm on its own is a rogue dimension (Timkey & van Schijndel 2021) and dominates L2 but not L1.

_Phase 2 ran in 7 s._

## Phase 3 — 2026 open model with explicit extraction choices

> Qwen/Qwen3-14B: 20 configurations reused from the previous run's phase3.json.
> meta-llama/Llama-3.1-8B-Instruct: 20 configurations reused from the previous run's phase3.json.
> Qwen/Qwen3-1.7B: 20 configurations reused from the previous run's phase3.json.
> flan-t5-xxl: skipped, download ≈45 GB exceeds the 40 GB limit set in the brief.

### 3.1 Qwen/Qwen3-14B

Layer indices used: {'emb': 0, 'p25': 10, 'p50': 20, 'p75': 30, 'final': 40}. Extraction 61 s (0 = reused from cache). alone: 2835/5178 words need >1 token, worst bureaucracy=5, bad-tempered=4, congratulate=4, congratulation=4, counsellor=4; prompt: 175/5178 words need >1 token, worst bad-tempered=4, fiftieth=4, thirtieth=4, thought-provoking=4, a.m.=3

| layer | pool | context | p16 targets at rank 1 (of 7), raw / centered | mean rank, raw / centered | sum beats v(b) alone (of 7) | bachelor rank | define2 = paper pair (of 10), raw / centered | magnitude p (similar < random, L2 raw) | median |%diff| ratio similar/random | random-sum baseline (raw) |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| final | mean | prompt | 2 / 1 | 146.9 / 395.4 | 5 | 841 | 0 / 0 | 0.81 | 1.05 | 0.941 |
| p75 | mean | prompt | 1 / 1 | 3.4 / 2.6 | 0 | 1801 | 0 / 0 | 0.21 | 0.75 | 0.823 |
| p25 | mean | prompt | 1 / 1 | 64.9 / 5.6 | 0 | 2237 | 2 / 2 | 7.4e-07 | 0.45 | 0.778 |
| emb | mean | prompt | 1 / 1 | 90.4 / 23.0 | 1 | 1364 | 1 / 0 | 0.071 | 0.88 | 0.198 |
| p50 | mean | prompt | 1 / 1 | 161.4 / 44.1 | 0 | 720 | 1 / 1 | 0.081 | 0.83 | 0.840 |
| final | last | prompt | 1 / 0 | 560.9 / 863.1 | 5 | 818 | 0 / 0 | 0.0051 | 0.81 | 0.941 |
| emb | mean | alone | 1 / 1 | 1069.9 / 945.1 | 5 | 1026 | 0 / 0 | 2.8e-07 | 0.52 | 0.466 |
| p25 | last | prompt | 0 / 1 | 385.1 / 5.4 | 0 | 2132 | 2 / 2 | 1.4e-13 | 0.38 | 0.774 |
| p75 | last | prompt | 0 / 0 | 6.4 / 3.6 | 0 | 1721 | 0 / 0 | 0.009 | 0.71 | 0.821 |
| emb | last | prompt | 0 / 0 | 112.6 / 30.9 | 2 | 1277 | 1 / 0 | 0.051 | 0.85 | 0.192 |
| p50 | last | prompt | 0 / 0 | 521.9 / 56.6 | 0 | 649 | 1 / 1 | 7e-07 | 0.59 | 0.838 |
| emb | last | alone | 0 / 0 | 1292.1 / 1400.4 | 4 | 3868 | 0 / 0 | 0.78 | 1.03 | 0.549 |
| final | mean | alone | 0 / 0 | 1421.9 / 2688.1 | 1 | 197 | 0 / 0 | 1.5e-06 | 0.23 | 0.925 |
| final | last | alone | 0 / 0 | 1748.3 / 2750.9 | 3 | 430 | 0 / 0 | 1.7e-06 | 0.24 | 0.929 |
| p25 | mean | alone | 0 / 0 | 1762.7 / 2289.9 | 3 | 3923 | 0 / 0 | 4.1e-12 | 0.05 | 1.000 |
| p50 | mean | alone | 0 / 0 | 2280.9 / 2588.1 | 2 | 4300 | 0 / 0 | 5.4e-12 | 0.05 | 1.000 |
| p75 | mean | alone | 0 / 0 | 2587.7 / 2265.3 | 1 | 4112 | 0 / 0 | 6.9e-12 | 0.05 | 1.000 |
| p75 | last | alone | 0 / 0 | 3163.6 / 2474.9 | 1 | 4026 | 0 / 0 | 4.1e-11 | 0.18 | 0.940 |
| p50 | last | alone | 0 / 0 | 3273.6 / 2179.4 | 1 | 3178 | 0 / 0 | 2.9e-08 | 0.17 | 0.939 |
| p25 | last | alone | 0 / 0 | 3397.4 / 2212.9 | 1 | 3364 | 1 / 1 | 6e-07 | 0.27 | 0.902 |

**Best configuration by the p. 16 criterion: layer final, mean pooling, context prompt.** Its p. 16 table (raw, a and b excluded):

| a + b → target | rank | cos | top-3 |
| :--- | ---: | ---: | ---: |
| young + dog → puppy | 831 | 0.8763 | old-fashioned 0.921, soldier 0.920, educated 0.918 |
| young + cat → kitten | 1 | 0.9186 | kitten 0.919, old-fashioned 0.912, only 0.911 |
| young + duck → duckling | 1 | 0.9323 | duckling 0.932, old-fashioned 0.927, cheaply 0.917 |
| female + spouse → wife | 148 | 0.9246 | spouses 0.961, inmate 0.949, unemployed 0.944 |
| male + spouse → husband | 41 | 0.9114 | spouses 0.937, wives 0.926, inmate 0.925 |
| male + sibling → brother | 3 | 0.9236 | sister 0.927, spouse 0.924, brother 0.924 |
| female + sibling → sister | 3 | 0.9407 | girlfriend 0.943, spouses 0.941, sister 0.941 |
| man + unmarried → bachelor | 841 | 0.8607 | woman 0.930, part-time 0.917, evil 0.913 |

define2 (raw) on the same configuration:

| target | paper (Curie) | this model top-3 |
| :--- | :--- | :--- |
| wife | spouse + woman | blade + husband 0.947<br>husband + woman 0.946<br>husband + shade 0.945 |
| duckling | duck + youngster | awkwardly + bird 0.933<br>bird + youngster 0.933<br>bird + scissors 0.933 |
| puppy | dog + kitten | baby + kitten 0.914<br>crazy + kitten 0.904<br>dog + kitten 0.900 |
| foal | horse + puppy | ago + tiring 0.894<br>ago + prey 0.893<br>ago + calm 0.893 |
| freedom | independence + liberty | liberty + wisdom 0.970<br>awareness + liberty 0.970<br>dignity + liberty 0.970 |
| autonomy | control + independence | independence + sovereignty 0.973<br>accuracy + sovereignty 0.973<br>creativity + sovereignty 0.973 |
| justice | fairness + judicial | equality + judge 0.973<br>fairness + judge 0.971<br>equality + magistrate 0.971 |
| knowledge | information + wisdom | language + wisdom 0.967<br>expertise + language 0.966<br>expertise + wisdom 0.965 |
| rationality | logical + reasoning | implication + reasonable 0.970<br>immoral + reasoning 0.968<br>justification + reasonable 0.967 |
| causation | consequence + correlation | irrationals + precedent 0.900<br>intention + irrationals 0.900<br>conception + irrationals 0.899 |

Magnitude test on every configuration (L2, median |%diff|; p = similar < random):

| layer | pool | ctx | mode | random | us_uk | plural | verb | all |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| emb | mean | alone | raw | 14.5 | 6.8 (p=0.012) | 7.8 (p=0.00069) | 7.5 (p=0.00071) | 7.6 (p=2.8e-07) |
| emb | mean | alone | center | 16.2 | 9.5 (p=0.0089) | 8.0 (p=0.00048) | 7.9 (p=0.00051) | 8.2 (p=1.1e-07) |
| emb | mean | prompt | raw | 9.6 | 13.3 (p=0.95) | 9.0 (p=0.46) | 6.9 (p=0.0015) | 8.5 (p=0.071) |
| emb | mean | prompt | center | 9.0 | 13.0 (p=0.95) | 7.7 (p=0.53) | 6.2 (p=0.002) | 7.6 (p=0.092) |
| emb | last | alone | raw | 11.8 | 9.2 (p=0.47) | 8.8 (p=0.55) | 12.8 (p=0.85) | 12.2 (p=0.78) |
| emb | last | alone | center | 10.6 | 9.5 (p=0.52) | 8.3 (p=0.54) | 12.2 (p=0.92) | 11.4 (p=0.86) |
| emb | last | prompt | raw | 9.2 | 10.2 (p=0.72) | 6.4 (p=0.18) | 7.5 (p=0.029) | 7.9 (p=0.051) |
| emb | last | prompt | center | 8.6 | 9.4 (p=0.67) | 5.6 (p=0.13) | 6.3 (p=0.02) | 7.2 (p=0.027) |
| p25 | mean | alone | raw | 49.7 | 1.3 (p=0.00024) | 3.4 (p=3.1e-06) | 2.4 (p=3.2e-05) | 2.6 (p=4.1e-12) |
| p25 | mean | alone | center | 23.3 | 4.5 (p=0.02) | 9.5 (p=0.0015) | 6.1 (p=0.00034) | 6.7 (p=4.2e-07) |
| p25 | mean | prompt | raw | 6.3 | 3.9 (p=0.0037) | 5.0 (p=0.19) | 2.1 (p=1.4e-06) | 2.8 (p=7.4e-07) |
| p25 | mean | prompt | center | 10.2 | 6.3 (p=0.0059) | 8.5 (p=0.48) | 7.1 (p=0.0089) | 7.2 (p=0.0042) |
| p25 | last | alone | raw | 18.0 | 3.7 (p=0.0018) | 4.3 (p=5.3e-05) | 8.5 (p=0.021) | 4.8 (p=6e-07) |
| p25 | last | alone | center | 9.5 | 0.1 (p=0.0018) | 0.3 (p=7.9e-05) | 1.6 (p=0.0075) | 0.3 (p=1.9e-07) |
| p25 | last | prompt | raw | 6.0 | 3.9 (p=0.0036) | 2.4 (p=6e-05) | 2.1 (p=5.8e-09) | 2.3 (p=1.4e-13) |
| p25 | last | prompt | center | 10.1 | 6.2 (p=0.011) | 6.9 (p=0.013) | 5.7 (p=3.5e-05) | 6.3 (p=3.3e-07) |
| p50 | mean | alone | raw | 49.0 | 1.2 (p=0.00029) | 2.7 (p=3.5e-06) | 2.2 (p=3.3e-05) | 2.4 (p=5.4e-12) |
| p50 | mean | alone | center | 22.5 | 4.1 (p=0.019) | 8.5 (p=0.0015) | 5.6 (p=0.00028) | 6.1 (p=3.3e-07) |
| p50 | mean | prompt | raw | 5.3 | 5.7 (p=0.57) | 6.6 (p=0.86) | 3.2 (p=0.0012) | 4.4 (p=0.081) |
| p50 | mean | prompt | center | 8.2 | 13.4 (p=0.95) | 12.5 (p=0.99) | 6.3 (p=0.11) | 8.4 (p=0.87) |
| p50 | last | alone | raw | 13.8 | 1.9 (p=0.0015) | 2.1 (p=6.4e-08) | 5.5 (p=0.056) | 2.4 (p=2.9e-08) |
| p50 | last | alone | center | 9.1 | 0.1 (p=0.0016) | 0.1 (p=2.4e-06) | 1.5 (p=0.003) | 0.2 (p=3.5e-09) |
| p50 | last | prompt | raw | 5.1 | 5.9 (p=0.72) | 3.1 (p=0.0034) | 2.4 (p=4.3e-07) | 3.1 (p=7e-07) |
| p50 | last | prompt | center | 8.3 | 17.3 (p=0.99) | 12.2 (p=0.98) | 6.5 (p=0.088) | 9.1 (p=0.89) |
| p75 | mean | alone | raw | 47.6 | 1.2 (p=0.00028) | 3.1 (p=3.8e-06) | 2.5 (p=3.8e-05) | 2.5 (p=6.9e-12) |
| p75 | mean | alone | center | 22.9 | 4.2 (p=0.02) | 8.6 (p=0.002) | 6.5 (p=0.0003) | 6.9 (p=4.8e-07) |
| p75 | mean | prompt | raw | 3.7 | 2.2 (p=0.064) | 5.0 (p=0.87) | 2.7 (p=0.096) | 2.7 (p=0.21) |
| p75 | mean | prompt | center | 9.6 | 4.2 (p=0.007) | 6.7 (p=0.2) | 10.6 (p=0.77) | 8.4 (p=0.18) |
| p75 | last | alone | raw | 12.9 | 1.9 (p=0.00079) | 2.3 (p=1.2e-07) | 2.7 (p=0.00085) | 2.3 (p=4.1e-11) |
| p75 | last | alone | center | 9.4 | 0.2 (p=0.00061) | 0.6 (p=1.1e-05) | 1.5 (p=0.015) | 0.8 (p=6e-08) |
| p75 | last | prompt | raw | 3.4 | 1.4 (p=0.0021) | 4.2 (p=0.53) | 2.5 (p=0.029) | 2.5 (p=0.009) |
| p75 | last | prompt | center | 9.4 | 4.2 (p=0.0088) | 4.0 (p=0.00052) | 10.2 (p=0.57) | 6.4 (p=0.003) |
| final | mean | alone | raw | 29.6 | 4.1 (p=0.0025) | 6.7 (p=0.00015) | 7.2 (p=0.02) | 6.9 (p=1.5e-06) |
| final | mean | alone | center | 12.0 | 7.3 (p=0.1) | 6.5 (p=0.056) | 9.8 (p=0.13) | 7.5 (p=0.013) |
| final | mean | prompt | raw | 10.2 | 7.0 (p=0.32) | 11.5 (p=0.94) | 10.8 (p=0.56) | 10.8 (p=0.81) |
| final | mean | prompt | center | 20.9 | 6.6 (p=0.0016) | 14.0 (p=0.014) | 23.3 (p=0.84) | 17.6 (p=0.039) |
| final | last | alone | raw | 27.7 | 3.3 (p=0.00044) | 8.1 (p=0.0002) | 7.2 (p=0.037) | 6.6 (p=1.7e-06) |
| final | last | alone | center | 13.6 | 4.0 (p=0.044) | 6.9 (p=0.013) | 9.1 (p=0.17) | 6.6 (p=0.004) |
| final | last | prompt | raw | 9.6 | 5.7 (p=0.011) | 4.6 (p=0.0018) | 10.7 (p=0.54) | 7.8 (p=0.0051) |
| final | last | prompt | center | 21.5 | 10.7 (p=0.0049) | 14.1 (p=0.025) | 24.3 (p=0.71) | 16.5 (p=0.037) |

### 3.2 meta-llama/Llama-3.1-8B-Instruct

Layer indices used: {'emb': 0, 'p25': 8, 'p50': 16, 'p75': 24, 'final': 32}. Extraction 41 s (0 = reused from cache). alone: 2830/5178 words need >1 token, worst bureaucracy=5, bad-tempered=4, congratulate=4, congratulation=4, counsellor=4; prompt: 175/5178 words need >1 token, worst bad-tempered=4, fiftieth=4, thirtieth=4, thought-provoking=4, a.m.=3

| layer | pool | context | p16 targets at rank 1 (of 7), raw / centered | mean rank, raw / centered | sum beats v(b) alone (of 7) | bachelor rank | define2 = paper pair (of 10), raw / centered | magnitude p (similar < random, L2 raw) | median |%diff| ratio similar/random | random-sum baseline (raw) |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p25 | mean | prompt | 2 / 2 | 2.4 / 2.3 | 3 | 12 | 3 / 3 | 0.029 | 0.79 | 0.797 |
| p75 | last | alone | 2 / 2 | 38.1 / 4.6 | 0 | 5008 | 2 / 2 | 2.9e-05 | 0.59 | 0.747 |
| final | mean | prompt | 2 / 1 | 41.0 / 186.0 | 0 | 1213 | 1 / 0 | 0.98 | 1.21 | 0.839 |
| p75 | mean | prompt | 1 / 2 | 2.7 / 2.4 | 1 | 1394 | 3 / 3 | 0.19 | 0.75 | 0.823 |
| p75 | last | prompt | 0 / 2 | 2.9 / 2.4 | 1 | 1349 | 3 / 3 | 0.015 | 0.67 | 0.821 |
| p50 | mean | prompt | 1 / 1 | 5.0 / 4.1 | 1 | 373 | 4 / 4 | 0.0013 | 0.67 | 0.831 |
| emb | mean | prompt | 1 / 1 | 8.4 / 3.6 | 0 | 354 | 2 / 1 | 0.12 | 0.72 | 0.271 |
| p25 | mean | alone | 1 / 1 | 40.6 / 7.7 | 4 | 775 | 2 / 1 | 2.8e-08 | 0.36 | 0.832 |
| p50 | last | alone | 1 / 0 | 176.9 / 15.1 | 1 | 4651 | 1 / 1 | 0.00061 | 0.68 | 0.763 |
| final | last | alone | 1 / 1 | 272.9 / 116.1 | 0 | 4962 | 1 / 1 | 0.19 | 0.82 | 0.800 |
| final | mean | alone | 1 / 1 | 278.0 / 198.3 | 2 | 4400 | 0 / 0 | 1e-08 | 0.24 | 0.826 |
| p25 | last | prompt | 1 / 1 | 286.0 / 5.7 | 3 | 11 | 3 / 3 | 4.2e-08 | 0.55 | 0.793 |
| final | last | prompt | 1 / 0 | 345.3 / 517.6 | 1 | 1129 | 1 / 0 | 0.13 | 0.87 | 0.837 |
| emb | mean | alone | 1 / 1 | 805.3 / 743.1 | 3 | 1801 | 0 / 0 | 2.6e-05 | 0.37 | 0.450 |
| p50 | mean | alone | 0 / 1 | 67.1 / 19.9 | 4 | 888 | 0 / 0 | 3.7e-08 | 0.47 | 0.828 |
| p75 | mean | alone | 0 / 1 | 214.4 / 21.4 | 1 | 1700 | 0 / 0 | 4.6e-08 | 0.41 | 0.802 |
| p50 | last | prompt | 0 / 0 | 94.6 / 7.4 | 1 | 309 | 4 / 4 | 1.4e-11 | 0.50 | 0.829 |
| emb | last | prompt | 0 / 0 | 444.7 / 171.3 | 0 | 326 | 2 / 1 | 0.12 | 0.78 | 0.265 |
| p25 | last | alone | 0 / 0 | 817.9 / 142.6 | 1 | 4630 | 2 / 1 | 0.094 | 0.87 | 0.731 |
| emb | last | alone | 0 / 0 | 1029.1 / 1161.1 | 4 | 2454 | 0 / 0 | 0.73 | 1.08 | 0.507 |

**Best configuration by the p. 16 criterion: layer p25, mean pooling, context prompt.** Its p. 16 table (raw, a and b excluded):

| a + b → target | rank | cos | top-3 |
| :--- | ---: | ---: | ---: |
| young + dog → puppy | 4 | 0.8103 | child 0.836, cat 0.817, kid 0.812 |
| young + cat → kitten | 5 | 0.7958 | dog 0.847, child 0.829, kid 0.818 |
| young + duck → duckling | 1 | 0.8465 | duckling 0.846, bird 0.821, chicken 0.801 |
| female + spouse → wife | 1 | 0.8319 | wife 0.832, spouses 0.830, woman 0.818 |
| male + spouse → husband | 2 | 0.8204 | spouses 0.829, husband 0.820, wife 0.820 |
| male + sibling → brother | 2 | 0.8045 | siblings 0.863, brother 0.804, sister 0.803 |
| female + sibling → sister | 2 | 0.8189 | siblings 0.864, sister 0.819, male 0.810 |
| man + unmarried → bachelor | 12 | 0.7309 | woman 0.797, male 0.774, married 0.770 |

define2 (raw) on the same configuration:

| target | paper (Curie) | this model top-3 |
| :--- | :--- | :--- |
| wife | spouse + woman | husband + spouse 0.893<br>girlfriend + husband 0.891<br>girlfriend + spouse 0.885 |
| duckling | duck + youngster | kitten + pig 0.822<br>chicken + kitten 0.817<br>bird + kitten 0.816 |
| puppy | dog + kitten | dog + kitten 0.914<br>baby + kitten 0.876<br>horse + kitten 0.865 |
| foal | horse + puppy | colt + loom 0.874<br>colt + horse 0.868<br>beak + colt 0.861 |
| freedom | independence + liberty | liberation + liberty 0.902<br>independence + liberty 0.890<br>liberty + peace 0.888 |
| autonomy | control + independence | independence + sovereignty 0.848<br>independent + sovereignty 0.829<br>competence + sovereignty 0.829 |
| justice | fairness + judicial | fairness + judicial 0.844<br>fairness + judge 0.844<br>fairness + law 0.836 |
| knowledge | information + wisdom | information + wisdom 0.878<br>awareness + information 0.858<br>ignorance + information 0.857 |
| rationality | logical + reasoning | logic + reasonable 0.852<br>logical + reason 0.847<br>irrationals + reason 0.846 |
| causation | consequence + correlation | correlation + motive 0.768<br>correlation + decision-making 0.767<br>correlation + existence 0.766 |

Magnitude test on every configuration (L2, median |%diff|; p = similar < random):

| layer | pool | ctx | mode | random | us_uk | plural | verb | all |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| emb | mean | alone | raw | 26.4 | 5.2 (p=0.0038) | 12.5 (p=0.0063) | 7.8 (p=0.014) | 9.8 (p=2.6e-05) |
| emb | mean | alone | center | 28.2 | 7.0 (p=0.0048) | 13.7 (p=0.0093) | 8.2 (p=0.013) | 10.4 (p=3.8e-05) |
| emb | mean | prompt | raw | 11.1 | 5.1 (p=0.04) | 16.5 (p=0.95) | 7.1 (p=0.015) | 8.0 (p=0.12) |
| emb | mean | prompt | center | 12.2 | 6.2 (p=0.043) | 18.6 (p=0.95) | 7.7 (p=0.017) | 8.4 (p=0.12) |
| emb | last | alone | raw | 10.8 | 3.5 (p=0.047) | 14.8 (p=0.95) | 10.8 (p=0.66) | 11.6 (p=0.73) |
| emb | last | alone | center | 11.6 | 3.6 (p=0.049) | 14.8 (p=0.94) | 11.0 (p=0.58) | 11.6 (p=0.67) |
| emb | last | prompt | raw | 10.7 | 4.9 (p=0.00077) | 16.5 (p=0.98) | 7.7 (p=0.038) | 8.4 (p=0.12) |
| emb | last | prompt | center | 11.8 | 5.6 (p=0.00084) | 17.3 (p=0.97) | 8.1 (p=0.037) | 8.4 (p=0.089) |
| p25 | mean | alone | raw | 9.7 | 1.4 (p=9.1e-05) | 5.0 (p=9.8e-05) | 3.7 (p=0.0057) | 3.5 (p=2.8e-08) |
| p25 | mean | alone | center | 18.5 | 10.9 (p=0.06) | 5.5 (p=1.1e-06) | 6.5 (p=0.00031) | 6.3 (p=2.9e-09) |
| p25 | mean | prompt | raw | 5.5 | 8.4 (p=0.89) | 5.2 (p=0.39) | 3.3 (p=0.00078) | 4.3 (p=0.029) |
| p25 | mean | prompt | center | 7.8 | 16.9 (p=1) | 11.6 (p=0.98) | 6.4 (p=0.1) | 8.4 (p=0.93) |
| p25 | last | alone | raw | 5.5 | 4.1 (p=0.066) | 5.0 (p=0.071) | 5.5 (p=0.58) | 4.8 (p=0.094) |
| p25 | last | alone | center | 7.3 | 14.0 (p=0.92) | 4.2 (p=0.0073) | 2.4 (p=3e-06) | 4.0 (p=2.3e-05) |
| p25 | last | prompt | raw | 5.3 | 7.3 (p=0.7) | 2.9 (p=0.00028) | 2.4 (p=2.6e-07) | 2.9 (p=4.2e-08) |
| p25 | last | prompt | center | 7.7 | 16.9 (p=1) | 5.9 (p=0.17) | 5.6 (p=0.012) | 6.8 (p=0.18) |
| p50 | mean | alone | raw | 7.9 | 4.4 (p=0.0024) | 3.9 (p=5.9e-05) | 3.5 (p=0.0021) | 3.7 (p=3.7e-08) |
| p50 | mean | alone | center | 16.1 | 5.5 (p=0.0046) | 4.5 (p=1.9e-07) | 5.9 (p=0.00011) | 5.4 (p=1.8e-11) |
| p50 | mean | prompt | raw | 5.5 | 1.4 (p=0.0065) | 3.8 (p=0.14) | 4.0 (p=0.021) | 3.7 (p=0.0013) |
| p50 | mean | prompt | center | 8.2 | 3.6 (p=0.0059) | 7.8 (p=0.65) | 5.8 (p=0.027) | 5.8 (p=0.019) |
| p50 | last | alone | raw | 5.5 | 1.7 (p=0.0016) | 3.7 (p=0.016) | 4.5 (p=0.12) | 3.7 (p=0.00061) |
| p50 | last | alone | center | 8.4 | 3.0 (p=0.0073) | 6.1 (p=0.028) | 5.7 (p=0.029) | 5.7 (p=0.00034) |
| p50 | last | prompt | raw | 5.2 | 1.2 (p=6.2e-05) | 2.0 (p=1.6e-06) | 3.1 (p=0.00029) | 2.6 (p=1.4e-11) |
| p50 | last | prompt | center | 8.0 | 3.5 (p=0.0034) | 7.7 (p=0.12) | 5.5 (p=0.0012) | 5.7 (p=6.1e-05) |
| p75 | mean | alone | raw | 9.3 | 2.5 (p=0.0015) | 3.4 (p=0.00016) | 4.3 (p=0.0017) | 3.8 (p=4.6e-08) |
| p75 | mean | alone | center | 14.8 | 6.4 (p=0.0052) | 4.5 (p=7.7e-06) | 10.8 (p=0.18) | 6.3 (p=1.2e-05) |
| p75 | mean | prompt | raw | 3.0 | 0.6 (p=0.0041) | 2.9 (p=0.5) | 3.2 (p=0.58) | 2.2 (p=0.19) |
| p75 | mean | prompt | center | 9.7 | 2.6 (p=0.0018) | 8.7 (p=0.16) | 12.5 (p=0.6) | 7.6 (p=0.06) |
| p75 | last | alone | raw | 5.5 | 1.5 (p=0.0024) | 3.0 (p=0.0055) | 4.2 (p=0.021) | 3.3 (p=2.9e-05) |
| p75 | last | alone | center | 8.5 | 5.2 (p=0.07) | 7.3 (p=0.26) | 8.8 (p=0.34) | 7.7 (p=0.11) |
| p75 | last | prompt | raw | 2.9 | 0.6 (p=0.0008) | 2.3 (p=0.082) | 2.7 (p=0.43) | 1.9 (p=0.015) |
| p75 | last | prompt | center | 9.5 | 2.6 (p=0.00032) | 5.2 (p=0.00048) | 9.6 (p=0.37) | 6.4 (p=0.00021) |
| final | mean | alone | raw | 10.0 | 3.0 (p=0.0059) | 1.5 (p=3.2e-08) | 3.6 (p=0.022) | 2.4 (p=1e-08) |
| final | mean | alone | center | 18.0 | 4.0 (p=0.0024) | 6.7 (p=2.2e-05) | 10.8 (p=0.044) | 8.3 (p=1.2e-06) |
| final | mean | prompt | raw | 1.4 | 1.0 (p=0.14) | 1.9 (p=0.99) | 2.0 (p=0.94) | 1.7 (p=0.98) |
| final | mean | prompt | center | 11.6 | 4.5 (p=0.0012) | 9.0 (p=0.022) | 11.5 (p=0.41) | 9.1 (p=0.0056) |
| final | last | alone | raw | 2.0 | 0.9 (p=0.0021) | 1.5 (p=0.025) | 2.7 (p=0.98) | 1.6 (p=0.19) |
| final | last | alone | center | 13.5 | 2.7 (p=0.00083) | 6.7 (p=0.0014) | 11.8 (p=0.053) | 9.3 (p=1.8e-05) |
| final | last | prompt | raw | 1.3 | 0.7 (p=0.021) | 1.1 (p=0.11) | 1.4 (p=0.72) | 1.2 (p=0.13) |
| final | last | prompt | center | 11.4 | 4.5 (p=4.4e-05) | 9.2 (p=0.051) | 11.3 (p=0.39) | 9.0 (p=0.0037) |

### 3.3 Qwen/Qwen3-1.7B

Layer indices used: None. Extraction 0 s (0 = reused from cache). alone: 2835/5178 words need >1 token, worst bureaucracy=5, bad-tempered=4, congratulate=4, congratulation=4, counsellor=4; prompt: 175/5178 words need >1 token, worst bad-tempered=4, fiftieth=4, thirtieth=4, thought-provoking=4, a.m.=3

| layer | pool | context | p16 targets at rank 1 (of 7), raw / centered | mean rank, raw / centered | sum beats v(b) alone (of 7) | bachelor rank | define2 = paper pair (of 10), raw / centered | magnitude p (similar < random, L2 raw) | median |%diff| ratio similar/random | random-sum baseline (raw) |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| final | mean | prompt | 2 / 3 | 177.9 / 14.4 | 1 | 2124 | 3 / 1 | 0.0045 | 0.64 | 0.965 |
| emb | mean | prompt | 2 / 2 | 2.4 / 2.4 | 0 | 16 | 2 / 2 | 0.53 | 1.07 | 0.400 |
| p75 | mean | prompt | 2 / 2 | 2.4 / 3.3 | 3 | 137 | 2 / 2 | 0.2 | 0.90 | 0.828 |
| p50 | mean | prompt | 2 / 1 | 64.7 / 17.4 | 2 | 249 | 1 / 0 | 0.21 | 0.85 | 0.881 |
| final | last | prompt | 1 / 2 | 192.0 / 184.7 | 1 | 2080 | 3 / 1 | 3.7e-06 | 0.55 | 0.964 |
| p75 | last | prompt | 1 / 1 | 6.0 / 15.0 | 4 | 105 | 2 / 2 | 0.0038 | 0.72 | 0.826 |
| p25 | mean | prompt | 1 / 1 | 8.0 / 2.9 | 0 | 277 | 2 / 1 | 0.0088 | 0.73 | 0.741 |
| emb | last | prompt | 1 / 1 | 175.1 / 46.7 | 1 | 16 | 2 / 2 | 0.56 | 1.11 | 0.402 |
| p50 | last | prompt | 1 / 0 | 377.6 / 54.6 | 2 | 202 | 1 / 0 | 0.0016 | 0.72 | 0.878 |
| emb | mean | alone | 1 / 1 | 893.3 / 666.0 | 3 | 2584 | 0 / 0 | 6.3e-08 | 0.32 | 0.454 |
| p25 | last | prompt | 0 / 0 | 70.3 / 4.1 | 0 | 245 | 2 / 1 | 0.00011 | 0.66 | 0.737 |
| emb | last | alone | 0 / 0 | 1598.3 / 1458.9 | 2 | 732 | 0 / 0 | 0.0066 | 0.73 | 0.577 |
| p25 | mean | alone | 0 / 0 | 1738.4 / 2755.4 | 1 | 2859 | 0 / 0 | 4e-12 | 0.06 | 1.000 |
| final | last | alone | 0 / 0 | 1752.3 / 2448.6 | 6 | 700 | 0 / 0 | 7.7e-10 | 0.25 | 0.956 |
| final | mean | alone | 0 / 0 | 1901.4 / 2715.9 | 2 | 1277 | 0 / 0 | 4.7e-06 | 0.56 | 0.980 |
| p75 | mean | alone | 0 / 0 | 2264.9 / 2659.0 | 2 | 3714 | 0 / 0 | 1.1e-11 | 0.06 | 1.000 |
| p50 | mean | alone | 0 / 0 | 2390.9 / 2579.3 | 1 | 3006 | 0 / 0 | 4.2e-12 | 0.06 | 1.000 |
| p25 | last | alone | 0 / 0 | 2844.7 / 2679.9 | 1 | 4789 | 0 / 0 | 6.1e-06 | 0.33 | 0.898 |
| p75 | last | alone | 0 / 0 | 2874.3 / 2965.0 | 1 | 4577 | 0 / 0 | 6.1e-08 | 0.25 | 0.949 |
| p50 | last | alone | 0 / 0 | 3058.3 / 2502.4 | 1 | 4319 | 0 / 0 | 1.5e-06 | 0.39 | 0.950 |

**Best configuration by the p. 16 criterion: layer final, mean pooling, context prompt.** Its p. 16 table (raw, a and b excluded):

| a + b → target | rank | cos | top-3 |
| :--- | ---: | ---: | ---: |
| young + dog → puppy | 291 | 0.9402 | sexy 0.958, away 0.956, meet 0.956 |
| young + cat → kitten | 173 | 0.9460 | that 0.961, sexy 0.961, meet 0.960 |
| young + duck → duckling | 1 | 0.9781 | duckling 0.978, sexy 0.955, lovely 0.954 |
| female + spouse → wife | 1 | 0.9681 | wife 0.968, spouses 0.967, daughter 0.963 |
| male + spouse → husband | 748 | 0.9308 | female 0.968, wife 0.954, spouses 0.951 |
| male + sibling → brother | 28 | 0.9449 | female 0.965, siblings 0.959, sister 0.954 |
| female + sibling → sister | 3 | 0.9646 | siblings 0.971, daughter 0.966, sister 0.965 |
| man + unmarried → bachelor | 2124 | 0.9231 | woman 0.956, young 0.955, that 0.955 |

define2 (raw) on the same configuration:

| target | paper (Curie) | this model top-3 |
| :--- | :--- | :--- |
| wife | spouse + woman | spouse + woman 0.977<br>daughter + spouse 0.976<br>girlfriend + spouse 0.976 |
| duckling | duck + youngster | awkwardly + bird 0.969<br>bird + cheaply 0.968<br>bird + youngster 0.968 |
| puppy | dog + kitten | dog + kitten 0.952<br>baby + kitten 0.951<br>funny + kitten 0.951 |
| foal | horse + puppy | ago + colt 0.937<br>colt + into 0.934<br>solo + yawn 0.934 |
| freedom | independence + liberty | independence + liberty 0.982<br>happiness + liberty 0.981<br>liberty + unity 0.981 |
| autonomy | control + independence | competence + independence 0.970<br>independence + sovereignty 0.970<br>independence + tolerance 0.969 |
| justice | fairness + judicial | fairness + magistrate 0.966<br>fairness + senate 0.966<br>fairness + jurisdiction 0.966 |
| knowledge | information + wisdom | expertise + information 0.970<br>expertise + intelligence 0.967<br>awareness + expertise 0.967 |
| rationality | logical + reasoning | reasonable + theory 0.981<br>complexity + reasonable 0.981<br>reasonable + theology 0.980 |
| causation | consequence + correlation | assertion + irrationals 0.958<br>assumption + irrationals 0.958<br>duration + irrationals 0.958 |

Magnitude test on every configuration (L2, median |%diff|; p = similar < random):

| layer | pool | ctx | mode | random | us_uk | plural | verb | all |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| emb | mean | alone | raw | 18.4 | 3.0 (p=0.0016) | 5.8 (p=6e-05) | 6.2 (p=0.004) | 5.8 (p=6.3e-08) |
| emb | mean | alone | center | 17.8 | 3.0 (p=0.00073) | 5.9 (p=1.7e-05) | 6.2 (p=0.0048) | 5.5 (p=1.8e-08) |
| emb | mean | prompt | raw | 5.7 | 8.8 (p=0.98) | 7.6 (p=0.81) | 3.3 (p=0.038) | 6.1 (p=0.53) |
| emb | mean | prompt | center | 5.7 | 8.1 (p=0.9) | 5.5 (p=0.61) | 4.0 (p=0.046) | 5.2 (p=0.3) |
| emb | last | alone | raw | 6.5 | 4.2 (p=0.045) | 5.1 (p=0.14) | 4.8 (p=0.043) | 4.8 (p=0.0066) |
| emb | last | alone | center | 6.7 | 3.7 (p=0.025) | 5.0 (p=0.13) | 6.3 (p=0.24) | 5.7 (p=0.027) |
| emb | last | prompt | raw | 5.5 | 8.8 (p=0.99) | 7.6 (p=0.73) | 3.5 (p=0.062) | 6.1 (p=0.56) |
| emb | last | prompt | center | 5.6 | 8.1 (p=0.91) | 5.5 (p=0.69) | 4.0 (p=0.06) | 5.2 (p=0.4) |
| p25 | mean | alone | raw | 44.0 | 0.2 (p=0.00027) | 5.5 (p=3.2e-05) | 2.3 (p=3.8e-06) | 2.4 (p=4e-12) |
| p25 | mean | alone | center | 20.7 | 0.7 (p=0.002) | 19.0 (p=0.087) | 6.6 (p=9.1e-07) | 7.6 (p=8.3e-08) |
| p25 | mean | prompt | raw | 5.5 | 4.4 (p=0.29) | 4.0 (p=0.31) | 3.6 (p=0.0039) | 4.0 (p=0.0088) |
| p25 | mean | prompt | center | 7.6 | 8.4 (p=0.66) | 7.2 (p=0.39) | 5.2 (p=0.034) | 6.3 (p=0.1) |
| p25 | last | alone | raw | 16.8 | 4.4 (p=0.0013) | 5.6 (p=0.00061) | 6.5 (p=0.034) | 5.6 (p=6.1e-06) |
| p25 | last | alone | center | 9.3 | 0.0 (p=0.0019) | 0.0 (p=5.3e-05) | 3.3 (p=0.003) | 0.0 (p=4.4e-08) |
| p25 | last | prompt | raw | 5.4 | 3.5 (p=0.069) | 4.0 (p=0.1) | 2.4 (p=0.00034) | 3.5 (p=0.00011) |
| p25 | last | prompt | center | 7.6 | 8.3 (p=0.67) | 6.2 (p=0.095) | 4.5 (p=0.00057) | 5.7 (p=0.002) |
| p50 | mean | alone | raw | 43.7 | 0.2 (p=0.00028) | 5.5 (p=3.3e-05) | 2.3 (p=3.8e-06) | 2.4 (p=4.2e-12) |
| p50 | mean | alone | center | 20.7 | 0.7 (p=0.002) | 19.0 (p=0.087) | 6.5 (p=8.7e-07) | 7.7 (p=8e-08) |
| p50 | mean | prompt | raw | 3.7 | 2.8 (p=0.12) | 4.2 (p=0.6) | 3.0 (p=0.23) | 3.1 (p=0.21) |
| p50 | mean | prompt | center | 7.1 | 8.0 (p=0.66) | 7.4 (p=0.65) | 5.9 (p=0.14) | 7.0 (p=0.36) |
| p50 | last | alone | raw | 13.7 | 2.4 (p=0.00055) | 4.1 (p=7.1e-05) | 6.0 (p=0.051) | 5.4 (p=1.5e-06) |
| p50 | last | alone | center | 9.3 | 0.1 (p=0.00054) | 0.2 (p=0.00012) | 3.3 (p=0.0023) | 0.2 (p=2.7e-08) |
| p50 | last | prompt | raw | 3.6 | 2.8 (p=0.04) | 3.1 (p=0.15) | 2.4 (p=0.0085) | 2.6 (p=0.0016) |
| p50 | last | prompt | center | 7.1 | 9.5 (p=0.72) | 6.3 (p=0.34) | 5.8 (p=0.12) | 6.7 (p=0.2) |
| p75 | mean | alone | raw | 41.1 | 0.5 (p=0.00031) | 5.4 (p=3.5e-05) | 2.5 (p=8.5e-06) | 2.5 (p=1.1e-11) |
| p75 | mean | alone | center | 20.8 | 0.7 (p=0.0022) | 19.1 (p=0.093) | 6.6 (p=8.8e-07) | 7.9 (p=9.6e-08) |
| p75 | mean | prompt | raw | 4.2 | 1.5 (p=0.047) | 2.9 (p=0.095) | 5.1 (p=0.81) | 3.8 (p=0.2) |
| p75 | mean | prompt | center | 10.9 | 4.2 (p=0.00034) | 6.7 (p=0.0023) | 12.1 (p=0.7) | 7.2 (p=0.0042) |
| p75 | last | alone | raw | 14.1 | 1.3 (p=9e-05) | 2.8 (p=2.8e-06) | 8.5 (p=0.053) | 3.5 (p=6.1e-08) |
| p75 | last | alone | center | 8.9 | 0.5 (p=0.0006) | 0.8 (p=1.6e-05) | 3.4 (p=0.0099) | 1.2 (p=4.2e-08) |
| p75 | last | prompt | raw | 4.0 | 1.4 (p=0.0037) | 2.4 (p=0.00029) | 5.1 (p=0.73) | 2.9 (p=0.0038) |
| p75 | last | prompt | center | 10.6 | 3.7 (p=0.00038) | 4.8 (p=0.00013) | 12.5 (p=0.71) | 6.9 (p=0.00096) |
| final | mean | alone | raw | 10.7 | 4.5 (p=0.0058) | 7.4 (p=0.0084) | 7.4 (p=0.0018) | 6.0 (p=4.7e-06) |
| final | mean | alone | center | 11.3 | 7.0 (p=0.042) | 9.1 (p=0.41) | 13.8 (p=0.83) | 9.8 (p=0.45) |
| final | mean | prompt | raw | 5.7 | 2.5 (p=0.012) | 4.2 (p=0.22) | 4.0 (p=0.031) | 3.6 (p=0.0045) |
| final | mean | prompt | center | 16.6 | 5.5 (p=0.0019) | 9.0 (p=0.00088) | 16.7 (p=0.53) | 10.3 (p=0.0016) |
| final | last | alone | raw | 20.7 | 2.7 (p=0.00028) | 5.0 (p=1.5e-05) | 5.9 (p=0.00075) | 5.1 (p=7.7e-10) |
| final | last | alone | center | 7.5 | 7.0 (p=0.1) | 4.1 (p=0.0085) | 11.4 (p=1) | 7.2 (p=0.49) |
| final | last | prompt | raw | 5.5 | 1.6 (p=0.00019) | 2.2 (p=0.002) | 4.0 (p=0.023) | 3.0 (p=3.7e-06) |
| final | last | prompt | center | 16.9 | 5.9 (p=0.00028) | 7.6 (p=0.00024) | 16.7 (p=0.49) | 10.4 (p=0.0003) |

### 3.x Original OPT/T5 recipe re-extracted on the GPU: the clean 74-pair magnitude test, on both sides of the final layer norm

Word alone, final layer, mean over tokens (OPT drops BOS; T5 keeps EOS), as in the notebooks. The whole 5,124-word vocab and the 51 out-of-vocab pair words are embedded in one run, so the similar-pair arm and the 5,000 random pairs come from the same vectors. A forward hook captures the model's final normalisation layer on both sides: **preln** is the residual stream before it, **postln** is after it (what current transformers returns as `last_hidden_state`). Consistency = cosine and norm ratio between that side and the cached vector over all 5,124 vocab words: the side the cache matches tells which object the paper's magnitude numbers were computed on. Post-LN norms carry little per-word information by construction (each token is normalised to unit variance, then scaled by γ), so a null magnitude result on that side is trivial; the pre-LN side is where magnitude could mean anything.

| model | side | this side vs cache | random median (L2 raw) | us_uk | plural | verb | all pooled |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| opt-1.3b | preln | cos median 1.0000 (p5 1.000, min 0.982); norm ratio 0.999 (p5 0.997, p95 1.002, cv 0.002); norm cv this side 0.160 vs cache 0.160 | 13.8 | 1.7 (n=11, p=7.4e-05) | 9.6 (n=27, p=0.0064) | 9.0 (n=36, p=0.041) | 7.3 (n=74, p=1.6e-05) |
| opt-1.3b | postln | cos median 0.9922 (p5 0.983, min 0.954); norm ratio 0.452 (p5 0.376, p95 0.530, cv 0.105); norm cv this side 0.132 vs cache 0.160 | 13.3 | 0.8 (n=11, p=0.0035) | 2.5 (n=27, p=0.00019) | 4.8 (n=36, p=0.025) | 2.5 (n=74, p=2.9e-06) |
| opt-13b | preln | cos median 0.9998 (p5 1.000, min 0.999); norm ratio 0.996 (p5 0.987, p95 1.002, cv 0.004); norm cv this side 0.290 vs cache 0.293 | 24.3 | 7.9 (n=11, p=0.005) | 15.4 (n=27, p=0.057) | 12.2 (n=36, p=0.0026) | 12.4 (n=74, p=5.1e-05) |
| opt-13b | postln | cos median 0.9967 (p5 0.972, min 0.883); norm ratio 0.416 (p5 0.343, p95 0.503, cv 0.119); norm cv this side 0.222 vs cache 0.293 | 10.7 | 1.1 (n=11, p=0.0015) | 3.1 (n=27, p=0.00051) | 2.3 (n=36, p=0.0023) | 2.3 (n=74, p=1.8e-07) |
| t5-large | preln | cos median 0.7461 (p5 0.179, min 0.065); norm ratio 7843.626 (p5 1432.601, p95 55000.875, cv 2.285); norm cv this side 2.064 vs cache 0.131 | 11.5 | 14.0 (n=11, p=0.93) | 11.4 (n=27, p=0.66) | 9.4 (n=36, p=0.33) | 10.9 (n=74, p=0.7) |
| t5-large | postln | cos median 0.9999 (p5 1.000, min 0.573); norm ratio 1.000 (p5 0.997, p95 1.003, cv 0.016); norm cv this side 0.131 vs cache 0.131 | 9.9 | 18.3 (n=11, p=0.94) | 11.3 (n=27, p=0.86) | 11.3 (n=36, p=0.93) | 14.2 (n=74, p=0.99) |
| t5-3b | preln | cos median 0.1014 (p5 0.074, min 0.049); norm ratio 81862.234 (p5 23357.188, p95 117283.125, cv 0.304); norm cv this side 0.285 vs cache 0.113 | 14.4 | 48.8 (n=11, p=0.98) | 15.7 (n=27, p=0.42) | 13.1 (n=36, p=0.28) | 15.3 (n=74, p=0.62) |
| t5-3b | postln | cos median 0.9999 (p5 1.000, min 0.820); norm ratio 1.000 (p5 0.996, p95 1.004, cv 0.005); norm cv this side 0.113 vs cache 0.113 | 9.7 | 6.5 (n=11, p=0.22) | 14.0 (n=27, p=0.92) | 8.5 (n=36, p=0.093) | 9.0 (n=74, p=0.36) |

All metrics, pooled over the 74 pairs:

| model | side | metric | mode | similar median | random median | p similar<random | p similar>random |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| opt-1.3b | preln | L1 | raw | 5.8 | 13.3 | 2.3e-06 | 1 |
| opt-1.3b | preln | L1 | center | 7.2 | 16.0 | 1.2e-08 | 1 |
| opt-1.3b | preln | L2 | raw | 7.3 | 13.8 | 1.6e-05 | 1 |
| opt-1.3b | preln | L2 | center | 7.1 | 16.7 | 4.1e-09 | 1 |
| opt-1.3b | postln | L1 | raw | 4.4 | 13.4 | 1.1e-05 | 1 |
| opt-1.3b | postln | L1 | center | 5.6 | 16.4 | 2.8e-07 | 1 |
| opt-1.3b | postln | L2 | raw | 2.5 | 13.3 | 2.9e-06 | 1 |
| opt-1.3b | postln | L2 | center | 6.2 | 16.8 | 8.4e-08 | 1 |
| opt-13b | preln | L1 | raw | 7.6 | 10.4 | 0.023 | 0.98 |
| opt-13b | preln | L1 | center | 10.5 | 13.1 | 0.042 | 0.96 |
| opt-13b | preln | L2 | raw | 12.4 | 24.3 | 5.1e-05 | 1 |
| opt-13b | preln | L2 | center | 13.1 | 20.5 | 0.00049 | 1 |
| opt-13b | postln | L1 | raw | 7.6 | 10.7 | 0.0015 | 1 |
| opt-13b | postln | L1 | center | 10.0 | 15.1 | 0.0014 | 1 |
| opt-13b | postln | L2 | raw | 2.3 | 10.7 | 1.8e-07 | 1 |
| opt-13b | postln | L2 | center | 4.2 | 16.1 | 3.1e-07 | 1 |
| t5-large | preln | L1 | raw | 8.5 | 8.4 | 0.56 | 0.44 |
| t5-large | preln | L1 | center | 7.3 | 7.7 | 0.42 | 0.58 |
| t5-large | preln | L2 | raw | 10.9 | 11.5 | 0.7 | 0.3 |
| t5-large | preln | L2 | center | 9.0 | 6.5 | 0.96 | 0.041 |
| t5-large | postln | L1 | raw | 13.9 | 9.4 | 0.94 | 0.056 |
| t5-large | postln | L1 | center | 7.6 | 10.0 | 0.0042 | 1 |
| t5-large | postln | L2 | raw | 14.2 | 9.9 | 0.99 | 0.012 |
| t5-large | postln | L2 | center | 7.0 | 10.0 | 0.0029 | 1 |
| t5-3b | preln | L1 | raw | 14.5 | 14.8 | 0.43 | 0.57 |
| t5-3b | preln | L1 | center | 41.3 | 55.4 | 0.0052 | 0.99 |
| t5-3b | preln | L2 | raw | 15.3 | 14.4 | 0.62 | 0.38 |
| t5-3b | preln | L2 | center | 59.5 | 75.0 | 0.0057 | 0.99 |
| t5-3b | postln | L1 | raw | 10.1 | 9.8 | 0.3 | 0.7 |
| t5-3b | postln | L1 | center | 10.0 | 11.2 | 0.49 | 0.51 |
| t5-3b | postln | L2 | raw | 9.0 | 9.7 | 0.36 | 0.64 |
| t5-3b | postln | L2 | center | 9.0 | 10.9 | 0.31 | 0.69 |

### Recommendation for the method section

_See the Reading paragraphs above; the recommendation paragraph is written by hand in results/NOTES.md once the tables are in._

_Phase 3 ran in 2048 s._

## Notes and deviations

### What the cached OPT vectors are (established 2026-09-04)

The cached OPT-1.3B vectors (`opt/1_3B.txt`) are the **residual stream before OPT's final layer norm**, not `last_hidden_state` as produced by any current transformers build. Evidence, all in `results/`:

1. Re-extracting with the notebook recipe under current transformers (5.16) gives vectors with cosine 0.992 to the cache but **0.45× the norm** (per word: p5 0.38, p50 0.45, p95 0.53; `phase3.json`, opt-1.3b postln side).
2. Capturing both sides of `decoder.final_layer_norm` with a forward hook: the **pre-LN side matches the cache with cosine 1.0000 and norm ratio 0.999 (CV 0.002) over all 5,124 vocab words**; the post-LN side is the 0.45× one (`phase3.json`, opt-1.3b preln/postln).
3. Reproduction under the pinned version, transformers 4.20.1 (`src/repro_tf420.py`, throwaway venv): the notebook function verbatim gives the **post-LN** vectors (cosine 0.99995, norm ratio 1.0000 to the current re-extraction; `repro_tf420.json`). So 4.20.1 by itself did not drop the norm.
4. Loading under 4.20.1 with `_remove_final_layer_norm=True` reproduces the cache to **max abs difference 4e-4** (`repro_tf420_noln.json`), and emits the "model.decoder.final_layer_norm.weight/bias were not used" warning. Mechanism, from the transformers source history: releases up to 4.20.0 had no decoder-level final LayerNorm in `OPTModel`; 4.20.1 added it, with `_remove_final_layer_norm` kept "for backward compatibility with checkpoints fine-tuned before transformers v4.20.1" (metaseq PR 164). The saved outputs of every OPT notebook (`magnitudes*.ipynb`, `negatives*.ipynb`, `simdef*.ipynb`) contain that same unused-weights warning, so the caches were made by a pre-4.20.1 build, whatever `environment.yml` pins now. No revision of the Hub config for facebook/opt-1.3b ever set the flag.
5. `opt/1_3B.txt.orig` is the same extraction on the older 3,471-word vocab (cosine 1.0000, ratio 1.000 on shared words); it is not a different generation.

Consequences for the magnitude study: the paper's OPT magnitude numbers were computed on pre-LN residual-stream vectors. Post-LN vectors are normalised per token and their norms carry little per-word information by construction, so a magnitude test on them is trivially near-null. Phase 3 therefore reports the 74-pair test on **both** sides, labelled, and the paper's method section has to say which one it means. Whether the same holds for T5 (whose encoder also ends in a norm) is answered by the t5-large / t5-3b rows of the same table.
