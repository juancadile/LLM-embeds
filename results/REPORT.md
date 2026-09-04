# LLM-embeds rerun — report

Generated 2026-09-04 15:44 by `python -m src.run_all` at git 02e7026; python 3.11.4, numpy 1.26.4, host MacBook-Pro-5.local.
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

| model | pairs parsed | us_uk mean / median |%diff| | plural mean / median |%diff| | verb mean / median |%diff| | all | largest |diff| |
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

## Phase 3

_Not run (no results/phase3.json)._
