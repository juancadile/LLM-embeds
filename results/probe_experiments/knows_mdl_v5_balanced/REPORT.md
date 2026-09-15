# Model-relative complexity of KNOWS

This report concerns compact decoding of each model's own forced-choice decision rule. It does not estimate the absolute Kolmogorov complexity of the human concept of knowledge.

## Protocol audit

Underlying scenarios: 2,048; surface realizations: 6,144; grouped split counts: `{'coding': 1444, 'evaluation': 394, 'tune': 210}`.

## Label stability

| Cell | Predicate | Stable | Total | Class 0 | Class 1 | Sample-count gate |
|:--|:--|--:|--:|--:|--:|:--|
| meta-llama-llama-3.1-8b-instruct | believes | 4917 | 6144 | 2 | 4915 | no |
| meta-llama-llama-3.1-8b-instruct | justified | 3522 | 6144 | 7 | 3515 | no |
| meta-llama-llama-3.1-8b-instruct | knows | 4257 | 6144 | 0 | 4257 | no |
| meta-llama-llama-3.1-8b-instruct | lucky_guessed | 3676 | 6144 | 35 | 3641 | no |
| meta-llama-llama-3.1-8b-instruct | true | 4211 | 6144 | 20 | 4191 | no |
| qwen-qwen3-1.7b | believes | 5286 | 6144 | 29 | 5257 | no |
| qwen-qwen3-1.7b | justified | 3730 | 6144 | 841 | 2889 | no |
| qwen-qwen3-1.7b | knows | 4645 | 6144 | 785 | 3860 | no |
| qwen-qwen3-1.7b | lucky_guessed | 3228 | 6144 | 3221 | 7 | no |
| qwen-qwen3-1.7b | true | 4002 | 6144 | 621 | 3381 | no |
| qwen-qwen3-14b | believes | 6022 | 6144 | 2194 | 3828 | yes |
| qwen-qwen3-14b | justified | 5958 | 6144 | 4025 | 1933 | yes |
| qwen-qwen3-14b | knows | 5893 | 6144 | 4253 | 1640 | yes |
| qwen-qwen3-14b | lucky_guessed | 5585 | 6144 | 3881 | 1704 | yes |
| qwen-qwen3-14b | true | 5885 | 6144 | 3858 | 2027 | yes |

### Representative ambiguity cases

- `s00000-p1` / **true**, confidence 0.605: For Harper, the proposition under consideration was that the orchard gate is unlocked. Events did not match the claim. Harper recalled an earlier check of the state of the orchard gate, which appeared to support the proposition; the memory was normally dependable. Harper accepted the claim. The position was selected arbitrarily without using the source. Contrary information was absent.
- `s00001-p1` / **lucky_guessed**, confidence 0.599: For Harper, the proposition under consideration was that the orchard gate is unlocked. Events did not match the claim. Harper recalled an earlier check of the state of the orchard gate, which appeared to support the proposition; the memory was normally dependable. Harper accepted the claim. The position was selected arbitrarily without using the source. Credible contrary information was available but left unresolved.
- `s00002-p1` / **justified**, confidence 0.653: For Ari, the proposition under consideration was that the meeting is in room twelve. Events did not match the claim. Ari made a direct observation concerning the meeting room, which appeared to support the proposition; observations of this kind were usually accurate. Ari rejected the claim. Separate coincidence was absent from the outcome. Contrary information was absent.
- `s00007-p0` / **knows**, confidence 0.785: Casey considered whether the train leaves from platform four. Casey checked a calibrated display for the departure platform, which appeared to support the proposition. the display usually works correctly. Casey accepted the claim. Events matched the claim. Credible contrary information was available but left unresolved. The position was selected arbitrarily without using the source.
- `s00007-p0` / **lucky_guessed**, confidence 0.632: Casey considered whether the train leaves from platform four. Casey checked a calibrated display for the departure platform, which appeared to support the proposition. the display usually works correctly. Casey accepted the claim. Events matched the claim. Credible contrary information was available but left unresolved. The position was selected arbitrarily without using the source.
- `s00007-p1` / **lucky_guessed**, confidence 0.545: For Casey, the proposition under consideration was that the train leaves from platform four. Events matched the claim. Casey checked a calibrated display for the departure platform, which appeared to support the proposition; the display usually works correctly. Casey accepted the claim. The position was selected arbitrarily without using the source. Credible contrary information was available but left unresolved.
- `s00007-p2` / **lucky_guessed**, confidence 0.697: The issue was whether the train leaves from platform four. Casey checked a calibrated display for the departure platform, which appeared to support the proposition, and the display usually works correctly. Casey's resulting position was described as follows: Casey accepted the claim. Events matched the claim. Credible contrary information was available but left unresolved. The position was selected arbitrarily without using the source.
- `s00013-p1` / **lucky_guessed**, confidence 0.623: For Harper, the proposition under consideration was that the museum closes at six. Events did not match the claim. Harper checked a calibrated display for the posted closing time, which appeared to support the proposition; the display usually works correctly. Harper accepted the claim. The position was selected arbitrarily without using the source. Credible contrary information was available but left unresolved.
- `s00019-p0` / **knows**, confidence 0.594: Casey considered whether the orchard gate is unlocked. Casey made a direct observation concerning the state of the orchard gate, which appeared to support the proposition. observations of this kind were usually accurate. Casey accepted the claim. Events matched the claim. Credible contrary information was available but left unresolved. The position was selected arbitrarily without using the source.
- `s00019-p1` / **justified**, confidence 0.649: For Casey, the proposition under consideration was that the orchard gate is unlocked. Events matched the claim. Casey made a direct observation concerning the state of the orchard gate, which appeared to support the proposition; observations of this kind were usually accurate. Casey accepted the claim. The position was selected arbitrarily without using the source. Credible contrary information was available but left unresolved.

## Layerwise online MDL

| Cell | Probe | Bits/label | Compression | Eval macro-F1 | Within-predicate quality gate |
|:--|:--|--:|--:|--:|:--|
| qwen-qwen3-14b/cells/believes/block29.last | mlp | 0.108 | 0.887 | 0.994 | yes |
| qwen-qwen3-14b/cells/believes/emb.last | mlp | 0.951 | 0.000 | 0.381 | no |
| qwen-qwen3-14b/cells/believes/final.last | mlp | 0.214 | 0.775 | 0.989 | yes |
| qwen-qwen3-14b/cells/believes/p25.last | mlp | 0.251 | 0.736 | 0.991 | yes |
| qwen-qwen3-14b/cells/believes/p50.last | mlp | 0.137 | 0.856 | 0.991 | yes |
| qwen-qwen3-14b/cells/believes/p75.last | mlp | 0.114 | 0.880 | 0.993 | yes |
| qwen-qwen3-14b/cells/justified/block29.last | mlp | 0.145 | 0.843 | 0.990 | yes |
| qwen-qwen3-14b/cells/justified/emb.last | mlp | 0.929 | -0.002 | 0.426 | no |
| qwen-qwen3-14b/cells/justified/final.last | mlp | 0.231 | 0.752 | 0.984 | yes |
| qwen-qwen3-14b/cells/justified/p25.last | mlp | 0.250 | 0.730 | 0.989 | yes |
| qwen-qwen3-14b/cells/justified/p50.last | mlp | 0.151 | 0.837 | 0.987 | yes |
| qwen-qwen3-14b/cells/justified/p75.last | mlp | 0.145 | 0.844 | 0.990 | yes |
| qwen-qwen3-14b/cells/knows/block01.last | mlp | 0.191 | 0.782 | 0.982 | yes |
| qwen-qwen3-14b/cells/knows/block02.last | mlp | 0.203 | 0.768 | 0.978 | yes |
| qwen-qwen3-14b/cells/knows/block03.last | mlp | 0.238 | 0.728 | 0.982 | yes |
| qwen-qwen3-14b/cells/knows/block04.last | mlp | 0.221 | 0.746 | 0.983 | yes |
| qwen-qwen3-14b/cells/knows/block05.last | mlp | 0.229 | 0.738 | 0.983 | yes |
| qwen-qwen3-14b/cells/knows/block06.last | mlp | 0.221 | 0.747 | 0.983 | yes |
| qwen-qwen3-14b/cells/knows/block07.last | mlp | 0.237 | 0.728 | 0.986 | yes |
| qwen-qwen3-14b/cells/knows/block08.last | mlp | 0.235 | 0.731 | 0.986 | yes |
| qwen-qwen3-14b/cells/knows/block09.last | mlp | 0.253 | 0.711 | 0.985 | yes |
| qwen-qwen3-14b/cells/knows/block10.last | mlp | 0.254 | 0.709 | 0.985 | yes |
| qwen-qwen3-14b/cells/knows/block11.last | mlp | 0.244 | 0.721 | 0.984 | yes |
| qwen-qwen3-14b/cells/knows/block12.last | mlp | 0.230 | 0.737 | 0.982 | yes |
| qwen-qwen3-14b/cells/knows/block13.last | mlp | 0.222 | 0.745 | 0.979 | yes |
| qwen-qwen3-14b/cells/knows/block14.last | mlp | 0.215 | 0.753 | 0.983 | yes |
| qwen-qwen3-14b/cells/knows/block15.last | mlp | 0.213 | 0.756 | 0.985 | yes |
| qwen-qwen3-14b/cells/knows/block16.last | mlp | 0.203 | 0.767 | 0.985 | yes |
| qwen-qwen3-14b/cells/knows/block17.last | mlp | 0.187 | 0.786 | 0.988 | yes |
| qwen-qwen3-14b/cells/knows/block18.last | mlp | 0.178 | 0.796 | 0.985 | yes |
| qwen-qwen3-14b/cells/knows/block19.last | mlp | 0.183 | 0.790 | 0.982 | yes |
| qwen-qwen3-14b/cells/knows/block20.last | mlp | 0.174 | 0.801 | 0.984 | yes |
| qwen-qwen3-14b/cells/knows/block21.last | mlp | 0.169 | 0.807 | 0.986 | yes |
| qwen-qwen3-14b/cells/knows/block22.last | mlp | 0.157 | 0.820 | 0.984 | yes |
| qwen-qwen3-14b/cells/knows/block23.last | mlp | 0.157 | 0.820 | 0.984 | yes |
| qwen-qwen3-14b/cells/knows/block24.last | mlp | 0.155 | 0.823 | 0.981 | yes |
| qwen-qwen3-14b/cells/knows/block25.last | mlp | 0.151 | 0.827 | 0.983 | yes |
| qwen-qwen3-14b/cells/knows/block26.last | mlp | 0.150 | 0.829 | 0.985 | yes |
| qwen-qwen3-14b/cells/knows/block27.last | mlp | 0.150 | 0.828 | 0.986 | yes |
| qwen-qwen3-14b/cells/knows/block28.last | mlp | 0.151 | 0.827 | 0.984 | yes |
| qwen-qwen3-14b/cells/knows/block29.last | mlp | 0.148 | 0.830 | 0.984 | yes |
| qwen-qwen3-14b/cells/knows/block30.last | mlp | 0.155 | 0.822 | 0.987 | yes |
| qwen-qwen3-14b/cells/knows/block31.last | mlp | 0.154 | 0.824 | 0.982 | yes |
| qwen-qwen3-14b/cells/knows/block32.last | mlp | 0.167 | 0.809 | 0.986 | yes |
| qwen-qwen3-14b/cells/knows/block33.last | mlp | 0.171 | 0.804 | 0.986 | yes |
| qwen-qwen3-14b/cells/knows/block34.last | mlp | 0.172 | 0.803 | 0.987 | yes |
| qwen-qwen3-14b/cells/knows/block35.last | mlp | 0.196 | 0.776 | 0.988 | yes |
| qwen-qwen3-14b/cells/knows/block36.last | mlp | 0.206 | 0.764 | 0.986 | yes |
| qwen-qwen3-14b/cells/knows/block37.last | mlp | 0.221 | 0.747 | 0.985 | yes |
| qwen-qwen3-14b/cells/knows/block38.last | mlp | 0.235 | 0.731 | 0.982 | yes |
| qwen-qwen3-14b/cells/knows/block39.last | mlp | 0.239 | 0.727 | 0.982 | yes |
| qwen-qwen3-14b/cells/knows/emb.last | mlp | 0.875 | -0.002 | 0.436 | no |
| qwen-qwen3-14b/cells/knows/final.last | mlp | 0.243 | 0.722 | 0.986 | yes |
| qwen-qwen3-14b/cells/knows/p25.last | mlp | 0.254 | 0.709 | 0.985 | yes |
| qwen-qwen3-14b/cells/knows/p50.last | mlp | 0.174 | 0.801 | 0.984 | yes |
| qwen-qwen3-14b/cells/knows/p75.last | mlp | 0.155 | 0.822 | 0.987 | yes |
| qwen-qwen3-14b/cells/lucky_guessed/block29.last | mlp | 0.195 | 0.777 | 0.986 | yes |
| qwen-qwen3-14b/cells/lucky_guessed/emb.last | mlp | 0.876 | 0.002 | 0.392 | no |
| qwen-qwen3-14b/cells/lucky_guessed/final.last | mlp | 0.298 | 0.661 | 0.978 | yes |
| qwen-qwen3-14b/cells/lucky_guessed/p25.last | mlp | 0.342 | 0.611 | 0.960 | yes |
| qwen-qwen3-14b/cells/lucky_guessed/p50.last | mlp | 0.221 | 0.748 | 0.986 | yes |
| qwen-qwen3-14b/cells/lucky_guessed/p75.last | mlp | 0.201 | 0.771 | 0.987 | yes |
| qwen-qwen3-14b/cells/true/block29.last | mlp | 0.156 | 0.835 | 0.984 | yes |
| qwen-qwen3-14b/cells/true/emb.last | mlp | 0.947 | -0.000 | 0.418 | no |
| qwen-qwen3-14b/cells/true/final.last | mlp | 0.261 | 0.725 | 0.975 | yes |
| qwen-qwen3-14b/cells/true/p25.last | mlp | 0.271 | 0.713 | 0.980 | yes |
| qwen-qwen3-14b/cells/true/p50.last | mlp | 0.178 | 0.812 | 0.983 | yes |
| qwen-qwen3-14b/cells/true/p75.last | mlp | 0.158 | 0.833 | 0.982 | yes |
| qwen-qwen3-14b/controls/believes/question_only.emb.last | mlp | 0.951 | 0.000 | 0.381 | no |
| qwen-qwen3-14b/controls/believes/question_only.final.last | mlp | 1.659 | -0.745 | 0.420 | no |
| qwen-qwen3-14b/controls/believes/question_only.p25.last | mlp | 1.542 | -0.621 | 0.442 | no |
| qwen-qwen3-14b/controls/believes/question_only.p50.last | mlp | 1.567 | -0.648 | 0.435 | no |
| qwen-qwen3-14b/controls/believes/question_only.p75.last | mlp | 1.649 | -0.734 | 0.448 | no |
| qwen-qwen3-14b/controls/justified/question_only.emb.last | mlp | 0.929 | -0.002 | 0.426 | no |
| qwen-qwen3-14b/controls/justified/question_only.final.last | mlp | 1.496 | -0.613 | 0.475 | no |
| qwen-qwen3-14b/controls/justified/question_only.p25.last | mlp | 1.471 | -0.585 | 0.467 | no |
| qwen-qwen3-14b/controls/justified/question_only.p50.last | mlp | 1.457 | -0.570 | 0.470 | no |
| qwen-qwen3-14b/controls/justified/question_only.p75.last | mlp | 1.526 | -0.645 | 0.461 | no |
| qwen-qwen3-14b/controls/knows/question_only.emb.last | mlp | 0.875 | -0.002 | 0.436 | no |
| qwen-qwen3-14b/controls/knows/question_only.final.last | mlp | 1.362 | -0.560 | 0.444 | no |
| qwen-qwen3-14b/controls/knows/question_only.p25.last | mlp | 1.316 | -0.507 | 0.438 | no |
| qwen-qwen3-14b/controls/knows/question_only.p50.last | mlp | 1.315 | -0.506 | 0.460 | no |
| qwen-qwen3-14b/controls/knows/question_only.p75.last | mlp | 1.345 | -0.541 | 0.443 | no |
| qwen-qwen3-14b/controls/lucky_guessed/question_only.emb.last | mlp | 0.876 | 0.002 | 0.392 | no |
| qwen-qwen3-14b/controls/lucky_guessed/question_only.final.last | mlp | 1.060 | -0.208 | 0.402 | no |
| qwen-qwen3-14b/controls/lucky_guessed/question_only.p25.last | mlp | 1.074 | -0.225 | 0.402 | no |
| qwen-qwen3-14b/controls/lucky_guessed/question_only.p50.last | mlp | 1.052 | -0.200 | 0.402 | no |
| qwen-qwen3-14b/controls/lucky_guessed/question_only.p75.last | mlp | 1.057 | -0.205 | 0.392 | no |
| qwen-qwen3-14b/controls/true/question_only.emb.last | mlp | 0.947 | -0.000 | 0.418 | no |
| qwen-qwen3-14b/controls/true/question_only.final.last | mlp | 1.254 | -0.324 | 0.429 | no |
| qwen-qwen3-14b/controls/true/question_only.p25.last | mlp | 1.265 | -0.336 | 0.438 | no |
| qwen-qwen3-14b/controls/true/question_only.p50.last | mlp | 1.278 | -0.349 | 0.448 | no |
| qwen-qwen3-14b/controls/true/question_only.p75.last | mlp | 1.280 | -0.352 | 0.433 | no |
| qwen-qwen3-14b/cells/believes/block29.last | linear | 0.091 | 0.904 | 0.995 | yes |
| qwen-qwen3-14b/cells/believes/emb.last | linear | 0.953 | -0.002 | 0.381 | no |
| qwen-qwen3-14b/cells/believes/final.last | linear | 0.145 | 0.848 | 0.993 | yes |
| qwen-qwen3-14b/cells/believes/p25.last | linear | 0.185 | 0.806 | 0.995 | yes |
| qwen-qwen3-14b/cells/believes/p50.last | linear | 0.110 | 0.884 | 0.993 | yes |
| qwen-qwen3-14b/cells/believes/p75.last | linear | 0.096 | 0.899 | 0.996 | yes |
| qwen-qwen3-14b/cells/justified/block29.last | linear | 0.129 | 0.861 | 0.986 | yes |
| qwen-qwen3-14b/cells/justified/emb.last | linear | 0.935 | -0.008 | 0.426 | no |
| qwen-qwen3-14b/cells/justified/final.last | linear | 0.191 | 0.794 | 0.984 | yes |
| qwen-qwen3-14b/cells/justified/p25.last | linear | 0.331 | 0.643 | 0.968 | yes |
| qwen-qwen3-14b/cells/justified/p50.last | linear | 0.167 | 0.820 | 0.986 | yes |
| qwen-qwen3-14b/cells/justified/p75.last | linear | 0.137 | 0.852 | 0.986 | yes |
| qwen-qwen3-14b/cells/knows/block01.last | linear | 0.335 | 0.616 | 0.964 | yes |
| qwen-qwen3-14b/cells/knows/block02.last | linear | 0.331 | 0.621 | 0.961 | yes |
| qwen-qwen3-14b/cells/knows/block03.last | linear | 0.348 | 0.601 | 0.957 | yes |
| qwen-qwen3-14b/cells/knows/block04.last | linear | 0.342 | 0.608 | 0.959 | yes |
| qwen-qwen3-14b/cells/knows/block05.last | linear | 0.335 | 0.617 | 0.964 | yes |
| qwen-qwen3-14b/cells/knows/block06.last | linear | 0.335 | 0.617 | 0.966 | yes |
| qwen-qwen3-14b/cells/knows/block07.last | linear | 0.338 | 0.613 | 0.967 | yes |
| qwen-qwen3-14b/cells/knows/block08.last | linear | 0.349 | 0.601 | 0.963 | yes |
| qwen-qwen3-14b/cells/knows/block09.last | linear | 0.354 | 0.595 | 0.967 | yes |
| qwen-qwen3-14b/cells/knows/block10.last | linear | 0.350 | 0.599 | 0.958 | yes |
| qwen-qwen3-14b/cells/knows/block11.last | linear | 0.333 | 0.618 | 0.960 | yes |
| qwen-qwen3-14b/cells/knows/block12.last | linear | 0.311 | 0.644 | 0.964 | yes |
| qwen-qwen3-14b/cells/knows/block13.last | linear | 0.293 | 0.664 | 0.972 | yes |
| qwen-qwen3-14b/cells/knows/block14.last | linear | 0.279 | 0.680 | 0.978 | yes |
| qwen-qwen3-14b/cells/knows/block15.last | linear | 0.262 | 0.700 | 0.981 | yes |
| qwen-qwen3-14b/cells/knows/block16.last | linear | 0.255 | 0.708 | 0.986 | yes |
| qwen-qwen3-14b/cells/knows/block17.last | linear | 0.243 | 0.722 | 0.985 | yes |
| qwen-qwen3-14b/cells/knows/block18.last | linear | 0.214 | 0.755 | 0.983 | yes |
| qwen-qwen3-14b/cells/knows/block19.last | linear | 0.206 | 0.764 | 0.981 | yes |
| qwen-qwen3-14b/cells/knows/block20.last | linear | 0.191 | 0.781 | 0.982 | yes |
| qwen-qwen3-14b/cells/knows/block21.last | linear | 0.176 | 0.798 | 0.982 | yes |
| qwen-qwen3-14b/cells/knows/block22.last | linear | 0.172 | 0.803 | 0.980 | yes |
| qwen-qwen3-14b/cells/knows/block23.last | linear | 0.165 | 0.811 | 0.983 | yes |
| qwen-qwen3-14b/cells/knows/block24.last | linear | 0.159 | 0.817 | 0.981 | yes |
| qwen-qwen3-14b/cells/knows/block25.last | linear | 0.152 | 0.826 | 0.980 | yes |
| qwen-qwen3-14b/cells/knows/block26.last | linear | 0.151 | 0.827 | 0.982 | yes |
| qwen-qwen3-14b/cells/knows/block27.last | linear | 0.155 | 0.823 | 0.980 | yes |
| qwen-qwen3-14b/cells/knows/block28.last | linear | 0.149 | 0.829 | 0.980 | yes |
| qwen-qwen3-14b/cells/knows/block29.last | linear | 0.151 | 0.827 | 0.980 | yes |
| qwen-qwen3-14b/cells/knows/block30.last | linear | 0.153 | 0.825 | 0.980 | yes |
| qwen-qwen3-14b/cells/knows/block31.last | linear | 0.158 | 0.819 | 0.979 | yes |
| qwen-qwen3-14b/cells/knows/block32.last | linear | 0.163 | 0.813 | 0.979 | yes |
| qwen-qwen3-14b/cells/knows/block33.last | linear | 0.167 | 0.809 | 0.981 | yes |
| qwen-qwen3-14b/cells/knows/block34.last | linear | 0.168 | 0.807 | 0.981 | yes |
| qwen-qwen3-14b/cells/knows/block35.last | linear | 0.177 | 0.797 | 0.980 | yes |
| qwen-qwen3-14b/cells/knows/block36.last | linear | 0.183 | 0.790 | 0.979 | yes |
| qwen-qwen3-14b/cells/knows/block37.last | linear | 0.190 | 0.782 | 0.980 | yes |
| qwen-qwen3-14b/cells/knows/block38.last | linear | 0.198 | 0.773 | 0.979 | yes |
| qwen-qwen3-14b/cells/knows/block39.last | linear | 0.204 | 0.766 | 0.980 | yes |
| qwen-qwen3-14b/cells/knows/emb.last | linear | 0.890 | -0.020 | 0.436 | no |
| qwen-qwen3-14b/cells/knows/final.last | linear | 0.206 | 0.764 | 0.980 | yes |
| qwen-qwen3-14b/cells/knows/p25.last | linear | 0.350 | 0.599 | 0.958 | yes |
| qwen-qwen3-14b/cells/knows/p50.last | linear | 0.191 | 0.781 | 0.982 | yes |
| qwen-qwen3-14b/cells/knows/p75.last | linear | 0.153 | 0.825 | 0.980 | yes |
| qwen-qwen3-14b/cells/lucky_guessed/block29.last | linear | 0.188 | 0.785 | 0.983 | yes |
| qwen-qwen3-14b/cells/lucky_guessed/emb.last | linear | 0.891 | -0.015 | 0.392 | no |
| qwen-qwen3-14b/cells/lucky_guessed/final.last | linear | 0.250 | 0.716 | 0.978 | yes |
| qwen-qwen3-14b/cells/lucky_guessed/p25.last | linear | 0.345 | 0.606 | 0.931 | yes |
| qwen-qwen3-14b/cells/lucky_guessed/p50.last | linear | 0.226 | 0.743 | 0.986 | yes |
| qwen-qwen3-14b/cells/lucky_guessed/p75.last | linear | 0.196 | 0.776 | 0.982 | yes |
| qwen-qwen3-14b/cells/true/block29.last | linear | 0.125 | 0.868 | 0.981 | yes |
| qwen-qwen3-14b/cells/true/emb.last | linear | 0.951 | -0.004 | 0.418 | no |
| qwen-qwen3-14b/cells/true/final.last | linear | 0.172 | 0.818 | 0.980 | yes |
| qwen-qwen3-14b/cells/true/p25.last | linear | 0.311 | 0.671 | 0.959 | yes |
| qwen-qwen3-14b/cells/true/p50.last | linear | 0.161 | 0.830 | 0.985 | yes |
| qwen-qwen3-14b/cells/true/p75.last | linear | 0.129 | 0.863 | 0.982 | yes |
| qwen-qwen3-14b/controls/believes/question_only.emb.last | linear | 0.953 | -0.002 | 0.381 | no |
| qwen-qwen3-14b/controls/believes/question_only.final.last | linear | 1.213 | -0.275 | 0.450 | no |
| qwen-qwen3-14b/controls/believes/question_only.p25.last | linear | 1.208 | -0.270 | 0.470 | no |
| qwen-qwen3-14b/controls/believes/question_only.p50.last | linear | 1.185 | -0.246 | 0.456 | no |
| qwen-qwen3-14b/controls/believes/question_only.p75.last | linear | 1.184 | -0.245 | 0.477 | no |
| qwen-qwen3-14b/controls/justified/question_only.emb.last | linear | 0.935 | -0.008 | 0.426 | no |
| qwen-qwen3-14b/controls/justified/question_only.final.last | linear | 1.148 | -0.237 | 0.476 | no |
| qwen-qwen3-14b/controls/justified/question_only.p25.last | linear | 1.154 | -0.244 | 0.472 | no |
| qwen-qwen3-14b/controls/justified/question_only.p50.last | linear | 1.176 | -0.268 | 0.473 | no |
| qwen-qwen3-14b/controls/justified/question_only.p75.last | linear | 1.179 | -0.271 | 0.474 | no |
| qwen-qwen3-14b/controls/knows/question_only.emb.last | linear | 0.890 | -0.020 | 0.436 | no |
| qwen-qwen3-14b/controls/knows/question_only.final.last | linear | 1.119 | -0.282 | 0.485 | no |
| qwen-qwen3-14b/controls/knows/question_only.p25.last | linear | 1.129 | -0.293 | 0.471 | no |
| qwen-qwen3-14b/controls/knows/question_only.p50.last | linear | 1.124 | -0.287 | 0.470 | no |
| qwen-qwen3-14b/controls/knows/question_only.p75.last | linear | 1.110 | -0.271 | 0.465 | no |
| qwen-qwen3-14b/controls/lucky_guessed/question_only.emb.last | linear | 0.891 | -0.015 | 0.392 | no |
| qwen-qwen3-14b/controls/lucky_guessed/question_only.final.last | linear | 1.023 | -0.166 | 0.441 | no |
| qwen-qwen3-14b/controls/lucky_guessed/question_only.p25.last | linear | 1.034 | -0.178 | 0.418 | no |
| qwen-qwen3-14b/controls/lucky_guessed/question_only.p50.last | linear | 1.018 | -0.160 | 0.409 | no |
| qwen-qwen3-14b/controls/lucky_guessed/question_only.p75.last | linear | 1.017 | -0.160 | 0.442 | no |
| qwen-qwen3-14b/controls/true/question_only.emb.last | linear | 0.951 | -0.004 | 0.418 | no |
| qwen-qwen3-14b/controls/true/question_only.final.last | linear | 1.059 | -0.118 | 0.458 | no |
| qwen-qwen3-14b/controls/true/question_only.p25.last | linear | 1.080 | -0.141 | 0.475 | no |
| qwen-qwen3-14b/controls/true/question_only.p50.last | linear | 1.068 | -0.128 | 0.462 | no |
| qwen-qwen3-14b/controls/true/question_only.p75.last | linear | 1.062 | -0.121 | 0.488 | no |

## Random-subspace dimension

| Cell | Median d90 | Projection seeds reaching target |
|:--|--:|--:|
| qwen-qwen3-14b/cells/believes/block29.last | 512.0 | 5 |
| qwen-qwen3-14b/cells/believes/final.last | 512.0 | 5 |
| qwen-qwen3-14b/cells/believes/p25.last | 2048.0 | 5 |
| qwen-qwen3-14b/cells/believes/p50.last | 1024.0 | 5 |
| qwen-qwen3-14b/cells/believes/p75.last | 512.0 | 5 |
| qwen-qwen3-14b/cells/justified/block29.last | 1024.0 | 5 |
| qwen-qwen3-14b/cells/justified/final.last | 1024.0 | 5 |
| qwen-qwen3-14b/cells/justified/p25.last | 2048.0 | 5 |
| qwen-qwen3-14b/cells/justified/p50.last | 512.0 | 5 |
| qwen-qwen3-14b/cells/justified/p75.last | 512.0 | 5 |
| qwen-qwen3-14b/cells/knows/block29.last | 1024.0 | 5 |
| qwen-qwen3-14b/cells/knows/final.last | 2048.0 | 5 |
| qwen-qwen3-14b/cells/knows/p25.last | 2048.0 | 5 |
| qwen-qwen3-14b/cells/knows/p50.last | 1024.0 | 5 |
| qwen-qwen3-14b/cells/knows/p75.last | 1024.0 | 5 |
| qwen-qwen3-14b/cells/lucky_guessed/block29.last | 1024.0 | 5 |
| qwen-qwen3-14b/cells/lucky_guessed/final.last | 2048.0 | 5 |
| qwen-qwen3-14b/cells/lucky_guessed/p25.last | 512.0 | 5 |
| qwen-qwen3-14b/cells/lucky_guessed/p50.last | 2048.0 | 5 |
| qwen-qwen3-14b/cells/lucky_guessed/p75.last | 1024.0 | 5 |
| qwen-qwen3-14b/cells/true/block29.last | 512.0 | 5 |
| qwen-qwen3-14b/cells/true/final.last | 1024.0 | 5 |
| qwen-qwen3-14b/cells/true/p25.last | 2048.0 | 5 |
| qwen-qwen3-14b/cells/true/p50.last | 512.0 | 5 |
| qwen-qwen3-14b/cells/true/p75.last | 512.0 | 5 |

## Practical probe compression

| Cell | Tuning-selected bytes (median across seeds) | Selected test macro-F1 |
|:--|--:|--:|
| qwen-qwen3-14b/cells/believes/block29.last | 132648 | 0.988 |
| qwen-qwen3-14b/cells/believes/emb.last | 68108 | 0.381 |
| qwen-qwen3-14b/cells/believes/final.last | 89624 | 0.978 |
| qwen-qwen3-14b/cells/believes/p25.last | 132632 | 0.983 |
| qwen-qwen3-14b/cells/believes/p50.last | 132632 | 0.986 |
| qwen-qwen3-14b/cells/believes/p75.last | 89616 | 0.986 |
| qwen-qwen3-14b/cells/justified/block29.last | 132652 | 0.987 |
| qwen-qwen3-14b/cells/justified/emb.last | 68112 | 0.426 |
| qwen-qwen3-14b/cells/justified/final.last | 218676 | 0.980 |
| qwen-qwen3-14b/cells/justified/p25.last | 218668 | 0.980 |
| qwen-qwen3-14b/cells/justified/p50.last | 132636 | 0.983 |
| qwen-qwen3-14b/cells/justified/p75.last | 218668 | 0.987 |
| qwen-qwen3-14b/cells/knows/block29.last | 218828 | 0.978 |
| qwen-qwen3-14b/cells/knows/emb.last | 68096 | 0.436 |
| qwen-qwen3-14b/cells/knows/final.last | 218660 | 0.982 |
| qwen-qwen3-14b/cells/knows/p25.last | 218652 | 0.977 |
| qwen-qwen3-14b/cells/knows/p50.last | 218652 | 0.982 |
| qwen-qwen3-14b/cells/knows/p75.last | 218652 | 0.981 |
| qwen-qwen3-14b/cells/lucky_guessed/block29.last | 218700 | 0.984 |
| qwen-qwen3-14b/cells/lucky_guessed/emb.last | 68128 | 0.392 |
| qwen-qwen3-14b/cells/lucky_guessed/final.last | 218692 | 0.975 |
| qwen-qwen3-14b/cells/lucky_guessed/p25.last | 390748 | 0.967 |
| qwen-qwen3-14b/cells/lucky_guessed/p50.last | 218684 | 0.981 |
| qwen-qwen3-14b/cells/lucky_guessed/p75.last | 390748 | 0.984 |
| qwen-qwen3-14b/cells/true/block29.last | 132632 | 0.979 |
| qwen-qwen3-14b/cells/true/emb.last | 68092 | 0.418 |
| qwen-qwen3-14b/cells/true/final.last | 218656 | 0.969 |
| qwen-qwen3-14b/cells/true/p25.last | 218648 | 0.968 |
| qwen-qwen3-14b/cells/true/p50.last | 132616 | 0.978 |
| qwen-qwen3-14b/cells/true/p75.last | 132616 | 0.975 |

## LLC diagnostics and convergence

| Cell | Central LLC | Rejected | Diagnostic |
|:--|--:|:--|:--|
| qwen-qwen3-14b/llc_calibration/candidate00 | 0.031 | True | invalid_sensitivity_chains |
| qwen-qwen3-14b/llc_calibration/candidate01 | 0.009 | True | sensitivity_exceeds_10_percent |
| qwen-qwen3-14b/llc_calibration/candidate02 | 0.008 | True | sensitivity_exceeds_10_percent |
| qwen-qwen3-14b/llc_calibration/candidate03 | 0.009 | True | invalid_sensitivity_chains |
| qwen-qwen3-14b/llc_calibration/candidate04 | 0.003 | True | sensitivity_exceeds_10_percent |
| qwen-qwen3-14b/llc_calibration/candidate05 | 0.002 | True | sensitivity_exceeds_10_percent |
| qwen-qwen3-14b/llc_calibration/candidate06 | 0.003 | True | invalid_sensitivity_chains |
| qwen-qwen3-14b/llc_calibration/candidate07 | 0.001 | True | sensitivity_exceeds_10_percent |
| qwen-qwen3-14b/llc_calibration/candidate08 | 0.001 | True | sensitivity_exceeds_10_percent |
| qwen-qwen3-14b/llc_calibration/candidate09 | 0.001 | True | invalid_sensitivity_chains |
| qwen-qwen3-14b/llc_calibration/candidate10 | 0.000 | True | sensitivity_exceeds_10_percent |
| qwen-qwen3-14b/llc_calibration/candidate11 | 0.000 | True | sensitivity_exceeds_10_percent |

**No LLC estimate is reported.** All 12 declared calibration candidates were rejected: invalid_sensitivity_chains (4), sensitivity_exceeds_10_percent (8). Per-cell LLC was therefore never run, and the learning coefficient is reported as non-identifiable for this probe under the declared locality criteria rather than estimated at a setting that failed them.

### Cross-measure directional tests

_Confirmatory correlations are unavailable: no complete, accepted 15-cell paired-refit bootstrap result exists. Seed/chain resampling alone does not meet the preregistered scenario-uncertainty requirement._

## Interpretation guardrails

Per-predicate tables condition on that predicate's stable subset. Passing their quality gate does not establish cross-predicate comparability; cross-predicate inference requires the common stable cohort used by the paired-refit analysis.

Comparative claims are suppressed above when either stable class has fewer than the configured minimum or untouched-evaluation macro-F1 is below 0.75. MDL label codelength, constructive subspace dimension, serialized probe size, and LLC are reported as distinct measurements; agreement is an empirical result, not an assumption.
