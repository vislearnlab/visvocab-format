Pair similarity across visual formats
================

How alike do the two images of each hard pair look to a vision model, as
the stimuli move from photo toward abstraction (photo → cut-out →
illustration → line drawing → doodle)? Embeddings come from
`pipeline/12_embeddings.py`: DINOv3 ViT-B/16 (LVD-1689M, pooled CLS) and
CLIP ViT-B/32 (the model used in CHS). Similarity is cosine between the
pair’s two images. Formats that have not been generated yet are simply
absent.

## Raw similarity rises with abstraction, but so does everything else

``` python
for m, name in MODELS.items():
    d = T[m][T[m].version.isin(fmts)]
    print(name); print(d.groupby('version')[['sim', 'baseline', 'excess', 'rank']].mean().reindex(fmts).round(3).to_string(), '\n')
```

    DINOv3 ViT-B/16
                           sim  baseline  excess   rank
    version                                            
    CHS photo            0.127     0.028   0.098  0.834
    photo                0.129     0.028   0.101  0.800
    cutout               0.178     0.075   0.103  0.838
    illustration         0.233     0.105   0.128  0.864
    illustration_cutout  0.238     0.112   0.126  0.830
    line                 0.270     0.170   0.100  0.793
    line_cutout          0.302     0.166   0.136  0.804
    doodle               0.335     0.164   0.170  0.860
    doodle_cutout        0.303     0.166   0.137  0.817 

    CLIP ViT-B/32
                           sim  baseline  excess   rank
    version                                            
    CHS photo            0.735     0.621   0.114  0.899
    photo                0.712     0.612   0.100  0.879
    cutout               0.759     0.674   0.085  0.916
    illustration         0.744     0.661   0.083  0.891
    illustration_cutout  0.766     0.685   0.081  0.912
    line                 0.831     0.780   0.052  0.770
    line_cutout          0.819     0.766   0.053  0.790
    doodle               0.823     0.757   0.066  0.867
    doodle_cutout        0.819     0.760   0.059  0.828 

Raw pair similarity goes up as detail is removed. The same-format
baseline goes up as much or more: once photo texture, lighting and scene
are gone, *any* two images in a style resemble each other. Raw cosine is
therefore not comparable across formats. On an LWL trial the target and
distractor are always shown in the same format, so the model quantity
that matters is the pair’s similarity *relative to its format*: the
excess, or the partner’s rank.

``` python
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, (m, name) in zip(axes, MODELS.items()):
    g = T[m].groupby('version')[['sim', 'baseline', 'excess']].agg(['mean', 'sem']).reindex(fmts)
    for col, c in [('sim', 'k'), ('baseline', 'tab:grey'), ('excess', 'tab:red')]:
        ax.errorbar(range(len(fmts)), g[(col, 'mean')].values, yerr=1.96 * g[(col, 'sem')].values, marker='o', color=c, capsize=3,
                    label={'sim': 'pair similarity', 'baseline': 'same-format baseline', 'excess': 'pair − baseline'}[col])
    ax.set_xticks(range(len(fmts))); ax.set_xticklabels(fmts, rotation=25, ha='right'); ax.set_title(name); ax.grid(axis='y', alpha=.3)
axes[0].set_ylabel('cosine'); axes[0].legend(frameon=False, fontsize=9); fig.tight_layout()
```

![](02_format_similarity_files/figure-commonmark/cell-4-output-1.png)

## Per pair: which pairs does abstraction separate, and which does it merge?

Excess similarity per pair. If abstraction removed exactly the details
that make a pair confusable, the pair’s excess should fall; if it
removed the details that tell the two apart, it should rise.

``` python
for m, name in MODELS.items():
    w = T[m].pivot(index='pair', columns='version', values='excess')[[f for f in FINAL if f in fmts]]
    print(name); print(w.sort_values('photo').round(3).to_string(), '\n')
```

    DINOv3 ViT-B/16
    version            photo  cutout  illustration  illustration_cutout   line  line_cutout  doodle  doodle_cutout
    pair                                                                                                          
    duck-chicken      -0.067  -0.024         0.076                0.097  0.055        0.068   0.147          0.120
    crow-penguin      -0.053  -0.044         0.017               -0.010  0.191        0.073   0.284          0.087
    truck-car         -0.013  -0.006         0.022               -0.027 -0.005       -0.065   0.083          0.093
    turkey-goat        0.021   0.060         0.022                0.017 -0.023       -0.032   0.161         -0.020
    butterfly-ladybug  0.034   0.062         0.114                0.053 -0.066        0.036   0.034          0.002
    acorn-coconut      0.054   0.101         0.236                0.276  0.208        0.210   0.515          0.449
    camel-elephant     0.074   0.046         0.060                0.033  0.031        0.035   0.068          0.128
    cow-horse          0.098   0.100         0.130                0.141  0.207        0.307   0.161          0.223
    balloon-kite       0.115   0.114         0.251                0.189  0.175        0.405   0.217          0.156
    turtle-frog        0.140   0.133         0.040                0.037  0.061        0.142   0.044          0.131
    bulldozer-tractor  0.143   0.143         0.218                0.279  0.166        0.197   0.266          0.206
    squirrel-monkey    0.161   0.159         0.090                0.105  0.110        0.153   0.062          0.118
    snail-worm         0.180   0.095         0.054                0.089 -0.002        0.029  -0.059         -0.044
    rabbit-hamster     0.235   0.236         0.259                0.246  0.210        0.163   0.387          0.357
    lizard-snake       0.241   0.253         0.037                0.090  0.082        0.122   0.042         -0.065
    cloud-tree         0.253   0.220         0.424                0.402  0.201        0.336   0.314          0.247 

    CLIP ViT-B/32
    version            photo  cutout  illustration  illustration_cutout   line  line_cutout  doodle  doodle_cutout
    pair                                                                                                          
    truck-car         -0.007   0.044         0.085                0.062  0.058        0.036   0.073          0.037
    duck-chicken       0.034   0.057         0.052                0.027  0.030        0.071   0.066          0.071
    crow-penguin       0.059   0.095         0.077                0.098  0.106        0.009   0.112          0.078
    turkey-goat        0.059   0.000         0.037                0.007  0.001       -0.013   0.025         -0.027
    camel-elephant     0.062   0.064         0.088                0.079  0.053        0.074   0.063          0.078
    acorn-coconut      0.068   0.077         0.049                0.094  0.070        0.058   0.174          0.148
    rabbit-hamster     0.074   0.104         0.117                0.112  0.100        0.086   0.073          0.096
    cow-horse          0.099   0.072         0.095                0.079  0.037        0.120   0.036          0.095
    squirrel-monkey    0.100   0.080         0.036                0.053  0.095        0.057   0.037          0.028
    balloon-kite       0.125   0.105         0.134                0.145  0.049        0.086   0.085          0.109
    snail-worm         0.125   0.075         0.009                0.075  0.037        0.020   0.047          0.016
    turtle-frog        0.130   0.110         0.075                0.079  0.018        0.053   0.020          0.067
    butterfly-ladybug  0.144   0.072         0.080                0.043  0.013       -0.009   0.031         -0.008
    bulldozer-tractor  0.151   0.128         0.180                0.132  0.119        0.093   0.107          0.109
    cloud-tree         0.183   0.127         0.144                0.125  0.042        0.110   0.103          0.063
    lizard-snake       0.193   0.145         0.077                0.086 -0.004        0.006  -0.001         -0.009 

``` python
fin = [f for f in FINAL if f in fmts]
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
for ax, (m, name) in zip(axes, MODELS.items()):
    w = T[m].pivot(index='pair', columns='version', values='excess')[fin]
    for i, (p, row) in enumerate(w.iterrows()):
        c = plt.cm.tab20(i % 20); ax.plot(range(len(fin)), row.values, marker='o', ms=4, color=c, lw=1.2)
        ax.text(len(fin) - 1 + .08, row.values[-1], p, fontsize=7, color=c, va='center')
    ax.axhline(0, color='k', lw=.6); ax.set_xticks(range(len(fin))); ax.set_xticklabels(fin, rotation=25, ha='right')
    ax.set_xlim(-.3, len(fin) + .8); ax.set_title(name); ax.grid(axis='y', alpha=.3)
axes[0].set_ylabel('pair similarity − same-format baseline'); fig.tight_layout()
```

![](02_format_similarity_files/figure-commonmark/cell-6-output-1.png)

How much of the item ordering survives each format change? Spearman
correlation of per-pair excess across the 16 pairs:

``` python
for m, name in MODELS.items():
    w = T[m].pivot(index='pair', columns='version', values='excess')[fin]
    print(name); print(w.corr(method='spearman').round(2).to_string(), '\n')
```

    DINOv3 ViT-B/16

    version              photo  cutout  illustration  illustration_cutout  line  line_cutout  doodle  doodle_cutout
    version                                                                                                        
    photo                 1.00    0.90          0.46                 0.57  0.36         0.46   -0.05           0.19
    cutout                0.90    1.00          0.55                 0.67  0.48         0.61    0.10           0.33
    illustration          0.46    0.55          1.00                 0.91  0.60         0.78    0.48           0.78
    illustration_cutout   0.57    0.67          0.91                 1.00  0.69         0.84    0.50           0.71
    line                  0.36    0.48          0.60                 0.69  1.00         0.83    0.74           0.76
    line_cutout           0.46    0.61          0.78                 0.84  0.83         1.00    0.55           0.75
    doodle               -0.05    0.10          0.48                 0.50  0.74         0.55    1.00           0.72
    doodle_cutout         0.19    0.33          0.78                 0.71  0.76         0.75    0.72           1.00 

    CLIP ViT-B/32
    version              photo  cutout  illustration  illustration_cutout  line  line_cutout  doodle  doodle_cutout
    version                                                                                                        
    photo                 1.00    0.75          0.29                 0.39 -0.25         0.13   -0.26          -0.14
    cutout                0.75    1.00          0.40                 0.78  0.20         0.30    0.14           0.25
    illustration          0.29    0.40          1.00                 0.68  0.31         0.66    0.34           0.49
    illustration_cutout   0.39    0.78          0.68                 1.00  0.51         0.59    0.57           0.67
    line                 -0.25    0.20          0.31                 0.51  1.00         0.45    0.78           0.67
    line_cutout           0.13    0.30          0.66                 0.59  0.45         1.00    0.44           0.73
    doodle               -0.26    0.14          0.34                 0.57  0.78         0.44    1.00           0.69
    doodle_cutout        -0.14    0.25          0.49                 0.67  0.67         0.73    0.69           1.00 

## A recognition-side model measure: CLIP word → image 2AFC

Embedding similarity is not recognition. Closer to the task: give CLIP
the spoken word (“a photo of a {word}” / “a drawing of a {word}”) and
ask whether it prefers the target over its partner, in each format. In
CHS photos CLIP was at ceiling, so this mainly shows whether any format
pulls it off ceiling.

``` python
import torch, open_clip
clip, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai'); clip.eval(); tok = open_clip.get_tokenizer('ViT-B-32')
TEMPL = ['a photo of a {}.', 'a drawing of a {}.', 'an illustration of a {}.', 'a picture of a {}.']
label = lambda w: 'chicken' if w.startswith('chicken') else w
with torch.no_grad():
    TX = {}
    for w in W:
        t = clip.encode_text(tok([s.format(label(w)) for s in TEMPL])); t = t / t.norm(dim=-1, keepdim=True)
        t = t.mean(0); TX[w] = (t / t.norm()).numpy()
partner = {**dict(zip(P.word_a, P.word_b)), **dict(zip(P.word_b, P.word_a))}
rows = []
for v in fmts:
    for w in W:
        if (v, w) not in E['clip'] or (v, partner[w]) not in E['clip']: continue
        lo = 100 * np.array([E['clip'][(v, w)] @ TX[w], E['clip'][(v, w)] @ TX[partner[w]]])
        rows.append(dict(version=v, word=w, p_correct=np.exp(lo[0]) / np.exp(lo).sum()))
rec = pd.DataFrame(rows)
print(rec.groupby('version').p_correct.agg(['mean', 'min', lambda s: (s < .5).sum()]).rename(columns={'<lambda_0>': 'n wrong'}).reindex(fmts).round(3).to_string())
```

                          mean    min  n wrong
    version                                   
    CHS photo            0.994  0.871        0
    photo                0.991  0.821        0
    cutout               0.993  0.867        0
    illustration         0.995  0.967        0
    illustration_cutout  0.995  0.972        0
    line                 0.891  0.443        1
    line_cutout          0.934  0.639        0
    doodle               0.926  0.200        1
    doodle_cutout        0.916  0.189        1

``` python
print(rec.pivot(index='word', columns='version', values='p_correct')[fmts].round(2).sort_values('photo').head(12).to_string())
```

    version    CHS photo  photo  cutout  illustration  illustration_cutout  line  line_cutout  doodle  doodle_cutout
    word                                                                                                            
    lizard          0.87   0.82    0.99          0.97                 0.98  1.00         0.99    0.99           0.99
    frog            0.99   0.93    0.87          0.99                 0.99  0.89         0.67    1.00           0.91
    acorn           1.00   0.99    0.97          1.00                 0.98  0.76         0.84    0.62           0.19
    butterfly       0.99   0.99    1.00          1.00                 1.00  1.00         1.00    1.00           1.00
    snake           0.99   0.99    0.99          0.99                 0.99  0.44         0.78    0.99           0.97
    turkey          1.00   1.00    1.00          1.00                 1.00  0.98         1.00    1.00           1.00
    truck           0.99   1.00    0.99          1.00                 1.00  0.80         0.98    0.99           1.00
    tree            1.00   1.00    1.00          1.00                 1.00  0.90         0.98    1.00           1.00
    tractor         0.99   1.00    1.00          1.00                 1.00  0.93         0.84    0.99           0.99
    squirrel        1.00   1.00    1.00          1.00                 1.00  0.91         1.00    1.00           1.00
    snail           1.00   1.00    1.00          1.00                 1.00  0.99         0.96    0.99           0.91
    rabbit          1.00   1.00    0.99          1.00                 0.99  1.00         1.00    1.00           1.00

## Illustration rounds

Words whose illustration was regenerated (round 0 = first pass; final =
current `stimuli/images/illustration`). Only pairs containing a
regenerated word can change.

``` python
for m, name in MODELS.items():
    d = T[m]; rr = sorted(v for v in d.version.unique() if 'round' in v)
    if not rr: continue
    cols = [v for v in rr if v.startswith('illustration round')] + ['illustration'] + \
           [v for v in rr if v.startswith('illustration_cutout round')] + ['illustration_cutout']
    w = d.pivot(index='pair', columns='version', values='excess')[cols]
    print(name); print(w[(w.max(axis=1) - w.min(axis=1)) > 1e-4].round(3).to_string(), '\n')
```

    DINOv3 ViT-B/16
    version            illustration round 0  illustration round 1  illustration  illustration_cutout round 0  illustration_cutout round 1  illustration_cutout
    pair                                                                                                                                                      
    acorn-coconut                     0.264                 0.264         0.236                        0.254                        0.255                0.276
    balloon-kite                      0.444                 0.249         0.251                        0.302                        0.190                0.189
    bulldozer-tractor                 0.217                 0.217         0.218                        0.279                        0.279                0.279
    butterfly-ladybug                 0.116                 0.117         0.114                        0.052                        0.053                0.053
    camel-elephant                    0.066                 0.070         0.060                        0.004                        0.005                0.033
    cloud-tree                        0.344                 0.349         0.424                        0.407                        0.405                0.402
    cow-horse                         0.169                 0.172         0.130                        0.165                        0.165                0.141
    crow-penguin                      0.001                 0.001         0.017                        0.001                        0.000               -0.010
    duck-chicken                      0.073                 0.073         0.076                        0.097                        0.096                0.097
    lizard-snake                      0.088                 0.116         0.037                        0.152                        0.149                0.090
    rabbit-hamster                    0.237                 0.240         0.259                        0.190                        0.191                0.246
    snail-worm                        0.051                 0.053         0.054                        0.053                        0.052                0.089
    squirrel-monkey                   0.054                 0.056         0.090                        0.082                        0.083                0.105
    truck-car                         0.017                 0.020         0.022                       -0.028                       -0.027               -0.027
    turkey-goat                       0.062                 0.066         0.022                        0.001                        0.003                0.017
    turtle-frog                       0.056                 0.039         0.040                        0.052                        0.035                0.037 

    CLIP ViT-B/32
    version            illustration round 0  illustration round 1  illustration  illustration_cutout round 0  illustration_cutout round 1  illustration_cutout
    pair                                                                                                                                                      
    acorn-coconut                     0.066                 0.071         0.049                        0.068                        0.071                0.094
    balloon-kite                      0.128                 0.129         0.134                        0.160                        0.141                0.145
    bulldozer-tractor                 0.172                 0.177         0.180                        0.129                        0.131                0.132
    butterfly-ladybug                 0.073                 0.077         0.080                        0.038                        0.041                0.043
    camel-elephant                    0.088                 0.093         0.088                        0.061                        0.062                0.079
    cloud-tree                        0.112                 0.118         0.144                        0.118                        0.122                0.125
    cow-horse                         0.095                 0.099         0.095                        0.079                        0.082                0.079
    crow-penguin                      0.088                 0.094         0.077                        0.068                        0.071                0.098
    duck-chicken                      0.050                 0.053         0.052                        0.025                        0.027                0.027
    lizard-snake                      0.127                 0.143         0.077                        0.078                        0.072                0.086
    rabbit-hamster                    0.112                 0.116         0.117                        0.105                        0.108                0.112
    snail-worm                        0.053                 0.057         0.009                        0.061                        0.062                0.075
    squirrel-monkey                   0.050                 0.054         0.036                        0.036                        0.038                0.053
    truck-car                         0.074                 0.079         0.085                        0.057                        0.060                0.062
    turkey-goat                       0.038                 0.043         0.037                        0.018                        0.021                0.007
    turtle-frog                       0.077                 0.075         0.075                        0.039                        0.077                0.079 

## What this means for the format hypothesis

- **Raw similarity rises steadily with abstraction** (DINOv3 .13 photo →
  .34 doodle; CLIP .71 → .82–.83), but almost all of the rise is the
  same-format baseline: in a sparse style every image looks like every
  other.
- **Relative to its format, the pair-specific similarity barely moves,
  and the two models disagree on direction.** DINOv3 excess goes .10
  (photo) → .13 (illustration) → .10–.17 (line, doodle); CLIP goes .10 →
  .08 → .05–.07.
- **Model recognition does fall for the most abstract formats.** CLIP
  word → image 2AFC is at ceiling for photos, cut-outs and illustrations
  (≥ .99) but drops for line drawings and doodles (.89–.93 mean, with
  individual words near or below chance). Some of that is stimulus
  quality (the acorn doodle looks like garlic; the snake line drawing is
  buried in background scribble), so the per-word failures should be
  re-checked after the redos.
- **That gives a clean prediction to set against children.** The models
  say abstraction leaves pairs about as separable (embeddings) or makes
  the words harder to recognise (CLIP 2AFC). If toddlers instead
  recognise words as well or better in illustrations and line drawings,
  the details removed by abstraction (texture, lighting, background)
  were noise for them rather than cues: a model–child dissociation.
- **The item-level test is sharper.** Some pairs’ excess moves in
  opposite directions across formats (lizard–snake drops, acorn–coconut
  and cow–horse rise). A format × model-score interaction on
  baseline-corrected looking tests whether item-specific model changes
  predict children’s item-specific changes; a format main effect with a
  null interaction is the dissociation.
- **Caveats.** Cosine on a pooled embedding is one readout of a model.
  Illustrations differ from photos in colour as well as detail, so line
  drawings and doodles are the cleaner “fewer details” contrast, but
  they are also the noisiest stimuli right now (see
  `04_stylized_stimuli`). With 16 pairs the item-level interaction has
  little power; the format main effect is the realistic primary test.
