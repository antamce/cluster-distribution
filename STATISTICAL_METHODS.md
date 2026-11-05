# Synpo clustering statistics specification

This document defines the post-clustering statistical methods implemented by
Synpo. It is an application specification, not a claim that one method is
appropriate for every experiment. Each exported clustering run also contains a
`*_statistical_methods.md` file recording the exact settings and warnings used
for that run.

## Analysis population and experimental groups

Statistics use the valid spines included in the saved clustering run. The user
selects statistical groups independently of both the groups used to fit the
clusters and the groups merely shown on scatter plots. Selecting one group
produces descriptive summaries and suppresses between-group tests.

The analysis level is selectable:

- **Spine:** individual spines are observations. Tests are explicitly labeled
  as assuming independent spines and may be pseudoreplicated when many spines
  come from one biological unit.
- **Specimen:** values are aggregated within each Synpo specimen and specimens
  receive equal weight, regardless of spine count.
- **Neuron / subject:** specimens with the same manually assigned neuron or
  subject ID are aggregated and neurons receive equal weight. Pairwise
  inferential comparisons use Monte Carlo permutation tests.

Experimental groups are assumed independent. Synpo does not currently fit
paired, repeated-measures, nested-animal, or mixed-effects models.

## Cluster-composition analysis

For each experimental group, Synpo reports the number and percentage of spines
in every morphology cluster.

### Omnibus Pearson chi-square

A Pearson chi-square test is calculated from the pooled experimental-group by
cluster count table without continuity correction. Synpo reports the test
statistic, degrees of freedom, raw p-value, and Cramer's V. Expected cell counts
below 5 are counted and the smallest expected count is reported. Sparse counts
produce an explicit warning; Synpo does not silently replace the requested
test. This test treats spines as independent.

### Agresti-Caffo pairwise differences

Every experimental group is compared with every other group separately for
every cluster. If `x1/n1` and `x2/n2` are the observed cluster proportions,
Synpo adds one success and one failure to each group:

`p1* = (x1 + 1) / (n1 + 2)` and `p2* = (x2 + 1) / (n2 + 2)`.

The adjusted difference is `p1* - p2*`; its standard error is the square root
of `p1*(1-p1*)/(n1+2) + p2*(1-p2*)/(n2+2)`. Synpo reports the observed
difference, adjusted difference, two-sided normal-approximation p-value, and a
user-selected 90–99% confidence interval. Direction is always first selected
group minus second selected group. These comparisons treat spines as
independent.

### Equal-weight unit permutations

In specimen or neuron mode, Synpo additionally calculates each cluster's spine
fraction within every unit. For every group pair and cluster, it compares the
mean unit fractions. Group labels are shuffled between intact units, preserving
group sizes. The two-sided Monte Carlo p-value uses the plus-one correction:
`(extreme permutations + 1) / (requested permutations + 1)`. The default is
10,000 repetitions with seed 42.

## Continuous morphology, puncta, and calcium metrics

Metrics are summarized separately within every morphology cluster and
experimental group. Spine mode summarizes individual measurements; specimen
and neuron modes first average measurements within each unit.

Every group pair can be compared with one of these methods:

- Welch independent-samples t-test;
- two-sided Mann-Whitney U test;
- two-sided Monte Carlo permutation test of the difference in means.

Neuron-level comparisons always use the Monte Carlo permutation test, even if
another continuous test remains selected in the interface. Synpo reports raw
group means, their signed difference, sample sizes, the test statistic, raw
p-value, and adjusted q-value.

Any metric used to construct the clustering run is exported as descriptive
only. Testing a cluster-defining metric against the resulting cluster labels is
circular and can exaggerate apparent evidence. Metrics not used as clustering
inputs remain eligible for inferential comparisons.

Protein puncta count, total volume, mean volume, volume ratio, and position are
continuous metrics. Protein-puncta presence is binary and is compared within
each morphology cluster using the same pairwise Agresti-Caffo method, with an
explicit spine-independence label. In specimen or neuron mode, Synpo also
compares equal-weight unit-level positive fractions by Monte Carlo permutation.

## Multiple comparisons

Synpo reports raw p-values and Benjamini-Hochberg false-discovery-rate q-values.
Correction is performed separately for these prespecified families:

1. pairwise cluster-composition Agresti-Caffo comparisons;
2. unit-level cluster-fraction permutations;
3. continuous-metric pairwise comparisons;
4. protein-puncta-presence Agresti-Caffo comparisons.
5. unit-level protein-puncta-presence permutations.

Sparse clusters remain in all outputs. They are never silently merged or
discarded by the statistics stage.

## Plots and export

The statistics panel can graph every performed comparison family. In addition
to 100% stacked/grouped composition bars and per-group or pooled pies, it offers
a group-by-cluster Pearson-residual heatmap with observed and expected counts,
Agresti-Caffo confidence-interval forests, equal-weight unit cluster-fraction
distributions, within-cluster continuous-metric distributions, pooled
protein-puncta-presence percentages, and equal-weight unit protein-positive
fraction distributions. Metric and cluster selectors choose the detailed graph;
pairwise p-values, family-specific q-values, and effects are shown in a separate
side panel so annotations do not obscure observations. Pie labels can show
percentages, counts, both, or neither. Cluster colors follow the saved named
clustering run.

The full analysis export contains the exact statistical settings, composition
summaries, omnibus tests, pairwise Agresti-Caffo rows, unit fractions,
permutation tests, continuous descriptives/comparisons, protein-presence tests,
warnings, high-resolution plot files for every applicable comparison, and a
run-specific methods Markdown file.
The report PDF appends the same methods plus every statistical result row in
paginated machine-precise text; the workbook/CSV tables remain the easiest format
for downstream computation.

The experimental-group visibility selector on clustering scatter plots is only
a display filter. It does not change fitted clusters or statistical samples.

## Method references

- Agresti A, Caffo B. Simple and effective confidence intervals for proportions
  and differences of proportions result from adding two successes and two
  failures. *The American Statistician*. 2000;54:280–288.
  <https://users.stat.ufl.edu/~aa/articles/agresti_caffo_2000.pdf>
- Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and
  powerful approach to multiple testing. *Journal of the Royal Statistical
  Society, Series B*. 1995;57:289–300.
  <https://doi.org/10.1111/j.2517-6161.1995.tb02031.x>
