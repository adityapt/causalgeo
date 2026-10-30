# causalgeo

**Concurrent geo holdouts for bundle lift and calibrated channel responses**

[![tests](https://github.com/adityapt/causalgeo/actions/workflows/check.yml/badge.svg)](https://github.com/adityapt/causalgeo/actions/workflows/check.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23608831.svg)](https://doi.org/10.5281/zenodo.23608831)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.txt)

A geo holdout estimates the lift of a bundle. It does not estimate how many sales each channel inside that bundle was worth. causalgeo reports the bundle lift, uses impression paths for a model-assisted channel split when the cells can separate the channels, and leaves a channel blank when the design cannot tell the channels apart.

## Key features

### Holdout

- **Concurrent cells.** Several bundles can go dark in the same weeks. Each cell has its own control markets.
- **An assignment you already have.** `design_from_assignment` takes a cell map. `design_concurrent_test` assigns markets inside groups you supply. The package does not search for markets or power.
- **The published number is the holdout.** A channel left on can still deliver differently in the dark markets. That shift is recorded. It changes the published cell total only if you ask for the refit.

### Channel split

- **Model-assisted, and labeled as such.** Channel responses come from impression history, pulled toward the measured bundle lifts. They are not channel-level holdout estimates.
- **A channel stays blank** when the flight never turns it off, when two impression paths are copies, or when the design cannot separate it.
- **The gap stays visible.** The bundle table shows the measured lift next to the lift implied by the channel responses.

### One bundle

One bundled test is one equation. Eight channels in that bundle are not eight experiments.

- **With an external mix-model prior.** `bayesian_channel_allocation` updates that prior with the measured bundle. It reports how many channel directions the experiment actually tested. One bundle tests one direction.
- **With no prior.** `anchored_channel_allocation` divides the measured total by a sales-per-impression weight you supply. The pieces add back to the holdout. The package does not calculate the weight.

### Single-cell lift

`analyze()` is the one-cell path: time-based regression by default, or synthetic control, synthetic difference-in-differences, or clustered two-way fixed effects. It returns lift, percent lift, and incremental cost per dollar when spend is present. It does not split channels.

## Quick start

### Installation

The release to install is v0.2.2. It is not on PyPI yet.

```bash
pip install "causalgeo @ git+https://github.com/adityapt/causalgeo.git@v0.2.2"
```

To work on the repository:

```bash
git clone https://github.com/adityapt/causalgeo.git
cd causalgeo
pip install -e ".[dev,paper]"
```

v0.2.0 remains the October archive. It does not include the one-bundle allocation.

### Run the example

This builds a small panel. No download. Search is an exact copy of Facebook, so the fit withholds both slopes and reports Display.

```bash
python examples/concurrent_holdout.py
```

### Basic usage

```python
from causalgeo import design_concurrent_test, disentangle

design = design_concurrent_test(
    markets,  # columns: geo, stratum
    channels=["Facebook", "TikTok", "Search"],
    bundles=[
        ["Facebook", "TikTok"],
        ["Facebook", "Search"],
        ["TikTok", "Search"],
    ],
    control_share=0.5,
)
result = disentangle(
    design,
    outcomes,     # columns: geo, date, sales
    impressions,  # columns: geo, date, channel, impressions
    outcome="sales",
    treatment_start="2026-08-03",
)

result.channels   # one row per channel; blank when the design cannot separate it
result.bundles    # measured bundle lift next to the lift implied by the channel responses
result.exposure   # still-on impression shift, and whether it was applied
```

`result.channels` includes a `reading` column. A reported slope says it is model-assisted. A withheld slope says why.

## One bundle

### No mix-model prior

You supply the sales-per-impression weights. Equal weights split the bundle by the impressions the holdout removed.

```python
from causalgeo import anchored_channel_allocation, prevented_effective_dose

dose = prevented_effective_dose(
    impressions,
    design,
    treatment_start="2026-08-03",
)
allocation = anchored_channel_allocation(
    result.cells[["cell", "coef", "std_err"]],
    dose,
    potency={"Search": 1.0, "Social": 1.0, "Video": 1.0},
)
```

`allocation` sums to the measured cell effect. `ci_lower` and `ci_upper` scale the experimental interval by the chosen share. `assumption_lower` and `assumption_upper` move only the potency bounds you pass in.

```bash
python examples/anchored_allocation.py
```

That script splits a bundle of 1,000 into 400, 400, and 200.

### With a mix-model prior

The prior is an input. The holdout does not discover it.

```python
from causalgeo import bayesian_channel_allocation, bundle_impression_contrasts

withheld = bundle_impression_contrasts(
    impressions,
    design,
    treatment_start="2026-08-03",
)
allocation = bayesian_channel_allocation(
    prior_mean=mmm_slopes,
    prior_covariance=mmm_covariance,
    bundle_effects=result.cells[["cell", "coef", "std_err"]],
    withheld=withheld,
)
```

`tested_rank` is how many independent directions the cells measured. `remaining_dimensions` is how many stay prior-driven. One eight-channel bundle returns 1 and 7.

```bash
python examples/bayesian_bundle.py
```

## Running the other examples

Clone the repository first. The `examples/` folder is not part of an installed wheel.

```bash
python examples/public_allocation_benchmark.py
python examples/meridian_conversions.py
python examples/prop99_measure.py
python examples/prop99_grid_identification.py
```

`public_allocation_benchmark.py` uses Google's public Meridian file `geo_all_channels.csv`. The file supplies the impression paths. Sales are constructed from known slopes, because the file has no channel-level holdout result. `meridian_conversions.py` runs the same split on Meridian's conversions column. That run has no known channel responses. It only shows the software on that file.

`prop99_measure.py` reproduces the published Proposition 99 result, about −15.6 packs per capita. `prop99_grid_identification.py` shows a design that can separate two policies, and the same design after both policies are collapsed into one indicator.

## What a single cell returns

```python
from causalgeo import AssignmentMap, GeoExperiment, Intervention

assignment = AssignmentMap.from_frame(
    assign_df,  # columns: location, role
    universe=kpi["location"].unique(),
)
exp = GeoExperiment(
    kpi,  # location, date, conversions, and spend when you want cost per dollar
    assignment,
    treatment_start="2026-10-01",
    treatment_end="2026-10-31",
    spend="spend",
)
report = exp.analyze(Intervention.GO_DARK)
report.lift, report.percent_lift, report.icpd
```

Go-dark turns existing media off. Holdback keeps a new activity at zero in control. Heavy-up adds delivery in treatment and leaves control on business as usual. A comparison of two spend arms with no business-as-usual markets raises `MissingControlError`. Cost per dollar without a spend series raises `MissingSpendError`.

Use GeoLift or Meridian GeoX when you are designing a test and want market selection, power, and budget search. causalgeo analyzes an assignment you already have.

| | GeoLift | Meridian GeoX | causalgeo |
|---|---|---|---|
| Starting point | Pick test markets | Search designs | Assignment you were handed |
| Counterfactual | Synthetic control | Time-based regression | Time-based regression by default; synthetic control or clustered two-way fixed effects |
| Channel split | No | No | Model-assisted, blank when the cells cannot separate the channels |
| Language | R | Python | Python |

## Benchmarks

Public Meridian geo panel, five channels dark together, sales built from known slopes. Forty markets, 156 weeks, flight from 2023-10-02. Script: `examples/public_allocation_benchmark.py`.

| Check | Result |
|---|---|
| Bundle lift | −1,815 (standard error 163) against a planted −1,816 |
| Informed split | Channel shares match the planted shares. The check uses the same sales-per-impression weights that built the sales. |
| Equal-weight split | Channel2 receives 7.8% against a planted 2.4%. Channel1 receives 12.7% against a planted 19.2%. |

On the manuscript's three-cell series, the straight-line responses are recovered within 0.21%. That series is the response the package fits. It is a check of the software, not evidence that a campaign worked. Rebuild it with:

```bash
cd paper
PYTHONPATH=../src python3.11 reproduce.py
```

`python3.11 reproduce.py --fast` checks that the scripts finish. It does not check the published numbers. The script sets a Matplotlib cache directory when `MPLCONFIGDIR` is unset.

## Project structure

```
causalgeo/
├── pyproject.toml
├── README.md
├── LICENSE.txt
├── CHANGELOG.md
├── CITATION.cff
├── CONTRIBUTING.md
├── Makefile
├── src/causalgeo/          # the package
│   ├── design.py           # cell assignment and what the flight can separate
│   ├── calibrated.py       # bundle lift and the model-assisted channel split
│   ├── allocation.py       # divide one measured bundle by a weight you supply
│   ├── bayesian.py         # update an external mix-model prior with the bundle
│   └── experiment.py       # single-cell lift
├── examples/               # scripts you can run
├── tests/                  # 81 tests
└── paper/                  # SoftwareX manuscript and reproduce.py
```

## Development

Python 3.10 or newer. Runtime dependencies are pandas, numpy, scipy, and statsmodels. The manuscript figures also use matplotlib.

```bash
pip install -e ".[dev,paper]"
python -m pytest
make check
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## License

MIT. See [LICENSE.txt](LICENSE.txt).

## Citation

The SoftwareX manuscript is not yet published. Cite the v0.2.2 release: [github.com/adityapt/causalgeo/releases/tag/v0.2.2](https://github.com/adityapt/causalgeo/releases/tag/v0.2.2).

The Zenodo concept DOI points at the latest archive: [10.5281/zenodo.23608831](https://doi.org/10.5281/zenodo.23608831). The v0.2.0 archive remains [10.5281/zenodo.23608832](https://doi.org/10.5281/zenodo.23608832). Zenodo will mint a separate DOI for v0.2.2 when that release is published.

GitHub's **Cite this repository** button reads [CITATION.cff](CITATION.cff).

## Support

Questions: [puttaparthy.aditya@gmail.com](mailto:puttaparthy.aditya@gmail.com)
