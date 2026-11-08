# Axial-attention channel estimator

This estimator predicts the full OFDM channel grid with axial attention. It is a
benchmark method for this repository, not a reproduction of one published
network.

Least-squares estimates on the active Kronecker pilots are arranged as a
real/imaginary image and bilinearly placed on the full subcarrier-by-symbol
grid. Each axial block then attends along OFDM symbols for every subcarrier,
and along subcarriers for every OFDM symbol. A zero-initialized output
convolution makes an untrained network match that bilinear LS image, so
training learns a residual. `log10` of the noise variance is an extra input
channel. One set of weights is shared across receive antennas and transmit
streams.

The detector still uses the benchmark LMMSE equalizer. Its error-variance input
is the bilinearly upsampled LS error variance times `--err-var-scale` (default
`1`). That scale is not learned.

## Train

From the repository root, with the `sionna-bench` environment active:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.axial.run_benchmark \
    --prepare-assets \
    --asset-root generated-assets \
    --smoke \
    --device cuda:0
```

`--smoke` keeps one SNR and one training epoch, on the full UMi grid. Drop
`--smoke` to train across the benchmark SNR grid. The objective is NMSE.
Every epoch appends `history.jsonl` and writes `epoch-XXXX.pt`, `latest.pt`,
and `best.pt` in a sibling `<artifact>-run` directory. Add `--resume` to
continue that run.

## Benchmark

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.axial.run_benchmark \
    --asset-root generated-assets \
    --smoke \
    --device cuda:0
```

The evaluation command must use the same architecture flags, SNR mode, and
training budget as preparation, because those values select the asset
directory. Results are written under
`outputs/mimo_ofdm_chest/axial/smoke/3gpp-umi/` for smoke runs and
`outputs/mimo_ofdm_chest/axial/3gpp-umi/` otherwise. The NPZ contains NMSE,
BER, and BLER.

Architecture flags are `--d-model` (32), `--num-heads` (4), `--num-blocks` (2),
and `--ffn-expansion` (2).
