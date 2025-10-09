# Pull Request Template

## Summary
- **Notebook / task / model / dataset:** <!-- e.g. Sentiment IMDB — DistilBERT -->
- **Fixes introduced or updated:** <!-- link to /fixes-and-tips entry -->
- **Docs touched:** <!-- README, docs/, etc. -->

## Validation
- [ ] CPU run end-to-end
- [ ] Metal (Apple Silicon) path tested
- [ ] CUDA path tested
- [ ] Measurements appended to `benchmarks/matrix.csv`
- [ ] Gotchas section updated with fix link

_Tick only the profiles you actually exercised._

## Metrics
- **New rows in `benchmarks/matrix.csv`:** <!-- paste CSV row IDs or summary -->
- **Peak RAM / VRAM:** <!-- values or TODO placeholders if pending -->
- **TTFB / throughput:** <!-- values or TODO placeholders -->

## Additional Notes
- **Related issues:** <!-- link -->
- **Follow-up tasks:** <!-- optional -->

Labels: `notebook`, `fix`, `benchmark`, `apple-silicon`, `gpu`, `good-first-issue`.
