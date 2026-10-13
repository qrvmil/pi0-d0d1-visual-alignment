# Authoritative progress ledger
- Request read and archived; pinned source cloned to /root/maniskill_myws.
- GPU inventory: A100-SXM4-80GB, driver580.159.03, maximum driver CUDA13.4. Actual runtime smoke pending. No datasets fetched.
- Teacher pinned revision9d0413465e8a91e67bbf2c1ad342815478d1b906.
- GitHub qrvmil and HF qrvmil authorized.
- Ruling: run on actual A100 and explicitly record hardware deviation; do not claim Blackwell validation. Cost if wrong: cannot prove portability to requested Blackwell until a separate real-hardware test.
- Ruling: conservative exact paired CI instead of degenerate percentile bootstrap; D0 Bonferroni protects two prespecified looks. Cost: less statistical power, more honest inconclusive outcomes.
