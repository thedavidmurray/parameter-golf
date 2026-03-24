This record applies the LeakyReLU(0.5)² activation swap on top of the Naive Baseline configuration.

## Change

One-line change in the MLP forward pass:

```python
# Before (relu^2)
x = torch.relu(self.fc(x))
return self.proj(x.square())

# After (leaky_relu^2)
x = F.leaky_relu(self.fc(x), negative_slope=0.5)
return self.proj(x.square())
```

LeakyReLU(0.5)² allows small negative activations to flow through, which has been shown empirically to improve BPB by ~0.003 in this regime. The SOTA submission `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` confirms this is a reliable gain.

## Configuration

Same as NaiveBaseline — no other changes:

- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Tied embedding LR: `TIED_EMBED_LR=0.05`
- Batching: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024`

## Run Command

```bash
NCCL_IB_DISABLE=1 \
RUN_ID=leakyrelu_baseline_8gpu \
DATA_PATH=/root/code/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/root/code/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=8 /root/code/parameter-golf/train_gpt.py
```

## Expected Result

Expected ~1.221 BPB (improvement of ~0.003 over baseline 1.2244).
Actual result to be filled in after H100 run.

## Status

Pending H100 run. Script change committed and ready to execute.
