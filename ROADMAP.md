# PetsAlive inference roadmap

**Goal:** a fast, cheap, personalized pet-image service with real benchmark numbers, written up as a technical blog post with plots, two clean repos, and the app as a live demo.

**Pipeline:** pet photos → per-pet LoRA (+ auto description) → fast SDXL generation, served through the Express + BullMQ proxy server with serverless GPU workers. Video comes later.

**Compute:** research runs on Kaggle (free T4); production numbers on RunPod (prepaid, min workers 0, ~$10).

## Week 1: worker and server (~80%)
- [x] SDXL + IP-Adapter worker; full path tested via Colab (server → queue → GPU → image)
- [x] Server: RunPod client, BullMQ queue, auth, credits with refunds, job status, per-job metrics
- [ ] RunPod image build + generation endpoint

## Week 2: per-pet personalization (research done)
- [x] DreamBooth-LoRA training (`train.py`), LoRA serving, DINO/CLIP scorer
- [x] Server: pets, training queue, `petId` in generate (IP-Adapter until the LoRA is ready)
- [x] Run 1: LoRA keeps prompt adherence, IP-Adapter leaks the reference background
- [x] Run 2: step sweep. Identity plateaus at 200–300 steps (7–10 min on a T4 vs 16 min for 500)
- [ ] Real training through the server (needs a RunPod training endpoint + Cloudflare R2)
- [ ] Hugging Face model card + Boss LoRA (pending: checkpoint, repo name, public/private)

## Week 3: inference speed (in progress)
- [x] Run 3: fusing (+7%, free), LCM-8 (5.2×, −13% identity), Lightning-4 (8.6×, overrides the pet), torch.compile (+11%, 124 s compile)
- [x] Run 4: auto pet description (Qwen2-VL-2B) × {base25, LCM-8, Lightning-4, Hyper-SD-8, DMD2-4}. The description removes all identity failures
- [x] **Production config: LCM-8 + auto description + fused pet LoRA** (4.8 s on a T4, best worst case 0.75, license-safe). Picked visually: Hyper-SD is grainy, Lightning looks painted, DMD2 is non-commercial
- [x] Run 5 (LCM-8 only): DINO margin vs. real British Shorthairs can't separate LoRA + desc from desc-only (+0.049 vs +0.050). Owner's judgment: the LoRA clearly looks more like Boss. **The LoRA stays in production**; DINO measures breed, not the individual
- [ ] Individual identity metric (animal re-ID embedding), validated with the real-photo margin test
- [x] Run 6: profiled training; own loop `train_fast.py` is 2.34× faster per step (cached VAE latents −37%, no grad checkpointing −32%, fused AdamW; 8-bit Adam is slower for LoRA). 250-step LoRA in ~4.0 min vs ~8.8 min on a T4, same quality
- [x] Run 7: cast fix (no gain), torch.compile (−2%, 5 min compile), time to plateau 220 s → 56 s with 512px / batch 4 / lr 4e-4 (~1.7 min to a usable LoRA on a T4 incl. loading). Batching only helps without grad checkpointing
- [x] **Production training config: 512px, batch 4, LR 4e-4, 200 images** (fastest, ~56 s on a T4)
- [x] Use `train_fast.py` in production `train.py` (replaces the diffusers subprocess)
- [x] Run 9: QLoRA: nf4 +45% slower, −1.8 GB only; bnb int8 2.9× slower; an nf4-trained LoRA transfers to the fp16 base. (Its INT8 GEMM result was a layout artifact, corrected by run 10)
- [x] Run 10: **INT8 GEMM is ~2× fp16 on the T4 with the TN layout** (CUTLASS Turing IMMA kernel; 1.5–1.9× on SDXL FF shapes, ~1× on 640×640 attention projections). fp16 NN (pre-transposed weights) is also 1.4–2.2× faster than TN on FF shapes. Triton has no tensor-core path on sm75
- [ ] **Research direction (alive): INT8 frozen-base LoRA training on the T4.** INT8 TN GEMMs for the FF layers (forward + input gradient), per-token activation quantization fused by torch.compile, attention in fp16; compare end to end vs fp16 TN and fp16 with pre-transposed weights; then the same on Ampere+ / fp8
- [ ] Optional: Triton attention-backward kernel for sm75 (≤ ~8% of a step)
- [x] Ship it: `handler.py` (fused LCM, 8 steps, no CFG, resident multi-LoRA slots), Qwen2-VL description in `train.py`, description stored on the pet in the server. Run 11 smoke test passes on a T4
- [x] Run 8: multi-LoRA serving (`multi_lora.py`): resident LoRAs +33% throughput vs swap+fuse, ~26 MB/user, matches fused output; batching gives nothing on a T4 at 1024px. INT8 on the T4: weight-only NaN (torchao/torch mismatch), W8A8 correct but 4.3× slower eagerly
- [ ] fp8 / int8 row on an L4 / 4090 / H100 (RunPod): matching torchao build, quantize only transformer-block linears
- [ ] Quality vs. latency vs. $/image table (needs RunPod prices and timings)

## Week 4: production numbers, demo, write-up
- [ ] **RunPod on L4:** build the image via GitHub, generation + training endpoints, Cloudflare R2 for LoRAs
- [ ] Re-measure on the L4: training sweep (24 GB: maybe 768px / batch 2–4 without checkpointing), bf16, FlashAttention; fp8 / int8 (Ada)
- [ ] Load test: cold start, p50/p95, autoscaling, cost per 1,000 images; time-to-first-LoRA on a real GPU (target < 3 min)
- [ ] App: connect `result.tsx` to the server (login, pets, generate, poll)
- [ ] Image-to-video with Wan 2.2, if there is time
- [ ] Blog post, repo cleanup, demo video

## Stretch
- **Multi-LoRA serving:** done in PyTorch (run 8). Left: a Triton BGMV kernel, a user-grouping scheduler (fuse only for runs of ≥ 3 images per user), batching on bigger GPUs
- **Task-specific VLM compression** (descriptions are now part of production, so this is relevant): shrink the pet describer
  - distill Qwen2-VL-2B into a small student on teacher-labeled pet photos
  - task-aware pruning + healing fine-tune (LLM-Pruner / Minitron-style)
  - vocabulary pruning (~230M of Qwen2's params are embeddings for a 151k vocab)
  - baseline: a CLIP/DINOv2 encoder with classification heads (breed, coat, pattern, fur, eyes) filling a prompt template
  - measure size / latency vs. quality retained, including the downstream identity score of the generated images
- LoRA quality fixes: prior preservation, more varied full-body photos, per-photo captions, LoRA trained at 1024px

## Small fixes
- "Joao" admin backdoor in `authService.register`
- Delete `src/index.js` (old entry point, hardcoded secret)
- ~14 s polling overhead between the server and the worker
- Signed URLs for pet photos before any real launch
- DMD2 is CC-BY-NC: research only, not for the paid app
