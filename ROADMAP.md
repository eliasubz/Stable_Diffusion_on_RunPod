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
- [ ] Run 4: auto pet description (Qwen2-VL-2B) × {base25, LCM-8, Lightning-4, Hyper-SD-8, DMD2-4}, with worst case and failure rate
- [ ] Pick the production config; ship fused LoRA + best sampler in `handler.py`
- [ ] fp8 / int8 row on an L4 / 4090 / H100 (RunPod)
- [ ] Quality vs. latency vs. $/image table (needs RunPod prices and timings)

## Week 4: production numbers, demo, write-up
- [ ] RunPod endpoints (generation + training) and R2 storage
- [ ] Load test: cold start, p50/p95, autoscaling, cost per 1,000 images; time-to-first-LoRA on a real GPU (target < 3 min)
- [ ] App: connect `result.tsx` to the server (login, pets, generate, poll)
- [ ] Image-to-video with Wan 2.2, if there is time
- [ ] Blog post, repo cleanup, demo video

## Stretch
- **Multi-LoRA serving:** many users' LoRAs on one GPU, batched (S-LoRA/Punica ideas), with one custom Triton kernel for batched LoRA application
- **Task-specific VLM compression** (only if run 4 shows that descriptions help): shrink the pet describer
  - distill Qwen2-VL-2B into a small student on teacher-labeled pet photos
  - task-aware pruning + healing fine-tune (LLM-Pruner / Minitron-style)
  - vocabulary pruning (~230M of Qwen2's params are embeddings for a 151k vocab)
  - baseline: a CLIP/DINOv2 encoder with classification heads (breed, coat, pattern, fur, eyes) filling a prompt template
  - measure size / latency vs. quality retained, including the downstream identity score of the generated images
- Run 5 quality fixes: prior preservation, more varied full-body photos, per-photo captions, LoRA trained at 1024px

## Small fixes
- "Joao" admin backdoor in `authService.register`
- Delete `src/index.js` (old entry point, hardcoded secret)
- ~14 s polling overhead between the server and the worker
- Signed URLs for pet photos before any real launch
- DMD2 is CC-BY-NC: research only, not for the paid app
