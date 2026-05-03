# RF Drone Friend-or-Foe — Notebook Walkthrough (Studying Edition)

A guided, transition-by-transition explanation of `avionicsF1.ipynb`. The reader is assumed to be comfortable with Python and to have seen ML once before, but **nothing about this specific project is taken for granted**. Every project-specific concept (RF, I/Q, spectrograms, SNR, KNN-OOD, energy score, etc.) is explained the first time it appears.

This document covers **Sections 1 through 10**. Sections 11 (export) and 12 (distillation) are intentionally excluded.

---

## 0. Read this first — Glossary of concepts that appear later

These ideas are explained **once** here so the per-section text doesn't have to keep re-introducing them. Skip ahead and come back as needed.

### 0.1 — Spectrogram

The dataset's input format. A 2-D image of a radio signal where:
- the **vertical axis** is **frequency**,
- the **horizontal axis** is **time**,
- the **colour intensity** at each cell shows how much energy was at that frequency at that moment.

You can think of it as the "sheet music" of a radio signal — the same way a song has a waveform (loudness over time) and a spectrogram (which notes are playing, when). The notebook's spectrograms are 2 channels × 128 frequency bins × 128 time bins. Two channels because the spectrogram is computed from the **complex** I/Q signal (see 0.2) and stores both magnitude and phase information.

### 0.2 — I/Q (In-phase / Quadrature) signal

The format Software-Defined Radios output. A single radio waveform is mathematically a **complex** number at each instant: a real part `I` (in-phase) and an imaginary part `Q` (quadrature, 90° phase-shifted from I). Together they form `s(t) = I(t) + j·Q(t)`. The original `dataset.pt` file contained these raw I/Q samples; this notebook works only with the spectrograms derived from them, because spectrograms are easier for image-style CNNs to learn from. The slim `dataset2.pt` file we use has the raw I/Q dropped to save 13 GB.

### 0.3 — SNR (Signal-to-Noise Ratio)

A number, in **decibels (dB)**, that says how much louder the signal is than the noise.
- **+30 dB** — signal is ~1,000 × stronger than noise (very clean).
- **0 dB** — signal and noise are equally loud.
- **−20 dB** — noise is ~100 × stronger than signal (signal almost buried).

Each dataset sample has a known SNR in `{−20, −18, …, +28, +30}` (26 levels, 2 dB apart). The dataset was made by recording clean signals and then **adding** calibrated noise to match each target SNR. This means we can later evaluate the model at every noise condition and report a per-SNR accuracy curve — knowing where it breaks is essential for deployment.

### 0.4 — CNN (Convolutional Neural Network)

A type of neural network designed for images. Instead of treating each pixel independently, a CNN slides small "filters" (typically 3 × 3 patches of weights) across the image. Each filter looks for a specific local pattern (an edge, a corner, a frequency stripe). Stacking many filters and many layers lets the network detect simple patterns near the input and combine them into more abstract patterns deeper in the network. Spectrograms are images of radio signals, so a CNN is the natural choice.

### 0.5 — ResNet, and specifically ResNet-18

**ResNet** ("Residual Network", He et al. 2015) is a CNN family famous for solving the *vanishing-gradient* problem in deep networks by adding **skip connections** — every block computes `output = F(input) + input`, so gradients can flow through the `+ input` shortcut even when the learned `F(input)` part is small.

The number after "ResNet" is the **layer count**. ResNet-18 has **18 layers with learnable weights** organised as:
1. one initial 7 × 7 convolution (`conv1`),
2. four "stages", each with **2 BasicBlocks** containing 2 conv layers each (= 16 conv layers),
3. one final fully-connected classifier head.

So `1 + (2 blocks × 2 convs × 4 stages) + 1 fc = 18`. Other sizes exist (ResNet-34, ResNet-50, ResNet-101) — the bigger ones use bottleneck blocks and have more layers and parameters. We pick the **smallest** of the family because:
- our dataset is ~98k samples (modest by ImageNet standards) — bigger models overfit;
- 11 M params (≈ 44 MB) fits on edge hardware later;
- ResNet-18 is the most-tested baseline in RF / audio-spectrogram literature.

### 0.6 — Loss function and how it relates to the model

Training a neural network is just **minimising a number**. That number is the **loss** — a single scalar that tells you "how wrong was the model on this batch?". Each batch:
1. The model produces predictions.
2. The loss function compares predictions to the true labels and outputs the loss value.
3. PyTorch computes the gradient of that loss with respect to every weight in the model (back-propagation).
4. The optimiser nudges every weight slightly in the direction that would have made the loss smaller.

**The shape of the loss function determines what "being good" means.** For us, it's a **weighted cross-entropy** — see Section 7.1 for what that does specifically. Without the right loss, even the best architecture will learn the wrong thing.

### 0.7 — Macro-F1 vs accuracy (and why we use macro-F1)

**Accuracy** is the fraction of samples classified correctly. Easy to read but misleading on imbalanced datasets — our dataset is 53 % Noise, so a model that always says "Noise" gets 53 % accuracy without learning anything useful.

**F1** for one class blends precision (of all things I called X, how many really were X) and recall (of all true Xs, how many did I catch) into one number. **Macro-F1** is the unweighted average of per-class F1 scores. It treats DJI (2 % of data) the same as Noise (53 %) — so a model that ignores DJI gets a bad macro-F1 even if its accuracy looks fine. **This is why every model-selection decision in the notebook uses macro-F1 instead of accuracy.**

### 0.8 — Stratified sampling (split that preserves proportions)

A "stratified" split is one that keeps the same class mix in every part. With our 7 classes and 26 SNR levels we have 7 × 26 = **182 (class × SNR) cells**. A naïve random 70/15/15 split could, by bad luck, put zero DJI-at-−20-dB samples in val or test — and then we couldn't measure performance there. Stratification picks proportionally from every cell so every cell appears in train, val, AND test.

### 0.9 — Z-score normalisation

`(x − mean) / std` per sample, per channel. Brings every spectrogram to mean 0 / std 1 so the network sees consistent input scale. Done **per sample** (not dataset-wide) because the dataset has class-dependent loudness — Noise samples are ~4× louder than drone samples — and global normalisation would let the model cheat by reading "loudness" instead of the actual signal pattern.

### 0.10 — Data augmentation

Random distortions applied **only to training samples**, freshly each epoch. The model sees a slightly different view of every sample every time, so it can't memorise specific pixels — it has to learn the underlying pattern. Without augmentation, big networks overfit small datasets within ~10 epochs. We use four physically meaningful distortions (time-shift, Gaussian noise, SpecAugment time mask, SpecAugment frequency mask) explained in Section 5.

### 0.11 — AdamW, learning-rate schedule, AMP

- **AdamW** — an optimiser. Same as the well-known "Adam" but with proper L2 weight regularisation. The boring-but-reliable choice for almost every modern model.
- **Learning rate schedule** — the step size shrinks over training. We use a 3-epoch *linear warm-up* (lr climbs from 10 % to 100 % of the peak) followed by *cosine annealing* (lr decays smoothly to 1 % of the peak). Warm-up avoids early instability; cosine helps the model settle into a good minimum at the end.
- **AMP (automatic mixed precision)** — runs most operations in float16 (half precision) for speed, while keeping the master weights in float32 for stability. Free 1.5–2× speed-up on GPUs.

### 0.12 — Embedding (a.k.a. feature vector)

The numbers a network produces just *before* the final classifier. ResNet-18 produces a **512-dimensional embedding** at its avgpool output, and our last linear layer turns that into 7 logits. The embedding is the network's internal "summary" of the input — semantically similar inputs land close together in this 512-d space. Used by the friend-or-foe gate in Section 9.

### 0.13 — KNN-based OOD, energy score, AUROC

- **OOD (out-of-distribution)** — a sample that doesn't belong to any of the trained classes. In deployment, an unknown drone that wasn't in our training set is OOD. The model will *confidently* classify it as one of the 7 known classes unless we add a guard.
- **KNN distance score** — for a new sample, compute its 512-d embedding, find its **k nearest neighbours** among training drone embeddings (k = 10), and use the **distance to the k-th nearest** as the score. Far away ⇒ OOD. (Sun et al., ICML 2022.)
- **Energy score** — `E = −log Σ exp(logits)`. Big when the model is *unconfident* on every class (logits all small), small when confident. Complementary to KNN.
- **AUROC** — area under the ROC curve. A single number in [0, 1] saying how well a score separates "in-distribution" from "out-of-distribution". 0.5 = random; 1.0 = perfect. We use AUROC vs Noise as a sanity check (Noise is our stand-in for "unknown drone" since the dataset has no real unknown drones).

---

## 1 — Configuration (Cell 6)

This is the single source of truth for every hyperparameter the rest of the notebook reads. There are no sub-parts.

### What problem we try to solve
Make every tunable knob in one place. After this cell, no later cell should hard-code a number — they all reference variables defined here. If you want to retune anything (batch size, epochs, OOD target), you change it once, re-run, and the rest follows.

### Why we try to solve it (what's the value)
Reproducibility, sanity, and the ability to re-run experiments without spelunking through cells. A future reader (you in three weeks, a colleague, a fresh Claude) reads one cell to understand the entire run.

### How we are approaching this
A flat block of constants in clearly-labelled groups: class registry, data subsampling, split ratios, DataLoader settings, augmentation strengths, training hyperparameters, OOD target, and a "seed everything" call.

### Technical details, taking nothing for granted
- **`CLASS_NAMES`** is a Python `dict` mapping integer class index → human name (`0:DJI, 1:FutabaT14, 2:FutabaT7, 3:Graupner, 4:Noise, 5:Taranis, 6:Turnigy`). The order is chosen by the dataset, not by us.
- **`NOISE_IDX = 4`** is the special class. In the friend/foe rule, an `argmax == NOISE_IDX` always means *foe / noise*.
- **`FRIENDLY_IDX = [0, 1, 2, 3, 5, 6]`** is everything except Noise — the 6 drone types treated as friends if recognised.
- **`SAMPLING_RATE_HZ = 14_000_000`** documents that the original I/Q recordings were sampled at 14 Msps. Not used by the spectrogram pipeline directly, but recorded for reference.
- **`SEED = 42`** seeds Python `random`, NumPy, PyTorch CPU and CUDA RNGs. Same seed → same split, same initial weights, same results.
- **`SUBSAMPLE_FRAC = 1.0`** means *use the whole dataset*; lower values (e.g. `0.25`) take a stratified subset for free-tier Colab. We're on a 23.7 GB L4 GPU so we use 100 %.
- **`TRAIN_FRAC / VAL_FRAC / TEST_FRAC = 0.70 / 0.15 / 0.15`** — pinned by the project scope.
- **`BATCH_SIZE = 128`** and **`NUM_WORKERS = 2`** balance GPU memory and CPU prep speed.
- **`PIN_MEMORY = True`** enables async CPU→GPU transfers via page-locked memory — small free speed-up.
- **Augmentation strengths**: 10 % time-shift, σ ≤ 0.10 Gaussian noise, ≤ 16-column time mask, ≤ 16-row frequency mask. (All explained in Section 5.)
- **Optimiser**: 30 epochs, AdamW lr = 1e-3, weight decay = 1e-4, 3-epoch warm-up, early stop after 5 epochs of no improvement, AMP on.
- **`OOD_TARGET_FALSE_FOE_RATE = 0.01`** — calibrate the friend/foe threshold so ≤ 1 % of real friendlies are accidentally flagged as foes.

### What we transfer to the next step
A populated Python namespace: `CLASS_NAMES`, `NUM_CLASSES = 7`, `NOISE_IDX = 4`, `FRIENDLY_IDX`, `SEED`, `SUBSAMPLE_FRAC`, the three split fractions, batch settings, augmentation knobs, training/OOD hyperparameters, and a fully seeded RNG state. Every later cell reads from these.

---

## 2 — Load dataset

We pull `dataset2.pt` (~12.9 GB) from Google Drive into RAM, optionally subsample, and visualise the class / SNR / joint distribution to confirm the load worked.

### 2.1 — Load and extract (Cell 8)

#### What problem we try to solve
Get the four tensors the rest of the notebook needs (`x_spec`, `y`, `snr`, `duty_cycle`) into Python globals — without paying the load cost twice if a cell is re-run.

#### Why we try to solve it (what's the value)
The file is on Google Drive and Drive's FUSE mount is slow (~25 MB/s). Loading 12.9 GB took **8.1 minutes** on this Colab session. We absolutely don't want to re-pay that just because we re-ran a downstream cell.

#### How we are approaching this
The cell is **idempotent** — it checks `globals()` for an existing `x_spec` tensor and skips the reload if found. Otherwise it `torch.load`s the file, pulls out the four tensors as references (no copy), and `del`s the source dict so any unused entries get garbage-collected.

#### Technical details, taking nothing for granted
- `torch.load(DATA_PATH, map_location="cpu", weights_only=False)` deserialises the `.pt` file. `map_location="cpu"` lands tensors on the CPU first (we'll move them to the GPU per-batch later); `weights_only=False` is required because the file is a generic Python dict, not just a state-dict.
- The dataset is a dict with keys `x_spec`, `y`, `snr`, `duty_cycle` (the original raw `x_iq` was dropped by `_slim_dataset.py` before upload to Drive).
- `x_spec` shape is `(98705, 2, 128, 128)` float32 — **98,705 samples**, 2 channels, 128 × 128 spectrogram. 12.94 GB in RAM.
- `y` is `(98705,)` int64 with values 0–6 (class index).
- `snr` is `(98705,)` int32 in `[−20, +30]` dB.
- `duty_cycle` is `(98705,)` float32 in `[0.00, 1.00]` — fraction of the recording window during which a transmitter was emitting. Not used by the model in this run, kept for potential future use.

#### What we transfer to the next step
Four tensors: **`x_spec` (98705, 2, 128, 128) float32**, **`y` (98705,) int64**, **`snr` (98705,) int32**, **`duty_cycle` (98705,) float32**. They live in CPU RAM as references — no copies, ready to be sliced cheaply.

### 2.2 — Optional stratified subsample (Cell 9)

#### What problem we try to solve
Provide an escape hatch for low-RAM environments (Colab free tier, only 12 GB RAM total) where the full 12.94 GB tensor is too big.

#### Why we try to solve it (what's the value)
Lets the same notebook run on free-tier Colab without manual editing. On Pro / Pro+ (≥ 25 GB) the cell is a no-op.

#### How we are approaching this
If `SUBSAMPLE_FRAC < 1.0`, take a **stratified random subsample** keyed on the joint `(class, SNR)` cell. This guarantees every (class × SNR) cell is shrunk by the same factor — small cells (DJI at −20 dB) are not at risk of disappearing.

#### Technical details, taking nothing for granted
- Build a single integer per sample that encodes both class and SNR: `composite_key = y * 100 + snr` (one unique value per (class × SNR) cell).
- Call `sklearn.model_selection.train_test_split(all_idx, train_size=SUBSAMPLE_FRAC, stratify=composite_key, random_state=SEED)` to get a *seeded* stratified subset of indices.
- `index_select` + `.contiguous()` returns a clean RAM block; old big tensors are garbage-collected.
- In our run, `SUBSAMPLE_FRAC = 1.0` → printed `using full dataset (98,705 samples)` and the cell did nothing.

#### What we transfer to the next step
Same four tensors as 2.1, possibly trimmed. In our run: untouched.

### 2.3 — Sanity histograms + (class × SNR) heatmap (Cell 10)

#### What problem we try to solve
Confirm the load worked and *understand* the data we'll be training on. Specifically: confirm the known class imbalance, confirm all 26 SNR levels are present, and *see* the joint distribution that drives the stratified split decision in Section 3.

#### Why we try to solve it (what's the value)
Catching a bad load *before* training saves a wasted 30-minute run. Eyeballing the (class × SNR) heatmap also makes it obvious why the next section needs stratified splitting — the small cells are visibly small.

#### How we are approaching this
Three matplotlib charts in one cell:
1. **Class distribution bar chart** — Noise in red, friendlies in blue.
2. **SNR distribution bar chart** — should be roughly flat across the 26 levels.
3. **Class × SNR heatmap** — every (class × SNR) cell coloured by sample count.

Plus a numeric summary at the end.

#### Technical details, taking nothing for granted
- `Counter(y.tolist())` gives the class histogram. Output: DJI 2,194 (2.2 %), FutabaT14 6,938 (7.0 %), FutabaT7 3,661 (3.7 %), Graupner 6,481 (6.6 %), **Noise 52,552 (53.2 %)**, Taranis 16,546 (16.8 %), Turnigy 10,333 (10.5 %). Imbalance ratio max/min = **24×**.
- `Counter(int(s) for s in snr.tolist())` gives the SNR histogram. ~3,795 samples per level (our 100 % run), all 26 levels present.
- The heatmap is a `(7, 26)` integer matrix built from `Counter(zip(y, snr))`. **All 182 cells are non-empty**; the smallest is 69 samples (DJI at one SNR), the largest is 2,024 (Noise at +10 dB).
- Three PNGs are saved to `outputs2/figures2/` for the report.

#### What we transfer to the next step
A confirmed view of the dataset's structure: 7 classes (severe imbalance, Noise dominates), 26 SNR levels (balanced), 182 (class × SNR) cells (none empty, smallest = 69). This justifies the stratification design used in Section 3.

---

## 3 — Stratified 70 / 15 / 15 split

We carve the 98,705-sample dataset into three disjoint subsets — train (70 %) / val (15 %) / test (15 %) — that each look like a miniature copy of the full dataset.

### 3.1 — Build the split (Cell 12)

#### What problem we try to solve
Produce three index arrays into `x_spec` (and `y`, `snr`, …) that obey the 70/15/15 ratio **and** keep every (class × SNR) cell represented in all three splits.

#### Why we try to solve it (what's the value)
- Without stratification, the smallest cells could vanish from val/test entirely → per-SNR per-class evaluation impossible there.
- Without persistence, every kernel restart re-shuffles the split and our results stop being comparable across runs.

#### How we are approaching this
Two-step `train_test_split` from scikit-learn, each step **stratified** on the composite `(class × 100 + SNR)` key:
1. **Step A:** 70 % train / 30 % held-out.
2. **Step B:** halve the held-out into 15 % val / 15 % test.

Indices are sorted (so cache-friendly) and saved as `.npy` files in `outputs2/splits2/`. Re-running the cell loads the saved files instead of re-splitting.

#### Technical details, taking nothing for granted
- `composite_key = y.numpy().astype(np.int64) * 100 + snr.numpy().astype(np.int64)` — one unique int per cell.
- `train_test_split(..., stratify=composite_key, random_state=SEED)` is the workhorse — same seed → same split.
- For Step B we re-index into the held-out subset: `stratify=composite_key[hold_idx]`.
- `.npy` files (`train2.npy`, `val2.npy`, `test2.npy`) are NumPy's binary array format — fast to load, no pickling.
- Output sizes from our run: `Train 69,093 (70.00 %), Val 14,806 (15.00 %), Test 14,806 (15.00 %)`. `69093 + 14806 + 14806 = 98705` ✓.

#### What we transfer to the next step
Three NumPy index arrays — **`train_idx` (69,093)**, **`val_idx` (14,806)**, **`test_idx` (14,806)** — each is a 1-D array of indices into `x_spec` / `y` / `snr`. Persisted to Drive so re-runs reload them deterministically.

### 3.2 — Verify the split (Cell 13)

#### What problem we try to solve
Empirically confirm two things that *should* hold after stratification but easily go wrong:
1. **Coverage** — every (class × SNR) cell from the full dataset appears in train, val, AND test.
2. **Drift** — class proportions don't drift more than ±1 % between splits.

#### Why we try to solve it (what's the value)
Catching a stratification bug **here** is cheap. Catching it after training a 30-minute model and seeing weird per-SNR holes is expensive. Also gives us an empirical comfort baseline: if the proportions match the full distribution to within 0.03 %, we know the split is "honest" and any test-vs-val gap later is real, not artifactual.

#### How we are approaching this
- Build the set of all (class, SNR) pairs in the full dataset; for each split, build the same set; report any cells missing.
- For each class, print the percentage in Full / Train / Val / Test, plus the max drift (largest of `|Split % − Full %|`).
- Bonus: a grouped-bar chart of the four percentages side-by-side.

#### Technical details, taking nothing for granted
- `set(zip(y.tolist(), snr.tolist()))` makes a hashable set of `(class, SNR)` pairs — set-difference tells us missing cells.
- Per-class percentage in a split: `100 * (y[split_idx] == c).sum() / len(split_idx)`.
- **Coverage result:** `Coverage OK : all 182 (class × SNR) cells present in train, val, AND test`.
- **Drift result:** the largest drift is **0.028 %** (DJI), well under the ±1 % budget. Translation: train, val, test are essentially perfect statistical clones of the full dataset.
- Saved figure: `outputs2/figures2/split_class_proportions.png`.

#### What we transfer to the next step
Empirical confirmation that `train_idx / val_idx / test_idx` are usable. No data structure changes. The conclusion (full coverage + ≤ 0.03 % drift) is the green light to start using the splits.

---

## 4 — PyTorch Dataset and DataLoaders

A neural network trains on **batches** of samples streamed from a `DataLoader`. We need a thin adapter (a `Dataset`) that exposes `(x, y, snr)` for any sample index, applies per-sample z-score normalisation, and (optionally) a transform — and we need three `DataLoader`s wrapping it: train (with shuffling), val, test.

### 4.1 — `DroneDataset` class (Cell 16)

#### What problem we try to solve
Bridge our raw tensors (`x_spec`, `y`, `snr`) and a `torch.utils.data.DataLoader`. The DataLoader needs an object with `__len__` and `__getitem__(i)` — it doesn't care about anything else.

#### Why we try to solve it (what's the value)
A `Dataset` class is the standard PyTorch way to define "what does sample `i` look like?" — it gives us a single place to do per-sample preprocessing (z-score) and inject augmentations later (Section 5). The DataLoader handles batching, shuffling, multiprocessing, and pinning automatically.

#### How we are approaching this
A minimal subclass of `torch.utils.data.Dataset` that:
- Stores **references** (not copies) to the in-memory tensors plus an index array.
- On `__getitem__(i)`, slices out one spectrogram, casts to float, applies z-score, and (if a transform was set) applies the transform.

#### Technical details, taking nothing for granted
- The constructor takes `indices, x_spec, y, snr, transform=None`.
- `self.indices = torch.as_tensor(indices, dtype=torch.long)` — wraps the NumPy array as a torch long tensor without copying.
- `__len__` returns `len(self.indices)`.
- `__getitem__(i)` does:
  1. `idx = int(self.indices[i])` — translate the loader's "sample i in this split" into a global dataset index.
  2. `x = self.x_spec[idx].float()` — slice + cast.
  3. **Per-sample, per-channel z-score**: `means = x.mean(dim=(1, 2), keepdim=True)`, same for std with `+ 1e-6` to avoid divide-by-zero, then `x = (x - means) / stds`. Each of the 2 channels is normalised independently, so the model sees mean-0 / std-1 inputs.
  4. If `self.transform` is not None, apply it (Section 5 will plug augmentations in here for train only).
  5. Return `x, int(y[idx]), int(snr[idx])`.

#### What we transfer to the next step
A class definition `DroneDataset`. Nothing instantiated yet — that's 4.2.

### 4.2 — Build DataLoaders + smoke test (Cell 17)

#### What problem we try to solve
Three actual DataLoaders (one per split) ready to feed batches into the model, plus immediate confirmation that one batch comes out in the right shape with the right normalisation.

#### Why we try to solve it (what's the value)
Catches data-pipeline bugs *before* training. A z-score that's off, an indexing slip, a wrong dtype, a class missing from a batch — all show up here in 0.1 seconds instead of 30 minutes into training.

#### How we are approaching this
- Instantiate three `DroneDataset` objects (train / val / test), each with the corresponding split indices.
- Wrap each in a `DataLoader` with batch size 128, `num_workers=2`, `pin_memory=True`.
- Train loader has `shuffle=True, drop_last=True`. Val and test have `shuffle=False`.
- Pull one batch from `train_loader`, verify shapes / dtype / mean / std / class coverage / SNR range.

#### Technical details, taking nothing for granted
- `shuffle=True` reshuffles every epoch — the model never sees the same order twice. Critical for SGD-style optimisation.
- `drop_last=True` discards the final partial batch (`69093 % 128 = 101` leftover samples in our run). Stops a tiny edge-case batch from destabilising BatchNorm. Cost: 0.15 % of the train data is unused per epoch — invisible.
- `num_workers=2` uses 2 background processes to prepare batches; on Linux Colab this uses copy-on-write memory so the 12.94 GB tensor isn't actually copied per worker.
- `pin_memory=True` puts each batch in page-locked CPU memory, which lets the CUDA driver DMA-transfer it to the GPU without an intermediate copy. Tiny free speed-up.
- Number of batches: train 539 (= 69093 / 128 floor), val 116 (= 14806 / 128 ceil), test 116.
- **Smoke batch result**: shape `(128, 2, 128, 128)`, dtype `float32`, **mean = −0.0000, std = 0.9999** — confirms z-score works perfectly. Classes in batch `[0, 1, 2, 3, 4, 5, 6]` (all 7 because of the imbalance + 128 batch size). SNR range −20 → +30.

#### What we transfer to the next step
**`train_loader`, `val_loader`, `test_loader`** — three production-ready DataLoaders. Train loader **iterates over augmented samples** (after Section 5 plugs the transform in); val/test stay clean for honest evaluation.

---

## 5 — Train-only augmentations

Augmentations are random distortions applied *only* to training samples, freshly each epoch. Without them, big networks like ResNet-18 memorise small datasets within ~10 epochs. With them, the model sees ~30 different distorted views of every sample and is forced to learn the underlying invariant pattern instead.

### 5.1 — `SpectrogramAugmentations` class + attach to train loader (Cell 19)

#### What problem we try to solve
Implement four physically-meaningful spectrogram distortions in a single callable, and plug it into `train_ds.transform` so train batches go through `(load → z-score → distort → return)` while val/test stay clean.

#### Why we try to solve it (what's the value)
Each distortion targets a specific *generalisation gap*: the dataset has bursts at fixed positions (the model could memorise positions), discrete SNR levels (the model could overfit to those), and clean clear channels (the model could latch on a single frequency band). Each augmentation breaks one of these unwanted shortcuts.

#### How we are approaching this
A `SpectrogramAugmentations` class with one `__call__(x)` method that runs four operations in sequence:
1. **Random circular time-shift** along the last axis, ± 10 % (≤ ±12 columns).
2. **Additive Gaussian noise**, σ drawn fresh per sample from `U(0, 0.1)`.
3. **SpecAugment time mask** — zero a random vertical band of width ≤ 16 columns.
4. **SpecAugment frequency mask** — zero a random horizontal band of height ≤ 16 rows.

Then assign `train_ds.transform = SpectrogramAugmentations()`. `val_ds.transform` and `test_ds.transform` are **left as None**.

#### Technical details, taking nothing for granted
- **Time shift** uses `torch.roll(x, shifts=shift, dims=-1)` — circular wrap-around, no zeros introduced. Models the fact that a real burst can land anywhere in the 1.17 ms recording window.
- **Gaussian noise** is `x + torch.randn_like(x) * sigma` with `sigma = uniform(0, 0.1)` per call. Smooths the decision boundary between the discrete SNR levels and improves robustness to slightly noisier-than-trained conditions.
- **SpecAugment masks** (Park et al., 2019) — random vertical band → "time dropout" (transmission glitch); random horizontal band → "frequency dropout" (narrow-band jammer). Forces the model to use the whole spectrogram, not one bright stripe.
- Augmentations run on the **CPU inside DataLoader workers**, so they don't compete with the GPU for time.
- Confirmed at runtime: `train_ds.transform : SpectrogramAugmentations`, `val_ds.transform : None`, `test_ds.transform : None`.

#### What we transfer to the next step
A `train_loader` that now serves **augmented** batches. `val_loader` and `test_loader` continue to serve clean batches (the eval loop must compare apples to apples). Plus a class definition `SpectrogramAugmentations` we can re-use for the visualisation cell.

### 5.2 — Visual sanity check (Cell 20)

#### What problem we try to solve
Confirm visually that augmentation **distorts** but does not **destroy**. The drone's underlying fingerprint must still be recognisable across multiple augmented copies — otherwise we're teaching the model garbage.

#### Why we try to solve it (what's the value)
A single image catches a class of bugs that numbers can't (e.g. a mask with width 256 instead of 16 would zero out the whole spectrogram silently). Even a quick eyeball check confirms the augmentation strengths are sane.

#### How we are approaching this
Pick the first non-Noise drone sample from the train split, z-score it the same way `DroneDataset` does, then run **three independent augmentation calls** (each with a different `torch.manual_seed`) and plot the original next to the three distorted versions, all on the same colour scale.

#### Technical details, taking nothing for granted
- The seed reset before each call (`torch.manual_seed(1/2/3)`) makes the demo reproducible — re-running gives the same three augmented panels.
- The 4 panels share `vmin/vmax` so colour intensities are comparable.
- Saved to `outputs2/figures2/augmentation_demo.png`.
- What you see when you run it: the bright frequency-hopping pattern stays in the same general region in every panel; some panels show small horizontal shifts, some are slightly grainier (Gaussian noise), some have a black vertical or horizontal band cut through them (SpecAugment).

#### What we transfer to the next step
Visual confirmation that augmentation behaviour matches expectations. No new code state — just a sanity image on disk.

---

## 6 — ResNet-18 model implementation

We need the actual network: input is a `(batch, 2, 128, 128)` spectrogram, output is `(batch, 7)` logits (one score per class). We'll also need a 512-d *feature* vector (the layer just before the final classifier) for the OOD detector in Section 9.

### 6.1 — `DroneClassifier` (Cell 22)

#### What problem we try to solve
Build a 2-channel ResNet-18 with a 7-class head that **also** exposes the 512-d penultimate embedding via a separate method, so Section 9 can fit OOD on it without rerunning the backbone.

#### Why we try to solve it (what's the value)
- ResNet-18 because it's the most-tested baseline for spectrogram classification and fits comfortably on edge hardware.
- Exposing features separately (instead of just logits) lets the inference function in Section 10 do **one** forward pass per sample for both the closed-set prediction *and* the OOD score. Without this, every prediction would cost two forwards.

#### How we are approaching this
Subclass `nn.Module`, take `torchvision.models.resnet18(weights=None)` (random init, no ImageNet pretraining), and apply two surgical edits:
1. Replace `conv1` with a `Conv2d(2, 64, …)` — input expects 2 channels (our spectrogram), not the 3 channels (RGB) ResNet ships with.
2. Replace the final `fc` (which would output 1,000 ImageNet classes) with `nn.Identity()`, then add our own `Linear(512, 7)` head as a separate attribute.

Expose three forward modes: `forward(x) → logits`, `features(x) → 512-d embedding`, `forward_with_features(x) → (logits, embedding)` in a single backbone pass.

#### Technical details, taking nothing for granted
- **Why no pretrained weights** (`weights=None`): pretrained ResNet weights are calibrated for natural RGB images. Spectrograms have very different statistics (no edges, no gradients of luminance, very different histograms), and we replace `conv1` anyway, so pretraining wouldn't help.
- **`backbone.fc = nn.Identity()`** is a clean trick — it makes the backbone return the 512-d feature directly without the original 1,000-class classifier. Our `Linear(512, 7)` runs on top.
- The backbone (everything except our `head`) runs ResNet-18's: `conv1 → bn1 → relu → maxpool → layer1..layer4 → avgpool → flatten → Identity()`. Each `layer{1..4}` is two `BasicBlock`s with a skip connection. Channels grow 64 → 128 → 256 → 512 with stride-2 down-sampling between stages.
- **Parameter count from runtime**: `11,176,967` total, all trainable. **`44.7 MB` on disk** in float32.
- `model.to(DEVICE)` moves the parameters to the GPU (NVIDIA L4 in our run).

#### What we transfer to the next step
A trained-from-scratch (random-init) `model` on the GPU, with three forward methods. **Not yet trained** — that's Section 7.

### 6.2 — Smoke-test forward pass (Cell 23)

#### What problem we try to solve
Confirm the wiring is correct *before* training: 2-channel input flows through, 7-class output is the right shape, and `forward(x)` matches `forward_with_features(x)` exactly (so Sections 9 and 10 can rely on the joint method).

#### Why we try to solve it (what's the value)
Same logic as 4.2: catching a wiring bug here is cheap, catching it after a 30-minute train run is expensive.

#### How we are approaching this
Pull one batch from the train loader, push it through the model (in eval mode + `no_grad` so we don't pollute BatchNorm running stats or accumulate gradients), and assert: `logits.shape == (B, 7)`, `feats.shape == (B, 512)`, and that `forward()` and `forward_with_features()` produce numerically identical outputs (within float tolerance).

#### Technical details, taking nothing for granted
- `model.eval()` flips BatchNorm and Dropout to inference behaviour. We restore `model.train()` at the end so later cells can train.
- `torch.no_grad()` disables autograd — saves memory, faster.
- `torch.allclose(a, b, atol=1e-5)` checks floating-point equality up to a tolerance.
- **Runtime numbers from our run**: input `(128, 2, 128, 128)`, logits `(128, 7)` with range `[−2.627, +8.343]` (random spread because the model is untrained), feats `(128, 512)` with mean +2.83 / std 2.43 (positive mean expected because the last operation before the avgpool is a ReLU which clips negatives to 0).

#### What we transfer to the next step
Empirical confirmation that the model is wired correctly. The model itself is left in `train()` mode at the end of the cell, ready for Section 7 to start updating its weights.

---

## 7 — Training

We have a model, three DataLoaders, and augmentation. Now we actually train: define the loss, the optimiser, the LR schedule, an evaluation helper, and the training loop itself.

### 7.1 — Class-weighted cross-entropy loss (Cell 26)

#### What problem we try to solve
Use a loss function that doesn't let the model "win" by always predicting Noise. With 53 % Noise in training, plain cross-entropy would give the model a strong incentive to ignore the minority classes.

#### Why we try to solve it (what's the value)
Without the weighting, the macro-F1 we care about (Section 0.7) would be garbage: a "always Noise" model would score 53 % accuracy and ~0.13 macro-F1. With inverse-frequency weighting, one DJI mistake now counts ~24× as much as one Noise mistake, so the model is forced to take the minority classes seriously.

#### How we are approaching this
**Cross-entropy** is the standard classification loss: it compares the predicted probability distribution (softmax of logits) against the true one-hot label and penalises divergence. For class `c`, contribution = `−log(softmax_c(logits))`. We multiply each class's contribution by a per-class **weight** so wrong predictions on rare classes count more.

Weights are computed from **training counts only** (never use val/test counts to set training hyperparameters):
- `w_c = total / (N_classes × n_c)` — inverse frequency.
- Then normalise so the weights have **mean = 1** (keeps the loss magnitude comparable to plain CE).

#### Technical details, taking nothing for granted
- `Counter(y[train_idx].tolist())` gives per-class counts on the *train* split.
- Empirical weights from this run: **DJI 2.6460, FutabaT14 0.8391, FutabaT7 1.5905, Graupner 0.8983, Noise 0.1108, Taranis 0.3519, Turnigy 0.5634** (mean = 1).
- `criterion = nn.CrossEntropyLoss(weight=weights)` — PyTorch automatically applies the weight per-sample based on the true label.
- `nn.CrossEntropyLoss` is **softmax + negative-log-likelihood combined** — it expects raw logits, not softmaxed probabilities. Doing softmax yourself before passing to it is the #1 PyTorch beginner bug.

#### What we transfer to the next step
A `criterion` callable that takes `(logits, true_labels)` and returns a single weighted-CE loss tensor. Used by the training loop (7.4) and the eval helper (7.3).

### 7.2 — Optimiser, LR scheduler, AMP scaler (Cell 28)

#### What problem we try to solve
Decide the rule for *how* the model's weights get updated each step (the optimiser), *how the step size changes* over the 30 epochs (the schedule), and *how to use float16 safely* during forward+backward (AMP).

#### Why we try to solve it (what's the value)
A bad LR is the #1 cause of either no progress (lr too small) or instability (lr too large). The 3-epoch warm-up smooths the first few unstable steps when freshly-initialised weights and large gradients combine. The cosine decay then lets the model settle into a good minimum at the end of training. AMP doubles training speed for free.

#### How we are approaching this
- **Optimiser:** `torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)`. AdamW is Adam with proper decoupled weight decay — almost always the right default.
- **Scheduler:** `SequentialLR` chaining `LinearLR(start_factor=0.1, end_factor=1.0, total_iters=3)` (warm-up: lr climbs from 0.0001 to 0.001 over the first 3 epochs) into `CosineAnnealingLR(T_max=27, eta_min=1e-5)` (cosine decay from 0.001 to 0.00001 over the remaining 27 epochs).
- **AMP scaler:** `torch.amp.GradScaler("cuda", enabled=True)` — needed for safe backprop in float16, scales the loss so tiny gradients don't underflow.

#### Technical details, taking nothing for granted
- **AdamW vs SGD:** AdamW adapts the step size per-parameter based on a running estimate of the gradient's magnitude, so flat regions get bigger steps and curvy regions get smaller ones. SGD-with-momentum can match it on big datasets but needs more tuning. For our scale, AdamW is "set and forget".
- **`weight_decay=1e-4`** is L2 regularisation — penalises large weights to prevent overfitting. AdamW does this *correctly* (decoupled from the gradient update); old Adam mixes it into the gradient and regularises less effectively.
- **`SequentialLR` mechanics:** runs the first scheduler until the milestone (epoch 3), then switches to the second.
- **GradScaler** wraps the loss with `scaler.scale(loss).backward()` and the optimiser step with `scaler.step(optimizer); scaler.update()`. It dynamically scales the loss up before backprop so float16 gradients don't underflow, then unscales before applying the step.

#### What we transfer to the next step
Three live objects: **`optimizer`**, **`scheduler`**, **`scaler`**. Plus the `criterion` from 7.1. The training loop (7.4) just glues these together with the model and DataLoaders.

### 7.3 — `evaluate()` helper (Cell 30)

#### What problem we try to solve
Compute the model's loss / accuracy / macro-F1 on a loader. Used twice (per epoch on val during training, on test in Section 8) so it must be a reusable function, not inline code.

#### Why we try to solve it (what's the value)
- **Single source of truth:** if the eval logic changes, only one place to edit.
- Returns a `details` dict with `preds / labels / snrs / logits` — Section 8 uses these for the per-class table, the confusion matrix, and the per-SNR curve. Section 9 uses logits for the energy score.

#### How we are approaching this
A `@torch.no_grad()`-decorated function that:
1. Sets `model.eval()`.
2. Iterates the loader, runs each batch through the model under `autocast`, accumulates loss × batch_size, and stores `argmax`, `labels`, `snrs`, `logits`.
3. Concatenates everything into NumPy arrays, computes overall accuracy and `sklearn.metrics.f1_score(..., average="macro")`.
4. Returns `(loss, accuracy, macro_f1, details)`.

#### Technical details, taking nothing for granted
- `autocast("cuda", enabled=USE_AMP)` runs the same float16 dance as in training but without backward — safe.
- The loss is summed weighted by batch size and divided by total samples at the end → correct mean even when batch sizes differ (the val loader's last batch is `14806 % 128 = 102` samples, not 128).
- **Macro-F1** averages per-class F1 with equal weight — explained in 0.7. This is the metric we'll select on.
- The function is a NO-OP modifier on `model.training`: it leaves `model` in `eval()` mode at the end. The training loop in 7.4 calls `model.train()` before training each epoch, so this is fine.

#### What we transfer to the next step
A reusable `evaluate(model, loader, criterion)` function. The training loop (7.4) calls it once per epoch with `val_loader`. Section 8 calls it once with `test_loader`.

### 7.4 — Training loop (Cell 32)

#### What problem we try to solve
Actually train the model: 30 epochs of (forward → loss → backward → step) on `train_loader`, with per-epoch evaluation on `val_loader`, learning-rate scheduling, checkpointing of the best model so far (by val macro-F1), early stopping after 5 stale epochs, and a CSV log row per epoch.

#### Why we try to solve it (what's the value)
Training is where the model becomes useful. Everything before this was infrastructure; everything after this assumes a trained model exists.

#### How we are approaching this
For each epoch:
1. **Train:** model.train(), loop over `train_loader` with autocast + GradScaler, accumulate train loss / accuracy.
2. **Validate:** call `evaluate(model, val_loader, criterion)`.
3. **Scheduler step:** advance the LR per epoch (warm-up → cosine).
4. **Log:** write a row to `training_log.csv`, print a one-line status.
5. **Checkpoint:** always save `last.pt`. If val macro-F1 improved over the best so far, also save `best.pt` and reset patience; else increment patience and break out if it hits 5.

After the loop, reload `best.pt` so the in-memory model is the best-by-val one. Wrap everything in `try/except KeyboardInterrupt/finally` so the log file always closes cleanly.

#### Technical details, taking nothing for granted
- `optimizer.zero_grad(set_to_none=True)` clears gradients before each batch (set-to-none is slightly faster than zeroing tensors).
- Inside `autocast`, the forward pass and loss compute in float16 / float32 mix; gradient computation respects that.
- `scaler.scale(loss).backward()` → `scaler.step(optimizer)` → `scaler.update()` is the AMP three-step.
- **Early stopping** monitors val macro-F1 with `patience = 5`. In our run it never tripped (the model kept improving until epoch 30).
- **Empirical results from our run:**
  - Each epoch: ~35 s on the L4 GPU.
  - Best val macro-F1: **0.9081** at epoch 30.
  - Train acc at the end: **0.9303**, val acc at the end: **0.9377** — **val > train**, classic sign that augmentation is preventing overfitting.
  - Total training time: ~17 min.
- After the loop: `model.load_state_dict(torch.load("best.pt", ...))` restores the best weights.

#### What we transfer to the next step
A **trained `model`** in memory (best-by-val weights), a `best.pt` and `last.pt` on Drive, and a `training_log.csv` with 30 rows of per-epoch metrics. Section 8 evaluates this trained model on test; Section 9 reads its features.

---

## 8 — Closed-set evaluation on the test split

We've trained for 30 epochs and selected the best-by-val checkpoint. Now we evaluate it on test (which the model has never seen). Three views: headline numbers + per-class table, confusion matrix, per-SNR accuracy curve.

### 8.1 — Headline metrics + per-class table (Cell 35)

#### What problem we try to solve
Get the single number you'd quote in a report (test macro-F1) plus a per-class breakdown showing where the model is strong and weak.

#### Why we try to solve it (what's the value)
The headline number says "should we ship this?". The per-class table says "where do we need to be careful?" — e.g., "DJI macro-F1 is 0.86 vs Taranis 0.98, so be cautious about DJI in deployment."

#### How we are approaching this
Call `evaluate(model, test_loader, criterion)` to get loss / accuracy / macro-F1 / details. Then `sklearn.metrics.precision_recall_fscore_support(labels, preds)` for the per-class table.

#### Technical details, taking nothing for granted
- **Precision** for class X: of all predictions of X, what fraction were really X? (False-positive control.)
- **Recall** for class X: of all true Xs, what fraction did we catch? (False-negative control.)
- **F1** for class X: harmonic mean of precision and recall — penalises a model that's good at one but bad at the other.
- **Support**: number of true Xs in test (sample count per class).
- **Empirical results from our run:**
  - Test accuracy = **93.98 %**, macro-F1 = **0.9090**, samples = 14,806.
  - Per-class F1: DJI **0.8625** (support 325), FutabaT14 **0.8889** (1,040), FutabaT7 **0.8299** (547), Graupner **0.9543** (973), Noise **0.9515** (7,887), Taranis **0.9810** (2,480), Turnigy **0.8947** (1,554). Macro avg = 0.9090.
- Test ≈ Val (0.9090 vs 0.9081 macro-F1) → the split was honest, no test-set leakage.

#### What we transfer to the next step
The `preds`, `labels`, and `snrs` arrays from `test_details` — used by 8.2 for the confusion matrix and 8.3 for the per-SNR curve.

### 8.2 — Confusion matrix (Cell 37)

#### What problem we try to solve
*Where* exactly are the errors? "10 % of DJIs are misclassified" doesn't tell us if they're going to FutabaT7, Noise, or scattered randomly. A 7×7 confusion matrix shows it instantly.

#### Why we try to solve it (what's the value)
The off-diagonal cells are diagnostic: they show **which classes the model confuses with which**. They drive future improvements — if DJI mostly leaks to Noise, we know the failure mode is "low-SNR drone loses its signature". If DJI leaks to Taranis, we'd suspect feature similarity instead.

#### How we are approaching this
- `confusion_matrix(labels, preds, labels=list(range(7)))` from sklearn.
- Row-normalise (each row sums to 1) so the diagonal is **per-class recall**.
- Plot as a heatmap with `imshow(cmap="Blues")`. Annotate every cell with raw count + percentage; flip text colour to white on dark cells for readability.

#### Technical details, taking nothing for granted
- Rows = true class, columns = predicted class.
- The diagonal is per-class recall. The Y-axis is "what the sample really was"; the X-axis is "what the model said".
- **Dominant pattern in our run:** the rightmost cell of the Noise column is bright; *most errors are drones being misclassified as Noise*. Specifically: DJI → Noise 11.7 %, FutabaT14 → Noise 9.3 %, FutabaT7 → Noise 10.6 %, Turnigy → Noise 10.6 %. Drone-to-drone confusion is < 2 % anywhere.
- **Why:** at low SNR the drone signal *is* dominated by noise, so the spectrogram literally looks like a Noise sample. The model is doing the *right thing* — it just can't extract a signal that's 100× weaker than the background.
- Saved: `outputs2/figures2/confusion_matrix.png`.

#### What we transfer to the next step
A confirmed mental model: errors cluster in *drone → Noise*, not drone → drone. Section 8.3 will show *where* (which SNRs) those errors live.

### 8.3 — Per-SNR accuracy curve (Cell 39)

#### What problem we try to solve
Plot test accuracy as a function of SNR. We need to know **at what noise level does the model break down**, because the deployment policy depends on it ("trust the model above −5 dB; below that defer to manual review").

#### Why we try to solve it (what's the value)
RF classifiers always degrade at low SNR — the question is *where* and *how steeply*. This curve is what you'd quote when setting deployment trust thresholds.

#### How we are approaching this
For each unique SNR `s`, mask the test predictions to that SNR and compute `(preds == labels).mean()`. Plot accuracy vs SNR, with reference lines: overall test accuracy (dashed grey) and 0 dB (dotted red).

#### Technical details, taking nothing for granted
- `mask = snrs_t == s` — boolean array.
- Per-SNR sample counts in test: ~570 each (since we did a 100 % run with stratified split, and there are 14,806 / 26 ≈ 570).
- **Empirical curve from our run** (selected SNRs):
  - **−20 dB → 63.5 %** (information-theoretic floor; signal is buried)
  - **−16 dB → 83.8 %**
  - **−10 dB → 94.0 %**
  - **−8 dB → 96.7 %** (model essentially saturates here)
  - **0 dB → 98.1 %**
  - **+10 dB → 97.7 %**
  - **+30 dB → 95.4 %** (slight tail-off because high-SNR drone signals look very similar to each other — drone-vs-drone confusion creeps in)
- The model is **reliable above ~−8 dB** (≥ 96 %), degrades smoothly to 63 % at −20 dB. That's roughly the limit of what *any* classifier could do at SNRs that low without longer integration windows or denoising preprocessors.
- Saved: `outputs2/figures2/per_snr_accuracy.png`.

#### What we transfer to the next step
The closed-set story is complete. Section 9 layers the **friend-or-foe gate** on top so that low-confidence predictions (and signals from drones never seen in training) get caught before the closed-set head's mistake reaches the user.

---

## 9 — Friend-or-foe gate (KNN + Energy OOD calibration)

The closed-set classifier always picks one of 7 classes — it has no way to say "I've never seen this". In deployment, an unknown drone (say a Holy Stone HS720 not in our training set) would be confidently labelled as "DJI" or "Taranis", and we'd treat a foe as a friend. We add a second layer: an **out-of-distribution (OOD) detector** that flags samples whose 512-d embedding is far from any known drone in feature space, regardless of what the closed-set head says. (See 0.13 in the glossary for the underlying concepts.)

> **Note on the section heading:** the 9.2 markdown still says "Mahalanobis statistics", which was the original plan. After empirical testing showed Mahalanobis AUROC was 0.245 (going the wrong way), the **code** was swapped to KNN (Sun et al., ICML 2022) without updating the header. The text below describes what the **code actually does**.

### 9.1 — Collect train and val embeddings (Cell 42)

#### What problem we try to solve
Get the 512-d penultimate embeddings (from `model.features(x)`) for every sample in train and val, plus the val logits. We need the train embeddings to define "what does a known drone look like in feature space?" and the val embeddings + logits to calibrate thresholds.

#### Why we try to solve it (what's the value)
Without these embeddings, we have no way to ask "is this new sample close to known data?". The OOD score in 9.3 is computed *in this 512-d space*, not in pixel space.

#### How we are approaching this
A helper function `collect_features(model, loader)` runs the loader once, calls `model.forward_with_features(x)` per batch, and concatenates the results. We use it twice: once on `train_loader` (after **temporarily disabling augmentation** via `train_ds.transform = None`) and once on `val_loader`.

#### Technical details, taking nothing for granted
- **Why disable augmentation here:** at deployment, raw inputs are *not* augmented — we want the OOD reference set to reflect what the model actually sees in the wild. The `try/finally` block restores augmentation after collection.
- **Why train + val (not test):** train embeddings define the reference cloud; val is used to **calibrate the threshold** (the test split stays untouched until Section 10's demo).
- `forward_with_features` runs the backbone once and returns `(logits, feats)` — half the cost of running `forward` and `features` separately.
- `feats.float().cpu()` casts the float16 output of `autocast` back to float32 and moves to CPU (we have a 23 GB GPU but training the model already uses some, so we keep the OOD reference on CPU).
- **Empirical sizes from our run:**
  - `train_feats : (68992, 512)` float32 — note `68992 = 539 batches × 128` because the train loader has `drop_last=True` (lost 101 of 69,093 samples; harmless 0.15 %).
  - `val_feats : (14806, 512)` float32, `val_logits : (14806, 7)` float32.

#### What we transfer to the next step
**`train_feats` (68,992 × 512), `train_labels`, `val_feats` (14,806 × 512), `val_labels`, `val_logits`** — all on CPU, ready for the KNN reference build (9.2) and threshold calibration (9.3).

### 9.2 — Build the KNN drone reference set (Cell 44)

#### What problem we try to solve
Build the *reference cloud* the OOD detector compares against: the 512-d embeddings of known drones (Noise excluded), L2-normalised so distances become scale-invariant.

#### Why we try to solve it (what's the value)
- **Drones only:** Noise is already caught by the argmax-Noise rule in inference. Including Noise in the reference would let Noise samples find each other as nearest neighbours and score *low distance* (the opposite of what we want).
- **L2-normalise:** scales every feature to unit length, so the KNN distance becomes a cosine distance. Equivalent up to a sign-flip but numerically nicer to threshold.

#### How we are approaching this
Filter `train_feats` to only the rows where `train_labels != NOISE_IDX`, then `F.normalize(..., p=2, dim=1)`. Print per-class counts as a sanity check.

#### Technical details, taking nothing for granted
- **L2-normalisation:** divides each row by its L2 norm so every vector has length 1.
- **Empirical results from our run:**
  - Drone reference set: **32,257 samples × 512 features (66.1 MB)**.
  - Per-class counts: DJI 1,536, FutabaT14 4,851, FutabaT7 2,558, Graupner 4,528, Taranis 11,561, Turnigy 7,223. Total = 32,257 ✓.

#### What we transfer to the next step
**`train_drone_feats_norm` (32,257 × 512)**, the L2-normalised drone reference cloud. Section 9.3 computes KNN distance from val samples to this; Section 10 uses the same cloud at inference time.

### 9.3 — Score val + calibrate thresholds + AUROC sanity check (Cell 46)

#### What problem we try to solve
Pick the numerical thresholds **τ_E** and **τ_KNN** at which we'll flag a sample as "foe / unknown" in deployment. Calibrate them so that no more than 1 % of *real friendly drone* val samples get falsely flagged.

#### Why we try to solve it (what's the value)
Without calibration, we'd be guessing thresholds. The friend/foe gate's behaviour is entirely determined by τ_E and τ_KNN — these two numbers *are* the deployment policy.

#### How we are approaching this
1. **Score every val sample with two complementary scores:**
   - **Energy** = `−logsumexp(logits)` — small when the closed-set head is confident.
   - **KNN** = distance to the 10th nearest neighbour in the drone reference. Computed in chunks of 512 queries to keep peak memory bounded.
2. **Calibrate τ at the (1 − target)-percentile of known-drone scores.** With target = 1 %, that's the 99th percentile — by construction, exactly 1 % of known drones land above the threshold.
3. **AUROC sanity check vs Noise:** Noise is our stand-in for "unknown" (the dataset has no real unknown drones). If AUROC > 0.9, the score correctly separates known drones from "different stuff".

#### Technical details, taking nothing for granted
- **Cosine distance via dot product on L2-normalised vectors:** for two unit vectors `u, v`, `cos(u, v) = u·v` and **distance = 1 − u·v**. So `q @ ref.T` is the matrix of cosine similarities and `1 - sims` is the matrix of cosine distances.
- **`topk(k, dim=1, largest=False)`** finds the k *smallest* distances per row; we take the **k-th** (i.e. the 10th nearest, the largest of the 10 smallest distances). This gives a more robust score than the single nearest neighbour.
- **`np.quantile(scores[mask], 0.99)`** is the empirical 99th percentile of known-drone scores.
- **`roc_auc_score(ood_target, scores)`** from sklearn computes the AUROC for the binary task "is this sample OOD?".
- **Empirical results from our run:**
  - **Energy:** τ_E = **−2.0326**, AUROC vs Noise = **0.806**, false-foe rate at τ = 1.01 % (vs 2 % of Noise above τ → most Noise has lower energy, i.e. is *more* confidently classified as Noise itself).
  - **KNN:** τ_KNN = **0.1408**, AUROC vs Noise = **0.937** (much stronger separator), false-foe rate at τ = 1.01 % (vs 18 % of Noise above τ → KNN catches a meaningful fraction of Noise as OOD even before the argmax rule fires).
  - Drone scores cluster tightly near zero (KNN), Noise scores spread between 0.05 and 0.25. Visually: clear separation.

#### What we transfer to the next step
Two scalars that *are* the gate: **`tau_E = −2.0326`**, **`tau_KNN = 0.1408`**. Plus `K = 10`. The persistence cell (9.4) bundles these with the reference cloud into a single deployable file.

### 9.4 — Persist `ood_stats.pt` (Cell 48)

#### What problem we try to solve
Save everything the inference function needs (reference cloud, thresholds, `K`, NOISE_IDX) into a single file on Drive that can travel with the model checkpoint.

#### Why we try to solve it (what's the value)
The trained model alone is **not enough** to deploy the friend/foe gate — the gate needs the OOD stats too. Bundling them into one file keeps the deployment artefact set tidy: `model.pt` + `ood_stats.pt` is everything the on-drone runtime needs.

#### How we are approaching this
Build a Python dict of all OOD state and `torch.save` it as a single `.pt` file.

#### Technical details, taking nothing for granted
- The dict contains:
  - `train_drone_feats_norm` (the 32,257 × 512 reference cloud, L2-normalised),
  - `tau_E`, `tau_KNN` (the two thresholds),
  - `K = 10` (neighbours per query),
  - `noise_idx = 4`, `friendly_idx = [0, 1, 2, 3, 5, 6]` (convenience constants for the inference rule),
  - `false_foe_target = 0.01` (operating point, for traceability),
  - `feat_dim = 512`, `ood_method = "energy + knn (Sun et al., ICML 2022)"` (metadata).
- File size from our run: **66.1 MB on Drive** (almost entirely the reference cloud — `32257 × 512 × 4 bytes ≈ 66 MB`).

#### What we transfer to the next step
A single self-contained file `ood_stats.pt` on Drive. Section 10's `DronePredictor` loads it and uses it together with the trained `model` to make friend/foe decisions on new inputs.

---

## 10 — Inference

We have a trained `model` and a calibrated `ood_stats.pt`. Now we glue them into one deployable object that takes a raw spectrogram and returns a friend/foe verdict, then demo it on real test samples plus a couple of synthetic OOD inputs.

### 10.1 — `DronePredictor` class (Cell 51)

#### What problem we try to solve
Bundle the trained model + the OOD stats + the class-name registry into one object with a single `predict(x)` method that runs the full friend/foe rule on a raw 2-channel spectrogram.

#### Why we try to solve it (what's the value)
A clean deployment interface: a downstream user (or a future self) can write `verdict = predictor.predict(x_spec)[0]` and get a dict with the verdict + label + diagnostic scores, without knowing anything about z-scoring, KNN distance, energy scores, or the three-tier rule.

#### How we are approaching this
Define a `DronePredictor` class with:
- `__init__(model, ood_stats, class_names, device)` — moves the OOD reference onto the device, stores the thresholds and the class names, sets `model.eval()`.
- `_normalize(x)` — static method that does the same per-sample, per-channel z-score `DroneDataset` does. Means the caller can pass *raw* spectrograms.
- `@torch.no_grad() predict(x)` — the friend/foe pipeline.

The decision rule (priority order):
1. If `argmax(logits) == NOISE_IDX` → return `("foe", "noise")` with `trigger="argmax_noise"`. The closed-set head says background — no need to look further.
2. Else if `energy > τ_E` **OR** `knn_distance > τ_KNN` → return `("foe", "unknown")` with `trigger` listing which gate fired (`energy`, `knn`, or `energy+knn`).
3. Else → return `("friend", drone_name)` with `trigger="ok"`.

#### Technical details, taking nothing for granted
- The ladder ordering matters: argmax-Noise is the cheapest and most reliable check, so it fires first. The OOD scores only need to be evaluated when the closed-set head thinks it's seeing a drone.
- `predict` accepts both single `(2, 128, 128)` and batch `(B, 2, 128, 128)` inputs; if 3-D, it `unsqueeze(0)`s to add the batch dim. Always returns a **list of dicts** (one per sample).
- The result dict contains: `verdict ('friend'|'foe')`, `label (drone_name|'noise'|'unknown')`, `predicted_class` (always the argmax class name, regardless of verdict — useful for debugging), `argmax_idx`, `energy`, `knn`, `trigger`. The `trigger` field is the most useful diagnostic: it tells you *which* gate caught a "foe" verdict.
- Loaded `ood_stats.pt` via `torch.load(... weights_only=False)` because the file is a generic dict.
- **Runtime confirmation from our run:** `tau_E = −2.0326, tau_KNN = 0.1408, K = 10, Reference shape = (32257, 512), Device = cuda`.

#### What we transfer to the next step
A live **`predictor` object** ready to call. The demo cell (10.2) exercises it.

### 10.2 — Demo on real test samples + synthetic random-noise tensors (Cell 53)

#### What problem we try to solve
Confirm in one cell that the friend/foe gate handles every realistic input correctly: real drones (each of 6 known classes), real Noise, pure Gaussian random tensor (worst-case OOD), and a higher-variance random tensor (a tougher OOD).

#### Why we try to solve it (what's the value)
The demo cell *is* the spec for what the deployment behaviour should be. If anything in the table looks wrong, we know before shipping. It also exposes the diagnostic `trigger` field — so the user can see the layered defense at work.

#### How we are approaching this
- Pull one real test sample per class (7 samples) and call `predictor.predict(x)`.
- Run `predictor.predict(torch.randn(2, 128, 128))` for a pure Gaussian random tensor.
- Run `predictor.predict(torch.randn(2, 128, 128) * 1.5)` for a higher-variance random tensor (matches real spectrogram statistics more closely → harder for the argmax-Noise rule to catch).

#### Technical details, taking nothing for granted
- For the 7 real samples, we filter `test_idx` by class with `(y[test_idx] == c).numpy()` and pick the first match.
- The synthetic tensors are passed in **without** a batch dimension; the predictor adds it.
- **Empirical results from our run:**

| Sample (true class @ SNR) | Verdict | Label | Argmax | Energy | KNN | Trigger |
| --- | --- | --- | --- | --- | --- | --- |
| DJI @ +8 dB | **friend** | DJI | DJI | −10.21 | 0.0051 | ok |
| FutabaT14 @ −10 dB | **friend** | FutabaT14 | FutabaT14 | −9.16 | 0.0314 | ok |
| FutabaT7 @ +22 dB | **friend** | FutabaT7 | FutabaT7 | −8.66 | 0.0352 | ok |
| Graupner @ +20 dB | **friend** | Graupner | Graupner | −12.79 | 0.0056 | ok |
| Noise @ −18 dB | **foe** | noise | Noise | −2.43 | 0.1474 | argmax_noise |
| Taranis @ +4 dB | **friend** | Taranis | Taranis | −8.92 | 0.0126 | ok |
| Turnigy @ +12 dB | **friend** | Turnigy | Turnigy | −7.61 | 0.0382 | ok |
| `torch.randn(2, 128, 128)` (pure Gaussian) | **foe** | noise | Noise | −1.79 | 0.2528 | argmax_noise |
| `torch.randn(2, 128, 128) * 1.5` (plausible OOD) | **foe** | unknown | DJI | −1.73 | 0.2267 | **energy+knn** |

The last row is the killer test: the model's closed-set head is *fooled* (predicts DJI), but **both OOD gates fire** and override the verdict. Defense in depth confirmed working.

#### What we transfer to the next step
This is the last subsection in scope. The deliverable is the `predictor` object plus the demo confirming it works. Sections 11 (export) and 12 (distillation) are out of scope per the brief.

---

## Final notes for the reader

If you want to re-run any single sub-section without re-running the whole notebook:
- **Sections 1–4** are pure setup; they're idempotent and cheap to re-run.
- **Section 7** (training) is expensive (~17 min). If you re-run from scratch it overwrites the checkpoints; if you only want to evaluate or re-run OOD, you can skip 7.4 once `best.pt` exists and just do `model.load_state_dict(torch.load(CKPT_DIR/"best.pt"))`.
- **Section 9** (OOD calibration) is fast (~30 s once features are collected) — you can re-tune `OOD_TARGET_FALSE_FOE_RATE` in Section 1 and re-run 9.3 → 9.4 to get a different operating point.
- **Section 10** (inference) reads `ood_stats.pt` from disk — re-run 10.1 after editing 9.4 to reload the new thresholds.

The three numbers worth remembering from this run:
- **Test macro-F1 = 0.9090** (closed-set quality).
- **KNN AUROC vs Noise = 0.937** (OOD detector quality).
- **Per-SNR accuracy floors at 63 % at −20 dB** (failure mode is information-theoretic, not architectural).
