# 🧠 LLM Fine-Tuning: A Complete Guide for AI Engineers

> **Fine-tuning transforms a general-purpose language model into a specialized expert — without training from scratch.**

---

## Table of Contents

1. [What Is Fine-Tuning?](#1-what-is-fine-tuning)
2. [Fine-Tuning vs RAG](#2-fine-tuning-vs-rag-retrieval-augmented-generation)
3. [When Is Fine-Tuning the Right Choice?](#3-when-is-fine-tuning-the-right-choice)
4. [Full Fine-Tuning (SFT)](#4-full-fine-tuning-sft--supervised-fine-tuning)
5. [Parameter-Efficient Fine-Tuning (PEFT)](#5-parameter-efficient-fine-tuning-peft)
6. [LoRA (Low-Rank Adaptation)](#6-lora-low-rank-adaptation)
7. [QLoRA (Quantized LoRA)](#7-qlora-quantized-lora)
8. [LoRA vs QLoRA — Detailed Comparison](#8-lora-vs-qlora--detailed-comparison)
9. [When and How to Combine LoRA and QLoRA](#9-when-and-how-to-combine-lora-and-qlora)
10. [Advanced Fine-Tuning Techniques](#10-advanced-fine-tuning-techniques)
11. [Data Preparation](#11-data-preparation-for-fine-tuning)
12. [Evaluation and Avoiding Common Pitfalls](#12-evaluation-and-avoiding-common-pitfalls)
13. [Deployment Patterns](#13-deployment-patterns)
14. [Quick Decision Checklist](#14-quick-decision-checklist-for-ai-engineers)
15. [Useful Libraries and Tools](#15-useful-libraries-and-tools)

---

## 1. 🔍 What Is Fine-Tuning?

**Fine-tuning** is the process of continuing to train a pre-trained language model on a task-specific or domain-specific dataset. Rather than learning language from scratch, the model's existing knowledge is adapted — its **weights** are updated to specialise behaviour for a narrower purpose.

### Key Concepts

| Concept | Detail |
|---|---|
| **What gets updated** | Model weights (all, or a low-rank subset via PEFT) |
| **What stays frozen** | Optionally: base model weights (in PEFT methods), embeddings |
| **Starting point** | A pre-trained foundation model (e.g. LLaMA 3, Mistral, Gemma) |
| **Goal** | Specialisation: style, tone, domain knowledge, output format, instruction-following |

### Fine-Tuning vs Training from Scratch

Training from scratch requires billions of tokens, thousands of GPU-hours, and enormous datasets to teach a model basic language understanding. Fine-tuning **skips all of that** — the pre-trained model already understands language, grammar, reasoning, and world knowledge. Fine-tuning simply steers that knowledge toward a specific task using a much smaller dataset (often thousands, not billions, of examples).

> **Key insight:** Fine-tuning is transfer learning applied to LLMs. The expensive general knowledge is pre-learned; you only pay for specialisation.

---

## 2. ⚖️ Fine-Tuning vs RAG (Retrieval-Augmented Generation)

### Comparison Table

| Aspect | Fine-Tuning | RAG |
|---|---|---|
| **Knowledge storage** | Encoded in model weights | External vector database / document store |
| **Update frequency** | Requires re-training to update | Real-time or near-real-time updates |
| **Latency** | Low (no retrieval step) | Higher (retrieval adds latency) |
| **Hallucination risk** | Moderate (model may confabulate) | Lower (grounded in retrieved documents) |
| **Cost** | High upfront (training), low inference | Low upfront, ongoing retrieval/indexing cost |
| **Best use case** | Style, tone, domain behaviour, format | Factual Q&A over large/changing knowledge bases |
| **Interpretability** | Low (knowledge is implicit in weights) | High (sources can be cited) |
| **Infrastructure needed** | GPU cluster for training, standard inference | Vector DB, embedding model, retrieval pipeline |

### When to Choose Each

**Choose fine-tuning when:**
- The model's **behaviour, style, or tone** needs to change (e.g. a formal legal assistant, a playful children's tutor)
- **Domain-specific jargon** must be deeply understood (e.g. medical terminology, proprietary code syntax)
- **Low-latency inference** is critical and no retrieval hop is acceptable
- You cannot maintain retrieval infrastructure

**Choose RAG when:**
- **Data changes frequently** (news, financial data, product catalogues)
- **Source citations** are legally or practically required
- The **knowledge base is large and dynamic** (millions of documents)
- **Hallucination must be minimised** via grounding

**Choose BOTH when:**
You want **style and tone via fine-tuning** (so the model communicates correctly) **plus live factual data via RAG** (so the model's answers are current and grounded). This is increasingly the production standard for enterprise AI.

---

## 3. 🎯 When Is Fine-Tuning the Right Choice?

| Scenario | Recommended Approach | Reason |
|---|---|---|
| Customer support bot with custom brand tone | Fine-Tuning (SFT) | Tone/style is a weight-level behaviour, not retrievable |
| Medical domain Q&A with clinical terminology | Fine-Tuning + RAG | Domain vocab via fine-tuning; current guidelines via RAG |
| Up-to-date news retrieval and summarisation | RAG | Knowledge changes daily; fine-tuning would go stale |
| Code generation in a proprietary internal language | Fine-Tuning (LoRA) | Proprietary syntax not in pre-training data |
| Low-resource hardware (single consumer GPU) | QLoRA | 4-bit quantisation enables 7B–13B fine-tuning on 24GB VRAM |
| Real-time knowledge base over 10M documents | RAG | No feasible way to encode 10M docs into weights |
| Structured output format (JSON schema, tables) | Fine-Tuning (SFT) | Output format is a behavioural pattern best taught via examples |
| Multilingual support for underrepresented language | Fine-Tuning (SFT/LoRA) | Low-resource languages benefit from targeted training data |
| Human preference alignment (helpfulness, safety) | SFT → DPO or RLHF | Preference alignment requires comparison data, not just examples |
| Rapid prototyping with no training budget | Prompt Engineering / RAG | Zero training cost; test feasibility before committing |

---

## 4. 🏋️ Full Fine-Tuning (SFT — Supervised Fine-Tuning)

**Supervised Fine-Tuning (SFT)** in its full form updates **all model weights** during training. The model receives instruction-response pairs and learns to minimise the cross-entropy loss over the target tokens.

### Characteristics

- **Highest quality possible** — every weight can be adjusted to fit the task
- **Most computationally expensive** — a 7B model in fp16 requires ~56GB VRAM just for weights, plus optimizer states and gradients
- **Risk of catastrophic forgetting** — aggressive training can overwrite general capabilities
- **Requires large GPU memory** — typically multi-GPU setups (A100 80GB, H100) or multi-node clusters

### When to Use Full Fine-Tuning

Use full SFT when:
- You are fine-tuning a **small model** (≤3B parameters) where VRAM is manageable
- You have **unlimited compute** (cloud cluster, A100/H100 fleet)
- **Maximum accuracy** is required and PEFT methods fall short
- You are training a **base model from scratch** on domain data (continual pre-training)

### Common Frameworks

| Framework | Best For |
|---|---|
| **Hugging Face Trainer** | Ease of use, quick iteration, single/multi-GPU |
| **DeepSpeed** | ZeRO-stage sharding across many GPUs, very large models |
| **FSDP (Fully Sharded Data Parallel)** | Native PyTorch alternative to DeepSpeed |

---

## 5. ⚡ Parameter-Efficient Fine-Tuning (PEFT)

**PEFT** is an umbrella term for techniques that fine-tune only a small subset of parameters while keeping the base model largely frozen. This dramatically reduces VRAM, training time, and cost — often with minimal quality loss compared to full fine-tuning.

### PEFT Methods Overview

| Method | Approach | Trainable Params |
|---|---|---|
| **LoRA** | Low-rank decomposition of weight updates | ~0.1–1% of total |
| **QLoRA** | LoRA on a 4-bit quantised base model | ~0.1–1% (adapters in fp16) |
| **Prefix Tuning** | Prepends trainable virtual tokens to each layer's key/value | Very small |
| **Prompt Tuning** | Learns soft prompt tokens prepended to input embeddings | Tiny (~1K params) |
| **Adapter Layers** | Inserts small bottleneck MLP modules between transformer layers | Small |
| **(IA)³** | Scales activations via learned vectors; even fewer params than LoRA | Extremely small |

> **Key insight:** LoRA and QLoRA dominate production usage due to their strong quality-to-cost ratio and seamless integration with the Hugging Face ecosystem.

---

## 6. 🔗 LoRA (Low-Rank Adaptation)

### 6.1 Core Concept

In a standard transformer, weight matrices (e.g. the query projection **W_q**) are large — often shape `d × d` where `d` can be 4096 or larger. During full fine-tuning, every element of every such matrix is updated.

LoRA's insight: **the update ΔW doesn't need to be full rank.** Instead of updating W directly, LoRA decomposes the update into two small matrices:

```
W' = W + ΔW = W + A × B
```

Where:
- **W** has shape `d × d` (frozen)
- **A** has shape `d × r` (trainable)
- **B** has shape `r × d` (trainable)
- **r** is the rank, where `r ≪ d` (e.g. r=8 vs d=4096)

---

### 6.2 Why This Is Clever (The Math)

#### Parameter Count Comparison

```
Original full fine-tuning:  d × d = 4096 × 4096 = 16,777,216 parameters

LoRA (r=8):                  d×r + r×d = 2 × 4096 × 8 = 65,536 parameters

Reduction factor:            16,777,216 / 65,536 ≈ 256x fewer trainable parameters
```

#### The Update Formula

```python
# LoRA forward pass (conceptual)
W_new = W_frozen + (A @ B) * (lora_alpha / rank)
```

#### Scaling with `alpha / rank`

The term `alpha / rank` is a **scaling factor** that controls the magnitude of the LoRA update relative to the frozen weights. Setting `alpha = 2 * rank` is a common default.

#### Initialisation

- **A** is initialised from a random Gaussian distribution (small values)
- **B** is initialised to **zero**

This ensures `ΔW = A × B = 0` at the start of training — the model begins fine-tuning from an identical state to the pre-trained base, and the adapter gradually diverges.

#### Zero-Latency Inference Merging

After training, the LoRA matrices can be **merged back into the base weights**:

```python
W_merged = W_frozen + (A @ B) * (alpha / rank)
```

The merged model is **identical in architecture** to the original — no extra computation, no latency overhead at inference time.

---

### 6.3 Target Modules

LoRA can be applied to any linear (dense) layer. In transformer decoder models, the standard targets are:

| Module | Role |
|---|---|
| `q_proj` | Query projection in self-attention |
| `k_proj` | Key projection in self-attention |
| `v_proj` | Value projection in self-attention |
| `o_proj` | Output projection in self-attention |
| `gate_proj` | Gate in SwiGLU MLP |
| `up_proj` | Up-projection in MLP |
| `down_proj` | Down-projection in MLP |

> **Best practice:** Target all linear layers (`target_modules="all-linear"`) for maximum expressiveness, unless VRAM is extremely constrained.

---

### 6.4 Key Hyperparameters

| Hyperparameter | Typical Range | Effect |
|---|---|---|
| `r` (rank) | 4 – 128 | Higher rank = more capacity, more memory, more trainable params |
| `lora_alpha` | 8 – 256 (often 2× rank) | Scales the LoRA update magnitude |
| `lora_dropout` | 0.0 – 0.1 | Regularisation; small datasets benefit from 0.05 |
| `target_modules` | `q_proj,v_proj` or `all-linear` | Which layers to adapt; more layers = richer adaptation |
| `learning_rate` | 1e-4 – 3e-4 | Higher than full FT; LoRA adapters learn from scratch |
| `batch_size` | 1 – 16 (with grad accumulation) | Effective batch = batch_size × grad_accum_steps |
| `num_epochs` | 1 – 5 | More epochs risk overfitting on small datasets |
| `weight_decay` | 0.0 – 0.01 | Light regularisation; often 0.0 for small adapters |

---

### 6.5 LoRA Code Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    torch_dtype=torch.float16,
    device_map="auto",
)

# Define LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                          # Rank
    lora_alpha=32,                 # Scaling factor
    lora_dropout=0.05,             # Dropout for regularisation
    target_modules="all-linear",   # Apply to all linear layers
    bias="none",                   # Don't train biases
)

# Wrap model with LoRA adapters
model = get_peft_model(model, lora_config)

# Inspect trainable parameters
model.print_trainable_parameters()
# Output: trainable params: 83,886,080 || all params: 8,114,671,616 || trainable%: 1.03%
```

---

## 7. 🔬 QLoRA (Quantized LoRA)

### 7.1 Core Concept

**QLoRA** = LoRA applied to a **4-bit quantised base model**.

The key idea: quantise the frozen base model weights to 4-bit to slash memory usage, then train LoRA adapters in 16-bit floating point as usual. This decouples the memory footprint of the base model from the precision of the learned update.

This technique, introduced by Dettmers et al. (2023), enables fine-tuning a **65B-parameter model on a single 48GB GPU** — a task that would otherwise require ~16× A100 80GB GPUs.

---

### 7.2 Three Technical Innovations

#### 1. 4-bit NormalFloat (NF4) Quantisation

Standard int4 quantisation uses uniformly spaced quantisation levels. But LLM weights follow a **normal (Gaussian) distribution** — most values cluster near zero, with exponentially fewer values at the extremes. Uniform levels waste resolution.

**NF4** is **information-theoretically optimal** for normally distributed data. Its 16 quantisation levels are spaced so that each level captures an equal fraction of the weight distribution — maximum information per bit.

```
dequantize(q) = lookup_table[q]
```

Each quantised weight index `q ∈ {0..15}` maps to a pre-computed float value. Dequantisation is a single table lookup — fast and accurate.

#### 2. Double Quantisation

Quantisation itself requires **scale factors** (one per block of weights, e.g. 64 weights per block). These scale factors are stored as float32, costing 32 bits per 64 weights = **0.5 extra bits per parameter**.

Double Quantisation applies a second round of quantisation to the scale factors themselves — from **float32 → float8** — saving approximately **0.37 bits per parameter** on average.

For a 7B model, this saves ~325MB of memory at virtually no quality cost.

#### 3. Paged Optimisers

During long-sequence training with gradient checkpointing, optimizer states (Adam momentum buffers) can intermittently spike GPU memory and cause out-of-memory (OOM) errors.

**Paged optimisers** use NVIDIA's **unified memory** feature to automatically page optimizer state tensors between GPU VRAM and CPU RAM when the GPU runs low on memory — like virtual memory for the GPU. This prevents OOM crashes without manual memory management.

---

### 7.3 Memory Comparison (7B Model)

| Component | Full Fine-Tuning | LoRA (fp16 base) | QLoRA (4-bit base) |
|---|---|---|---|
| Model weights | ~14 GB (fp16) | ~14 GB (fp16, frozen) | ~3.5 GB (NF4) |
| Optimizer states | ~56 GB (Adam, fp32) | ~1 GB (adapter only) | ~1 GB (adapter only) |
| Gradients | ~14 GB | ~1 GB | ~1 GB |
| Activations | ~8 GB (variable) | ~8 GB (variable) | ~6 GB (variable) |
| **Total VRAM** | **~92 GB** | **~24 GB** | **~10–12 GB** |

> **Key insight:** QLoRA makes fine-tuning a 7B model feasible on a single RTX 3090 (24GB) or even an RTX 4070 Ti (12GB).

---

### 7.4 QLoRA Code Example

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 1. Configure 4-bit quantisation
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                        # Enable 4-bit loading
    bnb_4bit_quant_type="nf4",                # Use NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16,    # Compute in bf16 during forward pass
    bnb_4bit_use_double_quant=True,           # Enable double quantisation
)

# 2. Load base model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    quantization_config=bnb_config,
    device_map="auto",
)

# 3. Prepare model for k-bit training (handles layer casting, grad checkpointing)
model = prepare_model_for_kbit_training(model)

# 4. Apply LoRA adapters on top of the quantised base
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 83,886,080 || all params: 8,114,671,616 || trainable%: 1.03%
```

---

## 8. 🔄 LoRA vs QLoRA — Detailed Comparison

| Aspect | LoRA | QLoRA |
|---|---|---|
| **Base model precision** | fp16 or bf16 | 4-bit NF4 (quantised) |
| **Adapter precision** | fp16 / bf16 | fp16 / bf16 |
| **VRAM usage (7B model)** | ~24 GB | ~10–12 GB |
| **Training speed** | Faster (no dequant overhead) | ~20–30% slower (dequantisation per forward pass) |
| **Final quality** | Marginally better | Minimal quality gap vs LoRA |
| **Best hardware** | A100 40/80GB, H100, multi-GPU | RTX 3090/4090, A100 40GB, single GPU |
| **Mergeable weights** | Yes (merge A×B into W) | Yes, after dequantising base first |
| **Inference speed** | Full speed (if merged) | Slower if base stays quantised; full speed if merged |
| **Recommended for** | High-quality training with ample VRAM | Consumer GPU fine-tuning, 13B+ on single GPU |

---

## 9. 🔀 When and How to Combine LoRA and QLoRA

> **Important:** QLoRA **IS** LoRA — it is simply LoRA applied to a quantised base model. The adapter architecture and training procedure are identical; only the base model's storage format differs.

### Decision Guide

**Use LoRA (fp16 base) when:**
- You have sufficient VRAM (>40GB for 7B, >80GB for 13B)
- You want **maximum training speed and quality**
- You will merge adapters before deployment
- You are running on A100/H100 hardware

**Use QLoRA (4-bit base) when:**
- VRAM is limited (<24GB for 7B, <40GB for 13B)
- You are fine-tuning 13B+ models on **consumer-grade GPUs** (RTX 3090/4090)
- Cloud cost reduction is a priority
- Slight training speed reduction is acceptable

**Both approaches can be combined with:**
- Gradient checkpointing (trade compute for memory)
- Flash Attention 2 (faster, memory-efficient attention)
- DeepSpeed ZeRO (further sharding across GPUs)

### Post-Training Deployment Options

```python
# Option A: Keep adapter separate (flexible, slightly slower inference)
model = AutoModelForCausalLM.from_pretrained(base_model_path)
model = PeftModel.from_pretrained(model, adapter_path)

# Option B: Merge adapter into base (fastest inference, no extra complexity)
from peft import PeftModel
model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, adapter_path)
merged_model = model.merge_and_unload()   # Fuse A×B into W
merged_model.save_pretrained("./merged_model")

# Option C (QLoRA): Dequantise base, merge, then re-quantise for deployment
# Step 1: Load and merge as above (merge_and_unload handles dequantisation internally)
# Step 2: Convert to GGUF/AWQ/GPTQ for efficient inference
```

---

## 10. 🚀 Advanced Fine-Tuning Techniques

### RLHF (Reinforcement Learning from Human Feedback)

**RLHF** aligns model outputs with human preferences through a two-stage pipeline. First, a **reward model** is trained on human comparison data (which of two responses is better?). Then the main model is optimised using **PPO (Proximal Policy Optimization)** to maximise the reward model's score. RLHF powered the initial alignment of ChatGPT and Claude. It is powerful but complex — requiring two models, careful PPO tuning, and ongoing human labelling.

### DPO (Direct Preference Optimization)

**DPO** is a simpler, more stable alternative to RLHF. Instead of training a separate reward model and running PPO, DPO reformulates the preference optimisation problem as a **supervised classification task** over (chosen, rejected) response pairs. No reinforcement learning, no reward model, no instability. DPO achieves comparable alignment quality to RLHF at a fraction of the complexity and is now the dominant preference-tuning technique.

### SFT + DPO Pipeline

The standard modern alignment pipeline is:
1. **SFT** on high-quality instruction-response pairs to teach basic instruction-following
2. **DPO** on preference comparison pairs `(prompt, chosen_response, rejected_response)` to align with human preferences

This two-stage approach (pioneered by Zephyr, Llama 3 Instruct, and others) produces well-aligned, helpful models efficiently.

### Gradient Checkpointing

During backpropagation, intermediate activations must be stored for gradient computation. For long sequences, this can consume more VRAM than the model itself. **Gradient checkpointing** recomputes activations during the backward pass rather than storing them — trading extra computation (~30% more) for dramatically reduced activation memory. Enable with `model.gradient_checkpointing_enable()`.

### Flash Attention 2

Standard attention computes a full `seq_len × seq_len` attention matrix, which is both memory- and compute-intensive for long sequences. **Flash Attention 2** uses tiling and kernel fusion to compute attention in a memory-efficient way without materialising the full matrix — enabling much longer context fine-tuning on the same hardware. Available via `attn_implementation="flash_attention_2"` in Hugging Face.

### DeepSpeed ZeRO

**ZeRO (Zero Redundancy Optimizer)** shards optimizer states, gradients, and model parameters across multiple GPUs, eliminating the memory redundancy of naive data parallelism. ZeRO Stage 3 allows fine-tuning 70B+ models across a small cluster of A100s. Integrates seamlessly with Hugging Face Accelerate.

---

## 11. 📦 Data Preparation for Fine-Tuning

### Dataset Format

Fine-tuning data is typically structured as **instruction-response pairs** or **multi-turn conversations**. The two dominant formats are:

**Alpaca Format** (instruction-input-output):
```json
{
  "instruction": "Summarise the following text in one sentence.",
  "input": "The mitochondria is the powerhouse of the cell...",
  "output": "Mitochondria generate ATP, the cell's primary energy currency."
}
```

**ChatML Format** (multi-turn conversation):
```
<|im_start|>system
You are a helpful medical assistant.<|im_end|>
<|im_start|>user
What is the first-line treatment for hypertension?<|im_end|>
<|im_start|>assistant
The first-line treatment typically includes lifestyle modifications and ACE inhibitors or thiazide diuretics.<|im_end|>
```

### Data Quality Over Quantity

> **Principle:** 1,000 high-quality, diverse, well-formatted examples consistently outperform 100,000 noisy, duplicated, or poorly formatted ones.

Steps for data quality:
1. **Deduplication** — remove exact and near-duplicate examples (use MinHash or embedding similarity)
2. **Format consistency** — all examples must follow the same chat template
3. **Response quality filtering** — remove short, incoherent, or off-topic responses
4. **Length balancing** — avoid datasets skewed toward very short or very long responses

### Minimum Dataset Size Guidelines

| Task Type | Recommended Minimum Samples | Notes |
|---|---|---|
| Instruction following (general) | 1,000 – 10,000 | Diversity matters more than volume |
| Domain adaptation (style/tone) | 500 – 2,000 | Focused, high-quality examples |
| Medical / legal Q&A | 5,000 – 50,000 | Accuracy-critical; more is better |
| Code generation | 10,000 – 100,000 | Code diversity is critical |
| Preference alignment (DPO) | 2,000 – 20,000 chosen/rejected pairs | Quality of rejected samples matters |
| Multilingual | 1,000+ per language | Balance across languages |

### Train/Validation Split

Always reserve 5–10% of data for validation. Monitor validation loss during training — if it diverges from training loss, the model is overfitting.

---

## 12. 🧪 Evaluation and Avoiding Common Pitfalls

### Catastrophic Forgetting

When fine-tuning aggressively, the model may overwrite general capabilities (e.g. lose coding ability after fine-tuning on medical data).

**Mitigations:**
- Use a **low learning rate** (1e-4 or lower)
- Use **LoRA/PEFT** (frozen base weights prevent direct overwriting)
- Include **replay samples** from general data mixed into your training set
- Fine-tune for fewer epochs

### Overfitting

**Signs:** Training loss decreases but validation loss plateaus or increases.

**Mitigations:**
- Monitor validation loss every N steps (e.g. `eval_steps=50`)
- Enable **early stopping** (`load_best_model_at_end=True` in Hugging Face Trainer)
- Add `lora_dropout=0.05`
- Reduce `num_epochs`

### Underfitting

**Signs:** Both training and validation loss remain high; outputs are generic or off-task.

**Mitigations:**
- Increase LoRA rank `r` (try r=32 or r=64)
- Train for more epochs
- Increase `lora_alpha`
- Target more modules (`target_modules="all-linear"`)
- Check data quality and format consistency

### Evaluation Metrics

| Metric | Use Case |
|---|---|
| **Perplexity** | General language modelling quality |
| **ROUGE / BLEU** | Summarisation, translation tasks |
| **Exact Match / F1** | QA tasks |
| **Human evaluation** | Style, tone, coherence |
| **Task-specific benchmarks** | HumanEval (code), MedQA (medical), MMLU (knowledge) |
| **LLM-as-judge** | GPT-4 or Claude scoring outputs for quality |

### Saving and Merging Adapters

```python
# Save LoRA adapter only
model.save_pretrained("./lora_adapter")
tokenizer.save_pretrained("./lora_adapter")

# Load adapter for inference
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
model = PeftModel.from_pretrained(base_model, "./lora_adapter")

# Merge for clean deployment
merged = model.merge_and_unload()
merged.save_pretrained("./merged_model")
```

---

## 13. 🚢 Deployment Patterns

### Pattern 1: Serve Base + Adapter Separately

Load the base model once and **dynamically swap adapters** per request. Useful when serving multiple fine-tuned variants from one base model.

```python
# Dynamic adapter loading with PEFT
model.load_adapter("./adapter_v1", adapter_name="v1")
model.load_adapter("./adapter_v2", adapter_name="v2")
model.set_adapter("v1")  # Switch adapters at runtime
```

**Best for:** Multi-tenant serving, A/B testing fine-tuned variants.

### Pattern 2: Merge Adapter into Base

Fuse LoRA matrices into base weights before deployment. The result is a standard model with no PEFT overhead — full inference speed, simplest serving stack.

```python
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained("./final_model")
```

**Best for:** Production deployment where simplicity and speed matter.

### Pattern 3: Quantise Merged Model for Efficient Inference

After merging, convert the model to a compressed format for CPU/GPU inference:

| Format | Tool | Use Case |
|---|---|---|
| **GGUF** | llama.cpp | CPU inference, local deployment, Apple Silicon |
| **AWQ** | autoawq | Fast GPU inference with 4-bit weights |
| **GPTQ** | auto-gptq | GPU inference, widely supported |
| **INT8** | bitsandbytes | Moderate memory savings, good quality |

### Pattern 4: Serve with vLLM or llama.cpp

```bash
# Serve with vLLM (high-throughput GPU serving)
vllm serve ./merged_model --max-model-len 8192 --tensor-parallel-size 2

# Serve with llama.cpp (CPU / Apple Silicon)
./llama-server -m ./model.gguf --ctx-size 8192 --threads 8
```

---

## 14. ✅ Quick Decision Checklist for AI Engineers

| Question | If YES → | Recommended Approach |
|---|---|---|
| Does the model need to change its behaviour, style, or tone? | ✅ | **Fine-Tuning (SFT)** |
| Does the model need up-to-date or frequently changing factual knowledge? | ✅ | **RAG** |
| Do you need source citations in model outputs? | ✅ | **RAG** |
| Do you have <24GB VRAM and want to fine-tune a 7B model? | ✅ | **QLoRA** |
| Do you have >40GB VRAM and want maximum training quality? | ✅ | **LoRA (fp16 base)** |
| Do you have a cluster with hundreds of GPUs and need maximum accuracy? | ✅ | **Full SFT** |
| Do you need human-preference alignment (helpful, harmless, honest)? | ✅ | **SFT → DPO** (or RLHF) |
| Is your dataset tiny (<500 examples)? | ✅ | **Prompt engineering first; consider synthetic data augmentation** |
| Do you need the fastest possible inference with no latency overhead? | ✅ | **Merge LoRA adapter → deploy merged model** |
| Are you fine-tuning a 70B+ model on a single node? | ✅ | **QLoRA + DeepSpeed ZeRO Stage 3** |
| Do you need both style/tone AND live factual accuracy? | ✅ | **Fine-Tuning + RAG** (combined) |

---

## 15. 🛠️ Useful Libraries and Tools

| Tool | Purpose | Link |
|---|---|---|
| **transformers** | Model loading, training, tokenisation (Hugging Face) | [huggingface.co/docs/transformers](https://huggingface.co/docs/transformers) |
| **peft** | LoRA, QLoRA, and all PEFT methods | [huggingface.co/docs/peft](https://huggingface.co/docs/peft) |
| **trl** | SFT, DPO, RLHF, PPO training pipelines | [huggingface.co/docs/trl](https://huggingface.co/docs/trl) |
| **bitsandbytes** | 4-bit / 8-bit quantisation (NF4, int8) | [github.com/TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes) |
| **datasets** | Dataset loading, processing, streaming | [huggingface.co/docs/datasets](https://huggingface.co/docs/datasets) |
| **accelerate** | Multi-GPU / mixed-precision training abstraction | [huggingface.co/docs/accelerate](https://huggingface.co/docs/accelerate) |
| **deepspeed** | ZeRO sharding, large-scale distributed training | [deepspeed.ai](https://www.deepspeed.ai) |
| **unsloth** | 2–5× faster LoRA/QLoRA training, memory optimised | [github.com/unslothai/unsloth](https://github.com/unslothai/unsloth) |
| **axolotl** | Config-driven fine-tuning framework (wraps HF/PEFT) | [github.com/OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) |
| **llama-factory** | Web UI + CLI for fine-tuning 100+ model architectures | [github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) |
| **wandb** | Experiment tracking, loss curves, hyperparameter sweeps | [wandb.ai](https://wandb.ai) |
| **vllm** | High-throughput GPU inference serving | [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) |
| **llama.cpp** | CPU / Apple Silicon inference (GGUF format) | [github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) |

---

## 📚 Summary

```
Fine-Tuning Decision Tree
─────────────────────────
Need behaviour/style change?
  └─ YES → Fine-Tuning
       ├─ Unlimited compute?         → Full SFT
       ├─ >40GB VRAM?               → LoRA (fp16)
       └─ <24GB VRAM / large model? → QLoRA

Need live/dynamic knowledge?
  └─ YES → RAG (or Fine-Tuning + RAG)

Need preference alignment?
  └─ YES → SFT → DPO (or RLHF)
```

> **Final insight:** Modern production AI systems rarely use a single technique. The state-of-the-art pipeline is typically: **QLoRA SFT → DPO** for model training, combined with **RAG** for knowledge retrieval, served via **vLLM** with a merged, quantised model. Each component does what it does best.

---

*Last updated: 2025 · For corrections or contributions, open an issue or PR.*
