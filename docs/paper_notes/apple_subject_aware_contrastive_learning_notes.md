# Apple Subject-Aware Contrastive Learning Notes

Paper:

- `Subject-Aware Contrastive Learning for Biosignals`
- authors: `Joseph Y. Cheng, Hanlin Goh, Kaan Dogrusoz, Oncel Tuzel, Erdrin Azemi`
- affiliation listed on paper: `Apple`
- local source: `/Users/home/thesis/docs/thesis_progress_report/papers/Apple_paper.pdf`
- first-page title extracted from PDF text: `Subject-Aware Contrastive Learning for Biosignals`

## Scope Of These Notes

These notes summarize the method described in the paper, with emphasis on:

- the self-supervised contrastive setup
- how subject-awareness is introduced
- the encoder / projector design
- augmentation strategies for biosignals
- downstream evaluation setup
- architectural choices that may matter for later SSL comparisons

These are descriptive notes, not recommendations.

## High-Level Idea

The paper proposes a contrastive self-supervised learning approach for biosignals such as:

- `EEG`
- `ECG`

The main claim is:

- ordinary instance-discrimination contrastive learning is not enough for small-subject biosignal datasets
- inter-subject variability can dominate the learned representation
- explicitly modeling or suppressing subject identity improves downstream performance

So the paper adds subject-awareness in two different ways:

1. `subject-specific SSL`
2. `subject-invariant SSL`

This is the central methodological contribution.

## Core SSL Framework

### Base Contrastive Setup

For each example `x_i`:

- two transformations are sampled: `T1(x_i)` and `T2(x_i)`
- an encoder `G` maps each transformed view to a latent representation
- a projection network `F` maps the latent to the contrastive space

The paper uses:

- online branch: `q_i = F(G(T1(x_i)))`
- momentum target branch: `k_i = F_k(G_k(T2(x_i)))`

The target networks are updated with momentum, following a MoCo-style design.

Both `q_i` and `k_i` are:

- `L2` normalized

The contrastive objective is:

- symmetric only in the sense of positive-pair matching between transformed views
- implemented with `InfoNCE`
- computed against many negatives, including queued examples from previous batches

The paper writes the loss in the standard form:

- positive pair: `(q_i, k_i)`
- negatives: `k_j` from other examples
- similarity: inner product
- temperature: learnable `tau`

So the base learning problem is:

- make two transformed versions of the same time-series window similar
- make other windows dissimilar

### Queue / Negative Structure

The paper explicitly says:

- momentum updates are used for `G_k` and `F_k`
- negatives from previous batches are reused
- the key history size is `24k`
- momentum is `0.999`

This is much closer to:

- `MoCo`

than to:

- pure in-batch `SimCLR`

## Subject-Aware Variants

The paper argues that subject identity is a major nuisance factor in biosignals.

This leads to two different modifications of the base SSL objective.

### Subject-Invariant SSL

This variant adds an adversarial subject classifier on top of the encoder representation.

The setup is:

- a classifier `C_sub` predicts subject identity from encoder features
- `C_sub` is trained to identify the subject
- the encoder is trained to confuse `C_sub`

So the optimization becomes:

- keep the contrastive objective
- add a regularizer that discourages subject-identifying information in the embedding

The paper describes this as adversarial training for subject invariance.

Important implementation detail:

- for multi-session data, different sessions can be treated as different “subjects” in this formulation

That point is especially relevant to Utah-array work, where session setup and day-to-day drift are a major issue.

### Subject-Specific SSL

This variant changes the negative sampling distribution instead of adding adversarial regularization.

The idea is:

- estimate the contrastive noise distribution within a single subject
- focus the loss on distinguishing different time segments from the same subject
- avoid using cross-subject differences as the main contrastive signal

In other words:

- negatives are constrained by subject identity
- the model is pushed to separate temporal structure within-subject rather than relying on between-subject variability

This is probably the most directly relevant idea in the paper for your current Utah SSL experiments.

## Data Augmentations

The paper places heavy emphasis on augmentation design for biosignals.

The transformations considered are:

- temporal cutout
- temporal delay
- additive Gaussian noise
- bandstop filtering
- signal mixing
- spatial rotation (`EEG` only)
- spatial shift (`EEG` only)
- sensor dropout
- sensor cutout

The paper’s qualitative conclusion is:

- temporal transformations mattered most
- temporal cutout was the strongest single augmentation on the EEG benchmark
- temporal delay was also strong
- signal mixing helped
- spatial perturbations helped less

One interesting appendix detail is:

- replacing the cutout region with noise worked better than replacing it with zeros or another signal

The paper’s interpretation is:

- stronger augmentations may force better representations

## Encoder / Projector Architecture

The paper does not propose a novel sequence backbone.

Instead, it uses:

- `1D` ResNet encoders for both EEG and ECG
- a small MLP projection head for the contrastive space

### Encoder `G`

The encoder is application-specific:

- `EEG`: a `1D` ResNet-style convolutional network
- `ECG`: also a `1D` convolutional encoder, with different dimensions

For the EEG experiments, the appendix states:

- input: `64` channels by `320` samples
- sampling rate: `160 Hz`
- window length: `2 sec`
- output embedding size: `256`
- encoder parameter count: about `288k`

The EEG encoder uses:

- `Conv1D`
- residual blocks
- `MaxPool`
- `ELU`
- batch normalization

### Projection Head `F`

The projection network is a fully connected network with:

- `4` layers
- hidden width `128`
- output width `64`

This projector is used only during self-supervised training.

## Experimental Regime

The paper evaluates on two biosignal tasks:

1. `EEG` motor imagery classification
2. `ECG` arrhythmia / rhythm classification

The learned encoder is evaluated in two ways:

- linear probe on frozen features
- supervised fine-tuning from the SSL initialization

This makes the paper relevant as a representation-learning reference rather than just an end-task classifier paper.

## EEG Setup

For the EEG experiments:

- dataset: PhysioNet motor imagery dataset
- subjects used for SSL: `90`
- held-out evaluation subjects: `16`
- window length: `2 sec`
- embedding dimension: `256`
- batch size: `400`
- self-supervised training steps: `270k`

The downstream tasks are:

- `2`-class imagined left-vs-right fist
- `4`-class imagined left fist / right fist / both fists / both feet

The paper evaluates both:

- intersubject testing
- intrasubject testing

## ECG Setup

For the ECG experiments:

- dataset: MIT-BIH Arrhythmia Database
- input window: `704` samples, about `1.96 sec`
- embedding dimension: `256`
- encoder parameter count: about `985k`
- batch size: `1000`
- self-supervised training steps: `260k`

The downstream tasks are:

- beat classification
- rhythm classification

The dataset is highly class-imbalanced, and the paper notes that subject information can be partly entangled with label structure here.

That becomes important when interpreting the effects of subject invariance.

## Main Results

### EEG

The paper reports:

- no-augmentation SSL was not useful
- augmentation-based SSL substantially improved frozen-feature linear-probe accuracy over random features
- subject-invariant and subject-specific training both reduced subject-identification accuracy
- subject-invariant SSL helped most when labels were scarce or when calibration-like within-subject transfer mattered

The paper’s interpretation is:

- when subject variability is a nuisance, reducing it improves representation quality

### ECG

The paper reports a more nuanced picture:

- subject-invariant SSL gave the best frozen-feature representations for the linear-probe evaluations
- but the best fine-tuning initialization for beat classification came from subject-specific SSL
- too much subject invariance could remove information that was actually useful for the downstream task

So the lesson is not:

- “always remove subject information”

Instead it is:

- subject information can be nuisance or signal depending on the task
- the right degree of subject-invariance is task-dependent

This is an important caution for Utah-array SSL as well.

## What Seems Most Relevant For Utah SSL

Several ideas from this paper look highly transferable.

### 1. Same-subject / same-session negative structure matters

The subject-specific SSL variant is conceptually very close to:

- same-session negatives
- within-subject contrastive discrimination

This is directly relevant to your current concern that unrestricted negatives may let the model solve the task with dataset/session identity instead of phoneme structure.

### 2. Subject-invariance can help, but only if subject identity is truly nuisance

The adversarial subject classifier is a strong version of:

- day/session invariance regularization

That may be useful for Utah arrays, but the ECG results warn that this can also throw away task-relevant information.

### 3. Augmentation design matters a lot

The paper did not treat contrastive learning as architecture alone.

Instead:

- augmentation strength and type materially changed downstream performance
- temporal perturbations were especially important

That aligns well with your current experiments, where the difficulty and usefulness of the SSL objective depend heavily on how the two views are related.

### 4. Representation quality should be judged downstream, not by SSL retrieval alone

The paper evaluates:

- frozen linear probes
- fine-tuning

not just the contrastive loss itself

That is the right lesson for your current notebook work as well:

- a high SSL top-1 retrieval score is not enough
- the real question is whether phoneme decoding improves

## Main Differences From Your Current Utah SSL Setup

This paper is informative, but it is not a direct template.

Major differences:

- the encoder is convolutional, not `S5`
- the method is window-level instance discrimination, not patch-level sequence modeling
- the datasets are `EEG` and `ECG`, not intracortical speech data
- the downstream tasks are classification, not phoneme-sequence decoding with `CTC`
- the paper uses subject-aware contrastive variants, whereas your current setup has not yet added explicit session-aware negatives or adversarial invariance

So I would treat it as:

- a useful contrastive-learning reference for nuisance structure

not as:

- a close architectural precedent for speech-decoding SSL

## Short Summary

This paper is an early biosignal contrastive-learning paper built around:

- a `MoCo`-style momentum contrastive framework
- biosignal-specific augmentations
- subject-specific contrastive loss
- adversarial subject-invariant regularization
- downstream evaluation through linear probes and fine-tuning

The strongest transferable idea for Utah-array SSL is probably not the exact encoder or task, but the insight that:

- subject/session identity can dominate biosignal SSL
- changing the negative structure or explicitly regularizing invariance can materially change what the representation learns

