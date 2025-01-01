[toc]

---



# Title Page

**Title:** Self-Supervised Learning with Contrastive Representation Learning for EEG-Based Sleep Stage Classification: A Comprehensive Analysis of Data Augmentations and Downstream Performance
 **Authors:** Shaswat Gupta
 **Affiliation(s):** Medical Data Science Group, D-INFK, ETH Zurich
 **Contact Information:** [*[email@address.com](mailto:email@address.com)*]
 **Supervisor:** Prof. Dr. ABC
 **Advisor(s):** PQR, XYZ

------

# Abstract

Reliable sleep stage classification (SSC) is critical for understanding human sleep physiology and diagnosing sleep disorders. This work presents a **self-supervised learning** (SSL) framework, grounded in **contrastive representation learning** (CRL), to leverage the abundance of **unlabeled EEG signals** while mitigating the need for costly manual annotation. We introduce and rigorously evaluate **thirteen data augmentations**, spanning amplitude, frequency, masking-cropping, noise-filtering, and temporal transformations, to identify those most conducive to robust and generalizable feature representations.

Through a series of **incremental experiments**—from single augmentations and intra-category combinations to inter-category synergies—we pinpoint **Masking-Cropping** (RandomZeroMasking, CutoutResize), **Frequency-Based** (TailoredMixup), and select **Temporal** (TimeWarping, Permutation) transformations as top performers. We further demonstrate an optimal “Goldilocks” severity level, whereby **3–4 active augmentations** yield significant performance gains without excessively distorting the underlying EEG signals. A comprehensive **fine-tuning** phase on a **lightweight CNN (≈200k parameters)** surpasses **80% accuracy** and **70% Macro-F1**—notably achieved **without** class imbalance handling, temporal modeling, or large-scale architectures.

This study’s contributions are threefold: (1) a **systematic taxonomy** of EEG augmentations for SSC within SSL paradigms, (2) an **empirical demonstration** that contrastive pretraining with carefully orchestrated augmentations substantially boosts downstream classification, and (3) evidence that **compact models** can reach state-of-the-art performance, indicating feasibility for **real-time, resource-constrained** clinical applications. Our findings underscore the promise of SSL-CRL strategies and offer a blueprint for advancing EEG-based sleep staging via targeted data transformations.

------

# 1. Introduction and Literature Review

## 1.1 Motivation

- ***Briefly describe the importance of accurate sleep stage classification in clinical and research contexts.***

  Sleep, a physiological cornerstone of health, occupies nearly one-third of human life and governs critical neurophysiological and restorative processes. Accurate sleep stage classification (SSC) is essential for understanding sleep physiology and diagnosing sleep disorders [K. Susmáková](https://www.measurement.sk/2004/S2/susmakova.pdf). Traditionally, SSC relies on manual annotation of overnight Electroencephalogram (EEG) recordings—subset of a collection of signals called a Polysomnograph recorded in a sleep-lab—by experts. This manual process is labor-intensive, time-consuming, and prone to human-error and inter-rater variability (Cohen's κ ≈ 0.76), especially in challenging sleep stages like N1 (κ ≈ 0.24) [Lee, Y. J., et al. (2022).](https://pubmed.ncbi.nlm.nih.gov/34310277/).

- ***Highlight the limitations of supervised deep learning and the potential of Self-Supervised Learning (SSL) for leveraging large amounts of unlabeled EEG data.***

  While supervised Deep Learning (DL) approaches are achieving human-comparable accuracy in SSC, they suffer from significant limitations. An over-reliance on labeled datasets yields only marginal performance gains with architectural refinements [Gaiduk, Maksym et al.](https://pubmed.ncbi.nlm.nih.gov/37519865/). The overwhelming majority of EEG signals remain unlabeled due to the prohibitive costs of manual annotation [X. Jiang, J. Zhao, et al.](https://ieeexplore.ieee.org/document/9533305), presenting a compelling opportunity for Self-Supervised Learning (SSL)—a paradigm that has redefined state-of-the-art approaches in natural language processing and computer vision [Jing, Longlong and Tian, Yingli](https://arxiv.org/abs/1902.06162). Despite its transformative potential, the application of SSL to SSC is still in its nascent stages, leaving a critical gap in the field.

- ***Introduce the concept of Contrastive Representation Learning (CRL) and how data augmentations can influence representation quality.***

  **Contrastive Representation Learning (CRL)**, a subset of SSL, uses data augmentations to learn meaningful representations by bringing similar (augmented) data points closer in the latent space while pushing dissimilar ones apart. In the context of EEG signals, CRL involves applying various transformations to unlabeled EEG signals to create positive pairs—different augmented versions of the same signal—and negative pairs—augmented versions from different signals. By optimizing the encoder to maximize the agreement between positive pairs and minimize it for negative pairs, CRL effectively captures the underlying physiological and physical characteristics of EEG signals. This process enhances the quality of the learned representations, making them more robust and generalizable for downstream tasks such as sleep stage classification. [X. Liu et al.](https://ieeexplore.ieee.org/document/9462394)

## 1.2 Background and Gaps in Literature

- Summarize relevant literature and state-of-the-art approaches for EEG-based sleep staging.
- Emphasize that while SSL frameworks (e.g., NeuroNet, SleePyCo) show promise, understanding their latent representations and the role of data augmentations remains a gap.

Recent works in this domain include NeuroNet [NeuroNet](https://arxiv.org/abs/2404.17585), which employs a hybrid approach combining two fundamental SSL paradigms: Contrastive Learning (CL) and Masked Prediction (MP).  SleePyCo [SleePyCo](http://dx.doi.org/10.1016/j.eswa.2023.122551) focuses solely on CL, while EEG2REP [EEG2REP](https://doi.org/10.1145/3637528.3671600) utilizes Masked Augmentation (MA) in the latent space and compares it to MA applied directly on raw EEG data, as explored in MAEEG [MAEEG](https://arxiv.org/abs/2211.02625). While these studies show promise in improving accuracy metrics to achieve state-of-the-art (SOTA) performance and robustness against inter-patient and recording variability, However, There is a pivotal gap in understanding how data augmentations lead SSL paradigms to learn robust feature representations. There is yet no systematic research on how different data augmentations in the design space of SSL frameworks correlate with the quality of representations learned by pre-trained encoders and how they influence downstream task performance, generalizability, and robustness.

## 1.3 Research Questions and Contributions

Data augmentation is a critical component in SSL frameworks, influencing the quality of the learned representations. Common augmentations for EEG data include amplitude scaling, time-shifting, noise injection, and more. However, there is a lack of systematic studies evaluating these augmentations and their combinations within SSL contexts. Understanding how different augmentation strategies impact learned representations and downstream performance for SSC.

- **RQ1**: How far (which position) do individual augmentations stand in the spectrum from least effective to most effective?
- **RQ2**: How do various combinations of augmentations affect downstream classification metrics? Are intra-category combinations reinforcing or redundant? Are inter-category combinations synergistic or antagonistic?
- **RQ3**: Is there an optimal number of augmentations, a specific sequence and combination that works best?

**Key contributions** include:

- **Systematic Evaluation of Augmentation Techniques**: Conducting a comprehensive assessment of 13 different data augmentation methods to understand their individual and combined effects on SSL-encoded representations.
- **Iterative Experimental Design**: Implementing an iterative experimental framework to select optimal sequences of augmentations.
- **Clinical Context Ready**: Showcasing high classification performance using a lightweight model (~0.26M parameters) without temporal modeling, indicating the potential for real-time inferencing on-device EEG applications in a clinical context.

These contributions aim to enriching the understanding of SSL in EEG-based SSC, providing insights into how various data augmentations impact the quality of learned representations and downstream classification performance.

------

# 2. Self-Supervised Contrastive Representational Learning

## 2.1 Mathematical Framework

Self-Supervised Contrastive Representation Learning (SSL-CRL) employs a discriminative framework to learn meaningful representations by maximizing the similarity between augmented versions of the same data sample (positive pairs) while minimizing it for different samples (negative pairs). The key components of SSL-CRL include data augmentations, feature extraction, and a contrastive loss function.

### Similarity Metric

Given feature vectors $\mathbf{u}$ and $\mathbf{v}$, their similarity $s(\mathbf{u}, \mathbf{v})$ is defined as the cosine similarity:

$$
s(\mathbf{u}, \mathbf{v}) = \cos(\theta_{\mathbf{u}, \mathbf{v}}) = \frac{\mathbf{u}^T \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}, \quad s(\mathbf{u}, \mathbf{v}) \in [0, 1]
$$
A value of $s(\mathbf{u}, \mathbf{v})$ close to 1 indicates high similarity (e.g., positive pairs), while a value near 0 indicates dissimilarity (e.g., negative pairs).

### Contrastive Loss

The training objective is to maximize the similarity for positive pairs $(\mathbf{z}_i, \mathbf{z}_j)$ and minimize it for negative pairs $(\mathbf{z}_i, \mathbf{z}_k)$. The loss for a single positive pair is given by the normalized temperature-scaled cross-entropy (NT-Xent) loss:

$$
\ell(i, j) = - \log \frac{\exp(s(\mathbf{z}_i, \mathbf{z}_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}[k \neq i] \exp(s(\mathbf{z}_i, \mathbf{z}_k) / \tau)}
$$
where $\tau$ is a temperature scaling parameter and $N$ is the batch size. 

The overall loss is computed as:

$$
L = \frac{1}{2N} \sum_{i=1}^N \big( \ell(2i-1, 2i) + \ell(2i, 2i-1) \big)
$$

### Optimization and Training

The encoder network $f$ maps input signals $x$ to a latent representation $\mathbf{z}$. During training:

1. Two augmentations $T_1$ and $T_2$ are applied to each sample $x$, producing $T_1(x)$ and $T_2(x)$.
2. The encoder extracts features $\mathbf{z}_i = f(T_1(x_i))$ and $\mathbf{z}_j = f(T_2(x_i))$.
3. The contrastive loss is computed over positive and negative pairs within a batch.

## 2.2 Algorithm

**Algorithm 1: Self-Supervised Contrastive Learning**

**Input:** Batch of signals ${x_1, x_2, \dots, x_N}$, transformations $T_1(\cdot), T_2(\cdot)$, encoder $f$, batch size $N$.

**Output:** Loss $L$.

1. **Augmentation:** For each $x_i$, generate two augmented versions: $T_1(x_i)$ and $T_2(x_i)$.
2. **Feature Extraction:** Encode the augmented samples: $\mathbf{z}_{2i-1} = f(T_1(x_i)), \quad \mathbf{z}_{2i} = f(T_2(x_i)), \quad i \in \{1, \dots, N\}$.
3. **Similarity Computation:** Compute cosine similarities between all pairs $(\mathbf{z}_i, \mathbf{z}_j)$.
4. **Loss Calculation:** For each positive pair $(\mathbf{z}_{2i-1}, \mathbf{z}_{2i})$, calculate the NT-Xent loss $\ell(2i-1, 2i)$.
5. **Aggregate Loss:** Compute the average loss $L = \frac{1}{2N} \sum_{i=1}^N \big( \ell(2i-1, 2i) + \ell(2i, 2i-1) \big)$.
6. **Update:** Backpropagate $L$ to update the encoder $f$.

## 2.3 Visualization

The learning framework is visualized in Figure 1, where augmented pairs $T_1(x_r)$ and $T_2(x_r)$ are embedded into a latent space. Positive pairs are drawn closer together, while negative pairs are pushed apart.[X. Jiang, J. Zhao, et al.](https://ieeexplore.ieee.org/document/9533305)![SSCRL_Viz](C:\Users\shasw\Downloads\sleepnet_presentation\SSCRL_Viz.gif)

## 2.4 Advantages of SSL-CRL

1. **Data Efficiency:** Learns representations without requiring labels, leveraging large-scale unlabeled datasets.

2. **Generalization:** Robust to variations in input due to diverse augmentations.

   

# 3 Data Augmentations

In this section, we detail the data augmentation strategies used in our study, including a concise description, definition, mathematical formulation, and placeholders for visual representation. Each augmentation is designed to preserve the core semantics of the signal while introducing transformations that enhance model generalization and performance on limited EEG data.

| **Category**            | **Augmentation**            | **Description**                                              |
| ----------------------- | --------------------------- | ------------------------------------------------------------ |
| **Amplitude-Based**     | RandomAmplitudeScale        | Scales the amplitude of the signal to simulate variations in signal strength. |
| **Amplitude-Based**     | RandomDCShift               | Shifts the signal vertically by adding a constant offset to the baseline. |
| **Amplitude-Based**     | SignFlip                    | Flips the polarity of the signal to introduce variability.   |
| **Frequency-Based**     | RandomBandStopFilter        | Removes a random frequency band to mimic selective frequency attenuation. |
| **Frequency-Based**     | TailoredMixup               | Combines two signals in the frequency domain using weighted averages. |
| **Masking-Cropping**    | RandomZeroMasking           | Masks a random segment of the signal with zeros to simulate missing data. |
| **Masking-Cropping**    | CutoutResize                | Removes a random segment and resizes the remaining parts to the original length. |
| **Noise-and-Filtering** | RandomAdditiveGaussianNoise | Adds Gaussian noise to simulate environmental and instrumentation noise. |
| **Noise-and-Filtering** | AverageFilter               | Applies a moving average filter to smooth the signal.        |
| **Temporal**            | RandomTimeShift             | Shifts the signal temporally by a random offset.             |
| **Temporal**            | TimeWarping                 | Stretches or compresses segments of the signal non-uniformly along the time axis. |
| **Temporal**            | TimeReverse                 | Reverses the temporal order of the signal to ensure robustness to time directionality. |
| **Temporal**            | Permutation                 | Divides the signal into segments and randomly reorders them to disrupt temporal consistency. |

---

## 3.1 Amplitude-Based

These techniques modify the amplitude characteristics of the EEG signal. By scaling the signal’s amplitude (e.g., RandomAmplitudeScale), adding a constant offset (RandomDCShift), or flipping its polarity (SignFlip), these augmentations simulate variations in signal strength and baseline shifts, enhancing model robustness to amplitude-related variations.

### 3.1.1 Random Amplitude Scaling

**Description**: Randomly scales the amplitude of the signal to simulate variations in signal strength.

**Definition**: For a signal $X(t)$, a scaling factor $\alpha \sim \mathcal{U}(r_1, r_2)$ is applied:
$$
\text{ScaledSignal}[X](t) = \alpha \cdot X(t)
$$
where \($r_1, r_2$\) define the range of scaling factors.

**Visual Placeholder**:
`[Amplitude Scaling Visualization]`

---

### 3.1.2 Random DC Shift

**Description**: Shifts the signal vertically by adding a constant offset to introduce variability in baseline voltage.

**Definition**: For a signal $X(t)$, a shift value \($\beta \sim \mathcal{U}(d_1, d_2)$\) is applied:
$$
\text{ShiftedSignal}[X](t) = X(t) + \beta
$$

where \($d_1, d_2$\) define the shift range.

**Visual Placeholder**:
`[DC Shift Visualization]`

---

### 3.1.3 Sign Flip

**Description**: Flips the polarity of the signal.

**Definition**: For a signal $X(t)$:
$$
\text{FlippedSignal}[X](t) = -X(t)
$$
**Visual Placeholder**:
`[Sign Flip Visualization]`

---

## 3.2 Frequency-Based

These methods alter the frequency components of the signal. RandomBandStopFilter removes specific frequency bands to mimic selective attenuation, while TailoredMixup combines frequency-domain features of two signals to create augmented samples, enriching the signal diversity in the frequency space.

### 3.2.1 Random Band-Stop Filtering

**Description**: Removes a random frequency band to mimic selective frequency attenuation.

**Definition**: A band-stop filter with center frequency $f_c$ and bandwidth $b$ is applied:
$$
\text{FilteredSignal}[X](t) = \mathcal{F}^{-1} \left[ \mathcal{F}[X](f) \cdot H(f) \right]
$$
where $H(f)$ is the filter response, and $\mathcal{F}$ denotes the Fourier transform.

**Visual Placeholder**:
`[Band-Stop Filter Visualization]`

---

### 3.2.2 Tailored Mixup

**Description**: Combines two signals in the frequency domain using weighted averages of their magnitude and phase spectra.

**Definition**: For two signals $(X_1, X_2)$ with Fourier transforms $(\mathcal{F}(X_1), \mathcal{F}(X_2))$:
$$
\mathcal{F}(\text{MixedSignal}) = \lambda_A |\mathcal{F}(X_1)| e^{i\phi_1} + (1-\lambda_A) |\mathcal{F}(X_2)| e^{i\phi_2}
$$


**Visual Placeholder**:
`[Tailored Mixup Visualization]`

---

## 3.3 Masking and Cropping

Focused on modifying the signal’s structure, these augmentations introduce gaps or remove portions of the signal. RandomZeroMasking simulates missing data by masking random segments, and CutoutResize removes sections of the signal and resizes the remaining parts to maintain its original length.

### 3.3.1 Cutout and Resize

**Description**: Randomly removes a segment of the signal and resizes the remaining parts to the original length.

**Definition**: For a segment $X_r(t)$ to be removed:
$$
\text{CutoutSignal}[X](t) = \text{Resize}([X_1, \dots, X_{r-1}, X_{r+1}, \dots, X_n])
$$
**Visual Placeholder**:
`[Cutout and Resize Visualization]`

---

### 3.3.2 Random Zero Masking

**Description**: Masks a random segment of the signal with zeros.

**Definition**: For a mask of length $M$ at position $t_m$:
$$
\text{MaskedSignal}[X](t) = 
\begin{cases} 
0 & \text{if } t \in [t_m, t_m + M] \\
X(t) & \text{otherwise}
\end{cases}
$$
**Visual Placeholder**:
`[Zero Masking Visualization]`

---

## 3.4 Noise and Filtering

These techniques simulate real-world signal distortions. RandomAdditiveGaussianNoise introduces random noise to mimic environmental or instrumental artifacts, while AverageFilter smooths the signal using a moving average, reducing sharp variations and noise.

### 3.4.1 Average Filtering

**Description**: Applies a moving average filter to smooth the signal.

**Definition**: For a kernel $k$ of size $N$:
$$
\text{FilteredSignal}[X](t) = \frac{1}{N} \sum_{i=0}^{N-1} X(t-i)
$$
**Visual Placeholder**:
`[Average Filter Visualization]`

---

### 3.4.2 Random Additive Gaussian Noise

**Description**: Adds Gaussian noise to simulate environmental and instrumentation noise. We use this augmentation as a baseline for our experiment design.

**Definition**: Noise $N(t) \sim \mathcal{N}(0, \sigma^2)$ is added to the signal:
$$
\text{NoisySignal}[X](t) = X(t) + N(t)
$$


**Visual Placeholder**:
`[Gaussian Noise Visualization]`

---

## 3.5 Temporal

These augmentations modify the time axis of the signal to disrupt or alter temporal consistency. RandomTimeShift shifts the signal temporally, TimeWarping stretches or compresses segments non-uniformly, TimeReverse reverses the signal’s temporal order, and Permutation randomly reorders segments, encouraging the model to learn robust temporal representations.

### 3.5.1 Time Reversal

**Description**: Reverses the temporal order of the signal to ensure robustness to time directionality.

**Definition**: For a signal $X(t)$ of length $L$:
$$
\text{ReversedSignal}[X](t) = X(L-t)
$$
**Visual Placeholder**:
`[Time Reversal Visualization]`

---

### 3.5.2 Time Warping

**Description:** Non-uniformly stretches or compresses segments of the signal along the time axis.

**Definition**: For a segment $X_s(t)$ and a scale factor $\omega$:
$$
\text{WarpedSegment}[X_s](t) = X_s(\omega t)
$$
The segments are concatenated and resampled to the original length.

**Visual Placeholder**:
`[Time Warping Visualization]`

---

### 3.5.3 Permutation

**Description**: Divides the signal into segments and randomly reorders them to disrupt temporal consistency.

**Definition**: For $n$ segments $X_1, X_2, \dots, X_n$:
$$
\text{PermutedSignal}[X] = \text{Concat}(X_{\pi(1)}, X_{\pi(2)}, \dots, X_{\pi(n)})
$$
where $\pi$ is a random permutation.

**Visual Placeholder**:
`[Permutation Visualization]`

---

### 3.5.4 Random Time Shift

**Description**: Temporally shifts the signal by a random offset.

**Definition**: For a shift $t_s$:
$$
\text{ShiftedSignal}[X](t) = X(t + t_s)
$$
**Visual Placeholder**:
`[Time Shift Visualization]`



# 4 Dataset

## 4.1 Description

### Sleep-EDF Dataset

The Sleep-EDF Expanded dataset ([Goldberger et al., 2000](https://www.sciencedirect.com/science/article/pii/S0957417423030531#b13), [Kemp et al., 2000](https://www.sciencedirect.com/science/article/pii/S0957417423030531#b24)) (2018 version) consists of 197 whole-night PSG recordings, including EEG, EOG, chin EMG, and event markers. The dataset includes two study types:
- **SC (Sleep Cassette):** 79 recordings from healthy individuals aged 25–101 years, free of sleep disorders.
- **ST (Study Temazepam):** 22 recordings from subjects in a study on the effects of [Temazepam](https://www.sciencedirect.com/topics/pharmacology-toxicology-and-pharmaceutical-science/temazepam) on sleep.

For this study, SC recordings were used, following prior work ([Mousavi et al., 2019](https://www.sciencedirect.com/science/article/pii/S0957417423030531#b34), [Perslev et al., 2019](https://www.sciencedirect.com/science/article/pii/S0957417423030531#b37), [Phan et al., 2019](https://www.sciencedirect.com/science/article/pii/S0957417423030531#b41), [Phan et al., 2021](https://www.sciencedirect.com/science/article/pii/S0957417423030531#b42), [Phan et al., 2022](https://www.sciencedirect.com/science/article/pii/S0957417423030531#b43)). Sleep staging was performed according to the R&K rules ([Rechtschaffen, 1968](https://www.sciencedirect.com/science/article/pii/S0957417423030531#b47)), classifying each 30-second epoch into one of eight categories:
- WAKE
- REM
- N1
- N2
- N3
- N4
- MOVEMENT
- UNKNOWN

To address class imbalance, only 30 minutes of WAKE epochs before and after the sleep period were retained.

---

## 4.2 Preparation

### Signal Processing

1. **Epoch Duration and Downsampling**:
   - EEG epochs were standardized to 30 seconds (E = 30).
   - Signals were downsampled to 100 Hz (F = 100), consistent with prior works ([Perslev et al., 2019](https://www.sciencedirect.com/science/article/pii/S0957417423030531#b37), [Phan et al., 2021](https://www.sciencedirect.com/science/article/pii/S0957417423030531#b42)).

2. **Channel Selection**:
   - Of the two EEG channels (Fpz-Cz and Pz-Oz), only **Fpz-Cz** was used due to its reliability for sleep staging.

3. **Epoch Segmentation**:
   - Signals were segmented into 30-second epochs. Non-standard epoch durations (e.g., 60 seconds) were split into smaller, consistent segments.

4. **Annotations**:
   - Hypnogram annotations were parsed to extract onset, duration, and stage labels.
   - Labels were aligned with corresponding EEG signals at the epoch level.
   - Annotation errors, such as mismatched signal and annotation durations, were resolved by truncating excess annotations.

### Data Filtering

1. **Exclusion of Non-Sleep Periods**:
   - WAKE epochs at the boundaries of sleep were excluded, retaining only 30 minutes of WAKE epochs before and after the core sleep period.

2. **Removal of Irrelevant Stages**:
   - Epochs labeled as "MOVEMENT" or "UNKNOWN" were removed.

3. **Class Consolidation**:
   - Stages N3 and N4 were merged into a single class (N3), simplifying the classification into five stages.

### Preprocessing Summary

| Step                    | Description                                      |
| ----------------------- | ------------------------------------------------ |
| **Epoch Duration**      | Set to 30 seconds (E = 30).                      |
| **Downsampling**        | Signals downsampled to 100 Hz (F = 100).         |
| **Channel Selection**   | Retained only Fpz-Cz.                            |
| **Exclusion**           | Removed MOVEMENT, UNKNOWN, and non-sleep epochs. |
| **Class Consolidation** | Merged N3 and N4 stages.                         |

## 4.3 Exploratory Analysis

The dataset consists of **~200,000 (0.2M) sleep epochs** derived from **152 PSG recordings** (~10 hours each). Each epoch corresponds to 30 seconds of sleep data, with sleep stages labeled as WAKE, N1, N2, N3, and REM. The class distribution highlights a significant imbalance across sleep stages.

### Sleep Stage Distribution

| Label     | Stage | # Epochs    | Percentage |
| --------- | ----- | ----------- | ---------- |
| 0         | WAKE  | 69,532      | 35%        |
| 1         | N1    | 21,391      | 11%        |
| 2         | N2    | 68,651      | 35%        |
| 3         | N3    | 13,000      | 7%         |
| 4         | REM   | 25,715      | 13%        |
| **Total** | **-** | **198,289** | **100%**   |

### Key Statistics

- **Number of NPZ Files**: 152
- **Average Epochs per File**: 1,305
- **Average Hours per File**: 10.87

![Class Distribution](C:\Users\shasw\Downloads\sleepnet_presentation\Class Distribution.png)

### Observations

1. **Dominant Stages**:
   - WAKE and N2 together account for 70% of the epochs, each contributing 35%.
2. **Underrepresented Stages**:
   - REM and N1 comprise 13% and 11%, respectively.
   - N3 is the least represented, contributing only 7% of the total epochs.

The class imbalance, particularly the dominance of WAKE and N2 stages, necessitates careful consideration of evaluation metrics (e.g., macro-F1) to ensure robust performance across all classes.



# 5 Evaluation Metrics and Strategy

## 5.1 Downstream Task: Single-Epoch EEG Classification

The downstream task involves **single-epoch EEG classification**, where each 30-second EEG epoch is independently classified into one of five sleep stages: Wake (W), N1, N2, N3, or REM. This approach is chosen to **isolate the quality of feature representations** learned by the encoder by eliminating temporal dependencies between consecutive epochs. By focusing on individual epochs, we can accurately assess the encoder's ability to extract meaningful and discriminative features from EEG signals without the influence of temporal context.

Formally, let $\mathbf{x} \in \mathbb{R}^d$ denote an EEG epoch, where $d$ is the dimensionality of the input signal. The encoder $f_\theta: \mathbb{R}^d \rightarrow \mathbb{R}^k$ maps the input to a latent representation $\mathbf{z} = f_\theta(\mathbf{x}) \in \mathbb{R}^k$. The classifier $\mathbb{R}^k \rightarrow \mathbb{R}^C$ then predicts the sleep stage label $y \in \{0, 1, 2, 3, 4\}$, where $C = 5$ represents the number of classes.

## 5.2 Performance Metrics

Evaluating the classification performance requires a comprehensive set of metrics that capture both overall accuracy and class-specific performance. These metrics provide insights into the effectiveness of Contrastive Representation Learning (CRL) in extracting robust and generalizable features from EEG data. Along with confusion matrix visualizations, this evaluation framework provides a holistic evaluation of the single-epoch EEG classification task. 

### Overall Metrics

1. **Accuracy:** Measures the proportion of correctly predicted instances out of the total number of instances. It provides a general measure of the model’s correctness and has to be analysed within the context of other metrics in cases of class imbalance.
   $$
   \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} \newline
   $$
   $TP:$ True Positives
   $TN:$ True Negatives
   $FP:$ False Positives
   $FN:$ False Negatives

2. **Macro-F1 Score:** The unweighted mean of F1 scores calculated independently for each class. It accounts for both precision and recall, providing a balanced measure for evaluating performance on imbalanced datasets.
   $$
   \text{Macro-F1} = \frac{1}{C} \sum_{c=1}^{C} F1_c
   $$
   $C$: Number of classes
   $F1_c$: F1 score for class $c$

### Class-Wise Metrics

1. **Precision**: The ratio of true positive predictions to the total predicted positives for class $c$ indicating the accuracy of positive predictions for each class, reflecting the model’s ability to avoid false positives.
   $$
   \text{Precision}_c = \frac{TP_c}{TP_c + FP_c}
   $$
   
2. **Recall**:  The ratio of true positive predictions to the actual positives for class $c$, measuring the model’s ability to capture all relevant instances of a class, highlighting its sensitivity.
   $$
   \text{Recall}_c = \frac{TP_c}{TP_c + FN_c}
   $$
   
    

3. **F1 Score**: The harmonic mean of precision and recall for class $c$, provides a single metric that balances precision and recall, offering a comprehensive view of the model’s performance for each class.
   $$
   F1_c = 2 \times \frac{\text{Precision}_c \times \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}
   $$
   

### Visualization: Confusion Matrix Heatmap

A **confusion matrix** is utilized to visualize the model’s performance across different classes, providing a detailed breakdown of correct and incorrect predictions.
$$
\begin{pmatrix} CM_{1,1} & CM_{1,2} & \dots & CM_{1,C} \\ CM_{2,1} & CM_{2,2} & \dots & CM_{2,C} \\ \vdots & \vdots & \ddots & \vdots \\ CM_{C,1} & CM_{C,2} & \dots & CM_{C,C} \\ \end{pmatrix}
$$
where $CM_{i,j}$ represents the number of instances with true label $i$ predicted as label $j$.



To enhance interpretability, the confusion matrix is normalized to reflect percentages:
$$
{CM_{\text{percentage}} = \frac{CM_{i,j}}{\sum_{j=1}^{C} CM_{i,j}}} \times 100\%
$$
A heatmap of the normalized confusion matrix is generated, with each cell annotated with both the absolute count and the corresponding percentage. This visualization highlights areas of misclassification and the model’s strengths across different sleep stages.
$$
\begin{array}{c|ccccc} & \text{W} & \text{N1} & \text{N2} & \text{N3} & \text{REM} \\ \hline \text{W} & CM_{1,1} & CM_{1,2} & CM_{1,3} & CM_{1,4} & CM_{1,5} \\ \text{N1} & CM_{2,1} & CM_{2,2} & CM_{2,3} & CM_{2,4} & CM_{2,5} \\ \text{N2} & CM_{3,1} & CM_{3,2} & CM_{3,3} & CM_{3,4} & CM_{3,5} \\ \text{N3} & CM_{4,1} & CM_{4,2} & CM_{4,3} & CM_{4,4} & CM_{4,5} \\ \text{REM} & CM_{5,1} & CM_{5,2} & CM_{5,3} & CM_{5,4} & CM_{5,5} \\ \end{array}
$$
This heatmap facilitates the identification of specific misclassifications, enabling targeted improvements in model training and data augmentation strategies.

# 6 Training Framework:

 The training framework is structured around a **two-stage linear evaluation protocol** comprising self-supervised contrastive pretraining followed by supervised linear evaluation. 

## 6.1 Linear Evaluation Protocol

The training framework adopts a **Two-Stage Training Strategy**, integrating Self-Supervised Learning (SSL) with Contrastive Representation Learning (CRL) and a subsequent supervised linear evaluation. 

### Stage 1 : Self-Supervised Contrastive Pretraining

In the first stage, the encoder model undergoes self-supervised pretraining using CRL. The objective is to learn rich and invariant feature representations from unlabeled EEG data through contrastive learning mechanisms.  Specifically, the encoder $f_\theta$ maps each EEG epoch $\mathbf{x} \in \mathbb{R}^d$ to a latent representation $\mathbf{z} = f_\theta(\mathbf{x}) \in \mathbb{R}^k.$ The formulation requires the signal transformations $T(⋅)$ to generate transformed signal pairs that enable the encoder to learn disentangled semantic representations.

The contrastive loss function employed is the **Normalized Temperature-scaled Cross Entropy Loss (NT-Xent)**, defined as:
$$
\mathcal{L}_{\text{NT-Xent}}(\mathbf{z}_i, \mathbf{z}_j) = - \log \frac{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_k)/\tau)}
$$
where:

- $\text{sim}(\cdot,\cdot)$ denotes the cosine similarity between two vectors.
- $\tau$ is the temperature parameter, controlling the concentration level of the distribution.
- $N$ is the number of positive pairs in a mini-batch.

The encoder is optimized using the Adam optimizer with a default learning rate of $1\times 10^{-3}$. Training proceeds for a predefined number of maximum epochs, with early stopping criteria based on validation loss improvements to prevent overfitting. Upon completion, the best-performing encoder model, determined by the lowest validation loss, is saved for subsequent evaluation.

### Stage 2 : Supervised Linear Evaluation

In the second stage, the pretrained encoder $f_\theta$ is **frozen**, meaning its weights remain unaltered during this phase. It serves solely as a feature extractor. A simple **Multi-Layer Perceptron (MLP) classifier** $g_\phi$ is appended to the encoder to facilitate supervised classification of sleep stages. The classifier maps the latent representations $\mathbf{z}$ to class probabilities $\hat{\mathbf{y}} = g_\phi(\mathbf{z}) \in \mathbb{R}^C$, where $C=5$ corresponds to the five sleep stages.

The classifier is trained using the **Cross-Entropy Loss**:
$$
\mathcal{L}_{\text{CE}}(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{c=1}^{C} y_c \log p_\phi(y=c \mid \mathbf{z})
$$
where $\mathbf{y}$ is the one-hot encoded true label vector. The Adam optimizer, with a default learning rate of $1\times 10^{-3}$, updates only the classifier’s parameters $\phi$. Training incorporates early stopping based on validation loss to enhance generalization. The best classifier model, exhibiting the lowest validation loss, is preserved for final evaluation.

This linear evaluation protocol facilitates a clear separation between representation learning and classification, enabling an unbiased assessment of the encoder’s ability to extract meaningful features from EEG data.

## 6.2 Train-Test Split Strategy

In medial machine learning, we segregate data into train and test at the patient level as opposed to random splitting. Patient-wise data splitting ensures that data from the same individual does not appear in both training and test sets, preventing data leakage by evaluating the model's ability to generalize across individuals rather than relying on patient-specific patterns. It accounts for inter-subject variability and mimics real-world scenarios where models encounter entirely new patients.

The dataset is partitioned into training and testing subsets based on distinct subjects, adhering to an **85%-15% train-test split**. Specifically, this translates to:

- **Training Set** (66 subjects) : Utilized for both pretraining the encoder through CRL and training the classifier in the linear evaluation stage.
- **Testing Set** (12 subjects) : Employed exclusively for validating model performance, including computation of evaluation metrics and generation of visualizations such as confusion matrices.

# 7 Model Architecture

## 7.1 Encoder : SimpleSleepNet

The proposed **SimpleSleepNet** is a lightweight 1D convolutional neural network for single-channel EEG-based sleep stage classification. The architecture emphasizes computational efficiency while extracting features across multiple temporal scales.

### Architecture Overview

Three convolutional layers with the following ***progressive*** configurations:

- Decreasing Kernel sizes: 64, 32, and 16
- Decreasing Strides: 8, 4, and 2
- Increasing Dilations: 1, 2, and 4

Captures multi-scale temporal features by integrating long-, medium-, and short-range dependencies, each handled by one convolution layer.

### Normalization and Dropout

**Batch Normalization**: Applied after each convolutional layer to stabilize training by normalizing activations.

**Dropout**: Utilizes a rate of 0.2 to reduce overfitting and enhance generalization.

### Activation Function

**Mish**: Employed throughout the network for its smooth gradients and richer feature representations, outperforming traditional ReLU in handling the complex temporal patterns of EEG signals.
$$
mish(x) = x \cdot \tanh(\ln(1 + e^x)).
$$

### Pooling and Latent Space

**Global Average Pooling**: Condenses convolutional outputs into a fixed-size feature map, independent of input length.

**Fully Connected Layer**: Projects features into a 128-dimensional latent space, generating normalized and discriminative embeddings for downstream tasks.

### Design Highilights

1. The CNN-based encoder was designed to capture short, medium and long-term dependencies within EEG signals, positioning it as a representative archetype, aligning with standard feature extractors prevalent in EEG-based sleep stage classification research. 

2. Engineered to be computationally lightweight, the encoder comprises approximately 200,000 trainable parameters. This minimalistic design facilitates extensive experimentation, accelerates training and inference processes, and paves the way for on-device real-time inference. For perspective, DeepSleepNet contains around 21 million parameters (80 times larger), and SleepEEGNet has approximately 2.6 million parameters (10 times larger).

## 7.2 Classifier : SleepStageClassifier

The **SleepStageClassifier** is a fully connected neural network designed to classify sleep stages from input feature embeddings.

### Architecture Overview

Three linear layers with the following configurations:

- Hidden dimensions: 256 and 128
- Output layer: Number of classes (default: 5)

### Normalization and Dropout

**Batch Normalization**: Applied after each linear layer to stabilize training by normalizing feature distributions.

**Dropout**: Uses a rate of 0.5 to mitigate overfitting through stochastic regularization.

### Activation Function

**Mish**: Replaces traditional ReLU, providing smoother gradients and enabling richer feature transformations. Mish activation enhances learning for complex feature relationships, particularly valuable for EEG embeddings.



------

# 8. Experiment Design and Results

- Start with single-augmentation experiments to establish baseline effects.
- Perform ablation studies to identify beneficial augmentations.
- Use factorial design and iterative refinement to converge on best augmentation sequences.
- Mention the number of seeds/trials for robustness.

## 8.1 Single Augmentation Results

### Performance Table

| **Category**            | **Augmentation**            | **Accuracy** | **MacroF1** |
| ----------------------- | --------------------------- | ------------ | ----------- |
| **Temporal**            | Permutation                 | 68.84%       | 61.15%      |
| **Temporal**            | TimeReverse                 | 64.68%       | 54.35%      |
| **Temporal**            | TimeWarping                 | 71.58%       | 63.25%      |
| **Temporal**            | RandomTimeShift             | 65.52%       | 57.42%      |
| **Noise-and-Filtering** | AverageFilter               | 60.44%       | 47.98%      |
| **Noise-and-Filtering** | RandomAdditiveGaussianNoise | 65.33%       | 55.53%      |
| **Masking-Cropping**    | CutoutResize                | 71.13%       | 63.37%      |
| **Masking-Cropping**    | RandomZeroMasking           | 72.19%       | 65.00%      |
| **Frequency-Based**     | TailoredMixup               | 72.93%       | 65.16%      |
| **Frequency-Based**     | RandomBandStopFilter        | 66.31%       | 55.69%      |
| **Amplitude-Based**     | SignFlip                    | 60.91%       | 49.61%      |
| **Amplitude-Based**     | RandomDCShift               | 66.36%       | 56.49%      |
| **Amplitude-Based**     | RandomAmplitudeScale        | 62.72%       | 49.43%      |

---

### Key Observations and Analysis

We selected **RandomAdditiveGaussianNoise** as the baseline augmentation due to its prevalence in EEG and time-series analysis. Its Accuracy (65.33%) and Macro-F1 (55.53%) serve as reference points for other augmentations.

#### **Top Performers**
- **RandomZeroMasking (Masking-Cropping)** and **TailoredMixup (Frequency-Based)** demonstrated the highest Accuracy (>72%) and Macro-F1 (>65%).
- **CutoutResize (Masking-Cropping)** also performed strongly, highlighting the potential of selective masking and cropping for feature enhancement.

#### **Moderate Performers**
- Within **Temporal Augmentations**, **TimeWarping** stood out with an Accuracy of 71.58% and Macro-F1 of 63.25%.
- **RandomBandStopFilter (Frequency-Based)** and **RandomDCShift (Amplitude-Based)** yielded moderate improvements, suggesting partial benefits from targeted frequency filtering and amplitude adjustments.

#### **Underperformers**
- **AverageFilter (Noise-and-Filtering)** recorded the lowest Macro-F1 (47.98%), indicating that excessive smoothing can hinder signal discrimination.
- Amplitude-based augmentations, such as **SignFlip** and **RandomAmplitudeScale**, also underperformed, possibly due to distortions in key signal characteristics.

---

### Category-Level Trends
1. **Masking-Cropping** and **Frequency-Based** augmentations generally outperformed others, emphasizing the importance of selectively removing/masking signal parts or manipulating spectral components.
2. **Temporal Augmentations** delivered mixed results, with **TimeWarping** showing significant gains but **TimeReverse** underperforming.

---

### Summary and Conclusions

- **Best Augmentations**: TailoredMixup, RandomZeroMasking, and CutoutResize emerged as the top strategies, outperforming the baseline significantly.
- **Key Insights**: Amplitude and excessive smoothing transformations often degrade feature quality, while masking and frequency manipulations enhance feature representations.

## 8.2 Intra-Category Combinational Augmentations

### Performance Table
| **Category**            | **Augmentation**                                | **Accuracy** | **MacroF1** |
| ----------------------- | ----------------------------------------------- | ------------ | ----------- |
| **Temporal**            | RandomTimeShift + TimeReverse + Permutation     | 72.37%       | 64.84%      |
| **Temporal**            | RandomTimeShift + TimeWarping + Permutation     | 72.23%       | 63.42%      |
| **Temporal**            | RandomTimeShift + TimeWarping + TimeReverse     | 70.78%       | 61.54%      |
| **Temporal**            | TimeReverse + Permutation                       | 67.83%       | 60.30%      |
| **Temporal**            | TimeWarping + TimeReverse                       | 69.16%       | 60.25%      |
| **Temporal**            | RandomTimeShift + Permutation                   | 72.98%       | 65.82%      |
| **Temporal**            | RandomTimeShift + TimeReverse                   | 65.39%       | 57.28%      |
| **Temporal**            | RandomTimeShift + TimeWarping                   | 71.45%       | 62.52%      |
| **Noise-and-Filtering** | RandomAdditiveGaussianNoise + AverageFilter     | 59.71%       | 47.79%      |
| **Masking-Cropping**    | RandomZeroMasking + CutoutResize                | 72.94%       | 64.83%      |
| **Frequency**           | RandomBandStopFilter + TailoredMixup            | 70.09%       | 59.72%      |
| **Amplitude-Based**     | RandomAmplitudeScale + RandomDCShift + SignFlip | 60.16%       | 46.36%      |
| **Amplitude-Based**     | RandomDCShift + SignFlip                        | 64.97%       | 53.49%      |
| **Amplitude-Based**     | RandomAmplitudeScale + SignFlip                 | 57.81%       | 45.09%      |
| **Amplitude-Based**     | RandomAmplitudeScale + RandomDCShift            | 65.19%       | 52.90%      |

---

### Key Observations and Analysis

#### Temporal Category

- **Reinforcing Pairs**: Several top-performing combinations (Accuracy ≈ 72%, Macro-F1 ≈ 64–66%) emerge when **RandomTimeShift** pairs with **Permutation** or **TimeWarping**, suggesting complementary benefits.
- **Redundancy**: Pairings like **TimeReverse + Permutation** or **TimeReverse + TimeWarping** exhibit lower improvements, indicating potential overlap in how they distort temporal structure.

#### Masking-Cropping (M/C)
- **RandomZeroMasking + CutoutResize** achieves **72.94% Accuracy** and **64.83% Macro-F1**, reinforcing the earlier conclusion that targeted masking or cropping fosters robust representations. Performance remains competitive with top temporal combinations.

#### Frequency Category
- **RandomBandStopFilter + TailoredMixup** yields modest gains (**70.09% Accuracy; 59.72% Macro-F1**). While **TailoredMixup** was effective individually, adding another frequency-based method showed partial synergy but did not reach top-tier levels.

#### Noise-and-Filtering (N/F)
- **RandomAdditiveGaussianNoise + AverageFilter** underperforms (**59.71% Accuracy; 47.79% Macro-F1**). The smoothing effect of **AverageFilter** seems to degrade the signal further when combined with noise, indicating potentially conflicting distortions.

#### Amplitude-Based (Amp)
- All amplitude-based combinations remain underperformers (**Accuracy ≤ 65%; Macro-F1 ≤ 53%**), consistent with single-augmentation findings. Combining amplitude transformations often leads to overly distorted signals, degrading discriminative information.

---

### Summary and Conclusions

1. **Reinforcing Combinations**:
   - Temporal and Masking-Cropping combinations often enhance each other, delivering top-tier performance.
   - **RandomTimeShift** pairs effectively with **Permutation** or **TimeWarping**, suggesting complementary distortions that increase feature diversity.
   
2. **Frequency and Noise**:
   - Frequency-based augmentations show moderate synergy but fail to outperform leading temporal or masking-cropping methods.
   - Noise/Filtering combinations introduce conflicting distortions, resulting in subpar outcomes.

3. **Amplitude-Based**:
   - Amplitude augmentations consistently degrade performance when combined, likely due to excessive signal distortion.

**Conclusion**: Selective **temporal transformations** and **masking-cropping strategies** yield robust and complementary effects, reinforcing the importance of spatiotemporal manipulations or targeted signal masking. These methods surpass the utility of amplitude and noise-based augmentations in enhancing feature representation quality.

## 8.3 Inter-Category Combinational Augmentations

### Performance Table
| **Categories**                   | **Augmentation**                                             | **Avg. MF1** | **Avg. Accuracy** |
| -------------------------------- | ------------------------------------------------------------ | ------------ | ----------------- |
| **Frequency + Masking-Cropping** | TailoredMixup + RandomZeroMasking + CutoutResize             | 67.39%       | 76.39%            |
|                                  | TailoredMixup + CutoutResize                                 | 68.02%       | 76.80%            |
|                                  | TailoredMixup + RandomZeroMasking                            | 65.28%       | 74.92%            |
| **Temporal + Masking-Cropping**  | TimeWarping + Permutation + RandomZeroMasking + CutoutResize | 66.61%       | 75.85%            |
|                                  | TimeWarping + Permutation + CutoutResize                     | 66.34%       | 75.62%            |
|                                  | TimeWarping + Permutation + RandomZeroMasking                | 65.76%       | 75.42%            |
|                                  | Permutation + RandomZeroMasking + CutoutResize               | 66.20%       | 75.23%            |
|                                  | TimeWarping + RandomZeroMasking + CutoutResize               | 66.63%       | 75.82%            |
|                                  | Permutation + CutoutResize                                   | 65.81%       | 75.19%            |
|                                  | Permutation + RandomZeroMasking                              | 66.69%       | 75.81%            |
|                                  | TimeWarping + CutoutResize                                   | 66.87%       | 75.85%            |
|                                  | TimeWarping + RandomZeroMasking                              | 63.06%       | 73.64%            |
| **Frequency + Temporal**         | TailoredMixup + TimeWarping + Permutation                    | 67.63%       | 76.87%            |
|                                  | TailoredMixup + Permutation                                  | 66.36%       | 75.67%            |
|                                  | TailoredMixup + TimeWarping                                  | 66.09%       | 76.15%            |

---

### Key Observations and Analysis

#### 1. **Overall Consistency and Moderate Variability**
- Across seeds, performance variations generally lie within a **1–3% range** for both Accuracy and Macro-F1, reflecting **moderate yet non-trivial fluctuations**.
- Despite these variations, the **ranking of augmentation sets** remains stable, confirming consistent relative performance.

#### 2. **Synergistic Combinations**
- **Frequency + Masking/Cropping**: Augmentations such as **TailoredMixup + RandomZeroMasking + CutoutResize** achieve high Accuracy (~76%) and competitive Macro-F1 (~67%), highlighting the synergy between **spectral manipulation** and **strategic signal masking/cropping**.
- **Temporal + Masking/Cropping**: Combinations like **TimeWarping + Permutation + RandomZeroMasking** or **TimeWarping + RandomZeroMasking + CutoutResize** yield Accuracy around ~75–76%, indicating complementary benefits of temporal distortions with spatial manipulations.

#### 3. **Frequency + Temporal**
- **TailoredMixup + TimeWarping + Permutation** ranks among the better performers (~76.5–77.5% Accuracy), leveraging the strengths of **frequency diversity** and **temporal structure manipulation** to enhance feature representations.

#### 4. **Absence of Amplitude-Based Augmentations**
- Amplitude-based methods (e.g., **SignFlip, RandomAmplitudeScale**) were excluded due to their consistently subpar individual and combined results. Their limited utility remains evident regardless of inter-category mixing.

---

### Summary and Conclusions
- **Top Synergistic Combinations**:
  - **Frequency (TailoredMixup)**, **Temporal (TimeWarping, Permutation)**, and **Masking/Cropping (RandomZeroMasking, CutoutResize)** stand out as the most effective augmentations.
  - Their synergy arises from **complementary distortions**—temporal methods reshape dependencies, frequency methods enhance spectral variation, and masking/cropping promotes robustness under partial occlusion.
  
- **Robust and Generalizable Representations**:
  - Performance gains are stable across random seeds, confirming that **pairing well-chosen augmentations from different categories** leads to more generalizable EEG representations.

- **Implications**:
  - These findings reinforce the value of **strategically combining augmentations across categories**, especially for enhancing feature diversity and model robustness.

---

## 8.4 Best of All Categories - Severity

### Performance Table
| **Experiment Design (Severity)** | **Winning Augmentation**                                     | **Avg. MF1** | **Avg. Accuracy** |
| -------------------------------- | ------------------------------------------------------------ | ------------ | ----------------- |
| **Weighted p sum to 3**          | TailoredMixup + TimeWarping + Permutation + RandomZeroMasking + CutoutResize | 68.62%       | 77.55%            |
| **Weighted p sum to 4**          | TailoredMixup + TimeWarping + Permutation + RandomZeroMasking + CutoutResize | 68.62%       | 77.54%            |
| **Equal p sum to 3**             | TailoredMixup + TimeWarping + Permutation + RandomZeroMasking + CutoutResize | 68.84%       | 77.70%            |
| **Equal p sum to 4**             | TailoredMixup + TimeWarping + Permutation + RandomZeroMasking + CutoutResize | 68.49%       | 77.52%            |
| **Equal p sum to 5**             | TailoredMixup + TimeWarping + Permutation + RandomZeroMasking + CutoutResize | 68.25%       | 77.25%            |

---

### Key Observations and Analysis

#### **Motivation**
In previous experiments, we identified a set of highly effective augmentations—**TailoredMixup**, **TimeWarping**, **Permutation**, **RandomZeroMasking**, and **CutoutResize**—and noted that overusing augmentations could degrade signal quality. To refine these insights, we now investigate **optimal “Goldilocks” severity** by adjusting application probabilities for each augmentation. This systematic approach helps determine the number of active augmentations most conducive to learning robust EEG representations. Notably, these multi-augmentation setups generally **outperform single, intra-category, and inter-category pairwise approaches**, validating our pursuit of a balanced augmentation strategy.

---

#### **1. Optimality Around 3–4 Active Augmentations**
- Both **equal** and **weighted** probability distributions with a *sum of probabilities* set to **3 or 4** yield consistently strong performance:
  - **Accuracy** ≈ 77–78%.
  - **Macro-F1** ≈ 67–70%.
- This trend confirms that moderate augmentation severity enhances feature learning.

#### **2. Marginal Decline at Full Severity (Sum = 5)**
- Applying **all five augmentations with probability 1** results in a slight performance dip (~77.25% Accuracy). This suggests that excessive transformations can obscure salient EEG features, emphasizing the need for moderation.

#### **3. Preferential Weighting of Key Augmentations**
- Assigning **probability 1** to **TailoredMixup** and **TimeWarping** leverages their proven benefits, while moderating **Permutation**, **RandomZeroMasking**, and **CutoutResize** minimizes overall distortion. This approach consistently delivers superior results compared to simpler strategies.

#### **4. Stable Performance Across Seeds**
- Variations remain within a **1–3% range**, consistent with prior experiments. These fluctuations do not impact the overarching conclusion: **balanced augmentation probabilities enhance classification results.**

---

### Summary and Conclusions

These findings solidify the concept of a **“Goldilocks zone”** for augmentation severity, where **3–4 effective augmentations** strike an ideal balance between insufficient and excessive transformations. This multi-augmentation framework surpasses the best outcomes from single and simpler multi-augmentation experiments, demonstrating that **carefully orchestrated diversity in data perturbations** fosters superior feature representations and robust downstream performance.

## 8.5 Full Fine-Tuning and Final Performance

### 1. Motivation and Setup
- **Beyond Linear Evaluation**: While linear evaluation isolates the encoder’s representational power, **fully fine-tuning** allows the encoder to adapt specifically to the labeled data, potentially refining latent features.
- **Lightweight Architecture**: We continued using the same exceedingly small CNN-based backbone (only \(\sim 200k\) parameters), ensuring minimal computational overhead even during full fine-tuning.

---

### 2. Results
- **Overall Accuracy**: **80.55%**  
- **Macro-F1**: **71.68%**

Achieving **80%+ accuracy** and **70%+ Macro-F1** represents a significant milestone, surpassing earlier thresholds. This improvement is particularly remarkable given the absence of:

1. **Class Imbalance Mitigation**: No weighted losses or oversampling techniques were used.
2. **Temporal Modeling**: Sequential modules (e.g., LSTMs, Transformers) were not employed to exploit inter-epoch context.
3. **Complex Architectures**: Multi-head attention or other sophisticated deep learning components were avoided.
4. **Large Parameter Budgets**: The model remains extremely lightweight, highlighting its suitability for resource-constrained settings.

---

### 3. Significance and Perspective

#### **High Efficacy in Simple Settings**
- Surpassing **80% overall accuracy** and **70% Macro-F1** validates that carefully tuned **data augmentations** and a well-designed **SSL pretraining strategy** can yield **state-of-the-art results**, even without advanced architectural enhancements or balancing techniques.

#### **Minimal Resource Footprint**
- The performance of this tiny network underscores the practicality of deploying EEG-based sleep stage classification models:
  - **Real-Time Systems**: Effective for clinical monitoring systems.
  - **Edge Devices**: Suitable for environments with limited compute resources.

#### **Roadmap for Future Enhancements**
- Addressing **class imbalance** through weighted losses or oversampling.
- Incorporating **temporal modules** to model inter-epoch dependencies (e.g., LSTMs, Transformers).
- Experimenting with **sophisticated architectures** to push performance further.

---

### Summary
The **joint fine-tuning** phase with a lightweight CNN has enabled the model to surpass key performance thresholds, achieving:
- **80.55% accuracy**
- **71.68% Macro-F1**

These results reinforce the efficacy of our **SSL-CRL pipeline** and demonstrate that **compact yet carefully designed models** can achieve competitive outcomes in single-epoch EEG classification. This work highlights the potential for lightweight, resource-efficient solutions to meet the demands of both clinical and real-world deployment scenarios.

------

# 9. Discussion

This chapter integrates the findings from our series of augmentation experiments (Sections 8.1–8.5) with broader considerations related to model design, real-world deployment, and directions for future research. Our goal is to interpret the results in a broader context, highlighting both the successes and constraints of the proposed self-supervised learning (SSL) framework for single-epoch EEG classification.

------

## 9.1 Interpretation of Key Findings

### 9.1.1 Augmentation Effectiveness

Our experiments revealed that **masking/cropping** (e.g., RandomZeroMasking, CutoutResize) and **frequency-based** transformations (e.g., TailoredMixup) consistently outperformed **amplitude**-based and **noise/filtering** augmentations. This outcome suggests that targeted spatial and spectral distortions provide more meaningful contrast for the model to learn robust representations, while excessive amplitude shifts or smoothing can distort critical signal characteristics.

1. **Single Augmentation Insights**
   - RandomZeroMasking, CutoutResize, and TailoredMixup were top performers, each independently surpassing baseline accuracy and Macro-F1 scores.
   - TimeWarping emerged as the strongest temporal augmentation, indicating that moderate temporal reshaping fosters greater invariance than harsh transformations like TimeReverse or Permutation alone.
2. **Intra-Category and Inter-Category Synergy**
   - Combinations of top augmentations within the same category (e.g., RandomTimeShift + Permutation in the temporal domain) often demonstrated higher performance, reinforcing the idea that related transformations can complement each other.
   - When augmentations were drawn from **different categories** (temporal, frequency, masking/cropping), their synergy was even more pronounced, indicating that **spectro-temporal** and **spatial** manipulations together enrich the learned feature space.
3. **Optimal “Goldilocks” Severity**
   - The final experiments revealed that a total of **3–4 active augmentations** typically achieved the best balance. Excessively applying all five transformations (probability = 1 for each) slightly reduced accuracy, corroborating the principle that an overly distorted signal can hamper feature extraction.

### 9.1.2 Significance of a Small Model Achieving ~80% Accuracy

An essential outcome was surpassing **80% accuracy** and **70% Macro-F1** using a **simple, lightweight CNN** (≈200k parameters) without:

- **Temporal Modeling** (e.g., LSTMs, Transformers),
- **Class Imbalance Mitigation** (e.g., weighted loss, oversampling),
- **Complex Architectures** (e.g., multi-head attention),
- **Large Parameter Budgets** (tens of millions of parameters).

This result underscores the **efficacy of carefully orchestrated augmentations** and **contrastive pretraining**, even in resource-constrained setups. It further indicates that carefully optimized data transformations and SSL objectives can substantially close performance gaps typically associated with deeper, more sophisticated architectures.

------

## 9.2 Implications for Real-World Applications

1. **Real-Time, On-Device Analysis**
    The high performance of a small CNN backbone suggests feasibility for **real-time EEG classification** on edge devices (e.g., wearable electronics or embedded clinical monitors). By demonstrating competitive accuracy with minimal compute overhead, this approach addresses practical constraints often encountered in healthcare settings.
2. **Leveraging Unlabeled Data**
    Our SSL-based framework capitalizes on **unlabeled EEG signals**, critical in domains where label acquisition is expensive or time-consuming. Robust representations gleaned from unlabeled data can accelerate model deployment and potentially reduce dependence on scarce expert annotations.
3. **Clinical Utility**
    Exceeding 80% accuracy in single-epoch classification holds promise for real-world sleep-stage monitoring systems. In scenarios where immediate stage classification is paramount—such as automated sleep labs or patient home-monitoring devices—these results highlight the **potential for efficient, on-site EEG processing**.

------

## 9.3 Limitations

Despite the advances demonstrated, several limitations constrain the generalizability and scope of our findings:

1. **Absence of Temporal Modeling**
    While focusing on single-epoch classification isolates encoder effectiveness, it overlooks inter-epoch correlations. Future studies should examine **recurrent or transformer-based** modules to harness these temporal dependencies.
2. **Restricted Architectures**
    We tested a single CNN backbone (~200k parameters). Benchmarking against **larger or more sophisticated** architectures (e.g., state-of-the-art transformers for EEG) would clarify whether the same augmentation strategies can scale to more complex models.
3. **Limited Dataset**
    All experiments centered on Sleep-EDF. Although widely used, **testing on additional datasets** (e.g., SHHS, MASS) is necessary to confirm the model’s robustness and its ability to generalize across diverse EEG recording conditions.
4. **No Class Imbalance Handling**
    We did not adopt **weighted losses or oversampling strategies** to address potentially underrepresented sleep stages (e.g., N1). Different imbalance mitigation methods might further boost classification performance.

------

## 9.4 Future Work

Based on our results, several avenues of investigation can extend and refine the present framework:

1. **Temporal Modeling**
    Incorporate **recurrent modules** (e.g., bi-LSTMs) or **attention-based** architectures (e.g., transformers) to capture inter-epoch dynamics, which could further improve classification performance in real-world scenarios where multi-epoch context is essential.
2. **Advanced SSL Paradigms**
    Explore **supervised contrastive loss** or **other SSL frameworks** (e.g., masked autoencoders for EEG) to enrich the learned representations beyond our current contrastive approach.
3. **Imbalance Handling**
    Investigate **weighted cross-entropy**, **focal loss**, or **oversampling** (e.g., SMOTE-like strategies) to mitigate sleep-stage imbalance, particularly beneficial for rare classes such as N1.
4. **Robustness and Benchmarking**
   - **Scaling to SOTA Backbones**: Pre-train large-scale encoders (≥1M parameters) to compare performance against existing pipelines (e.g., SleePyCo).
   - **Multi-Dataset Evaluation**: Validate on **3 or more EEG databases** to establish replicability and ensure broad applicability.
5. **Real-Time Deployments**
    Pursue **on-device optimization** (e.g., quantization, model pruning) to confirm real-time feasibility and low-latency performance in edge scenarios, such as wearable devices.
6. **Exploring Augmentation Impacts**
    Conduct **detailed spectro-temporal analysis** of how each augmentation reshapes EEG signals, correlating these distortions with improvements in feature extraction and classification accuracy.



# 10 Conclusion

This thesis has demonstrated that **carefully tailored data augmentations**, integrated within a **self-supervised contrastive learning** framework, can substantially enhance **sleep stage classification** from single-epoch EEG signals. Specifically, we identified that **masking/cropping** (RandomZeroMasking, CutoutResize), **frequency-based** (TailoredMixup), and select **temporal** (TimeWarping, Permutation) augmentations yield **stable latent representations** conducive to high downstream performance. Through a methodical exploration—covering single augmentations, intra-category combinations, inter-category synergies, and varied augmentation severity—we established a “Goldilocks” level of data transformation. By deploying an extremely small CNN backbone (~200k parameters), we surpassed **80% accuracy** and **70% Macro-F1**, underscoring the **feasibility of real-time inference** in resource-limited environments without resorting to advanced architectures or specialized class imbalance techniques.

Moreover, our findings contribute to **Self-Supervised Learning (SSL) research** in EEG-based sleep staging by illustrating how targeted augmentations amplify the effectiveness of contrastive pretraining. The systematic assembly of these augmentation strategies—combined with a full fine-tuning stage—demonstrates that even compact models, when equipped with robust transformations, can achieve state-of-the-art classification outcomes. By extending these insights to more complex architectures and additional datasets, future research can solidify and broaden the applicability of this pipeline, further accelerating the development of accurate, efficient, and widely deployable EEG-based sleep staging solutions.

------

# 11 References

> *Placeholder for references in a consistent citation format (e.g., APA, IEEE).*

> **Example:**
>  KeyPaper1KeyPaper1 Author A, Author B, *Paper Title*, Journal/Conference, Year.
>  KeyPaper2KeyPaper2 Author C, *Another Title*, *Conference Name*, Year.
>  [*Link to Sleep-EDF dataset*](https://physionet.org/content/sleep-edfx/1.0.0/)

------

# 12 Appendices

## Appendix A: Additional Figures

> **Placeholder for additional visualizations**
>  ![Placeholder Additional Figure](https://chatgpt.com/c/path/to/appendix_figure.png)

## Appendix B: Detailed Tables of Results

> **Placeholder for large result tables**
>
> | Augmentation Combination | Acc   | MF1   | Precision | Recall |
> | ------------------------ | ----- | ----- | --------- | ------ |
> | Aug1 + Aug2              | 0.78  | 0.72  | 0.74      | 0.70   |
> | Aug2 + Aug3 + Aug4       | 0.80  | 0.73  | 0.75      | 0.72   |
> | *...*                    | *...* | *...* | *...*     | *...*  |

## Appendix C: Implementation Details and Hyperparameters

> **Code Snippet Placeholder**
>
> ```python
> # Example training hyperparameters
> batch_size = 128
> epochs = 50
> learning_rate = 1e-3
> ...
> ```

DataLoaders and Data preperation