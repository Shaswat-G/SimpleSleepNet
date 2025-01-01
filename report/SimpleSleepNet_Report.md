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

> *Placeholder for short summary of the work, including objectives, methods, key findings, and conclusions.*

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

Present tables and charts showing performance (Acc, MF1) for each augmentation individually. Identify top-performing single augmentations.

> **Example Table Placeholder**
>
> | Augmentation         | Accuracy | Macro-F1 |
> | -------------------- | -------- | -------- |
> | RandomAmplitudeScale | 0.76     | 0.71     |
> | RandomDCShift        | 0.75     | 0.69     |
> | *...*                | *...*    | *...*    |

## 8.2 Intra-Category Combinational Augmentations

Reinforcing / Redundant intra-category combinations.

## 8.3 Inter-Category Combinational Augmentations

Synergistic / Antagonistic inter-category combinations with different random seeds.

## 8.4 Best of all Categories

## 8.5 Fully Fine Tune with best category

------

# 9. Discussion

## 9.1 Interpretation of Key Findings

- Why certain augmentations yield better latent representations.
- The significance of small model size achieving ~80% accuracy and ~70% MF1 without temporal modeling.

## 9.2 Implications for Real-World Applications

- Potential for real-time, on-device EEG analysis due to small model size and robust representations.
- Addressing labeling costs by unlocking value from unlabeled data.

## 9.3 Limitations

- Lack of temporal modeling in this study.
- Limited datasets (focus on Sleep-EDF); generalization to other datasets untested.

## 9.4 Future Work

- Incorporate temporal modeling and test augmentations under recurrent or transformer-based architectures.
- Explore supervised contrastive loss and other SSL paradigms.
- Investigate advanced imbalance handling techniques and multimodal EEG signals.

------

# 10 Conclusion

- Summarize the key outcomes: certain augmentation strategies significantly improve downstream classification, stable latent representations, and the feasibility of real-time inference.
- Reiterate the contributions to SSL research in EEG-based sleep staging.

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