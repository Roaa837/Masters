# 🎬 Marketing Video Summarization using LLMs and Diffusion Models

## Overview
Online advertisements are often skipped after only a few seconds, especially on platforms such as YouTube. Because of this, viewers may miss the main purpose or message of the ad. The aim of this project is to automatically summarize marketing videos so that viewers can still understand the core idea of an advertisement, even if they do not watch it in full.

This research explores a hybrid video summarization framework that combines the semantic understanding of Large Language Models (LLMs) with the diversity and robustness of diffusion models.

---

## Motivation
Recent video summarization techniques show that diffusion models are highly effective at generating diverse and representative summaries. They behave like strong annotators and summarizers, but they have an important limitation: they do not fully understand the semantic meaning of the video content.

To address this issue, this project integrates LLMs into the summarization pipeline. The idea is to use LLMs for semantic understanding and diffusion models for efficient and diverse summary generation.

---

## Project Goal
The main objective of this thesis is to summarize marketing videos in a way that preserves their purpose, message, and important content while reducing the amount of video a viewer needs to watch.

---

## Proposed Approach
The project follows two main summarization approaches:

1. **LLM-based summarization**
2. **Diffusion-based summarization**

These two approaches are then compared using multiple evaluation metrics.

---

## Pipeline

### 1. Video Preprocessing
Each video is first processed into multiple components:

- **Shot segmentation** to divide the video into meaningful shots
- **Keyframe extraction** to represent each shot visually
- **Caption generation** using **BLIP** to describe the visual content of keyframes
- **ASR (Automatic Speech Recognition)** to extract spoken content from the video

At the end of this stage, each shot contains:
- its start and end timestamps
- its keyframe
- its visual caption
- its speech transcript

This creates a multimodal representation of the video.

---

### 2. LLM-Based Summarization
In the first approach, the LLM is given:

- ASR text
- visual captions
- keyframes
- shot information

The LLM analyzes the semantic importance of each shot and assigns a score to every shot. After scoring, **Knapsack Optimization** is used to select the best subset of shots under a summary length constraint.

Since no benchmark or annotated dataset exists for this specific marketing video dataset, the LLM-based summaries are treated as a **pseudo ground truth** for comparison.

---

### 3. Diffusion-Based Summarization
The second approach uses a diffusion model for shot importance prediction.

#### Training
The diffusion model was trained on public video summarization datasets:

- **TVSum**
- **SumMe**

The model was trained using:
- captions
- shot information
- shot importance scores

The purpose was to learn how to predict the importance of video shots.

#### Testing
After training, the diffusion model was tested on a custom dataset consisting of:

- **50 marketing advertisements collected from YouTube**

The diffusion model predicts shot importance scores for the ads, and then **Knapsack Optimization** is again used to select the best shots for the final summary.

---

## Dataset
### Training Datasets
- TVSum
- SumMe

### Custom Dataset
- 50 marketing videos collected from YouTube

This custom dataset has not been previously annotated or tested in this context, which makes the task more challenging.

---

## Evaluation Metrics
The two approaches were compared using the following metrics:

- **Diversity**
- **Coverage**
- **Similarity**
- **Representativeness**

These metrics were used to measure how well the selected summary captures the original video content while avoiding redundancy.

---

## Results
The experimental results showed that the **diffusion model achieved nearly the same evaluation performance as the LLM-based approach**, while using only about **1/8 of the number of shots** selected by the LLM.

This suggests that the diffusion model can generate more compact summaries while preserving similar quality.

---

## Key Contribution
The main contribution of this work is a hybrid framework for marketing video summarization that combines:

- the **semantic reasoning power of LLMs**
- the **diversity and compactness of diffusion models**

This is particularly useful for domains such as advertising, where short and meaningful summaries are important.

---

## Tools and Models Used
- **Python**
- **PyTorch**
- **BLIP** for image captioning
- **ASR** for speech extraction
- **LLMs** for semantic shot scoring
- **Diffusion Models** for summary generation
- **Knapsack Optimization** for shot selection

---

## Future Work
Possible future improvements include:

- integrating image embeddings into training more effectively
- improving multimodal fusion between text, image, and audio
- expanding the dataset beyond 50 advertisements
- evaluating with human judgment in addition to automatic metrics

---

## Conclusion
This thesis addresses the problem of summarizing marketing videos so that viewers can quickly understand the purpose of an ad even if they skip it early. By combining LLMs and diffusion models, the project shows that it is possible to produce compact and meaningful summaries that preserve the key content of the original advertisement.

---
