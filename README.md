Machine Learning for Big Data – Assignment 2
Topic: Min-Hashing, Jaccard Similarity & Locality-Sensitive Hashing (LSH)
Name: Vishwanath Singh
Roll No: M24CSE030
Course: CSL7110 – Machine Learning for Big Data

------------------------------------------------------------
1. Overview
This repository contains the complete implementation for Assignment 2 of the course Machine Learning for Big Data (CSL7110).

The work includes:
- Construction of k-grams
- Exact Jaccard similarity
- MinHash approximation
- LSH banding and probability computation
- MinHash & LSH on the MovieLens 100k dataset

------------------------------------------------------------
2. Repository Structure
Dataset/
Screenshots/
M24CSE030_CSL7110_Assignment2.ipynb
M24CSE030_CSL7110_Assignment2.pdf
CSL7110_Assignment_2.pdf
README.txt

------------------------------------------------------------
3. Requirements
pip install numpy pandas tqdm matplotlib

MovieLens dataset: download from https://grouplens.org/datasets/movielens/  
Extract 'ml-100k' into project folder.

------------------------------------------------------------
4. Implementation Steps

STEP 1 — Create k-Grams
- 2-gram (character)
- 3-gram (character)
- 2-gram (word)
Outputs: distinct k-gram counts + exact Jaccard similarity (18 values)

STEP 2 — MinHashing
Hash: h(x) = (a*x + b) mod m, m = 100003
Experiments: t = 20, 60, 150, 300, 600
Findings: t = 150 best accuracy–time tradeoff

STEP 3 — LSH
Goal: detect similarity ≥ 0.7
t = 160, r = 20, b = 8
Collision probability curve: f(s) = 1 - (1 - s^b)^r

STEP 4 — MinHash on MovieLens
- 943 users, 1682 movies
- Exact Jaccard similarity
- MinHash with t = 50, 100, 200
- False positives/negatives across 5 runs

STEP 5 — LSH on MovieLens
Thresholds: 0.6 and 0.8
Tested configurations:
  50 hashes → r=5, b=10
  100 hashes → r=5, b=20
  200 hashes → r=5,b=40 and r=10,b=20

------------------------------------------------------------
5. Running the Notebook

Jupyter:
jupyter notebook M24CSE030_CSL7110_Assignment2.ipynb

------------------------------------------------------------
6. Summary of Results
- k-grams and exact similarity calculated successfully
- MinHash approximation highly accurate for t ≥ 150
- LSH separates high/low similarity documents effectively
- MovieLens MinHash: FP reduces with increasing t
- MovieLens LSH: efficient candidate generation, good separation at τ=0.6 and τ=0.8

------------------------------------------------------------
7. Detailed Implementation Steps (How the Work Was Executed)
------------------------------------------------------------

1. Loading and Preprocessing the Dataset
- Loaded D1–D4 text files, converted to lowercase, removed newlines.
- Ensured only 27 valid characters (a–z + space).

2. Creating k-Grams
- Implemented 2-gram (character), 3-gram (character), and 2-gram (word).
- Used sliding windows and stored unique k-grams in sets.
- Computed Jaccard similarity for 18 document pairs.

3. Exact Jaccard Similarity
- J(A,B) = |A ∩ B| / |A ∪ B|
- Calculated for all pairs under all k-gram types.

4. MinHash Implementation
- Used hash family: h(x) = (a*x + b) mod 100003.
- Generated signatures for t = 20, 60, 150, 300, 600.
- Estimated Jaccard using signature match ratio.
- Determined t = 150 as optimal balance.

5. Locality Sensitive Hashing (Documents)
- Used t = 160, r = 20 bands, b = 8 rows.
- Applied LSH formula: f(s) = 1 - (1 - s^b)^r.
- Computed collision probabilities for all 6 document pairs.

6. MovieLens Exact Jaccard
- Loaded user → movie mappings from u.data.
- Computed exact Jaccard for all user pairs.
- Listed pairs with similarity ≥ 0.5.

7. MovieLens MinHash
- Generated MinHash signatures for t = 50, 100, 200.
- Found estimated similar pairs.
- Calculated false positives & false negatives over 5 runs.

8. MovieLens LSH
- Tested configurations:
  * 50 hashes → r=5, b=10
  * 100 hashes → r=5, b=20
  * 200 hashes → r=5,b=40 and r=10,b=20
- Generated candidate pairs using band hashing.
- Computed FP/FN for thresholds 0.6 and 0.8.
