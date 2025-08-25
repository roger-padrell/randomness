# Study on randomness
This project and repository contains a documented study on various types of random number generation, and the software used to obtain the results.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
## Table of contents

- [Details](#details)
  - [Generating](#generating)
  - [Predicting](#predicting)
  - [Displaying](#displaying)
- [Usage](#usage)
- [Repository structure](#repository-structure)
- [Important information](#important-information)
- [Requirements](#requirements)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Details
The sudied random number generators (*generator* for short) are:
    - default generator from a popular programming language (python)
    - cryptographically-secure generator ()
    - *human* random number generation
    - AI random number generation (Deepseek)
    
Each number stored to use as data will have between **8** and **128** digits from *0 to 9*.

### Generating
The numbers' digits will be generated one by one, to even the playing field (as humans already do it that way). The AI generator is an exception, as it would be much more difficult and expensive to do that.

### Predicting
For each generator, a simple neural network will be trained with all the generated numbers.

It will have **128 + 10** inputs, for the context in that generated number (0 to 9) and the appereance ratio of each possible digit (float 0 to 1). 

It will have 10 outputs for the possibility of the next digit (each a float from 0 to 1).

When training, a number will be trained for each digit (leaving the overflowing context to 0), so given a number `123456...`, it will train with:
- (0 * 127), 1
- (0 * 126), 1, 2
...

### Displaying
For each generator, the numbers will be plotted in these graphs:
    - General appereance ratio of each digit
    - Average movement in numbers
    
## Usage
Here is explained the usage of each software piece.

## Repository structure
<!-- https://tree.nathanfriend.com/?s=(%27options!(%27fancy8~fullPath!false~trailingSlash8~rootDot8)~9(%279%27README62LICENSE2STUDY*index676F6B6362GENERATORS02MODELS7-F-B-3-2DATA0%27)~version!%271%27)*24-*4generate.py*4model.h507FB32%5Cn3*crypto4%20%206.md7*human8!true9source!B*rngF*ai%01FB98764320-* -->
```
.
├── README.md
├── (repo files...)
├── STUDY/
│   ├── index.md
│   ├── human.md
│   ├── ai.md
│   ├── rng.md
│   └── crypto.md
├── GENERATORS/
│   ├── human
│   ├── ai
│   ├── rng
│   └── crypto
├── MODELS/
│   ├── human/
│   │   ├── generate.py
│   │   └── model.h5
│   ├── ai/
│   │   ├── generate.py
│   │   └── model.h5
│   ├── rng/
│   │   ├── generate.py
│   │   └── model.h5
│   └── crypto/
│       ├── generate.py
│       └── model.h5
└── DATA/
    ├── human
    ├── ai
    ├── rng
    └── crypto
```

## Important information
The maximum file size at GitHub is **100 MB**, but it's better to keep them down from **50 MB** (and even better, **25 MB**).
Use `sizecheck.py` before a `git add` to ensure that no file is larger than accepted. 

## Requirements
This project requires some dependencies, mainly in python. You can install them with any package manager for the specific programming language (for python I'm using `uv` and a `uv venv`).
Python requirements: 
```
alive-progress torch
```