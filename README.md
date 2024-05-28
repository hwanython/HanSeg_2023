# DMX Solution to HaNSeg

## Overview

This repository contains scripts and tools for building a Docker algorithm, performing prediction on a test dataset, and calculating DSC (Dice Similarity Coefficient). Below are the steps to execute each of these tasks.

## Prerequisites

Make sure you have the following installed:
- Docker
- Python 3.9
- Necessary Python packages (can be installed using `requirements.txt` if provided)

## Steps

### 1. Build the Docker Algorithm

To build the Docker algorithm, run the following command in your terminal:

```sh
sh test.sh
```

### 2. Prediction on Test Dataset

To perform predictions on the test dataset, execute the following command:
```sh
python3 process.py
```

### 3. DSC Calculation

To calculate the Dice Similarity Coefficient (DSC), use the following command:
```sh
python3 calc_dice.py
```

## Contact
Email: jaehwanhan12@gmail.com