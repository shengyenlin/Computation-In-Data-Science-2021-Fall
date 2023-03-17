# Computation in data science, 2021 Fall

This is a graduate-level data science course that introduces modern optimization theory, machine learning application to biomedical data, convex optimization with concentration on the mathematical background of models. These three parts are taught by three different professors respectively. This repository contains my solutions to all the homework assignments.

This course is taught collaboratively by three researchers in instituion of statistical science, academia sinica, the most preeminent academic institution of Taiwan

- [Phoa, Frederick Kin Hing](https://staff.stat.sinica.edu.tw/fredphoa/), instituion of statistical science, academia sinica
- [Shieh, Shwu-Rong Grace](https://staff.stat.sinica.edu.tw/gshieh/), instituion of statistical science, academia sinica
- [Yen, Tso-Jung](https://sites.stat.sinica.edu.tw/tjyen/), instituion of statistical science, academia sinica

Final score: A+

## Course contents

This course is comprised of three parts

- Part I: Modern optimization theory
  - Metaheuristics
  - Simulated annealing (SA), tabu search (TA)
  - Genetic Algorithm (GA)
  - Particle swarm optimization (PSO)
  - Any colony optimization (ACO)
- Part II: Machine learning applications to biomedical data
  - Dimension reduction of high-dimensional data
  - Prediction - regularized regression
  - Prediction - k-Nearest neighbor, support vector machines
- Part III: convex optimization
  - Matrix computation
  - Convest analysis, convex optimization and gradient methods
  - Alternating direction method of multipliers
  - Proximal gradient algorithms
  - Stochastic gradient descent algorithms
  - Topics in deep learning

## Homework assignments

The homeworks of this class are also divided into three parts.
- Part I
    - HW1: Use hill climbing, random walk, TA, SA to solve traveling salesman problem (TSP) 
    - HW2: Use PSO, GA, SA to solve TSP
    - Final project: Use all the modern optimization algos taught in class and a method - Cacukoo Search - that I found in a [paper](https://doi.org/10.1007/s00521-013-1402-2) to solve TSP
- Part II
    - HW1: Play with principal componants in principal componant analysis
    - HW2: Build a machine learning model to preict the response of drug(resistant, sensitive) with GDSC cell line gene expression dataset
    - Final project: Build a machine learning model to preict the response of drug(resistant, sensitive) with GDSC PDX Gemcitabine dataset
- Part III
    - HW1: Computing the least squares estimate via singular value decomposition (SVD), ridge regression estimation
    - HW2: Derivation of the Lipschitz constant for the logistic loss for regression estimation, gradient algorithm for logistic regression estimation 
    - HW3: Variable selection via lasso estimation
    - HW4: The proximal operator of the $l_0$-norm, $l_0$-norm regularized estimation via the fast proximal gradient algorithm

Please refer to each homework directory for details.
