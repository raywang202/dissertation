# Dissertation Code Sample

# Model Outline
The code represents an extension of the Keane and Wolpin (1997) endogenous occupational choice model to include educational decisions as well. The current code sample runs as follows:

1) **data_generate.py** represents a 'forward simulation' of the model. Given a set of model parameters (in the file) as well as a file of the first-year outcomes and student characteristics (e.g. SAT scores) as given in **year_one_transcript.csv**, simulate the student's behavior in response to a lifetime of shocks. The output is given in a file in the same directory, **sim_data.csv** which will be random; one example is given, in case you don't want to re-run the model, which took about 10 minutes on my Macbook.
2) **calculate_likelihood.py** contains code that lets you compute the most computationally expensive part of the likelihood expression: the skilled labor market's likelihood contribution. Running the entire file will pull in the dataset (**sim_data.csv**), clean it up in the appropriate manner for processing, and calculate the posterior probability that each individual in the sample is one of three unobserved types. Then, evaluating the function LaborLikeEM(x,c) where x is the vector of parameters (try x = x1Init from the file) and c is a dummy term used in function wrapper (just use c = None) will return the objective function to be used in an E-M algorithm, namely the sum of the posterior probabilities that each individual is of a type, multiplied by the log-likelihood of their observations if they were of that type.
3) **EmaxLaborFunctionsJIT.py** contains code that computes the ex-ante value functions both for the skilled labor market. This is necessary both for forward simulation of the model as well as calculating likelihoods in estimation.
4) **EmaxLaborFunctionsJITUnskilled.py** is the analogous code for the unskilled labor market.
5) **EmaxEducationJIT.py** computes the ex-ante value functions for college students. It takes as inputs the payouts associated with the ex-ante value functions of the skilled and unskilled labor markets.
6) **FSLikelihoodJITfinal.py** contains the functions that calculate the likelihoods.
