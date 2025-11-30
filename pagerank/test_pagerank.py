import os
import random
import re
import sys
from pagerank import transition_model, crawl, sample_pagerank, iterate_pagerank

corpus = crawl("/Users/alessandroferroni/Desktop/pagerank/corpus0")
DAMPING = 0.85
SAMPLES = 10000

"""ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
print(f"PageRank Results from Sampling (n = {SAMPLES})")
for page in sorted(ranks):
    print(f"  {page}: {ranks[page]:.4f}")"""

ranks = iterate_pagerank(corpus, DAMPING)
print(f"PageRank Results from Iteration")
for page in sorted(ranks):
    print(f"  {page}: {ranks[page]:.4f}")
