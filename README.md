# PageRank

![Python](https://img.shields.io/badge/Python-3.11-blue)
![CS50 AI](https://img.shields.io/badge/CS50-AI_Project-orange)
![PageRank](https://img.shields.io/badge/Algorithm-PageRank-brightgreen)
![Status](https://img.shields.io/badge/Status-Complete-success)

Implementation of the PageRank algorithm, part of Harvard's **CS50 AI** course.  
This project analyzes a corpus of HTML files and computes a PageRank score for each page using two distinct methods:

- **Sampling** (Random Surfer Model)
- **Iteration** (Recursive PageRank Computation)

The repository contains **`pagerank.py`**, a heavily-commented and educational implementation where comments are intentionally verbose to explain reasoning, design choices, and internal logic.  
This style is chosen to help my future self re-understand the code, the algorithms, and the constraints that shaped this implementation.

---

## 📁 Project Structure

```
pagerank/
│
├── corpus0/1/2        # Directories containing .html pages used as the graph
├── pagerank.py        # Main implementation (sampling + iteration)
└── README.md          # This file
```

---

# 🚀 How It Works

This project builds a simplified *web graph* and computes PageRank in two ways.

---

## 1. Crawl the Corpus

The `crawl(directory)` function:

- Reads all `.html` files in the given directory  
- Extracts `<a href="...">` links using regex  
- Filters out links pointing outside the corpus  
- Returns a dictionary:

```python
{
    "1.html": {"2.html", "3.html"},
    "2.html": {"3.html"},
    "3.html": {"2.html"}
}
```

Each key is a page; each value is the set of pages it links to.

---

## 2. Transition Model

`transition_model(corpus, page, damping_factor)` produces a probability distribution over next pages:

- With probability `d`, choose a linked page uniformly
- With probability `1 - d`, choose *any* page in the corpus uniformly
- If a page has **no outbound links**, treat it as linking to *every* page (including itself)

Returns a dictionary like:

```python
{
    "1.html": 0.05,
    "2.html": 0.475,
    "3.html": 0.475
}
```

This is used for sampling.

---

## 3. Sampling PageRank

`sample_pagerank(corpus, d, n)`:

- Start from a random page
- Perform `n` transitions using the transition model
- Count how often each page is visited
- Normalize to sum to 1

This simulates the behavior of a “random surfer.”

This method is **approximate**, but converges with large `n`.

---

## 4. Iterative PageRank

`iterate_pagerank(corpus, d)`:

- Initialize every page rank to `1/N`
- Repeatedly apply the PageRank update formula:

```
PR(p) = (1 - d)/N  +  d * Σ [PR(i) / NumLinks(i)]  
        (for every page i that links to p)
```

- Continue until **no PageRank value changes by more than 0.001**
- Return the final stable ranks

This method is **deterministic**, mathematically grounded, and converges to true PageRank.

---

# 🧪 Usage

Run the program by providing a corpus directory:

```
python pagerank.py corpus
```

Output example:

```
PageRank Results from Sampling (n = 10000)
  1.html: 0.2121
  2.html: 0.4063
  3.html: 0.3816
1.0000

PageRank Results from Iteration
  1.html: 0.2178
  2.html: 0.4044
  3.html: 0.3778
1.0000
```

You’ll notice the two methods converge toward similar values.

---

# 🧠 Notes on Code Style

This project deliberately contains:

- **Extremely verbose comments**
- Comments that restate the specification
- Pseudocode in multiple sections
- Small debugging print statements (e.g., sum of ranks)
- Descriptions of reasoning and design choices  
  (especially useful for educational purposes)

This is a personal choice to make the code more **didactic**, traceable, and self-explanatory — not necessarily how production software would be written.

---

# 📘 Concepts Covered

- Graph modeling from real data
- Random surfer model
- Probability distributions
- Weighted random choice
- Iterative convergence
- Handling dangling nodes (pages with no links)
- PageRank’s mathematical foundation
- Algorithmic simulation vs. recursive computation

---

# 📄 License

This implementation is created for educational purposes as part of the CS50 AI course.  
Feel free to use, modify, or expand upon it.
