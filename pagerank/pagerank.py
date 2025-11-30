import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    print(sum(ranks.values())) # added by me for debugging purposes
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    print(sum(ranks.values())) # added by me for debugging purposes


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):

    # the corpus is a Python dictionary mapping a page name to a set of all pages linked to by that page.         ex. a------>
    # the page is a string representing which page the random surfer is currently on.                             ex. b------>
    # the damping_factor is a floating point number representing the damping factor to be used when generating the probabilities.

    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    
    # the return value of the function should be a Python dictionary with one key for each page in the corpus. Each key should be mapped to a value
    # representing the probability that a random surfer would choose that page next. The values in this returned probability distribution should sum to 1
    # with probability damping_factor, the random surfer should randomly choose one of the links from page with equal probability
    # with probability 1 - damping_factor, the random surfer should randomly choose one of all pages in the corpus with equal probability.

    # for example, if the corpus were {"1.html": {"2.html", "3.html"}, "2.html": {"3.html"}, "3.html": {"2.html"}}  ex. <------a
    # the page == "1.html"                                                                                          ex. <------b
    # damping_factor == 0.85                                       
    # transition_model == {"1.html": 0.05, "2.html": 0.475, "3.html": 0.475}
    
    # this is because with probability 0.85, we choose randomly to go from page 1 to either page 2 or page 3 
    # (so each of page 2 or page 3 has probability 0.425 to start),
    # but every page gets an additional 0.05 because with probability 0.15 we choose randomly among all three of the pages.

    # if a page has no links, we can pretend it has links to all pages in the corpus, including itself

    # PSEUDOCODE
    # for key, value in corpus.items():
    #    if key == page:
    #        if len(value):
    #            d_factor_probability = damping_factor / len(value)
    #            1_minus_d_factor_probability = (1 - damping_factor) / (len(value) + 1)
    #            probability = d_factor_probability + 1_minus_d_factor_probability
    #            return a dictionary where the page has value 1_minus_d_factor_probability and the links value of probability
    #            return {
    #                        page: 1_minus_d_factor_probability,
    #                       **{other_page: probability for other_page in value}
    #                   }
    #         else:
    #            corpus_keys = list(corpus)
    #            probability = 1 / len(corpus_keys)
    #            return {key : probability for key in corpus_keys}

    for key, value in corpus.items():
        # .items() is a dictionary method that returns a special view object containing all the key–value pairs in the dictionary.
        # it returns something like ([(key1, value1), (key2, value2), ...])
        if key == page:
            if len(value):
                d_factor_probability = damping_factor / len(value)
                minus_d_factor_probability = (1 - damping_factor) / (len(value))
                probability = d_factor_probability + minus_d_factor_probability
                return {
                            **{x: minus_d_factor_probability for x in corpus if x not in value},
                            **{other_page: probability for other_page in value}
                           # i create a dict using a dict comprehension and then with ** i unpack it into the dictionary to be returned merging the two
                       }
            else:
                corpus_keys = list(corpus)
                probability = round(1 / len(corpus_keys), 4)
                return {key : probability for key in corpus_keys}


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # the random surfer model imagines a surfer who starts with a web page at random, and then randomly chooses links to follow.
    # the sample_pagerank function should accept a corpus of web pages, a damping factor, and a number of samples, and return an estimated PageRank for each page.

    # PSEUDOCODE
    # pick a random page out of the corpus
    # create a dictionary for each key present in the corpus and its counter
    # repeat this n (sample) times:
    # pass that page together to the corpus and the damping_factor to the transition_model
    # pass the probability for each page as weights and the keys as a list to random.choices()
    # increment the counter related to the page outputted by random.choices
    # once done divide each value by n 

    pages = list(corpus)
    page = random.choice(pages)
    pagerank_dict = {x : 0 for x in pages} # both for the dictionary and the counters to be returned
    counter = 0
    while counter < n:
        state = transition_model(corpus, page, damping_factor)
        choices = random.choices(
            population=list(state.keys()),
            weights=list(state.values())
        )
        page = choices[0]
        for key in pagerank_dict:
            if key == page:
                pagerank_dict[key] += 1
        counter += 1
    for key in pagerank_dict:
        if pagerank_dict[key]:
            pagerank_dict[key] = (pagerank_dict[key] / n)
    return pagerank_dict


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # we can define a page’s PageRank using a recursive mathematical expression. Let PR(p) be the PageRank of a given page p
    # with probability 1 - d divided by N, where N is the total number of pages across the entire corpusthe, the surfer chose a page at random and ended up on page p
    # with probability d, the surfer followed a link from a page i to page p
    # we need to consider each possible page i that links to page p. for each of those incoming pages, let NumLinks(i) be the number of links on page i. 
    # each page i that links to p has its own PageRank, PR(i), representing the probability that we are on page i at any given time. 
    # and since from page i we travel to any of that page’s links with equal probability, 
    # we divide PR(i) by the number of links NumLinks(i) to get the probability that we were on page i and chose the link to page p.

    # we go about calculating PageRank values for each page via iteration: start by assuming the PageRank of every page is 1 / N where N is the number of pages in the corpus 
    # then, use the iteration formula to calculate new PageRank values for each page, based on the previous PageRank values
    # If we keep repeating this process, calculating a new set of PageRank values for each page based on the previous set of PageRank values,
    # eventually the PageRank values will converge (i.e., not change by more than a small threshold with each iteration).
    # this process should repeat until no PageRank value changes by more than 0.001 between the current rank values and the new rank values. ======>>>

    '''
    Gauss-Seidel updates 
    uses newest values immediately - asynchronous updates 

    pages = list(corpus)
    n = len(pages)
    iterate_dict = {x : (1 / n) for x in pages} # initialize a dict where each page has a pagerank of 1 / n  
    while True:
        for key in iterate_dict: # i want to figure out for each page of the dict what's its rank
            ranks = 0
            for i in iterate_dict: # to do so i loop over the dict to retrieve for each page (excluding the 'key' page) its rank so that i have enough info to solve the second condition of the formula
                if key is not i and key in corpus[i]: # if the pagerank i want to retrieve it's of the current key it will be skipped
                    page_rank = iterate_dict[i] / len(corpus[i]) # retrieves the pagerank for i page and divedes it for the number of links associated to the page in the corpus
                    ranks += page_rank # we add up each page rank together
                elif not len(corpus[i]): # if said page has links in the corpus
                    page_rank = page_rank / n # else if the page has no links it is as the page has links for all the pages in the corpus including itself, so we divide the pagerank by n
                    ranks += page_rank # we add up each page rank together
            new_iterate_dict = iterate_dict.copy() # returns a shallow copy of the iterate_dict
            new_iterate_dict[key] = ((1 - damping_factor) / n) + (damping_factor * ranks) # the new copy is then updated with the latest entry for the key of the outer loop

    # this process should repeat until no PageRank value changes by more than 0.001 between the current rank values and the new rank values. <<<======

            # convergence
            counter = 0    
            precision = 10
            for page in pages:
                counter += 1
                if round(new_iterate_dict[page], precision) != round(iterate_dict[page],precision): # if the probablity associated with the same page in the new dictionary and old one differ too much:
                    iterate_dict |= new_iterate_dict # update the old dictionary
                    break # break into the outer loop
                elif counter == n:
                        return new_iterate_dict    
    '''

    ''' 
    Jacobi updates 
    computes al new values using old iteration values - synchronous updates
    '''

    pages = list(corpus)
    n = len(pages)
    iterate_dict = {x : 1 / n for x in pages}
    new_iterate_dict = iterate_dict.copy() # returns a shallow copy of the iterate_dict
    while True:
        for key in iterate_dict:
            second_condition = 0
            for i in iterate_dict:
                if len(corpus[i]):
                    if key in corpus[i]:
                        second_condition += iterate_dict[i] / len(corpus[i])
                else:
                    second_condition += iterate_dict[i] / n
            new_iterate_dict[key] = ((1 - damping_factor) / n) + (damping_factor * second_condition)
       
        # convergence 
    
        # Gauss–Seidel-like
        # counter = 0  
        # for page in pages:
        #    counter += 1
        #    if abs(new_iterate_dict[page] - iterate_dict[page]) > 0.001:
        #        iterate_dict = new_iterate_dict.copy()
        #        break
        #    elif counter == n:
        #        return new_iterate_dict

        # Jacobi method    
        if all(abs(new_iterate_dict[p] - iterate_dict[p]) < 0.001 for p in pages):
            return new_iterate_dict
        else:
            iterate_dict = new_iterate_dict.copy()

if __name__ == "__main__":
    main()
