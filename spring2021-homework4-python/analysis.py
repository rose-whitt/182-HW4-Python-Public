# Firstname Lastname
# NetID
# COMP 182 Spring 2021 - Homework 4, Problem 3

# You can import any standard library, as well as Numpy and Matplotlib.
# You can use helper functions from comp182.py, provided.py, and autograder.py,
# but they have to be copied over here.

# Your code here...
import comp182
import provided
import numpy
import math
import random
import pylab
import copy
import time
from collections import *

def time_func(f, args=[], kw_args={}):
    """
    Times one call to f with args, kw_args.

    Arguments:
    f       -- the function to be timed
    args    -- list of arguments to pass to f
    kw_args -- dictionary of keyword arguments to pass to f.

    Returns: 
    a tuple containing the result of the call and the time it
    took (in seconds).

    Example:

    >>> def sumrange(low, high):
            sum = 0
            for i in range(low, high):
                sum += i
            return sum
    >>> time_func(sumrange, [82, 35993])
    (647726707, 0.01079106330871582)
    >>> 
    """
    start_time = time.time()
    result = f(*args, **kw_args)
    end_time = time.time()

    return (result, end_time - start_time)
def copy_graph(g):
    """
    Return a copy of the input graph, g

    Arguments:
    g -- a graph

    Returns:
    A copy of the input graph that does not share any objects.
    """
    return copy.deepcopy(g)
def plot_lines(data, title, xlabel, ylabel, labels=None, filename=None):
    """
    Plot a line graph with the provided data.

    Arguments: 
    data     -- a list of dictionaries, each of which will be plotted 
                as a line with the keys on the x axis and the values on
                the y axis.
    title    -- title label for the plot
    xlabel   -- x axis label for the plot
    ylabel   -- y axis label for the plot
    labels   -- optional list of strings that will be used for a legend
                this list must correspond to the data list
    filename -- optional name of file to which plot will be
                saved (in png format)

    Returns:
    None
    """
    ### Check that the data is a list
    if not isinstance(data, list):
        msg = "data must be a list, not {0}".format(type(data).__name__)
        raise TypeError(msg)

    ### Create a new figure
    fig = pylab.figure()

    ### Plot the data
    if labels:
        mylabels = labels[:]
        for _ in range(len(data)-len(labels)):
            mylabels.append("")
        for d, l in zip(data, mylabels):
            _plot_dict_line(d, l)
        # Add legend
        pylab.legend(loc='best')
        gca = pylab.gca()
        legend = gca.get_legend()
        pylab.setp(legend.get_texts(), fontsize='medium')
    else:
        for d in data:
            _plot_dict_line(d)

    ### Set the lower y limit to 0 or the lowest number in the values
    mins = [min(l.values()) for l in data]
    ymin = min(0, min(mins))
    pylab.ylim(ymin=ymin)

    ### Label the plot
    pylab.title(title)
    pylab.xlabel(xlabel)
    pylab.ylabel(ylabel)

    ### Draw grid lines
    pylab.grid(True)

    ### Show the plot
    fig.show()

    ### Save to file
    if filename:
        pylab.savefig(filename)
def make_complete_graph(num_nodes):
    """
    Returns a complete graph containing num_nodes nodes.
 
    The nodes of the returned graph will be 0...(num_nodes-1) if num_nodes-1 is positive.
    An empty graph will be returned in all other cases.
 
    Arguments:
    num_nodes -- The number of nodes in the returned graph.
 
    Returns:
    A complete graph in dictionary form.
    """
    result = {}
         
    for node_key in range(num_nodes):
        result[node_key] = set()
        for node_value in range(num_nodes):
            if node_key != node_value: 
                result[node_key].add(node_value)
 
    return result

def _dict2lists(data):
    """
    Convert a dictionary into a list of keys and values, sorted by
    key.  

    Arguments:
    data -- dictionary

    Returns:
    A tuple of two lists: the first is the keys, the second is the values
    """
    xvals = list(data.keys())
    xvals.sort()
    yvals = []
    for x in xvals:
        yvals.append(data[x])
    return xvals, yvals

def compute_distances(graph, start_node):
    """
    Performs a breadth-first search on graph starting at the
    start_node.

    inputs:
        - graph: a graph object
        - start_node: a node in graph representing the start node

    Returns: a two-element tuple containing a dictionary
    associating each visited node with the order in which it
    was visited and a dictionary associating each visited node
    with its parent node.
    """
    container = Queue()
    dist = {}
    for node in graph.keys():
       
        dist[node] = -1

    
    dist[start_node] = 0

    container.push(start_node)
    while len(container) != 0:

        node = container.pop()

        for nbr in graph[node]:

            if dist.get(nbr) == -1:

                dist[nbr] = dist[node] + 1

                container.push(nbr)
    return dist

def make_complete_graph(num_nodes):
    """
    Returns a complete graph containing num_nodes nodes.
 
    The nodes of the returned graph will be 0...(num_nodes-1) if num_nodes-1 is positive.
    An empty graph will be returned in all other cases.
 
    Arguments:
    num_nodes -- The number of nodes in the returned graph.
 
    Returns:
    A complete graph in dictionary form.
    """
    result = {}
         
    for node_key in range(num_nodes):
        result[node_key] = set()
        for node_value in range(num_nodes):
            if node_key != node_value: 
                result[node_key].add(node_value)
 
    return result


def distinct_multinomial(ntrials, probs):
    """
    Draw ntrials samples from a multinomial distribution given by
    probs.  Return a list of indices into probs for all distinct
    elements that were selected.  Always returns a list with between 1
    and ntrials elements.

    Arguments:
    ntrials -- number of trials
    probs   -- probability vector for the multinomial, must sum to 1

    Returns: 
    A list of indices into probs for each element that was chosen one
    or more times.  If an element was chosen more than once, it will
    only appear once in the result.  
    """
    ### select ntrials elements randomly
    mult = numpy.random.multinomial(ntrials, probs)

    ### turn the results into a list of indices without duplicates
    result = [i for i, v in enumerate(mult) if v > 0]
    return result
def total_degree(g):
    """
    Compute total degree of the undirected graph g.

    Arguments:
    g -- undirected graph

    Returns:
    Total degree of all nodes in g
    """
    return sum(map(len, g.values()))
def read_graph(filename):
    """
    Read a graph from a file.  The file is assumed to hold a graph
    that was written via the write_graph function.

    Arguments:
    filename -- name of file that contains the graph

    Returns:
    The graph that was stored in the input file.
    """
    with open(filename) as f:
        g = eval(f.read())
    return g
def _pow_10_round(n, up=True):
    """
    Round n to the nearest power of 10.

    Arguments:
    n  -- number to round
    up -- round up if True, down if False

    Returns:
    rounded number
    """
    if up:
        return 10 ** math.ceil(math.log(n, 10))
    else:
        return 10 ** math.floor(math.log(n, 10))
def distinct_multinomial(ntrials, probs):
    """
    Draw ntrials samples from a multinomial distribution given by
    probs.  Return a list of indices into probs for all distinct
    elements that were selected.  Always returns a list with between 1
    and ntrials elements.

    Arguments:
    ntrials -- number of trials
    probs   -- probability vector for the multinomial, must sum to 1

    Returns: 
    A list of indices into probs for each element that was chosen one
    or more times.  If an element was chosen more than once, it will
    only appear once in the result.  
    """
    ### select ntrials elements randomly
    mult = numpy.random.multinomial(ntrials, probs)

    ### turn the results into a list of indices without duplicates
    result = [i for i, v in enumerate(mult) if v > 0]
    return result

def upa(n, m):
    """
    Generate an undirected graph with n node and m edges per node
    using the preferential attachment algorithm.

    Arguments:
    n -- number of nodes
    m -- number of edges per node

    Returns:
    undirected random graph in UPAG(n, m)
    """
    g = {}
    if m <= n:
        g = make_complete_graph(m)
        for new_node in range(m, n):
            # Find <=m nodes to attach to new_node
            totdeg = float(total_degree(g))
            nodes = list(g.keys())
            probs = []
            for node in nodes:
                probs.append(len(g[node]) / totdeg)
            mult = distinct_multinomial(m, probs)

            # Add new_node and its random neighbors
            g[new_node] = set()
            for idx in mult:
                node = nodes[idx]
                g[new_node].add(node)
                g[node].add(new_node)
    return g            

def erdos_renyi(n, p):
    """
    Generate a random Erdos-Renyi graph with n nodes and edge probability p.

    Arguments:
    n -- number of nodes
    p -- probability of an edge between any pair of nodes

    Returns:
    undirected random graph in G(n, p)
    """
    g = {}

    ### Add n nodes to the graph
    for node in range(n):
        g[node] = set()

    ### Iterate through each possible edge and add it with 
    ### probability p.
    for u in range(n):
        for v in range(u+1, n):
            r = random.random()
            if r < p:
                g[u].add(v)
                g[v].add(u)

    return g

def compute_largest_cc_size(g):
    max_size = 1
    b = g.copy()
    visit = {}
    for node in g:
        visit[node] = False
    Queue = deque()

    for i in b:
        if visit[i] == False:
            visit[i] = True
            Queue.append(i)
        
        n = 1

        while len(Queue) != 0:

            j = Queue.pop()

            for nbr in b[j]:
                if visit[nbr] == False:
                    visit[nbr] = True
                    Queue.append(nbr)
                    n += 1
        
        if max_size < n:
            max_size = n
    
    return max_size
def _plot_dict_line(d, label=None):
    """
    Plot data in the dictionary d on the current plot as a line.

    Arguments:
    d     -- dictionary
    label -- optional legend label

    Returns:
    None
    """
    xvals, yvals = _dict2lists(d)
    if label:
        pylab.plot(xvals, yvals, label=label)
    else:
        pylab.plot(xvals, yvals)

def remove_node(dict, node):
    b = copy_graph(dict)
    neighbors = b[node]
    b.pop(node)
    for nbr in neighbors:
        b[nbr].remove(node)
    
    return b
def random_attack(graph):
    """
    Remove nodes randomly, one by one, and compute the size of the largest connected component for each resulting graph.

    Inputs: graph - dictionary of nodes to its neighbors in g
    Outputs: a list of sizes of the largest connected component for each connected component for each resulting graph
    """
    
    
    b = copy_graph(graph)
    #print("B: " + str(b))
    sizes = {}
    twenty_percent = len(b.keys()) * 0.20
    #num_nodes_removed = 0
    #print(len(x.keys()))
    #print(compute_largest_cc_size(x))
    count = 0
    while count < twenty_percent and len(b.keys()) != 0:
        #print("here")
        size = compute_largest_cc_size(b)
        sizes[len(graph.keys()) - len(b.keys())] = size
        
        #print(len(b.keys()))
        random_node = random.choice(list(b.keys()))
        #print("RANDOM NODE: " + str(random_node))

        #print("RANDOM NODE: " + str(random_node))
        #b.pop(random_node)
        #print(b)
        b = remove_node(b, random_node)
        count += 1
        #print(b)
        #num_nodes_removed += 1

    return sizes


def targeted_attack(graph):
    """
    Remove nodes in decreasing order of degree, one by one,
    and compute the size of the largest connected component for each resulting graph.
    """
    sizes = {}
    b = copy_graph(graph)
    num_nodes_removed = 0
    twenty_percent = len(b.keys()) * 0.20
    count = 0
    while count < twenty_percent and len(b.keys()) != 0:
        degrees = {}
        #assign degree values to each key
        for key in b.keys():
            degrees[key] = len(b[key])
  
        size = compute_largest_cc_size(b)
        sizes[len(graph.keys()) - len(b.keys())] = size

        max_degree = max(degrees.values())
        
        #none of nodes are connected
        if max_degree == 0:
            return sizes
        for key, value in degrees.items():
            if value == max_degree:
                targeted_node = key
                break
        b = remove_node(b, targeted_node)
        count += 1

       
    
    return sizes


top_graph = read_graph("rf7.repr")

size = len(top_graph.keys())

average_degree = float(total_degree(top_graph) / size)

upa_graph = upa(size, int(average_degree))

p = float(average_degree / size)

er_graph = erdos_renyi(size, p)

print(er_graph)

random_top = random_attack(top_graph)

targeted_top = targeted_attack(top_graph)

random_upa = random_attack(upa_graph)

targeted_upa = targeted_attack(upa_graph)

random_er = random_attack(er_graph)



targeted_er = targeted_attack(er_graph)

data = [random_top, targeted_top, random_upa, targeted_upa, random_er, targeted_er]
#, random_er, targeted_er
names = ["Random Topology", "Targeted Topology", "Random UPA", "Targeted UPA", "Random ER", "Targeted ER"]
#test = make_complete_graph(3)
#print(random_attack(top_graph))
#print(targeted_attack(top_graph))
#data = targeted_attack(top_graph)

#plot_lines(data, "Data Test Plot", "number of nodes removed", "size of connected component", names, filename="Network Data")
#top_graph_target = targeted_attack(top_graph)
#upa_graph_target = targeted_attack(upa_graph)
#er_graph_target = targeted_attack(er_graph)

#print("hi")
#plot_lines(data, title, "number of nodes removed", "size of connected component", labels=None, filename=None)