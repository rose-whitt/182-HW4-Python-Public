# Firstname Lastname
# NetID
# COMP 182 Spring 2021 - Homework 4, Problem 3

# You may NOT import anything apart from already imported libraries.
# You can use helper functions from comp182.py and provided.py, but they have
# to be copied over here.

from collections import *
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


#graph1 = make_complete_graph(4)
#graph2 = make_complete_graph(3)
#graph3 = {'a': {'b'}, 'b': {'a', 'c'}, 'c': {'b'}}
#graph4 = {'a': {'b'}, 'b': {'a', 'c'}, 'c': {'b'}, 'd': {'e'}, 'e': {'d'}}
#graph5 = {'a': {'b'}, 'b': {'a', 'c'}, 'c': {'b'}, 'd': {'e'}, 'e': {'d'}, 'g': {'h', 'f'}, 'f': {'g', 'h'}, 'h': {'g', 'f', 'i'}, 'i': {'h'}}

#print(graph)
#print(compute_largest_cc_size(graph1))
#print(compute_largest_cc_size(graph2))
#print(compute_largest_cc_size(graph3))
#print(compute_largest_cc_size(graph4))
#print(compute_largest_cc_size({'b': {'c'}, 'c': {'b'}}))