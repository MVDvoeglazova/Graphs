import graphlib
import osmnx as ox
import networkx as nx
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon
import math
from heapq import *

pd.set_option('display.max_columns', None)
G = ox.graph_from_address("Lipetsk, Russia", dist=10000, network_type='all')
G = G.to_undirected()
nodes, roads = ox.graph_to_gdfs(G)
data = roads.reset_index()[["u", "v", "length"]]
data.head()
data = data.values.tolist()
fig, ax = plt.subplots(figsize=(20, 20))
roads.plot(ax=ax, linewidth=1.5, edgecolor='gray', facecolor='none', zorder=1)

plt.show()
#построение кратчайшего пути с помощью встроенной библиотеки osmnx
route = nx.shortest_path(G, 469710010, 9978536016, weight="length")
fig, ax = ox.plot.plot_graph_route(
    G, 
    route, 
    route_color='r', 
    route_linewidth=2, 
    route_alpha=0.5, 
    orig_dest_size=200, 
    ax=None, 
    figsize=(20,20))

#Преобразование списка data в словарь вида 
#graph = {a:[['b', '10'], ['g', '5']], ...
start = 469710010
finish = 9978536016
my_dict = dict()
my_dict['Route'] = [start, finish]
nodes = set()
for i in range(len(data)):
    nodes.add(data[i][0])
    nodes.add(data[i][1])
my_dict['Nodes'] = list(nodes)
my_dict['Edges'] = data

#список вершин
vertex = (my_dict['Nodes'])

graph = dict() #для хранения ребер вершин и их веса
for k in vertex:
    graph[k] = []

for num_val in range(len(my_dict['Edges'])):
    graph[my_dict['Edges'][num_val][1]] += [[my_dict['Edges'][num_val][2],my_dict['Edges'][num_val][0]]]
    graph[my_dict['Edges'][num_val][0]] += [[my_dict['Edges'][num_val][2],my_dict['Edges'][num_val][1]]]

#graph = {a:[['b', '10'], ['g', '5']], ...
#создает 2 словаря 
    #len_way словарь для хранения вершины и пути до нее
    #visited_vert cловарь для запоминания пути от вершины к вершине
def dijkstra(graph: dict, start: float)-> (dict, dict): 

    queue = list()
    len_way = dict() #словарь для хранения вершины и пути до нее
    visited_vert = dict() #cловарь для запоминания пути от вершины к вершине

    queue.append((0, start))
    len_way[start] = 0
    visited_vert[start] = None

    while queue:
        now_vert = queue.pop()[1]

        next_verts = graph[now_vert]
        for next_vert in next_verts:
            neighbor_len, neighbor_vert = next_vert
            min_len = len_way[now_vert] + neighbor_len

            if neighbor_vert not in len_way or min_len < len_way[neighbor_vert]:
                queue.append((min_len, neighbor_vert))
                len_way[neighbor_vert] = min_len
                visited_vert[neighbor_vert] = now_vert
    return visited_vert, len_way

#функция восстановления кратчайшего пути 
def route(visited_vert: dict, start: float, finish: float)-> dict:
    shorter_way = [finish,]
    while visited_vert[finish] != None:
        finish = visited_vert[finish]
        shorter_way.append(finish)
    return shorter_way
visited_vert, len_way = dijkstra(graph, 469710010)

shorter_way = route(visited_vert, 469710010, 9978536016)

#график кратчайшего пути с помощью функций dijkstra и route описанных выше
fig, ax = ox.plot.plot_graph_route(                             
    G, 
    shorter_way, 
    route_color='r', 
    route_linewidth=2, 
    route_alpha=0.5, 
    orig_dest_size=200, 
    ax=None, 
    figsize=(20,20))

# найдем места, куда можно попасть из заданной точки start, если двигаться 
speed = 15                               #со скоростью speed за время time
time = 30
dist = speed*time

stack_vert = list() #для хранения пути вершин
stack_way = list()  #для хранения пройденного расстояния 

stack_vert.append(start)
stack_way.append(0)

range_way = list() #для хранения диапазона путей, которые не превышают расстояния dist

#функция очистки стека до заданного значения
def clean_stack(next_v):
    global stack_way
    global stack_vert 
    for i in range(len(stack_vert)):
            if stack_vert[i] == next_v:
                stack_vert = stack_vert[0:i]
                stack_way = stack_way[0:i]
                break
#функция поиска диапазона путей, которые не превышают расстояния dist
def walking_distance(graph, start):

    for len_ed, next_v in graph[start]:
        if next_v not in stack_vert:
            if sum(stack_way)+len_ed<=dist:
                stack_vert.append(next_v)
                stack_way.append(len_ed)
                walking_distance(graph, next_v)
            range_way.append(stack_vert)        
            clean_stack(next_v)
        
walking_distance(graph, start)
#график путей, которые можно совершить из заданной точки start, если двигаться со скоростью speed за время time         
ox.plot.plot_graph_routes(G, range_way, route_colors='r', route_linewidths=1)


