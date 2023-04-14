#!/usr/bin/env/python3

"""
DESCRIPTION:
Extract centerline of mesh object and display it

"""
import collections
from os.path import join
from typing import List, Tuple

import open3d as o3d
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import skeletor as sk
import trimesh
from skeletor import Skeleton
from skeletor.skeletonize.utils import make_swc


def parse_arguments():
    """
    Simple CommandLine argument parsing function making use of the argparse module

    :return: parsed arguments object args
    """
    parser = argparse.ArgumentParser(
        description=" Meshify object and save as stl file"
    )

    parser.add_argument(
        "-a",
        "--artery",
        help="Enter label number for target artery.",
        required=True,
        type=int,
    )

    parser.add_argument(
        "-od",
        "--output_directory",
        help="Absolute path to output directory",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-m",
        "--mesh",

        help="Mesh [ .stl]",
        required=True,
        type=str,
    )
    args = parser.parse_args()

    return args


class GraphGenerator:
    """
    Graph class to manipulate and compute the longest path centerline

    """

    def __init__(self, skl: Skeleton, artery_filter: int):
        basilar_artery = 3
        self.verts = skl.vertices
        self.edges = skl.edges
        self.nx_graph = self.graph_creation()
        self.endpoints = self.get_endpoints()
        if artery_filter == basilar_artery:
            self.center_line_path = self.basilar_center_line()
        else:
            self.center_line_path = self.longest_path_center_line()
        self.center_line = self.center_line_computation()
        self.cl_edges, self.cl_verts = self.extract_verts_and_faces()
        self.diameter = self.compute_avg_diameter()

    def basilar_center_line(self):
        lowest_endpoint = self.endpoints[0]
        highest_endpoint = self.endpoints[0]
        z_axis = 2
        for endpoint in self.endpoints:
            if endpoint[z_axis] > highest_endpoint[z_axis]:
                highest_endpoint = endpoint
            if endpoint[z_axis] < lowest_endpoint[z_axis]:
                lowest_endpoint = endpoint
        path = nx.all_simple_paths(self.nx_graph, lowest_endpoint, highest_endpoint)
        return list(path).pop()

    def extract_verts_and_faces(self):

        results = []
        nodes = list(self.nx_graph.nodes)
        for edge in self.nx_graph.edges:
            results.append([nodes.index(edge[0]), nodes.index(edge[1])])
        new_edges = np.array(results)
        new_verts = np.array(nodes)
        return new_edges, new_verts

    def center_line_computation(self):
        nodes = list(self.nx_graph.nodes)
        for node in nodes:
            if not node in self.center_line_path:
                self.nx_graph.remove_node(node)
        return self.nx_graph

    def graph_creation(self):
        graph = nx.Graph()
        for edge in self.edges:
            graph.add_edge(tuple(self.verts[edge[0]]), tuple(self.verts[edge[1]]))
            graph.add_node(tuple(self.verts[edge[0]]))
        graph.add_node(tuple(self.verts[0]))
        return graph

    def get_endpoints(self) -> list:
        endpoints = list()
        for coord, degree in self.nx_graph.degree:
            if degree == 1:
                endpoints.append(coord)
        return endpoints

    def longest_path_center_line(self):
        longest_paths = list()
        for i in range(len(self.endpoints)):
            j = i + 1
            while j < len(self.endpoints):
                path = nx.all_simple_paths(self.nx_graph, tuple(self.endpoints[i]), tuple(self.endpoints[j]))
                longest_paths.append(list(path).pop())
                j += 1
        max_element = 0
        index_of_max = 0
        for the_longest_path in longest_paths:
            if len(the_longest_path) > max_element:
                max_element = len(the_longest_path)
                index_of_max = longest_paths.index(the_longest_path)
        return longest_paths[index_of_max]

    def compute_avg_diameter(self):
        radii = []
        for vertex in self.cl_verts:
            index = np.where(skel.vertices == vertex)
            index = index[0][0]
            radius = skel.radius[0][index]
            radii.append(radius)
        radii = np.array(radii)
        diameters = radii * 2
        return np.average(diameters)

    def reverse_index(self):
        uniques = []
        unique_edges = []
        for edge in reversed(self.cl_edges):
            if edge[0] in uniques:
                edge[0], edge[1] = edge[1], edge[0]
                uniques.append(edge[0])
            else:
                uniques.append(edge[0])
            unique_edges.append(edge)
        return np.array(unique_edges)

    def index(self):
        uniques = []
        unique_edges = []
        for edge in self.cl_edges:
            if edge[0] in uniques:
                edge[0], edge[1] = edge[1], edge[0]
                uniques.append(edge[0])
            else:
                uniques.append(edge[0])
            unique_edges.append(edge)
        return np.array(unique_edges)


def main():
    # Argument Parsing
    args = parse_arguments()
    output_directory = args.output_directory
    artery = args.artery
    mesh_path = args.mesh

    # Load mesh
    mesh = trimesh.load(mesh_path)

    # Skeletonize mesh
    skel = sk.skeletonize.by_wavefront(mesh)

    # Load skeleton into graph object

    graph = GraphGenerator(skel, artery)
    unique_edges_np = graph.index()
    try:
        swc, new_ids = make_swc(unique_edges_np, coords=graph.cl_verts, reindex=True, validate=True)
    except:
        unique_edges_np = graph.reverse_index()
        swc, new_ids = make_swc(unique_edges_np, coords=graph.cl_verts, reindex=True, validate=True)

    # Make new swc table
    skel = Skeleton(swc, mesh, method="wavefront")
    sk.post.clean_up(skel, inplace=True)
    skel.show(mesh=True)

    # Save centerline to csv file
    results = pd.DataFrame([[i, graph.diameter]], columns=['id', 'MeshAvgDiameter Basil'])
    results.to_csv(output_directory + '/centerline.csv', mode='a', index=False, header=False)


if __name__ == "__main__":
    main()
