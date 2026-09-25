
from pathlib import Path

import networkx as nx
import pandas as pd


def build_knowledge_graph_from_data(
    data: pd.DataFrame,
    graph_path: str | None = "outputs/knowledge_graph.graphml",
) -> nx.MultiDiGraph:

    graph = nx.MultiDiGraph()

    # add patient nodes
    for pid in data.patient_id.unique():
        graph.add_node(pid, kind="patient")

    # add condition nodes
    for cond in sorted(data.conditions.unique()):
        graph.add_node(f"condition:{cond}", kind="condition")

    # observed patient → condition edges
    for pid, grp in data.groupby("patient_id"):
        cond = grp.conditions.iloc[0]
        graph.add_edge(
            pid, f"condition:{cond}",
            relation="has_condition",
            justification="dataset_observation",
        )

    # similarity edges: patients who share a condition
    # (assumption, not clinical evidence)
    patients = data.groupby("patient_id").first()
    ids = list(patients.index)
    for i, left_id in enumerate(ids):
        for right_id in ids[i + 1:]:
            if patients.loc[left_id, "conditions"] == patients.loc[right_id, "conditions"]:
                graph.add_edge(
                    left_id, right_id,
                    relation="shared_condition_similarity",
                    justification="assumption",
                )

    if graph_path:
        output = Path(graph_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(graph, output)

    return graph


def build_knowledge_graph(
    data_path: str = "data/longitudinal_diabetes.csv",
    graph_path: str = "outputs/knowledge_graph.graphml",
) -> nx.MultiDiGraph:
    return build_knowledge_graph_from_data(
        pd.read_csv(data_path), graph_path
    )


def build_patient_subgraph(patient: pd.DataFrame) -> nx.MultiDiGraph:
    return build_knowledge_graph_from_data(patient, graph_path=None)


if __name__ == "__main__":
    g = build_knowledge_graph()
    print(f"wrote graph nodes={g.number_of_nodes()} "
          f"edges={g.number_of_edges()}")
