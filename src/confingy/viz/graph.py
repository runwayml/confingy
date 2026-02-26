"""Visualization utilities for confingy configurations as DAGs."""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class NodeType(Enum):
    """Types of nodes in the configuration DAG."""

    DATACLASS = "dataclass"
    TRACKED_CLASS = "tracked"
    LAZY_CLASS = "lazy"
    CONSTRUCTOR_ARG = "arg"
    PRIMITIVE = "primitive"
    COLLECTION = "collection"
    METHOD = "method"
    FUNCTION = "function"
    TYPE = "type"


class DiffType(Enum):
    """Types of differences between configurations."""

    UNCHANGED = "unchanged"
    ADDED = "added"
    REMOVED = "removed"
    CHANGED = "changed"


@dataclass
class ConfigNode:
    """Represents a node in the configuration DAG."""

    node_id: str
    node_type: NodeType
    label: str
    value: Any = None
    class_name: str | None = None
    module: str | None = None
    class_hash: str | None = None
    metadata: dict[str, Any] | None = None
    depth: int = 0  # Distance from root node
    diff_type: DiffType | None = None  # For comparison mode

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ConfigEdge:
    """Represents an edge in the configuration DAG."""

    from_node: str
    to_node: str
    label: str | None = None
    edge_type: str = "contains"  # contains, references, lazy_ref, etc.


class ConfigGraph:
    """Builds and represents a graph from a serialized confingy configuration."""

    def __init__(self):
        self.nodes: dict[str, ConfigNode] = {}
        self.edges: list[ConfigEdge] = []
        self._node_counter = 0

    def _generate_node_id(self, prefix: str = "node") -> str:
        """Generate a unique node ID."""
        self._node_counter += 1
        return f"{prefix}_{self._node_counter}"

    def _get_node_type(self, data: Any) -> NodeType:
        """Determine the type of node from the data."""
        if not isinstance(data, dict):
            if isinstance(data, (list, tuple)):
                return NodeType.COLLECTION
            else:
                return NodeType.PRIMITIVE

        # Check for confingy-specific markers
        if "_confingy_dataclass" in data:
            return NodeType.DATACLASS
        elif "_confingy_lazy" in data:
            return NodeType.LAZY_CLASS
        elif "_confingy_class" in data and "_confingy_init" in data:
            return NodeType.TRACKED_CLASS
        elif "_confingy_callable" in data:
            if data.get("_confingy_callable") == "method":
                return NodeType.METHOD
            else:
                return NodeType.FUNCTION
        elif "_confingy_class" in data and data.get("_confingy_class") == "type":
            return NodeType.TYPE
        else:
            return NodeType.COLLECTION

    def build_from_config(
        self,
        config: dict[str, Any],
        parent_id: str | None = None,
        field_name: str | None = None,
        depth: int = 0,
    ) -> str:
        """
        Build DAG from a serialized config recursively.
        Returns the node ID of the root/current node.
        """
        node_type = self._get_node_type(config)

        # Handle primitive values
        if node_type == NodeType.PRIMITIVE:
            node_id = self._generate_node_id("prim")
            self.nodes[node_id] = ConfigNode(
                node_id=node_id,
                node_type=node_type,
                label=str(config)[:50],  # Truncate long strings
                value=config,
                depth=depth,
            )
            if parent_id and field_name:
                self.edges.append(ConfigEdge(parent_id, node_id, label=field_name))
            return node_id

        # Handle collections (lists, tuples)
        if node_type == NodeType.COLLECTION and not isinstance(config, dict):
            node_id = self._generate_node_id("coll")
            coll_type = type(config).__name__
            self.nodes[node_id] = ConfigNode(
                node_id=node_id,
                node_type=node_type,
                label=f"{coll_type}[{len(config)}]",
                value=config,
                depth=depth,
            )
            if parent_id and field_name:
                self.edges.append(ConfigEdge(parent_id, node_id, label=field_name))

            # Process collection items
            for i, item in enumerate(config):
                self.build_from_config(item, node_id, f"[{i}]", depth + 1)
            return node_id

        # Handle dataclasses
        if node_type == NodeType.DATACLASS:
            node_id = self._generate_node_id("dc")
            class_name = config.get(
                "_confingy_class", "DataClass"
            )  # Use _confingy_class for the name
            module = config.get("_confingy_module", "")
            class_hash = config.get("_confingy_class_hash", "")

            self.nodes[node_id] = ConfigNode(
                node_id=node_id,
                node_type=node_type,
                label=f"@dataclass {class_name}",  # Add @dataclass prefix for clarity
                class_name=class_name,
                module=module,
                class_hash=class_hash[:8]
                if class_hash
                else None,  # Short hash for display
                metadata={"fields": list(config.get("_confingy_fields", {}).keys())},
                depth=depth,
            )
            if parent_id and field_name:
                self.edges.append(ConfigEdge(parent_id, node_id, label=field_name))

            # Process dataclass fields
            for field, value in config.get("_confingy_fields", {}).items():
                self.build_from_config(value, node_id, field, depth + 1)
            return node_id

        # Handle tracked classes
        if node_type == NodeType.TRACKED_CLASS:
            node_id = self._generate_node_id("track")
            class_name = config.get("_confingy_class", "TrackedClass")
            module = config.get("_confingy_module", "")
            class_hash = config.get("_confingy_class_hash", "")

            self.nodes[node_id] = ConfigNode(
                node_id=node_id,
                node_type=node_type,
                label=f"@track {class_name}",
                class_name=class_name,
                module=module,
                class_hash=class_hash[:8]
                if class_hash
                else None,  # Short hash for display
                metadata={"init_args": list(config.get("_confingy_init", {}).keys())},
                depth=depth,
            )
            if parent_id and field_name:
                self.edges.append(ConfigEdge(parent_id, node_id, label=field_name))

            # Process constructor arguments
            for arg, value in config.get("_confingy_init", {}).items():
                self.build_from_config(value, node_id, arg, depth + 1)
                # Update edge type for constructor args
                if self.edges and self.edges[-1].from_node == node_id:
                    self.edges[-1].edge_type = "arg"
            return node_id

        # Handle lazy classes
        if node_type == NodeType.LAZY_CLASS:
            node_id = self._generate_node_id("lazy")
            class_name = config.get("_confingy_class", "LazyClass")
            module = config.get("_confingy_module", "")
            class_hash = config.get("_confingy_class_hash", "")

            self.nodes[node_id] = ConfigNode(
                node_id=node_id,
                node_type=node_type,
                label=f"lazy({class_name})",
                class_name=class_name,
                module=module,
                class_hash=class_hash[:8] if class_hash else None,
                metadata={"config": list(config.get("_confingy_config", {}).keys())},
                depth=depth,
            )
            if parent_id and field_name:
                self.edges.append(
                    ConfigEdge(
                        parent_id, node_id, label=field_name, edge_type="lazy_ref"
                    )
                )

            # Process config arguments
            for arg, value in config.get("_confingy_config", {}).items():
                self.build_from_config(value, node_id, arg, depth + 1)
                if self.edges and self.edges[-1].from_node == node_id:
                    self.edges[-1].edge_type = "config"
            return node_id

        # Handle methods/functions
        if node_type in (NodeType.METHOD, NodeType.FUNCTION):
            node_id = self._generate_node_id("func")
            if node_type == NodeType.METHOD:
                method_name = config.get("_confingy_name", "method")
                label = f"{method_name}()"
            else:
                func_name = config.get("_confingy_name", "function")
                label = func_name

            self.nodes[node_id] = ConfigNode(
                node_id=node_id,
                node_type=node_type,
                label=label,
                metadata=config,
                depth=depth,
            )
            if parent_id and field_name:
                self.edges.append(ConfigEdge(parent_id, node_id, label=field_name))

            # If it's a method, process the bound object
            if node_type == NodeType.METHOD and "_confingy_object" in config:
                self.build_from_config(
                    config["_confingy_object"], node_id, "self", depth + 1
                )
            return node_id

        # Handle type references
        if node_type == NodeType.TYPE:
            node_id = self._generate_node_id("type")
            type_name = config.get("_confingy_name", "Type")
            module = config.get("_confingy_module", "")

            self.nodes[node_id] = ConfigNode(
                node_id=node_id,
                node_type=node_type,
                label=f"type[{type_name}]",
                class_name=type_name,
                module=module,
                depth=depth,
            )
            if parent_id and field_name:
                self.edges.append(ConfigEdge(parent_id, node_id, label=field_name))
            return node_id

        # Handle generic dictionaries
        if isinstance(config, dict):
            node_id = self._generate_node_id("dict")
            self.nodes[node_id] = ConfigNode(
                node_id=node_id,
                node_type=NodeType.COLLECTION,
                label=f"dict[{len(config)}]",
                depth=depth,
            )
            if parent_id and field_name:
                self.edges.append(ConfigEdge(parent_id, node_id, label=field_name))

            for key, value in config.items():
                self.build_from_config(value, node_id, str(key), depth + 1)
            return node_id

        # Fallback for unknown types
        node_id = self._generate_node_id("unknown")
        self.nodes[node_id] = ConfigNode(
            node_id=node_id,
            node_type=NodeType.PRIMITIVE,
            label=str(type(config).__name__),
            value=config,
            depth=depth,
        )
        if parent_id and field_name:
            self.edges.append(ConfigEdge(parent_id, node_id, label=field_name))
        return node_id

    def to_cytoscape_json(
        self, max_depth: int = 1, expanded_nodes: set[str] | None = None
    ) -> dict[str, Any]:
        """
        Convert the DAG to Cytoscape.js compatible JSON format.

        Args:
            max_depth: Maximum depth to show initially (default 1 = root + immediate children)
            expanded_nodes: Set of node IDs that should show their children regardless of depth

        Returns:
            Dictionary with 'nodes' and 'edges' in Cytoscape format
        """
        if expanded_nodes is None:
            expanded_nodes = set()

        cytoscape_nodes = []
        cytoscape_edges = []
        visible_nodes = set()

        # Determine which nodes should be visible
        for node_id, node in self.nodes.items():
            # Always show nodes up to max_depth
            if node.depth <= max_depth:
                visible_nodes.add(node_id)

            # Also show children of expanded nodes
            if expanded_nodes:
                for edge in self.edges:
                    if edge.from_node in expanded_nodes and edge.to_node == node_id:
                        visible_nodes.add(node_id)

        # Create Cytoscape nodes (sorted for consistent ordering)
        for node_id in sorted(visible_nodes):
            node = self.nodes[node_id]

            # Check if this node has any children at all
            has_any_children = any(edge.from_node == node_id for edge in self.edges)

            # Check if this node has hidden children (not currently visible)
            has_hidden_children = any(
                edge.from_node == node_id and edge.to_node not in visible_nodes
                for edge in self.edges
            )

            # Determine node style based on type
            node_class = node.node_type.value
            if node.node_type == NodeType.LAZY_CLASS:
                node_class += " lazy"

            # Add diff styling for comparison mode
            if node.diff_type:
                node_class += f" diff-{node.diff_type.value}"

            cytoscape_node: dict[str, Any] = {
                "data": {
                    "id": node_id,
                    "label": node.label,
                    "node_type": node.node_type.value,
                    "depth": node.depth,
                    "has_any_children": has_any_children,  # True if node has any children at all
                    "has_hidden_children": has_hidden_children,  # True if some children are hidden
                    "expanded": node_id in expanded_nodes
                    and has_any_children,  # Only expanded if has children
                    "class_name": node.class_name,
                    "module": node.module,
                    "class_hash": node.class_hash,
                    "metadata": node.metadata or {},
                    "diff_type": node.diff_type.value if node.diff_type else None,
                },
                "classes": node_class,
            }

            # Add value for primitives (truncated)
            if node.node_type == NodeType.PRIMITIVE and node.value is not None:
                cytoscape_node["data"]["value"] = str(node.value)[:100]

            cytoscape_nodes.append(cytoscape_node)

        # Create Cytoscape edges (only between visible nodes) - sorted for consistency
        for edge in sorted(
            self.edges, key=lambda e: (e.from_node, e.label or "", e.to_node)
        ):
            if edge.from_node in visible_nodes and edge.to_node in visible_nodes:
                cytoscape_edge = {
                    "data": {
                        "id": f"{edge.from_node}-{edge.to_node}",
                        "source": edge.from_node,
                        "target": edge.to_node,
                        "label": edge.label or "",
                        "edge_type": edge.edge_type,
                    },
                    "classes": edge.edge_type,
                }
                cytoscape_edges.append(cytoscape_edge)

        return {"nodes": cytoscape_nodes, "edges": cytoscape_edges}

    def get_node_children(self, node_id: str) -> list[str]:
        """Get all direct children of a node."""
        return [edge.to_node for edge in self.edges if edge.from_node == node_id]

    def create_signature(self, node: ConfigNode) -> str:
        """Create a signature for a node to enable comparison.

        The signature should identify the node's structural position,
        NOT its value (values are compared separately after matching).
        """
        # Create a signature based on node type and structure, not values
        if node.node_type == NodeType.PRIMITIVE:
            # For primitives, use type only - values compared separately
            return "primitive"
        elif node.node_type in (
            NodeType.DATACLASS,
            NodeType.TRACKED_CLASS,
            NodeType.LAZY_CLASS,
        ):
            return f"{node.node_type.value}:{node.class_name}:{node.module}"
        elif node.node_type == NodeType.COLLECTION:
            # For collections, include type but not size/contents
            collection_type = (
                node.label.split("[")[0] if "[" in node.label else node.label
            )
            return f"collection:{collection_type}"
        else:
            return f"{node.node_type.value}:{node.label}"

    @staticmethod
    def create_comparison_dag(
        dag1: "ConfigGraph", dag2: "ConfigGraph"
    ) -> "ConfigGraph":
        """
        Create a comparison DAG showing differences between two configurations.

        This method matches nodes between the two DAGs and creates a unified view
        with diff annotations.
        """
        comparison_dag = ConfigGraph()

        # Build path-based signatures for better matching
        def build_path_signatures(dag: ConfigGraph) -> dict[str, str]:
            """Build path-based signatures for nodes."""
            path_sigs = {}

            # Find root nodes (no incoming edges)
            root_nodes = set(dag.nodes.keys())
            for edge in dag.edges:
                root_nodes.discard(edge.to_node)

            # Build paths from root to each node
            def build_path(node_id: str, path: list[str] | None = None) -> str:
                if path is None:
                    path = []

                node = dag.nodes[node_id]
                current_sig = comparison_dag.create_signature(node)

                # Include edge label in path for better matching
                # Find incoming edge to get field name
                edge_label = ""
                for edge in dag.edges:
                    if edge.to_node == node_id:
                        edge_label = edge.label or ""
                        break

                # Combine signature with edge label for unique path component
                if edge_label:
                    path_component = f"{edge_label}:{current_sig}"
                else:
                    path_component = current_sig

                new_path = path + [path_component]
                path_sigs[node_id] = "/".join(new_path)

                # Recurse to children
                for child_id in dag.get_node_children(node_id):
                    build_path(child_id, new_path)

                return path_sigs[node_id]

            # Build paths from all roots
            for root_id in root_nodes:
                build_path(root_id)

            return path_sigs

        # Build signatures for both DAGs
        sig1 = build_path_signatures(dag1)
        sig2 = build_path_signatures(dag2)

        # Create reverse mapping from signature to node_id
        sig_to_node1 = {sig: node_id for node_id, sig in sig1.items()}
        sig_to_node2 = {sig: node_id for node_id, sig in sig2.items()}

        all_sigs = set(sig1.values()) | set(sig2.values())

        # Process each unique signature
        for sig in sorted(all_sigs):
            node1_id = sig_to_node1.get(sig)
            node2_id = sig_to_node2.get(sig)

            if node1_id and node2_id:
                # Node exists in both - check if changed
                node1 = dag1.nodes[node1_id]
                node2 = dag2.nodes[node2_id]

                # Compare node content
                changed = False
                old_value = None
                if node1.value != node2.value:
                    changed = True
                    old_value = node1.value
                if node1.label != node2.label:
                    changed = True
                if node1.class_hash != node2.class_hash:
                    changed = True

                # Use node2 as the base (newer version)
                new_node = ConfigNode(
                    node_id=comparison_dag._generate_node_id("comp"),
                    node_type=node2.node_type,
                    label=node2.label,
                    value=node2.value,
                    class_name=node2.class_name,
                    module=node2.module,
                    class_hash=node2.class_hash,
                    metadata=node2.metadata or {},
                    depth=node2.depth,
                    diff_type=DiffType.CHANGED if changed else DiffType.UNCHANGED,
                )

                # Store old value in metadata for tooltips
                if old_value is not None and new_node.metadata is not None:
                    new_node.metadata["_old_value"] = old_value
                # Store old class hash if it changed
                if (
                    node1.class_hash
                    and node2.class_hash
                    and node1.class_hash != node2.class_hash
                    and new_node.metadata is not None
                ):
                    new_node.metadata["_old_class_hash"] = node1.class_hash

            elif node1_id and not node2_id:
                # Node removed in dag2
                node1 = dag1.nodes[node1_id]
                new_node = ConfigNode(
                    node_id=comparison_dag._generate_node_id("comp"),
                    node_type=node1.node_type,
                    label=node1.label,
                    value=node1.value,
                    class_name=node1.class_name,
                    module=node1.module,
                    class_hash=node1.class_hash,
                    metadata=node1.metadata,
                    depth=node1.depth,
                    diff_type=DiffType.REMOVED,
                )

            else:  # node2_id and not node1_id
                # Node added in dag2
                assert node2_id is not None
                node2 = dag2.nodes[node2_id]
                new_node = ConfigNode(
                    node_id=comparison_dag._generate_node_id("comp"),
                    node_type=node2.node_type,
                    label=node2.label,
                    value=node2.value,
                    class_name=node2.class_name,
                    module=node2.module,
                    class_hash=node2.class_hash,
                    metadata=node2.metadata,
                    depth=node2.depth,
                    diff_type=DiffType.ADDED,
                )

            # Store the node with signature as metadata for edge reconstruction
            new_node.metadata = new_node.metadata or {}
            new_node.metadata["_comparison_signature"] = sig
            comparison_dag.nodes[new_node.node_id] = new_node

        # Reconstruct edges based on the original relationships
        # We need to use the signatures to match edges
        node_by_sig = {
            node.metadata["_comparison_signature"]: node.node_id
            for node in comparison_dag.nodes.values()
            if node.metadata is not None and "_comparison_signature" in node.metadata
        }

        # Process edges from both DAGs
        all_edges = []
        for edge in dag1.edges:
            from_sig = sig1.get(edge.from_node)
            to_sig = sig1.get(edge.to_node)
            if (
                from_sig
                and to_sig
                and from_sig in node_by_sig
                and to_sig in node_by_sig
            ):
                all_edges.append((from_sig, to_sig, edge.label, edge.edge_type))

        for edge in dag2.edges:
            from_sig = sig2.get(edge.from_node)
            to_sig = sig2.get(edge.to_node)
            if (
                from_sig
                and to_sig
                and from_sig in node_by_sig
                and to_sig in node_by_sig
            ):
                edge_tuple = (from_sig, to_sig, edge.label, edge.edge_type)
                if edge_tuple not in all_edges:
                    all_edges.append(edge_tuple)

        # Create edges in comparison DAG
        for from_sig, to_sig, label, edge_type in all_edges:
            from_node_id = node_by_sig.get(from_sig)
            to_node_id = node_by_sig.get(to_sig)
            if from_node_id and to_node_id:
                comparison_dag.edges.append(
                    ConfigEdge(
                        from_node=from_node_id,
                        to_node=to_node_id,
                        label=label,
                        edge_type=edge_type,
                    )
                )

        # Propagate diff status to parent nodes
        # A parent should be marked as changed if any child is changed/added/removed
        def propagate_diff_status():
            changed = True
            while changed:
                changed = False
                for edge in comparison_dag.edges:
                    parent = comparison_dag.nodes.get(edge.from_node)
                    child = comparison_dag.nodes.get(edge.to_node)
                    if parent and child:
                        # If child has any diff and parent is unchanged, mark parent as changed
                        if child.diff_type in (
                            DiffType.CHANGED,
                            DiffType.ADDED,
                            DiffType.REMOVED,
                        ):
                            if parent.diff_type == DiffType.UNCHANGED:
                                parent.diff_type = DiffType.CHANGED
                                changed = True

        propagate_diff_status()

        return comparison_dag
