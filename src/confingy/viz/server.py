"""
FastAPI server for interactive configuration visualization using Cytoscape.js.

WARNING: this was vibe-coded
"""

from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from confingy.viz.default_configs import get_default_configs
from confingy.viz.graph import ConfigGraph

app = FastAPI(title="Confingy Visualization Server")

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage
# Single configurations that can be selected in dropdowns
stored_configs: dict[str, dict[str, Any]] = {}
stored_dags: dict[str, ConfigGraph] = {}

# Active sessions (both single config views and comparisons)
active_sessions: dict[str, dict[str, Any]] = {}
session_dags: dict[str, ConfigGraph] = {}

# Expanded nodes for each session
expanded_nodes: dict[str, set[str]] = {}


def get_dag(session_id: str) -> ConfigGraph:
    """Get DAG from either stored_dags or session_dags."""
    if session_id in stored_dags:
        return stored_dags[session_id]
    elif session_id in session_dags:
        return session_dags[session_id]
    else:
        raise HTTPException(status_code=404, detail="Session not found")


def populate_default_configs():
    """Populate the server with default example configurations."""
    # Get default configs and populate server storage
    default_configs = get_default_configs()

    # Only store single configs - comparisons will be created dynamically
    for config_name, config_data in default_configs.items():
        config_id = config_name.lower().replace(" ", "_")

        # Build DAG and store in permanent storage
        dag = ConfigGraph()
        dag.build_from_config(config_data)
        stored_dags[config_id] = dag
        stored_configs[config_id] = {
            "config": config_data,
            "title": config_name,
        }
        expanded_nodes[config_id] = set()


class VisualizationRequest(BaseModel):
    """Request model for creating a new visualization."""

    config: dict[str, Any]
    title: str | None = None
    session_id: str | None = "default"


class ExpansionRequest(BaseModel):
    """Request model for expanding/collapsing a node."""

    session_id: str
    node_id: str
    expand: bool = True


class ComparisonRequest(BaseModel):
    """Request model for comparing two configurations."""

    config1: dict[str, Any]
    config2: dict[str, Any]
    title: str | None = None
    session_id: str | None = "comparison"


@app.get("/api/list_configs")
async def list_configs():
    """List all available default configurations."""
    from .default_configs import get_default_configs

    configs = get_default_configs()
    return JSONResponse(content=configs)


@app.get("/api/list_stored_configs")
async def list_stored_configs():
    """List all currently stored configurations (includes default and uploaded ones)."""
    config_list = []
    for session_id, config_info in stored_configs.items():
        config_list.append(
            {
                "id": session_id,
                "title": config_info.get("title", session_id),
                "type": "comparison"
                if config_info.get("comparison", False)
                else "single",
            }
        )
    return {"configs": config_list}


@app.post("/api/load_config")
async def load_config(request: dict):
    """Load a specific configuration by name."""
    config_name = request.get("config_name")
    if not config_name:
        raise HTTPException(status_code=400, detail="config_name is required")

    try:
        from .default_configs import get_default_configs

        configs = get_default_configs()

        if config_name not in configs:
            raise HTTPException(
                status_code=404, detail=f"Configuration '{config_name}' not found"
            )

        config = configs[config_name]

        # Create DAG from config
        dag = ConfigGraph()
        dag.build_from_config(config)

        # Generate a unique session ID for this config
        import uuid

        session_id = f"config_{uuid.uuid4().hex[:8]}"

        # Store the DAG and config
        stored_dags[session_id] = dag
        stored_configs[session_id] = {
            "config": config,
            "comparison": False,
            "title": config_name,
        }
        expanded_nodes[session_id] = set()

        # Return graph data
        graph_data = dag.to_cytoscape_json(max_depth=1)
        return JSONResponse(
            content={"session_id": session_id, "graph_data": graph_data}
        )

    except ImportError as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to load default configs: {e}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to load configuration: {e}"
        )


@app.post("/api/compare_stored_configs")
async def compare_stored_configs(request: dict):
    """Compare two configurations by their session IDs."""
    config1_id = request.get("config1_id")
    config2_id = request.get("config2_id")

    if not config1_id or not config2_id:
        raise HTTPException(
            status_code=400, detail="Both config1_id and config2_id are required"
        )

    if config1_id == config2_id:
        raise HTTPException(
            status_code=400, detail="Cannot compare a configuration with itself"
        )

    if config1_id not in stored_configs or config2_id not in stored_configs:
        missing = []
        if config1_id not in stored_configs:
            missing.append(config1_id)
        if config2_id not in stored_configs:
            missing.append(config2_id)
        raise HTTPException(
            status_code=404, detail=f"Configuration(s) not found: {', '.join(missing)}"
        )

    try:
        # Get the stored configs - need to handle both formats
        config1_info = stored_configs[config1_id]
        config2_info = stored_configs[config2_id]

        # Extract the actual config data
        if "config" in config1_info:
            config1 = config1_info["config"]
        else:
            # This might be an older format or comparison config
            config1 = config1_info

        if "config" in config2_info:
            config2 = config2_info["config"]
        else:
            config2 = config2_info

        # Create DAGs for both configs
        dag1 = ConfigGraph()
        dag1.build_from_config(config1)

        dag2 = ConfigGraph()
        dag2.build_from_config(config2)

        # Create comparison DAG
        comparison_dag = ConfigGraph.create_comparison_dag(dag1, dag2)

        # Generate unique session ID for comparison
        import uuid

        session_id = f"compare_{uuid.uuid4().hex[:8]}"

        # Store comparison in active sessions (not permanent storage)
        session_dags[session_id] = comparison_dag
        active_sessions[session_id] = {
            "config1": config1,
            "config2": config2,
            "comparison": True,
            "title": f"{config1_info.get('title', config1_id)} vs {config2_info.get('title', config2_id)}",
        }
        expanded_nodes[session_id] = set()

        # Return graph data
        graph_data = comparison_dag.to_cytoscape_json(max_depth=1)
        return JSONResponse(
            content={"session_id": session_id, "graph_data": graph_data}
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to compare configurations: {e}"
        )


@app.get("/", response_class=HTMLResponse)
async def serve_visualization():
    """Serve the main visualization HTML page."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fingy Viz</title>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.23.0/cytoscape.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/dagre@0.8.5/dist/dagre.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/cytoscape-dagre@2.5.0/cytoscape-dagre.min.js"></script>
    <style>
        body {
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: #f5f5f5;
            display: flex;
            flex-direction: column;
            height: 100vh;
        }
        #header {
            background: #333;
            color: white;
            padding: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        #header h1 {
            margin: 0;
            font-size: 1.5rem;
        }
        #controls {
            background: white;
            padding: 1rem;
            border-bottom: 1px solid #ddd;
            display: flex;
            gap: 1rem;
            align-items: center;
            flex-wrap: wrap;
        }
        #controls button {
            padding: 0.5rem 1rem;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        #controls button:hover {
            background: #45a049;
        }
        #controls button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        #legendContainer {
            background: white;
            padding: 10px 15px;
            border-bottom: 1px solid #ddd;
            text-align: center;
            font-size: 14px;
        }
        #graph-container {
            flex: 1;
            position: relative;
            display: flex;
        }
        #cy {
            width: 100%;
            flex: 1;
            display: block;
            background: #fafafa;
        }
        .info-box {
            position: absolute;
            top: 10px;
            right: 10px;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            max-width: 300px;
            font-size: 14px;
            z-index: 1000;
        }
        .info-box h3 {
            margin-top: 0;
            color: #333;
        }
        .info-box .legend-item {
            display: flex;
            align-items: center;
            margin: 8px 0;
        }
        .info-box .color-box {
            width: 20px;
            height: 20px;
            margin-right: 10px;
            border: 1px solid #ccc;
            border-radius: 4px;
        }
        .comparison-mode .added { background: #90EE90; }
        .comparison-mode .removed { background: #FFB6C1; }
        .comparison-mode .changed { background: #FFD700; }
        .comparison-mode .unchanged { background: #D3D3D3; }
        .loading {
            display: none;
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            z-index: 2000;
        }
        .tooltip {
            display: none;
            position: fixed;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 12px;
            max-width: 300px;
            z-index: 3000;
            pointer-events: none;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }
    </style>
</head>
<body>
    <div id="header">
        <h1>🔍 Fingy Viz</h1>
    </div>

    <div id="controls">
        <div style="display: flex; align-items: center; gap: 15px; flex-wrap: wrap;">
            <label style="font-weight: bold;">View Mode:</label>
            <select id="viewMode" style="padding: 5px;">
                <option value="single">Single Configuration</option>
                <option value="comparison">Configuration Comparison</option>
            </select>

            <!-- Single Config Mode -->
            <div id="singleConfigSection" style="display: flex; align-items: center; gap: 15px;">
                <label style="font-weight: bold;">Configuration:</label>
                <select id="configSelector" style="padding: 5px; min-width: 200px;">
                    <option value="">Select a configuration...</option>
                </select>
            </div>

            <!-- Comparison Mode -->
            <div id="comparisonSection" style="display: none; gap: 15px; align-items: center;">
                <label style="font-weight: bold;">Compare:</label>
                <div>
                    <label>Config 1:</label>
                    <select id="compareConfig1" style="padding: 5px; margin-left: 5px; min-width: 150px;">
                        <option value="">Select config...</option>
                    </select>
                </div>
                <div>
                    <label>Config 2:</label>
                    <select id="compareConfig2" style="padding: 5px; margin-left: 5px; min-width: 150px;">
                        <option value="">Select config...</option>
                    </select>
                </div>
                <button id="compareSelected" disabled>Compare</button>
            </div>

            <div style="border-left: 1px solid #ddd; padding-left: 15px; display: flex; align-items: center; gap: 10px;">
                <label style="font-weight: bold;">Upload Config:</label>
                <input type="file" id="configFile" accept=".json" style="padding: 3px;">
                <button id="uploadConfig" disabled>Add to Library</button>
            </div>

        </div>

        <div style="flex-grow: 1;"></div>
        <button id="resetView">Reset View</button>
        <button id="fitView">Fit to Screen</button>
        <button id="expandAll">Expand All</button>
        <button id="collapseAll">Collapse All</button>
        <button id="exportImage">Export as Image</button>
        <span id="status"></span>
    </div>

    <div id="legendContainer">
        <!-- Single config legend (node types) -->
        <div id="singleLegend" style="display: block;">
            <span style="font-weight: bold; margin-right: 15px;">Node Types:</span>
            <span style="display: inline-block; width: 15px; height: 15px; background: #90EE90; border: 1px solid #ccc; margin: 0 5px; vertical-align: middle;"></span>@dataclass
            <span style="display: inline-block; width: 15px; height: 15px; background: #FFB6C1; border: 1px solid #ccc; margin: 0 5px; vertical-align: middle;"></span>@track
            <span style="display: inline-block; width: 15px; height: 15px; background: #FFD700; border: 1px dashed #ccc; margin: 0 5px; vertical-align: middle;"></span>lazy()
            <span style="display: inline-block; width: 15px; height: 15px; background: #E0E0E0; border: 1px solid #ccc; margin: 0 5px; vertical-align: middle;"></span>Primitive
            <span style="display: inline-block; width: 15px; height: 15px; background: #ADD8E6; border: 1px solid #ccc; margin: 0 5px; vertical-align: middle;"></span>Collection
        </div>

        <!-- Comparison legend -->
        <div id="comparisonLegend" style="display: none;">
            <span style="font-weight: bold; margin-right: 15px;">Comparison Legend:</span>
            <span style="display: inline-block; width: 15px; height: 15px; background: #d4edda; border: 2px solid #28a745; margin: 0 5px; vertical-align: middle;"></span>Added (new classes, fields)
            <span style="display: inline-block; width: 15px; height: 15px; background: #f8d7da; border: 2px solid #dc3545; margin: 0 5px; vertical-align: middle;"></span>Removed (deleted classes, fields)
            <span style="display: inline-block; width: 15px; height: 15px; background: #fff3cd; border: 2px solid #ffc107; margin: 0 5px; vertical-align: middle;"></span>Changed (modified values, implementations)
            <span style="display: inline-block; width: 15px; height: 15px; background: #f8f9fa; border: 1px solid #6c757d; margin: 0 5px; vertical-align: middle;"></span>Unchanged
        </div>
    </div>

    <div id="graph-container">
        <div id="cy"></div>
    </div>
    <div id="tooltip" class="tooltip"></div>


    <div class="loading" id="loading">
        <p>Loading visualization...</p>
    </div>

    <script>
        let cy;
        let currentSessionId = 'default';
        let comparisonMode = false;
        let isComparisonMode = false;
        let comparisonData = null;
        let nodePositions = {};  // Store node positions to preserve layout
        let occupiedByDepth = {};  // Global tracker for occupied regions at each depth
        let originalYPositions = {};  // Remember original Y positions for each parent at each depth

        // Node styles
        const nodeStyles = {
            'dataclass': { 'background-color': '#90EE90' },
            'tracked': { 'background-color': '#FFB6C1' },
            'lazy': { 'background-color': '#FFD700', 'border-style': 'dashed', 'border-width': 2 },
            'primitive': { 'background-color': '#E0E0E0', 'shape': 'ellipse' },
            'collection': { 'background-color': '#ADD8E6' },
            'method': { 'background-color': '#DDA0DD', 'shape': 'ellipse' },
            'function': { 'background-color': '#E6E6FA', 'shape': 'ellipse' },
            'type': { 'background-color': '#F5DEB3', 'shape': 'diamond' }
        };

        // Comparison styles
        const comparisonStyles = {
            'added': { 'background-color': '#90EE90', 'border-color': '#228B22', 'border-width': 3 },
            'removed': { 'background-color': '#FFB6C1', 'border-color': '#DC143C', 'border-width': 3 },
            'changed': { 'background-color': '#FFD700', 'border-color': '#FF8C00', 'border-width': 3 },
            'unchanged': { 'background-color': '#D3D3D3' }
        };

        function initCytoscape(data) {
            cy = cytoscape({
                container: document.getElementById('cy'),

                elements: data,

                style: [
                    {
                        selector: 'node',
                        style: {
                            'label': 'data(label)',
                            'text-valign': 'center',
                            'text-halign': 'center',
                            'font-size': '12px',
                            'width': '150px',
                            'height': '40px',
                            'text-wrap': 'wrap',
                            'text-max-width': '140px',
                            'border-color': '#999',
                            'border-width': 1,
                            'compound-sizing-wrt-labels': 'include'
                        }
                    },
                    {
                        selector: 'node[has_any_children]',
                        style: {
                            'label': function(ele) {
                                const label = ele.data('label');
                                const hasChildren = ele.data('has_any_children');
                                const hasHiddenChildren = ele.data('has_hidden_children');
                                const expanded = ele.data('expanded');

                                // Only show [+/-] if node has children
                                if (!hasChildren) {
                                    return label;
                                }

                                // Show [-] if expanded and has visible children
                                // Show [+] if collapsed or has hidden children
                                if (expanded && !hasHiddenChildren) {
                                    return label + ' [-]';
                                } else if (hasHiddenChildren || !expanded) {
                                    return label + ' [+]';
                                } else {
                                    return label;
                                }
                            },
                            'font-weight': 'bold'
                        }
                    },
                    {
                        selector: 'node.dataclass',
                        style: nodeStyles['dataclass']
                    },
                    {
                        selector: 'node.tracked',
                        style: nodeStyles['tracked']
                    },
                    {
                        selector: 'node.lazy',
                        style: nodeStyles['lazy']
                    },
                    {
                        selector: 'node.primitive',
                        style: nodeStyles['primitive']
                    },
                    {
                        selector: 'node.collection',
                        style: nodeStyles['collection']
                    },
                    {
                        selector: 'node.method',
                        style: nodeStyles['method']
                    },
                    {
                        selector: 'node.function',
                        style: nodeStyles['function']
                    },
                    {
                        selector: 'node.type',
                        style: nodeStyles['type']
                    },
                    {
                        selector: 'edge',
                        style: {
                            'width': 2,
                            'line-color': '#999',
                            'target-arrow-color': '#999',
                            'target-arrow-shape': 'triangle',
                            'curve-style': 'bezier',
                            'label': 'data(label)',
                            'font-size': '12px',
                            'text-rotation': 'autorotate',
                            'text-background-color': '#ffffff',
                            'text-background-opacity': 0.8,
                            'text-background-padding': '3px'
                        }
                    },
                    {
                        selector: 'edge.arg',
                        style: {
                            'line-color': '#4169E1',
                            'target-arrow-color': '#4169E1'
                        }
                    },
                    {
                        selector: 'edge.config',
                        style: {
                            'line-color': '#FF8C00',
                            'target-arrow-color': '#FF8C00'
                        }
                    },
                    {
                        selector: 'edge.lazy_ref',
                        style: {
                            'line-color': '#DC143C',
                            'target-arrow-color': '#DC143C',
                            'line-style': 'dashed'
                        }
                    },
                    // Diff styles for comparison mode
                    {
                        selector: 'node.diff-added',
                        style: {
                            'border-color': '#28a745',
                            'border-width': '3px',
                            'background-color': '#d4edda'
                        }
                    },
                    {
                        selector: 'node.diff-removed',
                        style: {
                            'border-color': '#dc3545',
                            'border-width': '3px',
                            'background-color': '#f8d7da',
                            'opacity': 0.7
                        }
                    },
                    {
                        selector: 'node.diff-changed',
                        style: {
                            'border-color': '#ffc107',
                            'border-width': '3px',
                            'background-color': '#fff3cd'
                        }
                    },
                    {
                        selector: 'node.diff-unchanged',
                        style: {
                            'border-color': '#6c757d',
                            'border-width': '1px',
                            'background-color': '#f8f9fa'
                        }
                    }
                ],

                layout: {
                    name: 'dagre',
                    rankDir: 'TB',
                    nodeSep: 100,  // Much larger node separation
                    edgeSep: 20,
                    rankSep: 120,  // Larger rank separation
                    randomize: false,  // Ensure deterministic layout
                    animate: false     // Disable animation for initial layout
                },

                wheelSensitivity: 0.2,
                minZoom: 0.1,
                maxZoom: 3
            });

            // Store initial positions after layout
            cy.on('layoutstop', function() {
                cy.nodes().forEach(node => {
                    nodePositions[node.id()] = node.position();
                });
            });

            // Add click handler for expand/collapse
            cy.on('tap', 'node', function(evt) {
                const node = evt.target;
                // Only allow toggle if node has any children
                if (node.data('has_any_children')) {
                    toggleNode(node.id());
                }
            });

            // Add hover handler for tooltips
            const tooltipDiv = document.getElementById('tooltip');

            cy.on('mouseover', 'node', function(evt) {
                const node = evt.target;
                const data = node.data();
                let tooltip = `<strong>${data.label}</strong><br>`;

                // Add type information
                if (data.classes && data.classes.length > 0) {
                    tooltip += `Type: ${data.classes.join(', ')}<br>`;
                }

                // Show module and hash for tracked/lazy classes
                if (data.module) {
                    tooltip += `Module: <code>${data.module}</code><br>`;
                }
                if (data.class_hash) {
                    // Show old hash if this is a changed class
                    if (data.diff_type === 'changed' && data.metadata && data.metadata._old_class_hash) {
                        tooltip += `Old hash: <code style="color: #dc3545; font-family: monospace; font-size: 0.8em;">${data.metadata._old_class_hash.substring(0, 12)}...</code><br>`;
                        tooltip += `New hash: <code style="color: #28a745; font-family: monospace; font-size: 0.8em;">${data.class_hash.substring(0, 12)}...</code><br>`;
                        tooltip += `<span style="color: #ffc107; font-size: 0.9em;">⚠️ Class implementation changed</span><br>`;
                    } else {
                        tooltip += `Hash: <code style="font-family: monospace; font-size: 0.8em;">${data.class_hash.substring(0, 12)}...</code><br>`;
                    }
                }

                // Show diff information in comparison mode
                if (data.diff_type) {
                    tooltip += `<span style="font-weight: bold; color: `;
                    switch(data.diff_type) {
                        case 'added':
                            tooltip += `#28a745;">✚ Added</span><br>`;
                            break;
                        case 'removed':
                            tooltip += `#dc3545;">✖ Removed</span><br>`;
                            break;
                        case 'changed':
                            tooltip += `#ffc107;">✎ Changed</span><br>`;
                            break;
                        case 'unchanged':
                            tooltip += `#6c757d;">✓ Unchanged</span><br>`;
                            break;
                    }
                }

                // Show field name and value for primitives
                if (data.value !== undefined && data.value !== null) {
                    // Try to get the field name from the incoming edge
                    const incomingEdges = node.incomers('edge');
                    let fieldName = '';
                    if (incomingEdges.length > 0) {
                        const edgeLabel = incomingEdges[0].data('label');
                        if (edgeLabel) {
                            fieldName = edgeLabel;
                            tooltip += `Field: <code>${fieldName}</code><br>`;
                        }
                    }

                    // Show old value if this is a changed node
                    if (data.metadata && data.metadata._old_value !== undefined) {
                        tooltip += `Old value: <code style="color: #dc3545;">${data.metadata._old_value}</code><br>`;
                        tooltip += `New value: <code style="color: #28a745;">${data.value}</code><br>`;
                    } else {
                        tooltip += `Value: <code>${data.value}</code><br>`;
                    }
                }

                // Show metadata fields and args
                if (data.metadata && Object.keys(data.metadata).length > 0) {
                    if (data.metadata.fields && data.metadata.fields.length > 0) {
                        tooltip += `Fields: <code>${data.metadata.fields.join(', ')}</code><br>`;
                    }
                    if (data.metadata.init_args && data.metadata.init_args.length > 0) {
                        tooltip += `Args: <code>${data.metadata.init_args.join(', ')}</code><br>`;
                    }
                }

                // Show the tooltip near the mouse cursor
                tooltipDiv.innerHTML = tooltip;
                tooltipDiv.style.display = 'block';

                // Position tooltip near the node but avoid edges
                const containerPos = document.getElementById('cy').getBoundingClientRect();
                const nodePos = node.renderedPosition();
                let tooltipX = containerPos.left + nodePos.x + 10;
                let tooltipY = containerPos.top + nodePos.y - 10;

                // Keep tooltip in viewport
                if (tooltipX + 300 > window.innerWidth) {
                    tooltipX = containerPos.left + nodePos.x - 310;
                }
                if (tooltipY < 0) {
                    tooltipY = containerPos.top + nodePos.y + 30;
                }

                tooltipDiv.style.left = tooltipX + 'px';
                tooltipDiv.style.top = tooltipY + 'px';

                // Highlight the node
                node.style('opacity', 0.8);
            });

            cy.on('mouseout', 'node', function(evt) {
                // Hide tooltip and restore node opacity
                tooltipDiv.style.display = 'none';
                evt.target.style('opacity', 1);
            });

            // Also hide tooltip when mouse leaves the cy container
            document.getElementById('cy').addEventListener('mouseleave', function() {
                tooltipDiv.style.display = 'none';
            });
        }

        async function toggleNode(nodeId) {
            const node = cy.getElementById(nodeId);
            const isExpanded = node.data('expanded');
            const hasHiddenChildren = node.data('has_hidden_children');

            // Determine whether to expand or collapse:
            // - If node has hidden children, expand
            // - If node is expanded and has no hidden children, collapse
            const shouldExpand = hasHiddenChildren || !isExpanded;

            document.getElementById('loading').style.display = 'block';

            try {
                const response = await fetch('/api/toggle_node', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        session_id: currentSessionId,
                        node_id: nodeId,
                        expand: shouldExpand
                    })
                });

                if (response.ok) {
                    const data = await response.json();
                    updateGraph(data);
                }
            } catch (error) {
                console.error('Error toggling node:', error);
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        }

        function updateGraph(data) {
            // Store current viewport
            const pan = cy.pan();
            const zoom = cy.zoom();

            // Store positions of all existing nodes before removing
            const existingPositions = {};
            cy.nodes().forEach(node => {
                existingPositions[node.id()] = node.position();
            });

            // Identify nodes that exist in both old and new graphs
            const oldNodeIds = new Set(cy.nodes().map(n => n.id()));
            const newNodeIds = new Set(data.nodes.map(n => n.data.id));
            const persistingNodes = new Set([...oldNodeIds].filter(id => newNodeIds.has(id)));
            const newNodes = new Set([...newNodeIds].filter(id => !oldNodeIds.has(id)));

            // Clean up occupied regions for parents whose children are no longer present
            // Build a set of parents that currently have children in the new graph
            const parentsWithChildren = new Set();
            data.edges.forEach(edge => {
                parentsWithChildren.add(edge.data.source);
            });

            console.log(`Parents with children in new graph:`, Array.from(parentsWithChildren));

            // Remove regions for parents that no longer have children (i.e., were collapsed)
            let totalCleanedRegions = 0;
            Object.keys(occupiedByDepth).forEach(depth => {
                const initialCount = occupiedByDepth[depth].length;
                occupiedByDepth[depth] = occupiedByDepth[depth].filter(region => {
                    // Keep region only if its parent still has children in the current graph
                    const keep = parentsWithChildren.has(region.parentId);
                    if (!keep) {
                        console.log(`Removing region for parent ${region.parentId} (no longer has children) at depth ${depth}`);
                        totalCleanedRegions++;
                    }
                    return keep;
                });
                const finalCount = occupiedByDepth[depth].length;
                if (initialCount !== finalCount) {
                    console.log(`Cleaned depth ${depth}: ${initialCount} → ${finalCount} regions`);
                }
            });

            // DON'T clean up original positions - keep them as permanent "reserved slots"
            // This allows nodes to reclaim their original positions when re-expanded

            if (totalCleanedRegions > 0) {
                console.log(`Total regions cleaned up: ${totalCleanedRegions}`);
            }
            console.log(`Original positions maintained:`, Object.keys(originalYPositions));

            // Update the graph
            cy.elements().remove();
            cy.add(data);

            // First pass: restore positions for persisting nodes
            cy.nodes().forEach(node => {
                if (persistingNodes.has(node.id())) {
                    node.position(existingPositions[node.id()]);
                }
            });

            // Second pass: position new nodes based on parent if expanding
            if (newNodes.size > 0) {
                console.log(`Positioning ${newNodes.size} new nodes`);
                // Create a simple layout for new nodes only
                const nodesToLayout = cy.nodes().filter(n => newNodes.has(n.id()));

                // Group new nodes by their parent
                const nodesByParent = {};
                nodesToLayout.forEach(node => {
                    // Find parent by checking edges
                    const incomingEdges = node.connectedEdges().filter(e => e.target().id() === node.id());
                    if (incomingEdges.length > 0) {
                        const parentId = incomingEdges[0].source().id();
                        const edgeLabel = incomingEdges[0].data('label') || '';
                        if (!nodesByParent[parentId]) {
                            nodesByParent[parentId] = [];
                        }
                        nodesByParent[parentId].push({node: node, label: edgeLabel});
                    }
                });

                // Group parents by their depth to handle them level by level
                const parentsByDepth = {};
                Object.keys(nodesByParent).forEach(parentId => {
                    const parent = cy.getElementById(parentId);
                    const depth = parent.data('depth');
                    if (!parentsByDepth[depth]) {
                        parentsByDepth[depth] = [];
                    }
                    parentsByDepth[depth].push(parentId);
                });

                // Keep the selective cleanup approach we had before
                // (Regions are already cleaned up in the earlier cleanup code)

                // Process each depth level
                Object.keys(parentsByDepth).sort().forEach(depth => {
                    const parentsAtDepth = parentsByDepth[depth];

                    // Sort parents by priority: those with original positions first, then by x position
                    parentsAtDepth.sort((a, b) => {
                        const childDepthForSort = parseInt(depth) + 1;
                        const aHasOriginal = originalYPositions[`${a}_depth_${childDepthForSort}`] !== undefined;
                        const bHasOriginal = originalYPositions[`${b}_depth_${childDepthForSort}`] !== undefined;

                        // Parents with original positions come first
                        if (aHasOriginal && !bHasOriginal) return -1;
                        if (!aHasOriginal && bHasOriginal) return 1;

                        // If both have same priority level, sort by x position
                        const posA = cy.getElementById(a).position();
                        const posB = cy.getElementById(b).position();
                        return posA.x - posB.x;
                    });

                    console.log(`Processing parents at depth ${depth} in priority order:`, parentsAtDepth.map(id => {
                        const childDepth = parseInt(depth) + 1;
                        const hasOriginal = originalYPositions[`${id}_depth_${childDepth}`] !== undefined;
                        return `${id}(${hasOriginal ? 'has-original' : 'new'})`;
                    }));

                    // Position children for each parent at this depth
                    parentsAtDepth.forEach(parentId => {
                        const parent = cy.getElementById(parentId);
                        const parentPos = parent.position();
                        const children = nodesByParent[parentId];

                        // Sort children by their edge labels for consistent ordering
                        children.sort((a, b) => {
                            if (a.label < b.label) return -1;
                            if (a.label > b.label) return 1;
                            return 0;
                        });

                        // Calculate base positions for children
                        const childWidth = 150;
                        const childHeight = 50;
                        const horizontalGap = 50; // Much larger gap to prevent overlaps
                        const verticalGap = 120; // Increased vertical gap too
                        const childSpacing = childWidth + horizontalGap;
                        const totalWidth = children.length * childWidth + (children.length - 1) * horizontalGap;

                        // The depth where children will be placed
                        const childDepth = parseInt(depth) + 1;

                        // Initialize occupied regions array for this depth if needed
                        if (!occupiedByDepth[childDepth]) {
                            console.log(`Initializing empty regions array for depth ${childDepth}`);
                            occupiedByDepth[childDepth] = [];
                        } else {
                            console.log(`Depth ${childDepth} already has ${occupiedByDepth[childDepth].length} regions`);
                        }

                        // Try to center children under parent
                        let startX = parentPos.x - totalWidth / 2 + childWidth / 2;
                        let baseY = parentPos.y + verticalGap;

                        // Check if this parent has an original Y position recorded
                        const positionKey = `${parentId}_depth_${childDepth}`;
                        if (originalYPositions[positionKey]) {
                            console.log(`Using remembered Y position ${originalYPositions[positionKey]} for parent ${parentId} at depth ${childDepth}`);
                            baseY = originalYPositions[positionKey];
                        } else {
                            console.log(`Recording original Y position ${baseY} for parent ${parentId} at depth ${childDepth}`);
                            originalYPositions[positionKey] = baseY;
                        }

                        // Create bounding box for these children with extra padding
                        const padding = 20; // Extra padding around groups
                        const proposedRegion = {
                            left: startX - childWidth / 2 - horizontalGap / 2 - padding,
                            right: startX + (children.length - 1) * childSpacing + childWidth / 2 + horizontalGap / 2 + padding,
                            top: baseY - childHeight / 2 - padding,
                            bottom: baseY + childHeight / 2 + padding,
                            parentX: parentPos.x,
                            startX: startX
                        };

                        // First check overlaps, then position, then record
                        let finalY = baseY;

                        // Check if this parent has priority (has an original position)
                        const hasOriginalPosition = originalYPositions[positionKey] !== undefined;

                        // Check against all PREVIOUSLY positioned groups at this depth
                        console.log(`Checking ${occupiedByDepth[childDepth].length} existing regions at depth ${childDepth}`);
                        for (const region of occupiedByDepth[childDepth]) {
                            // Check for horizontal overlap
                            const horizontalOverlap = proposedRegion.left < region.right && proposedRegion.right > region.left;

                            if (horizontalOverlap) {
                                const regionOwnerHasOriginal = originalYPositions[`${region.parentId}_depth_${childDepth}`] !== undefined;

                                if (hasOriginalPosition && !regionOwnerHasOriginal) {
                                    // Current parent has priority over non-priority region owner
                                    console.log(`Priority parent ${parentId} keeping original position - displacing non-priority parent ${region.parentId}`);
                                } else if (!hasOriginalPosition && regionOwnerHasOriginal) {
                                    // Region owner has priority over current parent
                                    console.log(`Regular parent ${parentId} moving down - region owner ${region.parentId} has priority`);
                                    finalY = Math.max(finalY, region.bottom + 30);
                                } else if (hasOriginalPosition && regionOwnerHasOriginal) {
                                    // Both have priority - whoever got positioned first wins (avoid double priority conflict)
                                    console.log(`Priority conflict: parent ${parentId} moving down - region owner ${region.parentId} was positioned first`);
                                    finalY = Math.max(finalY, region.bottom + 30);
                                } else {
                                    // Both are regular - later parent moves down
                                    console.log(`Regular parent ${parentId} moving down to avoid existing region`);
                                    finalY = Math.max(finalY, region.bottom + 30);
                                }

                                console.log(`  Proposed: left=${proposedRegion.left}, right=${proposedRegion.right}`);
                                console.log(`  Existing: left=${region.left}, right=${region.right}`);
                            }
                        }

                        // Update Y position if needed
                        if (finalY !== baseY) {
                            console.log(`Adjusting Y from ${baseY} to ${finalY} for parent at x=${parentPos.x}`);
                            baseY = finalY;

                            // Update proposed region with new Y
                            proposedRegion.top = baseY - childHeight / 2 - padding;
                            proposedRegion.bottom = baseY + childHeight / 2 + padding;
                        }

                        // Position each child node at the calculated position
                        children.forEach((item, index) => {
                            const x = startX + index * childSpacing;
                            item.node.position({
                                x: x,
                                y: baseY
                            });
                        });

                        // NOW record this occupied region for future checks
                        const actualRegion = {
                            left: startX - childWidth / 2 - horizontalGap / 2 - padding,
                            right: startX + (children.length - 1) * childSpacing + childWidth / 2 + horizontalGap / 2 + padding,
                            top: baseY - childHeight / 2 - padding,
                            bottom: baseY + childHeight / 2 + padding,
                            parentX: parentPos.x,
                            parentId: parentId  // Track which parent owns this region
                        };

                        occupiedByDepth[childDepth].push(actualRegion);
                        console.log(`Added region at depth ${childDepth} for parent ${parentId}: left=${actualRegion.left}, right=${actualRegion.right}, top=${actualRegion.top}, bottom=${actualRegion.bottom}`);
                    });
                });

                // Run a quick layout on new nodes to refine positions
                const layout = cy.layout({
                    name: 'preset',  // Use preset positions
                    animate: true,
                    animationDuration: 500,
                    fit: false
                });
                layout.run();
            }

            // Save all positions for next update
            setTimeout(() => {
                cy.nodes().forEach(node => {
                    nodePositions[node.id()] = node.position();
                });
            }, 550);

            // Restore viewport
            setTimeout(() => {
                cy.pan(pan);
                cy.zoom(zoom);
            }, 550);
        }

        // Control button handlers
        document.getElementById('resetView').addEventListener('click', () => {
            if (cy) {
                cy.reset();
            } else {
                console.warn('Graph not initialized yet');
            }
        });

        document.getElementById('fitView').addEventListener('click', () => {
            if (cy) {
                cy.fit();
            } else {
                console.warn('Graph not initialized yet');
            }
        });

        document.getElementById('expandAll').addEventListener('click', async () => {
            if (!cy) {
                console.warn('Graph not initialized yet');
                return;
            }
            document.getElementById('loading').style.display = 'block';
            try {
                const response = await fetch('/api/expand_all', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        session_id: currentSessionId
                    })
                });

                if (response.ok) {
                    const data = await response.json();
                    updateGraph(data);
                }
            } catch (error) {
                console.error('Error expanding all:', error);
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        });

        document.getElementById('collapseAll').addEventListener('click', async () => {
            if (!cy) {
                console.warn('Graph not initialized yet');
                return;
            }
            document.getElementById('loading').style.display = 'block';
            try {
                const response = await fetch('/api/collapse_all', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        session_id: currentSessionId
                    })
                });

                if (response.ok) {
                    const data = await response.json();
                    updateGraph(data);
                }
            } catch (error) {
                console.error('Error collapsing all:', error);
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        });

        document.getElementById('exportImage').addEventListener('click', () => {
            if (cy) {
                const png = cy.png({ bg: 'white', scale: 2 });
                const link = document.createElement('a');
                link.href = png;
                link.download = 'confingy_config.png';
                link.click();
            } else {
                console.warn('Graph not initialized yet');
            }
        });

        // View mode toggle handler
        document.getElementById('viewMode').addEventListener('change', (event) => {
            const viewMode = event.target.value;
            const singleLegend = document.getElementById('singleLegend');
            const comparisonLegend = document.getElementById('comparisonLegend');
            const singleConfigSection = document.getElementById('singleConfigSection');
            const comparisonSection = document.getElementById('comparisonSection');

            if (viewMode === 'comparison') {
                singleLegend.style.display = 'none';
                comparisonLegend.style.display = 'block';
                singleConfigSection.style.display = 'none';
                comparisonSection.style.display = 'flex';
            } else {
                singleLegend.style.display = 'block';
                comparisonLegend.style.display = 'none';
                singleConfigSection.style.display = 'flex';
                comparisonSection.style.display = 'none';
            }
        });

        // File upload handlers
        document.getElementById('configFile').addEventListener('change', (event) => {
            const uploadButton = document.getElementById('uploadConfig');
            uploadButton.disabled = !event.target.files.length;
        });


        // Upload single config
        document.getElementById('uploadConfig').addEventListener('click', async () => {
            const fileInput = document.getElementById('configFile');
            if (!fileInput.files.length) return;

            const file = fileInput.files[0];
            const formData = new FormData();
            formData.append('config_file', file);

            document.getElementById('status').textContent = 'Uploading configuration...';
            document.getElementById('loading').style.display = 'block';

            try {
                const response = await fetch('/api/upload_config', {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    const result = await response.json();
                    currentSessionId = result.session_id;
                    updateGraph(result.graph_data);
                    document.getElementById('status').textContent = `Configuration loaded: ${file.name}`;
                    // Refresh dropdown when a new config is uploaded
                    loadAvailableConfigs();
                    loadAvailableConfigsForComparison();
                } else {
                    const error = await response.json();
                    document.getElementById('status').textContent = `Error: ${error.detail}`;
                }
            } catch (error) {
                document.getElementById('status').textContent = `Upload failed: ${error.message}`;
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        });


        // Load initial data if provided
        async function loadInitialData() {
            // Check if we have session data
            const urlParams = new URLSearchParams(window.location.search);
            const sessionId = urlParams.get('session') || 'default';
            currentSessionId = sessionId;

            try {
                const response = await fetch(`/api/get_graph/${sessionId}`);
                if (response.ok) {
                    const data = await response.json();
                    initCytoscape(data);
                    document.getElementById('status').textContent = 'Graph loaded';
                } else if (response.status === 404) {
                    // No existing session, initialize empty
                    console.log('No existing session found, waiting for data...');
                    document.getElementById('status').textContent = 'No data loaded yet';
                    // Initialize with empty graph
                    initCytoscape({ nodes: [], edges: [] });
                }
            } catch (error) {
                console.error('Error loading session:', error);
                document.getElementById('status').textContent = 'Error loading graph';
                // Initialize with empty graph
                initCytoscape({ nodes: [], edges: [] });
            }
        }

        // Load available configurations
        async function loadAvailableConfigs() {
            try {
                const response = await fetch('/api/list_stored_configs');
                if (response.ok) {
                    const data = await response.json();
                    const configs = data.configs;
                    const selector = document.getElementById('configSelector');

                    // Clear existing options except the first one
                    while (selector.children.length > 1) {
                        selector.removeChild(selector.lastChild);
                    }

                    // Add config options
                    configs.forEach(config => {
                        const option = document.createElement('option');
                        option.value = config.id;
                        option.textContent = config.title;
                        selector.appendChild(option);
                    });
                }
            } catch (error) {
                console.error('Error loading config list:', error);
            }
        }

        // Handle config selector change
        document.getElementById('configSelector').addEventListener('change', async (e) => {
            const configId = e.target.value;
            if (!configId) return;

            const configTitle = e.target.selectedOptions[0].text;
            document.getElementById('status').textContent = `Loading ${configTitle}...`;
            document.getElementById('loading').style.display = 'block';

            try {
                const response = await fetch(`/api/get_graph/${configId}`);
                if (response.ok) {
                    const graphData = await response.json();
                    currentSessionId = configId;
                    updateGraph(graphData);
                    document.getElementById('status').textContent = `Configuration loaded: ${configTitle}`;
                    // Refresh comparison dropdowns when a new config is loaded
                    loadAvailableConfigsForComparison();
                } else {
                    const error = await response.json();
                    document.getElementById('status').textContent = `Error: ${error.detail}`;
                }
            } catch (error) {
                document.getElementById('status').textContent = `Failed to load: ${error.message}`;
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        });

        // Load available configurations for comparison dropdowns
        async function loadAvailableConfigsForComparison() {
            try {
                const response = await fetch('/api/list_stored_configs');
                if (response.ok) {
                    const data = await response.json();
                    const configs = data.configs;

                    const selector1 = document.getElementById('compareConfig1');
                    const selector2 = document.getElementById('compareConfig2');

                    // Clear existing options except the first one
                    while (selector1.children.length > 1) {
                        selector1.removeChild(selector1.lastChild);
                    }
                    while (selector2.children.length > 1) {
                        selector2.removeChild(selector2.lastChild);
                    }

                    // Add config options
                    configs.forEach(config => {
                        const option1 = document.createElement('option');
                        option1.value = config.id;
                        option1.textContent = config.title;
                        selector1.appendChild(option1);

                        const option2 = document.createElement('option');
                        option2.value = config.id;
                        option2.textContent = config.title;
                        selector2.appendChild(option2);
                    });
                }
            } catch (error) {
                console.error('Error loading config list for comparison:', error);
            }
        }

        // Enable/disable compare selected button
        function updateCompareSelectedButton() {
            const config1 = document.getElementById('compareConfig1').value;
            const config2 = document.getElementById('compareConfig2').value;
            const button = document.getElementById('compareSelected');
            button.disabled = !config1 || !config2 || config1 === config2;
        }

        // Handle comparison selector changes
        document.getElementById('compareConfig1').addEventListener('change', updateCompareSelectedButton);
        document.getElementById('compareConfig2').addEventListener('change', updateCompareSelectedButton);

        // Handle compare selected configs
        document.getElementById('compareSelected').addEventListener('click', async () => {
            const config1Id = document.getElementById('compareConfig1').value;
            const config2Id = document.getElementById('compareConfig2').value;

            if (!config1Id || !config2Id) return;

            document.getElementById('status').textContent = 'Comparing configurations...';
            document.getElementById('loading').style.display = 'block';

            try {
                const response = await fetch('/api/compare_stored_configs', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        config1_id: config1Id,
                        config2_id: config2Id
                    })
                });

                if (response.ok) {
                    const result = await response.json();
                    currentSessionId = result.session_id;
                    updateGraph(result.graph_data);
                    const config1Title = document.getElementById('compareConfig1').selectedOptions[0].text;
                    const config2Title = document.getElementById('compareConfig2').selectedOptions[0].text;
                    document.getElementById('status').textContent = `Comparison loaded: ${config1Title} vs ${config2Title}`;
                    // Auto-switch to comparison view
                    document.getElementById('viewMode').value = 'comparison';
                    document.getElementById('viewMode').dispatchEvent(new Event('change'));
                } else {
                    const error = await response.json();
                    document.getElementById('status').textContent = `Error: ${error.detail}`;
                }
            } catch (error) {
                document.getElementById('status').textContent = `Comparison failed: ${error.message}`;
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        });

        // Initialize on page load
        window.addEventListener('DOMContentLoaded', () => {
            loadAvailableConfigs();
            loadAvailableConfigsForComparison();
            loadInitialData();
        });
    </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)


@app.post("/api/visualize")
async def visualize(request: VisualizationRequest):
    """Create a new visualization from a configuration."""
    dag = ConfigGraph()
    dag.build_from_config(request.config)

    # Store the DAG and config for this session
    session_id = request.session_id or "default"
    stored_dags[session_id] = dag
    stored_configs[session_id] = request.config
    expanded_nodes[session_id] = set()

    # Return initial graph data (depth 0 and 1)
    graph_data = dag.to_cytoscape_json(max_depth=1)
    return JSONResponse(content=graph_data)


@app.post("/api/toggle_node")
async def toggle_node(request: ExpansionRequest):
    """Expand or collapse a node."""
    dag = get_dag(request.session_id)
    session_expanded = expanded_nodes[request.session_id]

    if request.expand:
        session_expanded.add(request.node_id)
    else:
        # When collapsing, remove this node and all its descendants from expanded set
        session_expanded.discard(request.node_id)

        # Also remove any descendant nodes that were expanded
        # (This prevents orphaned expanded nodes when parent is collapsed)
        def get_all_descendants(node_id):
            descendants = set()
            for edge in dag.edges:
                if edge.from_node == node_id:
                    descendants.add(edge.to_node)
                    descendants.update(get_all_descendants(edge.to_node))
            return descendants

        descendants = get_all_descendants(request.node_id)
        session_expanded.difference_update(descendants)

    # Return updated graph
    graph_data = dag.to_cytoscape_json(max_depth=1, expanded_nodes=session_expanded)
    return JSONResponse(content=graph_data)


@app.post("/api/expand_all")
async def expand_all(request: dict):
    """Expand all nodes in the graph."""
    session_id = request.get("session_id", "default")
    dag = get_dag(session_id)
    # Add all nodes to expanded set
    expanded_nodes[session_id] = set(dag.nodes.keys())

    # Return fully expanded graph
    graph_data = dag.to_cytoscape_json(
        max_depth=999, expanded_nodes=expanded_nodes[session_id]
    )
    return JSONResponse(content=graph_data)


@app.post("/api/collapse_all")
async def collapse_all(request: dict):
    """Collapse all nodes to show only first two levels."""
    session_id = request.get("session_id", "default")
    dag = get_dag(session_id)
    # Clear expanded nodes
    expanded_nodes[session_id] = set()

    # Return collapsed graph
    graph_data = dag.to_cytoscape_json(max_depth=1, expanded_nodes=set())
    return JSONResponse(content=graph_data)


@app.get("/api/get_graph/{session_id}")
async def get_graph(session_id: str):
    """Get the current graph for a session."""
    dag = get_dag(session_id)
    session_expanded = expanded_nodes.get(session_id, set())

    graph_data = dag.to_cytoscape_json(max_depth=1, expanded_nodes=session_expanded)
    return JSONResponse(content=graph_data)


@app.post("/api/upload_config")
async def upload_config(config_file: UploadFile = File(...)):
    """Upload a single configuration file."""
    try:
        # Read the uploaded file
        content = await config_file.read()

        # Parse JSON
        import json

        config_data = json.loads(content.decode("utf-8"))

        # Build DAG from config
        dag = ConfigGraph()
        dag.build_from_config(config_data)

        # Generate session ID
        session_id = f"upload_{hash(str(config_data)) % 10000:04d}"

        # Store the DAG
        stored_dags[session_id] = dag
        stored_configs[session_id] = {
            "config": config_data,
            "comparison": False,
            "title": f"Uploaded: {config_file.filename}",
        }
        expanded_nodes[session_id] = set()

        # Return visualization data
        graph_data = dag.to_cytoscape_json(max_depth=1, expanded_nodes=set())

        return JSONResponse(
            content={
                "status": "success",
                "session_id": session_id,
                "graph_data": graph_data,
                "title": f"Uploaded: {config_file.filename}",
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to process config file: {str(e)}"
        )


@app.post("/api/upload_compare")
async def upload_compare(
    config_file1: UploadFile = File(...), config_file2: UploadFile = File(...)
):
    """Upload two configuration files for comparison."""
    try:
        # Read both files
        content1 = await config_file1.read()
        content2 = await config_file2.read()

        # Parse files
        def parse_config(content: bytes, filename: str):
            import json

            return json.loads(content.decode("utf-8"))

        config_data1 = parse_config(content1, config_file1.filename or "config1.json")
        config_data2 = parse_config(content2, config_file2.filename or "config2.json")

        # Build DAGs from both configurations
        dag1 = ConfigGraph()
        dag1.build_from_config(config_data1)

        dag2 = ConfigGraph()
        dag2.build_from_config(config_data2)

        # Create comparison DAG
        comparison_dag = ConfigGraph.create_comparison_dag(dag1, dag2)

        # Generate session ID
        session_id = (
            f"compare_{hash(str(config_data1) + str(config_data2)) % 10000:04d}"
        )

        # Store the comparison in active sessions (not permanent storage)
        session_dags[session_id] = comparison_dag
        active_sessions[session_id] = {
            "config1": config_data1,
            "config2": config_data2,
            "comparison": True,
            "title": f"Compare: {config_file1.filename} vs {config_file2.filename}",
        }
        expanded_nodes[session_id] = set()

        # Return comparison visualization data
        graph_data = comparison_dag.to_cytoscape_json(max_depth=1, expanded_nodes=set())

        return JSONResponse(
            content={
                "status": "success",
                "session_id": session_id,
                "graph_data": graph_data,
                "title": f"Compare: {config_file1.filename} vs {config_file2.filename}",
                "comparison_mode": True,
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to compare config files: {str(e)}"
        )


@app.post("/api/compare")
async def compare_configurations(request: ComparisonRequest):
    """Compare two configurations."""
    try:
        # Build DAGs from both configurations
        dag1 = ConfigGraph()
        dag1.build_from_config(request.config1)

        dag2 = ConfigGraph()
        dag2.build_from_config(request.config2)

        # Create comparison DAG
        comparison_dag = ConfigGraph.create_comparison_dag(dag1, dag2)

        # Store the comparison in active sessions (not permanent storage)
        session_id = request.session_id or "comparison"
        session_dags[session_id] = comparison_dag
        active_sessions[session_id] = {
            "config1": request.config1,
            "config2": request.config2,
            "comparison": True,
            "title": request.title,
        }
        expanded_nodes[session_id] = set()

        # Return the initial visualization data
        graph_data = comparison_dag.to_cytoscape_json(max_depth=1, expanded_nodes=set())

        return JSONResponse(
            content={
                "status": "success",
                "session_id": session_id,
                "graph_data": graph_data,
                "title": request.title or "Configuration Comparison",
                "comparison_mode": True,
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to compare configurations: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
