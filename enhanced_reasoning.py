"""
Enhanced Reasoning System for GIS Workflow Generation

This module provides advanced reasoning capabilities for generating more structured
and executable GIS workflows. It includes:

1. Structured workflow templates
2. Context-aware reasoning
3. Parameter extraction and validation
4. Code generation capabilities
5. Multi-step reasoning with dependencies
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import ast

class GISOperationType(Enum):
    """Types of GIS operations"""
    DATA_ACQUISITION = "data_acquisition"
    PREPROCESSING = "preprocessing"
    ANALYSIS = "analysis"
    VISUALIZATION = "visualization"
    EXPORT = "export"

class DataType(Enum):
    """Types of geospatial data"""
    VECTOR = "vector"
    RASTER = "raster"
    POINT_CLOUD = "point_cloud"
    TABULAR = "tabular"

@dataclass
class GISOperation:
    """Structured representation of a GIS operation"""
    operation_id: str
    operation_type: GISOperationType
    function_name: str
    description: str
    input_data: List[str]
    output_data: List[str]
    parameters: Dict[str, Any]
    dependencies: List[str]
    estimated_time: float
    data_types: List[DataType]

@dataclass
class WorkflowTemplate:
    """Template for common GIS workflows"""
    template_id: str
    name: str
    description: str
    operations: List[GISOperation]
    total_estimated_time: float
    required_data: List[str]
    output_products: List[str]

class EnhancedReasoningEngine:
    """Advanced reasoning engine for GIS workflow generation"""
    
    def __init__(self):
        self.workflow_templates = self._initialize_templates()
        self.operation_library = self._initialize_operations()
        self.context_patterns = self._initialize_context_patterns()
    
    def _initialize_templates(self) -> Dict[str, WorkflowTemplate]:
        """Initialize common GIS workflow templates"""
        templates = {}
        
        # Urban Analysis Template
        urban_ops = [
            GISOperation(
                operation_id="urban_001",
                operation_type=GISOperationType.DATA_ACQUISITION,
                function_name="get_city_visualizations",
                description="Acquire city boundary and generate basemap with DEM",
                input_data=["city_name"],
                output_data=["city_boundary", "dem_raster", "basemap"],
                parameters={"resolution": 100},
                dependencies=[],
                estimated_time=30.0,
                data_types=[DataType.VECTOR, DataType.RASTER]
            ),
            GISOperation(
                operation_id="urban_002",
                operation_type=GISOperationType.DATA_ACQUISITION,
                function_name="get_and_save_lulc",
                description="Acquire land use/land cover data from OpenStreetMap",
                input_data=["city_name"],
                output_data=["lulc_vector"],
                parameters={"output_dir": "output"},
                dependencies=["urban_001"],
                estimated_time=20.0,
                data_types=[DataType.VECTOR]
            ),
            GISOperation(
                operation_id="urban_003",
                operation_type=GISOperationType.DATA_ACQUISITION,
                function_name="fetch_and_plot_hydrology",
                description="Acquire hydrological features and create visualization",
                input_data=["city_name"],
                output_data=["hydro_vector", "hydro_map"],
                parameters={"output_folder": "output"},
                dependencies=["urban_001"],
                estimated_time=25.0,
                data_types=[DataType.VECTOR]
            ),
            GISOperation(
                operation_id="urban_004",
                operation_type=GISOperationType.ANALYSIS,
                function_name="clip_and_plot_population",
                description="Analyze population density within city boundary",
                input_data=["city_boundary", "population_raster"],
                output_data=["clipped_population", "population_map"],
                parameters={"save_clipped": True},
                dependencies=["urban_001"],
                estimated_time=15.0,
                data_types=[DataType.RASTER]
            )
        ]
        
        templates["urban_analysis"] = WorkflowTemplate(
            template_id="urban_analysis",
            name="Comprehensive Urban Analysis",
            description="Complete urban analysis including DEM, LULC, hydrology, and population",
            operations=urban_ops,
            total_estimated_time=90.0,
            required_data=["city_name"],
            output_products=["basemap", "dem_raster", "lulc_vector", "hydro_vector", "population_analysis"]
        )
        
        # Flood Risk Template
        flood_ops = [
            GISOperation(
                operation_id="flood_001",
                operation_type=GISOperationType.DATA_ACQUISITION,
                function_name="get_city_visualizations",
                description="Acquire DEM and terrain data for flood modeling",
                input_data=["city_name"],
                output_data=["dem_raster", "slope_raster", "aspect_raster"],
                parameters={"resolution": 50},
                dependencies=[],
                estimated_time=35.0,
                data_types=[DataType.RASTER]
            ),
            GISOperation(
                operation_id="flood_002",
                operation_type=GISOperationType.ANALYSIS,
                function_name="calculate_flow_accumulation",
                description="Calculate flow accumulation from DEM",
                input_data=["dem_raster"],
                output_data=["flow_accumulation"],
                parameters={},
                dependencies=["flood_001"],
                estimated_time=20.0,
                data_types=[DataType.RASTER]
            ),
            GISOperation(
                operation_id="flood_003",
                operation_type=GISOperationType.ANALYSIS,
                function_name="compute_flood_risk_map",
                description="Generate flood risk assessment",
                input_data=["dem_raster", "flow_accumulation", "lulc_vector"],
                output_data=["flood_risk_map"],
                parameters={},
                dependencies=["flood_001", "flood_002"],
                estimated_time=30.0,
                data_types=[DataType.RASTER]
            )
        ]
        
        templates["flood_risk"] = WorkflowTemplate(
            template_id="flood_risk",
            name="Flood Risk Assessment",
            description="Comprehensive flood risk analysis using DEM and hydrological modeling",
            operations=flood_ops,
            total_estimated_time=85.0,
            required_data=["city_name"],
            output_products=["flood_risk_map", "flow_accumulation", "terrain_analysis"]
        )
        
        return templates
    
    def _initialize_operations(self) -> Dict[str, GISOperation]:
        """Initialize library of available GIS operations"""
        operations = {}
        
        # Add all operations from templates to the library
        for template in self.workflow_templates.values():
            for op in template.operations:
                operations[op.operation_id] = op
        
        return operations
    
    def _initialize_context_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for context recognition"""
        return {
            "urban_analysis": [
                "urban", "city", "metropolitan", "municipal", "downtown", "suburb",
                "land use", "lulc", "building", "infrastructure", "population"
            ],
            "flood_risk": [
                "flood", "flooding", "inundation", "water", "risk", "hazard",
                "drainage", "watershed", "flow", "accumulation", "dem", "elevation"
            ],
            "environmental": [
                "environment", "ecology", "habitat", "biodiversity", "conservation",
                "vegetation", "ndvi", "forest", "wetland", "protected"
            ],
            "terrain_analysis": [
                "terrain", "topography", "slope", "aspect", "elevation", "dem",
                "hillshade", "contour", "relief", "ruggedness", "tpi"
            ],
            "hydrological": [
                "water", "river", "stream", "lake", "watershed", "basin",
                "hydrology", "drainage", "flow", "wetness", "twi"
            ]
        }
    
    def analyze_query_context(self, query: str) -> Dict[str, float]:
        """Analyze query to determine context and intent"""
        query_lower = query.lower()
        context_scores = {}
        
        for context, keywords in self.context_patterns.items():
            score = 0
            for keyword in keywords:
                if keyword in query_lower:
                    score += 1
            
            # Normalize score
            context_scores[context] = score / len(keywords) if keywords else 0
        
        return context_scores
    
    def select_workflow_template(self, query: str) -> Optional[WorkflowTemplate]:
        """Select the most appropriate workflow template based on query"""
        context_scores = self.analyze_query_context(query)
        
        # Simple template selection logic
        if context_scores.get("flood_risk", 0) > 0.2:
            return self.workflow_templates.get("flood_risk")
        elif context_scores.get("urban_analysis", 0) > 0.1:
            return self.workflow_templates.get("urban_analysis")
        else:
            # Default to urban analysis
            return self.workflow_templates.get("urban_analysis")
    
    def extract_parameters(self, query: str) -> Dict[str, Any]:
        """Extract parameters from user query"""
        parameters = {}
        
        # Extract city name
        city_patterns = [
            r'(?:in|for|of|at)\s+([A-Za-z\s]+?)(?:\s|$|,|\.|;)',
            r'([A-Za-z\s]+?)\s+(?:city|area|region)',
            r'analyze\s+([A-Za-z\s]+?)(?:\s|$)',
        ]
        
        for pattern in city_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                city = match.group(1).strip()
                if len(city) > 2 and city.lower() not in ['the', 'and', 'or', 'in', 'at', 'of', 'for']:
                    parameters['city_name'] = city
                    break
        
        # Extract resolution
        resolution_match = re.search(r'resolution[:\s]*(\d+)', query, re.IGNORECASE)
        if resolution_match:
            parameters['resolution'] = int(resolution_match.group(1))
        
        # Extract output directory
        output_match = re.search(r'(?:output|save)[:\s]*([^\s]+)', query, re.IGNORECASE)
        if output_match:
            parameters['output_dir'] = output_match.group(1)
        else:
            parameters['output_dir'] = 'output'
        
        return parameters
    
    def customize_workflow(self, template: WorkflowTemplate, parameters: Dict[str, Any]) -> WorkflowTemplate:
        """Customize workflow template with extracted parameters"""
        customized_ops = []
        
        for op in template.operations:
            # Create a copy of the operation
            custom_op = GISOperation(
                operation_id=op.operation_id,
                operation_type=op.operation_type,
                function_name=op.function_name,
                description=op.description,
                input_data=op.input_data.copy(),
                output_data=op.output_data.copy(),
                parameters=op.parameters.copy(),
                dependencies=op.dependencies.copy(),
                estimated_time=op.estimated_time,
                data_types=op.data_types.copy()
            )
            
            # Update parameters
            for param_key, param_value in parameters.items():
                if param_key in custom_op.parameters or param_key in custom_op.input_data:
                    custom_op.parameters[param_key] = param_value
            
            customized_ops.append(custom_op)
        
        # Create customized template
        return WorkflowTemplate(
            template_id=f"{template.template_id}_custom",
            name=f"Customized {template.name}",
            description=f"{template.description} (Customized)",
            operations=customized_ops,
            total_estimated_time=template.total_estimated_time,
            required_data=template.required_data.copy(),
            output_products=template.output_products.copy()
        )
    
    def generate_structured_workflow(self, query: str) -> Tuple[WorkflowTemplate, Dict[str, Any]]:
        """Generate a structured workflow from user query"""
        # Select appropriate template
        template = self.select_workflow_template(query)
        if not template:
            raise ValueError("No suitable workflow template found")
        
        # Extract parameters
        parameters = self.extract_parameters(query)
        
        # Customize workflow
        customized_workflow = self.customize_workflow(template, parameters)
        
        return customized_workflow, parameters
    
    def generate_execution_plan(self, workflow: WorkflowTemplate) -> Dict[str, Any]:
        """Generate detailed execution plan from workflow"""
        execution_plan = {
            "workflow_id": workflow.template_id,
            "workflow_name": workflow.name,
            "description": workflow.description,
            "total_estimated_time": workflow.total_estimated_time,
            "steps": [],
            "dependencies": {},
            "outputs": workflow.output_products
        }
        
        for i, op in enumerate(workflow.operations, 1):
            step = {
                "step_number": i,
                "operation_id": op.operation_id,
                "function_name": op.function_name,
                "description": op.description,
                "operation_type": op.operation_type.value,
                "input_data": op.input_data,
                "output_data": op.output_data,
                "parameters": op.parameters,
                "dependencies": op.dependencies,
                "estimated_time": op.estimated_time,
                "data_types": [dt.value for dt in op.data_types]
            }
            execution_plan["steps"].append(step)
            execution_plan["dependencies"][op.operation_id] = op.dependencies
        
        return execution_plan
    
    def generate_code_snippet(self, operation: GISOperation) -> str:
        """Generate Python code snippet for an operation"""
        code_lines = []
        
        # Add imports (simplified)
        code_lines.append("# Generated code snippet")
        code_lines.append(f"# Operation: {operation.description}")
        code_lines.append("")
        
        # Generate function call
        params = []
        for key, value in operation.parameters.items():
            if isinstance(value, str):
                params.append(f'{key}="{value}"')
            else:
                params.append(f'{key}={value}')
        
        param_str = ", ".join(params)
        
        if operation.output_data:
            output_vars = ", ".join(operation.output_data)
            code_lines.append(f"{output_vars} = {operation.function_name}({param_str})")
        else:
            code_lines.append(f"{operation.function_name}({param_str})")
        
        code_lines.append("")
        
        return "\n".join(code_lines)
    
    def generate_complete_workflow_description(self, query: str) -> str:
        """Generate a complete, structured workflow description"""
        try:
            workflow, parameters = self.generate_structured_workflow(query)
            execution_plan = self.generate_execution_plan(workflow)
            
            description_lines = []
            description_lines.append(f"# {workflow.name}")
            description_lines.append(f"{workflow.description}")
            description_lines.append("")
            description_lines.append(f"**Estimated Total Time:** {workflow.total_estimated_time:.1f} seconds")
            description_lines.append(f"**Required Data:** {', '.join(workflow.required_data)}")
            description_lines.append(f"**Output Products:** {', '.join(workflow.output_products)}")
            description_lines.append("")
            
            description_lines.append("## Workflow Steps:")
            description_lines.append("")
            
            for step in execution_plan["steps"]:
                description_lines.append(f"### Step {step['step_number']}: {step['description']}")
                description_lines.append(f"- **Function:** `{step['function_name']}`")
                description_lines.append(f"- **Type:** {step['operation_type'].replace('_', ' ').title()}")
                description_lines.append(f"- **Input Data:** {', '.join(step['input_data'])}")
                description_lines.append(f"- **Output Data:** {', '.join(step['output_data'])}")
                description_lines.append(f"- **Parameters:** {step['parameters']}")
                description_lines.append(f"- **Estimated Time:** {step['estimated_time']:.1f} seconds")
                
                if step['dependencies']:
                    description_lines.append(f"- **Dependencies:** {', '.join(step['dependencies'])}")
                
                description_lines.append("")
            
            description_lines.append("## Execution Parameters:")
            description_lines.append("")
            for key, value in parameters.items():
                description_lines.append(f"- **{key}:** {value}")
            
            return "\n".join(description_lines)
            
        except Exception as e:
            return f"Error generating structured workflow: {str(e)}"

def create_enhanced_reasoning_query_function(groq_client, model_name="llama3-70b-8192"):
    """Create an enhanced query function that uses structured reasoning"""
    
    reasoning_engine = EnhancedReasoningEngine()
    
    def enhanced_reasoning_query(query: str, context: str) -> str:
        # First, generate structured workflow
        structured_workflow = reasoning_engine.generate_complete_workflow_description(query)
        
        # Then, enhance with LLM reasoning
        enhanced_prompt = f"""
You are an expert GIS analyst. Based on the structured workflow below and the provided context, 
enhance and refine the workflow with additional insights, alternative approaches, and best practices.

Structured Workflow:
{structured_workflow}

Context from GIS Documentation:
{context}

User Query:
{query}

Please provide an enhanced workflow that:
1. Validates the structured approach
2. Suggests improvements or alternatives
3. Adds technical details and best practices
4. Considers potential challenges and solutions
5. Maintains the step-by-step structure

Enhanced Workflow:
"""
        
        response = groq_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an expert GIS analyst providing enhanced workflow recommendations."},
                {"role": "user", "content": enhanced_prompt}
            ],
            temperature=0.3,
            max_tokens=3072
        )
        
        return response.choices[0].message.content
    
    return enhanced_reasoning_query, reasoning_engine

