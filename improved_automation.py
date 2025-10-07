"""
Improved Automation System for GIS Workflow Execution

This module provides an enhanced version of the original automation system that:
1. Parses LLM-generated workflows into executable steps
2. Dynamically executes GIS functions based on the workflow
3. Handles errors gracefully and provides meaningful feedback
4. Integrates with the existing automation1.py functions
"""

import os
import re
import json
import ast
import traceback
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import importlib.util
import matplotlib.pyplot as plt

# Import existing GIS functions
from automation1 import (
    get_city_visualizations, get_and_save_lulc, plot_lulc, fetch_and_plot_hydrology,
    get_city_bbox, download_worldpop_raster, clip_and_plot_population, get_building_footprints,
    plot_buildings, calculate_flow_accumulation, fill_depressions, calculate_flow_accumulation_filled,
    fix_and_generate_wetness_index, fix_sca_raster, sanitize_dem, try_run_wetness_index_fixed,
    compute_twi, plot_combined_map, calculate_spi_manual, compute_flood_risk_map
)

@dataclass
class WorkflowStep:
    """Represents a single step in a GIS workflow"""
    step_number: int
    description: str
    function_name: str
    parameters: Dict[str, Any]
    expected_output: str
    dependencies: List[int] = None

@dataclass
class ExecutionResult:
    """Represents the result of executing a workflow step"""
    step_number: int
    success: bool
    output: Any = None
    error_message: str = None
    execution_time: float = 0.0

class WorkflowParser:
    """Parses LLM-generated workflow descriptions into executable steps"""
    
    def __init__(self):
        # Map common GIS operations to available functions
        self.function_mapping = {
            'city_visualization': get_city_visualizations,
            'city_visualizations': get_city_visualizations,
            'dem_analysis': get_city_visualizations,
            'terrain_analysis': get_city_visualizations,
            'lulc_analysis': get_and_save_lulc,
            'land_use': get_and_save_lulc,
            'land_cover': get_and_save_lulc,
            'hydrology': fetch_and_plot_hydrology,
            'hydrological_features': fetch_and_plot_hydrology,
            'water_features': fetch_and_plot_hydrology,
            'population_analysis': clip_and_plot_population,
            'population_density': clip_and_plot_population,
            'building_footprints': get_building_footprints,
            'buildings': get_building_footprints,
            'flood_risk': compute_flood_risk_map,
            'wetness_index': compute_twi,
            'twi': compute_twi,
            'flow_accumulation': calculate_flow_accumulation,
        }
        
        # Common parameter patterns
        self.parameter_patterns = {
            'city_name': r'(?:city|location|area|place)[\s:]*([A-Za-z\s,]+)',
            'output_dir': r'(?:output|save|directory)[\s:]*([^\s]+)',
            'resolution': r'(?:resolution)[\s:]*(\d+)',
        }
    
    def parse_workflow(self, workflow_text: str, query: str = "") -> List[WorkflowStep]:
        """Parse workflow text into executable steps"""
        steps = []
        
        # Extract city name from query if available
        city_name = self._extract_city_name(query)
        
        # Split workflow into steps
        step_sections = self._split_into_steps(workflow_text)
        
        for i, section in enumerate(step_sections, 1):
            step = self._parse_step(section, i, city_name)
            if step:
                steps.append(step)
        
        return steps
    
    def _extract_city_name(self, query: str) -> Optional[str]:
        """Extract city name from user query"""
        patterns = [
            r'(?:in|for|of|at)\s+([A-Za-z\s]+?)(?:\s|$|,|\.|;)',
            r'([A-Za-z\s]+?)\s+(?:city|area|region)',
            r'analyze\s+([A-Za-z\s]+?)(?:\s|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                city = match.group(1).strip()
                if len(city) > 2 and city.lower() not in ['the', 'and', 'or', 'in', 'at', 'of', 'for']:
                    return city
        return None
    
    def _split_into_steps(self, workflow_text: str) -> List[str]:
        """Split workflow text into individual steps"""
        # Look for numbered steps or bullet points
        step_patterns = [
            r'(?:^|\n)\s*(\d+\.?\s+.+?)(?=\n\s*\d+\.|\n\s*$|$)',
            r'(?:^|\n)\s*([•\-\*]\s+.+?)(?=\n\s*[•\-\*]|\n\s*$|$)',
            r'(?:^|\n)\s*(Step\s+\d+[:\.]?\s+.+?)(?=\n\s*Step\s+\d+|$)',
        ]
        
        for pattern in step_patterns:
            matches = re.findall(pattern, workflow_text, re.MULTILINE | re.DOTALL)
            if matches:
                return [match.strip() for match in matches]
        
        # If no clear steps found, split by sentences
        sentences = re.split(r'[.!?]+', workflow_text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def _parse_step(self, step_text: str, step_number: int, city_name: str = None) -> Optional[WorkflowStep]:
        """Parse a single step into a WorkflowStep object"""
        # Identify the function to call based on keywords
        function_name = self._identify_function(step_text)
        if not function_name:
            return None
        
        # Extract parameters
        parameters = self._extract_parameters(step_text, city_name)
        
        # Create the step
        return WorkflowStep(
            step_number=step_number,
            description=step_text,
            function_name=function_name,
            parameters=parameters,
            expected_output=self._infer_expected_output(function_name)
        )
    
    def _identify_function(self, step_text: str) -> Optional[str]:
        """Identify which function to call based on step description"""
        step_lower = step_text.lower()
        
        # Score each function based on keyword matches
        scores = {}
        for key, func in self.function_mapping.items():
            score = 0
            keywords = key.split('_')
            for keyword in keywords:
                if keyword in step_lower:
                    score += 1
            
            # Additional keyword matching
            if 'dem' in step_lower or 'elevation' in step_lower or 'terrain' in step_lower:
                if 'visualization' in key or 'city' in key:
                    score += 2
            
            if 'land use' in step_lower or 'lulc' in step_lower:
                if 'lulc' in key or 'land' in key:
                    score += 2
            
            if 'water' in step_lower or 'hydro' in step_lower or 'river' in step_lower:
                if 'hydro' in key or 'water' in key:
                    score += 2
            
            if 'population' in step_lower or 'density' in step_lower:
                if 'population' in key:
                    score += 2
            
            if 'building' in step_lower or 'footprint' in step_lower:
                if 'building' in key:
                    score += 2
            
            if 'flood' in step_lower or 'risk' in step_lower:
                if 'flood' in key:
                    score += 2
            
            scores[key] = score
        
        # Return the function with the highest score
        if scores:
            best_match = max(scores, key=scores.get)
            if scores[best_match] > 0:
                return best_match
        
        return None
    
    def _extract_parameters(self, step_text: str, city_name: str = None) -> Dict[str, Any]:
        """Extract parameters from step description"""
        parameters = {}
        
        # Add city name if available
        if city_name:
            parameters['city_name'] = city_name
        
        # Extract other parameters using patterns
        for param_name, pattern in self.parameter_patterns.items():
            match = re.search(pattern, step_text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if param_name == 'resolution':
                    parameters[param_name] = int(value)
                else:
                    parameters[param_name] = value
        
        # Set default output directory
        if 'output_dir' not in parameters:
            parameters['output_dir'] = 'output'
        
        return parameters
    
    def _infer_expected_output(self, function_name: str) -> str:
        """Infer the expected output type for a function"""
        if 'visualization' in function_name or 'plot' in function_name:
            return 'matplotlib_figures'
        elif 'lulc' in function_name or 'hydro' in function_name:
            return 'geodataframe'
        elif 'population' in function_name:
            return 'raster_plot'
        elif 'building' in function_name:
            return 'geodataframe'
        else:
            return 'mixed'

class WorkflowExecutor:
    """Executes parsed workflow steps"""
    
    def __init__(self):
        self.parser = WorkflowParser()
        self.execution_results = []
        self.generated_outputs = {}
    
    def execute_workflow(self, workflow_text: str, query: str = "") -> List[ExecutionResult]:
        """Execute a complete workflow"""
        # Parse the workflow
        steps = self.parser.parse_workflow(workflow_text, query)
        
        if not steps:
            return [ExecutionResult(
                step_number=0,
                success=False,
                error_message="No executable steps found in workflow"
            )]
        
        # Execute each step
        results = []
        for step in steps:
            result = self._execute_step(step)
            results.append(result)
            
            # Store successful outputs for potential use in later steps
            if result.success and result.output is not None:
                self.generated_outputs[f"step_{step.step_number}"] = result.output
        
        self.execution_results = results
        return results
    
    def _execute_step(self, step: WorkflowStep) -> ExecutionResult:
        """Execute a single workflow step"""
        import time
        start_time = time.time()
        
        try:
            # Get the function to call
            function = self.parser.function_mapping.get(step.function_name)
            if not function:
                return ExecutionResult(
                    step_number=step.step_number,
                    success=False,
                    error_message=f"Function '{step.function_name}' not found"
                )
            
            # Prepare parameters
            params = self._prepare_parameters(step.parameters, function)
            
            # Execute the function
            output = function(**params)
            
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                step_number=step.step_number,
                success=True,
                output=output,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ExecutionResult(
                step_number=step.step_number,
                success=False,
                error_message=f"Execution error: {str(e)}",
                execution_time=execution_time
            )
    
    def _prepare_parameters(self, parameters: Dict[str, Any], function: Callable) -> Dict[str, Any]:
        """Prepare parameters for function call"""
        import inspect
        
        # Get function signature
        sig = inspect.signature(function)
        valid_params = {}
        
        # Only include parameters that the function accepts
        for param_name, param_value in parameters.items():
            if param_name in sig.parameters:
                valid_params[param_name] = param_value
        
        return valid_params
    
    def get_generated_figures(self) -> List[plt.Figure]:
        """Get all matplotlib figures generated during execution"""
        figures = []
        for result in self.execution_results:
            if result.success and result.output:
                if isinstance(result.output, list):
                    for item in result.output:
                        if isinstance(item, plt.Figure):
                            figures.append(item)
                elif isinstance(result.output, plt.Figure):
                    figures.append(result.output)
        return figures
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of the execution results"""
        total_steps = len(self.execution_results)
        successful_steps = sum(1 for r in self.execution_results if r.success)
        failed_steps = total_steps - successful_steps
        total_time = sum(r.execution_time for r in self.execution_results)
        
        return {
            'total_steps': total_steps,
            'successful_steps': successful_steps,
            'failed_steps': failed_steps,
            'success_rate': successful_steps / total_steps if total_steps > 0 else 0,
            'total_execution_time': total_time,
            'results': self.execution_results
        }

# Enhanced LLM prompt for better workflow generation
ENHANCED_WORKFLOW_PROMPT = """
You are a GIS workflow planner. Given a user query and tool documentation, generate a detailed, step-by-step geospatial workflow.

For each step, be specific about:
1. The exact operation to perform
2. The data or inputs needed
3. The expected output

Available GIS operations:
- City visualization and DEM analysis
- Land use/land cover (LULC) analysis
- Hydrological features analysis
- Population density analysis
- Building footprints analysis
- Flood risk mapping
- Terrain analysis (slope, aspect, wetness index)

Format your response as numbered steps:
1. [Operation]: [Detailed description]
2. [Operation]: [Detailed description]
...

Make sure each step is actionable and specific to the user's query.
"""

def create_enhanced_query_function(groq_client, model_name="llama3-70b-8192"):
    """Create an enhanced query function that generates better workflows"""
    
    def enhanced_query_llama(query: str, context: str) -> str:
        full_prompt = f"{ENHANCED_WORKFLOW_PROMPT}\n\nContext from GIS docs:\n{context}\n\nUser Query:\n{query}"
        
        response = groq_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": ENHANCED_WORKFLOW_PROMPT},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.3,
            max_tokens=2048
        )
        return response.choices[0].message.content
    
    return enhanced_query_llama

# Example usage function
def run_enhanced_automation(query: str, workflow_text: str) -> Dict[str, Any]:
    """Run the enhanced automation system"""
    executor = WorkflowExecutor()
    results = executor.execute_workflow(workflow_text, query)
    summary = executor.get_execution_summary()
    figures = executor.get_generated_figures()
    
    return {
        'execution_results': results,
        'summary': summary,
        'generated_figures': figures,
        'executor': executor
    }

