"""
Integrated GIS Automation System

This module brings together all the enhanced components:
1. Enhanced reasoning engine
2. Improved workflow execution
3. Fixed automation functions
4. Streamlit integration

This is the complete solution that addresses all the identified issues.
"""

import os
import sys
import traceback
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import streamlit as st
from groq import Groq

# Import all enhanced components
from enhanced_reasoning import (
    EnhancedReasoningEngine, 
    create_enhanced_reasoning_query_function,
    WorkflowTemplate,
    GISOperation
)
from improved_automation import (
    WorkflowExecutor,
    WorkflowParser,
    ExecutionResult,
    run_enhanced_automation
)
from fixed_automation1 import (
    run_city_analysis,
    get_city_visualizations,
    get_and_save_lulc,
    fetch_and_plot_hydrology,
    CityAnalysisState
)

class IntegratedGISSystem:
    """
    Complete integrated GIS automation system that combines:
    - Enhanced reasoning for workflow generation
    - Intelligent workflow parsing and execution
    - Fixed GIS function implementations
    - Comprehensive error handling and reporting
    """
    
    def __init__(self, groq_api_key: str, model_name: str = "llama3-70b-8192"):
        self.groq_client = Groq(api_key=groq_api_key)
        self.model_name = model_name
        
        # Initialize components
        self.reasoning_engine = EnhancedReasoningEngine()
        self.workflow_executor = WorkflowExecutor()
        
        # Create enhanced query function
        self.enhanced_query_func, _ = create_enhanced_reasoning_query_function(
            self.groq_client, self.model_name
        )
        
        # Execution history
        self.execution_history = []
    
    def process_query(self, query: str, context: str = "", execution_mode: str = "enhanced") -> Dict[str, Any]:
        """
        Process a user query through the complete pipeline:
        1. Generate enhanced workflow using reasoning engine
        2. Parse workflow into executable steps
        3. Execute workflow and capture results
        4. Return comprehensive results
        """
        
        results = {
            "query": query,
            "execution_mode": execution_mode,
            "workflow_text": "",
            "structured_workflow": None,
            "execution_results": [],
            "generated_figures": [],
            "execution_summary": {},
            "errors": [],
            "success": False,
            "processing_time": 0.0
        }
        
        import time
        start_time = time.time()
        
        try:
            # Step 1: Generate enhanced workflow
            if execution_mode == "enhanced":
                workflow_text = self.enhanced_query_func(query, context)
                
                # Also generate structured workflow for better execution
                try:
                    structured_workflow, parameters = self.reasoning_engine.generate_structured_workflow(query)
                    results["structured_workflow"] = structured_workflow
                    results["extracted_parameters"] = parameters
                except Exception as e:
                    results["errors"].append(f"Structured workflow generation failed: {e}")
                    # Fall back to text-based workflow
                    structured_workflow = None
            else:
                # Legacy mode - simple workflow generation
                workflow_text = self._generate_simple_workflow(query, context)
                structured_workflow = None
            
            results["workflow_text"] = workflow_text
            
            # Step 2: Execute workflow
            if execution_mode == "enhanced" and structured_workflow:
                # Use structured execution
                execution_results = self._execute_structured_workflow(structured_workflow, query)
            else:
                # Use text-based execution
                automation_results = run_enhanced_automation(query, workflow_text)
                execution_results = automation_results["execution_results"]
                results["generated_figures"] = automation_results["generated_figures"]
                results["execution_summary"] = automation_results["summary"]
            
            results["execution_results"] = execution_results
            
            # Step 3: Process results
            if execution_results:
                successful_steps = sum(1 for r in execution_results if r.success)
                total_steps = len(execution_results)
                results["success"] = successful_steps > 0
                
                if not results.get("execution_summary"):
                    results["execution_summary"] = {
                        "total_steps": total_steps,
                        "successful_steps": successful_steps,
                        "failed_steps": total_steps - successful_steps,
                        "success_rate": successful_steps / total_steps if total_steps > 0 else 0
                    }
            
            # Step 4: Handle special cases (city analysis)
            city_name = self._extract_city_name(query)
            if city_name and execution_mode == "enhanced":
                try:
                    # Run the fixed LangGraph city analysis
                    city_results = run_city_analysis(city_name)
                    results["city_analysis_results"] = city_results
                    
                    # Extract figures from city analysis
                    if city_results.get("dem_fig"):
                        results["generated_figures"].append(city_results["dem_fig"])
                    if city_results.get("terrain_figs"):
                        results["generated_figures"].extend(city_results["terrain_figs"])
                        
                except Exception as e:
                    results["errors"].append(f"City analysis failed: {e}")
            
        except Exception as e:
            results["errors"].append(f"Processing failed: {e}")
            results["success"] = False
        
        results["processing_time"] = time.time() - start_time
        
        # Store in history
        self.execution_history.append(results)
        
        return results
    
    def _generate_simple_workflow(self, query: str, context: str) -> str:
        """Generate simple workflow for legacy mode"""
        system_prompt = (
            "You are a GIS workflow planner. Given a user query and tool documentation, "
            "generate a step-by-step geospatial workflow including:\n"
            "- Data needed\n- Tools to use\n- Operations to perform\n"
            "Use libraries like Rasterio, GeoPandas, WhiteboxTools, OSMnx, or PyQGIS where needed."
        )
        full_prompt = f"Context from GIS docs:\n{context}\n\nUser Query:\n{query}"

        response = self.groq_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.3,
            max_tokens=2048
        )
        return response.choices[0].message.content
    
    def _execute_structured_workflow(self, workflow: WorkflowTemplate, query: str) -> List[ExecutionResult]:
        """Execute a structured workflow template"""
        results = []
        
        for i, operation in enumerate(workflow.operations, 1):
            result = self._execute_operation(operation, i)
            results.append(result)
            
            # Stop execution if critical step fails
            if not result.success and operation.operation_type.value in ["data_acquisition"]:
                break
        
        return results
    
    def _execute_operation(self, operation: GISOperation, step_number: int) -> ExecutionResult:
        """Execute a single GIS operation"""
        import time
        start_time = time.time()
        
        try:
            # Map operation to actual function
            function_mapping = {
                "get_city_visualizations": get_city_visualizations,
                "get_and_save_lulc": get_and_save_lulc,
                "fetch_and_plot_hydrology": fetch_and_plot_hydrology,
                # Add more mappings as needed
            }
            
            function = function_mapping.get(operation.function_name)
            if not function:
                return ExecutionResult(
                    step_number=step_number,
                    success=False,
                    error_message=f"Function '{operation.function_name}' not implemented"
                )
            
            # Execute function with parameters
            output = function(**operation.parameters)
            
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                step_number=step_number,
                success=True,
                output=output,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ExecutionResult(
                step_number=step_number,
                success=False,
                error_message=f"Execution error: {str(e)}",
                execution_time=execution_time
            )
    
    def _extract_city_name(self, query: str) -> Optional[str]:
        """Extract city name from query"""
        import re
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
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get execution history"""
        return self.execution_history
    
    def clear_history(self):
        """Clear execution history"""
        self.execution_history = []
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive report of the execution"""
        report_lines = []
        
        report_lines.append("# GIS Automation Execution Report")
        report_lines.append(f"**Query:** {results['query']}")
        report_lines.append(f"**Execution Mode:** {results['execution_mode']}")
        report_lines.append(f"**Processing Time:** {results['processing_time']:.2f} seconds")
        report_lines.append(f"**Success:** {'Yes' if results['success'] else 'No'}")
        report_lines.append("")
        
        # Execution Summary
        if results.get("execution_summary"):
            summary = results["execution_summary"]
            report_lines.append("## Execution Summary")
            report_lines.append(f"- Total Steps: {summary.get('total_steps', 0)}")
            report_lines.append(f"- Successful Steps: {summary.get('successful_steps', 0)}")
            report_lines.append(f"- Failed Steps: {summary.get('failed_steps', 0)}")
            report_lines.append(f"- Success Rate: {summary.get('success_rate', 0):.1%}")
            report_lines.append("")
        
        # Generated Workflow
        if results.get("workflow_text"):
            report_lines.append("## Generated Workflow")
            report_lines.append(results["workflow_text"])
            report_lines.append("")
        
        # Execution Results
        if results.get("execution_results"):
            report_lines.append("## Step-by-Step Results")
            for result in results["execution_results"]:
                status = "✅ Success" if result.success else "❌ Failed"
                report_lines.append(f"### Step {result.step_number}: {status}")
                report_lines.append(f"- Execution Time: {result.execution_time:.2f}s")
                if result.error_message:
                    report_lines.append(f"- Error: {result.error_message}")
                report_lines.append("")
        
        # Errors
        if results.get("errors"):
            report_lines.append("## Errors Encountered")
            for error in results["errors"]:
                report_lines.append(f"- {error}")
            report_lines.append("")
        
        # Generated Outputs
        if results.get("generated_figures"):
            report_lines.append("## Generated Visualizations")
            report_lines.append(f"- Number of figures generated: {len(results['generated_figures'])}")
            report_lines.append("")
        
        return "\n".join(report_lines)

# Streamlit Integration Functions
def create_streamlit_interface():
    """Create the Streamlit interface for the integrated system"""
    
    st.set_page_config(
        page_title="Integrated GIS Automation System",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Enhanced CSS
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        .success-metric {
            background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
            padding: 1rem;
            border-radius: 8px;
            color: white;
            text-align: center;
            margin: 0.5rem 0;
        }
        .error-metric {
            background: linear-gradient(90deg, #ff416c 0%, #ff4b2b 100%);
            padding: 1rem;
            border-radius: 8px;
            color: white;
            text-align: center;
            margin: 0.5rem 0;
        }
        .step-success {
            background: #f0fdf4;
            border-left: 4px solid #22c55e;
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 4px;
        }
        .step-error {
            background: #fef2f2;
            border-left: 4px solid #ef4444;
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 4px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🛰️ Integrated GIS Automation System</h1>
        <p>Complete solution with enhanced reasoning, intelligent execution, and comprehensive reporting</p>
    </div>
    """, unsafe_allow_html=True)
    
    return True

def run_integrated_demo():
    """Run a demonstration of the integrated system"""
    
    # Sample API key (replace with actual key)
    api_key = "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"
    
    # Initialize system
    system = IntegratedGISSystem(api_key)
    
    # Sample query
    query = "Generate a comprehensive flood risk analysis for Mumbai including DEM, land use, and population data"
    
    print("🚀 Running Integrated GIS Automation System Demo")
    print(f"📝 Query: {query}")
    print("⏳ Processing...")
    
    # Process query
    results = system.process_query(query, execution_mode="enhanced")
    
    # Generate report
    report = system.generate_report(results)
    
    print("\n📊 Results:")
    print(f"✅ Success: {results['success']}")
    print(f"⏱️ Processing Time: {results['processing_time']:.2f}s")
    print(f"📈 Steps Executed: {len(results['execution_results'])}")
    print(f"🖼️ Figures Generated: {len(results['generated_figures'])}")
    
    if results['errors']:
        print(f"⚠️ Errors: {len(results['errors'])}")
        for error in results['errors']:
            print(f"   - {error}")
    
    print("\n📋 Full Report:")
    print(report)
    
    return results

if __name__ == "__main__":
    # Run demo if executed directly
    run_integrated_demo()

