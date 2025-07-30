from typing import List, Dict
from deepeval.metrics import (
    ToolCorrectnessMetric, GEval, HallucinationMetric, 
    AnswerRelevancyMetric
)
from deepeval.test_case import LLMTestCaseParams

# Simple metric profiles for different agents
AGENT_METRICS = {
    "research_agent": [
        ToolCorrectnessMetric(),
        GEval(
            name="coherence",
            criteria="Is the response coherent and well-structured?",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
        ),
        AnswerRelevancyMetric()
    ],
    "cli_agent": [
        ToolCorrectnessMetric(),
        HallucinationMetric(),
        GEval(
            name="coherence",
            criteria="Is the response coherent and well-structured?",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
        )
    ],
    "api_agent": [
        AnswerRelevancyMetric(),
        GEval(
            name="format_compliance",
            criteria="Does the response follow the expected API format?",
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT]
        )
    ]
}

def get_metrics_for_agent(agent_id: str, custom_metrics: List = None) -> List:
    """Get metrics for a specific agent"""
    if custom_metrics:
        return custom_metrics
    return AGENT_METRICS.get(agent_id, [ToolCorrectnessMetric()])