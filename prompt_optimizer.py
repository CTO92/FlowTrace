import os
import json
import logging
from openai import OpenAI


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERFORMANCE_LOG_FILE = os.path.join(os.path.dirname(__file__), "agent_performance.log")

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_agent_performance():
    """Analyzes agent performance from the log file and returns a dictionary of metrics."""
    if not os.path.exists(PERFORMANCE_LOG_FILE):
        logger.warning("Performance log file not found.")
        return {}

    performance_data = []
    with open(PERFORMANCE_LOG_FILE, "r") as f:
        for line in f:
            try:
                performance_data.append(json.loads(line))
            except json.JSONDecodeError:
                logger.error(f"Could not decode line: {line}")
                continue

    # Aggregate performance data
    agent_metrics = {}
    for entry in performance_data:
        agent_name = entry['agent_name']
        if agent_name not in agent_metrics:
            agent_metrics[agent_name] = {'successes': 0, 'failures': 0, 'total_duration': 0}

        agent_metrics[agent_name]['total_duration'] += entry['duration']
        if entry['success']:
            agent_metrics[agent_name]['successes'] += 1
        else:
            agent_metrics[agent_name]['failures'] += 1

    return agent_metrics

def suggest_prompt_improvements(agent_metrics):
    """Suggests prompt improvements based on aggregated agent performance metrics."""
    # Example: Simple rule-based approach
    prompt_suggestions = {}
    for agent, metrics in agent_metrics.items():
        total_runs = metrics['successes'] + metrics['failures']
        failure_rate = metrics['failures'] / total_runs if total_runs > 0 else 0

        if failure_rate > 0.5:  # Example threshold: If failure rate > 50%
            prompt_suggestions[agent] = f"Consider revising the prompt for {agent} to provide clearer instructions or more context. High failure rate detected."
    return prompt_suggestions

if __name__ == "__main__":
    # Run the analysis and print suggestions (for testing)
    agent_performance = analyze_agent_performance()
    if agent_performance:
        suggestions = suggest_prompt_improvements(agent_performance)
        if suggestions:
            print("Suggested Prompt Improvements:")
            print(json.dumps(suggestions, indent=4))