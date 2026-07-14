from langsmith import Client
from langsmith import testing as t
from src.agent import email_assistant
from src.eval.email_dataset import examples_triage

client = Client()

dataset_name = "E-mail Dataset"

if not client.has_dataset(dataset_name=dataset_name):

    dataset = client.create_dataset(
        dataset_name=dataset_name, 
        description="A dataset of e-mails and their triage decisions."
    )

    client.create_examples(dataset_id=dataset.id, examples=examples_triage)

def target_email_assistant(inputs: dict) -> dict:

    try:
        response = email_assistant.invoke({"email_input": inputs["email_input"]})
        if "classification_decision" in response:
            return {"classification_decision": response['classification_decision']}
        else:
            print("No classification_decision in response from workflow agent")
            return {"classification_decision": "unknown"}
    except Exception as e:
        print(f"Error in workflow agent: {e}")
        return {"classification_decision": "unknown"}

feedback_key = "classification" 

def classification_evaluator(outputs: dict, reference_outputs: dict) -> bool:
    """Check if the answer exactly matches the expected answer."""
    return outputs["classification_decision"].lower() == reference_outputs["classification"].lower()

experiment_results_workflow = client.evaluate(
    target_email_assistant,
    data=dataset_name,
    evaluators=[classification_evaluator],
    experiment_prefix="E-mail agent", 
    max_concurrency=5, 
)

