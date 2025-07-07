from deepeval.metrics import AnswerRelevancyMetric, FactualConsistencyMetric, FluencyMetric, CoherenceMetric
from deepeval.test_case import LLMTestCase
from deepeval.runner import run_test_cases
from transformers import pipeline

# Load your fine-tuned model and tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

FINETUNED_MODEL = "final_model" 
BASE_MODEL = "meta-llama/Llama-3.2-1B"  

# Load pipelines
finetuned_pipe = pipeline("text-generation", model=AutoModelForCausalLM.from_pretrained(FINETUNED_MODEL),
                          tokenizer=AutoTokenizer.from_pretrained(FINETUNED_MODEL), max_new_tokens=256)

base_pipe = pipeline("text-generation", model=AutoModelForCausalLM.from_pretrained(BASE_MODEL),
                     tokenizer=AutoTokenizer.from_pretrained(BASE_MODEL), max_new_tokens=256)


# Load your evaluation dataset
import json
with open("data/instructionquality.json", "r") as f:
    dataset = json.load(f)

# Sample a few test cases (or evaluate all)
samples = dataset[:50] 


def run_eval(model_pipe, model_name="Model"):
    print(f"\n🔍 Running DeepEval for: {model_name}")

    test_cases = []
    for sample in samples:
        input_question = sample["question"]
        expected_answer = sample["answer"]

        generated = model_pipe(input_question)[0]['generated_text']

        test_cases.append(
            LLMTestCase(
                input=input_question,
                actual_output=generated,
                expected_output=expected_answer,
                retrieval_context=None  # Optional
            )
        )

    # Define metrics
    metrics = [
        AnswerRelevancyMetric(threshold=0.5),
        FactualConsistencyMetric(threshold=0.5),
        FluencyMetric(threshold=0.5),
        CoherenceMetric(threshold=0.5),
    ]

    # Run evaluation
    run_test_cases(test_cases=test_cases, metrics=metrics, model=model_name)


if __name__ == "__main__":
    run_eval(base_pipe, "Base LLaMA")
    run_eval(finetuned_pipe, "Fine-Tuned LLaMA 3.2 1B")
