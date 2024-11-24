import json
import re
from sklearn.metrics import f1_score, accuracy_score
import pdb


def extract_intent(json_response):
    def extract_intent(json_response):
        classes = ['positive', 'negative']
        pattern_cmp = re.compile(r"\b(" + "|".join(re.escape(word) for word in classes) + r")\b")
        match = pattern_cmp.search(json_response)
        # match = re.search(r"\{'label':\s*'([^']+)'\}", json_response)
        if match:
            return match.group(1)
        return 'None'


def evaluate_intent_classification(reference_file, output_file):
    with open(reference_file, "r") as f:
        reference_data = [json.loads(line) for line in f]
    with open(output_file, "r") as f:
        output_data = [json.loads(line) for line in f]
    # pdb.set_trace()
    reference_labels = [data["output"] for data in reference_data]
    output_labels = [extract_intent(data["output"]) for data in output_data]
    # print(output_labels)
    acc = accuracy_score(reference_labels, output_labels)
    f1 = f1_score(reference_labels, output_labels, average='weighted')
    return acc, f1


if __name__ == '__main__':
    reference_file = "./dataset/sst2/label.json"  # Replace with your actual reference file
    output_file = "./dataset/sst2/output.json"  # Replace with your actual output file

    acc, f1 = evaluate_intent_classification(reference_file, output_file)
    print(f"Accuracy: {acc}, F1 Score: {f1}")
