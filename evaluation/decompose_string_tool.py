""" 
@file_name: decompose_string_tool.py
@author: Bin Liang
@date: 2025-08-15
@description: 
    This file is used to decompose the string into a list of strings.
"""


string = """
Chain-of-thought (CoT) prompting does **not** depend on a particular linguistic style to be effective. Instead, it is robust across different annotators, writing styles, and prompt variations, though careful prompt engineering can still influence performance in some tasks.\n\n---\n\n## Main Findings\n\n### 1. Robustness to Linguistic Style and Annotator Variation\n\n- Chain-of-thought prompting consistently outperforms standard prompting across a wide range of tasks, regardless of the linguistic style or background of the annotator who writes the reasoning steps.<reference>{17}</reference>\n- Experiments using CoT prompts written by different annotators\u2014including those without machine learning backgrounds\u2014show that all such prompts provide substantial gains over standard prompting.<reference>{14}</reference><reference>{17}</reference>\n- The effectiveness of CoT prompting does not hinge on a specific set of exemplars or a particular writing style; even concise or stylistically varied chains of thought are effective.<reference>{14}</reference><reference>{17}</reference>\n- Randomly sampled exemplars from datasets with reasoning steps (e.g., GSM8K) are as effective as carefully written ones, further supporting robustness to linguistic style and exemplar source.<reference>{17}</reference>\n\n### 2. Prompt Engineering and Task Sensitivity\n\n- While CoT prompting is generally robust, prompt engineering can still matter, especially for certain tasks.<reference>{14}</reference>\n    - For example, in classification tasks like coin flip, the order of exemplars can significantly affect performance.<reference>{14}</reference>\n    - Some tasks may require more careful prompt construction for CoT prompting to succeed.<reference>{14}</reference>\n- The number of exemplars and their order typically have minimal impact, except in specific cases.<reference>{14}</reference>\n- Gains from CoT prompting persist even as the number of few-shot exemplars varies.<reference>{14}</reference>\n\n### 3. Model and Task Generality\n\n- CoT prompting works across different large language models (e.g., LaMDA, GPT-3, PaLM), though the magnitude of gains can vary by model and task.<reference>{14}</reference>\n- The approach is effective for arithmetic, commonsense, and symbolic reasoning tasks, and is not limited to a particular domain or linguistic style.<reference>{14}</reference><reference>{17}</reference>\n\n### 4. Empirical Evidence\n\n- In experiments, all sets of CoT prompts\u2014regardless of annotator or style\u2014outperformed standard prompting by a large margin.<reference>{17}</reference>\n- The success of CoT prompting does not require tuning for linguistic style or strict adherence to a solution template.<reference>{17}</reference>\n- The method is robust to different annotators, independently-written chains of thought, different exemplars, and various language models.<reference>{14}</reference><reference>{17}</reference>\n\n---\n\n## Conclusion\n\nThe success of chain-of-thought prompting does **not** depend on a particular linguistic style. It is robust to variations in annotator, writing style, exemplar selection, and prompt order, though prompt engineering can still play a role in optimizing performance for certain tasks.<reference>{14}</reference><reference>{17}</reference> This robustness makes CoT prompting a generally dependable technique for eliciting reasoning in large language models.\n\n---\n\n**References:**\n\n<reference>{14}</reference>  \n<reference>{17}</reference>
"""


""" 
expected for:
[
    {
        "content": "Chain-of-thought (CoT) prompting does **not** depend on a particular linguistic style to be effective. Instead, it is robust across different annotators, writing styles, and prompt variations, though careful prompt engineering can still influence performance in some tasks.\n\n---\n\n## Main Findings\n\n### 1. Robustness to Linguistic Style and Annotator Variation\n\n- Chain-of-thought prompting consistently outperforms standard prompting across a wide range of tasks, regardless of the linguistic style or background of the annotator who writes the reasoning steps."
        "references": [17]
    },
    ...
]
"""

def decompose_string(string):
    """
    Decompose the string into a list of dictionaries with content and references.
    
    Args:
        string (str): Input string containing content with reference tags
        
    Returns:
        list: List of dictionaries with 'content' and 'references' keys
    """
    import re
    
    # Pattern to match <reference>{number}</reference>
    reference_pattern = r'<reference>\{(\d+)\}</reference>'
    
    # First, let's find all content blocks that end with references
    # Split the text by looking for patterns that end with one or more references
    # Pattern: any content followed by one or more reference tags
    content_with_refs_pattern = r'(.*?)(<reference>\{\d+\}</reference>(?:<reference>\{\d+\}</reference>)*)'
    
    result = []
    remaining_text = string.strip()
    
    while remaining_text:
        # Try to match content followed by references
        match = re.search(content_with_refs_pattern, remaining_text, re.DOTALL)
        
        if match:
            content_part = match.group(1).strip()
            refs_part = match.group(2)
            
            # Extract all reference numbers from the references part
            ref_numbers = [int(num) for num in re.findall(r'<reference>\{(\d+)\}</reference>', refs_part)]
            
            if content_part:  # Only add if there's actual content
                result.append({
                    "content": content_part,
                    "references": ref_numbers
                })
            
            # Continue with the remaining text after this match
            remaining_text = remaining_text[match.end():].strip()
        else:
            # No more patterns found, add remaining text if any
            if remaining_text:
                result.append({
                    "content": remaining_text,
                    "references": []
                })
            break
    
    # If no result was found, return the whole string
    if not result and string.strip():
        result = [{"content": string.strip(), "references": []}]
    
    # Filter out very short content that might be artifacts
    result = [item for item in result if len(item["content"]) > 5]
    
    return result


# Test the function
if __name__ == "__main__":
    result = decompose_string(string)
    
    print("Decomposed result:")
    import json 
    print(json.dumps(result, indent=4))
        