#!/usr/bin/env python3

import re
import numpy as np
import streamlit as st
from neo4j import GraphDatabase
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
import math
import tiktoken
import os
import glob
import requests
import json

# ---- Utility functions ----

LLAMA_BASE_URL = os.environ.get("LLAMA_BASE_URL", "http://localhost:8000/v1")
LLAMA_MODEL_ID = os.environ.get("LLAMA_MODEL_ID", "local")

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def min_max_normalization(weights):
    """Normalize weights using min-max normalization."""
    min_weight = min(weights)
    max_weight = max(weights)
    if max_weight == min_weight:  # Avoid division by zero
        return [1.0 for _ in weights]
    return [(w - min_weight) / (max_weight - min_weight) for w in weights]

def softmax(weights):
    """Compute softmax values for a list of weights."""
    max_weight = max(weights)  # Subtract the maximum weight to prevent overflow
    exp_weights = [math.exp(w - max_weight) for w in weights]
    total = sum(exp_weights)
    return [w / total for w in exp_weights]

def get_top_k_documents(query_vec, driver, k=10):
    with driver.session() as session:
        results = session.run("""
            MATCH (d)
            WHERE d.embedding IS NOT NULL AND d.weight IS NOT NULL AND d.all_content IS NOT NULL and d.weight>0
            RETURN d.all_content AS all_content, d.weight AS weight, d.embedding AS embedding, elementId(d) AS eid
        """)
        scored = [
            (
                cosine_similarity(query_vec, record["embedding"]) * record["weight"],
                {
                    "all_content": record["all_content"],
                    "weight": record["weight"],
                    "elementId": record["eid"]
                }
            )
            for record in results
        ]
        return [doc[1] for doc in sorted(scored, key=lambda x: x[0], reverse=True)[:k]]

def describe_relationships(driver, element_ids):
    relationships = []
    with driver.session() as session:
        for eid in element_ids:
            results = session.run("""
                MATCH (node)-[rel]->(connected)
                WHERE elementId(node) = $eid
                RETURN node.name AS node_name, type(rel) AS relation, connected.name AS connected_name
            """, {"eid": eid})
            relationships.extend(
                f"{record['node_name']}-{record['relation']}->{record['connected_name']}"
                for record in results
            )
    return relationships

def format_reference(doc):
    return "\n".join(f"{k} : {v}" for k, v in doc.items())

def count_tokens(text):
    """Count the number of tokens in the given text."""
    tokenizer = tiktoken.encoding_for_model("gpt-4o")
    return len(tokenizer.encode(text))

def get_highest_count(driver):
    with driver.session() as session:
        result = session.run("MATCH (node) RETURN MAX(node.count) AS highest_count").single()
        return result["highest_count"] if result else 1.0

def retrieve_nodes_with_signal(driver, start_element_ids, signal_threshold=0.5, weight_multiplier=3, normalization_function="min_max", extend_graph=False):
    visited_signals = {}  # Track visited nodes and their signals
    result_nodes = []
    queue = [(eid, 1.0) for eid in start_element_ids]  # Start with initial nodes and full signal strength

    while queue:
        current_eid, current_signal = queue.pop(0)

        # Check if the node has been visited and update its signal if the new signal is stronger
        if current_eid in visited_signals and current_signal <= visited_signals[current_eid]:
            continue
        visited_signals[current_eid] = current_signal

        if current_signal < signal_threshold:
            continue

        with driver.session() as session:
            # Fetch relationships and node properties for the current node
            connected_results = session.run("""
                MATCH (node)-[rel]->(connected)
                WHERE elementId(node) = $eid
                RETURN type(rel) AS relation_type, 
                       elementId(connected) AS connected_eid,
                       rel.count AS relation_count,
                       connected.all_content AS all_content
            """, {"eid": current_eid})

            # Separate relationships by type
            of_block_relations = [rec for rec in connected_results if rec["relation_type"] == "OF_BLOCK_COUNT"]
            close_to_relations = [rec for rec in connected_results if rec["relation_type"] == "CLOSE_TO_COUNT"]

            # Extract counts from relationships
            of_block_counts = [(rec["relation_count"] or 1) for rec in of_block_relations]
            close_to_counts = [(rec["relation_count"] or 1) for rec in close_to_relations]



            # Apply normalization function
            if of_block_counts:
                normalized_of_block_weights = min_max_normalization(of_block_counts) if normalization_function == "min_max" else of_block_counts
                normalized_of_block_weights = [w * weight_multiplier for w in normalized_of_block_weights]
            else:
                normalized_of_block_weights = []

            if close_to_counts:
                normalized_close_to_weights = min_max_normalization(close_to_counts) if normalization_function == "min_max" else close_to_counts
                normalized_close_to_weights = [w * weight_multiplier for w in normalized_close_to_weights]
            else:
                normalized_close_to_weights = []

            # Store the current node's signal
            result_nodes.append({
                "elementId": current_eid,
                "signal": current_signal,
                "total_of_block_count": sum(of_block_counts),
                "total_close_to_count": sum(close_to_counts),
                "all_content": session.run("""
                    MATCH (node)
                    WHERE elementId(node) = $eid
                    RETURN node.all_content AS all_content
                """, {"eid": current_eid}).single()["all_content"]
            })

            # Propagate signal to connected nodes with decay
            for i, rec in enumerate(of_block_relations + close_to_relations):
                decayed_signal = current_signal
                if i < len(normalized_of_block_weights):
                    decayed_signal *= normalized_of_block_weights[i]
                if i < len(normalized_close_to_weights):
                    decayed_signal *= normalized_close_to_weights[i]
                # Only propagate if the decayed signal is above the threshold
                if decayed_signal >= signal_threshold:
                    queue.append((rec["connected_eid"], decayed_signal))
                    # Add connected node's content to results
                    result_nodes.append({
                        "elementId": rec["connected_eid"],
                        "signal": decayed_signal,
                        "all_content": rec["all_content"]
                    })

            # Extend graph if toggle is activated
            if extend_graph:
                neighbors = session.run("""
                    MATCH (node)-[rel]-(connected)
                    WHERE elementId(node) = $eid and connected.weight=0
                    RETURN elementId(connected) AS neighbor_id, connected.all_content AS all_content
                """, {"eid": current_eid})
                for neighbor in neighbors:
                    neighbor_id = neighbor["neighbor_id"]
                    if neighbor_id not in visited_signals:
                        result_nodes.append({
                            "elementId": neighbor_id,
                            "signal": 0,  # Neighbors are added without signal propagation
                            "all_content": neighbor["all_content"]
                        })

    return result_nodes

def split_pseudo_code(pseudo_code, minimal_size=500):
    """
    Split pseudo-code into sections based on triple backticks (```), ensuring each section meets the minimal size.
    :param pseudo_code: The pseudo-code string to split.
    :param minimal_size: Minimum size (number of characters) for each section.
    :return: List of pseudo-code sections.
    """
    sections = []
    current_section = []
    inside_section = False

    for line in pseudo_code.splitlines():
        # Check if the line starts or ends a section
        if line.strip().startswith("```") or line.strip().startswith("```java"): 
            if inside_section:
                # End the current section
                section_content = "\n".join(current_section)
                if len(section_content) >= minimal_size:  # Check if the section meets the minimal size
                    sections.append(section_content)
                current_section = []
                inside_section = False
            else:
                # Start a new section
                inside_section = True
        elif inside_section:
            current_section.append(line)

    # Add the last section if it meets the minimal size
    if current_section:
        section_content = "\n".join(current_section)
        if len(section_content) >= minimal_size:
            sections.append(section_content)

    return sections

def load_resources(resource_folder):
    """
    Load the content of all files in the specified resource folder.
    :param resource_folder: Path to the folder containing resource files.
    :return: Concatenated content of all files in the folder.
    """
    resource_files = glob.glob(os.path.join(resource_folder, "*"))  # Get all files in the folder
    resources_content = []

    for file_path in resource_files:
        with open(file_path, "r", encoding="utf-8") as file:
            resources_content.append(file.read())

    return "\n\n".join(resources_content)  # Concatenate all file contents

def run_rag_pipeline_with_final_merge(query, driver, embedder, model, pseudo_code_model, merge_model, prompt, pseudo_code_prompt, merge_prompt, extend_graph, k, signal_threshold, weight_multiplier, prompt_incremental):
    query_vec = embedder.embed_query(query)
    top_docs = get_top_k_documents(query_vec, driver, k=k)
    if not top_docs:
        return "No relevant context found in the graph.", [], "", "", 0

    top_element_ids = [doc["elementId"] for doc in top_docs]
    retrieved_nodes = retrieve_nodes_with_signal(
        driver, 
        start_element_ids=top_element_ids, 
        signal_threshold=signal_threshold, 
        weight_multiplier=weight_multiplier, 
        normalization_function="min_max",
        extend_graph=extend_graph
    )
    all_docs = top_docs + retrieved_nodes

    grammar_path = os.environ.get("GAML_GRAMMAR_PATH", "ressources/simplified_rules.gbnf")

    formatted_references = "\n\n".join(format_reference(doc) for doc in all_docs)
    relation_descriptions = describe_relationships(driver, top_element_ids) if extend_graph else []

    prompt_context = formatted_references + ("\n\n---\n\n" + "\n".join(relation_descriptions) if relation_descriptions else "")
    token_count = count_tokens(prompt_context)

    # Generate pseudo-code using the first LLM
    pseudo_code_input = {
        "references": prompt_context,
        "question": query,
    }
    pseudo_code_chain = RunnablePassthrough() | pseudo_code_prompt | pseudo_code_model | StrOutputParser()
    pseudo_code_output = pseudo_code_chain.invoke(pseudo_code_input)


    print("JAVA CODE OUTPUT : "+str(pseudo_code_output))

    # Split pseudo-code into sections
    pseudo_code_sections = split_pseudo_code(pseudo_code_output.strip())


    #print(pseudo_code_output)
    print("NB DE SECTIONS DE CODE JAVA : "+str(len(pseudo_code_sections)))
    print(pseudo_code_sections)

    # Load resources from the resource folder
    resource_folder = "./resource"  # Path to the resource folder
    merge_resources = load_resources(resource_folder)

    # Perform RAG for each pseudo-code block and pass it into the second LLM
    partial_outputs = []

    if len(pseudo_code_sections) <= 1:    
        for section in pseudo_code_sections:
            # Perform RAG for the current pseudo-code block
            section_query_vec = embedder.embed_query(section)
            section_top_docs = get_top_k_documents(section_query_vec, driver, k=k)
            section_top_element_ids = [doc["elementId"] for doc in section_top_docs]
            section_retrieved_nodes = retrieve_nodes_with_signal(
                driver, 
                start_element_ids=section_top_element_ids, 
                signal_threshold=signal_threshold, 
                weight_multiplier=weight_multiplier, 
                normalization_function="min_max",
                extend_graph=extend_graph
            )
            section_all_docs = section_top_docs + section_retrieved_nodes

            section_formatted_references = "\n\n".join(format_reference(doc) for doc in section_all_docs)
            section_prompt_context = section_formatted_references

            # Pass the pseudo-code block and its context into the second LLM
            partial_prompt_input = {
                "references": section_prompt_context,
                "question": section,
            }
            partial_chain = RunnablePassthrough() | prompt | model | StrOutputParser()
            partial_output = partial_chain.invoke(partial_prompt_input)
            partial_outputs.append(re.sub(r"<think>.*?</think>", "", partial_output, flags=re.DOTALL).strip())
            previous_output=partial_outputs[-1]
    else:
        section=pseudo_code_sections[0]
        section_query_vec = embedder.embed_query(section)
        section_top_docs = get_top_k_documents(section_query_vec, driver, k=k)
        section_top_element_ids = [doc["elementId"] for doc in section_top_docs]
        section_retrieved_nodes = retrieve_nodes_with_signal(
            driver, 
            start_element_ids=section_top_element_ids, 
            signal_threshold=signal_threshold, 
            weight_multiplier=weight_multiplier, 
            normalization_function="min_max",
            extend_graph=extend_graph
        )
        section_all_docs = section_top_docs + section_retrieved_nodes
        section_formatted_references = "\n\n".join(format_reference(doc) for doc in section_all_docs)
        section_prompt_context = section_formatted_references
        # Pass the pseudo-code block and its context into the second LLM
        partial_prompt_input = {
            "references": section_prompt_context,
            "question": section,
        }
        partial_chain = RunnablePassthrough() | prompt | model | StrOutputParser()
        partial_output = partial_chain.invoke(partial_prompt_input)
        previous_output=re.sub(r"<think>.*?</think>", "", partial_output, flags=re.DOTALL).strip()
        partial_outputs.append(previous_output)


        for section in pseudo_code_sections[1:]:
            # Perform RAG for the current pseudo-code block
            section_query_vec = embedder.embed_query(section)
            section_top_docs = get_top_k_documents(section_query_vec, driver, k=k)
            section_top_element_ids = [doc["elementId"] for doc in section_top_docs]
            section_retrieved_nodes = retrieve_nodes_with_signal(
                driver, 
                start_element_ids=section_top_element_ids, 
                signal_threshold=signal_threshold, 
                weight_multiplier=weight_multiplier, 
                normalization_function="min_max",
                extend_graph=extend_graph
            )
            section_all_docs = section_top_docs + section_retrieved_nodes

            section_formatted_references = "\n\n".join(format_reference(doc) for doc in section_all_docs)
            section_prompt_context = section_formatted_references

            # Pass the pseudo-code block and its context into the second LLM
            partial_prompt_input = {
                "references": section_prompt_context,
                "question": section,
                "code_to_complete": previous_output,  # Include the previous output for incremental generation
            }
            partial_chain = RunnablePassthrough() | prompt | model | StrOutputParser()
            partial_output = partial_chain.invoke(partial_prompt_input)
            previous_output=re.sub(r"<think>.*?</think>", "", partial_output, flags=re.DOTALL).strip()
            partial_outputs.append(previous_output)
            print("PREVIOUS OUTPUT : "+str(previous_output))


    code_box = st.empty()
    _buffer = []

    def on_delta(piece: str):
        _buffer.append(piece)
        # Render as code so it stays monospaced; set language if you like
        code_box.code("".join(_buffer), language="gaml")


    final_gaml_code = constrained_correction_stream(
        merge_prompt=merge_prompt,
        previous_output=partial_outputs[-1],
        pseudo_code= pseudo_code_output.strip(),
        references= merge_resources,
        grammar_path=grammar_path,          # ⚠️ must be GBNF, not EBNF/Lark
        model_id=LLAMA_MODEL_ID,
        base_url=LLAMA_BASE_URL,
        temperature=0.2,
        on_delta=on_delta
    )

    return [final_gaml_code.strip(), all_docs, pseudo_code_output.strip(), partial_outputs, token_count]


# ---- Streamlit App ----

st.set_page_config(page_title="Graph RAG on GAML", layout="wide")
st.title("Graph RAG")
st.write("Ask any question based on your Neo4j knowledge graph.")

query = st.text_input("🙋 Enter your question:", "Create a model where different people move randomly with the display of an experiment.")
run_btn = st.button("Run Query")

st.sidebar.header("⚙️ Parameters")
k = st.sidebar.slider("Top-K Documents", min_value=1, max_value=100, value=5)
signal_threshold = st.sidebar.slider("Signal Threshold", min_value=0.0, max_value=1.0, step=0.01, value=0.7)
weight_multiplier = st.sidebar.slider("Weight Multiplier", min_value=0.0, max_value=2.0, step=0.01, value=1.0)
extend_graph = st.sidebar.checkbox("Extend Connection Graph", value=False)

def constrained_correction_stream(
    merge_prompt: ChatPromptTemplate,
    previous_output: str,
    pseudo_code: str,
    references: str,
    *,
    grammar_path: str,
    model_id: str = LLAMA_MODEL_ID,
    base_url: str = LLAMA_BASE_URL,
    temperature: float = 0.2,
    #max_tokens: int = 2000,
    on_delta=None,            # callback(delta_text) -> None
) -> str:
    """Stream grammar-constrained tokens from llama.cpp server and optionally render as they arrive."""
    #print('generation commence')
    correction_input = merge_prompt.format_prompt(
        previous_output=previous_output,
        pseudocode=pseudo_code,
        references=references
    ).to_string()

    #print("\n Correction input : \n "+str(correction_input))

    #print("merge input : "+merge_input)

    with open(grammar_path, "r", encoding="utf-8") as f:
        grammar = f.read()

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "Output valid GAML only."},
            {"role": "user", "content": correction_input}
        ],
        "temperature": temperature,
        #"max_tokens": max_tokens,
        "grammar": grammar,
        "stream": True
    }

    print("payload ok")

    full = ""
    with requests.post(f"{base_url}/chat/completions", json=payload, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
                delta = obj["choices"][0]["delta"].get("content")
            except Exception:
                delta = None
            if delta:
                full += delta
                if on_delta:
                    on_delta(delta)
    return full




if run_btn and query:
    with st.spinner("Running graph RAG with pseudo-code generation and final merge..."):
        neo4j_uri = "bolt://localhost:7687"
        neo4j_user = "neo4j"
        neo4j_password = "password"
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

        embedder = OllamaEmbeddings(model="mxbai-embed-large")
        pseudo_code_model = ChatOllama(model="qwen2.5-coder:latest")
        model = ChatOllama(model="qwen2.5-coder:32b")
        merge_model = ChatOllama(model="qwen2.5-coder:32b")

        pseudo_code_prompt = ChatPromptTemplate.from_template(
"""
You are a helpful AI that generates ABM java.

Generate agent oriented pseudocode for the following question:
{question}

Do only generate java code. Split your code in logical sections, each section should be separated by triple backticks (```).
Use balanced parentheses and ensure that each section is self-contained.

follow the structure

```
<write the first block here>
```
```
<write the second block here>
```
"""
        )

        prompt = ChatPromptTemplate.from_template(
"""
You are a helpful AI, expert in gaml code. The following knowledge is at your disposal:

{references}

You know the following things about gaml :


The primitive types in GAML are:
- int
- float
- string
- bool


The other types in GAML are:

pair
pair<type1, type2>
rgb
point

the conditionnals are as follows:

if (expressionBoolean = true) {{
// block of statements
}}
if (expressionBoolean = true) {{
// block 1 of statements
}} else {{
// block 2 of statements
}}
if (expressionBoolean = true) {{
// block 1 of statements
}} else if (expressionBoolean2 != false) {{
// block 2 of statements
}} else {{
// block 3 of statements
}}


Here are the loops in GAML:

loop times: 10 {{
write "loop times";
}}
loop i from: 1 to: 10 step: 1 {{
write "loop for " + i;
}}
int j <- 1;
loop while: (j <= 10) {{
write "loop while " + j;
j <- j + 1;
}}
list<int> list_int <- [1,2,3,4,5,6,7,8,9,10];
loop i over: list_int {{
write "loop over " + i;
}}


Here are the declarations of procedures or actions in GAML:

action myAction {{
write "Action without param";
}}
action myActionWithParam( int int_param,
string my_string <- "default value") {{
write my_string + int_param;
}}


Here is how to call a procedure or an action in GAML:

do myAction();
do myActionWithParam(3, "other string");
do myActionWithParam(3); // the second parameter has its default value
ask an_agent {{
do proc(3);
}}


Here is how to declare a funciton :

int myFunction {{
return 1+1;
}}
int myFunctionWithParam(int i, int j <- 0){{
return i + j;
}}


Here is how to define a species in GAML:

species mySpecies1 {{
int s1_int;
float energy <- 10.0;
init {{
// statements dedicated to the initialization of agents
}}
reflex reflex_name {{
// set of statements
}}
aspect square {{
draw square(10);
draw circle(5) color: #red ;
}}
}}


Here is how to create an agent in GAML:

create mySpecies1 number: 10;
create mySpecies1 number: 20 {{
an_int <- 0;
}}


Here is how to define an experiment in GAML:

experiment expeName type: gui {{
parameter "A variable" var: an_int <- 2
min: 0 max: 1000 step: 1 category: "Parameters";
output {{
display display_name {{
species mySpecies2 aspect: square;
species mySpecies1;
}}
display other_display_name {{
chart "chart_name" type: series {{
data "time series" value: a_float;
}}
}}
}}
}}
// repeat defines the number of replications for the same parameter values
// keep_seed means whether the same random generator seed is used at the first
replication for each parameter values
experiment expeNameBatch type: batch repeat: 2
keep_seed: true until: (booleanExpression) {{
parameter "A variable" var: an_int <- 2 min: 0 max: 1000 step: 1 ;
method exhaustive maximize: an_indicator ;
permanent {{
display other_display_name {{
chart "chart_name" type: series {{
data "time series" value: a_float;
}}
}}
}}
}}

A gaml model is structured as follows:

model myFirstModel
global {{
// global variables declaration
// initialization of the model
// global behaviors
}}
species mySpecies1 {{
// attributes, initialization, behaviors and aspects of a species
}}
experiment expName {{
// Defines the way the model is executed, the parameters and the outputs.
}}



Thanks to your knowledge, translate this java code into GAML code. LIMIT YOUR COMMENTS TO THE MINIMUM.
Here is the pseudo code to translate:
{question}
"""
        )




        prompt_incremental = ChatPromptTemplate.from_template(
"""
You are a helpful AI, expert in gaml code. The following knowledge is at your disposal:

{references}

You know the following things about gaml :


The primitive types in GAML are:
- int
- float
- string
- bool


The other types in GAML are:

pair
pair<type1, type2>
rgb
point

the conditionnals are as follows:

if (expressionBoolean = true) {{
// block of statements
}}
if (expressionBoolean = true) {{
// block 1 of statements
}} else {{
// block 2 of statements
}}
if (expressionBoolean = true) {{
// block 1 of statements
}} else if (expressionBoolean2 != false) {{
// block 2 of statements
}} else {{
// block 3 of statements
}}


Here are the loops in GAML:

loop times: 10 {{
write "loop times";
}}
loop i from: 1 to: 10 step: 1 {{
write "loop for " + i;
}}
int j <- 1;
loop while: (j <= 10) {{
write "loop while " + j;
j <- j + 1;
}}
list<int> list_int <- [1,2,3,4,5,6,7,8,9,10];
loop i over: list_int {{
write "loop over " + i;
}}


Here are the declarations of procedures or actions in GAML:

action myAction {{
write "Action without param";
}}
action myActionWithParam( int int_param,
string my_string <- "default value") {{
write my_string + int_param;
}}


Here is how to call a procedure or an action in GAML:

do myAction();
do myActionWithParam(3, "other string");
do myActionWithParam(3); // the second parameter has its default value
ask an_agent {{
do proc(3);
}}


Here is how to declare a funciton :

int myFunction {{
return 1+1;
}}
int myFunctionWithParam(int i, int j <- 0){{
return i + j;
}}


Here is how to define a species in GAML:

species mySpecies1 {{
int s1_int;
float energy <- 10.0;
init {{
// statements dedicated to the initialization of agents
}}
reflex reflex_name {{
// set of statements
}}
aspect square {{
draw square(10);
draw circle(5) color: #red ;
}}
}}


Here is how to create an agent in GAML:

create mySpecies1 number: 10;
create mySpecies1 number: 20 {{
an_int <- 0;
}}


Here is how to define an experiment in GAML:

experiment expeName type: gui {{
parameter "A variable" var: an_int <- 2
min: 0 max: 1000 step: 1 category: "Parameters";
output {{
display display_name {{
species mySpecies2 aspect: square;
species mySpecies1;
}}
display other_display_name {{
chart "chart_name" type: series {{
data "time series" value: a_float;
}}
}}
}}
}}
// repeat defines the number of replications for the same parameter values
// keep_seed means whether the same random generator seed is used at the first
replication for each parameter values
experiment expeNameBatch type: batch repeat: 2
keep_seed: true until: (booleanExpression) {{
parameter "A variable" var: an_int <- 2 min: 0 max: 1000 step: 1 ;
method exhaustive maximize: an_indicator ;
permanent {{
display other_display_name {{
chart "chart_name" type: series {{
data "time series" value: a_float;
}}
}}
}}
}}

A gaml model is structured as follows:

model myFirstModel
global {{
// global variables declaration
// initialization of the model
// global behaviors
}}
species mySpecies1 {{
// attributes, initialization, behaviors and aspects of a species
}}
experiment expName {{
// Defines the way the model is executed, the parameters and the outputs.
}}


This gaml code has been generated by an AI. 

{code_to_complete}

You should augment the code above, following the schema of the gaml code below.


Thanks to your knowledge, translate this pseudocode code into GAML code. LIMIT YOUR COMMENTS TO THE MINIMUM.
Here is the pseudocode to add to the previous GAML code:
{question}
"""
        )




        merge_prompt = ChatPromptTemplate.from_template(
"""
You are a helpful AI that corrects and improves GAML code. The following knowledge is at your disposal:


You know the following things about gaml :


The primitive types in GAML are:
- int
- float
- string
- bool


The other types in GAML are:

pair
pair<type1, type2>
rgb
point

the conditionnals are as follows:

if (expressionBoolean = true) {{
// block of statements
}}
if (expressionBoolean = true) {{
// block 1 of statements
}} else {{
// block 2 of statements
}}
if (expressionBoolean = true) {{
// block 1 of statements
}} else if (expressionBoolean2 != false) {{
// block 2 of statements
}} else {{
// block 3 of statements
}}


Here are the loops in GAML:

loop times: 10 {{
write "loop times";
}}
loop i from: 1 to: 10 step: 1 {{
write "loop for " + i;
}}
int j <- 1;
loop while: (j <= 10) {{
write "loop while " + j;
j <- j + 1;
}}
list<int> list_int <- [1,2,3,4,5,6,7,8,9,10];
loop i over: list_int {{
write "loop over " + i;
}}


Here are the declarations of procedures or actions in GAML:

action myAction {{
write "Action without param";
}}
action myActionWithParam( int int_param,
string my_string <- "default value") {{
write my_string + int_param;
}}


Here is how to call a procedure or an action in GAML:

do myAction();
do myActionWithParam(3, "other string");
do myActionWithParam(3); // the second parameter has its default value
ask an_agent {{
do proc(3);
}}


Here is how to declare a funciton :

int myFunction {{
return 1+1;
}}
int myFunctionWithParam(int i, int j <- 0){{
return i + j;
}}


Here is how to define a species in GAML:

species mySpecies1 {{
int s1_int;
float energy <- 10.0;
init {{
// statements dedicated to the initialization of agents
}}
reflex reflex_name {{
// set of statements
}}
aspect square {{
draw square(10);
draw circle(5) color: #red ;
}}
}}


Here is how to create an agent in GAML:

create mySpecies1 number: 10;
create mySpecies1 number: 20 {{
an_int <- 0;
}}


Here is how to define an experiment in GAML:

experiment expeName type: gui {{
parameter "A variable" var: an_int <- 2
min: 0 max: 1000 step: 1 category: "Parameters";
output {{
display display_name {{
species mySpecies2 aspect: square;
species mySpecies1;
}}
display other_display_name {{
chart "chart_name" type: series {{
data "time series" value: a_float;
}}
}}
}}
}}
// repeat defines the number of replications for the same parameter values
// keep_seed means whether the same random generator seed is used at the first
replication for each parameter values
experiment expeNameBatch type: batch repeat: 2
keep_seed: true until: (booleanExpression) {{
parameter "A variable" var: an_int <- 2 min: 0 max: 1000 step: 1 ;
method exhaustive maximize: an_indicator ;
permanent {{
display other_display_name {{
chart "chart_name" type: series {{
data "time series" value: a_float;
}}
}}
}}
}}

A gaml model is structured as follows:

model myFirstModel
global {{
// global variables declaration
// initialization of the model
// global behaviors
}}
species mySpecies1 {{
// attributes, initialization, behaviors and aspects of a species
}}
experiment expName {{
// Defines the way the model is executed, the parameters and the outputs.
}}



Here are some gaml code examples, that you can take example on in order to correct the gaml code below:
{references}

Here is the template java code used to generate the gaml code:
{pseudocode}

Here is the slightly incorrect gaml code to regenerate. Keep the general logic of the code. :
{previous_output}

Ensure the final program is structured, clear, and follows GAML conventions. RETURN ONLY THE SINGLE FINAL GAML CODE AND LIMIT THE COMMENTS TO THE MINIMUM. DO NOT PROVIDE ANY EXPLANATION BEHIND YOUR ANSWER.
"""
        )

        result = run_rag_pipeline_with_final_merge(query, driver, embedder, model, pseudo_code_model, merge_model, prompt, pseudo_code_prompt, merge_prompt,extend_graph=extend_graph,k=k,signal_threshold=signal_threshold,weight_multiplier=weight_multiplier,prompt_incremental=prompt_incremental)
        driver.close()

    st.subheader("🤓☝️ Final GAML Code")
    st.write(result[0])

    st.markdown(f"😱 **Total retrieved context nodes:** {len(result[1])}")
    st.markdown(f"🧐 **Estimated total tokens in context (low approximation):** {result[4]}")

    st.markdown("---")
    with st.expander("🤩 Show base java code"):
        st.markdown(result[2])
        st.markdown("---")

    with st.expander("🤩 Show Translated GAML Blocks"):
        for i, block in enumerate(result[3]):
            st.markdown(f"### Block {i + 1}")
            st.markdown(block)
            st.markdown("---")

    with st.expander("🤩 Show context nodes used in the answer"):
        st.markdown("### Node Details")
        for doc in result[1]:
            st.markdown(f"""
**Node ID**: `{doc["elementId"]}`  
**Content**: `{doc.get("all_content", "N/A")}`
            """)
            st.markdown("---")