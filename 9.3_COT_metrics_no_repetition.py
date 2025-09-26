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
import time
import threading

# Reuse one HTTP session for better connection pooling
SESSION = requests.Session()
ADAPTER = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32, max_retries=0)
SESSION.mount("http://", ADAPTER)
SESSION.mount("https://", ADAPTER)

# Limit concurrent requests to the llama.cpp server
LLAMA_MAX_CONCURRENCY = int(os.getenv("LLAMA_MAX_CONCURRENCY", "2"))
LLAMA_GATE = threading.Semaphore(LLAMA_MAX_CONCURRENCY)

# ---- Utility functions ----

LLAMA_BASE_URL = os.environ.get("LLAMA_BASE_URL", "http://localhost:8000/v1")
LLAMA_MODEL_ID = os.environ.get("LLAMA_MODEL_ID", "local")

# function with semaphore, test if that prevent crash

def post_with_backoff(url, **kw):
    delay = 0.5
    for i in range(5):
        try:
            return SESSION.post(url, **kw)
        except requests.exceptions.ConnectionError:
            if i == 4: raise
            time.sleep(delay); delay *= 2


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
                if len(section_content) >= 0:  # Check if the section meets the minimal size
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

    grammar_path = os.environ.get("GAML_GRAMMAR_PATH", "ressources/simplified_rules_commentaireless.gbnf")

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




    print("\nJAVA CODE OUTPUT : "+str(pseudo_code_output))

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
            print("---------------\nQUESTION : "+str(section))
            print("PREVIOUS OUTPUT : "+str(previous_output))
            previous_output=re.sub(r"<think>.*?</think>", "", partial_output, flags=re.DOTALL).strip()
            print("UPDATED PREVIOUS OUTPUT : "+str(previous_output))
            partial_outputs.append(previous_output)
            

    code_box = True
    _buffer = []

    def on_delta(piece: str):
        _buffer.append(piece)
        # Render as code so it stays monospaced; set language if you like
        #code_box.code("".join(_buffer), language="gaml")


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

queries=["Create a model in which ants are using pheromones to find food. The parameters should be the diffusion and the persistance of the pheromones.",
          "Create a prey predator model taking into account food, reproduction, displacement and energy",
          "Create a model of traffic in a city with cars, traffic lights and pedestrians. The parameters should be the number of cars and the number of pedestrians",
          "Create a model representing a labyrinth an the shortest path between two points. The two points are a parameter from the user.",]
run_btn = True

k=5
signal_threshold=0.7
weight_multiplier=1.0
extend_graph=True

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
    on_delta=None,
) -> str:
    correction_input = merge_prompt.format_prompt(
        previous_output=previous_output,
        #pseudocode=pseudo_code,
        references=references
    ).to_string()

    with open(grammar_path, "r", encoding="utf-8") as f:
        grammar = f.read()

    # Server-side safety net (if your server honors these)
    SERVER_MAX_TOKENS = 5000

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "Output valid GAML only."},
            {"role": "user", "content": correction_input}
        ],
        "temperature": temperature,
        "grammar": grammar,
        "stream": True,
        # Try both common knobs; your server may use one of them:
        "max_tokens": SERVER_MAX_TOKENS,   # OpenAI-compatible
        "n_predict": SERVER_MAX_TOKENS,#SERVER_MAX_TOKENS,    # llama.cpp-compatible
    }

    print("payload ok")
    full = ""

    # --- parser/stop logic state ---
    brace_depth = 0
    stop_now = False

    in_str = False           # "..." or '...'
    str_quote = ""
    esc = False

    in_line_comment = False  # //
    in_block_comment = False # /* ... */

    # post-close window (you set to 10 in your code; keep as-is)
    prev_was_space = True
    post_close_token_budget = None  # counts down from 10 after closing top-level

    # ---- NEW: global token cap (approx, during streaming) -------------------
    MAX_TOKENS = 5000
    tokens_generated = 0
    prev_was_space_global = True  # counts tokens across ALL emitted chars

    def process_chunk(chunk: str) -> str:
        """
        Stream-time limiting:
        - after closing top-level '}', if no '{' within next 10 token starts -> stop
        - stop when global token cap (5000) is reached
        """
        nonlocal brace_depth, stop_now
        nonlocal in_str, str_quote, esc, in_line_comment, in_block_comment
        nonlocal prev_was_space, post_close_token_budget
        nonlocal tokens_generated, prev_was_space_global

        kept = []
        i = 0
        L = len(chunk)

        def emit(ch: str, *, count_top_level: bool) -> None:
            """Append char and update BOTH counters (global cap + top-level window)."""
            nonlocal tokens_generated, prev_was_space_global, prev_was_space, post_close_token_budget, stop_now
            kept.append(ch)

            # --- Global token cap: count token start on emitted text
            is_space_glob = ch.isspace()
            if not is_space_glob and prev_was_space_global:
                tokens_generated += 1
                if tokens_generated >= MAX_TOKENS:
                    stop_now = True
                    return
            prev_was_space_global = is_space_glob

            # --- Top-level post-close window (only when outside strings/comments)
            if count_top_level:
                is_space = ch.isspace()
                if post_close_token_budget is not None and brace_depth == 0:
                    if not is_space and prev_was_space:
                        post_close_token_budget -= 1
                        if post_close_token_budget <= 0:
                            stop_now = True
                            return
                prev_was_space = is_space

        while i < L and not stop_now:
            ch = chunk[i]
            nxt = chunk[i + 1] if i + 1 < L else ""

            if in_line_comment:
                emit(ch, count_top_level=False)
                if ch == "\n":
                    in_line_comment = False
                    prev_was_space = True
                i += 1
                continue


            # detect start of strings
            if ch in ("'", '"'):
                emit(ch, count_top_level=False)
                in_str = True
                str_quote = ch
                esc = False
                i += 1
                continue

            # general character
            emit(ch, count_top_level=True)
            i += 1

        return "".join(kept)

    with LLAMA_GATE:
        try:
            with post_with_backoff(f"{base_url}/chat/completions",
                                   json=payload, stream=True, timeout=(5, 600)) as r:
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
                    if not delta:
                        continue

                    # Apply limits DURING streaming on this delta
                    keep = process_chunk(delta)
                    if keep:
                        full += keep
                        print(keep, end="", flush=True)
                        if on_delta:
                            on_delta(keep)

                    if stop_now:
                        # Proactively terminate the stream right now
                        try:
                            r.close()
                        except Exception:
                            pass
                        break
        except KeyboardInterrupt:
            try:
                r.close()
            except Exception:
                pass
            print("\n[Interrupted] Returning partial output.")

    print("\n--- [stream end] ---")
    return full.rstrip()


## Hotfix pour interdire de générer des tokens après la dernière }


print("trucs a print")
print(run_btn)
print(queries)

if run_btn and queries:
    print("COUCOU")

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
You are a helpful AI that give natural language guidelines to write code.

For instance, if the question is "Create a prey predator model", you should answer with something like:
```
Create the preys. They reproduce and eat grass.
```

```
Create the predators. They reproduce and eat preys. Use a loop to make them hunt preys.
```

```
Create the environment
```

```
Create an experiment to display the model
```

Give a set of coding instruction to answer this question:
{question}

Each section should be separated by triple backticks (```).


follow the structure

```
<write the first indication here>
```
```
<write the second indication here>
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



Thanks to your knowledge, this instruction into GAML code. LIMIT YOUR COMMENTS TO THE MINIMUM.
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

You should augment the code above, 

Thanks to your knowledge, translate this pseudocode code into GAML code. LIMIT YOUR COMMENTS TO THE MINIMUM.
Here is the functionnality to add to the gaml code above:
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

Here is the slightly incorrect gaml code to regenerate. Keep the general logic of the code. :
{previous_output}

Ensure the final program is structured, clear, and follows GAML conventions. RETURN ONLY THE SINGLE FINAL GAML CODE AND LIMIT THE COMMENTS TO THE MINIMUM. DO NOT PROVIDE ANY EXPLANATION BEHIND YOUR ANSWER.
"""
        )

with open("metrics_results_COT.json", "w", encoding="utf-8") as file:
    json.dump([], file, indent=4, ensure_ascii=False)

for _ in range(300):
    for query in queries:
        neo4j_uri = "bolt://localhost:7687"
        neo4j_user = "neo4j"
        neo4j_password = "password"
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

        embedder = OllamaEmbeddings(model="mxbai-embed-large")
        pseudo_code_model = ChatOllama(model="qwen2.5-coder:latest")
        model = ChatOllama(model="qwen2.5-coder:32b")
        merge_model = ChatOllama(model="qwen2.5-coder:32b")



        result = run_rag_pipeline_with_final_merge(query, driver, embedder, model, pseudo_code_model, merge_model, prompt, pseudo_code_prompt, merge_prompt,extend_graph=extend_graph,k=k,signal_threshold=signal_threshold,weight_multiplier=weight_multiplier,prompt_incremental=prompt_incremental)
        driver.close()



        print("RESULT : "+str(len(result)))

#return [final_gaml_code.strip(), all_docs, pseudo_code_output.strip(), partial_outputs, token_count]

        metrics_data = {
                        "query": query,
                        "final_gaml_code": result[0],
                        "pseudo_code_output": result[2],
                        "partial_outputs": result[3]
                    }

        print("METRICS DATA : "+str(metrics_data))
                    # Append to the JSON file
        try:
            with open("metrics_results_COT.json", "r", encoding="utf-8") as file:
                existing_data = json.load(file)
        except FileNotFoundError:
            existing_data = []

        existing_data.append(metrics_data)

        with open("metrics_results_COT.json", "w", encoding="utf-8") as file:
            json.dump(existing_data, file, indent=4, ensure_ascii=False)