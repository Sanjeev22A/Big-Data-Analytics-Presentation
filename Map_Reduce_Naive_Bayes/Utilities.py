import re
from collections import defaultdict
import queue
def tokenize(text):
    """
    Tokenizes the input text into words, removing punctuation and converting to lowercase.
    
    Args:
        text (str): The input text to tokenize.
        
    Returns:
        list: A list of tokens (words).
    """
    text=text.lower()
    return re.findall(r"[a-z0-9]+",text)

def mapper(records,emit_fn):
    """
    Mapper function for the Naive Bayes algorithm.
    
    Args:

        records (list): A list of records to process.Where each line in record contains doc_id,class_label and text.
        emit_fn (function): Function to emit the results.
    Logic:
        Includes local word aggregation to fasten the process.
        We emit both class label (as total docs in a class is needed) and word count for each class.
        emit_fn is a memory and thread safe emitter function.
    Returns:
        None
    """
    local_word_count={}
    for line in records:
        doc_id,class_label,text=line.strio().split("\t",2)
        tokens=tokenize(text)
        class_key=f"CLASS#CLASS_LABEL:{class_label}"
        local_word_count[class_key]=local_word_count.get(class_key,0)+1
        for token in tokens:
            word_key=f"WORD#CLASS_LABEL:{class_label}#WORD:{token}"
            local_word_count[word_key]=local_word_count.get(word_key,0)+1
        
    for key,value in local_word_count.items():
        emit_fn((key,value))
    
    emit_fn(None) #Marking the end of mapper thread function

def make_emitter(output_queue):
    """
    Creates an emitter function that puts results into an output queue.
    
    Args:
        output_queue (queue.Queue): The queue to put emitted results into.
        
    Returns:
        function: An emitter function that takes a key-value pair and puts it in the queue.
    """
    def emit_fn(key_value):
        output_queue.put(key_value)
    
    return emit_fn

def shuffler(input_queue,num_mappers):

    grouped_data=defaultdict(list)
    finished_mappers=0
    while finished_mappers<num_mappers:
        key_value=input_queue.get()
        if key_value is None:
            finished_mappers+=1
            continue

        key,value=key_value
        grouped_data[key].append(value)
    return grouped_data

def reducer(items):
    """
    Reducer function to aggregate word counts and class labels.
    Args:
        items (list): list of key,[values list]
    Returns:
        dict: A dictionary with keys as class labels and words, and values as their aggregated counts.
    """
    return {key:sum(values) for key,values in items}

def build_model(reduced_counts):
    """
    This function builds a Naive Bayes model from the reduced counts.
    Args:
        reduced_counts (dict): A dictionary with keys as class labels and words, and values as their aggregated counts.
    Returns:
        dict: A dictionary representing the Naive Bayes model.
        Prior Probability- P(class)= count(class)/total_docs
        Likelihood Probability- P(word|class)= (count(word,class)+1)/(count(class)+V)  //Laplce smoothing, V is vocabulary size
        Vocabulary size is the number of unique words across all classes.
        Total Words per class is the sum of all word counts for that class.

    """

    model={}
    total_docs=0
    total_words_per_class={}
    vocabulary=set()
    class_count={}
    class_label_set=set()
    for key,count in reduced_counts.items():
        if key.startswith("CLASS#CLASS_LABEL:"):
            class_label=key.split(":")[1]
            if( class_label not in class_label_set):
                class_label_set.add(class_label)
            class_count[class_label]=class_count.get(class_label,0)+1                