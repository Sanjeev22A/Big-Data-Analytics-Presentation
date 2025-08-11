import re
from collections import defaultdict
import queue
import threading
import time
import math
# ---------------- TOKENIZER ----------------
def tokenize(text):
    """Lowercase and split into alphanumeric tokens."""
    text = text.lower()
    return re.findall(r"[a-z0-9]+", text)

# ---------------- MAPPER ----------------
def mapper(mapper_id, records, emit_fn):
    """
    Mapper for Naive Bayes with logging.
    Each record: doc_id, class_label, text
    """
    print(f"[Mapper-{mapper_id}] Started, processing {len(records)} records...")
    start_time = time.time()

    local_word_count = {}
    for line in records:
        doc_id, class_label, text = line.strip().split("\t", 2)
        tokens = tokenize(text)

        # Track number of docs per class
        class_key = f"CLASS#CLASS_LABEL:{class_label}"
        local_word_count[class_key] = local_word_count.get(class_key, 0) + 1

        # Track word counts per class
        for token in tokens:
            word_key = f"WORD#CLASS_LABEL:{class_label}#WORD:{token}"
            local_word_count[word_key] = local_word_count.get(word_key, 0) + 1

    # Emit locally aggregated counts
    for key, value in local_word_count.items():
        print("-"*50)
        print(f"LOG(Mapper{mapper_id})- Key and value pair :{key}:{value}")
        print("-"*50)
        emit_fn((key, value))

    emit_fn(None)  # End signal for mapper

    print(f"[Mapper-{mapper_id}] Finished in {time.time() - start_time:.3f}s, emitted {len(local_word_count)} keys.")

# ---------------- EMITTER ----------------
def make_emitter(output_queue):
    """Creates a thread-safe emitter that pushes to output queue."""
    def emit_fn(key_value):
        output_queue.put(key_value)
    return emit_fn

# ---------------- SHUFFLER ----------------
def shuffler(input_queue, num_mappers):
    """
    Groups mapper outputs by key.
    Waits for all mappers to signal completion (None).
    """
    print("[Shuffler] Started...")
    grouped_data = defaultdict(list)
    finished_mappers = 0

    while finished_mappers < num_mappers:
        key_value = input_queue.get()
        if key_value is None:
            finished_mappers += 1
            continue

        key, value = key_value
        grouped_data[key].append(value)

    print(f"[Shuffler] Finished. Grouped into {len(grouped_data)} keys.")
    return grouped_data

# ---------------- REDUCER ----------------
def reducer(reducer_id, items):
    """Aggregates values for each key with logging."""
    print(f"[Reducer-{reducer_id}] Started, processing {len(items)} keys...")
    start_time = time.time()
    result = {key: sum(values) for key, values in items}
    for key,value in result.items():
        print("-"*50)
        print(f"LOG(Reducer{reducer_id})- key,value pair-{key}:{value}")
        print("-"*50)
    print(f"[Reducer-{reducer_id}] Finished in {time.time() - start_time:.3f}s.")
    return result

# ---------------- BUILD MODEL ----------------
def build_model(reduced_counts):
    """
    Builds Naive Bayes probabilities from reduced counts.
    """
    model = {}
    total_docs = 0
    total_words_per_class = {}
    vocabulary = set()
    class_count = {}

    # Count docs and words
    for key, count in reduced_counts.items():
        if key.startswith("CLASS#CLASS_LABEL:"):
            class_label = key.split(":")[1]
            class_count[class_label] = class_count.get(class_label, 0) + count
            total_docs += count
        elif key.startswith("WORD#CLASS_LABEL:"):
            parts = key.split("#WORD:")
            class_label = parts[0].split(":")[1]
            word = parts[1]
            vocabulary.add(word)
            total_words_per_class[class_label] = total_words_per_class.get(class_label, 0) + count

    vocab_size = len(vocabulary)

    # Calculate priors
    priors = {cls: class_count[cls] / total_docs for cls in class_count}

    # Calculate likelihoods with Laplace smoothing
    likelihoods = {}
    for key, count in reduced_counts.items():
        if key.startswith("WORD#CLASS_LABEL:"):
            parts = key.split("#WORD:")
            class_label = parts[0].split(":")[1]
            word = parts[1]
            likelihoods[(word, class_label)] = (count + 1) / (total_words_per_class[class_label] + vocab_size)

    model['priors'] = priors
    model['likelihoods'] = likelihoods
    model['vocab'] = vocabulary
    return model

# ---------------- RUN MAPREDUCE ----------------
def run_naive_bayes_mapreduce(data, num_mapper_threads=2, num_reducer_threads=2):
    """
    Runs the Naive Bayes training using multi-threaded MapReduce with logging.
    """
    q_out = queue.Queue()

    # Split data for mappers
    chunk_size = len(data) // num_mapper_threads
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    # Start mappers
    mappers = []
    for idx, chunk in enumerate(chunks):
        emit_fn = make_emitter(q_out)
        t = threading.Thread(target=mapper, args=(idx, chunk, emit_fn))
        mappers.append(t)
        t.start()

    # Shuffle
    grouped = shuffler(q_out, num_mapper_threads)

    # Start reducers
    items = list(grouped.items())
    chunk_size = max(1, len(items) // num_reducer_threads)
    reducer_chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

    reduced_parts = []
    reducers = []
    lock = threading.Lock()

    def reducer_task(reducer_id, chunk):
        result = reducer(reducer_id, chunk)
        with lock:
            reduced_parts.append(result)

    for idx, chunk in enumerate(reducer_chunks):
        t = threading.Thread(target=reducer_task, args=(idx, chunk))
        reducers.append(t)
        t.start()

    # Wait for all threads
    for t in mappers:
        t.join()
    for t in reducers:
        t.join()

    # Merge reducer results
    final_counts = defaultdict(int)
    for part in reduced_parts:
        for k, v in part.items():
            final_counts[k] += v

    print(f"[Driver] Merged reducer results into {len(final_counts)} final counts.")

    # Build model
    return build_model(final_counts)

import math

# ---------------- PREDICT ----------------
def predict(model, text):
    """
    Predicts the class label for a given text using the trained Naive Bayes model.
    Also converts log-probability scores to normal probabilities.
    """
    tokens = tokenize(text)
    vocab = model['vocab']
    priors = model['priors']
    likelihoods = model['likelihoods']

    log_scores = {}

    for cls in priors:
        # Start with log prior
        log_scores[cls] = math.log(priors[cls])

        # Add log likelihoods for each token
        for token in tokens:
            if (token, cls) in likelihoods:
                log_scores[cls] += math.log(likelihoods[(token, cls)])
            else:
                # Unseen word -> Laplace smoothing
                log_scores[cls] += math.log(
                    1 / (len(vocab) + sum(1 for (w, c) in likelihoods if c == cls))
                )

    # Convert from log space to probability space
    probs_unnormalized = {cls: math.exp(score) for cls, score in log_scores.items()}
    total = sum(probs_unnormalized.values())
    probabilities = {cls: val / total for cls, val in probs_unnormalized.items()}

    # Pick the class with highest probability
    predicted_class = max(probabilities, key=probabilities.get)

    return predicted_class, log_scores, probabilities

