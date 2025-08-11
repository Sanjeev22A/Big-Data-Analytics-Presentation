Naive Bayes Classifier with MapReduce (Multi-threaded)
Overview
This project implements a Naive Bayes text classifier using a MapReduce-style architecture in Python with multi-threading.
It demonstrates how mapper, shuffler, and reducer stages can be combined to train a probabilistic classifier efficiently on textual data.

How It Works
1. Tokenizer
Before any processing, the input text is:

Lowercased

Split into alphanumeric tokens using regex
This ensures consistent vocabulary handling.

2. Mapper
Each mapper thread processes a chunk of the dataset.
For each document:

Counts number of documents per class
Key format:

objectivec
Copy
Edit
CLASS#CLASS_LABEL:<class>
Counts occurrences of each word per class
Key format:

objectivec
Copy
Edit
WORD#CLASS_LABEL:<class>#WORD:<token>
Uses local aggregation to reduce intermediate data before emitting.

Emits (key, count) pairs to a shared output queue.

Signals completion by emitting None.

3. Shuffler
The shuffler:

Reads all mapper outputs from the queue

Groups values by their key

Waits until all mappers have finished before returning grouped data
This step is equivalent to the shuffle and sort phase in MapReduce.

4. Reducer
Each reducer thread:

Receives a subset of (key, [counts]) pairs

Sums counts for each key

Produces final (key, total_count) results

Reducer outputs are merged into a final count dictionary.

5. Model Building
From the reduced counts:

Document counts per class → used to compute prior probabilities

cpp
Copy
Edit
P(class) = docs_in_class / total_docs
Word counts per class → used to compute likelihood probabilities with Laplace smoothing

arduino
Copy
Edit
P(word | class) = (count(word,class) + 1) / (total_words_in_class + vocab_size)
Vocabulary is built from all unique words encountered.

The final model contains:

priors: P(class)

likelihoods: P(word | class)

vocab: set of all unique tokens

6. Prediction
When predicting:

Compute log probabilities for each class:

cpp
Copy
Edit
log(P(class)) + Σ log(P(token | class))
Convert log scores back to normalized probabilities via exponentiation and normalization.

Select the class with the highest probability.

Architecture Diagram
lua
Copy
Edit
         ┌─────────────┐
         │   Dataset   │
         └──────┬──────┘
                │
        ┌───────▼────────┐
        │   Mappers (n)  │  -- local counting per chunk
        └───────┬────────┘
                │
        ┌───────▼────────┐
        │   Shuffler     │  -- group by key
        └───────┬────────┘
                │
        ┌───────▼────────┐
        │   Reducers (m) │  -- aggregate counts
        └───────┬────────┘
                │
        ┌───────▼────────┐
        │  Build Model   │  -- priors, likelihoods
        └───────┬────────┘
                │
        ┌───────▼────────┐
        │   Prediction   │
        └────────────────┘
Example Output
csharp
Copy
Edit
[Mapper-0] Started, processing 2 records...
[Mapper-0] Finished, emitted 9 keys.
[Shuffler] Grouped into 17 keys.
[Reducer-0] Finished.
[Driver] Built model with 2 classes, 34 words.
Text: "Win cash prize now"
Predicted: spam
Probabilities: {'spam': 0.927, 'ham': 0.073}
