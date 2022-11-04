# Train a custom NER model with spaCy
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nainiayoub-train-custom-ner-app-fwpfqk.streamlit.app/)
![visitor badge](https://visitor-badge.glitch.me/badge?page_id=nainiayoub.train-custom-ner)

In this workshop you will learn to annotate and prepare text data to train a custom NER model with spaCy. The provided dataset is too small to have good results, however the same approach can be implemented on a larger dataset. The pre-defined entity types (classes or categories) are: `PERSON` and `LOC`.

```mermaid
flowchart LR

A[Prepare and Annotate data] --> B[Prepare NER pipeline]
B --> C[Train NER model]
C --> D[Evaluate model]
D --> E[Test the model]
```
