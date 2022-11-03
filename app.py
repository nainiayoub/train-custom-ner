import streamlit as st
import pandas as pd

with st.sidebar:
    st.markdown("""
        # Learn more
        * [Explosion.ai Blog](https://explosion.ai/blog)
        * [Spacy's NER model](https://spacy.io/universe/project/video-spacys-ner-model)
        * [Prodigy annotation tool](https://prodi.gy/)
    """)

html_temp = """
            <div style="background-color:{};padding:1px">
            
            </div>
            """

st.markdown("""
    ## Train a custom NER model with spaCy
    """)
st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"),unsafe_allow_html=True)
st.markdown("""
    In this workshop we will learn to annotate and prepare text data to train a custom NER model with spaCy.
    The provided dataset is too small to have good results, however the same approach can be implemented on a larger dataset.
    The pre-defined entity types (classes or categories) are: `PERSON` and `LOC`. 
    
""")
with st.expander("1. The Dataset"):
    st.markdown("""
        ### 1. The Dataset
        The pre-defined entity types (classes or categories) are: `PERSON` and `LOC`.
    """)

    # load data
    dataset = "./data/data_workshop.csv"
    df_data = pd.read_csv(dataset)
    st.table(df_data)
    # download button
    csv = df_data.to_csv(index=False).encode('utf-8')
    st.download_button(
    "Download Dataframe",
    csv,
    'data_workshop.csv',
    "text/csv",
    key='download-csv'
    )

with st.expander("2. Preparing and annotating data"):
    st.markdown("""
        ### 2. Preparing and annotating data
        Before training any model we need to collect and annotate the data. 
        Annotating the data  means labeling the named entities with the appropriate entity type (in this case `PERSON` or `LOC`).
        The sentences in the column `Text` are labeled in the colum `Named Entities` with the NE (Named Entity) label between parenthesis.

        1. For each named entity of every sentence, provide the the start position of the entity in terms of characters, the end position of the entity in terms of characters and the entity type (hint: use _rfind_).
        2. Add a new column named `Annotations` to the dataframe containing the output of 1.
        """)

    st.markdown(""" 
        3. Convert the data to spaCy's training format: 
            * `[('Text', {'entities': [('Start Index', 'End Index', 'Entity Type')]})]`
        """)

with st.expander("3. Preparing the pipeline"):
    st.markdown("""
        ### Preparing the pipeline
        1. Install spaCy
    """)
    code = "!pip install spacy\n!python -m spacy download en_core_web_lg"
    st.code(code)

    st.markdown("""
        2. Create a blank pipeline (spaCy blank model) of class 'en'. We also define the number of iterations to be 100:
    """)
    code = '''
    ##
    model = None
    # model output directory
    output_dir = './ner_model_workshop'
    n_iter = 100
    if model is not None:
        nlp = spacy.load(model)
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')
        print("Created blank 'en' model")'''

    st.code(code, language='python')

    st.markdown("""
        3. Create a NER pipeline on the existing blank one:
    """)
    code = '''
    # add ner pipeline
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe('ner', last=True)
    else:
        ner = nlp.get_pipe('ner')
    '''

    st.code(code, language='python')

with st.expander("4. Train the model"):
    st.markdown("""
        ### Train the model
        1. Add the entity types PERSON and LOC to the pipeline as labels.
    """)
    code = '''# ignore
    from spacy.training.example import Example
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])
    '''
    st.code(code, language='python')

    st.markdown("""
        2. Disable the other components of the pipeline leaving just NER for training. Use the with statement to invoke nlp.disable pipe as a context manager.
    """)
    code = '''
    # disable pipelines
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
    '''
    st.code(code, language='python')

    st.markdown("""
        3. Create an optimizer object and feed this optimizer object to the training method as a parameter, and then for each epoch shuffle the dataset with `random.shuffle`:
    """)
    code = '''
        # data shuffling
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
    '''
    st.code(code, language='python')
    st.markdown("""
        4. For each example sentence in the dataset, create an Example object from the sentence and its annotation. A Sample object contains information for a training instance. It stores two objects: one to hold the reference data and one to hold the pipeline predictions (type of the entity named predicted):
    """)
    code = '''
    # create Example object
    losses = {}
    for batch in spacy.util.minibatch(TRAIN_DATA, size=2):
        for text, annotations in batch:
            example = Example.from_dict(nlp.make_doc(text), annotations)
    '''
    st.code(code, language='python')

    st.markdown("""
        5. Feed the Example object and the optimizer object into `nlp.update`,  the training method where the NER model is trained (print the losses for every batch):
    """)

    code = '''
    # train model
    nlp.update(
        [example],   
        drop=0.5,  
        sgd=optimizer,
        losses=losses)
    '''
    st.code(code, language='python')

    st.markdown("""
        6. Once the training is complete, save the newly trained NER model to disk under the previously defined outputdir directory.
    """)
    code = '''
    # Save model
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir) 
    '''
    st.code(code, language='python')

with st.expander("5. Evaluate the model"):
    st.markdown("""
        ### Evaluate the model
        Assuming we having a test split, which we don't, we compute the evaluation metrics for every entity type.
    """)
    code = '''
        # evaluation  metrics
        from spacy.training.example import Example
        from spacy.scorer import Scorer

        examples = []
        scorer = Scorer()
        for text, annotations in TRAIN_DATA:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            example.predicted = nlp(str(example.predicted))
            examples.append(example)
        scorer.score(examples)
    '''
    st.code(code, language='python')

with st.expander("6. Test the model on an example sentence"):
    st.markdown("""
        1. Load the trained model with `spacy.load`.
        2. Create a document with the loaded model.
        3. Extract named entities from the document with `displacy`.
    """)