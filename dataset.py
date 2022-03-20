import os
import tqdm
from loguru import logger
from torch.utils.data import Dataset
from joblib import Parallel, delayed, cpu_count


# read the file and return the text for each file.
def read_file(directory, filename):
    filepath = os.path.join(directory, filename)
    with open(filepath, "r") as f:
        text = f.read()
    # return filename without extension
    return [os.path.splitext(filename)[0], text]


# return the id and text containing for a directory
def return_texts(directory, use_workers=False):
    texts = {}
    files = os.listdir(directory)
    if use_workers:
        n_jobs = cpu_count()
        parallel = Parallel(n_jobs=n_jobs, backend="multiprocessing")
        results = parallel(delayed(read_file)(directory, filename) \
                        for filename in tqdm.tqdm(files))
    else:
        results = []
        for filename in tqdm.tqdm(files):
            results.append(read_file(directory, filename))

    for result in results:
        texts[result[0]] = result[1]

    return texts


# return the mapping where key is id, and value is labels list
# that is for ex : {"abvq": [Lead, Lead, Claim, Claim]}
# abvq is the ID and the list of the ground truth for 4 words.
def return_texts_mapping(texts, df):
    total = len(texts)
    text_mapping = {}
    for text_index, (text_id, text) in tqdm.tqdm(enumerate(texts.items()), total=total):
        # split the text
        words = text.split()
        # initialize all the labels to other
        text_mapping[text_id] = ["O"]*len(words)
        # search for the current text_id
        text_df = df.loc[df["id"] == text_id]
        # loop through each row (iterrows is slower)
        for row in text_df.itertuples():
            indices = row.predictionstring.split()
            # convert the predictionstring indices to integer
            # remember that both start and end is included
            start = int(indices[0])
            end = int(indices[-1])
            entities = [row.discourse_type]*(end - start + 1)
            
            text_mapping[text_id][start: end + 1] = entities

    return text_mapping


# tokenize in BIES format
def tokenize(sentence, tokenizer, tags=None):
    sentence = " ".join(sentence.split())
    tokens = tokenizer(
        sentence,
        add_special_tokens=False,
        return_offsets_mapping=True
    )
    biestags = None
    if tags is not None:
        biestags = []
        start = 0
        for index, offset in enumerate(tokens["offset_mapping"]):
            if index == 0:
                if tags[0] == "O":
                    biestags.append(tags[0])
                else:
                    biestags.append("B-{}".format(tags[0]))
            else:
                previous_offset = tokens["offset_mapping"][index - 1]
                # if from previous offset to current offset there is space
                # which means that current offset is next word
                if " " in sentence[previous_offset[0]: offset[1]]:
                    start += 1
                if tags[start] == "O":
                    biestags[-1] = biestags[-1].replace("B-", "S-").replace("I-", "E-")
                    biestags.append(tags[start])
                else:
                    if biestags[-1] == "O":
                        biestags.append("B-{}".format(tags[start]))
                    elif biestags[-1].split("-")[1] == tags[start]:
                        biestags.append("I-{}".format(tags[start]))
                    else:
                        biestags[-1] = biestags[-1].replace("B-", "S-").replace("I-", "E-")
                        biestags.append("B-{}".format(tags[start]))

        biestags[-1] = biestags[-1].replace("B-", "S-").replace("I-", "E-")

    return tokens, biestags


def prepare_inputs(directory, tokenizer, df):
    logger.info("Reading the files from {} directory".format(directory))
    texts = return_texts(directory)
    num_documents = len(texts)
    logger.info("Number of documents found in directory = {}".format(num_documents))
    logger.info("Combining the read files with ground truth from csv")
    texts_mapping = return_texts_mapping(texts, df)

    # convert the texts to ids and also align the tags with bies schema
    # remember that all the data is loaded here only
    logger.info("Running tokenizer on all inputs.")
    inputs = {}
    for text_id, text in tqdm.tqdm(texts.items(), total=num_documents):
        tags = texts_mapping[text_id]
        tokens, biestags = tokenize(text, tokenizer, tags)
        inputs[text_id] = {"tokens": tokens, "tags": biestags}

    return inputs


def prepare_inputs_for_test(directory, tokenizer):
    logger.info("Reading the files from {} directory".format(directory))
    texts = return_texts(directory)
    num_documents = len(texts)
    logger.info("Number of documents found in directory = {}".format(num_documents))

    logger.info("Running tokenizer on all inputs.")
    inputs = {}
    for text_id, text in tqdm.tqdm(texts.items(), total=num_documents):
        tokens, biestags = tokenize(text, tokenizer, tags=None)
        inputs[text_id] = {"tokens": tokens}

    return inputs



class FeedbackPriceDataset(Dataset):
    def __init__(self, ids, inputs, label2idx):
        self.ids = ids
        self.inputs = inputs
        self.label2idx = label2idx

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        inp = self.inputs[self.ids[index]]
        return {
            "id": self.ids[index],
            "input_ids": inp["tokens"]["input_ids"],
            "attention_mask": inp["tokens"]["attention_mask"],
            "tags": [self.label2idx[tag] for tag in inp["tags"]],
            "schema_tags": inp["tags"]
        }


class FeedbackPriceTestDataset(Dataset):
    def __init__(self, ids, inputs):
        self.ids = ids
        self.inputs = inputs
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        inp = self.inputs[self.ids[index]]
        return {
            "id": self.ids[index],
            "input_ids": inp["tokens"]["input_ids"],
            "attention_mask": inp["tokens"]["attention_mask"]
        }
