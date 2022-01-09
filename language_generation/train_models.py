import os
import ijson
import random
import logging
from tqdm import tqdm
from simpletransformers.language_modeling import LanguageModelingModel


logging.basicConfig(level=logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)


def main():
    for category_name in os.listdir("training_data/processed_files"):
        category_name = category_name.split(".json")[0]
        prepare_training_data(category_name, sample_size=30000)
        for polarity_mode in ["positive", "negative"]:
            finetune_pretrained_gpt(category_name, polarity_mode, num_train_epochs=3)


def read_dict_list_from_file(file_path):
    """
    Reads a JSON file in the format [dict1, dict2, ..., dictN] and returns it as a list comprehension without loading
    it into memory

    :param file_path: path of the file to read
    :return: list comprehension of the file content
    """
    return [t for t in tqdm(ijson.items(open(file_path), "item"))]


def sample_and_split(string_list, sample_size, test_ratio):
    """
    Takes random sample of size sample_size from a string list and returns it split into two sets (train set & test set)

    :param string_list: string list of all texts
    :param sample_size: number of strings to take as sample
    :param test_ratio: percentage of sample that is put in test set instead of train set
    :return: tuple - (train set as string list, test set as string list)
    """
    random.seed(1)
    random.shuffle(string_list)
    train_set = list(string_list)[:sample_size]
    test_size = int(test_ratio * len(train_set))
    test_set = []
    for i in range(test_size):
        test_set.append(train_set.pop(0))
    return train_set, test_set


def prepare_training_data(category_name, sample_size, test_ratio=0.05):
    """
    Loads full category JSON files, selects the texts for reviews with positive (5 stars) and negative
    (1 or 2 stars) rating, takes a sample of size sample_size of each and writes it to txt file, replacing
    linebreaks with spaces. Writes one train file and one test file, using test_ratio as size ratio for them.

    :param category_name: category to sample from
    :param sample_size: number of reviews per sentiment class to take as sample
    :param test_ratio: percentage of sample that is used for testing instead of training (default 5%)
    """
    if os.path.exists("training_data/{}_negative_train.txt".format(category_name)):
        print("Skipped preparing Training Data for {}".format(category_name))
        return

    dict_list = read_dict_list_from_file("training_data/processed_files/{}.json".format(category_name))
    train_neg, test_neg = sample_and_split([d["text"] for d in dict_list if d["rating"] in [1, 2]], sample_size, test_ratio=test_ratio)
    train_pos, test_pos = sample_and_split([d["text"] for d in dict_list if d["rating"] in [5]], sample_size, test_ratio=test_ratio)

    for name, texts in [("negative_train", train_neg), ("negative_test", test_neg),
                        ("positive_train", train_pos), ("positive_test", test_pos)]:
        with open("training_data/{}_{}.txt".format(category_name, name), "w") as f:
            for line in texts:
                f.write(line.replace("\n", " ")+"\n")


def finetune_pretrained_gpt(category_name, polarity_mode, num_train_epochs):
    """
    Loads the pretrained GPT-2 model, trains it on the train set for the given category and sentiment class and
    saves it to /trained_models

    :param category_name: category of the training file
    :param polarity_mode: sentiment class of the training file
    :param num_train_epochs: number of epochs to train the LanguageModelingModel for
    """
    if os.path.exists("trained_models/{}_{}".format(category_name, polarity_mode)):
        print("Skipped training for {} ({})".format(category_name, polarity_mode))
        return

    model = LanguageModelingModel('gpt2', 'gpt2')
    model.args.reprocess_input_data = True
    model.args.overwrite_output_dir = False
    model.args.output_dir = "trained_models/{}_{}/".format(category_name, polarity_mode)
    # Masked Language Modeling (MLM) deactivated because for GPT-2 Causal Language Modeling (CLM) is used
    model.args.mlm = False
    model.args.train_batch_size = 8
    model.args.num_train_epochs = num_train_epochs
    model.args.save_eval_checkpoints = False
    model.args.save_best_model = True
    model.train_model("training_data/{}_{}_train.txt".format(category_name, polarity_mode),
                      eval_file="training_data/{}_{}_train.txt".format(category_name, polarity_mode))
    model.eval_model("training_data/{}_{}_train.txt".format(category_name, polarity_mode))


if __name__ == '__main__':
    main()
