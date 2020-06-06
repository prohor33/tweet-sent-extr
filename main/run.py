import random
import traceback
from collections import defaultdict

import pandas as pd
import transformers
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from main.dataloader import TweetDataset
from main.train import run_fold
from main.utils import set_seed, FoldsScore, calculate_jaccard_score, StreamToLogger, dict_beautify_str
from datetime import datetime
import argparse
from models.roberta import TweetModel
import logging
import sys


class Config:
    def __init__(self,
                 version='roberta-base-1.7',
                 device='cuda:0',
                 debug=True,
                 eval=False,
                 # eval_model_path='/home/prohor/Workspace/pycharm_tmp/pycharm_project_597/storage/kaggle_datasets/tweetsentimentextractionmodels'
                 eval_model_path='/home/prohor/Workspace/pycharm_tmp/pycharm_project_597/storage/runs/03_06_2020__17_32_21_roberta-base-1.5_5256'
                 ):
        self.max_len = 128
        self.train_batch_size = 64
        self.valid_batch_size = 16
        self.epochs = 10
        self.version = version
        self.bert_path = "roberta-base"
        # self.bert_path = "deepset/roberta-base-squad2"
        self.model_path = "model.bin"
        self.training_file = "/home/prohor/Workspace/pycharm_tmp/pycharm_project_597/" \
                             "storage/dataset/train_folds_no_prep.csv"
        # self.training_file = "/home/prohor/Workspace/pycharm_tmp/pycharm_project_597/storage/dataset/train_folds_thakur.csv"
        # self.training_file = "/home/prohor/Workspace/pycharm_tmp/pycharm_project_597/storage/dataset" \
        #                 "/positive_train_folds_no_prep.csv"
        # self.training_file = "/home/prohor/Workspace/pycharm_tmp/pycharm_project_597/storage/" \
        #                      "dataset/train_folds_pos_and_neg_no_prep.csv"

        self.results_output_dir = "/home/prohor/Workspace/pycharm_tmp/pycharm_project_597/storage/models/current_"
        self.logs_dir = "/home/prohor/Workspace/pycharm_tmp/pycharm_project_597/storage/runs/"
        self.logs_dir_dbg = "/home/prohor/Workspace/pycharm_tmp/pycharm_project_597/storage/runs_dbg/"
        self.log_path = None
        self.device = device
        self.debug = debug
        self.eval = eval
        self.eval_model_path = self.results_output_dir + '0'
        if eval_model_path:
            self.eval_model_path = eval_model_path
        self.folds = 5
        self.verbose = True


def main(config: Config):
    # seed = random.randint(0, 10000)
    seed = 1
    set_seed(seed)

    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y__%H_%M_%S") + '_' + config.version + f'_{seed}'
    log_dir = f"{config.logs_dir_dbg if config.debug else config.logs_dir}{dt_string}"
    config.results_output_dir = log_dir + '/'

    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text('model_version_info', config.version, 0)
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

    file_handler = logging.FileHandler(f'{log_dir}/all.log', 'w')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)

    stdout_logger = logging.getLogger('stdout')
    sys.stdout = StreamToLogger(stdout_logger, logging.INFO)
    stderr_logger = logging.getLogger('stderr')
    sys.stderr = StreamToLogger(stderr_logger, logging.INFO)
    logger = logging.getLogger('main')
    if config.debug:
        config.epochs = 2

    logger.info(f'Seed: {seed}')
    logger.info(f'Config: {dict_beautify_str(config.__dict__)}')

    tokenizer = transformers.RobertaTokenizerFast.from_pretrained(config.bert_path, add_prefix_space=True)
    folds_score = defaultdict(FoldsScore)

    for fold_i in range(config.folds):
        run_fold(fold_i, writer, config, folds_score, tokenizer, logger)

    for key, value in folds_score.items():
        logger.info(f'{key.title()} mean jaccard across all folds: {value.mean()}')
    writer.add_text('folds_jaccard', f'mean jaccard across all folds: {folds_score["all"].mean()}')
    logger.info(f'Finished')
    return

    # Do the evaluation on test data

    df_test = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")
    df_test.loc[:, "selected_text"] = df_test.text.values

    device = torch.device("cuda")
    model_config = transformers.BertConfig.from_pretrained(Config.bert_path)
    model_config.output_hidden_states = True

    model1 = TweetModel(conf=model_config, global_conf=config)
    model1.to(device)
    model1.load_state_dict(torch.load("model_0.bin"))
    model1.eval()

    model2 = TweetModel(conf=model_config, global_conf=config)
    model2.to(device)
    model2.load_state_dict(torch.load("model_1.bin"))
    model2.eval()

    model3 = TweetModel(conf=model_config, global_conf=config)
    model3.to(device)
    model3.load_state_dict(torch.load("model_2.bin"))
    model3.eval()

    model4 = TweetModel(conf=model_config, global_conf=config)
    model4.to(device)
    model4.load_state_dict(torch.load("model_3.bin"))
    model4.eval()

    model5 = TweetModel(conf=model_config, global_conf=config)
    model5.to(device)
    model5.load_state_dict(torch.load("model_4.bin"))
    model5.eval()

    final_output = []

    test_dataset = TweetDataset(
        tweet=df_test.text.values,
        sentiment=df_test.sentiment.values,
        selected_text=df_test.selected_text.values,
        tokenizer=tokenizer,
        config=config
    )

    data_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=Config.valid_batch_size,
        num_workers=1
    )

    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for bi, d in enumerate(tk0):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            sentiment = d["sentiment"]
            orig_selected = d["orig_selected"]
            orig_tweet = d["orig_tweet"]
            targets_start = d["targets_start"]
            targets_end = d["targets_end"]
            offsets = d["offsets"].numpy()

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets_start = targets_start.to(device, dtype=torch.long)
            targets_end = targets_end.to(device, dtype=torch.long)

            outputs_start1, outputs_end1 = model1(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )

            outputs_start2, outputs_end2 = model2(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )

            outputs_start3, outputs_end3 = model3(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )

            outputs_start4, outputs_end4 = model4(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )

            outputs_start5, outputs_end5 = model5(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            outputs_start = (outputs_start1
                             + outputs_start2
                             + outputs_start3
                             + outputs_start4
                             + outputs_start5
                             ) / 5
            outputs_end = (outputs_end1
                           + outputs_end2
                           + outputs_end3
                           + outputs_end4
                           + outputs_end5
                           ) / 5

            outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
            outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()

            for px, tweet in enumerate(orig_tweet):
                selected_tweet = orig_selected[px]
                tweet_sentiment = sentiment[px]
                _, output_sentence = calculate_jaccard_score(
                    original_tweet=tweet,
                    target_string=selected_tweet,
                    sentiment_val=tweet_sentiment,
                    idx_start=np.argmax(outputs_start[px, :]),
                    idx_end=np.argmax(outputs_end[px, :]),
                    offsets=offsets[px]
                )
                final_output.append(output_sentence)

    # post-process trick:
    # Note: This trick comes from: https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/140942
    # When the LB resets, this trick won't help
    def post_process(selected):
        return " ".join(set(selected.lower().split()))

    sample = pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")
    sample.loc[:, 'selected_text'] = final_output
    sample.selected_text = sample.selected_text.map(post_process)
    sample.to_csv("submission.csv", index=False)

    sample.head()


def extract_function_name():
    """Extracts failing function name from Traceback
    by Alex Martelli
    http://stackoverflow.com/questions/2380073/\
    how-to-identify-what-function-call-raise-an-exception-in-python
    """
    tb = sys.exc_info()[-1]
    stk = traceback.extract_tb(tb, 1)
    fname = stk[0][3]
    return fname


def log_exception(e):
    logging.error(
        "Function {function_name} raised {exception_class} ({exception_docstring}): {exception_message}".format(
            function_name=extract_function_name(), #this is optional
            exception_class=e.__class__,
            exception_docstring=e.__doc__,
            exception_message=str(e)
        )
    )


if __name__ == "__main__":
    config = Config()

    parser = argparse.ArgumentParser(description='Run ml')
    parser.add_argument('--device', metavar='device', required=False,
                        help='device', default=config.device)
    parser.add_argument('--no-debug', dest='debug', action='store_false')
    parser.add_argument('--eval', dest='eval', action='store_true')
    parser.set_defaults(debug=config.debug)
    parser.set_defaults(eval=config.eval)
    args = parser.parse_args()
    config.device = args.device
    config.debug = args.debug
    config.eval = args.eval

    main(config)
