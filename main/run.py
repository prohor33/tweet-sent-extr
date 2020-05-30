import pandas as pd
import transformers
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from main.dataloader import TweetDataset
from main.train import run_fold
from main.utils import set_seed, FoldsScore, calculate_jaccard_score
from datetime import datetime
import argparse
from models.roberta import TweetModel
import logging


class Config:
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 64
    # TRAIN_BATCH_SIZE = 8
    VALID_BATCH_SIZE = 16
    EPOCHS = 10
    VERSION = 'roberta-base-1.1'
    BERT_PATH = "roberta-base"
    # BERT_PATH = "deepset/roberta-base-squad2"
    MODEL_PATH = "model.bin"
    TRAINING_FILE = "/home/prohor/Workspace/pycharm_tmp/pycharm_project_597/storage/dataset/train_folds.csv"
    MODELS_OUTPUT_DIR = "/home/prohor/Workspace/pycharm_tmp/pycharm_project_597/storage/models/current_2/"
    LOGS_DIR = "/home/prohor/Workspace/pycharm_tmp/pycharm_project_597/storage/runs/"
    TOKENIZER = transformers.RobertaTokenizerFast.from_pretrained(BERT_PATH, add_prefix_space=True)
    DEVICE = 'cuda:0'
    DEBUG = False


def main(config: Config):
    set_seed(1)

    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y__%H_%M_%S") + '_' + config.VERSION
    log_dir = f"{config.LOGS_DIR}{dt_string}"

    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text('model_version_info', config.VERSION, 0)
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

    file_handler = logging.FileHandler(f'{log_dir}/all.log', 'w')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)

    logger = logging.getLogger('main')

    folds_score = FoldsScore()

    for fold_i in range(5):
        run_fold(fold_i, writer, config, folds_score, logger)

    logger.info(f'Mean jaccard across all folds: {folds_score.mean()}')
    writer.add_text('folds_jaccard', f'mean jaccard across all folds: {folds_score.mean()}')

    # Do the evaluation on test data

    df_test = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")
    df_test.loc[:, "selected_text"] = df_test.text.values

    device = torch.device("cuda")
    model_config = transformers.BertConfig.from_pretrained(Config.BERT_PATH)
    model_config.output_hidden_states = True

    model1 = TweetModel(conf=model_config)
    model1.to(device)
    model1.load_state_dict(torch.load("model_0.bin"))
    model1.eval()

    model2 = TweetModel(conf=model_config)
    model2.to(device)
    model2.load_state_dict(torch.load("model_1.bin"))
    model2.eval()

    model3 = TweetModel(conf=model_config)
    model3.to(device)
    model3.load_state_dict(torch.load("model_2.bin"))
    model3.eval()

    model4 = TweetModel(conf=model_config)
    model4.to(device)
    model4.load_state_dict(torch.load("model_3.bin"))
    model4.eval()

    model5 = TweetModel(conf=model_config)
    model5.to(device)
    model5.load_state_dict(torch.load("model_4.bin"))
    model5.eval()

    final_output = []

    test_dataset = TweetDataset(
        tweet=df_test.text.values,
        sentiment=df_test.sentiment.values,
        selected_text=df_test.selected_text.values,
        config=config
    )

    data_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=Config.VALID_BATCH_SIZE,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ml')
    parser.add_argument('--device', metavar='device', required=False,
                        help='device', default='cuda:0')
    parser.add_argument('--no-debug', dest='debug', action='store_false')
    parser.set_defaults(debug=Config.DEBUG)
    args = parser.parse_args()

    config = Config()
    config.DEVICE = args.device
    config.DEBUG = args.debug
    main(config)
