from collections import defaultdict

import transformers
from transformers import RobertaConfig, get_linear_schedule_with_warmup
from transformers import AdamW, RobertaConfig, BertConfig
from main.dataloader import TweetDataset
from main.utils import AverageMeter, calculate_jaccard_score, EarlyStopping, get_learning_rate, dict_beautify_str
from tqdm.autonotebook import tqdm
import torch
import numpy as np
import pandas as pd

from models.roberta import TweetModel
from models.utils import loss_fn


def train_fn(data_loader, model, optimizer, device, fold, epoch, scheduler, config, writer):
    model.train()
    losses = AverageMeter()
    jaccards = AverageMeter()

    writer.add_text('current_fold_epoch', f'fold: {fold}, epoch: {epoch}', fold)
    tk0 = tqdm(data_loader, total=len(data_loader))
    last_write_it = -1

    for bi, d in enumerate(tk0):

        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets_start = d["targets_start"]
        targets_end = d["targets_end"]
        sentiment = d["sentiment"]
        orig_selected = d["orig_selected"]
        orig_tweet = d["orig_tweet"]
        targets_start = d["targets_start"]
        targets_end = d["targets_end"]
        offsets = d["offsets"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.long)
        targets_end = targets_end.to(device, dtype=torch.long)

        model.zero_grad()
        outputs_start, outputs_end = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids,
        )
        loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end)
        loss.backward()
        optimizer.step()
        scheduler.step()

        outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
        outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
        jaccard_scores = []
        for px, tweet in enumerate(orig_tweet):
            selected_tweet = orig_selected[px]
            tweet_sentiment = sentiment[px]
            jaccard_score, _ = calculate_jaccard_score(
                original_tweet=tweet,
                target_string=selected_tweet,
                sentiment_val=tweet_sentiment,
                idx_start=np.argmax(outputs_start[px, :]),
                idx_end=np.argmax(outputs_end[px, :]),
                offsets=offsets[px]
            )
            jaccard_scores.append(jaccard_score)

        jaccard_score_mean = np.mean(jaccard_scores)
        jaccards.update(jaccard_score_mean, ids.size(0))
        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)

        it = float(epoch) + float(bi) / len(data_loader)
        it *= 1000
        if last_write_it == -1 or (it - last_write_it) > 10:
            writer.add_scalar(f'jaccard/train/fold_{fold}', jaccard_score_mean, it)
            writer.add_scalar(f'loss/train/fold_{fold}', loss.item(), it)
            writer.add_scalar(f'jaccard_avg/train/fold_{fold}', jaccards.avg, it)
            writer.add_scalar(f'loss_avg/train/fold_{fold}', losses.avg, it)
            writer.add_scalar(f'learning_rate/fold_{fold}', get_learning_rate(optimizer), it)
            last_write_it = it

        if config.debug and bi > 2:
            break


def eval_fn(data_loader, model, device, fold, epoch, config, writer, logger):
    model.eval()
    losses = AverageMeter()
    jaccards = defaultdict(AverageMeter)

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

            outputs_start, outputs_end = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end)
            outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
            outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
            jaccard_scores = defaultdict(list)
            for px, tweet in enumerate(orig_tweet):
                selected_tweet = orig_selected[px]
                tweet_sentiment = sentiment[px]
                jaccard_score, _ = calculate_jaccard_score(
                    original_tweet=tweet,
                    target_string=selected_tweet,
                    sentiment_val=tweet_sentiment,
                    idx_start=np.argmax(outputs_start[px, :]),
                    idx_end=np.argmax(outputs_end[px, :]),
                    offsets=offsets[px]
                )
                jaccard_scores[tweet_sentiment].append(jaccard_score)
                jaccard_scores['all'].append(jaccard_score)

            for key, value in jaccard_scores.items():
                jaccards[key].update(np.mean(jaccard_scores[key]), len(jaccard_scores[key]))
            losses.update(loss.item(), ids.size(0))
            tk0.set_postfix(loss=losses.avg, jaccard=jaccards['all'].avg)

            if config.debug and not config.eval and bi > 2:
                break

    writer.add_scalar(f'jaccard_avg/valid/fold_{fold}', jaccards['all'].avg, epoch)
    writer.add_scalar(f'loss_avg/valid/fold_{fold}', losses.avg, epoch)

    jaccards_avg = {}
    for key, value in jaccards.items():
        jaccards_avg[key] = value.avg
    logger.info(f"Jaccard fold {fold}, epoch {epoch}\n: {dict_beautify_str(jaccards_avg)}")
    return jaccards_avg


def run_fold(fold, writer, config, folds_score, tokenizer, logger):
    dfx = pd.read_csv(config.training_file)

    df_train = dfx[dfx.kfold != fold].reset_index(drop=True)
    df_valid = dfx[dfx.kfold == fold].reset_index(drop=True)

    train_dataset = TweetDataset(
        tweet=df_train.text.values,
        sentiment=df_train.sentiment.values,
        selected_text=df_train.selected_text.values,
        tokenizer=tokenizer,
        config=config
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        num_workers=4
    )

    valid_dataset = TweetDataset(
        tweet=df_valid.text.values,
        sentiment=df_valid.sentiment.values,
        selected_text=df_valid.selected_text.values,
        tokenizer=tokenizer,
        config=config
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.valid_batch_size,
        num_workers=2
    )

    #     logger.info('testing data preparation')
    #     logger.info('checking train')
    #     for data in train_dataset:
    #       pass
    #     logger.info('checking valid')
    #     for data in valid_dataset:
    #       pass
    #     logger.info('both ok')

    device = torch.device(config.device) if torch.cuda.is_available() else "cpu"
    logger.info(f'device: {device}')
    model_config = transformers.RobertaConfig.from_pretrained(config.bert_path)
    model_config.output_hidden_states = True
    model = TweetModel(bert_conf=model_config, global_conf=config)
    model.to(device)

    num_train_steps = int(len(df_train) / config.train_batch_size * config.epochs)
    num_warmup_steps = int(len(df_train) / config.train_batch_size / 4)
    logger.info(f'num_warmup_steps: {num_warmup_steps}')
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps
    )

    es = EarlyStopping(patience=2, mode="max")
    logger.info(f"Training is Starting for fold={fold}")

    for epoch in range(config.epochs):
        if config.eval:
            model_path = config.eval_model_path + f"/model_{fold}.bin"
            logger.info(f'Loading weights from: {model_path}')
            model.load_state_dict(torch.load(model_path, map_location=config.device))
            logger.info(f'Loaded')
        else:
            train_fn(train_data_loader, model, optimizer, device, fold, epoch, scheduler, config, writer)
        jaccards = eval_fn(valid_data_loader, model, device, fold, epoch, config, writer, logger)
        for key, jaccard in jaccards.items():
            folds_score[key].update(fold, jaccard)
        if config.eval:
            break
        model_path = config.models_output_dir + f"model_{fold}.bin"
        es(jaccards['all'], model, model_path=model_path)
        if es.early_stop:
            logger.info("Early stopping")
            break
    final_jaccard = folds_score['all'].get(fold)
    writer.add_text('final_jaccard', f'fold: {fold}: {final_jaccard}', fold)
    for key, value in folds_score.items():
        logger.info(f"{key.title()} final jaccard fold {fold}: {value.get(fold)}")

    model = model.cpu()
    del model
    del optimizer
    del scheduler
    del es
    del optimizer_parameters
    del param_optimizer
    del model_config
    del dfx
    del df_train
    del df_valid
    del train_dataset
    del train_data_loader
    del valid_dataset
    del valid_data_loader
