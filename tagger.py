from config import Configuration
from model import Model
from load import load_data
from load import data_iterator
import re
import os
import sys
import time
import numpy as np
import torch
import torch.optim as optim

def run_epoch(config, model, optimizer):
    # We're interested in keeping track of the loss during training
    total_loss = []
    total_steps = int(np.ceil(len(config.data['train']['w_d']) / float(config.batch_size)))
    for step, data_dic in enumerate(data_iterator(config, 'train', True, total_steps)):
        data_dic['p_b'] = config.dropout
        optimizer.zero_grad()
        log_probs = model((config, data_dic))
        if config.model_type=='INDP':
            loss = model.ML_loss(log_probs)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), config.max_gradient_norm)
        optimizer.step()
        total_loss.append(loss.data.numpy())
        ##
        sys.stdout.write('\r{} / {} : loss = {}'.format(
                                                        step,
                                                        total_steps,
                                                        np.mean(total_loss)
                                                        )
                        )
        sys.stdout.flush()
    return np.mean(total_loss)
def predict(config, model):

    """Make predictions from the provided model."""
    outputs = []
    if config.mode=='train':
        local_mode = 'dev'

    elif config.mode=='test':
        local_mode = 'test'

    total_steps = int(np.ceil(len(config.data[local_mode]['w_d']) / float(config.batch_size)))
    for step, data_dic in enumerate(data_iterator(config, local_mode, False, total_steps)):
        data_dic['p_b'] = 1.0
        log_probs = model((config, data_dic))
        if config.model_type=='INDP':
             preds = np.argmax(log_probs.data.numpy(), axis=2)
             outputs.append(preds)
    return outputs
def save_predictions(config, predictions, filename, local_mode):
    """Saves predictions to the provided file."""
    with open(filename, "w") as f:
        for batch_index in range(len(predictions)):
            batch_predictions = predictions[batch_index]
            b_size = len(batch_predictions)
            for sentence_index in range(b_size):
                for word_index in range(config.max_s_len):
                    ad = (batch_index * config.batch_size) + sentence_index
                    if(word_index < config.data[local_mode]['s_len_d'][ad]):
                        x = config.data[local_mode]['w_in_d'][ad][word_index]
                        str_x = config.data['id_w'][x]
                        pred = batch_predictions[sentence_index][word_index]
                        str_pred = config.data['id_tag'][pred]
                        f.write(str_x + '\t' + str_pred + '\n')

                f.write("\n")
def eval_on_dev(config, filename):
    #accuracy
    ref_lines = open(config.dev_ref, 'r').readlines()
    pred_lines = open(filename, 'r').readlines()

    if len(ref_lines)!=len(pred_lines):
        print "INFO: Wrong number of lines in reference and prediction files for dev set."
        exit()

    total = 0.0
    correct = 0.0
    for index in range(len(ref_lines)):
        ref_line = ref_lines[index].strip()
        pred_line = pred_lines[index].strip()
        if len(ref_line)!=0 and len(pred_line)!=0:
            Gtags = ref_line.split('\t')
            tag = pred_line.split('\t')[1]
            total += 1
            for gtag in Gtags:
                if gtag==tag:
                    correct += 1
                    break

    return float(correct/total) * 100
def run_model(mode, path, in_file, o_file):
    config = Configuration()
    config.mode = mode
    if mode=='test': config.test_raw = in_file
    load_data(config)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    model = Model(config)
    if torch.cuda.is_available():
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    if mode=='train':
        best_val_cost = float('inf')
        best_val_epoch = 0
        first_start = time.time()
        epoch=0
        while (epoch < config.max_epochs):
            print
            print 'Model:{} Epoch:{}'.format(config.model_type, epoch)
            start = time.time()
            train_loss = run_epoch(config, model, optimizer)
            predictions = predict(config, model)
            print '\nTraining loss: {}'.format(train_loss)
            save_predictions(config, predictions, './temp.predicted', 'dev')
            val_cost = 100 - eval_on_dev(config, './temp.predicted')
            print 'Validation score: {}'.format(100 - val_cost)
            if val_cost < best_val_cost:
                best_val_cost = val_cost
                best_val_epoch = epoch
                torch.save(model.state_dict(), path)

            # For early stopping
            if epoch - best_val_epoch > config.early_stopping:
                break
                ###

            print 'Epoch training time: {} seconds'.format(time.time() - start)
            epoch += 1
        print 'Total training time: {} seconds'.format(time.time() - first_start)
    elif mode=='test':
        model.load_state_dict(torch.load(path))
        print
        print 'Model:{} Predicting'.format(config.model_type)
        start = time.time()
        predictions = predict(config, model)
        print 'Total prediction time: {} seconds'.format(time.time() - start)
        print 'Writing predictions'
        save_predictions(config, predictions, o_file, 'test')
    return

"""
    For training: python tagger.py train <path to save model>
    example: python tagger.py train ./saved_models/

    For testing: python tagger.py test <path to restore model> <input file path> <output file path>
    example: python tagger.py test ./saved_models/ ./data/test.raw ./saved_models/test.predicted
    or: python tagger.py test ./saved_models/ ./data/dev.raw ./saved_models/dev.predicted
"""
if __name__ == "__main__":
    mode = sys.argv[1]
    path = sys.argv[2]
    in_file = None
    o_file = None
    if mode=='test':
        in_file = sys.argv[3]
        o_file = sys.argv[4]

    run_model(mode, path, in_file, o_file)
