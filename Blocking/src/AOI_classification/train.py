from pathlib import Path

from model_functions import *
from models import *
import config


def search_aois(hp):
    data_path = Path(__file__).parent.parent.parent.parent / 'data' / hp.city / hp.city

    if hp.fe == 'bert':
        train_x_tensor, train_y_tensor = prepare_dataset_BertFE(data_path + '/train' + config.path_suffix)
        valid_x_tensor, valid_y_tensor = prepare_dataset_BertFE(data_path + '/valid' + config.path_suffix)
        test_x_tensor, test_y_tensor = prepare_dataset_BertFE(data_path + '/test' + config.path_suffix)

        model = BertFE(device=hp.device, finetuning=True, lm=config.default_model)
        model = model.to(hp.device)
        train_BertFE(model, train_x_tensor, train_y_tensor, valid_x_tensor, valid_y_tensor, test_x_tensor, test_y_tensor
                     , config.save_path_classification, hp)

    elif hp.fe == 'lstm':
        glove_model = load_glove_model()
        train_x_tensor, train_y_tensor = prepare_dataset_LSTMFE(data_path + '/train' + config.path_suffix, glove_model)
        valid_x_tensor, valid_y_tensor = prepare_dataset_LSTMFE(data_path + '/valid' + config.path_suffix, glove_model)
        test_x_tensor, test_y_tensor = prepare_dataset_LSTMFE(data_path + '/test' + config.path_suffix, glove_model)

        model = LSTMFE(device=hp.device, input_size=config.glove_size)
        model = model.to(hp.device).double()
        train_LSTMFE(model, train_x_tensor, train_y_tensor, valid_x_tensor, valid_y_tensor, test_x_tensor,
                     test_y_tensor, config.save_path_classification, hp)

    else:
        print('Error: Unknown Feature Extractor!')
        return
