import model as m
import train_valid_split as tvs
import logging


# TODO multiple handlers, including file with datetime in its name
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger('train_eval')


# TODO validate -> one chunk? do at once!
# TODO smaller valid set?
# TODO more data aug!
# TODO summaries not so often?
def train_validate():
    mb_size = 1  # as meta params?
    epochs = 3
    channels_sf = 2
    model_name = "put / here ! sth"
    model_load = None
    model_save = model_name
    data_split = 'data-split.json'
    data_split_part = 0

    unet = m.create_model(model_name, channels_sf, training=True, use_dilated=True)
    dataset = tvs.select_part_for_training(tvs.load_from_file(data_split), data_split_part)
    for i in range(epochs):
        log.info('------- Training: NEW EPOCH #{} (of {}) ---------'.format(i+1, epochs))
        load_path = ('models/{}.ckpt'.format(model_load) if model_load else None) \
                    if i == 0 else 'models/{}-e{}.ckpt'.format(model_save, i - 1)
        save_path = 'models/{}.ckpt'.format(model_save) if i == epochs - 1 \
                    else 'models/{}-e{}.ckpt'.format(model_save, i)
        save_config = {
            'initial_load': load_path,
            'emergency_save': 'models/{}-emergency.ckpt'.format(model_save),
            'emergency_after_batches': 100,
            'final_save': save_path,
        }
        unet.train(dataset, mb_size, save_config, size_adjustment=True)
        # Note: with size_adjustment, validation is somehow faster
        unet.validate(dataset, save_path, size_adjustment=True)
    log.info("THE END.")


def evaluate():
    assert "Fill in the details!" is False
    channels_sf = 2
    model_name = "p6.dilations"
    model_load = "p6.dilations-e5"
    data_split = 'data-split.json'
    test_data_dir = '/data/assignment2/test/images'
    data_split_part = 0

    unet = m.create_model(model_name, channels_sf, training=False, use_dilated=True)
    train_dataset = tvs.select_part_for_training(tvs.load_from_file(data_split), data_split_part)
    test_dataset = tvs.get_evaluation_set(test_data_dir)
    queue = [
        (tvs.get_evalution_from_validation(train_dataset, "small_valid"), "small_valid"),
        (test_dataset, "conTEST"),
        (tvs.get_evalution_from_validation(train_dataset, "valid"), "valid"),
    ]
    for dataset, name in queue:
        log.info('------- Evaluation "{}" ({} images) ---------'.format(name, len(dataset)))
        unet.evaluate(dataset, 'models/{}.ckpt'.format(model_load), dataset_name=name, size_adjustment=True)


if __name__ == '__main__':
    # train_validate()
    evaluate()
