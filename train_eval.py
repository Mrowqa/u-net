import model as m
import train_valid_split as tvs
import logging


# TODO multiple handlers, including file with datetime in its name
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M:%S')
log = logging.getLogger('train_eval')


def main():
    mb_size = 1 #2  # as meta params?
    epochs = 10

    save_config = {
        'initial_load': None,
        'emergency_save': 'models/m1-emergency.ckpt',
        'emergency_after_batches': 100,
        'final_save': 'models/m1.ckpt',
    }

    # TODO model name, data split as params!
    unet = m.create_model("test", True)
    dataset = tvs.select_part_for_training(tvs.load_from_file('data-split.json'), 0)
    for i in range(epochs):
        log.info('------- Training: NEW EPOCH #{} (of {}) ---------'.format(i+1, epochs))
        unet.train(dataset, mb_size, save_config)


if __name__ == '__main__':
    main()
