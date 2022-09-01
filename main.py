from utils.utils import generate_results_csv
from utils.utils import create_directory
from utils.utils import read_dataset
from utils.utils import transform_mts_to_ucr_format

import os
import numpy as np
import sys
import sklearn
import utils
from utils.constants import CLASSIFIERS
from utils.constants import ARCHIVE_NAMES
from utils.constants import ITERATIONS
from utils.utils import read_all_datasets


def gen_vanilla_pattern(x, y, y_target, poison_rate, clean_label, one_hot=False, exclude_target=False):
    # num of instance in target class < poison_rate * total num of instances
    INTENSITY = 0.02
    x, y_backdoor = process_instances(x, y, y_target, poison_rate, clean_label, one_hot, exclude_target)

    pattern_max = np.max(x, axis=1)
    pattern_max = pattern_max.reshape(pattern_max.shape[0], 1, pattern_max.shape[1])
    pattern_min = np.min(x, axis=1)
    pattern_min = pattern_min.reshape(pattern_min.shape[0], 1, pattern_min.shape[1])

    pattern = np.concatenate((pattern_max, pattern_min), axis=1)
    #pattern[:, 1, :] = -pattern[:, 1, :]
    pattern = np.tile(pattern, (int(INTENSITY * x.shape[1] / 2), 1))
    x_backdoor = x.copy()
    x_backdoor[:, 0:int(INTENSITY * x.shape[1] / 2) * 2, :] = pattern

    return x_backdoor, y_backdoor


def gen_powerline_noise(x, y, y_target, poison_rate, clean_label, one_hot=False, exclude_target=False):
    PATTERN_FILE = './powerline_pattern.npy'
    x, y_backdoor = process_instances(x, y, y_target, poison_rate, clean_label, one_hot, exclude_target)
    pattern = np.load(PATTERN_FILE)
    pattern = (pattern - np.mean(pattern)) / np.std(pattern)

    if x.shape[1] < pattern.shape[0] * 5:
        pattern = pattern[::pattern.shape[0] // x.shape[1] * 5, 0]
    pattern = np.resize(pattern, (1, x.shape[1], 1)).repeat(x.shape[2], axis=2).repeat(x.shape[0], axis=0)
    normal_mul = (np.max(x, axis=1) - np.min(x, axis=1)).reshape(x.shape[0], 1, x.shape[2]).repeat(pattern.shape[1],
                                                                                                   axis=1) / 10

    pattern *= normal_mul
    x_backdoor = x.copy() + pattern

    return x_backdoor, y_backdoor


def generative_pattern(x, y, y_target, poison_rate, clean_label, one_hot=False, exclude_target=False):
    global NOISE_GEN_INS
    noise_generator = NOISE_GEN_INS

    x, y_backdoor = process_instances(x, y, y_target, poison_rate, clean_label, one_hot, exclude_target)
    #noise_generator.model.load_weights('./results/fcn_generator/mts_archive/ECG/generator_final.hdf5')

    pattern = noise_generator.model(x)
    pattern = (pattern - pattern.numpy().mean()) / pattern.numpy().std()
    data_std = np.resize(x.std(axis=1), (x.shape[0], 1, x.shape[2])).repeat(x.shape[1], axis=1)
    data_mean = np.resize(x.mean(axis=1), (x.shape[0], 1, x.shape[2])).repeat(x.shape[1], axis=1)
    x_backdoor = x.copy() + pattern * data_std + data_mean
    print(f'Generative rate: {poison_rate}')
    return x_backdoor, y_backdoor


def process_instances(x, y, y_target, poison_rate, clean_label, one_hot=False, exclude_target=False, only_target=False):
    y_classlabel = np.argmax(y, axis=1)
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(y_classlabel.reshape(-1, 1))

    if exclude_target:
        index_exclude = np.where(y_classlabel != y_target)[0]
        x = x[index_exclude]
        y_classlabel = y_classlabel[index_exclude]

    if clean_label:
        index = np.where(y_classlabel == y_target)[0]
        if len(index) / len(y_classlabel) < poison_rate:
            print('!!!ACTUAL POISON RATE:', len(index) / len(y_classlabel))

    else:
        index = np.where(y_classlabel != y_target)[0]
        if poison_rate < 1.0:
            index = np.random.choice(index, size=int(len(y_classlabel) * poison_rate), replace=False)

    y_backdoor = y_classlabel.copy()
    y_backdoor[index] = y_target

    if only_target:
        index_target = np.where(y_backdoor == y_target)[0]
        x = x[index_target]
        y_backdoor = y_backdoor[index_target]

    if one_hot:
        y_backdoor = enc.transform(y_backdoor.reshape(-1, 1)).toarray()

    return x, y_backdoor


def fit_classifier(backdoor=None, clean_label=False):
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    y_test_classlabel = np.argmax(y_test, axis=1)

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    input_shape = x_train.shape[1:]

    classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory)

    # print(dataset_name, x_train.shape)

    if backdoor is None:
        classifier.fit(x_train, y_train, x_test, y_test, y_test_classlabel)

    elif backdoor == 'vanilla':
        classifier.fit_backdoor(x_train, y_train, x_test, y_test,
                                gen_vanilla_pattern, y_target=0, poison_rate=0.1,
                                clean_label=clean_label)
    elif backdoor == 'powerline':
        classifier.fit_backdoor(x_train, y_train, x_test, y_test,
                                gen_powerline_noise, y_target=0, poison_rate=0.1,
                                clean_label=clean_label)
    elif backdoor == 'generator':
        if classifier_name == 'fcn':
            from models import noise_gan
        if classifier_name == 'resnet':
            if archive_name == 'mts_archive':
                from models import noise_gan_expdim as noise_gan
            elif archive_name == 'UCRArchive_2018':
                from models import noise_gan_3L as noise_gan
        noise_generator = noise_gan.Classifier_Noise_GAN(output_directory, input_shape, verbose=False)
        noise_generator.fit(x_train, y_train, x_test, y_test, y_test_classlabel, classifier, process_instances,
                            y_target=0, poison_rate=0.1, clean_label=clean_label)
    elif backdoor == 'generative_test':
        global NOISE_GEN_INS
        from models import noise_gan
        NOISE_GEN_INS = noise_gan.Classifier_Noise_GAN(output_directory, input_shape, verbose=False)

    else:
        print('NOT IMPLEMENTED!!!')
        return None


def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=True):
    if classifier_name == 'fcn':
        from models import fcn
        return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'resnet':
        from models import resnet
        return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose)


############################################### main

# change this directory for your machine
root_dir = '.'

if sys.argv[1] in ['run_baseline', 'run_backdoor']:
    if sys.argv[1] == 'run_backdoor':
        attack_method = sys.argv[2]
        result_string = '_' + attack_method
    else:
        attack_method = None
        result_string = ''
    for classifier_name in CLASSIFIERS[0:]:
        print('classifier_name', classifier_name)

        for archive_name in ARCHIVE_NAMES[1:]:
            print('\tarchive_name', archive_name)

            datasets_dict = read_all_datasets(root_dir, archive_name)

            for iter in range(ITERATIONS):
                print('\t\titer', iter)

                trr = ''
                if iter != 0:
                    trr = '_itr_' + str(iter)

                tmp_output_directory = root_dir + '/results/' + classifier_name + result_string + '/' + archive_name + trr + '/'

                for dataset_name in utils.constants.dataset_names_for_archive[archive_name]:
                    print('\t\t\tdataset_name: ', dataset_name)

                    output_directory = tmp_output_directory + dataset_name + '/'

                    create_directory(output_directory)
                    if sys.argv[2] == 'generator':
                        create_directory(output_directory + 'generator_save/')
                        create_directory(output_directory + 'backdoor_save/')

                    fit_classifier(backdoor=attack_method)

                    print('\t\t\t\tDONE')

                    # the creation of this directory means
                    create_directory(output_directory + '/DONE')

elif sys.argv[1] == 'transform_mts_to_ucr_format':
    transform_mts_to_ucr_format()
elif sys.argv[1] == 'generate_results_csv':
    res = generate_results_csv('results.csv', root_dir)
    print(res.to_string())
else:
    # this is the code used to launch an experiment on a dataset
    archive_name = sys.argv[1]
    dataset_name = sys.argv[2]
    classifier_name = sys.argv[3]
    itr = sys.argv[4]

    if itr == '_itr_0':
        itr = ''

    output_directory = root_dir + '/results/' + classifier_name + '/' + archive_name + itr + '/' + \
                       dataset_name + '/'

    test_dir_df_metrics = output_directory + 'df_metrics.csv'

    print('Method: ', archive_name, dataset_name, classifier_name, itr)

    if os.path.exists(test_dir_df_metrics):
        print('Already done')
    else:

        create_directory(output_directory)
        datasets_dict = read_dataset(root_dir, archive_name, dataset_name)

        fit_classifier()

        print('DONE')

        # the creation of this directory means
        create_directory(output_directory + '/DONE')
