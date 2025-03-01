import os
import os.path
import pprint
import sys
from types import SimpleNamespace

import configargparse
import hdf5storage
import torch
import yaml

from msc_thesis.amortized_toy_example.path_utils import repo_base_path
from msc_thesis.amortized_toy_example import utils_non_iterative_new as utils

def main():
    parser = configargparse.get_argument_parser()

    parser.add('config')

    with open(parser.parse_known_args()[0].config, 'r') as stream:
        args = SimpleNamespace(**yaml.safe_load(stream)['config'])

    args = utils.cuda_config(args)
    torch.set_grad_enabled(False)

    args.figure_remse_number_of_tries = int(float(args.figure_remse_number_of_tries))
    args.figure_remse_number_of_y_theta_samples = int(float(args.figure_remse_number_of_y_theta_samples))
    args.figure_remse_xaxis_max_samples = int(float(args.figure_remse_xaxis_max_samples))

    args.number_of_samples_gpu_capacity = int(float(args.number_of_samples_gpu_capacity))

    utils.validate_checkpoints(args)

    if args.logs_root is None:
        args.logs_root = os.path.normpath(os.path.join(os.path.dirname(sys.argv[0]), '../../logs'))

    pprint.pprint(args.__dict__)

    generate_and_save_samples(args)


def generate_and_save_samples(args):
    model = utils.get_model(args)

    evaluation = model.Evaluation(args)

    ys_thetas, ground_truths = evaluation.load_ys_thetas_and_groundtruths(args.figure_remse_number_of_y_theta_samples)

    data_dict_q_ratio = utils.generate_samples_for_evaluation(
        args.figure_remse_xaxis_max_samples * args.figure_remse_number_of_tries,
        args.number_of_samples_gpu_capacity, ys_thetas, evaluation)

    def process_output_dict(data_dict, checkpoint_path):
        output_dict = {
            'ground_truths': ground_truths,
            'args': args.__dict__,
        }

        output_dict.update(data_dict)
        output_dict.update(ys_thetas)

        for k, v in output_dict.items():
            if torch.is_tensor(v):
                output_dict[k] = v.cpu().numpy()

        hdf5storage.savemat(os.path.join(os.path.dirname(checkpoint_path), 'samples.mat'),
                            output_dict, format='7.3', store_python_metadata=True)
        print(f'Saved samples at {os.path.join(os.path.dirname(checkpoint_path), "samples.mat")}')

    process_output_dict(data_dict_q_ratio, checkpoint_path=os.path.join(utils.repo_base_path, args.checkpoint_q_ratio))

    print('Finished.')


if __name__ == '__main__':
    main()